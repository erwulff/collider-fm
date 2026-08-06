"""Panda-style per-point t-SNE visualization of the pretrained backbone.

Reproduces the Panda paper (arXiv 2512.01324, Figure 2): embed the backbone's per-point
features -- one dot per grid-sampled calorimeter hit, unit-normalized -- and scatter the
2D t-SNE colored by particle type and by event id. Particle type collapses the raw
``pdg_id`` (of the dominant contributing particle) to 7 coarse calorimetry-role buckets
(e± / γ / μ± / charged hadron / neutral hadron / nucleus / other) so the legend stays
readable; see :func:`collider_fm.evaluation_labels.pdg_bucket`.

Two feature spaces: the **full up-cast** (all pooling levels → input resolution, an
exact 1:1 coord bijection to raw hits) and **up_cast(2)** (the pretraining space, where
each point is a downsampled voxel inheriting its cluster's energy-dominant pdg). The
full up-cast is needed because ``up_cast(level=2)`` stops at downsampled stage-2, which
has no mapping back to raw hits. Augmentation is off (one clean base view per event),
matching the paper's ``model(data)`` on raw data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ._panda.structure import Point
from .evaluation_labels import PDG_BUCKET_NAMES, pdg_bucket
from .views import build_point_view_from_event

__all__ = [
    "TsnePointCollection",
    "PDG_BUCKET_COLORS",
    "full_up_cast",
    "match_to_input_coords",
    "collect_tsne_points",
    "tsne_plot",
    "make_tsne_plots",
]

# Display colors for the 7 calorimetry-role buckets, keyed by PDG_BUCKET_NAMES. A
# missing name raises at plot time, keeping the two lists from drifting.
PDG_BUCKET_COLORS: dict[str, str] = {
    "e±": "tab:red",
    "γ": "tab:orange",
    "μ±": "tab:purple",
    "charged hadron": "tab:blue",
    "neutral hadron": "tab:pink",
    "nucleus": "tab:olive",
    "other": "tab:gray",
}

# Per-event colors (cycles for >10 events); fine for the event-separation floor-check.
_EVENT_COLORS: list[str] = [
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
]


def full_up_cast(point: Point) -> Point:
    """Rebuild per-point features at input resolution by walking the full pooling chain.

    Mirrors ``Panda_repo/panda/model_base.py`` ``forward(upcast=True)``: repeatedly pop
    ``pooling_parent`` / ``pooling_inverse`` and concatenate each parent's feature with
    its children's features broadcast back via ``point.feat[inverse]``, until no parent
    remains. Unlike :meth:`SonataModel.up_cast` (which stops after ``up_cast_level``),
    this walks *all* levels and returns one feature per input point. Consumes the
    breadcrumb keys (run on a fresh enc output each call).
    """
    while "pooling_parent" in point.keys():
        if "pooling_inverse" not in point.keys():
            raise KeyError("Full up-cast requires traceable PTv3 pooling features.")
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    return point


def _up_cast_levels(point: Point, n_levels: int) -> Point:
    """Walk ``n_levels`` pooling levels up (coarse -> finer), mirroring SonataModel.up_cast.

    Consumes ``n_levels`` of the ``pooling_parent`` / ``pooling_inverse`` breadcrumbs.
    After ``n_levels`` steps the point sits ``n_levels`` resolutions finer than the
    deepest encoder stage, with features = ``cat`` of the levels walked. The remaining
    ``depth - n_levels`` breadcrumbs stay on the point for :func:`upcast2_input_map`.
    """
    for _ in range(n_levels):
        if (
            "pooling_parent" not in point.keys()
            or "pooling_inverse" not in point.keys()
        ):
            raise KeyError("Up-cast requires traceable PTv3 pooling features.")
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    return point


def upcast2_input_map(point: Point, up_cast_level: int = 2) -> tuple[Point, torch.Tensor]:
    """Up-cast ``up_cast_level`` levels and map each input point to its up-cast cluster.

    Runs :func:`_up_cast_levels` (consuming ``up_cast_level`` breadcrumbs), then inverts
    the *remaining* pooling chain to recover, for every stage-0 (input) point, which
    up-cast point it pooled into. This lets each up-cast point (a downsampled voxel)
    inherit a label from its cluster of input points. Verified exact: the coord-mean
    invariant of ``GridPooling`` (each cluster's coord = mean of its members' coords)
    holds to float noise (~1e-3).

    Args:
        point: the backbone output (deepest stage, full pooling chain intact).
        up_cast_level: levels to walk up (the pretraining ``up_cast_level``, default 2).

    Returns:
        ``(upcast_point, input_to_cluster)``: the up-cast :class:`Point`
        (``[N_up, D]`` features) and a ``[N0]`` long tensor mapping each input point to
        its up-cast cluster id in ``[0, N_up)``.
    """
    upcast_point = _up_cast_levels(point, up_cast_level)
    n_up = upcast_point.feat.shape[0]

    # Invert the remaining chain: each up-cast point starts as its own cluster, then
    # expand membership one level at a time (coarse -> finer) until stage-0.
    membership = torch.arange(n_up, device=upcast_point.feat.device)
    cur = upcast_point
    while "pooling_parent" in cur.keys():
        inverse = cur["pooling_inverse"]  # [N_finer] -> coarse cluster id in `cur`
        membership = membership[inverse]  # broadcast coarse membership to finer points
        cur = cur["pooling_parent"]
    return upcast_point, membership


def match_to_input_coords(
    out_coord: torch.Tensor, in_coord: torch.Tensor
) -> torch.Tensor:
    """For each output point, the index of its nearest input point by Euclidean distance.

    Recovers the input row (-> ``source_index`` / raw hit index) for each full-up-cast
    output point, which is at the same resolution but reordered by serialization.
    Returns ``[num_out]`` long indices into ``in_coord``; on the happy path (verified)
    this is a bijection within float noise.

    Uses ``scipy.spatial.cKDTree`` on CPU (O(N log N), bounded memory) rather than an
    O(N^2) ``cdist`` -- the per-event point count (~10^4) makes the dense distance
    matrix GPU-prohibitive on a shared machine.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(in_coord.float().cpu().numpy())
    _, idx = tree.query(out_coord.float().cpu().numpy(), k=1)
    return torch.as_tensor(idx, dtype=torch.long)


@dataclass
class TsnePointCollection:
    """Bounded per-point features + coloring channels for the t-SNE plots.

    All arrays are ``[M]`` (or ``[M, D]`` for features) over ``M <= max_points``
    subsampled points across ``num_events`` events. ``pdg_id`` uses ``-1`` for unknown;
    ``event_id`` is a per-event index in ``[0, num_events)``. Prototype coloring is
    absent: the diagnostics head is dimensionally bound to the ``up_cast(2)`` features,
    not the full-up-cast features embedded here.
    """

    features: torch.Tensor  # [M, D] unit-normalized features
    pdg_id: torch.Tensor  # [M] dominant particle's pdg_id (-1 if unknown)
    event_id: torch.Tensor  # [M] per-event index in [0, num_events) (-1 if none)
    num_events: int


@torch.no_grad()
def collect_tsne_points(
    model,
    calo_truth_dataset: Any,
    pid_to_pdg: Mapping[int, int] | None,
    device: torch.device,
    *,
    view_kwargs: Mapping[str, Any],
    max_events: int,
    max_points: int = 20000,
    seed: int = 0,
    feature_space: str = "full",
    up_cast_level: int = 2,
) -> TsnePointCollection:
    """Collect bounded per-point features + coloring channels for the t-SNE plots.

    For each event: build one augmentation-free base view
    (:func:`build_point_view_from_event`), run the deterministic teacher backbone, then
    extract features in one of two spaces:

    - ``"full"`` (default): :func:`full_up_cast` to input resolution (one feature per
      grid-sampled hit). Each output point is coord-matched back to its input row ->
      ``source_index`` (raw hit index) -> the dominant particle's ``pdg_id``.
    - ``"upcast2"``: up-cast only ``up_cast_level`` levels (the pretraining feature
      space). These points are downsampled vs input; each is a cluster of input points
      (recovered exactly via :func:`upcast2_input_map`) and inherits the
      **energy-dominant** pdg across its raw hits.

    ``event_id`` (per-event index) is assigned in both spaces. Points are subsampled
    proportionally across events to bound total memory at ``max_points``.
    """
    from .evaluation_labels import event_dominant_particles

    if feature_space not in {"full", "upcast2"}:
        raise ValueError(f"feature_space must be 'full' or 'upcast2', got {feature_space!r}")

    model.eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    feat_parts: list[torch.Tensor] = []
    pdg_parts: list[torch.Tensor] = []
    eid_parts: list[torch.Tensor] = []
    total = 0
    events_done = 0

    from tqdm import tqdm

    pbar = tqdm(
        calo_truth_dataset, total=max_events, desc=f"t-SNE [{feature_space}]",
        unit="event", mininterval=2.0,
    )
    for event_index, event in enumerate(pbar):
        if events_done >= max_events:
            break

        # Raw calo_truth yields flat events; build_point_view_from_event expects the
        # wrapped form with on-device tensors. Keep the flat event's contrib_* for the
        # label lookup below.
        wrapped = {
            "calo_hits": {
                "x": torch.as_tensor(event["x"], dtype=torch.float32, device=device),
                "y": torch.as_tensor(event["y"], dtype=torch.float32, device=device),
                "z": torch.as_tensor(event["z"], dtype=torch.float32, device=device),
                "total_energy": torch.as_tensor(
                    event["total_energy"], dtype=torch.float32, device=device
                ),
            }
        }
        view = build_point_view_from_event(wrapped, device=device, **dict(view_kwargs))
        in_coord = view["coord"]
        n_in = in_coord.shape[0]
        if n_in == 0:
            continue

        _, pdg_per_hit, _, _ = event_dominant_particles(event, pid_to_pdg)
        hit_energy = np.asarray(event["total_energy"], dtype=np.float64)

        point = Point(
            feat=view["feat"].float(),
            coord=in_coord.float(),
            origin_coord=view["origin_coord"].float(),
            offset=torch.tensor([n_in], dtype=torch.long, device=device),
            grid_size=view["grid_size"],
            mask=torch.zeros(n_in, dtype=torch.bool, device=device),
        )
        point = model.teacher["backbone"](point)

        if feature_space == "full":
            feats, pdg_lab = _full_space_labels(point, view, pdg_per_hit)
        else:
            feats, pdg_lab = _upcast2_space_labels(
                point, view, pdg_per_hit, hit_energy, up_cast_level
            )
        n_out = feats.shape[0]

        remaining_budget = max_points - total
        remaining_events = max(1, max_events - event_index)
        quota = max(1, int(round(remaining_budget / remaining_events)))
        k = min(n_out, quota, remaining_budget)
        if k < n_out:
            idx = torch.randperm(n_out, generator=generator)[:k]
        else:
            idx = torch.arange(n_out)
        feats = F.normalize(feats[idx].float().cpu(), dim=-1)

        feat_parts.append(feats)
        pdg_parts.append(pdg_lab[idx])
        eid_parts.append(torch.full((k,), events_done, dtype=torch.long))
        total += k
        events_done += 1

        pbar.set_postfix(points=total, refresh=False)

        if total >= max_points:
            break
    pbar.close()

    return TsnePointCollection(
        features=torch.cat(feat_parts) if feat_parts else torch.empty(0, 0),
        pdg_id=torch.cat(pdg_parts) if pdg_parts else torch.empty(0, dtype=torch.long),
        event_id=torch.cat(eid_parts) if eid_parts else torch.empty(0, dtype=torch.long),
        num_events=events_done,
    )


def _full_space_labels(point, view, pdg_per_hit):
    """Full-up-cast: input-resolution features, per-hit pdg via coord bijection."""
    point = full_up_cast(point)
    feats = point.feat  # [n_in, D] at input resolution
    out_coord = point.coord  # [n_in, 3], reordered vs input

    in_idx = match_to_input_coords(out_coord, view["coord"])
    raw_hit = view["source_index"][in_idx].cpu().numpy().astype(np.int64)
    pdg_lab = torch.from_numpy(pdg_per_hit[raw_hit])
    return feats, pdg_lab


def _upcast2_space_labels(point, view, pdg_per_hit, hit_energy, up_cast_level):
    """up_cast(2): downsampled features, energy-dominant pdg per cluster.

    Each up-cast point is a cluster of input points (recovered exactly by
    :func:`upcast2_input_map`) and inherits the energy-dominant pdg across its raw hits
    via :func:`_cluster_dominant_pdg`.
    """
    upcast_point, input_to_cluster = upcast2_input_map(point, up_cast_level)
    feats = upcast_point.feat  # [n_up, D]

    # Map each input point -> raw hit, then each cluster -> energy-dominant pdg.
    raw_hit = view["source_index"].cpu().numpy().astype(np.int64)
    n_up = feats.shape[0]
    pdg_lab = _cluster_dominant_pdg(
        input_to_cluster.cpu().numpy(), raw_hit, pdg_per_hit, hit_energy, n_up
    )
    pdg_lab = torch.from_numpy(pdg_lab)
    return feats, pdg_lab


def _cluster_dominant_pdg(
    input_to_cluster: np.ndarray,
    raw_hit: np.ndarray,
    pdg_per_hit: np.ndarray,
    hit_energy: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """Energy-dominant pdg per cluster (the particle type of the voxel).

    For each cluster, sum each contributing particle's cell energy across its raw hits
    and take the argmax pdg (``-1`` if none known) -- the dominant depositor, not the
    plurality of hits.
    """
    cluster_of = input_to_cluster  # [n_in] -> [0, n_clusters)
    hit_pdg = pdg_per_hit[raw_hit]  # [n_in] dominant pdg per input point's raw hit
    hit_eng = hit_energy[raw_hit]  # [n_in] that raw hit's total cell energy

    # "known" = has a pdg at all. The sentinel is -1 (dominant particle missing from
    # the pid->pdg map); antiparticles are valid negative pdgs (-11 e+, -211 pi-, ...),
    # so the filter is `!= -1`, NOT `>= 0` (which would wrongly drop antiparticles).
    known = hit_pdg != -1
    c = cluster_of[known]
    p = hit_pdg[known]
    e = hit_eng[known]
    out = np.full(n_clusters, -1, dtype=np.int64)
    if p.size == 0:
        return out

    # Weighted energy per (cluster, pdg): scatter-add e into [n_clusters, n_pdg_ids].
    pdg_ids = np.unique(p)
    pdg_to_col = {v: i for i, v in enumerate(pdg_ids.tolist())}
    accum = np.zeros((n_clusters, len(pdg_ids)), dtype=np.float64)
    cols = np.array([pdg_to_col[v] for v in p.tolist()], dtype=np.int64)
    np.add.at(accum, (c, cols), e)

    has_energy = accum.sum(axis=1) > 0
    out[has_energy] = pdg_ids[accum[has_energy].argmax(axis=1)]
    return out


def tsne_plot(
    features: torch.Tensor,
    color: torch.Tensor,
    path: str | Path,
    *,
    title: str = "",
    color_label: str = "label id",
    max_points: int = 8000,
    seed: int = 0,
    categories: Sequence[tuple[str, str]] | None = None,
) -> None:
    """2D t-SNE scatter of ``features`` colored by ``color``; saved to ``path``.

    PCA-reduces to 50-d first (standard practice for high-dim features, bounds the
    t-SNE cost), then ``sklearn.manifold.TSNE`` (Barnes-Hut). ``color`` rows with
    ``-1`` (unknown) are dropped. Subsampled to ``max_points`` for the fit.

    If ``categories`` is given (a list of ``(name, color)`` pairs), ``color`` is taken
    as integer bucket ids in ``[0, len(categories))`` and the plot uses a discrete
    colormap with a per-bucket legend instead of a continuous colorbar.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    X = features.float().cpu().numpy()
    y = color.long().cpu().numpy()
    known = y >= 0
    X = X[known]
    y = y[known]
    if X.shape[0] < 2:
        return
    if X.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        sel = rng.choice(X.shape[0], size=max_points, replace=False)
        X = X[sel]
        y = y[sel]

    # PCA -> 50-d (or fewer) before t-SNE.
    pca_dim = min(50, X.shape[1], X.shape[0] - 1)
    if pca_dim > 2:
        X = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)
    emb = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        method="barnes_hut",
        random_state=seed,
    ).fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    if categories is not None:
        from matplotlib.colors import ListedColormap

        cmap = ListedColormap([c for _, c in categories])
        k = len(categories)
        ax.scatter(
            emb[:, 0],
            emb[:, 1],
            c=y,
            cmap=cmap,
            s=2,
            alpha=0.5,
            vmin=-0.5,
            vmax=k - 0.5,
        )
        proxies = [
            plt.Line2D([], [], marker="o", linestyle="", markersize=5, color=c, label=n)
            for n, c in categories
        ]
        ax.legend(handles=proxies, loc="best", fontsize=8, framealpha=0.7)
    else:
        scatter = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap="tab20", s=2, alpha=0.5)
        fig.colorbar(scatter, ax=ax, label=color_label)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def make_tsne_plots(
    collection: TsnePointCollection,
    tsne_dir: str | Path,
    *,
    seed: int = 0,
    subdir: str | None = None,
) -> list[str]:
    """Write the pdg_id / event_id t-SNE PNGs to ``tsne_dir``.

    Returns the list of written paths; skips a plot silently if its channel is empty.
    pdg_id collapses to the 7 calorimetry-role buckets (:func:`pdg_bucket`) with a
    discrete legend; event_id is a discrete per-event colormap. ``subdir`` writes to
    ``tsne_dir/subdir`` (keeps the up_cast(2) plots separate from the full-up-cast ones).
    """
    out_dir = Path(tsne_dir) if subdir is None else Path(tsne_dir) / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    if collection.features.shape[0] == 0:
        return paths

    bucket = torch.as_tensor(pdg_bucket(collection.pdg_id.cpu().numpy()))
    pdg_categories = [(name, PDG_BUCKET_COLORS[name]) for name in PDG_BUCKET_NAMES]
    n_events = max(int(collection.event_id.max().item()) + 1, 1) if collection.num_events else 1
    event_categories = [
        (f"event {i}", _EVENT_COLORS[i % len(_EVENT_COLORS)]) for i in range(n_events)
    ]
    specs = [
        (bucket, "tsne_pdg_id.png", "t-SNE colored by particle type", "particle type", pdg_categories),
        (collection.event_id, "tsne_event_id.png", "t-SNE colored by event id", "event id", event_categories),
    ]
    for color, name, title, label, categories in specs:
        path = out_dir / name
        tsne_plot(
            collection.features,
            color,
            path,
            title=title,
            color_label=label,
            seed=seed,
            categories=categories,
        )
        if path.exists():
            paths.append(str(path))
    return paths

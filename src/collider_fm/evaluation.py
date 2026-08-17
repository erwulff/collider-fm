"""Label-free representation-quality metrics for ColliderFM pretraining.

Focus: **collapse detection** on the per-point 288-d backbone features that a
downstream head would consume (the direct ``point.feat`` output, not mean-pooled).

All metrics are computed on held-out validation events with the (deterministic)
EMA teacher backbone. Augmentation stays on (DINO-style: phi-rotation / crop /
jitter / dropout) and masking is off by construction -- the harness never calls
the training ``forward``, so no mask tokens are inserted and ``mean_pool_features``
sees every point.

The headline is :func:`stable_rank` (floor-free: collapse -> ~1, healthy -> ~D).
Per-event retrieval / alignment / uniformity are a secondary invariance lens.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ._panda.structure import Point
from .sonata_model import mean_pool_features
from .training_loop import prototype_entropy
from .views import build_sonata_batch, move_sonata_batch_to_device

__all__ = [
    "EmbeddingCollection",
    "stable_rank",
    "alignment",
    "uniformity",
    "nn_retrieval",
    "collect_embeddings",
    "summarize",
]


# ---------------------------------------------------------------------------
# Pure metric functions
# ---------------------------------------------------------------------------


def stable_rank(features: torch.Tensor) -> tuple[float, list[float]]:
    """Stable rank ``(sum sigma)^2 / sum sigma^2`` plus the sorted spectrum.

    Computed on the raw (uncentered) ``[N, D]`` feature matrix via singular
    values. Bounded in ``[1, min(N, D)]``: a rank-1 (collapsed) matrix -> 1, an
    isotropic matrix -> ``min(N, D)`` (288 for the per-point backbone output).
    The full descending singular-value spectrum is returned alongside so a
    dominant mean component (one large ``sigma`` masking healthy variation) is
    visible.

    This is the nuclear-norm participation ratio ``(sum sigma)^2 / sum sigma^2``
    -- intentionally *not* Roy-Vetterli's entropy-based "effective rank"
    ``exp(-sum p_i log p_i)``.

    Args:
        features (torch.Tensor): Feature matrix of shape `[N, D]`.

    Returns:
        tuple[float, list[float]]: The stable rank (float) and the descending
        singular-value spectrum (list of floats).

    Raises:
        ValueError: If `features` is not 2-D.
    """
    if features.ndim != 2:
        raise ValueError(f"stable_rank expects 2D features, got {features.ndim}D")
    matrix = features.float()
    singular = torch.linalg.svdvals(matrix).clamp_min(0.0)
    denom = (singular * singular).sum().clamp_min(1e-12)
    rank = (singular.sum() ** 2) / denom
    return float(rank.item()), [float(v) for v in singular.tolist()]


def alignment(crop0: torch.Tensor, crop1: torch.Tensor) -> float:
    """Wang & Isola alignment: mean squared distance of positive pairs.

    ``mean ||f(x) - f(y)||^2`` over the paired crops, on the unit hypersphere.
    Lower is better -- augmented twins of the same event should map close
    together. Inputs are L2-normalized internally.

    Args:
        crop0 (torch.Tensor): First set of crop embeddings, shape `[N, D]`.
        crop1 (torch.Tensor): Second set of crop embeddings, shape `[N, D]`,
            positionally paired with `crop0`.

    Returns:
        float: The alignment metric (lower is better).
    """
    x = F.normalize(crop0.float(), dim=-1)
    y = F.normalize(crop1.float(), dim=-1)
    diff = x - y
    return float((diff * diff).sum(dim=-1).mean().item())


def uniformity(embeddings: torch.Tensor, t: float = 2.0) -> float:
    """Wang & Isola uniformity: ``log E[exp(-t ||x - y||^2)]`` over all pairs.

    On the unit hypersphere ``||x - y||^2 = 2 - 2 x.y``. Lower (more negative)
    is better: points spread uniformly across the sphere drive the exponent
    down. Collapse -> 0 (all pairs coincide).

    This uses the canonical Wang-Isola sign (``-t``). A ``+t`` variant reduces
    to a soft-max-of-diameter rather than a uniformity measure, so it is not
    used here. Inputs are L2-normalized internally.

    Args:
        embeddings (torch.Tensor): Embedding vectors of shape `[N, D]`.
        t (float, optional): Temperature parameter. Defaults to 2.0.

    Returns:
        float: The uniformity metric (lower is better; 0.0 for fewer than 2
        points).
    """
    x = F.normalize(embeddings.float(), dim=-1)
    n = x.shape[0]
    if n < 2:
        return 0.0
    gram = x @ x.T  # [n, n]
    total = 0.0
    chunk = 1024
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sqdist = (2.0 - 2.0 * gram[start:end]).clamp_min(0.0)  # [chunk, n]
        # exclude self-pairs on the diagonal block (exp(-inf) = 0).
        rows = torch.arange(start, end, device=x.device)
        sqdist[torch.arange(end - start), rows] = float("inf")
        total += torch.exp(-t * sqdist).sum().item()
    mean = total / (n * (n - 1))  # ordered distinct pairs
    return float(torch.log(torch.tensor(max(mean, 1e-300))).item())


def nn_retrieval(
    crop0: torch.Tensor,
    crop1: torch.Tensor,
    k_values: Sequence[int] = (1, 5),
    chunk: int = 1024,
) -> dict[str, float]:
    """Nearest-neighbor view-retrieval Recall@k (DINO-style instance retrieval).

    For each event's ``crop0`` (query), rank all ``crop1`` (pool) by cosine
    similarity; the correct match is that event's own ``crop1`` (its augmented
    twin). Returns ``r_at_1``, ``r_at_5``, ... Random baseline ~ k / N;
    perfect = 1.0. Inputs are L2-normalized internally.

    Args:
        crop0 (torch.Tensor): Query crop embeddings, shape `[N, D]`.
        crop1 (torch.Tensor): Pool crop embeddings, shape `[N, D]`, positionally
            paired with `crop0`.
        k_values (Sequence[int], optional): k values for Recall@k. Defaults to
            `(1, 5)`.
        chunk (int, optional): Row chunk size for the similarity computation.
            Defaults to 1024.

    Returns:
        dict[str, float]: Recall@k metrics keyed `r_at_{k}`. Returns all zeros
        if `N == 0`.
    """
    q = F.normalize(crop0.float(), dim=-1)
    pool = F.normalize(crop1.float(), dim=-1)
    n = q.shape[0]
    if n == 0:
        return {f"r_at_{k}": 0.0 for k in k_values}
    # k may exceed the pool size (small eval sets): clamp the topk width, and
    # any k >= n is trivially satisfied (the target is always in the top-n).
    topk_width = min(max(k_values), n)
    hits = {k: 0 for k in k_values}
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sim = q[start:end] @ pool.T  # [chunk, n]
        topk = sim.topk(topk_width, dim=1).indices  # [chunk, topk_width]
        targets = torch.arange(start, end, device=q.device).unsqueeze(1)
        for k in k_values:
            effective = min(k, n)
            hit = (topk[:, :effective] == targets).any(dim=1)
            hits[k] += int(hit.sum().item())
    return {f"r_at_{k}": hits[k] / n for k in k_values}


# ---------------------------------------------------------------------------
# Bounded embedding collection
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingCollection:
    """Bounded held-out embeddings from the teacher backbone.

    - ``crop0`` / ``crop1``: per-event L2-normalized pooled vectors ``[N, 288]``
      (the two augmented global crops), for retrieval / alignment / uniformity.
    - ``point_subsample``: raw per-point features ``[M, 288]`` (M <= budget),
      for stable rank.
    - ``prototype_bincount``: ``[num_prototypes]`` assignment counts over the
      same subsample, for prototype usage / entropy.
    """

    crop0: torch.Tensor
    crop1: torch.Tensor
    point_subsample: torch.Tensor
    prototype_bincount: torch.Tensor


def _estimate_num_batches(dataloader, max_events: int | None) -> int:
    """Best-effort batch count for proportional per-batch subsampling."""
    batch_size = getattr(dataloader, "batch_size", None) or 1
    try:
        n = len(dataloader)
    except (TypeError, NotImplementedError):
        n = None
    if n is None:
        if max_events is not None:
            return max(1, (max_events + batch_size - 1) // batch_size)
        return 1
    if max_events is not None:
        n = min(n, (max_events + batch_size - 1) // batch_size)
    return max(1, n)


@torch.no_grad()
def collect_embeddings(
    model,
    dataloader,
    device: torch.device,
    *,
    num_prototypes: int,
    batch_kwargs: dict,
    point_subsample_budget: int = 50000,
    max_events: int | None = None,
    seed: int = 0,
) -> EmbeddingCollection:
    """Collect bounded held-out embeddings from the teacher backbone.

    Per-point features are subsampled (never fully accumulated) to bound memory:
    only ``2N x 288`` pooled + ``<=budget x 288`` subsample + ``[num_prototypes]``
    bincount are stored. The subsample is drawn proportionally across batches so
    it is not biased toward early events.

    Args:
        model: The Sonata model with a teacher backbone and diagnostics head.
        dataloader: DataLoader yielding raw ColliderML event batches.
        device (torch.device): Compute device.
        num_prototypes (int): Number of prototype clusters in the diagnostics head.
        batch_kwargs (dict): Keyword arguments for `build_sonata_batch`.
        point_subsample_budget (int, optional): Maximum per-point features to
            collect. Defaults to 50000.
        max_events (int | None, optional): Maximum events to process. Defaults
            to None (all).
        seed (int, optional): RNG seed for subsampling. Defaults to 0.

    Returns:
        EmbeddingCollection: Bounded embeddings and prototype counts.
    """
    model.eval()
    head = model._head_for_diagnostics(use_teacher=True)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    crop0_parts: list[torch.Tensor] = []
    crop1_parts: list[torch.Tensor] = []
    subsample_parts: list[torch.Tensor] = []
    bincount = torch.zeros(num_prototypes, dtype=torch.long, device="cpu")

    num_batches = _estimate_num_batches(dataloader, max_events)
    events_collected = 0
    subsample_total = 0

    from tqdm import tqdm

    pbar = tqdm(dataloader, total=num_batches, desc="embeddings", unit="batch", mininterval=2.0)
    for batch_index, events in enumerate(pbar):
        if max_events is not None and events_collected >= max_events:
            break

        batch = build_sonata_batch(events, device=device, **batch_kwargs)
        batch = move_sonata_batch_to_device(batch, device)
        point = Point(
            feat=batch["global_feat"].float(),
            coord=batch["global_coord"].float(),
            origin_coord=batch["global_origin_coord"].float(),
            offset=batch["global_offset"].long(),
            grid_size=torch.as_tensor(batch["grid_size"], dtype=torch.float32, device=device),
            mask=torch.zeros(batch["global_coord"].shape[0], dtype=torch.bool, device=device),
        )
        point = model.teacher["backbone"](point)
        point = model.up_cast(point)
        point_features = point.feat  # [P, 288]

        # Per-event pooled (secondary lens): two global crops, event-major.
        pooled = mean_pool_features(point_features, point.offset)  # [2N, 288]
        n_events_batch = pooled.shape[0] // 2
        if max_events is not None:
            keep = max(0, max_events - events_collected)
            if keep < n_events_batch:
                n_events_batch = keep
                pooled = pooled[: n_events_batch * 2]
        pooled = F.normalize(pooled, dim=-1).view(n_events_batch, 2, -1)
        crop0_parts.append(pooled[:, 0].cpu())
        crop1_parts.append(pooled[:, 1].cpu())
        events_collected += n_events_batch

        # Per-point subsample (headline): proportional quota across batches.
        P = point_features.shape[0]
        remaining_budget = point_subsample_budget - subsample_total
        remaining_batches = max(1, num_batches - batch_index)
        if remaining_budget > 0 and P > 0:
            quota = max(1, int(round(remaining_budget / remaining_batches)))
            k = min(P, quota, remaining_budget)
            idx = torch.randperm(P, generator=generator)[:k].to(device)
            sampled = point_features[idx]  # [k, 288] on device
            logits = head(sampled.float())  # [k, num_prototypes]
            assigns = logits.argmax(dim=-1)
            bincount += torch.bincount(assigns, minlength=num_prototypes).cpu()
            subsample_parts.append(sampled.float().cpu())
            subsample_total += k

        pbar.set_postfix(events=events_collected, points=subsample_total, refresh=False)

        if max_events is not None and events_collected >= max_events and subsample_total >= point_subsample_budget:
            break
    pbar.close()

    return EmbeddingCollection(
        crop0=torch.cat(crop0_parts, dim=0) if crop0_parts else torch.empty(0, 0),
        crop1=torch.cat(crop1_parts, dim=0) if crop1_parts else torch.empty(0, 0),
        point_subsample=(torch.cat(subsample_parts, dim=0) if subsample_parts else torch.empty(0, 0)),
        prototype_bincount=bincount,
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarize(collection: EmbeddingCollection, *, dead_fraction: float = 0.1) -> dict[str, object]:
    """Compute the full metric suite from an `EmbeddingCollection`.

    Prototype health is reported three ways:
      * ``prototype_entropy`` -- ``-sum p log p`` (max = ``log(K)``).
      * ``prototype_effective_count`` -- ``exp(entropy)`` (perplexity): the
        threshold-free effective number of prototypes in use (uniform usage ->
        ``K``; total collapse onto one prototype -> ``1``).
      * ``num_dead_prototypes`` / ``num_active_prototypes`` -- a prototype is
        "dead" when its usage is below ``dead_fraction`` (default 10%) of the
        *uniform* rate ``1/K``. The threshold is relative to ``1/K`` rather
        than an absolute fraction of the pool, because with large K the uniform
        rate itself (e.g. 0.024% for K=4096) falls below any fixed fraction, so
        an absolute threshold would flag a perfectly healthy, uniform prototype
        space as entirely dead. This groups the whole statistically negligible
        tail (e.g. 0- or 1-point prototypes at this sample size).
      * ``num_empty_prototypes`` -- the strict, parameter-free count of
        prototypes that receive literally zero points. This is the threshold=0
        variant of "dead": objective and conservative, but sample-size noisy at
        the tail and blind to moribund prototypes with 1 point. Report it
        alongside the relative-threshold ``num_dead_prototypes``; the
        threshold-free prototype-health headline is
        ``prototype_effective_count``.

    Args:
        collection (EmbeddingCollection): Bounded embeddings from
            `collect_embeddings`.
        dead_fraction (float, optional): Fraction of the uniform rate
            `1/K` below which a prototype is "dead". Defaults to 0.1.

    Returns:
        dict[str, object]: The full metric suite (stable rank, alignment,
        uniformity, retrieval, prototype health).
    """
    metrics: dict[str, object] = {}

    rank, spectrum = stable_rank(collection.point_subsample)
    metrics["stable_rank"] = rank
    metrics["stable_rank_spectrum"] = spectrum
    metrics["stable_rank_dim"] = int(collection.point_subsample.shape[1])
    metrics["point_subsample_size"] = int(collection.point_subsample.shape[0])

    total = collection.prototype_bincount.sum().clamp_min(1)
    probs = collection.prototype_bincount.float() / total
    num_prototypes = int(collection.prototype_bincount.shape[0])
    entropy = prototype_entropy(probs)
    metrics["num_prototypes"] = num_prototypes
    metrics["prototype_entropy"] = entropy
    metrics["prototype_effective_count"] = float(math.exp(entropy))
    uniform_rate = 1.0 / num_prototypes
    dead_mask = probs < (dead_fraction * uniform_rate)
    num_dead = int(dead_mask.sum().item())
    metrics["num_dead_prototypes"] = num_dead
    metrics["num_active_prototypes"] = num_prototypes - num_dead
    metrics["num_empty_prototypes"] = int((collection.prototype_bincount == 0).sum().item())

    metrics["num_events"] = int(collection.crop0.shape[0])
    metrics["alignment"] = alignment(collection.crop0, collection.crop1)
    metrics["uniformity"] = uniformity(torch.cat([collection.crop0, collection.crop1], dim=0))
    metrics.update(nn_retrieval(collection.crop0, collection.crop1))

    return metrics

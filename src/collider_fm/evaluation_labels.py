"""Truth-label loading for the dominance report.

Loads the raw ColliderML ``calo_hits`` config (preserving ``contrib_particle_ids`` /
``contrib_energies`` that :class:`ColliderMLDataset` drops via ``select_columns``) and
derives the contributor / dominance distribution that characterizes the oracle
label's noisiness: shared calorimeter cells accumulate contributors from showering, so
a per-hit "which particle made this hit" label is only clean for the majority of hits
with a single dominant contributor.

This is a pure-data report -- it does not touch the backbone and is independent of the
checkpoint, so it documents the label-noise floor of the held-out subset rather than
model quality.

Schema (confirmed empirically, offline load): ``contrib_particle_ids`` is
``list<list<uint64>>`` and ``contrib_energies`` is ``list<list<float32>>`` -- one inner
list per cell, positionally paired (``contrib_energies[i][j]`` is the energy from
particle ``contrib_particle_ids[i][j]`` for cell ``i``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .data import resolve_colliderml_split


def load_calo_truth(
    split: str,
    dataset_name: str = "CERN/ColliderML-Release-1",
    dataset_type: str = "ttbar",
    pu_config: str = "pu0",
    cache_dir: str = "/mnt/ceph/users/ewulff/data/hf",
    dataset_revision: str | None = None,
    local_files_only: bool = False,
) -> Any:
    """Load the raw calo_hits config preserving the truth columns.

    This mirrors :class:`ColliderMLDataset`'s ``load_dataset`` call but skips
    ``select_columns``, so ``contrib_particle_ids`` / ``contrib_energies`` /
    ``event_id`` are exposed alongside ``x/y/z/total_energy``.
    """
    from datasets import DownloadConfig, load_dataset

    config_name = f"{dataset_type}_{pu_config}_calo_hits"
    resolved_split = resolve_colliderml_split(split)
    print(f"Loading raw {config_name} (truth columns)...")
    return load_dataset(
        dataset_name,
        config_name,
        split=resolved_split,
        cache_dir=cache_dir,
        revision=dataset_revision,
        download_config=DownloadConfig(local_files_only=local_files_only),
    )


def load_particle_pdg(
    split: str,
    dataset_name: str = "CERN/ColliderML-Release-1",
    dataset_type: str = "ttbar",
    pu_config: str = "pu0",
    cache_dir: str = "/mnt/ceph/users/ewulff/data/hf",
    dataset_revision: str | None = None,
    local_files_only: bool = False,
) -> dict[int, int]:
    """Build a ``particle_id -> pdg_id`` map from the sibling particles config.

    ``pdg_id`` (particle type) is not stored in the calo config, only the raw
    ``contrib_particle_ids``. Row-aligned to :func:`load_calo_truth` via the same split
    string (index-join by shared slice). Returns a flat dict spanning all events;
    ``particle_id`` values are globally unique within a run, so cross-event collisions
    are not an issue.
    """
    from datasets import DownloadConfig, load_dataset

    config_name = f"{dataset_type}_{pu_config}_particles"
    resolved_split = resolve_colliderml_split(split)
    print(f"Loading {config_name} (for pdg_id join)...")
    particles = load_dataset(
        dataset_name,
        config_name,
        split=resolved_split,
        cache_dir=cache_dir,
        revision=dataset_revision,
        download_config=DownloadConfig(local_files_only=local_files_only),
    )
    pid_to_pdg: dict[int, int] = {}
    for event in particles:
        for pid, pdg in zip(event["particle_id"], event["pdg_id"]):
            pid_to_pdg[int(pid)] = int(pdg)
    return pid_to_pdg


def dominant_particle(
    contrib_particle_ids: Sequence[Sequence[int]],
    contrib_energies: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-hit dominant particle via argmax energy.

    Returns three ``[num_hits]`` arrays:
    - ``dominant_particle_id``: the particle_id with the largest contribution energy.
    - ``dominant_energy_fraction``: that particle's energy / total cell energy (0..1).
    - ``contributor_count``: number of contributing particles (``len(contrib_*)``).

    Hits with zero contributors map to ``particle_id = -1`` and ``fraction = 0``.
    """
    n_hits = len(contrib_particle_ids)
    dom_pid = np.full(n_hits, -1, dtype=np.int64)
    dom_frac = np.zeros(n_hits, dtype=np.float32)
    counts = np.zeros(n_hits, dtype=np.int64)
    for i in range(n_hits):
        ids = np.asarray(contrib_particle_ids[i], dtype=np.int64)
        ens = np.asarray(contrib_energies[i], dtype=np.float64)
        counts[i] = ids.shape[0]
        if ids.shape[0] == 0:
            continue
        total = ens.sum()
        if total <= 0:
            dom_pid[i] = int(ids[0])
            continue
        j = int(np.argmax(ens))
        dom_pid[i] = int(ids[j])
        dom_frac[i] = float(ens[j] / total)
    return dom_pid, dom_frac, counts


def event_dominant_particles(
    event: Mapping[str, Any],
    pid_to_pdg: Mapping[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dominant labels for one raw calo event.

    Returns ``(dominant_particle_id, pdg_id, dominant_energy_fraction,
    contributor_count)`` per cell, all ``[num_hits]``. ``pdg_id`` is filled from
    ``pid_to_pdg`` (``-1`` when the dominant particle has no known pdg_id).
    """
    dom_pid, dom_frac, counts = dominant_particle(
        event["contrib_particle_ids"], event["contrib_energies"]
    )
    if pid_to_pdg is None:
        pdg = np.full(dom_pid.shape, -1, dtype=np.int64)
    else:
        pdg = np.array(
            [pid_to_pdg.get(int(p), -1) for p in dom_pid], dtype=np.int64
        )
    return dom_pid, pdg, dom_frac, counts


# Coarse calorimetry-role buckets for collapsing pdg_id to a readable t-SNE legend.
# Each common long-lived type maps to one of 7 named buckets; rare / unlisted codes
# (resonances, heavy-flavor hadrons, short-lived states) fall to "other". Nuclei are
# matched by magnitude (the 10-digit PDG ion codes, >= 1e9). Built from the val-split
# pdg frequency, where e± / γ / π± / π0 / K / p / n / μ / nuclei cover ~99% of records.
PDG_BUCKETS: list[tuple[str, frozenset[int]]] = [
    ("e±", frozenset({11, -11})),
    ("γ", frozenset({22})),
    ("μ±", frozenset({13, -13})),
    ("charged hadron", frozenset({211, -211, 321, -321, 2212, -2212})),
    ("neutral hadron", frozenset({111, 2112, -2112, 130, 310, 311, -311, 3122, -3122})),
    ("nucleus", frozenset()),
    ("other", frozenset()),
]
PDG_BUCKET_NAMES: list[str] = [name for name, _ in PDG_BUCKETS]
_PDG_TO_BUCKET: dict[int, int] = {
    pdg: idx for idx, (_, pdgs) in enumerate(PDG_BUCKETS) for pdg in pdgs
}
_NUCLEUS_INDEX = PDG_BUCKET_NAMES.index("nucleus")
_OTHER_INDEX = PDG_BUCKET_NAMES.index("other")
_NUCLEUS_THRESHOLD = 1_000_000_000  # 10-digit PDG ion codes: 10NZZZAAA...


def pdg_bucket(pdg_id: Any) -> np.ndarray:
    """Map raw ``pdg_id`` values to coarse calorimetry-role bucket indices.

    Returns an ``int64`` array (same shape as ``pdg_id``) of bucket indices into
    :data:`PDG_BUCKET_NAMES` (0=e±, 1=γ, 2=μ±, 3=charged hadron, 4=neutral hadron,
    5=nucleus, 6=other). The unknown sentinel ``-1`` (no pdg / missing pid) maps to
    ``-1`` so callers that drop ``-1`` treat it as "no label". Nuclei are matched by
    magnitude (``abs(pdg) >= 1e9``); everything else unmatched falls to "other".
    """
    arr = np.asarray(pdg_id)
    flat = arr.ravel().astype(np.int64, copy=False)
    out = np.full(flat.shape, _OTHER_INDEX, dtype=np.int64)
    out[np.abs(flat) >= _NUCLEUS_THRESHOLD] = _NUCLEUS_INDEX
    for pdg, idx in _PDG_TO_BUCKET.items():
        out[flat == pdg] = idx
    out[flat == -1] = -1
    return out.reshape(arr.shape)


def dominance_report(
    contributor_counts: np.ndarray, dominant_fractions: np.ndarray
) -> dict[str, Any]:
    """Summarize the contributor/dominance distributions for one eval subset.

    Computed from the arrays :func:`dominant_particle` builds; logged once per run to
    characterize the oracle label's noisiness (shared calorimeter cells have many
    contributors from showering).
    """
    counts = np.asarray(contributor_counts)
    frac = np.asarray(dominant_fractions, dtype=np.float64)
    shared = counts > 1
    report: dict[str, Any] = {
        "num_hits": int(counts.shape[0]),
        "contributor_count_mean": float(counts.mean()),
        "contributor_count_median": float(np.median(counts)),
        "contributor_count_p99": float(np.percentile(counts, 99)),
        "contributor_count_max": int(counts.max()),
        "pct_single_contributor": float((counts == 1).mean() * 100.0),
        "pct_shared": float(shared.mean() * 100.0),
    }
    report["dominant_frac_median"] = float(np.median(frac))
    report["dominant_frac_mean"] = float(frac.mean())
    report["pct_dominant_ge_0.9"] = float((frac >= 0.9).mean() * 100.0)
    report["pct_dominant_ge_0.5"] = float((frac >= 0.5).mean() * 100.0)
    if int(shared.sum()) > 0:
        f_shared = frac[shared]
        report["shared_num_hits"] = int(shared.sum())
        report["shared_dominant_frac_median"] = float(np.median(f_shared))
        report["shared_dominant_frac_mean"] = float(f_shared.mean())
        report["shared_pct_dominant_ge_0.9"] = float((f_shared >= 0.9).mean() * 100.0)
        report["shared_pct_dominant_ge_0.5"] = float((f_shared >= 0.5).mean() * 100.0)
    return report


def format_dominance_report(report: Mapping[str, Any]) -> str:
    """Render a dominance report dict as human-readable plain text.

    Mirrors the console block printed by ``scripts/evaluate.py``: a header, the
    contributor-count summary, the single/shared split, the dominant-fraction
    distribution, and (when present) the shared-only sub-stats. Written to
    ``runs/eval_<run>/dominance_report.txt`` alongside the JSON summary.
    """
    def _pct(v: Any) -> str:
        return f"{float(v):.1f}%" if v is not None else "n/a"

    def _f(v: Any, p: int = 3) -> str:
        return f"{float(v):.{p}f}" if v is not None else "n/a"

    lines = [
        "Dominance report (label-noise floor)",
        "=" * 40,
        f"events scanned : {report.get('num_events_scanned')}",
        f"hits           : {report.get('num_hits')}",
        "",
        "contributor count",
        f"  mean   : {_f(report.get('contributor_count_mean'), 2)}",
        f"  median : {report.get('contributor_count_median')}",
        f"  p99    : {_f(report.get('contributor_count_p99'), 0)}",
        f"  max    : {report.get('contributor_count_max')}",
        "",
        "single vs shared cells",
        f"  single contributor : {_pct(report.get('pct_single_contributor'))}",
        f"  shared (>=2)        : {_pct(report.get('pct_shared'))}",
        "",
        "dominant energy fraction",
        f"  mean   : {_f(report.get('dominant_frac_mean'))}",
        f"  median : {_f(report.get('dominant_frac_median'))}",
        f"  >=0.9  : {_pct(report.get('pct_dominant_ge_0.9'))}",
        f"  >=0.5  : {_pct(report.get('pct_dominant_ge_0.5'))}",
    ]
    if "shared_num_hits" in report:
        lines += [
            "",
            f"shared-only ({report.get('shared_num_hits')} hits)",
            f"  frac median : {_f(report.get('shared_dominant_frac_median'))}",
            f"  frac mean   : {_f(report.get('shared_dominant_frac_mean'))}",
            f"  >=0.9       : {_pct(report.get('shared_pct_dominant_ge_0.9'))}",
            f"  >=0.5       : {_pct(report.get('shared_pct_dominant_ge_0.5'))}",
        ]
    return "\n".join(lines) + "\n"


def compute_dominance_report(
    calo_truth_dataset: Any, max_events: int | None = None
) -> tuple[dict[str, Any], int]:
    """Compute the dominance report over up to ``max_events`` raw calo events.

    Walks the dataset, derives per-hit ``(contributor_count, dominant_fraction)`` via
    :func:`dominant_particle`, concatenates across events, and returns the
    :func:`dominance_report` output plus the number of events scanned.
    """
    counts_parts: list[np.ndarray] = []
    frac_parts: list[np.ndarray] = []
    n = 0
    for event in calo_truth_dataset:
        if max_events is not None and n >= max_events:
            break
        _, frac, counts = dominant_particle(
            event["contrib_particle_ids"], event["contrib_energies"]
        )
        counts_parts.append(counts)
        frac_parts.append(frac)
        n += 1
    counts = (
        np.concatenate(counts_parts) if counts_parts else np.array([], dtype=np.int64)
    )
    frac = (
        np.concatenate(frac_parts)
        if frac_parts
        else np.array([], dtype=np.float32)
    )
    return dominance_report(counts, frac), n

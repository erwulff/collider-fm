from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .views import SSL_VIEW_ORDER


def summarize_ssl_view(view: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize one structured SSL view for logging or diagnostics."""
    modality_id = torch.as_tensor(view["modality_id"], dtype=torch.long)
    visible_point_mask = torch.as_tensor(view["visible_point_mask"], dtype=torch.bool)
    selected_point_count = int(torch.as_tensor(view["point_id"], dtype=torch.long).numel())
    tracker_point_count = int((modality_id == 0).sum().item())
    calo_point_count = int((modality_id == 1).sum().item())
    visible_point_count = int(visible_point_mask.sum().item())
    candidate_point_count = int(visible_point_mask.numel())
    hidden_point_count = candidate_point_count - visible_point_count

    return {
        "view_type": str(view["view_type"]),
        "num_events": int(torch.as_tensor(view["offset"], dtype=torch.long).numel()),
        "selected_point_count": selected_point_count,
        "visible_point_count": visible_point_count,
        "hidden_point_count": hidden_point_count,
        "tracker_point_count": tracker_point_count,
        "calo_point_count": calo_point_count,
        "tracker_fraction": tracker_point_count / max(selected_point_count, 1),
        "calo_fraction": calo_point_count / max(selected_point_count, 1),
        "visible_fraction": visible_point_count / max(candidate_point_count, 1),
    }


def summarize_ssl_view_set(view_set: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Aggregate structured SSL view summaries by view family."""

    def average(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    summary: dict[str, Any] = {}
    for view_type in SSL_VIEW_ORDER:
        views = list(view_set.get(view_type, []))
        view_summaries = [summarize_ssl_view(view) for view in views]
        summary[view_type] = {
            "num_views": len(view_summaries),
            "views": view_summaries,
            "mean_selected_point_count": average([entry["selected_point_count"] for entry in view_summaries]),
            "mean_visible_fraction": average([entry["visible_fraction"] for entry in view_summaries]),
            "mean_tracker_fraction": average([entry["tracker_fraction"] for entry in view_summaries]),
            "mean_calo_fraction": average([entry["calo_fraction"] for entry in view_summaries]),
        }
    return summary

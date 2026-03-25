from __future__ import annotations

from typing import Any

__all__ = [
    "ColliderMLDataset",
    "MultimodalPointBatch",
    "build_model_inputs",
    "build_multimodal_points",
    "create_small_multimodal_model",
    "CaloStem",
    "DistillationOutputs",
    "MultimodalPandaSSL",
    "ModalityFusion",
    "SSLViewConfig",
    "TrackerStem",
    "build_ssl_views",
]


def __getattr__(name: str) -> Any:
    if name == "ColliderMLDataset":
        from .data import ColliderMLDataset

        return ColliderMLDataset
    if name in {"MultimodalPointBatch", "build_model_inputs", "build_multimodal_points"}:
        from .features import MultimodalPointBatch, build_model_inputs, build_multimodal_points

        exports = {
            "MultimodalPointBatch": MultimodalPointBatch,
            "build_model_inputs": build_model_inputs,
            "build_multimodal_points": build_multimodal_points,
        }
        return exports[name]
    if name in {"CaloStem", "ModalityFusion", "TrackerStem"}:
        from .stems import CaloStem, ModalityFusion, TrackerStem

        exports = {
            "CaloStem": CaloStem,
            "ModalityFusion": ModalityFusion,
            "TrackerStem": TrackerStem,
        }
        return exports[name]
    if name in {"SSLViewConfig", "build_ssl_views"}:
        from .views import SSLViewConfig, build_ssl_views

        exports = {
            "SSLViewConfig": SSLViewConfig,
            "build_ssl_views": build_ssl_views,
        }
        return exports[name]
    if name in {"DistillationOutputs", "MultimodalPandaSSL", "create_small_multimodal_model"}:
        from .model import DistillationOutputs, MultimodalPandaSSL, create_small_multimodal_model

        exports = {
            "DistillationOutputs": DistillationOutputs,
            "MultimodalPandaSSL": MultimodalPandaSSL,
            "create_small_multimodal_model": create_small_multimodal_model,
        }
        return exports[name]
    raise AttributeError(f"module 'collider_fm' has no attribute {name!r}")

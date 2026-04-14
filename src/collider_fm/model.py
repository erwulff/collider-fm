from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .sonata_model import SonataSelfDistillation
from .views import POINT_FEATURE_DIM

SMALL_SONATA_MODEL_BACKBONE_KWARGS = {
    "enc_depths": (1, 1, 1, 1, 1),
    "enc_channels": (8, 12, 16, 24, 32),
    "enc_num_head": (1, 1, 2, 4, 4),
    "enc_patch_size": (4, 4, 4, 4, 4),
    "shuffle_orders": False,
    "enable_flash": False,
    "flash_backend": "torch",
    "upcast_attention": False,
    "upcast_softmax": False,
    "enable_rpe": False,
    "traceable": True,
    "enc_mode": True,
    "mask_token": True,
}

TRAINING_SONATA_MODEL_BACKBONE_KWARGS = {
    "enc_depths": (1, 1, 2, 2, 1),
    "enc_channels": (16, 32, 64, 96, 128),
    "enc_num_head": (1, 2, 4, 4, 8),
    "enc_patch_size": (8, 8, 8, 8, 8),
    "shuffle_orders": False,
    "enable_flash": False,
    "flash_backend": "torch",
    "upcast_attention": False,
    "upcast_softmax": False,
    "enable_rpe": False,
    "traceable": True,
    "enc_mode": True,
    "mask_token": True,
}


def create_small_sonata_model(
    device: torch.device | None = None,
    backbone_kwargs: Mapping[str, Any] | None = None,
    **model_kwargs: Any,
) -> SonataSelfDistillation:
    """Construct the compact Sonata model shared by smoke tests and diagnostics."""

    resolved_backbone_kwargs = dict(SMALL_SONATA_MODEL_BACKBONE_KWARGS)
    if backbone_kwargs is not None:
        resolved_backbone_kwargs.update(dict(backbone_kwargs))

    resolved_model_kwargs = {
        "in_channels": POINT_FEATURE_DIM,
        "grid_size": 0.002,
        "head_embed_channels": 64,
        "head_num_prototypes": 32,
        "num_global_view": 2,
        "num_local_view": 4,
        "mask_size_start": 0.02,
        "mask_size_base": 0.15,
        "mask_jitter_base": 0.01,
        "match_max_r": 0.004,
    }
    resolved_model_kwargs.update(model_kwargs)

    model = SonataSelfDistillation(
        backbone_kwargs=resolved_backbone_kwargs,
        **resolved_model_kwargs,
    )
    if device is not None:
        model = model.to(device)
    return model


def create_training_sonata_model(
    device: torch.device | None = None,
    backbone_kwargs: Mapping[str, Any] | None = None,
    **model_kwargs: Any,
) -> SonataSelfDistillation:
    """Construct the Sonata model for the main training loop."""

    resolved_backbone_kwargs = dict(TRAINING_SONATA_MODEL_BACKBONE_KWARGS)
    if backbone_kwargs is not None:
        resolved_backbone_kwargs.update(dict(backbone_kwargs))

    resolved_model_kwargs = {
        "in_channels": POINT_FEATURE_DIM,
        "grid_size": 0.002,
        "head_embed_channels": 512,
        "head_num_prototypes": 4096,
        "num_global_view": 2,
        "num_local_view": 4,
        "mask_size_start": 0.02,
        "mask_size_base": 0.15,
        "mask_jitter_base": 0.01,
        "match_max_r": 0.004,
    }
    resolved_model_kwargs.update(model_kwargs)

    model = SonataSelfDistillation(
        backbone_kwargs=resolved_backbone_kwargs,
        **resolved_model_kwargs,
    )
    if device is not None:
        model = model.to(device)
    return model


def create_small_model(
    device: torch.device | None = None,
    backbone_kwargs: Mapping[str, Any] | None = None,
    **model_kwargs: Any,
) -> SonataSelfDistillation:
    """Construct the compact Sonata model (smoke tests, diagnostics)."""
    return create_small_sonata_model(
        device=device,
        backbone_kwargs=backbone_kwargs,
        **model_kwargs,
    )


def create_training_model(
    device: torch.device | None = None,
    backbone_kwargs: Mapping[str, Any] | None = None,
    **model_kwargs: Any,
) -> SonataSelfDistillation:
    """Construct the Sonata model for training."""
    return create_training_sonata_model(
        device=device,
        backbone_kwargs=backbone_kwargs,
        **model_kwargs,
    )

from __future__ import annotations

"""Model and loss helpers for the calo-only masked distillation pipeline."""

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._panda.model_base import PointTransformerV3
from ._panda.structure import Point
from .sonata_model import SonataSelfDistillation
from .views import DEFAULT_POINT_GRID_SIZE, POINT_FEATURE_DIM, PointView

SMALL_MODEL_BACKBONE_KWARGS = {
    "enc_depths": (1, 1, 1, 1, 1),
    "enc_channels": (8, 12, 16, 24, 32),
    "enc_num_head": (1, 1, 2, 4, 4),
    "enc_patch_size": (4, 4, 4, 4, 4),
    "shuffle_orders": False,
    "enable_flash": False,
    "flash_backend": "torch",
}

TRAINING_MODEL_BACKBONE_KWARGS = {
    "enc_depths": (1, 1, 2, 2, 1),
    "enc_channels": (16, 32, 64, 96, 128),
    "enc_num_head": (1, 2, 4, 4, 8),
    "enc_patch_size": (8, 8, 8, 8, 8),
    "shuffle_orders": False,
    "enable_flash": False,
    "flash_backend": "torch",
}

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


def as_point_cloud(
    view: Mapping[str, Any], default_grid_size: float = DEFAULT_POINT_GRID_SIZE
) -> Point:
    """Convert a validated point-view mapping into Panda's `Point` structure."""
    data = dict(view)

    if "coord" not in data or "feat" not in data:
        raise KeyError("Each view must provide 'coord' and 'feat' tensors.")

    coord = torch.as_tensor(data["coord"], dtype=torch.float32)
    feat = torch.as_tensor(data["feat"], dtype=torch.float32, device=coord.device)
    if coord.ndim != 2 or coord.shape[1] != 3:
        raise ValueError("'coord' must have shape [num_points, 3].")
    if feat.ndim != 2:
        raise ValueError("'feat' must have shape [num_points, num_features].")
    if feat.shape[0] != coord.shape[0]:
        raise ValueError("'coord' and 'feat' must describe the same number of points.")

    data["coord"] = coord
    data["feat"] = feat

    if "offset" in data:
        offset = torch.as_tensor(
            data["offset"], dtype=torch.long, device=coord.device
        ).flatten()
        if offset.numel() == 0:
            raise ValueError("'offset' must contain at least one event boundary.")
        if offset[-1].item() != coord.shape[0]:
            raise ValueError("The final offset must equal the number of points.")
        counts = torch.diff(offset, prepend=offset.new_zeros(1))
        if torch.any(counts <= 0):
            raise ValueError("'offset' must be a strictly increasing cumulative count.")
        data["offset"] = offset
    elif "batch" in data:
        batch = torch.as_tensor(
            data["batch"], dtype=torch.long, device=coord.device
        ).flatten()
        if batch.shape[0] != coord.shape[0]:
            raise ValueError("'batch' must contain one assignment per point.")
        data["batch"] = batch
    else:
        data["offset"] = torch.tensor(
            [coord.shape[0]], dtype=torch.long, device=coord.device
        )

    grid_size = data.get("grid_size", default_grid_size)
    data["grid_size"] = torch.as_tensor(
        grid_size, dtype=coord.dtype, device=coord.device
    )
    return Point(data)


def mean_pool_features(feat: torch.Tensor, offset: torch.Tensor | None) -> torch.Tensor:
    """Pool per-point features into one embedding per event."""
    if feat.ndim != 2:
        raise ValueError("'feat' must have shape [num_points, channels].")

    if offset is None:
        return feat.mean(dim=0, keepdim=True)

    offset = torch.as_tensor(offset, dtype=torch.long, device=feat.device).flatten()
    if offset.numel() == 0:
        raise ValueError("'offset' must contain at least one event boundary.")
    if offset[-1].item() != feat.shape[0]:
        raise ValueError("The final offset must equal the number of points.")

    counts = torch.diff(offset, prepend=offset.new_zeros(1))
    if torch.any(counts <= 0):
        raise ValueError("'offset' must be a strictly increasing cumulative count.")

    batch_index = torch.arange(offset.numel(), device=feat.device).repeat_interleave(
        counts
    )
    pooled = feat.new_zeros((offset.numel(), feat.shape[1]))
    pooled.scatter_add_(0, batch_index.unsqueeze(1).expand(-1, feat.shape[1]), feat)
    return pooled / counts.unsqueeze(1)


def gather_by_offset(values: torch.Tensor, offset: torch.Tensor) -> list[torch.Tensor]:
    """Split a batched point tensor into one tensor per event."""
    offset = torch.as_tensor(offset, dtype=torch.long, device=values.device).flatten()
    starts = torch.cat([offset.new_zeros(1), offset[:-1]], dim=0)
    return [values[start:end] for start, end in zip(starts.tolist(), offset.tolist())]


def masked_mean_pool(
    point_projection: torch.Tensor, mask: torch.Tensor, offset: torch.Tensor
) -> torch.Tensor:
    """Pool masked points per event, falling back to all points when no mask is active."""
    chunks = gather_by_offset(point_projection, offset)
    mask_chunks = gather_by_offset(mask.to(dtype=torch.bool), offset)
    pooled = []
    for event_projection, event_mask in zip(chunks, mask_chunks):
        if event_projection.shape[0] == 0:
            pooled.append(point_projection.new_zeros((1, point_projection.shape[1])))
            continue
        if torch.any(event_mask):
            pooled.append(event_projection[event_mask].mean(dim=0, keepdim=True))
        else:
            pooled.append(event_projection.mean(dim=0, keepdim=True))
    return torch.cat(pooled, dim=0)


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class PandaEncoderBackbone(nn.Module):
    """Small adapter around PTv3.

    By default we ask PTv3 to upcast the deep features back to the original point
    resolution. That gives us one learned embedding per input point, which keeps the
    point-level loss easy to explain and inspect.
    """

    def __init__(self, upcast: bool = True, **backbone_kwargs: Any) -> None:
        super().__init__()
        self.upcast = upcast
        self.backbone = PointTransformerV3(**backbone_kwargs)

    def forward(self, point: Any) -> Any:
        return self.backbone(point, upcast=self.upcast)


class IdentityBackbone(nn.Module):
    """Tiny backbone used in tests.

    It simply projects features with one linear layer and returns them at the same
    point resolution, which makes the point-level training code easy to unit test.
    """

    def __init__(
        self, in_channels: int = POINT_FEATURE_DIM, output_dim: int = 8, **_: Any
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.proj = nn.Linear(in_channels, output_dim, bias=False)

    def forward(self, point: Any) -> Any:
        point = as_point_cloud(point)
        encoded: Any = SimpleNamespace()
        encoded.feat = self.proj(point.feat)
        encoded.offset = point.offset
        return encoded


class PandaSelfDistillation(nn.Module):
    model_recipe = "legacy"

    """Small, readable Panda-style student-teacher model for calo point clouds.

    The current training objective is point-level. Teacher views produce prototype
    targets for the unmasked points, and student views try to match those targets on
    the same `source_index` points. We also compute a masked-point summary per event
    so the model still learns from the points hidden by the student mask.
    """

    def __init__(
        self,
        in_channels: int = POINT_FEATURE_DIM,
        embed_channels: int = 64,
        num_prototypes: int = 1024,
        projection_dim: int = 256,
        prediction_dim: int = 256,
        temp_student: float = 0.1,
        temp_teacher: float = 0.04,
        center_momentum: float = 0.9,
        grid_size: float = DEFAULT_POINT_GRID_SIZE,
        backbone_cls: type[nn.Module] = PandaEncoderBackbone,
        backbone_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()

        # Keep the PTv3 kwargs readable and close to the project-level concepts:
        # a small hierarchical point encoder, traceable pooling, and point-level
        # upcasting back to the original input resolution.
        default_backbone_kwargs = {
            "in_channels": in_channels,
            "order": ("z", "z-trans"),
            "stride": (2, 2, 2, 2),
            "enc_depths": (2, 2, 2, 6, 2),
            "enc_channels": (
                embed_channels,
                embed_channels * 2,
                embed_channels * 4,
                embed_channels * 8,
                embed_channels * 16,
            ),
            "enc_num_head": (
                max(1, embed_channels // 16),
                max(1, embed_channels // 8),
                max(1, embed_channels // 4),
                max(1, embed_channels // 2),
                max(1, embed_channels),
            ),
            "enc_patch_size": (48, 48, 48, 48, 48),
            "enc_mode": True,
            "enable_flash": True,
            "mask_token": True,
            "traceable": True,
        }
        if backbone_kwargs is not None:
            default_backbone_kwargs.update(dict(backbone_kwargs))

        uses_multiscale_upcast = bool(
            default_backbone_kwargs.get("enc_mode", False)
        ) and bool(default_backbone_kwargs.get("traceable", False))
        backbone_dim = int(
            default_backbone_kwargs.get(
                "output_dim",
                sum(default_backbone_kwargs["enc_channels"])
                if uses_multiscale_upcast
                else default_backbone_kwargs["enc_channels"][-1],
            )
        )
        self.grid_size = grid_size
        self.temp_student = temp_student
        self.temp_teacher = temp_teacher
        self.center_momentum = center_momentum
        self.num_prototypes = num_prototypes
        self.flash_attention_enabled = bool(
            default_backbone_kwargs.get("enable_flash", False)
        )
        self.flash_attention_backend = (
            str(default_backbone_kwargs.get("flash_backend", "flash_attn"))
            if self.flash_attention_enabled
            else "disabled"
        )

        self.student_backbone = backbone_cls(**default_backbone_kwargs)
        self.teacher_backbone = backbone_cls(**default_backbone_kwargs)

        self.student_projector = _build_mlp(backbone_dim, backbone_dim, projection_dim)
        self.teacher_projector = _build_mlp(backbone_dim, backbone_dim, projection_dim)
        self.student_predictor = _build_mlp(
            projection_dim, prediction_dim, projection_dim
        )
        self.prototype_head = nn.Linear(projection_dim, num_prototypes, bias=False)

        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_projector.load_state_dict(self.student_projector.state_dict())
        for module in (self.teacher_backbone, self.teacher_projector):
            for parameter in module.parameters():
                parameter.requires_grad = False

        self.register_buffer("center", torch.zeros(1, num_prototypes))

    def normalize_prototypes(self) -> None:
        """Keep prototype weights on a unit sphere to reduce collapse."""
        with torch.no_grad():
            weight = self.prototype_head.weight.data
            self.prototype_head.weight.data = F.normalize(weight, dim=1)

    def _encode_view(
        self,
        view: Mapping[str, Any] | PointView,
        backbone: nn.Module,
        projector: nn.Module,
        predictor: nn.Module | None = None,
    ) -> dict[str, Any]:
        """Encode one teacher or student view into point and pooled outputs."""

        point = as_point_cloud(view, default_grid_size=self.grid_size)
        encoded = backbone(point)
        point_features = encoded.feat
        point_projection = projector(point_features)
        if predictor is not None:
            point_projection = predictor(point_projection)
        point_projection = F.normalize(point_projection, dim=-1)
        point_logits = self.prototype_head(point_projection)

        mask = torch.as_tensor(
            view.get(
                "mask",
                torch.zeros(point_features.shape[0], device=point_features.device),
            ),
            dtype=torch.bool,
            device=point_features.device,
        )
        pooled = mean_pool_features(point_features, point.offset)
        masked_pooled_projection = masked_mean_pool(
            point_projection, mask, point.offset
        )
        masked_logits = self.prototype_head(masked_pooled_projection)
        return {
            "point_features": point_features,
            "point_projection": point_projection,
            "point_logits": point_logits,
            "pooled": pooled,
            "masked_pooled_projection": masked_pooled_projection,
            "masked_logits": masked_logits,
            "offset": point.offset,
            "source_index": torch.as_tensor(
                view["source_index"], dtype=torch.long, device=point_features.device
            ),
            "mask": mask,
            "view_kind": str(view.get("view_kind", "unknown")),
        }

    def encode_student_view(
        self, view: Mapping[str, Any] | PointView
    ) -> dict[str, Any]:
        return self._encode_view(
            view, self.student_backbone, self.student_projector, self.student_predictor
        )

    def encode_teacher_view(
        self, view: Mapping[str, Any] | PointView
    ) -> dict[str, Any]:
        return self._encode_view(view, self.teacher_backbone, self.teacher_projector)

    def forward(
        self, batch: Mapping[str, Sequence[Mapping[str, Any] | PointView]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Encode all student and teacher views from one distillation batch."""

        student_views = list(batch["student_views"])
        teacher_views = list(batch["teacher_views"])
        if len(student_views) < 1 or len(teacher_views) < 1:
            raise ValueError(
                "Training requires at least one student view and one teacher view."
            )

        student_outputs = [self.encode_student_view(view) for view in student_views]
        with torch.no_grad():
            teacher_outputs = [self.encode_teacher_view(view) for view in teacher_views]
        return student_outputs, teacher_outputs

    def distillation_loss(
        self,
        student_outputs: Sequence[dict[str, Any]],
        teacher_outputs: Sequence[dict[str, Any]],
    ) -> torch.Tensor:
        """Average point-level and masked pooled losses across all view pairs."""

        if not student_outputs or not teacher_outputs:
            raise ValueError("Student and teacher outputs must both be non-empty.")

        total_loss = student_outputs[0]["point_logits"].new_tensor(0.0)
        num_terms = 0

        for student_output in student_outputs:
            for teacher_output in teacher_outputs:
                total_loss = total_loss + pointwise_panda_loss(
                    student_logits=student_output["point_logits"],
                    teacher_logits=teacher_output["point_logits"],
                    student_source_index=student_output["source_index"],
                    teacher_source_index=teacher_output["source_index"],
                    student_mask=student_output["mask"],
                    center=cast(torch.Tensor, self.center),
                    temp_s=self.temp_student,
                    temp_t=self.temp_teacher,
                )
                num_terms += 1

                total_loss = total_loss + panda_loss(
                    student_output["masked_logits"],
                    teacher_output["masked_logits"].detach(),
                    cast(torch.Tensor, self.center),
                    self.temp_student,
                    self.temp_teacher,
                )
                num_terms += 1

        if num_terms == 0:
            raise ValueError("Need at least one student/teacher term for distillation.")
        return total_loss / num_terms

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        """Apply the EMA student-to-teacher update for one optimization step."""

        if not 0.0 <= momentum <= 1.0:
            raise ValueError("Momentum must be between 0 and 1.")

        for student_param, teacher_param in zip(
            self.student_backbone.parameters(), self.teacher_backbone.parameters()
        ):
            teacher_param.data.mul_(momentum).add_(
                student_param.data, alpha=1.0 - momentum
            )
        for student_param, teacher_param in zip(
            self.student_projector.parameters(), self.teacher_projector.parameters()
        ):
            teacher_param.data.mul_(momentum).add_(
                student_param.data, alpha=1.0 - momentum
            )
        self.normalize_prototypes()

    @torch.no_grad()
    def update_center(self, teacher_outputs: Sequence[dict[str, Any]]) -> None:
        """Update the running teacher-logit center used for stabilization."""

        if not teacher_outputs:
            raise ValueError("Teacher outputs must be non-empty.")

        teacher_logits = torch.cat(
            [output["point_logits"] for output in teacher_outputs], dim=0
        )
        batch_center = teacher_logits.mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(
            batch_center, alpha=1.0 - self.center_momentum
        )


def create_small_panda_model(
    device: torch.device | None = None,
    backbone_cls: type[nn.Module] = PandaEncoderBackbone,
    backbone_kwargs: Mapping[str, Any] | None = None,
    **model_kwargs: Any,
) -> PandaSelfDistillation:
    """Construct the compact model shared by smoke tests and diagnostics."""
    resolved_backbone_kwargs = dict(SMALL_MODEL_BACKBONE_KWARGS)
    if backbone_kwargs is not None:
        resolved_backbone_kwargs.update(dict(backbone_kwargs))

    resolved_model_kwargs = {
        "in_channels": POINT_FEATURE_DIM,
        "embed_channels": 8,
        "num_prototypes": 32,
        "projection_dim": 8,
        "prediction_dim": 16,
    }
    resolved_model_kwargs.update(model_kwargs)

    model = PandaSelfDistillation(
        backbone_cls=backbone_cls,
        backbone_kwargs=resolved_backbone_kwargs,
        **resolved_model_kwargs,
    )
    model.normalize_prototypes()
    if device is not None:
        model = model.to(device)
    return model


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
        "head_embed_channels": 64,
        "head_num_prototypes": 32,
        "num_global_view": 2,
        "num_local_view": 4,
        "mask_size_start": 100.0,
        "mask_size_base": 750.0,
        "mask_jitter_base": 50.0,
        "match_max_r": 20.0,
    }
    resolved_model_kwargs.update(model_kwargs)

    model = SonataSelfDistillation(
        backbone_kwargs=resolved_backbone_kwargs,
        **resolved_model_kwargs,
    )
    if device is not None:
        model = model.to(device)
    return model


def create_training_panda_model(
    device: torch.device | None = None,
    backbone_cls: type[nn.Module] = PandaEncoderBackbone,
    backbone_kwargs: Mapping[str, Any] | None = None,
    **model_kwargs: Any,
) -> PandaSelfDistillation:
    """Construct a slightly larger model for the real training loop."""
    resolved_backbone_kwargs = dict(TRAINING_MODEL_BACKBONE_KWARGS)
    if backbone_kwargs is not None:
        resolved_backbone_kwargs.update(dict(backbone_kwargs))

    resolved_model_kwargs = {
        "in_channels": POINT_FEATURE_DIM,
        "embed_channels": 16,
        "num_prototypes": 256,
        "projection_dim": 48,
        "prediction_dim": 64,
    }
    resolved_model_kwargs.update(model_kwargs)

    model = PandaSelfDistillation(
        backbone_cls=backbone_cls,
        backbone_kwargs=resolved_backbone_kwargs,
        **resolved_model_kwargs,
    )
    model.normalize_prototypes()
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
        "head_embed_channels": 256,
        "head_num_prototypes": 256,
        "num_global_view": 2,
        "num_local_view": 4,
        "mask_size_start": 100.0,
        "mask_size_base": 750.0,
        "mask_jitter_base": 50.0,
        "match_max_r": 20.0,
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
    recipe: str,
    device: torch.device | None = None,
    backbone_cls: type[nn.Module] = PandaEncoderBackbone,
    backbone_kwargs: Mapping[str, Any] | None = None,
    **model_kwargs: Any,
) -> nn.Module:
    if recipe == "legacy":
        return create_small_panda_model(
            device=device,
            backbone_cls=backbone_cls,
            backbone_kwargs=backbone_kwargs,
            **model_kwargs,
        )
    if recipe == "sonata":
        return create_small_sonata_model(
            device=device,
            backbone_kwargs=backbone_kwargs,
            **model_kwargs,
        )
    raise ValueError(f"Unsupported model recipe: {recipe}.")


def create_training_model(
    recipe: str,
    device: torch.device | None = None,
    backbone_cls: type[nn.Module] = PandaEncoderBackbone,
    backbone_kwargs: Mapping[str, Any] | None = None,
    **model_kwargs: Any,
) -> nn.Module:
    if recipe == "legacy":
        return create_training_panda_model(
            device=device,
            backbone_cls=backbone_cls,
            backbone_kwargs=backbone_kwargs,
            **model_kwargs,
        )
    if recipe == "sonata":
        return create_training_sonata_model(
            device=device,
            backbone_kwargs=backbone_kwargs,
            **model_kwargs,
        )
    raise ValueError(f"Unsupported model recipe: {recipe}.")


def panda_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    center: torch.Tensor,
    temp_s: float,
    temp_t: float,
) -> torch.Tensor:
    """Cross-entropy between centered teacher probabilities and student logits."""
    student_logits = student_logits.float()
    teacher_logits = teacher_logits.float()
    center = center.float()
    teacher_probs = F.softmax((teacher_logits - center) / temp_t, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temp_s, dim=-1)
    return -(teacher_probs * student_log_probs).sum(dim=-1).mean()


def pointwise_panda_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_source_index: torch.Tensor,
    teacher_source_index: torch.Tensor,
    student_mask: torch.Tensor,
    center: torch.Tensor,
    temp_s: float,
    temp_t: float,
) -> torch.Tensor:
    """Compute point-level distillation on shared source indices.

    We keep this implementation intentionally small and explicit: teacher and student
    points are aligned by `source_index`, and only shared points contribute to the
    point-level loss. Student masks simply restrict the student side to masked points
    when they exist; otherwise we fall back to all shared points.
    """
    if teacher_source_index.numel() == 0:
        return student_logits.new_tensor(0.0)

    candidate_rows = torch.nonzero(student_mask, as_tuple=False).flatten()
    if candidate_rows.numel() == 0:
        candidate_rows = torch.arange(
            student_source_index.shape[0], device=student_source_index.device
        )
    if candidate_rows.numel() == 0:
        return student_logits.new_tensor(0.0)

    candidate_source_index = student_source_index[candidate_rows]
    sorted_teacher_index, teacher_order = torch.sort(teacher_source_index)
    teacher_positions = (
        torch.searchsorted(sorted_teacher_index, candidate_source_index, right=True) - 1
    )
    valid_positions = teacher_positions >= 0
    if not torch.any(valid_positions):
        return student_logits.new_tensor(0.0)

    safe_teacher_positions = teacher_positions.clamp_min(0)
    matched_values = sorted_teacher_index[safe_teacher_positions]
    valid_positions = valid_positions & (matched_values == candidate_source_index)

    if not torch.any(valid_positions):
        return student_logits.new_tensor(0.0)

    matched_student_rows = candidate_rows[valid_positions]
    matched_teacher_rows = teacher_order[safe_teacher_positions[valid_positions]]
    matched_student = student_logits[matched_student_rows]
    matched_teacher = teacher_logits[matched_teacher_rows].detach()
    return panda_loss(matched_student, matched_teacher, center, temp_s, temp_t)

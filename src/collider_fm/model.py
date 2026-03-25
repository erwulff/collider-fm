from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._panda.model_base import PointTransformerV3
from ._panda.structure import Point
from .views import DEFAULT_POINT_GRID_SIZE, PointView


SMALL_MODEL_BACKBONE_KWARGS = {
    "enc_depths": (1, 1, 1, 1, 1),
    "enc_channels": (8, 12, 16, 24, 32),
    "enc_num_head": (1, 1, 2, 4, 4),
    "enc_patch_size": (4, 4, 4, 4, 4),
    "shuffle_orders": False,
    "enable_flash": False,
}


def as_point_cloud(view: Mapping[str, Any], default_grid_size: float = DEFAULT_POINT_GRID_SIZE) -> Point:
    """Convert a generic view mapping into the Panda point-cloud structure."""
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
        offset = torch.as_tensor(data["offset"], dtype=torch.long, device=coord.device).flatten()
        if offset.numel() == 0:
            raise ValueError("'offset' must contain at least one event boundary.")
        if offset[-1].item() != coord.shape[0]:
            raise ValueError("The final offset must equal the number of points.")
        counts = torch.diff(offset, prepend=offset.new_zeros(1))
        if torch.any(counts <= 0):
            raise ValueError("'offset' must be a strictly increasing cumulative count.")
        data["offset"] = offset
    elif "batch" in data:
        batch = torch.as_tensor(data["batch"], dtype=torch.long, device=coord.device).flatten()
        if batch.shape[0] != coord.shape[0]:
            raise ValueError("'batch' must contain one assignment per point.")
        data["batch"] = batch
    else:
        data["offset"] = torch.tensor([coord.shape[0]], dtype=torch.long, device=coord.device)

    grid_size = data.get("grid_size", default_grid_size)
    data["grid_size"] = torch.as_tensor(grid_size, dtype=coord.dtype, device=coord.device)
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

    batch_index = torch.arange(offset.numel(), device=feat.device).repeat_interleave(counts)
    pooled = feat.new_zeros((offset.numel(), feat.shape[1]))
    pooled.scatter_add_(0, batch_index.unsqueeze(1).expand(-1, feat.shape[1]), feat)
    return pooled / counts.unsqueeze(1)


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class PandaEncoderBackbone(nn.Module):
    """Adapter that exposes PTv3 encoder features without the multiscale upcast path."""

    def __init__(self, **backbone_kwargs: Any) -> None:
        super().__init__()
        self.backbone = PointTransformerV3(**backbone_kwargs)

    def forward(self, point: Any) -> Any:
        return self.backbone(point, upcast=False)


class PandaSelfDistillation(nn.Module):
    """Simplified Panda-style self-distillation model for ColliderML point clouds."""

    def __init__(
        self,
        in_channels: int = 6,
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
            "enable_flash": False,
        }
        if backbone_kwargs is not None:
            default_backbone_kwargs.update(dict(backbone_kwargs))

        backbone_dim = default_backbone_kwargs["enc_channels"][-1]
        self.grid_size = grid_size
        self.temp_student = temp_student
        self.temp_teacher = temp_teacher
        self.center_momentum = center_momentum

        self.student_backbone = backbone_cls(**default_backbone_kwargs)
        self.teacher_backbone = backbone_cls(**default_backbone_kwargs)

        self.student_projector = _build_mlp(backbone_dim, backbone_dim, projection_dim)
        self.teacher_projector = _build_mlp(backbone_dim, backbone_dim, projection_dim)
        self.student_predictor = _build_mlp(projection_dim, prediction_dim, projection_dim)
        self.prototype_head = nn.Linear(projection_dim, num_prototypes, bias=False)

        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_projector.load_state_dict(self.student_projector.state_dict())
        for module in (self.teacher_backbone, self.teacher_projector):
            for parameter in module.parameters():
                parameter.requires_grad = False

        self.register_buffer("center", torch.zeros(1, num_prototypes))

    def _encode_view(
        self,
        view: Mapping[str, Any] | PointView,
        backbone: nn.Module,
        projector: nn.Module,
        predictor: nn.Module | None = None,
    ) -> torch.Tensor:
        point = as_point_cloud(view, default_grid_size=self.grid_size)
        encoded = backbone(point)
        pooled = mean_pool_features(encoded.feat, getattr(encoded, "offset", None))
        projection = projector(pooled)
        if predictor is not None:
            projection = predictor(projection)
        projection = F.normalize(projection, dim=-1)
        return self.prototype_head(projection)

    def forward(self, views: Sequence[Mapping[str, Any] | PointView]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if len(views) < 2:
            raise ValueError("Self-distillation requires at least two augmented views.")

        student_outputs = [self._encode_view(view, self.student_backbone, self.student_projector, self.student_predictor) for view in views]

        with torch.no_grad():
            teacher_outputs = [self._encode_view(view, self.teacher_backbone, self.teacher_projector) for view in views[:2]]

        return student_outputs, teacher_outputs

    def distillation_loss(
        self,
        student_outputs: Sequence[torch.Tensor],
        teacher_outputs: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if not student_outputs or not teacher_outputs:
            raise ValueError("Student and teacher outputs must both be non-empty.")

        total_loss = student_outputs[0].new_tensor(0.0)
        num_terms = 0
        for teacher_index, teacher_logits in enumerate(teacher_outputs):
            for student_index, student_logits in enumerate(student_outputs):
                if student_index == teacher_index:
                    continue
                total_loss = total_loss + panda_loss(
                    student_logits,
                    teacher_logits.detach(),
                    self.center,
                    self.temp_student,
                    self.temp_teacher,
                )
                num_terms += 1

        if num_terms == 0:
            raise ValueError("Need at least one non-matching student/teacher pair for distillation.")
        return total_loss / num_terms

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("Momentum must be between 0 and 1.")

        for student_param, teacher_param in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)
        for student_param, teacher_param in zip(self.student_projector.parameters(), self.teacher_projector.parameters()):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)

    @torch.no_grad()
    def update_center(self, teacher_outputs: Sequence[torch.Tensor]) -> None:
        if not teacher_outputs:
            raise ValueError("Teacher outputs must be non-empty.")

        batch_center = torch.cat(list(teacher_outputs), dim=0).mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1.0 - self.center_momentum)


def create_small_panda_model(
    device: torch.device | None = None,
    backbone_cls: type[nn.Module] = PandaEncoderBackbone,
    backbone_kwargs: Mapping[str, Any] | None = None,
) -> PandaSelfDistillation:
    """Construct the compact Panda-style model shared by train, smoke, and diagnostics scripts."""
    resolved_backbone_kwargs = dict(SMALL_MODEL_BACKBONE_KWARGS)
    if backbone_kwargs is not None:
        resolved_backbone_kwargs.update(dict(backbone_kwargs))

    model = PandaSelfDistillation(
        in_channels=6,
        embed_channels=8,
        num_prototypes=32,
        projection_dim=8,
        prediction_dim=16,
        backbone_cls=backbone_cls,
        backbone_kwargs=resolved_backbone_kwargs,
    )
    if device is not None:
        model = model.to(device)
    return model


def panda_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    center: torch.Tensor,
    temp_s: float,
    temp_t: float,
) -> torch.Tensor:
    """Cross-entropy between centered teacher probabilities and student logits."""
    teacher_probs = F.softmax((teacher_logits - center) / temp_t, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temp_s, dim=-1)
    return -(teacher_probs * student_log_probs).sum(dim=-1).mean()

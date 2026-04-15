from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from itertools import chain
from typing import Any

from packaging import version
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from timm.models.layers import trunc_normal_

from ._panda.model_base import PointTransformerV3
from ._panda.structure import Point
from ._panda.utils import batch2offset, offset2batch, offset2bincount
from .views import DEFAULT_POINT_GRID_SIZE, POINT_FEATURE_DIM


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


class CosineScheduler:
    def __init__(
        self,
        base_value: float,
        final_value: float,
        total_iters: int,
        start_value: float = 0.0,
        warmup_iters: int = 0,
        freeze_value: float | None = None,
        freeze_iters: int = 0,
    ) -> None:
        if total_iters <= 0:
            raise ValueError("'total_iters' must be positive.")
        self.base_value = base_value
        self.final_value = final_value
        self.total_iters = total_iters

        warmup_schedule = torch.linspace(start_value, base_value, warmup_iters).tolist()
        if freeze_value is None:
            freeze_value = final_value
        freeze_schedule = [freeze_value] * freeze_iters

        schedule_length = total_iters - warmup_iters - freeze_iters
        if schedule_length > 0:
            iters = torch.arange(schedule_length, dtype=torch.float32)
            denom = max(1, schedule_length - 1)
            schedule = (
                final_value
                + 0.5
                * (base_value - final_value)
                * (1 + torch.cos(torch.pi * iters / denom))
            ).tolist()
        else:
            schedule = []
        self.schedule = warmup_schedule + schedule + freeze_schedule
        self.iter = 0

    def get(self, iteration: int) -> float:
        if iteration >= self.total_iters:
            return float(self.final_value)
        return float(self.schedule[iteration])

    def step(self) -> float:
        value = self.get(self.iter)
        self.iter += 1
        return value


class SonataBackbone(nn.Module):
    """PTv3 wrapper that leaves Sonata in charge of multiscale up-casting."""

    def __init__(self, **backbone_kwargs: Any) -> None:
        super().__init__()
        self.backbone = PointTransformerV3(**backbone_kwargs)

    def forward(self, point: Point) -> Point:
        return self.backbone(point, upcast=False)


class OnlineCluster(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 4096,
        embed_channels: int = 512,
        num_prototypes: int = 4096,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, embed_channels),
        )
        self.apply(self._init_weights)
        if version.parse(torch.__version__) >= version.parse("2.1.0"):
            self.prototype = torch.nn.utils.parametrizations.weight_norm(
                nn.Linear(embed_channels, num_prototypes, bias=False)
            )
            self.prototype.parametrizations.weight.original0.data.fill_(1)
            self.prototype.parametrizations.weight.original0.requires_grad = False
        else:
            self.prototype = torch.nn.utils.weight_norm(
                nn.Linear(embed_channels, num_prototypes, bias=False)
            )
            self.prototype.weight_g.data.fill_(1)
            self.prototype.weight_g.requires_grad = False

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def embed(self, feat: torch.Tensor) -> torch.Tensor:
        feat = self.mlp(feat)
        eps = 1e-6 if feat.dtype == torch.float16 else 1e-12
        return F.normalize(feat, dim=-1, p=2, eps=eps)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.prototype(self.embed(feat))


def mean_pool_features(feat: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    counts = torch.diff(offset, prepend=offset.new_zeros(1))
    batch_index = torch.arange(offset.numel(), device=feat.device).repeat_interleave(
        counts
    )
    pooled = feat.new_zeros((offset.numel(), feat.shape[1]))
    pooled.scatter_add_(0, batch_index.unsqueeze(1).expand(-1, feat.shape[1]), feat)
    return pooled / counts.unsqueeze(1)


def gather_by_offset(values: torch.Tensor, offset: torch.Tensor) -> list[torch.Tensor]:
    starts = torch.cat([offset.new_zeros(1), offset[:-1]], dim=0)
    return [values[start:end] for start, end in zip(starts.tolist(), offset.tolist())]


def masked_mean_pool(
    point_projection: torch.Tensor, mask: torch.Tensor, offset: torch.Tensor
) -> torch.Tensor:
    chunks = gather_by_offset(point_projection, offset)
    mask_chunks = gather_by_offset(mask.to(dtype=torch.bool), offset)
    pooled = []
    for event_projection, event_mask in zip(chunks, mask_chunks):
        if event_projection.shape[0] == 0:
            pooled.append(point_projection.new_zeros((1, point_projection.shape[1])))
        elif torch.any(event_mask):
            pooled.append(event_projection[event_mask].mean(dim=0, keepdim=True))
        else:
            pooled.append(event_projection.mean(dim=0, keepdim=True))
    return torch.cat(pooled, dim=0)


class SonataSelfDistillation(nn.Module):
    model_recipe = "sonata"

    def __init__(
        self,
        in_channels: int = POINT_FEATURE_DIM,
        grid_size: float = 0.002,
        backbone_kwargs: Mapping[str, Any] | None = None,
        head_in_channels: int | None = None,
        head_hidden_channels: int = 4096,
        head_embed_channels: int = 512,
        head_num_prototypes: int = 4096,
        teacher_custom: Mapping[str, Any] | None = None,
        num_global_view: int = 2,
        num_local_view: int = 4,
        mask_size_start: float = 0.02,
        mask_size_base: float = 0.15,
        mask_size_warmup_ratio: float = 0.05,
        mask_ratio_start: float = 0.5,
        mask_ratio_base: float = 0.9,
        mask_ratio_warmup_ratio: float = 0.05,
        mask_jitter: float | None = None,
        mask_jitter_start: float = 0.0,
        mask_jitter_base: float = 0.01,
        mask_jitter_warmup_ratio: float = 0.05,
        teacher_temp_start: float = 0.04,
        teacher_temp_base: float = 0.07,
        teacher_temp_warmup_ratio: float = 0.05,
        student_temp: float = 0.1,
        mask_loss_weight: float = 2 / 8,
        roll_mask_loss_weight: float = 2 / 8,
        unmask_loss_weight: float = 4 / 8,
        momentum_base: float = 0.994,
        momentum_final: float = 1.0,
        match_max_k: int = 1,
        match_max_r: float = 0.004,
        up_cast_level: int = 2,
    ) -> None:
        super().__init__()
        if backbone_kwargs is None:
            backbone_kwargs = {}
        if teacher_custom is None:
            teacher_custom = {}

        resolved_backbone_kwargs = {
            "in_channels": in_channels,
            "order": ("z", "z-trans"),
            "stride": (2, 2, 2, 2),
            "enc_depths": (1, 1, 2, 2, 1),
            "enc_channels": (16, 32, 64, 96, 128),
            "enc_num_head": (1, 2, 4, 4, 8),
            "enc_patch_size": (8, 8, 8, 8, 8),
            "shuffle_orders": False,
            "enable_flash": False,
            "flash_backend": "flash_attn",
            "upcast_attention": False,
            "upcast_softmax": False,
            "enable_rpe": False,
            "traceable": True,
            "enc_mode": True,
            "mask_token": True,
        }
        resolved_backbone_kwargs.update(dict(backbone_kwargs))
        teacher_backbone_kwargs = dict(resolved_backbone_kwargs)
        teacher_backbone_kwargs.update(dict(teacher_custom))

        if head_in_channels is None:
            enc_channels = tuple(
                int(channel) for channel in resolved_backbone_kwargs["enc_channels"]
            )
            head_in_channels = sum(enc_channels[-(up_cast_level + 1) :])

        self.grid_size = float(grid_size)
        self.mask_loss_weight = float(mask_loss_weight)
        self.roll_mask_loss_weight = float(roll_mask_loss_weight)
        self.unmask_loss_weight = float(unmask_loss_weight)
        self.num_global_view = int(num_global_view)
        self.num_local_view = int(num_local_view)
        self.mask_size = float(mask_size_start)
        self.mask_size_start = float(mask_size_start)
        self.mask_size_base = float(mask_size_base)
        self.mask_size_warmup_ratio = float(mask_size_warmup_ratio)
        self.mask_ratio = float(mask_ratio_start)
        self.mask_ratio_start = float(mask_ratio_start)
        self.mask_ratio_base = float(mask_ratio_base)
        self.mask_ratio_warmup_ratio = float(mask_ratio_warmup_ratio)
        self.mask_jitter = mask_jitter
        self.mask_jitter_start = float(mask_jitter_start)
        self.mask_jitter_base = float(mask_jitter_base)
        self.mask_jitter_warmup_ratio = float(mask_jitter_warmup_ratio)
        self.teacher_temp = float(teacher_temp_start)
        self.teacher_temp_start = float(teacher_temp_start)
        self.teacher_temp_base = float(teacher_temp_base)
        self.teacher_temp_warmup_ratio = float(teacher_temp_warmup_ratio)
        self.student_temp = float(student_temp)
        self.momentum = float(momentum_base)
        self.momentum_base = float(momentum_base)
        self.momentum_final = float(momentum_final)
        self.match_max_k = int(match_max_k)
        self.match_max_r = float(match_max_r)
        self.up_cast_level = int(up_cast_level)
        self.flash_attention_enabled = bool(
            resolved_backbone_kwargs.get("enable_flash", False)
        )
        self.flash_attention_backend = (
            str(resolved_backbone_kwargs.get("flash_backend", "flash_attn"))
            if self.flash_attention_enabled
            else "disabled"
        )
        self.num_prototypes = int(head_num_prototypes)
        self.last_monitoring_state: dict[str, Any] = {
            "student_logits": None,
            "point_features": None,
            "masked_fraction": 0.0,
        }

        assert (
            self.unmask_loss_weight + self.mask_loss_weight + self.roll_mask_loss_weight
            > 0
        )
        assert self.num_global_view > 1 or self.roll_mask_loss_weight == 0
        assert self.num_global_view in {1, 2}

        student_model_dict: dict[str, nn.Module] = {}
        teacher_model_dict: dict[str, nn.Module] = {}
        student_model_dict["backbone"] = SonataBackbone(**resolved_backbone_kwargs)
        teacher_model_dict["backbone"] = SonataBackbone(**teacher_backbone_kwargs)

        head_factory = partial(
            OnlineCluster,
            in_channels=head_in_channels,
            hidden_channels=head_hidden_channels,
            embed_channels=head_embed_channels,
            num_prototypes=head_num_prototypes,
        )
        if self.mask_loss_weight > 0 or self.roll_mask_loss_weight > 0:
            student_model_dict["mask_head"] = head_factory()
            teacher_model_dict["mask_head"] = head_factory()
        if self.unmask_loss_weight > 0:
            student_model_dict["unmask_head"] = head_factory()
            teacher_model_dict["unmask_head"] = head_factory()

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)
        for key in self.student:
            self.teacher[key].load_state_dict(self.student[key].state_dict())
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False

        self.mask_size_scheduler: CosineScheduler | None = None
        self.mask_ratio_scheduler: CosineScheduler | None = None
        self.teacher_temp_scheduler: CosineScheduler | None = None
        self.momentum_scheduler: CosineScheduler | None = None
        self.mask_jitter_scheduler: CosineScheduler | None = None

    def setup_schedulers(self, total_steps: int, current_step: int = 0) -> None:
        self.mask_size_scheduler = CosineScheduler(
            start_value=self.mask_size_start,
            base_value=self.mask_size_base,
            final_value=self.mask_size_base,
            warmup_iters=int(total_steps * self.mask_size_warmup_ratio),
            total_iters=total_steps,
        )
        self.mask_size_scheduler.iter = current_step

        self.mask_ratio_scheduler = CosineScheduler(
            start_value=self.mask_ratio_start,
            base_value=self.mask_ratio_base,
            final_value=self.mask_ratio_base,
            warmup_iters=int(total_steps * self.mask_ratio_warmup_ratio),
            total_iters=total_steps,
        )
        self.mask_ratio_scheduler.iter = current_step

        self.teacher_temp_scheduler = CosineScheduler(
            start_value=self.teacher_temp_start,
            base_value=self.teacher_temp_base,
            final_value=self.teacher_temp_base,
            warmup_iters=int(total_steps * self.teacher_temp_warmup_ratio),
            total_iters=total_steps,
        )
        self.teacher_temp_scheduler.iter = current_step

        self.momentum_scheduler = CosineScheduler(
            base_value=self.momentum_base,
            final_value=self.momentum_final,
            total_iters=total_steps,
        )
        self.momentum_scheduler.iter = current_step

        if self.mask_jitter is None:
            self.mask_jitter_scheduler = CosineScheduler(
                start_value=self.mask_jitter_start,
                base_value=self.mask_jitter_start,
                final_value=self.mask_jitter_base,
                warmup_iters=int(total_steps * self.mask_jitter_warmup_ratio),
                total_iters=total_steps,
            )
            self.mask_jitter_scheduler.iter = current_step

    def step_schedules(self) -> dict[str, float]:
        if (
            self.mask_size_scheduler is None
            or self.mask_ratio_scheduler is None
            or self.teacher_temp_scheduler is None
            or self.momentum_scheduler is None
        ):
            raise RuntimeError(
                "Sonata schedulers are not configured. Call setup_schedulers() first."
            )

        self.mask_size = self.mask_size_scheduler.step()
        self.mask_ratio = self.mask_ratio_scheduler.step()
        self.teacher_temp = self.teacher_temp_scheduler.step()
        self.momentum = self.momentum_scheduler.step()
        if self.mask_jitter_scheduler is not None:
            self.mask_jitter = self.mask_jitter_scheduler.step()
        return {
            "mask_size": self.mask_size,
            "mask_ratio": self.mask_ratio,
            "teacher_temperature": self.teacher_temp,
            "teacher_momentum": self.momentum,
            "mask_jitter": float(self.mask_jitter or 0.0),
        }

    @torch.no_grad()
    def update_teacher(self, momentum: float | None = None) -> None:
        if momentum is not None:
            self.momentum = float(momentum)
        for student_param, teacher_param in zip(
            self.student.parameters(), self.teacher.parameters()
        ):
            teacher_param.data.mul_(self.momentum).add_(
                student_param.data, alpha=1 - self.momentum
            )

    @staticmethod
    def sinkhorn_knopp(
        feat: torch.Tensor, temp: float, num_iter: int = 3
    ) -> torch.Tensor:
        if feat.shape[0] == 0:
            return feat.new_zeros((0, feat.shape[1]))

        feat = feat.float()
        q = torch.exp(feat / temp).t()
        k = q.shape[0]
        n_local = q.shape[1]
        sum_q_local = q.sum()
        if get_world_size() > 1:
            scalars = torch.stack([q.new_tensor(float(n_local)), sum_q_local])
            dist.all_reduce(scalars)
            n, sum_q = scalars[0], scalars[1]
        else:
            n, sum_q = q.new_tensor(float(n_local)), sum_q_local
        q = q / sum_q.clamp_min(1e-12)

        for _ in range(num_iter):
            q_row_sum = q.sum(dim=1, keepdim=True)
            if get_world_size() > 1:
                dist.all_reduce(q_row_sum)
            q = q / q_row_sum.clamp_min(1e-12) / k
            q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-12) / n
        q *= n
        return q.t()

    def generate_mask(
        self, coord: torch.Tensor, offset: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = offset2batch(offset)
        min_coord = torch_scatter.segment_coo(coord, batch, reduce="min")
        grid_coord = ((coord - min_coord[batch]) // self.mask_size).int()
        grid_coord = torch.cat([batch.unsqueeze(-1), grid_coord], dim=-1)
        unique, point_cluster, _ = torch.unique(
            grid_coord, dim=0, sorted=True, return_inverse=True, return_counts=True
        )
        patch_num = unique.shape[0]
        mask_patch_num = int(patch_num * self.mask_ratio)
        if patch_num == 0 or mask_patch_num == 0:
            return (
                torch.zeros(coord.shape[0], dtype=torch.bool, device=coord.device),
                point_cluster,
            )
        patch_index = torch.randperm(patch_num, device=coord.device)
        mask_patch_index = patch_index[:mask_patch_num]
        point_mask = torch.isin(point_cluster, mask_patch_index)
        return point_mask, point_cluster

    @torch.no_grad()
    def match_neighbour(
        self,
        view1_coord: torch.Tensor,
        view1_offset: torch.Tensor,
        view2_coord: torch.Tensor,
        view2_offset: torch.Tensor,
    ) -> torch.Tensor:
        view1_starts = torch.cat([view1_offset.new_zeros(1), view1_offset[:-1]], dim=0)
        view2_starts = torch.cat([view2_offset.new_zeros(1), view2_offset[:-1]], dim=0)
        if view1_offset.shape[0] != view2_offset.shape[0]:
            raise ValueError("Neighbour matching requires aligned batch boundaries.")
        matched_indices = []
        for view1_start, view1_end, view2_start, view2_end in zip(
            view1_starts.tolist(),
            view1_offset.tolist(),
            view2_starts.tolist(),
            view2_offset.tolist(),
        ):
            query_coord = view1_coord[view1_start:view1_end]
            reference_coord = view2_coord[view2_start:view2_end]
            if query_coord.shape[0] == 0 or reference_coord.shape[0] == 0:
                continue
            distance = torch.cdist(query_coord.float(), reference_coord.float())
            min_distance, index2 = distance.min(dim=1)
            index1 = torch.arange(
                view1_start,
                view1_end,
                device=view1_coord.device,
                dtype=torch.long,
            )
            index2 = index2.to(dtype=torch.long) + view2_start
            batch_index = torch.stack([index1, index2], dim=1)
            matched_indices.append(batch_index[min_distance < self.match_max_r])

        if not matched_indices:
            return torch.empty((0, 2), dtype=torch.long, device=view1_coord.device)
        return torch.cat(matched_indices, dim=0)

    @torch.no_grad()
    def roll_point(self, point: Point) -> Point:
        counts = offset2bincount(point.offset).tolist()
        bs = len(point.offset) // self.num_global_view
        data_dict: dict[str, torch.Tensor] = {}
        for key in point.keys():
            if key not in {"feat", "coord", "origin_coord", "batch"}:
                continue
            value = point[key].split(counts)
            value = chain(
                *[
                    value[self.num_global_view * b : self.num_global_view * (b + 1)][
                        ::-1
                    ]
                    for b in range(bs)
                ]
            )
            value_list = list(value)
            if key == "batch":
                value_list = [
                    torch.ones_like(chunk) * i for i, chunk in enumerate(value_list)
                ]
            data_dict[key] = torch.cat(value_list, dim=0)
        return Point(data_dict)

    def up_cast(self, point: Point) -> Point:
        for _ in range(self.up_cast_level):
            if (
                "pooling_parent" not in point.keys()
                or "pooling_inverse" not in point.keys()
            ):
                raise KeyError("Sonata up-cast requires traceable PTv3 features.")
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        return point

    def _grid_size_value(self, data_dict: Mapping[str, Any]) -> torch.Tensor:
        grid_size = data_dict["grid_size"]
        device = next(self.parameters()).device
        grid_size_tensor = torch.as_tensor(
            grid_size, dtype=torch.float32, device=device
        )
        if grid_size_tensor.ndim == 0:
            return grid_size_tensor
        return grid_size_tensor.flatten()[0]

    def _head_for_diagnostics(self, use_teacher: bool) -> OnlineCluster:
        module_dict = self.teacher if use_teacher else self.student
        if "unmask_head" in module_dict:
            return module_dict["unmask_head"]
        return module_dict["mask_head"]

    @torch.no_grad()
    def encode_view(
        self, view: Mapping[str, Any], use_teacher: bool
    ) -> dict[str, torch.Tensor]:
        point = Point(
            feat=torch.as_tensor(
                view["feat"], dtype=torch.float32, device=view["coord"].device
            ),
            coord=torch.as_tensor(
                view["coord"], dtype=torch.float32, device=view["coord"].device
            ),
            origin_coord=torch.as_tensor(
                view.get("origin_coord", view["coord"]),
                dtype=torch.float32,
                device=view["coord"].device,
            ),
            offset=torch.as_tensor(
                view["offset"], dtype=torch.long, device=view["coord"].device
            ),
            grid_size=torch.as_tensor(
                view.get("grid_size", self.grid_size),
                dtype=torch.float32,
                device=view["coord"].device,
            ),
            mask=torch.as_tensor(
                view.get(
                    "mask", torch.zeros(len(view["coord"]), device=view["coord"].device)
                ),
                dtype=torch.bool,
                device=view["coord"].device,
            ),
        )
        backbone_module = (
            self.teacher["backbone"] if use_teacher else self.student["backbone"]
        )
        head_module = self._head_for_diagnostics(use_teacher)
        point = backbone_module(point)
        point = self.up_cast(point)
        point_features = point.feat
        point_logits = head_module(point_features)
        pooled = mean_pool_features(point_features, point.offset)
        mask = (
            point.mask
            if "mask" in point.keys()
            else torch.zeros(
                point.feat.shape[0], dtype=torch.bool, device=point.feat.device
            )
        )
        masked_pooled_projection = masked_mean_pool(point_features, mask, point.offset)
        masked_logits = head_module(masked_pooled_projection)
        return {
            "point_features": point_features,
            "point_projection": point_features,
            "point_logits": point_logits,
            "pooled": pooled,
            "masked_pooled_projection": masked_pooled_projection,
            "masked_logits": masked_logits,
            "offset": point.offset,
            "source_index": torch.as_tensor(
                view.get(
                    "source_index",
                    torch.arange(point.feat.shape[0], device=point.feat.device),
                ),
                dtype=torch.long,
                device=point.feat.device,
            ),
            "mask": mask,
        }

    def encode_student_view(self, view: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        return self.encode_view(view, use_teacher=False)

    def encode_teacher_view(self, view: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        return self.encode_view(view, use_teacher=True)

    def forward(
        self, data_dict: Mapping[str, Any], return_point: bool = False
    ) -> dict[str, torch.Tensor]:
        grid_size = self._grid_size_value(data_dict)
        if return_point:
            point = Point(
                feat=data_dict["feat"],
                coord=data_dict["coord"],
                origin_coord=data_dict.get("origin_coord", data_dict["coord"]),
                offset=data_dict["offset"],
                grid_size=grid_size,
                mask=data_dict.get("mask"),
            )
            point = self.teacher["backbone"](point)
            point = self.up_cast(point)
            return {"point": point}

        with torch.no_grad():
            global_point = Point(
                feat=data_dict["global_feat"],
                coord=data_dict["global_coord"],
                origin_coord=data_dict["global_origin_coord"],
                offset=data_dict["global_offset"],
                grid_size=grid_size,
            )
            global_mask, _ = self.generate_mask(global_point.coord, global_point.offset)
            mask_global_coord = global_point.coord.clone().detach()
            if self.mask_jitter is not None and torch.any(global_mask):
                jitter = torch.randn_like(mask_global_coord[global_mask]).mul(
                    self.mask_jitter
                )
                mask_global_coord[global_mask] += torch.clip(
                    jitter, max=self.mask_jitter * 2
                )

            mask_global_point = Point(
                feat=data_dict["global_feat"],
                coord=mask_global_coord,
                origin_coord=data_dict["global_origin_coord"],
                mask=global_mask,
                offset=data_dict["global_offset"],
                grid_size=grid_size,
            )
            local_point = Point(
                feat=data_dict["local_feat"],
                coord=data_dict["local_coord"],
                origin_coord=data_dict["local_origin_coord"],
                offset=data_dict["local_offset"],
                grid_size=grid_size,
            )

            result_dict: dict[str, torch.Tensor | list[torch.Tensor]] = {"loss": []}
            global_point_ = self.teacher["backbone"](global_point)
            global_point_ = self.up_cast(global_point_)
            global_feat = global_point_.feat

        monitor_logits: torch.Tensor | None = None
        monitor_features: torch.Tensor | None = None
        monitor_masked_fraction = (
            float(global_mask.float().mean().item()) if global_mask.numel() > 0 else 0.0
        )
        monitor_global_mask = global_mask.detach()
        monitor_cosine_similarities: torch.Tensor | None = None

        if self.mask_loss_weight > 0 or self.roll_mask_loss_weight > 0:
            with torch.no_grad():
                global_point_.feat = self.teacher["mask_head"](global_feat)
            mask_global_point_ = self.student["backbone"](mask_global_point)
            mask_global_point_ = self.up_cast(mask_global_point_)

            with torch.no_grad():
                mask_match_index = (
                    self.match_neighbour(
                        mask_global_point_.origin_coord,
                        mask_global_point_.offset,
                        global_point_.origin_coord,
                        global_point_.offset,
                    )
                    if self.mask_loss_weight > 0
                    else torch.empty(
                        (0, 2), dtype=torch.long, device=mask_global_point_.coord.device
                    )
                )
                if self.roll_mask_loss_weight > 0:
                    roll_global_point_ = self.roll_point(global_point_)
                    roll_match_index = self.match_neighbour(
                        mask_global_point_.origin_coord,
                        mask_global_point_.offset,
                        roll_global_point_.origin_coord,
                        roll_global_point_.offset,
                    )
                else:
                    roll_match_index = torch.empty(
                        (0, 2), dtype=torch.long, device=mask_global_point_.coord.device
                    )

            all_student_idx = torch.cat(
                [mask_match_index[:, 0], roll_match_index[:, 0]]
            )
            if all_student_idx.numel() > 0:
                unique_idx, inverse = torch.unique(all_student_idx, return_inverse=True)
                mask_pred_sim = self.student["mask_head"](
                    mask_global_point_.feat[unique_idx]
                )
                n_mask_matched = mask_match_index.shape[0]
                mask_inverse = inverse[:n_mask_matched]
                roll_inverse = inverse[n_mask_matched:]
            else:
                mask_pred_sim = torch.empty(
                    (0, self.num_prototypes), device=mask_global_point_.coord.device
                )
                unique_idx = torch.empty(
                    0, dtype=torch.long, device=mask_global_point_.coord.device
                )
                n_mask_matched = 0
                mask_inverse = torch.empty(
                    0, dtype=torch.long, device=mask_global_point_.coord.device
                )
                roll_inverse = torch.empty(
                    0, dtype=torch.long, device=mask_global_point_.coord.device
                )

            monitor_logits = mask_pred_sim.detach()
            monitor_features = mask_global_point_.feat[unique_idx].detach()

            if self.mask_loss_weight > 0:
                if mask_match_index.shape[0] > 0:
                    monitor_cosine_similarities = F.cosine_similarity(
                        mask_global_point_.feat[mask_match_index[:, 0]],
                        global_feat[mask_match_index[:, 1]],
                        dim=-1,
                    ).detach()
                    with torch.no_grad():
                        mask_target_sim = self.sinkhorn_knopp(
                            global_point_.feat[mask_match_index[:, 1]],
                            self.teacher_temp,
                        )
                    mask_loss = -torch.sum(
                        mask_target_sim
                        * F.log_softmax(
                            mask_pred_sim[mask_inverse] / self.student_temp, dim=-1
                        ),
                        dim=-1,
                    )
                    mask_loss = torch_scatter.segment_coo(
                        mask_loss,
                        index=mask_global_point_.batch[mask_match_index[:, 0]],
                        reduce="mean",
                    ).mean()
                else:
                    mask_loss = mask_pred_sim.new_tensor(0.0)
                result_dict["mask_loss"] = mask_loss
                result_dict["loss"].append(mask_loss * self.mask_loss_weight)

            if self.roll_mask_loss_weight > 0:
                if roll_match_index.shape[0] > 0:
                    with torch.no_grad():
                        roll_mask_target_sim = self.sinkhorn_knopp(
                            roll_global_point_.feat[roll_match_index[:, 1]],
                            self.teacher_temp,
                        )
                    roll_mask_loss = -torch.sum(
                        roll_mask_target_sim
                        * F.log_softmax(
                            mask_pred_sim[roll_inverse] / self.student_temp, dim=-1
                        ),
                        dim=-1,
                    )
                    roll_mask_loss = torch_scatter.segment_coo(
                        roll_mask_loss,
                        index=mask_global_point_.batch[roll_match_index[:, 0]],
                        reduce="mean",
                    ).mean()
                else:
                    roll_mask_loss = mask_pred_sim.new_tensor(0.0)
                result_dict["roll_mask_loss"] = roll_mask_loss
                result_dict["loss"].append(roll_mask_loss * self.roll_mask_loss_weight)

        if self.unmask_loss_weight > 0:
            with torch.no_grad():
                global_point_.feat = self.teacher["unmask_head"](global_feat)
            local_point_ = self.student["backbone"](local_point)
            local_point_ = self.up_cast(local_point_)

            with torch.no_grad():
                principal_view_mask = global_point_.batch % self.num_global_view == 0
                principal_view_batch = (
                    global_point_.batch[principal_view_mask] // self.num_global_view
                )
                local_principal_offset = local_point_.offset[
                    self.num_local_view - 1 :: self.num_local_view
                ]
                unmask_match_index = self.match_neighbour(
                    local_point_.origin_coord,
                    local_principal_offset,
                    global_point_.origin_coord[principal_view_mask],
                    batch2offset(principal_view_batch),
                )

            if unmask_match_index.shape[0] > 0:
                unmask_pred_sim = self.student["unmask_head"](
                    local_point_.feat[unmask_match_index[:, 0]]
                )
                if monitor_logits is None:
                    monitor_logits = unmask_pred_sim.detach()
                    monitor_features = local_point_.feat[
                        unmask_match_index[:, 0]
                    ].detach()
                with torch.no_grad():
                    unmask_target_sim = self.sinkhorn_knopp(
                        global_point_.feat[principal_view_mask][
                            unmask_match_index[:, 1]
                        ],
                        self.teacher_temp,
                    )
                unmask_loss = -torch.sum(
                    unmask_target_sim
                    * F.log_softmax(unmask_pred_sim / self.student_temp, dim=-1),
                    dim=-1,
                )
                unmask_loss = torch_scatter.segment_coo(
                    unmask_loss,
                    index=local_point_.batch[unmask_match_index[:, 0]],
                    reduce="mean",
                ).mean()
            else:
                unmask_loss = local_point_.feat.new_tensor(0.0)
            result_dict["unmask_loss"] = unmask_loss
            result_dict["loss"].append(unmask_loss * self.unmask_loss_weight)

        total_loss = (
            sum(result_dict["loss"])
            if result_dict["loss"]
            else global_feat.new_tensor(0.0)
        )
        result_dict["loss"] = total_loss
        result_dict["total_loss"] = total_loss.detach().clone()

        if get_world_size() > 1:
            for key in list(result_dict.keys()):
                if key == "loss":
                    continue
                synced_loss = result_dict[key].detach()
                dist.all_reduce(synced_loss, op=dist.ReduceOp.SUM)
                synced_loss.div_(get_world_size())
                result_dict[key] = synced_loss

        self.last_monitoring_state = {
            "student_logits": monitor_logits,
            "point_features": monitor_features,
            "masked_fraction": monitor_masked_fraction,
            "global_mask": monitor_global_mask,
            "cosine_similarities": monitor_cosine_similarities,
        }
        return result_dict  # type: ignore[return-value]

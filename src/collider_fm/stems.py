from __future__ import annotations

import torch
import torch.nn as nn


def _build_stem_mlp(input_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.GELU(),
        nn.Linear(input_dim, output_dim),
        nn.GELU(),
        nn.Linear(output_dim, output_dim),
    )


def _require_categorical(categorical: dict[str, torch.Tensor], key: str, length: int) -> torch.Tensor:
    if key not in categorical:
        raise KeyError(f"Missing required categorical field '{key}'.")
    values = torch.as_tensor(categorical[key], dtype=torch.long).flatten()
    if values.shape[0] != length:
        raise ValueError(f"Categorical field '{key}' must contain one value per point.")
    return values


class TrackerStem(nn.Module):
    """Encode tracker hits from continuous measurements and detector metadata."""

    def __init__(
        self,
        continuous_dim: int,
        embed_dim: int,
        detector_vocab: int,
        volume_vocab: int,
        layer_vocab: int,
        surface_vocab: int | None = None,
        output_dim: int = 64,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.continuous_proj = nn.Linear(continuous_dim, output_dim)
        self.detector_embedding = nn.Embedding(detector_vocab, embed_dim)
        self.volume_embedding = nn.Embedding(volume_vocab, embed_dim)
        self.layer_embedding = nn.Embedding(layer_vocab, embed_dim)
        self.surface_embedding = nn.Embedding(surface_vocab, embed_dim) if surface_vocab is not None else None

        categorical_dim = embed_dim * (4 if self.surface_embedding is not None else 3)
        self.output_mlp = _build_stem_mlp(output_dim + categorical_dim, output_dim)

    def forward(self, continuous: torch.Tensor, categorical: dict[str, torch.Tensor]) -> torch.Tensor:
        if continuous.ndim != 2:
            raise ValueError("TrackerStem expects continuous inputs with shape [num_points, features].")

        num_points = continuous.shape[0]
        pieces = [self.continuous_proj(continuous)]
        pieces.append(self.detector_embedding(_require_categorical(categorical, "detector", num_points).to(continuous.device)))
        pieces.append(self.volume_embedding(_require_categorical(categorical, "volume_id", num_points).to(continuous.device)))
        pieces.append(self.layer_embedding(_require_categorical(categorical, "layer_id", num_points).to(continuous.device)))
        if self.surface_embedding is not None:
            surface = _require_categorical(categorical, "surface_id", num_points).to(continuous.device)
            pieces.append(self.surface_embedding(surface))
        return self.output_mlp(torch.cat(pieces, dim=1))


class CaloStem(nn.Module):
    """Encode calorimeter hits from coordinates, energy, and subsystem metadata."""

    def __init__(
        self,
        continuous_dim: int,
        embed_dim: int,
        subsystem_vocab: int,
        output_dim: int = 64,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.continuous_proj = nn.Linear(continuous_dim, output_dim)
        self.subsystem_embedding = nn.Embedding(subsystem_vocab, embed_dim)
        self.output_mlp = _build_stem_mlp(output_dim + embed_dim, output_dim)

    def forward(self, continuous: torch.Tensor, categorical: dict[str, torch.Tensor]) -> torch.Tensor:
        if continuous.ndim != 2:
            raise ValueError("CaloStem expects continuous inputs with shape [num_points, features].")

        num_points = continuous.shape[0]
        subsystem = _require_categorical(categorical, "detector", num_points).to(continuous.device)
        pieces = [self.continuous_proj(continuous), self.subsystem_embedding(subsystem)]
        return self.output_mlp(torch.cat(pieces, dim=1))


class ModalityFusion(nn.Module):
    """Fuse tracker and calo stem outputs into one sparse feature tensor."""

    def __init__(self, feature_dim: int, modality_vocab: int = 2) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.modality_embedding = nn.Embedding(modality_vocab, feature_dim)

    def forward(
        self,
        tracker_features: torch.Tensor,
        calo_features: torch.Tensor,
        modality_id: torch.Tensor,
        tracker_index: torch.Tensor | None = None,
        calo_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tracker_features.ndim != 2 or calo_features.ndim != 2:
            raise ValueError("ModalityFusion expects 2D feature tensors for both modalities.")
        if tracker_features.shape[1] != self.feature_dim or calo_features.shape[1] != self.feature_dim:
            raise ValueError("Tracker and calo features must match the configured fusion width.")

        total_points = tracker_features.shape[0] + calo_features.shape[0]
        modality_id = torch.as_tensor(modality_id, dtype=torch.long, device=tracker_features.device).flatten()
        if modality_id.shape[0] != total_points:
            raise ValueError("'modality_id' must contain one entry per fused point.")

        if tracker_index is None:
            tracker_index = torch.arange(tracker_features.shape[0], device=tracker_features.device, dtype=torch.long)
        else:
            tracker_index = torch.as_tensor(tracker_index, dtype=torch.long, device=tracker_features.device).flatten()
        if calo_index is None:
            calo_index = torch.arange(calo_features.shape[0], device=tracker_features.device, dtype=torch.long) + tracker_features.shape[0]
        else:
            calo_index = torch.as_tensor(calo_index, dtype=torch.long, device=tracker_features.device).flatten()

        fused = tracker_features.new_zeros((total_points, self.feature_dim))
        fused[tracker_index] = tracker_features
        fused[calo_index] = calo_features
        return fused + self.modality_embedding(modality_id)

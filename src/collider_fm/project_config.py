from __future__ import annotations

"""Shared OmegaConf-based configuration helpers for ColliderFM."""

import argparse
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
DEFAULT_DATASET_REVISION = "e28a24cc9c1641a478ae4e5bc3b376eb624b7283"


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    if config_path is None:
        return DEFAULT_CONFIG_PATH
    candidate = Path(config_path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def load_project_config(config_path: str | Path | None = None) -> DictConfig:
    base_config = OmegaConf.load(DEFAULT_CONFIG_PATH)
    resolved_path = resolve_config_path(config_path)
    if resolved_path == DEFAULT_CONFIG_PATH:
        return base_config
    override_config = OmegaConf.load(resolved_path)
    return OmegaConf.merge(base_config, override_config)


def load_project_config_from_cli(
    argv: list[str] | None = None,
) -> tuple[DictConfig, Path]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args, _ = parser.parse_known_args(argv)
    resolved_path = resolve_config_path(args.config)
    return load_project_config(resolved_path), resolved_path


def to_plain_container(config: Any) -> Any:
    return OmegaConf.to_container(config, resolve=True)


def model_factory_kwargs(
    model_config: DictConfig | dict[str, Any] | None,
) -> dict[str, Any]:
    if model_config is None:
        return {}
    plain_config = dict(to_plain_container(model_config))
    backbone_config = plain_config.pop("backbone", None)
    if backbone_config is not None:
        plain_config["backbone_kwargs"] = backbone_config
    return plain_config

from __future__ import annotations

"""Shared OmegaConf-based configuration helpers for ColliderFM."""

import argparse
from pathlib import Path
from typing import Any, Sequence

from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    if config_path is None:
        return DEFAULT_CONFIG_PATH
    candidate = Path(config_path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def load_project_config(
    config_path: str | Path | None = None,
    overrides: Sequence[str] | None = None,
) -> DictConfig:
    base_config = OmegaConf.load(DEFAULT_CONFIG_PATH)
    resolved_path = resolve_config_path(config_path)
    merged_config = base_config
    if resolved_path != DEFAULT_CONFIG_PATH:
        override_config = OmegaConf.load(resolved_path)
        merged_config = OmegaConf.merge(merged_config, override_config)
    if overrides:
        invalid_overrides = [override for override in overrides if "=" not in override]
        if invalid_overrides:
            invalid_str = ", ".join(invalid_overrides)
            raise ValueError(
                "OmegaConf CLI overrides must use key=value syntax. "
                f"Invalid overrides: {invalid_str}. "
                "Example: training.batch_size=16"
            )
        merged_config = OmegaConf.merge(
            merged_config, OmegaConf.from_dotlist(list(overrides))
        )
    return merged_config


def build_config_arg_parser(
    description: str,
    *,
    epilog: str | None = None,
    config_sections: Sequence[str] | None = None,
) -> argparse.ArgumentParser:
    combined_epilog = epilog or ""
    if config_sections:
        config = load_project_config()
        override_lines: list[str] = []
        for section in config_sections:
            section_value = config
            for part in section.split("."):
                section_value = section_value[part]
            override_lines.extend(
                _collect_override_lines(section, to_plain_container(section_value))
            )
        override_block = "Available overrides:\n" + "\n".join(
            f"  {line}" for line in override_lines
        )
        combined_epilog = (
            f"{combined_epilog}\n\n{override_block}"
            if combined_epilog
            else override_block
        )

    parser = argparse.ArgumentParser(
        description=description,
        epilog=combined_epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to an OmegaConf YAML file to merge on top of config/default.yaml.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides such as training.batch_size=16 or data.local_files_only=true.",
    )
    return parser


def to_plain_container(config: Any) -> Any:
    if not OmegaConf.is_config(config):
        return config
    return OmegaConf.to_container(config, resolve=True)


def _format_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        return "[" + ",".join(_format_override_value(item) for item in value) + "]"
    return str(value)


def _collect_override_lines(prefix: str, value: Any) -> list[str]:
    if isinstance(value, dict):
        lines: list[str] = []
        for key, nested_value in value.items():
            lines.extend(_collect_override_lines(f"{prefix}.{key}", nested_value))
        return lines
    return [f"{prefix}={_format_override_value(value)}"]


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

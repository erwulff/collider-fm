from __future__ import annotations

"""Shared OmegaConf-based configuration helpers for ColliderFM."""

import argparse
import shutil
import warnings
from pathlib import Path
from typing import Any, Sequence

from omegaconf import DictConfig, OmegaConf

from .experiment_logging import timestamp_suffix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    """Resolve a config path relative to the repository root when needed.

    Args:
        config_path (str | Path | None, optional): Path to a config file. If
            None, returns the default config path. Relative paths are resolved
            against the repository root; absolute paths are used as-is.
            Defaults to None.

    Returns:
        Path: The resolved absolute config file path.
    """

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
    """Load the default config, optionally merge a file, then apply dotlist overrides.

    Args:
        config_path (str | Path | None, optional): Path to an OmegaConf YAML
            file merged on top of the default config. Defaults to None.
        overrides (Sequence[str] | None, optional): OmegaConf dotlist overrides
            such as `["training.batch_size=16"]`. Defaults to None.

    Returns:
        DictConfig: The merged configuration.

    Raises:
        ValueError: If any override string lacks `=` (key=value syntax
            required).
    """

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
            raise ValueError("OmegaConf CLI overrides must use key=value syntax. " f"Invalid overrides: {invalid_str}. " "Example: training.batch_size=16")
        merged_config = OmegaConf.merge(merged_config, OmegaConf.from_dotlist(list(overrides)))
    return merged_config


def build_config_arg_parser(
    description: str,
    *,
    epilog: str | None = None,
    config_sections: Sequence[str] | None = None,
) -> argparse.ArgumentParser:
    """Build an argparse parser with `--config` and dotlist override support.

    When `config_sections` is provided, the epilog lists every available
    override key for the requested sections, pulled from the loaded default
    config.

    Args:
        description (str): Parser description shown in `--help`.
        epilog (str | None, optional): Extra epilog text appended after the
            override listing. Defaults to None.
        config_sections (Sequence[str] | None, optional): Dotted config section
            names (e.g. `["training", "model.training"]`) whose keys are listed
            as available overrides in the epilog. Defaults to None.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """

    combined_epilog = epilog or ""
    if config_sections:
        config = load_project_config()
        override_lines: list[str] = []
        for section in config_sections:
            section_value = config
            for part in section.split("."):
                section_value = section_value[part]
            override_lines.extend(_collect_override_lines(section, to_plain_container(section_value)))
        override_block = "Available overrides:\n" + "\n".join(f"  {line}" for line in override_lines)
        combined_epilog = f"{combined_epilog}\n\n{override_block}" if combined_epilog else override_block

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
    """Convert OmegaConf containers into plain Python values.

    Args:
        config (Any): An OmegaConf config node or a plain Python value.

    Returns:
        Any: The resolved plain Python value (dict/list/scalar), or the input
        unchanged if it is not an OmegaConf config.
    """

    if not OmegaConf.is_config(config):
        return config
    return OmegaConf.to_container(config, resolve=True)


def resolve_run_identity(
    project_root: Path,
    experiment_dir: str | None = None,
    run_name: str | None = None,
) -> tuple[Path, str]:
    """Resolve the run directory and run name for a training run.

    Args:
        project_root (Path): Repository root path.
        experiment_dir (str | None, optional): Explicit experiment directory.
            If None, defaults to `<project_root>/runs`. Defaults to None.
        run_name (str | None, optional): Explicit run name. If None, a
            timestamped name `run_<YYYYMMDD_HHMMSS>` is generated. Defaults to
            None.

    Returns:
        tuple[Path, str]: The resolved run directory and run name.
    """

    resolved_run_name = str(run_name) if run_name else f"run_{timestamp_suffix()}"
    resolved_experiment_dir = Path(experiment_dir) if experiment_dir is not None else project_root / "runs"
    resolved_run_dir = resolved_experiment_dir / resolved_run_name
    return resolved_run_dir, resolved_run_name


def resolve_run_lifecycle(
    project_root: Path,
    *,
    ray_storage_path: str | Path,
    experiment_dir: str | None = None,
    run_name: str | None = None,
    resume: bool = False,
    overwrite: bool = False,
) -> tuple[Path, str]:
    """Resolve run directory/name with resume and overwrite semantics.

    Validates that a fresh run does not collide with existing state, that
    resume targets an existing run, and that overwrite removes existing state
    before starting fresh.

    Args:
        project_root (Path): Repository root path.
        ray_storage_path (str | Path): Ray Train checkpoint storage root.
        experiment_dir (str | None, optional): Explicit experiment directory.
            Defaults to None (uses `<project_root>/runs`).
        run_name (str | None, optional): Explicit run name. Defaults to None
            (timestamped).
        resume (bool, optional): Whether to resume an existing run. Defaults to
            False.
        overwrite (bool, optional): Whether to remove existing run state and
            start fresh. Defaults to False.

    Returns:
        tuple[Path, str]: The resolved run directory and run name.

    Raises:
        ValueError: If `resume` and `overwrite` are both True, if `resume` is
            True without a `run_name`, or if a fresh run collides with existing
            state.
    """

    resolved_run_dir, resolved_run_name = resolve_run_identity(project_root, experiment_dir=experiment_dir, run_name=run_name)
    ray_run_dir = Path(ray_storage_path) / resolved_run_name

    if resume and overwrite:
        raise ValueError("training.resume=true cannot be combined with training.overwrite=true.")

    if overwrite:
        warnings.warn(
            f"Overwriting existing run state for {resolved_run_name}. Removing {resolved_run_dir} and {ray_run_dir} before starting fresh.",
            stacklevel=2,
        )
        if resolved_run_dir.exists():
            shutil.rmtree(resolved_run_dir)
        if ray_run_dir.exists():
            shutil.rmtree(ray_run_dir)
        return resolved_run_dir, resolved_run_name

    if resume:
        if run_name is None:
            raise ValueError("training.resume=true requires an explicit training.run_name.")
        if not resolved_run_dir.exists():
            raise ValueError(f"Run {resolved_run_name} cannot be resumed because local run directory " f"{resolved_run_dir} does not exist.")
        if not ray_run_dir.exists():
            raise ValueError(f"Run {resolved_run_name} cannot be resumed because Ray storage directory " f"{ray_run_dir} does not exist.")
        return resolved_run_dir, resolved_run_name

    if resolved_run_dir.exists() or ray_run_dir.exists():
        raise ValueError(
            f"Run {resolved_run_name} already exists. Set training.resume=true to continue, "
            "choose a new training.run_name, or use training.overwrite=true to start fresh."
        )

    return resolved_run_dir, resolved_run_name


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
    """Translate a model config block into the model factory's kwargs shape.

    Moves the `backbone` sub-config to `backbone_kwargs` and flattens the
    remaining keys into a plain dict.

    Args:
        model_config (DictConfig | dict[str, Any] | None): The model config
            block, or None.

    Returns:
        dict[str, Any]: Keyword arguments for the model factory (empty dict if
        `model_config` is None).
    """

    if model_config is None:
        return {}
    plain_config = dict(to_plain_container(model_config))
    backbone_config = plain_config.pop("backbone", None)
    if backbone_config is not None:
        plain_config["backbone_kwargs"] = backbone_config
    return plain_config


def select_model_config(config: DictConfig, flavor: str) -> DictConfig:
    """Select one named model config block from the shared project config.

    Args:
        config (DictConfig): The full project config.
        flavor (str): Model flavor to select; one of `"training"` or
            `"diagnostics"`.

    Returns:
        DictConfig: The selected model config block.

    Raises:
        ValueError: If `flavor` is not `"training"` or `"diagnostics"`.
    """

    if flavor not in {"training", "diagnostics"}:
        raise ValueError(f"Unsupported model flavor: {flavor}.")
    return config.model[flavor]


def point_view_kwargs(config: DictConfig, flavor: str, *, max_calo_hits: int | None) -> dict[str, Any]:
    """Build `build_point_view_from_event()` kwargs from the shared config.

    Args:
        config (DictConfig): The full project config.
        flavor (str): Model flavor (`"training"` or `"diagnostics"`); selects
            the grid_size to use.
        max_calo_hits (int | None): Optional cap on calorimeter hits per event.

    Returns:
        dict[str, Any]: Keyword arguments for `build_point_view_from_event()`.
    """

    view_config = config.views
    model_config = select_model_config(config, flavor)
    return {
        "max_calo_hits": max_calo_hits,
        "grid_size": float(model_config.grid_size),
        "coord_center": view_config.coord_center,
        "coord_scale": view_config.coord_scale,
        "energy_transform": view_config.energy_transform,
        "energy_min": view_config.energy_min,
        "energy_max": view_config.energy_max,
        "grid_sample_enabled": bool(view_config.get("grid_sample_enabled", False)),
        "grid_sample_size": float(view_config.get("grid_sample_size", 0.002)),
    }


def sonata_batch_kwargs(config: DictConfig, flavor: str, *, max_calo_hits: int | None) -> dict[str, Any]:
    """Build `build_sonata_batch()` kwargs from the shared config.

    Extends `point_view_kwargs` with augmentation and crop fields.

    Args:
        config (DictConfig): The full project config.
        flavor (str): Model flavor (`"training"` or `"diagnostics"`).
        max_calo_hits (int | None): Optional cap on calorimeter hits per event.

    Returns:
        dict[str, Any]: Keyword arguments for `build_sonata_batch()`.
    """

    view_config = config.views
    batch_kwargs = point_view_kwargs(
        config,
        flavor,
        max_calo_hits=max_calo_hits,
    )
    batch_kwargs.update(
        {
            "coord_noise_scale": view_config.coord_noise_scale,
            "feat_noise_scale": view_config.energy_jitter_scale,
            "phi_rotation_max": float(view_config.phi_rotation_max),
            "point_dropout": view_config.point_dropout,
            "num_global_views": view_config.num_global_views,
            "num_local_views": view_config.num_local_views,
            "global_crop_min_ratio": view_config.global_crop_min_ratio,
            "global_crop_max_ratio": view_config.global_crop_max_ratio,
            "local_crop_min_ratio": view_config.local_crop_min_ratio,
            "local_crop_max_ratio": view_config.local_crop_max_ratio,
            "constrain_to_principal": bool(view_config.get("constrain_to_principal", True)),
        }
    )
    return batch_kwargs

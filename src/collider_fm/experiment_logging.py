from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image


class NullLogger:
    """No-op logger used when all external logging is disabled."""

    def log_params(self, params: Mapping[str, Any]) -> None:
        """No-op parameter logging."""
        del params

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        step: int | None = None,
        epoch: int | None = None,
    ) -> None:
        """No-op metric logging."""
        del metrics, step, epoch

    def log_image(self, name: str, image_data: np.ndarray, step: int | None = None) -> None:
        """No-op image logging."""
        del name, image_data, step

    def finish(self) -> None:
        """No-op finish hook."""
        return


class JsonlLogger:
    """Local run logger that writes metrics JSONL and PNG visualizations."""

    def __init__(self, run_dir: Path) -> None:
        self.step_metrics_path = run_dir / "metrics_step.jsonl"
        self.epoch_metrics_path = run_dir / "metrics_epoch.jsonl"
        self.viz_dir = run_dir / "viz"

    def log_params(self, params: Mapping[str, Any]) -> None:
        """No-op; JsonlLogger does not log params."""
        del params

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        step: int | None = None,
        epoch: int | None = None,
    ) -> None:
        """Append a metrics record to the step or epoch JSONL file.

        Records with `record_type == "epoch_metrics"` are written to the epoch
        file; all others to the step file. `step` and `epoch` are inserted only
        if not already present in the record.

        Args:
            metrics (Mapping[str, Any]): Metric key-value pairs to record.
            step (int | None, optional): Global step. Defaults to None.
            epoch (int | None, optional): Epoch number. Defaults to None.
        """
        record = dict(metrics)
        if step is not None:
            record.setdefault("step", step)
        if epoch is not None:
            record.setdefault("epoch", epoch)
        record_type = str(record.get("record_type", "metrics"))
        metrics_path = self.epoch_metrics_path if record_type == "epoch_metrics" else self.step_metrics_path
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def log_image(self, name: str, image_data: np.ndarray, step: int | None = None) -> None:
        """Save a numpy RGB image array to the run's viz/ directory.

        Args:
            name (str): Image file basename (without extension).
            image_data (np.ndarray): RGB image array.
            step (int | None, optional): Step suffix appended to the filename.
                Defaults to None.
        """
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        step_suffix = f"_step{step}" if step is not None else ""
        path = self.viz_dir / f"{name}{step_suffix}.png"
        Image.fromarray(image_data).save(path)

    def finish(self) -> None:
        """No-op finish hook."""
        return


class CometLogger:
    """Thin adapter over a Comet experiment instance."""

    def __init__(self, experiment: Any) -> None:
        self.experiment = experiment

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Forward params to the Comet experiment.

        Args:
            params (Mapping[str, Any]): Parameter key-value pairs.
        """
        self.experiment.log_parameters(dict(params))

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        step: int | None = None,
        epoch: int | None = None,
    ) -> None:
        """Forward metrics to the Comet experiment.

        Args:
            metrics (Mapping[str, Any]): Metric key-value pairs.
            step (int | None, optional): Global step. Defaults to None.
            epoch (int | None, optional): Epoch number. Defaults to None.
        """
        self.experiment.log_metrics(dict(metrics), step=step, epoch=epoch)

    def log_image(self, name: str, image_data: np.ndarray, step: int | None = None) -> None:
        """Log a numpy RGB image array to the Comet experiment dashboard.

        Args:
            name (str): Image name.
            image_data (np.ndarray): RGB image array.
            step (int | None, optional): Step number. Defaults to None.
        """
        self.experiment.log_image(image_data, name=name, step=step, overwrite=True)

    def finish(self) -> None:
        """End the Comet experiment."""
        self.experiment.end()


class CompositeLogger:
    """Broadcast logging calls to multiple backends."""

    def __init__(self, loggers: list[Any]) -> None:
        self.loggers = loggers

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Broadcast `log_params` to all backends.

        Args:
            params (Mapping[str, Any]): Parameter key-value pairs.
        """
        for logger in self.loggers:
            logger.log_params(params)

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        step: int | None = None,
        epoch: int | None = None,
    ) -> None:
        """Broadcast `log_metrics` to all backends.

        Args:
            metrics (Mapping[str, Any]): Metric key-value pairs.
            step (int | None, optional): Global step. Defaults to None.
            epoch (int | None, optional): Epoch number. Defaults to None.
        """
        for logger in self.loggers:
            logger.log_metrics(metrics, step=step, epoch=epoch)

    def log_image(self, name: str, image_data: np.ndarray, step: int | None = None) -> None:
        """Broadcast `log_image` to all backends.

        Args:
            name (str): Image name.
            image_data (np.ndarray): RGB image array.
            step (int | None, optional): Step number. Defaults to None.
        """
        for logger in self.loggers:
            logger.log_image(name, image_data, step=step)

    def finish(self) -> None:
        """Call `finish` on all backends."""
        for logger in self.loggers:
            logger.finish()


def comet_config_path(home_dir: Path | None = None) -> Path:
    """Return the path to the Comet config file.

    Args:
        home_dir (Path | None, optional): Home directory to resolve the config
            path in. If None, uses the current user's home. Defaults to None.

    Returns:
        Path: Path to `~/.comet.config`.
    """
    resolved_home = Path.home() if home_dir is None else home_dir
    return resolved_home / ".comet.config"


def comet_is_configured(env: Mapping[str, str] | None = None, home_dir: Path | None = None) -> bool:
    """Check whether Comet is configured.

    Returns True if the `COMET_API_KEY` environment variable is set or a Comet
    config file exists.

    Args:
        env (Mapping[str, str] | None, optional): Environment mapping. If None,
            uses `os.environ`. Defaults to None.
        home_dir (Path | None, optional): Home directory for config file lookup.
            Defaults to None.

    Returns:
        bool: True if Comet is configured.
    """
    env = os.environ if env is None else env
    return bool(env.get("COMET_API_KEY")) or comet_config_path(home_dir).exists()


def resolve_comet_config(
    env: Mapping[str, str] | None = None,
) -> dict[str, str | None]:
    """Resolve the minimal Comet settings used by this project.

    Args:
        env (Mapping[str, str] | None, optional): Environment mapping. If None,
            uses `os.environ`. Defaults to None.

    Returns:
        dict[str, str | None]: Dict with `api_key`, `project_name` (default
        `"collider-fm"`), and `workspace` keys.
    """

    env = os.environ if env is None else env
    return {
        "api_key": env.get("COMET_API_KEY") or None,
        "project_name": env.get("COMET_PROJECT_NAME", "collider-fm"),
        "workspace": env.get("COMET_WORKSPACE") or None,
    }


def _default_comet_experiment_factory(
    *,
    api_key: str | None,
    project_name: str,
    workspace: str | None,
    run_name: str,
) -> Any:
    from comet_ml import Experiment

    experiment_kwargs: dict[str, Any] = {
        "project_name": project_name,
        "workspace": workspace,
        "auto_output_logging": "simple",
        "auto_metric_logging": False,
        "parse_args": False,
        "log_code": False,
        "log_env_cpu": True,
        "log_env_details": True,
        "log_env_gpu": True,
        "log_git_metadata": False,
        "log_git_patch": False,
        "log_graph": False,
    }
    if api_key is not None:
        experiment_kwargs["api_key"] = api_key

    experiment = Experiment(**experiment_kwargs)
    experiment.set_name(run_name)
    return experiment


def create_experiment_logger(
    backend: str,
    run_dir: Path,
    env: Mapping[str, str] | None = None,
    experiment_factory: Any | None = None,
    home_dir: Path | None = None,
) -> Any:
    """Create the requested logging backend for a training run.

    Args:
        backend (str): Logging backend; one of `"none"`, `"jsonl"`, `"auto"`,
            or `"comet"`. `"auto"` enables JSONL plus Comet when configured.
        run_dir (Path): Run output directory for local files.
        env (Mapping[str, str] | None, optional): Environment mapping for Comet
            config resolution. Defaults to None.
        experiment_factory (Any | None, optional): Callable producing a Comet
            experiment. If None, uses the default factory. Defaults to None.
        home_dir (Path | None, optional): Home directory for Comet config
            lookup. Defaults to None.

    Returns:
        Any: A logger instance (`NullLogger`, `JsonlLogger`, `CometLogger`, or
        `CompositeLogger`).

    Raises:
        ValueError: If `backend` is unsupported, or if `"comet"` is requested
            but Comet is not configured.
    """

    if backend not in {"none", "jsonl", "auto", "comet"}:
        raise ValueError(f"Unsupported log backend: {backend}")

    loggers: list[Any] = []
    if backend != "none":
        loggers.append(JsonlLogger(run_dir))

    wants_comet = backend in {"auto", "comet"}
    if wants_comet:
        if not comet_is_configured(env=env, home_dir=home_dir):
            if backend == "comet":
                raise ValueError("Comet logging requested, but neither COMET_API_KEY nor ~/.comet.config was found.")
        else:
            run_name = run_dir.name
            comet_config = resolve_comet_config(env)
            factory = experiment_factory or _default_comet_experiment_factory
            experiment = factory(
                api_key=comet_config["api_key"],
                project_name=str(comet_config["project_name"]),
                workspace=comet_config["workspace"],
                run_name=run_name,
            )
            loggers.append(CometLogger(experiment))

    if not loggers:
        return NullLogger()
    if len(loggers) == 1:
        return loggers[0]
    return CompositeLogger(loggers)


def timestamp_suffix(timestamp: datetime | None = None) -> str:
    """Format a timestamp as `YYYYMMDD_HHMMSS`.

    Args:
        timestamp (datetime | None, optional): Timestamp to format. If None,
            uses the current time. Defaults to None.

    Returns:
        str: The formatted timestamp string.
    """
    resolved_timestamp = datetime.now() if timestamp is None else timestamp
    return resolved_timestamp.strftime("%Y%m%d_%H%M%S")


def write_run_config(run_dir: Path, config: Mapping[str, Any]) -> Path:
    """Write the run config dict to `run_dir/config.json`.

    Args:
        run_dir (Path): Run output directory.
        config (Mapping[str, Any]): Config to serialize as JSON.

    Returns:
        Path: Path to the written `config.json` file.
    """
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(dict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config_path

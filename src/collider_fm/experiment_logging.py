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
        del params

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        step: int | None = None,
        epoch: int | None = None,
    ) -> None:
        del metrics, step, epoch

    def log_image(
        self, name: str, image_data: np.ndarray, step: int | None = None
    ) -> None:
        del name, image_data, step

    def finish(self) -> None:
        return


class JsonlLogger:
    """Local run logger that writes metrics JSONL and PNG visualizations."""

    def __init__(self, run_dir: Path) -> None:
        self.metrics_path = run_dir / "metrics.jsonl"
        self.viz_dir = run_dir / "viz"

    def log_params(self, params: Mapping[str, Any]) -> None:
        del params

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        step: int | None = None,
        epoch: int | None = None,
    ) -> None:
        record = dict(metrics)
        if step is not None:
            record.setdefault("step", step)
        if epoch is not None:
            record.setdefault("epoch", epoch)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def log_image(
        self, name: str, image_data: np.ndarray, step: int | None = None
    ) -> None:
        """Save a numpy RGB image array to the run's viz/ directory."""
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        step_suffix = f"_step{step}" if step is not None else ""
        path = self.viz_dir / f"{name}{step_suffix}.png"
        Image.fromarray(image_data).save(path)

    def finish(self) -> None:
        return


class CometLogger:
    """Thin adapter over a Comet experiment instance."""

    def __init__(self, experiment: Any) -> None:
        self.experiment = experiment

    def log_params(self, params: Mapping[str, Any]) -> None:
        self.experiment.log_parameters(dict(params))

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        step: int | None = None,
        epoch: int | None = None,
    ) -> None:
        self.experiment.log_metrics(dict(metrics), step=step, epoch=epoch)

    def log_image(
        self, name: str, image_data: np.ndarray, step: int | None = None
    ) -> None:
        """Log a numpy RGB image array to the Comet experiment dashboard."""
        self.experiment.log_image(image_data, name=name, step=step, overwrite=True)

    def finish(self) -> None:
        self.experiment.end()


class CompositeLogger:
    """Broadcast logging calls to multiple backends."""

    def __init__(self, loggers: list[Any]) -> None:
        self.loggers = loggers

    def log_params(self, params: Mapping[str, Any]) -> None:
        for logger in self.loggers:
            logger.log_params(params)

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        step: int | None = None,
        epoch: int | None = None,
    ) -> None:
        for logger in self.loggers:
            logger.log_metrics(metrics, step=step, epoch=epoch)

    def log_image(
        self, name: str, image_data: np.ndarray, step: int | None = None
    ) -> None:
        for logger in self.loggers:
            logger.log_image(name, image_data, step=step)

    def finish(self) -> None:
        for logger in self.loggers:
            logger.finish()


def comet_config_path(home_dir: Path | None = None) -> Path:
    resolved_home = Path.home() if home_dir is None else home_dir
    return resolved_home / ".comet.config"


def comet_is_configured(
    env: Mapping[str, str] | None = None, home_dir: Path | None = None
) -> bool:
    env = os.environ if env is None else env
    return bool(env.get("COMET_API_KEY")) or comet_config_path(home_dir).exists()


def resolve_comet_config(
    env: Mapping[str, str] | None = None,
) -> dict[str, str | None]:
    """Resolve the minimal Comet settings used by this project."""

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
        "log_env_cpu": False,
        "log_env_details": False,
        "log_env_gpu": False,
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
    """Create the requested logging backend for a training run."""

    if backend not in {"none", "jsonl", "auto", "comet"}:
        raise ValueError(f"Unsupported log backend: {backend}")

    loggers: list[Any] = []
    if backend != "none":
        loggers.append(JsonlLogger(run_dir))

    wants_comet = backend in {"auto", "comet"}
    if wants_comet:
        if not comet_is_configured(env=env, home_dir=home_dir):
            if backend == "comet":
                raise ValueError(
                    "Comet logging requested, but neither COMET_API_KEY nor ~/.comet.config was found."
                )
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
    resolved_timestamp = datetime.now() if timestamp is None else timestamp
    return resolved_timestamp.strftime("%Y%m%d_%H%M%S")


def write_run_config(run_dir: Path, config: Mapping[str, Any]) -> Path:
    config_path = run_dir / "config.json"
    config_path.write_text(
        json.dumps(dict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return config_path

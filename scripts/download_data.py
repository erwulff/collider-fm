from __future__ import annotations

"""Download ColliderML subsets into the shared Hugging Face cache."""

import argparse
import sys
from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from collider_fm.project_config import (
    build_config_arg_parser,
    load_project_config,
    to_plain_container,
)


def build_arg_parser() -> argparse.ArgumentParser:
    return build_config_arg_parser(
        description="Download selected ColliderML dataset configurations.",
        epilog=(
            "Examples:\n"
            "  uv run python scripts/download_data.py\n"
            "  uv run python scripts/download_data.py download.num_proc=12\n"
            "  uv run python scripts/download_data.py download.object_types=[calo_hits,tracker_hits]"
        ),
    )


def main() -> None:
    cli_args = build_arg_parser().parse_args()
    config = to_plain_container(
        load_project_config(cli_args.config, cli_args.overrides)
    )
    data_config = config["data"]
    download_config = config["download"]

    for dataset_type in download_config["dataset_types"]:
        for obj_type in download_config["object_types"]:
            dataset_name = f"{dataset_type}_{download_config['pu_config']}_{obj_type}"
            print(f"Downloading {dataset_name}...")
            load_dataset(
                data_config["dataset_name"],
                dataset_name,
                cache_dir=download_config["cache_dir"],
                revision=download_config["dataset_revision"],
                num_proc=download_config["num_proc"],
            )


if __name__ == "__main__":
    main()

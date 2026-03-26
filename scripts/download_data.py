from __future__ import annotations

"""Download ColliderML subsets into the shared Hugging Face cache."""

import argparse
from argparse import SUPPRESS
import sys
from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from collider_fm.project_config import (
    DEFAULT_CONFIG_PATH,
    load_project_config_from_cli,
    to_plain_container,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download selected ColliderML dataset configurations."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--dataset-name", default=SUPPRESS)
    parser.add_argument(
        "--object-types",
        nargs="+",
        default=SUPPRESS,
        help="List of object types to download (default: ['calo_hits']).",
    )
    parser.add_argument(
        "--dataset-types",
        nargs="+",
        default=SUPPRESS,
        help="List of dataset types (default: ['ttbar'], options: 'ttbar', 'dihiggs', 'ggf', 'higgs_portal', 'zee', 'zmumu').",
    )
    parser.add_argument(
        "--pu-config",
        default=SUPPRESS,
        help="PU configuration for the dataset (default: 'pu200', options: 'pu0').",
    )
    parser.add_argument(
        "--cache-dir",
        default=SUPPRESS,
        help="Directory to cache the datasets (default: '/mnt/ceph/users/ewulff/data/hf').",
    )
    parser.add_argument(
        "--dataset-revision",
        default=SUPPRESS,
        help="Pinned Hugging Face dataset revision (tag or commit hash).",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=SUPPRESS,
        help="Number of processes to use for downloading (default: 4).",
    )
    return parser


def resolve_args(argv: list[str] | None = None) -> argparse.Namespace:
    project_config, resolved_config_path = load_project_config_from_cli(argv)
    parser = build_arg_parser()
    parsed_args = parser.parse_args(argv)
    cli_values = vars(parsed_args)
    download_config = to_plain_container(project_config.download)
    merged = {
        "config": str(resolved_config_path),
        "dataset_name": to_plain_container(project_config.data)["dataset_name"],
        "object_types": download_config["object_types"],
        "dataset_types": download_config["dataset_types"],
        "pu_config": download_config["pu_config"],
        "cache_dir": download_config["cache_dir"],
        "dataset_revision": download_config["dataset_revision"],
        "num_proc": download_config["num_proc"],
    } | cli_values
    return argparse.Namespace(**merged)


def main() -> None:
    args = resolve_args()

    for dataset_type in args.dataset_types:
        for obj_type in args.object_types:
            dataset_name = f"{dataset_type}_{args.pu_config}_{obj_type}"
            print(f"Downloading {dataset_name}...")
            load_dataset(
                args.dataset_name,
                dataset_name,
                cache_dir=args.cache_dir,
                revision=args.dataset_revision,
                num_proc=args.num_proc,
            )


if __name__ == "__main__":
    main()

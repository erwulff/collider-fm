from __future__ import annotations

import argparse

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ColliderML calorimeter tables."
    )
    parser.add_argument(
        "--object-types",
        nargs="+",
        default=["calo_hits"],
        help="Object types to download.",
    )
    parser.add_argument(
        "--dataset-types",
        nargs="+",
        default=["ttbar"],
        help="Physics processes to download.",
    )
    parser.add_argument("--pu-config", default="pu0", help="Pileup configuration.")
    parser.add_argument(
        "--cache-dir",
        default="/mnt/ceph/users/ewulff/data/hf",
        help="Hugging Face cache directory.",
    )
    parser.add_argument(
        "--num-proc", type=int, default=4, help="Number of worker processes."
    )
    args = parser.parse_args()

    for dataset_type in args.dataset_types:
        for object_type in args.object_types:
            config_name = f"{dataset_type}_{args.pu_config}_{object_type}"
            print(f"Downloading {config_name}...")
            load_dataset(
                "CERN/ColliderML-Release-1",
                config_name,
                cache_dir=args.cache_dir,
                num_proc=args.num_proc,
            )


if __name__ == "__main__":
    main()

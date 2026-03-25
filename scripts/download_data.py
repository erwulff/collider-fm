import argparse
from datasets import load_dataset


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Download datasets with configurable options.")
    parser.add_argument(
        "--object-types",
        nargs="+",
        default=["calo_hits"],
        help="List of object types to download (default: ['calo_hits']).",
    )
    parser.add_argument(
        "--dataset-types",
        nargs="+",
        default=["ttbar"],
        help="List of dataset types (default: ['ttbar'], options: 'ttbar', 'dihiggs', 'ggf', 'higgs_portal').",
    )
    parser.add_argument(
        "--pu-config",
        default="pu200",
        help="PU configuration for the dataset (default: 'pu200', options: 'pu0').",
    )
    parser.add_argument(
        "--cache-dir",
        default="/mnt/ceph/users/ewulff/data/hf",
        help="Directory to cache the datasets (default: '/mnt/ceph/users/ewulff/data/hf').",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=4,
        help="Number of processes to use for downloading (default: 4).",
    )
    # Parse arguments
    args = parser.parse_args()

    # Download datasets
    for dataset_type in args.dataset_types:
        for obj_type in args.object_types:
            dataset_name = f"{dataset_type}_{args.pu_config}_{obj_type}"
            print(f"Downloading {dataset_name}...")
            load_dataset(
                "CERN/ColliderML-Release-1",
                dataset_name,
                cache_dir=args.cache_dir,
                num_proc=args.num_proc,  # Use configurable number of processes for faster downloading
            )


if __name__ == "__main__":
    main()

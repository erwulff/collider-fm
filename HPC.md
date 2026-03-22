# HPC resources and SLURM

This project is developed on an HPC cluster. Use this document for cluster-specific setup notes, resource references, and SLURM usage.

## General guidance

- Prefer SLURM jobs for heavy downloads, dependency builds, GPU validation, and training runs.
- Write SLURM logs to `logs_slurm/`.
- Avoid requesting more CPUs or memory than needed, especially on shared GPU partitions.

## uv on the cluster

The cluster provides a `uv` module starting with `modules/2.4`.

```bash
module load uv
```

The module:

- sets `UV_CACHE_DIR=$HOME/.cache/uv`
- appends its own `uv` binary to `PATH` instead of prepending it

That means a user-installed `uv` can still stay first in `PATH` while using the cache configuration provided by the module.

After syncing dependencies, format Python files with:

```bash
uv run black .
```

## Project-specific cluster notes

- The documented Hugging Face cache location for this project is `/mnt/ceph/users/ewulff/data/hf`.
- Heavy dataset download and GPU validation workflows are intended to run through scripts in `slurm/`.
- Some packages such as `torch-scatter` may need to build on compute nodes instead of the login node.

## Example SBATCH directives

Example 1:

```bash
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpuxl
#SBATCH --reservation=rocky9
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=96
#SBATCH --constraint=h100
```

Example 2:

```bash
#SBATCH -t 168:00:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --constraint=h100
```

Example 3:

```bash
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --constraint=a100-80gb&sxm4
```

Example 4:

```bash
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p genx
#SBATCH --cpus-per-task=12
```

## Compute node reference

### CPU partitions

| Nodes | CPU type | Cores | Memory | Notes |
| --- | --- | --- | --- | --- |
| 432 | genoa | 96 | 1.5 TB | `-C genoa` |
| 216 | icelake | 64 | 1 TB | `-C icelake` |
| 640 | rome | 128 | 1 TB | `-C rome` |
| 4 | cascadelake/cooperlake | 96-192 | 3-6 TB | `-p mem` |

### GPU partitions

| Nodes | GPU | CPU cores | Memory | Example SLURM flags |
| --- | --- | --- | --- | --- |
| 15 | 8x RTX6000Pro Blackwell-96B | 144 | 1 TB | `-p gpu -C rtxblackwell --reservation=rocky9` |
| 24 | 8x H200-144GB | 96 | 2 TB | `-p gpuxl -C h200 --reservation=rocky9` |
| 24 | 4x H100-94GB | 96 | 1.5 TB | `-p gpuxl -C h100 --reservation=rocky9` |
| 18 | 8x H100-80GB | 64 | 1 TB | `-C h100` |
| 36 | 4x A100-80GB | 64 | 1 TB | `-C a100-80gb` |
| 36 | 4x A100-40GB | 64 | 1 TB | `-C a100-40gb` |
| 6 | 4x V100-32GB | 36 | 768 GB | `-C v100` |

## GPU usage notes

- CoreSite, NJ hosts the Blackwell, H200, H100, and A100 systems.
- FI hosts the V100 systems.
- On non-exclusive partitions, avoid over-requesting system memory per GPU or cores per GPU.

Reference values:

| Node type | Constraint or partition | Max cores per GPU | Max system memory per GPU |
| --- | --- | --- | --- |
| Quad A100-40GB | `-C a100-40gb` | 16 | 256 GB |
| Quad A100-80GB | `-C a100-80gb` | 16 | 256 GB |
| Octo H100-80GB | `-C h100` | 8 | 128 GB |
| Quad H100-94GB | `-p gpuxl` | 24 | 384 GB |
| Octo H200-141GB | `-p gpup -C h200` | 12 | 256 GB |
| Octo H200-144GB | `-p gpu -C rtxblackwell --reservation=rocky9` | 18 | 125 GB |

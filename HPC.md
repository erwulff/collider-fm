# HPC resources and SLURM

## Compute nodes

#Nodes	CPU type	#Cores	Memory	GPU	Fabric	SLURM
432	genoa	96	1.5TB	no	InfiniBand	-C genoa
216	icelake	64	1TB	no	InfiniBand	-C icelake
640	rome	128	1TB	no	InfiniBand	-C rome
15	graniterapids	144	1TB	8x Nvidia RTX6000Pro Blackwell-96B	InfiniBand	-p gpu -C rtxblackwell --reservation=rocky9 --mem-per-cpu=<mem_per_core>
24	emeraldrapids	96	2TB	8x Nvidia H200-144GB	InfiniBand	-p gpuxl -C h200 --reservation=rocky9
24	genoa	96	1.5TB	4x Nvidia H100-94GB	InfiniBand	-p gpuxl -C h100 --reservation=rocky9
18	icelake	64	1TB	8x Nvidia H100-80GB	InfiniBand	-C h100
36	icelake	64	1TB	4x Nvidia A100-80GB	InfiniBand	-C a100-80gb
36	icelake	64	1TB	4x Nvidia A100-40GB	InfiniBand	-C a100-40gb
6	skylake	36	768GB	4x Nvidia V100-32GB	InfiniBand	-C v100
4	cascadelake, cooperlake	96-192	3-6TB	no	n/a	-p mem

## More info on GPU nodes

This wiki page has examples Slurm scripts and introductory information for using the GPU nodes: Using the GPU nodes
There are multiple generations of NVIDIA GPUs:
Located in CoreSite, NJ:
15 with 8x 96GB RTX6000Pro Blackwell Server Edition (Blackwell) (-p gpu --reservation=rocky9 -C rtxblackwell --gpus=<4N> --mem=<memory>) 1000GB system memory, 144 cores workergpu171 to workergpu185
16 with 8x 141GB H200 (Hopper SXM5) (-p eval -C h200 --gpus=<4N>) 2000GB system memory, 96 cores workergpu301 to workergpu318
24 with 4x 94GB H100 (Hopper SXM5) (-p gpuxl --gpus=<4N>) 1500GB system memory, 96 cores workergpu201 to workergpu224
18 with 8x 80GB H100 (Hopper) (--gpus=ib-h100p:<N>, --constraint=ib-h100p) 1000GB system memory, 64 cores workergpu151 to workergpu170
36 with 4x 80GB A100 (Ampere) (--gpus=a100-sxm4-80gb:<N>, --constraint=a100-80gb&sxm4) 1000GB system memory, 64 cores workergpu037 to workergpu072
36 with 4x 40GB A100 (Ampere) (--gpus=a100-sxm4-40gb:<N>, --constraint=a100-40gb&sxm4) 1000GB system memory, 64 cores. workergpu001 to workergpu036
Located at FI:
6 with 4x NVLinked 32GB V100 (--gpus=v100-sxm2-32gb:<N>, --constraint=v100, --constraint=v100-32gb), 768GB system memory,40 cores workergpu083 to workergpu142
See Slurm Partitions and Constraints if you are unsure how to use these.
On non-exclusive partitions, you should make sure that you are not accidentally locking down resources, so avoid requesting more system memory per GPU, or cores per GPU than needed. (we usually recommend not setting memory requirements on FI clusters)
Node type	Constraints	Max cores per GPU	Max system memory per GPU
Quad A100-40GB	-C a100-40gb	16	256
Quad A100-80GB	-C a100-80gb	16	256
Octo H100-80GB	-C h100	8	128
Quad H100-94GB	-p gpuxl	24	384
Octo H200-141GB	-p gpup -C h200	12	256
Octo H200-144GB	-p gpu -C rtxblackwell --reservation=rocky9	18	125


## Example SBATCH directives

Example 1
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpuxl
#SBATCH --reservation=rocky9
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=96
#SBATCH --constraint=h100

Example 2
Example 1
#SBATCH -t 168:00:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --constraint=h100

Example 3
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p gpu
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --constraint=a100-80gb&sxm4

Example 4
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p genx
#SBATCH --cpus-per-task=12

## uv on the cluster

The cluster provides a `uv` module starting with `modules/2.4`.

To use it:

```bash
module load uv
```

The module:

- sets `UV_CACHE_DIR=$HOME/.cache/uv`
- appends its own `uv` binary to `PATH` instead of prepending it

That means users can still install their own `uv` and keep it first in `PATH` while benefiting from the cache configuration provided by the module.

## SLURM guidelines

You don't have access to slurm commands like sbatch. Ask the user to run them for you.

Always write slurm logs to logs_slurm/.

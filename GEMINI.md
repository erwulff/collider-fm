# Legacy Project Context

`AGENTS.md` is now the primary operational guide for this repository.

Use this file for project background only. If anything in this file conflicts with the current repository contents or `AGENTS.md`, prefer the current repository state and `AGENTS.md`.

This document outlines the project background for adapting and scaling methods from the Panda paper on the ColliderML dataset.

## Project Goal

The primary goal of this project is to explore the application of self-distillation techniques for learning reusable representations from high-energy physics data. We will adapt the Panda model and scale it up to work with the large-scale ColliderML dataset.

## Key Papers and Resources

*   **Panda Paper:** Young, S., & Terao, K. (2025). *Panda: Self-distillation of Reusable Sensor-level Representations for High Energy Physics*. arXiv preprint arXiv:2512.01324. https://arxiv.org/abs/2512.01324
*   **ColliderML Dataset:** A large-scale, fully simulated benchmark dataset for machine learning in high-energy physics.
    *   Hugging Face: https://huggingface.co/datasets/ColliderML/ColliderML
    *   Website: https://colliderml.github.io/

## Methodology

The core of this project will be to re-implement or adapt the Panda model architecture and its self-distillation learning strategy. The initial focus will be on reproducing the key results from the paper on a smaller scale, before scaling up to the full ColliderML dataset.

## Dataset

We will use the ColliderML dataset. This dataset provides one million events with high pile-up (µ=200), which is a realistic simulation of the High-Luminosity Large Hadron Collider (HL-LHC) conditions, as well as and three million events with zero pile-up. The dataset is available in HDF5 format and can be accessed using a dedicated Python library.

This dataset contains simulated high-energy physics collision events generated using the Open Data Detector (ODD) geometry within the Key4hep and ACTS (A Common Tracking Software) frameworks, representing a generic collider detector similar to those at the HL-LHC.

### Dataset Summary

Collision Energy: 14 TeV (proton-proton)
Detector: Open Data Detector (ODD)
Simulation: DD4hep + Geant4 + ACTS
Format: Apache Parquet with list columns for variable-length data
License: CC-BY-4.0

### Available Configurations

The dataset is organized into multiple configurations, each representing a combination of:

Physics process (e.g., ttbar, ggf, dihiggs)
Pileup condition (pu0 = no pileup, pu200 = HL-LHC pileup)
Object type (particles, tracker_hits, calo_hits, tracks)

## Development Plan

A detailed work plan is available in [PLAN.md](PLAN.md). The high-level phases are:

1.  **Phase 1: Foundation and Initial Implementation**
2.  **Phase 2: Scaling and Evaluation**

When plan items are completed, update `PLAN.md` to keep the roadmap accurate.

## Commands and Conventions

*   **Hugging Face Cache:** The Hugging Face cache directory is located at `/mnt/ceph/users/ewulff/data/hf`.
*   **uv:** On this cluster, load `uv` with `module load uv` to set `UV_CACHE_DIR=$HOME/.cache/uv`. The module appends its own `uv` binary to `PATH`, so a user-installed `uv` still has precedence. Docs for `uv` are at https://docs.astral.sh/uv/.
*   **Panda dependency:** A `Panda_repo` git submodule is present in this repository and is used by the current model scaffold.

*(This section will be populated with more useful commands and project conventions as they are established.)*

## HPC and SLURM

This project is running on the login node of an HPC center. Do not run compute-heavy tasks directly in the terminal.
Compute-heavy tasks should be run on compute nodes via SLURM.

Detailed documentation on the available HPC resources and SLURM can be found in [HPC.md](HPC.md).

## CUDA

CUDA is available in the following versions. Make sure you pick a version that is compatible with all dependencies such as PyTorch etc.

cuda/12.0.0
cuda/12.1.1
cuda/12.3.2
cuda/12.5.1
cuda/12.8.0

## Container Environment

Gemini is running in an Apptainer container. The container was built using the script `apptainer/build_gemini_container.sh`.
This ensures a consistent development environment across different machines.

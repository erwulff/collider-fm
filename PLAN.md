# Project Plan: Panda on ColliderML

This document tracks the implementation roadmap for the project. Mark completed work here as the codebase evolves.

## Phase 1: Foundation and Initial Implementation

- **Stage 1: Environment and Data**
  - [x] Set up a dedicated Python environment using `uv`.
  - [x] Install core libraries: `pytorch`, `torchvision`, `torchaudio`, `h5py`, `huggingface_hub`, `datasets`.
  - [x] Write `scripts/download_data.py` to download a subset of the ColliderML dataset from Hugging Face.
  - [x] Create a PyTorch `Dataset` and `DataLoader` in `src/collider_fm/data.py` to read the HDF5 files.
  - [x] Implement basic data inspection and visualization to verify the data loader.

- **Stage 2: Model Architecture**
  - [ ] Review the Panda paper and repository (https://github.com/DeepLearnPhysics/Panda) to understand the hierarchical sparse 3D encoder in detail. The `Panda_repo` submodule is available here for reference.
  - [ ] Implement the core components of the Panda model in `src/collider_fm/model.py`.
  - [x] Start with a simplified version where needed.
  - [x] Write unit tests for the model components to ensure correctness.
  - Progress note: `src/collider_fm/model.py` now has a PTv3-backed point-cloud adapter, simplified student/teacher projection heads, and lightweight unit coverage. The `slurm/test_model.slurm` smoke test now completes on GPU with `Student output shape: (1, 32)` and `Distillation loss: 5.7166`. Full paper-parity architecture work is still pending.

- **Stage 3: Training and Loss Implementation**
  - [x] Implement the self-distillation loss function described in the paper.
  - [x] Create `scripts/train.py` to handle the training loop.
  - [x] Implement basic logging of training and validation loss.
  - [x] Train the model on a small subset of the data to ensure the training loop runs without errors.
  - Progress note: point-view construction now lives in `src/collider_fm/views.py`, the GPU smoke test uses `scripts/smoke_test_model.py`, and `scripts/train.py` supports a tiny train and validation run on cached ColliderML hits. `slurm/train_small.slurm` now completes successfully on GPU with `train loss: 7.4539` and `val loss: 6.9485`.

## Phase 2: Scaling and Evaluation

- **Stage 4: Evaluation and Refinement**
  - [ ] Create a diagnostics plotting script to visualize raw events, model inputs, per-point backbone features, pooled event embeddings, and prototype outputs.
  - [ ] Implement evaluation metrics based on the paper's methodology.
  - [ ] Create `scripts/evaluate.py` to run the evaluation.
  - [ ] Refine the model and training process based on initial results.
  - [ ] Experiment with hyperparameters such as learning rate and batch size.

- **Stage 5: Scaling Up**
  - [ ] Modify the data loader to handle the full ColliderML dataset.
  - [ ] Set up distributed training if necessary, for example with `torch.distributed`.
  - [ ] Start a long-running training job on the full dataset.
  - [ ] Implement checkpointing to save and resume training.

- **Stage 6: Analysis and Documentation**
  - [ ] Analyze the results from the scaled-up training.
  - [ ] Create visualizations of the learned representations.
  - [ ] Compare the results to the paper's benchmarks if possible.
  - [ ] Document the final results and findings.

## Future Work

- [ ] Explore different data augmentation techniques.
- [ ] Experiment with different model architectures or loss functions.
- [ ] Apply the learned representations to downstream tasks such as jet tagging or particle identification.

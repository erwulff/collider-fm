# Project Plan: Panda on ColliderML

This document provides a detailed work plan for the project.

## Phase 1: Foundation and Initial Implementation (Weeks 1-3)

*   **Week 1: Environment and Data**
    *   [x] Set up a dedicated Python environment using uv.
    *   [x] Install core libraries: `pytorch`, `torchvision`, `torchaudio`, `h5py`, `huggingface_hub`, `datasets`.
    *   [x] Write a script (`scripts/download_data.py`) to download a subset of the ColliderML dataset from Hugging Face.
    *   [x] Create a PyTorch `Dataset` and `DataLoader` (`src/collider_fm/data.py`) to read the HDF5 files.
    *   [x] Implement basic data inspection and visualization to verify the data loader.

*   **Week 2: Model Architecture**
    *   [ ] Review the Panda paper and github repo (https://github.com/DeepLearnPhysics/Panda) to understand the model architecture in detail (hierarchical sparse 3D encoder). The `Panda_repo` submodule is available in this repository for reference.
    *   [ ] Implement the core components of the Panda model in `src/collider_fm/model.py`.
    *   [ ] Start with a simplified version if necessary.
    *   [ ] Write unit tests for the model components to ensure correctness.

*   **Week 3: Training and Loss Implementation**
    *   [ ] Implement the self-distillation loss function as described in the paper.
    *   [ ] Create a `scripts/train.py` script to handle the training loop.
    *   [ ] Implement basic logging of training and validation loss.
    *   [ ] Train the model on a small subset of the data to ensure the training loop runs without errors.

## Phase 2: Scaling and Evaluation (Weeks 4-6)

*   **Week 4: Evaluation and Refinement**
    *   [ ] Implement evaluation metrics based on the paper's methodology.
    *   [ ] Create an `scripts/evaluate.py` script to run the evaluation.
    *   [ ] Refine the model and training process based on initial results.
    *   [ ] Experiment with hyperparameters (learning rate, batch size, etc.).

*   **Week 5: Scaling Up**
    *   [ ] Modify the data loader to handle the full ColliderML dataset.
    *   [ ] Set up for distributed training if necessary (e.g., using `torch.distributed`).
    *   [ ] Start a long-running training job on the full dataset.
    *   [ ] Implement checkpointing to save and resume training.

*   **Week 6: Analysis and Documentation**
    *   [ ] Analyze the results from the scaled-up training.
    *   [ ] Create visualizations of the learned representations.
    *   [ ] Compare the results to the paper's benchmarks if possible.
    *   [ ] Document the final results and findings.

## Future Work

*   [ ] Explore different data augmentation techniques.
*   [ ] Experiment with different model architectures or loss functions.
*   [ ] Apply the learned representations to downstream tasks (e.g., jet tagging, particle identification).

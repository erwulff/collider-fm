## Agent brief

Build a high-quality Python package, optimized for ease of use and hackability, for self-supervised pretraining of a reusable backbone on ColliderML Release 1 using only `calo_hits`.

### Hard scope limits

- Use only calorimeter hits as model input in this phase.
- Ignore tracker hits, tracks, particles-as-labels, and all downstream heads.
- Support both ECal and HCal hits from the start, because ColliderML includes both calorimeters and their granularity differs substantially, with 5.1 mm ECal cells and 30 mm HCal cells.
- Train only a reusable backbone and pretraining heads.
- Keep the codebase modular so tracker hits can be added later as a second modality without breaking the calo-only path.

## Package tree

Use this exact package layout as the starting point:

```text
colliderml_panda/
├── README.md
├── pyproject.toml
├── .gitignore
├── .env.example
├── configs/
│   ├── data/
│   │   ├── colliderml_calo_streaming.yaml
│   │   ├── colliderml_calo_cached.yaml
│   │   └── colliderml_single_particle_debug.yaml
│   ├── model/
│   │   ├── ptv3_tiny.yaml
│   │   ├── ptv3_small.yaml
│   │   └── ptv3_base.yaml
│   ├── pretrain/
│   │   ├── panda_ssl_base.yaml
│   │   ├── panda_ssl_debug.yaml
│   │   └── panda_ssl_multinode.yaml
│   ├── runtime/
│   │   ├── local_cpu_debug.yaml
│   │   ├── single_gpu.yaml
│   │   ├── multi_gpu_ddp.yaml
│   │   └── slurm.yaml
│   └── experiments/
│       ├── exp01_calo_ssl_debug.yaml
│       ├── exp02_calo_ssl_small.yaml
│       └── exp03_calo_ssl_base.yaml
├── scripts/
│   ├── run_pretrain.py
│   ├── inspect_dataset.py
│   ├── build_event_stats.py
│   ├── smoke_test_dataloader.py
│   ├── export_embeddings.py
│   ├── visualize_views.py
│   └── launch_slurm.sh
├── src/
│   └── colliderml_panda/
│       ├── __init__.py
│       ├── version.py
│       ├── cli/
│       │   ├── __init__.py
│       │   ├── main.py
│       │   ├── pretrain_cli.py
│       │   └── inspect_cli.py
│       ├── config/
│       │   ├── __init__.py
│       │   ├── schema.py
│       │   ├── loader.py
│       │   └── defaults.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── hf_loader.py
│       │   ├── dataset_splits.py
│       │   ├── calo_schema.py
│       │   ├── event_builder.py
│       │   ├── event_filter.py
│       │   ├── normalization.py
│       │   ├── voxelization.py
│       │   ├── collate.py
│       │   ├── views.py
│       │   ├── masking.py
│       │   ├── augmentations.py
│       │   ├── datamodule.py
│       │   └── stats.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── layers/
│       │   │   ├── __init__.py
│       │   │   ├── sparse_ops.py
│       │   │   ├── serialization.py
│       │   │   ├── attention.py
│       │   │   ├── mlp.py
│       │   │   ├── pooling.py
│       │   │   └── norm.py
│       │   ├── backbones/
│       │   │   ├── __init__.py
│       │   │   ├── ptv3_backbone.py
│       │   │   ├── stem.py
│       │   │   ├── encoder_stage.py
│       │   │   ├── feature_pyramid.py
│       │   │   └── positional_encoding.py
│       │   ├── heads/
│       │   │   ├── __init__.py
│       │   │   ├── projection_head.py
│       │   │   ├── prototype_head.py
│       │   │   └── masked_token.py
│       │   ├── student_teacher.py
│       │   └── factory.py
│       ├── losses/
│       │   ├── __init__.py
│       │   ├── cross_entropy.py
│       │   ├── sinkhorn.py
│       │   ├── ssl_loss.py
│       │   └── regularizers.py
│       ├── engine/
│       │   ├── __init__.py
│       │   ├── trainer.py
│       │   ├── loops.py
│       │   ├── ema.py
│       │   ├── optim.py
│       │   ├── schedulers.py
│       │   ├── checkpointing.py
│       │   ├── logging.py
│       │   ├── distributed.py
│       │   ├── precision.py
│       │   └── seed.py
│       ├── eval/
│       │   ├── __init__.py
│       │   ├── prototype_usage.py
│       │   ├── embedding_stats.py
│       │   ├── tsne_umap.py
│       │   ├── nearest_neighbors.py
│       │   └── linear_probe_stub.py
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── io.py
│       │   ├── paths.py
│       │   ├── registry.py
│       │   ├── timers.py
│       │   └── rich_logging.py
│       └── tests/
│           ├── __init__.py
│           ├── test_config.py
│           ├── test_hf_loader.py
│           ├── test_event_builder.py
│           ├── test_augmentations.py
│           ├── test_masking.py
│           ├── test_backbone_shapes.py
│           ├── test_ssl_loss.py
│           └── test_smoke_train_step.py
├── notebooks/
│   ├── 01_dataset_inspection.ipynb
│   ├── 02_event_statistics.ipynb
│   └── 03_embedding_visualization.ipynb
└── outputs/
    ├── checkpoints/
    ├── logs/
    ├── plots/
    └── embeddings/
```

This tree separates data handling, backbone code, self-distillation heads, training engine, and representation diagnostics, which matches Panda's split between a reusable backbone and pretraining-specific objective code.
It also gives the agent a clean place to handle ColliderML's parquet ingestion and Hugging Face access path without mixing dataset logic into model code.

## File responsibilities

`src/colliderml_panda/data/hf_loader.py` must load ColliderML from Hugging Face using the `datasets` library and expose a consistent iterable interface for streaming or cached modes, because the dataset is published on Hugging Face and is designed to work with that interface.
`src/colliderml_panda/data/calo_schema.py` must define the exact calo-hit fields the pipeline expects, centered on event ID, detector ID, position, energy, and optional truth-contribution lists, because Release 1 documents those as the calorimeter-hit features.
`src/colliderml_panda/data/event_builder.py` must convert one event into a sparse point-cloud sample with coordinates and per-point features, because Panda consumes raw 3D points with an associated scalar signal rather than dense images or high-level objects.

`src/colliderml_panda/data/normalization.py` must handle coordinate and feature normalization without erasing detector structure, because ColliderML's ODD geometry contains distinct barrel and endcap calorimeters and different ECal or HCal cell scales.
`src/colliderml_panda/data/augmentations.py`, `views.py`, and `masking.py` must generate two global views plus local and masked student views, because Panda's pretraining objective aligns student predictions on local or masked views to teacher predictions on unmasked global views.
`src/colliderml_panda/models/backbones/ptv3_backbone.py` must implement a Point Transformer V3-style sparse hierarchical encoder with multi-scale outputs, because Panda's backbone is a five-stage U-Net-style sparse 3D hierarchy with sparse convolutions, point serialization, local self-attention, and stride-2 pooling between stages.

`src/colliderml_panda/models/heads/projection_head.py` must map backbone point embeddings to a 256-dimensional latent space, because Panda projects upcast point features into a low-dimensional latent before prototype assignment.
`src/colliderml_panda/models/heads/prototype_head.py` must implement normalized learnable prototypes and temperature-scaled logits, because Panda uses 4096 learnable prototypes on a unit hypersphere with cosine-similarity-style assignment.
`src/colliderml_panda/models/student_teacher.py` plus `engine/ema.py` must implement student and EMA teacher networks with shared architecture but separate parameter update rules, because Panda updates the teacher as an exponential moving average of the student.

`src/colliderml_panda/losses/sinkhorn.py` and `ssl_loss.py` must implement teacher balancing, student-teacher cross-entropy, and separate local and mask losses, because Panda uses Sinkhorn-Knopp-style balancing plus local-view and masked-view cross-entropy losses to prevent collapse and learn stable prototypes.
`src/colliderml_panda/eval/prototype_usage.py`, `embedding_stats.py`, and `tsne_umap.py` must provide collapse and representation diagnostics, because Panda evaluates representation geometry and explicitly discusses collapse prevention during pretraining.
`src/colliderml_panda/eval/linear_probe_stub.py` should exist now as a placeholder interface, even if it only exports embeddings, because Panda uses frozen-feature probing during pretraining selection and you will want that hook later.

## Implementation spec

### 1. Dataset and sample definition

Represent each event as a set of points with at minimum `coords = [x, y, z]` in millimeters and `features = [energy, detector_one_hot_or_id]`.
Do not use tracker hits yet, and do not require truth labels for pretraining, even though ColliderML stores truth-contribution lists for calo cells.
Make detector identity explicit so the model can distinguish ECal and HCal from the start, because the two calorimeters have different segmentation and sampling behavior.

Use a configurable event filter to reject empty events, cap extreme point counts, and optionally apply a stricter energy cut only for memory control.
Keep raw released energies by default, because ColliderML's calorimeter values are already thresholded and time-window filtered during digitization, while calibration is documented as a separate optional recipe.
Add a config flag for region-based calibration factors later, but leave it disabled in the default pretraining path.

### 2. Geometry and preprocessing

Support both absolute Cartesian coordinates and derived cylindrical helper coordinates internally, but feed the backbone one consistent coordinate system at a time.
Do not random-rotate events in arbitrary 3D, because Panda's original rotations and flips were safe for LArTPC charge clouds, while ColliderML uses a structured cylindrical detector with barrel and endcap regions.
Use detector-aware preprocessing that preserves local shower geometry and coarse global placement.

Implement sparse voxelization as a configurable preprocessing option, but keep a direct point-cloud path available.
Start with a calo-friendly voxel size that is coarse enough for GPU memory yet fine enough not to erase ECal structure; expose this entirely through config because ECal and HCal granularity differs strongly.
Make `build_event_stats.py` compute point-count, energy-sum, detector-fraction, and spatial-range statistics before finalizing default voxel size and crop parameters.

### 3. Backbone

Implement a Panda-inspired Point Transformer V3 backbone with a sparse input stem, 4 to 5 encoder stages, stride-2 pooling between stages, sparse convolution plus local attention blocks, and multi-scale feature aggregation at the output.
Do not implement any decoder in this phase, because Panda's self-distillation pretraining is decoder-free and works directly from backbone point features.
Provide at least `tiny`, `small`, and `base` variants in config so the agent can debug on one GPU before scaling up.

Use a reusable feature interface such as `{"coords": ..., "features": ..., "embeddings": ..., "multiscale": ...}` returned by the backbone forward pass.
Keep backbone code free of any task-specific assumptions about clustering, particle IDs, or reconstruction heads, because the point of Panda is to learn a general reusable sensor-level representation first.
Ensure the backbone can export frozen point embeddings to disk for later diagnostics and downstream transfer.

### 4. Panda-style pretraining

Implement a student-teacher SSL loop with one student backbone, one EMA teacher backbone, a shared projection space, and a shared prototype layer.
The projection head must output 256-dimensional normalized point embeddings, and the prototype head should default to 4096 prototypes with weight normalization and cosine-style logits, because that is Panda's published design.
Do not add a `[cls]` token or event-level summary token, because Panda explicitly avoids it when events contain multiple unrelated local topologies, which also matches high-pileup ColliderML events.

Generate two global views per event, several smaller local views, and masked global variants.
For ColliderML calo hits, use safe augmentations such as azimuthal rotation around the beam axis, small coordinate jitter, small multiplicative energy jitter, random contiguous local cropping, patch masking, and point dropout.
Do not use arbitrary flips across detector axes as a default, because those can break detector semantics more easily here than in Panda's original LArTPC setting.

Implement two SSL losses:
- Local-view loss, where student local crops match the corresponding region of an unmasked teacher global view.
- Masked-view loss, where student masked global views match the teacher on masked points only.

Use low student temperature, teacher temperature warmup, Sinkhorn-style balancing or equivalent centering, and EMA teacher momentum scheduling, because Panda relies on those mechanisms to prevent uniform or single-prototype collapse.
Default the training recipe to AdamW with 5 percent warmup and cosine decay, using Panda-like schedules as the baseline before tuning for ColliderML.
Expose all of these in YAML rather than hardcoding them.

### 5. Runtime, outputs, and acceptance criteria

The training command must support local debug, single-GPU, and multi-GPU DDP execution from the same entrypoint.
Save checkpoints for student backbone, teacher backbone, optimizer, scheduler, and EMA state.
Write outputs to `outputs/checkpoints`, `outputs/logs`, `outputs/plots`, and `outputs/embeddings`.

A run is considered minimally successful when all of the following are true:
- `inspect_dataset.py` can read a chosen ColliderML calo split and print valid schema, event counts, and point statistics.
- `smoke_test_dataloader.py` can build multi-view batches from calo hits only.
- `test_smoke_train_step.py` can run one SSL optimization step without NaNs.
- `run_pretrain.py` can train for at least several hundred steps and save recoverable checkpoints.
- Diagnostic plots show non-collapsed prototype usage and embedding variance over time, which is necessary because Panda-style self-distillation can otherwise collapse without balancing and temperature control.

## Copy-paste prompt

Use this as the exact instruction block for the coding agent:

> Build a Python package named `colliderml_panda` for Panda-style self-supervised pretraining on ColliderML Release 1 using only calorimeter hits.
> Use the file tree provided above exactly unless a missing dependency forces a small change.
> Start with dataset inspection, schema validation, event building, multi-view augmentation, and a calo-only sparse point-cloud dataloader.
> Then implement a Panda-inspired Point Transformer V3-style backbone, projection head, prototype head, student-teacher wrapper, EMA update, and SSL loss with local-view and masked-view objectives.
> Use ColliderML calo hits only; ignore tracker hits and all downstream tasks.
> Keep configuration fully YAML-driven and make the package runnable in debug, single-GPU, and DDP modes.
> Prioritize correctness, debuggability, tests, and checkpoint recovery over peak speed.
> Deliver working scripts for dataset inspection, dataloader smoke tests, SSL pretraining, view visualization, and embedding export.
> The first finished milestone is a stable calo-only pretrained backbone with representation diagnostics, not a supervised reconstruction result.

Would you like a second version of this spec rewritten as a strict engineering ticket with numbered implementation tasks and acceptance tests?
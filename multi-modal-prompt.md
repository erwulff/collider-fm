## Plan: Multimodal Panda V0 File Tree And Execution Order

Reorient the repo around a multimodal Panda-style v0 by introducing modality-aware data assembly, separate tracker and calo stems, a shared Panda/PTv3 backbone, detector-aware multiview SSL, and modality-split diagnostics. The main implementation strategy is to add the new architecture alongside the current scaffold first, switch scripts and tests once the new path is working, and only then retire the old simplified fused-feature assumptions.

**Target package and file tree**
1. Keep the current package root /mnt/home/ewulff/repositories/collider-fm/src/collider_fm/, but split responsibilities more explicitly so data contract, stems, views, and SSL logic are no longer coupled in one or two files.
2. Recommended package layout:
3. /mnt/home/ewulff/repositories/collider-fm/src/collider_fm/data.py — keep ColliderML dataset access and raw event loading, but restrict it to loading tables and typed raw event dictionaries rather than constructing final model features.
4. /mnt/home/ewulff/repositories/collider-fm/src/collider_fm/features.py — new module for multimodal feature assembly and model-facing point payload construction.
5. /mnt/home/ewulff/repositories/collider-fm/src/collider_fm/views.py — redesign as detector-aware crop, mask, and augmentation generation over the new multimodal point payloads.
6. /mnt/home/ewulff/repositories/collider-fm/src/collider_fm/stems.py — new module for TrackerStem, CaloStem, and fusion helpers.
7. /mnt/home/ewulff/repositories/collider-fm/src/collider_fm/model.py — shared multimodal SSL model wrapper, teacher/student orchestration, projection heads, prototype heads, loss helpers, and checkpoint-facing configuration.
8. /mnt/home/ewulff/repositories/collider-fm/src/collider_fm/ssl.py — optional new module if loss scheduling and view-role logic start to overcrowd model.py. Recommendation: only split this out once local and masked losses are implemented.
9. /mnt/home/ewulff/repositories/collider-fm/src/collider_fm/diagnostics.py — keep as the shared diagnostics layer, but expand it around modality-split summaries and prototype occupancy.
10. /mnt/home/ewulff/repositories/collider-fm/scripts/train.py — promote from smoke-scale loop into resumable multimodal SSL training entrypoint.
11. /mnt/home/ewulff/repositories/collider-fm/scripts/smoke_test_model.py — keep lightweight; later adapt it to build a tiny multimodal batch and validate the new forward and loss path.
12. /mnt/home/ewulff/repositories/collider-fm/scripts/plot_diagnostics.py — reuse the shared diagnostics layer and add modality-split reports.
13. /mnt/home/ewulff/repositories/collider-fm/notebooks/plot_diagnostics_explorer.ipynb — keep secondary to Python modules and re-read before edits because it has already diverged recently.
14. /mnt/home/ewulff/repositories/collider-fm/tests/test_data.py — extend rather than replace.
15. /mnt/home/ewulff/repositories/collider-fm/tests/test_features.py — new test file for multimodal feature assembly.
16. /mnt/home/ewulff/repositories/collider-fm/tests/test_views.py — heavily refactor to focus on crops, masks, point IDs, and modality balance.
17. /mnt/home/ewulff/repositories/collider-fm/tests/test_stems.py — new test file for TrackerStem, CaloStem, and fusion shape checks.
18. /mnt/home/ewulff/repositories/collider-fm/tests/test_model.py — refactor around the new multimodal model path.
19. /mnt/home/ewulff/repositories/collider-fm/tests/test_diagnostics.py — extend for modality-split summaries.

**Recommended classes and method signatures**
1. In /mnt/home/ewulff/repositories/collider-fm/src/collider_fm/data.py:
2. RawTrackerHits — typed event-side mapping or TypedDict with x, y, z, time, detector, volume_id, layer_id, optional surface_id, and truth-linked fields reserved for diagnostics.
3. RawCaloHits — typed event-side mapping or TypedDict with x, y, z, total_energy, detector or subsystem code, and contribution fields reserved for diagnostics.
4. RawColliderEvent — typed mapping containing tracker_hits and calo_hits.
5. ColliderMLDataset.__getitem__(index: int) -> RawColliderEvent
6. Recommendation: keep model-facing tensors out of data.py so feature policy does not get buried inside dataset loading.

7. In /mnt/home/ewulff/repositories/collider-fm/src/collider_fm/features.py:
8. MultimodalPointBatch — TypedDict or small dataclass with coord, tracker_continuous, calo_continuous, tracker_categorical, calo_categorical, modality_id, point_id, event_id, offset, and grid_size.
9. build_multimodal_points(event: RawColliderEvent, device: torch.device, max_tracker_hits: int | None = None, max_calo_hits: int | None = None) -> MultimodalPointBatch
10. sample_tracker_hits(...) -> dict[str, torch.Tensor]
11. sample_calo_hits(...) -> dict[str, torch.Tensor]
12. assign_point_ids(event: RawColliderEvent, tracker_indices: torch.Tensor, calo_indices: torch.Tensor) -> torch.Tensor
13. build_model_inputs(points: MultimodalPointBatch) -> dict[str, torch.Tensor]
14. Recommendation: store truth-linked fields only in optional diagnostics payloads, not in build_model_inputs output.

15. In /mnt/home/ewulff/repositories/collider-fm/src/collider_fm/stems.py:
16. class TrackerStem(nn.Module)
17. TrackerStem.__init__(continuous_dim: int, embed_dim: int, detector_vocab: int, volume_vocab: int, layer_vocab: int, surface_vocab: int | None = None, output_dim: int = 64)
18. TrackerStem.forward(continuous: torch.Tensor, categorical: dict[str, torch.Tensor]) -> torch.Tensor
19. class CaloStem(nn.Module)
20. CaloStem.__init__(continuous_dim: int, embed_dim: int, subsystem_vocab: int, output_dim: int = 64)
21. CaloStem.forward(continuous: torch.Tensor, categorical: dict[str, torch.Tensor]) -> torch.Tensor
22. class ModalityFusion(nn.Module)
23. ModalityFusion.__init__(feature_dim: int, modality_vocab: int = 2)
24. ModalityFusion.forward(tracker_features: torch.Tensor, calo_features: torch.Tensor, tracker_mask: torch.Tensor, calo_mask: torch.Tensor, modality_id: torch.Tensor) -> torch.Tensor
25. Recommendation: do not mix stems into model.py directly; make them independently testable.

26. In /mnt/home/ewulff/repositories/collider-fm/src/collider_fm/views.py:
27. SSLView — TypedDict or dataclass with coord, continuous features, categorical features, modality_id, point_id, offset, grid_size, view_type, and visible_point_mask.
28. ViewSet — TypedDict with teacher_global, student_global, student_local, and student_masked entries, where some values are lists.
29. build_ssl_views(events: Sequence[RawColliderEvent], device: torch.device, max_tracker_hits: int, max_calo_hits: int, config: SSLViewConfig) -> ViewSet
30. build_global_view(points: MultimodalPointBatch, config: SSLViewConfig) -> SSLView
31. build_local_view(points: MultimodalPointBatch, anchor_strategy: str, config: SSLViewConfig) -> SSLView
32. build_masked_view(points: MultimodalPointBatch, config: SSLViewConfig) -> SSLView
33. apply_phi_rotation(view: SSLView, angle: float) -> SSLView
34. apply_tracker_time_jitter(view: SSLView, scale: float) -> SSLView
35. apply_calo_energy_jitter(view: SSLView, scale: float) -> SSLView
36. mask_spatial_patches(view: SSLView, config: SSLViewConfig) -> SSLView
37. Recommendation: preserve point_id through every transformation and make that a hard invariant in tests.

38. In /mnt/home/ewulff/repositories/collider-fm/src/collider_fm/model.py:
39. DistillationOutputs — NamedTuple or dataclass with student_logits, teacher_logits, student_point_features, teacher_point_features, student_visible_ids, teacher_visible_ids, and optional pooled summaries.
40. class MultimodalPandaSSL(nn.Module)
41. MultimodalPandaSSL.__init__(tracker_stem: TrackerStem, calo_stem: CaloStem, backbone_cls: type[nn.Module], num_prototypes: int = 4096, projection_dim: int = 256, prediction_dim: int = 256, grid_size: float = 10.0, temp_student: float = 0.1, temp_teacher: float = 0.04, center_momentum: float = 0.9)
42. MultimodalPandaSSL.encode_view(view: SSLView, use_teacher: bool = False) -> dict[str, torch.Tensor]
43. MultimodalPandaSSL.forward(view_set: ViewSet) -> DistillationOutputs
44. MultimodalPandaSSL.local_loss(outputs: DistillationOutputs) -> torch.Tensor
45. MultimodalPandaSSL.masked_loss(outputs: DistillationOutputs) -> torch.Tensor
46. MultimodalPandaSSL.total_loss(outputs: DistillationOutputs, masked_weight: float = 0.5) -> torch.Tensor
47. MultimodalPandaSSL.update_teacher(momentum: float) -> None
48. MultimodalPandaSSL.update_center(teacher_logits: Sequence[torch.Tensor]) -> None
49. create_small_multimodal_model(device: torch.device | None = None, backbone_kwargs: Mapping[str, Any] | None = None) -> MultimodalPandaSSL
50. Recommendation: keep the current PandaSelfDistillation class only as a transitional scaffold until train and tests switch over.

51. In /mnt/home/ewulff/repositories/collider-fm/src/collider_fm/diagnostics.py:
52. summarize_prototype_usage_by_modality(logits: torch.Tensor, modality_id: torch.Tensor) -> dict[str, Any]
53. summarize_mixed_neighborhoods(coord: torch.Tensor, modality_id: torch.Tensor, radius: float) -> dict[str, Any]
54. encode_view(...) should align with MultimodalPandaSSL.encode_view rather than copying private model logic.

55. In /mnt/home/ewulff/repositories/collider-fm/scripts/train.py:
56. build_training_model(args: argparse.Namespace, device: torch.device) -> MultimodalPandaSSL
57. save_checkpoint(model: MultimodalPandaSSL, optimizer: AdamW, epoch: int, step: int, run_dir: Path) -> Path
58. load_checkpoint_if_requested(model: MultimodalPandaSSL, optimizer: AdamW | None, checkpoint_path: str | None) -> dict[str, Any]
59. run_epoch(...) -> tuple[float, dict[str, float]]
60. Recommendation: log local and masked losses separately from the start.

**Phase-by-phase execution order optimized for minimal breakage**
1. Phase A — Introduce the new data and feature contract without touching the current training path.
2. Add /mnt/home/ewulff/repositories/collider-fm/src/collider_fm/features.py and corresponding tests first. Keep /mnt/home/ewulff/repositories/collider-fm/src/collider_fm/data.py working as-is while adding richer raw field preservation and model-safe feature builders beside it.
3. Extend /mnt/home/ewulff/repositories/collider-fm/tests/test_data.py and add /mnt/home/ewulff/repositories/collider-fm/tests/test_features.py to lock down detector metadata preservation, point_id assignment, modality_id creation, and exclusion of truth-only fields from model-facing tensors.
4. Why first: this creates a stable contract the rest of the architecture can target without breaking the existing simplified model immediately.

5. Phase B — Add stems as isolated modules.
6. Create /mnt/home/ewulff/repositories/collider-fm/src/collider_fm/stems.py and /mnt/home/ewulff/repositories/collider-fm/tests/test_stems.py.
7. Implement TrackerStem, CaloStem, and ModalityFusion against synthetic tensors before integrating them into the shared backbone.
8. Why second: stems are easy to validate in isolation and do not require the full SSL pipeline to exist.

9. Phase C — Introduce the new multimodal model in parallel with the existing model.
10. Add MultimodalPandaSSL to /mnt/home/ewulff/repositories/collider-fm/src/collider_fm/model.py without deleting PandaSelfDistillation yet.
11. Reuse the current Panda/PTv3 backbone wrapper from /mnt/home/ewulff/repositories/collider-fm/src/collider_fm/_panda/model_base.py and nearby modules, but change only the input preparation and output handling on the new model path.
12. Add model tests that validate a synthetic multimodal batch can pass through the new stems, backbone, projection path, and teacher/student update logic.
13. Why third: parallel model paths let the repo stay runnable while the new path stabilizes.

14. Phase D — Replace the current view pipeline with the new SSL view builder.
15. Rework /mnt/home/ewulff/repositories/collider-fm/src/collider_fm/views.py to generate structured teacher global, student local, and student masked views over MultimodalPointBatch inputs.
16. Add tests for detector-aware crops, masking, point_id preservation, and modality balance before wiring the new views into train.py.
17. Why fourth: the new model can be smoke-tested on direct synthetic inputs before the crop and mask complexity lands.

18. Phase E — Switch the training script to the new path.
19. Update /mnt/home/ewulff/repositories/collider-fm/scripts/train.py to build the multimodal model, call build_ssl_views, compute local and masked losses separately, and save resumable checkpoints.
20. Keep /mnt/home/ewulff/repositories/collider-fm/scripts/smoke_test_model.py lightweight; update it only after train.py proves the new forward path works.
21. Why fifth: this limits operational churn while the core architecture is still moving.

22. Phase F — Expand diagnostics and smoke validation.
23. Update /mnt/home/ewulff/repositories/collider-fm/src/collider_fm/diagnostics.py and /mnt/home/ewulff/repositories/collider-fm/scripts/plot_diagnostics.py to inspect tracker-only, calo-only, and mixed-neighborhood prototype behavior.
24. Re-read and then update /mnt/home/ewulff/repositories/collider-fm/notebooks/plot_diagnostics_explorer.ipynb only after the shared diagnostics helpers are stable.
25. Why sixth: diagnostics should follow the stable model and view interfaces, not lead them.

26. Phase G — Retire or quarantine the old simplified path.
27. Once train.py, smoke_test_model.py, and tests use MultimodalPandaSSL, either remove the old six-feature PandaSelfDistillation path or label it clearly as legacy scaffold code.
28. Update /mnt/home/ewulff/repositories/collider-fm/README.md, /mnt/home/ewulff/repositories/collider-fm/PLAN.md, and /mnt/home/ewulff/repositories/collider-fm/HPC.md last.
29. Why last: docs should describe the architecture that actually survived implementation.

**Verification**
1. After Phase A, verify raw events preserve detector hierarchy and model-facing tensors exclude truth-only fields.
2. After Phase B, verify TrackerStem and CaloStem produce the expected shared-width outputs on synthetic batches.
3. After Phase C, verify the new multimodal model can run a synthetic forward pass and produce finite prototype losses with EMA updates.
4. After Phase D, verify point IDs survive all crops and masks and that teacher/student correspondence is exact for the visible subsets.
5. After Phase E, run a short GPU training smoke job with checkpoint save and reload.
6. After Phase F, verify tracker-only and calo-only prototype usage are both non-degenerate on a small cached sample.

**Decisions**
- Preserve the current package root and Panda runtime subpackage, but introduce new feature and stem modules instead of overloading data.py and model.py further.
- Add the new multimodal path alongside the existing scaffold first, then migrate scripts and tests, then remove legacy assumptions.
- Keep the notebook subordinate to shared Python modules because it has already drifted and should not become the source of truth.
- Keep downstream task heads out of scope for this implementation sequence.

**Further Considerations**
1. If point-level exact matching proves too heavy for the first working checkpoint, preserve point_id now and allow the first loss implementation to reduce over matched visible subsets rather than full dense point-wise correspondence.
2. If categorical vocab sizes such as surface_id are too large or sparse initially, stage them behind detector, volume_id, and layer_id rather than blocking the first stem implementation.
3. If model.py becomes crowded while adding local and masked loss paths, split loss and view-role bookkeeping into ssl.py after Phase C rather than before.

# 0001 Review

## Current Decision

Keep `vfm_topology_scorer` as the v1 integration path. `mock_l1` validates the scoring plumbing, `cached_edge_l1` validates the offline cache contract, and the optional DINOv2 builder now validates that real VFM patch-token caches can be generated in this environment. Training has not yet consumed DINOv2 feature maps, so this is still an integration milestone rather than a VFM quality claim.

## Findings

- `vfm_topology_scorer` now returns the same `(importance_score, pruning_score)` contract as `fastgs_photometric`.
- The scorer uses SH0 rendering when `vfm_use_albedo_sh0=true`, creates a backend pixel error map, thresholds it into `metric_map`, and lets `render_fastgs(..., get_flag=True)` accumulate per-Gaussian hits.
- `fastgs_photometric` normalization was hardened to avoid NaNs when a metric map or pruning score has zero dynamic range.
- On the 2026-04-28 bicycle smoke run, mock v1 was metric-neutral against the baseline: PSNR 20.3459 vs 20.3464, SSIM 0.4294 vs 0.4294, LPIPS 0.6010 vs 0.6021.
- The mock v1 run produced 78,375 Gaussians vs 78,633 for baseline, so the conservative fusion did not explode point count in the short validation.
- `vfm_gs.cli.build_vfm_cache` now writes a manifest keyed by `Camera.image_name`, with backend name, source shape, cache shape, dtype, normalization, and per-entry checksum.
- `cached_edge_l1` completed the same train/render/metrics smoke path: PSNR 20.3265, SSIM 0.4291, LPIPS 0.6005, 78,605 Gaussians.
- Training output `cfg_args` now records optimization and pipeline parameters in addition to model parameters, so scorer/backend settings are recoverable from each run directory.
- `npz_uint8` compact storage reduced the bicycle edge cache from about 189MB to 35MB, and `vfm_gs.cli.validate_vfm_cache` passed checksum/source-image validation.
- The compact run completed train/render/metrics with 78,682 Gaussians, PSNR 20.1588, SSIM 0.4275, LPIPS 0.5993. The larger PSNR drop suggests edge quantization can move early densification decisions.
- Cached backends now run a training preflight before Scene construction. Good caches pass early; missing manifests fail before camera loading or densification.
- `vfm_gs.cli.vfm_backend_probe` records runtime compatibility and estimates DINOv2 cache sizes. On this machine, ViT-S/14 and ViT-B/14 at 518-640px cache width are the safest real VFM candidates.
- `vfm_gs.cli.build_vfm_cache` now supports optional `dinov2_vits14` and `dinov2_vitb14` cache generation. DINOv2 caches store L2-normalized patch-token grids with `feature=dinov2_patchtokens`.
- A 4-image ViT-S/14 smoke cache at `max_width=224` wrote 4 `npy_float16` entries, first entry shape `10x16x384`, total size about 500K, and passed `vfm_gs.cli.validate_vfm_cache`.
- Remote torch.hub listing hit GitHub rate limits during probing, but a local clone at `output/0001/external/dinov2` plus pretrained weight download worked. The builder reports a clear hint to pass `--dinov2_repo` when remote torch.hub loading fails.
- PyTorch 1.12.1 lacks the public SDPA API expected by current DINOv2, so the cache builder includes a narrow compatibility shim over the private 1.12 attention function. This should remain isolated to optional DINOv2 cache generation.

## Limitations

- `mock_l1` is deliberately not a real visual foundation model signal.
- `cached_edge_l1` is also a proxy; it only tests cache mechanics and edge-alignment behavior.
- DINOv2 cache generation is proven only as an offline artifact path. The scorer does not yet align rendered features with DINO patch tokens during training.
- The smoke run uses `-r 8` and 220 iterations, so it only validates integration health, not final reconstruction quality.
- Compact storage helps disk use, but `npz_uint8` is not proven metric-neutral. Keep float32 and compact cache variants available for ablation.
- The DINOv2 smoke used `--limit 4` and `max_width=224`; full-scene cache time, disk use, and training-time scorer overhead still need measurement.

## Next Version Plan

1. Add a DINO feature-map scorer path that consumes `dinov2_patchtokens` caches and compares them with a deterministic projection of rendered RGB or SH0 features.
2. Decide the first projection strategy explicitly: start with fixed color/edge descriptors pooled to the DINO patch grid, then add a learned lightweight adapter only if the fixed path is stable.
3. Build a full-scene `dinov2_vits14` cache at `max_width=518` or `640`, record cache time and disk use, and keep the `max_width=224` smoke as a fast regression target.
4. Run a longer bicycle ablation with matched baseline, `cached_edge_l1`, compact edge, and DINOv2 scorer schedules.
5. Compare PSNR/SSIM/LPIPS, Gaussian count, render FPS, cache build time, scorer overhead, cache size, and visual floaters before promoting DINOv2 to the default VFM backend.

# 0001 Review

## Current Decision

Keep `vfm_topology_scorer` as the v1 integration path. `mock_l1` validates the scoring plumbing, and `cached_edge_l1` now validates the offline cache contract. Neither backend proves VFM quality gains yet; they are stepping stones toward a real cached VFM backend.

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

## Limitations

- `mock_l1` is deliberately not a real visual foundation model signal.
- `cached_edge_l1` is also a proxy; it only tests cache mechanics and edge-alignment behavior.
- The smoke run uses `-r 8` and 220 iterations, so it only validates integration health, not final reconstruction quality.
- Compact storage helps disk use, but `npz_uint8` is not proven metric-neutral. Keep float32 and compact cache variants available for ablation.

## Next Version Plan

1. Add an optional DINOv2 cache builder using the existing manifest contract, starting with `dinov2_vits14`.
2. Keep the builder dependency-isolated: fail with a clear message when torch.hub weights or optional dependencies are unavailable.
3. Add a DINO feature-map scorer path that compares cached GT features against SH0 rendered features after aligned resizing/projection.
4. Run a longer bicycle ablation with matched baseline, `cached_edge_l1`, and DINOv2 schedules.
5. Compare PSNR/SSIM/LPIPS, Gaussian count, render FPS, cache build time, scorer overhead, cache size, and visual floaters.

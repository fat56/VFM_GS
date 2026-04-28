# 0001 Review

## Current Decision

Keep `vfm_topology_scorer` as the v1 integration path, but treat the current `mock_l1` backend only as a pipeline validator. It proves the score path can be swapped through the registry and can survive densification, rendering, and metrics. It does not prove VFM quality gains yet.

## Findings

- `vfm_topology_scorer` now returns the same `(importance_score, pruning_score)` contract as `fastgs_photometric`.
- The scorer uses SH0 rendering when `vfm_use_albedo_sh0=true`, creates a backend pixel error map, thresholds it into `metric_map`, and lets `render_fastgs(..., get_flag=True)` accumulate per-Gaussian hits.
- `fastgs_photometric` normalization was hardened to avoid NaNs when a metric map or pruning score has zero dynamic range.
- On the 2026-04-28 bicycle smoke run, mock v1 was metric-neutral against the baseline: PSNR 20.3459 vs 20.3464, SSIM 0.4294 vs 0.4294, LPIPS 0.6010 vs 0.6021.
- The mock v1 run produced 78,375 Gaussians vs 78,633 for baseline, so the conservative fusion did not explode point count in the short validation.

## Limitations

- `mock_l1` is deliberately not a real visual foundation model signal.
- The smoke run uses `-r 8` and 220 iterations, so it only validates integration health, not final reconstruction quality.
- `vfm_cache_dir` is exposed for config compatibility but no cache manifest or real feature reader is implemented yet.
- Train-time provenance is currently stronger in this docs record than in `cfg_args`, because the legacy output config logs model params only.

## Next Version Plan

1. Define a cache manifest format keyed by `Camera.image_name`, including backend name, source image size, tensor shape, normalization, and checksum.
2. Add an offline cache builder CLI for one real low-risk backend, preferably an edge/depth proxy first and DINOv2 only after dependency and memory checks.
3. Replace `mock_l1` in `configs/experiments/0001_vfm_topology_scorer.yaml` with a cached backend, while keeping `mock_l1` as the CI/smoke fallback.
4. Log scorer name, VFM backend, threshold, weight, and densification summary into each training output directory.
5. Run a longer bicycle ablation with matched baseline and VFM schedules, then compare PSNR/SSIM/LPIPS, Gaussian count, render FPS, and visual floaters.

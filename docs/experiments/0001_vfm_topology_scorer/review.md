# 0001 Review

## Current Decision

Keep `vfm_topology_scorer` as the v1 integration path. `mock_l1` validates the scoring plumbing, `cached_edge_l1` validates the offline cache contract, and `dinov2_token_edge_l1` now validates that training can consume real DINOv2 patch-token caches. The DINOv2 path is still a conservative token-edge projection, not a final semantic feature scorer. Full 30k runs, not short smoke metrics, are the decision gate from here.

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
- A full-scene ViT-S/14 cache at `max_width=224` wrote 194 `npy_float16` entries to `output/0001/vfm_cache/bicycle_dinov2_vits14`, took 15s to build, occupied about 24M, and passed validation.
- `dinov2_token_edge_l1` projects cached DINO patch tokens into a scalar token-edge topology map, pools SH0-rendered luminance edges to the same grid, and returns an upsampled pixel error map for the existing metric-map scorer path.
- The DINO token-edge smoke run completed train/render/metrics: PSNR 20.2913, SSIM 0.4272, LPIPS 0.6006, 77,761 Gaussians, 1.72s training time, 410.94 FPS on rendered test frames.
- DINO scorer preflight now accepts DINO cache manifests (`dinov2_vits14` or `dinov2_vitb14`) while rejecting mismatched cache features/backends before Scene construction.
- A matched 30k `-r 8` ablation is now the main quality signal. Baseline reached PSNR 26.7032, SSIM 0.8067, LPIPS 0.2278, 240,394 Gaussians, and 334.36 FPS.
- Compact cached edge improved the 30k run to PSNR 26.8864, SSIM 0.8229, LPIPS 0.1972, but grew to 408,925 Gaussians and 196.43 FPS.
- DINO token-edge gave the best 30k metrics: PSNR 27.0577, SSIM 0.8345, LPIPS 0.1767, with 490,832 Gaussians and 193.46 FPS.
- The full-run result reverses the short-run impression: DINO token-edge looked neutral/slightly worse at 220 iterations but becomes the strongest metric variant after normal densification has time to operate.
- A first budget-control probe with `vfm_loss_thresh=0.75` and `vfm_weight=0.10` did not bring VFM point counts near baseline. Edge stayed at 409,028 Gaussians; DINO dropped to 422,506 Gaussians but remained about 76% above baseline.
- The budget-control probe kept good metrics, but below default DINO: DINO t075/w010 reached PSNR 26.9586, SSIM 0.8258, LPIPS 0.1935.
- `vfm_importance_weight` now separates densification strength from pruning fusion. The default is `1.0`, preserving previous behavior.
- With `vfm_importance_weight=0.25`, edge reached PSNR 26.9439, SSIM 0.8244, LPIPS 0.1958, and 413,301 Gaussians; DINO reached PSNR 26.9261, SSIM 0.8259, LPIPS 0.1928, and 418,073 Gaussians.
- Explicit importance weighting reduced the default DINO point count by about 14.8%, but still did not reach a baseline-like budget. Edge remained effectively above 400k Gaussians.
- `vfm_importance_mode` now supports `max`, `weighted`, and `rgb_only`. The default `max` preserves previous behavior.
- 30k `rgb_only` ablations completed. Edge reached PSNR 26.9574, SSIM 0.8243, LPIPS 0.1961, and 413,914 Gaussians; DINO reached PSNR 26.9310, SSIM 0.8237, LPIPS 0.1962, and 413,223 Gaussians.
- `rgb_only` did not recover a baseline-like Gaussian count. VFM pruning-score fusion alone can still preserve or reshape enough points to keep final count about 72% above baseline.

## Limitations

- `mock_l1` is deliberately not a real visual foundation model signal.
- `cached_edge_l1` is also a proxy; it only tests cache mechanics and edge-alignment behavior.
- `dinov2_token_edge_l1` consumes DINO patch tokens, but it compares scalar topology projections rather than full semantic feature vectors.
- The 220-iteration smoke run validates integration health, not final reconstruction quality. It should not drive scorer selection now that 30k runs are cheap enough.
- Compact storage helps disk use, but `npz_uint8` is not proven metric-neutral. Keep float32 and compact cache variants available for ablation.
- The current DINO cache is built at `max_width=224`; full `max_width=518` or `640` cache time, disk use, and scorer behavior still need measurement.
- The best 30k DINO result is not budget-controlled: it used about 2.04x the baseline Gaussian count. The next result must separate feature-signal quality from simply allowing denser reconstructions.
- Existing knobs do not provide full budget control. `vfm_weight` affects pruning fusion, `vfm_importance_weight` affects direct VFM densification strength, and `vfm_importance_mode=rgb_only` can disable direct VFM densification, but none of them match the baseline point count.
- `target_gaussian_count` is a new final-prune control, but its quality behavior is not measured yet. It must be validated in 30k runs before it is treated as the budget-matching answer.
- The current 30k quality gains remain entangled with a larger Gaussian budget. A fair scorer comparison needs target-count results.

## Next Version Plan

1. Run cached edge and DINO token-edge at 30k `-r 8` with `--target_gaussian_count 240394`, matching the baseline count.
2. Compare budget-matched PSNR/SSIM/LPIPS/FPS against the baseline and against the unpruned VFM runs.
3. Add a no-effect control mode such as `vfm_weight=0` + `vfm_importance_mode=rgb_only` to measure VFM scorer overhead without changing pruning or densification decisions.
4. Keep 30k `-r 8` as the minimum quality gate; use 220 iterations only for smoke checks after code changes.
5. After budget matching is available, build a full-scene `dinov2_vits14` cache at `max_width=518` or `640`, then compare it with the `max_width=224` cache.

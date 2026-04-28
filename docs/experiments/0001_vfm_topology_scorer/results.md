# 0001 Results

## 2026-04-28 Mock v1 Smoke

Dataset: `datasets/mipnerf360/bicycle`, test split, `-r 8`, 220 iterations, `densify_from_iter=50`, `densification_interval=50`.

| Artifact | Variant | Scorer | Backend | PSNR | SSIM | LPIPS | Train Time | Gaussian Count | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `output/0001/baseline_bicycle_smoke` | `fastgs_baseline` | `fastgs_photometric` | n/a | 20.3464 | 0.4294 | 0.6021 | 0.96s | 78,633 | Control run |
| `output/0001/vfm_mock_bicycle_smoke` | `0001_vfm_topology_scorer` | `vfm_topology_scorer` | `mock_l1` | 20.3459 | 0.4294 | 0.6010 | 1.05s | 78,375 | SH0 mock VFM branch validated |
| `output/0001/vfm_cached_edge_bicycle_smoke` | `0001_vfm_topology_cached_edge` | `vfm_topology_scorer` | `cached_edge_l1` | 20.3265 | 0.4291 | 0.6005 | 1.17s | 78,605 | Offline cache read path validated |
| `output/0001/vfm_cached_edge_compact_bicycle_smoke` | `0001_vfm_topology_cached_edge_compact` | `vfm_topology_scorer` | `cached_edge_l1` / `npz_uint8` | 20.1588 | 0.4275 | 0.5993 | 1.16s | 78,682 | Compact cache and validation path validated |

## Interpretation

- The mock VFM scorer completed train, render, and metric evaluation without shape, device, or optimizer-state failures.
- Quality metrics are effectively tied in this short smoke run. This is expected because `mock_l1` is a plumbing proxy, not a real VFM signal.
- The mock branch added about 9% measured training time in this tiny run, but the number is dominated by short-run overhead and should be remeasured on a longer schedule.
- Gaussian count stayed close to baseline, which suggests the conservative `max(rgb_importance, vfm_importance)` and weighted pruning fusion did not destabilize densification.
- The cached edge proxy produced a small PSNR drop and a small LPIPS improvement in the smoke run. This is not a quality claim; it confirms that cached GT features can be read, resized, compared to SH0 render features, and fused without breaking training.
- Cache artifact: `output/0001/vfm_cache/bicycle_edge`, 194 entries, about 189MB when built from `images_8` with `--max_width 640`.
- Compact cache artifact: `output/0001/vfm_cache/bicycle_edge_u8`, 194 entries, about 35MB with `--storage npz_uint8`; `vfm_gs.cli.validate_vfm_cache` passed with checksum and source-image checks.
- The compact cache run stayed stable but shifted PSNR downward more than the float32 cache. That points to quantization changing thresholded edge masks enough to affect early densification, so longer runs should compare cache precision as an ablation rather than assuming compact storage is metric-neutral.

## 2026-04-28 Cache Preflight

- `vfm_gs.cli.validate_vfm_cache` passed on `output/0001/vfm_cache/bicycle_edge_u8` with 194 `cached_edge_l1` entries.
- `vfm_topology_scorer.preflight` passed on the same compact cache before Scene construction.
- A negative train-entry check with `--vfm_cache_dir output/0001/vfm_cache/does_not_exist` failed before camera loading with a structured `VFM cache preflight failed` error.

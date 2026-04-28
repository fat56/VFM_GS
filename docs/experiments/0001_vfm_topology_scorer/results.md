# 0001 Results

## 2026-04-28 Mock v1 Smoke

Dataset: `datasets/mipnerf360/bicycle`, test split, `-r 8`, 220 iterations, `densify_from_iter=50`, `densification_interval=50`.

| Artifact | Variant | Scorer | Backend | PSNR | SSIM | LPIPS | Train Time | Gaussian Count | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `output/0001/baseline_bicycle_smoke` | `fastgs_baseline` | `fastgs_photometric` | n/a | 20.3464 | 0.4294 | 0.6021 | 0.96s | 78,633 | Control run |
| `output/0001/vfm_mock_bicycle_smoke` | `0001_vfm_topology_scorer` | `vfm_topology_scorer` | `mock_l1` | 20.3459 | 0.4294 | 0.6010 | 1.05s | 78,375 | SH0 mock VFM branch validated |
| `output/0001/vfm_cached_edge_bicycle_smoke` | `0001_vfm_topology_cached_edge` | `vfm_topology_scorer` | `cached_edge_l1` | 20.3265 | 0.4291 | 0.6005 | 1.17s | 78,605 | Offline cache read path validated |

## Interpretation

- The mock VFM scorer completed train, render, and metric evaluation without shape, device, or optimizer-state failures.
- Quality metrics are effectively tied in this short smoke run. This is expected because `mock_l1` is a plumbing proxy, not a real VFM signal.
- The mock branch added about 9% measured training time in this tiny run, but the number is dominated by short-run overhead and should be remeasured on a longer schedule.
- Gaussian count stayed close to baseline, which suggests the conservative `max(rgb_importance, vfm_importance)` and weighted pruning fusion did not destabilize densification.
- The cached edge proxy produced a small PSNR drop and a small LPIPS improvement in the smoke run. This is not a quality claim; it confirms that cached GT features can be read, resized, compared to SH0 render features, and fused without breaking training.
- Cache artifact: `output/0001/vfm_cache/bicycle_edge`, 194 entries, about 189MB when built from `images_8` with `--max_width 640`.

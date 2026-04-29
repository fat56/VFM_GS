# 0001 Results

## 2026-04-28 Mock v1 Smoke

Dataset: `datasets/mipnerf360/bicycle`, test split, `-r 8`, 220 iterations, `densify_from_iter=50`, `densification_interval=50`.

| Artifact | Variant | Scorer | Backend | PSNR | SSIM | LPIPS | Train Time | Gaussian Count | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `output/0001/baseline_bicycle_smoke` | `fastgs_baseline` | `fastgs_photometric` | n/a | 20.3464 | 0.4294 | 0.6021 | 0.96s | 78,633 | Control run |
| `output/0001/vfm_mock_bicycle_smoke` | `0001_vfm_topology_scorer` | `vfm_topology_scorer` | `mock_l1` | 20.3459 | 0.4294 | 0.6010 | 1.05s | 78,375 | SH0 mock VFM branch validated |
| `output/0001/vfm_cached_edge_bicycle_smoke` | `0001_vfm_topology_cached_edge` | `vfm_topology_scorer` | `cached_edge_l1` | 20.3265 | 0.4291 | 0.6005 | 1.17s | 78,605 | Offline cache read path validated |
| `output/0001/vfm_cached_edge_compact_bicycle_smoke` | `0001_vfm_topology_cached_edge_compact` | `vfm_topology_scorer` | `cached_edge_l1` / `npz_uint8` | 20.1588 | 0.4275 | 0.5993 | 1.16s | 78,682 | Compact cache and validation path validated |
| `output/0001/vfm_dinov2_token_edge_bicycle_smoke` | `0001_vfm_topology_dinov2_token_edge` | `vfm_topology_scorer` | `dinov2_token_edge_l1` | 20.2913 | 0.4272 | 0.6006 | 1.72s | 77,761 | DINOv2 token-edge cache consumed by training |

## Interpretation

- The mock VFM scorer completed train, render, and metric evaluation without shape, device, or optimizer-state failures.
- Quality metrics are effectively tied in this short smoke run. This is expected because `mock_l1` is a plumbing proxy, not a real VFM signal.
- The mock branch added about 9% measured training time in this tiny run, but the number is dominated by short-run overhead and should be remeasured on a longer schedule.
- Gaussian count stayed close to baseline, which suggests the conservative `max(rgb_importance, vfm_importance)` and weighted pruning fusion did not destabilize densification.
- The cached edge proxy produced a small PSNR drop and a small LPIPS improvement in the smoke run. This is not a quality claim; it confirms that cached GT features can be read, resized, compared to SH0 render features, and fused without breaking training.
- Cache artifact: `output/0001/vfm_cache/bicycle_edge`, 194 entries, about 189MB when built from `images_8` with `--max_width 640`.
- Compact cache artifact: `output/0001/vfm_cache/bicycle_edge_u8`, 194 entries, about 35MB with `--storage npz_uint8`; `vfm_gs.cli.validate_vfm_cache` passed with checksum and source-image checks.
- The compact cache run stayed stable but shifted PSNR downward more than the float32 cache. That points to quantization changing thresholded edge masks enough to affect early densification, so longer runs should compare cache precision as an ablation rather than assuming compact storage is metric-neutral.
- The first DINOv2-consuming scorer stayed stable and produced slightly lower PSNR/SSIM than baseline in the short run, with LPIPS similar to prior VFM variants. It should be read as a successful real-cache training integration, not as evidence that the token-edge projection is the right quality signal.
- DINO token-edge training time was 1.72s vs 1.17s for cached edge in the same 220-iteration schedule, so the derived DINO projection adds visible but still modest scorer overhead at this tiny scale.

## 2026-04-28 30k Matched Ablation

Dataset: `datasets/mipnerf360/bicycle`, test split, `-r 8`, 30,000 iterations. These runs use the normal FastGS densification schedule and are more informative than the 220-iteration smoke checks.

| Artifact | Variant | Scorer | Backend | PSNR | SSIM | LPIPS | Train Time | Render FPS | Gaussian Count | Output Size | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `output/0001/baseline_bicycle_30k_r8` | `fastgs_baseline` | `fastgs_photometric` | n/a | 26.7032 | 0.8067 | 0.2278 | 116.92s | 334.36 | 240,394 | 82M | Control run |
| `output/0001/vfm_cached_edge_compact_bicycle_30k_r8` | `0001_vfm_topology_cached_edge_compact` | `vfm_topology_scorer` | `cached_edge_l1` / `npz_uint8` | 26.8864 | 0.8229 | 0.1972 | 159.77s | 196.43 | 408,925 | 122M | Edge cache improves metrics but grows point count |
| `output/0001/vfm_dinov2_token_edge_bicycle_30k_r8` | `0001_vfm_topology_dinov2_token_edge` | `vfm_topology_scorer` | `dinov2_token_edge_l1` | 27.0577 | 0.8345 | 0.1767 | 166.11s | 193.46 | 490,832 | 142M | Best metrics, densest reconstruction |

Cache artifacts used:

| Cache | Backend | Entries | Storage | Size | Validation |
|---|---|---:|---|---:|---|
| `output/0001/vfm_cache/bicycle_edge_u8` | `cached_edge_l1` | 194 | `npz_uint8` | 35M | passed |
| `output/0001/vfm_cache/bicycle_dinov2_vits14` | `dinov2_vits14` | 194 | `npy_float16` | 24M | passed |

Interpretation:

- The 220-iteration smoke metrics were too weak to guide quality decisions. They remain useful for integration health, but the 30k runs changed the signal meaningfully.
- `cached_edge_l1` improved over baseline by +0.1832 PSNR, +0.0163 SSIM, and -0.0306 LPIPS, but increased Gaussian count by about 70% and reduced test render FPS by about 41%.
- `dinov2_token_edge_l1` improved over baseline by +0.3544 PSNR, +0.0278 SSIM, and -0.0511 LPIPS, but increased Gaussian count by about 104% and reduced test render FPS by about 42%.
- The DINO token-edge scorer is now a promising direction on full-length low-resolution training, but the gain is entangled with a much larger Gaussian budget. The next comparison needs a budget-controlled run or a stronger final-prune setting before calling it a pure quality improvement.
- `cached_edge_l1` remains a strong deterministic proxy baseline: it is less semantically meaningful than DINO, but it gave a clear metric gain with a smaller cache and lower training overhead than DINO token-edge.

## 2026-04-28 Budget-Control Probe

Dataset and schedule match the 30k ablation above. These runs only override `vfm_loss_thresh=0.75` and `vfm_weight=0.10` to test whether existing knobs can bring VFM runs closer to the baseline Gaussian count.

| Artifact | Backend | PSNR | SSIM | LPIPS | Train Time | Render FPS | Gaussian Count | Output Size | Delta vs Baseline Count | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `output/0001/vfm_cached_edge_t075_w010_bicycle_30k_r8` | `cached_edge_l1` / `npz_uint8` | 26.9372 | 0.8238 | 0.1979 | 166.81s | 337.93 | 409,028 | 122M | +70.2% | Existing knobs did not reduce edge point count |
| `output/0001/vfm_dinov2_token_edge_t075_w010_bicycle_30k_r8` | `dinov2_token_edge_l1` | 26.9586 | 0.8258 | 0.1935 | 166.60s | 377.72 | 422,506 | 125M | +75.8% | Lower than default DINO point count, but still far above baseline |

Interpretation:

- Raising `vfm_loss_thresh` and lowering `vfm_weight` is not sufficient budget control. `vfm_weight` currently affects pruning-score fusion, while densification still uses `max(rgb_importance, vfm_importance)`.
- The DINO budget probe reduced Gaussian count by about 14% relative to the default DINO 30k run, but it also gave back part of the quality gain.
- Edge proxy point count was effectively unchanged under these knobs. The next implementation should separate VFM densification strength from VFM pruning strength, for example with a dedicated `vfm_importance_weight` or `vfm_importance_mode`.
- Render FPS varied enough that it should be interpreted together with Gaussian count and repeated where the budget comparison is close.

## 2026-04-28 Explicit Importance Weight Probe

Code change: add `vfm_importance_weight`, defaulting to `1.0` for backward compatibility. It scales VFM densification counts before `max(rgb_importance, vfm_importance)` and is independent from `vfm_weight`, which continues to control pruning-score fusion.

Dataset and schedule match the 30k ablation above. These runs use `vfm_importance_weight=0.25` with each variant's default `vfm_loss_thresh` and `vfm_weight`.

| Artifact | Backend | PSNR | SSIM | LPIPS | Train Time | Render FPS | Gaussian Count | Output Size | Delta vs Baseline Count | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `output/0001/vfm_cached_edge_i025_bicycle_30k_r8` | `cached_edge_l1` / `npz_uint8` | 26.9439 | 0.8244 | 0.1958 | 166.85s | 168.15 | 413,301 | 123M | +71.9% | Explicit importance scaling did not reduce edge count |
| `output/0001/vfm_dinov2_token_edge_i025_bicycle_30k_r8` | `dinov2_token_edge_l1` | 26.9261 | 0.8259 | 0.1928 | 159.84s | 279.87 | 418,073 | 124M | +73.9% | Reduced default DINO count by 14.8%, but not enough for budget matching |

Interpretation:

- `vfm_importance_weight=0.25` is useful but insufficient as the only budget lever.
- DINO i0.25 gives a better Gaussian budget than default DINO, but also gives back most of the default DINO metric advantage.
- Edge remains insensitive to this control, which suggests its extra points are not easily suppressed by simple post-count scaling.
- The next implementation should add a harder mode such as `vfm_importance_mode=rgb_only|max|weighted`, where `rgb_only` lets VFM affect pruning but not densification.

## 2026-04-28 Importance Mode Probe

Code change: add `vfm_importance_mode`, defaulting to `max` for backward compatibility. Available modes are `max`, `weighted`, and `rgb_only`. In `rgb_only`, VFM still contributes to pruning-score fusion through `vfm_weight`, but densification importance uses the RGB/FastGS importance counts directly.

Dataset and schedule match the 30k ablation above. These runs use `vfm_importance_mode=rgb_only` with each variant's default `vfm_loss_thresh`, `vfm_weight`, and `vfm_importance_weight`.

| Artifact | Backend | PSNR | SSIM | LPIPS | Train Time | Render FPS | Gaussian Count | Output Size | Delta vs Baseline Count | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `output/0001/vfm_cached_edge_rgb_only_bicycle_30k_r8` | `cached_edge_l1` / `npz_uint8` | 26.9574 | 0.8243 | 0.1961 | 158.76s | 282.64 | 413,914 | 123M | +72.2% | Disabling direct VFM densification did not reduce edge count |
| `output/0001/vfm_dinov2_token_edge_rgb_only_bicycle_30k_r8` | `dinov2_token_edge_l1` | 26.9310 | 0.8237 | 0.1962 | 159.52s | 185.85 | 413,223 | 123M | +71.9% | Lower than default DINO count, but still far above baseline |

Interpretation:

- `rgb_only` proves that short 220-iteration scores are too weak for choosing scorer behavior. The full 30k runs expose model-budget shifts that short runs cannot see.
- Direct VFM densification is not the only source of extra Gaussians. VFM pruning-score fusion alone can preserve or reshape enough points that final count remains about 1.72x baseline.
- DINO `rgb_only` reduces the default DINO point count by about 15.8%, but gives back most of the default DINO metric advantage.
- Edge `rgb_only` is effectively the same budget as previous edge probes, which means simple importance-mode controls are not sufficient for budget matching.
- The next version needs an explicit Gaussian budget mechanism, such as a final prune/target-count pass or a hard cap in the densify/prune loop. A pure signal-quality claim should wait until VFM and baseline are compared under a matched point budget.

## 2026-04-28 Target Budget Negative Probe

Code change: add `target_gaussian_count`, a final-prune budget control. The first implementation pruned the highest pruning-score Gaussians down to the target count. It matched the budget exactly but destroyed quality, so this is recorded as a negative result and the implementation was changed to prune the lowest-score Gaussians first.

Dataset and schedule match the 30k ablation above. The target count is the baseline 30k count: 240,394 Gaussians.

| Artifact | Backend | Target Strategy | PSNR | SSIM | LPIPS | Train Time | Render FPS | Gaussian Count | Output Size | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `output/0001/vfm_cached_edge_budget240394_bicycle_30k_r8` | `cached_edge_l1` / `npz_uint8` | prune highest score | 11.1494 | 0.2037 | 0.6308 | 155.51s | 143.74 | 240,394 | 79M | Budget matched, quality collapsed |
| `output/0001/vfm_dinov2_token_edge_budget240394_bicycle_30k_r8` | `dinov2_token_edge_l1` | prune highest score | 10.2215 | 0.1360 | 0.6769 | 162.33s | 406.19 | 240,394 | 76M | Budget matched, quality collapsed |

Interpretation:

- Exact count matching is not enough. The prune ordering matters more than the count target.
- Bulk-pruning highest-score Gaussians removes hard or high-error structure that the reconstruction still needs.
- `target_gaussian_count` should prune the lowest support/score Gaussians first for budget matching. High-score pruning can remain useful only as a small outlier-removal rule, not as the main budget cutter.

## 2026-04-28 Low-Score Target Budget Probe

Code change: `target_gaussian_count` now prunes the lowest pruning/support scores first for bulk budget control. This fixes the score direction from the negative probe above, but still applies the count correction only once after training.

Dataset and schedule match the 30k ablation above. The target count is the baseline 30k count: 240,394 Gaussians.

| Artifact | Backend | Target Strategy | PSNR | SSIM | LPIPS | Train Time | Render FPS | Gaussian Count | Output Size | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `output/0001/vfm_cached_edge_budget240394_lowscore_bicycle_30k_r8` | `cached_edge_l1` / `npz_uint8` | prune lowest score | 23.7729 | 0.7307 | 0.2685 | 153.06s | 283.32 | 240,394 | 81M | Exact budget, but below baseline quality |
| `output/0001/vfm_dinov2_token_edge_budget240394_lowscore_bicycle_30k_r8` | `dinov2_token_edge_l1` | prune lowest score | 23.5571 | 0.7087 | 0.2797 | 157.34s | 357.34 | 240,394 | 81M | Exact budget, worse than edge under final prune |

Interpretation:

- Low-score pruning avoids the catastrophic failure of high-score pruning, but one-shot final pruning is still too destructive at a baseline-sized target.
- Relative to the baseline 30k run, edge low-score target pruning is -2.9303 PSNR, -0.0760 SSIM, and +0.0407 LPIPS; DINO is -3.1461 PSNR, -0.0980 SSIM, and +0.0519 LPIPS.
- Final-pruned VFM runs have matched point counts and smaller output artifacts, but they no longer retain the unpruned VFM quality gains. This suggests the extra VFM-driven points are not simply removable duplicates after convergence.
- The next budget method should be staged during training, not a single final cut: enforce a soft cap after densification/pruning events, or run a post-prune fine-tune schedule after target pruning.

## 2026-04-29 Staged Target Budget Probe

Code change: add `target_gaussian_staged`, `target_gaussian_stage_margin`, `target_gaussian_stage_start`, and `target_gaussian_stage_interval`. These controls periodically prune the lowest-score Gaussians during training toward `target_gaussian_count * margin`, then still write an exact-budget final PLY.

Dataset and schedule match the 30k ablation above. These runs use `target_gaussian_count=240394`, `target_gaussian_staged=true`, `target_gaussian_stage_margin=1.2`, and `target_gaussian_stage_interval=500`. The staged cap is 288,473 Gaussians.

| Artifact | Backend | PSNR | SSIM | LPIPS | Train Time | Render FPS | Gaussian Count | Output Size | Staged Prunes | Final Prune | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `output/0001/vfm_cached_edge_budget240394_staged120_bicycle_30k_r8` | `cached_edge_l1` / `npz_uint8` | 25.7979 | 0.7747 | 0.2537 | 171.49s | 514.79 | 240,394 | 82M | 24 | 276,611 -> 240,394 | Better than final-only, still below baseline |
| `output/0001/vfm_dinov2_token_edge_budget240394_staged120_bicycle_30k_r8` | `dinov2_token_edge_l1` | 25.3529 | 0.7727 | 0.2493 | 172.39s | 524.63 | 240,394 | 82M | 25 | 288,127 -> 240,394 | Best LPIPS among strict-budget VFM runs, still below baseline |

Interpretation:

- Staged pruning is a clear improvement over one-shot final pruning, gaining +2.0251 PSNR for edge and +1.7958 PSNR for DINO at the same exact 240,394 Gaussian budget.
- Strict 240k budget still underperforms the baseline: edge is -0.9053 PSNR, -0.0320 SSIM, and +0.0259 LPIPS; DINO is -1.3503 PSNR, -0.0340 SSIM, and +0.0215 LPIPS.
- The unpruned VFM quality gain appears to need more than baseline point count on this scene, or it needs recovery training after aggressive budget pruning.
- Next comparison should use a looser staged budget, around 300k Gaussians, before spending effort on a more complex scorer.

## 2026-04-29 300k Staged Target Budget Probe

Dataset and schedule match the 30k ablation above. These runs use `target_gaussian_count=300000`, `target_gaussian_staged=true`, `target_gaussian_stage_margin=1.15`, and `target_gaussian_stage_interval=500`. The staged cap is 345,000 Gaussians.

| Artifact | Backend | PSNR | SSIM | LPIPS | Train Time | Render FPS | Gaussian Count | Output Size | Staged Prunes | Final Prune | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `output/0001/vfm_cached_edge_budget300000_staged115_bicycle_30k_r8` | `cached_edge_l1` / `npz_uint8` | 26.2327 | 0.7866 | 0.2412 | 165.98s | 445.07 | 300,000 | 96M | 23 | 314,704 -> 300,000 | Much closer to baseline, still below on all metrics |
| `output/0001/vfm_dinov2_token_edge_budget300000_staged115_bicycle_30k_r8` | `dinov2_token_edge_l1` | 25.9089 | 0.7925 | 0.2291 | 154.13s | 474.78 | 300,000 | 96M | 24 | 333,724 -> 300,000 | LPIPS nearly matches baseline, PSNR/SSIM remain lower |

Interpretation:

- Loosening the staged target from 240k to 300k recovers another +0.4348 PSNR for edge and +0.5561 PSNR for DINO.
- Edge 300k is still below baseline by -0.4705 PSNR, -0.0201 SSIM, and +0.0134 LPIPS.
- DINO 300k is below baseline by -0.7943 PSNR and -0.0142 SSIM, but LPIPS is only +0.0013 worse than baseline. DINO's perceptual signal remains the most promising under a constrained budget.
- The budget-quality curve has not crossed baseline yet. A 350k staged target should be tested before adding more training machinery; if it still underperforms, the next code path should be post-prune fine-tuning or a less edge-like DINO descriptor scorer.

## 2026-04-28 Cache Preflight

- `vfm_gs.cli.validate_vfm_cache` passed on `output/0001/vfm_cache/bicycle_edge_u8` with 194 `cached_edge_l1` entries.
- `vfm_topology_scorer.preflight` passed on the same compact cache before Scene construction.
- A negative train-entry check with `--vfm_cache_dir output/0001/vfm_cache/does_not_exist` failed before camera loading with a structured `VFM cache preflight failed` error.

## 2026-04-28 Backend Feasibility

- `vfm_gs.cli.vfm_backend_probe` was added and run on the current environment.
- Current runtime: Python 3.10.20, PyTorch 1.12.1+cu116, CUDA 11.6, RTX 4090 D 23.52GB.
- Optional VFM packages are not installed: `transformers`, `timm`, `xformers`, `opencv-python`.
- DINOv2 ViT-S/14 and ViT-B/14 are feasible first targets for offline cache building at `max_width` 518-640; raw float32 features at 640x426 are estimated at 0.40GB and 0.79GB for 194 images.
- Details: `docs/experiments/0001_vfm_topology_scorer/vfm_backend_feasibility.md`.

## 2026-04-28 DINOv2 Cache Smoke

Command shape:

```bash
uv run --active python -m vfm_gs.cli.build_vfm_cache \
  -s datasets/mipnerf360/bicycle \
  -i images_8 \
  -o output/0001/vfm_cache/bicycle_dinov2_vits14_smoke \
  --backend dinov2_vits14 \
  --dinov2_repo output/0001/external/dinov2 \
  --max_width 224 \
  --storage npy_float16 \
  --limit 4
```

| Artifact | Backend | Feature | Entries | Storage | Patch Grid | Size | Validation |
|---|---|---|---:|---|---|---:|---|
| `output/0001/vfm_cache/bicycle_dinov2_vits14_smoke` | `dinov2_vits14` | `dinov2_patchtokens` | 4 | `npy_float16` | first entry `10x16x384` | 500K | passed |

- The official DINOv2 repository was cloned under ignored output state at `output/0001/external/dinov2` and loaded through `torch.hub` with `source="local"`.
- Pretrained ViT-S/14 weights downloaded successfully and produced normalized patch-token maps through `forward_features`.
- The current PyTorch 1.12.1 runtime does not expose public `torch.nn.functional.scaled_dot_product_attention`; the builder adds a narrow compatibility shim over the private 1.12 function so the official DINOv2 code can run in this environment.
- DINOv2 imports warn that `xformers` is unavailable, but the smoke run falls back cleanly and does not require installing it for cache building.
- Regression checks also passed for storage defaults: `cached_edge_l1` defaults to `npy_float32`, while DINOv2 defaults to `npy_float16` when `--storage` is omitted.
- This cache-only smoke validated the real VFM artifact path before the training scorer consumed DINOv2 maps. The token-edge scorer smoke below is the first DINO-consuming training run.

## 2026-04-28 DINOv2 Token-Edge Scorer Smoke

Cache command:

```bash
uv run --active python -m vfm_gs.cli.build_vfm_cache \
  -s datasets/mipnerf360/bicycle \
  -i images_8 \
  -o output/0001/vfm_cache/bicycle_dinov2_vits14 \
  --backend dinov2_vits14 \
  --dinov2_repo output/0001/external/dinov2 \
  --max_width 224
```

Train command used `configs/experiments/0001_vfm_topology_dinov2_token_edge.yaml` with the same `-r 8`, 220-iteration smoke schedule as previous rows.

| Artifact | Value |
|---|---|
| Cache | `output/0001/vfm_cache/bicycle_dinov2_vits14` |
| Cache entries | 194 |
| Cache storage | `npy_float16` |
| Cache size | 24M |
| Cache build time | 15s |
| Validation | `vfm_gs.cli.validate_vfm_cache --backend dinov2_vits14` passed |
| Training preflight | 194 `dinov2_vits14` entries passed before Scene construction |
| Render FPS | 410.94 FPS on 25 test frames |

- `dinov2_token_edge_l1` converts cached DINO patch tokens into a scalar token-edge topology map and compares it with SH0-rendered luminance edges pooled to the same patch grid.
- This is the first scorer variant in the plan that consumes real DINOv2 cache data during training.
- It avoids online DINO inference in the training loop, so the runtime cost is cache loading, token-edge derivation, pooling, and an extra `render_fastgs(..., get_flag=True)` pass.

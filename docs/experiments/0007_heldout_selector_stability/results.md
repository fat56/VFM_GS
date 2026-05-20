# 0007 Held-out Selector Stability 结果

## 当前状态

Round 1 已完成。先用 0006 候选表做 held-out train-view selector：

- selector views：采样 train views 的 even index
- holdout views：采样 train views 的 odd index
- test metrics：复用 0006 candidate table 中的 official test summary

输出：

- `output/0007/heldout_selector/mipnerf360/heldout_candidate_metrics.csv`
- `output/0007/heldout_selector/mipnerf360/heldout_selector_averages.csv`
- `output/0007/heldout_selector/mipnerf360/heldout_selector_recommendations.csv`
- `output/0007/heldout_selector/cross/heldout_candidate_metrics.csv`
- `output/0007/heldout_selector/cross/heldout_selector_averages.csv`
- `output/0007/heldout_selector/cross/heldout_selector_recommendations.csv`

## Round 1：MipNeRF360

平均结果：

| selector | selector PSNR | holdout PSNR | test PSNR | test SSIM | test LPIPS | GS | Δtest PSNR | Δtest SSIM | Δtest LPIPS | ΔGS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 29.0763 | 29.5908 | 27.9590 | 0.820268 | 0.215657 | 1,161,786 |  |  |  |  |
| selector_best_psnr | 29.2309 | 29.7862 | 27.9576 | 0.823637 | 0.207608 | 1,384,316 | -0.0014 | +0.003369 | -0.008049 | +222,530 |
| selector_qcgi | 29.1628 | 29.6909 | 27.9523 | 0.820878 | 0.214215 | 1,200,811 | -0.0067 | +0.000610 | -0.001442 | +39,025 |

逐场景选择：

| scene | best_psnr pick | qcgi pick | qcgi Δholdout PSNR | qcgi Δtest PSNR | qcgi ΔGS |
|---|---|---|---:|---:|---:|
| `bicycle` | `depth_rgb_rerank_start9000` | `baseline` | +0.0000 | +0.0000 | +0 |
| `bonsai` | `depth_rgb_rerank_start9000` | `baseline` | +0.0000 | +0.0000 | +0 |
| `counter` | `baseline` | `baseline` | +0.0000 | +0.0000 | +0 |
| `flowers` | `depth_rgb_rerank_start9000` | `baseline` | +0.0000 | +0.0000 | +0 |
| `garden` | `depth_auto_topk` | `depth_auto_topk` | -0.0042 | +0.0181 | -9,524 |
| `kitchen` | `depth_rgb_rerank_start9000` | `depth_start24000_weight015` | +0.1108 | -0.0745 | +1,083 |
| `room` | `depth_rgb_rerank_start9000` | `depth_auto_topk` | +0.0269 | -0.0691 | +179 |
| `stump` | `depth_rgb_rerank_start9000` | `depth_rgb_rerank_start9000` | +0.7414 | +0.0737 | +353,964 |
| `treehill` | `depth_rgb_rerank_start9000` | `depth_auto_topk` | +0.0262 | -0.0087 | +5,519 |

观察：

- `selector_best_psnr` 仍主要偏向 `depth_rgb_rerank_start9000`，test LPIPS/SSIM 很好，但 Gaussian 增长约 222k，test PSNR 仍低于 baseline。
- `selector_qcgi` 明显更保守，会回退 `bicycle/bonsai/flowers/counter` 到 baseline；但 `kitchen/room/treehill` 的 holdout 正向没能迁移到 test。
- MipNeRF360 的 held-out 均值对 selector 偏乐观：`selector_qcgi` holdout PSNR 高于 baseline +0.1001，但 test PSNR 低于 baseline -0.0067。

## Round 1：DB/Tandt

平均结果：

| selector | selector PSNR | holdout PSNR | test PSNR | test SSIM | test LPIPS | GS | Δtest PSNR | Δtest SSIM | Δtest LPIPS | ΔGS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 32.1879 | 32.0911 | 27.3643 | 0.884519 | 0.206637 | 593,359 |  |  |  |  |
| selector_best_psnr | 32.2251 | 32.0994 | 27.3490 | 0.884418 | 0.207063 | 592,640 | -0.0153 | -0.000101 | +0.000427 | -719 |
| selector_qcgi | 32.2251 | 32.0994 | 27.3490 | 0.884418 | 0.207063 | 592,640 | -0.0153 | -0.000101 | +0.000427 | -719 |

逐场景选择：

| scene | best_psnr pick | qcgi pick | qcgi Δholdout PSNR | qcgi Δtest PSNR | qcgi ΔGS |
|---|---|---|---:|---:|---:|
| `drjohnson` | `depth_topk010` | `depth_topk010` | -0.0079 | -0.0337 | +938 |
| `playroom` | `depth_auto_topk` | `depth_auto_topk` | +0.0325 | -0.0649 | -1,055 |
| `train` | `baseline` | `baseline` | +0.0000 | +0.0000 | +0 |
| `truck` | `depth_topk005` | `depth_topk005` | +0.0088 | +0.0374 | -2,758 |

观察：

- DB/Tandt 上 `selector_best_psnr` 与 `selector_qcgi` 完全一致。
- held-out PSNR 只比 baseline 高 +0.0084，但 official test PSNR 低 -0.0153，SSIM/LPIPS 也略差。
- `playroom` 是典型反例：holdout +0.0325，但 test -0.0649。

## Round 1 决策

Held-out train-view split 没有解决 0006 的泛化问题：

- train 内部的 holdout view 对 selector 仍偏乐观；
- `selector_qcgi` 的容量控制是有效的，但 test PSNR 没过 baseline；
- `selector_best_psnr` 的视觉指标收益主要由高容量 RGB rerank 分支贡献，不适合默认化。

结论：validation-driven selector 主线收束。Depth prior 分支可以继续作为离线诊断/scene-conditioned 分析工具，但不进入默认 FastGS 策略。下一步应转向 0006 已暴露出的 online residual 方向，优先做 orientation-aware gating，而不是继续调 selector 权重。

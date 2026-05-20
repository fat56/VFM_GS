# 0006 Validation Selector + Online Depth Residual 结果

## 当前状态

Round 1 已启动准备：先复用 0002 / 0004 的既有 checkpoint，构建 validation-driven selector 候选表，并用双卡 tmux 分别评估 MipNeRF360 与 DB/Tandt。

## Round 1：Validation-driven selector

### 候选来源

候选表由 `scripts/build_0006_selector_candidates.py` 生成。FastGS big 基线统一记为 `baseline`，其他方法保留为短名：

- `depth_auto_topk`
- `depth_topk010`
- `depth_rgb_rerank_start9000`
- `depth_start24000_weight015`
- `depth_start24000_topk005`
- `depth_topk005`
- `depth_weight015_topk010`

### 运行状态

启动时间：2026-05-20 16:47 Asia/Shanghai。

候选计数：

| 分组 | 场景数 | 候选行数 | 方法数 |
|---|---:|---:|---:|
| MipNeRF360 | 9 | 44 | 6 |
| DB/Tandt | 4 | 16 | 5 |

tmux：

- `0006_selector_mip_g0`：GPU0，`output/0006/debug_logs/selector_mip_g0.log`
- `0006_selector_cross_g1`：GPU1，`output/0006/debug_logs/selector_cross_g1.log`

两个任务均已完成。

### MipNeRF360 已完成

输出：

- `output/0006/validation_selector/mipnerf360_train_selector/train_selector_averages.csv`
- `output/0006/validation_selector/mipnerf360_train_selector/train_selector_recommendations.csv`

与 MipNeRF360 baseline 均值比较：

| 方法 | PSNR | SSIM | LPIPS | GS |
|---|---:|---:|---:|---:|
| baseline | 27.9590 | 0.820268 | 0.215657 | 1,161,786 |
| train_best_psnr selector | 27.9602 | 0.823621 | 0.207624 | 1,384,362 |
| Δ best_psnr - baseline | +0.0012 | +0.003354 | -0.008033 | +222,576 |
| train_qcgi selector | 27.9506 | 0.820745 | 0.214239 | 1,200,922 |
| Δ qcgi - baseline | -0.0084 | +0.000477 | -0.001418 | +39,136 |

`train_qcgi` 逐场景选择：

| 场景 | 选择 |
|---|---|
| `bicycle` | `baseline` |
| `bonsai` | `depth_auto_topk` |
| `counter` | `depth_start24000_topk005` |
| `flowers` | `baseline` |
| `garden` | `depth_auto_topk` |
| `kitchen` | `depth_start24000_weight015` |
| `room` | `depth_auto_topk` |
| `stump` | `depth_rgb_rerank_start9000` |
| `treehill` | `depth_auto_topk` |

结论：`train_best_psnr` 主要被 RGB rerank 的训练视角质量吸引，LPIPS/SSIM 看起来好，但 Gaussian 增长太大。`train_qcgi` 控制容量后仍比 baseline 低 0.0084 PSNR，虽然 LPIPS 略好，不能默认化。

### DB/Tandt 已完成

输出：

- `output/0006/validation_selector/cross_train_selector/train_selector_averages.csv`
- `output/0006/validation_selector/cross_train_selector/train_selector_recommendations.csv`

`train_best_psnr` 与 `train_qcgi` 选择完全一致：

| 场景 | 选择 |
|---|---|
| `db/drjohnson` | `depth_auto_topk` |
| `db/playroom` | `depth_auto_topk` |
| `tandt/train` | `baseline` |
| `tandt/truck` | `depth_topk005` |

与 cross baseline 均值比较：

| 方法 | PSNR | SSIM | LPIPS | GS |
|---|---:|---:|---:|---:|
| baseline | 27.3643 | 0.884519 | 0.206637 | 593,359 |
| train_qcgi selector | 27.3563 | 0.884513 | 0.207116 | 592,792 |
| Δ selector - baseline | -0.0080 | -0.000006 | +0.000479 | -567 |

结论：DB/Tandt 上 train-split selector 会选出看似合理的 depth 分支，但 test 均值仍略低于 baseline。这是一个轻微过拟合信号，后续不能只用训练视角 proxy 作为默认策略依据。

### Round 1 决策

Validation-driven selector 作为“当前实现”不通过默认化门槛：

- MipNeRF360：`train_qcgi` 容量可控，但 PSNR 低于 baseline，且仍增加约 39k Gaussian。
- DB/Tandt：`train_qcgi` 与 `train_best_psnr` 一致，但 test PSNR / LPIPS 均略差。
- train split proxy 有用，但要升级成真正的 held-out selector 或 scene-conditioned policy，不能直接拿训练视角选择结果替换默认 FastGS。

### 决策口径

- 主看 `train_qcgi`，辅看 `train_best_psnr`。
- 与 baseline 比较时同时检查 PSNR / SSIM / LPIPS / Gaussian 数量，避免只追单一 PSNR。
- 如果 selector 只在个别场景赢，而均值和跨数据集不稳，则不进入默认策略，只保留为 scene-conditioned 工具。

## Round 2：Online depth residual

已加入 `scripts/diagnose_0006_online_depth_residual_proxy.py`。脚本先用 baseline checkpoint 做：

- RGB render error；
- Gaussian center z-buffer proxy depth；
- Depth Anything depth cache；
- proxy depth / inverse-depth residual 与 RGB error top-k 的重叠诊断。

单视角 smoke check：`stump` 的 proxy valid coverage 从无 splat 的约 18.5% 提升到 `--splat-radius 1` 的约 63.8%，说明 proxy 至少能覆盖足够多的图像区域用于第一轮诊断。

启动计划：

- `0006_depth_proxy_indoor_g1`：GPU1，`room kitchen bonsai`，已完成
- `0006_depth_proxy_mixed_g0`：GPU0，`stump counter`，已完成

### Proxy smoke 结果

输出：

- `output/0006/online_depth_residual_proxy/indoor_g1/summary.json`
- `output/0006/online_depth_residual_proxy/mixed_g0/summary.json`

整体：

| 分组 | 场景 | views | valid coverage | prior/RGB IoU | residual-depth/RGB IoU | residual-inv/RGB IoU | prior/edge IoU | residual-depth/edge IoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| indoor | bonsai/kitchen/room | 24 | 0.551 | 0.0748 | 0.0755 | 0.0406 | 0.0667 | 0.0695 |
| mixed | counter/stump | 15 | 0.539 | 0.0443 | 0.0431 | 0.0454 | 0.0646 | 0.0645 |

逐场景观察：

| 场景 | prior/RGB IoU | residual-depth/RGB IoU | residual-inv/RGB IoU | 观察 |
|---|---:|---:|---:|---|
| `bonsai` | 0.1197 | 0.1213 | 0.0452 | residual-depth 略高于静态 prior，但幅度很小 |
| `kitchen` | 0.0560 | 0.0548 | 0.0381 | RGB 上没有超过 prior；GT edge 上 residual-depth 更高 |
| `room` | 0.0488 | 0.0504 | 0.0385 | residual-depth 略高于 prior，方向正确但很弱 |
| `counter` | 0.0477 | 0.0484 | 0.0361 | residual-depth 略高于 prior，GT edge 也略高 |
| `stump` | 0.0409 | 0.0377 | 0.0547 | inverse-depth residual 对 RGB error 更强，但 edge 对齐更弱 |

结论：proxy 覆盖率足够做第一轮诊断，但 online residual 还不是一个“拿来就能替代静态 prior”的信号。更准确的说，它暴露了两个变量：深度方向需要按场景/视角校准，且 residual-depth 主要改善 edge 对齐，residual-inv 只在 `stump` 的 RGB error 上更强。下一步若继续，应做 orientation-aware residual selector，而不是直接把某一个 residual 图接入 pruning。

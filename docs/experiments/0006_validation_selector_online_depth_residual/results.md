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

待填：tmux 任务完成后记录 selector 平均指标、逐场景选择。

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

- `0006_depth_proxy_indoor_g1`：GPU1，`room kitchen bonsai`
- `0006_depth_proxy_mixed_g0`：GPU0，`stump counter`，等待 MipNeRF360 selector 释放 GPU0 后启动

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

### 决策口径

- 主看 `train_qcgi`，辅看 `train_best_psnr`。
- 与 baseline 比较时同时检查 PSNR / SSIM / LPIPS / Gaussian 数量，避免只追单一 PSNR。
- 如果 selector 只在个别场景赢，而均值和跨数据集不稳，则不进入默认策略，只保留为 scene-conditioned 工具。

## Round 2：Online depth residual

尚未启动。Round 1 完成后先做 proxy smoke；只有 smoke 证明 residual 能解释 0004 的正负例差异，才进入 CUDA/rasterizer 深度输出实现。

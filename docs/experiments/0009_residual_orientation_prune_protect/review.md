# 0009 Residual Orientation Prune-Protect Review

## Checklist

- [x] 新 backend 是否只影响 `depth_anything_residual_orientation`，不改变旧 depth prior 配置。
- [x] Smoke 是否覆盖 scorer preflight、train、render、metrics。
- [x] Pilot 是否在双卡 tmux 运行。
- [x] 是否按轮次 commit/push。
- [x] 若全 9 场景整体不稳，是否停止训练接入。

## Smoke Review

- 新参数已出现在 `vfm_gs.cli.train --help`，配置加载路径正常。
- 700-step train/render/metrics 全链路通过。
- 直接 scorer smoke 在 `DENSIFY=False` 下打印 late protect 日志，说明 `vfm_prune_protect_mode=rgb_prune_auto_topk` 与 residual orientation counts 已连通。
- Smoke 只证明代码路径健康，不说明 30k 质量收益。

## 风险

- 训练期 center z-buffer depth 仍不是 alpha-weighted renderer depth。
- `edge_iou` selector 使用 GT image edge，是诊断型 proxy，不是纯部署信号。
- Late protect 会改变 pruning，而不是 densification；若 baseline 的后期收益来自 pruning 重分配，保护过强仍可能伤害质量。
- 当前存在 Phase0 baseline 与 checkpoint-curve baseline 两套历史口径；0009 必须同时报告，避免把 run-to-run 差异误读为方法收益。

## Round 1 Review

三场景 pilot 已在双卡 tmux 完成，训练、render、metrics 均通过。主口径相对 Phase0 baseline 为：PSNR +0.0093、SSIM -0.00033、LPIPS +0.00045、Gaussian -3,815。相对 0004 使用的 checkpoint-curve baseline，则是 PSNR -0.0180。

逐场景看，`treehill` 符合 0008 proxy 预期，PSNR 薄正且 LPIPS 略好；`room` 在 Phase0 口径薄正，但在 curve 口径仍为 -0.0625 PSNR，说明 residual orientation 没有解决 late protect 伤害室内自然后期收益的问题；`stump` 相对 Phase0 负向，且没有复现 0004 static prior 的 +0.0357 PSNR。

判断：Round 1 不足以证明训练接入成立，但也没有明显容量失控或质量崩坏。继续补跑剩余 6 个 MipNeRF360 场景，只把 Round 2 当作诊断扩样；若全 9 均值不能超过 Phase0 baseline 与 0002 `depth_auto_topk`，就停止 0009 训练主线。

## Round 2 Review

补跑剩余 6 场景后，全 9 场景为 27.9662 / 0.82017 / 0.21585、1,160,135 点。相对 Phase0 baseline：PSNR +0.0072、SSIM -0.00010、LPIPS +0.00019、GS -1,651；相对 checkpoint-curve baseline：PSNR -0.0028；相对 0002 `depth_auto_topk`：PSNR -0.0061、SSIM +0.00003、LPIPS -0.00030、GS -1,958。

这不是稳定胜出，而是指标之间的薄 tradeoff。`counter/treehill/room/garden` 给了正向证据，但 `bonsai/stump` 负向明显；`kitchen` 相对 Phase0 略正，却比 0002 auto-topk 低 -0.1339 PSNR，也比 curve baseline 低 -0.1113。说明 0008 的离线 orientation gating 上限没有可靠传导到训练期 pruning 生命周期。

最终判断：停止 0009 训练接入主线。当前 backend 保留为诊断/未来 selector 特征，不继续做固定 top-k、weight 或 start-iter 扫描。

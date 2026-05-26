# 0021 Taming Importance FastGS Big Full9

## 核心问题

FastGS acknowledgement 提到 3DGS、Taming-3DGS、Speedy-Splat、Abs-GS 等前人代码/方法。本实验要隔离验证：

> FastGS big 的指标优势是否主要来自多视角 VCD/VCP scoring，而不是来自融合前人代码后的其它工程收益？

## 具体对照

固定 FastGS big 30K 训练流程，只替换 densification 阶段消费的 per-Gaussian importance：

- Baseline：`fastgs_big` 原始 multi-view photometric scorer。
- 0021：`taming_importance_fastgs_prune`，用 Taming-3DGS-style primitive/loss score 生成 densification importance。

保持不变：

- MipNeRF360 原图 1.6K auto-resolution 口径。
- `fastgs_big` scene overrides。
- densification interval = 100。
- clone / split / densify-stage prune gate。
- 15K 后 FastGS final-prune tail。
- FastGS pruning score。0021 只换 densification importance，不换 pruning score。

## 实现口径

Taming-3DGS 官方实现的 score 包含 pixel loss/edge、gradient、opacity、depth、radii、scale、dist/blend/count 等项。当前 FastGS rasterizer 只暴露 `accum_metric_counts`，没有 Taming 的 full weighted accumulators，因此 0021 采用当前管线可获得的 Taming-style score：

- GT edge + render/GT L1 loss map 形成 Taming pixel saliency proxy。
- Gaussian primitive terms：gradient、opacity、camera depth、screen radii、scale。
- 使用 Taming-3DGS 默认 score coefficients。
- 将 raw score 归一到 FastGS `densify_metric_thresh=5.0` 的阈值契约，避免改动下游 densify 逻辑。

这个实验不是重跑完整 Taming-3DGS；它是 FastGS 内部的 controlled ablation：只替换 importance 来源，观察 multi-view VCD/VCP importance 被移除后质量、点数和训练时间如何变化。

## 数据与指标

数据集：MipNeRF360 全 9 场景。

指标：

- PSNR
- SSIM
- LPIPS
- GS_num
- train_time

对照 baseline：

```text
output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_combined/summary.csv
```

## 判定

若 0021 显著低于 `fastgs_big`，说明 FastGS 的优势确实依赖当前 multi-view VCD/VCP importance，而不是单纯继承 Taming/Speedy/Abs-GS 的工程组件。

若 0021 接近或超过 `fastgs_big`，则需要重新审视：FastGS 的收益可能更多来自 clone/split/prune schedule、scene overrides、优化器/学习率或 inherited implementation details，而非 VCD/VCP scoring 本身。

# 0014 Phase0 Fingerprint Cross-Method Selector

## 核心问题

0013 说明 Depth / residual / DINO 在 MipNeRF360 不同场景上存在互补上界：test oracle 可到 +0.0646 QCGI。但这还不是可部署方案，因为 oracle 直接用了 test 指标。

0014 不新增训练，只问一个更靠近部署的问题：

> 只看 Phase0 baseline 训练曲线形成的低成本 scene fingerprint，能否在 leave-one-scene-out 下选择 0013 中更合适的方法？

## 输入

- Phase0 checkpoint curve：`output/0002/fastgs_big_baseline_checkpoint_curve/mipnerf360/check.csv`
- 0013 cross-method comparison：`output/0013/cross_method_scene_oracle/method_comparison_vs_phase0.csv`

## 对照策略

- 固定方法：phase0 / depth_auto_topk / residual_orientation / 三个 DINO 变体
- test oracle：逐场景选择 QCGI 最大方法，仅作为上界
- in-sample Phase0 curve stump：单特征单阈值，两侧各选择一个方法，仅作过拟合上界
- LOOCV Phase0 curve stump：每次留一场景，训练单特征单阈值方法选择器
- LOOCV nearest-neighbor oracle copy：用 Phase0 曲线特征找最近训练场景，拷贝该训练场景的 oracle 方法

## 判定

如果 LOOCV selector 明显低于固定 `depth_auto_topk`，则 Phase0-only fingerprint 不足以支撑 cross-method 默认化；下一步不应继续堆 selector，而应寻找新的训练期 proxy 或回到方法本身的稳定性改造。

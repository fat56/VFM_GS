# 0012 Scene Selector Proxy Audit

## 核心问题

0010 的 scene oracle 显示，如果只在 `descriptor_i050_until8000` 正收益场景启用 descriptor，QCGI 可以从 -0.0171 提高到 +0.0336。0011 则说明继续缩短全局 active window 不能解决问题。

本实验不新增训练，只审计已有结果和 Phase0 checkpoint curve：

> 是否存在一个足够轻、可解释的场景级开关，能接近 0010 oracle，而不需要额外训练完整 descriptor variant？

## 输入

- Phase0 full 30k summary：`output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_combined/summary.csv`
- Phase0 checkpoint curve：`output/0002/fastgs_big_baseline_checkpoint_curve/mipnerf360/check.csv`
- 0010 full i0.50 direct result
- 0010 until8000 full9 result
- 0011 until4000 full9 result

## 候选策略

- all until8000：固定全局启用。
- oracle：只在 until8000 相对 Phase0 QCGI 正向场景启用。
- full i0.50 QCGI sign：用完整 i0.50 的响应符号预测 until8000 是否启用。
- until4000 QCGI sign：用更短 window 的响应符号预测 until8000 是否启用。
- Phase0 checkpoint curve 单阈值：仅使用 Phase0 早期/曲线特征，搜索单个阈值规则，并做 leave-one-out 估计。

## 判定

如果 Phase0-only 的 LOOCV 单阈值能接近 oracle，后续做真实 selector；如果只有 full/4000 response proxy 有效，则不值得再训练复杂 selector，因为代理本身已经很贵。

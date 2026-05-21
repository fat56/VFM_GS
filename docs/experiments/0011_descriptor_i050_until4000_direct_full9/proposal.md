# 0011 Descriptor i0.50 until4000 Direct Full9

## 核心问题

0010 说明 `descriptor_i050_until8000` 比 full i0.50 更省，但 full 9 相对 Phase0 仍为负 QCGI（-0.0171）。本实验只问一个更窄的问题：

> 把 descriptor active window 从 8000 缩到 4000，是否能保留质量收益，同时把 Gaussian 增长压到足够低？

## 口径

- 数据集：MipNeRF360 全 9 场景
- 训练：`fastgs_big + images + -r -1`
- VFM cache：沿用 `output/0001/vfm_cache/{scene}_dinov2_vits14`
- 基线：`output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_combined/summary.csv`
- 对照：0010 `descriptor_i050_until8000` full9

## 方法

- 配置：`configs/experiments/0011_descriptor_i050_active_until4000.yaml`
- 机制：`vfm_active_until_iter=4000`
- 其他 descriptor 参数保持 0010 until8000 一致：
  - DINOv2 descriptor cosine
  - top-k 0.25
  - weighted i0.50
  - densify-only, `vfm_weight=0.0`

## 判定

- 若相对 Phase0 的 QCGI 转正，且平均 GS 增长显著低于 0010 until8000 full9 的 +33,372，则 4000 可作为新的全局窗口候选。
- 若质量收益被明显吃掉或 QCGI 仍为负，则停止全局窗口调参，转向 0010 scene-adaptive/selector。

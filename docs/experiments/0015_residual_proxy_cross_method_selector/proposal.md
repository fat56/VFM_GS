# 0015 Residual Proxy Cross-Method Selector

## 核心问题

0014 只用 Phase0 checkpoint curve，LOOCV 树桩可到 +0.0165 QCGI，但误选 `treehill/room/bonsai`，说明纯 Phase0 曲线信号不够。

0015 继续不新增训练，加入 0008 已有 residual-orientation proxy summary 作为更直接的训练期 proxy：

> Phase0 曲线 + residual proxy fingerprint 能否更稳地预测 0013 的 cross-method oracle？

## 输入

- Phase0 checkpoint curve：`output/0002/fastgs_big_baseline_checkpoint_curve/mipnerf360/check.csv`
- 0008 residual orientation summary：`output/0008/residual_orientation_gating_mipnerf360/scene_orientation_summary.csv`
- 0013 cross-method comparison：`output/0013/cross_method_scene_oracle/method_comparison_vs_phase0.csv`

## 对照策略

- 固定方法：phase0 / depth_auto_topk / residual_orientation / 三个 DINO 变体
- test oracle：逐场景选择 QCGI 最大方法，仅作为上界
- in-sample Phase0 + residual proxy stump
- LOOCV Phase0 + residual proxy stump
- LOOCV Phase0 + residual proxy nearest-neighbor oracle copy

## 判定

如果 residual proxy 特征不能明显超过 0014 的 Phase0-only LOOCV，则说明已有 proxy 只适合解释 residual 方法本身，不足以作为跨 Depth/DINO/residual 的总选择器。

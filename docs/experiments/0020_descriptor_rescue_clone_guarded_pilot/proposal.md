# 0020 descriptor rescue clone guarded pilot

## 核心假设

0019 证明 descriptor 作为 0-15K clone OR 信号有场景依赖收益，但无保护 OR 会让 `room`、`garden` 明显退化。0020 把 descriptor 从“并列触发条件”降级为“少量 rescue 候选”：

- RGB/FastGS clone 条件仍完整保留。
- descriptor-only clone 候选必须同时通过弱 RGB gate。
- descriptor-only clone 候选每轮最多为 RGB clone 候选数的 20%。
- 生效窗口从 `0-15K` 收窄为 `3K-12K`，避免太早影响粗糙几何，也避免后期继续扰动。

要验证的问题不是“是否 full9 默认提升”，而是：加保护后能否保留 `stump/treehill` 的收益，同时修掉 `room/garden` 的退化。

## 变体 / 配置

- 变体：`fastgs_big`
- 配置：`configs/experiments/0020_descriptor_rescue_clone_guarded_pilot.yaml`
- 打分器：`vfm_topology_scorer`
- backend：`dinov2_descriptor_cosine`
- runner：`scripts/run_0019_descriptor_or_rgb_clone_baseline30_full9_eval.py`，通过 0020 参数复用汇总逻辑
- tmux：`scripts/run_0020_descriptor_rescue_clone_guarded_pilot_tmux.sh`

关键开关：

```yaml
vfm_clone_or_rgb_enabled: true
vfm_clone_or_rgb_from_iter: 3000
vfm_clone_or_rgb_until_iter: 12000
vfm_clone_or_rgb_rgb_gate_ratio: 0.5
vfm_clone_or_rgb_extra_clone_ratio: 0.2
```

## 数据集

- 数据集：MipNeRF360
- 场景：`garden`、`room`、`stump`、`treehill`
- 分辨率：沿用 scene overrides / 原实验口径

场景选择：

- 0019 正例：`stump`、`treehill`
- 0019 负例：`room`、`garden`

## 指标

- 15K / 20K / 25K / 30K summary。
- 20K / 30K 相对 FastGS checkpoint curve。
- 30K 相对 FastGS 30K baseline。
- 重点判定：
  - `room`、`garden` 的 PSNR/SSIM 退化是否显著收敛。
  - `stump`、`treehill` 的 SSIM/LPIPS 收益是否保留。
  - Gaussian 增量是否低于 0019。

## 决策

- 若四场景平均 30K PSNR 转正，且 `room/garden` 不再明显负向，再考虑 full9。
- 若 `room/garden` 仍明显退化，停止 clone-rescue 训练接入，descriptor 只保留为离线诊断或 selector 特征。
- 若正例收益消失但负例修复，则说明保护过强，不继续做大扫，只考虑更晚期/更小容量的单场景 rescue。

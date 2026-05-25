# 0019 Descriptor OR RGB Clone Baseline30 Full9

## 核心问题

0018 的补充对照说明，35K 最终质量收益主要来自更长、更后移的 schedule，而不能把最终质量提升单独归因给 descriptor。0019 回到更干净的标准 FastGS 30K baseline 流程，只在原始 0-15K densification 窗口里让 descriptor 作为 clone 的额外触发信号：

> clone 候选满足 `RGB/FastGS clone 条件 OR descriptor clone 条件` 即可复制；split 与 densify 阶段中的 prune 继续使用 RGB/FastGS 原逻辑；15K 之后回到标准 FastGS final-prune/fine-tune。

这个实验回答的问题是：descriptor 能否作为 FastGS baseline 的轻量 clone 增强信号，而不是另开 extra schedule 或延长训练。

## 口径

- 数据集：MipNeRF360 全 9 场景：`bicycle` / `flowers` / `garden` / `stump` / `treehill` / `room` / `counter` / `kitchen` / `bonsai`。
- 训练起点：从 COLMAP/input point cloud 正常从 0 开始。
- 总迭代：30K，保持标准 FastGS baseline schedule。
- densification：仅 0-15K。
- final prune：15K-30K，沿用 FastGS 30K baseline 节奏。
- 保存与评测点：15K / 20K / 25K / 30K。
- 对照：`output/0002/fastgs_big_baseline_checkpoint_curve/mipnerf360/check.csv` 中的 FastGS checkpoint curve，最终判断优先看 30K。

## 方法

- 配置：`configs/experiments/0019_descriptor_or_rgb_clone_baseline30_full9.yaml`
- runner：`scripts/run_0019_descriptor_or_rgb_clone_baseline30_full9_eval.py`
- tmux：`scripts/run_0019_descriptor_or_rgb_clone_baseline30_full9_tmux.sh`
- scorer：`vfm_topology_scorer`
- descriptor backend：`dinov2_descriptor_cosine`
- descriptor metric：top-k25，token smooth kernel 3，与 0017/0018 descriptor clone 口径一致。
- `vfm_weight=0.0`，descriptor 不参与 pruning score。
- 新增开关：`vfm_clone_or_rgb_enabled=true`
  - clone mask：`rgb_metric_mask OR descriptor_metric_mask`
  - split mask：`rgb_metric_mask`
  - densify-stage prune score：RGB/FastGS pruning score

## 判定

优先看 30K 相对 FastGS 30K baseline：

- 若 PSNR/SSIM/LPIPS 平均正向，且 Gaussian 数量没有显著膨胀，说明 descriptor 作为 clone-only OR 信号有默认化潜力。
- 若质量正向但点数明显增加，需要继续做容量约束或 scene-adaptive 开关。
- 若质量不如 baseline，说明 descriptor clone 的有效时机仍不适合直接并入原始 0-15K densification，应回到后期辅助或 selector 路线。

中间点用于诊断：

- 15K：观察 descriptor OR clone 是否已经改变 early densification 轨迹。
- 20K/25K：观察标准 final prune 是否能吸收额外 clone。
- 30K：最终 baseline 对齐结论。

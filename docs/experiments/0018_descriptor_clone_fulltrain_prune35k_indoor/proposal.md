# 0018 Descriptor Clone Full-Train Prune35K Indoor

## 核心问题

0017 证明了 `descriptor clone@16K` 相对 16K 起点是稳定正向的，但它是从 16K PLY 续跑，optimizer state 是新建的，且没有接上后续 final prune。0018 改成完整训练：

> 从 0 开始训练四个室内场景，0-15K 完全按 FastGS baseline；15K-20K 接入 DINO descriptor clone-only 作为额外致密化阶段；20K-35K 回到 FastGS photometric final prune tail。这样同时验证 extra clone 的收益能否被后续 prune/fine-tune 吸收，并避免“从 PLY 重启导致 optimizer state 变化”的混淆。

## 口径

- 数据集：MipNeRF360 四个室内场景：`room` / `counter` / `kitchen` / `bonsai`。
- 训练起点：从 COLMAP/input point cloud 正常从 0 开始，不调用 16K baseline checkpoint。
- 总迭代：35K。
- 保存与评测点：15K / 20K / 25K / 30K / 35K。
- 训练阶段只保存 checkpoint；训练结束后统一 render 和 metric。
- 30K baseline 对照：使用 `output/0002/fastgs_big_baseline_checkpoint_curve/mipnerf360/check.csv` 中的 FastGS 30K 指标。

## 方法

- 配置：`configs/experiments/0018_descriptor_clone_fulltrain_prune35k_indoor.yaml`
- scorer：`vfm_topology_scorer`
- 0-15K：VFM inactive，scorer 自动回落为 FastGS photometric；clone/split/prune 均沿用 FastGS。
- 15K-20K：VFM active，`dinov2_descriptor_cosine` 只控制 clone；关闭 split 与 densify-prune；关闭 opacity reset。
- 20K-35K：VFM inactive，final prune 使用 FastGS photometric scorer。
- optimizer schedule 后移 5K：
  - `optimizer_dense_until_iter=20000`
  - `optimizer_sparse32_until_iter=25000`
  - 25K 之后进入 64-step sparse cadence。
- final prune window 后移：
  - `final_prune_from_iter=20000`
  - `final_prune_until_iter=35000`
  - `final_prune_interval=3000`

## 注意

训练循环中保存发生在同一 iteration 的 final prune 之前。因此：

- 25K checkpoint 反映 21K/24K prune 后的状态。
- 30K checkpoint 反映 21K/24K/27K prune 后的状态。
- 35K checkpoint 反映 21K/24K/27K/30K/33K prune 后再训练到 35K 的状态。

这个时序与原 FastGS 保存逻辑一致，0018 不额外改保存顺序。

## 判定

优先看两个问题：

- 相对 20K descriptor clone end，25K/30K/35K 的 PSNR/SSIM/LPIPS 是否继续上升，Gaussian 数量是否被 prune tail 合理回收。
- 相对标准 FastGS 30K baseline，0018 的 30K/35K 是否能持平或更好；若 35K 明显优于 baseline 30K，说明额外 clone 的增长能通过后续 prune tail 转化为有效质量。

若 35K 仍弱于 baseline 30K，则 descriptor clone 更可能只是延迟/扰动了 FastGS 原本的高效 densification，下一步应转向 RGB/FastGS 候选保留、descriptor rerank 或容量 cap，而不是继续延长 clone-only。

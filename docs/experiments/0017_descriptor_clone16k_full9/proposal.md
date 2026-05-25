# 0017 Descriptor Clone16K Full9

## 核心问题

0016 的补充 smoke 里，`desc_clone_16k` 是唯一有正信号的分支：3 场景 +2K 平均 `+0.1522` PSNR、`+0.00140` SSIM、`-0.00227` LPIPS，且 kitchen / bonsai 几乎不增点。0017 只追这条线：

> DINO descriptor 只控制 clone、从 FastGS baseline 16K checkpoint 接入，续跑 5K 到 21K 后，在 MipNeRF360 full9 上是否仍然稳定正向？

## 口径

- 数据集：MipNeRF360 全 9 场景。
- 起点：`output/0002/fastgs_big_baseline_checkpoint_curve/mipnerf360/{scene}/fastgs_big_30k_curve_r_auto` 的 `iteration_16000` PLY。
- 续跑窗口：16K -> 21K，共 5K iteration。
- 评测点：16K / 17K / 18K / 19K / 20K / 21K。
- 训练效率：训练阶段只保存 17K-21K 的中间 PLY；训练结束后统一 render 6 个 checkpoint，再统一 metric。

## 方法

- 配置：`configs/experiments/0017_descriptor_clone16k_full9.yaml`
- backend：`dinov2_descriptor_cosine`
- cache：`output/0001/vfm_cache/{scene}_dinov2_vits14`
- scorer：`vfm_topology_scorer`
- importance：`vfm_importance_mode=vfm_only`
- 分支：只开 clone，关闭 split。
- prune：关闭 densify prune、final prune、opacity reset。
- densification interval：100，沿用 FastGS 多 camera 采样和多视图判别流程。
- scene overrides：沿用 `fastgs_big` MipNeRF360 场景级 overrides。

## 判定

优先看相对 16K 起点的曲线，而不是只看 21K 单点：

- 若 full9 在 +5K 平均 PSNR/SSIM/LPIPS 均为正向，且没有明显单场景崩盘，则 `descriptor clone@16K` 可以进入容量约束或 RGB gate 版本。
- 若正向只集中在 kitchen / bonsai，或 flowers/garden 等室外场景以大量增点换很小收益，则下一步必须先做 candidate 分布和容量 cap。
- 若 +1K/+2K 正、后续回落，则保留 early-stop 可能性，不直接默认跑满 5K。

# 0009 Residual Orientation Prune-Protect

## 核心假设

0008 说明 `prior / residual_depth / residual_inv` 的 per-view orientation gating 在离线 proxy 上有稳定上限：全 MipNeRF360 9 场景中，`edge_proxy_pick` 相对 static prior 的 RGB IoU 平均 +0.0088、edge IoU 平均 +0.0104，并且 9/9 场景 RGB delta 为正。

0009 只验证一个很保守的训练接入：

- RGB/FastGS 继续决定 densification；
- residual orientation 只在 late pruning 阶段保护 RGB pruning candidate；
- 不做硬 selector，不扩大 Gaussian 生命周期主导权；
- 先看 small pilot 是否比 0004 static prior late protect 稳。

## 方法

新增 backend：`depth_anything_residual_orientation`。

每个 scorer view 内：

1. 加载 Depth Anything relative depth prior；
2. 用 Gaussian center z-buffer proxy 生成当前模型 depth；
3. 构造 `prior`、`abs(depth_norm - prior)`、`abs(inv_depth_norm - prior)`；
4. 用 GT edge top-k IoU 在三者里选一个作为该 view 的 metric map；
5. rasterizer 累积 per-Gaussian VFM counts；
6. 复用已有 `rgb_prune_auto_topk`，仅保护 late RGB pruning candidates。

## 配置

主配置：

- `configs/experiments/0009_residual_orientation_protect_start24000_auto_topk005.yaml`

关键参数：

- `vfm_backend: depth_anything_residual_orientation`
- `vfm_metric_topk: 0.10`
- `vfm_prune_protect_mode: rgb_prune_auto_topk`
- `vfm_prune_protect_rgb_auto_max_topk: 0.005`
- `vfm_active_from_iter: 24000`
- `vfm_residual_orientation_selector: edge_iou`

## 数据集

Round 1 pilot：

- `room`：0004 late protect 的失败样本，baseline 24k 后仍有明显自然收益；
- `treehill`：0008 full proxy 中 `edge_proxy` 与 RGB oracle 完全一致，且 static prior 弱；
- `stump`：0008 中 match rate 最低，作为压力测试。

若 Round 1 不负，再扩到 MipNeRF360 全 9 场景。

## 判定

- 先看相对 `fastgs_big` baseline 的 PSNR / SSIM / LPIPS / Gaussian 数；
- 同时对照 0004 `start24000_auto_topk005`；
- 若 pilot 仍和 0004 一样只局部正向，则停止训练接入，把 residual orientation 保留为离线诊断；
- 只有 pilot 三场景整体不负，才进入全 9 场景。

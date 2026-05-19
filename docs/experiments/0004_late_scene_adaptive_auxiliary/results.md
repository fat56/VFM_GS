# 0004 Late Scene-Adaptive Auxiliary 结果

## 当前状态

已完成前置 baseline curve 诊断，0004 正式训练待启动。

## 前置 Baseline Curve

来源：`output/0002/fastgs_big_baseline_checkpoint_curve/mipnerf360/check.md`

| 场景集合 | 16k -> 30k PSNR | 20k -> 30k PSNR | 24k -> 30k PSNR | 30k GS |
|---|---:|---:|---:|---:|
| MipNeRF360 全 9 场景平均 | +0.3316 | +0.1525 | +0.0630 | 1,161,267 |

逐场景 24k -> 30k：

| 场景 | 24k -> 30k PSNR | 30k PSNR | 30k SSIM | 30k LPIPS | 30k GS |
|---|---:|---:|---:|---:|---:|
| bicycle | +0.0268 | 25.2646 | 0.7556 | 0.2447 | 1,558,080 |
| bonsai | +0.2177 | 32.9616 | 0.9531 | 0.1601 | 844,773 |
| counter | +0.1049 | 29.5411 | 0.9179 | 0.1765 | 471,258 |
| flowers | +0.0099 | 21.6213 | 0.6022 | 0.3407 | 1,134,832 |
| garden | +0.0201 | 27.6123 | 0.8643 | 0.1099 | 2,631,613 |
| kitchen | +0.0952 | 32.4183 | 0.9394 | 0.1043 | 1,179,861 |
| room | +0.1058 | 32.3007 | 0.9307 | 0.1881 | 570,190 |
| stump | -0.0194 | 27.1382 | 0.7863 | 0.2398 | 1,052,292 |
| treehill | +0.0055 | 22.8626 | 0.6323 | 0.3769 | 1,008,508 |

结论：0004 pilot 应优先关注后期窗口差异。`room` 这种室内场景 24k 后仍有明显 PSNR 空间；`stump` 这种场景 PSNR 在 22k 达峰但 LPIPS 仍改善；`bicycle` 后期收益薄但稳定。只看 30k 平均容易掩盖 intervention timing 的场景差异。

## 预期观察

- 先看小 pilot 是否能比 0002 / 0003 更稳地控制 Gaussian 数量。
- 再看场景间差异是否真的能被 policy 吸收，而不是只换一种失败方式。

## 记录表

| 场景 / 数据集 | 配置 | PSNR | SSIM | LPIPS | Gaussian 数量 | 备注 |
|---|---|---:|---:|---:|---:|---|
| MipNeRF360 pilot | `0004_late_scene_adaptive_auxiliary` | TBD | TBD | TBD | TBD | 待跑 |

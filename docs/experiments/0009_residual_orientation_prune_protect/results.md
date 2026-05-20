# 0009 Residual Orientation Prune-Protect 结果

## 当前状态

实现、配置和 smoke 已完成。下一步启动 `room/treehill/stump` 三场景 pilot。

## Smoke

已完成：

- `py_compile` 通过：`src/vfm_gs/scorers/vfm_topology.py`、`src/vfm_gs/config/legacy_args.py`。
- `treehill -r 8` 700-step 训练通过，并在第 600 step 实际触发 `depth_anything_residual_orientation` scorer。
- Render / metrics 通过：700-step `treehill` 为 PSNR 20.7703 / SSIM 0.5537 / LPIPS 0.5305。
- 直接 scorer smoke 覆盖 late prune-protect 分支：`[VFM PRUNE PROTECT] iter=24000 mode=rgb_prune_auto_topk weight=0.2500 protected=239 rgb_candidates=271 mean=0.000138 max=1.000000`。

Smoke 输出：

- `output/0009/residual_orientation_protect_smoke/treehill/smoke_700_r8`

## Round 1 Pilot

待记录：

| scene | baseline PSNR | 0009 PSNR | ΔPSNR | baseline SSIM | 0009 SSIM | ΔSSIM | baseline LPIPS | 0009 LPIPS | ΔLPIPS | baseline GS | 0009 GS | ΔGS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| room | | | | | | | | | | | | |
| treehill | | | | | | | | | | | | |
| stump | | | | | | | | | | | | |

## 决策

待填。

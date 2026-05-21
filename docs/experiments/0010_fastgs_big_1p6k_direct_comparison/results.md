# 0010 实验结果

## 当前状态

本实验用于补齐同尺度比较：只接受 `fastgs_big + images + -r -1`，并直接对照 0002 Phase 0 FastGS big 1.6K baseline。

第一批 `descriptor_i050_fastgs_big_legacy_cache` 已完成 MipNeRF360 全 9 场景。结论：在 direct `fastgs_big` 1.6K 口径下，0001 的 DINO descriptor weighted i0.50 不再是清晰正向默认方案。它相对 Phase 0 baseline 的均值为 -0.0027 PSNR、+0.0011 SSIM、LPIPS -0.0032，但平均多 53,628 Gaussians；QCGI 均值 -0.1015，只有 3/9 场景 QCGI 正向。

第二批已启动准备：`descriptor_i050_until8000` 只让 descriptor i0.50 在 8000 iter 前参与 densification，后续回到 FastGS/RGB importance。目标不是提高质量上界，而是检查能否保留 SSIM/LPIPS 收益并压回 Gaussian 增长。

已完成第一批：

- `descriptor_i050_fastgs_big_legacy_cache`
  - `configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i050.yaml`
  - VFM cache: `output/0001/vfm_cache/{scene}_dinov2_vits14`
  - 输出：`output/0010/descriptor_i050_fastgs_big_legacy_cache/`

第一批完成后暂不直接扩展：

- `descriptor_max_fastgs_big_legacy_cache`
- `descriptor_i050_fastgs_big_images1600_cache`

原因：第一批的主要失败是容量效率，而不是 descriptor 信号完全无效。直接跑 `descriptor max` 很可能提高质量但进一步增点，因此优先进入容量受控 pilot。

## 口径说明

0001 旧表中的 `dataset_quality_policy`、`dataset_fixed_policy`、`DINO descriptor top-k25 max` 全数据集均值，大多不是 `fastgs_big` 1.6K 直接对照；它们不能和 0002-0009 的 Phase 0 baseline delta 混排。本实验完成后，所有“最好结果”表应拆成两张：

- `fastgs_baseline / -r 8` 历史验证表；
- `fastgs_big / images / -r -1` direct comparison 表。

## Descriptor i0.50 legacy cache full MipNeRF360

输出：

- `output/0010/descriptor_i050_fastgs_big_legacy_cache/mipnerf360_combined/summary.csv`
- `output/0010/descriptor_i050_fastgs_big_legacy_cache/mipnerf360_combined/comparison_vs_phase0.csv`
- `output/0010/descriptor_i050_fastgs_big_legacy_cache/mipnerf360_combined/summary_stats.json`

| 场景 | PSNR | SSIM | LPIPS | GS | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGS | QCGI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | 25.3147 | 0.7609 | 0.2361 | 1,610,066 | +0.0578 | +0.0056 | -0.0089 | +49,857 | +0.1639 |
| flowers | 21.6225 | 0.6019 | 0.3418 | 1,079,919 | -0.0059 | -0.0004 | +0.0014 | -42,896 | -0.0219 |
| garden | 27.4946 | 0.8627 | 0.1109 | 2,564,955 | -0.1400 | -0.0018 | +0.0013 | -69,861 | -0.1818 |
| stump | 27.1850 | 0.7898 | 0.2315 | 1,189,517 | +0.0066 | +0.0030 | -0.0079 | +124,657 | -0.0937 |
| treehill | 22.8091 | 0.6316 | 0.3773 | 942,547 | -0.0185 | -0.0007 | +0.0003 | -66,563 | -0.0349 |
| room | 32.1754 | 0.9322 | 0.1823 | 615,504 | -0.0382 | +0.0018 | -0.0058 | +43,897 | -0.0176 |
| counter | 29.6029 | 0.9197 | 0.1723 | 551,186 | +0.0769 | +0.0016 | -0.0043 | +80,609 | +0.0507 |
| kitchen | 32.4246 | 0.9395 | 0.1036 | 1,290,842 | +0.1436 | +0.0006 | -0.0015 | +112,854 | +0.0113 |
| bonsai | 32.9780 | 0.9537 | 0.1561 | 1,094,195 | -0.1067 | -0.0000 | -0.0037 | +250,102 | -0.7891 |
| **平均** | **27.9563** | **0.8213** | **0.2124** | **1,215,415** | **-0.0027** | **+0.0011** | **-0.0032** | **+53,628** | **-0.1015** |

判断：

- direct 1.6K 口径下，descriptor i0.50 保留了 SSIM/LPIPS 平均收益，但 PSNR 没有超过 Phase 0。
- 正例主要是 `bicycle/counter/kitchen`；`stump` 三项质量正向但增点过多，QCGI 负；`bonsai` 是主要容量负例。
- 因此 0001 的旧口径“全 9 场景强正向”不能直接外推为 `fastgs_big` 1.6K 默认方案。
- 下一步若继续补 0010，应优先试更严格的 early/limited active window 或 scene-adaptive 版本，而不是直接扩 `descriptor max`。`descriptor max` 预计质量可能更高但容量更贵，不太可能解决 direct 1.6K 默认化问题。

## Descriptor i0.50 until8000 pilot

配置：

- `configs/experiments/0010_descriptor_i050_active_until8000.yaml`

输出：

- `output/0010/descriptor_i050_until8000_pilot/mip_g0/summary.csv`
- `output/0010/descriptor_i050_until8000_pilot/mip_g1/summary.csv`
- `output/0010/descriptor_i050_until8000_pilot/mipnerf360_pilot_combined/summary.csv`
- `output/0010/descriptor_i050_until8000_pilot/mipnerf360_pilot_combined/comparison_vs_phase0.csv`
- `output/0010/descriptor_i050_until8000_pilot/mipnerf360_pilot_combined/comparison_vs_i050_full.csv`
- `output/0010/descriptor_i050_until8000_pilot/mipnerf360_pilot_combined/summary_stats.json`

场景：

- GPU0：`bicycle/garden/stump`
- GPU1：`counter/kitchen/bonsai`

判定：

- 首先看相对 Phase0 的 QCGI 是否回正，且平均 ΔGS 是否明显低于第一批的 +53,628。
- 再看相对 full i0.50 是否以较小质量损失换来足够容量下降。
- 若 `bonsai/stump` 的容量负例不改善，或 `bicycle/counter/kitchen` 的正向质量被吃掉，则不继续 early-window 固定策略，转向 scene-adaptive / selector。

当前状态：已准备配置和 tmux launcher，待运行。

# 0010 实验结果

## 当前状态

本实验用于补齐同尺度比较：只接受 `fastgs_big + images + -r -1`，并直接对照 0002 Phase 0 FastGS big 1.6K baseline。

第一批 `descriptor_i050_fastgs_big_legacy_cache` 已完成 MipNeRF360 全 9 场景。结论：在 direct `fastgs_big` 1.6K 口径下，0001 的 DINO descriptor weighted i0.50 不再是清晰正向默认方案。它相对 Phase 0 baseline 的均值为 -0.0027 PSNR、+0.0011 SSIM、LPIPS -0.0032，但平均多 53,628 Gaussians；QCGI 均值 -0.1015，只有 3/9 场景 QCGI 正向。

第二批 `descriptor_i050_until8000` pilot 已完成 6 场景。它相对 Phase 0 质量均值略正向（+0.0239 PSNR、+0.0012 SSIM、LPIPS -0.0029），但平均仍多 52,337 Gaussians，QCGI 均值 -0.0232；相对 full i0.50 平均少 39,032 Gaussians，QCGI 近乎持平（+0.0018）。结论：early-window 确实比 full i0.50 更省，但固定 8000 iter 仍不能成为 direct `fastgs_big` 1.6K 默认方案。

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

| 场景 | PSNR | SSIM | LPIPS | GS | ΔPSNR vs P0 | ΔSSIM vs P0 | ΔLPIPS vs P0 | ΔGS vs P0 | QCGI vs P0 | ΔPSNR vs full | ΔGS vs full | QCGI vs full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | 25.2764 | 0.7580 | 0.2410 | 1,581,406 | +0.0195 | +0.0027 | -0.0040 | +21,197 | +0.0731 | -0.0383 | -28,660 | -0.1195 |
| garden | 27.5721 | 0.8648 | 0.1095 | 2,590,798 | -0.0625 | +0.0003 | -0.0001 | -44,018 | -0.0554 | +0.0775 | +25,843 | +0.1006 |
| stump | 27.2062 | 0.7891 | 0.2335 | 1,141,851 | +0.0278 | +0.0022 | -0.0058 | +76,991 | +0.0243 | +0.0212 | -47,666 | -0.0037 |
| counter | 29.5782 | 0.9190 | 0.1735 | 514,084 | +0.0522 | +0.0010 | -0.0030 | +43,507 | +0.0437 | -0.0247 | -37,102 | -0.0440 |
| kitchen | 32.4306 | 0.9395 | 0.1037 | 1,241,014 | +0.1497 | +0.0005 | -0.0015 | +63,026 | +0.1047 | +0.0061 | -49,828 | +0.0050 |
| bonsai | 33.0416 | 0.9544 | 0.1569 | 997,414 | -0.0430 | +0.0006 | -0.0029 | +153,321 | -0.3298 | +0.0636 | -96,781 | +0.0721 |
| **平均** | **29.1842** | **0.8708** | **0.1697** | **1,344,428** | **+0.0239** | **+0.0012** | **-0.0029** | **+52,337** | **-0.0232** | **+0.0176** | **-39,032** | **+0.0018** |

判断：

- 相对 Phase0，early-window 保留了 SSIM/LPIPS 收益，PSNR 均值也转正；但平均增点几乎没有低于第一批 full i0.50 的 +53,628，QCGI 仍为负。
- 相对 full i0.50，`garden/bonsai` 质量和 QCGI 明显改善，`kitchen` 也更好；`bicycle/counter` 则用容量下降换掉了一部分原本质量收益。
- `bonsai` 仍是主要容量负例：即使缩短到 8000 iter，仍多 153,321 Gaussians，QCGI -0.3298。`garden` 虽少点但质量不足，QCGI -0.0554。
- 固定 early-window 策略不继续默认化；下一轮应转向 scene-adaptive/selector：只在 `bicycle/stump/counter/kitchen` 这类正 QCGI 场景启用 descriptor early-window，或者寻找可由训练前/早期统计预测的开关，避免 `garden/bonsai` 这类负例。

## Descriptor i0.50 until8000 full9 completion

目的：补齐 `flowers/treehill/room`，并与第二批 6 场景合并为 full 9，以便后续 scene-adaptive/selector 有完整标签。

输出：

- `output/0010/descriptor_i050_until8000_full9_extra/`
- `output/0010/descriptor_i050_until8000_full9_combined/summary.csv`
- `output/0010/descriptor_i050_until8000_full9_combined/comparison_vs_phase0.csv`
- `output/0010/descriptor_i050_until8000_full9_combined/comparison_vs_i050_full.csv`
- `output/0010/descriptor_i050_until8000_full9_combined/summary_stats.json`

当前状态：已准备配置和 tmux launcher，待运行。

# 0010 FastGS Big 1.6K Direct Comparison

## 核心问题

前面实验里存在两种评测口径：

- 早期 0001 主结果多为 `fastgs_baseline`、`images_8`、`-r 8`。
- 0002 之后多数结果为 `fastgs_big`、`images`、`-r -1`，即 FastGS 原始大图自动缩放到 1.6K。

为了避免把不同尺度的结果混排，本补充实验只回答一个问题：

> 0001 中最有价值的 DINO descriptor 方法，在原始 `fastgs_big` 1.6K 口径下，相对原始 FastGS big baseline 是否仍然正向？

## 对照基线

统一使用 0002 Phase 0 的 5090 修复后 baseline：

- MipNeRF360: `output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_combined/summary.csv`
- DB/Tandt: `output/0002/phase0_5090_fastgs_big_baseline_fix1/all_combined/summary.csv`

该 baseline 使用：

- `--variant fastgs_big`
- `-i images`
- `-r -1`
- FastGS 大图自动缩放到 1.6K
- 0002/0001 fastgs_big runner 的 per-scene overrides

## 第一批补充

优先补 MipNeRF360 全 9 场景：

1. `descriptor_i050_fastgs_big_legacy_cache`
   - 配置：`configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i050.yaml`
   - 训练口径：`fastgs_big + images + -r -1`
   - VFM cache：沿用 0001 `output/0001/vfm_cache/{scene}_dinov2_vits14`
   - 目的：复验 0001 容量受控档是否在 direct FastGS big 口径下仍正向。

2. `descriptor_max_fastgs_big_legacy_cache`
   - 配置：`configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025.yaml`
   - 训练口径：同上
   - 目的：给出 DINO descriptor densify-only 的质量上界，但重点关注容量代价。

## 解释边界

第一批补充保持 VFM cache 与 0001 主线一致，因此它是“训练/渲染/基线口径对齐”的 direct comparison。若该批仍显示正向，再补更严格的 `images` / `max_width=1600` DINO descriptor cache，以区分“训练尺度收益”和“VFM 特征尺度收益”。

## 判定

每个候选必须报告：

- PSNR / SSIM / LPIPS
- Gaussian 数量
- 相对 Phase 0 FastGS big 的 delta
- QCGI

若全 9 场景均值不能超过 Phase 0 baseline，或只靠大幅增点换来薄收益，则不再把 0001 的旧口径最佳结果放进“fastgs_big direct”主表。

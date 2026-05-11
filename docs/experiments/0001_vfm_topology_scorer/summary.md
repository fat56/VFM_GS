# 0001 实验总结

## 定位

0001 的核心目标已经从“VFM 分支效果不好时回退 FastGS”修订为“验证视觉基础模型先验能否提升 Gaussian Splatting 训练质量”。最终应把 0001 视为 VFM_GS 的第一阶段验证实验：它证明 VFM 输出可以通过现有 FastGS 的 `pixel_error_map -> metric_map -> accum_metric_counts -> importance_score/pruning_score` 链路进入 densification/pruning，并且在多个数据集上产生可复现质量收益。

0001 的主要贡献不是单一最佳配置，而是把问题拆清楚：

- VFM scorer 的工程链路已经跑通，包括离线 cache、训练前 preflight、DINO token cache、在线 descriptor residual、metric-map 聚合和 Gaussian 级别计数。
- 语义/结构先验指导复制是有效的，最干净证据来自 DINO descriptor densify-only 系列。
- 直接把 VFM 介入 pruning 或用硬预算截断候选，很容易破坏训练轨迹；容量控制应转向场景自适应或几何边界先验。
- Tandt 这类负例不能靠最终容量下限、高权重或关闭 pruning fusion 单点修复；后续需要回退策略或新的几何先验。

## 最终主结论

0001 的 VFM_GS 初步验证应采用 DINO descriptor densify-only top-k25 weighted i0.50/i0.70 作为核心结果，而不是继续把体量更大的 token-edge selector 当作唯一主线。

这条 descriptor 线有三个优点：

- `vfm_weight=0.0`，不改变 pruning score，只让 VFM descriptor residual 影响 densification。
- 结论更接近方法贡献：“VFM 先验指导新增 Gaussians 的位置，从而提升质量”。
- top-k25 `max`、weighted i0.50 和 weighted i0.70 构成质量上界、容量受控档、质量折中档三档证据。

关键结果如下，均相对 matched FastGS densify100，且 LPIPS 为越低越好：

| 方法 | 范围 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | 结论 |
|---|---|---:|---:|---:|---:|---|
| descriptor top-k25 `max` | MipNeRF360 9 场景 | +0.1066 | +0.0050 | -0.0093 | +50,131 | 无回退质量优先档，9/9 场景三项正向 |
| descriptor top-k25 `max` | DB 2 场景 | +0.0085 | +0.0002 | -0.0011 | +9,189 | 数据集均值弱正向，`playroom` 单场景负向 |
| descriptor top-k25 `max` | Tandt 2 场景 | +0.1004 | +0.0017 | -0.0016 | +2,081 | 修复 token-edge 在 Tandt 低于 baseline 的问题 |
| descriptor top-k25 weighted i0.50 | 四场景初筛 | +0.0465 | +0.0040 | -0.0078 | +31,351 | 容量受控正向档，4/4 三项正向 |
| descriptor top-k25 weighted i0.70 | 四场景初筛 | +0.0884 | +0.0047 | -0.0086 | +35,511 | 质量折中档，较 i0.50 明显提升 PSNR |
| descriptor top-k25 weighted i0.65 | 四场景初筛 | +0.0282 | +0.0043 | -0.0083 | +35,190 | 不推荐，两个场景 PSNR 转负 |

因此，0001 的收束判断是：

- `top-k25 max` 保留为强制启用 VFM 的质量证据。
- `top-k25 weighted i0.50` 保留为容量受控正向档。
- `top-k25 weighted i0.70` 保留为质量-容量折中档。
- `i0.65`、soft budget、硬 candidate cap、final/staged target-prune 不进入主线。

## Token-Edge 与 Selector 结论

DINO token-edge top-k25 weighted 系列仍是有价值的第一版策略线，但它更适合作为 selector/工程策略结果，而不是“无回退 VFM 先验有效”的核心证据。

按公开数据集拆分的非 oracle 策略如下：

| 策略 | MipNeRF360 | DB | Tandt | 结论 |
|---|---|---|---|---|
| `dataset_fixed_policy` | fixed weighted i0.50 | fixed DINO weighted i0.90 | baseline 回退 | 保守展示线 |
| `dataset_quality_policy` | weighted QCGI 场景选择 | fixed DINO weighted i0.90 | baseline 回退 | 质量展示线 |

主要结果：

| 数据集 | 策略 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian |
|---|---|---:|---:|---:|---:|
| MipNeRF360 | fixed weighted i0.50 | +0.1979 | +0.0109 | -0.0223 | +81,395 |
| MipNeRF360 | weighted QCGI | +0.2114 | +0.0116 | -0.0231 | +82,481 |
| DB | fixed i0.90 | +0.4894 | +0.0051 | -0.0038 | +8,320 |
| Tandt | baseline 回退 | 0.0000 | 0.0000 | 0.0000 | 0 |

这条线的边界也很明确：

- 固定 i0.75/i0.90 不是跨数据集默认档。
- Tandt 上 i0.50/i0.75/i0.90 都低于 baseline，必须回退或换新先验。
- train-side render 指标无法替代 test oracle；自动 selector 后续需要 validation split、预先固定的数据集级策略或训练过程信号。

## Proxy 与负结果

`cached_edge_l1` 应保留为 proxy 正向控制组，而不是最终方法贡献。它在 MipNeRF360 和 DB 平均正向，但 Tandt 负向，说明纯边缘代理泛化不足。

已经收束的负结果：

- Tandt 容量保护只能恢复部分 cached-edge 损伤，不能超过 baseline。
- DINO weighted + 自动容量下限、prunemin-only、关闭 VFM pruning fusion 都不能修复 Tandt。
- Final prune、staged target prune 和 post-densify 大裁剪会破坏训练轨迹。
- 全局、3D 空间、clone/split、screen-support 等硬 candidate cap 能控点，但都会破坏有效 densification 分布。
- `adaptive_weighted + quadratic 430k` 在 bicycle 上正向，但 treehill 复验失败，不扩展为主线。
- COLMAP sparse depth-edge L1 和 sparse depth-edge prior 都低于 baseline，说明稀疏几何 proxy 覆盖不足。

## 高分辨率迁移

high-res descriptor weighted i0.50 在 MipNeRF360 9 场景均值正向，相对 FastGS big 为 +0.0615 PSNR、+0.0020 SSIM、LPIPS -0.0035，平均多 56,312 个点，QCGI 为 +0.0633。这说明 descriptor residual 在 1.6K 自动缩放口径下仍有效。

但 high-res 也暴露出下一阶段问题：

- `stump/bonsai` 容量过强。
- `flowers/treehill/truck` 出现质量或 QCGI 负例。
- 固定 metric 阈值不能跨场景泛化；stump metric6 正向，bonsai metric6 明显负向。
- 自适应 metric budget 在 stump 上优于固定阈值，但 bonsai 没有修复。

high-res 结论是：DINO descriptor 已经有迁移信号，但容量与边界结构需要新先验或自适应控制。

## 对 0002 的交接

0002 不应继续膨胀 0001 的 DINO 权重、top-k、candidate cap 或 staged target 扫描。下一阶段应改问一个新问题：

> Dense depth prior 能否补上 DINO descriptor 在几何边界、遮挡边界和 high-res 容量控制上的不足？

0002 的起点应是 Depth Anything dense depth prior：

- 先构建 GT 图像 dense depth cache。
- 再把渲染图或 SH0/albedo 图送入同一 depth 后端，得到 dense depth residual 或 depth-edge residual。
- 输出仍收敛为 `pixel_error_map`，复用 0001 已验证的 metric-map 和 per-Gaussian count 链路。
- 第一阶段只影响 densification，保持 `vfm_weight=0.0`，避免把 pruning 变量重新混入。
- 与 0001 的 descriptor top-k25 weighted i0.50/i0.70 做 matched 对照。

0001 的最终定位是：VFM_GS 的语义先验初步成立；0002 开始验证几何先验能否进一步提升结构边界和容量效率。

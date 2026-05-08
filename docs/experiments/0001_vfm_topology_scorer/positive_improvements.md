# 0001 正向改进汇总

本文只保留截至目前有正向效果或明确采用价值的 30k 实验结果；负向方案不进入主推荐表，只在末尾简述排除原因。除特别说明外，实验均为 `-r 8`、30,000 iterations、`--eval`。`ΔLPIPS` 为负数代表改善。

## MipNeRF360 全场景 v1

`cached_edge_l1 + npz_uint8 cache + staged target ~= 1.42x baseline count` 在 MipNeRF360 9 个场景上平均正向，可作为 0001 的 proxy 正向控制组。

| 方法 | PSNR | ΔPSNR | SSIM | ΔSSIM | LPIPS | ΔLPIPS | Gaussian 数量 | ΔGaussian | 训练时间 | Δ训练时间 | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 28.6527 | - | 0.8551 | - | 0.1620 | - | 173,341 | - | 125.38s | - | 控制组 |
| cached edge v1 | 28.7213 | +0.0686 | 0.8579 | +0.0028 | 0.1551 | -0.0068 | 215,869 | +42,528 | 139.33s | +13.95s | 保留为正向 proxy |

逐场景正向性：

| 场景 | baseline PSNR/SSIM/LPIPS | cached edge v1 PSNR/SSIM/LPIPS | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | 备注 |
|---|---|---|---:|---:|---:|---:|---:|---|
| bicycle | 26.6987 / 0.8061 / 0.2284 | 26.7886 / 0.8069 / 0.2240 | +0.0899 | +0.0009 | -0.0044 | +92,591 | +16.59s | 正向 |
| counter | 29.5411 / 0.9311 / 0.0806 | 29.6428 / 0.9321 / 0.0787 | +0.1017 | +0.0010 | -0.0018 | -1,898 | +7.85s | 少点且三项正向 |
| flowers | 22.7542 / 0.6723 / 0.3187 | 22.9676 / 0.6890 / 0.2862 | +0.2134 | +0.0167 | -0.0325 | +85,886 | +18.29s | 感知收益大 |
| garden | 28.7256 / 0.8893 / 0.1132 | 28.9003 / 0.8962 / 0.1002 | +0.1747 | +0.0069 | -0.0130 | +51,897 | +6.10s | 正向 |
| kitchen | 33.0920 / 0.9672 / 0.0379 | 33.3102 / 0.9691 / 0.0350 | +0.2182 | +0.0019 | -0.0029 | -10,196 | +9.41s | 少点且三项正向 |
| stump | 27.1756 / 0.7934 / 0.2327 | 27.2475 / 0.7932 / 0.2302 | +0.0718 | -0.0002 | -0.0026 | +71,719 | +22.49s | PSNR/LPIPS 正向 |

`bonsai` 和 `room` 的 SSIM/LPIPS 正向但 PSNR 小幅下降，适合作为感知改善观察；`treehill` 三项均负向，不作为正向结论。

## Bicycle 30k DINO/Edge 消融

该表展示从 baseline 到 edge、DINO top-k、partial importance 和 weighted 的主要正向点。`fastgs_densify100` 是 cadence control，用来拆分“更频繁 densification”与 VFM 信号本身的影响。

| 方案 | PSNR | ΔPSNR vs baseline | SSIM | ΔSSIM | LPIPS | ΔLPIPS | Gaussian 数量 | ΔGaussian | 训练时间 | 相对上一正向点 | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| baseline | 26.7032 | - | 0.8067 | - | 0.2278 | - | 240,394 | - | 116.92s | - | 控制组 |
| fastgs densify100 | 26.9287 | +0.2254 | 0.8241 | +0.0174 | 0.1964 | -0.0314 | 412,078 | +171,684 | 165.89s | - | cadence control |
| cached edge compact | 26.8864 | +0.1832 | 0.8229 | +0.0163 | 0.1972 | -0.0306 | 408,925 | +168,531 | 159.77s | 低于 cadence PSNR，但 edge proxy 正向 | proxy 正向 |
| DINO token-edge 默认 | 27.0577 | +0.3544 | 0.8345 | +0.0278 | 0.1767 | -0.0511 | 490,832 | +250,438 | 166.11s | 比 edge +0.1713 PSNR / -0.0205 LPIPS | 高质量但点数大 |
| DINO top-k 15% | 27.0223 | +0.3191 | 0.8322 | +0.0255 | 0.1810 | -0.0468 | 464,998 | +224,604 | 140.40s | 比默认少 25,834 点、质量小回落 | 更省点 |
| DINO top-k 25% | 27.0636 | +0.3603 | 0.8354 | +0.0288 | 0.1748 | -0.0530 | 497,328 | +256,934 | 146.76s | 比 top-k15 +0.0413 PSNR / -0.0062 LPIPS | bicycle 质量上界 |
| DINO top-k25 i0.25 | 26.9515 | +0.2483 | 0.8262 | +0.0196 | 0.1920 | -0.0358 | 420,361 | +179,967 | 143.26s | 比 top-k25 少 76,967 点 | 弱 VFM 引导有效 |
| DINO top-k25 i0.50 | 26.9966 | +0.2934 | 0.8303 | +0.0237 | 0.1842 | -0.0435 | 440,071 | +199,677 | 141.34s | 比 i0.25 +0.0451 PSNR / -0.0078 LPIPS | 当前质量/点数折中 |
| DINO top-k25 i0.75 | 27.0284 | +0.3251 | 0.8332 | +0.0265 | 0.1788 | -0.0490 | 472,164 | +231,770 | 155.18s | 比 i0.50 +0.0317 PSNR / -0.0055 LPIPS | 边际收益放缓 |
| DINO top-k25 weighted i0.50 | 26.9756 | +0.2724 | 0.8288 | +0.0221 | 0.1867 | -0.0411 | 415,158 | +174,764 | 141.21s | 比 i0.50 少 24,913 点，质量小幅回落 | 近预算效率候选 |

结论：Bicycle 上 DINO top-k 25% 给出最高质量；`importance_weight=0.50` 是较好的质量/点数折中；`weighted + i0.50` 进一步接近 cadence control 的点数，同时相对 cadence control 仍有 +0.0469 PSNR、+0.0047 SSIM、-0.0097 LPIPS，最有预算效率价值。

## DINO importance / weighted 模式

### MipNeRF360 全场景 DINO i0.50

`DINO token-edge top-k 25% + importance_weight=0.50` 已完成 MipNeRF360 9 场景复验，平均质量超过 baseline 和 cached edge v1；代价是平均 Gaussian 数量仍比 baseline 高约 52.1%。

| 方法 | PSNR | ΔPSNR vs baseline | SSIM | ΔSSIM | LPIPS | ΔLPIPS | Gaussian 数量 | ΔGaussian | 训练时间 | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline 平均 | 28.6527 | - | 0.8551 | - | 0.1620 | - | 173,341 | - | 125.38s | 控制组 |
| cached edge v1 平均 | 28.7213 | +0.0686 | 0.8579 | +0.0028 | 0.1551 | -0.0068 | 215,869 | +42,528 | 139.33s | proxy 正向 |
| DINO top-k25 i0.50 平均 | 28.8577 | +0.2051 | 0.8666 | +0.0115 | 0.1385 | -0.0234 | 263,572 | +90,231 | 140.47s | 当前全场景质量候选 |

相对 cached edge v1，DINO i0.50 平均继续提升 +0.1365 PSNR、+0.0087 SSIM、LPIPS 改善 -0.0166；同时平均多 47,703 个 Gaussians。

### 单场景正向复验

| 场景 | 方案 | PSNR | ΔPSNR vs baseline | SSIM | ΔSSIM | LPIPS | ΔLPIPS | Gaussian 数量 | ΔGaussian | 判断 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| counter | DINO top-k25 i0.50 | 29.7174 | +0.1763 | 0.9338 | +0.0026 | 0.0751 | -0.0055 | 119,695 | +6,672 | 少量增点、三项正向 |
| counter | DINO weighted i0.50 | 29.6650 | +0.1239 | 0.9333 | +0.0022 | 0.0752 | -0.0054 | 119,273 | +6,250 | 仍正向，但只比普通 i0.50 少 422 点且质量回落 |
| kitchen | DINO top-k25 i0.50 | 33.3358 | +0.2438 | 0.9693 | +0.0021 | 0.0344 | -0.0035 | 161,347 | -7,629 | 少点且三项正向 |
| room | DINO top-k25 i0.50 | 33.0721 | +0.0945 | 0.9622 | +0.0025 | 0.0574 | -0.0037 | 103,820 | +12,506 | 三项正向 |
| stump | DINO top-k25 i0.50 | 27.6106 | +0.4350 | 0.8168 | +0.0234 | 0.1935 | -0.0393 | 365,584 | +194,825 | 最大 PSNR 正例 |
| stump | DINO weighted i0.50 | 27.6147 | +0.4391 | 0.8170 | +0.0236 | 0.1934 | -0.0393 | 354,046 | +183,287 | 比普通 i0.50 少 11,538 点且质量略升 |
| bonsai | DINO top-k25 i0.50 | 32.4920 | +0.1346 | 0.9640 | +0.0044 | 0.0500 | -0.0123 | 136,305 | +12,963 | 三项正向 |
| flowers | DINO top-k25 i0.50 | 23.0134 | +0.2592 | 0.6960 | +0.0237 | 0.2747 | -0.0440 | 350,421 | +141,774 | 感知收益大 |
| garden | DINO top-k25 i0.50 | 28.9644 | +0.2388 | 0.8986 | +0.0092 | 0.0954 | -0.0177 | 262,385 | +65,878 | 三项正向 |

`treehill` 的 DINO i0.50 和 weighted 版本均改善 SSIM/LPIPS，但 PSNR 仍低于 baseline，因此不放入主推荐表；它是压力场景/边界观察。

## 跨数据集正向结果

### DB cached edge v1

DB 两个场景均为正向，说明 `cached_edge_l1` 不只在 MipNeRF360 上有效；但该结论不能直接外推到 Tandt。

| 范围 | 方法 | PSNR | ΔPSNR | SSIM | ΔSSIM | LPIPS | ΔLPIPS | Gaussian 数量 | ΔGaussian | 训练时间 | Δ训练时间 | 判断 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DB 平均 | baseline | 30.1179 | - | 0.9324 | - | 0.0658 | - | 54,685 | - | 121.22s | - | 控制组 |
| DB 平均 | cached edge v1 | 30.5631 | +0.4451 | 0.9361 | +0.0037 | 0.0637 | -0.0021 | 62,092 | +7,408 | 133.78s | +12.56s | 跨数据集正向 |
| drjohnson | cached edge v1 | 30.6034 | +0.1055 | 0.9283 | +0.0018 | 0.0726 | -0.0029 | 78,899 | +7,937 | 130.54s | +8.06s | 正向 |
| playroom | cached edge v1 | 30.5228 | +0.7847 | 0.9439 | +0.0055 | 0.0548 | -0.0013 | 45,286 | +6,878 | 137.03s | +17.07s | 强正向 |

### Tandt 容量保护的采用价值

Tandt 的 cached edge v1 低于 baseline，不作为正向质量方案；但 `prune_min_gaussian_count` 能显著恢复原始负例，适合作为默认关闭的诊断/回退保护。

| 范围 | 方法 | PSNR | ΔPSNR vs cached edge v1 | SSIM | ΔSSIM | LPIPS | ΔLPIPS | Gaussian 数量 | ΔGaussian | 判断 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Tandt 平均 | cached edge v1 | 25.5799 | - | 0.9316 | - | 0.0608 | - | 31,562 | - | 负例基准 |
| Tandt 平均 | cached edge + 容量保护 | 25.7806 | +0.2007 | 0.9337 | +0.0021 | 0.0585 | -0.0023 | 50,370 | +18,808 | 可采用为保护机制 |

容量保护后仍低于 Tandt baseline 的 25.9551 / 0.9377 / 0.0541，因此它是防线，不是主质量方案。

## 已排除或仅作边界观察

| 方案 | 结论 |
|---|---|
| Tandt cached edge v1 | `train`、`truck` 均负向，平均 PSNR -0.3752、SSIM -0.0061、LPIPS +0.0067；不作为推荐项。 |
| 严格 baseline-count final prune / high-score prune | 即使点数精确匹配，也会显著破坏质量；high-score final prune 尤其不稳定。 |
| descriptor staged / rgb_only / soft-topk staged / dense recovery | unpruned descriptor 有质量收益，但接近预算或恢复后仍未稳定超过 cadence control；暂不作为主路线。 |
| support_ratio 与 prune-protect | Bicycle 上没有优于普通 DINO i0.50；收束为非主方向。 |
| budget-aware 420k/430k 曲线 | 比固定低权重略好，但没有保住普通 i0.50 质量；不继续手工追加相近曲线单点。 |
| treehill weighted | 相比普通 treehill i0.50 少 14,986 点且 SSIM/LPIPS 对 baseline 正向，但 PSNR 仍低于 baseline；作为压力场景观察，不作为正向主结论。 |
| counter weighted | 相比 baseline 和 cached-edge v1 仍三项正向，但相对普通 counter i0.50 只少 422 点且 PSNR 回落 -0.0524；用于说明 weighted 不适合默认替代低增点场景的普通 i0.50。 |

## 简短结论

目前最值得保留的改进有三类：第一，`cached_edge_l1 + staged target ~= 1.42x` 是稳定的 proxy 正向控制组，在 MipNeRF360 平均和 DB 平均均正向；第二，`DINO token-edge top-k25 + importance_weight=0.50` 是当前 MipNeRF360 全场景质量最强候选，平均相对 baseline 提升 +0.2051 PSNR、+0.0115 SSIM、-0.0234 LPIPS；第三，`weighted + importance_weight=0.50` 是最值得继续验证的近预算效率点，已在 bicycle 上接近 cadence control 点数，在 stump 上还比普通 i0.50 更省点且质量略升，但 counter 说明它不适合无条件替代普通 i0.50。

仍未解决的问题是预算机制和跨数据集稳健性：DINO i0.50 平均 Gaussian 数量仍比 baseline 多约 52.1%，`treehill` PSNR 仍有压力，Tandt 上 cached edge v1 明确负向，容量保护只能恢复一部分质量。

下一步建议优先做两件事：把 `weighted + i0.50` 扩展到更多 MipNeRF360 正例和压力场景，确认它是否能保留 DINO i0.50 的主要收益；同时设计自动场景容量保护或预算感知 scorer，避免 Tandt 这类场景被 pruning/cadence 组合压得过稀。

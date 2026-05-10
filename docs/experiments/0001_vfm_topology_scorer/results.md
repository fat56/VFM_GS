# 0001 实验结果

## 2026-05-07 MipNeRF360 全场景 v1 评估

评估范围：`datasets/mipnerf360` 全 9 个场景，`-r 8`，30,000 iterations，`--eval`。每个场景统一重跑 `fastgs_baseline` 与当前 v1 正向版本 `cached_edge_l1 + npz_uint8 cache + staged target`。v1 的 `target_gaussian_count` 设为该场景 baseline Gaussian 数量的 `1.42x`，`target_gaussian_stage_margin=1.10`，`target_gaussian_stage_interval=500`。测试指标来自 test split render 后的 `metrics`，原始汇总产物在 `output/0001/full_mipnerf360_v1/summary.csv`。

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| bicycle | baseline | 26.6987 | 0.8061 | 0.2284 | 241,383 | 125.44s |
| bicycle | cached edge v1 | 26.7886 | 0.8069 | 0.2240 | 333,974 | 142.03s |
| bonsai | baseline | 32.3574 | 0.9596 | 0.0623 | 123,342 | 129.53s |
| bonsai | cached edge v1 | 32.2266 | 0.9600 | 0.0588 | 111,795 | 139.83s |
| counter | baseline | 29.5411 | 0.9311 | 0.0806 | 113,023 | 124.27s |
| counter | cached edge v1 | 29.6428 | 0.9321 | 0.0787 | 111,125 | 132.12s |
| flowers | baseline | 22.7542 | 0.6723 | 0.3187 | 208,647 | 123.97s |
| flowers | cached edge v1 | 22.9676 | 0.6890 | 0.2862 | 294,533 | 142.26s |
| garden | baseline | 28.7256 | 0.8893 | 0.1132 | 196,507 | 128.93s |
| garden | cached edge v1 | 28.9003 | 0.8962 | 0.1002 | 248,404 | 135.03s |
| kitchen | baseline | 33.0920 | 0.9672 | 0.0379 | 168,976 | 132.09s |
| kitchen | cached edge v1 | 33.3102 | 0.9691 | 0.0350 | 158,780 | 141.51s |
| room | baseline | 32.9776 | 0.9597 | 0.0612 | 91,314 | 121.77s |
| room | cached edge v1 | 32.9685 | 0.9620 | 0.0578 | 98,899 | 135.43s |
| stump | baseline | 27.1756 | 0.7934 | 0.2327 | 170,759 | 121.11s |
| stump | cached edge v1 | 27.2475 | 0.7932 | 0.2302 | 242,478 | 143.60s |
| treehill | baseline | 24.5517 | 0.7175 | 0.3228 | 246,117 | 121.30s |
| treehill | cached edge v1 | 24.4394 | 0.7126 | 0.3253 | 342,829 | 142.14s |
| **平均** | **baseline** | **28.6527** | **0.8551** | **0.1620** | **173,341** | **125.38s** |
| **平均** | **cached edge v1** | **28.7213** | **0.8579** | **0.1551** | **215,869** | **139.33s** |

逐场景差值：

| 场景 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 |
|---|---:|---:|---:|---:|---:|
| bicycle | +0.0899 | +0.0009 | -0.0044 | +92,591 | +16.59s |
| bonsai | -0.1308 | +0.0004 | -0.0035 | -11,547 | +10.30s |
| counter | +0.1017 | +0.0010 | -0.0018 | -1,898 | +7.85s |
| flowers | +0.2134 | +0.0167 | -0.0325 | +85,886 | +18.29s |
| garden | +0.1747 | +0.0069 | -0.0130 | +51,897 | +6.10s |
| kitchen | +0.2182 | +0.0019 | -0.0029 | -10,196 | +9.41s |
| room | -0.0091 | +0.0023 | -0.0034 | +7,585 | +13.66s |
| stump | +0.0718 | -0.0002 | -0.0026 | +71,719 | +22.49s |
| treehill | -0.1123 | -0.0048 | +0.0025 | +96,712 | +20.84s |

解读：

- 全 9 场景平均结果为正向：PSNR +0.0686、SSIM +0.0028、LPIPS -0.0068。
- 平均 Gaussian 数量从 173,341 增加到 215,869，约 +24.5%；平均训练时间从 125.38s 增加到 139.33s，约 +11.1%。
- 6/9 场景 PSNR 提升；8/9 场景 SSIM 提升；8/9 场景 LPIPS 改善。`treehill` 是三项指标同时变差的主要负例，`bonsai` 与 `room` 的 PSNR 小幅下降但 SSIM/LPIPS 改善。
- `counter` 与 `kitchen` 在 Gaussian 数量少于 baseline 的情况下仍提升三项指标，说明 v1 的正向结果不完全依赖更大点数预算。
- `flowers` 和 `garden` 收益最明显，尤其 LPIPS 分别改善 -0.0325 和 -0.0130，说明 edge-alignment proxy 对复杂植被/纹理边界有稳定帮助。
- `treehill` 同时增加约 96,712 个 Gaussians 但质量下降，说明固定 `1.42x` ratio target 并非所有场景稳健。下一版应引入场景自适应 budget 或把 edge/VFM 信号改为 prune-protect，而不是直接推动更多 densification。
- 因此，`cached_edge_l1` 可以作为 0001 v1 的 MipNeRF360 全场景正向控制组，但它仍是边缘代理，不是最终语义 VFM scorer；更进一步的研究应解决场景自适应预算和语义 descriptor 预算高效化。

## 2026-05-07 Tandt/DB 全场景 v1 评估

评估范围：`datasets/tandt_db/db` 的 `drjohnson`、`playroom`，以及 `datasets/tandt_db/tandt` 的 `train`、`truck`。训练设置与 MipNeRF360 全场景评估保持一致：`-r 8`，30,000 iterations，`--eval`，每个场景重跑 `fastgs_baseline` 与 `cached_edge_l1 + npz_uint8 cache + staged target`。这两个数据集没有 `images_8` 目录，因此训练仍用 `-i images`，edge cache 也从 `images` 构建。测试指标来自 test split render 后的 `metrics`；原始汇总产物在 `output/0001/full_tandt_db_v1/db/summary.csv` 和 `output/0001/full_tandt_db_v1/tandt/summary.csv`。

`db` 结果：

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| drjohnson | baseline | 30.4978 | 0.9265 | 0.0755 | 70,962 | 122.48s |
| drjohnson | cached edge v1 | 30.6034 | 0.9283 | 0.0726 | 78,899 | 130.54s |
| playroom | baseline | 29.7381 | 0.9384 | 0.0561 | 38,408 | 119.96s |
| playroom | cached edge v1 | 30.5228 | 0.9439 | 0.0548 | 45,286 | 137.03s |
| **平均** | **baseline** | **30.1179** | **0.9324** | **0.0658** | **54,685** | **121.22s** |
| **平均** | **cached edge v1** | **30.5631** | **0.9361** | **0.0637** | **62,092** | **133.78s** |

`tandt` 结果：

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| train | baseline | 23.6772 | 0.9154 | 0.0752 | 58,788 | 145.92s |
| train | cached edge v1 | 23.4054 | 0.9081 | 0.0837 | 35,322 | 148.42s |
| truck | baseline | 28.2330 | 0.9601 | 0.0330 | 41,952 | 134.59s |
| truck | cached edge v1 | 27.7543 | 0.9550 | 0.0379 | 27,802 | 141.28s |
| **平均** | **baseline** | **25.9551** | **0.9377** | **0.0541** | **50,370** | **140.26s** |
| **平均** | **cached edge v1** | **25.5799** | **0.9316** | **0.0608** | **31,562** | **144.85s** |

逐场景差值：

| 数据集 | 场景 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 |
|---|---|---:|---:|---:|---:|---:|
| db | drjohnson | +0.1055 | +0.0018 | -0.0029 | +7,937 | +8.06s |
| db | playroom | +0.7847 | +0.0055 | -0.0013 | +6,878 | +17.07s |
| tandt | train | -0.2718 | -0.0072 | +0.0085 | -23,466 | +2.50s |
| tandt | truck | -0.4787 | -0.0051 | +0.0049 | -14,150 | +6.68s |

数据集平均差值：

| 数据集 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 |
|---|---:|---:|---:|---:|---:|
| db | +0.4451 | +0.0037 | -0.0021 | +7,407.5 | +12.56s |
| tandt | -0.3752 | -0.0061 | +0.0067 | -18,808.0 | +4.59s |
| db+tandt | +0.0350 | -0.0012 | +0.0023 | -5,700.3 | +8.58s |

解读：

- `db` 两个场景均为正向，平均提升 +0.4451 PSNR、+0.0037 SSIM、LPIPS 改善 -0.0021；平均 Gaussian 数量增加约 13.5%，训练时间增加约 10.4%。
- `tandt` 两个场景均为负向，平均下降 -0.3752 PSNR、-0.0061 SSIM、LPIPS 变差 +0.0067；同时平均 Gaussian 数量下降约 37.3%，说明负向结果不是由点数膨胀导致，而更可能是当前 edge/pruning 组合在这两个场景上过度抑制了结构。
- 四个新增场景合并后，PSNR 只微幅正向（+0.0350），但 SSIM 和 LPIPS 负向，不能把 `cached_edge_l1` 简单推广为 Tandt/DB 的稳定正向方案。
- 与 MipNeRF360 全场景结果对比，`db` 更接近室内重建收益模式，而 `tandt` 暴露出跨数据集泛化风险。下一版应加入场景自适应 target 或 pruning 强度控制，特别是当 edge v1 的自然结束点数显著低于 baseline 时，应触发保护机制或回退到 baseline pruning。

## 2026-05-07 Tandt 容量保护诊断

目标：解释 Tandt `train` 和 `truck` 中 `cached_edge_l1` 负向且 Gaussian 数量显著低于 baseline 的原因，并测试最小容量保护是否能恢复质量。所有实验继续使用 `datasets/tandt_db/tandt`、`-r 8`、30,000 iterations、`--eval`，测试指标来自 test split。

代码变更：增加 `prune_min_gaussian_count`，默认值为 `0`，默认关闭。该参数会限制 FastGS 训练期抽样裁剪和最终一致性裁剪的最大删除量，避免点数低于指定下限。

`train` 诊断：

| 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 | 说明 |
|---|---:|---:|---:|---:|---:|---|
| baseline | 23.6772 | 0.9154 | 0.0752 | 58,788 | 145.92s | 原始 FastGS |
| cached edge v1 | 23.4054 | 0.9081 | 0.0837 | 35,322 | 148.42s | v1 负例 |
| cached edge, `vfm_weight=0.0` | 23.2628 | 0.9070 | 0.0820 | 35,698 | 146.52s | 关闭 pruning fusion 仍未恢复容量 |
| FastGS, `densification_interval=100` | 23.6091 | 0.9128 | 0.0770 | 43,488 | 139.01s | 只改变 densification cadence 已低于 baseline |
| cached edge, `densification_interval=500` | 23.6147 | 0.9094 | 0.0824 | 40,419 | 137.65s | 回到 baseline cadence 后 PSNR 接近 cadence 控制，但感知指标仍弱 |
| cached edge + `prune_min_gaussian_count=58788` | 23.5970 | 0.9104 | 0.0804 | 58,788 | 148.36s | 容量恢复，质量部分恢复但未超过 baseline |

`tandt` 两场景容量保护结果：

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| train | baseline | 23.6772 | 0.9154 | 0.0752 | 58,788 | 145.92s |
| train | cached edge v1 | 23.4054 | 0.9081 | 0.0837 | 35,322 | 148.42s |
| train | cached edge + 容量保护 | 23.5970 | 0.9104 | 0.0804 | 58,788 | 148.36s |
| truck | baseline | 28.2330 | 0.9601 | 0.0330 | 41,952 | 134.59s |
| truck | cached edge v1 | 27.7543 | 0.9550 | 0.0379 | 27,802 | 141.28s |
| truck | cached edge + 容量保护 | 27.9641 | 0.9570 | 0.0366 | 41,952 | 139.71s |
| **平均** | **baseline** | **25.9551** | **0.9377** | **0.0541** | **50,370** | **140.26s** |
| **平均** | **cached edge v1** | **25.5799** | **0.9316** | **0.0608** | **31,562** | **144.85s** |
| **平均** | **cached edge + 容量保护** | **25.7806** | **0.9337** | **0.0585** | **50,370** | **144.04s** |

解读：

- `prune_min_gaussian_count` 能把 Tandt 两个场景的最终 Gaussian 数量拉回 baseline 水平，并相对原始 cached edge v1 恢复 +0.2007 PSNR、+0.0021 SSIM、LPIPS 改善 -0.0023。
- 容量保护后的平均结果仍低于 baseline：PSNR -0.1745、SSIM -0.0040、LPIPS 差 +0.0044。因此 Tandt 负例不只是“删得太多”，也包含 edge signal、densification cadence 与 pruning trajectory 对结构分布的影响。
- `vfm_weight=0.0` 仍保持低点数，说明单独关闭 VFM pruning fusion 不足以修复 Tandt；`densification_interval=100` 本身也会让 `train` 的点数和质量低于 baseline。下一版应把容量保护作为防线，但主改动应转向场景自适应 scorer/cadence，而不是简单固定最小点数。
- 该参数默认关闭，适合做诊断和回退保护；若用于正式方案，需要自动估计 `min_gaussian_count`，例如由 baseline 预跑、场景尺度或在线增长曲线预测。

## 2026-05-08 Tandt 自动容量下限复验

目标：把手动填写的 `prune_min_gaussian_count` 推进一步，改为通过 `prune_min_gaussian_target_ratio` 从 `target_gaussian_count` 自动派生。`configs/experiments/0001_vfm_topology_cached_edge_auto_prunemin.yaml` 使用 ratio `0.7042253521126761`；当 target 仍取 `baseline * 1.42` 时，该 ratio 等价于 baseline 最终点数。显式 `prune_min_gaussian_count` 仍优先。

代码行为：训练开始时会计算有效 `prune_min_gaussian_count`，写回 `opt`，因此训练期抽样裁剪、最终一致性裁剪和输出目录中的 `cfg_args` 都记录同一个有效下限。`train` 自动派生为 58,788，`truck` 自动派生为 41,952。

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 | 输出路径 |
|---|---|---:|---:|---:|---:|---:|---|
| train | cached edge + 自动容量下限 | 23.5604 | 0.9088 | 0.0826 | 58,788 | 151.27s | `output/0001/full_tandt_db_v1/tandt/train/vfm_cached_edge_autoprunemin_staged142_30k_r8` |
| truck | cached edge + 自动容量下限 | 27.9342 | 0.9573 | 0.0363 | 41,952 | 143.17s | `output/0001/full_tandt_db_v1/tandt/truck/vfm_cached_edge_autoprunemin_staged142_30k_r8` |
| **平均** | **cached edge + 自动容量下限** | **25.7473** | **0.9330** | **0.0594** | **50,370** | **147.22s** | - |

对比：

| 参照 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 |
|---|---:|---:|---:|---:|---:|
| 相对 cached edge v1 | +0.1674 | +0.0014 | -0.0014 | +18,808 | +2.37s |
| 相对手动容量保护 | -0.0333 | -0.0007 | +0.0009 | +0 | +3.18s |
| 相对 baseline | -0.2078 | -0.0047 | +0.0053 | +0 | +6.97s |

解读：

- 自动下限复验达成机制目标：不再手填 `prune_min_gaussian_count`，仍能从 staged target 派生出与 baseline 容量一致的保护下限，并在 `cfg_args` 中记录有效值。
- 质量相对原始 cached edge v1 仍有恢复，但略低于 2026-05-07 的手动容量保护 run。两组最终 Gaussian 数量相同，因此差异主要来自运行方差或训练轨迹微小差别；不能把自动下限解读成新的质量提升。
- 结论保持不变：容量下限是 Tandt 负例的必要防线，但不是完整解法。下一步应让 ratio 或下限来源于 baseline 预跑、在线增长曲线或场景尺度估计，而不是固定绑定 `baseline * 1.42` 这个诊断比例。

## 2026-05-08 Tandt DINO weighted i0.50 跨数据集复验

目标：检查 MipNeRF360 上的预算效率候选 `dinov2_token_edge_l1 + top-k25 + weighted + importance_weight=0.50` 能否迁移到 Tandt。实验使用 `scripts/run_0001_dino_weighted_eval.py`，数据集为 `datasets/tandt_db/tandt`，图像目录为 `images`，训练仍为 `-r 8`、30,000 iterations、`--eval`。DINO cache 使用 `dinov2_vits14`、`max_width=224`、`npy_float16`，输出在 `output/0001/dino_weighted_i050_tandt`。

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 | 输出路径 |
|---|---|---:|---:|---:|---:|---:|---|
| train | DINO weighted i0.50 | 23.5947 | 0.9122 | 0.0788 | 42,160 | 155.17s | `output/0001/dino_weighted_i050_tandt/train/vfm_dinov2_token_edge_topk025_weighted_i050_30k_r8` |
| truck | DINO weighted i0.50 | 27.9092 | 0.9571 | 0.0363 | 34,628 | 146.92s | `output/0001/dino_weighted_i050_tandt/truck/vfm_dinov2_token_edge_topk025_weighted_i050_30k_r8` |
| **平均** | **DINO weighted i0.50** | **25.7519** | **0.9346** | **0.0575** | **38,394** | **151.05s** | - |

对比：

| 参照 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| 相对 cached edge v1 | +0.1721 | +0.0031 | -0.0033 | +6,832 | +6.20s | 两个场景均三项指标正向，说明 DINO topology 比 edge proxy 更能修复 Tandt 负例 |
| 相对 baseline | -0.2032 | -0.0031 | +0.0035 | -11,976 | +10.79s | 平均质量仍低于 baseline，不能作为 Tandt 默认方案 |

逐场景差值：

| 场景 | 参照 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 |
|---|---|---:|---:|---:|---:|---:|
| train | baseline | -0.0825 | -0.0032 | +0.0036 | -16,628 | +9.26s |
| train | cached edge v1 | +0.1893 | +0.0040 | -0.0049 | +6,838 | +6.75s |
| truck | baseline | -0.3238 | -0.0030 | +0.0033 | -7,324 | +12.33s |
| truck | cached edge v1 | +0.1549 | +0.0021 | -0.0016 | +6,826 | +5.65s |

解读：

- DINO weighted i0.50 在 Tandt 上相对 `cached_edge_l1` 是清晰正向：`train` 和 `truck` 都提升 PSNR/SSIM，并降低 LPIPS。这说明 DINO token topology 的跨数据集鲁棒性优于纯边缘 proxy。
- 它仍低于 Tandt baseline，且训练时间更长。与容量保护不同，它没有把点数拉回 baseline，而是在少于 baseline 平均 11,976 个 Gaussians 的条件下修复一部分 cached-edge 损伤。
- 因此该结果应作为“VFM 分支之间的正向替代”记录，而不是 Tandt 的最终默认方案。下一步更值得继续跑 DB 两场景：如果 DB 上也相对 cached-edge 或 baseline 保持正向，就能更清楚地区分室内/受控场景与 Tandt 户外轨迹场景的选择规则。

## 2026-05-08 DB DINO weighted i0.50 跨数据集复验

目标：用同一脚本复验 DB 两场景，判断 DINO weighted i0.50 是否能在更接近室内/受控轨迹的数据集上超过 baseline 或替代 cached-edge v1。实验使用 `datasets/tandt_db/db`，场景为 `drjohnson` 和 `playroom`，其他设置与 Tandt 复验一致。

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 | 输出路径 |
|---|---|---:|---:|---:|---:|---:|---|
| drjohnson | DINO weighted i0.50 | 30.6889 | 0.9299 | 0.0706 | 80,243 | 143.40s | `output/0001/dino_weighted_i050_db/drjohnson/vfm_dinov2_token_edge_topk025_weighted_i050_30k_r8` |
| playroom | DINO weighted i0.50 | 30.0316 | 0.9420 | 0.0576 | 44,279 | 139.56s | `output/0001/dino_weighted_i050_db/playroom/vfm_dinov2_token_edge_topk025_weighted_i050_30k_r8` |
| **平均** | **DINO weighted i0.50** | **30.3603** | **0.9360** | **0.0641** | **62,261** | **141.48s** | - |

对比：

| 参照 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| 相对 baseline | +0.2423 | +0.0035 | -0.0017 | +7,576 | +20.26s | 平均三项指标正向，但 `playroom` 的 LPIPS 单项变差 |
| 相对 cached edge v1 | -0.2028 | -0.0001 | +0.0004 | +169 | +7.70s | 平均略低于 cached-edge，不能替代 DB proxy 正向控制组 |

逐场景差值：

| 场景 | 参照 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 |
|---|---|---:|---:|---:|---:|---:|
| drjohnson | baseline | +0.1911 | +0.0034 | -0.0050 | +9,281 | +20.92s |
| drjohnson | cached edge v1 | +0.0856 | +0.0016 | -0.0020 | +1,344 | +12.86s |
| playroom | baseline | +0.2935 | +0.0036 | +0.0015 | +5,871 | +19.60s |
| playroom | cached edge v1 | -0.4912 | -0.0019 | +0.0028 | -1,007 | +2.54s |

解读：

- DB 上 DINO weighted i0.50 相对 baseline 的平均结果是正向的，尤其 `drjohnson` 同时超过 baseline 和 cached-edge v1，说明 DINO topology 在室内/受控场景中仍有可用质量信号。
- `playroom` 暴露了边界：它相对 baseline 提升 PSNR/SSIM，但 LPIPS 变差；相对 cached-edge v1 则三项指标均回落。因此 DB 的当前默认正向控制组仍应保留 `cached_edge_l1`，DINO weighted i0.50 不能无条件替代。
- 结合 Tandt 结果，跨数据集选择规则更清楚：DINO weighted i0.50 比 cached-edge 更适合修复 Tandt 这类 edge proxy 负例，但在 DB 上 cached-edge 仍更强。下一版应把 scene-level 选择规则纳入批量评估，而不是假设单一 VFM 后端跨数据集最优。

## 2026-05-08 跨数据集后端选择汇总

新增 `scripts/summarize_0001_cross_dataset_selector.py`，把 MipNeRF360 9 场景、Tandt 2 场景和 DB 2 场景中的 baseline、cached-edge v1、DINO weighted i0.50 汇总成统一选择表。该脚本不启动训练，只读取已完成的 summary 文件，输出到 `output/0001/cross_dataset_selector`。

按公开数据集分别求平均：

| 数据集 | 方法 | 场景数 | PSNR | ΔPSNR vs baseline | SSIM | ΔSSIM | LPIPS | ΔLPIPS | Gaussian 数量 | ΔGaussian | 训练时间 | Δ训练时间 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MipNeRF360 | baseline | 9 | 28.6527 | 0.0000 | 0.8551 | 0.0000 | 0.1620 | 0.0000 | 173,341 | 0 | 125.38s | 0.00s |
| MipNeRF360 | cached-edge v1 | 9 | 28.7213 | +0.0686 | 0.8579 | +0.0028 | 0.1551 | -0.0068 | 215,869 | +42,528 | 139.33s | +13.95s |
| MipNeRF360 | DINO weighted i0.50 | 9 | 28.8505 | +0.1979 | 0.8660 | +0.0109 | 0.1397 | -0.0223 | 254,736 | +81,395 | 137.60s | +12.22s |
| DB | baseline | 2 | 30.1179 | 0.0000 | 0.9324 | 0.0000 | 0.0658 | 0.0000 | 54,685 | 0 | 121.22s | 0.00s |
| DB | cached-edge v1 | 2 | 30.5631 | +0.4451 | 0.9361 | +0.0037 | 0.0637 | -0.0021 | 62,092 | +7,408 | 133.78s | +12.56s |
| DB | DINO weighted i0.50 | 2 | 30.3603 | +0.2423 | 0.9360 | +0.0035 | 0.0641 | -0.0017 | 62,261 | +7,576 | 141.48s | +20.26s |
| Tandt | baseline | 2 | 25.9551 | 0.0000 | 0.9377 | 0.0000 | 0.0541 | 0.0000 | 50,370 | 0 | 140.26s | 0.00s |
| Tandt | cached-edge v1 | 2 | 25.5799 | -0.3752 | 0.9316 | -0.0061 | 0.0608 | +0.0067 | 31,562 | -18,808 | 144.85s | +4.59s |
| Tandt | DINO weighted i0.50 | 2 | 25.7519 | -0.2032 | 0.9346 | -0.0031 | 0.0575 | +0.0035 | 38,394 | -11,976 | 151.05s | +10.79s |

选择器输出中的 `all` 行只作为脚本调试和覆盖检查，不作为实验结论口径。论文式比较必须按 MipNeRF360、DB、Tandt 三个数据集分别报告平均值。

QCGI 定义：

```text
quality_gain = ΔPSNR + 20 * ΔSSIM - 5 * ΔLPIPS
gs_penalty = 0.01 * min(max(ΔGS, 0), 100000) / 10000
           + 0.04 * max(ΔGS - 100000, 0) / 10000
QCGI = quality_gain - gs_penalty
```

该指数的出发点是支持正向且少量的 Gaussian 增长，抑制低效膨胀。当前分档为：`sub_0.01M` 轻量增长、`0.01M_to_0.10M` 可接受增长、`gte_0.10M` 重惩罚增长。典型结果是：DB `playroom` 的 cached-edge v1 只增加 6,878 个点且质量明显提升，QCGI 为正；MipNeRF360 `treehill` 的 DINO weighted i0.50 虽然改善 SSIM/LPIPS，但增加 171,417 个点且 PSNR 低于 baseline，QCGI 为负。

逐场景推荐摘要：

| 推荐类型 | 场景 | 数量 | 说明 |
|---|---|---:|---|
| `best_psnr_method = dino_weighted_i050` | DB `drjohnson`；MipNeRF360 `bicycle/bonsai/counter/garden/kitchen/room/stump` | 8 | DINO weighted i0.50 是这些场景的 PSNR 最优后端 |
| `best_psnr_method = cached_edge_staged142` | DB `playroom`；MipNeRF360 `flowers` | 2 | edge proxy 仍是这些场景的 PSNR 最优 VFM 后端 |
| `best_psnr_method = baseline` | MipNeRF360 `treehill`；Tandt `train/truck` | 3 | 当前不应默认启用 VFM 后端；Tandt 中 DINO weighted 只适合修复 cached-edge 负例 |

解读：

- 分数据集看，MipNeRF360 与 DB 都存在明确 VFM 正向平均收益；Tandt 两场景的平均指标仍以 baseline 为上界，DINO weighted i0.50 只能作为相对 cached-edge 的修复分支。
- 单一后端不能解释所有场景：DB `playroom` 和 MipNeRF360 `flowers` 更偏 cached-edge，Tandt 两场景和 MipNeRF360 `treehill` 的 PSNR 最优仍是 baseline。
- 下一版不应继续只追加固定后端单点，而应做场景级选择或自动回退：先判断是否启用 VFM，再在 cached-edge 与 DINO weighted 之间选择。如果只能保守上线，应以 baseline 作为预算安全回退，以 DINO weighted/cached-edge 作为有证据的场景级增强。

## 2026-05-09 Tandt/DB DINO weighted 多档复验

目标：把 MipNeRF360 上已经完成的 `weighted i0.75/i0.90` 扩展到 `datasets/tandt_db`，验证更激进的 VFM importance 是否能改善跨数据集结果。实验仍使用 `-r 8`、30,000 iterations、`--eval`，DINO cache 复用 `output/0001/vfm_cache`，输出分别在 `output/0001/dino_weighted_i075_tandt`、`output/0001/dino_weighted_i090_tandt`、`output/0001/dino_weighted_i075_db` 和 `output/0001/dino_weighted_i090_db`。

Tandt 结果：

| 方法 | 场景数 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 2 | 25.9551 | 0.9377 | 0.0541 | 50,370 | 140.26s | Tandt 当前质量上界 |
| cached-edge v1 | 2 | 25.5799 | 0.9316 | 0.0608 | 31,562 | 144.85s | 明确负例 |
| DINO weighted i0.50 | 2 | 25.7519 | 0.9346 | 0.0575 | 38,394 | 151.05s | 相对 cached-edge 修复，但仍低于 baseline |
| DINO weighted i0.75 | 2 | 25.6201 | 0.9328 | 0.0585 | 37,986 | 214.65s | 低于 i0.50，不保留为默认 |
| DINO weighted i0.90 | 2 | 25.5329 | 0.9323 | 0.0611 | 37,523 | 212.83s | 继续退化，不保留 |

DB 结果：

| 方法 | 场景数 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 2 | 30.1179 | 0.9324 | 0.0658 | 54,685 | 121.22s | 控制组 |
| cached-edge v1 | 2 | 30.5631 | 0.9361 | 0.0637 | 62,092 | 133.78s | 早期 proxy 正向控制组 |
| DINO weighted i0.50 | 2 | 30.3603 | 0.9360 | 0.0641 | 62,261 | 141.48s | 相对 baseline 正向，低于 cached-edge |
| DINO weighted i0.75 | 2 | 30.5446 | 0.9358 | 0.0633 | 63,034 | 181.81s | 接近 cached-edge，两个场景都高于 baseline |
| DINO weighted i0.90 | 2 | 30.6074 | 0.9376 | 0.0620 | 63,006 | 197.47s | 同时超过 baseline 和 cached-edge，是 DB 当前 DINO 高质量档 |

DB i0.90 逐场景对比：

| 场景 | 参照 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 |
|---|---|---:|---:|---:|---:|---:|
| drjohnson | baseline | +0.1004 | +0.0037 | -0.0063 | +12,281 | +76.96s |
| drjohnson | cached-edge v1 | -0.0052 | +0.0019 | -0.0033 | +4,344 | +68.91s |
| playroom | baseline | +0.8785 | +0.0066 | -0.0013 | +4,360 | +75.52s |
| playroom | cached-edge v1 | +0.0938 | +0.0011 | -0.0000 | -2,518 | +58.46s |

解读：

- Tandt 高权重档位是负向证据。i0.75/i0.90 都比 i0.50 更低，且训练时间明显增加，因此 Tandt 当前策略应回退 baseline，或只把 i0.50 作为“相对 cached-edge 修复”的诊断分支。
- DB 高权重档位是正向证据。i0.90 在两个场景上都相对 baseline 三项正向，平均也超过 cached-edge v1；`playroom` 尤其说明高权重 DINO 可以修复 i0.50 低于 cached-edge 的问题。
- 同一个 fixed weight 不能跨数据集通吃。真正有价值的是把 i0.50/i0.75/i0.90 作为场景候选池，再由 QCGI 或严格质量门槛选择。

## 2026-05-09 跨数据集多档 selector 汇总

`scripts/summarize_0001_cross_dataset_selector.py` 已扩展为读取 MipNeRF360/Tandt/DB 的 DINO weighted i0.50/i0.75/i0.90，并把 `best_dino_method`、`best_dino_vs_baseline_status`、`dino_weighted_i075_vs_baseline_status` 和 `dino_weighted_i090_vs_baseline_status` 写入 `recommendations.csv`。最终产物仍在 `output/0001/cross_dataset_selector`。

固定方法均值按公开数据集分别报告如下。`all` 合并行仍由脚本生成，但不作为主结论。

| 数据集 | 方法 | 场景数 | PSNR | ΔPSNR vs baseline | SSIM | ΔSSIM | LPIPS | ΔLPIPS | Gaussian 数量 | ΔGaussian | 训练时间 | Δ训练时间 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MipNeRF360 | baseline | 9 | 28.6527 | 0.0000 | 0.8551 | 0.0000 | 0.1620 | 0.0000 | 173,341 | 0 | 125.38s | 0.00s |
| MipNeRF360 | DINO weighted i0.50 | 9 | 28.8505 | +0.1979 | 0.8660 | +0.0109 | 0.1397 | -0.0223 | 254,736 | +81,395 | 137.60s | +12.22s |
| MipNeRF360 | DINO weighted i0.75 | 9 | 28.8396 | +0.1869 | 0.8663 | +0.0112 | 0.1395 | -0.0224 | 257,715 | +84,374 | 168.35s | +42.97s |
| MipNeRF360 | DINO weighted i0.90 | 9 | 28.8238 | +0.1711 | 0.8661 | +0.0109 | 0.1394 | -0.0226 | 253,687 | +80,347 | 195.95s | +70.57s |
| DB | baseline | 2 | 30.1179 | 0.0000 | 0.9324 | 0.0000 | 0.0658 | 0.0000 | 54,685 | 0 | 121.22s | 0.00s |
| DB | DINO weighted i0.50 | 2 | 30.3603 | +0.2423 | 0.9360 | +0.0035 | 0.0641 | -0.0017 | 62,261 | +7,576 | 141.48s | +20.26s |
| DB | DINO weighted i0.75 | 2 | 30.5446 | +0.4267 | 0.9358 | +0.0033 | 0.0633 | -0.0026 | 63,034 | +8,350 | 181.81s | +60.59s |
| DB | DINO weighted i0.90 | 2 | 30.6074 | +0.4894 | 0.9376 | +0.0051 | 0.0620 | -0.0038 | 63,006 | +8,320 | 197.47s | +76.24s |
| Tandt | baseline | 2 | 25.9551 | 0.0000 | 0.9377 | 0.0000 | 0.0541 | 0.0000 | 50,370 | 0 | 140.26s | 0.00s |
| Tandt | DINO weighted i0.50 | 2 | 25.7519 | -0.2032 | 0.9346 | -0.0031 | 0.0575 | +0.0035 | 38,394 | -11,976 | 151.05s | +10.79s |
| Tandt | DINO weighted i0.75 | 2 | 25.6201 | -0.3350 | 0.9328 | -0.0049 | 0.0585 | +0.0044 | 37,986 | -12,384 | 214.65s | +74.39s |
| Tandt | DINO weighted i0.90 | 2 | 25.5329 | -0.4222 | 0.9323 | -0.0054 | 0.0611 | +0.0070 | 37,522 | -12,848 | 212.83s | +72.57s |

逐场景 PSNR 最优分布：

| 最优方法 | 场景 | 数量 |
|---|---|---:|
| DINO weighted | DB `drjohnson/playroom`；MipNeRF360 `bicycle/bonsai/counter/garden/kitchen/room/stump` | 9 |
| cached-edge v1 | MipNeRF360 `flowers` | 1 |
| baseline | MipNeRF360 `treehill`；Tandt `train/truck` | 3 |

解读：

- 分数据集看，MipNeRF360 固定 i0.50 最均衡，i0.75/i0.90 不应替代它作为默认档；DB 固定 i0.90 是最强质量档；Tandt 三个 DINO 档位均低于 baseline，必须回退。
- 固定 i0.75/i0.90 不是跨数据集答案。多档策略的收益来自“按数据集/场景选择或回退”，不是全局提高权重。
- QCGI 仍适合作为“质量-容量综合指标”，后续可用于自适应 density/prune 强度；但文档展示指标必须按 MipNeRF360、DB、Tandt 分开报告。

## 2026-05-09 DB train-side selector 泄漏检查

目标：验证上面的多档 selector 是否能只依赖训练侧信号，而不是使用 test split 结果做事后选择。新增 `scripts/evaluate_0001_train_selector.py`，它不保存渲染图，而是在内存中对 train split 均匀抽样视角渲染并计算 PSNR/SSIM/LPIPS；随后只用 train 指标选择候选，再回到已有 test summary 查看最终测试表现。快速检查先覆盖 DB 两场景，每个 run 抽样 20 个 train 视角，输出在 `output/0001/train_selector`。

命令：

```bash
uv run --active python scripts/evaluate_0001_train_selector.py \
  --datasets db \
  --max-views 20 \
  --resume
```

train-side 选择结果落到 test split 后的表现：

| 选择器 | 场景数 | Test PSNR | Test SSIM | Test LPIPS | Test Gaussian 数 | Test 训练时间 |
|---|---:|---:|---:|---:|---:|---:|
| train best-PSNR | 2 | 30.5605 | 0.9370 | 0.0620 | 64,264 | 168.24s |
| train QCGI | 2 | 30.5605 | 0.9370 | 0.0620 | 64,264 | 168.24s |

逐场景选择：

| 场景 | train best-PSNR / QCGI 选择 | 对应 test PSNR | test 最优 PSNR 方法 | test 最优 PSNR |
|---|---|---:|---|---:|
| drjohnson | DINO weighted i0.90 | 30.5982 | DINO weighted i0.75 | 30.7254 |
| playroom | cached-edge v1 | 30.5228 | DINO weighted i0.90 | 30.6165 |

解读：

- 这个快速检查是负向结果：train 渲染指标没有复现 DB 上的 test 最优选择。它在 `drjohnson` 过度偏向 i0.90，在 `playroom` 又保守地选择 cached-edge。
- train-side 选择的 DB 平均 test PSNR 为 30.5605，略低于 DB cached-edge v1 的 30.5631，也低于 fixed DINO weighted i0.90 的 30.6074，更低于 test oracle 的 30.6710。
- 因此不能把当前 `validated_policy` 包装成无泄漏的自动选择器。下一步如果要形成严格方案，需要独立 validation split、预先固定的数据集级策略，或使用与 test 指标无关的训练过程信号；单纯 train render PSNR/QCGI 不够。

## 2026-05-09 Tandt DINO weighted 容量/时序诊断

目标：验证 Tandt 上 DINO weighted i0.50 低于 baseline 是否主要来自最终 Gaussian 数量不足。实验使用 `configs/experiments/0001_vfm_topology_dinov2_token_edge_weighted_i050_auto_prunemin.yaml`，在 `target_gaussian_count=baseline*1.42` 的同时启用 `prune_min_gaussian_target_ratio=0.7042253521126761`，因此容量下限自动等价于 baseline 最终点数。运行入口为 `scripts/run_0001_dino_weighted_eval.py`，输出在 `output/0001/dino_weighted_i050_auto_prunemin_tandt`。

| 场景 | 方法 | target | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|---:|
| train | DINO weighted i0.50 + 自动容量下限 | 83,479 | 23.5176 | 0.9091 | 0.0834 | 58,788 | 213.78s |
| truck | DINO weighted i0.50 + 自动容量下限 | 59,572 | 27.6980 | 0.9557 | 0.0386 | 41,952 | 207.22s |
| **平均** | **DINO weighted i0.50 + 自动容量下限** | - | **25.6078** | **0.9324** | **0.0610** | **50,370** | **210.50s** |

对比：

| 参照 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| 相对 baseline | -0.3473 | -0.0053 | +0.0069 | +0 | +70.24s | 容量相同但质量明显更低 |
| 相对 DINO weighted i0.50 | -0.1441 | -0.0022 | +0.0035 | +11,976 | +59.45s | 补容量反而降低质量并显著变慢 |
| 相对 cached edge v1 | +0.0279 | +0.0008 | +0.0002 | +18,808 | +65.65s | 只剩很小 PSNR/SSIM 修复，LPIPS 未改善 |

解读：

- 自动容量下限机制按预期生效：最终 `train/truck` 分别保留 58,788 和 41,952 个 Gaussians，正好回到 baseline 容量。
- 质量没有恢复，且低于原始 DINO weighted i0.50。这说明 Tandt 的 DINO weighted 负例不是单纯“最终点数太少”。
- 训练日志显示早期 staged target pruning 仍发生了大幅裁剪，例如 `train` 在 1,000 iteration 从 184,878 裁到 91,827。容量下限只约束训练期 FastGS prune 和最终一致性裁剪，不会阻止 staged target 在早期改变结构轨迹。
- 下一步不再继续“最终补容量”支线。若还要修复 Tandt，应测试推迟或取消早期 staged target、降低早期 VFM pruning/densification 介入，或直接回退 baseline；当前更有价值的主线仍是跨场景选择/回退和 QCGI 容量约束。

随后增加 `configs/experiments/0001_vfm_topology_dinov2_token_edge_weighted_i050_prunemin_only.yaml`，只启用容量下限，不启用 staged target。runner 使用 `--target-ratio-from-reference 1.0 --target-reference-method baseline`，因此 `target_gaussian_count` 与 `prune_min_gaussian_count` 都等于 baseline 最终 Gaussian 数量，输出在 `output/0001/dino_weighted_i050_prunemin_only_tandt`。

| 场景 | 方法 | target | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|---:|
| train | DINO weighted i0.50 + 容量下限，不启用 staged target | 58,788 | 23.4287 | 0.9111 | 0.0775 | 58,788 | 218.14s |
| truck | DINO weighted i0.50 + 容量下限，不启用 staged target | 41,952 | 27.8573 | 0.9572 | 0.0357 | 41,952 | 214.33s |
| **平均** | **DINO weighted i0.50 + 容量下限，不启用 staged target** | - | **25.6430** | **0.9341** | **0.0566** | **50,370** | **216.24s** |

关键对比：

| 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 相对 baseline | 结论 |
|---|---:|---:|---:|---:|---|---|
| baseline | 25.9551 | 0.9377 | 0.0541 | 50,370 | - | Tandt 当前默认回退 |
| DINO weighted i0.50 | 25.7519 | 0.9346 | 0.0575 | 38,394 | -0.2032 / -0.0031 / +0.0035 | 相对 cached-edge 修复，但仍低于 baseline |
| DINO weighted i0.50 + staged 自动容量下限 | 25.6078 | 0.9324 | 0.0610 | 50,370 | -0.3473 / -0.0053 / +0.0069 | staged target 早期裁剪后，补容量仍不能恢复 |
| DINO weighted i0.50 + 容量下限，不启用 staged target | 25.6430 | 0.9341 | 0.0566 | 50,370 | -0.3121 / -0.0036 / +0.0025 | LPIPS/SSIM 恢复一部分，但 PSNR 仍低 |

解读：

- 取消 staged target 后，平均结果相对 `staged 自动容量下限` 提升 +0.0352 PSNR、+0.0017 SSIM，LPIPS 改善 -0.0044，说明早期 staged target 对 Tandt 的感知质量确实有负面影响。
- 该结果仍低于 baseline，也低于原始 DINO weighted i0.50 的 PSNR。换言之，`prune_min` 可以防止最终过稀，取消 staged target 可以修复部分感知指标，但 DINO weighted 在 Tandt 上仍不是默认方案。

最后测试 `configs/experiments/0001_vfm_topology_dinov2_token_edge_weighted_i050_prune0.yaml`：保留 DINO weighted densification，但把 `vfm_weight=0.0`，即关闭 VFM pruning score 融合，不设置 target 或容量下限。输出在 `output/0001/dino_weighted_i050_prune0_tandt`。

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| train | DINO weighted i0.50, `vfm_weight=0.0` | 23.5517 | 0.9110 | 0.0773 | 42,287 | 215.13s |
| truck | DINO weighted i0.50, `vfm_weight=0.0` | 27.8393 | 0.9567 | 0.0370 | 34,057 | 205.44s |
| **平均** | **DINO weighted i0.50, `vfm_weight=0.0`** | **25.6955** | **0.9338** | **0.0572** | **38,172** | **210.29s** |

关键对比：

| 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 相对 baseline | 结论 |
|---|---:|---:|---:|---:|---|---|
| DINO weighted i0.50 | 25.7519 | 0.9346 | 0.0575 | 38,394 | -0.2032 / -0.0031 / +0.0035 | 当前 Tandt 最好的 DINO weighted 诊断分支 |
| DINO weighted i0.50, `vfm_weight=0.0` | 25.6955 | 0.9338 | 0.0572 | 38,172 | -0.2596 / -0.0039 / +0.0031 | LPIPS 略好，但 PSNR/SSIM 低于原始 i0.50 |

解读：

- 关闭 VFM pruning fusion 后，相对 cached-edge v1 仍保持正向，说明 DINO weighted densification 本身能修复一部分 Tandt 负例。
- 该变体没有解决容量偏低问题，平均 Gaussian 数量仍约 38k；PSNR/SSIM 也低于原始 DINO weighted i0.50。
- Tandt 支线到此收束：三类诊断都没有超过 baseline。正式策略应选择 baseline 回退；DINO weighted 只作为“相对 cached-edge 的恢复分支”和跨数据集 selector 的候选，不作为 Tandt 默认方法。

新增 `scripts/summarize_0001_tandt_diagnostics.py`，把上述 Tandt 结果汇总到 `output/0001/tandt_diagnostics`。关键产物：

| 产物 | 作用 |
|---|---|
| `summary.csv` | baseline、cached-edge、DINO weighted 三档和三条诊断的逐场景指标 |
| `averages.csv` | Tandt 两场景平均指标 |
| `comparisons.csv` | 相对 baseline、cached-edge、DINO weighted i0.50 的逐场景差值 |
| `scene_policy.csv` | 每个场景的策略选择和回退原因 |
| `policy.json` | 数据集级策略结论 |

`policy.json` 当前结论为 `dataset_policy_pick=baseline`，原因是所有 VFM 候选/诊断的平均指标都没有三项同时超过 baseline。`scene_policy.csv` 中 `train` 和 `truck` 也都回退 baseline。

## 2026-05-09 数据集级预设策略汇总

目标：把目前已经收束的规则整理成非 oracle 展示线，避免把 test split 的逐场景最优选择误解为真实自动 selector。新增 `scripts/summarize_0001_dataset_policies.py`，只读取已有 summary 与 recommendation 产物，输出到 `output/0001/dataset_policies`。

策略定义：

| 策略 | MipNeRF360 | DB | Tandt |
|---|---|---|---|
| `dataset_fixed_policy` | 固定 `weighted_i050` | 固定 `dino_weighted_i090` | baseline 回退 |
| `dataset_quality_policy` | weighted QCGI 场景选择 | 固定 `dino_weighted_i090` | baseline 回退 |

按公开数据集分别求平均：

| 数据集 | 策略 | 场景数 | PSNR | ΔPSNR vs baseline | SSIM | ΔSSIM | LPIPS | ΔLPIPS | Gaussian 数量 | ΔGaussian | 训练时间 | Δ训练时间 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MipNeRF360 | baseline | 9 | 28.6527 | 0.0000 | 0.8551 | 0.0000 | 0.1620 | 0.0000 | 173,341 | 0 | 125.38s | 0.00s |
| MipNeRF360 | `dataset_fixed_policy` | 9 | 28.8505 | +0.1979 | 0.8660 | +0.0109 | 0.1397 | -0.0223 | 254,736 | +81,395 | 137.60s | +12.22s |
| MipNeRF360 | `dataset_quality_policy` | 9 | 28.8641 | +0.2114 | 0.8667 | +0.0116 | 0.1388 | -0.0231 | 255,822 | +82,481 | 158.78s | +33.40s |
| DB | baseline | 2 | 30.1179 | 0.0000 | 0.9324 | 0.0000 | 0.0658 | 0.0000 | 54,685 | 0 | 121.22s | 0.00s |
| DB | `dataset_fixed_policy` | 2 | 30.6074 | +0.4894 | 0.9376 | +0.0051 | 0.0620 | -0.0038 | 63,006 | +8,320 | 197.47s | +76.24s |
| DB | `dataset_quality_policy` | 2 | 30.6074 | +0.4894 | 0.9376 | +0.0051 | 0.0620 | -0.0038 | 63,006 | +8,320 | 197.47s | +76.24s |
| Tandt | baseline | 2 | 25.9551 | 0.0000 | 0.9377 | 0.0000 | 0.0541 | 0.0000 | 50,370 | 0 | 140.26s | 0.00s |
| Tandt | `dataset_fixed_policy` | 2 | 25.9551 | 0.0000 | 0.9377 | 0.0000 | 0.0541 | 0.0000 | 50,370 | 0 | 140.26s | 0.00s |
| Tandt | `dataset_quality_policy` | 2 | 25.9551 | 0.0000 | 0.9377 | 0.0000 | 0.0541 | 0.0000 | 50,370 | 0 | 140.26s | 0.00s |

解读：

- `dataset_fixed_policy` 和 `dataset_quality_policy` 都应按三个公开数据集分别解读：MipNeRF360 正向、DB 正向、Tandt 回退后与 baseline 持平。
- `dataset_fixed_policy` 更适合作为当前第一版的保守展示线：MipNeRF360 与 DB 三项指标均正向，Tandt 不启用负向候选。
- `dataset_quality_policy` 通过 MipNeRF360 weighted QCGI 场景选择进一步提高质量，但训练时间增加更明显。它适合作为质量优先线，而不是默认效率线。

## 2026-04-28 Mock v1 快速验证

数据集：`datasets/mipnerf360/bicycle`，test split，`-r 8`，220 iterations，`densify_from_iter=50`，`densification_interval=50`。

| 产物 | 变体 | 打分器 | 后端 | PSNR | SSIM | LPIPS | 训练时间 | Gaussian 数量 | 备注 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `output/0001/baseline_bicycle_smoke` | `fastgs_baseline` | `fastgs_photometric` | n/a | 20.3464 | 0.4294 | 0.6021 | 0.96s | 78,633 | 控制组 |
| `output/0001/vfm_mock_bicycle_smoke` | `0001_vfm_topology_scorer` | `vfm_topology_scorer` | `mock_l1` | 20.3459 | 0.4294 | 0.6010 | 1.05s | 78,375 | SH0 mock VFM 分支已验证 |
| `output/0001/vfm_cached_edge_bicycle_smoke` | `0001_vfm_topology_cached_edge` | `vfm_topology_scorer` | `cached_edge_l1` | 20.3265 | 0.4291 | 0.6005 | 1.17s | 78,605 | 离线 cache 读取链路已验证 |
| `output/0001/vfm_cached_edge_compact_bicycle_smoke` | `0001_vfm_topology_cached_edge_compact` | `vfm_topology_scorer` | `cached_edge_l1` / `npz_uint8` | 20.1588 | 0.4275 | 0.5993 | 1.16s | 78,682 | compact cache 与校验链路已验证 |
| `output/0001/vfm_dinov2_token_edge_bicycle_smoke` | `0001_vfm_topology_dinov2_token_edge` | `vfm_topology_scorer` | `dinov2_token_edge_l1` | 20.2913 | 0.4272 | 0.6006 | 1.72s | 77,761 | 训练已消费 DINOv2 token-edge cache |

## 解读

- mock VFM scorer 已完成 train、render 和 metric evaluation，没有 shape、device 或 optimizer-state 失败。
- 这个短跑快速验证中质量指标基本打平。符合预期，因为 `mock_l1` 只是链路占位后端，不是真实 VFM 信号。
- mock 分支在这次极短 run 中训练时间约增加 9%，但该数字受短跑固定开销主导，应在更长 schedule 上重新测。
- Gaussian count 接近 baseline，说明保守的 `max(rgb_importance, vfm_importance)` 和 weighted pruning fusion 没有让 densification 失稳。
- cached edge proxy 在快速验证中带来小幅 PSNR 下降和小幅 LPIPS 改善。这不是质量结论，只确认缓存的 GT 特征可以读取、resize、与 SH0 渲染特征对比并融合，且不会破坏训练。
- Cache 产物：`output/0001/vfm_cache/bicycle_edge`，194 个 entries，从 `images_8` 以 `--max_width 640` 构建时约 189MB。
- Compact cache 产物：`output/0001/vfm_cache/bicycle_edge_u8`，194 个 entries，`--storage npz_uint8` 下约 35MB；`vfm_gs.cli.validate_vfm_cache` 通过 checksum 和 source-image 检查。
- compact cache run 保持稳定，但 PSNR 相比 float32 cache 下降更明显。这说明 quantization 可能改变 thresholded edge masks，进而影响早期 densification；长跑应把 cache precision 作为 ablation，而不是默认 compact storage metric-neutral。
- 首个消费 DINOv2 的 scorer 保持稳定，短跑 PSNR/SSIM 略低于 baseline，LPIPS 接近之前 VFM 变体。它应被解读为真实 cache 训练集成成功，而不是 token-edge projection 已经是正确质量信号。
- 同一 220-iteration schedule 下，DINO token-edge 训练时间为 1.72s，cached edge 为 1.17s，因此派生的 DINO projection 在这个小规模上带来可见但仍温和的 scorer 开销。

## 2026-04-28 30k 匹配消融

数据集：`datasets/mipnerf360/bicycle`，test split，`-r 8`，30,000 iterations。这组使用正常 FastGS densification schedule，比 220-iteration 快速验证更能指导质量判断。

| 产物 | 变体 | 打分器 | 后端 | PSNR | SSIM | LPIPS | 训练时间 | 渲染 FPS | Gaussian 数量 | 输出大小 | 备注 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `output/0001/baseline_bicycle_30k_r8` | `fastgs_baseline` | `fastgs_photometric` | n/a | 26.7032 | 0.8067 | 0.2278 | 116.92s | 334.36 | 240,394 | 82M | 控制组 |
| `output/0001/vfm_cached_edge_compact_bicycle_30k_r8` | `0001_vfm_topology_cached_edge_compact` | `vfm_topology_scorer` | `cached_edge_l1` / `npz_uint8` | 26.8864 | 0.8229 | 0.1972 | 159.77s | 196.43 | 408,925 | 122M | Edge cache 改善指标，但点数增长 |
| `output/0001/vfm_dinov2_token_edge_bicycle_30k_r8` | `0001_vfm_topology_dinov2_token_edge` | `vfm_topology_scorer` | `dinov2_token_edge_l1` | 27.0577 | 0.8345 | 0.1767 | 166.11s | 193.46 | 490,832 | 142M | 指标最佳，重建最密 |

使用的 cache 产物：

| Cache | 后端 | 条目数 | 存储 | 大小 | 校验 |
|---|---|---:|---|---:|---|
| `output/0001/vfm_cache/bicycle_edge_u8` | `cached_edge_l1` | 194 | `npz_uint8` | 35M | 通过 |
| `output/0001/vfm_cache/bicycle_dinov2_vits14` | `dinov2_vits14` | 194 | `npy_float16` | 24M | 通过 |

解读：

- 220-iteration 快速验证指标太弱，不能指导质量决策。它仍适合判断集成健康，但 30k runs 明显改变了信号含义。
- `cached_edge_l1` 相比 baseline 提升 +0.1832 PSNR、+0.0163 SSIM、-0.0306 LPIPS，但 Gaussian count 增加约 70%，test render FPS 降低约 41%。
- `dinov2_token_edge_l1` 相比 baseline 提升 +0.3544 PSNR、+0.0278 SSIM、-0.0511 LPIPS，但 Gaussian count 增加约 104%，test render FPS 降低约 42%。
- DINO token-edge scorer 在完整低分辨率训练上是有希望的方向，但收益与更大的 Gaussian 预算纠缠。下次对比需要预算受控的 run 或更强 final-prune 设置，之后才能称为纯质量改进。
- `cached_edge_l1` 仍是强确定性 proxy baseline：语义性弱于 DINO，但 cache 更小、训练开销更低，并给出清晰指标增益。

## 2026-04-28 预算控制探测

数据集和 schedule 与上面的 30k ablation 一致。这组只覆盖 `vfm_loss_thresh=0.75` 和 `vfm_weight=0.10`，用于测试现有 knobs 能否让 VFM runs 接近 baseline Gaussian count。

| 产物 | 后端 | PSNR | SSIM | LPIPS | 训练时间 | 渲染 FPS | Gaussian 数量 | 输出大小 | 相对 baseline 点数差值 | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `output/0001/vfm_cached_edge_t075_w010_bicycle_30k_r8` | `cached_edge_l1` / `npz_uint8` | 26.9372 | 0.8238 | 0.1979 | 166.81s | 337.93 | 409,028 | 122M | +70.2% | 现有 knobs 未降低 edge 点数 |
| `output/0001/vfm_dinov2_token_edge_t075_w010_bicycle_30k_r8` | `dinov2_token_edge_l1` | 26.9586 | 0.8258 | 0.1935 | 166.60s | 377.72 | 422,506 | 125M | +75.8% | 低于默认 DINO 点数，但仍远高于 baseline |

解读：

- 提高 `vfm_loss_thresh` 并降低 `vfm_weight` 仍不足以控制预算。`vfm_weight` 当前影响 pruning-score fusion，而 densification 仍使用 `max(rgb_importance, vfm_importance)`。
- DINO budget probe 相比默认 DINO 30k run 将 Gaussian count 降低约 14%，但也交回了部分质量收益。
- Edge proxy 点数在这些 knobs 下基本不变。下一步实现应拆分 VFM densification strength 和 VFM pruning strength，例如增加独立的 `vfm_importance_weight` 或 `vfm_importance_mode`。
- 渲染 FPS 波动较大，应与 Gaussian count 一起解读；预算对比接近时需要重复实验。

## 2026-04-28 显式 Importance Weight 探测

代码变更：增加 `vfm_importance_weight`，默认 `1.0` 以保持向后兼容。它会在 `max(rgb_importance, vfm_importance)` 前缩放 VFM densification counts，并独立于继续控制 pruning-score fusion 的 `vfm_weight`。

数据集和 schedule 与上面的 30k ablation 一致。这组使用 `vfm_importance_weight=0.25`，其他 `vfm_loss_thresh` 和 `vfm_weight` 使用各变体默认值。

| 产物 | 后端 | PSNR | SSIM | LPIPS | 训练时间 | 渲染 FPS | Gaussian 数量 | 输出大小 | 相对 baseline 点数差值 | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `output/0001/vfm_cached_edge_i025_bicycle_30k_r8` | `cached_edge_l1` / `npz_uint8` | 26.9439 | 0.8244 | 0.1958 | 166.85s | 168.15 | 413,301 | 123M | +71.9% | 显式 importance scaling 未降低 edge 点数 |
| `output/0001/vfm_dinov2_token_edge_i025_bicycle_30k_r8` | `dinov2_token_edge_l1` | 26.9261 | 0.8259 | 0.1928 | 159.84s | 279.87 | 418,073 | 124M | +73.9% | 默认 DINO 点数降低 14.8%，但仍不足以 budget matching |

解读：

- `vfm_importance_weight=0.25` 有用，但单独作为预算杠杆仍不足。
- DINO i0.25 的 Gaussian budget 优于默认 DINO，但也交回了默认 DINO 的大部分指标优势。
- Edge 对这个控制不敏感，说明它的额外点数不能被简单 post-count scaling 轻易压下。
- 下一步应增加更硬的模式，例如 `vfm_importance_mode=rgb_only|max|weighted`，其中 `rgb_only` 允许 VFM 影响 pruning 但不影响 densification。

## 2026-04-28 Importance Mode 探测

代码变更：增加 `vfm_importance_mode`，默认 `max` 以保持向后兼容。可选模式为 `max`、`weighted` 和 `rgb_only`。在 `rgb_only` 中，VFM 仍通过 `vfm_weight` 参与 pruning-score fusion，但 densification importance 直接使用 RGB/FastGS importance counts。

数据集和 schedule 与上面的 30k ablation 一致。这组使用 `vfm_importance_mode=rgb_only`，其他 `vfm_loss_thresh`、`vfm_weight` 和 `vfm_importance_weight` 使用各变体默认值。

| 产物 | 后端 | PSNR | SSIM | LPIPS | 训练时间 | 渲染 FPS | Gaussian 数量 | 输出大小 | 相对 baseline 点数差值 | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `output/0001/vfm_cached_edge_rgb_only_bicycle_30k_r8` | `cached_edge_l1` / `npz_uint8` | 26.9574 | 0.8243 | 0.1961 | 158.76s | 282.64 | 413,914 | 123M | +72.2% | 禁用直接 VFM densification 未降低 edge 点数 |
| `output/0001/vfm_dinov2_token_edge_rgb_only_bicycle_30k_r8` | `dinov2_token_edge_l1` | 26.9310 | 0.8237 | 0.1962 | 159.52s | 185.85 | 413,223 | 123M | +71.9% | 低于默认 DINO 点数，但仍远高于 baseline |

解读：

- `rgb_only` 证明短 220-iteration 指标太弱，不能选择 scorer 行为。完整 30k runs 会暴露短跑看不到的 model-budget shift。
- 直接 VFM densification 不是额外 Gaussians 的唯一来源。单独的 VFM pruning-score fusion 也能保留或重塑足够多点，使最终点数保持在 baseline 的约 1.72x。
- DINO `rgb_only` 将默认 DINO 点数降低约 15.8%，但交回了大部分默认 DINO 指标优势。
- Edge `rgb_only` 与之前 edge probes 的预算几乎相同，说明简单 importance-mode 控制不足以 budget matching。
- 下一版需要显式 Gaussian budget 机制，例如 final prune/target-count pass，或 densify/prune loop 中的 hard cap。纯 signal-quality 结论应等到 VFM 和 baseline 在匹配点预算下比较后再下。

## 2026-04-28 目标预算负例探测

代码变更：增加 `target_gaussian_count` 作为 final-prune 预算控制。首版实现会把最高 pruning-score Gaussians 裁到目标点数。它精确匹配预算但破坏质量，因此记录为负例，并已将实现改为优先裁剪最低分 Gaussians。

数据集和 schedule 与上面的 30k ablation 一致。目标点数是 baseline 30k 点数：240,394 个 Gaussians。

| 产物 | 后端 | 目标策略 | PSNR | SSIM | LPIPS | 训练时间 | 渲染 FPS | Gaussian 数量 | 输出大小 | 备注 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `output/0001/vfm_cached_edge_budget240394_bicycle_30k_r8` | `cached_edge_l1` / `npz_uint8` | 裁剪最高分 | 11.1494 | 0.2037 | 0.6308 | 155.51s | 143.74 | 240,394 | 79M | 预算匹配，质量崩溃 |
| `output/0001/vfm_dinov2_token_edge_budget240394_bicycle_30k_r8` | `dinov2_token_edge_l1` | 裁剪最高分 | 10.2215 | 0.1360 | 0.6769 | 162.33s | 406.19 | 240,394 | 76M | 预算匹配，质量崩溃 |

解读：

- 只精确匹配点数不够。prune ordering 比 count target 更重要。
- bulk-pruning highest-score Gaussians 会删除重建仍需要的困难结构或高误差结构。
- `target_gaussian_count` 应优先裁剪最低 support/score Gaussians 来做预算匹配。high-score pruning 只能作为小规模异常点移除规则，而不能作为主预算裁剪策略。

## 2026-04-28 Low-Score 目标预算探测

代码变更：`target_gaussian_count` 现在会优先裁剪最低 pruning/support scores，用于 bulk budget control。这修正了上面负例中的 score 方向，但仍只在训练结束后执行一次 count correction。

数据集和 schedule 与上面的 30k ablation 一致。目标点数是 baseline 30k 点数：240,394 个 Gaussians。

| 产物 | 后端 | 目标策略 | PSNR | SSIM | LPIPS | 训练时间 | 渲染 FPS | Gaussian 数量 | 输出大小 | 备注 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `output/0001/vfm_cached_edge_budget240394_lowscore_bicycle_30k_r8` | `cached_edge_l1` / `npz_uint8` | 裁剪最低分 | 23.7729 | 0.7307 | 0.2685 | 153.06s | 283.32 | 240,394 | 81M | 精确预算，但质量低于 baseline |
| `output/0001/vfm_dinov2_token_edge_budget240394_lowscore_bicycle_30k_r8` | `dinov2_token_edge_l1` | 裁剪最低分 | 23.5571 | 0.7087 | 0.2797 | 157.34s | 357.34 | 240,394 | 81M | 精确预算，final prune 下弱于 edge |

解读：

- Low-score pruning 避免了 high-score pruning 的灾难性失败，但 baseline-sized target 下单次 final pruning 仍破坏性太强。
- 相比 baseline 30k run，edge low-score target pruning 为 -2.9303 PSNR、-0.0760 SSIM、+0.0407 LPIPS；DINO 为 -3.1461 PSNR、-0.0980 SSIM、+0.0519 LPIPS。
- Final-pruned VFM runs 点数匹配且输出产物更小，但不再保留 unpruned VFM 的质量收益。这说明额外 VFM-driven points 不是收敛后可简单删除的重复点。
- 下一种预算方法应在训练期 staged 执行，而不是最后单刀裁剪：在 densification/pruning events 后施加 soft cap，或在 target pruning 后跑 post-prune fine-tune schedule。

## 2026-04-29 Staged 目标预算探测

代码变更：增加 `target_gaussian_staged`、`target_gaussian_stage_margin`、`target_gaussian_stage_start` 和 `target_gaussian_stage_interval`。这些控制项会在训练期周期性地向 `target_gaussian_count * margin` 裁剪 lowest-score Gaussians，最终仍写出 exact-budget final PLY。

数据集和 schedule 与上面的 30k ablation 一致。这组使用 `target_gaussian_count=240394`、`target_gaussian_staged=true`、`target_gaussian_stage_margin=1.2` 和 `target_gaussian_stage_interval=500`。staged cap 为 288,473 个 Gaussians。

| 产物 | 后端 | PSNR | SSIM | LPIPS | 训练时间 | 渲染 FPS | Gaussian 数量 | 输出大小 | Staged 裁剪 | 最终裁剪 | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `output/0001/vfm_cached_edge_budget240394_staged120_bicycle_30k_r8` | `cached_edge_l1` / `npz_uint8` | 25.7979 | 0.7747 | 0.2537 | 171.49s | 514.79 | 240,394 | 82M | 24 | 276,611 -> 240,394 | 优于 final-only，但仍低于 baseline |
| `output/0001/vfm_dinov2_token_edge_budget240394_staged120_bicycle_30k_r8` | `dinov2_token_edge_l1` | 25.3529 | 0.7727 | 0.2493 | 172.39s | 524.63 | 240,394 | 82M | 25 | 288,127 -> 240,394 | 严格预算 VFM 中 LPIPS 最好，但仍低于 baseline |

解读：

- Staged pruning 相比单次 final pruning 明显更好，在同样精确 240,394 Gaussian budget 下，edge 提升 +2.0251 PSNR，DINO 提升 +1.7958 PSNR。
- strict 240k budget 仍低于 baseline：edge 为 -0.9053 PSNR、-0.0320 SSIM、+0.0259 LPIPS；DINO 为 -1.3503 PSNR、-0.0340 SSIM、+0.0215 LPIPS。
- unpruned VFM 质量收益在这个 scene 上似乎需要超过 baseline point count，或者需要在激进 budget pruning 后做 recovery training。
- 下一次对比应先使用更宽松的 staged budget，例如 300k Gaussians，然后再投入更复杂 scorer。

## 2026-04-29 300k Staged 目标预算探测

数据集和 schedule 与上面的 30k ablation 一致。这组使用 `target_gaussian_count=300000`、`target_gaussian_staged=true`、`target_gaussian_stage_margin=1.15` 和 `target_gaussian_stage_interval=500`。staged cap 为 345,000 个 Gaussians。

| 产物 | 后端 | PSNR | SSIM | LPIPS | 训练时间 | 渲染 FPS | Gaussian 数量 | 输出大小 | Staged 裁剪 | 最终裁剪 | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `output/0001/vfm_cached_edge_budget300000_staged115_bicycle_30k_r8` | `cached_edge_l1` / `npz_uint8` | 26.2327 | 0.7866 | 0.2412 | 165.98s | 445.07 | 300,000 | 96M | 23 | 314,704 -> 300,000 | 更接近 baseline，但所有指标仍低 |
| `output/0001/vfm_dinov2_token_edge_budget300000_staged115_bicycle_30k_r8` | `dinov2_token_edge_l1` | 25.9089 | 0.7925 | 0.2291 | 154.13s | 474.78 | 300,000 | 96M | 24 | 333,724 -> 300,000 | LPIPS 几乎匹配 baseline，PSNR/SSIM 仍低 |

解读：

- 将 staged target 从 240k 放宽到 300k 后，edge 进一步恢复 +0.4348 PSNR，DINO 进一步恢复 +0.5561 PSNR。
- Edge 300k 仍低于 baseline：-0.4705 PSNR、-0.0201 SSIM、+0.0134 LPIPS。
- DINO 300k 的 PSNR 比 baseline 低 -0.7943，SSIM 低 -0.0142，但 LPIPS 只差 +0.0013。受限预算下，DINO 的 perceptual signal 仍最有希望。
- budget-quality curve 还没超过 baseline。加更多训练机制前应先测 350k staged target；如果仍不达标，下一条代码路径应是 post-prune fine-tuning 或不那么 edge-like 的 DINO descriptor scorer。

## 2026-04-29 350k Staged 目标预算探测

数据集和 schedule 与上面的 30k ablation 一致。这组使用 `target_gaussian_count=350000`、`target_gaussian_staged=true`、`target_gaussian_stage_margin=1.10` 和 `target_gaussian_stage_interval=500`。staged cap 为 385,000 个 Gaussians。

| 产物 | 后端 | PSNR | SSIM | LPIPS | 训练时间 | 渲染 FPS | Gaussian 数量 | 输出大小 | Staged 裁剪 | 最终裁剪 | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `output/0001/vfm_cached_edge_budget350000_staged110_bicycle_30k_r8` | `cached_edge_l1` / `npz_uint8` | 26.7788 | 0.8089 | 0.2206 | 161.89s | 529.06 | 340,283 | 106M | 21 | 跳过，340,283 <= 350,000 | 首个所有指标超过 baseline 的预算受控 VFM run |
| `output/0001/vfm_dinov2_token_edge_budget350000_staged110_bicycle_30k_r8` | `dinov2_token_edge_l1` | 26.3634 | 0.8033 | 0.2188 | 166.69s | 517.33 | 350,000 | 108M | 23 | 360,043 -> 350,000 | 受限预算下 LPIPS 最好，但 PSNR/SSIM 低于 baseline |

解读：

- 350k staged edge 是首个预算受控正向结果：相比 baseline，在约 1.42x baseline Gaussian count 下达到 +0.0756 PSNR、+0.0022 SSIM、-0.0072 LPIPS。
- 相比默认 unpruned edge，staged 350k 减少约 16.8% Gaussians，并保持质量高于 baseline，但交回了一部分 unpruned edge 收益。
- DINO 350k 的 PSNR 仍比 baseline 低 -0.3398，SSIM 低 -0.0034，但 LPIPS 改善 -0.0090。当前 token-edge projection 在 budget control 下更偏 perceptual，而非 photometric。
- v1 正向结果应表述为 staged-budget `cached_edge_l1`，不是 DINO token-edge。DINO 需要更好的 descriptor scorer 或 recovery/fine-tune 路径，之后才能称为预算高效改进。

## 2026-04-29 Garden Edge 复验

复现场景：`datasets/mipnerf360/garden`，`-r 8`，30,000 iterations。edge run 使用新构建的 compact cache：`output/0001/vfm_cache/garden_edge_u8`，185 个 entries，大小 36M。budget target 设为 278,606，对齐 bicycle 正向结果约 `1.42x` baseline-count ratio。

| 产物 | 变体 | PSNR | SSIM | LPIPS | 训练时间 | 渲染 FPS | Gaussian 数量 | 输出大小 | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `output/0001/baseline_garden_30k_r8` | baseline | 28.7051 | 0.8889 | 0.1134 | 135.77s | 475.27 | 196,201 | 76M | Garden baseline |
| `output/0001/vfm_cached_edge_garden_budget278606_staged110_30k_r8` | `cached_edge_l1`, staged target | 28.9411 | 0.8964 | 0.1007 | 146.57s | 533.30 | 248,471 | 88M | natural final count 低于 278,606，因此跳过 target |

解读：

- staged edge 正向结果已在第二个 scene 复验。Garden edge 相比自身 baseline 提升 +0.2360 PSNR、+0.0074 SSIM、-0.0126 LPIPS。
- Garden 在所选 target 下不需要 staged 或 final target prune；FastGS 加 edge scorer 自然结束在 248,471 个 Gaussians，约为 garden baseline count 的 1.27x。
- 这降低了 bicycle 350k edge 结果只是单场景偶然的概率。当前 v1 正向结论仍应限定为 staged/ratio-aware budget control 下的 `cached_edge_l1`。

## 2026-05-06 Counter Edge 复验

复现场景：`datasets/mipnerf360/counter`，`-r 8`，30,000 iterations。edge run 使用新构建的 compact cache：`output/0001/vfm_cache/counter_edge_u8`，240 个 entries，大小 17M，并通过 `validate_vfm_cache`。budget target 设为 160,699，对齐 bicycle 正向结果约 `1.42x` baseline-count ratio；stage margin 为 1.10，cap 为 176,769。

| 产物 | 变体 | PSNR | SSIM | LPIPS | 训练时间 | Gaussian 数量 | 输出大小 | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `output/0001/baseline_counter_30k_r8` | baseline | 29.5346 | 0.9304 | 0.0815 | 122.75s | 113,168 | 41M | Counter baseline |
| `output/0001/vfm_cached_edge_counter_budget160699_staged110_30k_r8` | `cached_edge_l1`, staged target | 29.6316 | 0.9319 | 0.0791 | 132.67s | 111,116 | 40M | natural final count 低于 160,699，因此跳过 target |

解读：

- Counter edge 相比自身 baseline 提升 +0.0970 PSNR、+0.0015 SSIM、-0.0024 LPIPS。
- 这组 edge run 的最终 Gaussian count 比 baseline 少 2,052 个，约少 1.8%。因此这个正向结果不是由更多点数换来的。
- Counter 是第三个场景，且与 bicycle/garden 的室外场景不同。结合 bicycle 和 garden，`cached_edge_l1` 可以作为 v1 正向控制组固化：它证明 scorer/cache/pruning 路径能稳定产生正向信号，但它仍是边缘代理，不是最终 VFM 语义打分器。
- Garden 与 Counter 在所选 target 下都自然低于目标点数，没有真正触发最终预算裁剪；严格预算机制的正向证据主要来自 bicycle 350k staged run。

## 2026-05-06 No-Effect 与 Densification Cadence 控制

数据集和 schedule 与 bicycle 30k ablation 一致，`-r 8`，30,000 iterations。本组拆分两类影响：

- `fastgs_densify100`：使用 `fastgs_photometric`，不启用 VFM scorer/cache，只把 `densification_interval` 从 baseline 的 500 覆盖为 100。
- VFM zero-weight controls：启用 `vfm_topology_scorer`、cache preflight 和对应 backend，但设置 `vfm_weight=0.0`、`vfm_importance_mode=rgb_only`，用于检查 VFM scorer/cache 管线在不直接改变 importance/pruning 时的影响。

| 产物 | Scorer / Backend | densification interval | VFM 设置 | PSNR | SSIM | LPIPS | 训练时间 | 渲染 FPS | Gaussian 数量 | 输出大小 | 备注 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| `output/0001/fastgs_densify100_bicycle_30k_r8` | `fastgs_photometric` | 100 | off | 26.9287 | 0.8241 | 0.1964 | 165.89s | 398.04 | 412,078 | 123M | 严格 cadence 控制组 |
| `output/0001/vfm_cached_edge_noeffect_bicycle_30k_r8` | `vfm_topology_scorer` / `cached_edge_l1` | 100 | `vfm_weight=0`, `rgb_only` | 26.9312 | 0.8242 | 0.1968 | 142.62s | 477.44 | 410,330 | 122M | zero-weight VFM 管线控制 |
| `output/0001/vfm_dinov2_token_edge_noeffect_bicycle_30k_r8` | `vfm_topology_scorer` / `dinov2_token_edge_l1` | 100 | `vfm_weight=0`, `rgb_only` | 26.9536 | 0.8243 | 0.1968 | 162.60s | 507.36 | 412,037 | 123M | zero-weight DINO 管线控制 |

解读：

- 410k 级别的 Gaussian count 主要来自 `densification_interval=100` 的更高 densification cadence，而不是 VFM signal 本身。`fastgs_densify100` 已达到 412,078 个 Gaussians，与两个 zero-weight VFM run 基本一致。
- zero-weight cached edge 与 fastgs densify100 的差异很小：+0.0025 PSNR、+0.0002 SSIM、+0.0004 LPIPS，Gaussian count 反而少 1,748。
- zero-weight DINO 与 fastgs densify100 也接近：+0.0249 PSNR、+0.0002 SSIM、+0.0004 LPIPS，Gaussian count 少 41。
- 因此，之前 `rgb_only`/zero-weight run 不能用来说明 VFM 本身导致点数膨胀；需要把 cadence control 作为后续预算实验的固定对照。
- 当前 `vfm_enable` 是 config 中的单向布尔开关；如果加载同一个 VFM experiment yaml，CLI 不能直接把它显式关闭。严格 no-effect 应从 `fastgs_baseline` 出发，手动覆盖 `--densification_interval 100`，而不是加载 VFM config 后尝试关闭 VFM。

## 2026-05-06 最终裁剪后恢复训练探测

代码变更：增加 `post_prune_finetune_iterations`。当 `target_gaussian_count` 的最终裁剪实际删除 Gaussians 后，训练会清空残留梯度，继续执行指定步数的光度恢复训练，并保存到 `iterations + post_prune_finetune_iterations`。该参数默认 `0`，不改变既有 run。

快速验证：`output/0001/post_prune_finetune_smoke_bicycle_260_r8` 在 260-iteration 短跑中从 78,838 裁到 65,000，并继续 20 步保存到 `ours_280`；render 和 metrics 均能读取最新迭代。

2026-05-07 代码变更：为 post-prune recovery 增加独立的 `post_prune_finetune_step_interval`、`post_prune_finetune_sh_step_interval`、`post_prune_finetune_lr_mode`、`post_prune_finetune_lr_scale` 和 `post_prune_finetune_trigger`。默认值保持旧行为；设置 step interval 后，恢复阶段可以脱离 30k 后每 64 步才更新一次的主训练 cadence。`post_prune_finetune_trigger=staged_prune|any_prune|always` 还允许 staged pruning 后触发恢复训练。

Dense recovery 快速验证：`output/0001/post_prune_dense_finetune_smoke_bicycle_260_r8` 使用 `post_prune_finetune_step_interval=1`、`post_prune_finetune_sh_step_interval=1`、`post_prune_finetune_lr_mode=local` 和 `post_prune_finetune_lr_scale=0.25`。训练从 88,194 裁到 65,000，并继续 20 步保存到 `ours_280`；render 和 metrics 均读取最新迭代，达到 PSNR 20.4099、SSIM 0.4310、LPIPS 0.5961。该短跑只验证 dense recovery 调度、保存和评估链路，不作为质量结论。

完整探测使用 bicycle 30k `-r 8`，`cached_edge_l1`，目标点数为 baseline 30k 的 240,394；不启用 staged pruning，只在最终 low-score target prune 后恢复 4,096 步。

| 产物 | 后端 | 目标策略 | 恢复步数 | PSNR | SSIM | LPIPS | 训练时间 | Gaussian 数量 | 输出大小 | 备注 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `output/0001/post_prune_dense_finetune_smoke_bicycle_260_r8` | `fastgs_photometric` | 最终裁剪最低分后 dense recovery | 20 | 20.4099 | 0.4310 | 0.5961 | 1.60s | 65,000 | 51M | 快速验证，88,194 -> 65,000，保存 `ours_280` |
| `output/0001/vfm_cached_edge_budget240394_lowscore_finetune4096_bicycle_30k_r8` | `cached_edge_l1` | 最终裁剪最低分后恢复 | 4,096 | 26.0163 | 0.7760 | 0.2580 | 156.29s | 240,394 | 138M | 411,345 -> 240,394，保存 `ours_34096` |
| `output/0001/vfm_cached_edge_budget240394_lowscore_denseft4096_s1_lr025_bicycle_30k_r8` | `cached_edge_l1` | 最终裁剪最低分后 dense recovery | 4,096 | 26.2470 | 0.7813 | 0.2526 | 156.92s | 240,394 | 138M | 411,319 -> 240,394，保存 `ours_34096`；step interval 1，SH interval 16，局部 xyz LR x0.25 |

对比：

- dense recovery 快速验证证明新参数不会破坏裁剪、恢复、保存和 `--iteration -1` 评估路径。由于它是 260-step 短跑，不能和 30k 质量结果比较。
- 相比 final-only low-score target prune，恢复训练提升 +2.2434 PSNR、+0.0453 SSIM、-0.0105 LPIPS，说明最终裁剪后的结构损伤可以被一部分恢复。
- 相比 240k staged edge，post-prune fine-tune 的 PSNR/SSIM 更高：+0.2184 PSNR、+0.0013 SSIM；但 LPIPS 更差 +0.0043。
- 相比 baseline 30k，仍低 -0.6869 PSNR、-0.0307 SSIM、+0.0302 LPIPS。严格 240k 预算下，单次最终裁剪加 4,096 步恢复还不能构成正向结果。
- dense recovery 完整版验证了上面的猜测：将恢复阶段主 optimizer 改为每步更新、SH optimizer 每 16 步更新，并用局部 xyz LR 后，相比默认 cadence 的 4,096 步恢复提升 +0.2307 PSNR、+0.0053 SSIM、-0.0054 LPIPS。
- 相比 final-only low-score target prune，dense recovery 完整版提升 +2.4741 PSNR、+0.0506 SSIM、-0.0159 LPIPS；相比 240k staged edge，提升 +0.4491 PSNR、+0.0066 SSIM、-0.0011 LPIPS。
- 但严格 240k 下 dense recovery 仍低于 baseline 30k：-0.4562 PSNR、-0.0254 SSIM、+0.0248 LPIPS；也低于 350k staged edge 正向控制组：-0.5318 PSNR、-0.0276 SSIM、+0.0320 LPIPS。
- 因此，单次最终裁剪再恢复不是足够的预算高效方案。下一步应把 dense recovery 放到 staged pruning 后，优先验证 descriptor/top-k staged + `post_prune_finetune_trigger=any_prune`，因为这类 run 的结构损伤发生在训练中期，更可能被后续恢复吸收。
- dense recovery 完整版输出包含 `iteration_30000` 和 `iteration_34096` 两份 PLY，因此目录大小为 138M；render/metrics 使用 `ours_34096`。

## 2026-04-28 Cache 预检

- `vfm_gs.cli.validate_vfm_cache` 在 `output/0001/vfm_cache/bicycle_edge_u8` 上通过，包含 194 个 `cached_edge_l1` entries。
- `vfm_topology_scorer.preflight` 在 Scene 构建前对同一个 compact cache 通过。
- 使用 `--vfm_cache_dir output/0001/vfm_cache/does_not_exist` 的负例 train-entry 检查在 camera loading 前失败，并给出结构化的 `VFM cache preflight failed` 错误。

## 2026-04-28 后端可行性

- 已新增并运行 `vfm_gs.cli.vfm_backend_probe`，记录当前环境。
- 当前 runtime：Python 3.10.20，PyTorch 1.12.1+cu116，CUDA 11.6，RTX 4090 D 23.52GB。
- 可选 VFM packages 未安装：`transformers`、`timm`、`xformers`、`opencv-python`。
- DINOv2 ViT-S/14 和 ViT-B/14 是 `max_width` 518-640 下离线 cache build 的可行首选；640x426 下 raw float32 features 对 194 张图估算分别为 0.40GB 和 0.79GB。
- 详情见：`docs/experiments/0001_vfm_topology_scorer/vfm_backend_feasibility.md`。

## 2026-04-28 DINOv2 Cache 快速验证

命令形态：

```bash
uv run --active python -m vfm_gs.cli.build_vfm_cache \
  -s datasets/mipnerf360/bicycle \
  -i images_8 \
  -o output/0001/vfm_cache/bicycle_dinov2_vits14_smoke \
  --backend dinov2_vits14 \
  --dinov2_repo output/0001/external/dinov2 \
  --max_width 224 \
  --storage npy_float16 \
  --limit 4
```

| 产物 | 后端 | 特征 | 条目数 | 存储 | Patch Grid | 大小 | 校验 |
|---|---|---|---:|---|---|---:|---|
| `output/0001/vfm_cache/bicycle_dinov2_vits14_smoke` | `dinov2_vits14` | `dinov2_patchtokens` | 4 | `npy_float16` | 首个 entry `10x16x384` | 500K | 通过 |

- 官方 DINOv2 仓库 clone 到 ignored output state：`output/0001/external/dinov2`，并通过 `torch.hub` 以 `source="local"` 加载。
- Pretrained ViT-S/14 weights 成功下载，并通过 `forward_features` 产出归一化 patch-token maps。
- 当前 PyTorch 1.12.1 runtime 不暴露公开的 `torch.nn.functional.scaled_dot_product_attention`；builder 增加了一个窄兼容 shim，调用 1.12 私有函数，使官方 DINOv2 代码能在当前环境运行。
- DINOv2 import 时提示 `xformers` 不可用，但快速验证可以 clean fallback，不需要为了 cache building 安装它。
- storage defaults 的回归检查也通过：`cached_edge_l1` 未传 `--storage` 时默认 `npy_float32`，DINOv2 默认 `npy_float16`。
- 这个 cache-only 快速验证在训练 scorer 消费 DINOv2 maps 前验证了真实 VFM artifact 路径。下面的 token-edge scorer 快速验证是首个消费 DINO 的训练 run。

## 2026-04-28 DINOv2 Token-Edge 打分器快速验证

Cache 命令：

```bash
uv run --active python -m vfm_gs.cli.build_vfm_cache \
  -s datasets/mipnerf360/bicycle \
  -i images_8 \
  -o output/0001/vfm_cache/bicycle_dinov2_vits14 \
  --backend dinov2_vits14 \
  --dinov2_repo output/0001/external/dinov2 \
  --max_width 224
```

训练命令使用 `configs/experiments/0001_vfm_topology_dinov2_token_edge.yaml`，并沿用前面同样的 `-r 8`、220-iteration 快速验证 schedule。

| 项目 | 值 |
|---|---|
| Cache | `output/0001/vfm_cache/bicycle_dinov2_vits14` |
| Cache entries | 194 |
| Cache 存储 | `npy_float16` |
| Cache 大小 | 24M |
| Cache 构建时间 | 15s |
| 校验 | `vfm_gs.cli.validate_vfm_cache --backend dinov2_vits14` 通过 |
| 训练预检 | 194 个 `dinov2_vits14` entries 在 Scene 构建前通过 |
| 渲染 FPS | 25 个 test frames 上 410.94 FPS |

- `dinov2_token_edge_l1` 将 cached DINO patch tokens 转为标量 token-edge topology map，并与汇聚到同一 patch grid 的 SH0 渲染亮度边缘对比。
- 这是计划中第一个在训练期消费真实 DINOv2 cache data 的 scorer 变体。
- 它避免在训练循环中在线 DINO inference，因此运行开销来自 cache loading、token-edge derivation、pooling，以及额外一次 `render_fastgs(..., get_flag=True)` pass。

## 2026-05-08 DINOv2 Token-Edge Top-k 快速验证

代码变更：新增 `configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml`。它复用 `dinov2_token_edge_l1` 后端和 `output/0001/vfm_cache/bicycle_dinov2_vits14`，但把 metric map 从固定阈值改为 `vfm_metric_map_mode=topk`、`vfm_metric_topk=0.15`。目标是把 descriptor 分支中已经验证可用的 top-k 高误差区域选择迁移到当前最强的 DINO token-edge 路径上。

后续追加 `vfm_importance_normalizer=none|support_ratio`。默认值为 `none`，保持已有实验行为不变；`support_ratio` 会在 densification 打分时额外统计每个 Gaussian 的可见像素支持度，用 VFM 命中比例调节 VFM importance，避免高可见度 Gaussian 仅因曝光机会多而获得过高 VFM densification 权重。

再追加 `vfm_prune_protect_weight`、`vfm_prune_protect_mode`、`vfm_prune_protect_min_count`、`vfm_prune_protect_power`。这些参数默认关闭；开启后会把高置信 VFM 命中区域转成 pruning-side 保护分数，降低对应 Gaussian 的 `pruning_score`，让它们在训练期抽样裁剪、后期一致性裁剪和 target budget 裁剪中更不容易被删掉。`rgb_aware` 模式会额外乘以 `(1 - rgb_pruning)`，避免保护 RGB 侧已经明显不一致的点。

再追加 `target_gaussian_prune_order`。默认值为 `lowest_score`，保持已有 target-prune 实验行为不变；可选 `highest_score` 和 `lowest_opacity` 用于诊断最终预算裁剪的排序语义。该参数只在显式设置 `target_gaussian_count` 时生效。

再追加 `vfm_importance_budget_count`、`vfm_importance_budget_start_ratio` 和 `vfm_importance_budget_min_weight`。这些参数默认关闭；开启后会在训练期根据当前 Gaussian 数量动态衰减 VFM densification 权重。它不是训练结束后的硬裁剪，而是一个软预算机制：在当前点数接近 `budget_count * start_ratio` 时开始线性降低 VFM importance，到达 `budget_count` 时降到 `min_weight`。

再追加 `vfm_importance_budget_curve`，默认值为 `linear`，保持历史行为不变。可选 `quadratic` 和 `sqrt`，用于诊断非线性软预算：`quadratic` 在衰减早段更保留 VFM densification 权重，接近预算时再更快降低；`sqrt` 则更早压低权重。

主对照数据集：`datasets/mipnerf360/bicycle`，test split，`-r 8`。先用 620-step 验证链路健康，再跑 30,000 iterations 完整对照。后续用同一设置在 `garden`、`counter`、`treehill`、`bonsai`、`flowers`、`kitchen`、`room` 和 `stump` 做跨场景复验，训练仍用 train split，指标来自 test split render。

| 产物 | Backend | Metric map | Iterations | PSNR | SSIM | LPIPS | 训练时间 | Gaussian 数量 | 输出大小 | 备注 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `output/0001/vfm_dinov2_token_edge_topk015_bicycle_620_r8` | `dinov2_token_edge_l1` | top-k 15% | 620 | 20.8432 | 0.4752 | 0.5460 | 2.74s | 61,555 | 17M | 触发 token-edge top-k scoring 和 densification |
| `output/0001/vfm_dinov2_token_edge_topk015_bicycle_30k_r8` | `dinov2_token_edge_l1` | top-k 15% | 30,000 | 27.0223 | 0.8322 | 0.1810 | 140.40s | 464,998 | 136M | 完整对照，质量低于默认 DINO token-edge，但更省点更快 |
| `output/0001/vfm_dinov2_token_edge_topk025_bicycle_30k_r8` | `dinov2_token_edge_l1` | top-k 25% | 30,000 | 27.0636 | 0.8354 | 0.1748 | 146.76s | 497,328 | 144M | 当前 bicycle 30k 质量最佳 |
| `output/0001/vfm_dinov2_token_edge_topk025_budget490832_staged105_bicycle_30k_r8` | `dinov2_token_edge_l1` | top-k 25%, staged 490,832 | 30,000 | 27.0001 | 0.8286 | 0.1887 | 146.81s | 453,505 | 133M | 预算更低，但质量明显回落 |
| `output/0001/vfm_dinov2_token_edge_topk025_budget490832_final_bicycle_30k_r8` | `dinov2_token_edge_l1` | top-k 25%, final 490,832 | 30,000 | 26.8466 | 0.8244 | 0.1858 | 141.91s | 490,832 | 142M | 仅最终裁剪 6,723 个点，质量明显回落 |
| `output/0001/vfm_dinov2_token_edge_topk025_budget490832_highscore_final_bicycle_30k_r8` | `dinov2_token_edge_l1` | top-k 25%, final 490,832, high-score prune | 30,000 | 24.0554 | 0.7914 | 0.2040 | 147.65s | 490,832 | 142M | 最终裁剪高 pruning score，质量显著崩落 |
| `output/0001/vfm_dinov2_token_edge_topk025_rgb_only_bicycle_30k_r8` | `dinov2_token_edge_l1` | top-k 25%, `rgb_only` | 30,000 | 26.9340 | 0.8236 | 0.1981 | 139.06s | 411,539 | 123M | 预算贴近 cadence control，但质量不构成清晰正向 |
| `output/0001/vfm_dinov2_token_edge_topk025_i025_bicycle_30k_r8` | `dinov2_token_edge_l1` | top-k 25%, `importance_weight=0.25` | 30,000 | 26.9515 | 0.8262 | 0.1920 | 143.26s | 420,361 | 125M | 接近 cadence 预算，三项指标均小幅优于 cadence control |
| `output/0001/vfm_dinov2_token_edge_topk025_i050_bicycle_30k_r8` | `dinov2_token_edge_l1` | top-k 25%, `importance_weight=0.50` | 30,000 | 26.9966 | 0.8303 | 0.1842 | 141.34s | 440,071 | 130M | partial importance 曲线继续正向，当前预算效率最佳 |
| `output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_bicycle_30k_r8` | `dinov2_token_edge_l1` | top-k 25%, `weighted`, `importance_weight=0.50` | 30,000 | 26.9756 | 0.8288 | 0.1867 | 141.21s | 415,158 | 124M | 贴近 cadence control 点数，质量明显优于 `rgb_only` 和 cadence control |
| `output/0001/vfm_dinov2_token_edge_topk025_i050_garden_30k_r8` | `dinov2_token_edge_l1` | garden, top-k 25%, `importance_weight=0.50` | 30,000 | 28.9644 | 0.8986 | 0.0954 | 139.31s | 262,385 | 92M | 第二场景复验，超过 garden baseline 与 cached-edge v1 |
| `output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_garden_30k_r8` | `dinov2_token_edge_l1` | garden, top-k 25%, `weighted`, `importance_weight=0.50` | 30,000 | 28.9546 | 0.8977 | 0.0974 | 134.56s | 253,355 | 89M | weighted 中等增点场景复验，省点且仍超过 baseline/cached-edge |
| `output/0001/vfm_dinov2_token_edge_topk025_i050_counter_30k_r8` | `dinov2_token_edge_l1` | counter, top-k 25%, `importance_weight=0.50` | 30,000 | 29.7174 | 0.9338 | 0.0751 | 140.29s | 119,695 | 42M | 第三场景复验，点数接近 baseline 且超过 cached-edge v1 |
| `output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_counter_30k_r8` | `dinov2_token_edge_l1` | counter, top-k 25%, `weighted`, `importance_weight=0.50` | 30,000 | 29.6650 | 0.9333 | 0.0752 | 133.34s | 119,273 | 42M | weighted 低增点场景复验，仍正向但低于普通 i0.50 |
| `output/0001/vfm_dinov2_token_edge_topk025_i050_treehill_30k_r8` | `dinov2_token_edge_l1` | treehill, top-k 25%, `importance_weight=0.50` | 30,000 | 24.5173 | 0.7284 | 0.2822 | 142.14s | 432,520 | 120M | cached-edge 负例压力测试，SSIM/LPIPS 转正但 PSNR 略低于 baseline |
| `output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_treehill_30k_r8` | `dinov2_token_edge_l1` | treehill, top-k 25%, `weighted`, `importance_weight=0.50` | 30,000 | 24.5101 | 0.7281 | 0.2837 | 139.79s | 417,534 | 117M | weighted 压力复验，少点但质量小幅低于普通 i0.50 |
| `output/0001/vfm_dinov2_token_edge_topk025_i050_bonsai_30k_r8` | `dinov2_token_edge_l1` | bonsai, top-k 25%, `importance_weight=0.50` | 30,000 | 32.4920 | 0.9640 | 0.0500 | 139.32s | 136,305 | 50M | 室内小场景复验，超过 baseline 与 cached-edge v1 |
| `output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_bonsai_30k_r8` | `dinov2_token_edge_l1` | bonsai, top-k 25%, `weighted`, `importance_weight=0.50` | 30,000 | 32.5395 | 0.9642 | 0.0493 | 138.31s | 134,806 | 49M | weighted 中低增点场景复验，少点且质量微升 |
| `output/0001/vfm_dinov2_token_edge_topk025_weighted_i075_bonsai_30k_r8` | `dinov2_token_edge_l1` | bonsai, top-k 25%, `weighted`, `importance_weight=0.75` | 30,000 | 32.4848 | 0.9642 | 0.0497 | 138.14s | 138,808 | 50M | 高质量档位边界复验，仍优于 baseline/cached-edge，但低于 weighted i0.50 |
| `output/0001/vfm_dinov2_token_edge_topk025_i050_flowers_30k_r8` | `dinov2_token_edge_l1` | flowers, top-k 25%, `importance_weight=0.50` | 30,000 | 23.0134 | 0.6960 | 0.2747 | 145.30s | 350,421 | 107M | 植被/花丛场景复验，超过 baseline 与 cached-edge v1 |
| `output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_flowers_30k_r8` | `dinov2_token_edge_l1` | flowers, top-k 25%, `weighted`, `importance_weight=0.50` | 30,000 | 22.9636 | 0.6933 | 0.2791 | 141.18s | 339,267 | 104M | weighted 高增点植被场景复验，省点但质量回落更明显 |
| `output/0001/vfm_dinov2_token_edge_topk025_i050_kitchen_30k_r8` | `dinov2_token_edge_l1` | kitchen, top-k 25%, `importance_weight=0.50` | 30,000 | 33.3358 | 0.9693 | 0.0344 | 144.92s | 161,347 | 58M | 室内高基线场景复验，超过 baseline 与 cached-edge v1 |
| `output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_kitchen_30k_r8` | `dinov2_token_edge_l1` | kitchen, top-k 25%, `weighted`, `importance_weight=0.50` | 30,000 | 33.3234 | 0.9693 | 0.0347 | 141.71s | 157,804 | 57M | weighted 室内高基线场景复验，少点省时且接近普通 i0.50 |
| `output/0001/vfm_dinov2_token_edge_topk025_i050_room_30k_r8` | `dinov2_token_edge_l1` | room, top-k 25%, `importance_weight=0.50` | 30,000 | 33.0721 | 0.9622 | 0.0574 | 133.74s | 103,820 | 38M | 室内房间场景复验，超过 baseline 与 cached-edge v1 |
| `output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_room_30k_r8` | `dinov2_token_edge_l1` | room, top-k 25%, `weighted`, `importance_weight=0.50` | 30,000 | 33.1081 | 0.9626 | 0.0574 | 131.98s | 101,384 | 38M | weighted 室内房间场景复验，少点省时且质量微升 |
| `output/0001/vfm_dinov2_token_edge_topk025_i050_stump_30k_r8` | `dinov2_token_edge_l1` | stump, top-k 25%, `importance_weight=0.50` | 30,000 | 27.6106 | 0.8168 | 0.1935 | 137.90s | 365,584 | 103M | 室外树桩场景复验，超过 baseline 与 cached-edge v1 |
| `output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_stump_30k_r8` | `dinov2_token_edge_l1` | stump, top-k 25%, `weighted`, `importance_weight=0.50` | 30,000 | 27.6147 | 0.8170 | 0.1934 | 136.36s | 354,046 | 100M | weighted 大收益场景复验，少点且质量基本持平 |
| `output/0001/vfm_dinov2_token_edge_topk025_i075_bicycle_30k_r8` | `dinov2_token_edge_l1` | top-k 25%, `importance_weight=0.75` | 30,000 | 27.0284 | 0.8332 | 0.1788 | 155.18s | 472,164 | 137M | 高质量预算点，略优于 top-k 15% |
| `output/0001/vfm_dinov2_token_edge_topk025_weighted_i075_bicycle_30k_r8` | `dinov2_token_edge_l1` | top-k 25%, `weighted`, `importance_weight=0.75` | 30,000 | 26.9909 | 0.8309 | 0.1844 | 141.48s | 414,563 | 124M | 高质量档位 weighted，对比 weighted i0.50 少点且三项质量提升 |
| `output/0001/vfm_dinov2_token_edge_topk025_weighted_i075_stump_30k_r8` | `dinov2_token_edge_l1` | stump, top-k 25%, `weighted`, `importance_weight=0.75` | 30,000 | 27.6183 | 0.8178 | 0.1929 | 138.82s | 370,556 | 104M | 高质量档位第二场景复验，质量继续微升但点数增加 |
| `output/0001/vfm_dinov2_token_edge_topk025_weighted_i075_room_30k_r8` | `dinov2_token_edge_l1` | room, top-k 25%, `weighted`, `importance_weight=0.75` | 30,000 | 33.1334 | 0.9626 | 0.0575 | 120.08s | 101,965 | 38M | 高质量档位第三场景复验，PSNR 提升且点数近似不变 |
| `output/0001/vfm_dinov2_token_edge_topk025_supportnorm_i050_bicycle_620_r8` | `dinov2_token_edge_l1` | top-k 25%, `importance_weight=0.50`, `support_ratio` | 620 | 20.8465 | 0.4756 | 0.5468 | 2.96s | 61,517 | 35M | 支持度归一化链路快速验证 |
| `output/0001/vfm_dinov2_token_edge_topk025_supportnorm_i050_bicycle_30k_r8` | `dinov2_token_edge_l1` | top-k 25%, `importance_weight=0.50`, `support_ratio` | 30,000 | 26.9694 | 0.8286 | 0.1878 | 149.40s | 432,948 | 128M | 点数略降但质量回落，不优于 i0.50 |
| `output/0001/vfm_dinov2_token_edge_topk025_pruneprotect_i050_w025_bicycle_620_r8` | `dinov2_token_edge_l1` | top-k 25%, `importance_weight=0.50`, prune-protect w0.25 | 620 | 20.8808 | 0.4757 | 0.5457 | 2.72s | 61,604 | 35M | 高置信保护链路快速验证 |
| `output/0001/vfm_dinov2_token_edge_topk025_pruneprotect_i050_w025_bicycle_30k_r8` | `dinov2_token_edge_l1` | top-k 25%, `importance_weight=0.50`, prune-protect w0.25 | 30,000 | 26.9910 | 0.8302 | 0.1839 | 144.70s | 441,352 | 130M | 与 i0.50 基本持平，LPIPS 微幅改善但 PSNR/SSIM 回落 |
| `output/0001/vfm_dinov2_token_edge_budgetaware420k_i050_bicycle_620_r8` | `dinov2_token_edge_l1` | top-k 25%, `importance_weight=0.50`, soft budget 420k | 620 | 20.7641 | 0.4749 | 0.5459 | 2.81s | 61,590 | 35M | 预算感知 importance 链路快速验证 |
| `output/0001/vfm_dinov2_token_edge_budgetaware420k_i050_bicycle_30k_r8` | `dinov2_token_edge_l1` | top-k 25%, `importance_weight=0.50`, soft budget 420k | 30,000 | 26.9732 | 0.8273 | 0.1916 | 140.65s | 422,778 | 125M | 接近 i0.25 点数，质量优于 i0.25 但低于 i0.50 |
| `output/0001/vfm_dinov2_token_edge_budgetaware430k_s095_min010_i050_bicycle_30k_r8` | `dinov2_token_edge_l1` | top-k 25%, `importance_weight=0.50`, soft budget 430k, start 0.95, min 0.10 | 30,000 | 26.9750 | 0.8270 | 0.1919 | 139.38s | 419,513 | 125M | 放松衰减起点后未改善质量，点数也未上升 |
| `output/0001/vfm_dinov2_token_edge_budgetquad430k_i050_bicycle_30k_r8` | `dinov2_token_edge_l1` | top-k 25%, `importance_weight=0.50`, soft budget 430k, quadratic | 30,000 | 26.9402 | 0.8262 | 0.1918 | 140.00s | 418,137 | 124M | 非线性 late-decay 负例，质量低于线性软预算 |
| `output/0001/vfm_dinov2_token_edge_topk025_adaptive_weighted_i050_bicycle_620_r8` | `dinov2_token_edge_l1` | top-k 25%, `adaptive_weighted`, low 65k budget | 620 | 21.8205 | 0.5489 | 0.4514 | 3.11s | 180,605 | 65M | 新模式链路验证，低预算只用于触发分支 |
| `output/0001/vfm_dinov2_token_edge_topk025_adaptive_weighted_i050_budget430k_bicycle_30k_r8` | `dinov2_token_edge_l1` | top-k 25%, `adaptive_weighted`, 叠加旧权重衰减 | 30,000 | 26.9331 | 0.8248 | 0.1946 | 141.30s | 407,773 | 122M | 过度降权负例 |
| `output/0001/vfm_dinov2_token_edge_topk025_adaptive_weighted_i050_budget430k_v2_bicycle_30k_r8` | `dinov2_token_edge_l1` | top-k 25%, `adaptive_weighted`, linear 430k | 30,000 | 26.9724 | 0.8292 | 0.1859 | 141.18s | 421,472 | 125M | 修正后混合结果，LPIPS 好于 weighted 但 PSNR 略低 |
| `output/0001/vfm_dinov2_token_edge_topk025_adaptive_weighted_i050_budget430k_quad_bicycle_30k_r8` | `dinov2_token_edge_l1` | top-k 25%, `adaptive_weighted`, quadratic 430k | 30,000 | 26.9858 | 0.8302 | 0.1853 | 141.94s | 424,011 | 126M | 新的 bicycle 预算效率点 |
| `output/0001/vfm_dinov2_token_edge_topk025_adaptive_weighted_i050_budget430k_quad_treehill_30k_r8` | `dinov2_token_edge_l1` | treehill, top-k 25%, `adaptive_weighted`, quadratic 430k | 30,000 | 24.4393 | 0.7285 | 0.2821 | 140.96s | 420,283 | 117M | 第二场景复验未通过，PSNR 明显回落 |

解读：

- 训练预检、train、render 和 metrics 均通过，说明 token-edge top-k 配置可直接进入 30k 完整对照；`support_ratio` 支持度归一化也已通过 620-step 链路验证。
- 620-step 指标只说明集成健康，不作为最终质量判断；30k 完整结果相对原始 baseline 提升 +0.3191 PSNR、+0.0255 SSIM、LPIPS 改善 -0.0468，属于清晰正向。
- 相比 `fastgs_densify100` cadence control，token-edge top-k 15% 提升 +0.0936 PSNR、+0.0081 SSIM、LPIPS 改善 -0.0154，同时多 52,920 个 Gaussians，训练少 25.49s。
- 相比默认 DINO token-edge，top-k 15% 少 25,834 个 Gaussians，训练少 25.71s，但质量下降 -0.0354 PSNR、-0.0023 SSIM、LPIPS 差 +0.0043。它是更高效的 DINO token-edge 变体，但没有刷新质量上界。
- top-k 25% 刷新了当前 bicycle 30k 质量上界。相比默认 DINO token-edge，它提升 +0.0059 PSNR、+0.0009 SSIM、LPIPS 改善 -0.0019，但多 6,496 个 Gaussians；相比原始 baseline，它提升 +0.3604 PSNR、+0.0287 SSIM、LPIPS 改善 -0.0530。
- top-k 25% staged 490,832 从 iteration 7500 到 14500 触发 15 次 staged pruning，最终自然落到 453,505 个 Gaussians，低于 target 因而跳过最终裁剪。它相比 top-k 25% 完整对照少 43,823 个点，但质量下降 -0.0634 PSNR、-0.0068 SSIM、LPIPS 差 +0.0139。
- top-k 25% final 490,832 不做中期裁剪，训练结束时从 497,555 个 Gaussians 只裁到 490,832，实际删除 6,723 个点。但它相比 top-k 25% 完整对照下降 -0.2170 PSNR、-0.0110 SSIM、LPIPS 差 +0.0110；相比默认 DINO token-edge 也低 -0.2111 PSNR、-0.0101 SSIM、LPIPS 差 +0.0091。
- final 490,832 仍优于原始 baseline（+0.1434 PSNR、+0.0177 SSIM、LPIPS 改善 -0.0420），但相对 `fastgs_densify100` cadence control 变成 PSNR 更低、SSIM 基本持平、LPIPS 更好。这个负例说明当前 final target-prune 的排序即使只裁约 1.35% Gaussians，也会误删对结构质量敏感的点。
- high-score final 490,832 显式改为裁剪最高 `pruning_score`。训练从 496,264 个 Gaussians 裁到 490,832，只删除 5,432 个点，但 PSNR/SSIM/LPIPS 变为 24.0554、0.7914、0.2040，比 low-score final 更差。这说明 target-prune 不能简单复用训练期/最终一致性裁剪的高分删除语义；`pruning_score` 在终局小比例批量裁剪里不能稳定表达“可安全删除”。
- top-k 25% `rgb_only` 关闭直接 VFM densification，只保留 VFM support/pruning 侧影响。它最终 411,539 个 Gaussians，几乎贴住 `fastgs_densify100` 的 412,078；相比 cadence control，PSNR 只高 +0.0053，SSIM 低 -0.0005，LPIPS 差 +0.0017。因此“只做 prune-protect / support 重排”可以控制预算，但没有保住 top-k 25% 完整对照的质量收益。
- top-k 25% `importance_weight=0.25` 保留少量 VFM densification，而不是像 `rgb_only` 一样完全关闭。它最终 420,361 个 Gaussians，比 cadence control 多 8,283 个；相比 cadence control，PSNR 高 +0.0228、SSIM 高 +0.0021、LPIPS 改善 -0.0044。相比 `rgb_only`，它多 8,822 个点，但 PSNR 高 +0.0175、SSIM 高 +0.0027、LPIPS 改善 -0.0061，说明 partial VFM importance 是比完全关闭 VFM densification 更合理的预算方向。
- top-k 25% `importance_weight=0.50` 进一步把预算推到 440,071 个 Gaussians，相比 `importance_weight=0.25` 多 19,710 个点，但换来 +0.0451 PSNR、+0.0041 SSIM、LPIPS 改善 -0.0078。相比 cadence control，它多 27,993 个点，PSNR 高 +0.0679、SSIM 高 +0.0062、LPIPS 改善 -0.0122，并且训练时间更短。这个点是当前 partial importance 曲线上最好的预算效率折中。
- top-k 25% `weighted + importance_weight=0.50` 把 RGB 与 VFM importance 做加权平均，而不是先缩放 VFM 再与 RGB 取最大值。它最终 415,158 个 Gaussians，只比 cadence control 多 3,080 个；相比 cadence control，PSNR 高 +0.0469、SSIM 高 +0.0047、LPIPS 改善 -0.0097，训练少 24.68s。相比普通 `max + importance_weight=0.50`，它少 24,913 个点，PSNR 低 -0.0210、SSIM 低 -0.0015、LPIPS 差 +0.0025。这个结果说明 `weighted` 是目前最接近 cadence 预算且仍保住大部分 DINO 收益的融合方式，值得做第二场景压力测试。
- `garden` 复验先构建并验证 `output/0001/vfm_cache/garden_dinov2_vits14`，共 185 张图对应的 DINOv2 ViT-S/14 cache，目录大小约 23M。`top-k 25% + importance_weight=0.50` 完成 30k train、render 和 metrics，最终 262,385 个 Gaussians。
- 在 `garden` 上，DINO token-edge partial importance 相比本地 baseline `output/0001/baseline_garden_30k_r8` 提升 +0.2593 PSNR、+0.0097 SSIM、LPIPS 改善 -0.0179。若对比 full MipNeRF360 v1 统一表中的 garden baseline，则提升 +0.2388 PSNR、+0.0092 SSIM、LPIPS 改善 -0.0177，同时多 65,878 个 Gaussians，训练多 10.38s。
- 相比 full MipNeRF360 v1 的 garden cached-edge v1，DINO token-edge partial importance 仍提升 +0.0641 PSNR、+0.0024 SSIM、LPIPS 改善 -0.0048；代价是多 13,981 个 Gaussians，训练多 4.28s。这说明 i0.50 不是只在 bicycle 生效，但它仍不是 budget-neutral 结论。
- `garden weighted + importance_weight=0.50` 最终 253,355 个 Gaussians，比普通 garden i0.50 少 9,030 个点，训练少 4.75s；PSNR 低 -0.0098、SSIM 低 -0.0009，LPIPS 差 +0.0020。相比 baseline，它仍提升 +0.2290 PSNR、+0.0084 SSIM、LPIPS 改善 -0.0158；相比 cached-edge v1，也提升 +0.0543 PSNR、+0.0015 SSIM、LPIPS 改善 -0.0028。这个结果说明在中等点数增长场景中，`weighted` 可以省点省时并保留主要收益，但感知指标回落比 stump 更明显。
- `counter` 复验先构建并验证 `output/0001/vfm_cache/counter_dinov2_vits14`，共 240 张图对应的 DINOv2 ViT-S/14 cache，目录大小约 30M。`top-k 25% + importance_weight=0.50` 完成 30k train、render 和 metrics，最终 119,695 个 Gaussians。
- 在 `counter` 上，DINO token-edge partial importance 相比 full MipNeRF360 v1 的 counter baseline 提升 +0.1763 PSNR、+0.0026 SSIM、LPIPS 改善 -0.0055；Gaussian 数量只多 6,672 个，约 +5.9%，训练多 16.02s。相比 counter cached-edge v1，仍提升 +0.0746 PSNR、+0.0017 SSIM、LPIPS 改善 -0.0037，点数多 8,570 个，训练多 8.16s。
- `counter weighted + importance_weight=0.50` 最终 119,273 个 Gaussians，比普通 counter i0.50 少 422 个点，训练少 6.95s；但 PSNR 低 -0.0524、SSIM 低 -0.0005，LPIPS 差 +0.0001。相比 baseline，它仍提升 +0.1239 PSNR、+0.0022 SSIM、LPIPS 改善 -0.0054；相比 cached-edge v1，也提升 +0.0222 PSNR、+0.0012 SSIM、LPIPS 改善 -0.0035。这个结果说明在 `counter` 这种普通 i0.50 已经接近 baseline 点数的场景中，`weighted` 的省点空间很小，优先级不如普通 i0.50。
- `treehill` 是 full MipNeRF360 v1 中 cached-edge v1 的明确负例。DINO 复验先构建并验证 `output/0001/vfm_cache/treehill_dinov2_vits14`，共 141 张图对应的 DINOv2 ViT-S/14 cache，目录大小约 18M。`top-k 25% + importance_weight=0.50` 完成 30k train、render 和 metrics，最终 432,520 个 Gaussians。
- 在 `treehill` 上，DINO token-edge partial importance 相比 baseline 的 PSNR 低 -0.0344，但 SSIM 高 +0.0110，LPIPS 改善 -0.0407；相比 cached-edge v1，PSNR 高 +0.0779、SSIM 高 +0.0158、LPIPS 改善 -0.0431。这个结果说明 DINO token-edge 比 edge proxy 更稳，但它通过 +75.7% Gaussian 数量换取感知/结构修复，仍不能作为预算受控的正例。
- `treehill weighted + importance_weight=0.50` 最终 417,534 个 Gaussians，比普通 treehill i0.50 少 14,986 个点，训练少 2.35s；PSNR 低 -0.0072、SSIM 低 -0.0003、LPIPS 差 +0.0015。相比 baseline，它仍是 PSNR 低 -0.0416、SSIM 高 +0.0106、LPIPS 改善 -0.0391；相比 cached-edge v1，三项指标仍正向。该结果说明 weighted 融合能省掉一部分点且基本保住普通 i0.50 的压力场景收益，但在 treehill 上还没有解决相对 baseline 的 PSNR 负向，也没有把点数拉回接近 baseline。
- `bonsai` 复验先构建并验证 `output/0001/vfm_cache/bonsai_dinov2_vits14`，共 292 张图对应的 DINOv2 ViT-S/14 cache，目录大小约 36M。`top-k 25% + importance_weight=0.50` 完成 30k train、render 和 metrics，最终 136,305 个 Gaussians。
- 在 `bonsai` 上，DINO token-edge partial importance 相比 full MipNeRF360 v1 的 bonsai baseline 提升 +0.1346 PSNR、+0.0044 SSIM、LPIPS 改善 -0.0123；Gaussian 数量多 12,963 个，训练多 9.79s。相比 bonsai cached-edge v1，PSNR 高 +0.2654、SSIM 高 +0.0040、LPIPS 改善 -0.0088，点数多 24,510 个，训练时间基本持平且略少 0.51s。
- `bonsai weighted + importance_weight=0.50` 最终 134,806 个 Gaussians，比普通 bonsai i0.50 少 1,499 个点，训练少 1.01s；PSNR 高 +0.0475、SSIM 高 +0.0002，LPIPS 改善 -0.0007。相比 baseline，它提升 +0.1821 PSNR、+0.0046 SSIM、LPIPS 改善 -0.0130；相比 cached-edge v1，也提升 +0.3129 PSNR、+0.0042 SSIM、LPIPS 改善 -0.0095。这个结果说明 weighted 在中低增点场景不一定像 counter 那样退化；当场景仍受益于 DINO token topology 时，它可以同时省点、提速并略微提升质量。
- `flowers` 复验先构建并验证 `output/0001/vfm_cache/flowers_dinov2_vits14`，共 173 张图对应的 DINOv2 ViT-S/14 cache，目录大小约 22M。`top-k 25% + importance_weight=0.50` 完成 30k train、render 和 metrics，最终 350,421 个 Gaussians。
- 在 `flowers` 上，DINO token-edge partial importance 相比 full MipNeRF360 v1 的 flowers baseline 提升 +0.2592 PSNR、+0.0237 SSIM、LPIPS 改善 -0.0440；Gaussian 数量多 141,774 个，训练多 21.33s。相比 flowers cached-edge v1，仍提升 +0.0459 PSNR、+0.0069 SSIM、LPIPS 改善 -0.0115，点数多 55,888 个，训练多 3.05s。
- `flowers weighted + importance_weight=0.50` 最终 339,267 个 Gaussians，比普通 flowers i0.50 少 11,154 个点，训练少 4.12s；PSNR 低 -0.0498、SSIM 低 -0.0027，LPIPS 差 +0.0044。相比 baseline，它仍提升 +0.2094 PSNR、+0.0210 SSIM、LPIPS 改善 -0.0396；相比 cached-edge v1，PSNR 基本持平但低 -0.0040，SSIM 高 +0.0043，LPIPS 改善 -0.0071。这个结果说明 weighted 在复杂植被/高增点场景仍有正向折中价值，但质量回落比 garden/stump 更明显，不适合无条件替代普通 i0.50。
- `kitchen` 复验先构建并验证 `output/0001/vfm_cache/kitchen_dinov2_vits14`，共 279 张图对应的 DINOv2 ViT-S/14 cache，目录大小约 34M。`top-k 25% + importance_weight=0.50` 完成 30k train、render 和 metrics，最终 161,347 个 Gaussians。
- 在 `kitchen` 上，DINO token-edge partial importance 相比 full MipNeRF360 v1 的 kitchen baseline 提升 +0.2438 PSNR、+0.0021 SSIM、LPIPS 改善 -0.0035；Gaussian 数量少 7,629 个，训练多 12.83s。相比 kitchen cached-edge v1，仍提升 +0.0256 PSNR、+0.0002 SSIM、LPIPS 改善 -0.0006，点数只多 2,567 个，训练多 3.41s。
- `kitchen weighted + importance_weight=0.50` 最终 157,804 个 Gaussians，比普通 kitchen i0.50 少 3,543 个点，训练少 3.21s；PSNR 低 -0.0124、SSIM 基本持平、LPIPS 差 +0.0003。相比 baseline，它仍提升 +0.2314 PSNR、+0.0021 SSIM、LPIPS 改善 -0.0032，并且少 11,172 个点；相比 cached-edge v1，也小幅提升 +0.0132 PSNR、+0.0002 SSIM、LPIPS 改善 -0.0003。这个结果把 weighted 在室内高基线场景中的角色定为省点省时的折中，而不是质量上界替代。
- `room` 复验先构建并验证 `output/0001/vfm_cache/room_dinov2_vits14`，共 311 张图对应的 DINOv2 ViT-S/14 cache，目录大小约 38M。`top-k 25% + importance_weight=0.50` 完成 30k train、render 和 metrics，最终 103,820 个 Gaussians。
- 在 `room` 上，DINO token-edge partial importance 相比 full MipNeRF360 v1 的 room baseline 提升 +0.0945 PSNR、+0.0025 SSIM、LPIPS 改善 -0.0037；Gaussian 数量多 12,506 个，训练多 11.97s。相比 room cached-edge v1，仍提升 +0.1037 PSNR、+0.0002 SSIM、LPIPS 改善 -0.0004，点数多 4,921 个，训练少 1.69s。
- `room weighted + importance_weight=0.50` 最终 101,384 个 Gaussians，比普通 room i0.50 少 2,436 个点，训练少 1.76s；PSNR 高 +0.0360、SSIM 高 +0.0004，LPIPS 基本持平。相比 baseline，它提升 +0.1305 PSNR、+0.0029 SSIM、LPIPS 改善 -0.0038；相比 cached-edge v1，也提升 +0.1396 PSNR、+0.0006 SSIM、LPIPS 改善 -0.0004。该结果把 weighted 的室内小场景结论从 kitchen 的“低风险折中”推进到“少点省时且质量微升”。
- `stump` 复验先构建并验证 `output/0001/vfm_cache/stump_dinov2_vits14`，共 125 张图对应的 DINOv2 ViT-S/14 cache，目录大小约 16M。`top-k 25% + importance_weight=0.50` 完成 30k train、render 和 metrics，最终 365,584 个 Gaussians。
- 在 `stump` 上，DINO token-edge partial importance 相比 full MipNeRF360 v1 的 stump baseline 提升 +0.4350 PSNR、+0.0234 SSIM、LPIPS 改善 -0.0393；Gaussian 数量多 194,825 个，训练多 16.79s。相比 stump cached-edge v1，仍提升 +0.3632 PSNR、+0.0236 SSIM、LPIPS 改善 -0.0367，点数多 123,106 个，训练少 5.70s。
- `stump weighted + importance_weight=0.50` 最终 354,046 个 Gaussians，比普通 stump i0.50 少 11,538 个点，训练少 1.54s；PSNR 高 +0.0041、SSIM 高 +0.0002，LPIPS 改善 -0.0001。相比 baseline，它提升 +0.4391 PSNR、+0.0236 SSIM、LPIPS 改善 -0.0393；相比 cached-edge v1，仍提升 +0.3672 PSNR、+0.0238 SSIM、LPIPS 改善 -0.0368。这个结果说明 `weighted` 在大收益场景中不仅能省点，还能基本完整保留普通 i0.50 的质量收益。
- MipNeRF360 全 9 场景 candidate 均值为 PSNR 28.8577、SSIM 0.8666、LPIPS 0.1385、263,572 个 Gaussians、训练 140.47s。相比 9 场景 baseline 均值，DINO i0.50 平均提升 +0.2051 PSNR、+0.0115 SSIM、LPIPS 改善 -0.0234，但平均多 90,231 个 Gaussians、训练多 15.09s；相比 cached-edge v1 均值，平均提升 +0.1365 PSNR、+0.0087 SSIM、LPIPS 改善 -0.0166，平均多 47,703 个 Gaussians、训练多 1.15s。
- 因此 `top-k 25% + importance_weight=0.50` 可以作为 0001 v1 candidate：它在 MipNeRF360 全场景上比 baseline 和 cached-edge v1 都更强，9/9 场景相对 cached-edge v1 三项指标均正向，8/9 场景相对 baseline 的 PSNR 正向，9/9 场景相对 baseline 的 SSIM/LPIPS 正向。但它仍不是预算受控版本，下一版应进入自动容量保护和预算感知 scorer，而不是继续追加同类 top-k 单点。
- MipNeRF360 全 9 场景 weighted i0.50 均值为 PSNR 28.8505、SSIM 0.8660、LPIPS 0.1397、254,736 个 Gaussians、训练 137.60s。相比 baseline 平均提升 +0.1978 PSNR、+0.0109 SSIM、LPIPS 改善 -0.0223；相比普通 i0.50 平均少 8,836 个 Gaussians、训练少 2.87s，质量只小幅回落 -0.0072 PSNR、-0.0006 SSIM、LPIPS 差 +0.0012。因此 weighted i0.50 已形成完整的全场景预算效率候选，但质量上界仍保留给普通 i0.50。
- MipNeRF360 全 9 场景 fixed weighted i0.75 均值为 PSNR 28.8396、SSIM 0.8663、LPIPS 0.1395、257,715 个 Gaussians、训练 168.35s。相比 weighted i0.50，PSNR 低 -0.0109，SSIM 高 +0.0003，LPIPS 改善 -0.0001，平均多 2,978 个 Gaussians、训练多 30.75s。因此固定 i0.75 不能作为全场景默认档位。
- MipNeRF360 全 9 场景 fixed weighted i0.90 均值为 PSNR 28.8238、SSIM 0.8661、LPIPS 0.1394、253,687 个 Gaussians、训练 195.95s。相比 weighted i0.50，PSNR 低 -0.0268，SSIM 基本持平，LPIPS 改善 -0.0003，平均少 1,049 个 Gaussians，但训练多 58.35s。因此固定 i0.90 也不能作为全场景默认档位。
- 严格 `quality_pick` 已扩展为在 i0.50/i0.75/i0.90 之间选择：bicycle、garden、room、stump 选 i0.75，counter 选 i0.90，bonsai、flowers、kitchen、treehill 回退 i0.50。该选择均值为 PSNR 28.8641、SSIM 0.8665、LPIPS 0.1392、257,326 个 Gaussians、训练 151.84s。相比 weighted i0.50，三项质量均提升：+0.0136 PSNR、+0.0004 SSIM、LPIPS 改善 -0.0005，只多 2,589 个 Gaussians；相比普通 i0.50，PSNR 高 +0.0064，点数少 6,246。
- `QCGI pick` 使用质量-容量收益指数选择档位：除严格 `quality_pick` 已选择的场景外，treehill 也切到 i0.90，因为它几乎不损 PSNR（-0.0003），同时 SSIM 提升 +0.0024、LPIPS 改善 -0.0030，并减少 13,534 个 Gaussians。QCGI 选择均值为 PSNR 28.8641、SSIM 0.8667、LPIPS 0.1388、255,822 个 Gaussians、训练 158.78s；相比 weighted i0.50 提升 +0.0136 PSNR、+0.0007 SSIM、LPIPS 改善 -0.0008，只多 1,086 个 Gaussians。因此本轮的正向结论是“场景级档位选择 + QCGI 约束”，不是固定 i0.75 或 i0.90。
- top-k 25% `importance_weight=0.75` 继续改善质量，但收益开始变慢：相比 `importance_weight=0.50` 多 32,093 个 Gaussians，换来 +0.0317 PSNR、+0.0028 SSIM、LPIPS 改善 -0.0055。它相对 top-k 15% 多 7,166 个点，PSNR 高 +0.0061、SSIM 高 +0.0010、LPIPS 改善 -0.0022，因此可作为高质量预算点，但预算效率不如 `importance_weight=0.50` 清晰。
- top-k 25% `weighted + importance_weight=0.75` 在 bicycle 上最终 414,563 个 Gaussians，PSNR 26.9909、SSIM 0.8309、LPIPS 0.1844，训练 141.48s。相比 `weighted i0.50`，它少 595 个点，PSNR 高 +0.0153、SSIM 高 +0.0021、LPIPS 改善 -0.0023；相比普通 i0.75，少 57,601 个点、训练少 13.70s，但 PSNR 低 -0.0375、SSIM 低 -0.0023、LPIPS 差 +0.0056。因此它是新的 bicycle 近预算正向点，下一步可优先在 `stump/room` 这类 weighted i0.50 已经微升质量的场景复验。
- `stump weighted + importance_weight=0.75` 最终 370,556 个 Gaussians，PSNR 27.6183、SSIM 0.8178、LPIPS 0.1929，训练 138.82s。相比 `stump weighted i0.50`，它多 16,510 个点、训练多 2.46s，但 PSNR 高 +0.0036、SSIM 高 +0.0008、LPIPS 改善 -0.0005；相比普通 stump i0.50，它多 4,972 个点、训练多 0.92s，PSNR 高 +0.0077、SSIM 高 +0.0010、LPIPS 改善 -0.0006。这个结果说明 `weighted i0.75` 的质量档位在第二个大收益场景也成立，但不再具备 bicycle 那种“少点且质量提升”的预算优势。
- `room weighted + importance_weight=0.75` 最终 101,965 个 Gaussians，PSNR 33.1334、SSIM 0.9626、LPIPS 0.0575，训练 120.08s。相比 `room weighted i0.50`，它只多 581 个点，训练少 11.90s，PSNR 高 +0.0253、SSIM 基本持平、LPIPS 差 +0.0001；相比普通 room i0.50，它少 1,855 个点、训练少 13.66s，PSNR 高 +0.0613、SSIM 高 +0.0004、LPIPS 差 +0.0000。这个结果把 `weighted i0.75` 从 bicycle/stump 扩展到室内小场景，说明它可以作为高质量档位，但 LPIPS 不是稳定占优指标。
- `bonsai weighted + importance_weight=0.75` 最终 138,808 个 Gaussians，PSNR 32.4848、SSIM 0.9642、LPIPS 0.0497，训练 138.14s。相比 `bonsai weighted i0.50`，它多 4,002 个点、训练少 0.17s，但 PSNR 低 -0.0547、SSIM 基本持平、LPIPS 差 +0.0004；相比普通 bonsai i0.50，它多 2,503 个点、训练少 1.18s，PSNR 低 -0.0072、SSIM 高约 +0.0002、LPIPS 改善约 -0.0003。它仍明显优于 bonsai baseline 与 cached-edge v1，但没有通过“高质量档位应优于 weighted i0.50”的门槛。因此 bonsai 的推荐档位保持 `weighted i0.50`，也说明 i0.75 不能做无条件质量默认值。
- 新补齐的 i0.75 五场景显示出明显场景差异：counter 相比 i0.50 提升 +0.0531 PSNR 且只多 2,313 个点；garden 提升 +0.0200 PSNR、SSIM/LPIPS 同步改善，只多 3,574 个点；flowers、kitchen、treehill 则未通过质量门槛，其中 kitchen 虽少 1,348 个点但 PSNR 低 -0.1152。该结果修正了单场景筛选口径：单场景负例不应直接否定候选，但固定超参也不能只凭少数正例升级，最终以数据集均值和场景选择均值为准。
- `support_ratio` 快速验证在 620-step 下得到 61,517 个 Gaussians，PSNR 20.8465、SSIM 0.4756、LPIPS 0.5468；它与历史 top-k 15% 快速验证的 61,555 个点基本一致，说明额外的 support-count raster pass 没有破坏 densification 链路。
- `support_ratio + importance_weight=0.50` 30k 最终 432,948 个 Gaussians，比普通 i0.50 少 7,123 个点，但质量回落 -0.0272 PSNR、-0.0018 SSIM、LPIPS 差 +0.0036，训练时间也多 8.05s。它仍优于 cadence control（+0.0407 PSNR、+0.0045 SSIM、LPIPS 改善 -0.0086），但不如直接 partial importance i0.50，因此支持度归一化不是当前最佳预算效率方向。
- 高置信 VFM 区域保护的 620-step 快速验证使用 `vfm_prune_protect_weight=0.25`、`vfm_prune_protect_mode=rgb_aware`、`vfm_prune_protect_min_count=5`、`vfm_prune_protect_power=2.0`。它完成 train、render 和 metrics，最终 61,604 个 Gaussians，PSNR 20.8808、SSIM 0.4757、LPIPS 0.5457，说明 protection score 可以同时接入 densification 评分和后期 pruning-only 评分路径。
- 高置信 VFM 区域保护的 30k 正式对照最终 441,352 个 Gaussians，比普通 `importance_weight=0.50` 多 1,281 个点；PSNR 低 -0.0056、SSIM 低 -0.0001，LPIPS 仅改善 -0.0003，训练时间多 3.36s。它没有形成比 i0.50 更好的预算效率点，说明简单保护高 VFM 命中区域不能解决当前的预算-质量矛盾。
- 预算感知 importance 的 620-step 快速验证使用 `vfm_importance_budget_count=420000`、`vfm_importance_budget_start_ratio=0.90`、`vfm_importance_budget_min_weight=0.0`。短程点数远低于软预算区间，因此它主要验证参数读取和 scorer 分支健康；train、render 和 metrics 均通过，结果为 61,590 个 Gaussians，PSNR 20.7641、SSIM 0.4749、LPIPS 0.5459。
- 预算感知 importance 的 30k 对照最终 422,778 个 Gaussians，接近固定 `importance_weight=0.25` 的 420,361 个点；相比 i0.25 提升 +0.0217 PSNR、+0.0011 SSIM、LPIPS 改善 -0.0004，训练少 2.61s。相比普通 i0.50，它少 17,293 个点，但质量回落 -0.0234 PSNR、-0.0031 SSIM、LPIPS 差 +0.0074。这个结果说明动态软预算优于固定低权重，但当前线性衰减还没有保住 i0.50 的主要质量收益。
- 放松软预算的 430k 对照使用 `vfm_importance_budget_count=430000`、`vfm_importance_budget_start_ratio=0.95`、`vfm_importance_budget_min_weight=0.10`，最终 419,513 个 Gaussians，PSNR 26.9750、SSIM 0.8270、LPIPS 0.1919，训练 139.38s。相比 420k 软预算，它少 3,265 个点，PSNR 只高 +0.0018，SSIM 低 -0.0003，LPIPS 差 +0.0003；相比普通 i0.50，仍少 20,558 个点但质量回落 -0.0216 PSNR、-0.0034 SSIM、LPIPS 差 +0.0077。因此“更晚开始衰减 + 保留 10% 最小 VFM 权重”没有成为新主结果，短期不迁移全场景。
- quadratic 430k 对照最终 418,137 个 Gaussians，PSNR 26.9402、SSIM 0.8262、LPIPS 0.1918，训练 140.00s。相比 420k 线性软预算，点数少 4,641，但 PSNR 低 -0.0330、SSIM 低 -0.0011，LPIPS 只好 +0.0002；相比 430k 放松衰减，PSNR 低 -0.0348、SSIM 低 -0.0008，LPIPS 只好 +0.0001。这个结果说明“直接衰减 VFM 权重”的二次曲线没有改善预算质量曲线，至少在 bicycle 上不是值得迁移的方向。
- `adaptive_weighted` 代码路径把预算进度用于 importance 融合语义，而不是直接衰减 VFM 权重：软预算区间前使用接近 `max` 的 partial VFM importance，接近预算时平滑过渡到 RGB/VFM weighted average。首个 65k 低预算 620-step 快速验证完成 train、render 和 metrics，说明分支可运行；指标不作为质量判断。
- 首个 430k 30k 对照错误地叠加了旧的 budget-aware weight decay，最终只有 407,773 个 Gaussians，PSNR 26.9331、SSIM 0.8248、LPIPS 0.1946。它说明“过渡到 weighted”与“继续衰减 VFM weight”不能同时使用，否则会过度削弱 VFM densification。
- 修正后的 `adaptive_weighted + quadratic 430k` 在 bicycle 上优于 `weighted i0.50`，但 treehill 第二场景复验没有复制这个收益：最终 420,283 个 Gaussians，PSNR 24.4393、SSIM 0.7285、LPIPS 0.2821。相比普通 treehill i0.50，PSNR 低 -0.0780、SSIM 基本持平、LPIPS 仅改善 -0.0001；相比 `weighted i0.50`，点数多 2,749 个但 PSNR 低 -0.0708、SSIM 高 +0.0004、LPIPS 改善 -0.0016。因此它不进入正向改进表，也不替代全场景 `weighted i0.50` 结论。
- 修正后的 `adaptive_weighted` linear 430k 对照最终 421,472 个 Gaussians，PSNR 26.9724、SSIM 0.8292、LPIPS 0.1859。相比 `weighted i0.50`，点数多 6,314，PSNR 低 -0.0032，但 SSIM 高 +0.0004、LPIPS 改善 -0.0008；这是混合结果，还不足以替代 weighted。
- `adaptive_weighted` quadratic 430k 对照最终 424,011 个 Gaussians，PSNR 26.9858、SSIM 0.8302、LPIPS 0.1853，训练 141.94s。相比 `weighted i0.50` 多 8,853 个点，但三项质量均更好：+0.0102 PSNR、+0.0014 SSIM、LPIPS 改善 -0.0014；相比普通 i0.50 少 16,060 个点，PSNR 低 -0.0108，但 SSIM 低 -0.0001、LPIPS 差 +0.0011。它是新的 bicycle 预算效率点，但还需要至少在 `stump/room/kitchen/treehill` 上复验，才能替代全场景 weighted 结论。
- staged 490,832、final 490,832、high-score final 490,832 和 `rgb_only` 分别暴露了四类问题：中期反复压 cap 会损伤结构生长，终局一次性低分裁剪排序不够可靠，终局高分裁剪更不可靠，完全关闭 VFM densification 又会交回主要质量收益。`importance_weight=0.25/0.50/0.75` 给出连续正向曲线，`weighted + importance_weight=0.50` 则把点数进一步拉回 cadence 预算附近，因此 partial VFM importance 与加权融合是当前最有效的预算控制方向。
- 这一路径不引入在线 DINO inference，成本明显低于 descriptor 系列。当前 0001 在 bicycle 上的最佳质量结论仍定为 top-k 25% 完整对照；预算方向把 `weighted + importance_weight=0.50` 作为已完成全场景验证的效率候选，把 `max + importance_weight=0.50` 作为全场景质量 candidate。MipNeRF360 全场景复验支持 i0.50 的跨场景有效性，尤其 `counter` 在仅 +5.9% Gaussian 的条件下获得三项指标提升，`kitchen` 在少于 baseline 点数下仍获得三项指标提升，`room` 证明高基线小室内场景上也能略超 cached-edge v1，`stump` 则给出全场景中最大的 PSNR 增益；`treehill` 显示 DINO i0.50 能修复 cached-edge 负例的 SSIM/LPIPS，但预算膨胀明显且 PSNR 仍略低于 baseline。weighted 全场景复验显示：固定 i0.50 平均质量几乎贴住普通 i0.50，同时少 8,836 个 Gaussians、训练少 2.87s；固定 i0.75/i0.90 都不是全场景默认值，但三档 `quality_pick` 和 `QCGI pick` 平均 PSNR 均超过普通 i0.50 与 weighted i0.50，其中 QCGI 版本只比 weighted i0.50 多 1,086 个点。因此下一步主线应转向场景级档位选择，而不是继续寻找单一固定 importance weight。adaptive weighted 在 treehill 第二场景未通过，后续不升级为主线。

## 2026-05-07 DINOv2 Descriptor 打分器快速验证

代码变更：增加 `dinov2_descriptor_cosine` 后端和 `configs/experiments/0001_vfm_topology_dinov2_descriptor.yaml`。该后端复用 GT 侧 `dinov2_patchtokens` cache，但在 densification/pruning 节点对 SH0 渲染图在线运行同一个 DINOv2 ViT-S/14，再用 patch-token cosine distance 生成 `pixel_error_map`。这比 `dinov2_token_edge_l1` 更接近 proposal 中的语义特征误差，但会增加训练期 scorer 开销。

使用 cache：`output/0001/vfm_cache/bicycle_dinov2_vits14`，194 个 entries，`npy_float16`，24M；DINOv2 本地仓库：`output/0001/external/dinov2`。

| 产物 | 后端 | 迭代 | PSNR | SSIM | LPIPS | 训练时间 | Gaussian 数量 | 输出大小 | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `output/0001/vfm_dinov2_descriptor_smoke_bicycle_80_r8` | `dinov2_descriptor_cosine` | 80 | 18.5433 | 0.3634 | 0.6927 | 1.66s | 57,709 | 16M | 最小链路验证，触发一次 descriptor scoring |
| `output/0001/vfm_dinov2_descriptor_bicycle_smoke` | `dinov2_descriptor_cosine` | 220 | 20.0193 | 0.4233 | 0.6018 | 2.55s | 77,060 | 20M | 与历史快速验证 schedule 对齐 |
| `output/0001/vfm_dinov2_descriptor_t030_bicycle_smoke` | `dinov2_descriptor_cosine`, `vfm_loss_thresh=0.30` | 220 | 20.2853 | 0.4276 | 0.6011 | 2.58s | 78,935 | 39M | 接近 0.35，但 LPIPS 较差 |
| `output/0001/vfm_dinov2_descriptor_t035_bicycle_smoke` | `dinov2_descriptor_cosine`, `vfm_loss_thresh=0.35` | 220 | 20.2897 | 0.4287 | 0.5993 | 2.65s | 79,120 | 39M | 三个阈值点中最好 |
| `output/0001/vfm_dinov2_descriptor_t040_bicycle_smoke` | `dinov2_descriptor_cosine`, `vfm_loss_thresh=0.40` | 220 | 20.2550 | 0.4267 | 0.6008 | 2.64s | 76,773 | 39M | 低于 0.35 |
| `output/0001/vfm_dinov2_descriptor_t065_bicycle_smoke` | `dinov2_descriptor_cosine`, `vfm_loss_thresh=0.65` | 220 | 20.2162 | 0.4253 | 0.6034 | 2.46s | 77,037 | 39M | PSNR/SSIM 好于默认，LPIPS 较差 |
| `output/0001/vfm_dinov2_descriptor_w518_t035_bicycle_smoke` | `dinov2_descriptor_cosine`, `max_width=518`, `vfm_loss_thresh=0.35` | 220 | 20.3059 | 0.4272 | 0.5999 | 2.69s | 78,927 | 21M | 高分辨率 DINO cache 复测 |

解读：

- descriptor scorer 已完成 train、render 和 metrics，证明“渲染图在线 DINO descriptor vs GT cache descriptor”的真实语义特征路径可以接入现有 `pixel_error_map -> metric_map -> accum_metric_counts` 管线。
- 220-iteration 指标低于 token-edge 快速验证：PSNR -0.2720、SSIM -0.0039、LPIPS +0.0012。这个差距不能直接否定 descriptor 路径，因为短跑主要验证集成健康，且 descriptor 后端的阈值、cache 分辨率和投影策略尚未调参。
- 阈值小网格显示 `vfm_loss_thresh=0.35` 明显优于默认 0.50：+0.2704 PSNR、+0.0054 SSIM、-0.0025 LPIPS。它也接近 token-edge 快速验证的 PSNR/SSIM，并在 LPIPS 上略优。
- 细扫 0.30 和 0.40 后，0.35 仍是当前最优短跑阈值。0.30 的 PSNR 接近 0.35，但 SSIM/LPIPS 更弱；0.40 三项指标都低于 0.35。
- `vfm_loss_thresh=0.65` 的 PSNR/SSIM 也优于默认，但 LPIPS 变差。下一轮应优先补高分辨率 DINO cache 或直接用 0.35 跑 30k，而不是继续扩大低分辨率阈值网格。
- `max_width=518` DINO cache 构建耗时 9.90s，194 entries，大小 127M，首个 entry 形状为 `24x37x384`，并通过 `validate_vfm_cache`。相比 `max_width=224` 的 24M cache，它带来更密的 descriptor grid。
- `max_width=518` + `vfm_loss_thresh=0.35` 的 PSNR 比 `224` cache 高 +0.0162，但 SSIM 低 -0.0015，LPIPS 差 +0.0007。高分辨率 cache 没有在短跑中形成全面优势，但训练时间保持接近，说明 30k 成本风险不大。
- 当前实现每个 scorer 视角都在线运行 DINOv2。短跑训练时间从 token-edge 的 1.72s 增到 2.55s，仍可接受，但完整训练还需要单独记录成本。
- DINOv2 import 继续提示 `xformers` 不可用，但官方实现能 fallback，训练未失败。

## 2026-05-07 DINOv2 Descriptor 30k 完整训练

数据集：`datasets/mipnerf360/bicycle`，test split，`-r 8`，30,000 iterations。配置使用 `configs/experiments/0001_vfm_topology_dinov2_descriptor.yaml`，并覆盖 `vfm_loss_thresh=0.35`；cache 使用 `output/0001/vfm_cache/bicycle_dinov2_vits14`，即 `max_width=224`、194 entries、24M 的 DINOv2 ViT-S/14 patch-token cache。该配置的 `densification_interval=100`，因此应优先与 cadence control 和 DINO token-edge 对比，而不是只与原始 baseline 对比。

| 产物 | Scorer / Backend | PSNR | SSIM | LPIPS | 训练时间 | Gaussian 数量 | 输出大小 | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `output/0001/baseline_bicycle_30k_r8` | `fastgs_photometric` | 26.7032 | 0.8067 | 0.2278 | 116.92s | 240,394 | 82M | 原始 baseline |
| `output/0001/fastgs_densify100_bicycle_30k_r8` | `fastgs_photometric`, `densification_interval=100` | 26.9287 | 0.8241 | 0.1964 | 165.89s | 412,078 | 123M | cadence control |
| `output/0001/vfm_cached_edge_compact_bicycle_30k_r8` | `cached_edge_l1` / `npz_uint8` | 26.8864 | 0.8229 | 0.1972 | 159.77s | 408,925 | 122M | edge proxy |
| `output/0001/vfm_dinov2_token_edge_bicycle_30k_r8` | `dinov2_token_edge_l1` | 27.0577 | 0.8345 | 0.1767 | 166.11s | 490,832 | 142M | 当前 DINO token-edge 最强完整 run |
| `output/0001/vfm_dinov2_descriptor_t035_bicycle_30k_r8` | `dinov2_descriptor_cosine`, `vfm_loss_thresh=0.35` | 26.9770 | 0.8298 | 0.1850 | 190.59s | 461,846 | 135M | 完整 descriptor 语义特征路径 |

解读：

- descriptor 30k 完整训练通过 train、render 和 metrics，最终写出 `iteration_30000`，点云 PLY 约 110M；包含 25 个 test renders 后目录约 135M。
- 相比原始 baseline，descriptor 提升 +0.2738 PSNR、+0.0231 SSIM、-0.0428 LPIPS，但 Gaussian 数量增加约 92.1%。这说明它有正向信号，但不能作为严格同预算结论。
- 相比更公平的 `fastgs_densify100` cadence control，descriptor 提升 +0.0483 PSNR、+0.0057 SSIM、-0.0114 LPIPS，同时多 49,768 个 Gaussians，训练多约 24.70s。
- 相比 `cached_edge_l1`，descriptor 三项指标更好，但训练更慢、点数更多。它比 edge proxy 更贴近 proposal 语义特征误差，但首版并不更省预算。
- 相比默认 DINO token-edge，descriptor 少 28,986 个 Gaussians，却低 -0.0807 PSNR、-0.0047 SSIM、LPIPS 差 +0.0083，并且训练更慢。当前实现中，在线 DINO descriptor 还没有超过更简单的 token-edge projection。
- 结论是：`dinov2_descriptor_cosine` 已完成第一版真实 descriptor 路径验证，并在完整训练中优于 cadence/no-effect 控制；但它暂时不应取代 `cached_edge_l1` v1 正向控制组，也不应取代 `dinov2_token_edge_l1` 作为 DINO 完整训练上界。下一步若继续 descriptor，应优先做 410k 左右的 staged budget 对齐和 scorer 设计改进，而不是继续扩大短跑阈值网格。

## 2026-05-07 DINOv2 Descriptor Staged 预算对齐

数据集和 schedule 与上一节一致。这组使用 `target_gaussian_count=410000`、`target_gaussian_staged=true`、`target_gaussian_stage_margin=1.05` 和 `target_gaussian_stage_interval=500`。staged cap 为 430,500 个 Gaussians；该 target 约等于 `fastgs_densify100` 的 412,078 个点，用于检查 descriptor 相对 cadence control 的收益能否在接近预算时保留。

| 产物 | Scorer / Backend | PSNR | SSIM | LPIPS | 训练时间 | Gaussian 数量 | 输出大小 | Staged 裁剪 | 最终裁剪 | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `output/0001/fastgs_densify100_bicycle_30k_r8` | `fastgs_photometric`, `densification_interval=100` | 26.9287 | 0.8241 | 0.1964 | 165.89s | 412,078 | 123M | n/a | n/a | cadence control |
| `output/0001/vfm_dinov2_descriptor_t035_bicycle_30k_r8` | `dinov2_descriptor_cosine`, no target | 26.9770 | 0.8298 | 0.1850 | 190.59s | 461,846 | 135M | 0 | 跳过 | 完整 descriptor 结果 |
| `output/0001/vfm_dinov2_descriptor_budget410000_staged105_bicycle_30k_r8` | `dinov2_descriptor_cosine`, staged target | 26.9064 | 0.8208 | 0.2021 | 161.73s | 381,726 | 116M | 21 | 跳过，381,726 <= 410,000 | 预算对齐后低于 cadence control |

解读：

- 该 run 从 iteration 4500 到 14500 共触发 21 次 staged pruning，把中期点数压到 430,500 cap 附近；训练结束时自然低于 410,000，因此 final target prune 跳过。
- 相比 unpruned descriptor，staged 版本减少 80,120 个 Gaussians，训练少 28.86s，目录小 19M，但质量下降 -0.0706 PSNR、-0.0089 SSIM、LPIPS 差 +0.0171。
- 相比 `fastgs_densify100` cadence control，staged descriptor 少 30,352 个 Gaussians、训练少 4.16s，但指标也更低：-0.0223 PSNR、-0.0033 SSIM、LPIPS 差 +0.0057。
- 因此，descriptor 在 unpruned 30k 中相对 cadence control 的小幅收益没有经受住 staged budget 对齐。当前瓶颈更像 scorer mask/aggregation 设计，而不是继续调 target count。
- 下一步不应继续在同一个 `dinov2_descriptor_cosine` 阈值形态上做更多短跑或单点 target；更有价值的是改 descriptor error map 的构造方式，例如 percentile/top-k token mask、token-grid smoothing、多视角聚合，或把 descriptor 信号只用于 pruning 而不直接驱动 densification。

## 2026-05-07 DINOv2 Descriptor `rgb_only` 保守接入

数据集和 schedule 与 descriptor 30k 完整训练一致。这组只覆盖 `vfm_importance_mode=rgb_only`，即 descriptor 仍以 `vfm_weight=0.25` 参与 pruning-score fusion，但 densification importance 回到 RGB/FastGS importance counts。它用于检查 unpruned descriptor 的收益是否来自直接 VFM densification，还是来自 pruning/support score 的保守重排。

| 产物 | Scorer / Backend | PSNR | SSIM | LPIPS | 训练时间 | Gaussian 数量 | 输出大小 | 备注 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `output/0001/fastgs_densify100_bicycle_30k_r8` | `fastgs_photometric`, `densification_interval=100` | 26.9287 | 0.8241 | 0.1964 | 165.89s | 412,078 | 123M | cadence control |
| `output/0001/vfm_dinov2_descriptor_t035_bicycle_30k_r8` | `dinov2_descriptor_cosine`, `max` importance | 26.9770 | 0.8298 | 0.1850 | 190.59s | 461,846 | 135M | 完整 descriptor 结果 |
| `output/0001/vfm_dinov2_descriptor_budget410000_staged105_bicycle_30k_r8` | `dinov2_descriptor_cosine`, staged target | 26.9064 | 0.8208 | 0.2021 | 161.73s | 381,726 | 116M | 预算对齐负例 |
| `output/0001/vfm_dinov2_descriptor_rgb_only_bicycle_30k_r8` | `dinov2_descriptor_cosine`, `rgb_only` | 26.9370 | 0.8239 | 0.1972 | 191.54s | 407,201 | 122M | 禁用直接 descriptor densification |

解读：

- `rgb_only` run 完成 train、render 和 metrics，最终写出 407,201 个 Gaussians；`iteration_30000` 点云约 97M，包含 test renders 后目录约 122M。
- 相比 `fastgs_densify100` cadence control，descriptor `rgb_only` 少 4,877 个 Gaussians，PSNR 高 +0.0083，但 SSIM 低 -0.0002，LPIPS 差 +0.0008，训练多 25.65s。整体只能视为基本持平或轻微混合结果，不能作为清晰正向。
- 相比默认 descriptor，它少 54,645 个 Gaussians，但指标退回 -0.0400 PSNR、-0.0059 SSIM、LPIPS 差 +0.0122。这说明直接 descriptor densification 确实贡献了 unpruned 结果中的一部分质量，也带来了额外点数。
- 相比 staged 410k 预算对齐，`rgb_only` 多 25,475 个 Gaussians，PSNR 高 +0.0306、SSIM 高 +0.0030、LPIPS 好 -0.0050。它比 staged hard cap 温和，但仍没有超过 cadence control 的 LPIPS/SSIM。
- 结论是：保守接入能避免 descriptor 点数膨胀到 460k，但也几乎交回了 descriptor 的质量优势。下一步应优先改 `pixel_error_map -> metric_map` 的 mask/aggregation，而不是继续只调接入强度。

## 2026-05-07 Descriptor Top-k / Smoothing 集成验证

代码变更：增加 `vfm_metric_map_mode=threshold|percentile|topk|soft_topk`，并保留 `threshold` 为默认行为。新增 `vfm_metric_topk`、`vfm_metric_percentile`、`vfm_metric_soft_levels` 和 `vfm_descriptor_token_smooth_kernel`，用于把 descriptor cosine error 先在 DINO token grid 上平滑，再选择固定比例、分位数或多层 top-k 高误差区域。`soft_topk` 仍复用现有整数 CUDA 计数接口：它生成多层嵌套二值 masks，并按层权重累加 per-Gaussian counts。

新增配置：`configs/experiments/0001_vfm_topology_dinov2_descriptor_topk.yaml`，默认使用 `vfm_metric_map_mode=topk`、`vfm_metric_topk=0.15`、`vfm_descriptor_token_smooth_kernel=3`。另新增 `configs/experiments/0001_vfm_topology_dinov2_descriptor_soft_topk.yaml`，默认使用 `vfm_metric_map_mode=soft_topk`、`vfm_metric_topk=0.15`、`vfm_metric_soft_levels=3`。`configs/experiments/0001_vfm_topology_dinov2_descriptor_percentile.yaml` 使用 `vfm_metric_map_mode=percentile`、`vfm_metric_percentile=0.90` 和同样的 token-grid smoothing。

| 产物 | Scorer / Backend | Iterations | Metric map | PSNR | SSIM | LPIPS | 训练时间 | Gaussian 数量 | 输出大小 | 备注 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| `output/0001/vfm_dinov2_descriptor_topk015_smooth3_bicycle_120_r8` | `dinov2_descriptor_cosine` | 120 | top-k 15%, token smooth 3 | 19.3201 | 0.3804 | 0.6716 | 2.00s | 58,605 | 33M | 触发一次 descriptor scoring 和 densification |
| `output/0001/vfm_dinov2_descriptor_topk015_smooth3_bicycle_30k_r8` | `dinov2_descriptor_cosine` | 30,000 | top-k 15%, token smooth 3 | 27.0274 | 0.8330 | 0.1805 | 191.30s | 484,229 | 140M | 完整 30k 对照，接近 DINO token-edge |
| `output/0001/vfm_dinov2_descriptor_topk008_smooth3_bicycle_30k_r8` | `dinov2_descriptor_cosine` | 30,000 | top-k 8%, token smooth 3 | 26.9931 | 0.8301 | 0.1849 | 150.34s | 456,567 | 133M | 降低 top-k 比例，质量接近默认 descriptor |
| `output/0001/vfm_dinov2_descriptor_topk015_smooth3_budget410000_staged105_bicycle_30k_r8` | `dinov2_descriptor_cosine` | 30,000 | top-k 15%, token smooth 3, staged 410k | 26.9047 | 0.8219 | 0.1998 | 160.53s | 389,250 | 117M | 预算对齐后低于 cadence control |
| `output/0001/vfm_dinov2_descriptor_topk015_smooth3_budget410000_staged105_denseft4096_anyprune_s1_lr025_bicycle_30k_r8` | `dinov2_descriptor_cosine` | 34,096 | top-k 15%, token smooth 3, staged 410k, dense recovery | 26.8472 | 0.8223 | 0.1974 | 178.91s | 387,109 | 209M | `any_prune` 触发 4,096 步 dense recovery，仍低于 cadence control |
| `output/0001/vfm_dinov2_descriptor_topk008_smooth3_budget410000_staged105_bicycle_30k_r8` | `dinov2_descriptor_cosine` | 30,000 | top-k 8%, token smooth 3, staged 410k | 26.8783 | 0.8208 | 0.2013 | 165.01s | 382,035 | 116M | 更低 top-k 的预算对齐负例 |
| `output/0001/vfm_dinov2_descriptor_topk015_smooth3_rgb_only_bicycle_30k_r8` | `dinov2_descriptor_cosine` | 30,000 | top-k 15%, token smooth 3, `rgb_only` | 26.9117 | 0.8237 | 0.1977 | 154.16s | 412,317 | 123M | 只保留 descriptor support/pruning，仍低于 cadence control |
| `output/0001/vfm_dinov2_descriptor_percentile090_smooth3_bicycle_30k_r8` | `dinov2_descriptor_cosine` | 30,000 | percentile 90%, token smooth 3 | 27.0036 | 0.8313 | 0.1827 | 206.60s | 464,425 | 135M | 分位点表达完整对照，质量介于 top-k 15% 和 top-k 8% 之间 |
| `output/0001/vfm_dinov2_descriptor_soft_topk015_l3_smooth3_bicycle_620_r8` | `dinov2_descriptor_cosine` | 620 | soft top-k 15%, 3 levels, token smooth 3 | 20.8610 | 0.4747 | 0.5481 | 4.47s | 61,344 | 35M | 触发真实 descriptor scoring 和多层计数 |
| `output/0001/vfm_dinov2_descriptor_soft_topk015_l3_smooth3_bicycle_30k_r8` | `dinov2_descriptor_cosine` | 30,000 | soft top-k 15%, 3 levels, token smooth 3 | 26.9875 | 0.8305 | 0.1844 | 212.26s | 462,696 | 135M | 完整 30k 对照，质量正向但成本偏高 |
| `output/0001/vfm_dinov2_descriptor_soft_topk015_l3_smooth3_budget410000_staged105_bicycle_30k_r8` | `dinov2_descriptor_cosine` | 30,000 | soft top-k 15%, 3 levels, token smooth 3, staged 410k | 26.8848 | 0.8201 | 0.2029 | 231.87s | 383,528 | 116M | 预算对齐负例 |

解读：

- 120-step train、render 和 metrics 均通过。训练在 iteration 100 触发一次 DINO descriptor scorer，Gaussian 数量从 54,275 增到 58,605，说明新 top-k/smoothing 分支已进入真实 densification 计数链路。
- 该结果只作为集成健康检查，不作为质量选择依据。前面完整 30k 结果已经说明短跑指标对最终质量指导能力弱；这组的价值是确认下一版 30k 实验入口可用。
- soft top-k 620-step train、render 和 metrics 均通过。训练在 iteration 600 触发一次 DINO descriptor scorer 和 3 层嵌套 top-k 计数，Gaussian 数量从 54,275 增到 61,344，说明多层计数路径已进入真实 densification 链路。
- 620-step soft top-k 指标只说明评估链路健康，不作为质量结论。它的下一步判断门槛仍是 bicycle 30k `-r 8`，并且要优先与 top-k 15%、top-k 8% 和 `fastgs_densify100` cadence control 对比。
- soft top-k 30k 完成 train、render 和 metrics，`iteration_30000` 点云约 110M，包含 test renders 后目录约 135M。
- 相比 top-k 15% 完整对照，soft top-k 少 21,533 个 Gaussians，但质量下降 -0.0399 PSNR、-0.0024 SSIM、LPIPS 差 +0.0039，训练反而多 20.97s。多层计数没有超过单层 top-k 15%。
- 相比 top-k 8% 完整对照，soft top-k 多 6,129 个 Gaussians、训练多 61.93s，PSNR 低 -0.0056，SSIM 高 +0.0004，LPIPS 好 -0.0006。它基本落在 top-k 8% 附近，但计算成本明显更高。
- 相比 `fastgs_densify100` cadence control，soft top-k 提升 +0.0589 PSNR、+0.0064 SSIM、LPIPS 好 -0.0120，但多 50,618 个 Gaussians，训练多 46.37s。它是质量正向结果，不是预算高效结果。
- 相比默认 DINO token-edge，soft top-k 少 28,136 个 Gaussians，但 PSNR 低 -0.0702、SSIM 低 -0.0040、LPIPS 差 +0.0077。soft top-k 没有改变 descriptor 仍低于 token-edge 上界的判断。
- soft top-k staged 410k 完成 train、render 和 metrics；从 iteration 5000 到 14500 共触发 20 次 staged pruning，把中期点数压到 430,500 cap，训练结束时自然低于 410,000，因此 final target prune 跳过。`iteration_30000` 点云约 91M，包含 test renders 后目录约 116M。
- 相比 soft top-k 完整对照，staged 版本减少 79,168 个 Gaussians，但质量下降 -0.1027 PSNR、-0.0104 SSIM、LPIPS 差 +0.0186，训练反而多 19.60s。说明 soft top-k 的完整训练收益同样没有经受住 staged 预算约束。
- 相比 top-k 15% staged，soft top-k staged 少 5,722 个 Gaussians，但质量更低：-0.0199 PSNR、-0.0018 SSIM、LPIPS 差 +0.0031，训练多 71.34s。多层 soft 计数没有改善 staged 预算结果。
- 相比 top-k 8% staged，soft top-k staged 多 1,493 个 Gaussians，PSNR 高 +0.0065，但 SSIM 低 -0.0007、LPIPS 差 +0.0016，训练多 66.86s。它只是一个更慢、质量近似的 staged 负例。
- 相比 `fastgs_densify100` cadence control，soft top-k staged 少 28,550 个 Gaussians，但指标低 -0.0439 PSNR、-0.0040 SSIM、LPIPS 差 +0.0065，训练多 65.98s。因此它不是预算高效方案。
- percentile 90% 完整对照完成 train、render 和 metrics，最终写出 464,425 个 Gaussians；`iteration_30000` 点云约 110M，包含 test renders 后目录约 135M。
- 当前 percentile 实现按每张归一化 error map 的第 90 分位点取 `>` 阈值，本质上仍接近固定比例 high-error mask。它验证了配置和代码路径，但不构成新的预算控制机制。
- 相比 top-k 15%，percentile 90% 少 19,804 个 Gaussians，但质量低 -0.0238 PSNR、-0.0017 SSIM、LPIPS 差 +0.0022，训练多 15.30s；它没有超过 top-k 15% 的完整上界。
- 相比 top-k 8%，percentile 90% 多 7,858 个 Gaussians，质量高 +0.0105 PSNR、+0.0012 SSIM、LPIPS 好 -0.0023，但训练多 56.27s。它基本落在 top-k 8% 和 top-k 15% 之间。
- 相比 `fastgs_densify100` cadence control，percentile 90% 提升 +0.0749 PSNR、+0.0072 SSIM、LPIPS 好 -0.0137，但多 52,347 个 Gaussians，训练多 40.71s。它是质量正向结果，不是预算高效结果。
- 相比 soft top-k 30k，percentile 90% 多 1,729 个 Gaussians，但 PSNR 高 +0.0161、SSIM 高 +0.0007、LPIPS 好 -0.0017，训练少 5.66s。因此如果只看 unpruned descriptor，percentile 90% 优于 soft top-k，但没有改变预算结论。
- top-k/smoothing 30k 完成 train、render 和 metrics，`iteration_30000` 点云约 115M，包含 test renders 后目录约 140M。
- 相比默认 descriptor，top-k/smoothing 提升 +0.0504 PSNR、+0.0032 SSIM、LPIPS 好 -0.0045，但多 22,383 个 Gaussians，训练时间基本持平。这说明 mask/aggregation 改动确实改善了 descriptor scorer 质量。
- 相比 `fastgs_densify100` cadence control，top-k/smoothing 提升 +0.0987 PSNR、+0.0089 SSIM、LPIPS 好 -0.0159，但多 72,151 个 Gaussians，训练多 25.41s。它是 descriptor 方向目前最强完整结果，但不是预算受控结果。
- 相比 DINO token-edge，top-k/smoothing 少 6,603 个 Gaussians，PSNR 低 -0.0303、SSIM 低 -0.0015、LPIPS 差 +0.0038。它已经接近 token-edge 上界，但仍未超过。
- 相比 descriptor `rgb_only`，top-k/smoothing 三项指标明显更好，但点数多 77,028。说明这版 top-k mask 主要通过更强 densification 换取质量，下一步必须做 staged budget 对齐，不能直接把它作为预算高效结论。
- top-k 8% 完成 train、render 和 metrics，`iteration_30000` 点云约 108M，包含 test renders 后目录约 133M。
- 相比 top-k 15%，top-k 8% 减少 27,662 个 Gaussians，训练少 40.96s，但质量下降 -0.0343 PSNR、-0.0029 SSIM、LPIPS 差 +0.0045。这说明降低 top-k 比例确实能缓解过密重建，但也会交回一部分 top-k 15% 的质量收益。
- 相比默认 descriptor，top-k 8% 少 5,279 个 Gaussians，训练少 40.25s，同时 PSNR 高 +0.0161、SSIM 高 +0.0003、LPIPS 基本持平。这是 descriptor mask 方向里更好的完整对照点。
- 相比 `fastgs_densify100` cadence control，top-k 8% 仍多 44,489 个 Gaussians，但指标更好：+0.0644 PSNR、+0.0060 SSIM、LPIPS 好 -0.0115。它是质量正向但预算未贴合的结果。
- top-k/smoothing staged 410k 完成 train、render 和 metrics；从 iteration 4500 到 14500 共触发 21 次 staged pruning，把中期点数压到 430,500 cap，训练结束时自然低于 410,000，因此 final target prune 跳过。`iteration_30000` 点云约 93M，包含 test renders 后目录约 117M。
- 相比 unpruned top-k/smoothing，staged 版本减少 94,979 个 Gaussians，训练少 30.77s，但质量下降 -0.1227 PSNR、-0.0111 SSIM、LPIPS 差 +0.0193。
- 相比 `fastgs_densify100` cadence control，staged top-k/smoothing 少 22,828 个 Gaussians、训练少 5.36s，但指标也更低：-0.0240 PSNR、-0.0022 SSIM、LPIPS 差 +0.0034。
- 相比默认 descriptor staged 410k，top-k/smoothing staged 多 7,524 个 Gaussians，PSNR 低 -0.0017，SSIM 高 +0.0011，LPIPS 好 -0.0023。它只是在 descriptor staged 负例上小幅改善感知指标，没有改变预算对齐结论。
- top-k/smoothing staged + dense recovery 完成 train、render 和 metrics。训练中仍按 410k target 与 1.05 margin 做 staged pruning，训练结束后因为 `post_prune_finetune_trigger=any_prune` 触发 4,096 步恢复，并保存 `ours_34096`；最终 Gaussian 数量为 387,109。
- 相比无恢复的 top-k/smoothing staged 410k，dense recovery 的 PSNR 低 -0.0575，SSIM 高 +0.0004，LPIPS 好 -0.0024。它对感知指标有轻微修复，但没有恢复 photometric 指标。
- 相比 `fastgs_densify100` cadence control，dense recovery 版本少 24,969 个 Gaussians，但仍低 -0.0815 PSNR、-0.0018 SSIM，LPIPS 差 +0.0010；训练多 13.02s。
- 因此，训练结束后追加一次 dense recovery 不是 descriptor 预算高效化的解法。它更像是 LPIPS 局部修补，而不是能把 staged pruning 损伤整体恢复的机制。
- top-k 8% staged 410k 完成 train、render 和 metrics；最终点数为 382,035，低于 410,000 target，因此 final target prune 跳过。`iteration_30000` 点云约 91M，包含 test renders 后目录约 116M。
- 相比 top-k 8% 完整对照，staged 版本减少 74,532 个 Gaussians，但质量下降 -0.1149 PSNR、-0.0093 SSIM、LPIPS 差 +0.0164，训练反而多 14.67s。说明在这一路径上，中期 staged pruning 比最终点数本身更伤质量。
- 相比 top-k 15% staged，top-k 8% staged 少 7,215 个 Gaussians，但质量更低：-0.0264 PSNR、-0.0011 SSIM、LPIPS 差 +0.0016。降低 top-k ratio 没有改善预算对齐结果。
- 相比 `fastgs_densify100` cadence control，top-k 8% staged 少 30,043 个 Gaussians、训练少 0.88s，但指标低 -0.0504 PSNR、-0.0033 SSIM、LPIPS 差 +0.0049。因此它是明确的预算对齐负例。
- top-k/smoothing `rgb_only` 完成 train、render 和 metrics；它保留 descriptor top-k mask 对 pruning/support 的影响，但 densification importance 使用 RGB/FastGS 计数。最终点数为 412,317，`iteration_30000` 点云约 98M，包含 test renders 后目录约 123M。
- 相比 `fastgs_densify100` cadence control，top-k/smoothing `rgb_only` 多 239 个 Gaussians，训练少 11.73s，但质量更低：-0.0170 PSNR、-0.0004 SSIM、LPIPS 差 +0.0014。它是一个预算贴合的负例。
- 相比普通 descriptor `rgb_only`，top-k/smoothing `rgb_only` 多 5,116 个 Gaussians、训练少 37.38s，但质量略低：-0.0253 PSNR、-0.0002 SSIM、LPIPS 差 +0.0006。说明 top-k mask 单独用于 support/pruning 时没有保住质量收益。
- 结论是：top-k/smoothing 能改善 unpruned descriptor 质量；top-k 8% 与 percentile 90% 都比默认 descriptor 更均衡，但仍高于 cadence control 预算。soft top-k 30k 质量正向，但基本落在 top-k 8% 附近且成本更高。在接近 410k budget、降低 top-k ratio、关闭直接 descriptor densification、percentile mask、soft top-k 多层计数或训练结束后 dense recovery 后，收益都没有保住。0001 的 descriptor 分支应收束为“真实语义 descriptor 路径已打通，但当前预算机制未转正”；下一版需要改变预算行为本身，而不是继续增加同类 mask ratio 变体。

## 2026-05-09 大分辨率 ViT-L/14 token-edge 链路探测

目标：按 FastGS 原代码的原图裁切规则验证大分辨率训练链路，即训练使用 `-i images -r -1`，当宽度超过 1.6K 时自动缩到 1.6K。本轮优先尝试更大的 DINOv2 ViT-L/14；DINO 只在离线 cache 构建阶段运行，训练阶段读取 token-edge cache，不在每次 densification 中重新前向 ViT-L。

代码变更：`build_vfm_cache` 和 `vfm_topology_scorer` 已支持 `dinov2_vitl14`；新增 `--project_token_edge`，用于把 DINO patch tokens 直接投影成 `dinov2_token_edge` 2D cache。这样保持训练端消费的 topology signal 不变，但避免全量保存 1.6K patch-token 特征。`dinov2_token_edge_l1` 现在同时接受 `feature=dinov2_patchtokens` 和 `feature=dinov2_token_edge` 两种 cache manifest。

| 项目 | 结果 |
|---|---|
| cache 后端 | `dinov2_vitl14` |
| cache 分辨率 | `--max_width 1600` |
| cache 特征 | `dinov2_token_edge` |
| cache 存储 | `npz_uint8` |
| bicycle entries | 194 |
| cache 大小 | 1.9M |
| 首个 entry shape | `75x114` |
| 校验 | 通过 |

短训练探测使用 `configs/experiments/0001_vfm_topology_dinov2_token_edge_weighted_i050.yaml`，覆盖 `-i images -r -1` 和上述 ViT-L token-edge cache。日志确认触发 FastGS 原始裁切提示：`Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.` 620 steps 训练完成，最终 61,265 个 Gaussians，训练时间 3.15s，未出现 OOM。

正式 30k bicycle 大分辨率探针已经完成。训练使用 `-i images -r -1`，日志确认沿用 FastGS 原始 1.6K 自动缩放规则；cache 使用 ViT-L/14 在 1.6K 下投影出的 token-edge 2D 图，训练阶段不再运行 DINO。

| 场景 | 方法 | 分辨率规则 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---|---:|---:|---:|---:|---:|
| bicycle | ViT-L token-edge weighted i0.50 | `-r -1`，自动缩到 1.6K | 25.0785 | 0.7394 | 0.2733 | 1,033,601 | 187.28s |

结论：

- 30k 全程未出现 OOM。训练期显存观测约 7.2GB，说明在当前 24GB 4090 D 上，ViT-L token-edge 离线 cache + 1.6K FastGS 训练有充足余量。
- 最终 Gaussian 数量为 1.0336M，低于用户提供的 FastGS `densify100` 原图裁切参考约 1.15M。至少在 bicycle 上，这个 VFM_GS 大分辨率版本没有靠超过参考点数来换取结果。
- 本结果只能说明 bicycle 链路和资源可行。正式有效性判断需要按 MipNeRF360、DB、Tandt 三个数据集分别统计平均值，并与用户手头的 FastGS 原始数据在相同 1.6K 裁切口径下比较。
- 下一步已准备全场景批量入口：继续使用同一功能模块 `vfm_topology_scorer + dinov2_token_edge_l1 + weighted importance i0.50`，只改变数据集与场景。全场景汇总必须分开报告 MipNeRF360、DB、Tandt 平均值，不再合并成 13 场景总平均作为主结论。

## 2026-05-09 大分辨率 ViT-L/14 全场景评估

评估范围：`datasets/mipnerf360` 全 9 场景、`datasets/tandt_db/db` 全 2 场景、`datasets/tandt_db/tandt` 全 2 场景。统一设置为 `-i images -r -1`，即训练和测试都沿用 FastGS 原始大图处理逻辑，宽度超过 1.6K 时自动缩到 1.6K。方法保持同一个功能模块：`vfm_topology_scorer + dinov2_token_edge_l1 + weighted importance i0.50`，cache 使用 `dinov2_vitl14`、`--max_width 1600`、`--project_token_edge` 和 `npz_uint8`。

完整批次耗时 5,830s，约 1h37m；输出目录 `output/0001/large_res_vitl_full` 约 2.9G，ViT-L token-edge cache 目录 `output/0001/vfm_cache_large` 约 24M。原始汇总文件分别为：

- `output/0001/large_res_vitl_full/mipnerf360/summary.csv`
- `output/0001/large_res_vitl_full/db/summary.csv`
- `output/0001/large_res_vitl_full/tandt/summary.csv`

MipNeRF360：

| 场景 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---:|---:|---:|---:|---:|
| bicycle | 25.1098 | 0.7403 | 0.2727 | 1,039,441 | 189.06s |
| bonsai | 31.3437 | 0.9403 | 0.1945 | 279,294 | 161.96s |
| counter | 28.6755 | 0.9036 | 0.2078 | 218,291 | 164.20s |
| flowers | 21.3540 | 0.5789 | 0.3710 | 794,588 | 179.81s |
| garden | 26.7939 | 0.8219 | 0.2006 | 641,564 | 175.73s |
| kitchen | 31.2784 | 0.9244 | 0.1357 | 297,933 | 169.60s |
| room | 31.6938 | 0.9208 | 0.2162 | 229,772 | 149.84s |
| stump | 26.7909 | 0.7658 | 0.2741 | 795,355 | 179.50s |
| treehill | 22.6285 | 0.6196 | 0.3827 | 935,157 | 182.99s |
| **平均** | **27.2965** | **0.8017** | **0.2506** | **581,266** | **172.52s** |

DB：

| 场景 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---:|---:|---:|---:|---:|
| drjohnson | 29.4648 | 0.9025 | 0.2537 | 425,990 | 142.73s |
| playroom | 30.5957 | 0.9126 | 0.2504 | 255,942 | 142.36s |
| **平均** | **30.0302** | **0.9076** | **0.2521** | **340,966** | **142.55s** |

Tandt：

| 场景 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---:|---:|---:|---:|---:|
| train | 21.5675 | 0.7981 | 0.2470 | 207,211 | 141.09s |
| truck | 25.3946 | 0.8749 | 0.1775 | 263,557 | 139.18s |
| **平均** | **23.4810** | **0.8365** | **0.2123** | **235,384** | **140.13s** |

解读：

- 资源层面：ViT-L/14 在 24GB 4090 D 上可用。cache 构建阶段显存峰值观察约 16.9GB，训练阶段约 7GB；全场景没有 OOM。
- 存储层面：`--project_token_edge` 使 1.6K ViT-L cache 保持很小，全 13 场景 token-edge cache 约 24M，避免了保存完整 patch-token 特征。
- 评价口径：上述结果是严格 test split 指标，测试集不参与训练；训练和测试都使用 `-r -1` 的同一 1.6K 自动缩放口径。
- 相比用户给出的 bicycle FastGS `densify100` 约 1.15M GS 参考，本方法 bicycle full-run 为 1.039M GS，点数没有超过该参考。但是否质量正向必须等待同裁切口径的 baseline 指标对齐，不能仅凭绝对 PSNR 判断。
- 大分辨率下 MipNeRF360 的点数分布差异很大：bicycle、stump、treehill 接近或超过 0.8M，室内场景多在 0.2M 到 0.3M。下一步若迁移 r8 阶段的 QCGI 或数据集策略，应优先关注高点数场景的质量-容量收益。
- DB 和 Tandt 的 LPIPS 数值整体偏高，和 r8 结果不可直接横向比较；需要用用户手头的 FastGS 原始 1.6K 裁切数据作为唯一公平 baseline。

## 2026-05-09 大分辨率 FastGS big 基线复核

目标：针对“论文中 MipNeRF360 `densify100` PSNR 约 27.90，而当前大分辨率 VFM 均值为 27.2965”这一疑点，补跑同裁切口径的纯 FastGS 基线。命令使用 `scripts/run_0001_fastgs_big_eval.py`，训练/渲染/测试均为 `-i images -r -1`，沿用 FastGS 原始大图自动缩放到 1.6K 的逻辑；基线配置使用 `fastgs_big`、`densification_interval=100`，并复用 `scripts/train_big.sh` 中的 MipNeRF360 场景级 `dense`、`grad_abs_thresh`、`highfeature_lr` 等超参。

本轮同时核对了当前大分辨率 VFM 结果的实际 `cfg_args`：`densification_interval=100`、`scorer='vfm_topology_scorer'`、`vfm_enable=True`。因此这次复核的公平主基线是 `fastgs_big/densify100`，不是 `fastgs_baseline` 的 `densification_interval=500`。

产物：

- `output/0001/large_res_fastgs_big_baseline/mipnerf360/summary.csv`
- `output/0001/large_res_fastgs_big_baseline/mipnerf360/averages.json`

逐场景对比：

| 场景 | FastGS PSNR | VFM PSNR | ΔPSNR | FastGS SSIM | VFM SSIM | ΔSSIM | FastGS LPIPS | VFM LPIPS | ΔLPIPS | FastGS GS | VFM GS | ΔGS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | 25.2532 | 25.1098 | -0.1434 | 0.7554 | 0.7403 | -0.0150 | 0.2446 | 0.2727 | +0.0281 | 1,560,079 | 1,039,441 | -520,638 |
| bonsai | 32.9863 | 31.3437 | -1.6426 | 0.9512 | 0.9403 | -0.0109 | 0.1600 | 0.1945 | +0.0345 | 842,636 | 279,294 | -563,342 |
| counter | 29.5268 | 28.6755 | -0.8514 | 0.9180 | 0.9036 | -0.0144 | 0.1763 | 0.2078 | +0.0315 | 473,200 | 218,291 | -254,909 |
| flowers | 21.6166 | 21.3540 | -0.2626 | 0.6017 | 0.5789 | -0.0229 | 0.3403 | 0.3710 | +0.0307 | 1,140,260 | 794,588 | -345,672 |
| garden | 27.6137 | 26.7939 | -0.8198 | 0.8645 | 0.8219 | -0.0426 | 0.1098 | 0.2006 | +0.0908 | 2,624,164 | 641,564 | -1,982,600 |
| kitchen | 32.2700 | 31.2784 | -0.9916 | 0.9391 | 0.9244 | -0.0147 | 0.1044 | 0.1357 | +0.0312 | 1,178,795 | 297,933 | -880,862 |
| room | 32.1323 | 31.6938 | -0.4385 | 0.9298 | 0.9208 | -0.0090 | 0.1881 | 0.2162 | +0.0281 | 570,779 | 229,772 | -341,007 |
| stump | 27.1310 | 26.7909 | -0.3401 | 0.7862 | 0.7658 | -0.0204 | 0.2406 | 0.2741 | +0.0335 | 1,062,281 | 795,355 | -266,926 |
| treehill | 22.8339 | 22.6285 | -0.2054 | 0.6318 | 0.6196 | -0.0122 | 0.3770 | 0.3827 | +0.0057 | 998,983 | 935,157 | -63,826 |
| **平均** | **27.9293** | **27.2965** | **-0.6328** | **0.8198** | **0.8017** | **-0.0180** | **0.2157** | **0.2506** | **+0.0349** | **1,161,242** | **581,266** | **-579,976** |

解读：

- 纯 FastGS big/densify100 的 MipNeRF360 大分辨率均值为 27.9293 PSNR，和用户提供的论文参考 27.90 基本一致。这说明当前仓库迁移、1.6K 自动缩放和 test split 指标链路是健康的。
- 当前大分辨率 VFM i0.50 在 9 个场景上均低于 FastGS big/densify100，平均少约 580k 个 Gaussians。它不是“少量多点换质量”的情况，而是明显的容量压缩型结果。
- VFM 大分辨率批量脚本使用了 densify100，但没有套 `train_big.sh` 中的场景级基线超参；FastGS big 基线使用了这些场景级超参。因此下一步公平修复不是继续和 `fastgs_baseline` 比，而是补一组“VFM + FastGS big 场景级超参”的大分辨率 MipNeRF360 对照。
- 在这个对照完成前，当前大分辨率 ViT-L token-edge i0.50 只能作为资源可行性和过强筛选诊断结果，不能作为 VFM_GS 大分辨率有效性结论。

## 2026-05-09 大分辨率 VFM + FastGS big 场景超参复验

目标：复核上一节发现的公平性问题。上一轮大分辨率 VFM 虽然使用 `densification_interval=100`，但没有套 `scripts/train_big.sh` 的每场景 `dense`、`grad_abs_thresh`、`highfeature_lr`、`loss_thresh` 等设置；这会让 VFM 与论文口径的 FastGS big 基线不在同一 recipe 下比较。本轮在 `scripts/run_0001_dino_weighted_eval.py` 中增加 `--use-scene-overrides`，让 VFM 训练复用与 FastGS big 相同的场景级超参。DINO cache 仍使用已有 ViT-L/14 token-edge cache，训练/渲染/测试仍为 `-i images -r -1`。

命令核心参数：

```bash
uv run --active python scripts/run_0001_dino_weighted_eval.py \
  --dataset-name mipnerf360 \
  --dataset-root datasets/mipnerf360 \
  --output-root output/0001/large_res_vitl_big_overrides/mipnerf360 \
  --scenes bicycle bonsai counter flowers garden kitchen room stump treehill \
  --train-images images \
  --cache-images images \
  --resolution -1 \
  --cache-root output/0001/vfm_cache_large \
  --cache-max-width 1600 \
  --cache-storage npz_uint8 \
  --project-token-edge \
  --dino-backend dinov2_vitl14 \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_weighted_i050.yaml \
  --method-name large_res_vitl14_i050_big_overrides \
  --run-name vfm_dinov2_vitl14_token_edge_weighted_i050_big_overrides_30k_r_auto \
  --use-scene-overrides
```

产物：

- `output/0001/large_res_vitl_big_overrides/mipnerf360/summary.csv`
- `output/0001/large_res_vitl_big_overrides/mipnerf360/averages.json`

与同口径 FastGS big 基线对比：

| 场景 | FastGS PSNR | VFM+big PSNR | ΔPSNR | FastGS SSIM | VFM+big SSIM | ΔSSIM | FastGS LPIPS | VFM+big LPIPS | ΔLPIPS | FastGS GS | VFM+big GS | ΔGS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bicycle | 25.2532 | 25.0779 | -0.1753 | 0.7554 | 0.7398 | -0.0156 | 0.2446 | 0.2728 | +0.0282 | 1,560,079 | 1,037,111 | -522,968 |
| bonsai | 32.9863 | 33.0865 | +0.1002 | 0.9512 | 0.9545 | +0.0033 | 0.1600 | 0.1554 | -0.0046 | 842,636 | 1,185,620 | +342,984 |
| counter | 29.5268 | 29.6163 | +0.0895 | 0.9180 | 0.9201 | +0.0020 | 0.1763 | 0.1705 | -0.0058 | 473,200 | 589,307 | +116,107 |
| flowers | 21.6166 | 21.6316 | +0.0150 | 0.6017 | 0.6025 | +0.0008 | 0.3403 | 0.3407 | +0.0004 | 1,140,260 | 1,183,435 | +43,175 |
| garden | 27.6137 | 27.4459 | -0.1678 | 0.8645 | 0.8628 | -0.0018 | 0.1098 | 0.1113 | +0.0016 | 2,624,164 | 2,476,925 | -147,239 |
| kitchen | 32.2700 | 32.4097 | +0.1397 | 0.9391 | 0.9397 | +0.0007 | 0.1044 | 0.1030 | -0.0014 | 1,178,795 | 1,332,720 | +153,925 |
| room | 32.1323 | 32.1015 | -0.0308 | 0.9298 | 0.9327 | +0.0029 | 0.1881 | 0.1807 | -0.0074 | 570,779 | 680,350 | +109,571 |
| stump | 27.1310 | 27.1803 | +0.0493 | 0.7862 | 0.7898 | +0.0036 | 0.2406 | 0.2321 | -0.0086 | 1,062,281 | 1,193,465 | +131,184 |
| treehill | 22.8339 | 22.8913 | +0.0574 | 0.6318 | 0.6374 | +0.0056 | 0.3770 | 0.3665 | -0.0105 | 998,983 | 1,064,340 | +65,357 |
| **平均** | **27.9293** | **27.9379** | **+0.0086** | **0.8198** | **0.8199** | **+0.0002** | **0.2157** | **0.2148** | **-0.0009** | **1,161,242** | **1,193,697** | **+32,455** |

与上一版未套场景超参的 VFM 相比，本轮平均提升 +0.6414 PSNR、+0.0182 SSIM，LPIPS 改善 -0.0358，同时平均多 612,431 个 Gaussians。这说明上一轮大分辨率 VFM 的低分主要来自 FastGS big recipe 未对齐和容量被压得过低，而不是 VFM token-edge 信号失效。

解读：

- 这是当前大分辨率 MipNeRF360 上第一组相对同口径 FastGS big/densify100 三项平均指标都正向的 VFM 结果。平均 PSNR 增益只有 +0.0086，属于很小但方向一致的提升；SSIM 和 LPIPS 也小幅正向。
- 平均 Gaussian 数量只比 FastGS big 多 32,455，约 +2.8%。这符合“允许少量正向 GS 增长”的实验原则；但 bonsai、counter、kitchen、room、stump 的单场景增量超过 0.1M，需要后续用 QCGI 或容量收益门槛做自适应约束。
- bicycle 和 garden 仍是负例。bicycle 没有额外 scene override，因此仍近似复现上一版；garden 虽然从 26.7939 恢复到 27.4459，但仍低于 FastGS big 27.6137。下一步不应继续盲目提高 ViT-L 权重，而应针对这两个负例做回退或降低 VFM 介入强度。
- 该结果把大分辨率方向重新转为“有希望但需要选择/回退”的状态：默认展示可以报告 MipNeRF360 平均小幅正向，但下一版计划必须补 DB/Tandt 同 recipe 复验，并设计不依赖 test oracle 的场景级容量/质量选择规则。

## 2026-05-09 大分辨率三数据集同 recipe 复验

目标：把 `fastgs_big/densify100` 与 `VFM + FastGS big 场景超参` 的同 recipe 对照扩展到 DB 与 Tandt，并按三个公开数据集分别报告平均值。DB 数据路径为 `datasets/tandt_db/db`，Tandt 数据路径为 `datasets/tandt_db/tandt`。两组都使用 `-i images -r -1`，继续沿用 FastGS 1.6K 自动缩放；场景级超参复用 `scripts/train_big.sh` 并适配当前数据路径。

产物：

- `output/0001/large_res_fastgs_big_baseline/db/summary.csv`
- `output/0001/large_res_fastgs_big_baseline/tandt/summary.csv`
- `output/0001/large_res_vitl_big_overrides/db/summary.csv`
- `output/0001/large_res_vitl_big_overrides/tandt/summary.csv`

分数据集平均对照：

| 数据集 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MipNeRF360 | FastGS big | 27.9293 | 0.8198 | 0.2157 | 1,161,242 | 236.23s | - | - | - | - |
| MipNeRF360 | VFM+big | 27.9379 | 0.8199 | 0.2148 | 1,193,697 | 255.38s | +0.0086 | +0.0002 | -0.0009 | +32,455 |
| DB | FastGS big | 30.2073 | 0.9112 | 0.2402 | 650,194 | 167.36s | - | - | - | - |
| DB | VFM+big | 30.2236 | 0.9108 | 0.2362 | 740,387 | 197.25s | +0.0163 | -0.0004 | -0.0040 | +90,192 |
| Tandt | FastGS big | 24.3557 | 0.8573 | 0.1745 | 540,578 | 171.16s | - | - | - | - |
| Tandt | VFM+big | 24.3691 | 0.8569 | 0.1739 | 517,316 | 179.29s | +0.0134 | -0.0004 | -0.0006 | -23,262 |

DB 逐场景：

| 场景 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGS |
|---|---:|---:|---:|---:|
| drjohnson | +0.0062 | -0.0002 | -0.0040 | +134,438 |
| playroom | +0.0264 | -0.0007 | -0.0040 | +45,947 |

Tandt 逐场景：

| 场景 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGS |
|---|---:|---:|---:|---:|
| train | +0.1533 | +0.0004 | +0.0010 | -30,542 |
| truck | -0.1264 | -0.0011 | -0.0022 | -15,981 |

解读：

- 三个公开数据集分开看，VFM+big 的平均 PSNR 均小幅高于同 recipe FastGS big，LPIPS 也均小幅改善；SSIM 在 DB 和 Tandt 上略低。它比上一轮未对齐 recipe 的大分辨率结论稳定得多。
- 平均 Gaussian 数量没有出现 0.1M 级别以上的失控增长：MipNeRF360 +32,455，DB +90,192，Tandt -23,262。但单场景仍需要约束，例如 MipNeRF360 bonsai +342,984、kitchen +153,925、stump +131,184，DB drjohnson +134,438。
- Tandt 仍是混合数据集：`train` 的 PSNR 正向但 LPIPS 略差，`truck` 的 PSNR/SSIM 负向但 LPIPS 正向。该数据集不能用单一场景判断，应继续按数据集平均和质量-容量收益共同决策。
- 当前最合理的阶段性结论是：大分辨率 VFM_GS 在同 recipe 下已经具备跨 MipNeRF360、DB、Tandt 的平均弱正向证据，但提升幅度很小；下一版重点不是继续扩大 DINO，而是把容量收益门槛、场景回退和不泄漏 test 的选择规则工程化。

## 2026-05-09 DINO descriptor densify-only 复制先验探测

目标：把 VFM_GS 的下一步实验从“工程回退”转向“证明 VFM 先验能提升质量”。本轮只让 DINO descriptor residual 介入 densification，不改变 pruning score，从而隔离验证“语义结构先验能否指导新增 GS 的位置”。配置为 `configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only.yaml`：`dinov2_descriptor_cosine`、top-k 15%、token smoothing 3、`vfm_weight=0.0`、`vfm_importance_mode=max`。训练与对照均为 `-r 8`、30,000 iterations、`--eval`，并使用 matched `fastgs_photometric + densification_interval=100` 作为控制组。

当前批次先跑 MipNeRF360 的 `bicycle`、`garden`、`stump`、`bonsai`。四个场景覆盖户外大场景、植被/复杂结构、此前收益明显场景和室内小物体场景。原始产物在 `output/0001/descriptor_densify_only_probe/summary.csv`、`output/0001/descriptor_densify_only_probe/comparisons.csv` 和 `output/0001/descriptor_densify_only_probe/averages.json`。

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| bicycle | FastGS densify100 | 26.9221 | 0.8237 | 0.1970 | 413,084 | 129.72s |
| bicycle | DINO descriptor densify-only | 27.0211 | 0.8329 | 0.1794 | 486,197 | 163.30s |
| garden | FastGS densify100 | 28.8736 | 0.8961 | 0.1003 | 248,094 | 136.19s |
| garden | DINO descriptor densify-only | 29.0409 | 0.9005 | 0.0932 | 283,487 | 146.37s |
| stump | FastGS densify100 | 27.5457 | 0.8106 | 0.2042 | 295,290 | 134.25s |
| stump | DINO descriptor densify-only | 27.6200 | 0.8186 | 0.1900 | 401,485 | 153.74s |
| bonsai | FastGS densify100 | 32.4327 | 0.9625 | 0.0545 | 127,978 | 125.08s |
| bonsai | DINO descriptor densify-only | 32.4214 | 0.9643 | 0.0495 | 138,094 | 149.36s |
| **平均** | **FastGS densify100** | **28.9435** | **0.8732** | **0.1390** | **271,112** | **131.31s** |
| **平均** | **DINO descriptor densify-only** | **29.0259** | **0.8791** | **0.1280** | **327,316** | **153.19s** |

差值：

| 场景 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 |
|---|---:|---:|---:|---:|---:|
| bicycle | +0.0991 | +0.0092 | -0.0176 | +73,113 | +33.58s |
| garden | +0.1673 | +0.0044 | -0.0071 | +35,393 | +10.18s |
| stump | +0.0743 | +0.0080 | -0.0142 | +106,195 | +19.49s |
| bonsai | -0.0113 | +0.0017 | -0.0050 | +10,116 | +24.29s |
| **平均** | **+0.0823** | **+0.0058** | **-0.0110** | **+56,204** | **+21.88s** |

解读：

- 这是“descriptor residual 只指导复制”路线的第一组四场景正向结果。由于 `vfm_weight=0.0`，提升不能归因于 VFM pruning fusion，而更直接指向 DINO descriptor residual 对 densification 候选排序的帮助。
- 四场景平均 PSNR +0.0823、SSIM +0.0058、LPIPS 改善 -0.0110；SSIM 和 LPIPS 为 4/4 场景正向，PSNR 为 3/4 场景正向。`bonsai` 的 PSNR 轻微下降，但 SSIM 和 LPIPS 仍改善。
- 平均 Gaussian 数量增加 56,204，约 +20.7%。`bicycle`、`garden`、`bonsai` 的增量低于 0.1M 关注阈值；`stump` 多 106,195 个点，略高于阈值但三项指标改善明显，说明这里额外容量大概率落在有效结构区域。
- 该结果比“容量保护”更接近 VFM_GS 的方法贡献：DINO descriptor 不负责最终删点，只负责把新增 GS 引导到语义/结构残差高的位置。下一轮应继续做 descriptor 复制先验强度扫描，例如提高 top-k 覆盖率或改为 weighted importance，以判断质量增益和点数增长是否能进一步优化。

## 2026-05-09 DINO descriptor top-k25 质量优先档

目标：在 top-k 15% 的 descriptor densify-only 已经四场景正向后，检查更大的 descriptor 残差覆盖率是否能继续提高质量。本轮配置为 `configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025.yaml`：仍使用 `dinov2_descriptor_cosine`、token smoothing 3、`vfm_weight=0.0`、`vfm_importance_mode=max`，只把 `vfm_metric_topk` 从 0.15 提高到 0.25。对照继续使用上一轮同口径 matched FastGS densify100，不重复训练控制组。原始产物在 `output/0001/descriptor_densify_only_topk025_probe/summary.csv`、`comparisons.csv` 和 `averages.json`。

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| bicycle | FastGS densify100 | 26.9221 | 0.8237 | 0.1970 | 413,084 | 129.72s |
| bicycle | DINO descriptor top-k25 | 27.0613 | 0.8355 | 0.1741 | 518,848 | 154.23s |
| garden | FastGS densify100 | 28.8736 | 0.8961 | 0.1003 | 248,094 | 136.19s |
| garden | DINO descriptor top-k25 | 29.0934 | 0.9018 | 0.0912 | 295,276 | 151.27s |
| stump | FastGS densify100 | 27.5457 | 0.8106 | 0.2042 | 295,290 | 134.25s |
| stump | DINO descriptor top-k25 | 27.7338 | 0.8222 | 0.1852 | 440,028 | 152.83s |
| bonsai | FastGS densify100 | 32.4327 | 0.9625 | 0.0545 | 127,978 | 125.08s |
| bonsai | DINO descriptor top-k25 | 32.5997 | 0.9648 | 0.0489 | 142,867 | 154.30s |
| **平均** | **FastGS densify100** | **28.9435** | **0.8732** | **0.1390** | **271,112** | **131.31s** |
| **平均** | **DINO descriptor top-k25** | **29.1220** | **0.8811** | **0.1248** | **349,255** | **153.16s** |

相对 FastGS densify100：

| 场景 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 |
|---|---:|---:|---:|---:|---:|
| bicycle | +0.1392 | +0.0118 | -0.0229 | +105,764 | +24.51s |
| garden | +0.2197 | +0.0057 | -0.0091 | +47,182 | +15.08s |
| stump | +0.1881 | +0.0116 | -0.0190 | +144,738 | +18.58s |
| bonsai | +0.1670 | +0.0022 | -0.0056 | +14,889 | +29.22s |
| **平均** | **+0.1785** | **+0.0078** | **-0.0141** | **+78,143** | **+21.85s** |

相对 top-k 15% descriptor densify-only：

| 场景 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian |
|---|---:|---:|---:|---:|
| bicycle | +0.0401 | +0.0026 | -0.0053 | +32,651 |
| garden | +0.0524 | +0.0013 | -0.0020 | +11,789 |
| stump | +0.1138 | +0.0036 | -0.0048 | +38,543 |
| bonsai | +0.1783 | +0.0005 | -0.0006 | +4,773 |
| **平均** | **+0.0962** | **+0.0020** | **-0.0032** | **+21,939** |

解读：

- top-k25 是当前最清楚的 descriptor 质量优先档：四个场景 PSNR、SSIM、LPIPS 全部相对 FastGS densify100 正向，且相对 top-k15 也全部保持正向。
- `bonsai` 在 top-k15 中 PSNR 轻微负向，top-k25 后转为 +0.1670 PSNR，说明扩大 descriptor 残差覆盖率不只是增强复杂户外场景，也能修复小室内场景的弱点。
- 平均 Gaussian 数量比 baseline 多 78,143，仍低于 0.1M 平均关注阈值；但 `bicycle` 多 105,764、`stump` 多 144,738，已经进入需要容量收益约束的区间。这里的质量收益足以证明“多出的 GS 大多是正向的”，但下一轮必须验证 `weighted i0.50` 或类似机制能否压低高增点场景的无效容量。
- 方法贡献层面，本轮依然没有让 VFM 改 pruning score，因此更强结论仍是：DINO descriptor residual 可以独立指导 densification，使新增 GS 更集中在有语义/结构残差的位置。

## 2026-05-10 DINO descriptor weighted i0.50 预算效率探测

目标：在 top-k25 已证明质量优先档有效后，检查更保守的 RGB/VFM importance 加权平均能否显著降低 Gaussian 增量，同时保留 descriptor 复制收益。本轮配置为 `configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_weighted_i050.yaml`：仍使用 `dinov2_descriptor_cosine`、top-k 15%、token smoothing 3、`vfm_weight=0.0`，但将 `vfm_importance_mode` 改为 `weighted`，`vfm_importance_weight=0.50`。对照继续使用 matched FastGS densify100。原始产物在 `output/0001/descriptor_densify_only_weighted_i050_probe/summary.csv`、`comparisons.csv` 和 `averages.json`。

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| bicycle | FastGS densify100 | 26.9221 | 0.8237 | 0.1970 | 413,084 | 129.72s |
| bicycle | DINO descriptor weighted i0.50 | 26.9017 | 0.8256 | 0.1931 | 399,796 | 146.74s |
| garden | FastGS densify100 | 28.8736 | 0.8961 | 0.1003 | 248,094 | 136.19s |
| garden | DINO descriptor weighted i0.50 | 28.9371 | 0.8964 | 0.1002 | 251,096 | 147.62s |
| stump | FastGS densify100 | 27.5457 | 0.8106 | 0.2042 | 295,290 | 134.25s |
| stump | DINO descriptor weighted i0.50 | 27.5343 | 0.8125 | 0.2017 | 320,554 | 147.68s |
| bonsai | FastGS densify100 | 32.4327 | 0.9625 | 0.0545 | 127,978 | 125.08s |
| bonsai | DINO descriptor weighted i0.50 | 32.4330 | 0.9635 | 0.0507 | 129,269 | 151.99s |
| **平均** | **FastGS densify100** | **28.9435** | **0.8732** | **0.1390** | **271,112** | **131.31s** |
| **平均** | **DINO descriptor weighted i0.50** | **28.9515** | **0.8745** | **0.1365** | **275,179** | **148.51s** |

相对 FastGS densify100：

| 场景 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 |
|---|---:|---:|---:|---:|---:|
| bicycle | -0.0203 | +0.0019 | -0.0038 | -13,288 | +17.03s |
| garden | +0.0634 | +0.0003 | -0.0000 | +3,002 | +11.43s |
| stump | -0.0114 | +0.0019 | -0.0024 | +25,264 | +13.43s |
| bonsai | +0.0003 | +0.0010 | -0.0037 | +1,291 | +26.91s |
| **平均** | **+0.0080** | **+0.0013** | **-0.0025** | **+4,067** | **+17.20s** |

解读：

- weighted i0.50 成功控制了容量：平均只比 baseline 多 4,067 个 Gaussians，`bicycle` 甚至少 13,288 个点。相比 top-k15 的 +56,204 和 top-k25 的 +78,143，它确实是预算效率方向。
- 质量收益明显弱于 top-k15/top-k25。平均 PSNR 只提升 +0.0080，且 `bicycle`、`stump` 的 PSNR 轻微负向；SSIM 和 LPIPS 四场景全部正向，但幅度小。
- 该结果说明 `weighted i0.50 + top-k15` 过于保守，适合作为“接近 baseline 容量的低风险档”，但不能作为证明 VFM_GS 质量贡献的主结果。
- 下一步应测试 `top-k25 + weighted i0.50`：保留 top-k25 更强 descriptor 覆盖率，同时用 weighted 融合控制 `bicycle`、`stump` 的高增点。

## 2026-05-10 DINO descriptor top-k25 + weighted i0.50 容量受控档

目标：组合上一轮两个方向：用 top-k25 保留更大的 descriptor residual 覆盖率，用 `weighted + importance_weight=0.50` 抑制 `max` importance 带来的高增点。本轮配置为 `configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i050.yaml`：`dinov2_descriptor_cosine`、top-k 25%、token smoothing 3、`vfm_weight=0.0`、`vfm_importance_mode=weighted`、`vfm_importance_weight=0.50`。对照继续复用 matched FastGS densify100。原始产物在 `output/0001/descriptor_topk025_weighted_i050_probe/summary.csv`、`comparisons.csv` 和 `averages.json`。

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| bicycle | FastGS densify100 | 26.9221 | 0.8237 | 0.1970 | 413,084 | 129.72s |
| bicycle | DINO descriptor top-k25 weighted i0.50 | 26.9644 | 0.8301 | 0.1843 | 440,867 | 152.20s |
| garden | FastGS densify100 | 28.8736 | 0.8961 | 0.1003 | 248,094 | 136.19s |
| garden | DINO descriptor top-k25 weighted i0.50 | 28.9969 | 0.8989 | 0.0961 | 267,569 | 146.43s |
| stump | FastGS densify100 | 27.5457 | 0.8106 | 0.2042 | 295,290 | 134.25s |
| stump | DINO descriptor top-k25 weighted i0.50 | 27.5631 | 0.8165 | 0.1941 | 367,363 | 149.01s |
| bonsai | FastGS densify100 | 32.4327 | 0.9625 | 0.0545 | 127,978 | 125.08s |
| bonsai | DINO descriptor top-k25 weighted i0.50 | 32.4357 | 0.9635 | 0.0501 | 134,051 | 149.39s |
| **平均** | **FastGS densify100** | **28.9435** | **0.8732** | **0.1390** | **271,112** | **131.31s** |
| **平均** | **DINO descriptor top-k25 weighted i0.50** | **28.9901** | **0.8773** | **0.1312** | **302,463** | **149.26s** |

相对 FastGS densify100：

| 场景 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 |
|---|---:|---:|---:|---:|---:|
| bicycle | +0.0424 | +0.0064 | -0.0126 | +27,783 | +22.48s |
| garden | +0.1233 | +0.0028 | -0.0041 | +19,475 | +10.24s |
| stump | +0.0175 | +0.0058 | -0.0101 | +72,073 | +14.77s |
| bonsai | +0.0030 | +0.0010 | -0.0044 | +6,073 | +24.31s |
| **平均** | **+0.0465** | **+0.0040** | **-0.0078** | **+31,351** | **+17.95s** |

相对 top-k25 `max` 质量优先档：

| 场景 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian |
|---|---:|---:|---:|---:|
| bicycle | -0.0968 | -0.0054 | +0.0103 | -77,981 |
| garden | -0.0964 | -0.0029 | +0.0049 | -27,707 |
| stump | -0.1707 | -0.0057 | +0.0088 | -72,665 |
| bonsai | -0.1640 | -0.0012 | +0.0012 | -8,816 |
| **平均** | **-0.1320** | **-0.0038** | **+0.0063** | **-46,792** |

相对 top-k15 `weighted i0.50`：

| 场景 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian |
|---|---:|---:|---:|---:|
| bicycle | +0.0627 | +0.0045 | -0.0088 | +41,071 |
| garden | +0.0599 | +0.0025 | -0.0041 | +16,473 |
| stump | +0.0288 | +0.0040 | -0.0077 | +46,809 |
| bonsai | +0.0027 | +0.0000 | -0.0007 | +4,782 |
| **平均** | **+0.0385** | **+0.0028** | **-0.0053** | **+27,284** |

解读：

- `top-k25 + weighted i0.50` 四个场景相对 FastGS densify100 都是三项指标正向，说明扩大 descriptor 覆盖后，即使采用加权 importance 抑制容量增长，复制先验仍然有效。
- 它把 top-k25 `max` 的平均 Gaussian 增量从 +78,143 压到 +31,351；`bicycle` 从 +105,764 降到 +27,783，`stump` 从 +144,738 降到 +72,073。容量控制目标达成。
- 代价是质量明显低于 top-k25 `max`：平均 PSNR 少 0.1320、SSIM 少 0.0038、LPIPS 差 0.0063。因此它不是新的质量优先档，而是“容量受控正向档”。
- 相比 top-k15 `weighted i0.50`，本轮平均多 27,284 个点，但换来 +0.0385 PSNR、+0.0028 SSIM、LPIPS 改善 -0.0053，且修复了 top-k15 weighted 中 `bicycle/stump` 的 PSNR 轻微负向。
- 下一步应尝试中间权重，例如 `top-k25 + weighted i0.65` 或 i0.70。目标不是回退 FastGS，而是在不超过约 0.1M 平均增点的前提下，尽量靠近 top-k25 `max` 的质量收益。

## 2026-05-10 DINO descriptor top-k25 + weighted i0.70 质量折中档

目标：在 i0.50 容量受控档和 top-k25 `max` 质量优先档之间寻找中间点。本轮配置为 `configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i070.yaml`：保持 `dinov2_descriptor_cosine`、top-k 25%、token smoothing 3、`vfm_weight=0.0` 和 `vfm_importance_mode=weighted`，仅把 `vfm_importance_weight` 从 0.50 提高到 0.70。原始产物在 `output/0001/descriptor_topk025_weighted_i070_probe/summary.csv`、`comparisons.csv` 和 `averages.json`。

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| bicycle | FastGS densify100 | 26.9221 | 0.8237 | 0.1970 | 413,084 | 129.72s |
| bicycle | DINO descriptor top-k25 weighted i0.70 | 26.9848 | 0.8309 | 0.1840 | 442,608 | 210.12s |
| garden | FastGS densify100 | 28.8736 | 0.8961 | 0.1003 | 248,094 | 136.19s |
| garden | DINO descriptor top-k25 weighted i0.70 | 28.9733 | 0.8991 | 0.0950 | 274,374 | 196.27s |
| stump | FastGS densify100 | 27.5457 | 0.8106 | 0.2042 | 295,290 | 134.25s |
| stump | DINO descriptor top-k25 weighted i0.70 | 27.6225 | 0.8177 | 0.1928 | 374,854 | 224.10s |
| bonsai | FastGS densify100 | 32.4327 | 0.9625 | 0.0545 | 127,978 | 125.08s |
| bonsai | DINO descriptor top-k25 weighted i0.70 | 32.5470 | 0.9643 | 0.0495 | 134,654 | 218.09s |
| **平均** | **FastGS densify100** | **28.9435** | **0.8732** | **0.1390** | **271,112** | **131.31s** |
| **平均** | **DINO descriptor top-k25 weighted i0.70** | **29.0319** | **0.8780** | **0.1303** | **306,623** | **212.15s** |

相对 FastGS densify100：

| 场景 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 |
|---|---:|---:|---:|---:|---:|
| bicycle | +0.0627 | +0.0072 | -0.0130 | +29,524 | +80.40s |
| garden | +0.0997 | +0.0030 | -0.0053 | +26,280 | +60.08s |
| stump | +0.0768 | +0.0070 | -0.0114 | +79,564 | +89.85s |
| bonsai | +0.1143 | +0.0017 | -0.0049 | +6,676 | +93.01s |
| **平均** | **+0.0884** | **+0.0047** | **-0.0086** | **+35,511** | **+80.84s** |

相对 i0.50 容量受控档：

| 场景 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 |
|---|---:|---:|---:|---:|---:|
| bicycle | +0.0203 | +0.0008 | -0.0004 | +1,741 | +57.92s |
| garden | -0.0236 | +0.0002 | -0.0011 | +6,805 | +49.84s |
| stump | +0.0594 | +0.0012 | -0.0012 | +7,491 | +75.09s |
| bonsai | +0.1113 | +0.0008 | -0.0005 | +603 | +68.70s |
| **平均** | **+0.0419** | **+0.0007** | **-0.0008** | **+4,160** | **+62.89s** |

相对 top-k25 `max` 质量优先档：

| 场景 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 |
|---|---:|---:|---:|---:|---:|
| bicycle | -0.0765 | -0.0046 | +0.0099 | -76,240 | +55.89s |
| garden | -0.1200 | -0.0027 | +0.0038 | -20,902 | +45.00s |
| stump | -0.1113 | -0.0045 | +0.0076 | -65,174 | +71.27s |
| bonsai | -0.0527 | -0.0005 | +0.0007 | -8,213 | +63.79s |
| **平均** | **-0.0901** | **-0.0031** | **+0.0055** | **-42,632** | **+58.99s** |

解读：

- i0.70 四场景相对 FastGS densify100 仍全部三项指标正向，平均 PSNR +0.0884、SSIM +0.0047、LPIPS 改善 -0.0086，强于 i0.50 的 +0.0465 / +0.0040 / -0.0078。
- 容量仍受控：平均只比 FastGS 多 35,511 个 Gaussians，比 i0.50 多 4,160 个点，远低于 top-k25 `max` 的 +78,143 平均增量。`stump` 仍是最大增点场景，但 +79,564 低于 0.1M 关注阈值。
- i0.70 没有追回 top-k25 `max` 的质量上界，平均仍低 -0.0901 PSNR、SSIM -0.0031、LPIPS 差 +0.0055。因此它应定位为质量折中档。
- 主要问题是训练时间：平均训练时间比 FastGS 多 80.84s，比 i0.50 多 62.89s，甚至比 top-k25 `max` 多 58.99s。该现象需要后续确认是否来自运行方差、DINO descriptor 调用频率、或本轮机器负载；在确认前，i0.70 不应作为默认效率档。
- 下一步建议优先测试 `top-k25 + weighted i0.60` 或 i0.65，目标是保留 i0.70 的质量提升趋势，同时检查训练时间是否回落。如果训练时间仍异常，则应先做 profiling，而不是继续提高权重。

## 2026-05-10 DINO descriptor top-k25 + weighted i0.65 边界探测

目标：在 i0.50 和 i0.70 之间补一个中间权重，判断 i0.70 的质量提升是否可以用更低权重保留，同时确认训练时间异常是否回落。本轮配置为 `configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i065.yaml`：保持 `dinov2_descriptor_cosine`、top-k 25%、token smoothing 3、`vfm_weight=0.0` 和 `vfm_importance_mode=weighted`，仅将 `vfm_importance_weight` 设为 0.65。原始产物在 `output/0001/descriptor_topk025_weighted_i065_probe/summary.csv`、`comparisons.csv` 和 `averages.json`。

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| bicycle | FastGS densify100 | 26.9221 | 0.8237 | 0.1970 | 413,084 | 129.72s |
| bicycle | DINO descriptor top-k25 weighted i0.65 | 26.9874 | 0.8309 | 0.1838 | 440,747 | 215.06s |
| garden | FastGS densify100 | 28.8736 | 0.8961 | 0.1003 | 248,094 | 136.19s |
| garden | DINO descriptor top-k25 weighted i0.65 | 28.9759 | 0.8992 | 0.0951 | 271,696 | 210.07s |
| stump | FastGS densify100 | 27.5457 | 0.8106 | 0.2042 | 295,290 | 134.25s |
| stump | DINO descriptor top-k25 weighted i0.65 | 27.5256 | 0.8163 | 0.1939 | 378,158 | 218.56s |
| bonsai | FastGS densify100 | 32.4327 | 0.9625 | 0.0545 | 127,978 | 125.08s |
| bonsai | DINO descriptor top-k25 weighted i0.65 | 32.3981 | 0.9640 | 0.0499 | 134,605 | 213.47s |
| **平均** | **FastGS densify100** | **28.9435** | **0.8732** | **0.1390** | **271,112** | **131.31s** |
| **平均** | **DINO descriptor top-k25 weighted i0.65** | **28.9717** | **0.8776** | **0.1307** | **306,302** | **214.29s** |

相对 FastGS densify100：

| 场景 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 |
|---|---:|---:|---:|---:|---:|
| bicycle | +0.0653 | +0.0072 | -0.0132 | +27,663 | +85.34s |
| garden | +0.1022 | +0.0031 | -0.0051 | +23,602 | +73.87s |
| stump | -0.0201 | +0.0057 | -0.0103 | +82,868 | +84.32s |
| bonsai | -0.0346 | +0.0014 | -0.0046 | +6,627 | +88.39s |
| **平均** | **+0.0282** | **+0.0043** | **-0.0083** | **+35,190** | **+82.98s** |

相对 i0.50 容量受控档：

| 场景 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian |
|---|---:|---:|---:|---:|
| bicycle | +0.0230 | +0.0007 | -0.0005 | -120 |
| garden | -0.0210 | +0.0003 | -0.0010 | +4,127 |
| stump | -0.0375 | -0.0002 | -0.0002 | +10,795 |
| bonsai | -0.0376 | +0.0004 | -0.0002 | +554 |
| **平均** | **-0.0183** | **+0.0003** | **-0.0005** | **+3,839** |

相对 i0.70 质量折中档：

| 场景 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 |
|---|---:|---:|---:|---:|---:|
| bicycle | +0.0026 | -0.0000 | -0.0002 | -1,861 | +4.95s |
| garden | +0.0025 | +0.0000 | +0.0001 | -2,678 | +13.80s |
| stump | -0.0969 | -0.0013 | +0.0011 | +3,304 | -5.53s |
| bonsai | -0.1489 | -0.0003 | +0.0004 | -49 | -4.62s |
| **平均** | **-0.0602** | **-0.0004** | **+0.0003** | **-321** | **+2.14s** |

解读：

- i0.65 的平均指标仍相对 FastGS densify100 正向：PSNR +0.0282、SSIM +0.0043、LPIPS 改善 -0.0083，平均多 35,190 个 Gaussians。但它没有保持 i0.70 的 4/4 PSNR 正向，`stump` 和 `bonsai` 的 PSNR 都转为负向。
- 相比 i0.50，i0.65 平均只多 3,839 个点，但 PSNR 低 -0.0183；SSIM 和 LPIPS 只极小幅改善。这个权重没有形成比 i0.50 更好的质量-容量折中。
- 相比 i0.70，i0.65 平均点数几乎相同，训练时间也没有回落，反而略高 2.14s；同时 PSNR 低 -0.0602。这说明 i0.70 的时间开销不是单纯由权重大小导致，i0.65 不能作为性能修复点。
- 结论：i0.65 记录为边界负例，不推荐继续作为主线。当前 descriptor 分支的清晰结论保持不变：top-k25 `max` 是质量优先档，top-k25 weighted i0.50 是容量受控正向档，top-k25 weighted i0.70 是质量折中档但需要 profiling。下一步优先检查 descriptor scoring 的调用频率和耗时，而不是继续在相邻权重上做密集扫描。

## 2026-05-10 DINO descriptor scorer profiling 诊断

目标：解释 i0.65/i0.70 训练时间偏高的来源，并确认是否需要继续扫描相邻权重。新增默认关闭的 profiling 参数：`--vfm_profile_scorer` 和 `--vfm_profile_interval`。默认训练不启用 profiling，不会额外同步 CUDA；只有显式传参时才打印 `[VFM PROFILE]`。

先用 bicycle 620/820-step 链路确认 profiling 可用，再用 bicycle i0.70 30k 复跑采样诊断。30k 命令仍使用 `configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i070.yaml`、`-r 8`、`vfm_weight=0.0`，输出到 `output/0001/descriptor_profile_i070_bicycle_30k_r8`。

30k 复跑结果：

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| bicycle | DINO descriptor top-k25 weighted i0.70 profile | 26.9835 | 0.8314 | 0.1831 | 445,245 | 184.47s |
| bicycle | 原 i0.70 记录 | 26.9848 | 0.8309 | 0.1840 | 442,608 | 210.12s |
| bicycle | FastGS densify100 对照 | 26.9221 | 0.8237 | 0.1970 | 413,084 | 129.72s |

采样 profile：

| 调用 | 当前 GS | 总耗时 | RGB score | SH0 渲染 | DINO descriptor error | metric map | count raster |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 415,470 | 171.56ms | 63.24ms | 24.69ms | 51.90ms | 3.81ms | 26.38ms |
| 100 | 479,580 | 188.26ms | 80.85ms | 25.36ms | 50.11ms | 3.81ms | 26.60ms |

短程首轮加载诊断：

| 运行 | 调用 | 总耗时 | RGB score | DINO descriptor error | 说明 |
|---|---:|---:|---:|---:|---|
| 620-step | 1 | 2511.96ms | 85.07ms | 2367.03ms | 包含首次 DINO 模型加载，不代表稳定训练开销 |
| 820-step | 1 | 2953.07ms | 85.63ms | 2800.05ms | 同上 |
| 820-step | 2 | 218.22ms | 85.07ms | 69.31ms | 稳定调用 |
| 820-step | 3 | 218.66ms | 85.17ms | 69.92ms | 稳定调用 |

解读：

- i0.70 复跑质量稳定：PSNR 基本不变，SSIM/LPIPS 略好；Gaussian 数量比原记录多 2,637 个。它仍是质量折中档，不是偶然正例。
- 训练时间从原记录 210.12s 降到 184.47s，说明早先 210s 中存在运行波动；但它仍明显慢于 i0.50 的 149.26s 和 FastGS densify100 的 129.72s。
- 稳定 scorer 单次调用约 172-188ms，主要由两部分组成：FastGS 原始 multi-view score 约 63-81ms，在线 DINO descriptor error 约 50-52ms；其余是 SH0 渲染和 metric-map count raster。
- descriptor 30k 在 densification 阶段会多次调用 scorer，因此在线 DINO descriptor 是可解释的主要额外成本。下一步优化优先级应是降低 descriptor 参与频率或缩短参与窗口，而不是继续细扫 i0.60/i0.65/i0.70 一类相邻权重。
- 值得测试的下一版方向：`descriptor_warm_window`，只在较早 densification 阶段使用 DINO descriptor，例如 600-8000 iteration；后半段回到 RGB/FastGS score 或 token-edge 轻量 proxy。目标是在保留早期结构引导收益的同时减少 30k 额外耗时。

## 2026-05-10 DINO descriptor warm window 机制验证

目标：把 profiling 中发现的在线 DINO descriptor 开销转化为可控参数。新增 `vfm_active_from_iter` 和 `vfm_active_until_iter`，默认均为 0，表示全程启用；当当前训练 iteration 不在窗口内时，`vfm_topology_scorer` 只返回原始 FastGS score，跳过 SH0 渲染、在线 DINO descriptor、metric map 和 VFM count raster。训练循环在 densification、15k 后 pruning 和 target prune 调用 scorer 前写入 `current_iteration`，避免窗口判断依赖外部状态。

新增配置：`configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i070_warm8000.yaml`。配置保持 `dinov2_descriptor_cosine`、top-k25、token smoothing 3、`vfm_weight=0.0`、`vfm_importance_mode=weighted`、`vfm_importance_weight=0.70`，只增加 `vfm_active_until_iter=8000`。设计意图是：早期 densification 继续使用 DINO descriptor residual 引导新增 GS，后半段回到 FastGS score，验证质量保留和训练耗时下降。

短程窗口验证使用 bicycle `-r 8`、820 iterations，并临时覆盖 `--vfm_active_until_iter 650`，输出在 `output/0001/descriptor_warm650_i070_bicycle_820_r8`。该设置会让 iteration 600 的 densification 走 DINO descriptor，iteration 700/800 回到 FastGS。

| 调用 | iteration 区间 | active | 当前 GS | 总耗时 | RGB score | DINO descriptor error | 说明 |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | 600 | true | 54,275 | 2041.38ms | 79.74ms | 1905.36ms | 包含首次 DINO 模型加载 |
| 2 | 700 | false | 60,350 | 78.08ms | 78.06ms | - | 已跳过 DINO/VFM 路径 |
| 3 | 800 | false | 70,553 | 76.59ms | 76.57ms | - | 已跳过 DINO/VFM 路径 |

短程结果：820 iteration 结束时 Gaussian 数量为 85,013，训练耗时 6.69s。该结果只用于验证窗口机制，不作为质量判断。

解读：

- `active=false` 时 scorer 耗时几乎等于 FastGS 原始 multi-view score，说明窗口确实绕过了在线 DINO descriptor 和 VFM raster 计数。
- 窗口机制不改变默认行为；旧配置未设置窗口时仍全程启用 VFM。
- 下一步应跑 bicycle 30k `warm8000`，直接比较三组：FastGS densify100、原 i0.70/profile i0.70、warm8000 i0.70。若 PSNR/SSIM/LPIPS 基本保留且训练时间明显下降，再扩展到 garden/stump/bonsai 四场景。

### 30k bicycle warm8000 结果

配置：`configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i070_warm8000.yaml`，输出目录 `output/0001/descriptor_warm8000_i070_bicycle_30k_r8`。测试仍为 bicycle `-r 8`、30,000 iterations、`--eval`，与 i0.70/profile i0.70、i0.50 和 FastGS densify100 保持同一 matched 口径。

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| bicycle | FastGS densify100 | 26.9221 | 0.8237 | 0.1970 | 413,084 | 129.72s |
| bicycle | DINO descriptor top-k25 weighted i0.50 | 26.9644 | 0.8301 | 0.1843 | 440,867 | 152.20s |
| bicycle | DINO descriptor top-k25 weighted i0.70 profile | 26.9835 | 0.8314 | 0.1831 | 445,245 | 184.47s |
| bicycle | DINO descriptor top-k25 weighted i0.70 warm8000 | 26.9633 | 0.8285 | 0.1874 | 432,336 | 177.80s |

相对 FastGS densify100：

| 方法 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 |
|---|---:|---:|---:|---:|---:|
| i0.50 | +0.0423 | +0.0064 | -0.0127 | +27,783 | +22.48s |
| i0.70 profile | +0.0614 | +0.0077 | -0.0139 | +32,161 | +54.75s |
| i0.70 warm8000 | +0.0412 | +0.0048 | -0.0097 | +19,252 | +48.08s |

相对 i0.70 profile：

| 指标 | warm8000 - i0.70 profile |
|---|---:|
| PSNR | -0.0202 |
| SSIM | -0.0029 |
| LPIPS | +0.0043 |
| Gaussian 数量 | -12,909 |
| 训练时间 | -6.67s |

30k profile 采样：

| 调用 | active | 当前 GS | 总耗时 | RGB score | DINO descriptor error | 说明 |
|---:|---|---:|---:|---:|---:|---|
| 50 | true | 418,533 | 199.50ms | 78.95ms | 51.82ms | 8000 iteration 前仍使用 DINO descriptor |
| 100 | false | 477,175 | 80.28ms | 80.26ms | - | 8000 iteration 后回到 FastGS score |

解读：

- warm8000 机制有效，但质量-效率折中不优。它相对 FastGS 仍三项指标正向，且比 i0.70 少 12,909 个点；但 PSNR/SSIM/LPIPS 都低于 i0.70，也低于 i0.50。
- 训练时间只比 i0.70 profile 少 6.67s，远小于预期。原因是 8000 iteration 之前已经覆盖了主要 densification 增长阶段，且 15k 后 pruning 仍需要 FastGS 原始 score；仅截断后半段 DINO descriptor 不能充分降低总训练耗时。
- 结论：`warm8000` 记录为机制可行但实验负例，不扩展四场景。下一轮不继续缩短窗口做细扫；更应把资源放回已经正向的 descriptor top-k25 `max` 或 weighted i0.70 多场景/全场景验证，判断数据集平均收益，而不是用单场景窗口优化追求小幅省时。

## 2026-05-10 DINO descriptor top-k25 max MipNeRF360 全场景扩展

目标：验证质量优先档 `dinov2_descriptor_topk25_max` 是否只是在最初四个场景上有效，还是能在 MipNeRF360 全 9 场景平均上稳定提升。该方案保持 `vfm_weight=0.0`，只让 DINO descriptor residual 参与 densification importance，不改变 pruning score，因此是当前最干净的“VFM 指导复制提升质量”证据。

新增 5 场景使用命令：

```bash
source .venv/bin/activate && uv run --active python scripts/run_0001_descriptor_quality_probe.py \
  --dataset-name mipnerf360 \
  --dataset-root datasets/mipnerf360 \
  --output-root output/0001/descriptor_topk025_mipnerf360_full \
  --scenes counter flowers kitchen room treehill \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025.yaml \
  --descriptor-method dinov2_descriptor_topk25_max \
  --descriptor-run-name vfm_dinov2_descriptor_topk25_max_30k_r8 \
  --baseline-run-name fastgs_densify100_30k_r8 \
  --cache-root output/0001/vfm_cache \
  --cache-max-width 224 \
  --cache-storage npy_float16 \
  --dino-backend dinov2_vits14 \
  --dinov2-repo output/0001/external/dinov2 \
  --resolution 8 \
  --iterations 30000 \
  --densification-interval 100
```

合并结果输出在 `output/0001/descriptor_topk025_mipnerf360_full_merged/summary.csv`、`comparisons.csv`、`averages.json` 和 `dataset_comparison.json`。旧四场景来自 `output/0001/descriptor_densify_only_topk025_probe/summary.csv`，新五场景来自 `output/0001/descriptor_topk025_mipnerf360_full/summary.csv`。

全 9 场景结果：

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| bicycle | FastGS densify100 | 26.9221 | 0.8237 | 0.1970 | 413,084 | 129.72s |
| bicycle | DINO descriptor top-k25 max | 27.0613 | 0.8355 | 0.1741 | 518,848 | 154.23s |
| bonsai | FastGS densify100 | 32.4327 | 0.9625 | 0.0545 | 127,978 | 125.08s |
| bonsai | DINO descriptor top-k25 max | 32.5997 | 0.9648 | 0.0489 | 142,867 | 154.30s |
| counter | FastGS densify100 | 29.6104 | 0.9316 | 0.0795 | 110,956 | 128.98s |
| counter | DINO descriptor top-k25 max | 29.7090 | 0.9342 | 0.0737 | 123,800 | 150.09s |
| flowers | FastGS densify100 | 22.9911 | 0.6933 | 0.2772 | 321,358 | 143.28s |
| flowers | DINO descriptor top-k25 max | 23.0339 | 0.6982 | 0.2724 | 363,631 | 154.08s |
| garden | FastGS densify100 | 28.8736 | 0.8961 | 0.1003 | 248,094 | 136.19s |
| garden | DINO descriptor top-k25 max | 29.0934 | 0.9018 | 0.0912 | 295,276 | 151.27s |
| kitchen | FastGS densify100 | 33.3742 | 0.9692 | 0.0350 | 158,819 | 147.13s |
| kitchen | DINO descriptor top-k25 max | 33.3927 | 0.9699 | 0.0337 | 169,214 | 157.19s |
| room | FastGS densify100 | 33.0262 | 0.9619 | 0.0584 | 99,304 | 124.06s |
| room | DINO descriptor top-k25 max | 33.0628 | 0.9621 | 0.0572 | 105,528 | 165.00s |
| stump | FastGS densify100 | 27.5457 | 0.8106 | 0.2042 | 295,290 | 134.25s |
| stump | DINO descriptor top-k25 max | 27.7338 | 0.8222 | 0.1852 | 440,028 | 152.83s |
| treehill | FastGS densify100 | 24.3962 | 0.7246 | 0.2900 | 408,149 | 144.20s |
| treehill | DINO descriptor top-k25 max | 24.4452 | 0.7298 | 0.2757 | 475,021 | 182.65s |
| **平均** | **FastGS densify100** | **28.7969** | **0.8637** | **0.1440** | **242,559** | **134.77s** |
| **平均** | **DINO descriptor top-k25 max** | **28.9035** | **0.8687** | **0.1347** | **292,690** | **157.96s** |

逐场景差值：

| 场景 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 |
|---|---:|---:|---:|---:|---:|
| bicycle | +0.1392 | +0.0118 | -0.0229 | +105,764 | +24.51s |
| bonsai | +0.1670 | +0.0022 | -0.0056 | +14,889 | +29.22s |
| counter | +0.0986 | +0.0026 | -0.0058 | +12,844 | +21.11s |
| flowers | +0.0428 | +0.0049 | -0.0047 | +42,273 | +10.80s |
| garden | +0.2197 | +0.0057 | -0.0091 | +47,182 | +15.08s |
| kitchen | +0.0185 | +0.0006 | -0.0013 | +10,395 | +10.06s |
| room | +0.0366 | +0.0002 | -0.0012 | +6,224 | +40.94s |
| stump | +0.1881 | +0.0116 | -0.0190 | +144,738 | +18.58s |
| treehill | +0.0489 | +0.0052 | -0.0143 | +66,872 | +38.45s |
| **平均** | **+0.1066** | **+0.0050** | **-0.0093** | **+50,131** | **+23.20s** |

解读：

- 这是目前 descriptor densify-only 分支最强的全场景证据：MipNeRF360 9/9 场景 PSNR、SSIM、LPIPS 全部正向，数据集平均 PSNR +0.1066、SSIM +0.0050、LPIPS 改善 -0.0093。
- 由于 `vfm_weight=0.0`，该实验不改变 pruning score；收益来自 DINO descriptor residual 对新增 Gaussian 位置/区域的引导，而不是剪枝回退或测试集选择。
- 平均 Gaussian 数量增加 50,131，低于 0.1M 的平均关注阈值，说明整体容量增长仍可接受；但 `bicycle` 和 `stump` 单场景分别增加 105,764 和 144,738，后续需要容量收益门槛或自适应 importance 控制。
- 新补的 `counter/flowers/kitchen/room/treehill` 五场景全部正向，尤其 `treehill` 在过去多个分支中经常是压力场景，本轮仍获得 +0.0489 PSNR、+0.0052 SSIM、LPIPS -0.0143，说明 descriptor top-k25 max 的鲁棒性强于 warm-window 和若干 weighted 边界档。
- 代价主要是训练时间：平均 +23.20s。该成本来自在线 DINO descriptor 对渲染图与 GT cache 的比较，后续不应再用 warm8000 这类质量损失较大的窗口截断，而应优先研究更轻量的 descriptor 近似、cache/投影复用或自适应调用频率。
- 下一轮建议把 top-k25 max 作为 MipNeRF360 `-r 8` 的质量优先证据保留，同时在 DB/Tandt 做同一功能模块的跨数据集验证；若 DB/Tandt 也正向，再进入高分辨率或 Depth Anything 几何先验分支。

## 2026-05-10 DINO descriptor top-k25 max DB/Tandt 跨数据集验证

目标：沿用同一个 `dinov2_descriptor_topk25_max` 功能模块，不做数据集级回退，也不调节超参，检查 descriptor densify-only 是否能从 MipNeRF360 推广到 `datasets/tandt_db/db` 和 `datasets/tandt_db/tandt`。该方案仍保持 `vfm_weight=0.0`，因此 VFM 只影响 densification importance，不改变 pruning score。

DB 命令：

```bash
source .venv/bin/activate && uv run --active python scripts/run_0001_descriptor_quality_probe.py \
  --dataset-name db \
  --dataset-root datasets/tandt_db/db \
  --output-root output/0001/descriptor_topk025_db_full \
  --scenes drjohnson playroom \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025.yaml \
  --descriptor-method dinov2_descriptor_topk25_max \
  --descriptor-run-name vfm_dinov2_descriptor_topk25_max_30k_r8 \
  --baseline-run-name fastgs_densify100_30k_r8 \
  --cache-root output/0001/vfm_cache \
  --cache-max-width 224 \
  --cache-storage npy_float16 \
  --dino-backend dinov2_vits14 \
  --dinov2-repo output/0001/external/dinov2 \
  --resolution 8 \
  --iterations 30000 \
  --densification-interval 100
```

Tandt 命令：

```bash
source .venv/bin/activate && uv run --active python scripts/run_0001_descriptor_quality_probe.py \
  --dataset-name tandt \
  --dataset-root datasets/tandt_db/tandt \
  --output-root output/0001/descriptor_topk025_tandt_full \
  --scenes train truck \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025.yaml \
  --descriptor-method dinov2_descriptor_topk25_max \
  --descriptor-run-name vfm_dinov2_descriptor_topk25_max_30k_r8 \
  --baseline-run-name fastgs_densify100_30k_r8 \
  --cache-root output/0001/vfm_cache \
  --cache-max-width 224 \
  --cache-storage npy_float16 \
  --dino-backend dinov2_vits14 \
  --dinov2-repo output/0001/external/dinov2 \
  --resolution 8 \
  --iterations 30000 \
  --densification-interval 100
```

DB 结果：

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| drjohnson | FastGS densify100 | 30.5704 | 0.9287 | 0.0714 | 78,984 | 139.81s |
| drjohnson | DINO descriptor top-k25 max | 30.6938 | 0.9306 | 0.0690 | 92,273 | 153.76s |
| playroom | FastGS densify100 | 30.6169 | 0.9447 | 0.0548 | 47,574 | 126.62s |
| playroom | DINO descriptor top-k25 max | 30.5105 | 0.9432 | 0.0551 | 52,662 | 142.59s |
| **平均** | **FastGS densify100** | **30.5937** | **0.9367** | **0.0631** | **63,279** | **133.22s** |
| **平均** | **DINO descriptor top-k25 max** | **30.6022** | **0.9369** | **0.0620** | **72,468** | **148.18s** |

DB 逐场景差值：

| 场景 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 |
|---|---:|---:|---:|---:|---:|
| drjohnson | +0.1234 | +0.0020 | -0.0024 | +13,289 | +13.95s |
| playroom | -0.1064 | -0.0015 | +0.0003 | +5,088 | +15.97s |
| **平均** | **+0.0085** | **+0.0002** | **-0.0011** | **+9,189** | **+14.96s** |

Tandt 结果：

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| train | FastGS densify100 | 23.5886 | 0.9110 | 0.0792 | 43,476 | 144.24s |
| train | DINO descriptor top-k25 max | 23.5941 | 0.9132 | 0.0771 | 44,034 | 169.54s |
| truck | FastGS densify100 | 27.9625 | 0.9582 | 0.0349 | 35,851 | 135.54s |
| truck | DINO descriptor top-k25 max | 28.1578 | 0.9593 | 0.0338 | 39,454 | 176.24s |
| **平均** | **FastGS densify100** | **25.7755** | **0.9346** | **0.0571** | **39,664** | **139.89s** |
| **平均** | **DINO descriptor top-k25 max** | **25.8759** | **0.9363** | **0.0554** | **41,744** | **172.89s** |

Tandt 逐场景差值：

| 场景 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 |
|---|---:|---:|---:|---:|---:|
| train | +0.0055 | +0.0022 | -0.0021 | +558 | +25.31s |
| truck | +0.1953 | +0.0011 | -0.0011 | +3,603 | +40.70s |
| **平均** | **+0.1004** | **+0.0017** | **-0.0016** | **+2,081** | **+33.00s** |

分数据集汇总：

| 数据集 | 方法 | 场景数 | PSNR | ΔPSNR | SSIM | ΔSSIM | LPIPS | ΔLPIPS | Gaussian 数 | ΔGaussian | 训练时间 | Δ训练时间 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MipNeRF360 | DINO descriptor top-k25 max | 9 | 28.9035 | +0.1066 | 0.8687 | +0.0050 | 0.1347 | -0.0093 | 292,690 | +50,131 | 157.96s | +23.20s |
| DB | DINO descriptor top-k25 max | 2 | 30.6022 | +0.0085 | 0.9369 | +0.0002 | 0.0620 | -0.0011 | 72,468 | +9,189 | 148.18s | +14.96s |
| Tandt | DINO descriptor top-k25 max | 2 | 25.8759 | +0.1004 | 0.9363 | +0.0017 | 0.0554 | -0.0016 | 41,744 | +2,081 | 172.89s | +33.00s |

解读：

- 这是目前最干净的“同一功能模块强制启用 VFM”证据：三个公开数据集的平均 PSNR、SSIM、LPIPS 都相对各自 FastGS densify100 正向，且不依赖数据集级回退。
- MipNeRF360 的证据最强，9/9 场景三项指标全部正向；Tandt 的两场景也全部正向，说明 descriptor residual 与此前 token-edge weighted 分支不同，能在 Tandt 上提供有效复制引导。
- DB 均值虽然正向，但强依赖 `drjohnson`，`playroom` 单场景三项回落。因此 DB 上 descriptor top-k25 max 只能算弱正向，不应替代 DB 目前更强的 DINO weighted i0.90 或 cached-edge 正例。
- Gaussian 增长整体可接受：DB 平均只多 9,189 个点，Tandt 平均只多 2,081 个点；MipNeRF360 平均多 50,131 个点，但 `bicycle` 和 `stump` 已超过 0.1M 单场景关注阈值。
- 下一步应把 descriptor top-k25 max 作为“强制 VFM、无回退”的第一版核心证据之一。若继续推进质量，应优先在高分辨率同口径上做小范围 bicycle/garden/truck 试验；若推进效率，应研究 descriptor 调用频率和近似缓存，而不是继续调相邻 top-k 比例。

## 2026-05-10 DINO descriptor top-k25 max 高分辨率 bicycle 探针

目标：检查低分辨率 `-r 8` 中已经正向的 descriptor top-k25 max，在原图输入并沿用 FastGS 1.6K 自动缩放规则时是否仍能提升质量。该实验只跑 MipNeRF360 `bicycle`，训练使用 `fastgs_big` recipe，配置仍为 `dinov2_descriptor_cosine + top-k25 + vfm_weight=0.0`。VFM cache 复用 `output/0001/vfm_cache/bicycle_dinov2_vits14`，即 DINO-S/14、`max_width=224`、`npy_float16`；这次不改变 DINO 尺寸，先隔离“分辨率/recipe 迁移”变量。

命令：

```bash
source .venv/bin/activate && uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/descriptor_topk025_big_bicycle/vfm_dinov2_descriptor_topk25_max_big_30k_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/bicycle_dinov2_vits14 \
  -r -1
```

训练日志确认沿用 FastGS 原始大图规则：

```text
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
```

对照使用已完成的 `output/0001/large_res_fastgs_big_baseline/mipnerf360/bicycle/fastgs_big_densify100_30k_r_auto`。

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| bicycle | FastGS big densify100 | 25.2532 | 0.7554 | 0.2446 | 1,560,079 | 234.94s |
| bicycle | DINO descriptor top-k25 max + FastGS big | 25.3279 | 0.7646 | 0.2277 | 1,809,292 | 291.88s |

相对 FastGS big：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 |
|---:|---:|---:|---:|---:|
| +0.0748 | +0.0093 | -0.0169 | +249,213 | +56.94s |

解读：

- high-res bicycle 同口径下 descriptor top-k25 max 仍三项质量正向，说明该语义 residual 不是只在 `-r 8` 缩图上有效。
- 显存没有形成阻塞：训练阶段观测约 9.5GB，低于 24GB 上限；DINO-S/224 descriptor 路径可在 1.6K 自动缩放口径下运行。
- 但 Gaussian 增长达到 +249,213，明显超过 0.1M 的单场景关注阈值；训练时间也多 56.94s。因此这不是可直接扩全场景的默认方案，而是“质量可迁移、容量待控制”的探针。
- 下一步不应立即扩大到全 MipNeRF360；应先在 bicycle 跑更保守的 high-res descriptor weighted 或预算/频率控制版本，目标是在保持 PSNR/SSIM/LPIPS 正向的同时把 ΔGS 压回 0.1M 以内。

## 2026-05-10 DINO descriptor top-k25 + weighted i0.50 高分辨率 bicycle

目标：在上一个 high-res `top-k25 max` 质量正向但 ΔGS 过高的基础上，测试 `importance_mode=weighted + importance_weight=0.50` 能否保留质量收益并把容量增长压回 0.1M 以内。训练仍使用 `fastgs_big` recipe、`-i images -r -1` 和同一个 DINO-S/224 descriptor cache。

命令：

```bash
source .venv/bin/activate && uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i050.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/descriptor_topk025_weighted_i050_big_bicycle/vfm_dinov2_descriptor_topk25_weighted_i050_big_30k_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/bicycle_dinov2_vits14 \
  -r -1
```

训练日志同样确认沿用 FastGS 原始大图规则：

```text
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
```

结果：

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| bicycle | FastGS big densify100 | 25.2532 | 0.7554 | 0.2446 | 1,560,079 | 234.94s |
| bicycle | DINO descriptor top-k25 max + FastGS big | 25.3279 | 0.7646 | 0.2277 | 1,809,292 | 291.88s |
| bicycle | DINO descriptor top-k25 weighted i0.50 + FastGS big | 25.2937 | 0.7606 | 0.2361 | 1,606,190 | 268.87s |

相对 FastGS big：

| 方法 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---|---:|---:|---:|---:|---:|---:|
| top-k25 max | +0.0748 | +0.0093 | -0.0169 | +249,213 | +56.94s | -0.3525 |
| top-k25 weighted i0.50 | +0.0406 | +0.0053 | -0.0085 | +46,111 | +33.93s | +0.1422 |

相对 top-k25 max：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 |
|---:|---:|---:|---:|---:|
| -0.0342 | -0.0040 | +0.0084 | -203,102 | -23.01s |

解读：

- `weighted i0.50` 达成当前 high-res 容量目标：相对 FastGS big 只多 46,111 个 Gaussians，低于 0.1M 单场景关注阈值；相对 `max` 少 203,102 个点。
- 质量仍三项正向：PSNR +0.0406、SSIM +0.0053、LPIPS 改善 -0.0085。虽然低于 `max` 的质量上界，但不再是高增点方案。
- 按当前 QCGI，`max` 因 ΔGS 进入重惩罚区间而为负，`weighted i0.50` 为正。因此 high-res 后续扩展应优先采用 `top-k25 weighted i0.50` 作为容量受控候选，而不是直接扩展 `max`。
- 下一步建议先在 high-res `garden` 或 `truck` 上复验 `weighted i0.50`。若第二、第三个场景也保持三项指标正向且 ΔGS < 0.1M，再进入 MipNeRF360/DB/Tandt 全量 high-res 评估。

## 2026-05-10 DINO descriptor top-k25 + weighted i0.50 高分辨率 truck 复验

目标：用 Tandt `truck` 检查 high-res `top-k25 weighted i0.50` 是否能从 bicycle 外推到第二个公开数据集场景。训练使用 `fastgs_big` recipe，并对齐 FastGS big 的 truck 场景超参：`--highfeature_lr 0.04 --grad_abs_thresh 0.0004 --mult 0.7`。输入仍为 `-i images -r -1`；该场景日志没有触发 1.6K 自动缩放提示，说明原图尺寸没有超过 FastGS 的自动缩放阈值。

命令：

```bash
source .venv/bin/activate && uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i050.yaml \
  -s datasets/tandt_db/tandt/truck \
  -i images \
  -m output/0001/descriptor_topk025_weighted_i050_big_truck/vfm_dinov2_descriptor_topk25_weighted_i050_big_30k_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/truck_dinov2_vits14 \
  -r -1 \
  --highfeature_lr 0.04 \
  --grad_abs_thresh 0.0004 \
  --mult 0.7
```

对照使用已完成的 `output/0001/large_res_fastgs_big_baseline/tandt/truck/fastgs_big_densify100_30k_r_auto`。

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| truck | FastGS big densify100 | 26.1085 | 0.8894 | 0.1394 | 623,129 | 172.65s |
| truck | DINO descriptor top-k25 weighted i0.50 + FastGS big | 26.0251 | 0.8886 | 0.1374 | 727,663 | 227.00s |

相对 FastGS big：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| -0.0835 | -0.0008 | -0.0019 | +104,534 | +54.34s | -0.2088 |

解读：

- truck 是 high-res `weighted i0.50` 的边界负例：LPIPS 小幅改善 -0.0019，但 PSNR 和 SSIM 分别回落 -0.0835、-0.0008。
- Gaussian 数量增加 104,534，略高于 0.1M 单场景关注阈值；训练时间增加 54.34s。该容量增长没有换来足够的三项质量收益。
- 因此 high-res `weighted i0.50` 不能直接扩展到三个公开数据集全量评估。当前更稳妥的下一步是增加第三个 MipNeRF360 场景，判断负例是否集中在 Tandt truck；若 MipNeRF360 继续稳定，再考虑先扩 MipNeRF360，而不是直接扩三个公开数据集。

## 2026-05-10 DINO descriptor top-k25 + weighted i0.50 高分辨率 garden 复验

目标：补一个 MipNeRF360 内部的大场景，判断 high-res `top-k25 weighted i0.50` 的正向结果是否只发生在 `bicycle`。训练使用 `fastgs_big` recipe，并对齐 FastGS big 的 garden 场景超参：`--highfeature_lr 0.02 --loss_thresh 0.06 --grad_abs_thresh 0.0003`。输入仍为 `-i images -r -1`，cache 使用 `output/0001/vfm_cache/garden_dinov2_vits14`。

命令：

```bash
source .venv/bin/activate && uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i050.yaml \
  -s datasets/mipnerf360/garden \
  -i images \
  -m output/0001/descriptor_topk025_weighted_i050_big_garden/vfm_dinov2_descriptor_topk25_weighted_i050_big_30k_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/garden_dinov2_vits14 \
  -r -1 \
  --highfeature_lr 0.02 \
  --loss_thresh 0.06 \
  --grad_abs_thresh 0.0003
```

训练日志确认沿用 FastGS 原始大图规则：

```text
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
```

对照使用已完成的 `output/0001/large_res_fastgs_big_baseline/mipnerf360/garden/fastgs_big_densify100_30k_r_auto`。

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| garden | FastGS big densify100 | 27.6137 | 0.8645 | 0.1098 | 2,624,164 | 411.34s |
| garden | DINO descriptor top-k25 weighted i0.50 + FastGS big | 27.6376 | 0.8648 | 0.1094 | 2,570,661 | 424.24s |

相对 FastGS big：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| +0.0239 | +0.0002 | -0.0004 | -53,503 | +12.90s | +0.0307 |

解读：

- garden 是 high-res `weighted i0.50` 的第二个 MipNeRF360 正例：PSNR、SSIM、LPIPS 均小幅优于 FastGS big。
- 与 bicycle 不同，garden 没有带来 Gaussian 增长，反而少 53,503 个点；这说明 high-res descriptor weighted 档并不一定依赖增点换质量。
- 当前 high-res 结论应分开看：MipNeRF360 内的 bicycle/garden 为正向，Tandt truck 为边界负例。下一步更适合继续补 MipNeRF360 场景以判断数据集平均，而不是因为 truck 单场景负例立即停止 high-res descriptor 线。

## 2026-05-10 DINO descriptor top-k25 + weighted i0.50 高分辨率 stump 复验

目标：继续补 MipNeRF360 high-res 场景，检查 `top-k25 weighted i0.50` 在大收益但易增点的 `stump` 上是否仍能提升质量。训练使用 `fastgs_big` recipe，并对齐 FastGS big 的 stump 场景超参：`--dense 0.004 --grad_abs_thresh 0.001`。输入为 `-i images -r -1`，cache 使用 `output/0001/vfm_cache/stump_dinov2_vits14`。

命令：

```bash
source .venv/bin/activate && uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i050.yaml \
  -s datasets/mipnerf360/stump \
  -i images \
  -m output/0001/descriptor_topk025_weighted_i050_big_stump/vfm_dinov2_descriptor_topk25_weighted_i050_big_30k_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/stump_dinov2_vits14 \
  -r -1 \
  --dense 0.004 \
  --grad_abs_thresh 0.001
```

训练日志确认沿用 FastGS 原始大图规则：

```text
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
```

对照使用已完成的 `output/0001/large_res_fastgs_big_baseline/mipnerf360/stump/fastgs_big_densify100_30k_r_auto`。

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| stump | FastGS big densify100 | 27.1310 | 0.7862 | 0.2406 | 1,062,281 | 189.72s |
| stump | DINO descriptor top-k25 weighted i0.50 + FastGS big | 27.2233 | 0.7903 | 0.2317 | 1,196,350 | 283.46s |

相对 FastGS big：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| +0.0923 | +0.0042 | -0.0089 | +134,069 | +93.74s | -0.0162 |

解读：

- stump 是 high-res `weighted i0.50` 的第三个 MipNeRF360 质量正例：PSNR、SSIM、LPIPS 三项均优于 FastGS big。
- 但该场景新增 134,069 个 Gaussians，超过 0.1M 单场景关注阈值；按当前 QCGI 规则，质量收益没有完全抵消容量惩罚。
- 这说明 high-res MipNeRF360 内部不是简单的“越多点越好”：bicycle 是小幅增点正向，garden 是少点正向，stump 是明显增点换质量。下一步应继续补一个中等/室内场景，估计 MipNeRF360 high-res 的数据集均值，同时扫描 `i0.35` 或更强容量约束来处理 stump 这类高增点场景。

## 2026-05-10 DINO descriptor top-k25 + weighted i0.50 高分辨率 counter 复验

目标：补一个 MipNeRF360 室内/中等规模场景，降低 high-res 结论只来自 bicycle/garden/stump 这类户外或复杂植被场景的偏差。训练使用 `fastgs_big` recipe，并对齐 FastGS big 的 counter 场景超参：`--highfeature_lr 0.02 --grad_abs_thresh 0.0004`。输入为 `-i images -r -1`，cache 使用 `output/0001/vfm_cache/counter_dinov2_vits14`。

命令：

```bash
source .venv/bin/activate && uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i050.yaml \
  -s datasets/mipnerf360/counter \
  -i images \
  -m output/0001/descriptor_topk025_weighted_i050_big_counter/vfm_dinov2_descriptor_topk25_weighted_i050_big_30k_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/counter_dinov2_vits14 \
  -r -1 \
  --highfeature_lr 0.02 \
  --grad_abs_thresh 0.0004
```

训练日志确认沿用 FastGS 原始大图规则：

```text
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
```

对照使用已完成的 `output/0001/large_res_fastgs_big_baseline/mipnerf360/counter/fastgs_big_densify100_30k_r_auto`。

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| counter | FastGS big densify100 | 29.5268 | 0.9180 | 0.1763 | 473,200 | 180.41s |
| counter | DINO descriptor top-k25 weighted i0.50 + FastGS big | 29.5838 | 0.9194 | 0.1723 | 551,300 | 200.97s |

相对 FastGS big：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| +0.0570 | +0.0014 | -0.0041 | +78,100 | +20.56s | +0.0274 |

解读：

- counter 是 high-res `weighted i0.50` 的第四个 MipNeRF360 质量正例，也是第三个 QCGI 为正的容量效率正例。
- 新增 78,100 个 Gaussians，低于 0.1M 单场景关注阈值；质量收益足以覆盖当前 QCGI 的容量惩罚。
- high-res MipNeRF360 已覆盖 bicycle/garden/stump/counter 四个场景，四场景三项质量均正向；其中 bicycle/garden/counter 通过 QCGI，stump 是质量正向但容量偏高的边界样本。下一步优先继续补 kitchen/room 这类室内高基线场景，或对 stump 做更强容量约束扫描。

## 2026-05-10 DINO descriptor top-k25 + weighted i0.50 高分辨率 kitchen 复验

目标：继续补 MipNeRF360 室内高基线场景，判断 high-res `top-k25 weighted i0.50` 在 already-strong baseline 上是否仍能提供正向质量收益。训练使用 `fastgs_big` recipe，并对齐 FastGS big 的 kitchen 场景超参：`--highfeature_lr 0.02 --grad_abs_thresh 0.0002`。输入为 `-i images -r -1`，cache 使用 `output/0001/vfm_cache/kitchen_dinov2_vits14`。

命令：

```bash
source .venv/bin/activate && uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i050.yaml \
  -s datasets/mipnerf360/kitchen \
  -i images \
  -m output/0001/descriptor_topk025_weighted_i050_big_kitchen/vfm_dinov2_descriptor_topk25_weighted_i050_big_30k_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/kitchen_dinov2_vits14 \
  -r -1 \
  --highfeature_lr 0.02 \
  --grad_abs_thresh 0.0002
```

训练日志确认沿用 FastGS 原始大图规则：

```text
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
```

对照使用已完成的 `output/0001/large_res_fastgs_big_baseline/mipnerf360/kitchen/fastgs_big_densify100_30k_r_auto`。

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| kitchen | FastGS big densify100 | 32.2700 | 0.9391 | 0.1044 | 1,178,795 | 335.11s |
| kitchen | DINO descriptor top-k25 weighted i0.50 + FastGS big | 32.4350 | 0.9398 | 0.1036 | 1,286,004 | 383.90s |

相对 FastGS big：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| +0.1650 | +0.0007 | -0.0008 | +107,209 | +48.79s | +0.0546 |

解读：

- kitchen 是 high-res `weighted i0.50` 的第五个 MipNeRF360 质量正例，也是在室内高基线场景上的正向样本。
- 新增 107,209 个 Gaussians，略高于 0.1M 单场景关注阈值；但 PSNR 提升 +0.1650 足以让 QCGI 保持正值。
- 该结果支持“少量超过 0.1M 的 GS 增长不应机械否决”：当质量收益足够明确时，新增点可视为有效容量。下一步仍需要补 `room` 或扫描 `stump i0.35`，区分高增点正向和低效增点。

## 2026-05-10 DINO descriptor top-k25 + weighted i0.50 高分辨率 room 复验

目标：继续补 MipNeRF360 室内高基线场景，与 kitchen/counter 一起判断 high-res descriptor weighted 档在室内场景上的稳定性。训练使用 `fastgs_big` recipe，并对齐 FastGS big 的 room 场景超参：`--highfeature_lr 0.02 --grad_abs_thresh 0.0004`。输入为 `-i images -r -1`，cache 使用 `output/0001/vfm_cache/room_dinov2_vits14`。

命令：

```bash
source .venv/bin/activate && uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i050.yaml \
  -s datasets/mipnerf360/room \
  -i images \
  -m output/0001/descriptor_topk025_weighted_i050_big_room/vfm_dinov2_descriptor_topk25_weighted_i050_big_30k_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/room_dinov2_vits14 \
  -r -1 \
  --highfeature_lr 0.02 \
  --grad_abs_thresh 0.0004
```

训练日志确认沿用 FastGS 原始大图规则：

```text
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
```

对照使用已完成的 `output/0001/large_res_fastgs_big_baseline/mipnerf360/room/fastgs_big_densify100_30k_r_auto`。

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| room | FastGS big densify100 | 32.1323 | 0.9298 | 0.1881 | 570,779 | 164.20s |
| room | DINO descriptor top-k25 weighted i0.50 + FastGS big | 32.1919 | 0.9324 | 0.1822 | 615,908 | 197.77s |

相对 FastGS big：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| +0.0596 | +0.0025 | -0.0059 | +45,129 | +33.57s | +0.0958 |

解读：

- room 是 high-res `weighted i0.50` 的第六个 MipNeRF360 质量正例，且 QCGI 明确为正。
- 新增 45,129 个 Gaussians，低于 0.1M 单场景关注阈值；SSIM 和 LPIPS 的收益比 PSNR 更突出。
- 目前 high-res MipNeRF360 已覆盖 6 个场景，6/6 三项质量正向，其中 5/6 QCGI 为正；唯一 QCGI 负例是 stump。下一步可以补剩余 flowers/bonsai/treehill，或先对 stump 做 `i0.35` 容量约束复验。

## 2026-05-10 DINO descriptor top-k25 + weighted i0.50 高分辨率 bonsai 复验

目标：继续补 MipNeRF360 室内/小物体高基线场景，检查 high-res `top-k25 weighted i0.50` 在 bonsai 这类容易增点的场景上是否仍具备质量收益。训练使用 `fastgs_big` recipe，并对齐 FastGS big 的 bonsai 场景超参：`--highfeature_lr 0.02 --grad_abs_thresh 0.0002`。输入为 `-i images -r -1`，cache 使用 `output/0001/vfm_cache/bonsai_dinov2_vits14`。

命令：

```bash
source .venv/bin/activate && uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i050.yaml \
  -s datasets/mipnerf360/bonsai \
  -i images \
  -m output/0001/descriptor_topk025_weighted_i050_big_bonsai/vfm_dinov2_descriptor_topk25_weighted_i050_big_30k_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/bonsai_dinov2_vits14 \
  -r -1 \
  --highfeature_lr 0.02 \
  --grad_abs_thresh 0.0002
```

训练日志确认沿用 FastGS 原始大图规则：

```text
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
```

对照使用已完成的 `output/0001/large_res_fastgs_big_baseline/mipnerf360/bonsai/fastgs_big_densify100_30k_r_auto`。

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| bonsai | FastGS big densify100 | 32.9863 | 0.9512 | 0.1600 | 842,636 | 212.97s |
| bonsai | DINO descriptor top-k25 weighted i0.50 + FastGS big | 33.1160 | 0.9549 | 0.1560 | 1,094,114 | 265.81s |

相对 FastGS big：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| +0.1297 | +0.0037 | -0.0040 | +251,478 | +52.84s | -0.4824 |

解读：

- bonsai 是 high-res `weighted i0.50` 的第七个 MipNeRF360 三项质量正例，说明 descriptor residual 在该场景上仍能提升测试集渲染质量。
- 但新增 251,478 个 Gaussians，远高于 0.1M 单场景关注阈值；按当前 QCGI 计算，容量惩罚明显超过质量收益。
- 因此 bonsai 应记录为“质量正向但容量过强”的边界样本，不放入正向效率主表。后续更适合对 bonsai/stump 扫描 `i0.35` 或加入自适应容量约束，而不是直接扩大同一档位。

## 2026-05-10 DINO descriptor top-k25 + weighted i0.50 高分辨率 flowers 复验

目标：补 MipNeRF360 植被/细碎纹理场景，检查 high-res `top-k25 weighted i0.50` 在低 PSNR、高感知难度的 flowers 上是否能延续低分辨率 descriptor 正例。训练使用 `fastgs_big` recipe，并对齐 FastGS big 的 flowers 场景超参：`--dense 0.005 --grad_abs_thresh 0.001`。输入为 `-i images -r -1`，cache 使用 `output/0001/vfm_cache/flowers_dinov2_vits14`。

命令：

```bash
source .venv/bin/activate && uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i050.yaml \
  -s datasets/mipnerf360/flowers \
  -i images \
  -m output/0001/descriptor_topk025_weighted_i050_big_flowers/vfm_dinov2_descriptor_topk25_weighted_i050_big_30k_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/flowers_dinov2_vits14 \
  -r -1 \
  --dense 0.005 \
  --grad_abs_thresh 0.001
```

训练日志确认沿用 FastGS 原始大图规则：

```text
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
```

对照使用已完成的 `output/0001/large_res_fastgs_big_baseline/mipnerf360/flowers/fastgs_big_densify100_30k_r_auto`。

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| flowers | FastGS big densify100 | 21.6166 | 0.6017 | 0.3403 | 1,140,260 | 207.76s |
| flowers | DINO descriptor top-k25 weighted i0.50 + FastGS big | 21.6293 | 0.6022 | 0.3412 | 1,091,531 | 278.68s |

相对 FastGS big：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| +0.0127 | +0.0004 | +0.0008 | -48,729 | +70.92s | +0.0172 |

解读：

- flowers 不是三项质量正例：PSNR 和 SSIM 小幅提升，LPIPS 小幅变差。
- Gaussian 数量减少 48,729，因此 QCGI 仍为正；但训练时间增加 70.92s，说明 online descriptor 成本在该场景没有换来足够清晰的感知收益。
- 该结果应作为 high-res `weighted i0.50` 的混合样本保留，不放入正向效率主表。后续若要提升 flowers，优先尝试更高 descriptor 强度或 quality-first `max/i0.70`，而不是继续降低容量。

## 2026-05-10 DINO descriptor top-k25 + weighted i0.50 高分辨率 treehill 复验与 MipNeRF360 汇总

目标：补齐 MipNeRF360 high-res 最后一个场景 treehill，并形成同一功能模块在 9 个 MipNeRF360 场景上的数据集均值。treehill 训练使用 `fastgs_big` recipe，并对齐 FastGS big 场景超参：`--dense 0.01 --grad_abs_thresh 0.0018`。输入为 `-i images -r -1`，cache 使用 `output/0001/vfm_cache/treehill_dinov2_vits14`。

命令：

```bash
source .venv/bin/activate && uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i050.yaml \
  -s datasets/mipnerf360/treehill \
  -i images \
  -m output/0001/descriptor_topk025_weighted_i050_big_treehill/vfm_dinov2_descriptor_topk25_weighted_i050_big_30k_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/treehill_dinov2_vits14 \
  -r -1 \
  --dense 0.01 \
  --grad_abs_thresh 0.0018
```

训练日志确认沿用 FastGS 原始大图规则：

```text
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
```

treehill 单场景结果：

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| treehill | FastGS big densify100 | 22.8339 | 0.6318 | 0.3770 | 998,983 | 189.59s |
| treehill | DINO descriptor top-k25 weighted i0.50 + FastGS big | 22.8069 | 0.6317 | 0.3774 | 945,930 | 240.89s |

相对 FastGS big：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| -0.0270 | -0.0001 | +0.0004 | -53,053 | +51.29s | -0.0318 |

treehill 是 high-res `weighted i0.50` 的明确质量负例。它虽然少 53,053 个 Gaussians，但 PSNR/SSIM/LPIPS 三项都低于 FastGS big，说明在该场景上降低容量没有带来有效质量收益。后续如果继续处理 treehill，应优先试 `max/i0.70` 或加入几何/深度先验，而不是继续压低 weighted 强度。

MipNeRF360 high-res 9 场景汇总：

| 场景 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---|---:|---:|---:|---:|---:|---:|
| bicycle | +0.0406 | +0.0053 | -0.0085 | +46,111 | +33.93s | +0.1422 |
| flowers | +0.0127 | +0.0004 | +0.0008 | -48,729 | +70.92s | +0.0172 |
| garden | +0.0239 | +0.0002 | -0.0004 | -53,503 | +12.90s | +0.0307 |
| stump | +0.0923 | +0.0042 | -0.0089 | +134,069 | +93.74s | -0.0162 |
| treehill | -0.0270 | -0.0001 | +0.0004 | -53,053 | +51.29s | -0.0318 |
| counter | +0.0570 | +0.0014 | -0.0041 | +78,100 | +20.56s | +0.0274 |
| kitchen | +0.1650 | +0.0007 | -0.0008 | +107,209 | +48.79s | +0.0546 |
| room | +0.0596 | +0.0025 | -0.0059 | +45,129 | +33.57s | +0.0948 |
| bonsai | +0.1297 | +0.0037 | -0.0040 | +251,478 | +52.84s | -0.4824 |
| **平均** | **+0.0615** | **+0.0020** | **-0.0035** | **+56,312** | **+46.50s** | **+0.0633** |

绝对均值：

| 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---:|---:|---:|---:|---:|
| FastGS big densify100 | 27.9293 | 0.8198 | 0.2157 | 1,161,242 | 236.23s |
| DINO descriptor top-k25 weighted i0.50 + FastGS big | 27.9908 | 0.8218 | 0.2122 | 1,217,554 | 282.73s |

解读：

- 数据集均值是正向的：PSNR +0.0615、SSIM +0.0020、LPIPS 改善 -0.0035，说明同一 VFM descriptor densify-only 模块在 MipNeRF360 high-res 口径下具备整体质量收益。
- 场景分布不是全胜：bicycle/garden/counter/kitchen/room/bonsai/stump 三项或主要质量指标正向，flowers 是混合样本，treehill 是明确负例。
- 平均多 56,312 个 Gaussians，处于 `0.01M_to_0.10M` 可接受增长区间；但训练时间平均多 46.50s，且 bonsai/stump 出现容量边界问题。下一步应针对 bonsai/stump 做更强容量约束，针对 flowers/treehill 做更高 descriptor 强度或几何先验，而不是把 i0.50 直接视为最终收束。

## 2026-05-10 DINO descriptor top-k25 max 高分辨率 treehill 复验

目标：复查 high-res `weighted i0.50` 在 treehill 上的质量负例是否来自 descriptor 强度不足。该实验改用 `top-k25 max` 质量优先档，仍保持 `vfm_weight=0.0`，只让 DINO descriptor residual 参与 densification，不改变 FastGS pruning score。训练使用 `fastgs_big` recipe，并对齐 FastGS big 的 treehill 场景超参：`--dense 0.01 --grad_abs_thresh 0.0018`。输入为 `-i images -r -1`，cache 使用 `output/0001/vfm_cache/treehill_dinov2_vits14`。

命令：

```bash
source .venv/bin/activate && uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025.yaml \
  -s datasets/mipnerf360/treehill \
  -i images \
  -m output/0001/descriptor_topk025_max_big_treehill/vfm_dinov2_descriptor_topk25_max_big_30k_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/treehill_dinov2_vits14 \
  -r -1 \
  --dense 0.01 \
  --grad_abs_thresh 0.0018
```

训练日志确认沿用 FastGS 原始大图规则：

```text
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
```

treehill 单场景结果：

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| treehill | FastGS big densify100 | 22.8339 | 0.6318 | 0.3770 | 998,983 | 189.59s |
| treehill | DINO descriptor top-k25 weighted i0.50 + FastGS big | 22.8069 | 0.6317 | 0.3774 | 945,930 | 240.89s |
| treehill | DINO descriptor top-k25 max + FastGS big | 22.8700 | 0.6367 | 0.3665 | 1,129,614 | 219.73s |

`max` 相对 FastGS big：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| +0.0362 | +0.0049 | -0.0105 | +130,631 | +30.14s | -0.0360 |

`max` 相对 `weighted i0.50`：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 |
|---:|---:|---:|---:|---:|
| +0.0632 | +0.0050 | -0.0109 | +183,684 | -21.15s |

解读：

- treehill 的负例可以被更强 descriptor 强度修复：`max` 相对 FastGS big 三项质量指标全部正向，也明显优于 `weighted i0.50`。
- 该修复依赖额外 130,631 个 Gaussians，超过 0.1M 容量关注阈值；按当前 QCGI 计算仍为负。因此它是“质量修复但容量代价偏高”的证据，不放入正向效率主表。
- 结论上，treehill 更像 descriptor 强度不足问题，而不是 descriptor residual 完全无效。下一步如果继续处理 treehill，应优先尝试介于 `weighted i0.50` 和 `max` 之间的 high-res `i0.70`，或者引入容量自适应权重，而不是直接固定使用 `max`。

## 2026-05-10 DINO descriptor top-k25 weighted i0.70 高分辨率 treehill 复验

目标：验证 `weighted i0.70` 是否能在 high-res treehill 上取得介于 `weighted i0.50` 和 `max` 之间的折中：尽量恢复 `max` 的质量，同时避免超过 0.1M 的 Gaussian 增长。训练继续使用 `fastgs_big` recipe，并对齐 FastGS big 的 treehill 场景超参：`--dense 0.01 --grad_abs_thresh 0.0018`。输入为 `-i images -r -1`，cache 使用 `output/0001/vfm_cache/treehill_dinov2_vits14`。

命令：

```bash
source .venv/bin/activate && uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i070.yaml \
  -s datasets/mipnerf360/treehill \
  -i images \
  -m output/0001/descriptor_topk025_weighted_i070_big_treehill/vfm_dinov2_descriptor_topk25_weighted_i070_big_30k_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/treehill_dinov2_vits14 \
  -r -1 \
  --dense 0.01 \
  --grad_abs_thresh 0.0018
```

训练日志确认沿用 FastGS 原始大图规则：

```text
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
```

treehill 对照：

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| treehill | FastGS big densify100 | 22.8339 | 0.6318 | 0.3770 | 998,983 | 189.59s |
| treehill | DINO descriptor top-k25 weighted i0.50 + FastGS big | 22.8069 | 0.6317 | 0.3774 | 945,930 | 240.89s |
| treehill | DINO descriptor top-k25 weighted i0.70 + FastGS big | 22.8614 | 0.6305 | 0.3813 | 877,180 | 221.12s |
| treehill | DINO descriptor top-k25 max + FastGS big | 22.8700 | 0.6367 | 0.3665 | 1,129,614 | 219.73s |

`weighted i0.70` 相对 FastGS big：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| +0.0276 | -0.0014 | +0.0043 | -121,803 | +31.53s | -0.0212 |

`weighted i0.70` 相对 `weighted i0.50`：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| +0.0546 | -0.0012 | +0.0039 | -68,750 | -19.77s | +0.0107 |

解读：

- `weighted i0.70` 没有形成理想折中。它比 `weighted i0.50` 省点、省时且 PSNR 更高，但 SSIM 和 LPIPS 更差。
- 相对 FastGS big，它只在 PSNR 上正向，SSIM/LPIPS 均负向；因此不能作为 treehill 的 high-res 推荐档，也不应进入正向改进表。
- treehill 当前结论收敛为两端分化：`max` 能修复三项质量但容量代价偏高，`weighted i0.70` 能压低容量但无法保住感知质量。后续 treehill 更适合转向自适应容量/几何先验，而不是继续密集扫描相邻 importance。

## 2026-05-10 DINO descriptor top-k25 weighted i0.70 高分辨率 flowers 复验

目标：flowers 在 high-res `weighted i0.50` 下 PSNR/SSIM 微升但 LPIPS 变差。本轮尝试 `weighted i0.70`，检查更强 descriptor residual 是否能修复感知指标。训练使用 `fastgs_big` recipe，并对齐 FastGS big 的 flowers 场景超参：`--dense 0.005 --grad_abs_thresh 0.001`。输入为 `-i images -r -1`，cache 使用 `output/0001/vfm_cache/flowers_dinov2_vits14`。

命令：

```bash
source .venv/bin/activate && uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i070.yaml \
  -s datasets/mipnerf360/flowers \
  -i images \
  -m output/0001/descriptor_topk025_weighted_i070_big_flowers/vfm_dinov2_descriptor_topk25_weighted_i070_big_30k_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/flowers_dinov2_vits14 \
  -r -1 \
  --dense 0.005 \
  --grad_abs_thresh 0.001
```

训练日志确认沿用 FastGS 原始大图规则：

```text
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
```

flowers 对照：

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| flowers | FastGS big densify100 | 21.6166 | 0.6017 | 0.3403 | 1,140,260 | 207.76s |
| flowers | DINO descriptor top-k25 weighted i0.50 + FastGS big | 21.6293 | 0.6022 | 0.3412 | 1,091,531 | 278.68s |
| flowers | DINO descriptor top-k25 weighted i0.70 + FastGS big | 21.5801 | 0.6001 | 0.3442 | 1,045,864 | 234.69s |

`weighted i0.70` 相对 FastGS big：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| -0.0365 | -0.0017 | +0.0039 | -94,396 | +26.93s | -0.0895 |

`weighted i0.70` 相对 `weighted i0.50`：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| -0.0492 | -0.0021 | +0.0031 | -45,667 | -43.99s | -0.1067 |

解读：

- `weighted i0.70` 没有修复 flowers 的 LPIPS，反而使 PSNR、SSIM、LPIPS 三项都低于 FastGS big。
- 它比 i0.50 少 45,667 个 Gaussians、训练少 43.99s，但质量同步下降，说明该场景不是简单提高 weighted importance 就能解决。
- flowers 后续如果继续处理，应改试 `max` 质量上界或引入几何/深度 residual；`i0.70` 不再扩展。

## 2026-05-10 DINO descriptor top-k25 max 高分辨率 flowers 复验

目标：在 flowers 上检查 `max` 质量优先档是否能修复 `weighted i0.50/i0.70` 的 LPIPS 问题，并判断该场景是否存在 DINO descriptor residual 的质量上界。训练继续使用 `fastgs_big` recipe，并对齐 FastGS big 的 flowers 场景超参：`--dense 0.005 --grad_abs_thresh 0.001`。输入为 `-i images -r -1`，cache 使用 `output/0001/vfm_cache/flowers_dinov2_vits14`。

命令：

```bash
source .venv/bin/activate && uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025.yaml \
  -s datasets/mipnerf360/flowers \
  -i images \
  -m output/0001/descriptor_topk025_max_big_flowers/vfm_dinov2_descriptor_topk25_max_big_30k_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/flowers_dinov2_vits14 \
  -r -1 \
  --dense 0.005 \
  --grad_abs_thresh 0.001
```

训练日志确认沿用 FastGS 原始大图规则：

```text
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
```

flowers 对照：

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| flowers | FastGS big densify100 | 21.6166 | 0.6017 | 0.3403 | 1,140,260 | 207.76s |
| flowers | DINO descriptor top-k25 weighted i0.50 + FastGS big | 21.6293 | 0.6022 | 0.3412 | 1,091,531 | 278.68s |
| flowers | DINO descriptor top-k25 weighted i0.70 + FastGS big | 21.5801 | 0.6001 | 0.3442 | 1,045,864 | 234.69s |
| flowers | DINO descriptor top-k25 max + FastGS big | 21.6394 | 0.6039 | 0.3386 | 1,273,570 | 275.79s |

`max` 相对 FastGS big：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| +0.0228 | +0.0021 | -0.0017 | +133,310 | +68.03s | -0.1595 |

`max` 相对 `weighted i0.50`：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| +0.0101 | +0.0017 | -0.0026 | +182,039 | -2.89s | -0.3716 |

解读：

- `max` 修复了 flowers 的感知指标：相对 FastGS big 三项质量都正向，也优于 `weighted i0.50/i0.70`。
- 该收益依赖额外 133,310 个 Gaussians，QCGI 为负；因此它是质量上界证据，而不是容量高效方案。
- flowers 的结论与 treehill 类似：DINO descriptor residual 有效，但固定 weighted 档不能在 high-res 下稳定取得质量-容量折中。后续应尝试自适应容量控制或 Depth Anything 几何/边界 residual，而不是继续扫描相邻权重。

## 2026-05-10 DINO descriptor top-k25 weighted i0.35 高分辨率 stump 容量探针

目标：stump 在 high-res `weighted i0.50` 下三项质量正向，但多 134,069 个 Gaussians，QCGI 为负。本轮尝试更保守的 `weighted i0.35`，检查是否能压低容量并保留主要质量收益。训练使用 `fastgs_big` recipe，并对齐 FastGS big 的 stump 场景超参：`--dense 0.01 --grad_abs_thresh 0.00015`。输入为 `-i images -r -1`，cache 使用 `output/0001/vfm_cache/stump_dinov2_vits14`。

命令：

```bash
source .venv/bin/activate && uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i035.yaml \
  -s datasets/mipnerf360/stump \
  -i images \
  -m output/0001/descriptor_topk025_weighted_i035_big_stump/vfm_dinov2_descriptor_topk25_weighted_i035_big_30k_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/stump_dinov2_vits14 \
  -r -1 \
  --dense 0.01 \
  --grad_abs_thresh 0.00015
```

训练日志确认沿用 FastGS 原始大图规则：

```text
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
```

stump 对照：

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| stump | FastGS big densify100 | 27.1310 | 0.7862 | 0.2406 | 1,062,281 | 189.72s |
| stump | DINO descriptor top-k25 weighted i0.50 + FastGS big | 27.2233 | 0.7903 | 0.2317 | 1,196,350 | 283.46s |
| stump | DINO descriptor top-k25 weighted i0.35 + FastGS big | 26.7484 | 0.7768 | 0.2305 | 3,035,777 | 385.31s |

`weighted i0.35` 相对 FastGS big：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| -0.3826 | -0.0093 | -0.0101 | +1,973,496 | +195.59s | -8.1125 |

`weighted i0.35` 相对 `weighted i0.50`：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| -0.4749 | -0.0135 | -0.0012 | +1,839,427 | +101.86s | -7.7963 |

解读：

- `weighted i0.35` 没有压低容量，反而触发容量失控，最终达到 3.04M Gaussians。
- 质量也没有保住：PSNR/SSIM 明显低于 FastGS big 和 i0.50，仅 LPIPS 相对 baseline 有轻微改善。
- 该结果说明 high-res stump 上 `vfm_importance_weight` 与最终 Gaussian 数量不是单调关系；简单降低权重不是可靠的容量控制方式。后续应使用显式自适应容量约束、target 保护或训练过程点数反馈，而不是继续扫更低固定权重。

## 2026-05-10 DINO descriptor top-k25 weighted i0.50 + soft budget 高分辨率 stump 探针

目标：在不改变 i0.50 质量档的前提下，尝试用现有 `vfm_importance_budget_count` 软预算机制约束 stump 的容量增长。本轮设置 `vfm_importance_budget_count=1150000`、`vfm_importance_budget_start_ratio=0.90`、`vfm_importance_budget_min_weight=0.0`、`vfm_importance_budget_curve=linear`。训练使用 `fastgs_big` recipe，并对齐 FastGS big 的 stump 场景超参：`--dense 0.01 --grad_abs_thresh 0.00015`。输入为 `-i images -r -1`。

命令：

```bash
source .venv/bin/activate && uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i050.yaml \
  -s datasets/mipnerf360/stump \
  -i images \
  -m output/0001/descriptor_topk025_weighted_i050_budget1150k_big_stump/vfm_dinov2_descriptor_topk25_weighted_i050_budget1150k_big_30k_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/stump_dinov2_vits14 \
  -r -1 \
  --dense 0.01 \
  --grad_abs_thresh 0.00015 \
  --vfm_importance_budget_count 1150000 \
  --vfm_importance_budget_start_ratio 0.90 \
  --vfm_importance_budget_min_weight 0.0 \
  --vfm_importance_budget_curve linear
```

训练日志确认沿用 FastGS 原始大图规则：

```text
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
```

stump 对照：

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| stump | FastGS big densify100 | 27.1310 | 0.7862 | 0.2406 | 1,062,281 | 189.72s |
| stump | DINO descriptor top-k25 weighted i0.50 + FastGS big | 27.2233 | 0.7903 | 0.2317 | 1,196,350 | 283.46s |
| stump | DINO descriptor top-k25 weighted i0.50 + soft budget 1.15M | 26.6930 | 0.7743 | 0.2376 | 2,247,258 | 341.23s |

soft budget 相对 FastGS big：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| -0.4379 | -0.0119 | -0.0030 | +1,184,977 | +151.51s | -5.1007 |

soft budget 相对 `weighted i0.50`：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| -0.5303 | -0.0160 | +0.0059 | +1,050,908 | +57.78s | -4.7844 |

解读：

- 现有软预算机制不足以约束 high-res stump。它只衰减 VFM densification 权重，无法约束 FastGS/RGB 路径自身的 densification 轨迹，最终仍增长到 2.25M Gaussians。
- 质量也明显低于 i0.50 和 FastGS big，说明单纯软衰减会改变训练轨迹，但没有形成有效容量-质量折中。
- 后续如果继续做容量控制，需要更硬的训练期点数反馈，例如在超过预算后同时降低 RGB/VFM densification、缩短 densification 窗口、提高 clone/split 门槛，或在 densify 后立即执行小步 staged prune 与恢复；仅靠当前 `vfm_importance_budget_count` 不够。

## 2026-05-10 DINO descriptor top-k25 weighted i0.50 + staged target 1.15M 高分辨率 stump 探针

目标：检验更硬的训练期点数反馈是否能解决 stump 的容量边界问题。本轮在 `weighted i0.50` 基础上设置 `target_gaussian_count=1150000`，并启用 staged target：从 iteration 9000 开始，每 500 step 将点数裁到 `1.02 * target`，最终再裁到 1.15M。训练仍使用 `fastgs_big` recipe，对齐 stump 的 `--dense 0.01 --grad_abs_thresh 0.00015`，输入为 `-i images -r -1`。

命令：

```bash
source .venv/bin/activate && uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i050.yaml \
  -s datasets/mipnerf360/stump \
  -i images \
  -m output/0001/descriptor_topk025_weighted_i050_staged1150k_big_stump/vfm_dinov2_descriptor_topk25_weighted_i050_staged1150k_big_30k_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/stump_dinov2_vits14 \
  -r -1 \
  --dense 0.01 \
  --grad_abs_thresh 0.00015 \
  --target_gaussian_count 1150000 \
  --target_gaussian_staged \
  --target_gaussian_stage_margin 1.02 \
  --target_gaussian_stage_start 9000 \
  --target_gaussian_stage_interval 500 \
  --target_gaussian_prune_order lowest_score
```

训练日志确认沿用 FastGS 原始大图规则：

```text
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
```

staged pruning 轨迹：

| iteration | 裁剪前 | 裁剪后 | 删除数量 |
|---:|---:|---:|---:|
| 9000 | 3,702,269 | 1,173,000 | 2,529,269 |
| 9500 | 1,372,895 | 1,173,000 | 199,895 |
| 10000 | 1,365,028 | 1,173,000 | 192,028 |
| 10500 | 1,440,307 | 1,173,000 | 267,307 |
| 11000 | 1,468,639 | 1,173,000 | 295,639 |
| 11500 | 1,566,211 | 1,173,000 | 393,211 |
| 12000 | 1,562,803 | 1,173,000 | 389,803 |
| 12500 | 1,590,424 | 1,173,000 | 417,424 |
| 13000 | 1,455,595 | 1,173,000 | 282,595 |
| 13500 | 1,461,691 | 1,173,000 | 288,691 |
| 14000 | 1,499,122 | 1,173,000 | 326,122 |
| 14500 | 1,485,451 | 1,173,000 | 312,451 |
| final | 1,157,762 | 1,150,000 | 7,762 |

stump 对照：

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| stump | FastGS big densify100 | 27.1310 | 0.7862 | 0.2406 | 1,062,281 | 189.72s |
| stump | DINO descriptor top-k25 weighted i0.50 + FastGS big | 27.2233 | 0.7903 | 0.2317 | 1,196,350 | 283.46s |
| stump | DINO descriptor top-k25 weighted i0.50 + staged target 1.15M | 25.9591 | 0.7401 | 0.2799 | 1,150,000 | 303.26s |

staged target 相对 FastGS big：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| -1.1719 | -0.0460 | +0.0393 | +87,719 | +113.53s | -2.3764 |

staged target 相对 `weighted i0.50`：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| -1.2642 | -0.0502 | +0.0482 | -46,350 | +19.80s | -2.5087 |

解读：

- staged target 成功把最终点数压到 1.15M，但质量严重下降，明显低于 FastGS big 和自然结束的 i0.50。
- 关键问题不是最终小幅裁剪，而是 iteration 9000 首次从 3.70M 直接裁到 1.17M，随后在 9500-14500 间连续大批量裁剪。这会破坏训练中期已经形成的结构，并且后续 30k 内没有恢复回来。
- 该结果把容量控制方向收窄为：不要早期大幅 staged prune；下一轮应测试更晚启动、更小步的 target 反馈，例如接近 i0.50 自然点数的 1.18M late target，或 staged prune 后立即加短恢复阶段。

## 2026-05-10 DINO descriptor top-k25 weighted i0.50 + late target 1.18M 高分辨率 stump 诊断

目标：尝试更晚启动的容量反馈，避免上一轮 iteration 9000 的大幅 staged prune。本轮设置 `target_gaussian_count=1180000`、`target_gaussian_stage_start=24000`、`target_gaussian_stage_interval=3000`。训练参数仍对齐 high-res stump 的 `fastgs_big` recipe。

命令：

```bash
source .venv/bin/activate && uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i050.yaml \
  -s datasets/mipnerf360/stump \
  -i images \
  -m output/0001/descriptor_topk025_weighted_i050_late1180k_big_stump/vfm_dinov2_descriptor_topk25_weighted_i050_late1180k_big_30k_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/stump_dinov2_vits14 \
  -r -1 \
  --dense 0.01 \
  --grad_abs_thresh 0.00015 \
  --target_gaussian_count 1180000 \
  --target_gaussian_staged \
  --target_gaussian_stage_margin 1.02 \
  --target_gaussian_stage_start 24000 \
  --target_gaussian_stage_interval 3000 \
  --target_gaussian_prune_order lowest_score
```

训练日志确认沿用 FastGS 原始大图规则：

```text
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
```

实际裁剪轨迹：

| 阶段 | 裁剪前 | 裁剪后 | 删除数量 |
|---|---:|---:|---:|
| final target prune | 3,322,015 | 1,180,000 | 2,142,015 |

stump 对照：

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| stump | FastGS big densify100 | 27.1310 | 0.7862 | 0.2406 | 1,062,281 | 189.72s |
| stump | DINO descriptor top-k25 weighted i0.50 + FastGS big | 27.2233 | 0.7903 | 0.2317 | 1,196,350 | 283.46s |
| stump | DINO descriptor top-k25 weighted i0.50 + late target 1.18M | 22.4721 | 0.6603 | 0.3091 | 1,180,000 | 425.10s |

late target 相对 FastGS big：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| -4.6589 | -0.1258 | +0.0685 | +117,719 | +235.38s | -7.6888 |

late target 相对 `weighted i0.50`：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| -4.7512 | -0.1300 | +0.0774 | -16,350 | +141.64s | -7.7379 |

解读：

- 这轮没有真正执行 late staged prune。原因是当前实现把 staged target 放在 densification 分支内部；当 `target_gaussian_stage_start=24000` 晚于 `densify_until_iter=15000` 时，staged target 不会触发。
- 最终实际变成一次性 target prune，从 3.32M 直接裁到 1.18M，质量比上一轮 1.15M staged target 更差。
- 结论：final target-prune 已明确不适合 high-res descriptor 容量控制；需要先修正 staged target 触发位置，允许其在 densification 结束后继续运行，才能验证真正的后期轻量反馈。

## 2026-05-10 DINO descriptor top-k25 weighted i0.50 + post-densify staged target 1.18M 高分辨率 stump 诊断

目标：在新增 `target_gaussian_stage_after_densify` 后，验证 staged target 能否在 densification 结束后继续触发，并观察“后期一次容量反馈”能否保住质量。本轮使用与 late target 相同的 1.18M target，只额外开启 `--target_gaussian_stage_after_densify`。

命令：

```bash
source .venv/bin/activate && uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i050.yaml \
  -s datasets/mipnerf360/stump \
  -i images \
  -m output/0001/descriptor_topk025_weighted_i050_postdensify1180k_big_stump/vfm_dinov2_descriptor_topk25_weighted_i050_postdensify1180k_big_30k_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/stump_dinov2_vits14 \
  -r -1 \
  --dense 0.01 \
  --grad_abs_thresh 0.00015 \
  --target_gaussian_count 1180000 \
  --target_gaussian_staged \
  --target_gaussian_stage_margin 1.02 \
  --target_gaussian_stage_start 24000 \
  --target_gaussian_stage_interval 3000 \
  --target_gaussian_stage_after_densify \
  --target_gaussian_prune_order lowest_score
```

训练日志确认沿用 FastGS 原始大图规则：

```text
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
```

实际裁剪轨迹：

| iteration | 裁剪前 | 裁剪后 | 删除数量 |
|---:|---:|---:|---:|
| 24000 | 3,316,554 | 1,203,600 | 2,112,954 |
| final | 1,176,624 | 1,176,624 | 0 |

stump 对照：

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| stump | FastGS big densify100 | 27.1310 | 0.7862 | 0.2406 | 1,062,281 | 189.72s |
| stump | DINO descriptor top-k25 weighted i0.50 + FastGS big | 27.2233 | 0.7903 | 0.2317 | 1,196,350 | 283.46s |
| stump | DINO descriptor top-k25 weighted i0.50 + post-densify staged target 1.18M | 26.5097 | 0.7595 | 0.2646 | 1,176,624 | 380.42s |

post-densify staged target 相对 FastGS big：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| -0.6213 | -0.0267 | +0.0240 | +114,343 | +190.70s | -1.4320 |

post-densify staged target 相对 `weighted i0.50`：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| -0.7137 | -0.0308 | +0.0329 | -19,726 | +96.97s | -1.4946 |

解读：

- 新增开关生效：iteration 24000 触发了 `Post-densify staged target Gaussian prune`，最终点数 1.18M，且没有再执行 final target prune。
- 相比 late target final-only 的 22.4721 / 0.6603 / 0.3091，post-densify staged target 明显恢复到 26.5097 / 0.7595 / 0.2646，说明训练结束前保留 6k iteration 恢复确实有帮助。
- 但 24k 时仍然是从 3.32M 一次性裁到 1.20M，质量仍显著低于 FastGS big 和自然 i0.50。下一步不应继续把裁剪推得更晚，而应在 densification 窗口内就限制容量增长，或把 staged target start 提前但 target 设得更高、更平滑，避免单次删除 2M 级 Gaussians。

## 2026-05-10 DINO descriptor top-k25 weighted i0.50 + staged target 1.45M 高分辨率 stump 诊断

目标：把容量反馈前移到 densification 窗口内，并放宽 target 到 1.45M，尝试避免 1.18M 方案的大幅质量破坏。本轮从 iteration 9000 开始，每 3000 step 执行 staged target，stage 上限为 `1.02 * 1.45M = 1.479M`，并保留 post-densify staged target 开关。

命令：

```bash
source .venv/bin/activate && uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i050.yaml \
  -s datasets/mipnerf360/stump \
  -i images \
  -m output/0001/descriptor_topk025_weighted_i050_staged1450k_big_stump/vfm_dinov2_descriptor_topk25_weighted_i050_staged1450k_big_30k_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/stump_dinov2_vits14 \
  -r -1 \
  --dense 0.01 \
  --grad_abs_thresh 0.00015 \
  --target_gaussian_count 1450000 \
  --target_gaussian_staged \
  --target_gaussian_stage_margin 1.02 \
  --target_gaussian_stage_start 9000 \
  --target_gaussian_stage_interval 3000 \
  --target_gaussian_stage_after_densify \
  --target_gaussian_prune_order lowest_score
```

训练日志确认沿用 FastGS 原始大图规则：

```text
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
```

实际裁剪轨迹：

| iteration | 裁剪前 | 裁剪后 | 删除数量 |
|---:|---:|---:|---:|
| 9000 | 3,703,938 | 1,479,000 | 2,224,938 |
| 12000 | 2,988,769 | 1,479,000 | 1,509,769 |
| 15000 | 2,929,055 | 1,479,000 | 1,450,055 |
| final | 1,262,974 | 1,262,974 | 0 |

stump 对照：

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| stump | FastGS big densify100 | 27.1310 | 0.7862 | 0.2406 | 1,062,281 | 189.72s |
| stump | DINO descriptor top-k25 weighted i0.50 + FastGS big | 27.2233 | 0.7903 | 0.2317 | 1,196,350 | 283.46s |
| stump | DINO descriptor top-k25 weighted i0.50 + staged target 1.45M | 26.5426 | 0.7636 | 0.2562 | 1,262,974 | 288.31s |

staged target 1.45M 相对 FastGS big：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| -0.5884 | -0.0226 | +0.0156 | +200,693 | +98.59s | -1.6212 |

staged target 1.45M 相对 `weighted i0.50`：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| -0.6807 | -0.0268 | +0.0245 | +66,624 | +4.85s | -1.4051 |

解读：

- 1.45M target 相比 1.18M post-densify 稍好，但仍明显低于 FastGS big 和自然 i0.50。
- 即使 target 放宽到 1.45M，iteration 9000 首次仍需删除 2.22M Gaussians，说明 high-res stump 的 descriptor densification 在早期已经出现 3M 级容量激增。
- 结论：`prune_to_target` 系列已经不适合作为 stump 的主容量解法。下一步应转向生成侧控制：按当前点数动态提高 densification 门槛、降低 `dense`、缩短 descriptor 参与窗口，或对 clone/split 分支分别加容量反馈。

## 2026-05-10 DINO descriptor top-k25 weighted i0.50 + warm8000 高分辨率 stump 诊断

目标：验证“早期使用 VFM descriptor，引发容量爆点前退出”是否能从生成侧缓解 high-res stump 的容量问题。本轮不启用 target prune，只覆盖 `--vfm_active_until_iter 8000`，让 8k 之后的 scorer 回到 FastGS/RGB 路径。

命令：

```bash
source .venv/bin/activate && uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i050.yaml \
  -s datasets/mipnerf360/stump \
  -i images \
  -m output/0001/descriptor_topk025_weighted_i050_warm8000_big_stump/vfm_dinov2_descriptor_topk25_weighted_i050_warm8000_big_30k_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/stump_dinov2_vits14 \
  -r -1 \
  --dense 0.01 \
  --grad_abs_thresh 0.00015 \
  --vfm_active_until_iter 8000
```

训练日志确认沿用 FastGS 原始大图规则：

```text
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
```

stump 对照：

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| stump | FastGS big densify100 | 27.1310 | 0.7862 | 0.2406 | 1,062,281 | 189.72s |
| stump | DINO descriptor top-k25 weighted i0.50 + FastGS big | 27.2233 | 0.7903 | 0.2317 | 1,196,350 | 283.46s |
| stump | DINO descriptor top-k25 weighted i0.50 + warm8000 | 26.8154 | 0.7780 | 0.2307 | 2,737,333 | 362.21s |

warm8000 相对 FastGS big：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| -0.3156 | -0.0082 | -0.0099 | +1,675,052 | +172.49s | -6.8294 |

warm8000 相对 `weighted i0.50`：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| -0.4079 | -0.0123 | -0.0010 | +1,540,983 | +78.76s | -6.5132 |

解读：

- warm8000 没有压住容量，最终达到 2.74M Gaussians，远高于 baseline 与自然 i0.50。
- 指标是混合结果：LPIPS 略优于 baseline 与 i0.50，但 PSNR/SSIM 均低，且容量代价极高。
- 该结果说明简单关闭后半段 VFM descriptor 不足以解决 stump 容量问题；8k 前的早期轨迹已经改变了后续 FastGS/RGB densification 与 pruning。下一步应改动生成侧阈值或 densification 门槛，而不是只调 active window。

## 2026-05-10 DINO descriptor top-k25 weighted i0.50 + densify budget 1.20M 高分辨率 stump 诊断

目标：验证新增的默认关闭 `densify_budget_count` 生成侧阈值门控是否比事后 target prune 更稳。本轮不删除已有点，而是在当前点数接近 1.20M 预算时，把 FastGS densification 的 metric 阈值从 5.0 逐步提高到 12.0。

命令：

```bash
source .venv/bin/activate && uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i050.yaml \
  -s datasets/mipnerf360/stump \
  -i images \
  -m output/0001/descriptor_topk025_weighted_i050_densifybudget1200k_big_stump/vfm_dinov2_descriptor_topk25_weighted_i050_densifybudget1200k_big_30k_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/stump_dinov2_vits14 \
  -r -1 \
  --dense 0.01 \
  --grad_abs_thresh 0.00015 \
  --densify_budget_count 1200000 \
  --densify_budget_start_ratio 0.90 \
  --densify_budget_max_metric_thresh 12.0
```

训练和渲染日志均确认沿用 FastGS 原始大图规则：

```text
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
```

stump 对照：

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| stump | FastGS big densify100 | 27.1310 | 0.7862 | 0.2406 | 1,062,281 | 189.72s |
| stump | DINO descriptor top-k25 weighted i0.50 + FastGS big | 27.2233 | 0.7903 | 0.2317 | 1,196,350 | 283.46s |
| stump | DINO descriptor top-k25 weighted i0.50 + densify budget 1.20M | 26.6899 | 0.7738 | 0.2371 | 2,378,891 | 315.61s |

densify budget 1.20M 相对 FastGS big：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| -0.4411 | -0.0124 | -0.0035 | +1,316,610 | +125.89s | -5.6382 |

densify budget 1.20M 相对 `weighted i0.50`：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| -0.5334 | -0.0166 | +0.0054 | +1,182,541 | +32.16s | -5.3219 |

解读：

- 生成侧阈值门控比 warm8000 的 2.74M 点有所降低，但最终仍有 2.38M 点，远高于 1.20M 预算和自然 i0.50 的 1.20M 结果。
- 质量低于 FastGS big 和自然 i0.50；LPIPS 仍略优于 baseline，但没有保住 i0.50 的感知质量。
- 结论：只把 metric 阈值从 5 提到 12 不能有效限制 high-res stump 的候选规模。下一步应采用更直接的候选数量门控，按剩余容量限制 clone/split 候选数；单纯继续调小 active window 或事后删除都不是主线。

## 2026-05-10 DINO descriptor top-k25 weighted i0.50 + candidate cap 1.20M 高分辨率 stump 诊断

目标：验证默认关闭的 `densify_budget_candidate_cap` 是否能在生成阶段直接按剩余容量限制 clone/split 候选数。与上一轮只抬高 metric 阈值不同，本轮在当前点数接近 1.20M 预算后直接截断候选集合，避免继续产生 2M 级冗余点。

命令：

```bash
source .venv/bin/activate && uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i050.yaml \
  -s datasets/mipnerf360/stump \
  -i images \
  -m output/0001/descriptor_topk025_weighted_i050_candidatecap1200k_big_stump/vfm_dinov2_descriptor_topk25_weighted_i050_candidatecap1200k_big_30k_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/stump_dinov2_vits14 \
  -r -1 \
  --dense 0.01 \
  --grad_abs_thresh 0.00015 \
  --densify_budget_count 1200000 \
  --densify_budget_start_ratio 0.90 \
  --densify_budget_max_metric_thresh 12.0 \
  --densify_budget_candidate_cap
```

训练和渲染日志均确认沿用 FastGS 原始大图规则：

```text
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
```

stump 对照：

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| stump | FastGS big densify100 | 27.1310 | 0.7862 | 0.2406 | 1,062,281 | 189.72s |
| stump | DINO descriptor top-k25 weighted i0.50 + FastGS big | 27.2233 | 0.7903 | 0.2317 | 1,196,350 | 283.46s |
| stump | DINO descriptor top-k25 weighted i0.50 + candidate cap 1.20M | 26.3268 | 0.7522 | 0.2712 | 1,076,427 | 212.35s |

candidate cap 1.20M 相对 FastGS big：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| -0.8042 | -0.0340 | +0.0306 | +14,146 | +22.63s | -1.6513 |

candidate cap 1.20M 相对自然 `weighted i0.50`：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| -0.8965 | -0.0381 | +0.0395 | -119,923 | -71.11s | -1.8572 |

解读：

- 候选数量门控已经有效控制容量：最终 1.076M Gaussians，接近 FastGS big 的 1.062M，也明显低于自然 i0.50 的 1.196M。
- 质量明显低于 FastGS big 和自然 i0.50，说明 1.20M 预算触发过早或过强，直接截断候选会压掉有效的 descriptor densification。
- 该结果是重要负例：问题不再是“控不住点数”，而是“控点太强时质量坍缩”。下一步应扫描更宽松的 `densify_budget_count=1.35M/1.45M`，寻找接近自然 i0.50 质量且仍低于 0.1M 级额外增长的拐点。

## 2026-05-10 DINO descriptor top-k25 weighted i0.50 + candidate cap 1.35M 高分辨率 stump 诊断

目标：在 1.20M candidate cap 已确认“控点有效但质量坍缩”后，放宽 `densify_budget_count` 到 1.35M，检查质量损失是否主要来自预算过紧。本轮仍使用相同 high-res stump 口径、相同 FastGS big 超参和相同 DINO descriptor 配置。

命令：

```bash
source .venv/bin/activate && uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i050.yaml \
  -s datasets/mipnerf360/stump \
  -i images \
  -m output/0001/descriptor_topk025_weighted_i050_candidatecap1350k_big_stump/vfm_dinov2_descriptor_topk25_weighted_i050_candidatecap1350k_big_30k_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/stump_dinov2_vits14 \
  -r -1 \
  --dense 0.01 \
  --grad_abs_thresh 0.00015 \
  --densify_budget_count 1350000 \
  --densify_budget_start_ratio 0.90 \
  --densify_budget_max_metric_thresh 12.0 \
  --densify_budget_candidate_cap
```

训练和渲染日志均确认沿用 FastGS 原始大图规则：

```text
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
```

stump 对照：

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| stump | FastGS big densify100 | 27.1310 | 0.7862 | 0.2406 | 1,062,281 | 189.72s |
| stump | DINO descriptor top-k25 weighted i0.50 + FastGS big | 27.2233 | 0.7903 | 0.2317 | 1,196,350 | 283.46s |
| stump | DINO descriptor top-k25 weighted i0.50 + candidate cap 1.20M | 26.3268 | 0.7522 | 0.2712 | 1,076,427 | 212.35s |
| stump | DINO descriptor top-k25 weighted i0.50 + candidate cap 1.35M | 26.3994 | 0.7570 | 0.2637 | 1,199,833 | 218.61s |

candidate cap 1.35M 相对 FastGS big：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| -0.7316 | -0.0292 | +0.0231 | +137,552 | +28.89s | -1.6803 |

candidate cap 1.35M 相对自然 `weighted i0.50`：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| -0.8239 | -0.0333 | +0.0320 | +3,483 | -64.85s | -1.6537 |

解读：

- 1.35M candidate cap 最终 1.200M Gaussians，几乎等于自然 i0.50 的 1.196M，并且训练时间比自然 i0.50 少 64.85s。
- 质量只比 1.20M cap 略有恢复，仍显著低于 FastGS big 和自然 i0.50；因此主要问题不只是“最终点数太少”，而是 cap 改变了 densification 过程中保留候选的空间/结构分布。
- 该结果说明当前按全局 `importance_score` top-k 截断候选不是可用容量解法。1.45M 边界检查继续验证这一判断：如果放宽后仍不能恢复质量，就应放弃全局 candidate cap，转向分视角/分区域配额、分 clone/split 配额、或 Depth Anything 几何边界辅助。

## 2026-05-10 DINO descriptor top-k25 weighted i0.50 + candidate cap 1.45M 高分辨率 stump 边界检查

目标：在 1.35M candidate cap 已经把最终点数恢复到自然 i0.50 水平但质量仍未恢复后，继续放宽 `densify_budget_count` 到 1.45M，检查全局候选截断是否存在可用质量-容量拐点。本轮仍使用 high-res stump、FastGS big 超参、FastGS 原始大图自动缩放规则和相同 DINO descriptor 配置。

命令：

```bash
source .venv/bin/activate && uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only_topk025_weighted_i050.yaml \
  -s datasets/mipnerf360/stump \
  -i images \
  -m output/0001/descriptor_topk025_weighted_i050_candidatecap1450k_big_stump/vfm_dinov2_descriptor_topk25_weighted_i050_candidatecap1450k_big_30k_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/stump_dinov2_vits14 \
  -r -1 \
  --dense 0.01 \
  --grad_abs_thresh 0.00015 \
  --densify_budget_count 1450000 \
  --densify_budget_start_ratio 0.90 \
  --densify_budget_max_metric_thresh 12.0 \
  --densify_budget_candidate_cap
```

训练和渲染日志均确认沿用 FastGS 原始大图规则：

```text
[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.
```

stump 对照：

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数量 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| stump | FastGS big densify100 | 27.1310 | 0.7862 | 0.2406 | 1,062,281 | 189.72s |
| stump | DINO descriptor top-k25 weighted i0.50 + FastGS big | 27.2233 | 0.7903 | 0.2317 | 1,196,350 | 283.46s |
| stump | DINO descriptor top-k25 weighted i0.50 + candidate cap 1.20M | 26.3268 | 0.7522 | 0.2712 | 1,076,427 | 212.35s |
| stump | DINO descriptor top-k25 weighted i0.50 + candidate cap 1.35M | 26.3994 | 0.7570 | 0.2637 | 1,199,833 | 218.61s |
| stump | DINO descriptor top-k25 weighted i0.50 + candidate cap 1.45M | 26.4697 | 0.7588 | 0.2611 | 1,283,793 | 226.10s |

candidate cap 1.45M 相对 FastGS big：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| -0.6613 | -0.0274 | +0.0205 | +221,512 | +36.38s | -1.8975 |

candidate cap 1.45M 相对自然 `weighted i0.50`：

| ΔPSNR | ΔSSIM | ΔLPIPS | ΔGaussian | Δ训练时间 | QCGI |
|---:|---:|---:|---:|---:|---:|
| -0.7536 | -0.0315 | +0.0294 | +87,443 | -57.36s | -1.6189 |

解读：

- 1.45M candidate cap 比 1.35M 多保留约 83,960 个 Gaussians，PSNR 只恢复 +0.0703，SSIM 只恢复 +0.0018，LPIPS 只恢复 -0.0026，仍显著低于 FastGS big 和自然 i0.50。
- 相对自然 i0.50，1.45M cap 反而多 87,443 个 Gaussians，却低 -0.7536 PSNR、-0.0315 SSIM、LPIPS 差 +0.0294。这说明问题不是最终容量不足，而是全局 top-k 截断在训练期改变了有效候选分布。
- 1.20M、1.35M、1.45M 三个点共同收敛到同一结论：全局 candidate cap 可以控制点数，但不是可用的质量-容量解法。下一轮不再继续扫描全局预算，应转向“保留分布”的生成侧控制，例如按空间区域/视角/clone-split 类型分配候选配额，或引入 Depth Anything 几何边界先验来决定哪些新增 GS 值得保留。

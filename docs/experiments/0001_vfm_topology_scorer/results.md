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

数据集：`datasets/mipnerf360/bicycle`，test split，`-r 8`。先用 620-step 验证链路健康，再跑 30,000 iterations 完整对照。

| 产物 | Backend | Metric map | Iterations | PSNR | SSIM | LPIPS | 训练时间 | Gaussian 数量 | 输出大小 | 备注 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `output/0001/vfm_dinov2_token_edge_topk015_bicycle_620_r8` | `dinov2_token_edge_l1` | top-k 15% | 620 | 20.8432 | 0.4752 | 0.5460 | 2.74s | 61,555 | 17M | 触发 token-edge top-k scoring 和 densification |
| `output/0001/vfm_dinov2_token_edge_topk015_bicycle_30k_r8` | `dinov2_token_edge_l1` | top-k 15% | 30,000 | 27.0223 | 0.8322 | 0.1810 | 140.40s | 464,998 | 136M | 完整对照，质量低于默认 DINO token-edge，但更省点更快 |
| `output/0001/vfm_dinov2_token_edge_topk025_bicycle_30k_r8` | `dinov2_token_edge_l1` | top-k 25% | 30,000 | 27.0636 | 0.8354 | 0.1748 | 146.76s | 497,328 | 144M | 当前 bicycle 30k 质量最佳 |
| `output/0001/vfm_dinov2_token_edge_topk025_budget490832_staged105_bicycle_30k_r8` | `dinov2_token_edge_l1` | top-k 25%, staged 490,832 | 30,000 | 27.0001 | 0.8286 | 0.1887 | 146.81s | 453,505 | 133M | 预算更低，但质量明显回落 |
| `output/0001/vfm_dinov2_token_edge_topk025_budget490832_final_bicycle_30k_r8` | `dinov2_token_edge_l1` | top-k 25%, final 490,832 | 30,000 | 26.8466 | 0.8244 | 0.1858 | 141.91s | 490,832 | 142M | 仅最终裁剪 6,723 个点，质量明显回落 |
| `output/0001/vfm_dinov2_token_edge_topk025_rgb_only_bicycle_30k_r8` | `dinov2_token_edge_l1` | top-k 25%, `rgb_only` | 30,000 | 26.9340 | 0.8236 | 0.1981 | 139.06s | 411,539 | 123M | 预算贴近 cadence control，但质量不构成清晰正向 |

解读：

- 训练预检、train、render 和 metrics 均通过，说明 token-edge top-k 配置可直接进入 30k 完整对照。
- 620-step 指标只说明集成健康；30k 完整结果相对原始 baseline 提升 +0.3191 PSNR、+0.0255 SSIM、LPIPS 改善 -0.0468，属于清晰正向。
- 相比 `fastgs_densify100` cadence control，token-edge top-k 15% 提升 +0.0936 PSNR、+0.0081 SSIM、LPIPS 改善 -0.0154，同时多 52,920 个 Gaussians，训练少 25.49s。
- 相比默认 DINO token-edge，top-k 15% 少 25,834 个 Gaussians，训练少 25.71s，但质量下降 -0.0354 PSNR、-0.0023 SSIM、LPIPS 差 +0.0043。它是更高效的 DINO token-edge 变体，但没有刷新质量上界。
- top-k 25% 刷新了当前 bicycle 30k 质量上界。相比默认 DINO token-edge，它提升 +0.0059 PSNR、+0.0009 SSIM、LPIPS 改善 -0.0019，但多 6,496 个 Gaussians；相比原始 baseline，它提升 +0.3604 PSNR、+0.0287 SSIM、LPIPS 改善 -0.0530。
- top-k 25% staged 490,832 从 iteration 7500 到 14500 触发 15 次 staged pruning，最终自然落到 453,505 个 Gaussians，低于 target 因而跳过最终裁剪。它相比 top-k 25% 完整对照少 43,823 个点，但质量下降 -0.0634 PSNR、-0.0068 SSIM、LPIPS 差 +0.0139。
- top-k 25% final 490,832 不做中期裁剪，训练结束时从 497,555 个 Gaussians 只裁到 490,832，实际删除 6,723 个点。但它相比 top-k 25% 完整对照下降 -0.2170 PSNR、-0.0110 SSIM、LPIPS 差 +0.0110；相比默认 DINO token-edge 也低 -0.2111 PSNR、-0.0101 SSIM、LPIPS 差 +0.0091。
- final 490,832 仍优于原始 baseline（+0.1434 PSNR、+0.0177 SSIM、LPIPS 改善 -0.0420），但相对 `fastgs_densify100` cadence control 变成 PSNR 更低、SSIM 基本持平、LPIPS 更好。这个负例说明当前 final target-prune 的排序即使只裁约 1.35% Gaussians，也会误删对结构质量敏感的点。
- top-k 25% `rgb_only` 关闭直接 VFM densification，只保留 VFM support/pruning 侧影响。它最终 411,539 个 Gaussians，几乎贴住 `fastgs_densify100` 的 412,078；相比 cadence control，PSNR 只高 +0.0053，SSIM 低 -0.0005，LPIPS 差 +0.0017。因此“只做 prune-protect / support 重排”可以控制预算，但没有保住 top-k 25% 完整对照的质量收益。
- staged 490,832、final 490,832 和 `rgb_only` 分别暴露了三类问题：中期反复压 cap 会损伤结构生长，终局一次性低分裁剪排序不够可靠，完全关闭 VFM densification 又会交回主要质量收益。因此预算约束下一步不应继续依赖最终硬裁剪，而应探索温和的预算感知 densification，例如 support-normalized score、partial VFM importance，或只保护高置信 VFM 区域而不是全量放大。
- 这一路径不引入在线 DINO inference，成本明显低于 descriptor 系列。当前 0001 在 bicycle 上的最佳质量结论仍定为 top-k 25% 完整对照；预算方向则保留 top-k 15% 作为更高效但略弱的选择，490,832 staged/final 两个预算 run 均作为负例。

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

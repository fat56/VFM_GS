# 0001 实验结果

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

完整探测使用 bicycle 30k `-r 8`，`cached_edge_l1`，目标点数为 baseline 30k 的 240,394；不启用 staged pruning，只在最终 low-score target prune 后恢复 4,096 步。

| 产物 | 后端 | 目标策略 | 恢复步数 | PSNR | SSIM | LPIPS | 训练时间 | Gaussian 数量 | 输出大小 | 备注 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `output/0001/vfm_cached_edge_budget240394_lowscore_finetune4096_bicycle_30k_r8` | `cached_edge_l1` | 最终裁剪最低分后恢复 | 4,096 | 26.0163 | 0.7760 | 0.2580 | 156.29s | 240,394 | 138M | 411,345 -> 240,394，保存 `ours_34096` |

对比：

- 相比 final-only low-score target prune，恢复训练提升 +2.2434 PSNR、+0.0453 SSIM、-0.0105 LPIPS，说明最终裁剪后的结构损伤可以被一部分恢复。
- 相比 240k staged edge，post-prune fine-tune 的 PSNR/SSIM 更高：+0.2184 PSNR、+0.0013 SSIM；但 LPIPS 更差 +0.0043。
- 相比 baseline 30k，仍低 -0.6869 PSNR、-0.0307 SSIM、+0.0302 LPIPS。严格 240k 预算下，单次最终裁剪加 4,096 步恢复还不能构成正向结果。
- 由于默认 `optimizer_step` 在 20k 之后每 64 iteration 才更新一次，4,096 步恢复实际只有约 64 次参数更新。这可能解释了恢复有限；下一版若继续这条路，应增加专门的 dense recovery step schedule 或在裁剪前后采用更平滑的 staged+fine-tune 组合。
- 本组输出包含 `iteration_30000` 和 `iteration_34096` 两份 PLY，因此目录大小为 138M；只看恢复后 PLY 约 57M。

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

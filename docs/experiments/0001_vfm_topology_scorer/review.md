# 0001 复盘

## 当前决策

继续保留 `vfm_topology_scorer` 作为 v1 集成路径。`mock_l1` 验证打分链路，`cached_edge_l1` 验证离线缓存契约，`dinov2_token_edge_l1` 验证训练可以消费真实 DINOv2 patch-token cache。当前 DINOv2 路径仍是保守的 token-edge projection，不是最终语义特征打分器。后续决策门槛以 30k full run 为准，不再用短跑 smoke 指标判断质量。

## 发现

- `vfm_topology_scorer` 已返回与 `fastgs_photometric` 一致的 `(importance_score, pruning_score)` 契约。
- 当 `vfm_use_albedo_sh0=true` 时，打分器使用 SH0 渲染，生成后端 pixel error map，再阈值化为 `metric_map`，最后让 `render_fastgs(..., get_flag=True)` 累积每个 Gaussian 的命中次数。
- `fastgs_photometric` 的归一化已加固，避免 metric map 或 pruning score 动态范围为 0 时产生 NaN。
- 2026-04-28 bicycle smoke 中，mock v1 与 baseline 基本指标持平：PSNR 20.3459 vs 20.3464，SSIM 0.4294 vs 0.4294，LPIPS 0.6010 vs 0.6021。
- mock v1 产生 78,375 个 Gaussians，baseline 为 78,633，说明保守融合在短跑验证中没有造成点数膨胀。
- `vfm_gs.cli.build_vfm_cache` 现在会写 manifest，并以 `Camera.image_name` 为 key，记录 backend name、source shape、cache shape、dtype、normalization 和每个 entry 的 checksum。
- `cached_edge_l1` 完成同样的 train/render/metrics smoke 路径：PSNR 20.3265，SSIM 0.4291，LPIPS 0.6005，78,605 个 Gaussians。
- 训练输出的 `cfg_args` 现在除 model 参数外，还记录 optimization 和 pipeline 参数，因此每个 run directory 都能恢复 scorer/backend 设置。
- `npz_uint8` compact storage 将 bicycle edge cache 从约 189MB 降到 35MB，且 `vfm_gs.cli.validate_vfm_cache` 通过 checksum/source-image 校验。
- compact run 完成 train/render/metrics，结果为 78,682 个 Gaussians，PSNR 20.1588，SSIM 0.4275，LPIPS 0.5993。PSNR 降幅更大，说明 edge quantization 可能改变早期 densification 决策。
- cached backends 现在会在 Scene 构建前执行训练 preflight。好 cache 提前通过，缺失 manifest 会在 camera loading 或 densification 前失败。
- `vfm_gs.cli.vfm_backend_probe` 已记录运行时兼容性并估算 DINOv2 cache 体积。在这台机器上，cache width 518-640px 的 ViT-S/14 和 ViT-B/14 是最稳妥的真实 VFM 候选。
- `vfm_gs.cli.build_vfm_cache` 已支持可选的 `dinov2_vits14` 和 `dinov2_vitb14` cache 生成。DINOv2 cache 保存 L2 归一化 patch-token grid，`feature=dinov2_patchtokens`。
- 4-image ViT-S/14 smoke cache 在 `max_width=224` 下写出 4 个 `npy_float16` entries，首个 entry 形状为 `10x16x384`，总大小约 500K，并通过 `vfm_gs.cli.validate_vfm_cache`。
- 探测时远程 `torch.hub` listing 触发 GitHub 限流，但本地 clone 到 `output/0001/external/dinov2` 后加载 pretrained weights 可用。builder 在远程 `torch.hub` 失败时会提示传入 `--dinov2_repo`。
- PyTorch 1.12.1 缺少当前 DINOv2 期望的公开 SDPA API，因此 cache builder 为可选 DINOv2 cache 路径增加了一个隔离的兼容 shim，底层使用 1.12 私有 attention 函数。
- full-scene ViT-S/14 cache 在 `max_width=224` 下向 `output/0001/vfm_cache/bicycle_dinov2_vits14` 写入 194 个 `npy_float16` entries，构建耗时 15s，大小约 24M，并通过校验。
- `dinov2_token_edge_l1` 将 cached DINO patch tokens 投影成标量 token-edge topology map，将 SH0 渲染亮度边缘汇聚到同一 grid，并返回上采样后的 pixel error map，供现有 metric-map scorer 路径使用。
- DINO token-edge smoke 完成 train/render/metrics：PSNR 20.2913，SSIM 0.4272，LPIPS 0.6006，77,761 个 Gaussians，训练时间 1.72s，测试帧渲染 410.94 FPS。
- DINO scorer preflight 现在接受 DINO cache manifests（`dinov2_vits14` 或 `dinov2_vitb14`），并会在 Scene 构建前拒绝 cache feature/backend 不匹配。
- matched 30k `-r 8` ablation 已成为主质量信号。baseline 达到 PSNR 26.7032，SSIM 0.8067，LPIPS 0.2278，240,394 个 Gaussians，334.36 FPS。
- compact cached edge 将 30k run 提升到 PSNR 26.8864，SSIM 0.8229，LPIPS 0.1972，但点数增长到 408,925，FPS 为 196.43。
- DINO token-edge 给出最佳 30k 指标：PSNR 27.0577，SSIM 0.8345，LPIPS 0.1767，但点数达到 490,832，FPS 为 193.46。
- full run 结果反转了短跑印象：DINO token-edge 在 220 iteration 看起来中性或略差，但在正常 densification 有时间发挥后成为指标最强变体。
- 首个 budget-control probe 使用 `vfm_loss_thresh=0.75` 和 `vfm_weight=0.10`，未能让 VFM 点数接近 baseline。edge 仍为 409,028 个 Gaussians；DINO 降到 422,506，但仍比 baseline 高约 76%。
- budget-control probe 仍保持较好指标，但低于默认 DINO：DINO t075/w010 达到 PSNR 26.9586，SSIM 0.8258，LPIPS 0.1935。
- `vfm_importance_weight` 现在将 densification 强度和 pruning fusion 分离。默认值为 `1.0`，保持向后兼容。
- `vfm_importance_weight=0.25` 时，edge 达到 PSNR 26.9439，SSIM 0.8244，LPIPS 0.1958，413,301 个 Gaussians；DINO 达到 PSNR 26.9261，SSIM 0.8259，LPIPS 0.1928，418,073 个 Gaussians。
- 显式 importance weighting 让默认 DINO 点数降低约 14.8%，但仍未达到 baseline-like budget。edge 仍稳定在 400k Gaussians 以上。
- `vfm_importance_mode` 现在支持 `max`、`weighted` 和 `rgb_only`。默认 `max` 保持旧行为。
- 30k `rgb_only` ablation 已完成。edge 达到 PSNR 26.9574，SSIM 0.8243，LPIPS 0.1961，413,914 个 Gaussians；DINO 达到 PSNR 26.9310，SSIM 0.8237，LPIPS 0.1962，413,223 个 Gaussians。
- `rgb_only` 未能恢复 baseline-like Gaussian count。单独的 VFM pruning-score fusion 仍可保留或重塑足够多的点，让最终点数保持在 baseline 的约 1.72x。
- 首个 `target_gaussian_count` probe 精确匹配 baseline 点数，但裁掉了最高分 Gaussians，导致质量崩溃：edge PSNR 11.1494，DINO PSNR 10.2215。
- target-count pruning 现在优先删除最低 score Gaussians。相比 high-score removal，这更符合 bulk budget control 的 score 语义。
- low-score target-count 30k runs 精确匹配 baseline Gaussian count，但质量仍低于 baseline：edge PSNR 23.7729 / SSIM 0.7307 / LPIPS 0.2685；DINO PSNR 23.5571 / SSIM 0.7087 / LPIPS 0.2797。
- 因此，一次性 final prune 不是有效的 budget-matched 质量对比方式。它删除了过多已收敛结构，却不给模型恢复机会。
- staged target-count 30k runs 在保持精确 baseline count 的同时，相比 one-shot final prune 明显恢复质量。edge 达到 PSNR 25.7979，SSIM 0.7747，LPIPS 0.2537；DINO 达到 PSNR 25.3529，SSIM 0.7727，LPIPS 0.2493。
- staged 240k 仍低于 baseline 质量，说明当前 VFM 变体下严格 baseline-count matching 过于激进。
- 300k staged target 继续恢复质量。edge 达到 PSNR 26.2327，SSIM 0.7866，LPIPS 0.2412；DINO 达到 PSNR 25.9089，SSIM 0.7925，LPIPS 0.2291。
- DINO 300k 的 LPIPS 几乎追平 baseline，但 PSNR/SSIM 仍落后。edge 300k 的 PSNR 更接近，但 LPIPS 更差。
- 350k staged target 产出首个 budget-controlled 正向结果。edge 在 340,283 个 Gaussians 下达到 PSNR 26.7788，SSIM 0.8089，LPIPS 0.2206，三项指标均超过 baseline。
- DINO 350k 在 350,000 个 Gaussians 下达到 PSNR 26.3634，SSIM 0.8033，LPIPS 0.2188。它的 LPIPS 优于 baseline，但 PSNR/SSIM 仍低。
- edge 正向结果已在 garden 复验。garden baseline 为 PSNR 28.7051，SSIM 0.8889，LPIPS 0.1134，196,201 个 Gaussians；staged edge 为 PSNR 28.9411，SSIM 0.8964，LPIPS 0.1007，248,471 个 Gaussians。

## 局限

- `mock_l1` 有意不是一个真实视觉基础模型信号。
- `cached_edge_l1` 也只是 proxy；它主要测试 cache 机制和 edge-alignment 行为。
- `dinov2_token_edge_l1` 消费 DINO patch tokens，但比较的是标量 topology projection，而不是完整语义特征向量。
- 220-iteration smoke run 只验证集成健康，不验证最终重建质量。既然 30k runs 已足够便宜，后续不应用短跑结果选择 scorer。
- compact storage 有助于节省磁盘，但 `npz_uint8` 尚未证明 metric-neutral。float32 与 compact cache 变体仍应保留用于 ablation。
- 当前 DINO cache 构建于 `max_width=224`；完整 `max_width=518` 或 `640` 的缓存时间、磁盘占用和 scorer 行为仍需测量。
- 最好的 30k DINO 结果不是 budget-controlled：它使用了约 2.04x baseline Gaussian count。下一步必须把 feature-signal 质量和允许更密 reconstruction 的收益拆开。
- 现有 knobs 不能提供完整 budget control。`vfm_weight` 影响 pruning fusion，`vfm_importance_weight` 影响直接 VFM densification 强度，`vfm_importance_mode=rgb_only` 可以关闭直接 VFM densification，但都不能匹配 baseline point count。
- `target_gaussian_count` 适合作为 count-control 诊断，但 baseline-sized budget 下的一次性 final pruning 破坏性太强。
- staged budget control 更健康，edge 已在 bicycle 和 garden 产生 budget-aware 正向结果。DINO token-edge 仍需要更好的 projection 或 recovery training，才能成为预算高效方案。

## 下一版计划

1. 增加 no-effect control，例如 `vfm_weight=0` + `vfm_importance_mode=rgb_only`，测量不改变 pruning 或 densification 决策时的 VFM scorer 开销。
2. 增加 final target pruning 后的 post-prune fine-tune 选项，对比 staged cap 和 final-prune-plus-fine-tune。
3. 用 patch-descriptor scorer 替换或增强 `dinov2_token_edge_l1`，因为当前 token-edge projection 在 budget control 下 PSNR/SSIM 弱于 edge。
4. 再跑一个 scene 的 staged-budget edge，然后再考虑把它从 v1 positive control 提升为更强结论。
5. 保持 30k `-r 8` 作为最小质量门槛；220 iteration 只用于代码变更后的 smoke checks。

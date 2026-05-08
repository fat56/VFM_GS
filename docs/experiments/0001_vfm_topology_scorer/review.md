# 0001 复盘

## 当前决策

继续保留 `vfm_topology_scorer` 作为 v1 集成路径。`mock_l1` 验证打分链路；`cached_edge_l1` 固化为早期 proxy 正向控制组，但 Tandt/DB 全场景评估显示它存在明显跨数据集差异：`db` 正向，`tandt` 负向。`dinov2_token_edge_l1` 已验证训练可以消费真实 DINOv2 patch-token cache，并且 `top-k 25% + importance_weight=0.50` 完成 MipNeRF360 全 9 场景评估，平均 PSNR 28.8577、SSIM 0.8666、LPIPS 0.1385，超过 baseline 与 cached-edge v1，可作为 0001 的 DINO 质量候选；但平均 Gaussian 数量比 baseline 多约 52.1%，还不是预算高效结论。`weighted + importance_weight=0.50` 已完成 MipNeRF360 全 9 场景复验，平均 PSNR 28.8505、SSIM 0.8660、LPIPS 0.1397，平均 254,736 个 Gaussians、训练 137.60s；它相对普通 i0.50 平均少 8,836 个点、训练少 2.87s，质量只小幅回落，因此是当前全场景预算效率候选。`weighted + importance_weight=0.75/0.90` 均已补齐 MipNeRF360 全 9 场景，固定均值都不作为默认档。严格三档 `quality_pick` 平均为 PSNR 28.8641、SSIM 0.8665、LPIPS 0.1392、257,326 个 Gaussians，比 fixed weighted i0.50 三项质量均正向且只多 2,589 个点；`QCGI pick` 平均为 28.8641、0.8667、0.1388、255,822 个 Gaussians，只多 1,086 个点。这个结果把 i0.75/i0.90 收束为“场景选择质量档”，也确认单个负例不应直接否决方法，应看数据集平均收益。`adaptive_weighted + quadratic 430k` 在 bicycle 上正向，但 treehill 第二场景复验未通过，后续不升级为主线。`dinov2_descriptor_cosine` 则打通了在线渲染图 descriptor 与 GT cache descriptor 的语义比较路径，但预算对齐后未转正。后续决策门槛以 30k 完整 run 和多场景平均为准，不再用短程验证指标判断质量。

Tandt 的 DINO weighted i0.50/i0.75/i0.90 跨数据集复验显示：i0.50 平均为 PSNR 25.7519、SSIM 0.9346、LPIPS 0.0575，较 cached-edge v1 分别改善 +0.1721、+0.0031、-0.0033；但它仍低于 Tandt baseline，平均差距为 -0.2032 PSNR、-0.0031 SSIM、LPIPS 差 +0.0035。继续提高权重没有帮助，i0.75 均值降到 25.6201 / 0.9328 / 0.0585，i0.90 降到 25.5329 / 0.9323 / 0.0611，训练时间也明显拉长。因此 Tandt 当前只把 DINO weighted 记录为“修复 cached-edge Tandt 负例的一部分”，默认策略应回退 baseline。

DINO weighted i0.50 + 自动容量下限进一步排除了“最终点数太少”这一解释。该诊断把 Tandt 两个场景最终 Gaussian 数量拉回 baseline 均值 50,370，但平均结果只有 PSNR 25.6078、SSIM 0.9324、LPIPS 0.0610，低于原始 DINO weighted i0.50 和 baseline。训练日志显示早期 staged target pruning 已在 1,000 iteration 附近发生大幅裁剪，因此容量下限只能防止最终过稀，不能修复已经被改写的训练轨迹。取消 staged target、仅保留容量下限后，平均结果为 25.6430 / 0.9341 / 0.0566、50,370 个 Gaussians；它相对 staged 版本修复了 LPIPS/SSIM，但 PSNR 仍低于原始 DINO weighted i0.50 和 baseline。下一步若继续处理 Tandt，应降低早期 VFM 介入，或把策略改成场景级 baseline 回退。

DB 的 DINO weighted 多档复验则给出相反信号。i0.50 平均为 30.3603 / 0.9360 / 0.0641，相对 baseline 正向但低于 cached-edge；i0.75 提升到 30.5446 / 0.9358 / 0.0633；i0.90 进一步达到 30.6074 / 0.9376 / 0.0620，平均 63,006 个 Gaussians。i0.90 相对 DB baseline 平均 +0.4894 PSNR、+0.0051 SSIM、LPIPS -0.0038；相对 DB cached-edge v1 也有 +0.0443 PSNR、+0.0015 SSIM、LPIPS -0.0017。因此 DB 上高权重 DINO weighted 是新的正向质量档，但它不应外推到 Tandt。

跨数据集选择汇总脚本已扩展到 13 个场景的 baseline、cached-edge v1 和 DINO weighted i0.50/i0.75/i0.90。固定三档中，i0.50/i0.75/i0.90 均值分别为 28.6061 / 0.8873 / 0.1154、28.6066 / 0.8872 / 0.1153、28.5919 / 0.8872 / 0.1154，说明固定权重不是主解。场景级 `validated_policy` 与 PSNR oracle 一致，均值提升到 28.6981 / 0.8872 / 0.1179、178,903 个 Gaussians；相对 baseline 为 +0.2350 PSNR、+0.0075 SSIM、LPIPS -0.0127。`qcgi_pick` 为 28.6930 / 0.8881 / 0.1147、188,542 个 Gaussians，更偏 SSIM/LPIPS 和容量综合收益。逐场景 PSNR 最优分布更新为：9 个场景选 DINO weighted，1 个场景选 cached-edge，3 个场景选 baseline。下一版应把 0001 收束为“自动场景选择/回退 + QCGI 容量约束”，而不是继续寻找单一固定后端。

## 发现

- `vfm_topology_scorer` 已返回与 `fastgs_photometric` 一致的 `(importance_score, pruning_score)` 契约。
- 当 `vfm_use_albedo_sh0=true` 时，打分器使用 SH0 渲染，生成后端 pixel error map，再阈值化为 `metric_map`，最后让 `render_fastgs(..., get_flag=True)` 累积每个 Gaussian 的命中次数。
- `fastgs_photometric` 的归一化已加固，避免 metric map 或 pruning score 动态范围为 0 时产生 NaN。
- 2026-04-28 bicycle 快速验证中，mock v1 与 baseline 基本指标持平：PSNR 20.3459 vs 20.3464，SSIM 0.4294 vs 0.4294，LPIPS 0.6010 vs 0.6021。
- mock v1 产生 78,375 个 Gaussians，baseline 为 78,633，说明保守融合在短跑验证中没有造成点数膨胀。
- `vfm_gs.cli.build_vfm_cache` 现在会写 manifest，并以 `Camera.image_name` 为 key，记录 backend name、source shape、cache shape、dtype、normalization 和每个 entry 的 checksum。
- `cached_edge_l1` 完成同样的 train/render/metrics 快速验证路径：PSNR 20.3265，SSIM 0.4291，LPIPS 0.6005，78,605 个 Gaussians。
- 训练输出的 `cfg_args` 现在除 model 参数外，还记录 optimization 和 pipeline 参数，因此每个 run directory 都能恢复 scorer/backend 设置。
- `npz_uint8` compact storage 将 bicycle edge cache 从约 189MB 降到 35MB，且 `vfm_gs.cli.validate_vfm_cache` 通过 checksum/source-image 校验。
- compact run 完成 train/render/metrics，结果为 78,682 个 Gaussians，PSNR 20.1588，SSIM 0.4275，LPIPS 0.5993。PSNR 降幅更大，说明 edge quantization 可能改变早期 densification 决策。
- cached backends 现在会在 Scene 构建前执行训练 preflight。好 cache 提前通过，缺失 manifest 会在 camera loading 或 densification 前失败。
- `vfm_gs.cli.vfm_backend_probe` 已记录运行时兼容性并估算 DINOv2 cache 体积。在这台机器上，cache width 518-640px 的 ViT-S/14 和 ViT-B/14 是最稳妥的真实 VFM 候选。
- `vfm_gs.cli.build_vfm_cache` 已支持可选的 `dinov2_vits14` 和 `dinov2_vitb14` cache 生成。DINOv2 cache 保存 L2 归一化 patch-token grid，`feature=dinov2_patchtokens`。
- 4-image ViT-S/14 快速验证 cache 在 `max_width=224` 下写出 4 个 `npy_float16` entries，首个 entry 形状为 `10x16x384`，总大小约 500K，并通过 `vfm_gs.cli.validate_vfm_cache`。
- 探测时远程 `torch.hub` listing 触发 GitHub 限流，但本地 clone 到 `output/0001/external/dinov2` 后加载 pretrained weights 可用。builder 在远程 `torch.hub` 失败时会提示传入 `--dinov2_repo`。
- PyTorch 1.12.1 缺少当前 DINOv2 期望的公开 SDPA API，因此 cache builder 为可选 DINOv2 cache 路径增加了一个隔离的兼容 shim，底层使用 1.12 私有 attention 函数。
- full-scene ViT-S/14 cache 在 `max_width=224` 下向 `output/0001/vfm_cache/bicycle_dinov2_vits14` 写入 194 个 `npy_float16` entries，构建耗时 15s，大小约 24M，并通过校验。
- `dinov2_token_edge_l1` 将 cached DINO patch tokens 投影成标量 token-edge topology map，将 SH0 渲染亮度边缘汇聚到同一 grid，并返回上采样后的 pixel error map，供现有 metric-map scorer 路径使用。
- DINO token-edge 快速验证完成 train/render/metrics：PSNR 20.2913，SSIM 0.4272，LPIPS 0.6006，77,761 个 Gaussians，训练时间 1.72s，测试帧渲染 410.94 FPS。
- DINO scorer preflight 现在接受 DINO cache manifests（`dinov2_vits14` 或 `dinov2_vitb14`），并会在 Scene 构建前拒绝 cache feature/backend 不匹配。
- `dinov2_descriptor_cosine` 已落地。它在 scorer 节点对 SH0 渲染图在线运行 DINOv2，再与 GT cache patch tokens 做 cosine distance，并上采样为 pixel error map。
- descriptor 快速验证完成 80-step 和 220-step train/render/metrics。220-step bicycle 结果为 PSNR 20.0193，SSIM 0.4233，LPIPS 0.6018，77,060 个 Gaussians，训练时间 2.55s。
- descriptor 快速验证指标略低于 token-edge 快速验证，但它是更贴近 proposal 语义特征误差的真实路径。下一步不应仅看短跑质量，而应先调 descriptor 阈值和 cache 分辨率，再进入 30k。
- descriptor 阈值小网格已完成。`vfm_loss_thresh=0.30/0.35/0.40/0.65` 均完成 220-step train/render/metrics，其中 0.35 最好：PSNR 20.2897，SSIM 0.4287，LPIPS 0.5993，79,120 个 Gaussians。
- 0.30 达到 PSNR 20.2853，SSIM 0.4276，LPIPS 0.6011；0.40 达到 PSNR 20.2550，SSIM 0.4267，LPIPS 0.6008；0.65 达到 PSNR 20.2162，SSIM 0.4253，LPIPS 0.6034。
- `vfm_loss_thresh=0.35` 是当前 descriptor 最优短跑点，明显优于默认 0.50，并在 LPIPS 上略优于 token-edge 快速验证。继续扩大低分辨率阈值网格的收益预计有限。
- `max_width=518` DINO ViT-S/14 cache 已构建并校验通过。cache 大小 127M，构建耗时 9.90s，首个 entry 为 `24x37x384`。
- `max_width=518` + `vfm_loss_thresh=0.35` 快速验证为 PSNR 20.3059，SSIM 0.4272，LPIPS 0.5999，78,927 个 Gaussians，训练时间 2.69s。它只在 PSNR 上略优于 224-cache 0.35，SSIM/LPIPS 没有同步提升。
- descriptor 30k 完整训练已完成。`dinov2_descriptor_cosine`、`vfm_loss_thresh=0.35`、`max_width=224` cache 达到 PSNR 26.9770，SSIM 0.8298，LPIPS 0.1850，461,846 个 Gaussians，训练时间 190.59s。
- descriptor 30k 相比原始 baseline 提升 +0.2738 PSNR、+0.0231 SSIM、-0.0428 LPIPS；相比 `fastgs_densify100` cadence control 提升 +0.0483 PSNR、+0.0057 SSIM、-0.0114 LPIPS。
- descriptor 30k 低于默认 DINO token-edge：-0.0807 PSNR、-0.0047 SSIM、LPIPS 差 +0.0083；但点数少 28,986。当前 descriptor 是可用的真实语义路径，还不是 DINO 主结果。
- descriptor staged 预算对齐已完成。`target_gaussian_count=410000`、`stage_margin=1.05` 后自然结束在 381,726 个 Gaussians，结果为 PSNR 26.9064，SSIM 0.8208，LPIPS 0.2021，训练时间 161.73s。
- 预算对齐后的 descriptor 低于 `fastgs_densify100` cadence control：-0.0223 PSNR、-0.0033 SSIM、LPIPS 差 +0.0057。这说明 unpruned descriptor 的小幅收益没有在接近预算下保住。
- descriptor `rgb_only` 保守接入已完成。禁用直接 descriptor densification 后，结果为 PSNR 26.9370，SSIM 0.8239，LPIPS 0.1972，407,201 个 Gaussians，训练时间 191.54s。
- descriptor `rgb_only` 相比 `fastgs_densify100` 只高 +0.0083 PSNR，但 SSIM 低 -0.0002，LPIPS 差 +0.0008；它能控制点数，却没有保住默认 descriptor 的主要质量收益。
- descriptor mask/aggregation 第一版改进已实现。`vfm_metric_map_mode` 支持 `threshold`、`percentile`、`topk` 和 `soft_topk`，descriptor 还可用 `vfm_descriptor_token_smooth_kernel` 在 DINO token grid 上平滑 patch error。
- top-k/smoothing 集成验证已完成。120-step bicycle run 触发一次 descriptor scoring 和 densification，达到 PSNR 19.3201，SSIM 0.3804，LPIPS 0.6716，58,605 个 Gaussians，训练时间 2.00s。该结果只说明链路可用，不作为质量判断。
- soft top-k 集成验证已完成。620-step bicycle run 触发一次 descriptor scoring、3 层嵌套 top-k 计数和 densification，达到 PSNR 20.8610，SSIM 0.4747，LPIPS 0.5481，61,344 个 Gaussians，训练时间 4.47s。该结果只说明链路可用，不作为质量判断。
- soft top-k 30k 完整对照已完成。`vfm_metric_map_mode=soft_topk`、`vfm_metric_topk=0.15`、`vfm_metric_soft_levels=3`、`vfm_descriptor_token_smooth_kernel=3` 达到 PSNR 26.9875，SSIM 0.8305，LPIPS 0.1844，462,696 个 Gaussians，训练时间 212.26s。
- soft top-k 相比 `fastgs_densify100` cadence control 质量正向：+0.0589 PSNR、+0.0064 SSIM、LPIPS 好 -0.0120，但多 50,618 个点、训练多 46.37s；相比 top-k 8% 几乎打平但更慢。因此它还不是预算高效结论。
- soft top-k staged 410k 已完成。结果为 PSNR 26.8848，SSIM 0.8201，LPIPS 0.2029，383,528 个 Gaussians，训练时间 231.87s。相比 `fastgs_densify100` cadence control 少 28,550 个点，但质量低 -0.0439 PSNR、-0.0040 SSIM、LPIPS 差 +0.0065，且训练更慢。
- soft top-k staged 相比 top-k 15% staged 更弱、相比 top-k 8% staged 近似但更慢。多层 soft 计数没有解决 descriptor 收益在预算对齐后消失的问题。
- percentile 90% 完整对照已完成。`vfm_metric_map_mode=percentile`、`vfm_metric_percentile=0.90`、`vfm_descriptor_token_smooth_kernel=3` 达到 PSNR 27.0036，SSIM 0.8313，LPIPS 0.1827，464,425 个 Gaussians，训练时间 206.60s。
- percentile 90% 质量介于 top-k 15% 和 top-k 8% 之间，优于 soft top-k 30k，但仍比 `fastgs_densify100` 多 52,347 个点、训练多 40.71s。由于当前 percentile 实现接近固定比例 high-error mask，它没有带来新的预算行为。
- top-k/smoothing 30k 对照已完成。`vfm_metric_map_mode=topk`、`vfm_metric_topk=0.15`、`vfm_descriptor_token_smooth_kernel=3` 达到 PSNR 27.0274，SSIM 0.8330，LPIPS 0.1805，484,229 个 Gaussians，训练时间 191.30s。
- top-k/smoothing 相比默认 descriptor 提升 +0.0504 PSNR、+0.0032 SSIM、LPIPS 好 -0.0045，并接近 DINO token-edge；但它比 `fastgs_densify100` 多 72,151 个点，因此还不是预算受控结论。
- top-k/smoothing 8% 完整对照已完成。结果为 PSNR 26.9931，SSIM 0.8301，LPIPS 0.1849，456,567 个 Gaussians，训练时间 150.34s。
- top-k 8% 相比 top-k 15% 少 27,662 个 Gaussians、训练少 40.96s，质量小幅下降；相比默认 descriptor 少 5,279 个 Gaussians，且 PSNR/SSIM 略高，是当前 descriptor 完整对照中更均衡的点。
- top-k 8% 相比 `fastgs_densify100` cadence control 仍多 44,489 个 Gaussians，但指标更好：+0.0644 PSNR、+0.0060 SSIM、LPIPS 好 -0.0115。下一步需要做 410k staged 对齐，确认这个收益能否在贴近预算时保住。
- top-k/smoothing 410k staged target 已完成。结果为 PSNR 26.9047，SSIM 0.8219，LPIPS 0.1998，389,250 个 Gaussians，训练时间 160.53s。
- top-k/smoothing staged 相比 `fastgs_densify100` 低 -0.0240 PSNR、-0.0022 SSIM、LPIPS 差 +0.0034；相比默认 descriptor staged 410k 只小幅改善 LPIPS。这说明 top-k/smoothing 的 unpruned 收益没有经受住预算对齐。
- top-k/smoothing staged + dense recovery 已完成。`post_prune_finetune_trigger=any_prune` 在训练中期 staged pruning 后触发 4,096 步恢复，最终为 PSNR 26.8472，SSIM 0.8223，LPIPS 0.1974，387,109 个 Gaussians，训练时间 178.91s。
- 相比无恢复的 top-k/smoothing staged，dense recovery 的 SSIM 高 +0.0004、LPIPS 好 -0.0024，但 PSNR 低 -0.0575；相比 `fastgs_densify100` 仍低 -0.0815 PSNR、-0.0018 SSIM，LPIPS 差 +0.0010。训练结束后统一恢复不能把 descriptor staged 预算结果转正。
- top-k 8% staged 410k 已完成。结果为 PSNR 26.8783，SSIM 0.8208，LPIPS 0.2013，382,035 个 Gaussians，训练时间 165.01s。
- top-k 8% staged 相比 `fastgs_densify100` 少 30,043 个 Gaussians，但质量低 -0.0504 PSNR、-0.0033 SSIM、LPIPS 差 +0.0049；相比 top-k 15% staged 也更弱。降低 top-k ratio 没有改善预算对齐结果。
- top-k/smoothing `rgb_only` 已完成，用于测试“descriptor 只参与 support/pruning，不直接提高 densification importance”。结果为 PSNR 26.9117，SSIM 0.8237，LPIPS 0.1977，412,317 个 Gaussians，训练时间 154.16s。
- top-k/smoothing `rgb_only` 与 `fastgs_densify100` 点数几乎一致，但指标低 -0.0170 PSNR、-0.0004 SSIM、LPIPS 差 +0.0014；相比普通 descriptor `rgb_only` 也略低。这条路线没有转正。
- matched 30k `-r 8` ablation 已成为主质量信号。baseline 达到 PSNR 26.7032，SSIM 0.8067，LPIPS 0.2278，240,394 个 Gaussians，334.36 FPS。
- compact cached edge 将 30k run 提升到 PSNR 26.8864，SSIM 0.8229，LPIPS 0.1972，但点数增长到 408,925，FPS 为 196.43。
- DINO token-edge 给出最佳 30k 指标：PSNR 27.0577，SSIM 0.8345，LPIPS 0.1767，但点数达到 490,832，FPS 为 193.46。
- full run 结果反转了短跑印象：DINO token-edge 在 220 iteration 看起来中性或略差，但在正常 densification 有时间发挥后成为指标最强变体。
- 首个 budget-control probe 使用 `vfm_loss_thresh=0.75` 和 `vfm_weight=0.10`，未能让 VFM 点数接近 baseline。edge 仍为 409,028 个 Gaussians；DINO 降到 422,506，但仍比 baseline 高约 76%。
- budget-control probe 仍保持较好指标，但低于默认 DINO：DINO t075/w010 达到 PSNR 26.9586，SSIM 0.8258，LPIPS 0.1935。
- `vfm_importance_weight` 现在将 densification 强度和 pruning fusion 分离。默认值为 `1.0`，保持向后兼容。
- `vfm_importance_weight=0.25` 时，edge 达到 PSNR 26.9439，SSIM 0.8244，LPIPS 0.1958，413,301 个 Gaussians；DINO 达到 PSNR 26.9261，SSIM 0.8259，LPIPS 0.1928，418,073 个 Gaussians。
- 显式 importance weighting 让默认 DINO 点数降低约 14.8%，但仍未达到接近 baseline 的预算。edge 仍稳定在 400k Gaussians 以上。
- `vfm_importance_mode` 现在支持 `max`、`weighted`、`adaptive_weighted` 和 `rgb_only`。默认 `max` 保持旧行为。
- 30k `rgb_only` ablation 已完成。edge 达到 PSNR 26.9574，SSIM 0.8243，LPIPS 0.1961，413,914 个 Gaussians；DINO 达到 PSNR 26.9310，SSIM 0.8237，LPIPS 0.1962，413,223 个 Gaussians。
- `rgb_only` 未能恢复接近 baseline 的 Gaussian count。单独的 VFM pruning-score fusion 仍可保留或重塑足够多的点，让最终点数保持在 baseline 的约 1.72x。
- 首个 `target_gaussian_count` probe 精确匹配 baseline 点数，但裁掉了最高分 Gaussians，导致质量崩溃：edge PSNR 11.1494，DINO PSNR 10.2215。
- target-count pruning 现在优先删除最低 score Gaussians。相比 high-score removal，这更符合批量预算控制的 score 语义。
- low-score target-count 30k runs 精确匹配 baseline Gaussian count，但质量仍低于 baseline：edge PSNR 23.7729 / SSIM 0.7307 / LPIPS 0.2685；DINO PSNR 23.5571 / SSIM 0.7087 / LPIPS 0.2797。
- 因此，一次性 final prune 不是有效的 budget-matched 质量对比方式。它删除了过多已收敛结构，却不给模型恢复机会。
- staged target-count 30k runs 在保持精确 baseline count 的同时，相比单次 final prune 明显恢复质量。edge 达到 PSNR 25.7979，SSIM 0.7747，LPIPS 0.2537；DINO 达到 PSNR 25.3529，SSIM 0.7727，LPIPS 0.2493。
- staged 240k 仍低于 baseline 质量，说明当前 VFM 变体下严格 baseline-count matching 过于激进。
- 300k staged target 继续恢复质量。edge 达到 PSNR 26.2327，SSIM 0.7866，LPIPS 0.2412；DINO 达到 PSNR 25.9089，SSIM 0.7925，LPIPS 0.2291。
- DINO 300k 的 LPIPS 几乎追平 baseline，但 PSNR/SSIM 仍落后。edge 300k 的 PSNR 更接近，但 LPIPS 更差。
- 350k staged target 产出首个预算受控正向结果。edge 在 340,283 个 Gaussians 下达到 PSNR 26.7788，SSIM 0.8089，LPIPS 0.2206，三项指标均超过 baseline。
- DINO 350k 在 350,000 个 Gaussians 下达到 PSNR 26.3634，SSIM 0.8033，LPIPS 0.2188。它的 LPIPS 优于 baseline，但 PSNR/SSIM 仍低。
- edge 正向结果已在 garden 复验。garden baseline 为 PSNR 28.7051，SSIM 0.8889，LPIPS 0.1134，196,201 个 Gaussians；staged edge 为 PSNR 28.9411，SSIM 0.8964，LPIPS 0.1007，248,471 个 Gaussians。
- edge 正向结果已在 counter 完成第三场景复验。counter baseline 为 PSNR 29.5346，SSIM 0.9304，LPIPS 0.0815，113,168 个 Gaussians；staged edge 为 PSNR 29.6316，SSIM 0.9319，LPIPS 0.0791，111,116 个 Gaussians。
- counter edge 的点数比 baseline 少 2,052 个，约少 1.8%。这说明至少在这个室内场景上，`cached_edge_l1` 的正向指标不是由更大的 Gaussian budget 解释的。
- MipNeRF360 全 9 场景统一重跑已完成。`cached_edge_l1 + staged target ~= 1.42x baseline count` 平均达到 PSNR 28.7213，SSIM 0.8579，LPIPS 0.1551；对应 baseline 平均为 PSNR 28.6527，SSIM 0.8551，LPIPS 0.1620。
- 全场景平均提升为 +0.0686 PSNR、+0.0028 SSIM、-0.0068 LPIPS；平均 Gaussian 数量从 173,341 增到 215,869，平均训练时间从 125.38s 增到 139.33s。
- 全场景中 6/9 场景 PSNR 提升，8/9 场景 SSIM 提升，8/9 场景 LPIPS 改善。`treehill` 是明确负例，三项指标同时变差；`bonsai` 和 `room` 的 PSNR 小幅下降但 SSIM/LPIPS 改善。
- `counter` 与 `kitchen` 在 Gaussian 数量少于 baseline 的情况下仍三项指标提升，说明 edge v1 并不只是靠更大点数预算取胜。`flowers` 与 `garden` 的感知指标收益最大，说明边缘代理对复杂纹理/植被边界有帮助。
- no-effect 控制已补齐。`fastgs_photometric + densification_interval=100` 达到 PSNR 26.9287，SSIM 0.8241，LPIPS 0.1964，412,078 个 Gaussians；zero-weight cached edge 和 zero-weight DINO 分别为 410,330 与 412,037 个 Gaussians，指标也基本一致。
- 这说明 410k 级别点数主要由 `densification_interval=100` 的 cadence 改变驱动，而不是 VFM zero-weight scorer/cache 本身。后续预算归因必须固定或显式报告 densification cadence。
- `post_prune_finetune_iterations` 已落地，默认关闭。训练会在最终 target prune 真的删除 Gaussians 后清空残留梯度，继续纯光度恢复训练，并保存到新的迭代号。
- 260-iteration 快速验证确认了裁剪、恢复保存和 `--iteration -1` 评估链路：78,838 -> 65,000，20 步后保存 `ours_280`。
- dense recovery 参数已落地，默认保持旧行为。新增 `post_prune_finetune_step_interval`、`post_prune_finetune_sh_step_interval`、`post_prune_finetune_lr_mode`、`post_prune_finetune_lr_scale` 和 `post_prune_finetune_trigger`，用于让恢复阶段脱离 30k 后的稀疏 optimizer cadence。
- dense recovery 260-iteration 快速验证已完成。`post_prune_finetune_step_interval=1`、`post_prune_finetune_sh_step_interval=1`、`post_prune_finetune_lr_mode=local`、`post_prune_finetune_lr_scale=0.25` 从 88,194 裁到 65,000，并保存 `ours_280`；render/metrics 达到 PSNR 20.4099，SSIM 0.4310，LPIPS 0.5961。该结果只说明链路可用。
- bicycle edge 30k 最终裁剪后 4,096 步恢复结果为 PSNR 26.0163，SSIM 0.7760，LPIPS 0.2580，240,394 个 Gaussians。相比 final-only target prune 明显恢复，但仍低于 baseline，也没有超过 240k staged 的 LPIPS。
- cached edge 严格 240k dense recovery 完整对照已完成。`post_prune_finetune_step_interval=1`、`post_prune_finetune_sh_step_interval=16` 和局部 xyz LR x0.25 后达到 PSNR 26.2470，SSIM 0.7813，LPIPS 0.2526，240,394 个 Gaussians。它比默认 cadence 恢复更好，但仍低于 baseline 与 350k staged edge 正向控制组。
- dense recovery 的结论是：恢复训练能修补 final-prune 或 staged-prune 的一部分损伤，但不能替代健康的预算机制。下一版应把恢复移动到 pruning 发生后的局部时序中，或改变 score 本身，避免先造成大结构损伤再尝试补救。

## 局限

- `mock_l1` 有意不是一个真实视觉基础模型信号。
- `cached_edge_l1` 也只是 proxy；它主要测试 cache 机制和 edge-alignment 行为。
- `dinov2_token_edge_l1` 消费 DINO patch tokens，但比较的是标量 topology projection，而不是完整语义特征向量。
- `dinov2_descriptor_cosine` 已比较完整 patch descriptor，但目前仍是在线 DINO 推理版本；30k 成本高于 token-edge。top-k/smoothing 已把 unpruned descriptor 质量推进到接近 DINO token-edge，top-k 8% 与 percentile 90% 给出更均衡的完整对照点；soft top-k 30k 质量正向但成本更高。staged 预算对齐、降低 top-k ratio、`rgb_only` support/pruning、percentile mask、soft top-k staged 和 staged 后 dense recovery 接入后仍低于 cadence control。
- 220-iteration 快速验证只验证集成健康，不验证最终重建质量。既然 30k runs 已足够便宜，后续不应用短跑结果选择 scorer。
- compact storage 有助于节省磁盘，但 `npz_uint8` 尚未证明 metric-neutral。float32 与 compact cache 变体仍应保留用于 ablation。
- 当前 DINO cache 构建于 `max_width=224`；完整 `max_width=518` 或 `640` 的缓存时间、磁盘占用和 scorer 行为仍需测量。
- 最好的 DINO 质量结果不是预算受控结果：bicycle top-k 25% 完整对照约为该场景 baseline 的 2.07x Gaussian count；全场景 DINO i0.50 candidate 平均也约为 baseline 的 1.52x。下一步必须把特征信号质量和允许更密 reconstruction 的收益拆开。
- 现有 knobs 不能提供完整 budget control。`vfm_weight` 影响 pruning fusion，`vfm_importance_weight` 影响直接 VFM densification 强度，`vfm_importance_mode=rgb_only` 可以关闭直接 VFM densification，但都不能匹配 baseline point count。
- `target_gaussian_count` 适合作为 count-control 诊断，但 baseline-sized budget 下的一次性 final pruning 破坏性太强。
- staged budget control 更健康，edge 已在 MipNeRF360 全 9 场景平均转正，但 `treehill` 仍是负例。DINO token-edge i0.50 已成为更强的 MipNeRF360 质量候选，但仍需要预算感知 scorer 或自动容量保护，才能成为预算高效方案。
- 全场景 edge 评估使用 `1.42x baseline count` 的比例感知 target；部分场景自然低于 target，没有真正触发最终预算裁剪。因此结论应表述为比例感知的 v1 正向控制组，而不是所有场景都完成了精确预算匹配。
- Tandt/DB 全场景评估已完成。`db` 的 `drjohnson` 和 `playroom` 均正向，平均提升 +0.4451 PSNR、+0.0037 SSIM、LPIPS 改善 -0.0021，平均 Gaussian 数量增加约 13.5%。
- `tandt` 的 `train` 和 `truck` 均负向，平均下降 -0.3752 PSNR、-0.0061 SSIM、LPIPS 变差 +0.0067，同时平均 Gaussian 数量下降约 37.3%。这说明 v1 在该数据集上的问题不是点数膨胀，而更像 edge/pruning 组合过度压低结构容量。
- 新增四场景合并后 PSNR 只微幅正向（+0.0350），但 SSIM/LPIPS 负向。因此 `cached_edge_l1` 可以作为 MipNeRF360 与 DB 的正向 proxy 控制组，但还不能视为跨数据集稳健方案。
- `vfm_enable` 当前没有 CLI 级显式关闭开关；若加载 VFM experiment yaml，就会触发 VFM scorer/preflight。严格 no-effect 需要从 baseline variant 出发手动覆盖非 VFM cadence 参数。
- post-prune fine-tune 已证明有恢复价值，但严格 240k 预算和 descriptor staged 410k 预算下都未转正；它现在只能作为局部补救机制，不能作为预算高效化主方案。
- `prune_min_gaussian_count` 已作为默认关闭的容量保护参数落地。它会限制训练期抽样裁剪和最终一致性裁剪的最大删除量，避免点数低于指定下限。
- Tandt 容量保护诊断显示，`train` 从 cached edge v1 的 PSNR 23.4054 / SSIM 0.9081 / LPIPS 0.0837 / 35,322 点恢复到 23.5970 / 0.9104 / 0.0804 / 58,788 点；`truck` 从 27.7543 / 0.9550 / 0.0379 / 27,802 点恢复到 27.9641 / 0.9570 / 0.0366 / 41,952 点。
- 容量保护后的 Tandt 平均为 PSNR 25.7806、SSIM 0.9337、LPIPS 0.0585，优于原始 cached edge v1 的 25.5799 / 0.9316 / 0.0608，但仍低于 baseline 的 25.9551 / 0.9377 / 0.0541。结论是：容量下限是必要防线，但不能独立解决 Tandt 负例。
- `prune_min_gaussian_target_ratio` 已完成 Tandt 自动下限复验。ratio `0.7042253521126761` 在 `target_gaussian_count=baseline*1.42` 时自动派生出 train 的 58,788 和 truck 的 41,952，并写入 `cfg_args`。两场景平均为 PSNR 25.7473、SSIM 0.9330、LPIPS 0.0594、50,370 点；相对原始 cached edge v1 有恢复，但略低于手动容量保护，不应解读为新的质量提升。
- `vfm_weight=0.0` 的 Tandt `train` 诊断仍只有 35,698 个点且 PSNR 23.2628；纯 FastGS `densification_interval=100` 也只有 43,488 个点且 PSNR 23.6091。负例同时包含 cadence、edge signal 和 pruning trajectory 的耦合，不应简单归因为某一个 fusion 权重。
- DINO token-edge `top-k 25% + importance_weight=0.50` 已完成 MipNeRF360 全 9 场景复验。候选均值为 PSNR 28.8577、SSIM 0.8666、LPIPS 0.1385、263,572 个 Gaussians；相对 baseline 平均提升 +0.2051 PSNR、+0.0115 SSIM、LPIPS 改善 -0.0234；相对 cached-edge v1 平均提升 +0.1365 PSNR、+0.0087 SSIM、LPIPS 改善 -0.0166。
- DINO i0.50 相对 cached-edge v1 在 9/9 场景三项指标均正向；相对 baseline 在 8/9 场景 PSNR 正向、9/9 场景 SSIM/LPIPS 正向。`treehill` 是唯一 PSNR 低于 baseline 的场景，但 SSIM/LPIPS 明显改善；`stump` 是 PSNR 增益最大的正例。
- DINO i0.50 的代价是平均 Gaussian 数量从 baseline 的 173,341 增到 263,572，约 +52.1%。它证明真实 DINO token topology 信号有效，但预算机制仍需单独解决。
- `weighted + importance_weight=0.50` bicycle 30k 对照最终 415,158 个 Gaussians，PSNR 26.9756、SSIM 0.8288、LPIPS 0.1867，训练 141.21s。相比 `fastgs_densify100` cadence control 只多 3,080 个点，却提升 +0.0469 PSNR、+0.0047 SSIM、LPIPS 改善 -0.0097；相比普通 `max + importance_weight=0.50` 少 24,913 个点，质量只小幅回落。这个结果说明 RGB/VFM 加权平均比软预算衰减更像可用的近预算融合方式。
- `weighted + importance_weight=0.50` treehill 压力复验最终 417,534 个 Gaussians，PSNR 24.5101、SSIM 0.7281、LPIPS 0.2837，训练 139.79s。相比普通 treehill i0.50 少 14,986 个点、训练少 2.35s，但 PSNR 低 -0.0072、SSIM 低 -0.0003、LPIPS 差 +0.0015；相比 baseline 仍保持 SSIM/LPIPS 明显正向但 PSNR 略低。它说明 weighted 能在压力场景省点并基本保住质量，但不是完整预算解法。
- `weighted + importance_weight=0.50` stump 大收益场景复验最终 354,046 个 Gaussians，PSNR 27.6147、SSIM 0.8170、LPIPS 0.1934，训练 136.36s。相比普通 stump i0.50 少 11,538 个点、训练少 1.54s，且 PSNR/SSIM/LPIPS 基本持平甚至微幅更好；相比 baseline 提升 +0.4391 PSNR、+0.0236 SSIM、LPIPS 改善 -0.0393。它把 weighted 从“bicycle 近预算点”推进为可跨场景保留收益的候选。
- `weighted + importance_weight=0.50` counter 低增点场景复验最终 119,273 个 Gaussians，PSNR 29.6650、SSIM 0.9333、LPIPS 0.0752，训练 133.34s。相比普通 counter i0.50 只少 422 个点、训练少 6.95s，但 PSNR 回落 -0.0524；相比 baseline 和 cached-edge v1 仍三项正向。它说明普通 i0.50 已贴近 baseline 点数时，weighted 不应默认替换普通 i0.50。
- `weighted + importance_weight=0.50` garden 中等点数增长复验最终 253,355 个 Gaussians，PSNR 28.9546、SSIM 0.8977、LPIPS 0.0974，训练 134.56s。相比普通 garden i0.50 少 9,030 个点、训练少 4.75s，PSNR/SSIM/LPIPS 小幅回落；相比 baseline 和 cached-edge v1 仍三项正向。它说明 weighted 在中等点数增长场景可作为省点折中，但 LPIPS 回落比 stump/counter 更明显。
- `weighted + importance_weight=0.50` flowers 高增点植被场景复验最终 339,267 个 Gaussians，PSNR 22.9636、SSIM 0.6933、LPIPS 0.2791，训练 141.18s。相比普通 flowers i0.50 少 11,154 个点、训练少 4.12s，但 PSNR 回落 -0.0498、SSIM 回落 -0.0027、LPIPS 差 +0.0044；相比 baseline 仍明显正向，相比 cached-edge v1 也保住 SSIM/LPIPS 正向。它说明高增点植被场景中 weighted 仍有折中价值，但质量损失比 garden/stump 更大。
- `weighted + importance_weight=0.50` bonsai 中低增点场景复验最终 134,806 个 Gaussians，PSNR 32.5395、SSIM 0.9642、LPIPS 0.0493，训练 138.31s。相比普通 bonsai i0.50 少 1,499 个点、训练少 1.01s，同时 PSNR/SSIM/LPIPS 均微幅更好；相比 baseline 和 cached-edge v1 三项指标也更强。它说明 weighted 并非只适合大增点场景，关键还取决于 DINO token topology 是否改善了该场景的结构选择。
- `weighted + importance_weight=0.50` kitchen 室内高基线场景复验最终 157,804 个 Gaussians，PSNR 33.3234、SSIM 0.9693、LPIPS 0.0347，训练 141.71s。相比普通 kitchen i0.50 少 3,543 个点、训练少 3.21s，PSNR 低 -0.0124、SSIM 基本持平、LPIPS 差 +0.0003；相比 baseline 和 cached-edge v1 仍三项正向，并且点数少于二者。它说明 weighted 在室内高基线场景可以作为低风险省点折中，但质量上界仍是普通 i0.50。
- `weighted + importance_weight=0.50` room 室内房间场景复验最终 101,384 个 Gaussians，PSNR 33.1081、SSIM 0.9626、LPIPS 0.0574，训练 131.98s。相比普通 room i0.50 少 2,436 个点、训练少 1.76s，同时 PSNR 高 +0.0360、SSIM 高 +0.0004、LPIPS 基本持平；相比 baseline 和 cached-edge v1 也三项正向。它说明 weighted 在小室内场景也可能少点省时且微升质量。
- `weighted + importance_weight=0.50` 全 9 场景均值为 PSNR 28.8505、SSIM 0.8660、LPIPS 0.1397、254,736 个 Gaussians、训练 137.60s。相比 baseline 仍提升 +0.1978 PSNR、+0.0109 SSIM、LPIPS 改善 -0.0223；相比普通 i0.50 平均少 8,836 点、训练少 2.87s，质量只低 -0.0072 PSNR、-0.0006 SSIM、LPIPS 差 +0.0012。因此 weighted 完整结论是“预算效率候选”，普通 i0.50 完整结论仍是“质量候选”。
- `weighted + importance_weight=0.75` 已从 bicycle/stump/room/bonsai 扩展到 MipNeRF360 全 9 场景。bicycle、stump、room 继续保持 PSNR 正向，bonsai 仍是边界负例，因此早期判断“i0.75 需要选择规则约束”成立。
- 新补的 5 个 i0.75 场景显示出更清楚的分化：counter 相比 i0.50 为 +0.0531 PSNR、+0.0006 SSIM、LPIPS -0.0001、+2,313 点；garden 为 +0.0200 PSNR、+0.0006 SSIM、LPIPS -0.0008、+3,574 点；flowers、kitchen、treehill 则未通过质量门槛，应回退 i0.50。
- 固定 `weighted i0.75` 全 9 场景均值为 PSNR 28.8396、SSIM 0.8663、LPIPS 0.1395、257,715 个 Gaussians、训练 168.35s。相比固定 `weighted i0.50`，PSNR 低 -0.0109，SSIM 高 +0.0003，LPIPS 改善 -0.0001，但多 2,978 点且训练多 30.75s，因此不能作为无条件默认档。
- 严格 `quality_pick` 按场景在 i0.50/i0.75/i0.90 之间选择，当前选中 bicycle、garden、room、stump 的 i0.75，counter 的 i0.90，其余场景保留 i0.50。全 9 场景均值为 PSNR 28.8641、SSIM 0.8665、LPIPS 0.1392、257,326 个 Gaussians，相比 fixed weighted i0.50 提升 +0.0136 PSNR、+0.0004 SSIM、LPIPS 改善 -0.0005，只多 2,589 点。`QCGI pick` 进一步让 treehill 选择 i0.90，均值为 28.8641 / 0.8667 / 0.1388、255,822 个 Gaussians，只多 1,086 点。它是当前 weighted 分支最值得保留的质量策略。
- `adaptive_weighted + quadratic 430k` bicycle 30k 对照最终 424,011 个 Gaussians，PSNR 26.9858、SSIM 0.8302、LPIPS 0.1853，训练 141.94s。相比 `weighted i0.50` 多 8,853 个点，但提升 +0.0102 PSNR、+0.0014 SSIM、LPIPS 改善 -0.0014；相比普通 i0.50 少 16,060 个点，质量仍小幅低一点。它是新的单场景效率候选，下一步必须跨场景复验后再决定是否替代 weighted。
- `adaptive_weighted + quadratic 430k` treehill 复验最终 420,283 个 Gaussians，PSNR 24.4393、SSIM 0.7285、LPIPS 0.2821，训练 140.96s。相比普通 treehill i0.50，PSNR 低 -0.0780，SSIM 基本持平，LPIPS 只微幅好 -0.0001；相比 treehill `weighted i0.50`，点数更多但 PSNR 低 -0.0708。它没有通过第二场景验证，不能替代全场景 weighted 结论。

## 下一版计划

1. 固化两个角色：`cached_edge_l1` 是 proxy 正向控制组，结论边界为 MipNeRF360 与 DB 正向、Tandt 负向；DINO token-edge i0.50 是 MipNeRF360 全场景质量候选，但不是预算高效候选。
2. 把 `prune_min_gaussian_count` 保留为诊断/回退保护，同时使用默认关闭的 `prune_min_gaussian_target_ratio` 从 staged target 自动派生容量下限。当前 ratio `0.7042253521126761` 在 target 为 `baseline * 1.42` 时等价于 baseline 最终点数，Tandt 复验已确认机制可复现容量保护；但 DINO weighted + 自动容量下限和 prunemin-only 诊断说明它不能单独修复 Tandt 轨迹损伤。下一版不要长期依赖人工填 baseline count，应继续转向 baseline 预跑、在线增长曲线或 scene scale 自动估计容量下限。
3. 为下一版增加场景自适应保护：当自然结束 Gaussian 数量显著低于 baseline 或 staged target 时，降低 pruning fusion 强度、回退到 baseline pruning，或触发容量保护，避免 Tandt 这类场景被压得过稀。
4. DINO 主线不再追加同类 top-k、percentile、soft-top-k 或单纯 final hard-prune 单点。下一步只做会改变预算行为的实验，例如 RGB/VFM 加权融合的高质量档位复验、按场景自动下调 VFM importance，或把 target budget 与恢复时序绑定。
5. 预算感知 importance cap 已完成 bicycle 420k、430k 放松衰减和 430k quadratic 三个 30k 对照：420k 软预算点为 422,778 个 Gaussians，PSNR/SSIM/LPIPS 为 26.9732 / 0.8273 / 0.1916；430k、start 0.95、min 0.10 为 419,513 个 Gaussians，PSNR/SSIM/LPIPS 为 26.9750 / 0.8270 / 0.1919；430k quadratic 为 418,137 个 Gaussians，PSNR/SSIM/LPIPS 为 26.9402 / 0.8262 / 0.1918。它们都没有保住 i0.50 质量；下一步不迁移全场景，也不继续手工追加相近曲线单点，先改为场景自适应预算或直接估计场景容量。
6. `weighted + importance_weight=0.50` 已完成全 9 场景复验，是预算效率默认候选；`weighted + importance_weight=0.75/0.90` 已补齐全 9 场景，固定均值都不超过 i0.50，但按场景选择后的 `quality_pick` 和 `QCGI pick` 三项质量均优于固定 i0.50。下一步把规则收束为：预算优先选 `weighted i0.50`；质量优先时用 QCGI 或同等质量门槛允许小幅 GS 增长，在 i0.75/i0.90 中按场景选择；单个场景负例只作为回退信号，不再直接否定整个方法，最终以数据集平均和场景分布判断。
7. 已验证的 `support_ratio` 与高置信 prune-protect 都没有优于普通 i0.50，因此下一版不把它们作为主方向；可以保留为诊断分支。
8. 重做恢复时序：不再只在训练结束后统一 dense recovery，而是在 staged pruning 发生后立即执行短局部恢复，观察是否能减少中期结构损伤。

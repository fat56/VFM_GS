# 0001 复盘

## 当前决策

继续保留 `vfm_topology_scorer` 作为 v1 集成路径。`mock_l1` 验证打分链路；`cached_edge_l1` 固化为早期 proxy 正向控制组，但 Tandt/DB 全场景评估显示它存在明显跨数据集差异：`db` 正向，`tandt` 负向。`dinov2_token_edge_l1` 已验证训练可以消费真实 DINOv2 patch-token cache，并且 `top-k 25% + importance_weight=0.50` 完成 MipNeRF360 全 9 场景评估，平均 PSNR 28.8577、SSIM 0.8666、LPIPS 0.1385，超过 baseline 与 cached-edge v1，可作为 0001 的 DINO 质量候选；但平均 Gaussian 数量比 baseline 多约 52.1%，还不是预算高效结论。`weighted + importance_weight=0.50` 已完成 MipNeRF360 全 9 场景复验，平均 PSNR 28.8505、SSIM 0.8660、LPIPS 0.1397，平均 254,736 个 Gaussians、训练 137.60s；它相对普通 i0.50 平均少 8,836 个点、训练少 2.87s，质量只小幅回落，因此是当前全场景预算效率候选。`weighted + importance_weight=0.75/0.90` 均已补齐 MipNeRF360 全 9 场景，固定均值都不作为默认档。严格三档 `quality_pick` 平均为 PSNR 28.8641、SSIM 0.8665、LPIPS 0.1392、257,326 个 Gaussians，比 fixed weighted i0.50 三项质量均正向且只多 2,589 个点；`QCGI pick` 平均为 28.8641、0.8667、0.1388、255,822 个 Gaussians，只多 1,086 个点。这个结果把 i0.75/i0.90 收束为“场景选择质量档”，也确认单个负例不应直接否决方法，应看数据集平均收益。`adaptive_weighted + quadratic 430k` 在 bicycle 上正向，但 treehill 第二场景复验未通过，后续不升级为主线。`dinov2_descriptor_cosine` 则打通了在线渲染图 descriptor 与 GT cache descriptor 的语义比较路径，但预算对齐后未转正。后续决策门槛以 30k 完整 run 和多场景平均为准，不再用短程验证指标判断质量。

Tandt 的 DINO weighted i0.50/i0.75/i0.90 跨数据集复验显示：i0.50 平均为 PSNR 25.7519、SSIM 0.9346、LPIPS 0.0575，较 cached-edge v1 分别改善 +0.1721、+0.0031、-0.0033；但它仍低于 Tandt baseline，平均差距为 -0.2032 PSNR、-0.0031 SSIM、LPIPS 差 +0.0035。继续提高权重没有帮助，i0.75 均值降到 25.6201 / 0.9328 / 0.0585，i0.90 降到 25.5329 / 0.9323 / 0.0611，训练时间也明显拉长。因此 Tandt 当前只把 DINO weighted 记录为“修复 cached-edge Tandt 负例的一部分”，默认策略应回退 baseline。

DINO weighted i0.50 + 自动容量下限进一步排除了“最终点数太少”这一解释。该诊断把 Tandt 两个场景最终 Gaussian 数量拉回 baseline 均值 50,370，但平均结果只有 PSNR 25.6078、SSIM 0.9324、LPIPS 0.0610，低于原始 DINO weighted i0.50 和 baseline。训练日志显示早期 staged target pruning 已在 1,000 iteration 附近发生大幅裁剪，因此容量下限只能防止最终过稀，不能修复已经被改写的训练轨迹。取消 staged target、仅保留容量下限后，平均结果为 25.6430 / 0.9341 / 0.0566、50,370 个 Gaussians；它相对 staged 版本修复了 LPIPS/SSIM，但 PSNR 仍低于原始 DINO weighted i0.50 和 baseline。关闭 VFM pruning fusion 后，平均为 25.6955 / 0.9338 / 0.0572，LPIPS 略优于原始 i0.50，但 PSNR/SSIM 仍低。三条诊断都没有让 Tandt 超过 baseline，因此策略应改成场景级 baseline 回退。

DB 的 DINO weighted 多档复验则给出相反信号。i0.50 平均为 30.3603 / 0.9360 / 0.0641，相对 baseline 正向但低于 cached-edge；i0.75 提升到 30.5446 / 0.9358 / 0.0633；i0.90 进一步达到 30.6074 / 0.9376 / 0.0620，平均 63,006 个 Gaussians。i0.90 相对 DB baseline 平均 +0.4894 PSNR、+0.0051 SSIM、LPIPS -0.0038；相对 DB cached-edge v1 也有 +0.0443 PSNR、+0.0015 SSIM、LPIPS -0.0017。因此 DB 上高权重 DINO weighted 是新的正向质量档，但它不应外推到 Tandt。

跨数据集选择汇总脚本已扩展到 MipNeRF360、DB、Tandt 三个公开数据集的 baseline、cached-edge v1 和 DINO weighted i0.50/i0.75/i0.90。后续文档不再把三个数据集混成一个总平均作为主结论，而是按数据集分别报告。固定三档的分数据集结论是：MipNeRF360 上 i0.50 最均衡，平均 28.8505 / 0.8660 / 0.1397，相对 baseline 为 +0.1979 PSNR、+0.0109 SSIM、LPIPS -0.0223；DB 上 i0.90 最强，平均 30.6074 / 0.9376 / 0.0620，相对 baseline 为 +0.4894 PSNR、+0.0051 SSIM、LPIPS -0.0038；Tandt 上三档均低于 baseline，必须回退。逐场景 PSNR 最优分布仍可作为诊断参考：9 个场景选 DINO weighted，1 个场景选 cached-edge，3 个场景选 baseline。下一版应把 0001 收束为“按数据集/场景自动选择或回退 + QCGI 容量约束”，而不是继续寻找单一固定后端。

为了避免把 test oracle 包装成自动方法，第一版同时固化了两个非事后展示策略，并按三个数据集分别比较。`dataset_fixed_policy` 使用 MipNeRF360 固定 weighted i0.50、DB 固定 DINO weighted i0.90、Tandt baseline 回退：MipNeRF360 平均相对 baseline 为 +0.1979 PSNR、+0.0109 SSIM、LPIPS -0.0223；DB 为 +0.4894、+0.0051、-0.0038；Tandt 与 baseline 持平。`dataset_quality_policy` 使用 MipNeRF360 weighted QCGI、DB 固定 i0.90、Tandt baseline 回退：MipNeRF360 为 +0.2114、+0.0116、-0.0231；DB 与 fixed policy 相同；Tandt 持平。这个口径更准确：0001 第一版已经证明 VFM_GS 在 MipNeRF360 和 DB 上按数据集平均有效，Tandt 当前需要回退或继续研究；下一版重点应从继续堆单点实验转向无泄漏 selector 与训练期自适应控制。

`DINO descriptor top-k25 max` 已完成 DB/Tandt 同模块验证，补上了一条不依赖回退的 VFM 证据线。该方案使用 `dinov2_descriptor_cosine + top-k25 + vfm_weight=0.0`，只让 descriptor residual 参与 densification，不改变 pruning score。分数据集平均结果为：MipNeRF360 相对 FastGS densify100 提升 +0.1066 PSNR、+0.0050 SSIM、LPIPS 改善 -0.0093；DB 提升 +0.0085、+0.0002、-0.0011；Tandt 提升 +0.1004、+0.0017、-0.0016。MipNeRF360 是强证据，Tandt 两个场景全部正向，说明 descriptor residual 能修复此前 token-edge weighted 在 Tandt 上低于 baseline 的问题；DB 是弱正向，`playroom` 单场景仍负向。该结果不替代当前最佳指标策略，但它更直接支撑“VFM 先验指导 GS 复制提升质量”的研究目标。

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
- 大分辨率链路探测已开始。`dinov2_vitl14` 已接入 cache builder 和 scorer preflight；`--project_token_edge` 可把 1.6K ViT-L patch tokens 投影成 `dinov2_token_edge` 2D cache。bicycle 全量 cache 共有 194 个 entries，`npz_uint8` 下约 1.9M，并通过校验。
- bicycle `-i images -r -1` 的 620-step ViT-L token-edge 短训练已通过，日志确认沿用 FastGS 原始 1.6K 自动缩放提示，最终 61,265 个 Gaussians，训练 3.15s，无 OOM。下一步先跑 bicycle 30k 大分辨率正式实验，再决定是否扩展到三个公开数据集全场景。
- bicycle 大分辨率 ViT-L token-edge 30k 正式探针已完成。测试指标为 PSNR 25.0785、SSIM 0.7394、LPIPS 0.2733，最终 1,033,601 个 Gaussians，训练 187.28s。训练沿用 `-r -1` 的 1.6K 自动缩放规则，显存观测约 7.2GB，低于 24GB 上限很多。
- 该结果的关键价值是资源与流程确认：ViT-L/14 是当前可承受的大 DINO 档位，1.6K token-edge cache 足够小，且最终点数低于用户提供的 FastGS `densify100` 约 1.15M 参考。质量是否优于 baseline 需要等待用户手头同裁切口径 FastGS 原始指标对照，不能只看单场景绝对 PSNR。
- 全场景大分辨率评测应按 MipNeRF360、DB、Tandt 三个数据集分别统计平均 PSNR/SSIM/LPIPS、Gaussian 数量和训练时间；不要再把 13 个场景合并成一个主平均值。脚本已支持 `--project-token-edge` 与 `--cache-storage npz_uint8`，可复用同一配置跑三套数据集。
- 大分辨率 ViT-L token-edge i0.50 全 13 场景评测已完成，总耗时 5,830s，约 1h37m；评估输出约 2.9G，13 场景 token-edge cache 约 24M。全程无 OOM，cache 阶段显存观察约 16.9GB，训练阶段约 7GB。
- 大分辨率分数据集均值为：MipNeRF360 PSNR 27.2965、SSIM 0.8017、LPIPS 0.2506、581,266 个 Gaussians；DB PSNR 30.0302、SSIM 0.9076、LPIPS 0.2521、340,966 个 Gaussians；Tandt PSNR 23.4810、SSIM 0.8365、LPIPS 0.2123、235,384 个 Gaussians。
- 已补跑 MipNeRF360 大分辨率 `fastgs_big/densify100` 同口径基线。该基线使用 `-i images -r -1`、FastGS 原始 1.6K 自动缩放，并复用 `scripts/train_big.sh` 的场景级超参；平均为 PSNR 27.9293、SSIM 0.8198、LPIPS 0.2157、1,161,242 个 Gaussians，和用户给出的论文参考 27.90 基本一致。
- 与该基线相比，当前大分辨率 VFM i0.50 平均低 -0.6328 PSNR、-0.0180 SSIM，LPIPS 差 +0.0349，同时少约 579,976 个 Gaussians。9 个 MipNeRF360 场景全部低于 FastGS big；这说明迁移/评测链路没有坏，主要问题是当前大分辨率 VFM 配置过度压低容量，且没有套 FastGS big 的场景级超参。
- 从容量角度看，bicycle VFM 为 1.039M GS，而同口径 FastGS big 为 1.560M GS；garden 差距最大，VFM 为 0.642M，FastGS big 为 2.624M。后续大分辨率策略应先补“VFM + FastGS big 场景级超参”的公平对照，再考虑 QCGI 或容量收益门槛；当前 ViT-L i0.50 全场景结果只能作为资源可行性和过强筛选诊断。
- “VFM + FastGS big 场景级超参”对照已完成。MipNeRF360 平均为 PSNR 27.9379、SSIM 0.8199、LPIPS 0.2148、1,193,697 个 Gaussians；相对 FastGS big/densify100 为 +0.0086 PSNR、+0.0002 SSIM、LPIPS 改善 -0.0009，平均多 32,455 个 Gaussians。它说明大分辨率 VFM 在 recipe 对齐后可以达到小幅正向，但提升很薄。
- 逐场景看，bonsai、counter、flowers、kitchen、stump、treehill PSNR 正向；room 的 PSNR 略低但 SSIM/LPIPS 正向；bicycle 和 garden 仍低于 FastGS big。单场景 GS 增长在 bonsai、counter、kitchen、room、stump 上超过 0.1M，因此下一步必须加入容量收益门槛或场景回退，而不能直接把该结果视为完全收束。
- DB/Tandt 同 recipe 复验也已完成。DB VFM+big 平均为 30.2236 / 0.9108 / 0.2362、740,387 个 Gaussians，相对 FastGS big 为 +0.0163 PSNR、-0.0004 SSIM、LPIPS -0.0040，平均多 90,192 个点。Tandt VFM+big 平均为 24.3691 / 0.8569 / 0.1739、517,316 个 Gaussians，相对 FastGS big 为 +0.0134 PSNR、-0.0004 SSIM、LPIPS -0.0006，平均少 23,262 个点。
- 三个公开数据集分开看，VFM+big 都有平均 PSNR 和 LPIPS 弱正向，但提升幅度很小，且 DB/Tandt 的 SSIM 略低。当前可以把大分辨率 v1 定义为“同 recipe 下弱正向、需要容量/回退选择”的结果，而不是强质量突破。
- DINO descriptor top-k25 `max` 已完成 MipNeRF360 全 9 场景扩展。该方案保持 `vfm_weight=0.0`，只影响 densification，不改变 pruning score；全 9 场景相对 FastGS densify100 平均为 PSNR +0.1066、SSIM +0.0050、LPIPS 改善 -0.0093，平均多 50,131 个 Gaussians、训练多 23.20s。9/9 场景 PSNR、SSIM、LPIPS 全部正向，其中新增的 `counter/flowers/kitchen/room/treehill` 五场景也全部正向。这个结果把 descriptor densify-only 从四场景正例推进为 MipNeRF360 数据集级正例，是目前最干净的“VFM 语义/结构 residual 指导复制提升质量”证据。限制是 `bicycle` 和 `stump` 单场景增点超过 0.1M，后续需要容量收益门槛、自适应 importance 或更轻量 descriptor 调用频率控制。
- DINO descriptor top-k25 `max` 已完成 DB/Tandt 扩展。DB 平均为 30.6022 / 0.9369 / 0.0620，相对 FastGS densify100 为 +0.0085 PSNR、+0.0002 SSIM、LPIPS -0.0011，平均多 9,189 个 Gaussians；但 `playroom` 单场景负向。Tandt 平均为 25.8759 / 0.9363 / 0.0554，相对 FastGS densify100 为 +0.1004、+0.0017、-0.0016，平均只多 2,081 个 Gaussians，且 `train/truck` 两场景全部正向。这条线是目前最重要的“无回退 VFM”跨数据集证据。
- DINO descriptor top-k25 `max` 已完成 high-res bicycle 探针。训练使用 `fastgs_big` recipe、`-i images -r -1`，日志确认沿用 FastGS 原始 1.6K 自动缩放。相对 FastGS big bicycle baseline，结果从 25.2532 / 0.7554 / 0.2446 提升到 25.3279 / 0.7646 / 0.2277，三项质量正向；最终 Gaussian 数量从 1,560,079 增到 1,809,292，训练时间从 234.94s 增到 291.88s。该结果说明 descriptor residual 在高分辨率口径下仍有效，但 +249,213 个点超过单场景容量关注阈值，因此不能直接扩全场景，下一步应先跑 high-res weighted/预算控制版本。
- DINO descriptor top-k25 `weighted i0.50` high-res bicycle 已完成。相对同一 FastGS big baseline，结果为 25.2937 / 0.7606 / 0.2361、1,606,190 个 Gaussians、训练 268.87s，对应 +0.0406 PSNR、+0.0053 SSIM、LPIPS -0.0085，且只多 46,111 个 Gaussians。它相比 `max` 少 203,102 个点、少 23.01s，代价是 PSNR -0.0342、SSIM -0.0040、LPIPS +0.0084。按当前 QCGI，`max` 因容量惩罚为负，`weighted i0.50` 为正，因此 high-res 后续扩展应优先用 weighted i0.50，而不是直接扩展 max。
- DINO descriptor top-k25 `weighted i0.50` high-res truck 复验完成。该场景对齐 FastGS big 的 truck 超参，结果为 26.0251 / 0.8886 / 0.1374、727,663 个 Gaussians、训练 227.00s；相对 FastGS big 为 -0.0835 PSNR、-0.0008 SSIM、LPIPS -0.0019，同时多 104,534 个 Gaussians、训练多 54.34s。这个结果说明 high-res i0.50 不是可直接全量扩展的稳定档；它目前是 bicycle 正例、truck 边界负例，下一步需要第三场景或权重扫描。
- DINO descriptor top-k25 `weighted i0.50` high-res garden 复验完成。该场景对齐 FastGS big 的 garden 超参，结果为 27.6376 / 0.8648 / 0.1094、2,570,661 个 Gaussians、训练 424.24s；相对 FastGS big 为 +0.0239 PSNR、+0.0002 SSIM、LPIPS -0.0004，同时少 53,503 个 Gaussians。它把 high-res i0.50 从单个 bicycle 正例推进为 MipNeRF360 内两个正例；truck 更像 Tandt 边界负例，而不是直接否定该 high-res descriptor 分支。
- DINO descriptor top-k25 `weighted i0.50` high-res stump 复验完成。该场景对齐 FastGS big 的 stump 超参，结果为 27.2233 / 0.7903 / 0.2317、1,196,350 个 Gaussians、训练 283.46s；相对 FastGS big 为 +0.0923 PSNR、+0.0042 SSIM、LPIPS -0.0089，同时多 134,069 个 Gaussians、训练多 93.74s。它是 MipNeRF360 内第三个 high-res 质量正例，但 QCGI 为 -0.0162，说明质量提升伴随的容量和时间代价已经偏高。该结果应记录为 high-res 质量证据和容量边界样本，不应放入只保留正向效率改进的主表。
- DINO descriptor top-k25 `weighted i0.50` high-res counter 复验完成。该场景对齐 FastGS big 的 counter 超参，结果为 29.5838 / 0.9194 / 0.1723、551,300 个 Gaussians、训练 200.97s；相对 FastGS big 为 +0.0570 PSNR、+0.0014 SSIM、LPIPS -0.0041，同时多 78,100 个 Gaussians、训练多 20.56s。它是 MipNeRF360 内第四个 high-res 三项质量正例，也是继 bicycle/garden 后第三个 QCGI 为正的容量效率正例。
- DINO descriptor top-k25 `weighted i0.50` high-res kitchen 复验完成。该场景对齐 FastGS big 的 kitchen 超参，结果为 32.4350 / 0.9398 / 0.1036、1,286,004 个 Gaussians、训练 383.90s；相对 FastGS big 为 +0.1650 PSNR、+0.0007 SSIM、LPIPS -0.0008，同时多 107,209 个 Gaussians、训练多 48.79s。它是 MipNeRF360 内第五个 high-res 三项质量正例；虽然 ΔGS 略高于 0.1M，但 QCGI 为 +0.0546，说明新增容量仍有质量收益支撑。
- DINO descriptor top-k25 `weighted i0.50` high-res room 复验完成。该场景对齐 FastGS big 的 room 超参，结果为 32.1919 / 0.9324 / 0.1822、615,908 个 Gaussians、训练 197.77s；相对 FastGS big 为 +0.0596 PSNR、+0.0025 SSIM、LPIPS -0.0059，同时多 45,129 个 Gaussians、训练多 33.57s。它是 MipNeRF360 内第六个 high-res 三项质量正例，且 QCGI 为 +0.0958。
- DINO descriptor top-k25 `weighted i0.50` high-res bonsai 复验完成。该场景对齐 FastGS big 的 bonsai 超参，结果为 33.1160 / 0.9549 / 0.1560、1,094,114 个 Gaussians、训练 265.81s；相对 FastGS big 为 +0.1297 PSNR、+0.0037 SSIM、LPIPS -0.0040，同时多 251,478 个 Gaussians、训练多 52.84s。它是 MipNeRF360 内第七个 high-res 三项质量正例，但 QCGI 为 -0.4824，说明容量增长明显过强，应作为容量边界样本，而不是正向效率改进。
- DINO descriptor top-k25 `weighted i0.50` high-res flowers 复验完成。该场景对齐 FastGS big 的 flowers 超参，结果为 21.6293 / 0.6022 / 0.3412、1,091,531 个 Gaussians、训练 278.68s；相对 FastGS big 为 +0.0127 PSNR、+0.0004 SSIM、LPIPS +0.0008，同时少 48,729 个 Gaussians、训练多 70.92s。它是 high-res i0.50 的混合样本：省点且 PSNR/SSIM 微升，但 LPIPS 未过线，不应放入三项质量正例。
- DINO descriptor top-k25 `weighted i0.50` high-res treehill 复验完成。该场景对齐 FastGS big 的 treehill 超参，结果为 22.8069 / 0.6317 / 0.3774、945,930 个 Gaussians、训练 240.89s；相对 FastGS big 为 -0.0270 PSNR、-0.0001 SSIM、LPIPS +0.0004，同时少 53,053 个 Gaussians、训练多 51.29s。它是 high-res i0.50 的明确质量负例。
- high-res MipNeRF360 9 场景汇总完成。`top-k25 weighted i0.50 + FastGS big` 平均为 27.9908 / 0.8218 / 0.2122、1,217,554 个 Gaussians、训练 282.73s；FastGS big 平均为 27.9293 / 0.8198 / 0.2157、1,161,242 个 Gaussians、训练 236.23s。数据集均值相对 FastGS big 为 +0.0615 PSNR、+0.0020 SSIM、LPIPS -0.0035，平均多 56,312 个点、训练多 46.50s，QCGI 为 +0.0633。结论是 high-res MipNeRF360 数据集均值正向，但 treehill/flowers/bonsai/stump 暴露出场景差异和容量/感知边界。
- high-res treehill `top-k25 max` 复验完成。相对 FastGS big，结果从 `weighted i0.50` 的 22.8069 / 0.6317 / 0.3774 恢复到 22.8700 / 0.6367 / 0.3665，三项质量指标全部正向；Gaussian 数量从 945,930 增至 1,129,614，相对 baseline 多 130,631 个点，QCGI 为 -0.0360。该结果说明 treehill 更像 descriptor 强度不足，而不是 descriptor residual 无效；但 `max` 容量代价偏高，下一步应扫 high-res `i0.70` 或自适应容量约束。
- high-res treehill `weighted i0.70` 复验完成。该档结果为 22.8614 / 0.6305 / 0.3813、877,180 个 Gaussians、训练 221.12s；相对 FastGS big 为 +0.0276 PSNR、-0.0014 SSIM、LPIPS +0.0043，同时少 121,803 个点，QCGI 为 -0.0212。它比 i0.50 省点省时且 PSNR 更高，但 SSIM/LPIPS 变差；说明 treehill 并不存在简单的 `i0.50 -> i0.70 -> max` 平滑折中，后续应转向自适应容量或几何先验。
- high-res flowers `weighted i0.70` 复验完成。结果为 21.5801 / 0.6001 / 0.3442、1,045,864 个 Gaussians、训练 234.69s；相对 FastGS big 为 -0.0365 PSNR、-0.0017 SSIM、LPIPS +0.0039，同时少 94,396 个点，QCGI 为 -0.0895。该档比 i0.50 少 45,667 个点、训练少 43.99s，但三项质量同步下降，说明 flowers 不能靠简单提高 weighted importance 修复。
- high-res flowers `top-k25 max` 复验完成。结果为 21.6394 / 0.6039 / 0.3386、1,273,570 个 Gaussians、训练 275.79s；相对 FastGS big 为 +0.0228 PSNR、+0.0021 SSIM、LPIPS -0.0017，同时多 133,310 个点，QCGI 为 -0.1595。它证明 flowers 存在 descriptor 质量上界，但容量收益比不足；后续应转向自适应容量或 Depth Anything，而不是继续扫描固定 weighted 权重。
- high-res stump `weighted i0.35` 容量探针完成。结果为 26.7484 / 0.7768 / 0.2305、3,035,777 个 Gaussians、训练 385.31s；相对 FastGS big 为 -0.3826 PSNR、-0.0093 SSIM、LPIPS -0.0101，同时多 1,973,496 个点，QCGI 为 -8.1125。它说明固定降低 `vfm_importance_weight` 不会单调压低容量，反而可能改变训练轨迹并造成容量失控；后续容量控制需要显式点数反馈或自适应机制。

## 下一版计划

0. 研究目标从“失败时回退 FastGS 的工程策略”修订为“证明 VFM 先验能提升 GS 质量”。后续不把 FastGS 回退作为主要贡献，而把它仅作为诊断 baseline。主线改为：DINO 负责语义/结构 residual，Depth Anything 负责几何/遮挡边界 residual；先验证 VFM 能否指导复制，再验证能否指导剪枝保护。
1. 固化两个角色：`cached_edge_l1` 是 proxy 正向控制组，结论边界为 MipNeRF360 与 DB 正向、Tandt 负向；DINO token-edge i0.50 是 MipNeRF360 全场景质量候选，但不是预算高效候选。
2. 把 `prune_min_gaussian_count` 保留为诊断/回退保护，同时使用默认关闭的 `prune_min_gaussian_target_ratio` 从 staged target 自动派生容量下限。当前 ratio `0.7042253521126761` 在 target 为 `baseline * 1.42` 时等价于 baseline 最终点数，Tandt 复验已确认机制可复现容量保护；但 DINO weighted + 自动容量下限和 prunemin-only 诊断说明它不能单独修复 Tandt 轨迹损伤。下一版不要长期依赖人工填 baseline count，应继续转向 baseline 预跑、在线增长曲线或 scene scale 自动估计容量下限。
3. 为下一版增加场景自适应保护：当自然结束 Gaussian 数量显著低于 baseline 或 staged target 时，降低 pruning fusion 强度、回退到 baseline pruning，或触发容量保护，避免 Tandt 这类场景被压得过稀。
4. DINO 主线不再追加同类 top-k、percentile、soft-top-k 或单纯 final hard-prune 单点。下一步只做会改变预算行为的实验，例如 RGB/VFM 加权融合的高质量档位复验、按场景自动下调 VFM importance，或把 target budget 与恢复时序绑定。
5. 预算感知 importance cap 已完成 bicycle 420k、430k 放松衰减和 430k quadratic 三个 30k 对照：420k 软预算点为 422,778 个 Gaussians，PSNR/SSIM/LPIPS 为 26.9732 / 0.8273 / 0.1916；430k、start 0.95、min 0.10 为 419,513 个 Gaussians，PSNR/SSIM/LPIPS 为 26.9750 / 0.8270 / 0.1919；430k quadratic 为 418,137 个 Gaussians，PSNR/SSIM/LPIPS 为 26.9402 / 0.8262 / 0.1918。它们都没有保住 i0.50 质量；下一步不迁移全场景，也不继续手工追加相近曲线单点，先改为场景自适应预算或直接估计场景容量。
6. `weighted + importance_weight=0.50` 已完成全 9 场景复验，是 MipNeRF360 预算效率默认候选；`weighted + importance_weight=0.75/0.90` 已补齐全 9 场景，固定均值都不超过 i0.50，但按场景选择后的 `quality_pick` 和 `QCGI pick` 三项质量均优于固定 i0.50。跨数据集收束后，当前第一版推荐两条展示线：保守线用 `dataset_fixed_policy`，质量线用 `dataset_quality_policy`。下一步把规则收束为：预算优先选数据集级固定策略；质量优先时用 QCGI 或同等质量门槛允许小幅 GS 增长；Tandt 这类无候选三项超过 baseline 的数据集直接回退 baseline。单个场景负例只作为回退信号，不再直接否定整个方法，最终以数据集平均和场景分布判断。
7. 已验证的 `support_ratio` 与高置信 prune-protect 都没有优于普通 i0.50，因此下一版不把它们作为主方向；可以保留为诊断分支。
8. 下一版 selector 必须避免 test 泄漏。短期优先级是独立 validation split、预先冻结的数据集级策略和训练过程容量信号；`evaluate_0001_train_selector.py` 已证明当前 train-side 渲染指标不足以替代 test oracle。
9. 重做恢复时序：不再只在训练结束后统一 dense recovery，而是在 staged pruning 发生后立即执行短局部恢复，观察是否能减少中期结构损伤。
10. 大分辨率 ViT-L token-edge i0.50 的三数据集同 recipe 复验已完成。VFM+big 在 MipNeRF360、DB、Tandt 三个数据集上平均 PSNR 和 LPIPS 都小幅优于 FastGS big，SSIM 在 DB/Tandt 略低。下一步设计不依赖 test oracle 的场景级容量/质量选择规则，并优先处理 bicycle/garden、DB drjohnson、Tandt truck 这类负例或高增点场景。
11. 新增 DINO descriptor densify-only 小范围实验线：配置 `configs/experiments/0001_vfm_topology_dinov2_descriptor_densify_only.yaml` 使用 `dinov2_descriptor_cosine`、top-k 15%、token smoothing 3、`vfm_weight=0.0` 和 `vfm_importance_mode=max`。它只让 DINO descriptor residual 影响复制，不改变 pruning score，用于隔离验证“语义结构先验指导新增 GS”的质量贡献。第一批 MipNeRF360 `-r 8` 场景为 bicycle、garden、stump、bonsai。四场景相对 matched FastGS densify100 平均提升 +0.0823 PSNR、+0.0058 SSIM、LPIPS 改善 -0.0110，平均 Gaussian 数量增加 56,204。SSIM/LPIPS 4/4 正向，PSNR 3/4 正向；唯一 PSNR 轻微负向的 bonsai 仍改善 SSIM 和 LPIPS。该结果是目前最干净的“VFM 只指导复制也能提升质量”证据，后续应继续做 descriptor 复制强度和预算效率扫描，而不是把贡献点放在效果不好时回退到 FastGS。
12. DINO descriptor top-k25 质量优先档已完成 MipNeRF360、DB、Tandt 三个公开数据集复验。它保持 `vfm_weight=0.0`，只把 descriptor residual 用于 densification，不改变 pruning score。MipNeRF360 全 9 场景平均提升 +0.1066 PSNR、+0.0050 SSIM、LPIPS 改善 -0.0093，9/9 场景三项指标全部正向；DB 平均弱正向但 `playroom` 单场景负向；Tandt 平均提升 +0.1004 PSNR、+0.0017 SSIM、LPIPS 改善 -0.0016，且两场景全部正向。high-res bicycle 上，`max` 相对 FastGS big 提升 +0.0748 PSNR、+0.0093 SSIM、LPIPS -0.0169，但多 249,213 个 Gaussians；`weighted i0.50` 仍提升 +0.0406 PSNR、+0.0053 SSIM、LPIPS -0.0085，且只多 46,111 个 Gaussians。high-res MipNeRF360 已完成同一 i0.50 全 9 场景，平均相对 FastGS big 为 +0.0615 PSNR、+0.0020 SSIM、LPIPS -0.0035，平均多 56,312 个点，QCGI 为 +0.0633。逐场景看，garden 少点增质，counter/kitchen/room QCGI 为正，stump/bonsai 质量正向但容量过强，flowers 是 PSNR/SSIM 微升但 LPIPS 变差的混合样本，treehill 是明确质量负例；high-res truck 则为 PSNR -0.0835、SSIM -0.0008、LPIPS -0.0019，且多 104,534 个点。该结果是当前 descriptor densify-only 的跨数据集主证据；下一步应对 stump/bonsai 扫描更强容量约束，对 flowers/treehill 扫描更强 descriptor 强度或引入 Depth Anything 几何先验。
13. DINO descriptor weighted i0.50 预算效率探测完成。它保持 top-k15、`vfm_weight=0.0`，把 importance 融合改为 `weighted + importance_weight=0.50`。四场景相对 matched FastGS densify100 平均只多 4,067 个 Gaussians，PSNR +0.0080、SSIM +0.0013、LPIPS 改善 -0.0025；`bicycle` 甚至少 13,288 个点。但 PSNR 在 `bicycle` 和 `stump` 轻微负向，说明该档位过于保守，适合作为接近 baseline 容量的低风险档，不适合作为主要质量贡献证据。
14. DINO descriptor top-k25 + weighted i0.50 容量受控档完成。它保持 `vfm_weight=0.0`，只影响 densification，不改变 pruning score。四场景相对 matched FastGS densify100 平均为 PSNR +0.0465、SSIM +0.0040、LPIPS 改善 -0.0078，平均多 31,351 个 Gaussians，且 4/4 场景三项指标均正向。相比 top-k25 `max` 质量优先档，它少 46,792 个点，但平均 PSNR 低 -0.1320、SSIM 低 -0.0038、LPIPS 差 +0.0063。因此它应定位为容量受控正向档，而不是质量最优档。下一步应跑 `top-k25 + weighted i0.65/i0.70`，寻找介于 top-k25 max 和 i0.50 之间的质量-容量折中。
15. DINO descriptor top-k25 + weighted i0.70 质量折中档完成。四场景相对 matched FastGS densify100 平均为 PSNR +0.0884、SSIM +0.0047、LPIPS 改善 -0.0086，平均多 35,511 个 Gaussians；4/4 场景三项指标均正向，且 `bicycle/stump/bonsai` 的 PSNR 相对 i0.50 明显提升。相比 i0.50，它平均多 4,160 个点并提升 +0.0419 PSNR；相比 top-k25 `max`，它少 42,632 个点但仍低 -0.0901 PSNR。因此 i0.70 是质量折中档。主要问题是训练时间异常偏高：平均比 i0.50 多 62.89s，比 top-k25 `max` 多 58.99s。下一步优先确认该时间开销是否可复现，再考虑 i0.60/i0.65。
16. DINO descriptor top-k25 + weighted i0.65 边界探测完成。四场景相对 matched FastGS densify100 平均为 PSNR +0.0282、SSIM +0.0043、LPIPS 改善 -0.0083，平均多 35,190 个 Gaussians；但 `stump` 和 `bonsai` 的 PSNR 转为负向。相比 i0.50，它平均多 3,839 个点但 PSNR 低 -0.0183；相比 i0.70，它点数几乎相同、训练时间没有回落，PSNR 低 -0.0602。因此 i0.65 不推荐作为主线，后续应优先 profiling descriptor scoring，而不是继续密集扫描相邻权重。
17. 已新增默认关闭的 `vfm_profile_scorer` 与 `vfm_profile_interval`，用于诊断 VFM scorer 内部耗时。bicycle i0.70 30k profiling 复跑为 PSNR 26.9835、SSIM 0.8314、LPIPS 0.1831、445,245 个 Gaussians、训练 184.47s；质量与原 i0.70 基本一致，训练时间低于原记录的 210.12s，但仍高于 i0.50。稳定 scorer 调用约 172-188ms，其中 FastGS 原始 multi-view score 约 63-81ms，在线 DINO descriptor error 约 50-52ms。结论是训练耗时由在线 descriptor 与原始 score 叠加造成，下一步优先做 descriptor 参与窗口或频率控制。
18. 已新增默认关闭的 `vfm_active_from_iter` 与 `vfm_active_until_iter`，用于控制 VFM scorer 介入训练的 iteration 窗口。bicycle 820-step 机制验证中，临时设置 `active_until=650` 后，iteration 600 的首次 DINO descriptor 调用耗时 2041.38ms；iteration 700/800 打印 `active=false`，耗时分别降到 78.08ms 和 76.59ms，基本等于 FastGS 原始 score。该机制确认可以在后半训练阶段跳过在线 DINO descriptor 与 VFM count raster。下一步已排队 bicycle 30k `warm8000`，判断早期 descriptor 引导能否保留 i0.70 的质量收益并降低训练耗时。
19. bicycle 30k `warm8000` 完成，结果为 PSNR 26.9633、SSIM 0.8285、LPIPS 0.1874、432,336 个 Gaussians、训练 177.80s。它相对 FastGS densify100 仍正向，但低于 i0.70 profile 的 26.9835 / 0.8314 / 0.1831，也低于 i0.50 的 26.9644 / 0.8301 / 0.1843；训练时间只比 i0.70 profile 少 6.67s。profile 显示 call=50 仍走 DINO descriptor，call=100 已 `active=false` 并降到 80.28ms，机制成立但收益不足。结论：warm8000 不扩展四场景，后续回到已经正向的 descriptor top-k25 `max` 或 weighted i0.70 多场景验证。

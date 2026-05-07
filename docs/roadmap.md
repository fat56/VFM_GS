# 路线图

## 进行中

- `0001_vfm_topology_scorer`: staged/ratio-aware cached edge 已在 bicycle、garden 和 counter 三个 scene 上超过各自 baseline；no-effect/cadence 控制、post-prune fine-tune、cached edge 严格 240k dense recovery、descriptor 30k 完整训练、descriptor staged 预算对齐、descriptor `rgb_only` 保守接入、descriptor top-k/smoothing 30k、top-k 8% 完整对照、top-k 8% staged 对齐、410k staged 对齐、top-k/smoothing `rgb_only`、soft top-k 30k 完整对照、soft top-k staged 410k 和 percentile 90% 完整对照均已完成。当前结论是 descriptor unpruned 可接近 DINO token-edge，top-k 8% 与 percentile 90% 是更均衡的完整对照点，soft top-k 质量正向但成本更高；预算对齐、降低 top-k ratio、只参与 support/pruning、percentile mask、多层 soft 计数或严格 240k dense recovery 后仍低于 cadence control。

## 排队

- 将 `cached_edge_l1` 固化为 0001 v1 正向控制组，并整理下一版实验入口。
- 继续 descriptor 时只做会改变预算行为的方案：验证 descriptor/top-k staged + `post_prune_finetune_trigger=any_prune`。
- 若 staged 后 dense recovery 仍低于 cadence control，则收束 0001 v1：保留 cached edge staged/ratio-aware 作为正向控制组，把语义 VFM descriptor 的预算高效化转入下一版计划。

## 阻塞

- DINOv2 token-edge 默认 30k 指标领先但预算过大；350k staged budget 下只在 LPIPS 上超过 baseline，PSNR/SSIM 仍低。当前 DINO token-edge 还不能作为预算高效的主结果。
- cached edge 严格 240k final-prune + dense recovery 已比 final-only 和 240k staged 更好，但仍低于 baseline 与 350k staged 正向控制组，说明 baseline-sized budget 对当前裁剪路径过紧。

## 已完成

- 将原生 FastGS 扁平结构迁移到 `src/vfm_gs` 包结构。
- 建立 `vfm_topology_scorer` mock v1，验证 scorer registry 与 FastGS baseline 的输出兼容。
- 建立 `build_vfm_cache` CLI 和 `cached_edge_l1` 后端，验证离线缓存读取链路。
- 为训练输出补充 scorer/backend/阈值/权重等 provenance 日志。
- 增加 `npz_uint8` compact cache storage，将 bicycle edge cache 从约 189MB 降到约 35MB。
- 增加 `validate_vfm_cache` CLI，支持 manifest、checksum、shape、source-image 和 backend 校验。
- 为 cached backend 增加训练前 preflight，提前暴露缺失 cache 或 backend 不匹配。
- 增加 `vfm_backend_probe` CLI，记录当前环境和 DINOv2 cache-size feasibility。
- 增加 optional `dinov2_vits14` / `dinov2_vitb14` cache builder，并在当前 PyTorch 1.12.1 环境完成 4-image ViT-S/14 cache 快速验证。
- 增加 `dinov2_token_edge_l1` scorer backend，完成 194 张图 DINOv2 cache build/validate，以及 220-iteration bicycle 快速验证的 train/render/metrics。
- 增加 `dinov2_descriptor_cosine` scorer backend，完成 80-step 和 220-step bicycle 快速验证的 train/render/metrics，打通在线渲染图 DINO descriptor 与 GT cache descriptor 比较路径。
- 完成 `dinov2_descriptor_cosine` 阈值细扫；`vfm_loss_thresh=0.35` 在 220-step 快速验证中优于 0.30、0.40、默认 0.50 和 0.65。
- 完成 `max_width=518` DINO ViT-S/14 cache build/validate；cache 为 127M，构建 9.90s，descriptor 快速验证只在 PSNR 上略优于 224-cache。
- 完成 `dinov2_descriptor_cosine`、`vfm_loss_thresh=0.35`、`max_width=224` cache 的 30k 完整训练；结果为 PSNR 26.9770，SSIM 0.8298，LPIPS 0.1850，461,846 个 Gaussians，优于 cadence control 但低于 DINO token-edge。
- 完成 descriptor staged 预算对齐；`target_gaussian_count=410000`、`stage_margin=1.05` 后自然结束在 381,726 个 Gaussians，PSNR 26.9064，SSIM 0.8208，LPIPS 0.2021，低于 `fastgs_densify100`。
- 完成 descriptor `rgb_only` 保守接入；禁用直接 descriptor densification 后达到 PSNR 26.9370，SSIM 0.8239，LPIPS 0.1972，407,201 个 Gaussians，与 cadence control 基本持平但不是清晰正向。
- 增加 descriptor metric-map 策略参数，支持 threshold、percentile、top-k 和 DINO token-grid smoothing；120-step top-k/smoothing 集成验证已触发真实 descriptor scoring 和 densification。
- 完成 descriptor top-k/smoothing 30k 对照；PSNR 27.0274，SSIM 0.8330，LPIPS 0.1805，484,229 个 Gaussians，质量接近 DINO token-edge 但预算偏高。
- 完成 descriptor top-k 8% / smoothing 30k 对照；PSNR 26.9931，SSIM 0.8301，LPIPS 0.1849，456,567 个 Gaussians，比 top-k 15% 更省点、更快，但仍高于 cadence control 预算。
- 完成 descriptor top-k/smoothing 410k staged target；PSNR 26.9047，SSIM 0.8219，LPIPS 0.1998，389,250 个 Gaussians，预算对齐后仍低于 cadence control。
- 完成 descriptor top-k 8% / smoothing 410k staged target；PSNR 26.8783，SSIM 0.8208，LPIPS 0.2013，382,035 个 Gaussians，低于 top-k 15% staged 和 cadence control。
- 完成 descriptor top-k/smoothing `rgb_only`；PSNR 26.9117，SSIM 0.8237，LPIPS 0.1977，412,317 个 Gaussians，与 cadence control 点数贴合但质量略低。
- 增加 descriptor `soft_topk` 多层近似 metric map；620-step 集成验证已触发真实 descriptor scoring、3 层嵌套 top-k 计数和 densification。
- 完成 descriptor soft top-k 30k 完整对照；PSNR 26.9875，SSIM 0.8305，LPIPS 0.1844，462,696 个 Gaussians，质量高于 cadence control，但成本和点数仍偏高。
- 完成 descriptor soft top-k staged 410k；PSNR 26.8848，SSIM 0.8201，LPIPS 0.2029，383,528 个 Gaussians，预算对齐后低于 cadence control。
- 完成 descriptor percentile 90% / smoothing 30k 完整对照；PSNR 27.0036，SSIM 0.8313，LPIPS 0.1827，464,425 个 Gaussians，质量介于 top-k 15% 和 top-k 8% 之间，但仍高于 cadence control 预算。
- 完成 baseline、compact cached edge、DINOv2 token-edge 的 30k `-r 8` matched ablation；DINO token-edge 指标最好，但点数和渲染成本也最高。
- 完成 t075/w010 budget-control probe；现有阈值/权重 knob 无法充分控制 VFM densification 点数。
- 增加 `vfm_importance_weight` 并完成 i0.25 30k probe；DINO 点数下降但仍未达成 budget matching。
- 增加 `vfm_importance_mode=max|weighted|rgb_only` 并完成 `rgb_only` 30k probe；直接关闭 VFM densification 仍未达成 budget matching。
- 增加 `target_gaussian_count` final-prune control；首版 high-score 批量裁剪是负例，已修正为 low-score 批量裁剪。
- 完成 low-score final target-prune 30k probe；点数精确匹配 baseline，但 edge/DINO 质量均低于 baseline，说明单次最终裁剪不是可用的公平预算方案。
- 增加 `target_gaussian_staged` / `target_gaussian_stage_margin` / `target_gaussian_stage_interval`，支持训练期分阶段预算控制。
- 完成 240k staged budget 30k probe；质量较 final-only 大幅恢复，但仍低于 baseline。
- 完成 300k staged budget 30k probe；DINO LPIPS 几乎追平 baseline，但 PSNR/SSIM 仍低。
- 完成 350k staged budget 30k probe；cached edge 在约 340k 点数下超过 baseline，DINO 350k 仅 LPIPS 超过 baseline。
- 完成 garden staged-budget edge 复验；edge 在第二个 scene 上继续超过 baseline。
- 完成 counter staged/ratio-aware edge 复验；edge 在第三个 scene 上继续超过 baseline，且 Gaussian count 低于自身 baseline。
- 完成 no-effect/cadence control；`fastgs_photometric + densification_interval=100` 与 zero-weight VFM runs 都在约 410k Gaussians，说明此前 no-effect 高点数主要来自 densification cadence。
- 增加 `post_prune_finetune_iterations` 并完成 final-prune-plus-fine-tune 探测；严格 240k 预算下质量明显优于 final-only，但仍低于 baseline 与 350k staged 正向控制组。
- 增加 dense recovery 调度参数；支持恢复阶段独立 optimizer step interval、SH step interval、局部 xyz LR 和 staged/any-prune 触发。260-step 快速验证已完成，能从 88,194 裁到 65,000 并保存 `ours_280`。
- 完成 cached edge 严格 240k final-prune + dense recovery 完整对照；PSNR 26.2470，SSIM 0.7813，LPIPS 0.2526，优于默认 cadence 恢复但仍低于 baseline。

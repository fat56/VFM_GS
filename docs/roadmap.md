# 路线图

## 进行中

- `0001 adaptive weighted`: 已在 bicycle 上完成 `adaptive_weighted + quadratic 430k` 30k 对照，结果为 PSNR 26.9858、SSIM 0.8302、LPIPS 0.1853、424,011 个 Gaussians。相比 `weighted i0.50` 三项质量微升但多 8,853 个点；相比普通 i0.50 少 16,060 个点但质量仍小幅低一点。treehill 第二场景复验为 PSNR 24.4393、SSIM 0.7285、LPIPS 0.2821、420,283 个 Gaussians，相比普通 treehill i0.50 的 PSNR 低 -0.0780，相比 treehill `weighted i0.50` 也低 -0.0708。该曲线没有通过第二场景验证，后续不再扩展为全场景候选。
- `0001_vfm_topology_scorer`: 第一版实验进入收束阶段。`cached_edge_l1 + staged target ~= 1.42x baseline count` 已完成 MipNeRF360 全 9 场景评估，平均 PSNR/SSIM/LPIPS 均超过 baseline，是 0001 v1 的正向控制组；但平均 Gaussian 数量增加约 24.5%，训练时间增加约 11.1%，且 `treehill` 是明确负例。Tandt/DB 新增全场景评估显示跨数据集差异：DB 两场景平均正向，Tandt 两场景平均负向且 Gaussian 数量下降约 37.3%。已增加默认关闭的 `prune_min_gaussian_count` 并完成 Tandt 容量保护诊断：平均结果从 cached edge v1 的 PSNR 25.5799、SSIM 0.9316、LPIPS 0.0608 恢复到 25.7806、0.9337、0.0585，但仍低于 baseline 的 25.9551、0.9377、0.0541。DINO token-edge `top-k25 + i0.50` 已完成 MipNeRF360 全 9 场景评估，均值 28.8577 / 0.8666 / 0.1385、263,572 个 Gaussians，相对 baseline 为 +0.2051 PSNR、+0.0115 SSIM、LPIPS -0.0234，是当前全场景质量候选。`weighted + i0.50` 已完成全 9 场景复验，均值 28.8505 / 0.8660 / 0.1397、254,736 个 Gaussians，相比普通 i0.50 少 8,836 点、训练少 2.87s，质量只小幅回落，因此是预算效率候选。`weighted + i0.75` 已在 bicycle、stump、room 和 bonsai 复验：bicycle 为 26.9909 / 0.8309 / 0.1844、414,563 点，相比 weighted i0.50 少 595 点且三项质量提升；stump 为 27.6183 / 0.8178 / 0.1929、370,556 点，相比 weighted i0.50 质量微升但多 16,510 点；room 为 33.1334 / 0.9626 / 0.0575、101,965 点，相比 weighted i0.50 PSNR +0.0253、点数只多 581、LPIPS 极小幅回落；bonsai 为 32.4848 / 0.9642 / 0.0497、138,808 点，仍优于 baseline/cached-edge，但比 weighted i0.50 低 -0.0547 PSNR 且多 4,002 点。因此 i0.75 是有条件高质量档位，不是默认省点档位，也不是无条件质量默认值。已新增 `0001_vfm_topology_dinov2_token_edge_weighted_i050.yaml` 和 `0001_vfm_topology_dinov2_token_edge_weighted_i075.yaml`，分别作为预算效率档和高质量档的可复用入口；`scripts/summarize_0001_weighted_candidates.py` 也已把 weighted 选择规则做成可重复生成的 summary/comparison/recommendation 表。support-ratio、高置信保护、软预算衰减和 adaptive weighted 都已验证但没有超过 `weighted i0.50` 的全场景结论；`adaptive_weighted + quadratic 430k` 在 treehill 第二场景未通过。下一步应把容量保护升级为自动场景自适应机制，并把 weighted 选择规则接入后续批量评估流程。
- `0001 Tandt DINO weighted`: 已完成 Tandt `train/truck` 的 `dinov2_token_edge_l1 + top-k25 + weighted i0.50` 复验，平均 PSNR/SSIM/LPIPS 为 25.7519 / 0.9346 / 0.0575，38,394 个 Gaussians，训练 151.05s。它相对 Tandt cached-edge v1 平均提升 +0.1721 PSNR、+0.0031 SSIM、LPIPS 改善 -0.0033，两个场景均三项指标正向；但相对 Tandt baseline 仍低 -0.2032 PSNR、-0.0031 SSIM、LPIPS 差 +0.0035。结论是 DINO weighted 能修复一部分 cached-edge Tandt 负例，但不能作为 Tandt 默认方案。下一步继续用同一脚本跑 DB 两场景，判断 DINO weighted 在 DB 的收益是相对 cached-edge 修复还是能直接超过 baseline。
- `0001 DB DINO weighted`: 已完成 DB `drjohnson/playroom` 的同条件复验，平均 PSNR/SSIM/LPIPS 为 30.3603 / 0.9360 / 0.0641，62,261 个 Gaussians，训练 141.48s。它相对 DB baseline 平均提升 +0.2423 PSNR、+0.0035 SSIM、LPIPS 改善 -0.0017，但相对 DB cached-edge v1 平均低 -0.2028 PSNR、-0.0001 SSIM、LPIPS 差 +0.0004。`drjohnson` 同时超过 baseline/cached-edge，`playroom` 则低于 cached-edge。结论是 DINO weighted i0.50 可以作为跨数据集 baseline 正向候选，但不能替代 DB 上已更强的 cached-edge proxy；下一步应把 backend 选择规则做成按场景评估，而不是继续寻找单一固定后端。
- 已新增 `prune_min_gaussian_target_ratio`，默认关闭。它会在训练开始时从 `target_gaussian_count` 自动派生有效 `prune_min_gaussian_count`，并写入 `cfg_args`；显式 `prune_min_gaussian_count` 仍优先。配套配置 `0001_vfm_topology_cached_edge_auto_prunemin.yaml` 使用 ratio `0.7042253521126761`，当 target 仍取 `baseline * 1.42` 时可自动还原 baseline 容量下限。Tandt `train/truck` 自动下限复验已完成，平均 PSNR/SSIM/LPIPS 为 25.7473 / 0.9330 / 0.0594，50,370 个 Gaussians；它相对原始 cached edge v1 有恢复，但略低于手动容量保护，结论是机制可用、质量不新增突破。

## 排队

- 下一版优先设计预算感知 scorer 和自动容量保护，而不是继续堆叠 descriptor mask ratio：候选方向包括当自然点数显著低于 baseline 预测值时回退 pruning 强度、在 staged pruning 后立即短恢复而非训练结束统一恢复。按 Gaussian 支持度归一化 VFM score 和高置信 VFM 区域保护都已完成 bicycle 30k 验证，均不优于普通 i0.50；`weighted` importance 已完成 MipNeRF360 全 9 场景 i0.50 复验，平均 PSNR 28.8505、SSIM 0.8660、LPIPS 0.1397、254,736 个 Gaussians、训练 137.60s。相比普通 i0.50，它平均少 8,836 个点、训练少 2.87s，质量只小幅回落 -0.0072 PSNR、-0.0006 SSIM、LPIPS 差 +0.0012。`weighted i0.75` 已在 bicycle/stump/room 复验为 PSNR 正向，但 bonsai 未超过 weighted i0.50。`prune_min_gaussian_target_ratio` 已把手工容量下限推进为默认关闭的自动派生机制；下一步考虑让 ratio 来自 baseline 预跑或在线增长曲线。后续不再优先做同类手工单点，改为固化选择规则：预算优先选 weighted i0.50，质量优先且可接受点数或 LPIPS 小幅波动时才选 weighted i0.75。
- 已增加默认关闭的 `vfm_importance_budget_count` 软预算机制，并完成 bicycle 620-step、420k 30k、430k 放松衰减和 430k quadratic 对照。420k 软预算点最终 422,778 个 Gaussians，PSNR/SSIM/LPIPS 为 26.9732 / 0.8273 / 0.1916；430k、start 0.95、min 0.10 对照最终 419,513 个 Gaussians，PSNR/SSIM/LPIPS 为 26.9750 / 0.8270 / 0.1919；430k quadratic 对照最终 418,137 个 Gaussians，PSNR/SSIM/LPIPS 为 26.9402 / 0.8262 / 0.1918。三者都未保住普通 i0.50 质量。下一步不继续手工追加相近预算曲线单点，应转向场景自适应预算或直接估计场景容量。
- 已增加 `adaptive_weighted` importance mode，用同一个预算进度在 `max` partial importance 和 `weighted` importance 之间平滑过渡。bicycle 620-step 快速验证完成 train/render/metrics；30k linear 430k 对照为 26.9724 / 0.8292 / 0.1859、421,472 个 Gaussians，属于混合结果。quadratic 430k 在 bicycle 上是正向单点，但 treehill 复验 PSNR 明显回落，因此不作为下一版主线。

## 阻塞

- DINOv2 token-edge i0.50 已成为 MipNeRF360 全场景质量更强的 v1 candidate，但平均 Gaussian 数量仍比 baseline 多约 52.1%。当前 DINO token-edge 还不能作为预算高效的主结果。
- cached edge 严格 240k final-prune + dense recovery 已比 final-only 和 240k staged 更好，但仍低于 baseline 与 350k staged 正向控制组，说明 baseline-sized budget 对当前裁剪路径过紧。
- descriptor staged 后 dense recovery 只带来轻微 LPIPS/SSIM 修复，PSNR 反而下降，仍低于 `fastgs_densify100`。这说明“训练结束后统一恢复”不能充分修复训练中期 staged pruning 的结构损伤。

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
- 完成 descriptor top-k/smoothing staged 410k + dense recovery；PSNR 26.8472，SSIM 0.8223，LPIPS 0.1974，387,109 个 Gaussians，LPIPS 较无恢复 staged 版本小幅改善但整体仍低于 cadence control。
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
- 完成 MipNeRF360 全 9 场景 baseline 与 cached-edge v1 统一评估；cached-edge v1 平均 PSNR 28.7213、SSIM 0.8579、LPIPS 0.1551，baseline 平均 PSNR 28.6527、SSIM 0.8551、LPIPS 0.1620。平均增益为 +0.0686 PSNR、+0.0028 SSIM、-0.0068 LPIPS。
- 完成 Tandt/DB 全场景 baseline 与 cached-edge v1 统一评估；DB 平均增益为 +0.4451 PSNR、+0.0037 SSIM、-0.0021 LPIPS，Tandt 平均变化为 -0.3752 PSNR、-0.0061 SSIM、+0.0067 LPIPS，并且 Tandt cached-edge v1 平均 Gaussian 数量比 baseline 少约 37.3%。
- 增加 `prune_min_gaussian_count`，默认关闭，用于容量保护诊断。Tandt 两场景保护后平均 PSNR/SSIM/LPIPS 从 cached edge v1 的 25.5799 / 0.9316 / 0.0608 恢复到 25.7806 / 0.9337 / 0.0585，并把平均 Gaussian 数量恢复到 baseline 的 50,370；但仍低于 baseline 质量，说明容量保护只是防线，不是完整解法。
- 增加 `0001_vfm_topology_dinov2_token_edge_topk.yaml`，把 DINO token-edge 的 metric map 改为 top-k 15%。bicycle 620-step 快速验证完成 train/render/metrics，结果为 PSNR 20.8432、SSIM 0.4752、LPIPS 0.5460，61,555 个 Gaussians；该结果只说明链路健康。
- 完成 DINO token-edge top-k 15% bicycle 30k 对照；PSNR 27.0223、SSIM 0.8322、LPIPS 0.1810，464,998 个 Gaussians，训练 140.40s。相比原始 baseline 清晰正向；相比默认 DINO token-edge 少 25,834 个点、训练少 25.71s，但质量略低。
- 完成 DINO token-edge top-k 25% bicycle 30k 对照；PSNR 27.0636、SSIM 0.8354、LPIPS 0.1748，497,328 个 Gaussians，训练 146.76s。相比默认 DINO token-edge 三项指标小幅提升，相比原始 baseline 提升 +0.3604 PSNR、+0.0287 SSIM、LPIPS 改善 -0.0530。
- 完成 DINO token-edge top-k 25% staged 490,832 预算探测；最终 453,505 个 Gaussians，PSNR 27.0001、SSIM 0.8286、LPIPS 0.1887。它仍优于 baseline 和 cadence control，但低于 top-k 15%、默认 DINO token-edge 和 top-k 25% 完整对照。
- 完成 DINO token-edge top-k 25% final 490,832 预算探测；训练结束从 497,555 裁到 490,832，只删除 6,723 个点，但 PSNR/SSIM/LPIPS 回落到 26.8466、0.8244、0.1858，说明当前 low-score final target-prune 排序不适合作为温和预算约束。
- 完成 DINO token-edge top-k 25% `rgb_only` 预算贴合探测；最终 411,539 个 Gaussians，PSNR 26.9340、SSIM 0.8236、LPIPS 0.1981。它贴近 `fastgs_densify100` 点数，但只在 PSNR 上微幅超过，SSIM/LPIPS 不占优。
- 完成 DINO token-edge top-k 25% `importance_weight=0.25` partial importance 探测；最终 420,361 个 Gaussians，PSNR 26.9515、SSIM 0.8262、LPIPS 0.1920。相比 `fastgs_densify100` cadence control，三项指标均小幅占优，说明完全关闭 VFM densification 不如保留弱 VFM 引导。
- 完成 DINO token-edge top-k 25% `importance_weight=0.50` partial importance 探测；最终 440,071 个 Gaussians，PSNR 26.9966、SSIM 0.8303、LPIPS 0.1842。相比 `importance_weight=0.25`，多 19,710 个点并带来 +0.0451 PSNR、+0.0041 SSIM、LPIPS 改善 -0.0078，说明 partial importance 曲线仍在有效区间内。
- 完成 DINO token-edge top-k 25% `importance_weight=0.50` garden 30k 复验；最终 262,385 个 Gaussians，PSNR 28.9644、SSIM 0.8986、LPIPS 0.0954。相比 garden cached-edge v1 仍提升 +0.0641 PSNR、+0.0024 SSIM、LPIPS 改善 -0.0048，但多 13,981 个点、训练多 4.28s。
- 完成 DINO token-edge top-k 25% `importance_weight=0.50` counter 30k 复验；最终 119,695 个 Gaussians，PSNR 29.7174、SSIM 0.9338、LPIPS 0.0751。相比 counter baseline 提升 +0.1763 PSNR、+0.0026 SSIM、LPIPS 改善 -0.0055，点数只多 6,672 个；相比 counter cached-edge v1 仍提升 +0.0746 PSNR、+0.0017 SSIM、LPIPS 改善 -0.0037。
- 完成 DINO token-edge top-k 25% `importance_weight=0.50` treehill 30k 压力测试；最终 432,520 个 Gaussians，PSNR 24.5173、SSIM 0.7284、LPIPS 0.2822。相比 treehill baseline，PSNR 低 -0.0344，但 SSIM 高 +0.0110、LPIPS 改善 -0.0407；相比 cached-edge v1 三项指标均正向，但点数更多。
- 完成 DINO token-edge top-k 25% `importance_weight=0.50` bonsai 30k 复验；最终 136,305 个 Gaussians，PSNR 32.4920、SSIM 0.9640、LPIPS 0.0500。相比 bonsai baseline 提升 +0.1346 PSNR、+0.0044 SSIM、LPIPS 改善 -0.0123；相比 cached-edge v1 仍提升 +0.2654 PSNR、+0.0040 SSIM、LPIPS 改善 -0.0088。
- 完成 DINO token-edge top-k 25% `importance_weight=0.50` flowers 30k 复验；最终 350,421 个 Gaussians，PSNR 23.0134、SSIM 0.6960、LPIPS 0.2747。相比 flowers baseline 提升 +0.2592 PSNR、+0.0237 SSIM、LPIPS 改善 -0.0440；相比 cached-edge v1 仍提升 +0.0459 PSNR、+0.0069 SSIM、LPIPS 改善 -0.0115。
- 完成 DINO token-edge top-k 25% `importance_weight=0.50` kitchen 30k 复验；最终 161,347 个 Gaussians，PSNR 33.3358、SSIM 0.9693、LPIPS 0.0344。相比 kitchen baseline 提升 +0.2438 PSNR、+0.0021 SSIM、LPIPS 改善 -0.0035，且 Gaussian 数量少 7,629 个；相比 cached-edge v1 仍提升 +0.0256 PSNR、+0.0002 SSIM、LPIPS 改善 -0.0006。
- 完成 DINO token-edge top-k 25% `importance_weight=0.50` room 30k 复验；最终 103,820 个 Gaussians，PSNR 33.0721、SSIM 0.9622、LPIPS 0.0574。相比 room baseline 提升 +0.0945 PSNR、+0.0025 SSIM、LPIPS 改善 -0.0037；相比 cached-edge v1 仍提升 +0.1037 PSNR、+0.0002 SSIM、LPIPS 改善 -0.0004，训练时间少 1.69s。
- 完成 DINO token-edge top-k 25% `importance_weight=0.50` stump 30k 复验；最终 365,584 个 Gaussians，PSNR 27.6106、SSIM 0.8168、LPIPS 0.1935。相比 stump baseline 提升 +0.4350 PSNR、+0.0234 SSIM、LPIPS 改善 -0.0393；相比 cached-edge v1 仍提升 +0.3632 PSNR、+0.0236 SSIM、LPIPS 改善 -0.0367，训练时间少 5.70s。
- 完成 DINO token-edge top-k 25% `importance_weight=0.50` MipNeRF360 全 9 场景 candidate 汇总；平均 PSNR 28.8577、SSIM 0.8666、LPIPS 0.1385、263,572 个 Gaussians。相对 baseline 平均提升 +0.2051 PSNR、+0.0115 SSIM、LPIPS 改善 -0.0234；相对 cached-edge v1 平均提升 +0.1365 PSNR、+0.0087 SSIM、LPIPS 改善 -0.0166，但平均 Gaussian 数量仍比 baseline 多 90,231。
- 完成 DINO token-edge top-k 25% `importance_weight=0.75` partial importance 探测；最终 472,164 个 Gaussians，PSNR 27.0284、SSIM 0.8332、LPIPS 0.1788。它略优于 top-k 15%，但相对 `importance_weight=0.50` 的边际收益放缓，因此 partial importance 单变量扫描在这里收束。
- 完成 DINO token-edge top-k 25% `weighted + importance_weight=0.50` bicycle 30k 对照；最终 415,158 个 Gaussians，PSNR 26.9756、SSIM 0.8288、LPIPS 0.1867，训练 141.21s。相比 cadence control 三项指标均正向，且点数只多 3,080；相比普通 `max + importance_weight=0.50` 少 24,913 个点但质量小幅回落，是新的近预算效率候选。
- 完成 DINO token-edge top-k 25% `weighted + importance_weight=0.50` treehill 30k 压力复验；最终 417,534 个 Gaussians，PSNR 24.5101、SSIM 0.7281、LPIPS 0.2837，训练 139.79s。相比普通 treehill i0.50 少 14,986 个点，质量小幅回落；相比 baseline 仍保住 SSIM/LPIPS 明显优势但 PSNR 略低。
- 完成 DINO token-edge top-k 25% `weighted + importance_weight=0.50` stump 30k 大收益场景复验；最终 354,046 个 Gaussians，PSNR 27.6147、SSIM 0.8170、LPIPS 0.1934，训练 136.36s。相比普通 stump i0.50 少 11,538 个点且质量基本持平；相比 baseline 提升 +0.4391 PSNR、+0.0236 SSIM、LPIPS 改善 -0.0393。
- 完成 DINO token-edge top-k 25% `weighted + importance_weight=0.50` counter 30k 低增点场景复验；最终 119,273 个 Gaussians，PSNR 29.6650、SSIM 0.9333、LPIPS 0.0752，训练 133.34s。相比 baseline 和 cached-edge v1 仍三项正向，但相比普通 counter i0.50 只少 422 个点且 PSNR 回落 -0.0524，因此 weighted 不适合默认替代低增点场景的普通 i0.50。
- 完成 DINO token-edge top-k 25% `weighted + importance_weight=0.50` garden 30k 中等点数增长复验；最终 253,355 个 Gaussians，PSNR 28.9546、SSIM 0.8977、LPIPS 0.0974，训练 134.56s。相比普通 garden i0.50 少 9,030 个点、训练少 4.75s，质量小幅回落；相比 baseline 和 cached-edge v1 仍三项正向。
- 完成 DINO token-edge top-k 25% `weighted + importance_weight=0.50` flowers 30k 高增点植被场景复验；最终 339,267 个 Gaussians，PSNR 22.9636、SSIM 0.6933、LPIPS 0.2791，训练 141.18s。相比普通 flowers i0.50 少 11,154 个点、训练少 4.12s，但质量回落更明显；相比 baseline 仍明显正向，相比 cached-edge v1 保住 SSIM/LPIPS 改善。
- 完成 DINO token-edge top-k 25% `weighted + importance_weight=0.50` bonsai 30k 中低增点场景复验；最终 134,806 个 Gaussians，PSNR 32.5395、SSIM 0.9642、LPIPS 0.0493，训练 138.31s。相比普通 bonsai i0.50 少 1,499 个点、训练少 1.01s，同时三项质量微升；相比 baseline 和 cached-edge v1 也三项正向。
- 完成 DINO token-edge top-k 25% `weighted + importance_weight=0.75` bonsai 30k 边界复验；最终 138,808 个 Gaussians，PSNR 32.4848、SSIM 0.9642、LPIPS 0.0497，训练 138.14s。相比 baseline 与 cached-edge v1 仍正向，但相比 bonsai weighted i0.50 多 4,002 个点且 PSNR 低 -0.0547、LPIPS 差 +0.0004，因此 bonsai 推荐保持 weighted i0.50。
- 完成 DINO token-edge top-k 25% `weighted + importance_weight=0.50` kitchen 30k 室内高基线场景复验；最终 157,804 个 Gaussians，PSNR 33.3234、SSIM 0.9693、LPIPS 0.0347，训练 141.71s。相比普通 kitchen i0.50 少 3,543 个点、训练少 3.21s，质量只小幅回落；相比 baseline 和 cached-edge v1 三项指标仍正向，且 Gaussian 数量更少。
- 完成 DINO token-edge top-k 25% `weighted + importance_weight=0.50` room 30k 室内房间场景复验；最终 101,384 个 Gaussians，PSNR 33.1081、SSIM 0.9626、LPIPS 0.0574，训练 131.98s。相比普通 room i0.50 少 2,436 个点、训练少 1.76s，同时 PSNR 和 SSIM 微升、LPIPS 基本持平。
- 完成 DINO token-edge top-k 25% `weighted + importance_weight=0.50` MipNeRF360 全 9 场景汇总；平均 PSNR 28.8505、SSIM 0.8660、LPIPS 0.1397、254,736 个 Gaussians、训练 137.60s。相比 baseline 平均提升 +0.1978 PSNR、+0.0109 SSIM、LPIPS 改善 -0.0223；相比普通 i0.50 平均少 8,836 个点、训练少 2.87s，质量只小幅回落。
- 增加 `vfm_importance_normalizer=none|support_ratio`，默认关闭。support-ratio 模式在 densification 打分时额外统计每个 Gaussian 的可见像素支持度，再用 VFM 命中比例调节 VFM importance；620-step bicycle 快速验证完成 train/render/metrics，PSNR 20.8465、SSIM 0.4756、LPIPS 0.5468，61,517 个 Gaussians，说明链路健康。
- 完成 `support_ratio + importance_weight=0.50` bicycle 30k 对照；最终 432,948 个 Gaussians，PSNR 26.9694、SSIM 0.8286、LPIPS 0.1878，训练 149.40s。它仍优于 cadence control，但比普通 i0.50 少点的同时质量回落，因此不作为下一版主方向。
- 增加 `vfm_prune_protect_weight` 等默认关闭参数，把高置信 VFM 命中区域转换为 pruning-side 保护分数。620-step bicycle 快速验证完成 train/render/metrics，PSNR 20.8808、SSIM 0.4757、LPIPS 0.5457，61,604 个 Gaussians，说明保护分数链路健康。
- 完成高置信 VFM 区域保护 30k bicycle 对照；最终 441,352 个 Gaussians，PSNR 26.9910、SSIM 0.8302、LPIPS 0.1839，训练 144.70s。它相对普通 i0.50 只微幅改善 LPIPS，PSNR/SSIM 和训练时间均不占优，因此该分支收束为负结果。
- 增加 `vfm_importance_budget_count`、`vfm_importance_budget_start_ratio` 和 `vfm_importance_budget_min_weight`，默认关闭。预算感知 importance 会在训练期根据当前 Gaussian 数量软衰减 VFM densification 权重，而不是训练结束后硬裁剪。620-step bicycle 快速验证完成 train/render/metrics，PSNR 20.7641、SSIM 0.4749、LPIPS 0.5459，61,590 个 Gaussians，说明链路健康。
- 完成预算感知 importance 420k bicycle 30k 对照；最终 422,778 个 Gaussians，PSNR 26.9732、SSIM 0.8273、LPIPS 0.1916，训练 140.65s。相比固定 i0.25 三项小幅正向，但相比普通 i0.50 质量回落，说明线性软预算优于固定低权重但还不是新主结果。
- 完成预算感知 importance 430k 放松衰减 bicycle 30k 对照；`start_ratio=0.95`、`min_weight=0.10` 后最终 419,513 个 Gaussians，PSNR 26.9750、SSIM 0.8270、LPIPS 0.1919，训练 139.38s。相比 420k 软预算没有形成质量提升，说明继续微调线性衰减起点收益有限。
- 增加 `vfm_importance_budget_curve=linear|quadratic|sqrt`，默认 `linear` 保持历史行为。完成 430k quadratic bicycle 30k 对照；最终 418,137 个 Gaussians，PSNR 26.9402、SSIM 0.8262、LPIPS 0.1918，训练 140.00s。它低于线性软预算点，说明 late-decay 二次曲线不是当前解法。
- 增加 `target_gaussian_prune_order`，默认保持 `lowest_score`，用于显式复现不同 target-prune 排序。完成 high-score final 490,832 bicycle 30k 对照；最终 PSNR 24.0554、SSIM 0.7914、LPIPS 0.2040，说明终局高分批量裁剪比 low-score final 更差，target-prune 不能简单复用训练期 pruning 的高分删除语义。
- 完成 no-effect/cadence control；`fastgs_photometric + densification_interval=100` 与 zero-weight VFM runs 都在约 410k Gaussians，说明此前 no-effect 高点数主要来自 densification cadence。
- 增加 `post_prune_finetune_iterations` 并完成 final-prune-plus-fine-tune 探测；严格 240k 预算下质量明显优于 final-only，但仍低于 baseline 与 350k staged 正向控制组。
- 增加 dense recovery 调度参数；支持恢复阶段独立 optimizer step interval、SH step interval、局部 xyz LR 和 staged/any-prune 触发。260-step 快速验证已完成，能从 88,194 裁到 65,000 并保存 `ours_280`。
- 完成 cached edge 严格 240k final-prune + dense recovery 完整对照；PSNR 26.2470，SSIM 0.7813，LPIPS 0.2526，优于默认 cadence 恢复但仍低于 baseline。

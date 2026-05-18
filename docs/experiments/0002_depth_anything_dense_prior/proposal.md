# 0002 Depth Anything Dense Depth Prior

## 核心假设

Depth Anything 这类 dense monocular depth prior 能提供比 COLMAP sparse edge 更完整的几何/遮挡边界信号。若将 dense depth residual 或 depth-edge prior 转换为 FastGS 已验证的 `metric_map`，并且只作用于 densification，则可以在不直接改变 pruning score 的前提下，补充 0001 DINO descriptor 对几何边界和 high-res 容量控制的不足。

0002 的第一阶段不追求直接替代 0001 的 DINO descriptor 主线，而是验证几何先验是否提供互补正向信号。

在正式推进 Depth Anything 前，必须先完成 phase 0：使用双卡 RTX 5090 在三个公开数据集的所有场景上重跑一次 FastGS big baseline，保持原图输入和 FastGS 原始 1.6K 自动缩放口径，确认 5090 环境与此前 4090D 结果相差无几。这个步骤用于隔离硬件、CUDA、PyTorch 和本地 CUDA extension 迁移带来的变量；未通过前不进入 Depth Anything 训练。

## 0001 继承结论

0001 已经确认 VFM_GS 的语义先验初步成立：

- DINO descriptor densify-only top-k25 `max` 在 MipNeRF360、DB、Tandt 三个公开数据集均值均正向。
- DINO descriptor densify-only top-k25 weighted i0.50 是容量受控正向档。
- DINO descriptor densify-only top-k25 weighted i0.70 是质量-容量折中档。
- 上述 descriptor 档位保持 `vfm_weight=0.0`，只影响 densification，不改变 pruning score。

0001 也排除了几个不应继续投入的方向：

- COLMAP sparse depth-edge L1 和 sparse depth-edge prior 覆盖太稀，均为负结果。
- final/staged target-prune 和硬 candidate cap 会破坏有效 densification 分布。
- Tandt 的 token-edge 负例不能靠最终容量保护、高权重或关闭 pruning fusion 单点修复。

因此 0002 应直接使用 dense depth prior，不再继续扫描 COLMAP sparse proxy。

## 变体 / 配置

- 变体：`fastgs_baseline`
- 打分器：`vfm_topology_scorer`
- 新后端候选：
  - `depth_anything_depth_edge_prior`：已实现并通过 high-res `bicycle` 620-step smoke 与 30k pilot；30k matched 对照为弱混合信号。
  - `depth_anything_depth_prior`：已实现并完成 high-res `bicycle` 30k，三项质量正向且点数少于 matched baseline；`fastgs_big + scene overrides` 下 `stump` 为三项质量正向，`playroom` 为薄正例，`bonsai` 为容量效率不足的边界样本，`truck` 为明确负例；当前不直接扩全数据集。
  - `depth_anything_depth_residual` / `depth_anything_depth_edge_residual`：保留为在线 residual 后续方向，尚未实现。
- 0001 对照：
  - `dinov2_descriptor_cosine + top-k25 + weighted i0.50 + vfm_weight=0.0`
  - `dinov2_descriptor_cosine + top-k25 + weighted i0.70 + vfm_weight=0.0`
- 初始原则：
  - 只影响 densification。
  - 默认 `vfm_weight=0.0`。
  - 输出统一为 `pixel_error_map`。
  - 仍通过 top-k metric map 和 `render_fastgs(..., get_flag=True)` 映射到 Gaussian counts。

已新增配置：

- `configs/experiments/0002_depth_anything_depth_edge_prior_densify_only_topk025_weighted_i050.yaml`
- `configs/experiments/0002_depth_anything_depth_prior_densify_only_topk025_weighted_i050.yaml`
- `configs/experiments/0002_depth_anything_depth_prior_rgb_rerank_final_topm_l025.yaml`
- `configs/experiments/0002_depth_anything_depth_prior_rgb_rerank_final_topm_l010.yaml`
- `configs/experiments/0002_depth_anything_depth_prior_rgb_rerank_final_topm_l005.yaml`
- `configs/experiments/0002_depth_anything_depth_prior_rgb_rerank_final_topm_l010_broad035.yaml`
- `configs/experiments/0002_depth_anything_depth_prior_rgb_rerank_final_topm_l010_broad035_start9000.yaml`
- `configs/experiments/0002_depth_anything_depth_prior_prune_protect_topk005.yaml`
- `configs/experiments/0002_depth_anything_depth_prior_prune_protect_weight015_topk010.yaml`

当前优先级已转为 direct relative depth prior，并在此基础上尝试 RGB-gated rerank 的保守扫描。depth-edge prior 只保留为弱混合信号对照，不继续扩展。2026-05-18 的 `l0.10 + broad035` 说明缩小 RGB broad candidate 能压低部分容量代价，但仍不能让 `stump/playroom/truck` 同时稳定正向，也不能解决 `truck` depth prior 与 RGB 瓶颈错位。`start9000` 后期介入能让 `playroom` 回到薄正向，但 `stump` 容量失控、`truck` 仍负向。按用户建议扩展到 MipNeRF360 全 9 场景后，均值质量小幅正向但平均多 237,716 个 Gaussians，9/9 场景 QCGI 为负；该结果不支持继续把当前 RGB-gated depth rerank 作为训练主线。2026-05-18 的 prune-protect 分支显示后期辅助裁剪比 densify rerank 稳定得多，但 `topk010`、`weight015`、`topk005` 三轮都没有形成跨场景稳定正收益：`topk010` 三场景平均接近持平，`weight015` 证明单独降保护权重会伤害 playroom，`topk005` 证明继续收窄 RGB proposal 会让 truck 变好但 playroom 变差。当前判断是 Depth Anything prune-side auxiliary 有轻量后验修正价值，但不应直接作为全局默认方案。

## Phase 0：5090 FastGS Big Baseline 复核

Phase 0 覆盖三个公开数据集所有场景：

| 数据集 | 场景 | 分辨率 | 方法 |
|---|---|---|---|
| MipNeRF360 | `bicycle/flowers/garden/stump/treehill/room/counter/kitchen/bonsai` | 原图，`-r -1`，1.6K 自动缩放 | `fastgs_big + densification_interval=100 + scene overrides` |
| DB | `drjohnson/playroom` | 原图，`-r -1`，1.6K 自动缩放 | `fastgs_big + densification_interval=100 + scene overrides` |
| Tandt | `train/truck` | 原图，`-r -1`，1.6K 自动缩放 | `fastgs_big + densification_interval=100 + scene overrides` |

建议复用 `scripts/run_0001_fastgs_big_eval.py`，将输出写到 `output/0002/phase0_5090_fastgs_big_baseline/`。双卡只用于场景级并行，不做单实验多卡训练；每张卡跑一组场景，避免输出目录重叠。

验收标准：

- 每个场景完成 train/render/metrics，并写出 `summary.csv`、`summary.json`、`averages.json`。
- 与 4090D 既有 FastGS big baseline 比较时，PSNR/SSIM/LPIPS 只允许出现小幅随机波动；若某场景出现明显偏移，先排查环境、随机种子、CUDA extension、scene override 和数据路径。
- Gaussian 数量、训练时间和渲染速度单独记录。训练时间可因 5090 提升而变化，不作为质量一致性的失败条件。
- 只有 phase 0 通过后，才开始 Depth Anything backend/cache 和 620-step smoke。

## Depth Anything 数据集计划

当前算力允许双卡并行，因此 0002 不再先走 `-r 8` 低分辨率探测。Phase 0 baseline 复核通过后，Depth Anything 的 smoke、pilot 和正式验证都直接使用原图输入与 FastGS 原始 1.6K 自动缩放口径，即 `-i images -r -1`。先在少数代表场景上确认 dense depth prior 有效；若 pilot 场景整体正向，再扩展到三个公开数据集全场景训练验证。

| 阶段 | 数据集 | 场景 | 分辨率 | 目的 |
|---|---|---|---|---|
| phase 0 | MipNeRF360/DB/Tandt | 全场景 | 原图，`-r -1`，1.6K 自动缩放 | 5090 FastGS big baseline 复核 |
| smoke | MipNeRF360 | bicycle | 原图，`-r -1`，620 step | 验证 cache/build/train/render/metrics 链路 |
| pilot | MipNeRF360 | bicycle/stump/bonsai | 原图，`-r -1`，30k | 已完成；stump 有效正向，bonsai 边界正向 |
| pilot | DB/Tandt | playroom/truck | 原图，`-r -1`，30k | 已完成；playroom 薄正向，truck 明确负向 |
| full | MipNeRF360/DB/Tandt | 全场景 | 原图，`-r -1`，30k | 仅在 pilot 多场景有效后推进 |

## 指标

主指标仍使用 test split 的全图指标：

- PSNR
- SSIM
- LPIPS
- Gaussian 数量
- 训练时间
- 渲染 FPS 或 metrics 运行时的有效渲染速度

同时记录 QCGI：

```text
quality_gain = ΔPSNR + 20 * ΔSSIM - 5 * ΔLPIPS
gs_penalty = 0.01 * min(max(ΔGS, 0), 100000) / 10000
           + 0.04 * max(ΔGS - 100000, 0) / 10000
QCGI = quality_gain - gs_penalty
```

0002 还应补充局部几何区域指标，但不阻塞第一阶段：

- depth-edge top-k 区域的 RGB/LPIPS 局部质量。
- DINO descriptor top-k 区域与 depth-edge top-k 区域的重叠率。
- depth prior 命中 Gaussian 的新增/保留分布。

## 运行命令草案

Depth Anything depth-edge prior 已落地。实际训练使用原图 1.6K 口径，不再加 `-r 8`。

```bash
source .venv/bin/activate

HF_HUB_DISABLE_XET=1 python -m vfm_gs.cli.build_vfm_cache \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -o output/0002/vfm_cache/bicycle_depth_anything_v2s_edge \
  --backend depth_anything_v2 \
  --max_width 1600 \
  --device cuda \
  --depth_anything_feature depth_edge \
  --storage npz_uint8

python -m vfm_gs.cli.validate_vfm_cache \
  -c output/0002/vfm_cache/bicycle_depth_anything_v2s_edge \
  -s datasets/mipnerf360/bicycle \
  -i images \
  --backend depth_anything_v2

python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0002_depth_anything_depth_edge_prior_densify_only_topk025_weighted_i050.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0002/depth_anything_edge_prior_bicycle_620_r_auto \
  --eval \
  --iterations 620 \
  --densify_from_iter 500 \
  --densify_until_iter 620 \
  --densification_interval 100 \
  --test_iterations 620 \
  --save_iterations 620 \
  --checkpoint_iterations 620 \
  --vfm_cache_dir output/0002/vfm_cache/bicycle_depth_anything_v2s_edge \
  -r -1

python -m vfm_gs.cli.render \
  -m output/0002/depth_anything_edge_prior_bicycle_620_r_auto \
  --iteration -1 \
  --skip_train \
  --quiet

python -m vfm_gs.cli.metrics \
  -m output/0002/depth_anything_edge_prior_bicycle_620_r_auto
```

## 成功标准

Phase 0 成功标准：

- 三个公开数据集共 13 个场景的 FastGS big baseline 都完成。
- 5090 结果与 4090D 既有结果在 PSNR/SSIM/LPIPS 上无系统性漂移。
- 若发现偏移，先记录到 `results.md` 和 `review.md`，暂停 Depth Anything。

Depth Anything 第一阶段成功标准：

- 620-step smoke 完成 train/render/metrics，且 cache preflight 能提前发现缺失或后端不匹配。
- high-res pilot 场景相对 phase 0 FastGS big baseline 至少两项质量正向，且 LPIPS 不变差。
- 若 Gaussian 增长超过 0.10M，必须有明显 QCGI 正收益，否则只记录为几何信号存在但容量低效。
- 与 0001 DINO descriptor weighted i0.50/i0.70 对照时，Depth Anything 至少应在某类几何/边界场景上形成互补，而不是只复现更弱的边缘 proxy。

## 失败记录

2026-05-11 Phase 0 首次 high-res FastGS big baseline 已启动，但 MipNeRF360 `bicycle` 与 `room` 分别在两张 RTX 5090 上提前报 CUDA illegal memory access。blocking 复跑仍失败；修补 rasterizer debug wrapper 后，`bicycle` debug 复现在 `rasterizer_impl.cu:422` 捕获 `operation not supported on global/shared address space`，位置紧跟 `identifyTileRanges` kernel。该失败发生在 Depth Anything 接入前，当前作为 5090 / CUDA extension 环境阻塞记录。

第一轮 rasterizer 修补已通过 MipNeRF360、DB、Tandt 全 13 场景 high-res FastGS big baseline：MipNeRF360 为 27.9590 / 0.8203 / 0.2157、1,161,786 个 Gaussians，DB 为 30.2331 / 0.9111 / 0.2397、646,600 个 Gaussians，Tandt 为 24.4955 / 0.8579 / 0.1736、540,119 个 Gaussians；三组都与 0001/4090D 同口径 baseline 基本贴合。Phase 0 已通过，可以进入 Depth Anything。

重点观察风险：

- Depth Anything 相对深度的 scale/shift 不稳定。
- 渲染 albedo 图送入 depth model 后，预测深度与 GT RGB depth prior 域差异过大。
- depth-edge prior 过度集中在少数遮挡边界，导致新增 Gaussian 分布偏置。
- dense depth cache 体积、依赖版本或推理耗时不可接受。

## 决策

2026-05-11：`depth_anything_depth_edge_prior` 作为 0002 第一条落地路径，high-res `bicycle` 620-step smoke 已通过 cache/build/preflight/train/render/metrics。620-step matched baseline 略优，因此该结果只证明链路健康，不作为质量正例。

2026-05-11：`depth_anything_depth_edge_prior` high-res `bicycle` 30k pilot 完成。相对 matched `fastgs_baseline + densify100` baseline，结果为 -0.0023 PSNR、+0.0017 SSIM、LPIPS -0.0035，Gaussian 数 +39,399；这是弱混合信号，不足以直接扩到全数据集。

2026-05-11：`depth_anything_depth_prior` high-res `bicycle` 30k pilot 完成。相对 matched `fastgs_baseline + densify100` baseline，结果为 +0.0628 PSNR、+0.0063 SSIM、LPIPS -0.0090，Gaussian 数 -17,959；这是 0002 首个明确正向 direct depth signal。

2026-05-14：`depth_anything_depth_prior` 在 `fastgs_big + scene overrides` 下完成 `stump/bonsai` 复验。`stump` 相对 phase 0 为 +0.0155 PSNR、+0.0027 SSIM、LPIPS -0.0072，GS +21,369；`bonsai` 为 -0.0113 PSNR、+0.0008 SSIM、LPIPS -0.0034，GS +87,481。该结果保留 direct depth 主线，但不支持直接全数据集扩展。所有后续 fastgs_big prior 必须显式带上 phase 0 scene overrides。

2026-05-14：`playroom/truck` 跨数据集 pilot 完成。`playroom` 为 +0.1061 PSNR、-0.0004 SSIM、LPIPS -0.0026，GS +17,201，QCGI +0.0936；`truck` 为 -0.1502 PSNR、-0.0026 SSIM、LPIPS +0.0021，GS -101,994，QCGI -0.2124。overlap 显示 `playroom` 改善不集中在 depth prior top-k，`truck` 的 depth prior top-k 与 RGB 瓶颈严重错位。

2026-05-17：`RGB broad candidate -> depth prior rerank -> final-topm` l0.25 pilot 完成。`truck` 从 direct depth 的明确负例翻成三项质量正向，说明先 RGB 后 depth 的两阶段策略是可行方向；但 `stump/playroom` 的 PSNR 退化、Gaussian 数明显上涨，三场景 QCGI 都为负。该结果支持继续做更保守的 RGB-gated 扫描，但不支持把 l0.25 直接升级为默认方案。

2026-05-17：`RGB broad candidate -> depth prior rerank -> final-topm` l0.10 pilot 完成。`stump` 和 `truck` 都回到三项质量正向，但 `playroom` 仍然 PSNR 负向，且三场景 QCGI 仍为负；这说明两阶段思路成立，但实现还需要继续收紧，不能直接扩全数据集。

2026-05-17：`RGB broad candidate -> depth prior rerank -> final-topm` l0.05 pilot 完成。`playroom` 从 l0.10 的 PSNR 负向翻回正向，但 `truck` 再次 PSNR 负向，且三场景 QCGI 仍为负；单纯继续降低 rerank strength 不能解决 broad candidate 与 depth prior 错位问题。

2026-05-18：`RGB broad candidate -> depth prior rerank -> final-topm` l0.10 broad035 pilot 完成。相对 top50 broad，`stump` 继续三项质量正向且 Gaussian 增量从约 +323k 降到 +256k，`playroom/truck` 的 Gaussian 增量也略降；但 `playroom` PSNR/SSIM 轻微负向，`truck` PSNR 负向，三场景 QCGI 仍为负。缩窄 candidate 入口只能缓和容量问题，不能把 depth prior 变成稳定的 RGB residual surrogate。

2026-05-18：`RGB broad candidate -> depth prior rerank -> final-topm` l0.10 broad035 start9000 pilot 完成。后期介入让 `playroom` 从 broad035 的轻微负向变成 PSNR/LPIPS 正向，但 `stump` Gaussian 增量扩大到 +355,826，`truck` 仍 PSNR 负向，三场景 QCGI 继续为负。单独延后介入不是稳定解。

2026-05-18：按用户建议将 `l0.10 broad035 start9000` 扩展到 MipNeRF360 全 9 场景。全场景均值为 27.9698 / 0.8238 / 0.2070、1,399,502 GS，相对 Phase 0 FastGS big 为 +0.0107 PSNR、+0.0036 SSIM、LPIPS -0.0087，但平均多 237,716 GS，QCGI 均值 -0.5602，且 9/9 场景 QCGI 均为负。结论是该策略有数据集均值质量信号，但容量效率不合格。

初始决策：0002 只推进 dense depth prior，不再扩展 COLMAP sparse depth-edge proxy。

## 下一步

暂停同配置 `depth_anything_depth_prior`、`RGB broad top50 -> depth rerank -> final-topm`、`broad035` 和 `start9000` 的继续扩展。若继续 0002，Depth Anything 应继续往后期辅助裁剪信号收缩，而不是再做 densify 主信号。

已完成的最小轮次是 `prune-protect-only`：FastGS/RGB 继续决定 densification，Depth Anything 只在 densification 结束后保护 RGB 高 pruning-score 候选。`stump/playroom/truck` 的三场景平均几乎持平，说明这条路比 RGB rerank 更像“轻量后验修正”，但还不是稳定默认解。`prune-protect weight015` 已验证单独降权重不够；`topk005` 已验证继续收窄 RGB proposal 也不够，三场景平均为 -0.0185 PSNR、-0.0003 SSIM、LPIPS +0.0010、GS -6,755，QCGI -0.0295。下一步若继续 0002，应把 `topk010` 视为当前最接近中性的 prune-protect 对照，优先考虑更多场景复验或场景自适应门控，而不是继续手工扫相邻 top-k/weight。长任务继续使用 tmux/detached 方式运行；每轮完成后更新文档、commit 并 push。

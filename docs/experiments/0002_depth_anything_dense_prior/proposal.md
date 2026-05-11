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
  - `depth_anything_depth_edge_prior`：已实现并通过 high-res `bicycle` 620-step smoke。
  - `depth_anything_depth_prior`：已实现配置入口，尚未跑正式 smoke。
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

当前优先验证 depth-edge prior，因为它能离线缓存并直接复用现有 prior-style scorer，变量少于在线 depth residual。

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
| pilot | MipNeRF360 | bicycle/stump/bonsai | 原图，`-r -1`，30k | 覆盖常规、几何收益和容量压力场景 |
| pilot | DB/Tandt | playroom/truck | 原图，`-r -1`，30k | 检查 dense depth 是否在室内和 Tandt 负例上互补 |
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

初始决策：0002 只推进 dense depth prior，不再扩展 COLMAP sparse depth-edge proxy。

## 下一步

推进 high-res `bicycle` 30k pilot；若 30k 不正向，再比较 `depth_anything_depth_prior` 与 depth-edge prior，或转向在线 depth residual。长任务继续使用 detached 方式运行；每轮完成后更新文档、commit 并 push。

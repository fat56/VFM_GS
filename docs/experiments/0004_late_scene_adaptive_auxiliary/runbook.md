# 0004 Late Scene-Adaptive Auxiliary 运行手册

## 环境

```bash
source .venv/bin/activate
```

0004 先复用现有 FastGS big 训练链路和 0002 / 0003 已验证的辅助后端。长跑建议继续用 `tmux` 或 `screen`。

## Phase 0：确定晚期辅助是否值得跑

先不要把 prior 放到 densification 早期。第一轮只验证一个小型异质 pilot：

- `bicycle`
- `stump`
- `room`

这三个场景分别代表中性、偏正、偏负的情况，足够看出辅助策略是不是在做真正的 scene-adaptive，而不是把所有场景都往同一个方向硬拽。

建议先用 `Depth Anything` 的 prune-side 方案做主 pilot，因为 0002 里它是目前最接近“减伤器”的候选；`0003` 的 DINO prune-protect-only 作为负例对照。

## Phase 1：晚期 prune-protect pilot

目标不是冲最好 PSNR，而是验证下面三件事：

1. 辅助只在晚期介入。
2. 辅助只在 RGB 候选内部 rerank / protect。
3. 不同场景会落到不同强度或不同开关状态。

运行时应记录：

- 每 2k iteration 的 `render`
- 每 2k iteration 的 `metrics`
- 每个 checkpoint 的 PSNR / SSIM / LPIPS
- 每个 checkpoint 的 Gaussian 数量
- 该 checkpoint 是否超出预算上限

建议把这些检查结果写成 `output/0004/.../check.md`，方便后面按场景比较曲线。

一个可直接起跑的小 pilot 示例是：

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 uv run --active python scripts/run_0001_fastgs_big_eval.py \
  --dataset-name mipnerf360 \
  --dataset-root datasets/mipnerf360 \
  --output-root output/0004/late_scene_adaptive_auxiliary/mipnerf360_gpu0 \
  --scenes bicycle stump room \
  --variant fastgs_big \
  --config configs/experiments/0004_late_scene_adaptive_auxiliary.yaml \
  --vfm-cache-template output/0002/vfm_cache/{scene}_depth_anything_v2s_depth \
  --resolution -1
```

如果后面切到 DINO prune-protect 对照，就复用 `configs/experiments/0003_dino_descriptor_prune_protect_only.yaml` 和 0003 的现成 cache，仍然保持晚期介入和预算约束。

## Phase 2：扩展到全量场景

如果小 pilot 没有出现明显退化，再扩到：

- MipNeRF360 全 9 场景
- DB 全场景
- Tandt 全场景

这一步只在小 pilot 说明“晚期辅助有机会”时做。

## 结果归档

每轮实验结束后更新：

- `docs/experiments/0004_late_scene_adaptive_auxiliary/results.md`
- `docs/experiments/0004_late_scene_adaptive_auxiliary/review.md`

如果实验方向再次收束，继续把结论同步回 `docs/roadmap.md`。

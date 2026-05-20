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

第一轮建议保留 `bicycle / stump / room`，因为 baseline curve 分别代表三种后期行为：后期小幅稳定收益、PSNR 早停但 LPIPS 后续改善、室内后期仍明显涨。一个可直接起跑的小 pilot 示例是：

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

若要和 baseline 曲线严格对齐，使用 checkpoint-curve runner：

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 uv run --active python scripts/run_fastgs_big_checkpoint_curve.py \
  --dataset-name mipnerf360 \
  --dataset-root datasets/mipnerf360 \
  --output-root output/0004/late_scene_adaptive_auxiliary/depth_prune_auto_topk_pilot_gpu1 \
  --scenes bicycle stump room \
  --iterations 30000 \
  --checkpoint-interval 2000 \
  --resolution -1 \
  --variant fastgs_big \
  --densification-interval 100 \
  --method-name depth_prune_auto_topk_late_aux \
  --run-name depth_prune_auto_topk_late_aux_30k_curve_r_auto \
  --config configs/experiments/0004_late_scene_adaptive_auxiliary.yaml \
  --vfm-cache-template output/0002/vfm_cache/{scene}_depth_anything_v2s_depth
```

第一轮 `start15001` 已完成，结果见：

- `output/0004/late_scene_adaptive_auxiliary/depth_prune_auto_topk_pilot_gpu1/check.md`
- `output/0004/late_scene_adaptive_auxiliary/depth_prune_auto_topk_indoor_gpu0/check.md`

## Phase 1B：更晚介入 start24000

第一轮显示 `room` 对 18k/21k protect 很敏感。下一轮只改变介入时机：`configs/experiments/0004_late_scene_adaptive_auxiliary_start24000.yaml` 会跳过 18k/21k，只在 24k/27k 的 pruning 中启用 protect。

GPU1 跑异质 pilot：

```bash
tmux new-session -d -s 0004_depth_start24_pilot_g1 "cd /home/m/project/ltm/VFM_GS && source .venv/bin/activate && mkdir -p output/0004/debug_logs && CUDA_VISIBLE_DEVICES=1 python scripts/run_fastgs_big_checkpoint_curve.py --dataset-name mipnerf360 --dataset-root datasets/mipnerf360 --output-root output/0004/late_scene_adaptive_auxiliary/depth_prune_auto_topk_start24000_pilot_gpu1 --scenes bicycle stump room --iterations 30000 --checkpoint-interval 2000 --resolution -1 --variant fastgs_big --densification-interval 100 --method-name depth_prune_auto_topk_start24000 --run-name depth_prune_auto_topk_start24000_30k_curve_r_auto --config configs/experiments/0004_late_scene_adaptive_auxiliary_start24000.yaml --vfm-cache-template output/0002/vfm_cache/{scene}_depth_anything_v2s_depth > output/0004/debug_logs/depth_prune_auto_topk_start24000_pilot_g1.log 2>&1"
```

GPU0 跑室内补充组：

```bash
tmux new-session -d -s 0004_depth_start24_indoor_g0 "cd /home/m/project/ltm/VFM_GS && source .venv/bin/activate && mkdir -p output/0004/debug_logs && CUDA_VISIBLE_DEVICES=0 python scripts/run_fastgs_big_checkpoint_curve.py --dataset-name mipnerf360 --dataset-root datasets/mipnerf360 --output-root output/0004/late_scene_adaptive_auxiliary/depth_prune_auto_topk_start24000_indoor_gpu0 --scenes bonsai counter kitchen --iterations 30000 --checkpoint-interval 2000 --resolution -1 --variant fastgs_big --densification-interval 100 --method-name depth_prune_auto_topk_start24000 --run-name depth_prune_auto_topk_start24000_30k_curve_r_auto --config configs/experiments/0004_late_scene_adaptive_auxiliary_start24000.yaml --vfm-cache-template output/0002/vfm_cache/{scene}_depth_anything_v2s_depth > output/0004/debug_logs/depth_prune_auto_topk_start24000_indoor_g0.log 2>&1"
```

监控：

```bash
tmux ls
tail -n 60 output/0004/debug_logs/depth_prune_auto_topk_start24000_pilot_g1.log
tail -n 60 output/0004/debug_logs/depth_prune_auto_topk_start24000_indoor_g0.log
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

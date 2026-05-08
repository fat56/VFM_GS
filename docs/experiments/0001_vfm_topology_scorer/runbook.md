# 0001 VFM 拓扑打分器运行手册

## 完整基线

```bash
uv run --active python -m vfm_gs.cli.train --variant fastgs_baseline -s <dataset>/<scene> -m output/0001_baseline/<scene> --eval
uv run --active python -m vfm_gs.cli.render -m output/0001_baseline/<scene> --skip_train
uv run --active python -m vfm_gs.cli.metrics -m output/0001_baseline/<scene>
```

## MipNeRF360 全场景 v1 评估

统一批次脚本会对 9 个 MipNeRF360 场景依次运行 baseline 与 `cached_edge_l1 + staged target ~= 1.42x baseline count`，并为每个 run 保存 train/render/metrics 日志。脚本可重复执行；已完成的训练、渲染和指标会跳过。

```bash
uv run --active python scripts/run_mipnerf360_v1_eval.py
```

主要产物：

- `output/0001/full_mipnerf360_v1/summary.csv`
- `output/0001/full_mipnerf360_v1/summary.json`
- `output/0001/full_mipnerf360_v1/averages.json`
- `output/0001/full_mipnerf360_v1/<scene>/logs/<method>/train.log`
- `output/0001/full_mipnerf360_v1/<scene>/logs/<method>/render.log`
- `output/0001/full_mipnerf360_v1/<scene>/logs/<method>/metrics.log`

## Tandt/DB 全场景 v1 评估

同一个批次脚本也支持指定数据集根目录、场景列表和 cache 图像目录。`datasets/tandt_db/db` 与 `datasets/tandt_db/tandt` 没有 `images_8`，因此这里显式使用 `--cache-images images`。训练仍使用 `-i images`、`--eval`、`-r 8`、30,000 iterations，并对每个场景运行 baseline 与 `cached_edge_l1 + staged target ~= 1.42x baseline count`。

```bash
uv run --active python scripts/run_mipnerf360_v1_eval.py \
  --dataset-name db \
  --dataset-root datasets/tandt_db/db \
  --output-root output/0001/full_tandt_db_v1/db \
  --scenes drjohnson playroom \
  --cache-images images

uv run --active python scripts/run_mipnerf360_v1_eval.py \
  --dataset-name tandt \
  --dataset-root datasets/tandt_db/tandt \
  --output-root output/0001/full_tandt_db_v1/tandt \
  --scenes train truck \
  --cache-images images
```

主要产物：

- `output/0001/full_tandt_db_v1/db/summary.csv`
- `output/0001/full_tandt_db_v1/db/summary.json`
- `output/0001/full_tandt_db_v1/db/averages.json`
- `output/0001/full_tandt_db_v1/tandt/summary.csv`
- `output/0001/full_tandt_db_v1/tandt/summary.json`
- `output/0001/full_tandt_db_v1/tandt/averages.json`
- `output/0001/full_tandt_db_v1/<dataset>/<scene>/logs/<method>/train.log`
- `output/0001/full_tandt_db_v1/<dataset>/<scene>/logs/<method>/render.log`
- `output/0001/full_tandt_db_v1/<dataset>/<scene>/logs/<method>/metrics.log`

## Tandt 容量保护诊断

`prune_min_gaussian_count` 默认关闭。它用于诊断 `cached_edge_l1` 在 Tandt 上是否因为训练期或最终裁剪过强而低于 baseline 容量。下面两条命令分别把 `train` 和 `truck` 的最小 Gaussian 数量设为对应 baseline 的最终数量，其他条件仍保持 `-r 8`、30,000 iterations、`--eval` 和 v1 staged target。

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_cached_edge_compact.yaml \
  -s datasets/tandt_db/tandt/train \
  -i images \
  -m output/0001/full_tandt_db_v1/tandt/train/vfm_cached_edge_prunemin58788_staged142_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/full_tandt_db_v1/tandt/train/cache/edge_u8 \
  --prune_min_gaussian_count 58788 \
  --target_gaussian_count 83479 \
  --target_gaussian_staged \
  --target_gaussian_stage_margin 1.10 \
  --target_gaussian_stage_interval 500 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/full_tandt_db_v1/tandt/train/vfm_cached_edge_prunemin58788_staged142_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/full_tandt_db_v1/tandt/train/vfm_cached_edge_prunemin58788_staged142_30k_r8
```

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_cached_edge_compact.yaml \
  -s datasets/tandt_db/tandt/truck \
  -i images \
  -m output/0001/full_tandt_db_v1/tandt/truck/vfm_cached_edge_prunemin41952_staged142_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/full_tandt_db_v1/tandt/truck/cache/edge_u8 \
  --prune_min_gaussian_count 41952 \
  --target_gaussian_count 59572 \
  --target_gaussian_staged \
  --target_gaussian_stage_margin 1.10 \
  --target_gaussian_stage_interval 500 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/full_tandt_db_v1/tandt/truck/vfm_cached_edge_prunemin41952_staged142_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/full_tandt_db_v1/tandt/truck/vfm_cached_edge_prunemin41952_staged142_30k_r8
```

配套诊断命令：

```bash
# 关闭 VFM pruning fusion，检查负例是否只来自融合项。
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_cached_edge_compact.yaml \
  -s datasets/tandt_db/tandt/train \
  -i images \
  -m output/0001/full_tandt_db_v1/tandt/train/vfm_cached_edge_prune0_staged142_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/full_tandt_db_v1/tandt/train/cache/edge_u8 \
  --vfm_weight 0.0 \
  --target_gaussian_count 83479 \
  --target_gaussian_staged \
  --target_gaussian_stage_margin 1.10 \
  --target_gaussian_stage_interval 500 \
  -r 8

# 只改变 FastGS densification cadence，作为 Tandt 的 no-effect 参照。
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  -s datasets/tandt_db/tandt/train \
  -i images \
  -m output/0001/full_tandt_db_v1/tandt/train/fastgs_densify100_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --densification_interval 100 \
  -r 8
```

## 模拟 VFM 拓扑 v1

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_scorer.yaml \
  -s <dataset>/<scene> \
  -m output/0001_vfm/<scene> \
  --eval
```

当前 v1 使用 `vfm_topology_scorer` + `mock_l1` 后端。它不是真实 VFM 质量实验，而是验证 SH0 render、pixel error map、metric map、Gaussian 计数和 FastGS 分数融合链路。

## 离线边缘缓存代理 v1

`cached_edge_l1` 后端先用 GT 图像的归一化 luminance edge map 作为轻量离线缓存代理。它不是最终 VFM 后端，但能验证 `image_name -> cache entry -> pixel_error_map` 的真实缓存读取流程。
训练入口会在 Scene 加载前自动执行 cache preflight；这里仍显式运行 `validate_vfm_cache`，用于把缓存检查作为实验流程的一部分记录下来。

```bash
uv run --active python -m vfm_gs.cli.build_vfm_cache \
  -s datasets/mipnerf360/bicycle \
  -i images_8 \
  -o output/0001/vfm_cache/bicycle_edge_u8 \
  --max_width 640 \
  --storage npz_uint8

uv run --active python -m vfm_gs.cli.validate_vfm_cache \
  -c output/0001/vfm_cache/bicycle_edge_u8 \
  -s datasets/mipnerf360/bicycle \
  -i images_8 \
  --backend cached_edge_l1

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_cached_edge_compact.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_cached_edge/bicycle \
  --eval
```

## 可选 DINOv2 Cache 快速验证

DINOv2 cache build 是真实 VFM features 的离线产物路径。快速验证会在运行训练 scorer 前验证 `dinov2_patchtokens` cache 生成和 manifest 兼容性。

当 `torch.hub` 远程访问被限流时，将官方仓库 clone 到 ignored output state，并显式传入路径：

```bash
git clone https://github.com/facebookresearch/dinov2.git output/0001/external/dinov2

uv run --active python -m vfm_gs.cli.build_vfm_cache \
  -s datasets/mipnerf360/bicycle \
  -i images_8 \
  -o output/0001/vfm_cache/bicycle_dinov2_vits14_smoke \
  --backend dinov2_vits14 \
  --dinov2_repo output/0001/external/dinov2 \
  --max_width 224 \
  --storage npy_float16 \
  --limit 4

uv run --active python -m vfm_gs.cli.validate_vfm_cache \
  -c output/0001/vfm_cache/bicycle_dinov2_vits14_smoke \
  --backend dinov2_vits14
```

对 DINOv2 来说，省略 `--storage` 时默认使用 `npy_float16`。DINO patch-token caches 会有意拒绝 `npz_uint8`。

## DINOv2 Token-Edge 打分器 v1

`dinov2_token_edge_l1` 是第一版训练期消费 DINOv2 cache 的 scorer backend。它不在训练循环里跑 DINOv2，而是把离线 `dinov2_patchtokens` 转成 token-edge topology map，再和 SH0 渲染图的 pooled edge map 比较。

```bash
uv run --active python -m vfm_gs.cli.build_vfm_cache \
  -s datasets/mipnerf360/bicycle \
  -i images_8 \
  -o output/0001/vfm_cache/bicycle_dinov2_vits14 \
  --backend dinov2_vits14 \
  --dinov2_repo output/0001/external/dinov2 \
  --max_width 224

uv run --active python -m vfm_gs.cli.validate_vfm_cache \
  -c output/0001/vfm_cache/bicycle_dinov2_vits14 \
  --backend dinov2_vits14

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_bicycle_smoke \
  --eval \
  --iterations 220 \
  --densify_from_iter 50 \
  --densify_until_iter 220 \
  --densification_interval 50 \
  --test_iterations 220 \
  --save_iterations 220 \
  --checkpoint_iterations 220 \
  -r 8

uv run --active python -m vfm_gs.cli.render -m output/0001/vfm_dinov2_token_edge_bicycle_smoke --skip_train
uv run --active python -m vfm_gs.cli.metrics -m output/0001/vfm_dinov2_token_edge_bicycle_smoke
```

### Token-Edge Top-k 探测

`0001_vfm_topology_dinov2_token_edge_topk.yaml` 保持 `dinov2_token_edge_l1` 后端不变，只把 metric map 改为 top-k 15% 高误差区域。先用 620-step 验证链路，再进入 30k 完整对照。

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk015_bicycle_620_r8 \
  --eval \
  --iterations 620 \
  --densify_from_iter 500 \
  --densify_until_iter 620 \
  --densification_interval 100 \
  --test_iterations 620 \
  --save_iterations 620 \
  --checkpoint_iterations 620 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk015_bicycle_620_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk015_bicycle_620_r8
```

完整 30k 对照：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk015_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk015_bicycle_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk015_bicycle_30k_r8
```

更宽 top-k 质量上界探测：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_metric_topk 0.25 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_bicycle_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_bicycle_30k_r8
```

更宽 top-k 的 `rgb_only` 预算贴合探测：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_rgb_only_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_metric_topk 0.25 \
  --vfm_importance_mode rgb_only \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_rgb_only_bicycle_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_rgb_only_bicycle_30k_r8
```

更宽 top-k 的 partial importance 预算探测：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_i025_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_metric_topk 0.25 \
  --vfm_importance_weight 0.25 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_i025_bicycle_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_i025_bicycle_30k_r8
```

partial importance 曲线的更高权重点：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_i050_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_metric_topk 0.25 \
  --vfm_importance_weight 0.50 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_i050_bicycle_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_i050_bicycle_30k_r8
```

RGB/VFM 加权融合的预算贴合探测：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/bicycle_dinov2_vits14 \
  --vfm_metric_topk 0.25 \
  --vfm_importance_weight 0.50 \
  --vfm_importance_mode weighted \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_bicycle_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_bicycle_30k_r8
```

partial importance 第二场景复验：

```bash
uv run --active python -m vfm_gs.cli.build_vfm_cache \
  -s datasets/mipnerf360/garden \
  -i images_8 \
  -o output/0001/vfm_cache/garden_dinov2_vits14 \
  --backend dinov2_vits14 \
  --dinov2_repo output/0001/external/dinov2 \
  --max_width 224

uv run --active python -m vfm_gs.cli.validate_vfm_cache \
  -c output/0001/vfm_cache/garden_dinov2_vits14 \
  -s datasets/mipnerf360/garden \
  -i images_8 \
  --backend dinov2_vits14

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/garden \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_i050_garden_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/garden_dinov2_vits14 \
  --vfm_metric_topk 0.25 \
  --vfm_importance_weight 0.50 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_i050_garden_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_i050_garden_30k_r8
```

RGB/VFM 加权融合在中等点数增长场景的复验：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/garden \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_garden_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/garden_dinov2_vits14 \
  --vfm_metric_topk 0.25 \
  --vfm_importance_weight 0.50 \
  --vfm_importance_mode weighted \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_garden_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_garden_30k_r8
```

partial importance 室内/台面场景复验：

```bash
uv run --active python -m vfm_gs.cli.build_vfm_cache \
  -s datasets/mipnerf360/counter \
  -i images_8 \
  -o output/0001/vfm_cache/counter_dinov2_vits14 \
  --backend dinov2_vits14 \
  --dinov2_repo output/0001/external/dinov2 \
  --max_width 224

uv run --active python -m vfm_gs.cli.validate_vfm_cache \
  -c output/0001/vfm_cache/counter_dinov2_vits14 \
  -s datasets/mipnerf360/counter \
  -i images_8 \
  --backend dinov2_vits14

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/counter \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_i050_counter_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/counter_dinov2_vits14 \
  --vfm_metric_topk 0.25 \
  --vfm_importance_weight 0.50 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_i050_counter_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_i050_counter_30k_r8
```

RGB/VFM 加权融合在室内/台面场景的低增点复验：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/counter \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_counter_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/counter_dinov2_vits14 \
  --vfm_metric_topk 0.25 \
  --vfm_importance_weight 0.50 \
  --vfm_importance_mode weighted \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_counter_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_counter_30k_r8
```

partial importance 在 cached-edge 负例上的压力测试：

```bash
uv run --active python -m vfm_gs.cli.build_vfm_cache \
  -s datasets/mipnerf360/treehill \
  -i images_8 \
  -o output/0001/vfm_cache/treehill_dinov2_vits14 \
  --backend dinov2_vits14 \
  --dinov2_repo output/0001/external/dinov2 \
  --max_width 224

uv run --active python -m vfm_gs.cli.validate_vfm_cache \
  -c output/0001/vfm_cache/treehill_dinov2_vits14 \
  -s datasets/mipnerf360/treehill \
  -i images_8 \
  --backend dinov2_vits14

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/treehill \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_i050_treehill_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/treehill_dinov2_vits14 \
  --vfm_metric_topk 0.25 \
  --vfm_importance_weight 0.50 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_i050_treehill_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_i050_treehill_30k_r8
```

RGB/VFM 加权融合在 cached-edge 负例上的压力复验：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/treehill \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_treehill_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/treehill_dinov2_vits14 \
  --vfm_metric_topk 0.25 \
  --vfm_importance_weight 0.50 \
  --vfm_importance_mode weighted \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_treehill_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_treehill_30k_r8
```

partial importance 室内小场景复验：

```bash
uv run --active python -m vfm_gs.cli.build_vfm_cache \
  -s datasets/mipnerf360/bonsai \
  -i images_8 \
  -o output/0001/vfm_cache/bonsai_dinov2_vits14 \
  --backend dinov2_vits14 \
  --dinov2_repo output/0001/external/dinov2 \
  --max_width 224

uv run --active python -m vfm_gs.cli.validate_vfm_cache \
  -c output/0001/vfm_cache/bonsai_dinov2_vits14 \
  -s datasets/mipnerf360/bonsai \
  -i images_8 \
  --backend dinov2_vits14

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/bonsai \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_i050_bonsai_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/bonsai_dinov2_vits14 \
  --vfm_metric_topk 0.25 \
  --vfm_importance_weight 0.50 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_i050_bonsai_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_i050_bonsai_30k_r8
```

RGB/VFM 加权融合在中低增点室内/植物场景的复验：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/bonsai \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_bonsai_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/bonsai_dinov2_vits14 \
  --vfm_metric_topk 0.25 \
  --vfm_importance_weight 0.50 \
  --vfm_importance_mode weighted \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_bonsai_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_bonsai_30k_r8
```

partial importance 植被/花丛场景复验：

```bash
uv run --active python -m vfm_gs.cli.build_vfm_cache \
  -s datasets/mipnerf360/flowers \
  -i images_8 \
  -o output/0001/vfm_cache/flowers_dinov2_vits14 \
  --backend dinov2_vits14 \
  --dinov2_repo output/0001/external/dinov2 \
  --max_width 224

uv run --active python -m vfm_gs.cli.validate_vfm_cache \
  -c output/0001/vfm_cache/flowers_dinov2_vits14 \
  -s datasets/mipnerf360/flowers \
  -i images_8 \
  --backend dinov2_vits14

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/flowers \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_i050_flowers_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/flowers_dinov2_vits14 \
  --vfm_metric_topk 0.25 \
  --vfm_importance_weight 0.50 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_i050_flowers_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_i050_flowers_30k_r8
```

RGB/VFM 加权融合在植被/花丛场景的复验：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/flowers \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_flowers_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/flowers_dinov2_vits14 \
  --vfm_metric_topk 0.25 \
  --vfm_importance_weight 0.50 \
  --vfm_importance_mode weighted \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_flowers_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_flowers_30k_r8
```

partial importance 室内高基线场景复验：

```bash
uv run --active python -m vfm_gs.cli.build_vfm_cache \
  -s datasets/mipnerf360/kitchen \
  -i images_8 \
  -o output/0001/vfm_cache/kitchen_dinov2_vits14 \
  --backend dinov2_vits14 \
  --dinov2_repo output/0001/external/dinov2 \
  --max_width 224

uv run --active python -m vfm_gs.cli.validate_vfm_cache \
  -c output/0001/vfm_cache/kitchen_dinov2_vits14 \
  -s datasets/mipnerf360/kitchen \
  -i images_8 \
  --backend dinov2_vits14

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/kitchen \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_i050_kitchen_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/kitchen_dinov2_vits14 \
  --vfm_metric_topk 0.25 \
  --vfm_importance_weight 0.50 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_i050_kitchen_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_i050_kitchen_30k_r8
```

weighted importance 室内高基线场景复验：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/kitchen \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_kitchen_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/kitchen_dinov2_vits14 \
  --vfm_metric_topk 0.25 \
  --vfm_importance_weight 0.50 \
  --vfm_importance_mode weighted \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_kitchen_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_kitchen_30k_r8
```

partial importance 室内房间场景复验：

```bash
uv run --active python -m vfm_gs.cli.build_vfm_cache \
  -s datasets/mipnerf360/room \
  -i images_8 \
  -o output/0001/vfm_cache/room_dinov2_vits14 \
  --backend dinov2_vits14 \
  --dinov2_repo output/0001/external/dinov2 \
  --max_width 224

uv run --active python -m vfm_gs.cli.validate_vfm_cache \
  -c output/0001/vfm_cache/room_dinov2_vits14 \
  -s datasets/mipnerf360/room \
  -i images_8 \
  --backend dinov2_vits14

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/room \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_i050_room_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/room_dinov2_vits14 \
  --vfm_metric_topk 0.25 \
  --vfm_importance_weight 0.50 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_i050_room_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_i050_room_30k_r8
```

weighted importance 室内房间场景复验：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/room \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_room_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/room_dinov2_vits14 \
  --vfm_metric_topk 0.25 \
  --vfm_importance_weight 0.50 \
  --vfm_importance_mode weighted \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_room_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_room_30k_r8
```

partial importance 室外树桩场景复验：

```bash
uv run --active python -m vfm_gs.cli.build_vfm_cache \
  -s datasets/mipnerf360/stump \
  -i images_8 \
  -o output/0001/vfm_cache/stump_dinov2_vits14 \
  --backend dinov2_vits14 \
  --dinov2_repo output/0001/external/dinov2 \
  --max_width 224

uv run --active python -m vfm_gs.cli.validate_vfm_cache \
  -c output/0001/vfm_cache/stump_dinov2_vits14 \
  -s datasets/mipnerf360/stump \
  -i images_8 \
  --backend dinov2_vits14

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/stump \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_i050_stump_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/stump_dinov2_vits14 \
  --vfm_metric_topk 0.25 \
  --vfm_importance_weight 0.50 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_i050_stump_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_i050_stump_30k_r8
```

RGB/VFM 加权融合在室外树桩场景的收益保持复验：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/stump \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_stump_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/stump_dinov2_vits14 \
  --vfm_metric_topk 0.25 \
  --vfm_importance_weight 0.50 \
  --vfm_importance_mode weighted \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_stump_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_weighted_i050_stump_30k_r8
```

partial importance 曲线的高质量预算点：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_i075_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_metric_topk 0.25 \
  --vfm_importance_weight 0.75 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_i075_bicycle_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_i075_bicycle_30k_r8
```

预算感知 importance 的链路验证：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_budget_aware.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_budgetaware420k_i050_bicycle_620_r8 \
  --eval \
  --iterations 620 \
  --test_iterations 620 \
  --save_iterations 620 \
  --checkpoint_iterations 620 \
  --vfm_cache_dir output/0001/vfm_cache/bicycle_dinov2_vits14 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_budgetaware420k_i050_bicycle_620_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_budgetaware420k_i050_bicycle_620_r8
```

预算感知 importance 的 30k 正式对照：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_budget_aware.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_budgetaware420k_i050_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/bicycle_dinov2_vits14 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_budgetaware420k_i050_bicycle_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_budgetaware420k_i050_bicycle_30k_r8
```

预算感知 importance 的放松衰减对照：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_budget_aware.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_budgetaware430k_s095_min010_i050_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/bicycle_dinov2_vits14 \
  --vfm_importance_budget_count 430000 \
  --vfm_importance_budget_start_ratio 0.95 \
  --vfm_importance_budget_min_weight 0.10 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_budgetaware430k_s095_min010_i050_bicycle_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_budgetaware430k_s095_min010_i050_bicycle_30k_r8
```

预算感知 importance 的非线性衰减对照：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_budget_quadratic.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_budgetquad430k_i050_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/bicycle_dinov2_vits14 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_budgetquad430k_i050_bicycle_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_budgetquad430k_i050_bicycle_30k_r8
```

支持度归一化 importance 的链路验证：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_supportnorm_i050_bicycle_620_r8 \
  --eval \
  --iterations 620 \
  --densify_from_iter 500 \
  --densify_until_iter 620 \
  --densification_interval 100 \
  --test_iterations 620 \
  --save_iterations 620 \
  --checkpoint_iterations 620 \
  --vfm_metric_topk 0.25 \
  --vfm_importance_weight 0.50 \
  --vfm_importance_normalizer support_ratio \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_supportnorm_i050_bicycle_620_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_supportnorm_i050_bicycle_620_r8
```

支持度归一化 importance 的 30k 正式对照：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_supportnorm_i050_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_metric_topk 0.25 \
  --vfm_importance_weight 0.50 \
  --vfm_importance_normalizer support_ratio \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_supportnorm_i050_bicycle_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_supportnorm_i050_bicycle_30k_r8
```

高置信 VFM 区域保护的链路验证：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_pruneprotect_i050_w025_bicycle_620_r8 \
  --eval \
  --iterations 620 \
  --densify_from_iter 500 \
  --densify_until_iter 620 \
  --densification_interval 100 \
  --test_iterations 620 \
  --save_iterations 620 \
  --checkpoint_iterations 620 \
  --vfm_metric_topk 0.25 \
  --vfm_importance_weight 0.50 \
  --vfm_prune_protect_weight 0.25 \
  --vfm_prune_protect_mode rgb_aware \
  --vfm_prune_protect_min_count 5 \
  --vfm_prune_protect_power 2.0 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_pruneprotect_i050_w025_bicycle_620_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_pruneprotect_i050_w025_bicycle_620_r8
```

高置信 VFM 区域保护的 30k 正式对照：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_pruneprotect_i050_w025_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_metric_topk 0.25 \
  --vfm_importance_weight 0.50 \
  --vfm_prune_protect_weight 0.25 \
  --vfm_prune_protect_mode rgb_aware \
  --vfm_prune_protect_min_count 5 \
  --vfm_prune_protect_power 2.0 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_pruneprotect_i050_w025_bicycle_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_pruneprotect_i050_w025_bicycle_30k_r8
```

预算约束探测：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_budget490832_final_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_metric_topk 0.25 \
  --target_gaussian_count 490832 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_budget490832_final_bicycle_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_budget490832_final_bicycle_30k_r8
```

最终预算裁剪排序诊断：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_budget490832_highscore_final_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_metric_topk 0.25 \
  --target_gaussian_count 490832 \
  --target_gaussian_prune_order highest_score \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_budget490832_highscore_final_bicycle_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_budget490832_highscore_final_bicycle_30k_r8
```

中期 staged 预算约束探测：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_budget490832_staged105_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_metric_topk 0.25 \
  --target_gaussian_count 490832 \
  --target_gaussian_staged \
  --target_gaussian_stage_margin 1.05 \
  --target_gaussian_stage_interval 500 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_budget490832_staged105_bicycle_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_budget490832_staged105_bicycle_30k_r8
```

## DINOv2 Descriptor 打分器 v0

`dinov2_descriptor_cosine` 是下一版 DINO scorer 的最小真实语义路径。它复用 GT 侧 DINO patch-token cache，但在 densification/pruning 节点对 SH0 渲染图在线跑同一个 DINOv2 模型，再用 patch-token cosine distance 生成 `pixel_error_map`。它比 `dinov2_token_edge_l1` 更贴近 proposal 中的语义特征误差，但训练开销也更高。

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_descriptor_bicycle_smoke \
  --eval \
  --iterations 220 \
  --densify_from_iter 50 \
  --densify_until_iter 220 \
  --densification_interval 50 \
  --test_iterations 220 \
  --save_iterations 220 \
  --checkpoint_iterations 220 \
  -r 8

uv run --active python -m vfm_gs.cli.render -m output/0001/vfm_dinov2_descriptor_bicycle_smoke --skip_train
uv run --active python -m vfm_gs.cli.metrics -m output/0001/vfm_dinov2_descriptor_bicycle_smoke
```

如果本地 `output/0001/external/dinov2` 不存在，先按上一节 clone 官方 DINOv2 仓库，或用命令行覆盖 `--vfm_dinov2_repo <path>`。

完整 30k 评估使用短跑细扫后的 `vfm_loss_thresh=0.35`，并显式用 `--iteration -1` 渲染最新保存结果：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_descriptor_t035_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_loss_thresh 0.35 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_descriptor_t035_bicycle_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_descriptor_t035_bicycle_30k_r8
```

descriptor 预算对齐探测使用接近 `fastgs_densify100` 点数的 target。该 run 是边界/负向结果：staged pruning 后自然结束在 410k 以下，需要和 cadence control 一起解读。

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_descriptor_budget410000_staged105_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_loss_thresh 0.35 \
  --target_gaussian_count 410000 \
  --target_gaussian_staged \
  --target_gaussian_stage_margin 1.05 \
  --target_gaussian_stage_interval 500 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_descriptor_budget410000_staged105_bicycle_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_descriptor_budget410000_staged105_bicycle_30k_r8
```

descriptor 保守接入探测使用 `vfm_importance_mode=rgb_only`。该 run 会禁用直接 descriptor densification，但保留 descriptor pruning-score fusion，用于判断默认 descriptor 的收益是否依赖直接 VFM importance。

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_descriptor_rgb_only_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_loss_thresh 0.35 \
  --vfm_importance_mode rgb_only \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_descriptor_rgb_only_bicycle_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_descriptor_rgb_only_bicycle_30k_r8
```

descriptor mask/aggregation 改进探测使用 token-grid smoothing 加 top-k metric map。默认 descriptor 仍使用 threshold；这组显式设置 `vfm_metric_map_mode=topk`、`vfm_metric_topk=0.15` 和 `vfm_descriptor_token_smooth_kernel=3`，用于避免整幅 cosine error 归一化后直接硬阈值化。

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_topk.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_descriptor_topk015_smooth3_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_descriptor_topk015_smooth3_bicycle_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_descriptor_topk015_smooth3_bicycle_30k_r8
```

top-k/smoothing staged + dense recovery 是 0001 中 descriptor 预算恢复的最终诊断之一。该 run 使用接近 `fastgs_densify100` 的 410k target，并在任意 staged pruning 发生后触发 4,096 步 dense recovery。实验结果为 PSNR 26.8472、SSIM 0.8223、LPIPS 0.1974、387,109 个 Gaussians；它只小幅修复 LPIPS/SSIM，整体仍低于 cadence control。

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_topk.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_descriptor_topk015_smooth3_budget410000_staged105_denseft4096_anyprune_s1_lr025_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --target_gaussian_count 410000 \
  --target_gaussian_staged \
  --target_gaussian_stage_margin 1.05 \
  --target_gaussian_stage_interval 500 \
  --post_prune_finetune_iterations 4096 \
  --post_prune_finetune_trigger any_prune \
  --post_prune_finetune_step_interval 1 \
  --post_prune_finetune_sh_step_interval 16 \
  --post_prune_finetune_lr_mode local \
  --post_prune_finetune_lr_scale 0.25 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_descriptor_topk015_smooth3_budget410000_staged105_denseft4096_anyprune_s1_lr025_bicycle_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_descriptor_topk015_smooth3_budget410000_staged105_denseft4096_anyprune_s1_lr025_bicycle_30k_r8
```

soft top-k 近似使用多层嵌套 top-k 二值图来模拟连续 metric map。当前 CUDA 计数接口仍是整数命中计数，因此该模式不改 rasterizer；它会对同一个 descriptor error map 生成 `vfm_metric_soft_levels` 层 top-k masks，并把每层 per-Gaussian 命中按 `1 / levels` 累加。620-step 集成验证已确认 iteration 600 会触发真实 descriptor scoring 和多层计数。

完整 30k 对照已完成，质量高于 cadence control，但点数和训练成本仍偏高。staged 410k 预算诊断也已完成；该设置能把最终点数压到 383,528，但质量低于 cadence control，因此 soft top-k 不能作为预算高效方案。

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_soft_topk.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_descriptor_soft_topk015_l3_smooth3_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_descriptor_soft_topk015_l3_smooth3_bicycle_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_descriptor_soft_topk015_l3_smooth3_bicycle_30k_r8
```

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_soft_topk.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_descriptor_soft_topk015_l3_smooth3_budget410000_staged105_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --target_gaussian_count 410000 \
  --target_gaussian_staged \
  --target_gaussian_stage_margin 1.05 \
  --target_gaussian_stage_interval 500 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_descriptor_soft_topk015_l3_smooth3_budget410000_staged105_bicycle_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_descriptor_soft_topk015_l3_smooth3_budget410000_staged105_bicycle_30k_r8
```

percentile mask 探测使用每张 descriptor error map 的分位点阈值。当前实现与 top-k 固定比例很接近：`vfm_metric_percentile=0.90` 约等价于保留最高 10% 误差像素，但阈值由每张图自身误差分布给出。该 run 已完成完整 30k，对应 PSNR 27.0036、SSIM 0.8313、LPIPS 0.1827、464,425 个 Gaussians；结果介于 top-k 15% 和 top-k 8% 之间，不能作为新的预算控制机制。

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_descriptor_percentile.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_descriptor_percentile090_smooth3_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_descriptor_percentile090_smooth3_bicycle_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_descriptor_percentile090_smooth3_bicycle_30k_r8
```

## 30k 匹配消融

220-iteration runs 只作为快速验证。迭代 scorer 行为时，使用这组 30k `-r 8` 作为最低质量门槛：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/baseline_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  -r 8

uv run --active python -m vfm_gs.cli.render -m output/0001/baseline_bicycle_30k_r8 --skip_train
uv run --active python -m vfm_gs.cli.metrics -m output/0001/baseline_bicycle_30k_r8

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_cached_edge_compact.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_cached_edge_compact_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  -r 8

uv run --active python -m vfm_gs.cli.render -m output/0001/vfm_cached_edge_compact_bicycle_30k_r8 --skip_train
uv run --active python -m vfm_gs.cli.metrics -m output/0001/vfm_cached_edge_compact_bicycle_30k_r8

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  -r 8

uv run --active python -m vfm_gs.cli.render -m output/0001/vfm_dinov2_token_edge_bicycle_30k_r8 --skip_train
uv run --active python -m vfm_gs.cli.metrics -m output/0001/vfm_dinov2_token_edge_bicycle_30k_r8
```

## 预算控制探测

现有 knobs 可以从命令行覆盖。第一次 probe 使用更高 VFM threshold 和更低 pruning fusion weight：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_cached_edge_compact.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_cached_edge_t075_w010_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_loss_thresh 0.75 \
  --vfm_weight 0.10 \
  -r 8

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_t075_w010_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_loss_thresh 0.75 \
  --vfm_weight 0.10 \
  -r 8
```

这没有让 Gaussian counts 接近 baseline，因此下一步实现应做显式 VFM importance control，而不是继续跑更多短跑网格。

## 显式 Importance Weight 探测

`vfm_importance_weight` 会在 VFM densification counts 与 RGB importance 融合前进行缩放。它与控制 pruning-score fusion 的 `vfm_weight` 分离。

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_cached_edge_compact.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_cached_edge_i025_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_importance_weight 0.25 \
  -r 8

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_i025_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_importance_weight 0.25 \
  -r 8
```

## 重要性模式探测

`vfm_importance_mode=rgb_only` 会禁用直接 VFM densification，同时保留 VFM pruning-score fusion。这个 probe 显示必须跑完整 30k；短跑快速验证指标不能暴露最终 Gaussian-budget 影响。

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_cached_edge_compact.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_cached_edge_rgb_only_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_importance_mode rgb_only \
  -r 8

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_rgb_only_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_importance_mode rgb_only \
  -r 8
```

每次训练后运行 render 和 metrics：

```bash
uv run --active python -m vfm_gs.cli.render -m <run_dir> --skip_train
uv run --active python -m vfm_gs.cli.metrics -m <run_dir>
```

## No-Effect 与 Densification Cadence 控制

严格 no-effect 不能加载 VFM experiment config 后再“关掉 VFM”，因为当前 `vfm_enable` 没有 CLI 级显式 false 开关。应从 baseline variant 出发，只覆盖需要对齐 VFM 实验的非 VFM cadence 参数。

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/fastgs_densify100_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --densification_interval 100 \
  -r 8

uv run --active python -m vfm_gs.cli.render -m output/0001/fastgs_densify100_bicycle_30k_r8 --skip_train
uv run --active python -m vfm_gs.cli.metrics -m output/0001/fastgs_densify100_bicycle_30k_r8
```

zero-weight VFM 管线控制用于确认 cache/scorer 路径开启但不直接贡献 VFM score 时是否改变结果：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_cached_edge_compact.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_cached_edge_noeffect_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_weight 0.0 \
  --vfm_importance_mode rgb_only \
  -r 8

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_noeffect_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_weight 0.0 \
  --vfm_importance_mode rgb_only \
  -r 8
```

这三组要一起解读：如果 `fastgs_densify100` 与 zero-weight VFM 点数接近，说明高点数主要来自 `densification_interval=100`，不是 VFM signal。

## 目标 Gaussian 预算探测

`target_gaussian_count` 是 final budget control。当它大于 0 时，训练会在结束时计算配置 scorer 的 pruning score，将最低分 Gaussians 裁剪到请求点数，并在训练 iteration 写出 final target-pruned PLY。

先使用 baseline 30k 点数作为第一个 budget target：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_cached_edge_compact.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_cached_edge_budget240394_lowscore_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --target_gaussian_count 240394 \
  -r 8

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_budget240394_lowscore_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --target_gaussian_count 240394 \
  -r 8
```

更早不带 `_lowscore_` 后缀的 high-score target-prune 输出保留为负例控制组，不应作为预算匹配质量结果使用。

## 最终裁剪后恢复训练探测

`post_prune_finetune_iterations` 默认关闭。设置为正数时，训练会清空残留梯度并继续执行纯光度恢复训练。默认触发条件是最终 `target_gaussian_count` 裁剪确实删除了 Gaussians；也可用 `post_prune_finetune_trigger=staged_prune|any_prune|always` 改为 staged pruning 或任意裁剪后触发。恢复后的 PLY 保存到 `iterations + post_prune_finetune_iterations`，因此 render/metrics 使用 `--iteration -1` 会自动读取恢复后的最新迭代。

dense recovery 相关参数默认保持旧行为：

- `post_prune_finetune_step_interval=0`：沿用主训练 optimizer cadence；设为 `1` 表示恢复阶段每步更新主 optimizer。
- `post_prune_finetune_sh_step_interval=0`：沿用主训练 SH optimizer cadence；设为 `1` 表示恢复阶段每步更新 SH optimizer。
- `post_prune_finetune_lr_mode=continue`：沿用全局 iteration 的 xyz LR；`local` / `restart` 使用恢复阶段局部 step 计算 xyz LR。
- `post_prune_finetune_lr_scale=1.0`：对恢复阶段 xyz LR 乘缩放系数。

快速验证命令：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/post_prune_finetune_smoke_bicycle_260_r8 \
  --eval \
  --iterations 260 \
  --densify_from_iter 50 \
  --densify_until_iter 220 \
  --densification_interval 50 \
  --test_iterations 260 \
  --save_iterations 260 \
  --checkpoint_iterations 260 \
  --target_gaussian_count 65000 \
  --post_prune_finetune_iterations 20 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/post_prune_finetune_smoke_bicycle_260_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/post_prune_finetune_smoke_bicycle_260_r8
```

dense recovery 快速验证命令：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/post_prune_dense_finetune_smoke_bicycle_260_r8 \
  --eval \
  --iterations 260 \
  --densify_from_iter 50 \
  --densify_until_iter 260 \
  --densification_interval 50 \
  --test_iterations 260 \
  --save_iterations 260 \
  --checkpoint_iterations 260 \
  --target_gaussian_count 65000 \
  --post_prune_finetune_iterations 20 \
  --post_prune_finetune_step_interval 1 \
  --post_prune_finetune_sh_step_interval 1 \
  --post_prune_finetune_lr_mode local \
  --post_prune_finetune_lr_scale 0.25 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/post_prune_dense_finetune_smoke_bicycle_260_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/post_prune_dense_finetune_smoke_bicycle_260_r8
```

完整 30k 探测命令：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_cached_edge.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_cached_edge_budget240394_lowscore_finetune4096_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --target_gaussian_count 240394 \
  --post_prune_finetune_iterations 4096 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_cached_edge_budget240394_lowscore_finetune4096_bicycle_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_cached_edge_budget240394_lowscore_finetune4096_bicycle_30k_r8
```

dense recovery 完整 30k 探测命令：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_cached_edge.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_cached_edge_budget240394_lowscore_denseft4096_s1_lr025_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --target_gaussian_count 240394 \
  --post_prune_finetune_iterations 4096 \
  --post_prune_finetune_step_interval 1 \
  --post_prune_finetune_sh_step_interval 16 \
  --post_prune_finetune_lr_mode local \
  --post_prune_finetune_lr_scale 0.25 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_cached_edge_budget240394_lowscore_denseft4096_s1_lr025_bicycle_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_cached_edge_budget240394_lowscore_denseft4096_s1_lr025_bicycle_30k_r8
```

## 分阶段目标 Gaussian 预算探测

`target_gaussian_staged` 启用训练期 budget correction。densification events 后，它会周期性重新计算 scorer pruning/support score，将 lowest-score Gaussians 向 `target_gaussian_count * target_gaussian_stage_margin` 裁剪，并继续训练。最终 target prune 仍会写出 exact-budget PLY。

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_cached_edge_compact.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_cached_edge_budget240394_staged120_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --target_gaussian_count 240394 \
  --target_gaussian_staged \
  --target_gaussian_stage_margin 1.2 \
  --target_gaussian_stage_interval 500 \
  -r 8

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_budget240394_staged120_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --target_gaussian_count 240394 \
  --target_gaussian_staged \
  --target_gaussian_stage_margin 1.2 \
  --target_gaussian_stage_interval 500 \
  -r 8
```

如果 strict 240,394-budget 质量低于 baseline，先跑更宽松的 300k target，再改 scorer：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_cached_edge_compact.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_cached_edge_budget300000_staged115_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --target_gaussian_count 300000 \
  --target_gaussian_staged \
  --target_gaussian_stage_margin 1.15 \
  --target_gaussian_stage_interval 500 \
  -r 8

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_budget300000_staged115_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --target_gaussian_count 300000 \
  --target_gaussian_staged \
  --target_gaussian_stage_margin 1.15 \
  --target_gaussian_stage_interval 500 \
  -r 8
```

如果 300k 仍低于 baseline，继续跑下一个 budget-curve 点：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_cached_edge_compact.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_cached_edge_budget350000_staged110_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --target_gaussian_count 350000 \
  --target_gaussian_staged \
  --target_gaussian_stage_margin 1.10 \
  --target_gaussian_stage_interval 500 \
  -r 8

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_budget350000_staged110_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --target_gaussian_count 350000 \
  --target_gaussian_staged \
  --target_gaussian_stage_margin 1.10 \
  --target_gaussian_stage_interval 500 \
  -r 8
```

## Counter Edge 第三场景复验

Counter 用于验证 edge 正向控制组是否能跨到一个室内场景。先跑 baseline，再构建 compact edge cache，最后按 bicycle 正向结果的约 `1.42x` baseline 点数比例设置目标点数。

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  -s datasets/mipnerf360/counter \
  -i images \
  -m output/0001/baseline_counter_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/baseline_counter_30k_r8 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/baseline_counter_30k_r8

uv run --active python -m vfm_gs.cli.build_vfm_cache \
  -s datasets/mipnerf360/counter \
  -i images_8 \
  -o output/0001/vfm_cache/counter_edge_u8 \
  --max_width 640 \
  --storage npz_uint8

uv run --active python -m vfm_gs.cli.validate_vfm_cache \
  -c output/0001/vfm_cache/counter_edge_u8 \
  -s datasets/mipnerf360/counter \
  -i images_8 \
  --backend cached_edge_l1

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_cached_edge_compact.yaml \
  -s datasets/mipnerf360/counter \
  -i images \
  -m output/0001/vfm_cached_edge_counter_budget160699_staged110_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/counter_edge_u8 \
  --target_gaussian_count 160699 \
  --target_gaussian_staged \
  --target_gaussian_stage_margin 1.10 \
  --target_gaussian_stage_interval 500 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_cached_edge_counter_budget160699_staged110_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_cached_edge_counter_budget160699_staged110_30k_r8
```

本组 edge 运行自然结束在 111,116 个 Gaussians，低于目标点数 160,699，因此没有触发最终 target prune。它仍超过 counter baseline，且点数更少；这使 `cached_edge_l1` 足以作为 0001 的 v1 正向控制组。

## 2026-04-28 快速验证

同条件低分辨率短跑，用于确认 densification 分支实际触发 scorer：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/baseline_bicycle_smoke \
  --eval \
  --iterations 220 \
  --densify_from_iter 50 \
  --densify_until_iter 220 \
  --densification_interval 50 \
  --test_iterations 220 \
  --save_iterations 220 \
  --checkpoint_iterations 220 \
  -r 8

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_scorer.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_mock_bicycle_smoke \
  --eval \
  --iterations 220 \
  --densify_from_iter 50 \
  --densify_until_iter 220 \
  --densification_interval 50 \
  --test_iterations 220 \
  --save_iterations 220 \
  --checkpoint_iterations 220 \
  -r 8

uv run --active python -m vfm_gs.cli.render -m output/0001/baseline_bicycle_smoke --skip_train
uv run --active python -m vfm_gs.cli.metrics -m output/0001/baseline_bicycle_smoke

uv run --active python -m vfm_gs.cli.render -m output/0001/vfm_mock_bicycle_smoke --skip_train
uv run --active python -m vfm_gs.cli.metrics -m output/0001/vfm_mock_bicycle_smoke

uv run --active python -m vfm_gs.cli.build_vfm_cache \
  -s datasets/mipnerf360/bicycle \
  -i images_8 \
  -o output/0001/vfm_cache/bicycle_edge_u8 \
  --max_width 640 \
  --storage npz_uint8

uv run --active python -m vfm_gs.cli.validate_vfm_cache \
  -c output/0001/vfm_cache/bicycle_edge_u8 \
  -s datasets/mipnerf360/bicycle \
  -i images_8 \
  --backend cached_edge_l1

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_cached_edge_compact.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_cached_edge_compact_bicycle_smoke \
  --eval \
  --iterations 220 \
  --densify_from_iter 50 \
  --densify_until_iter 220 \
  --densification_interval 50 \
  --test_iterations 220 \
  --save_iterations 220 \
  --checkpoint_iterations 220 \
  -r 8

uv run --active python -m vfm_gs.cli.render -m output/0001/vfm_cached_edge_compact_bicycle_smoke --skip_train
uv run --active python -m vfm_gs.cli.metrics -m output/0001/vfm_cached_edge_compact_bicycle_smoke
```

## DINO Token-Edge Adaptive Weighted 预算探测

本组继续沿用 bicycle、MipNeRF360、`-r 8` 和 30k 完整训练作为预算方法有效性的对照。`adaptive_weighted` 与旧的 soft-budget weight decay 不同：它不直接衰减 VFM 权重，而是在 Gaussian 数量接近预算区间时，把 importance fusion 从 `max(rgb, scaled_vfm)` 平滑过渡到 `weighted(rgb, vfm)`。

快速验证只用于确认新分支会被 scorer 调用：

```bash
source .venv/bin/activate

uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_topk.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_adaptive_weighted_i050_bicycle_620_r8 \
  --eval \
  --iterations 620 \
  --test_iterations 620 \
  --save_iterations 620 \
  --checkpoint_iterations 620 \
  --densify_from_iter 50 \
  --densification_interval 50 \
  --vfm_cache_dir output/0001/vfm_cache/bicycle_dinov2_vits14 \
  --vfm_metric_topk 0.25 \
  --vfm_importance_weight 0.50 \
  --vfm_importance_mode adaptive_weighted \
  --vfm_importance_budget_count 65000 \
  --vfm_importance_budget_start_ratio 0.80 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_adaptive_weighted_i050_bicycle_620_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_adaptive_weighted_i050_bicycle_620_r8
```

正式 30k 对照：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_adaptive_weighted.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_adaptive_weighted_i050_budget430k_quad_bicycle_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/bicycle_dinov2_vits14 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_adaptive_weighted_i050_budget430k_quad_bicycle_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_adaptive_weighted_i050_budget430k_quad_bicycle_30k_r8
```

结果摘要：620-step 快速验证为 PSNR 21.8205、SSIM 0.5489、LPIPS 0.4514，最终 180,605 个 Gaussians，只说明链路健康。30k quadratic 版本为 PSNR 26.9858、SSIM 0.8302、LPIPS 0.1853，最终 424,011 个 Gaussians，训练 141.94s。相比 `weighted i0.50`，它多 8,853 个点但三项质量均微升；相比普通 i0.50，少 16,060 个点但质量仍小幅低一点。下一步优先在 `stump/room/kitchen/treehill` 复验，不直接替代全场景 weighted 结论。

treehill 第二场景复验：

```bash
uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0001_vfm_topology_dinov2_token_edge_adaptive_weighted.yaml \
  -s datasets/mipnerf360/treehill \
  -i images \
  -m output/0001/vfm_dinov2_token_edge_topk025_adaptive_weighted_i050_budget430k_quad_treehill_30k_r8 \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0001/vfm_cache/treehill_dinov2_vits14 \
  -r 8

uv run --active python -m vfm_gs.cli.render \
  -m output/0001/vfm_dinov2_token_edge_topk025_adaptive_weighted_i050_budget430k_quad_treehill_30k_r8 \
  --iteration -1 \
  --skip_train \
  --quiet

uv run --active python -m vfm_gs.cli.metrics \
  -m output/0001/vfm_dinov2_token_edge_topk025_adaptive_weighted_i050_budget430k_quad_treehill_30k_r8
```

treehill 结果为 PSNR 24.4393、SSIM 0.7285、LPIPS 0.2821，最终 420,283 个 Gaussians，训练 140.96s。相比普通 treehill i0.50，PSNR 下降 -0.0780，SSIM 基本持平，LPIPS 仅微幅改善 -0.0001；相比 treehill `weighted i0.50`，点数多 2,749 个但 PSNR 下降 -0.0708。因此 quadratic adaptive weighted 不能从 bicycle 单场景候选升级为全场景预算效率候选，后续不继续扩展这条曲线。

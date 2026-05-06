# 0001 VFM 拓扑打分器运行手册

## 完整基线

```bash
uv run --active python -m vfm_gs.cli.train --variant fastgs_baseline -s <dataset>/<scene> -m output/0001_baseline/<scene> --eval
uv run --active python -m vfm_gs.cli.render -m output/0001_baseline/<scene> --skip_train
uv run --active python -m vfm_gs.cli.metrics -m output/0001_baseline/<scene>
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

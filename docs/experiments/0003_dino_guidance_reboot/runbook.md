# 0003 DINO Guidance Reboot 运行手册

## 环境

```bash
source .venv/bin/activate
```

0003 会复用 0001 的 DINOv2 本地 repo：

```bash
output/0001/external/dinov2
```

## 已知 Cache

检查 0001 主线 cache：

```bash
.venv/bin/python - <<'PY'
import json
for p in [
    "output/0001/vfm_cache/bicycle_dinov2_vits14/manifest.json",
    "output/0001/vfm_cache/bicycle_dinov2_vits14_w518/manifest.json",
    "output/0001/vfm_cache_large/bicycle_dinov2_vitl14_token_edge_w1600/manifest.json",
]:
    with open(p) as f:
        m = json.load(f)
    print(p)
    print(m["images"], m["backend"], m["feature"], m["patch_size"], m["max_width"])
    first = next(iter(m["entries"].values()))
    print(first["shape"], first["source_shape"])
PY
```

## Phase 0：训练时同款 DINO Residual 诊断

当前 `scripts/diagnose_prior_overlap.py` 适合 2D prior map。若传入 3D descriptor cache，它只能用 channel norm 作为粗略 saliency proxy，不等价于训练时的 descriptor cosine residual。

0003 已新增 `scripts/diagnose_dino_descriptor_residual.py`，用于复现 `dinov2_descriptor_cosine` 训练后端的中间 error map：

```text
baseline render image -> DINO patch tokens
GT/source image cache  -> DINO patch tokens
residual = 1 - cosine(render_token, gt_token)
```

`output/0001/vfm_cache/*_dinov2_vits14` 已保存 GT/source image 的 DINO patch-token cache。诊断脚本只需要对 render 图重新跑 DINO；若 torch hub cache 缺 DINOv2 checkpoint，则允许重新下载权重。

输入：

- baseline model dir，例如 `output/0002/fastgs_baseline_bicycle_30k_densify100_r_auto`
- render image
- GT/source image 的 DINO patch-token cache
- DINO repo，例如 `output/0001/external/dinov2`
- token grid / output size

输出：

- render-vs-GT DINO cosine residual map
- RGB error map
- top-k overlap summary
- per-view CSV
- 可选 overlay 图

已完成的 bicycle 诊断命令：

```bash
python scripts/diagnose_dino_descriptor_residual.py \
  --baseline-model output/0002/fastgs_baseline_bicycle_30k_densify100_r_auto \
  --gt-cache output/0001/vfm_cache/bicycle_dinov2_vits14 \
  --dinov2-repo output/0001/external/dinov2 \
  --device cuda \
  --topk 0.25 \
  --rgb-topk 0.25 \
  --rgb-broad-topk 0.50 \
  --smooth-kernel 3 \
  --output output/0003/diagnostics/bicycle_dino_descriptor_residual_w224_topk25

python scripts/diagnose_dino_descriptor_residual.py \
  --baseline-model output/0002/fastgs_baseline_bicycle_30k_densify100_r_auto \
  --gt-cache output/0001/vfm_cache/bicycle_dinov2_vits14_w518 \
  --dinov2-repo output/0001/external/dinov2 \
  --device cuda \
  --topk 0.25 \
  --rgb-topk 0.25 \
  --rgb-broad-topk 0.50 \
  --smooth-kernel 3 \
  --output output/0003/diagnostics/bicycle_dino_descriptor_residual_w518_topk25

python scripts/diagnose_dino_descriptor_residual.py \
  --baseline-model output/0002/fastgs_baseline_bicycle_30k_densify100_r_auto \
  --backend dinov2_vits14 \
  --max-width 1600 \
  --dinov2-repo output/0001/external/dinov2 \
  --device cuda \
  --topk 0.25 \
  --rgb-topk 0.25 \
  --rgb-broad-topk 0.50 \
  --smooth-kernel 3 \
  --output output/0003/diagnostics/bicycle_dino_descriptor_residual_w1600_topk25
```

top-10 诊断同理，把 `--topk` 和 `--rgb-topk` 改成 `0.10`，输出目录后缀改为 `topk10`。

## 高分辨率 DINO Cache 试建

Phase 0 第一轮已经用即时提取方式完成 `max_width=1600` 诊断，不需要先构建全量 cache。若后续训练需要复用 high-res GT tokens，再在 `bicycle` 上构建 ViT-S/14 high-res patch-token cache：

```bash
.venv/bin/python -m vfm_gs.cli.build_vfm_cache \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -o output/0003/vfm_cache/bicycle_dinov2_vits14_w1600_patchtokens \
  --backend dinov2_vits14 \
  --dinov2_repo output/0001/external/dinov2 \
  --max_width 1600 \
  --storage npy_float16

.venv/bin/python -m vfm_gs.cli.validate_vfm_cache \
  -c output/0003/vfm_cache/bicycle_dinov2_vits14_w1600_patchtokens \
  -s datasets/mipnerf360/bicycle \
  -i images \
  --backend dinov2_vits14
```

预期 token grid 约为 `75x114x384`。这比 0001 的 `10x16x384` 更适合 high-res 局部引导，但不宜一开始扩全数据集。

## Phase 1：RGB 放宽候选 + DINO Rerank Smoke

已实现配置：

- `configs/experiments/0003_dino_descriptor_rgb_broad.yaml`
- `configs/experiments/0003_dino_descriptor_rgb_rerank_l025.yaml`

关键参数：

```yaml
loss_thresh: 0.05
vfm_backend: dinov2_descriptor_cosine
vfm_metric_map_mode: topk
vfm_metric_topk: 0.25
vfm_importance_mode: rgb_broad 或 rgb_rerank
vfm_rgb_broad_topk: 0.50
vfm_dino_rerank_lambda: 0.25
vfm_weight: 0.0
```

核心约束：

- RGB/FastGS 先给出放宽候选，例如 top-40% 或 top-50%；当前第一版使用 top-50%。
- DINO 只在 RGB 候选内部 rerank，不允许单独把 RGB 低误差区域拉进 densification。
- 第一组 start iter 扫描：`7000/9000/11000`，保持 `densify_until_iter=15000`。
- matched 对照必须包含 `RGB-only broad candidate`，否则无法判断 DINO rerank 是否真有贡献。

620-step 只验证链路健康；30k 才做质量判断。

已完成的 high-res `bicycle` 620-step smoke：

```bash
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0003_dino_descriptor_rgb_broad.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0003/rgb_broad_bicycle_620_r_auto \
  --eval \
  --iterations 620 \
  --densify_until_iter 620 \
  --test_iterations 620 \
  --save_iterations 620 \
  --checkpoint_iterations 620 \
  --quiet

python -m vfm_gs.cli.render \
  -m output/0003/rgb_broad_bicycle_620_r_auto \
  --iteration 620 \
  --skip_train \
  --quiet

python -m vfm_gs.cli.metrics \
  -m output/0003/rgb_broad_bicycle_620_r_auto

CUDA_VISIBLE_DEVICES=0 python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0003_dino_descriptor_rgb_rerank_l025.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0003/dino_rgb_rerank_l025_bicycle_620_r_auto \
  --eval \
  --iterations 620 \
  --densify_until_iter 620 \
  --test_iterations 620 \
  --save_iterations 620 \
  --checkpoint_iterations 620 \
  --vfm_active_from_iter 600 \
  --quiet

python -m vfm_gs.cli.render \
  -m output/0003/dino_rgb_rerank_l025_bicycle_620_r_auto \
  --iteration 620 \
  --skip_train \
  --quiet

python -m vfm_gs.cli.metrics \
  -m output/0003/dino_rgb_rerank_l025_bicycle_620_r_auto
```

结果：

- RGB broad control 620：19.4699 / 0.4046 / 0.6282，63,439 Gaussians。
- DINO RGB rerank l0.25 620：19.4483 / 0.4051 / 0.6281，63,442 Gaussians。

Phase 2 第一组 30k 建议命令：

```bash
CUDA_VISIBLE_DEVICES=0 python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0003_dino_descriptor_rgb_broad.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0003/rgb_broad_bicycle_30k_r_auto \
  --eval \
  --quiet

CUDA_VISIBLE_DEVICES=1 python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0003_dino_descriptor_rgb_rerank_l025.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0003/dino_rgb_rerank_l025_start9000_bicycle_30k_r_auto \
  --eval \
  --vfm_active_from_iter 9000 \
  --quiet
```

已完成结果：

- RGB broad control 30k：25.3627 / 0.7656 / 0.2273，1,883,915 Gaussians，`output/0003/rgb_broad_bicycle_30k_r_auto`。
- DINO RGB rerank l0.25 start7000 30k：25.3695 / 0.7663 / 0.2255，1,924,629 Gaussians，`output/0003/dino_rgb_rerank_l025_start7000_bicycle_30k_r_auto`。
- DINO RGB rerank l0.25 start9000 30k：25.3538 / 0.7659 / 0.2260，1,915,967 Gaussians，`output/0003/dino_rgb_rerank_l025_start9000_bicycle_30k_r_auto`。
- DINO RGB rerank l0.25 start11000 30k：25.3515 / 0.7660 / 0.2262，1,905,234 Gaussians，`output/0003/dino_rgb_rerank_l025_start11000_bicycle_30k_r_auto`。

已完成的 start7000/start11000 双卡扫描命令：

```bash
CUDA_VISIBLE_DEVICES=0 python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0003_dino_descriptor_rgb_rerank_l025.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0003/dino_rgb_rerank_l025_start7000_bicycle_30k_r_auto \
  --eval \
  --vfm_active_from_iter 7000 \
  --quiet

CUDA_VISIBLE_DEVICES=1 python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0003_dino_descriptor_rgb_rerank_l025.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0003/dino_rgb_rerank_l025_start11000_bicycle_30k_r_auto \
  --eval \
  --vfm_active_from_iter 11000 \
  --quiet
```

下一轮建议先降 DINO rerank 强度，不直接扩多场景：

```bash
CUDA_VISIBLE_DEVICES=0 python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0003_dino_descriptor_rgb_rerank_l025.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0003/dino_rgb_rerank_l010_start7000_bicycle_30k_r_auto \
  --eval \
  --vfm_active_from_iter 7000 \
  --vfm_dino_rerank_lambda 0.10 \
  --quiet

CUDA_VISIBLE_DEVICES=1 python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0003_dino_descriptor_rgb_rerank_l025.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0003/dino_rgb_rerank_l010_start9000_bicycle_30k_r_auto \
  --eval \
  --vfm_active_from_iter 9000 \
  --vfm_dino_rerank_lambda 0.10 \
  --quiet
```

## Phase 3：DINO Prune-Protect

只有 Phase 1/2 的 densification rerank 成立后再做 pruning 方向。第一版只做保护，不做主动删除：

```text
RGB pruning says bad AND DINO says important -> protect
```

不直接启用 `DINO says redundant -> prune`，因为语义不重要的 GS 仍可能承担遮挡、边界或细纹理贡献。

## 记录要求

每轮实验必须更新：

- `docs/experiments/0003_dino_guidance_reboot/results.md`
- `docs/experiments/0003_dino_guidance_reboot/review.md`
- `docs/roadmap.md`
- `docs/experiments/index.md`

每轮完成后 commit + push。

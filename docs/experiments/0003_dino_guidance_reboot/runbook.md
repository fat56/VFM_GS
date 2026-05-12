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

## Phase 0：真实 DINO Residual 诊断

当前 `scripts/diagnose_prior_overlap.py` 适合 2D prior map。若传入 3D descriptor cache，它只能用 channel norm 作为粗略 saliency proxy，不等价于训练时的 descriptor cosine residual。

0003 需要新增诊断脚本，输入：

- baseline model dir，例如 `output/0002/fastgs_baseline_bicycle_30k_densify100_r_auto`
- GT image / render image
- DINO patch-token cache
- DINO repo
- token grid / output size

输出：

- true DINO cosine residual map
- RGB error map
- top-k overlap summary
- per-view CSV
- 可选 overlay 图

建议输出路径：

```bash
output/0003/diagnostics/bicycle_dino_descriptor_true_residual_w224
output/0003/diagnostics/bicycle_dino_descriptor_true_residual_w518
output/0003/diagnostics/bicycle_dino_descriptor_true_residual_w1600
```

## 高分辨率 DINO Cache 试建

先在 `bicycle` 上构建 ViT-S/14 high-res patch-token cache，确认体积和速度：

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

## Phase 1：RGB-Gated DINO Smoke

待实现配置方向：

```text
vfm_backend: dinov2_descriptor_cosine_rgb_gated
vfm_metric_map_mode: topk
vfm_metric_topk: 0.25
vfm_dino_rgb_gate_mode: product
vfm_dino_rgb_alpha: 1.0
vfm_dino_beta: 1.0
vfm_weight: 0.0
vfm_importance_mode: weighted
vfm_importance_weight: 0.50
```

620-step 只验证链路健康；30k 才做质量判断。

## 记录要求

每轮实验必须更新：

- `docs/experiments/0003_dino_guidance_reboot/results.md`
- `docs/experiments/0003_dino_guidance_reboot/review.md`
- `docs/roadmap.md`
- `docs/experiments/index.md`

每轮完成后 commit + push。

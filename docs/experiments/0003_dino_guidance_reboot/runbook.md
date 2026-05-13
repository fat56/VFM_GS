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
- DINO RGB rerank l0.10 start7000 30k：25.3519 / 0.7660 / 0.2262，1,913,988 Gaussians，`output/0003/dino_rgb_rerank_l010_start7000_bicycle_30k_r_auto`。
- DINO RGB rerank l0.10 start9000 30k：25.3556 / 0.7660 / 0.2261，1,907,193 Gaussians，`output/0003/dino_rgb_rerank_l010_start9000_bicycle_30k_r_auto`。

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

已完成的 l0.10 双卡扫描命令：

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

已完成的局部区域诊断命令：

```bash
python scripts/diagnose_0003_local_regions.py \
  --reference-run rgb_broad=output/0003/rgb_broad_bicycle_30k_r_auto \
  --run rgb_broad=output/0003/rgb_broad_bicycle_30k_r_auto \
  --run l025_start7000=output/0003/dino_rgb_rerank_l025_start7000_bicycle_30k_r_auto \
  --run l010_start9000=output/0003/dino_rgb_rerank_l010_start9000_bicycle_30k_r_auto \
  --gt-cache output/0001/vfm_cache/bicycle_dinov2_vits14 \
  --dinov2-repo output/0001/external/dinov2 \
  --device cuda \
  --topk 0.25 0.10 \
  --rgb-broad-topk 0.50 \
  --smooth-kernel 3 \
  --output output/0003/diagnostics/bicycle_local_regions_rgb_broad_ref_l025_l010
```

输出：

- `output/0003/diagnostics/bicycle_local_regions_rgb_broad_ref_l025_l010/summary.json`
- `output/0003/diagnostics/bicycle_local_regions_rgb_broad_ref_l025_l010/per_view.csv`

关键结果：

- DINO/RGB top-25 IoU 为 0.1627，top-10 IoU 为 0.0716。
- `l0.25 start7000` 在 RGB top-25 区域 L1 改善 -0.011869、PSNR +0.5414；在 DINO/RGB intersection top-25 区域 L1 改善 -0.012438、PSNR +0.5276。
- `l0.25 start7000` 在 DINO-only top-25 区域 L1 反而 +0.004755、PSNR -2.6560；`l0.10 start9000` 同方向。
- 当前 LPIPS 工具只返回全图标量，局部诊断只报告 L1/MSE/PSNR。

该诊断推动了后续显式 final top-m：保持 RGB broad 候选总量/最终 densification 容量不变，只让 DINO 改变候选内部排序。final-topm 判别结果见 Phase 2B。

## Phase 2B：Final Top-M 容量锁定

已新增：

- `vfm_rgb_rerank_final_topm`
- `configs/experiments/0003_dino_descriptor_rgb_rerank_final_topm_l025.yaml`

机制：`rgb_rerank` 仍先用 `vfm_rgb_broad_topk=0.50` 生成 broad candidate，再用 `RGB_importance * (1 + lambda * DINO)` 排序；但实际 densification 前会用 RGB broad reference score 计算本 step 的参考候选数 `m`，最终只保留 rerank 后的 top-m。这样 RGB broad 决定容量，DINO 只换排序。

620-step preflight smoke：

```bash
CUDA_VISIBLE_DEVICES=0 python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0003_dino_descriptor_rgb_rerank_final_topm_l025.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0003/dino_rgb_rerank_finaltopm_l025_bicycle_620_r_auto \
  --eval \
  --iterations 620 \
  --densify_until_iter 620 \
  --test_iterations 620 \
  --save_iterations 620 \
  --checkpoint_iterations 620 \
  --vfm_active_from_iter 600 \
  --quiet

CUDA_VISIBLE_DEVICES=1 python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0003_dino_descriptor_rgb_rerank_final_topm_l025.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0003/dino_rgb_rerank_finaltopm_l010_bicycle_620_r_auto \
  --eval \
  --iterations 620 \
  --densify_until_iter 620 \
  --test_iterations 620 \
  --save_iterations 620 \
  --checkpoint_iterations 620 \
  --vfm_active_from_iter 600 \
  --vfm_dino_rerank_lambda 0.10 \
  --quiet
```

smoke 结果：

- final-topm l0.25 620：19.4840 / 0.4052 / 0.6290，63,443 Gaussians。
- final-topm l0.10 620：19.4752 / 0.4052 / 0.6280，63,446 Gaussians。

30k 双卡训练命令：

```bash
CUDA_VISIBLE_DEVICES=0 python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0003_dino_descriptor_rgb_rerank_final_topm_l025.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0003/dino_rgb_rerank_finaltopm_l025_start7000_bicycle_30k_r_auto \
  --eval \
  --vfm_active_from_iter 7000 \
  --quiet

CUDA_VISIBLE_DEVICES=1 python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0003_dino_descriptor_rgb_rerank_final_topm_l025.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0003/dino_rgb_rerank_finaltopm_l010_start9000_bicycle_30k_r_auto \
  --eval \
  --vfm_active_from_iter 9000 \
  --vfm_dino_rerank_lambda 0.10 \
  --quiet
```

30k 结果：

- final-topm l0.25 start7000：25.3564 / 0.7659 / 0.2266，1,896,839 Gaussians。
- final-topm l0.10 start9000：25.3692 / 0.7657 / 0.2270，1,893,008 Gaussians。

局部诊断命令：

```bash
python scripts/diagnose_0003_local_regions.py \
  --reference-run rgb_broad=output/0003/rgb_broad_bicycle_30k_r_auto \
  --run rgb_broad=output/0003/rgb_broad_bicycle_30k_r_auto \
  --run finaltopm_l025_start7000=output/0003/dino_rgb_rerank_finaltopm_l025_start7000_bicycle_30k_r_auto \
  --run finaltopm_l010_start9000=output/0003/dino_rgb_rerank_finaltopm_l010_start9000_bicycle_30k_r_auto \
  --run rerank_l025_start7000=output/0003/dino_rgb_rerank_l025_start7000_bicycle_30k_r_auto \
  --run rerank_l010_start9000=output/0003/dino_rgb_rerank_l010_start9000_bicycle_30k_r_auto \
  --gt-cache output/0001/vfm_cache/bicycle_dinov2_vits14 \
  --dinov2-repo output/0001/external/dinov2 \
  --device cuda \
  --topk 0.25 0.10 \
  --rgb-broad-topk 0.50 \
  --smooth-kernel 3 \
  --output output/0003/diagnostics/bicycle_local_regions_rgb_broad_ref_finaltopm
```

局部结论：final-topm 在 RGB top25 / RGB broad top50 区域仍有改善，但 DINO-only top25 继续退化。l0.25 final-topm 的 DINO-only top25 为 ΔL1 +0.005009、ΔPSNR -2.7328；l0.10 final-topm 为 ΔL1 +0.005027、ΔPSNR -2.7022。因此 final-topm 只证明容量可控，未证明 DINO selector 有独立贡献。

## Phase 3：DINO Prune-Protect

Phase 2 已显示 DINO rerank 不能成为可靠 densification selector。Phase 3 只做 pruning-side protection，不做主动删除：

```text
RGB pruning says bad AND DINO says important -> protect
```

不直接启用 `DINO says redundant -> prune`，因为语义不重要的 GS 仍可能承担遮挡、边界或细纹理贡献。

新增配置：

- `configs/experiments/0003_dino_descriptor_prune_protect_only.yaml`

关键机制：

```text
vfm_importance_mode=rgb_only
vfm_weight=0.0
vfm_active_from_iter=15001
vfm_prune_protect_mode=rgb_prune_candidate
vfm_prune_protect_rgb_min_score=0.90
```

这保证 15k 前 densification 完全由 FastGS/RGB 决定；DINO 只在 18k/21k/24k/27k 的 final pruning score 里保护 RGB high-prune candidate。

620-step smoke：

```bash
CUDA_VISIBLE_DEVICES=0 python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0003_dino_descriptor_prune_protect_only.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0003/dino_pruneprotect_only_bicycle_620_r_auto \
  --eval \
  --iterations 620 \
  --densify_from_iter 500 \
  --densify_until_iter 620 \
  --test_iterations 620 \
  --save_iterations 620 \
  --checkpoint_iterations 620 \
  -r -1

CUDA_VISIBLE_DEVICES=0 python -m vfm_gs.cli.render \
  -m output/0003/dino_pruneprotect_only_bicycle_620_r_auto \
  --iteration -1 \
  --skip_train \
  --quiet

CUDA_VISIBLE_DEVICES=0 python -m vfm_gs.cli.metrics \
  -m output/0003/dino_pruneprotect_only_bicycle_620_r_auto
```

18.1k prune-path smoke：

```bash
CUDA_VISIBLE_DEVICES=1 python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0003_dino_descriptor_prune_protect_only.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0003/dino_pruneprotect_only_bicycle_18100_r_auto \
  --eval \
  --iterations 18100 \
  --test_iterations 18100 \
  --save_iterations 18100 \
  --checkpoint_iterations 18100 \
  -r -1

CUDA_VISIBLE_DEVICES=1 python -m vfm_gs.cli.render \
  -m output/0003/dino_pruneprotect_only_bicycle_18100_r_auto \
  --iteration -1 \
  --skip_train \
  --quiet

CUDA_VISIBLE_DEVICES=1 python -m vfm_gs.cli.metrics \
  -m output/0003/dino_pruneprotect_only_bicycle_18100_r_auto
```

30k pilot：

```bash
CUDA_VISIBLE_DEVICES=0 python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0003_dino_descriptor_prune_protect_only.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0003/dino_pruneprotect_only_bicycle_30k_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  -r -1

CUDA_VISIBLE_DEVICES=0 python -m vfm_gs.cli.render \
  -m output/0003/dino_pruneprotect_only_bicycle_30k_r_auto \
  --iteration -1 \
  --skip_train \
  --quiet

CUDA_VISIBLE_DEVICES=0 python -m vfm_gs.cli.metrics \
  -m output/0003/dino_pruneprotect_only_bicycle_30k_r_auto
```

Detached wrapper：

```bash
setsid bash -lc 'cd /home/m/project/ltm/VFM_GS && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=0 python -m vfm_gs.cli.train ... && CUDA_VISIBLE_DEVICES=0 python -m vfm_gs.cli.render ... && CUDA_VISIBLE_DEVICES=0 python -m vfm_gs.cli.metrics ...' \
  > output/0003/logs/dino_pruneprotect_only_bicycle_30k_r_auto.driver.log 2>&1 < /dev/null &
```

本轮实际 detached 命令：

```bash
setsid bash -lc 'cd /home/m/project/ltm/VFM_GS && source .venv/bin/activate && export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src" && CUDA_VISIBLE_DEVICES=0 python -m vfm_gs.cli.train --variant fastgs_big --config configs/experiments/0003_dino_descriptor_prune_protect_only.yaml -s datasets/mipnerf360/bicycle -i images -m output/0003/dino_pruneprotect_only_bicycle_30k_r_auto --eval --iterations 30000 --test_iterations 30000 --save_iterations 30000 --checkpoint_iterations 30000 -r -1 > output/0003/logs/dino_pruneprotect_only_bicycle_30k_r_auto.train.log 2>&1 && CUDA_VISIBLE_DEVICES=0 python -m vfm_gs.cli.render -m output/0003/dino_pruneprotect_only_bicycle_30k_r_auto --iteration -1 --skip_train --quiet > output/0003/logs/dino_pruneprotect_only_bicycle_30k_r_auto.render.log 2>&1 && CUDA_VISIBLE_DEVICES=0 python -m vfm_gs.cli.metrics -m output/0003/dino_pruneprotect_only_bicycle_30k_r_auto > output/0003/logs/dino_pruneprotect_only_bicycle_30k_r_auto.metrics.log 2>&1' > output/0003/logs/dino_pruneprotect_only_bicycle_30k_r_auto.driver.log 2>&1 < /dev/null &
```

检查命令：

```bash
rg -n "VFM PRUNE PROTECT|Gaussian number|Training time|Training complete" \
  output/0003/logs/dino_pruneprotect_only_bicycle_30k_r_auto.train.log

cat output/0003/dino_pruneprotect_only_bicycle_30k_r_auto/results.json
head -n 4 output/0003/dino_pruneprotect_only_bicycle_30k_r_auto/point_cloud/iteration_30000/point_cloud.ply
```

30k pilot 结果：

- 输出：`output/0003/dino_pruneprotect_only_bicycle_30k_r_auto`
- 日志：`output/0003/logs/dino_pruneprotect_only_bicycle_30k_r_auto.{train,render,metrics}.log`
- 指标：25.2519 PSNR / 0.7554 SSIM / 0.2449 LPIPS
- Gaussians：1,555,224
- 训练时间：160.06s
- 对照 FastGS big baseline：25.2569 / 0.7553 / 0.2450，1,560,209 点，159.11s

Protection 日志：

```text
iter=18000 protected=2 rgb_candidates=2 max=0.154052
iter=21000 protected=1 rgb_candidates=1 max=0.303642
iter=24000 protected=1 rgb_candidates=1 max=0.072603
iter=27000 protected=1 rgb_candidates=1 max=0.153624
```

结论：prune-protect 链路安全、可复现，但当前 `rgb_pruning >= 0.90` 的候选空间极窄，30k 结果基本等同 baseline。下一轮若继续 pruning，应先放宽 proposal gate 或改为 top-k proposal，再做小场景 smoke。

RGB threshold gate 复核：

```bash
CUDA_VISIBLE_DEVICES=0 python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0003_dino_descriptor_prune_protect_rgb080.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0003/dino_pruneprotect_rgb080_bicycle_18100_r_auto \
  --eval \
  --iterations 18100 \
  --test_iterations 18100 \
  --save_iterations 18100 \
  --checkpoint_iterations 18100 \
  -r -1

CUDA_VISIBLE_DEVICES=1 python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0003_dino_descriptor_prune_protect_rgb070.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0003/dino_pruneprotect_rgb070_bicycle_18100_r_auto \
  --eval \
  --iterations 18100 \
  --test_iterations 18100 \
  --save_iterations 18100 \
  --checkpoint_iterations 18100 \
  -r -1
```

结果：

- rgb080：`output/0003/dino_pruneprotect_rgb080_bicycle_18100_r_auto`，25.0579 / 0.7501 / 0.2517，1,586,419 GS。日志为 `protected=1 / rgb_candidates=1`。
- rgb070：`output/0003/dino_pruneprotect_rgb070_bicycle_18100_r_auto`，25.0638 / 0.7504 / 0.2513，1,581,575 GS。日志为 `protected=1 / rgb_candidates=1`。

结论：把 `rgb_min_score` 从 0.90 放宽到 0.80/0.70 没有扩大 RGB pruning proposal。下一步应新增 top-k/top-p proposal mode，而不是继续扫绝对阈值。

2026-05-13 smoke 结果：

- 620 preflight：`output/0003/dino_pruneprotect_only_bicycle_620_r_auto`，19.3464 / 0.4003 / 0.6293，66,232 GS。只验证配置与 cache preflight，不触发 final pruning。
- 18.1k prune-path：`output/0003/dino_pruneprotect_only_bicycle_18100_r_auto`，25.0803 / 0.7510 / 0.2504，1,586,344 GS。iteration 18000 打出 `[VFM PRUNE PROTECT] iter=18000 mode=rgb_prune_candidate weight=0.2500 protected=1 rgb_candidates=1 mean=0.000000 max=0.132225`，证明 pruning-only protection 链路真实触发。

## 记录要求

每轮实验必须更新：

- `docs/experiments/0003_dino_guidance_reboot/results.md`
- `docs/experiments/0003_dino_guidance_reboot/review.md`
- `docs/roadmap.md`
- `docs/experiments/index.md`

每轮完成后 commit + push。

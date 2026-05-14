# 0002 Depth Anything Dense Depth Prior 运行手册

## 环境

推荐先激活当前环境：

```bash
source .venv/bin/activate
```

Depth Anything V2-S cache builder 依赖 Transformers / HuggingFace Hub。当前服务器已安装并验证：

```bash
uv pip install transformers huggingface-hub safetensors
```

如需重建环境，先用下面命令确认依赖和 GPU 状态：

```bash
python -m vfm_gs.cli.vfm_backend_probe --width 1600 --height 1066 --num_images 194
```

## Phase 0：双卡 5090 Baseline 复核

在 Depth Anything 实验前，先使用双卡 RTX 5090 跑完三个公开数据集所有场景的 FastGS big baseline。分辨率保持原图输入和 FastGS 原始 1.6K 自动缩放口径，即 `-i images -r -1`。目标是确认 5090 环境与此前 4090D 结果无明显质量漂移。

复用已有 runner：

```bash
source .venv/bin/activate
```

MipNeRF360 可以拆成两张卡并行：

```bash
setsid bash -lc 'cd /home/m/project/ltm/VFM_GS && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=0 uv run --active python scripts/run_0001_fastgs_big_eval.py \
  --dataset-name mipnerf360 \
  --dataset-root datasets/mipnerf360 \
  --output-root output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_gpu0 \
  --scenes flowers garden stump treehill' \
  > output/0002/debug_logs/phase0_fix1_mipnerf360_gpu0_detached.log 2>&1 < /dev/null &

setsid bash -lc 'cd /home/m/project/ltm/VFM_GS && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=1 uv run --active python scripts/run_0001_fastgs_big_eval.py \
  --dataset-name mipnerf360 \
  --dataset-root datasets/mipnerf360 \
  --output-root output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_gpu1 \
  --scenes room counter kitchen bonsai' \
  > output/0002/debug_logs/phase0_fix1_mipnerf360_gpu1_detached.log 2>&1 < /dev/null &
```

DB 与 Tandt 也可以并行：

```bash
setsid bash -lc 'cd /home/m/project/ltm/VFM_GS && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=0 uv run --active python scripts/run_0001_fastgs_big_eval.py \
  --dataset-name db \
  --dataset-root datasets/tandt_db/db \
  --output-root output/0002/phase0_5090_fastgs_big_baseline_fix1/db \
  --scenes drjohnson playroom' \
  > output/0002/debug_logs/phase0_fix1_db_detached.log 2>&1 < /dev/null &

setsid bash -lc 'cd /home/m/project/ltm/VFM_GS && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=1 uv run --active python scripts/run_0001_fastgs_big_eval.py \
  --dataset-name tandt \
  --dataset-root datasets/tandt_db/tandt \
  --output-root output/0002/phase0_5090_fastgs_big_baseline_fix1/tandt \
  --scenes train truck' \
  > output/0002/debug_logs/phase0_fix1_tandt_detached.log 2>&1 < /dev/null &
```

当前环境未安装 `screen`。长实验使用上面的 `setsid ... < /dev/null &` 保活方式；进程的 PPID 会变成 1，SSH 断开后仍继续运行。用下面命令监控：

```bash
ps -eo pid,ppid,stat,etime,cmd | rg 'run_0001_fastgs_big_eval|vfm_gs.cli.train|vfm_gs.cli.render|vfm_gs.cli.metrics'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
tail -n 80 output/0002/debug_logs/phase0_fix1_db_detached.log
tail -n 80 output/0002/debug_logs/phase0_fix1_tandt_detached.log
```

如果 SSH 断开期间出现截断产物，先把不完整场景目录移动到 `output/0002/debug_artifacts/` 留档，再重跑该场景。2026-05-11 的非 detached 补跑中，`treehill` PLY 截断并在 render 时报 `early end-of-file`，`bonsai` 停在半程；归档后 detached 重跑已通过。

这些命令默认使用：

- `--variant fastgs_big`
- `--densification-interval 100`
- `--resolution -1`
- `--train-images images`
- 脚本内置的 MipNeRF360/DB/Tandt scene overrides

Phase 0 产物：

- `output/0002/phase0_5090_fastgs_big_baseline/*/summary.csv`
- `output/0002/phase0_5090_fastgs_big_baseline/*/summary.json`
- `output/0002/phase0_5090_fastgs_big_baseline/*/averages.json`
- 每个场景的 `logs/<run-name>/{train,render,metrics}.log`

验收后再把摘要写入 `results.md`。若 5090 与 4090D 的 PSNR/SSIM/LPIPS 出现明显偏移，先暂停 Depth Anything，排查环境和编译链路。

### Phase 0 故障诊断

若 high-res FastGS big baseline 出现 CUDA illegal memory access，先不要继续调度全数据集。先收集日志：

```bash
rg -n "Traceback|illegal memory|AcceleratorError|CUDA error" \
  output/0002/phase0_5090_fastgs_big_baseline
```

用同步 CUDA 报错复现一个单场景短跑：

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_big \
  -s datasets/mipnerf360/room \
  -i images \
  -m output/0002/debug_5090_illegal_room_2500_r_auto \
  --eval \
  --iterations 2500 \
  --test_iterations 2500 \
  --save_iterations 2500 \
  --checkpoint_iterations 2500 \
  --densification_interval 100 \
  --optimizer_type default \
  --highfeature_lr 0.02 \
  --grad_abs_thresh 0.0004 \
  -r -1
```

若同步复现仍指向 rasterizer / knn / fused SSIM 等本地 CUDA extension，按当前 PyTorch/CUDA/RTX 5090 环境重编译后再复跑 Phase 0。重编译操作和结果需要写回 `review.md`。

2026-05-11 的 `room` 2500-step 同步诊断已完成且未复现 illegal memory access。若继续推进 Phase 0，可先用 blocking 口径复跑并把输出写入新目录，避免覆盖首次失败日志：

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 uv run --active python scripts/run_0001_fastgs_big_eval.py \
  --dataset-name mipnerf360 \
  --dataset-root datasets/mipnerf360 \
  --output-root output/0002/phase0_5090_fastgs_big_baseline_blocking/mipnerf360_gpu0 \
  --scenes bicycle flowers garden stump treehill

CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 uv run --active python scripts/run_0001_fastgs_big_eval.py \
  --dataset-name mipnerf360 \
  --dataset-root datasets/mipnerf360 \
  --output-root output/0002/phase0_5090_fastgs_big_baseline_blocking/mipnerf360_gpu1 \
  --scenes room counter kitchen bonsai
```

blocking 复跑若仍在 `render_fastgs` 的 `radii` 访问处失败，继续用 rasterizer debug 复现：

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_big \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0002/debug_5090_illegal_bicycle_9500_debug_r_auto \
  --eval \
  --iterations 9500 \
  --test_iterations 9500 \
  --save_iterations 9500 \
  --checkpoint_iterations 9500 \
  --densification_interval 100 \
  --optimizer_type default \
  --debug_from 9000 \
  -r -1
```

若错误在 `debug_from` 指定迭代前提前触发，改成从第一步启用 debug：

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 uv run --active python -m vfm_gs.cli.train \
  --variant fastgs_big \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0002/debug_5090_illegal_bicycle_5000_debug0_r_auto \
  --eval \
  --iterations 5000 \
  --test_iterations 5000 \
  --save_iterations 5000 \
  --checkpoint_iterations 5000 \
  --densification_interval 100 \
  --optimizer_type default \
  --debug_from 0 \
  -r -1
```

注意：2026-05-11 已修补 `diff_gaussian_rasterization_fastgs` Python wrapper 的 debug 分支，使其与当前 FastGS 扩展的 9 项返回值一致。若换环境或重装依赖后 debug 模式又出现 `ValueError: too many values to unpack (expected 7)`，先确认 `submodules/diff-gaussian-rasterization_fastgs/diff_gaussian_rasterization_fastgs/__init__.py` 和安装副本是否包含同一修补。

修补 wrapper 后的 `bicycle` high-res debug 复现已经在 1220/5000 捕获到真实 CUDA 检查点：

- 日志：`output/0002/debug_logs/bicycle_5000_debug_from_0_fixed_wrapper.log`
- 输出：`output/0002/debug_5090_illegal_bicycle_5000_debug0_fixed_wrapper_r_auto`
- snapshot：`output/0002/debug_artifacts/snapshot_fw_identify_tile_ranges_global_shared_20260511.dump`
- 位置：`submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/rasterizer_impl.cu:422`，紧跟 `identifyTileRanges` kernel
- 错误：`operation not supported on global/shared address space`

第一轮修补后，`bicycle` high-res 5000-step debug 验证已完成：

- 日志：`output/0002/debug_logs/bicycle_5000_debug_from_0_fix1.log`
- 输出：`output/0002/debug_5090_fix1_bicycle_5000_debug0_r_auto`
- 结果：5000/5000 完成，1,340,808 个 Gaussians，未再触发 `identifyTileRanges` / CUDA illegal memory access

这只代表原阻塞点通过 debug 验证。Phase 0 仍需正常 30k train/render/metrics 后才能视为通过，不要提前启动 Depth Anything 训练验证。

修补后的正常 30k 单场景验收命令：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --active python scripts/run_0001_fastgs_big_eval.py \
  --dataset-name mipnerf360 \
  --dataset-root datasets/mipnerf360 \
  --output-root output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_single_gpu0 \
  --scenes bicycle
```

2026-05-11 结果：`bicycle` train/render/metrics 完成，25.2569 / 0.7553 / 0.2450，1,560,209 个 Gaussians，训练 159.11s。该结果与 0001 同口径 baseline 基本贴合；可继续恢复双卡 MipNeRF360 全场景 baseline。

MipNeRF360 全场景 fix1 已完成：

- 汇总：`output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_combined/summary.csv`
- 均值：27.9590 / 0.8203 / 0.2157，1,161,786 个 Gaussians，训练 159.70s
- 结论：与 0001 high-res FastGS big baseline 27.9293 / 0.8198 / 0.2157、1,161,242 点基本一致；MipNeRF360 Phase 0 通过

DB/Tandt fix1 也已完成：

- DB 汇总：`output/0002/phase0_5090_fastgs_big_baseline_fix1/db/summary.csv`
- DB 均值：30.2331 / 0.9111 / 0.2397，646,600 个 Gaussians，训练 98.54s
- Tandt 汇总：`output/0002/phase0_5090_fastgs_big_baseline_fix1/tandt/summary.csv`
- Tandt 均值：24.4955 / 0.8579 / 0.1736，540,119 个 Gaussians，训练 105.37s
- 结论：与 0001 high-res FastGS big baseline DB 30.2073 / 0.9112 / 0.2402、650,194 点，以及 Tandt 24.3557 / 0.8573 / 0.1745、540,578 点基本一致；Phase 0 全部通过

## 目标流程

0002 沿用 0001 已验证的 VFM scorer 主链路：

```text
GT image -> dense depth cache
rendered SH0/albedo -> online depth prediction or depth-edge comparison
depth residual / depth edge prior -> pixel_error_map
pixel_error_map -> top-k metric_map
render_fastgs(..., get_flag=True) -> accum_metric_counts
counts -> densification importance
```

第一阶段保持 `vfm_weight=0.0`，只影响 densification。

## Prior/RGB 瓶颈诊断

每个新 prior 完成 train/render/metrics 后，先跑轻量重叠诊断，再决定是否扩场景。脚本会读取 render/gt、`cameras.json` 和 VFM cache，输出 `per_view.csv` 与 `summary.json`：

```bash
.venv/bin/python scripts/diagnose_prior_overlap.py \
  --baseline-model output/0002/fastgs_baseline_bicycle_30k_densify100_r_auto \
  --candidate-model output/0002/depth_anything_depth_prior_bicycle_30k_r_auto \
  --prior-cache output/0002/vfm_cache/bicycle_depth_anything_v2s_depth \
  --output-dir output/0002/diagnostics/bicycle_depth_prior_overlap \
  --topk 0.25 \
  --rgb-topk 0.25
```

建议同时跑 top-k 10% 敏感性检查：

```bash
.venv/bin/python scripts/diagnose_prior_overlap.py \
  --baseline-model output/0002/fastgs_baseline_bicycle_30k_densify100_r_auto \
  --candidate-model output/0002/depth_anything_depth_prior_bicycle_30k_r_auto \
  --prior-cache output/0002/vfm_cache/bicycle_depth_anything_v2s_depth \
  --output-dir output/0002/diagnostics/bicycle_depth_prior_overlap_topk10 \
  --topk 0.10 \
  --rgb-topk 0.10
```

没有 candidate 时也可以只诊断 prior 是否覆盖 baseline RGB 高误差区域：

```bash
.venv/bin/python scripts/diagnose_prior_overlap.py \
  --baseline-model output/0002/fastgs_baseline_bicycle_30k_densify100_r_auto \
  --prior-cache output/0001/vfm_cache_large/bicycle_dinov2_vitl14_token_edge_w1600 \
  --output-dir output/0002/diagnostics/bicycle_dino_token_edge_baseline_overlap \
  --topk 0.25 \
  --rgb-topk 0.25
```

重点看这些字段：

- `prior_rgb_topk_iou` / `prior_rgb_topk_recall`：prior 关心的区域是否覆盖 RGB 高误差区域。
- `baseline_l1_prior_topk` vs `baseline_l1_non_prior`：prior 区域是否确实更难。
- `delta_l1_prior_topk` vs `delta_l1_non_prior`：candidate 的改善是否真的落在 prior 区域。

## Depth Anything Smoke

Phase 0 通过且 Depth Anything backend 落地后，先跑 high-res bicycle 620-step。0002 不再使用 `-r 8` 低分辨率 smoke，保持原图输入和 FastGS 1.6K 自动缩放口径。

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

`HF_HUB_DISABLE_XET=1` 是当前服务器的必要绕过：首次默认 HuggingFace Xet 下载在模型权重阶段出现 `RemoteProtocolError`，禁用 Xet 后 `depth-anything/Depth-Anything-V2-Small-hf` 可正常加载。

2026-05-11 smoke 结果：

- cache：`output/0002/vfm_cache/bicycle_depth_anything_v2s_edge`，194 entries，48MB，validate 通过。
- Depth Anything V2-S depth-edge prior：19.4402 / 0.4039 / 0.6270，61,277 个 Gaussians，训练 1.90s。
- matched FastGS 620 baseline：19.4930 / 0.4046 / 0.6268，61,278 个 Gaussians，训练 1.81s。
- 结论：链路健康，但 620-step 短程指标略低于 matched baseline；质量判断必须看 high-res 30k pilot。

下面是早期 depth residual 草案，保留作后续变体参考；当前优先级低于已跑通的 depth-edge prior。

```bash
python -m vfm_gs.cli.build_vfm_cache \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -o output/0002/vfm_cache/bicycle_depth_anything \
  --backend depth_anything \
  --max_width 1600

python -m vfm_gs.cli.validate_vfm_cache \
  -c output/0002/vfm_cache/bicycle_depth_anything \
  -s datasets/mipnerf360/bicycle \
  -i images \
  --backend depth_anything

python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0002_depth_anything_depth_residual_densify_only_topk025_weighted_i050.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0002/depth_anything_depth_residual_bicycle_620_r_auto \
  --eval \
  --iterations 620 \
  --densify_from_iter 500 \
  --densify_until_iter 620 \
  --densification_interval 100 \
  --test_iterations 620 \
  --save_iterations 620 \
  --checkpoint_iterations 620 \
  --vfm_cache_dir output/0002/vfm_cache/bicycle_depth_anything \
  -r -1

python -m vfm_gs.cli.render \
  -m output/0002/depth_anything_depth_residual_bicycle_620_r_auto \
  --iteration -1 \
  --skip_train \
  --quiet

python -m vfm_gs.cli.metrics \
  -m output/0002/depth_anything_depth_residual_bicycle_620_r_auto
```

## Depth Anything Pilot 30k 对照

正式对照直接使用 high-res 1.6K 口径。先跑少数 pilot 场景；若多场景有效，再扩展全数据集。

```bash
setsid bash -lc 'cd /home/m/project/ltm/VFM_GS && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=0 python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0002_depth_anything_depth_edge_prior_densify_only_topk025_weighted_i050.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0002/depth_anything_edge_prior_bicycle_30k_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0002/vfm_cache/bicycle_depth_anything_v2s_edge \
  -r -1' > output/0002/debug_logs/depth_anything_edge_prior_bicycle_30k_train.log 2>&1 < /dev/null &

python -m vfm_gs.cli.render \
  -m output/0002/depth_anything_edge_prior_bicycle_30k_r_auto \
  --iteration -1 \
  --skip_train \
  --quiet

python -m vfm_gs.cli.metrics \
  -m output/0002/depth_anything_edge_prior_bicycle_30k_r_auto
```

为了公平解释 30k pilot，需补齐同 recipe matched baseline：

```bash
setsid bash -lc 'cd /home/m/project/ltm/VFM_GS && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=0 python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0002/fastgs_baseline_bicycle_30k_densify100_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --densification_interval 100 \
  -r -1' > output/0002/debug_logs/fastgs_baseline_bicycle_30k_densify100_train.log 2>&1 < /dev/null &

python -m vfm_gs.cli.render \
  -m output/0002/fastgs_baseline_bicycle_30k_densify100_r_auto \
  --iteration -1 \
  --skip_train \
  --quiet

python -m vfm_gs.cli.metrics \
  -m output/0002/fastgs_baseline_bicycle_30k_densify100_r_auto
```

2026-05-11 `bicycle` 30k 结果：Depth Anything V2-S depth-edge prior 为 25.0764 / 0.7387 / 0.2744、1,063,311 个 Gaussians；matched baseline 为 25.0787 / 0.7370 / 0.2779、1,023,912 个 Gaussians。结论是弱混合信号，不直接扩全数据集；下一轮优先跑 direct relative depth prior。

direct relative depth prior 的下一轮命令：

```bash
HF_HUB_DISABLE_XET=1 python -m vfm_gs.cli.build_vfm_cache \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -o output/0002/vfm_cache/bicycle_depth_anything_v2s_depth \
  --backend depth_anything_v2 \
  --max_width 1600 \
  --device cuda \
  --depth_anything_feature depth \
  --storage npz_uint8

python -m vfm_gs.cli.validate_vfm_cache \
  -c output/0002/vfm_cache/bicycle_depth_anything_v2s_depth \
  -s datasets/mipnerf360/bicycle \
  -i images \
  --backend depth_anything_v2

python -m vfm_gs.cli.train \
  --variant fastgs_baseline \
  --config configs/experiments/0002_depth_anything_depth_prior_densify_only_topk025_weighted_i050.yaml \
  -s datasets/mipnerf360/bicycle \
  -i images \
  -m output/0002/depth_anything_depth_prior_bicycle_30k_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0002/vfm_cache/bicycle_depth_anything_v2s_depth \
  -r -1

python -m vfm_gs.cli.render \
  -m output/0002/depth_anything_depth_prior_bicycle_30k_r_auto \
  --iteration -1 \
  --skip_train \
  --quiet

python -m vfm_gs.cli.metrics \
  -m output/0002/depth_anything_depth_prior_bicycle_30k_r_auto
```

2026-05-11 direct relative depth prior `bicycle` 30k 结果：25.1415 / 0.7434 / 0.2689、1,005,953 个 Gaussians、训练 128.82s；相对 matched baseline 为 +0.0628 PSNR、+0.0063 SSIM、LPIPS -0.0090，Gaussian 数 -17,959。该变体成为 0002 当前主线。

后续 pilot 场景按同一模板替换 dataset/output/cache 路径。注意：如果对照对象是 Phase 0 FastGS big baseline，必须同时复制 `scripts/run_0001_fastgs_big_eval.py` 中的 per-scene overrides；只写 `--variant fastgs_big` 不会自动带上这些场景超参。

```bash
HF_HUB_DISABLE_XET=1 python -m vfm_gs.cli.build_vfm_cache \
  -s datasets/mipnerf360/stump \
  -i images \
  -o output/0002/vfm_cache/stump_depth_anything_v2s_depth \
  --backend depth_anything_v2 \
  --max_width 1600 \
  --device cuda \
  --depth_anything_feature depth \
  --storage npz_uint8

python -m vfm_gs.cli.validate_vfm_cache \
  -c output/0002/vfm_cache/stump_depth_anything_v2s_depth \
  -s datasets/mipnerf360/stump \
  -i images \
  --backend depth_anything_v2

setsid bash -lc 'cd /home/m/project/ltm/VFM_GS && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=0 python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0002_depth_anything_depth_prior_densify_only_topk025_weighted_i050.yaml \
  -s datasets/mipnerf360/stump \
  -i images \
  -m output/0002/depth_anything_depth_prior_fastgs_big_stump_30k_scene_override_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0002/vfm_cache/stump_depth_anything_v2s_depth \
  --dense 0.004 \
  --grad_abs_thresh 0.001 \
  -r -1' > output/0002/debug_logs/depth_anything_depth_prior_fastgs_big_stump_30k_scene_override_train.log 2>&1 < /dev/null &
```

已验证的 MipNeRF360 scene overrides：

| 场景 | 必须附加参数 |
|---|---|
| stump | `--dense 0.004 --grad_abs_thresh 0.001` |
| bonsai | `--highfeature_lr 0.02 --grad_abs_thresh 0.0002` |

2026-05-14 的有效 fastgs_big direct depth prior 输出：

- `stump`: `output/0002/depth_anything_depth_prior_fastgs_big_stump_30k_scene_override_r_auto`
- `bonsai`: `output/0002/depth_anything_depth_prior_fastgs_big_bonsai_30k_scene_override_r_auto`
- `playroom`: `output/0002/depth_anything_depth_prior_fastgs_big_playroom_30k_scene_override_r_auto`
- `truck`: `output/0002/depth_anything_depth_prior_fastgs_big_truck_30k_scene_override_r_auto`

对应 overlap 诊断：

```bash
.venv/bin/python scripts/diagnose_prior_overlap.py \
  --baseline-model output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_gpu0/stump/fastgs_big_densify100_30k_r_auto \
  --candidate-model output/0002/depth_anything_depth_prior_fastgs_big_stump_30k_scene_override_r_auto \
  --prior-cache output/0002/vfm_cache/stump_depth_anything_v2s_depth \
  --output-dir output/0002/diagnostics/stump_depth_prior_fastgs_big_scene_override_overlap \
  --topk 0.25 \
  --rgb-topk 0.25

.venv/bin/python scripts/diagnose_prior_overlap.py \
  --baseline-model output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_gpu1/bonsai/fastgs_big_densify100_30k_r_auto \
  --candidate-model output/0002/depth_anything_depth_prior_fastgs_big_bonsai_30k_scene_override_r_auto \
  --prior-cache output/0002/vfm_cache/bonsai_depth_anything_v2s_depth \
  --output-dir output/0002/diagnostics/bonsai_depth_prior_fastgs_big_scene_override_overlap \
  --topk 0.25 \
  --rgb-topk 0.25
```

DB/Tandt scene overrides：

| 场景 | 必须附加参数 |
|---|---|
| playroom | `--highfeature_lr 0.0015 --dense 0.003 --mult 0.7 --grad_abs_thresh 0.0005` |
| truck | `--highfeature_lr 0.04 --grad_abs_thresh 0.0004 --mult 0.7` |

`playroom/truck` cache 与训练命令：

```bash
HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=0 python -m vfm_gs.cli.build_vfm_cache \
  -s datasets/tandt_db/db/playroom \
  -i images \
  -o output/0002/vfm_cache/playroom_depth_anything_v2s_depth \
  --backend depth_anything_v2 \
  --max_width 1600 \
  --device cuda \
  --depth_anything_feature depth \
  --storage npz_uint8

setsid bash -lc 'cd /home/m/project/ltm/VFM_GS && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=0 python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0002_depth_anything_depth_prior_densify_only_topk025_weighted_i050.yaml \
  -s datasets/tandt_db/db/playroom \
  -i images \
  -m output/0002/depth_anything_depth_prior_fastgs_big_playroom_30k_scene_override_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0002/vfm_cache/playroom_depth_anything_v2s_depth \
  --highfeature_lr 0.0015 \
  --dense 0.003 \
  --mult 0.7 \
  --grad_abs_thresh 0.0005 \
  -r -1' > output/0002/debug_logs/depth_anything_depth_prior_fastgs_big_playroom_30k_scene_override_train.log 2>&1 < /dev/null &

HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=1 python -m vfm_gs.cli.build_vfm_cache \
  -s datasets/tandt_db/tandt/truck \
  -i images \
  -o output/0002/vfm_cache/truck_depth_anything_v2s_depth \
  --backend depth_anything_v2 \
  --max_width 1600 \
  --device cuda \
  --depth_anything_feature depth \
  --storage npz_uint8

setsid bash -lc 'cd /home/m/project/ltm/VFM_GS && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=1 python -m vfm_gs.cli.train \
  --variant fastgs_big \
  --config configs/experiments/0002_depth_anything_depth_prior_densify_only_topk025_weighted_i050.yaml \
  -s datasets/tandt_db/tandt/truck \
  -i images \
  -m output/0002/depth_anything_depth_prior_fastgs_big_truck_30k_scene_override_r_auto \
  --eval \
  --iterations 30000 \
  --test_iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --vfm_cache_dir output/0002/vfm_cache/truck_depth_anything_v2s_depth \
  --highfeature_lr 0.04 \
  --grad_abs_thresh 0.0004 \
  --mult 0.7 \
  -r -1' > output/0002/debug_logs/depth_anything_depth_prior_fastgs_big_truck_30k_scene_override_train.log 2>&1 < /dev/null &
```

必须对照：

- Phase 0 FastGS big baseline。
- High-res 0001 DINO descriptor top-k25 weighted i0.50。
- High-res 0001 DINO descriptor top-k25 weighted i0.70。
- 若可用，再加 DINO descriptor top-k25 `max` 作为质量上界。

每一批实验完成后：

```bash
git add <changed-files>
git commit -m "..."
git push origin main
```

当前只有本服务器修改，按用户要求不再强制每轮先 `git pull`。若后续出现多人协作，再恢复 pull/rebase 检查。

## 结果回写

只把摘要、指标表和输出路径写入 `results.md`。原始 artifact 保留在：

- `output/0002/...`
- `eval/...`
- 外部存储

每次新增正式 30k 结果后，更新：

- `docs/experiments/0002_depth_anything_dense_prior/results.md`
- `docs/experiments/0002_depth_anything_dense_prior/review.md`
- `docs/roadmap.md`
- `docs/experiments/index.md`，如果实验状态发生阶段变化

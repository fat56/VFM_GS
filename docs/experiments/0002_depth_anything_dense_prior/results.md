# 0002 实验结果

## 当前状态

Phase 0 已开始。首次双卡 high-res FastGS big baseline 在 MipNeRF360 首个场景上均提前失败，失败发生在 Depth Anything 接入前，因此该阶段首先修复 5090 / CUDA 12.8 / Blackwell 下的 FastGS rasterizer 稳定性，而不是评价 dense depth prior。

已完成第一轮 rasterizer 修补、5000-step debug 验证、MipNeRF360 `bicycle` 单场景 30k 验收，以及 MipNeRF360/DB/Tandt 三个公开数据集全 13 场景 high-res FastGS big train/render/metrics。三数据集结果与 0001/4090D 同口径 high-res FastGS big baseline 基本贴合。当前结论：5090 环境 Phase 0 通过。

Depth Anything V2-S dense depth cache/backend 已完成最小接入，并通过 high-res `bicycle` 620-step smoke、depth-edge 30k pilot、direct relative depth 30k pilot，以及 `fastgs_big + scene overrides` 下的 `stump/bonsai/playroom/truck` pilot。30k matched `fastgs_baseline + densify100` 对照下，depth-edge prior 是弱混合信号；direct relative depth prior 在 `bicycle` 上三项质量正向且 Gaussian 数更少。`fastgs_big` 对齐场景超参后，`stump` 三项质量正向，`playroom` PSNR/LPIPS 正向但 SSIM 微负，`bonsai` PSNR 微负但 SSIM/LPIPS 正向且 QCGI 为负，`truck` 三项质量负向。2026-05-17 的 RGB-gated depth rerank 完成 l0.25/l0.10/l0.05：l0.25 把 `truck` 从 direct depth 的明确负例翻成三项质量正向，但三场景 QCGI 均为负；l0.10 把 `stump/truck` 拉回三项质量正向，但 `playroom` 仍负向；l0.05 让 `playroom` PSNR 转正，但 `truck` 又回到 PSNR 负向。2026-05-18 的 `l0.10 broad035` 继续保住 `stump`，但 `playroom/truck` 仍然没有稳定正向；`start9000` 能改善 `playroom`，但 `stump` 容量失控、`truck` 仍负向。按用户建议扩展到 MipNeRF360 全 9 场景后，均值为 27.9698 / 0.8238 / 0.2070、1,399,502 GS，相对 Phase 0 是 +0.0107 PSNR、+0.0036 SSIM、LPIPS -0.0087，但平均多 237,716 GS，QCGI 均值 -0.5602 且 9/9 场景 QCGI 为负。2026-05-18 的 `prune-protect-only` 小规模验证则更像后期辅助修正：`stump/playroom/truck` 平均 +0.0089 PSNR、-0.0003 SSIM、LPIPS +0.0008、GS -5,253，QCGI 近乎持平但略负；`playroom` 和 `truck` 是正例，`stump` 仍是负例。`weight015` sweep 平均为 -0.0480 PSNR、-0.0004 SSIM、LPIPS +0.0010、GS -1,636，说明单独降保护权重更差；`topk005` sweep 平均为 -0.0185 PSNR、-0.0003 SSIM、LPIPS +0.0010、GS -6,755，QCGI -0.0295，说明继续收窄 RGB proposal 也没有成为稳定解。`topk010` 扩到 MipNeRF360 全 9 场景后，均值为 27.9128 / 0.8196 / 0.2162、1,159,451 GS，相对 Phase 0 为 -0.0462 PSNR、-0.0007 SSIM、LPIPS +0.0005、GS -2,335，QCGI -0.0643。当前结论：prune-side auxiliary 比 RGB rerank 稳，但固定 `topk010/005` 和 `weight015` 都不能作为默认方案；0002 应暂停固定 Depth Anything protect 继续扩展，转向场景自适应门控或回到其他更直接的误差入口。

## Phase 0：5090 FastGS Big Baseline 复核

| 日期 | 数据集 | 场景数 | 输出路径 | PSNR/SSIM/LPIPS 是否对齐 4090D | 结论 |
|---|---|---:|---|---|---|
| 2026-05-11 | MipNeRF360 | 9 | `output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_combined` | 是；相对 0001 high-res baseline 为 +0.0297 PSNR / +0.0005 SSIM / LPIPS 约持平 / +544 GS | 通过；继续 DB/Tandt |
| 2026-05-11 | DB | 2 | `output/0002/phase0_5090_fastgs_big_baseline_fix1/db` | 是；相对 0001 high-res baseline 为 +0.0258 PSNR / -0.0001 SSIM / LPIPS -0.0005 / -3,595 GS | 通过 |
| 2026-05-11 | Tandt | 2 | `output/0002/phase0_5090_fastgs_big_baseline_fix1/tandt` | 是；相对 0001 high-res baseline 为 +0.1398 PSNR / +0.0006 SSIM / LPIPS -0.0009 / -459 GS | 通过 |

### 2026-05-11 首次 high-res baseline 启动

| 数据集 | 场景 | GPU | 输出路径 | 进度 | 失败位置 | 错误 | 结论 |
|---|---|---:|---|---:|---|---|---|
| MipNeRF360 | bicycle | 0 | `output/0002/phase0_5090_fastgs_big_baseline/mipnerf360_gpu0/bicycle/logs/fastgs_big_densify100_30k_r_auto/train.log` | 7190/30000 | `src/vfm_gs/cli/train.py:362`，`loss.item()` | `torch.AcceleratorError: CUDA error: an illegal memory access was encountered` | 无指标产出；暂停全数据集 baseline |
| MipNeRF360 | room | 1 | `output/0002/phase0_5090_fastgs_big_baseline/mipnerf360_gpu1/room/logs/fastgs_big_densify100_30k_r_auto/train.log` | 2200/30000 | `src/vfm_gs/gaussian_renderer/__init__.py:108`，`(radii > 0).nonzero()` | `torch.AcceleratorError: CUDA error: an illegal memory access was encountered` | 无指标产出；暂停全数据集 baseline |

两份日志都确认输入使用 FastGS 原始高分辨率口径：遇到宽度大于 1.6K 的原图后自动缩放到 1.6K。两张 5090 在不同场景、不同调用点都出现 illegal memory access，优先怀疑 CUDA 异步报错掩盖了更早的 kernel 问题，或本地 CUDA extension 尚未针对当前 PyTorch/CUDA/Blackwell 环境稳定编译。

后续环境检查：

| 日期 | 检查 | 结果 | 结论 |
|---|---|---|---|
| 2026-05-11 | PyTorch / GPU | PyTorch `2.10.0+cu128`，CUDA runtime `12.8`，两张 RTX 5090 capability `(12, 0)` | Python 环境识别 Blackwell 正常 |
| 2026-05-11 | 本地 CUDA extension | `diff_gaussian_rasterization_fastgs` / `simple_knn` / `fused_ssim` 均加载自 `.venv`；rasterizer `.so` 字符串包含 `Cuda compilation tools, release 12.8` 与 `-arch sm_120` | 扩展不是旧 Ada/Ampere binary，但仍可能存在异步 kernel 稳定性问题 |
| 2026-05-11 | 同步 CUDA 短复现 | `CUDA_LAUNCH_BLOCKING=1`，MipNeRF360 `room`，2500 step high-res FastGS big，输出 `output/0002/debug_5090_illegal_room_2500_r_auto_blocking` | 完成训练并保存点云，未复现 illegal memory access；下一步用 blocking 口径复跑 Phase 0 |
| 2026-05-11 | blocking Phase 0 复跑 | `CUDA_LAUNCH_BLOCKING=1`，MipNeRF360 `bicycle` / `room`，输出 `output/0002/phase0_5090_fastgs_big_baseline_blocking/` | 仍失败；`bicycle` 在 9300/30000、`room` 在 11450/30000，于 `render_fastgs` 读取 `radii` 时触发 CUDA illegal memory access；无 render/metrics |
| 2026-05-11 | rasterizer debug 复现 | `CUDA_LAUNCH_BLOCKING=1 --debug_from 9000`，MipNeRF360 `bicycle` 9500 step，输出 `output/0002/debug_5090_illegal_bicycle_9500_debug_r_auto` | 在 3680/9500 提前失败，仍停在 `render_fastgs` 的 `radii` 访问；`debug_from=9000` 未生效，无 `snapshot_fw.dump` |
| 2026-05-11 | rasterizer debug wrapper 检查 | `CUDA_LAUNCH_BLOCKING=1 --debug_from 0`，MipNeRF360 `bicycle` 5000 step，输出 `output/0002/debug_5090_illegal_bicycle_5000_debug0_r_auto` | 第一步即触发 Python wrapper 错误：debug 分支按 7 项解包，但当前 `_C.rasterize_gaussians` 返回 9 项；已修补源码和当前 `.venv` 安装副本，并将旧 snapshot 移到 `output/0002/debug_artifacts/snapshot_fw_debug_wrapper_mismatch_20260511.dump` |
| 2026-05-11 | rasterizer debug kernel 定位 | 修补 wrapper 后重跑 `CUDA_LAUNCH_BLOCKING=1 --debug_from 0`，MipNeRF360 `bicycle` 5000 step，输出 `output/0002/debug_5090_illegal_bicycle_5000_debug0_fixed_wrapper_r_auto` | 在 1220/5000 失败；`submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/rasterizer_impl.cu:422` 报 `operation not supported on global/shared address space`，位置紧跟 `identifyTileRanges` kernel；snapshot 已归档到 `output/0002/debug_artifacts/snapshot_fw_identify_tile_ranges_global_shared_20260511.dump` |
| 2026-05-11 | rasterizer fix1 + debug 验证 | 修补无效/越界 tile key 保护、低透明度/NaN 椭圆保护、`<cstdint>` 编译兼容，并用 CUDA 12.8 / `sm_120` 重编译安装；`CUDA_LAUNCH_BLOCKING=1 --debug_from 0`，MipNeRF360 `bicycle` 5000 step，输出 `output/0002/debug_5090_fix1_bicycle_5000_debug0_r_auto` | 完成 5000/5000，保存 1,340,808 个 Gaussians，训练时间 1350.99s；未再触发 `identifyTileRanges` / CUDA illegal memory access；下一步跑正常 30k baseline 验收 |
| 2026-05-11 | rasterizer fix1 + 30k 单场景验收 | 修补后正常模式，MipNeRF360 `bicycle` high-res FastGS big，输出 `output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_single_gpu0` | train/render/metrics 完成；25.2569 PSNR / 0.7553 SSIM / 0.2450 LPIPS，1,560,209 Gaussians，训练 159.11s；相对 0001 记录的 25.2532 / 0.7554 / 0.2446、1,560,079 点基本贴合 | 单场景 smoke 通过；恢复双卡全场景 Phase 0 |
| 2026-05-11 | rasterizer fix1 + MipNeRF360 全场景验收 | `setsid/nohup` detached 方式补跑，输出 `output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_{single_gpu0,gpu0,gpu1,combined}` | 9/9 场景完成 train/render/metrics；均值 27.9590 / 0.8203 / 0.2157，1,161,786 Gaussians，训练 159.70s；与 0001 同口径 27.9293 / 0.8198 / 0.2157、1,161,242 点基本一致 | MipNeRF360 Phase 0 通过；继续 DB/Tandt baseline |
| 2026-05-11 | rasterizer fix1 + DB/Tandt 全场景验收 | `setsid/nohup` detached 方式双卡并行，输出 `output/0002/phase0_5090_fastgs_big_baseline_fix1/{db,tandt}` | DB 2/2 完成，均值 30.2331 / 0.9111 / 0.2397，646,600 Gaussians；Tandt 2/2 完成，均值 24.4955 / 0.8579 / 0.1736，540,119 Gaussians；无 CUDA/PLY 错误 | Phase 0 全部通过；可启动 Depth Anything |

### 2026-05-11 MipNeRF360 fix1 全场景结果

统一汇总路径：`output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_combined/summary.csv`。

| 场景 | PSNR | SSIM | LPIPS | Gaussian 数 | 训练时间 |
|---|---:|---:|---:|---:|---:|
| bicycle | 25.2569 | 0.7553 | 0.2450 | 1,560,209 | 159.11s |
| flowers | 21.6284 | 0.6023 | 0.3404 | 1,122,815 | 145.69s |
| garden | 27.6346 | 0.8644 | 0.1096 | 2,634,816 | 273.13s |
| stump | 27.1784 | 0.7868 | 0.2393 | 1,064,860 | 131.01s |
| treehill | 22.8275 | 0.6323 | 0.3770 | 1,009,110 | 127.40s |
| room | 32.2136 | 0.9304 | 0.1882 | 571,607 | 113.52s |
| counter | 29.5259 | 0.9180 | 0.1766 | 470,577 | 122.00s |
| kitchen | 32.2810 | 0.9390 | 0.1051 | 1,177,988 | 219.85s |
| bonsai | 33.0846 | 0.9538 | 0.1598 | 844,093 | 145.59s |
| **平均** | **27.9590** | **0.8203** | **0.2157** | **1,161,786** | **159.70s** |

对照 0001 high-res FastGS big baseline 平均 27.9293 / 0.8198 / 0.2157、1,161,242 Gaussians。5090 fix1 的质量和容量均无系统性漂移；训练时间下降属于硬件差异，不作为质量结论。

注意：第一次非 detached 补跑期间，`treehill` 产出的 PLY 在 render 时触发 `early end-of-file`，`bonsai` 也停在半程训练目录。为保留证据，两个不完整目录已移动到 `output/0002/debug_artifacts/interrupted_mipnerf360_20260511_200949/`，随后用 `setsid/nohup` 重新补跑通过。后续长实验统一使用 detached 方式，避免 SSH 断开导致产物截断。

### 2026-05-11 DB/Tandt fix1 全场景结果

汇总路径：

- DB：`output/0002/phase0_5090_fastgs_big_baseline_fix1/db/summary.csv`
- Tandt：`output/0002/phase0_5090_fastgs_big_baseline_fix1/tandt/summary.csv`
- 全 13 场景：`output/0002/phase0_5090_fastgs_big_baseline_fix1/all_combined/summary.csv`

| 数据集 | 场景 | PSNR | SSIM | LPIPS | Gaussian 数 | 训练时间 |
|---|---|---:|---:|---:|---:|---:|
| DB | drjohnson | 29.7481 | 0.9074 | 0.2437 | 703,946 | 100.99s |
| DB | playroom | 30.7181 | 0.9148 | 0.2357 | 589,253 | 96.09s |
| DB | **平均** | **30.2331** | **0.9111** | **0.2397** | **646,600** | **98.54s** |
| Tandt | train | 22.8911 | 0.8263 | 0.2087 | 454,435 | 102.01s |
| Tandt | truck | 26.0998 | 0.8896 | 0.1385 | 625,803 | 108.74s |
| Tandt | **平均** | **24.4955** | **0.8579** | **0.1736** | **540,119** | **105.37s** |

对照 0001 high-res FastGS big baseline：DB 为 30.2073 / 0.9112 / 0.2402、650,194 Gaussians；Tandt 为 24.3557 / 0.8573 / 0.1745、540,578 Gaussians。两组都无系统性质量或容量漂移。

## Depth Anything High-Res Pilot

### 2026-05-11 Depth Anything V2-S depth-edge cache/backend smoke

实现与环境：

- `src/vfm_gs/cli/build_vfm_cache.py` 新增 `depth_anything` / `depth_anything_v2` cache builder，默认模型为 `depth-anything/Depth-Anything-V2-Small-hf`。
- `src/vfm_gs/scorers/vfm_topology.py` 新增 `depth_anything_depth_edge_prior` 和 `depth_anything_depth_prior` backend；本轮只验证 depth-edge prior。
- 新增配置：`configs/experiments/0002_depth_anything_depth_edge_prior_densify_only_topk025_weighted_i050.yaml`。
- 安装 optional 依赖：`transformers==5.8.0`、`huggingface-hub==1.14.0`、`safetensors==0.7.0`。
- HuggingFace 默认 Xet 下载在本机出现 `RemoteProtocolError: Server disconnected without sending a response`；设置 `HF_HUB_DISABLE_XET=1` 后模型下载和加载成功。
- 全量 cache：`output/0002/vfm_cache/bicycle_depth_anything_v2s_edge`，194 entries，48MB，`max_width=1600`，feature 为 `depth_anything_depth_edge`，validate 通过。

Smoke 与 matched baseline：

| 方法 | 配置 | PSNR | SSIM | LPIPS | Gaussian 数 | 训练时间 | 输出路径 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|---|
| FastGS matched 620 | `fastgs_baseline` + high-res + densify100 | 19.4930 | 0.4046 | 0.6268 | 61,278 | 1.81s | `output/0002/fastgs_baseline_bicycle_620_r_auto` | 620-step 参照线 |
| Depth Anything V2-S depth-edge prior 620 | top-k25 weighted i0.50, `vfm_weight=0.0` | 19.4402 | 0.4039 | 0.6270 | 61,277 | 1.90s | `output/0002/depth_anything_edge_prior_bicycle_620_r_auto` | 链路通过；短程指标略低于 matched baseline |

日志：

- cache build：`output/0002/debug_logs/depth_anything_bicycle_cache_build.log`
- train：`output/0002/debug_logs/depth_anything_edge_prior_bicycle_620_train.log`
- render：`output/0002/debug_logs/depth_anything_edge_prior_bicycle_620_render.log`
- metrics：`output/0002/debug_logs/depth_anything_edge_prior_bicycle_620_metrics.log`

判断：`depth_anything_depth_edge_prior` 能复用现有 `pixel prior -> top-k metric_map -> accum_metric_counts -> densification importance` 主链路，cache preflight 也能在训练前校验 backend/feature。620-step 指标本身不支持质量结论；下一步可以跑 high-res `bicycle` 30k pilot，并与 Phase 0 FastGS big baseline、0001 high-res DINO descriptor weighted i0.50/i0.70 对照。

### 2026-05-11 Depth Anything V2-S depth-edge 30k bicycle pilot

本轮使用与 620-step smoke 相同的 `fastgs_baseline + densification_interval=100` recipe，保持 high-res `-r -1` / 1.6K 自动缩放口径。为避免把 recipe 差异误读成方法差异，补跑了同配方 30k matched baseline；Phase 0 `fastgs_big` bicycle 只作为 5090 环境上限参考。

| 方法 | Recipe | PSNR | SSIM | LPIPS | Gaussian 数 | 训练时间 | 输出路径 |
|---|---|---:|---:|---:|---:|---:|---|
| FastGS matched 30k | `fastgs_baseline + densify100` | 25.0787 | 0.7370 | 0.2779 | 1,023,912 | 124.78s | `output/0002/fastgs_baseline_bicycle_30k_densify100_r_auto` |
| Depth Anything V2-S depth-edge prior 30k | top-k25 weighted i0.50, `vfm_weight=0.0` | 25.0764 | 0.7387 | 0.2744 | 1,063,311 | 131.21s | `output/0002/depth_anything_edge_prior_bicycle_30k_r_auto` |
| Phase 0 FastGS big bicycle | `fastgs_big + densify100 + scene overrides` | 25.2569 | 0.7553 | 0.2450 | 1,560,209 | 159.11s | `output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_single_gpu0/bicycle` |

相对 matched baseline：Depth Anything edge-prior 为 -0.0023 PSNR、+0.0017 SSIM、LPIPS -0.0035，Gaussian 数 +39,399，QCGI 约 +0.010。它说明 dense depth-edge prior 有一点结构性信号，但收益很弱，且 PSNR 没有转正。

相对 Phase 0 FastGS big：Depth Anything edge-prior 为 -0.1805 PSNR、-0.0166 SSIM、LPIPS +0.0294，Gaussian 数 -496,898。该比较混入了 recipe 差异，不能作为 depth-edge prior 的公平负例；但它提醒 0002 若要对齐用户要求的 big baseline，后续应 either 使用 `fastgs_big` recipe 接入 Depth Anything，or 在每个 pilot 中补齐 matched baseline。

日志：

- Depth Anything train：`output/0002/debug_logs/depth_anything_edge_prior_bicycle_30k_train.log`
- Depth Anything render：`output/0002/debug_logs/depth_anything_edge_prior_bicycle_30k_render.log`
- Depth Anything metrics：`output/0002/debug_logs/depth_anything_edge_prior_bicycle_30k_metrics.log`
- matched baseline train：`output/0002/debug_logs/fastgs_baseline_bicycle_30k_densify100_train.log`
- matched baseline render：`output/0002/debug_logs/fastgs_baseline_bicycle_30k_densify100_render.log`
- matched baseline metrics：`output/0002/debug_logs/fastgs_baseline_bicycle_30k_densify100_metrics.log`

判断：暂不扩展 depth-edge prior 到全数据集。下一轮优先跑 `depth_anything_depth_prior`，检验直接 relative depth map 是否比 edge map 更适合作为 densification prior；若 direct depth 仍只有弱混合信号，再考虑 `fastgs_big` recipe 下的 matched 接入或在线 render-depth residual。

### 2026-05-11 Depth Anything V2-S direct depth 30k bicycle pilot

本轮构建 direct relative depth cache：`output/0002/vfm_cache/bicycle_depth_anything_v2s_depth`，194 entries，46.21MB，feature 为 `depth_anything_relative_depth`，validate 通过。训练 recipe 与 depth-edge pilot 和 matched baseline 保持一致：`fastgs_baseline + densification_interval=100`，high-res `-r -1` / 1.6K 自动缩放，`vfm_weight=0.0`，top-k25 weighted i0.50，只影响 densification。

| 方法 | Recipe | PSNR | SSIM | LPIPS | Gaussian 数 | 训练时间 | 输出路径 |
|---|---|---:|---:|---:|---:|---:|---|
| FastGS matched 30k | `fastgs_baseline + densify100` | 25.0787 | 0.7370 | 0.2779 | 1,023,912 | 124.78s | `output/0002/fastgs_baseline_bicycle_30k_densify100_r_auto` |
| Depth Anything V2-S depth-edge prior 30k | top-k25 weighted i0.50, `vfm_weight=0.0` | 25.0764 | 0.7387 | 0.2744 | 1,063,311 | 131.21s | `output/0002/depth_anything_edge_prior_bicycle_30k_r_auto` |
| Depth Anything V2-S direct depth prior 30k | top-k25 weighted i0.50, `vfm_weight=0.0` | 25.1415 | 0.7434 | 0.2689 | 1,005,953 | 128.82s | `output/0002/depth_anything_depth_prior_bicycle_30k_r_auto` |

相对 matched baseline：direct depth prior 为 +0.0628 PSNR、+0.0063 SSIM、LPIPS -0.0090，Gaussian 数 -17,959，QCGI 约 +0.2345。它比 depth-edge prior 更干净：质量三项全正向，同时没有增加点数。

相对 Phase 0 FastGS big bicycle：direct depth prior 仍为 -0.1154 PSNR、-0.0119 SSIM、LPIPS +0.0239，Gaussian 数 -554,256。该差距仍受 recipe 差异影响；当前不能把它解释为 direct depth prior 相对 FastGS big 的最终结论。下一步 pilot 仍采用 matched `fastgs_baseline + densify100` 对照推进，多场景成立后再补 `fastgs_big` recipe 接入。

日志：

- cache build：`output/0002/debug_logs/depth_anything_bicycle_depth_cache_build.log`
- cache validate：`output/0002/debug_logs/depth_anything_bicycle_depth_cache_validate.log`
- train：`output/0002/debug_logs/depth_anything_depth_prior_bicycle_30k_train.log`
- render：`output/0002/debug_logs/depth_anything_depth_prior_bicycle_30k_render.log`
- metrics：`output/0002/debug_logs/depth_anything_depth_prior_bicycle_30k_metrics.log`

判断：`depth_anything_depth_prior` 成为 0002 当前主线。下一步扩到 `stump/bonsai/playroom/truck` 四个 pilot 场景，先验证 direct depth 是否跨场景成立；如果至少多数场景相对 matched baseline 正向，再进入三个公开数据集全场景训练验证。

### 2026-05-14 Depth Anything V2-S direct depth fastgs_big stump/bonsai pilot

本轮目标是把 `depth_anything_depth_prior` 从 `fastgs_baseline + densify100` matched recipe 推到用户要求的 `fastgs_big + 1.6K` 正式口径。这里有一个重要修正：第一次直接用 `--variant fastgs_big` 调度 `stump/bonsai` 时没有带上 phase 0 runner 的 per-scene overrides，因此该结果只作为无效诊断，不进入方法结论。有效复验必须显式对齐：

- `stump`: `--dense 0.004 --grad_abs_thresh 0.001`
- `bonsai`: `--highfeature_lr 0.02 --grad_abs_thresh 0.0002`

Cache 状态：

| 场景 | cache | entries | 体积 | validate |
|---|---|---:|---:|---|
| stump | `output/0002/vfm_cache/stump_depth_anything_v2s_depth` | 125 | 31MB | 通过 |
| bonsai | `output/0002/vfm_cache/bonsai_depth_anything_v2s_depth` | 292 | 55MB | 通过 |

主结果：

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数 | 训练时间 | 相对 phase 0 | QCGI | 输出路径 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| stump | Phase 0 FastGS big | 27.1784 | 0.7868 | 0.2393 | 1,064,860 | 131.01s | - | - | `output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_gpu0/stump/fastgs_big_densify100_30k_r_auto` |
| stump | Depth Anything direct depth prior, scene override | 27.1939 | 0.7895 | 0.2321 | 1,086,229 | 137.45s | +0.0155 / +0.0027 / -0.0072, GS +21,369 | +0.0835 | `output/0002/depth_anything_depth_prior_fastgs_big_stump_30k_scene_override_r_auto` |
| bonsai | Phase 0 FastGS big | 33.0846 | 0.9538 | 0.1598 | 844,093 | 145.59s | - | - | `output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_gpu1/bonsai/fastgs_big_densify100_30k_r_auto` |
| bonsai | Depth Anything direct depth prior, scene override | 33.0734 | 0.9546 | 0.1564 | 931,574 | 159.83s | -0.0113 / +0.0008 / -0.0034, GS +87,481 | -0.0651 | `output/0002/depth_anything_depth_prior_fastgs_big_bonsai_30k_scene_override_r_auto` |

无效诊断记录：

| 场景 | 无效原因 | PSNR | SSIM | LPIPS | Gaussian 数 | 输出路径 |
|---|---|---:|---:|---:|---:|---|
| stump | 未带 phase 0 `stump` overrides；`dense=0.001`、`grad_abs_thresh=0.0008` 与 baseline 不一致 | 27.0123 | 0.7805 | 0.2424 | 1,038,269 | `output/0002/depth_anything_depth_prior_fastgs_big_stump_30k_r_auto` |
| bonsai | 未带 phase 0 `bonsai` overrides；`highfeature_lr=0.005`、`grad_abs_thresh=0.0008` 与 baseline 不一致 | 31.8038 | 0.9438 | 0.1847 | 341,303 | `output/0002/depth_anything_depth_prior_fastgs_big_bonsai_30k_r_auto` |

判断：对齐 scene overrides 后，`stump` 从“负结果”恢复为小幅正向，说明上一轮失败主要混入了 recipe mismatch。`bonsai` 不是清晰正例：PSNR 微负，SSIM/LPIPS 正向，点数增加 87,481，QCGI 为负。当前不能直接扩全数据集；下一步应先补 `playroom/truck` 两个跨数据集场景，且每个场景必须同时报告 overlap 诊断。

### 2026-05-14 fastgs_big stump/bonsai prior-overlap 诊断

诊断对象为对齐 scene overrides 后的 `stump/bonsai` direct depth prior。输出：

- `output/0002/diagnostics/stump_depth_prior_fastgs_big_scene_override_overlap`
- `output/0002/diagnostics/stump_depth_prior_fastgs_big_scene_override_overlap_topk10`
- `output/0002/diagnostics/bonsai_depth_prior_fastgs_big_scene_override_overlap`
- `output/0002/diagnostics/bonsai_depth_prior_fastgs_big_scene_override_overlap_topk10`

Top-k 25%：

| 场景 | baseline L1 | prior top-k L1 | non-prior L1 | prior/RGB IoU | prior/RGB recall | candidate ΔL1 | prior top-k ΔL1 | non-prior ΔL1 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stump | 0.030519 | 0.034971 | 0.029035 | 0.1790 | 0.3009 | -0.000144 | -0.000280 | -0.000099 | prior 区域更难，改善更集中在 prior top-k |
| bonsai | 0.012907 | 0.016605 | 0.011675 | 0.2030 | 0.3328 | +0.000117 | +0.000106 | +0.000121 | prior 区域更难，但 RGB L1 没改善；全图正向主要来自 SSIM/LPIPS |

Top-k 10%：

| 场景 | prior top-k L1 | non-prior L1 | prior/RGB IoU | prior/RGB recall | candidate ΔL1 | prior top-k ΔL1 | non-prior ΔL1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| stump | 0.037759 | 0.029715 | 0.0818 | 0.1502 | -0.000144 | -0.000256 | -0.000132 |
| bonsai | 0.016663 | 0.012490 | 0.0978 | 0.1733 | +0.000117 | +0.000043 | +0.000125 |

判断：

- Depth Anything direct depth top-k 在两个场景上都覆盖更难区域，但与 RGB 高误差区域的 IoU 仍不高；它更像“几何难区 prior”，不是直接的 RGB loss surrogate。
- `stump` 的质量正向与 L1 诊断一致，prior top-k 区域改善更大。
- `bonsai` 的 SSIM/LPIPS 改善没有对应到 L1 改善，且点数增加明显；应作为边界样本，而不是 0002 成功证据。

### 2026-05-14 Depth Anything V2-S direct depth playroom/truck pilot

本轮补齐跨数据集 pilot，并继续使用 `fastgs_big + scene overrides`：

- `playroom`: `--highfeature_lr 0.0015 --dense 0.003 --mult 0.7 --grad_abs_thresh 0.0005`
- `truck`: `--highfeature_lr 0.04 --grad_abs_thresh 0.0004 --mult 0.7`

Cache 状态：

| 场景 | cache | entries | validate |
|---|---|---:|---|
| playroom | `output/0002/vfm_cache/playroom_depth_anything_v2s_depth` | 225 | 通过 |
| truck | `output/0002/vfm_cache/truck_depth_anything_v2s_depth` | 251 | 通过 |

主结果：

| 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数 | 训练时间 | 相对 phase 0 | QCGI | 输出路径 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| playroom | Phase 0 FastGS big | 30.7181 | 0.9148 | 0.2357 | 589,253 | 96.09s | - | - | `output/0002/phase0_5090_fastgs_big_baseline_fix1/db/playroom/fastgs_big_densify100_30k_r_auto` |
| playroom | Depth Anything direct depth prior, scene override | 30.8242 | 0.9144 | 0.2331 | 606,454 | 100.72s | +0.1061 / -0.0004 / -0.0026, GS +17,201 | +0.0936 | `output/0002/depth_anything_depth_prior_fastgs_big_playroom_30k_scene_override_r_auto` |
| truck | Phase 0 FastGS big | 26.0998 | 0.8896 | 0.1385 | 625,803 | 108.74s | - | - | `output/0002/phase0_5090_fastgs_big_baseline_fix1/tandt/truck/fastgs_big_densify100_30k_r_auto` |
| truck | Depth Anything direct depth prior, scene override | 25.9496 | 0.8870 | 0.1405 | 523,809 | 107.51s | -0.1502 / -0.0026 / +0.0021, GS -101,994 | -0.2124 | `output/0002/depth_anything_depth_prior_fastgs_big_truck_30k_scene_override_r_auto` |

Overlap 诊断：

| 场景 | top-k | baseline L1 | prior top-k L1 | non-prior L1 | prior/RGB IoU | prior/RGB recall | candidate ΔL1 | prior top-k ΔL1 | non-prior ΔL1 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| playroom | 25% | 0.019299 | 0.024015 | 0.017727 | 0.2149 | 0.3473 | -0.000150 | -0.000093 | -0.000168 | prior 区域更难，但 L1 改善更偏非-prior |
| playroom | 10% | 0.019299 | 0.027192 | 0.018422 | 0.1132 | 0.1987 | -0.000150 | -0.000071 | -0.000158 | top-10 同样不是改善集中区 |
| truck | 25% | 0.028265 | 0.022478 | 0.030195 | 0.1108 | 0.1970 | +0.000730 | +0.000587 | +0.000777 | prior 区域反而更容易，全图变差 |
| truck | 10% | 0.028265 | 0.024621 | 0.028670 | 0.0391 | 0.0741 | +0.000730 | +0.000946 | +0.000706 | prior/RGB 错位严重，prior top-k 变差更大 |

判断：`playroom` 是可保留的跨数据集薄正例，但它的 L1 改善没有集中在 Depth Anything top-k 区域，收益可能来自 densification 轨迹扰动而非精确命中几何 prior。`truck` 是明确负例：Depth Anything top-k 甚至不是 RGB 高误差区，top-10 IoU 只有 0.039，candidate 还在 prior top-k 区域变差更多。至此，direct depth prior 在 `bicycle/stump/playroom` 上有信号，在 `bonsai/truck` 上不稳或负向；不建议进入全数据集训练验证。

### 2026-05-17 RGB-gated depth rerank final-topm l0.25 pilot

本轮把 direct depth prior 改成 `RGB broad candidate -> Depth Anything rerank -> final-topm capacity lock`。配置为 `configs/experiments/0002_depth_anything_depth_prior_rgb_rerank_final_topm_l025.yaml`，仍使用 `fastgs_big + scene overrides`、Depth Anything relative depth top-k25、RGB broad top-50%，`vfm_dino_rerank_lambda=0.25`。这里复用已有 `rgb_rerank_final_topm` 机制；变量名里仍叫 `dino_rerank_lambda`，但本轮实际 rerank 分数来自 Depth Anything prior count。

620-step `truck` smoke 已通过 cache preflight/train/save，输出 `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l025_fastgs_big_truck_620_scene_override_r_auto`，只作为链路健康检查。

正式三场景结果：

| 场景 | Phase 0 FastGS big | RGB-gated depth rerank final-topm l0.25 | ΔPSNR / ΔSSIM / ΔLPIPS / ΔGS | QCGI | 输出 |
|---|---:|---:|---:|---:|---|
| stump | 27.1784 / 0.7868 / 0.2393, 1,064,860 GS | 27.1267 / 0.7922 / 0.2265, 1,388,160 GS, 154.22s | -0.0517 / +0.0054 / -0.0128 / +323,300 | -0.8724 | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l025_fastgs_big_stump_30k_scene_override_r_auto` |
| playroom | 30.7181 / 0.9148 / 0.2357, 589,253 GS | 30.6396 / 0.9143 / 0.2329, 746,231 GS, 113.16s | -0.0785 / -0.0005 / -0.0028 / +156,978 | -0.4021 | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l025_fastgs_big_playroom_30k_scene_override_r_auto` |
| truck | 26.0998 / 0.8896 / 0.1385, 625,803 GS | 26.2003 / 0.8918 / 0.1317, 845,866 GS, 129.50s | +0.1005 / +0.0022 / -0.0068 / +220,063 | -0.4017 | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l025_fastgs_big_truck_30k_scene_override_r_auto` |

Top-k 25% prior-overlap 诊断：

| 场景 | baseline L1 | prior top-k L1 | non-prior L1 | prior/RGB IoU | prior/RGB recall | candidate ΔL1 | prior top-k ΔL1 | non-prior ΔL1 | 输出 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stump | 0.030519 | 0.034971 | 0.029035 | 0.1790 | 0.3009 | +0.000520 | -0.000256 | +0.000778 | `output/0002/diagnostics/stump_depth_prior_rgb_rerank_final_topm_l025_overlap` |
| playroom | 0.019299 | 0.024015 | 0.017727 | 0.2149 | 0.3473 | +0.000208 | +0.000660 | +0.000058 | `output/0002/diagnostics/playroom_depth_prior_rgb_rerank_final_topm_l025_overlap` |
| truck | 0.028265 | 0.022478 | 0.030195 | 0.1108 | 0.1970 | -0.000399 | -0.000357 | -0.000413 | `output/0002/diagnostics/truck_depth_prior_rgb_rerank_final_topm_l025_overlap` |

Top-k 10% prior-overlap 诊断：

| 场景 | prior top-k L1 | non-prior L1 | prior/RGB IoU | prior/RGB recall | candidate ΔL1 | prior top-k ΔL1 | non-prior ΔL1 | 输出 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| stump | 0.037759 | 0.029715 | 0.0818 | 0.1502 | +0.000520 | -0.000171 | +0.000597 | `output/0002/diagnostics/stump_depth_prior_rgb_rerank_final_topm_l025_overlap_topk10` |
| playroom | 0.027192 | 0.018422 | 0.1132 | 0.1987 | +0.000208 | +0.001019 | +0.000118 | `output/0002/diagnostics/playroom_depth_prior_rgb_rerank_final_topm_l025_overlap_topk10` |
| truck | 0.024621 | 0.028670 | 0.0391 | 0.0741 | -0.000399 | -0.000377 | -0.000402 | `output/0002/diagnostics/truck_depth_prior_rgb_rerank_final_topm_l025_overlap_topk10` |

判断：RGB-gated rerank 确实修复了 `truck` 的 direct depth 错位：全图指标三项正向，L1 也从 direct depth 的 +0.000730 变为 -0.000399。但它并没有成为稳定方案。`stump` 的 prior top-k 区域改善仍在，但非 prior 区域变差更大，导致 PSNR 和全图 L1 退化；`playroom` 的 prior top-k 区域明显变差，说明 l0.25 rerank 改坏了原本的薄正例。三场景 Gaussian 都明显增加，QCGI 全负。下一轮应优先把 rerank 权重降到 l0.10，必要时再缩小 RGB broad top-k 或延后 `vfm_active_from_iter`，目标是保留 `truck` 修复而不过度改写 `stump/playroom` 的增长轨迹。

### 2026-05-17 RGB-gated depth rerank final-topm l0.10 pilot

这一轮把同一机制收紧到 `vfm_dino_rerank_lambda=0.10`。结果说明，降低权重确实把 `stump` 拉回到三项质量正向，也让 `truck` 维持三项质量正向，但对 `playroom` 还不够，且容量代价仍然太高。

正式三场景结果：

| 场景 | Phase 0 FastGS big | RGB-gated depth rerank final-topm l0.10 | ΔPSNR / ΔSSIM / ΔLPIPS / ΔGS | QCGI | 输出 |
|---|---:|---:|---:|---:|---|
| stump | 27.1784 / 0.7868 / 0.2393, 1,064,860 GS | 27.2355 / 0.7923 / 0.2258, 1,387,811 GS, 153.56s | +0.0571 / +0.0055 / -0.0135 / +322,951 | -0.7581 | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l010_fastgs_big_stump_30k_scene_override_r_auto` |
| playroom | 30.7181 / 0.9148 / 0.2357, 589,253 GS | 30.5282 / 0.9142 / 0.2326, 746,312 GS, 113.08s | -0.1899 / -0.0006 / -0.0031 / +157,059 | -0.5136 | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l010_fastgs_big_playroom_30k_scene_override_r_auto` |
| truck | 26.0998 / 0.8896 / 0.1385, 625,803 GS | 26.1558 / 0.8916 / 0.1313, 842,083 GS, 129.17s | +0.0560 / +0.0020 / -0.0072 / +216,280 | -0.4325 | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l010_fastgs_big_truck_30k_scene_override_r_auto` |

Top-k 25% prior-overlap 诊断：

| 场景 | baseline L1 | prior top-k L1 | non-prior L1 | prior/RGB IoU | prior/RGB recall | candidate ΔL1 | prior top-k ΔL1 | non-prior ΔL1 | 输出 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stump | 0.030519 | 0.034971 | 0.029035 | 0.1790 | 0.3009 | -0.000320 | -0.000480 | -0.000267 | `output/0002/diagnostics/stump_depth_prior_rgb_rerank_final_topm_l010_overlap` |
| playroom | 0.019299 | 0.024015 | 0.017727 | 0.2149 | 0.3473 | +0.000529 | +0.000431 | +0.000561 | `output/0002/diagnostics/playroom_depth_prior_rgb_rerank_final_topm_l010_overlap` |
| truck | 0.028265 | 0.022478 | 0.030195 | 0.1108 | 0.1970 | -0.000284 | -0.000087 | -0.000349 | `output/0002/diagnostics/truck_depth_prior_rgb_rerank_final_topm_l010_overlap` |

Top-k 10% prior-overlap 诊断：

| 场景 | prior top-k L1 | non-prior L1 | prior/RGB IoU | prior/RGB recall | candidate ΔL1 | prior top-k ΔL1 | non-prior ΔL1 | 输出 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| stump | 0.037759 | 0.029715 | 0.0818 | 0.1502 | -0.000320 | -0.000394 | -0.000312 | `output/0002/diagnostics/stump_depth_prior_rgb_rerank_final_topm_l010_overlap_topk10` |
| playroom | 0.027192 | 0.018422 | 0.1132 | 0.1987 | +0.000529 | +0.000150 | +0.000571 | `output/0002/diagnostics/playroom_depth_prior_rgb_rerank_final_topm_l010_overlap_topk10` |
| truck | 0.024621 | 0.028670 | 0.0391 | 0.0741 | -0.000284 | +0.000241 | -0.000342 | `output/0002/diagnostics/truck_depth_prior_rgb_rerank_final_topm_l010_overlap_topk10` |

判断：`lambda 0.10` 比 `0.25` 更像一个可用边界上的收缩版。`stump` 和 `truck` 的全图质量都回到了三项正向，但 `playroom` 仍然 PSNR 负向，且三场景 QCGI 依旧为负，说明容量代价没有被真正压住。`truck` 的 prior top-10 仍只有 0.039 的 IoU，错位问题没有消失，只是方向没有再被放大。下一轮如果继续，应该再试 `0.05`，或者保持 `0.10` 但缩小 `vfm_rgb_broad_topk`。

### 2026-05-17 RGB-gated depth rerank final-topm l0.05 pilot

`lambda 0.05` 进一步收缩后，`stump` 依旧保住三项质量正向，`playroom` 也从 l0.10 的 PSNR 负向翻回正向，但 `truck` 再次掉回 PSNR 负向。三场景 QCGI 仍然全负，说明单纯继续降 lambda 已经进入收益递减区，错位主要还在 broad candidate 入口，而不是 rerank 强度本身。

正式三场景结果：

| 场景 | Phase 0 FastGS big | RGB-gated depth rerank final-topm l0.05 | ΔPSNR / ΔSSIM / ΔLPIPS / ΔGS | QCGI | 输出 |
|---|---:|---:|---:|---:|---|
| stump | 27.1784 / 0.7868 / 0.2393, 1,064,860 GS | 27.2379 / 0.7923 / 0.2258, 1,388,153 GS, 153.55s | +0.0595 / +0.0055 / -0.0135 / +323,293 | -0.7570 | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l005_fastgs_big_stump_30k_scene_override_r_auto` |
| playroom | 30.7181 / 0.9148 / 0.2357, 589,253 GS | 30.8380 / 0.9147 / 0.2309, 746,168 GS, 112.77s | +0.1199 / -0.0001 / -0.0048 / +156,915 | -0.1863 | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l005_fastgs_big_playroom_30k_scene_override_r_auto` |
| truck | 26.0998 / 0.8896 / 0.1385, 625,803 GS | 26.0655 / 0.8900 / 0.1328, 850,023 GS, 129.97s | -0.0343 / +0.0004 / -0.0057 / +224,220 | -0.5939 | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l005_fastgs_big_truck_30k_scene_override_r_auto` |

Top-k 25% prior-overlap 诊断：

| 场景 | baseline L1 | prior top-k L1 | non-prior L1 | prior/RGB IoU | prior/RGB recall | candidate ΔL1 | prior top-k ΔL1 | non-prior ΔL1 | 输出 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stump | 0.030519 | 0.034971 | 0.029035 | 0.1790 | 0.3009 | -0.000386 | -0.000441 | -0.000368 | `output/0002/diagnostics/stump_depth_prior_rgb_rerank_final_topm_l005_overlap` |
| playroom | 0.019299 | 0.024015 | 0.017727 | 0.2149 | 0.3473 | -0.000179 | +0.000205 | -0.000306 | `output/0002/diagnostics/playroom_depth_prior_rgb_rerank_final_topm_l005_overlap` |
| truck | 0.028265 | 0.022478 | 0.030195 | 0.1108 | 0.1970 | +0.000352 | +0.000651 | +0.000252 | `output/0002/diagnostics/truck_depth_prior_rgb_rerank_final_topm_l005_overlap` |

Top-k 10% prior-overlap 诊断：

| 场景 | prior top-k L1 | non-prior L1 | prior/RGB IoU | prior/RGB recall | candidate ΔL1 | prior top-k ΔL1 | non-prior ΔL1 | 输出 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| stump | 0.037759 | 0.029715 | 0.0818 | 0.1502 | -0.000386 | -0.000381 | -0.000387 | `output/0002/diagnostics/stump_depth_prior_rgb_rerank_final_topm_l005_overlap_topk10` |
| playroom | 0.027192 | 0.018422 | 0.1132 | 0.1987 | -0.000179 | +0.000638 | -0.000269 | `output/0002/diagnostics/playroom_depth_prior_rgb_rerank_final_topm_l005_overlap_topk10` |
| truck | 0.024621 | 0.028670 | 0.0391 | 0.0741 | +0.000352 | +0.001177 | +0.000260 | `output/0002/diagnostics/truck_depth_prior_rgb_rerank_final_topm_l005_overlap_topk10` |

判断：l0.05 比 l0.10 更接近“保守修正”，但仍没有把三场景同时推到可扩全数据集的状态。`playroom` 只是在全图 PSNR 上翻正，prior top-k 仍然变差；`truck` 重新负向，且 top-10 IoU 仍只有 0.039。下一步如果继续 0002，优先缩小 `vfm_rgb_broad_topk`，或者把 depth prior 改成更后期的辅助裁剪，而不是继续单纯扫 lambda。

### 2026-05-18 RGB-gated depth rerank final-topm l0.10 broad035 pilot

这一轮保持 `vfm_dino_rerank_lambda=0.10`，只把 `vfm_rgb_broad_topk` 从 0.50 缩到 0.35，目的是验证“先放宽 RGB 候选，再让 depth prior 二次筛选”的入口是否能进一步压低错位和容量代价。结果表明它只能缓和，不能根治。

正式三场景结果：

| 场景 | Phase 0 FastGS big | RGB-gated depth rerank final-topm l0.10 broad035 | ΔPSNR / ΔSSIM / ΔLPIPS / ΔGS | QCGI | 输出 |
|---|---:|---:|---:|---:|---|
| stump | 27.1784 / 0.7868 / 0.2393, 1,064,860 GS | 27.2526 / 0.7917 / 0.2301, 1,321,343 GS, 127.29s | +0.0742 / +0.0049 / -0.0092 / +256,483 | -0.5075 | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l010_broad035_fastgs_big_stump_30k_scene_override_r_auto` |
| playroom | 30.7181 / 0.9148 / 0.2357, 589,253 GS | 30.7168 / 0.9144 / 0.2313, 734,580 GS, 110.58s | -0.0013 / -0.0004 / -0.0044 / +145,327 | -0.2691 | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l010_broad035_fastgs_big_playroom_30k_scene_override_r_auto` |
| truck | 26.0998 / 0.8896 / 0.1385, 625,803 GS | 26.0930 / 0.8900 / 0.1331, 838,229 GS, 127.29s | -0.0068 / +0.0004 / -0.0054 / +212,426 | -0.5211 | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l010_broad035_fastgs_big_truck_30k_scene_override_r_auto` |

Top-k 25% prior-overlap 诊断：

| 场景 | baseline L1 | prior top-k L1 | non-prior L1 | prior/RGB IoU | prior/RGB recall | candidate ΔL1 | prior top-k ΔL1 | non-prior ΔL1 | 输出 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stump | 0.030519 | 0.034971 | 0.029035 | 0.1790 | 0.3009 | -0.000340 | -0.000441 | -0.000306 | `output/0002/diagnostics/stump_depth_prior_rgb_rerank_final_topm_l010_broad035_overlap` |
| playroom | 0.019299 | 0.024015 | 0.017727 | 0.2149 | 0.3473 | +0.000036 | +0.000327 | -0.000061 | `output/0002/diagnostics/playroom_depth_prior_rgb_rerank_final_topm_l010_broad035_overlap` |
| truck | 0.028265 | 0.022478 | 0.030195 | 0.1108 | 0.1970 | +0.000242 | +0.000588 | +0.000127 | `output/0002/diagnostics/truck_depth_prior_rgb_rerank_final_topm_l010_broad035_overlap` |

Top-k 10% prior-overlap 诊断：

| 场景 | prior top-k L1 | non-prior L1 | prior/RGB IoU | prior/RGB recall | candidate ΔL1 | prior top-k ΔL1 | non-prior ΔL1 | 输出 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| stump | 0.037759 | 0.029715 | 0.0818 | 0.1502 | -0.000340 | -0.000390 | -0.000335 | `output/0002/diagnostics/stump_depth_prior_rgb_rerank_final_topm_l010_broad035_overlap_topk10` |
| playroom | 0.027192 | 0.018422 | 0.1132 | 0.1987 | +0.000036 | +0.000356 | +0.000000 | `output/0002/diagnostics/playroom_depth_prior_rgb_rerank_final_topm_l010_broad035_overlap_topk10` |
| truck | 0.024621 | 0.028670 | 0.0391 | 0.0741 | +0.000242 | +0.000952 | +0.000163 | `output/0002/diagnostics/truck_depth_prior_rgb_rerank_final_topm_l010_broad035_overlap_topk10` |

判断：`broad035` 比 top50 更像一个节流阀。`stump` 仍然保住三项质量正向，且 Gaussian 增量从约 +323k 压到 +256k；`playroom/truck` 的容量代价也有所下降。但 `playroom` 依旧没有转成稳定正向，`truck` 仍然是 prior/RGB 错位的薄壳，三场景 QCGI 全负。它说明入口缩窄能减少一部分浪费，却不能把 depth prior 变成稳定的 RGB 瓶颈代理。

### 2026-05-18 RGB-gated depth rerank final-topm l0.10 broad035 start9000 pilot

这一轮保持 `l0.10 + broad035`，但增加 `vfm_active_from_iter=9000`，让前 9000 iter 完全走 RGB/FastGS，depth prior 只介入 densification 后半段。注意默认 `densify_until_iter=15000`，所以不能把 start 设到 15001，否则 densification 阶段完全不会用到 prior。

正式三场景结果：

| 场景 | Phase 0 FastGS big | RGB-gated depth rerank final-topm l0.10 broad035 start9000 | ΔPSNR / ΔSSIM / ΔLPIPS / ΔGS | QCGI | 输出 |
|---|---:|---:|---:|---:|---|
| stump | 27.1784 / 0.7868 / 0.2393, 1,064,860 GS | 27.2377 / 0.7926 / 0.2243, 1,420,686 GS, 156.11s | +0.0593 / +0.0058 / -0.0150 / +355,826 | -0.8732 | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l010_broad035_start9000_fastgs_big_stump_30k_scene_override_r_auto` |
| playroom | 30.7181 / 0.9148 / 0.2357, 589,253 GS | 30.7635 / 0.9145 / 0.2310, 749,458 GS, 110.84s | +0.0454 / -0.0003 / -0.0047 / +160,205 | -0.2774 | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l010_broad035_start9000_fastgs_big_playroom_30k_scene_override_r_auto` |
| truck | 26.0998 / 0.8896 / 0.1385, 625,803 GS | 26.0805 / 0.8899 / 0.1330, 846,889 GS, 128.21s | -0.0193 / +0.0003 / -0.0055 / +221,086 | -0.5707 | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l010_broad035_start9000_fastgs_big_truck_30k_scene_override_r_auto` |

Top-k 25% prior-overlap 诊断：

| 场景 | baseline L1 | prior top-k L1 | non-prior L1 | prior/RGB IoU | prior/RGB recall | candidate ΔL1 | prior top-k ΔL1 | non-prior ΔL1 | 输出 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stump | 0.030519 | 0.034971 | 0.029035 | 0.1790 | 0.3009 | -0.000283 | -0.000277 | -0.000285 | `output/0002/diagnostics/stump_depth_prior_rgb_rerank_final_topm_l010_broad035_start9000_overlap` |
| playroom | 0.019299 | 0.024015 | 0.017727 | 0.2149 | 0.3473 | -0.000124 | +0.000166 | -0.000221 | `output/0002/diagnostics/playroom_depth_prior_rgb_rerank_final_topm_l010_broad035_start9000_overlap` |
| truck | 0.028265 | 0.022478 | 0.030195 | 0.1108 | 0.1970 | +0.000271 | +0.000584 | +0.000167 | `output/0002/diagnostics/truck_depth_prior_rgb_rerank_final_topm_l010_broad035_start9000_overlap` |

Top-k 10% prior-overlap 诊断：

| 场景 | prior top-k L1 | non-prior L1 | prior/RGB IoU | prior/RGB recall | candidate ΔL1 | prior top-k ΔL1 | non-prior ΔL1 | 输出 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| stump | 0.037759 | 0.029715 | 0.0818 | 0.1502 | -0.000283 | -0.000097 | -0.000304 | `output/0002/diagnostics/stump_depth_prior_rgb_rerank_final_topm_l010_broad035_start9000_overlap_topk10` |
| playroom | 0.027192 | 0.018422 | 0.1132 | 0.1987 | -0.000124 | +0.000226 | -0.000163 | `output/0002/diagnostics/playroom_depth_prior_rgb_rerank_final_topm_l010_broad035_start9000_overlap_topk10` |
| truck | 0.024621 | 0.028670 | 0.0391 | 0.0741 | +0.000271 | +0.000972 | +0.000193 | `output/0002/diagnostics/truck_depth_prior_rgb_rerank_final_topm_l010_broad035_start9000_overlap_topk10` |

判断：后期介入让 `playroom` 从 broad035 的轻微负向变成薄正向，但收益仍主要来自非-prior 区域；`stump` 的 LPIPS 很好但容量代价更糟，`truck` 的错位完全没有修复。这个结果排除了“只要晚一点接入就稳定”的假设，下一步应转向在线 depth residual 或 pruning-side 后处理。

### 2026-05-18 RGB-gated depth rerank final-topm l0.10 broad035 start9000 MipNeRF360 full

按用户建议，本轮虽然三场景 pilot 已经显示容量效率不足，仍进一步扩展到 MipNeRF360 全 9 场景，观察数据集均值是否存在总体正向。训练使用双卡 tmux 场景级并行，保持原图 `-r -1` / 1.6K 自动缩放、`fastgs_big + scene overrides`，配置为 `configs/experiments/0002_depth_anything_depth_prior_rgb_rerank_final_topm_l010_broad035_start9000.yaml`。`treehill` 首次 cache build 受 HuggingFace 临时连接失败影响中断；随后使用本地已缓存权重 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 补建 cache 并完成 train/render/metrics。

汇总路径：

- `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l010_broad035_start9000/mipnerf360_combined/summary.csv`
- `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l010_broad035_start9000/mipnerf360_combined/comparison_vs_phase0.csv`

| 场景 | Phase 0 FastGS big | Full MipNeRF360 candidate | ΔPSNR / ΔSSIM / ΔLPIPS / ΔGS | QCGI |
|---|---:|---:|---:|---:|
| bicycle | 25.2569 / 0.7553 / 0.2450, 1,560,209 GS | 25.3426 / 0.7657 / 0.2269, 1,886,181 GS | +0.0857 / +0.0105 / -0.0180 / +325,972 | -0.6185 |
| flowers | 21.6284 / 0.6023 / 0.3404, 1,122,815 GS | 21.6723 / 0.6055 / 0.3374, 1,403,416 GS | +0.0438 / +0.0031 / -0.0030 / +280,601 | -0.7008 |
| garden | 27.6346 / 0.8644 / 0.1096, 2,634,816 GS | 27.6315 / 0.8643 / 0.1099, 2,631,044 GS | -0.0032 / -0.0002 / +0.0003 / -3,772 | -0.0078 |
| stump | 27.1784 / 0.7868 / 0.2393, 1,064,860 GS | 27.2521 / 0.7927 / 0.2243, 1,418,824 GS | +0.0737 / +0.0059 / -0.0150 / +353,964 | -0.8492 |
| treehill | 22.8275 / 0.6323 / 0.3770, 1,009,110 GS | 22.8560 / 0.6401 / 0.3559, 1,269,544 GS | +0.0285 / +0.0078 / -0.0210 / +260,434 | -0.4525 |
| room | 32.2136 / 0.9304 / 0.1882, 571,607 GS | 32.2081 / 0.9335 / 0.1789, 765,178 GS | -0.0055 / +0.0030 / -0.0093 / +193,571 | -0.3732 |
| counter | 29.5259 / 0.9180 / 0.1766, 470,577 GS | 29.6565 / 0.9202 / 0.1709, 601,504 GS | +0.1306 / +0.0021 / -0.0056 / +130,927 | -0.0225 |
| kitchen | 32.2810 / 0.9390 / 0.1051, 1,177,988 GS | 32.3777 / 0.9400 / 0.1026, 1,415,081 GS | +0.0967 / +0.0011 / -0.0025 / +237,093 | -0.5173 |
| bonsai | 33.0846 / 0.9538 / 0.1598, 844,093 GS | 32.7310 / 0.9526 / 0.1561, 1,204,749 GS | -0.3536 / -0.0011 / -0.0037 / +360,656 | -1.5003 |
| **平均** | **27.9590 / 0.8203 / 0.2157, 1,161,786 GS** | **27.9698 / 0.8238 / 0.2070, 1,399,502 GS** | **+0.0107 / +0.0036 / -0.0087 / +237,716** | **-0.5602** |

判断：全 9 场景均值在三项质量指标上小幅正向，6/9 场景 PSNR 正向、7/9 场景 SSIM 正向、8/9 场景 LPIPS 正向；但容量代价过大，平均多 237,716 个 Gaussians，9/9 场景 QCGI 全负。`bonsai` 是主要质量负例，`garden/room` 也出现 PSNR 轻微负向。该 full run 说明 late RGB-gated depth rerank 可以制造数据集均值的感知质量增益，但不是一个容量可接受的 VFM_GS 训练策略。

### 2026-05-12 Prior/RGB 瓶颈重叠诊断

为避免继续盲目扩展 prior，新增 `scripts/diagnose_prior_overlap.py`，直接读取已有 render/gt、`cameras.json` 和 VFM cache，检查 prior top-k 区域是否也是 baseline RGB 高误差区域，并统计候选方法的 L1 改善是否集中在 prior 区域。该脚本不依赖 CUDA，可作为每个新 prior 的轻量体检；但它对 DINO descriptor cache 不能直接复现训练时的 render-vs-GT cosine residual，DINO token-edge 行只能作为 2D prior 对照。

诊断对象为 high-res `bicycle` matched baseline：

- baseline：`output/0002/fastgs_baseline_bicycle_30k_densify100_r_auto`
- direct depth candidate：`output/0002/depth_anything_depth_prior_bicycle_30k_r_auto`
- depth-edge candidate：`output/0002/depth_anything_edge_prior_bicycle_30k_r_auto`
- direct depth cache：`output/0002/vfm_cache/bicycle_depth_anything_v2s_depth`
- depth-edge cache：`output/0002/vfm_cache/bicycle_depth_anything_v2s_edge`
- DINO token-edge cache：`output/0001/vfm_cache_large/bicycle_dinov2_vitl14_token_edge_w1600`

Top-k 25% 结果：

| Prior | Candidate | baseline L1 | prior top-k L1 | non-prior L1 | prior/RGB top-k IoU | prior/RGB recall | candidate ΔL1 | prior top-k ΔL1 | non-prior ΔL1 | 输出 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Depth Anything relative depth | direct depth prior | 0.039367 | 0.050134 | 0.035778 | 0.2264 | 0.3675 | -0.000422 | -0.000656 | -0.000344 | `output/0002/diagnostics/bicycle_depth_prior_overlap` |
| Depth Anything depth edge | depth-edge prior | 0.039367 | 0.046330 | 0.037047 | 0.1749 | 0.2967 | -0.000023 | +0.000072 | -0.000055 | `output/0002/diagnostics/bicycle_depth_edge_prior_overlap` |
| DINO ViT-L token edge | none | 0.039367 | 0.041098 | 0.038791 | 0.1493 | 0.2593 | n/a | n/a | n/a | `output/0002/diagnostics/bicycle_dino_token_edge_baseline_overlap` |

Top-k 10% 结果：

| Prior | Candidate | prior top-k L1 | non-prior L1 | prior/RGB top-k IoU | prior/RGB recall | candidate ΔL1 | prior top-k ΔL1 | non-prior ΔL1 | 输出 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Depth Anything relative depth | direct depth prior | 0.056710 | 0.037440 | 0.1239 | 0.2188 | -0.000422 | -0.000770 | -0.000383 | `output/0002/diagnostics/bicycle_depth_prior_overlap_topk10` |
| Depth Anything depth edge | depth-edge prior | 0.053768 | 0.037767 | 0.1033 | 0.1864 | -0.000023 | +0.000098 | -0.000037 | `output/0002/diagnostics/bicycle_depth_edge_prior_overlap_topk10` |
| DINO ViT-L token edge | none | 0.042608 | 0.039007 | 0.0680 | 0.1269 | n/a | n/a | n/a | `output/0002/diagnostics/bicycle_dino_token_edge_baseline_overlap_topk10` |

判断：

- Direct relative depth prior 命中的 top-k 区域确实比非 prior 区域更难：top-25% L1 为 0.0501 vs 0.0358，top-10% L1 为 0.0567 vs 0.0374。candidate 的 L1 改善在 prior 区域也更大，说明 `depth_anything_depth_prior` 的 bicycle 正向不是纯随机波动。
- 但 direct depth 与 RGB 高误差 top-k 的重叠仍有限：top-25% IoU 只有 0.226，top-10% IoU 只有 0.124。这解释了为什么即使命中区域有效，全图 PSNR/SSIM 也只有小幅上涨。
- Depth-edge prior 的 top-k 区域也是较难区域，但 candidate 在这些区域反而微弱变差，改善主要来自非 prior 区域；这支持“不扩展 depth-edge prior”的决定。
- DINO token-edge 与 RGB 高误差 top-k 的重叠更低，top-10% recall 只有 0.127。它更像结构重要性信号，而不是当前全图 RGB 指标的主要误差瓶颈；但这只能评价 token-edge prior，不能解释或否定 0001 的 descriptor residual 结果。
- 后续 pilot 必须同时报告全图指标和 prior-overlap 诊断；如果某 prior 的 top-k 区域不覆盖 RGB 高误差区域，不能期待全图指标明显提升，只能转向局部结构指标或训练策略贡献。

| 日期 | 数据集 | 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数 | 训练时间 | 输出路径 | 结论 |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| 2026-05-11 | MipNeRF360 | bicycle | Depth Anything V2-S depth-edge prior 620 smoke | 19.4402 | 0.4039 | 0.6270 | 61,277 | 1.90s | `output/0002/depth_anything_edge_prior_bicycle_620_r_auto` | 链路通过；短程略低于 matched 620 baseline，不作为质量负例 |
| 2026-05-11 | MipNeRF360 | bicycle | FastGS matched 30k, `fastgs_baseline + densify100` | 25.0787 | 0.7370 | 0.2779 | 1,023,912 | 124.78s | `output/0002/fastgs_baseline_bicycle_30k_densify100_r_auto` | Depth Anything edge-prior 的公平对照 |
| 2026-05-11 | MipNeRF360 | bicycle | Depth Anything V2-S depth-edge prior 30k | 25.0764 | 0.7387 | 0.2744 | 1,063,311 | 131.21s | `output/0002/depth_anything_edge_prior_bicycle_30k_r_auto` | 弱混合信号：SSIM/LPIPS 小幅正向，PSNR 微负，点数 +39,399 |
| 2026-05-11 | MipNeRF360 | bicycle | Depth Anything V2-S direct depth prior 30k | 25.1415 | 0.7434 | 0.2689 | 1,005,953 | 128.82s | `output/0002/depth_anything_depth_prior_bicycle_30k_r_auto` | 当前主线：三项质量正向，且点数少于 matched baseline |
| 2026-05-14 | MipNeRF360 | stump | Depth Anything V2-S direct depth prior, `fastgs_big + scene override` | 27.1939 | 0.7895 | 0.2321 | 1,086,229 | 137.45s | `output/0002/depth_anything_depth_prior_fastgs_big_stump_30k_scene_override_r_auto` | 相对 Phase 0 三项正向，GS +21,369；有效 fastgs_big 正例 |
| 2026-05-14 | MipNeRF360 | bonsai | Depth Anything V2-S direct depth prior, `fastgs_big + scene override` | 33.0734 | 0.9546 | 0.1564 | 931,574 | 159.83s | `output/0002/depth_anything_depth_prior_fastgs_big_bonsai_30k_scene_override_r_auto` | PSNR 微负、SSIM/LPIPS 正向，GS +87,481；边界样本 |
| 2026-05-14 | DB | playroom | Depth Anything V2-S direct depth prior, `fastgs_big + scene override` | 30.8242 | 0.9144 | 0.2331 | 606,454 | 100.72s | `output/0002/depth_anything_depth_prior_fastgs_big_playroom_30k_scene_override_r_auto` | PSNR/LPIPS 正向，SSIM 微负，GS +17,201；薄正例 |
| 2026-05-14 | Tandt | truck | Depth Anything V2-S direct depth prior, `fastgs_big + scene override` | 25.9496 | 0.8870 | 0.1405 | 523,809 | 107.51s | `output/0002/depth_anything_depth_prior_fastgs_big_truck_30k_scene_override_r_auto` | 三项质量负向，虽少 101,994 点；明确负例 |
| 2026-05-17 | MipNeRF360 | stump | Depth Anything RGB-gated rerank final-topm l0.25, `fastgs_big + scene override` | 27.1267 | 0.7922 | 0.2265 | 1,388,160 | 154.22s | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l025_fastgs_big_stump_30k_scene_override_r_auto` | SSIM/LPIPS 正向但 PSNR 负向，GS +323,300，QCGI -0.8724 |
| 2026-05-17 | DB | playroom | Depth Anything RGB-gated rerank final-topm l0.25, `fastgs_big + scene override` | 30.6396 | 0.9143 | 0.2329 | 746,231 | 113.16s | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l025_fastgs_big_playroom_30k_scene_override_r_auto` | LPIPS 正向但 PSNR/SSIM 负向，GS +156,978，QCGI -0.4021 |
| 2026-05-17 | Tandt | truck | Depth Anything RGB-gated rerank final-topm l0.25, `fastgs_big + scene override` | 26.2003 | 0.8918 | 0.1317 | 845,866 | 129.50s | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l025_fastgs_big_truck_30k_scene_override_r_auto` | 从 direct depth 负例变为三项质量正向，但 GS +220,063，QCGI -0.4017 |
| 2026-05-17 | MipNeRF360 | stump | Depth Anything RGB-gated rerank final-topm l0.10, `fastgs_big + scene override` | 27.2355 | 0.7923 | 0.2258 | 1,387,811 | 153.56s | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l010_fastgs_big_stump_30k_scene_override_r_auto` | 三项质量正向，但 GS +322,951，QCGI -0.7581 |
| 2026-05-17 | DB | playroom | Depth Anything RGB-gated rerank final-topm l0.10, `fastgs_big + scene override` | 30.5282 | 0.9142 | 0.2326 | 746,312 | 113.08s | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l010_fastgs_big_playroom_30k_scene_override_r_auto` | PSNR/SSIM 负向，GS +157,059，QCGI -0.5136 |
| 2026-05-17 | Tandt | truck | Depth Anything RGB-gated rerank final-topm l0.10, `fastgs_big + scene override` | 26.1558 | 0.8916 | 0.1313 | 842,083 | 129.17s | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l010_fastgs_big_truck_30k_scene_override_r_auto` | 保住三项质量正向，但 GS +216,280，QCGI -0.4325 |
| 2026-05-17 | MipNeRF360 | stump | Depth Anything RGB-gated rerank final-topm l0.05, `fastgs_big + scene override` | 27.2379 | 0.7923 | 0.2258 | 1,388,153 | 153.55s | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l005_fastgs_big_stump_30k_scene_override_r_auto` | 三项质量正向，但 GS +323,293，QCGI -0.7570 |
| 2026-05-17 | DB | playroom | Depth Anything RGB-gated rerank final-topm l0.05, `fastgs_big + scene override` | 30.8380 | 0.9147 | 0.2309 | 746,168 | 112.77s | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l005_fastgs_big_playroom_30k_scene_override_r_auto` | PSNR 正向但 SSIM 轻微负向，GS +156,915，QCGI -0.1863 |
| 2026-05-17 | Tandt | truck | Depth Anything RGB-gated rerank final-topm l0.05, `fastgs_big + scene override` | 26.0655 | 0.8900 | 0.1328 | 850,023 | 129.97s | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l005_fastgs_big_truck_30k_scene_override_r_auto` | PSNR 负向，GS +224,220，QCGI -0.5939 |

## 结果表

| 日期 | 数据集 | 场景 | 方法 | PSNR | SSIM | LPIPS | Gaussian 数 | 训练时间 | 输出路径 | 结论 |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| 2026-05-18 | MipNeRF360 | stump | Depth Anything RGB-gated rerank final-topm l0.10 broad035, `fastgs_big + scene override` | 27.2526 | 0.7917 | 0.2301 | 1,321,343 | 127.29s | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l010_broad035_fastgs_big_stump_30k_scene_override_r_auto` | 三项质量正向，GS +256,483，QCGI -0.5075 |
| 2026-05-18 | DB | playroom | Depth Anything RGB-gated rerank final-topm l0.10 broad035, `fastgs_big + scene override` | 30.7168 | 0.9144 | 0.2313 | 734,580 | 110.58s | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l010_broad035_fastgs_big_playroom_30k_scene_override_r_auto` | PSNR/SSIM 轻微负向，GS +145,327，QCGI -0.2691 |
| 2026-05-18 | Tandt | truck | Depth Anything RGB-gated rerank final-topm l0.10 broad035, `fastgs_big + scene override` | 26.0930 | 0.8900 | 0.1331 | 838,229 | 127.29s | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l010_broad035_fastgs_big_truck_30k_scene_override_r_auto` | PSNR 负向、SSIM 微正向，GS +212,426，QCGI -0.5211 |
| 2026-05-18 | MipNeRF360 | stump | Depth Anything RGB-gated rerank final-topm l0.10 broad035 start9000, `fastgs_big + scene override` | 27.2377 | 0.7926 | 0.2243 | 1,420,686 | 156.11s | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l010_broad035_start9000_fastgs_big_stump_30k_scene_override_r_auto` | 三项质量正向但 GS +355,826，QCGI -0.8732 |
| 2026-05-18 | DB | playroom | Depth Anything RGB-gated rerank final-topm l0.10 broad035 start9000, `fastgs_big + scene override` | 30.7635 | 0.9145 | 0.2310 | 749,458 | 110.84s | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l010_broad035_start9000_fastgs_big_playroom_30k_scene_override_r_auto` | PSNR/LPIPS 正向、SSIM 轻微负向，QCGI -0.2774 |
| 2026-05-18 | Tandt | truck | Depth Anything RGB-gated rerank final-topm l0.10 broad035 start9000, `fastgs_big + scene override` | 26.0805 | 0.8899 | 0.1330 | 846,889 | 128.21s | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l010_broad035_start9000_fastgs_big_truck_30k_scene_override_r_auto` | PSNR 负向，prior top-k 仍变差，QCGI -0.5707 |
| 2026-05-18 | MipNeRF360 | 全 9 场景平均 | Depth Anything RGB-gated rerank final-topm l0.10 broad035 start9000, `fastgs_big + scene overrides` | 27.9698 | 0.8238 | 0.2070 | 1,399,502 | 181.20s | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l010_broad035_start9000/mipnerf360_combined` | 相对 Phase 0 为 +0.0107 / +0.0036 / -0.0087，但 GS +237,716，平均 QCGI -0.5602，9/9 QCGI 负向 |
| 2026-05-17 | MipNeRF360 | stump | Depth Anything RGB-gated rerank final-topm l0.05, `fastgs_big + scene override` | 27.2379 | 0.7923 | 0.2258 | 1,388,153 | 153.55s | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l005_fastgs_big_stump_30k_scene_override_r_auto` | 三项质量正向，但 GS +323,293，QCGI -0.7570 |
| 2026-05-17 | DB | playroom | Depth Anything RGB-gated rerank final-topm l0.05, `fastgs_big + scene override` | 30.8380 | 0.9147 | 0.2309 | 746,168 | 112.77s | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l005_fastgs_big_playroom_30k_scene_override_r_auto` | PSNR 正向但 SSIM 轻微负向，GS +156,915，QCGI -0.1863 |
| 2026-05-17 | Tandt | truck | Depth Anything RGB-gated rerank final-topm l0.05, `fastgs_big + scene override` | 26.0655 | 0.8900 | 0.1328 | 850,023 | 129.97s | `output/0002/depth_anything_depth_prior_rgb_rerank_final_topm_l005_fastgs_big_truck_30k_scene_override_r_auto` | PSNR 负向，GS +224,220，QCGI -0.5939 |

## 对照表

0002 的正式结果必须至少与以下 0001/基线结果比较：

| 对照 | 用途 |
|---|---|
| Phase 0 FastGS big baseline | 判断 high-res depth prior 是否超过同环境 baseline |
| High-res DINO descriptor top-k25 weighted i0.50 | 0001 容量受控 VFM_GS 初步验证 |
| High-res DINO descriptor top-k25 weighted i0.70 | 0001 质量-容量折中档 |
| DINO descriptor top-k25 `max` | 无回退 VFM 质量上界 |

## 2026-05-18 prune-protect-only pilot

Depth Anything prune-side auxiliary 这轮把 FastGS/RGB 保持为 densification 主信号，只在 densification 结束后保护 RGB 高 pruning-score 候选。三场景已完成，平均结果接近持平但略偏负：

| 数据集 | 场景 | PSNR | SSIM | LPIPS | Gaussian 数 | 训练时间 | 相对 Phase 0 | QCGI | 输出路径 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| MipNeRF360 | stump | 27.1460 | 0.7857 | 0.2406 | 1,052,302 | 134.01s | -0.0324 / -0.0011 / +0.0013, GS -12,558 | -0.0607 | `output/0002/depth_anything_depth_prior_prune_protect_topk010/mipnerf360/stump/fastgs_big_30k_scene_override_r_auto` |
| DB | playroom | 30.7653 | 0.9150 | 0.2360 | 588,069 | 95.96s | +0.0472 / +0.0002 / +0.0003, GS -1,184 | +0.0492 | `output/0002/depth_anything_depth_prior_prune_protect_topk010/db/playroom/fastgs_big_30k_scene_override_r_auto` |
| Tandt | truck | 26.1118 | 0.8895 | 0.1394 | 623,787 | 116.13s | +0.0120 / -0.0001 / +0.0009, GS -2,016 | +0.0060 | `output/0002/depth_anything_depth_prior_prune_protect_topk010/tandt/truck/fastgs_big_30k_scene_override_r_auto` |
| **平均** | **stump/playroom/truck** | **28.0077** | **0.8634** | **0.2053** | **754,719** | **115.37s** | **+0.0089 / -0.0003 / +0.0008, GS -5,253** | **-0.0018** | `output/0002/depth_anything_depth_prior_prune_protect_topk010/combined` |

结论：这条路线比 RGB rerank 稳得多，确实能把一部分场景拉回正向，同时不再制造大规模增点；但 stump 的负例说明它还不是全局默认解。

## 2026-05-18 prune-protect weight015 sweep

这一轮只把保护权重从 0.25 收到 0.15，`rgb_prune_topk` 仍然保持 1.0%。结果显示：`stump` 更接近 phase 0 了，但 `playroom` 明显退化，`truck` 基本持平却 LPIPS 变差。整体均值反而比上一轮差，说明单独降权重不是解法。

| 数据集 | 场景 | PSNR | SSIM | LPIPS | Gaussian 数 | 训练时间 | 相对 Phase 0 | QCGI | 输出路径 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| MipNeRF360 | stump | 27.1709 | 0.7862 | 0.2400 | 1,065,270 | 129.31s | -0.0076 / -0.0007 / +0.0007, GS +410 | -0.0246 | `output/0002/depth_anything_depth_prior_prune_protect_weight015_topk010/mipnerf360/stump/fastgs_big_30k_scene_override_r_auto` |
| DB | playroom | 30.5765 | 0.9146 | 0.2366 | 588,311 | 96.26s | -0.1416 / -0.0002 / +0.0009, GS -942 | -0.1494 | `output/0002/depth_anything_depth_prior_prune_protect_weight015_topk010/db/playroom/fastgs_big_30k_scene_override_r_auto` |
| Tandt | truck | 26.1049 | 0.8893 | 0.1397 | 621,427 | 109.16s | +0.0051 / -0.0003 / +0.0012, GS -4,376 | -0.0064 | `output/0002/depth_anything_depth_prior_prune_protect_weight015_topk010/tandt/truck/fastgs_big_30k_scene_override_r_auto` |
| **平均** | **stump/playroom/truck** | **27.9508** | **0.8634** | **0.2054** | **758,336** | **111.58s** | **-0.0480 / -0.0004 / +0.0010, GS -1,636** | **-0.0601** | `output/0002/depth_anything_depth_prior_prune_protect_weight015_topk010/combined` |

相比 `topk010`，这轮只把 stump 拉近了 baseline 一点点，但把 playroom 拉坏了，三场景均值也更差。当前更像是 proposal 空间该继续收，而不是保护权重该继续降。

## 2026-05-18 prune-protect topk005 sweep

这一轮保持保护权重 0.25，只把 RGB pruning proposal 从 top 1.0% 收窄到 top 0.5%。结果比 `weight015` 好，但不如原始 `topk010` 稳：`truck` 明显改善，`stump` 只小幅变好，`playroom` 从正例转为负例。

| 数据集 | 场景 | PSNR | SSIM | LPIPS | Gaussian 数 | 训练时间 | 相对 Phase 0 | QCGI | 输出路径 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| MipNeRF360 | stump | 27.1458 | 0.7860 | 0.2407 | 1,048,128 | 128.60s | -0.0327 / -0.0008 / +0.0014, GS -16,732 | -0.0555 | `output/0002/depth_anything_depth_prior_prune_protect_topk005/mipnerf360/stump/fastgs_big_30k_scene_override_r_auto` |
| DB | playroom | 30.6579 | 0.9146 | 0.2366 | 588,479 | 95.90s | -0.0602 / -0.0002 / +0.0009, GS -774 | -0.0675 | `output/0002/depth_anything_depth_prior_prune_protect_topk005/db/playroom/fastgs_big_30k_scene_override_r_auto` |
| Tandt | truck | 26.1372 | 0.8896 | 0.1392 | 623,045 | 109.56s | +0.0374 / +0.0000 / +0.0008, GS -2,758 | +0.0345 | `output/0002/depth_anything_depth_prior_prune_protect_topk005/tandt/truck/fastgs_big_30k_scene_override_r_auto` |
| **平均** | **stump/playroom/truck** | **27.9803** | **0.8634** | **0.2055** | **753,217** | **111.35s** | **-0.0185 / -0.0003 / +0.0010, GS -6,755** | **-0.0295** | `output/0002/depth_anything_depth_prior_prune_protect_topk005/combined` |

判断：`topk005` 没有支持“继续收窄 proposal 就能稳定”的假设。它确实更保护 truck，但 playroom 对过窄 proposal 很敏感；与 `topk010` 的平均 QCGI -0.0018 相比，`topk005` 反而退化到 -0.0295。当前最合理的 prune-protect 结论是：Depth Anything 可作为低成本后验保护信号，但需要场景自适应或更多场景统计，不能靠单一固定 top-k/weight 直接定版。

## 2026-05-19 prune-protect topk010 full MipNeRF360

为验证三场景近中性的 `topk010` 是否能扩展到更大范围，本轮用同一配置跑 MipNeRF360 全 9 场景。结果显示：它不是可用默认解。平均点数略少，但质量三项均值全部负向。

| 场景 | PSNR | SSIM | LPIPS | Gaussian 数 | 相对 Phase 0 | QCGI |
|---|---:|---:|---:|---:|---:|---:|
| bicycle | 25.2545 | 0.7553 | 0.2450 | 1,556,989 | -0.0024 / +0.0000 / +0.0000, GS -3,220 | -0.0024 |
| flowers | 21.6346 | 0.6024 | 0.3416 | 1,126,749 | +0.0062 / +0.0000 / +0.0012, GS +3,934 | -0.0029 |
| garden | 27.3385 | 0.8621 | 0.1120 | 2,618,309 | -0.2961 / -0.0024 / +0.0024, GS -16,507 | -0.3555 |
| stump | 27.1420 | 0.7859 | 0.2405 | 1,050,718 | -0.0365 / -0.0010 / +0.0011, GS -14,142 | -0.0615 |
| treehill | 22.8613 | 0.6324 | 0.3774 | 1,006,816 | +0.0338 / +0.0001 / +0.0005, GS -2,294 | +0.0336 |
| room | 32.1015 | 0.9301 | 0.1882 | 570,155 | -0.1122 / -0.0003 / +0.0001, GS -1,452 | -0.1190 |
| counter | 29.6040 | 0.9180 | 0.1768 | 471,617 | +0.0780 / -0.0001 / +0.0003, GS +1,040 | +0.0743 |
| kitchen | 32.3789 | 0.9392 | 0.1041 | 1,185,801 | +0.0979 / +0.0003 / -0.0010, GS +7,813 | +0.1007 |
| bonsai | 32.9002 | 0.9509 | 0.1598 | 847,904 | -0.1844 / -0.0029 / +0.0000, GS +3,811 | -0.2462 |
| **平均** | **27.9128** | **0.8196** | **0.2162** | **1,159,451** | **-0.0462 / -0.0007 / +0.0005, GS -2,335** | **-0.0643** |

判断：`treehill/counter/kitchen` 是正例，但 `garden/room/bonsai/stump` 的退化足以压过这些收益。固定 Depth Anything prune-protect 只能形成局部轻量修正，不能作为 MipNeRF360 默认策略。下一步若继续 0002，应停止固定 top-k/weight 扫描，改为场景自适应 protect 或在线/validation-driven 的误差入口。

## 失败记录

| 日期 | 阶段 | 范围 | 失败 | 后续处理 |
|---|---|---|---|---|
| 2026-05-11 | Phase 0 | MipNeRF360 `bicycle` / `room` high-res FastGS big baseline | 双卡均出现 CUDA illegal memory access，训练未完成，未产生 render/metrics | 用 `CUDA_LAUNCH_BLOCKING=1` 做单场景复现；检查并必要时重编译 `diff-gaussian-rasterization_fastgs`、`simple-knn`、`fused-ssim` 等本地 CUDA extension |
| 2026-05-11 | Phase 0 blocking 复跑 | MipNeRF360 `bicycle` / `room` high-res FastGS big baseline | blocking 口径仍失败，且都定位到 rasterizer 前向后 `radii` 访问 | 下一步用 `--debug_from` 打开 rasterizer debug 检查，优先定位 `diff_gaussian_rasterization_fastgs` 的具体 forward kernel |
| 2026-05-11 | rasterizer debug 复现 | MipNeRF360 `bicycle` high-res FastGS big | `--debug_from 9000` 之前已经在 3680 失败，说明触发点随随机视角/训练轨迹变化 | 改用 `--debug_from 0` 从第一步开始启用 rasterizer 内部 CUDA 检查 |
| 2026-05-11 | rasterizer debug wrapper 修补 | `diff_gaussian_rasterization_fastgs` Python wrapper | debug 分支未同步 FastGS 扩展的 9 项返回值，导致无法进入真实 CUDA debug | 已修补 debug 分支；重新运行 `--debug_from 0` |
| 2026-05-11 | rasterizer debug kernel 定位 | MipNeRF360 `bicycle` high-res FastGS big | 修补 wrapper 后，在 `rasterizer_impl.cu:422` 捕获 `operation not supported on global/shared address space`；对应 `identifyTileRanges` 后的 CUDA 检查 | 暂停 baseline 与 Depth Anything；优先排查 FastGS rasterizer 在 Blackwell / CUDA 12.8 / sm120 下的 `identifyTileRanges`、前置 sort/buffer 状态或编译兼容性 |
| 2026-05-11 | rasterizer fix1 debug 验证 | MipNeRF360 `bicycle` high-res FastGS big | 原失败点已通过 5000-step debug 验证；未产出 render/metrics | 继续正常 30k baseline；若通过再恢复双卡/全数据集 baseline |
| 2026-05-11 | rasterizer fix1 30k 单场景验收 | MipNeRF360 `bicycle` high-res FastGS big | 训练、渲染、指标全部完成，且与 0001 high-res bicycle baseline 基本一致 | 恢复 MipNeRF360 双卡全场景 baseline；Depth Anything 仍等待全场景 Phase 0 |
| 2026-05-11 | 非 detached 补跑中断 | MipNeRF360 `treehill` / `bonsai` | `treehill` 训练目录存在但 PLY 截断，render 报 `early end-of-file`；`bonsai` 训练停在半程，无 30000 checkpoint | 将不完整目录归档到 `output/0002/debug_artifacts/interrupted_mipnerf360_20260511_200949/`，改用 `setsid/nohup` detached 补跑并通过 |
| 2026-05-14 | fastgs_big direct depth prior 首次扩场景 | MipNeRF360 `stump` / `bonsai` | 直接 `--variant fastgs_big` 未带 phase 0 runner 的 per-scene overrides，导致 `stump`/`bonsai` 与 baseline recipe 不一致，指标不可用于方法结论 | 已用 scene overrides 重新补跑并写入正式结果；后续所有 fastgs_big prior 必须显式复用 `scripts/run_0001_fastgs_big_eval.py` 的场景超参 |

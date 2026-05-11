# 0002 复盘

## 当前决策

0002 的第一步是双卡 RTX 5090 baseline 复核，而不是直接开始 Depth Anything。先在三个公开数据集所有场景上跑 FastGS big baseline，保持原图输入和 FastGS 原始 1.6K 自动缩放口径，确认 5090 结果与此前 4090D 相差无几。

通过 phase 0 后，0002 再从 Depth Anything dense depth prior 开始，不继续扩展 0001 中已经收束为负结果的 COLMAP sparse depth-edge proxy。

由于当前双卡算力充足，0002 的 Depth Anything 验证不再先跑 `-r 8`。所有 smoke、pilot 和正式验证都使用原图输入与 FastGS 1.6K 自动缩放口径。先在少数代表场景上验证有效性；如果 pilot 多场景成立，再扩展三个公开数据集全场景。

第一阶段只做 densification prior，保持 `vfm_weight=0.0`，避免把 pruning score 变量重新混入。只有当 dense depth prior 在 high-res matched 对照中出现正向或互补信号后，才考虑 depth-edge protect 或 pruning-side 保护。

## 2026-05-11 Phase 0 首次启动

双卡 high-res FastGS big baseline 已启动，但在 MipNeRF360 首个场景上均提前失败：

- GPU0 `bicycle` 在 7190/30000 iteration 报 `CUDA error: an illegal memory access was encountered`，Python 栈停在 `loss.item()`。
- GPU1 `room` 在约 2200/30000 iteration 报同一 CUDA illegal memory access，Python 栈停在 `(radii > 0).nonzero()`。

两个失败点都属于 CUDA 异步错误的表层位置；结合两张卡、两个场景同时失败，当前优先判断为 5090 环境中的本地 CUDA extension / PyTorch / CUDA 编译链路稳定性问题。该失败发生在 Depth Anything backend 接入前，因此不能作为 0002 方法负结果。下一步先用 `CUDA_LAUNCH_BLOCKING=1` 单场景复现，并检查或重编译本地 CUDA extension。

补充诊断显示：当前环境为 PyTorch `2.10.0+cu128`，两张 RTX 5090 capability `(12, 0)`；rasterizer `.so` 字符串包含 `Cuda compilation tools, release 12.8` 与 `-arch sm_120`，说明扩展已按 Blackwell 目标编译过。`CUDA_LAUNCH_BLOCKING=1` 的 MipNeRF360 `room` high-res FastGS big 2500-step 诊断完成训练并保存点云，未复现 illegal memory access。当前采用保守推进：先用 blocking 口径复跑 Phase 0，若 30k 仍稳定，再继续全数据集 baseline；若仍失败，再重编译或定位 rasterizer kernel。

blocking Phase 0 复跑仍失败：`bicycle` 在 9300/30000 iteration、`room` 在 11450/30000 iteration 触发 CUDA illegal memory access；两者 Python 栈都停在 `src/vfm_gs/gaussian_renderer/__init__.py:108` 的 `(radii > 0).nonzero()`。结合错误发生在 rasterizer 返回后读取 `radii` 时，当前优先定位 `diff_gaussian_rasterization_fastgs` forward kernel，而不是继续调度全场景 baseline。

`bicycle` 的 `--debug_from 9000` 复现没有等到 debug 打开，训练在 3680/9500 提前失败，仍无 `snapshot_fw.dump`。这说明失败触发点受随机视角和训练轨迹影响，不是固定 iteration；下一步应从 iteration 1 开始开启 rasterizer debug，牺牲速度换取具体 kernel 检查点。

首次 `--debug_from 0` 没有进入真实 CUDA debug，而是暴露了 `diff_gaussian_rasterization_fastgs` Python wrapper 的接口漂移：debug 分支仍按旧 7 项返回值解包，当前 FastGS 扩展实际返回 9 项。已把源码和当前 `.venv` 安装副本的 debug 分支同步到 9 项，并把误生成的 snapshot 归档到 `output/0002/debug_artifacts/`。这个修补不解决 illegal memory access 本身，但恢复了后续 debug 能力。

修补 wrapper 后的 `--debug_from 0` 复现进入真实 rasterizer debug，并在 `bicycle` 1220/5000 iteration 失败。CUDA 检查点位于 `submodules/diff-gaussian-rasterization_fastgs/cuda_rasterizer/rasterizer_impl.cu:422`，紧跟 `identifyTileRanges` kernel，错误为 `operation not supported on global/shared address space`。本次 snapshot 已归档为 `output/0002/debug_artifacts/snapshot_fw_identify_tile_ranges_global_shared_20260511.dump`。这确认 Phase 0 尚未完成 baseline 验收，也没有可用于比较 4090D 的正常训练指标。

第一轮修补加入了三类防御：`identifyTileRanges` 对无效/越界 tile key 完整跳过，key/value buffer 在写入前初始化为 sentinel，`duplicateToTilesTouched` 对低透明度、非正阈值和 NaN/反序 bbox 做早退；同时补充 `<cstdint>` 以保证 CUDA 12.8 重编译。使用 `CUDA_HOME=/usr/local/cuda-12.8`、`TORCH_CUDA_ARCH_LIST=12.0` 重编译安装后，`bicycle --debug_from 0` high-res 5000-step 已完成，保存 1,340,808 个 Gaussians，未再触发 `identifyTileRanges` 或 CUDA illegal memory access。该结果说明修补方向有效，但还不是 30k baseline 验收。

修补后的正常 30k `bicycle` baseline 已完成 train/render/metrics：25.2569 PSNR、0.7553 SSIM、0.2450 LPIPS、1,560,209 个 Gaussians，训练 159.11s。0001 文档中同口径 4090D/high-res bicycle baseline 记录为 25.2532 / 0.7554 / 0.2446、1,560,079 个点，因此当前 5090 单场景结果没有明显质量或容量漂移。单场景 baseline smoke 通过，但还不能替代三个公开数据集全场景 Phase 0。

修补后的 MipNeRF360 全 9 场景 baseline 已完成 train/render/metrics。统一汇总为 `output/0002/phase0_5090_fastgs_big_baseline_fix1/mipnerf360_combined/summary.csv`，均值 27.9590 / 0.8203 / 0.2157、1,161,786 个 Gaussians，训练 159.70s。0001 同口径 high-res FastGS big baseline 是 27.9293 / 0.8198 / 0.2157、1,161,242 个点，因此 5090 fix1 在 MipNeRF360 上无系统性质量或容量漂移。

DB/Tandt baseline 也已完成。DB 均值为 30.2331 / 0.9111 / 0.2397、646,600 个 Gaussians，对照 0001 high-res FastGS big 的 30.2073 / 0.9112 / 0.2402、650,194 点；Tandt 均值为 24.4955 / 0.8579 / 0.1736、540,119 个 Gaussians，对照 0001 的 24.3557 / 0.8573 / 0.1745、540,578 点。三数据集全 13 场景 Phase 0 可以判为通过。

执行层面补充一条运行约束：当前环境未安装 `screen`，长实验改用 `setsid/nohup` detached 方式保活。一次非 detached 补跑中 `treehill` 出现截断 PLY，render 报 `early end-of-file`，`bonsai` 停在半程训练目录；两个不完整目录已归档到 `output/0002/debug_artifacts/interrupted_mipnerf360_20260511_200949/`，后续 detached 补跑通过。接下来 DB/Tandt 也应沿用 detached 方式，避免 SSH 断联导致训练产物损坏。

## 2026-05-11 Depth Anything V2-S smoke

已完成 dense depth prior 的最小工程接入。当前选择 `depth_anything_depth_edge_prior` 作为第一条可运行路径，而不是一开始做在线 depth residual：先离线用 Depth Anything V2-S 从 GT RGB 生成 dense depth-edge cache，再把该 2D prior 直接送入既有 `top-k metric_map -> accum_metric_counts -> densification importance` 链路。这样能先验证 cache、preflight、scorer 和训练调用栈，减少在线推理域差异和速度变量。

依赖状态：`.venv` 已安装 `transformers==5.8.0`、`huggingface-hub==1.14.0`、`safetensors==0.7.0`。首次模型下载被 HuggingFace Xet 通道的 `RemoteProtocolError` 打断；设置 `HF_HUB_DISABLE_XET=1` 后 `depth-anything/Depth-Anything-V2-Small-hf` 能正常加载。该环境变量应写入后续 cache build 命令。

`bicycle` high-res cache 已构建并校验通过：`output/0002/vfm_cache/bicycle_depth_anything_v2s_edge`，194 entries，48MB，`max_width=1600`，feature 为 `depth_anything_depth_edge`。训练 preflight 在 camera loading 前通过，说明 backend/feature 校验已经接入。

620-step smoke 完成 train/render/metrics：Depth Anything V2-S depth-edge prior 为 19.4402 / 0.4039 / 0.6270、61,277 个 Gaussians、训练 1.90s；matched high-res FastGS 620 baseline 为 19.4930 / 0.4046 / 0.6268、61,278 个 Gaussians、训练 1.81s。短程指标略低于 matched baseline，但差异很小，且 620-step 只用于链路健康，不作为方法质量负例。

30k `bicycle` pilot 已完成，并补跑了同 recipe matched baseline。Depth Anything V2-S depth-edge prior 30k 为 25.0764 / 0.7387 / 0.2744、1,063,311 个 Gaussians、训练 131.21s；matched `fastgs_baseline + densify100` 为 25.0787 / 0.7370 / 0.2779、1,023,912 个 Gaussians、训练 124.78s。相对 matched baseline，Depth Anything edge-prior 是 -0.0023 PSNR、+0.0017 SSIM、LPIPS -0.0035，Gaussian 数 +39,399，QCGI 约 +0.010。

这是一条弱混合信号：SSIM/LPIPS 指向几何边界 prior 可能有用，但 PSNR 没有转正，且需要更多点。相对 Phase 0 `fastgs_big` bicycle 的 25.2569 / 0.7553 / 0.2450 仍明显落后，不过该比较混入了 `fastgs_baseline` 与 `fastgs_big` recipe 差异，只能作为环境上限参考，不能当作 depth-edge prior 的公平负例。

direct relative depth cache 与 30k pilot 也已完成。cache 为 `output/0002/vfm_cache/bicycle_depth_anything_v2s_depth`，194 entries，46.21MB，feature 为 `depth_anything_relative_depth`，validate 通过。Depth Anything V2-S direct depth prior 30k 为 25.1415 / 0.7434 / 0.2689、1,005,953 个 Gaussians、训练 128.82s。相对同 recipe matched baseline，这是 +0.0628 PSNR、+0.0063 SSIM、LPIPS -0.0090，Gaussian 数 -17,959，QCGI 约 +0.2345。

当前决策：`depth_anything_depth_prior` 明显优于 depth-edge prior，成为 0002 当前主线。它已经在 `bicycle` 上给出三项质量正向且更少点数的信号；下一步扩到 `stump/bonsai/playroom/truck` 四个 pilot 场景。只有 direct depth 在多数 pilot 场景继续正向后，才进入全数据集训练验证；否则再回头考虑 `fastgs_big` recipe matched 接入、在线 render-depth residual 或局部 depth 区域指标。

## 2026-05-12 指标瓶颈诊断

为回应“DINO descriptor 改善很薄，瓶颈到底在哪里”的问题，新增 `scripts/diagnose_prior_overlap.py` 并先在 high-res `bicycle` 上补充验证。该诊断不重新训练，只读取已有 baseline/candidate render、GT、`cameras.json` 和 VFM cache，比较 prior top-k 与 RGB 高误差 top-k 的重叠，并看候选方法的 L1 改善是否真的落在 prior 区域。

关键结论是：当前 prior 并没有很好覆盖全图 RGB 指标的主要误差区域。Depth Anything relative depth 比 depth-edge 和 DINO token-edge 更接近 RGB 瓶颈，但重叠仍有限。

- Direct depth prior top-25% 区域的 baseline L1 为 0.0501，高于非 prior 区域 0.0358；candidate 在该区域 L1 改善 -0.000656，也大于非 prior 区域 -0.000344。这说明 `depth_anything_depth_prior` 的 bicycle 正向是有区域对应关系的。
- Direct depth prior 与 RGB 高误差 top-25% 的 IoU 只有 0.226，top-10% 只有 0.124。也就是说，它命中的是“相对更难”的区域，但不是 RGB loss 最大的那一小撮区域。
- Depth-edge prior top-25% 区域 baseline L1 为 0.0463，高于非 prior 区域 0.0370，但 candidate 在 prior 区域 L1 反而 +0.000072，改善主要来自非 prior 区域。这解释了 depth-edge 30k 只有弱混合信号。
- DINO ViT-L token-edge 与 RGB 高误差区域重叠最低：top-25% IoU 0.149，top-10% IoU 0.068。它更像结构重要性信号，不是全图 RGB 指标瓶颈的直接代理。

这条诊断改变后续优先级：不再把“换一个结构 prior”默认视为提高全图 PSNR/SSIM 的主路径。后续每个 pilot 都应先回答两个问题：prior top-k 是否覆盖 RGB 高误差区域；candidate 改善是否确实落在 prior 区域。如果答案是否定的，应该转向局部结构指标、validation selector 或更直接的优化入口，而不是继续扩大同类 prior 扫描。

## 继承自 0001 的边界

- DINO descriptor densify-only top-k25 weighted i0.50/i0.70 是 0001 对 VFM_GS 的初步验证主线。
- Depth Anything 的任务不是替代该主线，而是验证几何/遮挡边界是否提供互补收益。
- 稀疏 COLMAP depth-edge 覆盖不足，不再继续扩展。
- 硬 candidate cap、final prune 和早期 staged target-prune 不作为 0002 默认容量解法。

## 观察项

- 5090 与 4090D 的 PSNR/SSIM/LPIPS 是否存在系统性漂移。
- 本地 CUDA extension 在新架构上的 Gaussian 数量和训练轨迹是否稳定。
- 双卡场景级并行是否会引入输出目录或日志冲突。
- Depth Anything high-res cache 和在线推理是否造成不可接受的显存或时间开销。

## 下一步

扩展 direct relative depth prior 到 `stump/bonsai/playroom/truck` 四个 high-res 30k pilot 场景，但每个场景必须同步跑 `diagnose_prior_overlap.py`。如果 direct depth 在某场景全图指标正向，同时 prior top-k 区域更难且改善集中在 prior 区域，再进入下一轮；如果全图正向但 prior 区域不对应，先把它记录为训练轨迹偶然收益，不急着扩全数据集。长任务继续用 detached 方式运行；每轮实验完成后更新文档、commit 并 push；当前只有本服务器改动，按用户要求不再强制先 `git pull`。

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

当前决策：`depth_anything_depth_prior` 明显优于 depth-edge prior，成为 0002 当前主线。它已经在 `bicycle` 上给出三项质量正向且更少点数的信号。2026-05-14 进一步确认：在 `fastgs_big + scene overrides` 下，`stump` 是有效正例，`bonsai` 是 PSNR 微负但 SSIM/LPIPS 正向的边界样本。必须注意，所有 fastgs_big prior 都要显式复用 phase 0 的 per-scene overrides；不带 overrides 的 run 会混入 recipe mismatch，不能作为方法负例。2026-05-17 的 RGB-gated rerank final-topm l0.25 说明“先 RGB broad 再 depth rerank”确实能把 `truck` 从负例翻成三项质量正向，但它同时把 `stump/playroom` 的容量和 QCGI 拉得很差，因此只能作为方向验证，不能直接升级为默认方案。

## 2026-05-14 fastgs_big direct depth prior 复验

本轮先踩到了一个有价值的陷阱：直接用 `--variant fastgs_big` 跑 `stump/bonsai`，并不会自动带上 `scripts/run_0001_fastgs_big_eval.py` 中的场景超参。第一次 run 的 `stump` 使用了 `dense=0.001`、`grad_abs_thresh=0.0008`，而 phase 0 baseline 是 `dense=0.004`、`grad_abs_thresh=0.001`；`bonsai` 使用了 `highfeature_lr=0.005`、`grad_abs_thresh=0.0008`，而 phase 0 baseline 是 `highfeature_lr=0.02`、`grad_abs_thresh=0.0002`。这导致首次结果尤其是 bonsai 出现严重退化，必须标记为无效诊断。

对齐 scene overrides 后，`depth_anything_depth_prior` 的 fastgs_big 结果变成：

- `stump`: 27.1939 / 0.7895 / 0.2321、1,086,229 点；相对 phase 0 为 +0.0155 PSNR、+0.0027 SSIM、LPIPS -0.0072，GS +21,369，QCGI +0.0835。
- `bonsai`: 33.0734 / 0.9546 / 0.1564、931,574 点；相对 phase 0 为 -0.0113 PSNR、+0.0008 SSIM、LPIPS -0.0034，GS +87,481，QCGI -0.0651。

这改变了上一轮“fastgs_big 复现失败”的判断。更准确的说法是：Depth Anything direct depth 在 fastgs_big 下有弱正向信号，但收益不稳定。`stump` 的三项质量成立且 QCGI 为正；`bonsai` 只是感知/结构指标正向，PSNR 微负且容量增加接近 0.09M，QCGI 为负，不足以支持直接全数据集扩展。

overlap 诊断也支持这个分层判断：

- `stump` top-25% prior 区域 baseline L1 为 0.034971，高于非 prior 的 0.029035；candidate 全图 L1 改善 -0.000144，prior top-k 改善 -0.000280，大于非 prior 的 -0.000099。说明 `stump` 的收益确实更集中在 Depth Anything 关心的区域。
- `bonsai` top-25% prior 区域也更难，0.016605 vs 0.011675；但 candidate L1 为 +0.000117，prior top-k 也是 +0.000106。也就是说，bonsai 的 SSIM/LPIPS 正向没有对应到 RGB L1 改善，应视为边界样本。
- 两个场景的 prior/RGB 高误差 IoU 仍低：top-25% 为 0.1790/0.2030，top-10% 为 0.0818/0.0978。Depth Anything dense depth prior 比 DINO token-edge 更贴近几何难区，但仍不是 RGB loss 的直接替代信号。

当前下一步不应直接跑全数据集。更合理的是补 `playroom/truck` 两个跨数据集场景，用同样的 scene override 纪律和 overlap 诊断判断是否存在室内/道路场景互补。如果 `playroom/truck` 至少一个清晰正向，0002 再考虑扩展；如果二者都只是薄收益或负向，应暂停 direct prior 主线，转向在线 depth residual 或 RGB-gated depth prior。

## 2026-05-14 playroom/truck 跨数据集 pilot

跨数据集补验已完成，仍然严格使用 phase 0 scene overrides。

`playroom` direct depth prior 为 30.8242 / 0.9144 / 0.2331、606,454 点，训练 100.72s。相对 phase 0 FastGS big 的 30.7181 / 0.9148 / 0.2357、589,253 点，是 +0.1061 PSNR、-0.0004 SSIM、LPIPS -0.0026，GS +17,201，QCGI +0.0936。它是薄正例：PSNR/LPIPS 与 QCGI 成立，但 SSIM 没有转正。

`truck` direct depth prior 为 25.9496 / 0.8870 / 0.1405、523,809 点，训练 107.51s。相对 phase 0 FastGS big 的 26.0998 / 0.8896 / 0.1385、625,803 点，是 -0.1502 PSNR、-0.0026 SSIM、LPIPS +0.0021，GS -101,994，QCGI -0.2124。它是明确负例，少点不能弥补质量下降。

overlap 诊断进一步解释了分裂：

- `playroom` 的 prior top-k 区域确实更难：top-25% L1 0.024015 vs non-prior 0.017727；但 candidate 的 L1 改善更偏非-prior，prior top-k 为 -0.000093，non-prior 为 -0.000168。它说明 playroom 收益不完全来自精确命中 depth top-k 区域，更像 densification 轨迹扰动带来的薄收益。
- `truck` 的 prior top-k 区域不是 RGB 高误差区域：top-25% prior L1 0.022478 低于 non-prior 0.030195；top-10 prior/RGB IoU 只有 0.0391，recall 0.0741。candidate 全图 L1 变差 +0.000730，top-10 prior 区域还变差 +0.000946。这个结果和全图指标负向一致。

至此，direct depth prior 的 pilot 结论是局部成立但不稳：`bicycle/stump/playroom` 有正向或薄正向信号，`bonsai/truck` 是边界或负例。它不满足“多场景稳定后扩全数据集”的标准。下一步如果继续 0002，不应继续同配置铺全场景，而应改策略：优先考虑 RGB-gated depth prior，只在 RGB 高误差候选区域内用 Depth Anything 做二次权重；或者做在线 render-vs-GT depth residual，让 prior 直接对应当前重建误差，而不是离线 GT depth top-k prior。

## 2026-05-17 RGB-gated depth rerank final-topm l0.25 pilot

这轮把 direct depth prior 改成 `RGB broad candidate -> depth prior rerank -> final-topm`，目标是验证你提出的“先用 RGB 放宽候选，再让 depth prior 二次筛选”的思路是否能减少错位。结果说明这个思路是合理的，但 l0.25 还太激进。

`truck` 从 direct depth 的明确负例翻成三项质量正向，说明 prior 先验放在 RGB 候选后面是有意义的；但 `stump/playroom` 的 PSNR 退化、Gaussian 数显著上涨，QCGI 全负，说明当前权重把 prior 放大得过头了。更像是在修正候选分布，而不是稳定提升 densification 质量。

因此，RGB-gated 方向应该继续，但下一轮要降到更保守的 rerank 强度，必要时再缩小 `vfm_rgb_broad_topk` 或延后 `vfm_active_from_iter`。现在不能把 l0.25 当成可扩全场景的最终方案。

## 2026-05-12 指标瓶颈诊断

为回应“这些 prior 是否真的命中当前重建瓶颈”的问题，新增 `scripts/diagnose_prior_overlap.py` 并先在 high-res `bicycle` 上补充验证。该诊断不重新训练，只读取已有 baseline/candidate render、GT、`cameras.json` 和 VFM cache，比较 prior top-k 与 RGB 高误差 top-k 的重叠，并看候选方法的 L1 改善是否真的落在 prior 区域。DINO token-edge 行只是 2D prior 对照，不能替代 0001 真实 descriptor residual 诊断。

关键结论是：当前 prior 并没有很好覆盖全图 RGB 指标的主要误差区域。Depth Anything relative depth 比 depth-edge 和 DINO token-edge 更接近 RGB 瓶颈，但重叠仍有限。

- Direct depth prior top-25% 区域的 baseline L1 为 0.0501，高于非 prior 区域 0.0358；candidate 在该区域 L1 改善 -0.000656，也大于非 prior 区域 -0.000344。这说明 `depth_anything_depth_prior` 的 bicycle 正向是有区域对应关系的。
- Direct depth prior 与 RGB 高误差 top-25% 的 IoU 只有 0.226，top-10% 只有 0.124。也就是说，它命中的是“相对更难”的区域，但不是 RGB loss 最大的那一小撮区域。
- Depth-edge prior top-25% 区域 baseline L1 为 0.0463，高于非 prior 区域 0.0370，但 candidate 在 prior 区域 L1 反而 +0.000072，改善主要来自非 prior 区域。这解释了 depth-edge 30k 只有弱混合信号。
- DINO ViT-L token-edge 与 RGB 高误差区域重叠最低：top-25% IoU 0.149，top-10% IoU 0.068。它更像结构重要性信号，不是全图 RGB 指标瓶颈的直接代理；但这不能外推为“DINO descriptor residual 已经错位”，因为 token-edge 与训练时的 descriptor cosine residual 不是同一个 metric map。

这条诊断改变后续优先级：不再把“换一个结构 prior”默认视为提高全图 PSNR/SSIM 的主路径。后续每个 pilot 都应先回答两个问题：prior top-k 是否覆盖 RGB 高误差区域；candidate 改善是否确实落在 prior 区域。如果答案是否定的，应该转向局部结构指标、validation selector 或更直接的优化入口，而不是继续扩大同类 prior 扫描。对于 DINO descriptor，下一步应先补做真实 render-vs-GT descriptor residual overlap，再决定是否训练。

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

暂停同配置 direct depth prior 的全数据集扩展。下一步若继续 0002，应把 RGB-gated depth prior 再做一轮更保守的权重扫描，并继续用 `stump/playroom/truck` 作为三点诊断：`stump` 看能否保住原本正例，`playroom` 看薄收益是否还在，`truck` 看能否保住被修正的方向。长任务继续用 detached 方式运行；每轮实验完成后更新文档、commit 并 push；当前只有本服务器改动，按用户要求不再强制先 `git pull`。

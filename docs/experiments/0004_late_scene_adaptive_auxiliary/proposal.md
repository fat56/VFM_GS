# 0004 Late Scene-Adaptive Auxiliary

## 核心假设

0002 和 0003 已经说明：把 DINO / Depth Anything 当作 FastGS 的主增长信号，往往会出现区域错位、跨场景不稳、或者“结构上合理但指标涨得薄”的问题。0004 不再验证“prior 能不能直接拉动 densification”，而是验证另一条更保守的策略：

1. 先让 RGB / FastGS 自己把粗结构和候选区域建立起来。
2. 辅助 prior 只在后期介入，只在 RGB 候选内部 rerank 或 protect，不再单独拉新区域。
3. 辅助 prior 必须是场景自适应的，不能用一个全局固定阈值硬套所有场景。
4. 辅助 prior 必须受预算约束，最多只能改变一部分候选和一部分 pruning 结果，不能把总点数和训练时长推爆。

这次的真正对象是“辅助策略”，不是单纯的 DINO 或 Depth Anything 本身。先用 0002 里最接近中性的 Depth Anything prune-side 方案做主 pilot，再把 0003 的 DINO prune-protect-only 作为负例对照，确认“晚期 + 场景自适应 + 预算约束”这三个条件是否真的能把 prior 从主信号降级成可用辅助器。

## 变体 / 配置

- 变体：`fastgs_big`
- 主打分器：`depth_anything_depth_prior`
- 对照打分器：`dinov2_descriptor_cosine`
- 训练原则：`RGB` 先生成候选，辅助 prior 只 rerank / protect
- 介入时机：晚期，默认从 densify 窗口后半段开始
- 预算原则：候选比例、protect 数量、最终 Gaussian 数量都要设上限
- 代码版本：当前主分支 + 0002/0003 已验证的辅助后端

## 运行命令

```bash
python scripts/run_0001_fastgs_big_eval.py \
  --dataset-name mipnerf360 \
  --dataset-root datasets/mipnerf360 \
  --output-root output/0004/late_scene_adaptive_auxiliary/mipnerf360 \
  --scenes bicycle stump room \
  --variant fastgs_big \
  --config configs/experiments/0004_late_scene_adaptive_auxiliary.yaml \
  --vfm-cache-template output/0002/vfm_cache/{scene}_depth_anything_v2s_depth \
  --resolution -1
```

## 数据集

- 数据集：MipNeRF360 先行，必要时扩展到 DB / Tandt
- 场景：先选 `bicycle / stump / room` 做异质 pilot，再决定是否全量
- 分辨率：原图输入，保持 FastGS 原始 1.6K 自动缩放口径，不再使用 `-r 8`

## 指标

| 指标 | 基线 | 实验 | 差值 |
|---|---:|---:|---:|
| PSNR | TBD | TBD | TBD |
| SSIM | TBD | TBD | TBD |
| LPIPS | TBD | TBD | TBD |
| 训练时间 | TBD | TBD | TBD |
| Gaussian 数量 | TBD | TBD | TBD |

## 失败记录

- 若辅助 prior 仍然把增长拉到 RGB 低误差区域，说明它仍不是合格的 rerank/protect 信号。
- 若按场景切换后仍需要手工调很多阈值，说明 scene-adaptive 只是换了名字的超参扫。
- 若预算一收紧就明显掉质量，说明这个 prior 只能做诊断，不能做在线辅助。

## 决策

先验证“晚期 + 场景自适应 + 预算约束”这三件事是否能同时成立。若不能同时成立，就停止把 prior 当作在线辅助器，只保留离线诊断价值。

## 下一步

先跑 MipNeRF360 的小 pilot，记录每 2k iteration 的 render / metric / Gaussian 数量，再决定是否扩到全 9 场景。

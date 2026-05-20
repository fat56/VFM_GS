# 0006 Validation Selector + Online Depth Residual

## 背景

0002 / 0004 已经把固定 Depth Anything prune-protect 的主要边界暴露出来：

- `depth_auto_topk` 在 MipNeRF360 全 9 场景有很弱正向，但 DB/Tandt 交叉验证基本贴近 0，不能默认化。
- `start24000` 证明 late timing 是有效变量，但固定 `weight` / `topk` 仍会在 `room/bonsai/kitchen/counter` 之间互相伤害。
- 继续手工扫固定阈值的价值变低，更值得验证“按场景选择”和“让 prior 只响应当前模型残差”两条线。

这也对应 FastGS 相关工作的迭代脉络：3DGS 给出可训练显式表示，FastGS / Abs-GS 关注更敏感的增长与剪枝信号，Taming-3DGS / Speedy-Splat 关注容量和速度约束。我们现在的问题不是缺一个更强的静态 prior，而是缺一个能在不同场景、不同训练阶段自动降级或退出的策略。

## 路径 A：Validation-Driven Selector

### 假设

Depth prior 不是全局默认项，但可能是按场景可选项。用少量训练视角作为 validation proxy，给每个场景在 `baseline`、`depth_auto_topk`、`depth_topk010`、`late-start` 等候选之间做选择，应该比固定单一配置更稳。

### 第一轮候选

候选表由 `scripts/build_0006_selector_candidates.py` 从既有输出规整生成：

- MipNeRF360：baseline、Depth Anything auto-topk、Depth Anything topk010、RGB rerank start9000、0004 late-start 两个 partial 变体。
- DB/Tandt：baseline、auto-topk、topk010、以及已有 partial topk005 / weight0.15 topk010。

重要约定：FastGS big 基线统一重命名为 `baseline`，这样 `scripts/evaluate_0001_train_selector.py` 可以正确计算 QCGI selector。

### 成功标准

- `train_qcgi` selector 的测试集均值优于 baseline，且跨 MipNeRF360 与 DB/Tandt 不出现明显退化。
- 负向场景能回退到 `baseline`，而不是强行选择 prior 分支。
- 若 `train_best_psnr` 与 `train_qcgi` 分歧明显，以 `train_qcgi` 为主，因为它显式惩罚 Gaussian 增长。

## 路径 B：Online Depth Residual

### 假设

当前 `depth_anything_depth_prior` 本质是静态图像 prior：它只知道 Depth Anything 的单图深度/边缘，不知道当前 Gaussian 模型已经解释了什么。因此它容易把“先验强响应”误当成“当前模型需要保护/增长”的证据。

更合理的信号是 online residual：

```text
depth_residual = normalize(rendered_depth_current) - normalize(depth_anything_depth)
```

然后只在 late pruning 的 RGB candidate 内使用这个 residual 做 protect / rerank。这样 prior 的作用从“静态告诉模型哪里重要”降级为“当前模型和深度先验不一致时，提醒剪枝谨慎”。

### 最小实现路线

1. 先做不改 CUDA 的 smoke：用当前 Gaussian 中心投影构造近似 rendered-depth proxy，验证 residual 是否和失败场景相关。
2. 若 smoke 正向，再给 rasterizer 增加真正的 alpha-weighted depth 输出，替换 proxy。
3. 只从 late prune-protect 接入，不接 densification，避免回到 0002 早期 prior 主导增长的问题。

### 第一批场景

- `room`：0004 中对 timing 最敏感，用来检查 residual 是否避免 early/late 误伤。
- `kitchen` / `bonsai`：固定 topk 的负例，用来验证 residual 是否能少保护错误候选。
- `stump` / `counter`：曾经有小正向，用来检查新信号是否保留正例。

## 决策规则

- 先启动路径 A 的双卡 selector 评估，因为它直接复用已有 checkpoint，产出最快。
- 路径 B 不先大改 CUDA；先做 proxy smoke 和文档化失败/成功判据，再决定是否进入 renderer 深度输出改造。
- 每一轮实验结束后更新 `results.md` / `review.md`，随后 commit 并 push。

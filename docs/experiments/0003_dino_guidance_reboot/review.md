# 0003 复盘

## 当前判断

0001 最大的问题不是“DINO 完全无效”，而是证据链不够闭合：它证明了 DINO 介入 densification 后训练结果能变好，却没有充分证明 DINO metric map 与当前 FastGS 的重建瓶颈对齐。

这会带来两种混淆：

- 如果 DINO top-k 区域和 RGB 高误差区域不重叠，质量提升可能来自训练轨迹、容量变化或间接正则，而不是精准结构引导。
- 如果 DINO token 太粗，再上采样成像素级 map，Gaussian 复制会落在大块区域内，无法证明引导在结构边界附近精确生效。

## 0001 Token 粒度问题

0001 descriptor 主线实际使用的 token grid 是 `10x16` 量级。它在 `-r 8` 图像上已经偏粗，在 high-res 1.6K 复验中则明显过粗。`vfm_descriptor_token_smooth_kernel=3` 会在 `10x16` grid 上进一步平滑，相当于把少数 patch 的响应扩成大块区域。

这解释了为什么 top-k25 会带来全图轻微正向，但场景间容量/QCGI 不稳定：它可能在“粗结构区域”增加了 Gaussian，而不是在真正的当前误差瓶颈处做精细复制。

## DINO 特性重新理解

DINOv2 patch token 的优势是语义/结构一致性和一定程度的光照、纹理不变性；弱点是它不天然等价于 RGB 重建误差。对 Gaussian Splatting 来说，训练主损失仍是 photometric/SSIM，DINO 如果不被当前误差锚定，就可能去强调“语义上重要但已经重建得还可以”的区域。

因此 0003 的 DINO 用法应从：

```text
DINO says important -> densify
```

改为：

```text
RGB/SSIM says still bad AND DINO says structurally meaningful -> densify
```

## 优先修改方向

1. 先导出真实 DINO residual map  
   不能用 descriptor cache 的 channel norm 替代训练时的 cosine residual。应读取 baseline render 和 GT，复现 `dinov2_descriptor_cosine` 的 patch error，再做 overlap 诊断。

2. 使用高分辨率 patch tokens  
   high-res 实验至少要用 `max_width=1600` 的 patch-token cache 做小范围验证。ViT-S/14 高分辨率 patch tokens 存储较大，但在少数场景上可接受；如果全量存储压力太大，可以先只导出 residual/prior 2D map。

3. RGB-gated DINO  
   第一版训练候选应是 DINO residual 与 RGB error 的乘积或交集，而不是裸 DINO top-k。这样能让 DINO 指导“哪些高误差区域值得复制”，而不是单独决定复制区域。

4. Patch-aware map  
   初始阶段至少改成 nearest upsample 或 token-cell mask，避免双线性插值制造虚假的亚 token 精度。后续可改为 token-grid 级 Gaussian visibility 聚合。

5. 局部指标必须入表  
   每轮除了 PSNR/SSIM/LPIPS，还要报告 DINO top-k 区域、RGB 高误差区域和二者交集区域的 L1/LPIPS 改善。

## 对 0001 的重新定位

0001 可以保留为“DINO 介入 densification 有正向潜力”的工程证据，但不应继续作为“DINO metric map 已经有效定位结构瓶颈”的证据。0003 需要重建这个中间环节。

## 下一步

实现 true descriptor residual overlap 诊断，并在 `bicycle` 上比较 `max_width=224`、`518` 和 `1600` 三种 token 粒度。

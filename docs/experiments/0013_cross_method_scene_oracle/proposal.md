# 0013 Cross-Method Scene Oracle

## 核心问题

0002、0009、0010、0011 分别给出了 Depth Anything、residual orientation、DINO descriptor 的不同接入方式，但单个固定方法都没有稳定默认化。本实验不新增训练，只把已有方法放到同一 QCGI 口径下：

> 如果允许按场景从已有方法中选择，prior 接入路线还有多大上界？

## 输入方法

- Phase0 FastGS big baseline
- 0002 Depth Anything prune-protect auto-topk
- 0009 residual orientation protect
- 0010 DINO descriptor i0.50 full
- 0010 DINO descriptor i0.50 until8000
- 0011 DINO descriptor i0.50 until4000

## 判定

若 cross-method oracle 仍只带来薄收益，说明问题不是某个单方法阈值没调好，而是当前 prior 接入路线整体上界有限。

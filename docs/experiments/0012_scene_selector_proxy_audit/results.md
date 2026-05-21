# 0012 实验结果

## 当前状态

离线 selector proxy audit 已完成。结论：descriptor scene oracle 有小幅上界，但 Phase0-only 的低成本单阈值 selector 在 LOOCV 下失败；能转正的 proxy 都依赖额外 descriptor 响应（full i0.50 或 until4000），本身不够便宜。因此暂不进入复杂 selector 训练，DINO descriptor direct 1.6K 主线应收束。

输出：

- `output/0012/scene_selector_proxy_audit/feature_table.csv`
- `output/0012/scene_selector_proxy_audit/policy_summary.csv`
- `output/0012/scene_selector_proxy_audit/loocv_predictions.csv`
- `output/0012/scene_selector_proxy_audit/summary_stats.json`

| 策略 | 启用场景 | ΔPSNR | ΔSSIM | ΔLPIPS | ΔGS | QCGI |
|---|---|---:|---:|---:|---:|---:|
| all until8000 | bicycle/flowers/garden/stump/treehill/room/counter/kitchen/bonsai | +0.0156 | +0.0009 | -0.0018 | +33,372 | -0.0171 |
| oracle until8000 positive | bicycle/flowers/stump/counter/kitchen | +0.0323 | +0.0008 | -0.0015 | +21,925 | +0.0336 |
| full i0.50 QCGI sign | bicycle/counter/kitchen | +0.0246 | +0.0005 | -0.0009 | +14,192 | +0.0246 |
| until4000 QCGI sign | bicycle/garden/stump/counter/kitchen | +0.0207 | +0.0008 | -0.0016 | +17,856 | +0.0212 |
| best Phase0 curve threshold, in-sample | bicycle/flowers/stump/treehill/room/counter/kitchen | +0.0274 | +0.0008 | -0.0014 | +21,227 | +0.0257 |
| Phase0 curve single-threshold LOOCV | bicycle/flowers/garden/stump/treehill/room/bonsai | -0.0068 | +0.0008 | -0.0013 | +21,535 | -0.0336 |

LOOCV 失败点：

- `garden/treehill/room/bonsai` 这 4 个负例里，Phase0-only 单阈值误开了全部 4 个。
- `counter/kitchen` 这 2 个正例被误关。
- 说明 Phase0 checkpoint curve 的简单形态不足以预测 DINO descriptor early-window 是否有益。

判断：

- Oracle 上界存在，但幅度只有 +0.0336 QCGI；真实 selector 必须极低成本。
- `full_i050_qcgi_positive` 与 `until4000_qcgi_positive` 能转正，但它们依赖额外 descriptor 训练结果，成本接近再跑一个变体，不适合作为默认化机制。
- 当前证据不支持继续在 DINO descriptor direct 1.6K 上投入复杂 selector；更合理的是把 0010/0011/0012 作为 direct comparison 收束证据，回到其他更稳的方向。

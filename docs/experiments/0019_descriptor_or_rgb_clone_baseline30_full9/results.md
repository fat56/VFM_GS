# 0019 Results

## 启动记录

目标：在标准 FastGS 30K full9 流程中，只把 DINO descriptor 作为 0-15K clone 的额外 OR 触发条件；split、densify-stage prune 和 15K 后 final prune 均保持 RGB/FastGS 逻辑。

状态：2026-05-25 21:41:49 CST 已启动双卡 tmux full9。

预估完成：2026-05-25 22:10-22:20 CST。该估计基于 FastGS 30K checkpoint-curve 历史耗时、当前双卡分配和 0-15K descriptor OR 额外 scorer 开销；render/metrics 完成后 merge 会话会自动合并。

会话：

```text
0019_desc_or_rgb_clone_g0
0019_desc_or_rgb_clone_g1
0019_desc_or_rgb_clone_merge
```

启动前检查：

- `python -m py_compile` 通过。
- `git diff --check` 通过。
- `bicycle` 700-step smoke 通过，并触发了真实 descriptor/RGB clone OR densification。

配置与脚本：

```text
configs/experiments/0019_descriptor_or_rgb_clone_baseline30_full9.yaml
scripts/run_0019_descriptor_or_rgb_clone_baseline30_full9_eval.py
scripts/run_0019_descriptor_or_rgb_clone_baseline30_full9_tmux.sh
```

评测点：

```text
15K / 20K / 25K / 30K
```

输出位置：

```text
output/0019/descriptor_or_rgb_clone_baseline30_full9/mipnerf360_combined
```

## 完成结果

完整性：

- `summary_rows`: 36 / 36。
- `comparison_15k_rows`: 27 / 27。
- `baseline_curve_rows`: 18 / 18。
- `baseline30_rows`: 9 / 9。

30K 相对 FastGS 30K baseline 的 full9 平均 delta：

| metric | delta |
| --- | ---: |
| PSNR | -0.0253 |
| SSIM | +0.001414 |
| LPIPS | -0.002948 |
| Gaussian count | +48,510 |

逐场景 30K delta：

| scene | dPSNR | dSSIM | dLPIPS | dGS |
| --- | ---: | ---: | ---: | ---: |
| `stump` | +0.0676 | +0.004101 | -0.009399 | +161,073 |
| `bicycle` | +0.0390 | +0.004584 | -0.006939 | +46,011 |
| `bonsai` | +0.0360 | -0.000106 | -0.000325 | +10,903 |
| `counter` | +0.0306 | +0.000312 | -0.000215 | +5,246 |
| `flowers` | +0.0271 | +0.001422 | -0.000025 | +95,250 |
| `kitchen` | +0.0038 | -0.000017 | -0.000098 | -2,124 |
| `treehill` | +0.0026 | +0.004820 | -0.010017 | +112,705 |
| `garden` | -0.1367 | -0.000534 | -0.000027 | +423 |
| `room` | -0.2978 | -0.001855 | +0.000511 | +7,103 |

结论：

- 不是稳健全局提升：full9 平均 PSNR 被 `room` 和 `garden` 明显拉负。
- 5 / 9 个场景三项指标同时正向：`bicycle`、`flowers`、`stump`、`treehill`、`counter`。
- 比较明确的正例是 `stump`、`bicycle`、`treehill`。其中 `treehill` 的 PSNR 基本持平，但 SSIM/LPIPS 提升很明显。
- `room` 是最大负例，三项指标都退化；`garden` 主要是 PSNR/SSIM 退化，LPIPS 只有极小正向。
- 去掉 `room` 后，剩余 8 场景平均为 PSNR +0.0087、SSIM +0.001823、LPIPS -0.003381；再去掉 `garden` 后，剩余 7 场景平均为 PSNR +0.0295、SSIM +0.002159、LPIPS -0.003860。

20K/30K 相对 FastGS checkpoint curve 的变化：

| scene | 20K dPSNR / dSSIM / dLPIPS | 30K dPSNR / dSSIM / dLPIPS |
| --- | ---: | ---: |
| `bicycle` | +0.0326 / +0.004573 / -0.006833 | +0.0390 / +0.004584 / -0.006939 |
| `flowers` | +0.0331 / +0.001356 / +0.000226 | +0.0271 / +0.001422 / -0.000025 |
| `garden` | -0.0735 / -0.000246 / -0.000257 | -0.1367 / -0.000534 / -0.000027 |
| `stump` | +0.0735 / +0.004252 / -0.009138 | +0.0676 / +0.004101 / -0.009399 |
| `treehill` | +0.0229 / +0.005299 / -0.009992 | +0.0026 / +0.004820 / -0.010017 |
| `room` | -0.2864 / -0.001523 / +0.000441 | -0.2978 / -0.001855 / +0.000511 |
| `counter` | +0.0374 / +0.000266 / -0.000102 | +0.0306 / +0.000312 / -0.000215 |
| `kitchen` | +0.0025 / +0.000036 / -0.000209 | +0.0038 / -0.000017 / -0.000098 |
| `bonsai` | +0.0541 / -0.000015 / -0.000379 | +0.0360 / -0.000106 / -0.000325 |

Interpretation: 0019 更像是场景依赖的 clone-recall 增强，而不是可直接替代 FastGS 30K baseline 的稳健配置。收益集中在部分 outdoor/structure-heavy 场景，并且通常伴随更多 Gaussian；`room`/`garden` 说明 descriptor OR clone 仍可能引入对最终 PSNR 不利的点。

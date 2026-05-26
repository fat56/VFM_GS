# 0020 Results

## 启动记录

目标：在 0019 的 descriptor OR clone 基础上加入保护，只让 descriptor 作为少量 clone rescue 候选，验证是否能保留正例收益并修复 `room/garden` 退化。

启动：2026-05-25 23:23:24 CST 双卡 tmux 四场景 pilot。

完成：2026-05-25 23:46:25 CST merge 自动合并完成。

启动前检查：

- `python -m py_compile` 通过。
- `bash -n scripts/run_0020_descriptor_rescue_clone_guarded_pilot_tmux.sh` 通过。
- `git diff --check` 通过。
- `bicycle` 3.6K smoke 通过，覆盖 `3K-12K` rescue gate/cap 生效窗口并正常保存 checkpoint。

会话：

```text
0020_desc_rescue_g0
0020_desc_rescue_g1
0020_desc_rescue_merge
```

场景分配：

```text
GPU0: garden room
GPU1: stump treehill
```

配置：

```text
configs/experiments/0020_descriptor_rescue_clone_guarded_pilot.yaml
```

脚本：

```text
scripts/run_0020_descriptor_rescue_clone_guarded_pilot_tmux.sh
scripts/run_0019_descriptor_or_rgb_clone_baseline30_full9_eval.py
```

核心保护：

- descriptor-only clone 生效窗口：`3K-12K`。
- 弱 RGB gate：`rgb_score > 0.5 * densify_metric_thresh`。
- descriptor-only clone cap：每轮最多为 RGB clone 候选数的 `20%`。
- split 与 densify-stage prune 继续使用 RGB/FastGS 逻辑。

场景：

```text
garden room stump treehill
```

评测点：

```text
15K / 20K / 25K / 30K
```

输出位置：

```text
output/0020/descriptor_rescue_clone_guarded_pilot/mipnerf360_combined
```

## Pilot 完整性

四场景 pilot 当时合并输出完整；后续 full9 复验使用同一路径重新 merge，当前 `mipnerf360_combined` 已是 full9 36 行结果。

| table | rows | expected |
| --- | ---: | ---: |
| `summary` | 16 | 16 |
| `comparison_15k_to_eval` | 12 | 12 |
| `comparison_vs_baseline_curve` | 8 | 8 |
| `comparison_vs_baseline30` | 4 | 4 |

当前 full9 主要产物：

```text
output/0020/descriptor_rescue_clone_guarded_pilot/mipnerf360_combined/summary.csv
output/0020/descriptor_rescue_clone_guarded_pilot/mipnerf360_combined/comparison_15k_to_eval.csv
output/0020/descriptor_rescue_clone_guarded_pilot/mipnerf360_combined/comparison_vs_baseline_curve.csv
output/0020/descriptor_rescue_clone_guarded_pilot/mipnerf360_combined/comparison_vs_baseline30.csv
```

## 30K 相对 FastGS 30K baseline

四场景平均：

| metric | delta |
| --- | ---: |
| PSNR | +0.0147 |
| SSIM | +0.000732 |
| LPIPS | -0.001933 |
| Gaussian count | +21,649 |

逐场景：

| scene | dPSNR | dSSIM | dLPIPS | dGS |
| --- | ---: | ---: | ---: | ---: |
| `garden` | -0.0066 | +0.000197 | -0.000256 | +11,802 |
| `room` | -0.0036 | -0.000009 | -0.000496 | +5,121 |
| `stump` | +0.0333 | +0.000856 | -0.002493 | +34,185 |
| `treehill` | +0.0357 | +0.001886 | -0.004489 | +35,488 |

解读：

- `room/garden` 的 0019 大幅 PSNR 退化基本被修复：30K PSNR 只剩 -0.0036 / -0.0066，SSIM/LPIPS 已接近持平或小幅正向。
- `stump/treehill` 仍保持 30K 三项正向，尤其 `treehill` 的 PSNR 从 0019 的近持平变为 +0.0357。
- 但正例的 perceptual/structure 收益被保护策略明显削弱：`stump/treehill` 平均 SSIM/LPIPS 收益低于 0019。

## 20K/30K 相对 checkpoint curve

| iteration | avg dPSNR | avg dSSIM | avg dLPIPS | avg dGS |
| ---: | ---: | ---: | ---: | ---: |
| 20K | +0.0312 | +0.000993 | -0.002033 | +22,002 |
| 30K | +0.0147 | +0.000732 | -0.001933 | +21,649 |

逐场景 20K/30K：

| scene | 20K dPSNR / dSSIM / dLPIPS | 30K dPSNR / dSSIM / dLPIPS |
| --- | ---: | ---: |
| `garden` | +0.0379 / +0.000480 / -0.000576 | -0.0066 / +0.000197 / -0.000256 |
| `room` | +0.0252 / +0.000215 / -0.000490 | -0.0036 / -0.000009 / -0.000496 |
| `stump` | +0.0302 / +0.000926 / -0.002193 | +0.0333 / +0.000856 / -0.002493 |
| `treehill` | +0.0314 / +0.002351 / -0.004873 | +0.0357 / +0.001886 / -0.004489 |

20K 时四场景全部正向；到 30K，`garden/room` 的 PSNR 回落到接近 baseline，但没有复现 0019 的大幅负向。说明 0020 的 rescue 主要改善 early/mid trajectory，最终收益在负例上被 20K-30K 正常收敛部分吸收。

## 15K 到评测点

相对本实验 15K checkpoint 的平均变化：

| target | avg dPSNR | avg dSSIM | avg dLPIPS | avg dGS |
| ---: | ---: | ---: | ---: | ---: |
| 20K | +0.3525 | +0.005748 | -0.011293 | -193,820 |
| 25K | +0.3931 | +0.006215 | -0.013215 | -212,827 |
| 30K | +0.4156 | +0.006412 | -0.014239 | -218,606 |

15K 之后 final-prune tail 正常降低点数并继续提升质量；0020 没有出现 descriptor rescue 后期拖累收敛的迹象。

## 与 0019 同四场景对照

30K 相对 FastGS 30K baseline：

| group | experiment | dPSNR | dSSIM | dLPIPS | dGS |
| --- | --- | ---: | ---: | ---: | ---: |
| all 4 | 0019 OR clone | -0.0911 | +0.001633 | -0.004733 | +70,326 |
| all 4 | 0020 guarded rescue | +0.0147 | +0.000732 | -0.001933 | +21,649 |
| `garden/room` | 0019 OR clone | -0.2172 | -0.001195 | +0.000242 | +3,763 |
| `garden/room` | 0020 guarded rescue | -0.0051 | +0.000094 | -0.000376 | +8,462 |
| `stump/treehill` | 0019 OR clone | +0.0351 | +0.004460 | -0.009708 | +136,889 |
| `stump/treehill` | 0020 guarded rescue | +0.0345 | +0.001371 | -0.003491 | +34,837 |

0020 相对 0019 的变化：

| group | dPSNR change | dSSIM change | dLPIPS change | dGS change |
| --- | ---: | ---: | ---: | ---: |
| all 4 | +0.1058 | -0.000901 | +0.002800 | -48,677 |
| `garden/room` | +0.2121 | +0.001289 | -0.000618 | +4,699 |
| `stump/treehill` | -0.0006 | -0.003090 | +0.006217 | -102,053 |

结论很明确：

- 保护策略有效修复 0019 的负例。`room` 从 -0.2978 PSNR 拉回 -0.0036，`garden` 从 -0.1367 拉回 -0.0066。
- 四场景平均 PSNR 从 0019 的 -0.0911 变为 +0.0147，同时 Gaussian 增量从 +70,326 降到 +21,649。
- 正例不是完全保留。`stump/treehill` 的 PSNR 平均几乎持平于 0019，但 SSIM/LPIPS 收益只保留了约三分之一，说明 `rgb_gate=0.5 + cap=20% + 3K-12K` 对 descriptor-only rescue 偏保守。

## Pilot 判定

- Pilot 通过安全性门槛：四场景平均 30K PSNR 转正，`room/garden` 不再明显负向，Gaussian 增量低于 0019。
- 不足以直接默认化：正例的 SSIM/LPIPS 强收益被 cap/gate/window 明显削弱，当前配置更像“安全版 descriptor clone recall”，不是 0019 正例收益的完整替代。
- 因此进入 full9 复验，但保持 pilot 性质：若 full9 仍能维持平均 PSNR 非负、`room/garden` 近零、且不新增负例，可把 guarded rescue 作为候选分支；若 full9 平均被其它场景拉负，则停止默认化，只保留为场景自适应或 selector 候选。

## Full9 复验

启动：2026-05-26 10:34:00 CST 双卡 tmux full9。

完成：2026-05-26 11:13 CST 前后 merge 自动合并完成。

场景分配：

```text
GPU0: garden bicycle flowers room
GPU1: kitchen bonsai stump treehill counter
```

说明：full9 复验复用四场景 pilot 的已有产物，补跑 `bicycle`、`flowers`、`kitchen`、`bonsai`、`counter` 五个新场景后重新合并。

完整性：

| table | rows | expected |
| --- | ---: | ---: |
| `summary` | 36 | 36 |
| `comparison_15k_to_eval` | 27 | 27 |
| `comparison_vs_baseline_curve` | 18 | 18 |
| `comparison_vs_baseline30` | 9 | 9 |

## Full9 30K 相对 FastGS 30K baseline

full9 平均：

| metric | delta |
| --- | ---: |
| PSNR | -0.0146 |
| SSIM | +0.000545 |
| LPIPS | -0.001143 |
| Gaussian count | +15,141 |

逐场景：

| scene | dPSNR | dSSIM | dLPIPS | dGS |
| --- | ---: | ---: | ---: | ---: |
| `bicycle` | +0.0237 | +0.001966 | -0.002347 | +11,642 |
| `flowers` | -0.0356 | +0.000302 | -0.000390 | +25,569 |
| `garden` | -0.0066 | +0.000197 | -0.000256 | +11,802 |
| `stump` | +0.0333 | +0.000856 | -0.002493 | +34,185 |
| `treehill` | +0.0357 | +0.001886 | -0.004489 | +35,488 |
| `room` | -0.0036 | -0.000009 | -0.000496 | +5,121 |
| `counter` | +0.0209 | +0.000285 | -0.000532 | +3,835 |
| `kitchen` | -0.3100 | -0.001347 | +0.001454 | +24 |
| `bonsai` | +0.1110 | +0.000769 | -0.000740 | +8,600 |

Full9 结论：

- 5 / 9 场景 PSNR 正向：`bicycle`、`stump`、`treehill`、`counter`、`bonsai`；`garden/room` 接近持平但略负。
- `kitchen` 是决定性失败场景：30K dPSNR -0.3100、dSSIM -0.001347、dLPIPS +0.001454，且 Gaussian count 基本不变，仅 +24。该退化不是容量膨胀导致，而是 clone-rescue 信号本身扰动了收敛。
- `flowers` 也出现轻度 PSNR 负向：dPSNR -0.0356，但 SSIM/LPIPS 仍小幅正向。
- 如果去掉 `kitchen`，剩余 8 场景平均为 dPSNR +0.0223、dSSIM +0.000781、dLPIPS -0.001468、dGS +17,030，说明 full9 的负向主要由 `kitchen` 单场景触发。

## Full9 与 0019 对照

30K 相对 FastGS 30K baseline：

| group | experiment | dPSNR | dSSIM | dLPIPS | dGS |
| --- | --- | ---: | ---: | ---: | ---: |
| full9 | 0019 OR clone | -0.0253 | +0.001414 | -0.002948 | +48,510 |
| full9 | 0020 guarded rescue | -0.0146 | +0.000545 | -0.001143 | +15,141 |
| full9 without `kitchen` | 0019 OR clone | -0.0290 | +0.001593 | -0.003305 | +54,839 |
| full9 without `kitchen` | 0020 guarded rescue | +0.0223 | +0.000781 | -0.001468 | +17,030 |
| new 5 scenes | 0019 OR clone | +0.0273 | +0.001239 | -0.001520 | +31,057 |
| new 5 scenes | 0020 guarded rescue | -0.0380 | +0.000395 | -0.000511 | +9,934 |
| `garden/room` | 0019 OR clone | -0.2172 | -0.001195 | +0.000242 | +3,763 |
| `garden/room` | 0020 guarded rescue | -0.0051 | +0.000094 | -0.000376 | +8,462 |
| `stump/treehill` | 0019 OR clone | +0.0351 | +0.004460 | -0.009708 | +136,889 |
| `stump/treehill` | 0020 guarded rescue | +0.0345 | +0.001371 | -0.003491 | +34,837 |

0020 相对 0019 的 full9 变化：

| group | dPSNR change | dSSIM change | dLPIPS change | dGS change |
| --- | ---: | ---: | ---: | ---: |
| full9 | +0.0107 | -0.000869 | +0.001805 | -33,369 |
| full9 without `kitchen` | +0.0513 | -0.000812 | +0.001837 | -37,809 |
| new 5 scenes | -0.0653 | -0.000844 | +0.001010 | -21,123 |
| `garden/room` | +0.2121 | +0.001289 | -0.000618 | +4,699 |
| `stump/treehill` | -0.0006 | -0.003090 | +0.006217 | -102,053 |

相对 0019，0020 明显更省点，也修复了 `garden/room`。但它没有解决 full9 默认化问题：full9 平均 PSNR 仍为负，且 `kitchen` 从 0019 的近中性 `+0.0038` 变成 0020 的 `-0.3100`。

## Full9 20K/30K 相对 checkpoint curve

| iteration | avg dPSNR | avg dSSIM | avg dLPIPS | avg dGS |
| ---: | ---: | ---: | ---: | ---: |
| 20K | -0.0014 | +0.000688 | -0.001215 | +15,423 |
| 30K | -0.0146 | +0.000545 | -0.001143 | +15,141 |

20K 已经被 `kitchen` 拉到接近零，30K 继续转负。说明 `kitchen` 的退化不是 20K-30K 后期才出现，而是在 rescue window 后的 continuation 中持续存在。

## 最终决策

- 不默认化 0020 guarded descriptor rescue。Full9 平均 PSNR 仍为 -0.0146，未通过“平均 PSNR 非负且不新增明显负例”的门槛。
- 保留为候选分支或 selector 特征。它对 `garden/room` 的修复非常明确，且大幅降低 0019 的 Gaussian 增量，但必须能避开 `kitchen` 这类失败场景。
- 不继续做单纯 gate/cap/window 的全局调参。当前结果说明全局保护能压住旧负例，却会暴露新负例；下一步若继续，应优先做 scene-adaptive 触发或训练期 fingerprint，而不是再把 guarded rescue 直接扫 full9。

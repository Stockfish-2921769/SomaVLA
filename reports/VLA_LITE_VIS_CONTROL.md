# SomaVLA — VLA-lite vis 对齐对照：场的贡献是否 NCA 独有？

> 日期：2026-11-10
> 状态：实验完成；分支 `fix/vla-lite-g2-design`（基础 commit `465e51f`），未合并 main
> 关联：`VLA_LITE_PILOT.md`（试点主结果与协议）、`VLA_LITE_TASK.md` §13.4（vis 对齐定义）
> 代码：`soma/vla_lite/policy.py`、`scripts/train_vla_lite_low.py`、`scripts/eval_vla_lite.py`、
> `scripts/analyze_vla_lite_5seed.py`
>
> **结论先行**：在与 NCA **相同的 77 通道 ext 输入、32 通道场接口、readout/decoder、
> plan 监督与优化器配置**下，一个非迭代的 1×1 conv 场（仅 2,496 个臂专属参数）相对无场
> 仍有 **+13.5pp、5/5 seed 为正**（池化 95% CI [+9.9, +17.1]）。NCA 相对该 vis 控制的
> 增量为 **+8.1pp（3/5 seed，seed 层 t=1.30）**，方向为正但未达 seed 层面显著。
> 因此：**场路径本身的收益已可确认，但“NCA 迭代空间计算”的增量尚不能作为已验证的结构优势**；
> 12 层 CNN 与 1×1 conv 打平，说明单纯增加空间层数不是收益来源。

---

## 0. 摘要

| 问题 | 结论 | 证据强度 |
|---|---|---|
| 对齐接口后场还有用吗？ | 有，vis 场 +13.5pp | 强：5/5 seed，池化 CI 不含 0 |
| 收益是 NCA 独有的吗？ | 否，大部分可由同接口 1×1 conv 获得 | 强：vis 回收 NCA−none 的约 62% |
| NCA 额外贡献多少？ | +8.1pp，方向为正 | 弱：3/5 seed，t=1.30，主要来自 s4 |
| 更多空间层数（CNN）有优势吗？ | 没有，与 vis 打平（−2.6pp） | 中：2/5 seed，CI 跨 0 |
| 能否做容量归因？ | 不能，vis 与 NCA 参数量差 43× | — 需 K=1 消融 |

---

## 1. 背景与设计

试点报告（`VLA_LITE_PILOT.md`）确认 NCA 场相对无场有稳定收益，但 NCA−CNN 的
结构对比被 seed 方差（±15–25pp）与初始化混杂限制。为判断收益是否来自“场的接口/
监督路径”而非“NCA 迭代计算”，任务书 §13.4 定义了 vis 对齐对照：

- **输入**：与 NCA 完全相同的 77 通道 ext（适配器 64ch 特征 + 障碍概率 + 目标 one-hot +
  起点 one-hot + 任务槽）。
- **场生成**：单个 1×1 conv（77→32），逐像素线性变换，**无跨像素传播、无迭代**。
- **场接口**：输出 32 通道场，与 NCA 输出同形状，进入同一个 FieldReadout（32→256，
  16×16 token）与同一个 flow-matching 解码器。
- **监督**：同一动作 flow-matching 损失 + 同 plan 监督（`plan_weight=0.1`，
  vis 的通道 0 作为 plan）+ 同障碍损失；同 field-dropout 0.1。
- **优化器**：vis 专属参数使用 `lr_nca=1e-3`，其余共享模块分组与 NCA 相同。

臂专属参数量：**vis 2,496 / NCA 106,688 / 12 层 CNN 135,008**。
这是一个“同接口、无迭代、极小容量”的控制，不是严格等容量控制（见 §6）。

### 1.1 训练与评估

- 5 个训练 seed（0–4），每 seed 20,000 step、batch 64、bf16、AdamW，数据与场景划分
  与主试点一致（450/75/75，25,004 chunk）。
- 评估：200 个外部场景（scene seed 777），所有臂共享同一个预测高层 `high_s0`，
  det-noise、exec 4、`max_steps=1500`、`visit_cap=180`、grid 32、K=12。
- 所有配对均在**同一场景索引**上逐场景比较；统计见 §3。

### 1.2 同码重评

vis 由分支代码训练与评估。为使对比不受主干/分支评估代码差异影响，none/cnn/nca 的
15 个 checkpoint 全部用**同一分支评估代码**重评（`recheck_*`），最终表只用重评结果。
vis 的评估文件为 `conf_vis_*`。

旧评估与重评的成功数差异：

| 臂 | 逐 seed 差值（重评 − 旧） | 说明 |
|---|---|---|
| none | 0 / 0 / 0 / 0 / 0 | 逐场景完全一致 |
| cnn | 0 / 0 / +1 / 0 / 0 | 1 个场景，≈0.5pp |
| nca | −4 / −4 / +3 / +1 / −1 | ≤2pp，bf16 kernel 非确定性 |

因此评估本身的运行噪声约为 ±0.5–2pp；臂间比较统一使用重评集。

---

## 2. 主结果（同码，5 训练 seed × 200 场景）

成功率（%，3/3 placed）：

| 臂 | s0 | s1 | s2 | s3 | s4 | 均值 [min,max] |
|---|---|---|---|---|---|---|
| none | 25.5 | 33.0 | 20.5 | 14.5 | 28.0 | 24.3 [14.5,33.0] |
| cnn | 28.0 | 24.0 | 22.0 | 48.0 | 54.0 | 35.2 [22.0,54.0] |
| **vis（对齐 1×1 conv，2,496 参）** | **39.0** | **56.0** | **24.5** | **33.0** | **36.5** | **37.8 [24.5,56.0]** |
| nca | 40.5 | 55.0 | 36.0 | 30.5 | 67.5 | 45.9 [30.5,67.5] |

逐场景配对统计（精确 McNemar；seed 层 se 为 5 个 seed 差值的标准误；
池化 CI 为 seed 分层、20,000 次场景 bootstrap）：

| 对比 | 均值 | 方向 | seed se / t | 池化 95% CI | 逐 seed 差值（pp） |
|---|---|---|---|---|---|
| **vis − none** | **+13.5pp** | **5/5 正** | 3.4 / 3.98 | **[+9.9, +17.1]** | +13.5 / +23.0 / +4.0 / +18.5 / +8.5 |
| nca − vis | +8.1pp | 3/5 正 | 6.2 / 1.30 | [+4.3, +11.9] | +1.5 / −1.0 / +11.5 / −2.5 / +31.0 |
| cnn − vis | −2.6pp | 2/5 正 | 9.1 / −0.29 | [−6.5, +1.2] | −11.0 / −32.0 / −2.5 / +15.0 / +17.5 |
| nca − none | +21.6pp | 5/5 正 | 4.7 / 4.64 | [+17.9, +25.3] | +15.0 / +22.0 / +15.5 / +16.0 / +39.5 |
| cnn − none | +10.9pp | 4/5 正 | 8.0 / 1.36 | [+7.2, +14.6] | +2.5 / −9.0 / +1.5 / +33.5 / +26.0 |
| nca − cnn | +10.7pp | 4/5 正 | 7.8 / 1.36 | [+7.1, +14.5] | +12.5 / +31.0 / +14.0 / −17.5 / +13.5 |

逐 seed McNemar p（vis−none）：`.002 / .000 / .374 / .000 / .068`；
nca−vis：`.810 / .913 / .013 / .649 / .000`。

---

## 3. 解读与归因分解

1. **场收益不是 NCA 独有。** 对齐输入、场接口、readout、解码器、plan 监督与学习率后，
   1×1 conv 场仍有 +13.5pp，且 5/5 seed 为正。共享路径（结构化 ext 输入 + plan 监督 +
   32 通道场读入）本身即可获得大部分收益。
2. **NCA 增量方向为正但不稳健。** 均值 +8.1pp，但 3/5 seed 为正，seed 层 t=1.30
   （df=4，5% 临界≈2.78）；增量主要来自 s4（+31.0pp）与 s2（+11.5pp），其余 3 个 seed
   在 ±2.5pp 内。按主协议，不能宣称 NCA 结构优于同接口控制。
3. **CNN 与 vis 打平。** −2.6pp、2/5、CI [−6.5, +1.2]；12 层 CNN 在 s0/s1 大幅落后
   （−11/−32），在 s3/s4 领先（+15/+17.5），呈 seed 依赖。单纯增加空间层数没有稳定收益。
4. **分解**：NCA−none +21.6pp ≈ vis−none +13.5pp（对齐路径）+ nca−vis +8.1pp
   （迭代计算/容量）。vis 回收约 **62%** 的 NCA 总增益。

---

## 4. 与试点报告的关系

- 试点报告的核心可声称项“NCA 场显著优于无场（5/5 seed）”在同码重评下复现：
  +21.6pp，5/5，CI [+17.9, +25.3]（旧评估为 +22.1pp，CI [+18.5, +25.7]）。
- 试点报告“不能声称 NCA 结构优于 CNN”（+11.3pp，t≈1.4）在同码重评下方向不变
  （+10.7pp，t=1.36），仍不可声称。
- 新增 vis 对照把“场接口贡献”与“NCA 迭代计算/容量贡献”分开：前者强，后者弱。

---

## 5. 可以声称与不能声称

**可以声称**

1. 在相同 ext 输入、场接口、readout/解码器、plan 监督与优化器配置下，非迭代 1×1 conv
   场相对无场仍有 +13.5pp（5/5 seed，池化 CI 不含 0）。
2. NCA−none 的总增益中，约 62% 可由对齐接口的非迭代场复现。
3. NCA 相对 vis 的平均增量 +8.1pp 方向为正，但未达 seed 层面显著。
4. CNN 与 vis 在当前协议下统计打平。

**不能声称**

1. NCA 迭代空间计算优于同接口简单场（结构优势未成立）。
2. 增量来自“迭代计算”而非“参数量/容量”（vis 参数量少 43×，需 K=1 消融）。
3. 任何单 seed 的臂间排序（s4 对 nca−vis 均值贡献过大）。
4. 对任务 v2 / 多任务 / 障碍 OOD 的泛化（本对照只在原试点任务上）。

---

## 6. 限制

1. **容量混淆**：vis 2,496 参 vs NCA 106,688 参。+8.1pp 可能来自容量、迭代计算或二者叠加。
   需要在同参数量下做 NCA K=1（单步、不迭代）消融，以及容量匹配的 1×1/MLP 场对照。
2. **单次评估**：每个 checkpoint 只评估一次；bf16 kernel 非确定性带来约 ±0.5–2pp 的
   评估噪声（见 §1.2）。vis 与重评基线各只有一次评估。
3. **s4 主导**：nca−vis 的均值由 s4 的 +31pp 拉动；中位数为 +1.5pp。
4. **场通道接口**：vis 输出 32 通道并全部进入 readout，与 NCA 对齐；但 CNN 臂只使用
   通道 0（其余置零），因此 cnn−vis 的对比不是等带宽对比，只用于回答“更深空间计算是否有用”。
5. **环境外部效度**：与试点相同（墙可滑动、1500 步预算宽松），不强烈奖励规划。

---

## 7. 下一步（按用户指示暂缓，不自动启动）

1. NCA K=1 消融（同 107K 参数、单步推理）与容量匹配场，隔离“迭代计算 vs 容量”。
2. 确定性训练/评估或每配置多次评估，压低 ±2pp 噪声后再重估 nca−vis。
3. 任务 v2（非滑动墙/窄通道、紧预算、障碍 OOD）后再检验场收益是否随规划需求上升。
4. 其余已列实验（G64、多墙、VIN 全量训练等）保持暂缓。

---

## 附录：复现

```bash
PY=/home/ubuntu/miniconda3/envs/cerebvla/bin/python

# vis 训练（5 seed；k=12 对 vis 前向无影响，仅为命令行统一）
for s in 0 1 2 3 4; do
  $PY scripts/train_vla_lite_low.py --arm vis --seed $s --steps 20000 \
    --batch 64 --k-steps 12 --out ckpts/vla_lite/conf_vis_s$s
done

# vis 评估（200 场景，seed 777，预测高层）
for s in 0 1 2 3 4; do
  $PY scripts/eval_vla_lite.py --ckpt ckpts/vla_lite/conf_vis_s$s/last.pt \
    --high-ckpt ckpts/vla_lite/high_s0/last.pt --n-episodes 200 \
    --json-out logs/vla_lite/conf_vis_s$s_pred.json
done

# 基线同码重评（none/cnn/nca × 5 seed）
for arm in none cnn nca; do for s in 0 1 2 3 4; do
  $PY scripts/eval_vla_lite.py --ckpt ckpts/vla_lite/conf_${arm}_s$s/last.pt \
    --high-ckpt ckpts/vla_lite/high_s0/last.pt --n-episodes 200 \
    --json-out logs/vla_lite/recheck_${arm}_s$s_pred.json
done; done

# 汇总统计（输出本报告 §2 的配对结果与结果摘要 JSON）
$PY scripts/analyze_vla_lite_5seed.py \
  --json-out reports/VLA_LITE_VIS_CONTROL_results.json
```

结果文件：

| 路径 | 内容 | 是否入库 |
|---|---|---|
| `reports/VLA_LITE_VIS_CONTROL_results.json` | 逐 seed 成功计数、逐场景 0/1 串、配对统计 | 是 |
| `logs/vla_lite/conf_vis_s*_pred.json` | vis 评估原始 JSON（含 visit 日志） | 否（gitignored） |
| `logs/vla_lite/recheck_{none,cnn,nca}_s*_pred.json` | 基线同码重评原始 JSON | 否（gitignored） |
| `ckpts/vla_lite/conf_vis_s*/last.pt` | vis 低层 checkpoint | 否（gitignored） |

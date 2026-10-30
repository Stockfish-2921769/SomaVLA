# SomaVLA — VLA-lite 试点报告：NCA 规划场的贡献归因

> 日期：2026-10-30
> 状态：试点（pilot）完成，主协议修正待确认
> 关联：`VLA_LITE_TASK.md`（任务书）、`REPORT_FIELD_INJECTION.md`（E0 方案评估）、
> 代码 `soma/vla_lite/`、脚本 `scripts/*_vla_lite*.py`
>
> **结论先行**：在无 LLM 的 VLA-lite 上，NCA 场对低层策略有稳定正贡献
> （5/5 训练 seed 为正，+22.1pp，池化 95% CI [+18.5, +25.7]）；
> “有信息的场”才有效（GT/random 场显著有害）；NCA 相对同接口、同感受野的
> CNN 场平均 +11.3pp（4/5 seed），但受 seed 间波动限制，**尚不能作为已证实的
> 结构优势**。单 seed 结论一律不可引用；teacher 子目标协议已饱和，不能作为主指标。

---

## 0. 摘要

| 问题 | 结论 | 证据强度 |
|---|---|---|
| 场是否帮低层策略？ | 是，NCA 场 +22.1pp | 强：5/5 seed，最小 +14.0pp |
| 任意场都有用吗？ | 否，GT −9.0pp、random −8.5pp | 中：修复前 5-seed，配对显著 |
| NCA 结构是否优于 CNN？ | 方向支持（+11.3pp，4/5） | 弱–中：seed 间 se 8.2，未达显著 |
| 高层是否是瓶颈？ | 是 | 强：val xy 误差 3.8cm，31.7% 超抓取半径 |
| teacher 子目标协议能否用？ | 不能，已饱和（95.5–100%） | 强：clean teacher，GT 还显著低于 none |

---

## 1. 实验设置

### 1.1 任务与模型

- **环境**：双墙、3 个方块对 3 个同色目标，`success = 3/3 placed`；墙可滑动
  （`in_walls` 时尝试 x-only / y-only 运动）；`max_steps=1500`，`visit_cap=180`。
- **VLA-lite**：冻结 SigLIP-B/16（256px，16×16×768 token）→ 空间适配器
  （768→256，32×32，64ch，含障碍头）→ 场生成器（唯一变量）→ FieldReadout
  （32→256，16×16 网格 token）→ flow-matching 动作解码器（4 层，d=256，
  chunk 8，7D 绝对 plant target）+ 高层决策头（2 层 Transformer，输出
  `[x,y,z,open]`，每个 visit 查询一次）。**无 LLM**，语言为可学习任务槽。
- **数据**：低层 25,004 个 chunk / 600 场景（scene 级划分 450/75/75）；
  visit 数据 72,000（54,000/9,000/9,000）。外部评测固定 200 个新场景
  （seed 777），所有臂共享同批场景与同一个高层 checkpoint（`high_s0`）。

### 1.2 受控臂（只有场生成器不同）

| 臂 | 场来源 | 说明 |
|---|---|---|
| `nca` | NCA（C=32，K=12，共享 3×3 规则） | 主臂，端到端可微 |
| `cnn` | 12 层 3×3 CNN | 同接口、同感受野、同输入通道 |
| `gt` | 解析测地线场 + 障碍 | 信息上界 |
| `euclid` | 零参数目标势场 | “只需目标方向”的下界 |
| `random` | 平滑随机场 | 无信息控制 |
| `none` | 无场（token 置零） | 主基线 |
| `zero` | 付 NCA 算力但场对解码器隐藏 | 隔离“计算量”效应 |
| `vis` | 适配器 64ch → 1×1 conv → 32ch 场 | 等带宽感知控制 |

低层训练：20k step、batch 64、AdamW、bf16、field-dropout 0.1；
所有臂共享同一数据、标签、损失与优化器配置。

### 1.3 协议与统计

- **主协议**：预测高层（高层 BC 于 teacher visit command，闭环中自预测），
  非 teacher 子目标。
- 200 场景全臂配对；同 seed 内用精确 McNemar；跨 seed 用场景池化 bootstrap
  并同时报告“逐 seed 差值方向”；**禁止单 seed 推断**。

---

## 2. 主结果（修复初始化混杂后，5 训练 seed）

成功率（%，3/3 placed）：

| 臂 | s0 | s1 | s2 | s3 | s4 | 均值 [min,max] | 同 seed 配对差 vs none |
|---|---|---|---|---|---|---|---|
| none | 25.5 | 33.0 | 20.5 | 14.5 | 28.0 | 24.3 [14.5,33.0] | — |
| cnn | 28.0 | 24.0 | 21.5 | 48.0 | 54.0 | 35.1 [21.5,54.0] | +10.8pp [−9.0,+33.5]，4/5 正 |
| nca | 42.5 | 57.0 | 34.5 | 30.0 | 68.0 | 46.4 [30.0,68.0] | **+22.1pp [+14.0,+40.0]，5/5 正** |

- NCA−none：5/5 seed 为正；池化 95% CI **[+18.5, +25.7]**。
- CNN−none：4/5 为正，但 s1 显著为负（−9.0pp，p=.047）；CI [+7.2, +14.4]。
- **NCA−CNN**：逐 seed −18.0 / +14.5 / +33.0 / +13.0 / +14.0（4/5 正），
  均值 **+11.3pp**，seed 间 se 8.2（≈1.4σ）→ seed 层面未达显著。

修复初始化混杂前（pilot）5-seed 均值：none 24.0、nca 46.8、cnn 38.1、
euclid 27.8、gt 15.0、random 15.5；`vis`/`zero` 仅 s0（2.0 / 24.0，零臂≡none，p=1.0）。
修复后均值变化 ≤3pp，但逐 seed 排序变化大（如 cnn s0 45.5→28.0）。

---

## 3. Teacher 子目标协议：已饱和

clean teacher 子目标、seed 0：

| 臂 | none | random | gt | euclid | nca | cnn | vis |
|---|---|---|---|---|---|---|---|
| success % | 99.5 | 98.5 | 95.5 | 99.0 | 100 | 100 | 32 |

- 天花板在 95.5–100%，且 **GT 显著低于 none（p=.022）**：解析场与低层策略
  的接口/尺度不完全匹配。
- 因此任务书 G0/G1 的“teacher +10pp”判据在该任务/预算下不可达；teacher
  只能作为上限参考，主结论必须走预测高层协议。
- `vis` 仅 32%，说明原始视觉特征直接当“场”不可用，场需要结构化生成器。

---

## 4. iid 目标噪声压力测试（seed 0，单 seed，待复现）

对每个 visit 命令加高斯 xy 噪声（按场景/visit 序号固定，臂间共享）：

| σ | none | gt | euclid | nca | cnn | random |
|---|---|---|---|---|---|---|
| 2cm | 51 | 45 | 63 | 55 | 54.5 | 38 |
| 5cm | 0.5 | 1.0 | 3.5 | 0.5 | 2.0 | — |

- σ=2cm 时 euclid/nca/cnn 均高于 none，GT 反而更低（与 teacher 结果一致）。
- σ=5cm 时全部 ≤3.5%：低层本质是“目标跟踪器”，场无法修复大范围目标误差。
- 剂量排序仅单 seed，不能引用为结论。

---

## 5. 机制与瓶颈诊断

### 5.1 高层是主要瓶颈

- visit-val 命令 xy 误差：均值 3.81cm、中位 1.80cm、p90 7.50cm；
  **31.7% 超过 2.5cm 抓取半径**。
- 闭环 OOD（新场景）xyz L1 2.6–3.2cm，高于域内 1.57cm。
- 场景层面 corr(none 成功率, NCA−none 增益) = **−0.55**：场主要在基线会
  失败的场景起效，符合“场提供补救信息”的解释。

### 5.2 离线指标不能选模

- 17 个臂×seed 上，离线 val action loss 与闭环 success **负相关 r=−0.52**：
  离线更拟合 teacher 动作的臂（如 random）闭环更差，CNN 离线最差却闭环较好。
- success 与 `high_open_acc` 同步波动（高层查询状态分布在闭环中漂移）。

### 5.3 学习场抗噪但被解码器绕过

- 预测计划对“真实目标/扰动目标”的 MSE 比（σ=5cm）：nca 0.60、cnn 0.51、
  euclid 0.83 —— 学习场在离线能部分纠正目标误差。
- 但闭环仍完全依赖显式目标向量（解码器同时接收 `subgoals`）：目标被“喂两次”，
  场的视觉纠偏被旁路。若要让场真正参与闭环，需要 goal-dropout 或让高层读场。

### 5.4 NCA 自适应计算未成立

- 训练 K=12；推理 K=24/48 的计划 MSE：0.051 → 0.35–0.54 → 4.0–17.3（3/3 seed）。
- NCA 学到的是一条固定 12 步轨迹，不是可迭代到收敛的不动点；
  K=12 与 12 层 CNN 感受野相同，域内不应期待先验结构优势。

---

## 6. 方法学发现

### 6.1 跨臂初始化混杂（已修复，commit `1e2f216`）

`policy.py` 原先按臂条件构造参数化模块（CNNField、vis 的 1×1 conv），在共享
readout/decoder/language 之前消耗 RNG，导致同 seed 下 cnn/vis 臂的共享模块
初始化与 none 不同。

- 影响：CNN−none 与 NCA−CNN 的比较被初始化差异污染；NCA/euclid/random/zero−none 不受影响。
- 修复：共享模块先构造，臂专属模块后构造；新增 `test_arm_init_parity`
  （7 臂 shared 权重逐张量相等）。
- 修复对 5-seed 均值影响 ≤3pp，但逐 seed 排序变化明显 → 修复是必要的。

### 6.2 同 seed 重跑噪声（新发现）

nca s0 修复前后初始化完全相同（训练首步 loss 吻合到 1e-5），最终成功率
**33.0% → 42.5%**；none s0 24.0% → 25.5%。来源是 GPU 非确定性核
（bf16/cuDNN），不是初始化或数据顺序。

- 含义：除 ±15–25pp 的 seed 间波动外，还有约 ±5–10pp 的运行噪声。
- 结构归因需要确定性训练（`torch.use_deterministic_algorithms` +
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`）或每配置多次重跑平均。

---

## 7. 可以声称与不能声称的内容

**可以声称**

1. 在该任务/数据/预算下，NCA 场显著优于无场（5/5 seed）。
2. 场的收益不是“任意场”效应：GT 与 random 场显著有害。
3. 收益集中在基线会失败的场景，与“场提供目标导向的空间信息”一致。

**不能声称**

1. NCA 结构优于 CNN 场（方向支持，seed 层面未达显著）。
2. 任何单 seed 的臂间排序。
3. teacher 子目标协议下的增益。
4. NCA 的迭代计算/不动点优势（K>12 崩溃）。
5. 离线指标可作为选模依据。

**外部效度限制**：墙可滑动、1500 步预算宽松，环境不强烈奖励规划；
多任务泛化尚未进入本试点。

---

## 8. 下一步

1. **确定性训练重跑** NCA/CNN/none × 5 seed，消除 ±5–10pp 运行噪声后重估
   NCA−CNN。
2. **任务 v2**：非滑动墙/窄通道、紧预算（≤600 步）、4–8 障碍 OOD、G64；
   让规划有实际回报。
3. **高层改进**：噪声增广、DAgger/闭环状态训练、更多 visit 数据；
   评估把 NCA 场池化后喂给高层。
4. **NCA 假设检验**：训练 K∈[6,24] 随机 + 不动点损失，推理 K=48；
   若仍不收敛，则“自适应计算”假设被证伪。
5. **架构**：goal-dropout（防解码器绕过场）、readout 16→32。
6. 上述协议修正确认后写入 `VLA_LITE_TASK.md`。

---

## 附录：复现

```bash
PY=/home/ubuntu/miniconda3/envs/cerebvla/bin/python

# 数据与特征
$PY scripts/vla_lite_data.py
$PY scripts/precompute_siglip.py --data data/vla_lite_2w.npz \
  --out data/feat_low_fp16.npy
$PY scripts/precompute_siglip.py --data data/vla_lite_visit_2w.npz \
  --out data/feat_visit_fp16.npy

# 低层（每臂每 seed）
$PY scripts/train_vla_lite_low.py --arm nca --seed 0 \
  --out ckpts/vla_lite/conf_nca_s0
# 高层（一次，全臂共享）
$PY scripts/train_vla_lite_high.py --out ckpts/vla_lite/high_s0

# 闭环评估（200 场景，seed 777，预测高层）
$PY scripts/eval_vla_lite.py --ckpt ckpts/vla_lite/conf_nca_s0/last.pt \
  --high-ckpt ckpts/vla_lite/high_s0/last.pt \
  --json-out logs/vla_lite/conf_nca_s0_pred.json

# 单元测试（含跨臂初始化一致性）
$PY scripts/test_vla_lite.py
```

原始评估 JSON 位于 `logs/vla_lite/`（gitignored），本报告为其汇总。

# SomaVLA — 分层控制研究：架构、门禁与独立物理验证报告

> 项目代号 **SomaVLA**（本仓库）。日期：2026-09-01。
> 本报告覆盖 Round 12 新架构的全部已完成的验证阶段（门禁 a–h）、闭环组装、
> 以及 MuJoCo 独立物理交叉验证（Level A / A+）的诚实结论。

---

## 1. 项目概述

**目标**：验证一种"慢速高层规划 + 快速自组织底层控制"的分层架构能否从程序化
仿真轨迹与物理模态中涌现出**隐式动作规划**能力。核心主张是：一个**同质的
body-graph NCA（神经细胞自动机）**在小脑层做高速（数百 Hz）闭环控制，顶层
**MoE 路由器**在低频做事件驱动的技能组合，两者共同完成 pick-and-place 任务。

**关键设计选择**：
- **同质 NCA 专家**：7 个 EEF-pose DOF 细胞共享同一网络，无 cell-type embedding、
  无细胞分化——只利用 NCA 的**涌现**特性（邻域迭代 + 跨步形态素记忆）。
- **事件驱动**：skill 完成 → 路由器重推理 → 下一 skill；不锚定频率路线。
- **程序化 sim 训练 + 物理模态通道**：训练数据来自程序化轨迹生成器，闭环
  鲁棒性用 roll-in 训练获得；小脑每步观察一个可微的库仑物理上下文，从而学出
  "余量越小、动作越保守"的隐式速度计划。

---

## 2. 架构设计

### 2.1 分层结构

```
                 ┌─────────────────────────────┐
  scene/state ──▶│  MoE 路由器 (StateRouter)   │  低频，事件驱动
                 │  π0.5 dense backbone        │  skill + 边界条件
                 └──────────────┬──────────────┘
                                │ skill, goal, alpha_mask, duration
                 ┌──────────────▼──────────────┐
  每技能同质      │  BodyGraph-NCA 专家 ×6      │  高频（数百 Hz）
  body-graph     │  approach/grasp/lift/       │  7 DOF 细胞，共享权重
                 │  transport/place/release    │
                 └──────────────┬──────────────┘
                                ▼
                        绝对 target[7] → 阻尼 plant → 闭环
```

- **6 个技能**：approach / grasp / lift / transport / place / release，每个技能
  一个独立训练的 NCA 专家（同质结构，参数不同）。
- **控制器**（`soma/bodygraph_controller.py`）把路由器的"skill + 边界条件
  （goal、alpha activity mask、期望时长）"装配成专家的 relax 目标；完成判定用
  **max-based active-DOF 误差**（不是均值——均值会掩盖单轴偏差）。

### 2.2 NCA 细胞状态与更新

- 细胞状态三通道：`[alpha(1) | value(1) | morphogen(8)]`。
  - **alpha**：活动门控，由 MoE 设定，不参与读出/更新投影；
  - **value**：相对 reseed 的归一化位移（输出）；
  - **morphogen**：跨步热启动的轨迹记忆（warm-start 输入到下一 relax）。
- 稳定性三件套：**末层零初始化 + tanh 有界更新 + alpha>0.1 门控**；
  训练 firing=0.5 随机 / 推理 firing=1.0 确定性。
- `BodyGraphNCA`：**15,905 参数**，einsum 邻域聚合（比逐细胞 Python 循环快 65×）。

### 2.3 训练配方

- 随机 rollout 长度（interval ∈ [3,8]）+ **稀疏终态监督**（`‖target(last)−goal‖²`
  为主）+ 掩膜 loss（只 alpha>0 的 DOF）+ Adam + grad clip。
- **lr=1e-3 是回归任务的关键**：论文的 lr=1e-4（分类任务取值）会让专家学成
  echo（terminal err ≈ 注入 bias，approach 38.8mm/0%）；lr=1e-3 修复后 approach
  2.3mm/100%。dense drift loss 反而让 approach 变差，主训练用 terminal-only。
- **闭环 roll-in 重训练**（`checkpoints_rollin2/`）：稀疏终态下专家会学成
  `target≈reseed`（echo），闭环 plant 永不推进；`unroll_loop`（reseed 沿阻尼
  plant 前进）+ `loss_loop` 稠密 hold-at-goal 项（wd=0.3）修复，否则闭环卡死。

---

## 3. 门禁验证体系

### 3.1 专家收敛 — gate (a)

6 专家（lr=1e-3, 3000 步）全部收敛：100% 收敛、末端误差 0.0–1.0mm。

### 3.2 漂移重收敛 — gate (b)

中途 t0∈[30%,70%] 注入 ε∈{10,20}mm 扰动，推理 firing∈{1.0,0.5}：firing=0.5 下
末端仍 <2.5mm —— 同质 NCA 对扰动重收敛，对应论文的"损伤恢复"。

### 3.3 形态素涌现 — gate (c)

12 维形态素合格（跨步热启动记忆确实在形成）。

### 3.4 场景鲁棒 + MoE 路由 — gate (d)

- **场景参数化**（`scripts/proc_sim.py`）：obj/place 每 episode 随机；NCA 在
  参数化场景下 gate(a)(b) 全 PASS **无需重训**（transport dxy 117→8–251mm 仍
  收敛）——规则只吃归一化 `(goal−reseed)/σ`，scenario-agnostic 成立。
- **StateRouter**（`soma/moe_router.py`）：backbone(scene⊕state)→skill_head(6)
  + 每-skill 学习常量 `goal_base[6,7]` + scene→xy 读出 + duration_head。关键修正：
  常量目标（pz/openness）走纯标量、scene 派生只留 approach/transport 的 px,py
  ——否则留出场景泛化差 15mm。
- **训练**：BC，loss = CE + masked-MSE(硬头) + L1(duration)，lr=1e-3，需
  ~20k 步。**结果**：路由 98.6%、6 技能 active-pos 0.00mm、duration err 9.4 步；
  中段扰动探针显示 skill 临近完成时路由转下一 skill——**事件驱动可组合行为**。

### 3.5 SigLIP 感知前插 — gate (d) + VLM

- 用户选定 **SigLIP**（ViT-B-16, frozen）+ 99k 回归头；俯视表渲染 obj(红块)+
  place(蓝环)，视觉抖动使感知成为鲁棒回归。
- 冻结编码器 + MLP(768→128→4)，MSE 回归 lr=1e-3 20k 步，**收敛 ~3.7mm**
  （SigLIP 全局池化对亚像素回归的分辨率地板）。
- 端到端（渲染→SigLIP→scene_pred→路由器）：**路由 98.7%**（GT 基线 99.1%，
  −0.4pp）、approach/transport active-pos 2.38/2.51mm（<10mm 门）、duration
  9.6 步。场景派生 xy goal 误差 = SigLIP 定位误差线性传递，无需对路由器加
  scene 噪声增广。

### 3.6 闭环组装 — gates (e)(f)(g)

- **(e)** GT scene 20/20、SigLIP 20/20（mean 31–32 步）。
- **(f)** 鲁棒性：lift 中段 10mm 扰动 15/15。
- **(g)** 推理速度：464 Hz CPU / 960 Hz GPU（目标 ≥200/≥100）。
- **三个关键实测修复**：
  1. **Echo 病态 → roll-in 重训练**（见 2.3）；
  2. **过渡约束区分 completed**：完成位姿下 self-logit 占优会让路由器重路由回
     自己、链卡死；修复 = 完成→只允许规范 next / 超时→允许 {self,next}；
  3. **完成判定 mean→max**：transport 单轴偏 11mm 被 mean 掩盖 → 每 active DOF
     都在 tol 内才算完成，EEF 落点贴近目标，边际失败全部回收。

---

## 4. 物理模态 + 隐式动作规划 — gate (h) 与 Level A/A+ 独立验证

### 4.1 库仑物理模态通道（Phase 6）

- `soma/physics.py` 提供可微解析库仑模型：`F_n = F_max·(1−g)`（g 为手爪开度），
  切向载荷为重量与惯性载荷的组合，滑移当 `F_t > μ·F_n`。
- NCA 每步广播 **physics_ctx[9] = [μ/0.6, m/0.35, F_n/15, slip_risk/5] +
  onehot5(contact)**（relax update 54→63 维）。
- 滑移损失 = **累积滑移位移 hinge**（`relu(Σ relu(risk−1)·|Δx|·held/SLIP_THRESH
  − slip_safety)`），首版 relu² 均值版稀释梯度失败，累积位移版才让 expert 随
  margin 收缩。
- sim_env 库仑状态机：free→contact→grasped→slip→detach；两个实测修复：
  release 分支先于 F_n<held_req 检查；**位置感知 detach**（F_n 掉穿/滑移越阈只
  detach，成败由 obj_final 位置判定）——这是 aware 能恢复硬单元的关键。

### 4.2 gate (h) 初版（τ=0.1 标定）

- **(h1)** aware 119/120（99%）vs blind 106/120（88%）：aware 收缩步恢复硬尾
  单元；ctx 置零的 aware 也 119/120 → 能力恢复主要来自 slip-loss 训练注入的
  保守先验。
- **(h2)** aware+ctx transport 步随 margin 单调收缩 **12.2→23.8mm**（~2×），
  blind 平坦 ~53mm；ctx 置零的 aware 平坦 ~19mm → **margin-自适应计划来自模态
  通道（ctx→action 策略）**。
- **(h3)** morphogen 跨步热启动消融仅 119→117/120 → **NULL**：因 (μ,m) 每步都在
  ctx、回合内恒定，计划是策略编码，不依赖形态素跨步记忆。

### 4.3 Level A — MuJoCo 独立交叉验证（2026-08-31，诚实负面）

用 MuJoCo 3.11 接触求解器对同一策略轨迹做**物理完全独立**的裁决
（pseudo-force / co-moving frame 模型：固定 world pad + `opt.gravity = g − Ḧ`）。
结论：**gate (h) 的 outcome 主张不被刚性体物里复现**。

- 重放 730 段 transport（aware+blind，3 seed），**MuJoCo 0 段滑出**；sim 判盲
  基线 transport 掉件 14 段，**MuJoCo 复现 0/14**（最大 slip 0.15mm vs 阈 15mm）。
- **负面机制**（解析模型自身不诚实）：
  1. 惯性项 `a = |Δx|/τ² = 100·Δx` 高估 ~2×（plant 真实峰值 `a=k²·Δx≈48·Δx`）；
  2. 把整个标量载荷 `m(g0+a)` 记在摩擦上——但 transport 沿 pad 法向（x）的
     加速度由接触法向承载（capture），只有切向（y,z）是摩擦载荷；
  3. 诚实切向载荷是矢量和，且诚实静态边界 1.42–1.53 > 可行带上限 1.376 →
     **诚实物里下任何可行单元在 transport 都不滑**。

### 4.4 Level A+ — 诚实重建 + 重找滑移带（2026-09-01）

**诚实重建（`soma/physics.py`）**：
- τ 0.1→**0.05**（快臂 regime）；峰值加速度 `a = K2·|Δx|`，
  `K2 = (−ln(1−0.5)/0.05)² ≈ 192`；
- 切向载荷改**矢量和** `F_t = m·sqrt((g0·held)² + a²)`；
- `SLIP_THRESH` 5→15mm（对齐 MuJoCo 重放阈）；训练硬带 [1.10, 1.35]、
  评估带 [1.16, 1.26]；
- 训练 bug 修复：`sqrt(0)` 首步反传 `inf·0=nan` → `+eps` 入 sqrt。

**MuJoCo 容量校准（Level A "0 滑移"的真根因）**：
- MuJoCo 摩擦是干净 Coulomb，但 pad gap 沉降出 **F_n_total≈75N** 而非意图的
  30N：盒-盒软接触刚度 ~165kN/m（假设的 2.5×），容量 2.5× 高于解析 μ·F_max
  → 无步滑。修复：按 (mass, Fn) **二分搜索 pad gap** 使沉降 F_n_total=2·F_MAX。
- 校准后 MuJoCo 单步滑移 onset **对齐诚实 risk**（m/μ=1.35 在 D≥40mm 滑，
  m/μ=1.20 在 40–55mm，m/μ=0.50 永不滑）。

**重训 aware（`checkpoints_phys_honest/`，w_slip=8.0, hard-frac=0.5）**：

| 指标 | blind | aware | aware(ctx 置零) |
|---|---|---|---|
| (h1) 成功率 | 84/120 (**70%**) | **120/120 (100%)** | 120/120 |
| 掉件数 | 36 | 0 | 0 |
| transport 步 | ~53mm（平坦） | ~30mm（统一保守） | ~30mm |
| lift 步 | ~55mm（平坦） | **24.8→38.3mm 单调** | ~26mm（平坦） |

- **h1（最强诚实正结果）**：盲基线正好掉低余量单元（margin<~1.1N，即
  m/μ>~1.05）；诚实模型下硬尾是唯一判别区，aware 全部恢复。
- **h2 SPLIT**：lift 是真实 **ctx→action 隐式速度计划**（24.8→38.3mm 单调、
  ctx 置零压平到 26mm）；transport 是统一 ~30mm 保守步（hard-frac 训练先验，
  非 margin 条件、非 ctx 驱动）。
- **h3** NULL（120/120 vs 120/120）。

**MuJoCo 裁决（20 cells × 3 eps，worst-case 90° yaw）**：
- **OUTCOME NOT CONFIRMED**：sim 判盲基线 transport 掉 2 段，MuJoCo 复现 0/2
  （全 HELD）。根因是**滑移率不匹配**：解析累积 `(risk−1)·dx` 每步 ~6.2mm、
  2–3 步越阈；MuJoCo 刚体 ~0.17mm/step、需 ~90 步——而真实 transport 只有
  3–8 步，MuJoCo 累积 <1.5mm 永不到 15mm。τ 张力：无 τ 同时满足"短程掉件"+
  "易单元不滑"。
- **MECHANISM CONFIRMED（modest）**：判别带（m/μ>1.1）上 MuJoCo 滑移 blind
  **0.23mm** vs aware **0.10mm**（~2.3×，盲从未低于 aware）——aware 的收缩步
  确实产生更少的刚性体滑移，是真实、刚体有效的行为，但量级 sub-mm、短程
  transport 远低于 15mm 掉件阈。

**诚实结论**：gate (h) 的**掉件 outcome 主张不被 MuJoCo 复现**（现在是滑移率
伪差，容量已修、onset 已验证）；**机制主张被复现但量级 modest**（收缩步减少
刚性体滑移是真实行为）。h1 的 sim 能力（100%/70%）是最强诚实正结果。

---

## 5. 关键工程发现汇总

| # | 发现 | 影响 |
|---|---|---|
| 1 | 回归任务 lr=1e-4 学成 echo，lr=1e-3 修复 | 专家收敛 |
| 2 | 稀疏终态 → echo；roll-in 重训练 + 稠密 hold-at-goal 修复 | 闭环不卡死 |
| 3 | 完成判定 mean→max | SigLIP 闭环 85%→100% |
| 4 | 过渡约束 completed 区分（完成→only next / 超时→{self,next}） | 链不卡死 |
| 5 | SigLIP 感知 3.7mm，路由误差线性传递 | 闭环与感知解耦 |
| 6 | 解析载荷必须 plant 一致（a=K2·Δx + 矢量和） | 诚实物理裁决 |
| 7 | MuJoCo 容量校准（F_n_total 2.5× 高估 → 二分 pad gap） | onset 对齐 |
| 8 | 解析滑移率 ~40–100× 高估（`(risk−1)·dx` vs 刚体） | outcome 不可复现 |
| 9 | sqrt(0) 反传 NaN → +eps | 训练稳定 |

---

## 6. 当前状态与结论

- **已达成**：分层架构（MoE 路由 + 同质 NCA 专家）在程序化 sim 中完成全任务
  闭环（gate e 100%）、鲁棒（f 15/15）、高速（g 数百 Hz）；物理模态让 NCA 学出
  **margin 自适应的隐式速度计划**（lift 步 ctx 驱动单调收缩），硬尾单元 sim
  能力 100% vs 盲 70%。
- **诚实边界**：掉件 outcome 未被 MuJoCo 刚性体物里复现——解析模型作为
  **学习信号**仍然有效（aware 因此学到保守收缩），但作为**物理预言**其滑移率
  是伪差；可声称的是"学到 margin-自适应步长收缩（context→action 策略）+ 该
  收缩在独立刚性体物里确实减少滑移（机制级）"。

---

## 7. 复现指南

```bash
# 环境
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate cerebvla
# （MuJoCo 用 /home/ubuntu/miniconda3/envs/cerebvla/bin/python）

# 训练专家（每技能）与路由器
python scripts/train_bodygraph_nca.py --skill <sk> --out checkpoints_rollin2
python scripts/train_moe_router.py

# 感知 + 门禁
python scripts/train_perception.py
python scripts/eval_moe_router_vlm.py
python scripts/eval_closed_loop.py --ckpt-dir checkpoints_phys_honest

# 物理模态：gate (h)
python scripts/eval_implicit_planning.py --aware-dir checkpoints_phys_honest \
  --blind-dir checkpoints_rollin2 --n-cells 40 --eps 3 --hard-frac 0.5 --zero-ctx

# MuJoCo 独立裁决：单步滑移 onset（校准）+ 段重放
python scripts/eval_level_a.py --calib
python scripts/eval_level_a.py --aware-dir checkpoints_phys_honest \
  --n-cells 20 --eps 3
```

**检查点说明**：
- `checkpoints_rollin2/`：盲基线（6 技能全盲）；
- `checkpoints_phys/`：初版物理 aware（uniform 训练，gate (h) 原版）；
- `checkpoints_phys2/`：hard-frac 实验产物（弃用）；
- `checkpoints_phys_honest/`：诚实重建 aware（Level A+，现用）。

**主要代码文件**：`soma/bodygraph_nca.py`、`soma/moe_router.py`、
`soma/bodygraph_controller.py`、`soma/physics.py`、`soma/gates.py`、
`scripts/proc_sim.py`、`scripts/sim_env.py`、`scripts/eval_closed_loop.py`、
`scripts/eval_implicit_planning.py`、`scripts/eval_level_a.py`。

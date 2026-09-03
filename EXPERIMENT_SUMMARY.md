# SomaVLA — 实验总结（按版本 / Round，至 2026-09-03）

> 本文件把 SomaVLA 从 Round 12 至今的每一次架构调整与对应验证按版本组织成一条
> 可追溯的演进链。每个 Round 固定给：**目标 → 架构状态 → 本次架构调整 → 验证
> （方法 + 数字）→ 结论**。诚实负面的条目都明确标注，不粉饰。
>
> 与 `REPORT.md`（截至 Level A+，讲架构/门禁细节）互补；`EXPERIMENT_LOG.md` 是
> 流水账；本文是**版本化的架构演进摘要**。
>
> 评估口径贯穿全文：所有"outcome 主张"都要经 **sim + MuJoCo 3.11 独立裁决**双轨
> 验证；MuJoCo 只裁决策略轨迹、物理完全独立。参数量谱系（研究主线）：
> NCA 0.1M → VLA-tiny 2.44M → 分层 0.38M → SmolVLA 0.5B。

## 0. 一页时间线（速查）

| Round | 日期 | 架构调整（一句话） | 验证 / 裁决 | 状态 |
|---|---|---|---|---|
| R1 · Round 12 | 08-26~28 | MoE 路由 + 同质 body-graph NCA，自包含 sim | gates a–g | **PASS** |
| R2 · Phase 6 | 08-28/29 | + 库仑物理模态通道 physics_ctx | gate h（sim） | 部分 PASS（h3 NULL） |
| R3 · Level A | 08-31 | 验证升级：MuJoCo 独立刚性体裁决 | outcome 复现 | **诚实负面** |
| R4 · Level A+ | 09-01 | 诚实重建 physics.py + MuJoCo 容量校准 | gate h 重跑 + MuJoCo | h1 PASS / outcome NOT CONFIRMED |
| R5 · Phase 7 | 09-01/02 | 新方向：单一任务小 VLA；建立两分布 + NCA 锚点 + VLA-A 挑战 | 硬 + 长程 MuJoCo | VLA-A 长程失败 → 指向 B |
| R6 · 决策 B | 09-02 | 分层：SigLIP 规划器头 + aware-NCA 执行器（**无 oracle**） | 36/36 + 0/15 | **PASS（终选）** |
| R7 · 数据效率 | 09-02 | （不改架构）扫规划器数据预算 | 闭环 + 长程 | 安全 1.2k / 完成 ~10.8k |
| R8 · Track A | 09-02/03 | SmolVLA-0.5B 对照（不新增架构） | 同 harness 闭环 | **诚实负面**（Q1/Q2） |

架构演进速览（组件增减）：

```
R1  sim1.0  StateRouter ─▶ 6×同质 NCA 专家（无物理）           ← 全门禁过
R2  sim2.0  + physics_ctx[9]（可微库仑）→ aware 专家            ← 学出 margin 自适应
R3  不动架构，只换裁决（MuJoCo 刚体）                           ← outcome 复现失败
R4  sim2.1  physics.py 诚实重建（plant 一致）+ 容量校准          ← 只留机制级 + h1 sim 能力
R5  换问题：单任务小 VLA 竞赛。NCA 为锚、VLA-A 端到端挑战        ← 长程全掉
R6  sim4.0  SigLIP-planner(0.27M) ─▶ aware-NCA（去掉最后 oracle）← 无 oracle 复现 → 终选
R7  /       数据预算轴扫描（1.2k→10.8k 合成场景）               ← 1–3 min GPU 即可用
R8  /       SmolVLA-0.5B 真权重微调（与 R5 同数据同 harness）    ← 0% 闭环（state 捷径）
```

---

## R1 · Round 12（2026-08-26~28）— 架构 1.0：MoE 路由 + 同质 body-graph NCA

**目标**：废弃旧 CerebVLA 路线（OpenVLA 7B + stateless NCA 小脑，Rounds 8–11 净负面），
验证"慢速高层规划 + 快速自组织底层控制"。事件驱动、不锚定频率、纯涌现（无 cell-type
embedding、无细胞分化）。

**架构状态**：`StateRouter`（低频，技能选择 + 边界条件）→ 每技能独立同质 NCA 专家
（高频，7 EEF-pose DOF 细胞共享权重，15,905 参数）。细胞三通道 `[alpha|value|
morphogen(8)]`，末层零初始化 + tanh 有界更新 + alpha>0.1 门控。训练 = 程序化 sim
轨迹 + 稀疏终态监督（目标 `‖target(last)−goal‖²`）+ 掩膜 loss。

**本次架构调整**（R1 内部 6 次迭代，每次都是"调整→验证"闭环）：

| # | 调整 | 触发 | 验证 | 结果 |
|---|---|---|---|---|
| a | **lr 1e-4→1e-3**（论文值是分类任务取值） | 专家学成 echo：approach 38.8mm/0% | gate(a) 技能收敛 | 收敛 0.0–1.0mm，100% PASS |
| b | 推理 firing 0.5 随机化对照 | 漂移鲁棒性 | gate(b) 中途 ε=10/20mm 扰动 | firing=0.5 下末端 <2.5mm PASS |
| c | （无调整，测量） | 形态素是否形成 | gate(c) | 12 维形态素合格 PASS |
| d | **场景参数化 + StateRouter goal 装配改纯标量**：常量目标(pz/openness)走学习常量、scene 派生只留 approach/transport 的 px,py | 最初 139 维头用 scene 特征拟合常量目标 → (w,b) 退化、留出泛化差 15mm | gate(d) 路由质量 | 路由 98.6%、active-pos 0.00mm、duration err 9.4 步 PASS |
| e | **SigLIP 感知前插**（frozen SigLIP-B16 + 99k 回归头，替换 GT scene）；渲染加视觉抖动；open_clip 预处理改纯 torch 向量化 | 感知要成为鲁棒回归；逐图 PIL 是 CPU 瓶颈 | gate(d)+VLM | 收敛 **~3.7mm**（SigLIP 全局池化分辨率地板）；端到端路由 98.7%（GT 99.1%，−0.4pp）、approach/transport xy goal 2.38/2.51mm PASS |
| f | **闭环组装三修**：① echo 病态 → roll-in 重训练（`unroll_loop` + 稠密 hold-at-goal）；② completed 过渡约束（完成→only next / 超时→{self,next}）；③ 完成判定 mean→max | ① 闭环 plant 永不推进；② 重路由回自己、链卡死；③ mean 掩盖 transport 单轴 11mm 偏差 | gates (e)(f)(g) | (e) 20/20 GT + 20/20 SigLIP；(f) 鲁棒 15/15；(g) 464Hz CPU / 960Hz GPU。mean→max 让 SigLIP 闭环 85%→100% |

**结论**：sim 自包含系统全门禁 PASS，闭环、鲁棒、高速。**局限（事后看是命门）**：全部
验证都在解析 sim 内自证，物理真实性未独立裁决——这正是 R3 要补的。

产物：`checkpoints_rollin2/`（盲基线，现用）、`checkpoints_rollin/`、`checkpoints/`
（echo 旧专家，勿用）。

---

## R2 · Phase 6（2026-08-28/29）— 架构 2.0：库仑物理模态 + 隐式动作规划

**目标**：给 NCA 一个**可微物理模态通道**，让"余量越小、动作越保守"的隐式速度计划
从数据里长出来，而不是手写规则。

**本次架构调整**：
- `soma/physics.py`：解析库仑模型，`F_n = F_max·(1−g)`，切向载荷 `F_t = m·(g0·held +
  |Δx|/τ²)`，滑移当 F_t > μ·F_n。
- 每步广播 **physics_ctx[9] = [μ/0.6, m/0.35, F_n/15, slip_risk/5] + onehot(contact 5)**
  （relax 更新 54→63 维）。τ=0.1s 标定 → 满抓下盲基线 risk≈m/μ，真滑移 onset 于
  m/μ>1。
- **滑移损失** = 累积滑移位移 hinge `relu(Σ relu(risk−1)·|Δx|·held/SLIP_THRESH − 0.5)`
  （首版 relu² 均值版稀释梯度失败）。
- sim_env 库仑状态机 free→contact→grasped→slip→detach。

**内部迭代**（Round A → B）：
- **Round A**：gate(h) v1 blind 63/80 vs aware 64/80 **无 gap**。诊断出两个 bug：
  ① release 分支是死代码（`F_n<held_req` 检查先于 `g≥0.5`，g≥0.5 隐含 F_n≤7.5N
  < held_req，每个硬单元放置都被误判掉件）；② 均匀可行带抽样没覆盖盲基线的真实
  滑移区（m/μ>1）。
- **Round B**：① 调分支顺序（先检查意图释放）；② **位置感知 detach**——掉 F_n /
  滑移越阈只 detach，成败由 obj_final 落点判定（这是 aware 能恢复极端硬单元的关键）。

**验证**（uniform 训练 + 位置感知 detach，40 cells × 3 eps）：
- **h1**：aware 119/120 (99%) vs blind 106/120 (88%)，dropped 1 vs 14。注意
  **ctx 置零的 aware 也 119/120** → 硬尾恢复主要来自 slip-loss 注入的保守先验，
  不是推理期 ctx。
- **h2**：aware+ctx transport 步随 margin 单调收缩 **12.2→23.8mm（~2×）**；盲平坦
  ~53mm；ctx 置零的 aware 平坦 ~19mm → margin-自适应计划**由模态通道驱动**
  （同专家 on/off ctx 对照）。
- **h3（负面）**：morphogen 热启动消融 119→117/120 → 计划是 context→action 策略
  的每步重推导，不在跨步形态素记忆里。
- hard_frac 训练实验（`checkpoints_phys2/`）→ 全局保守化（transport 平坦 ~20mm、
  h2 消失）→ **弃用**，uniform 训练严格更优。

**结论**：同质 NCA 从物理模态学出 **margin-自适应的隐式速度计划** 并恢复硬尾
（99% vs 88%）。这是当时最强的正结果——但**全部在解析 sim 内**。

---

## R3 · Level A（2026-08-31）— 独立物理交叉验证 = 诚实负面

**目标**：把 outcome 裁决换成 MuJoCo 3.11 接触求解器——策略轨迹不变，物理完全独立。
诚实性条款：两方向的失败都是有效发现。

**本次架构/方法调整**：不动策略，只换裁决。
- 尝试 1 废弃：双 pad 力伺服抓取。**MuJoCo 切向摩擦对动态 body 是软的**，物体在
  动态 pad 上以 ~20mm/s 蠕动，淹没 transport slip 信号。
- 最终：**pseudo-force / co-moving frame**——pad 是固定 world geom（~0.12mm/s 干净
  保持），hand 运动作为惯性伪力 `opt.gravity = g − Ḧ`（Ḧ 从 plant 重构，
  k=−ln(1−0.5)/0.1=6.93）。Galilean 等价使 onset 与真实移动抓手一致。

**验证**：`--calib` 单步滑移 onset + 730 段 transport 重放（aware+blind，3 seed）。

**结果（诚实负面）**：
- **MuJoCo 重放 730 段，0 段滑出**（最大 slip 0.15mm vs 阈 15mm）；sim 判盲基线
  transport 掉件 14 段，**MuJoCo 复现 0/14**。
- **负面机制**（解析模型自身不诚实）：
  1. transport 沿 pad 法向（x）的加速度由**接触法向承载**（capture），非摩擦；
     解析模型把整个标量载荷 m(g0+a) 记在摩擦上。
  2. plant 真实峰值加速度 `a = k²·Δx ≈ 48·Δx`，解析的 `a = Δx/τ² = 100·Δx` 高估 ~2×。
  3. 诚实静态滑移边界 m/μ≈1.42–1.53 > 可行带 1.376 → **诚实物里下任何可行单元在
     transport 都不滑**。

**结论**：gate(h) 的 **outcome 主张（aware 收回复盲掉件）不成立**——诚实物里下盲
基线同样持稳。h2 收缩是真实学到的行为，但解决的是一个诚实物里下不存在的问题。
保留 `scripts/eval_level_a.py`（--calib + 重放）。

---

## R4 · Level A+（2026-09-01）— 诚实重建 + 重找滑移带

**目标**：把解析模型改造成 plant 一致，重标定 τ 到快臂 regime，找到真实且 MuJoCo
可复现的滑移带，重训 aware，重跑 gate(h)，MuJoCo 裁决。

**本次架构调整**（`soma/physics.py` 诚实重建）：
- τ 0.1→**0.05**；峰值加速度 `a = K2·|Δx|`，`K2=(−ln(1−0.5)/0.05)²≈192`（修 2× 高估）。
- 切向载荷改**矢量和** `F_t = m·sqrt((g0·held)² + a²)`（非标量）。
- `SLIP_THRESH` 5mm→15mm（对齐 MuJoCo 重放阈）；训练硬带 [1.10,1.35]、评估带 [1.16,1.26]。
- 训练 bug：`sqrt(0)` 首步反传 inf·0=nan → `+eps` 入 sqrt。

**MuJoCo 容量校准（Level A "0 滑移" 的真根因）**：
- MuJoCo 摩擦是干净 Coulomb，但 pad gap 沉降出 **F_n_total≈75N** 而非意图的 30N：
  盒-盒软接触刚度 ~165kN/m（假设的 2.5×），容量 2.5× 高于 μ·F_max → 无步滑。
- 修复：按 (mass, Fn) **二分搜索 pad gap** → 沉降 F_n_total=2·F_MAX=30N。校准后
  单步滑移 onset **对齐诚实 risk**（m/μ=1.35 D≥40mm 滑、m/μ=1.20 40–55mm onset、
  m/μ=0.50 永不滑）。

**验证**（重训 `checkpoints_phys_honest/`，w_slip=8.0, hard-frac=0.5，40 cells×3 eps）：

| 指标 | blind | aware | aware(ctx 置零) |
|---|---|---|---|
| (h1) 成功率 | 84/120 (**70%**) | **120/120 (100%)** | 120/120 |
| 掉件数 | 36 | 0 | 0 |
| transport 步 | ~53mm 平坦 | ~30mm 统一保守 | ~30mm |
| lift 步 | ~55mm 平坦 | **24.8→38.3mm 单调** | ~26mm 平坦 |

- **h1（最强诚实正结果）**：盲正好掉低余量单元（margin<~1.1N，即 m/μ>~1.05）；
  诚实模型下硬尾是唯一判别区，aware 全恢复。
- **h2 SPLIT**：lift 是真实 **ctx→action 隐式速度计划**（单调收缩、ctx 置零压平）；
  transport 是统一 ~30mm 保守步（hard-frac 0.5 训练先验，非 margin 条件、非 ctx 驱动）。
- **h3**：morphogen 消融 120/120 vs 120/120 —— NULL（不变）。

**MuJoCo 裁决**（20 cells × 3 eps，worst-case 90° yaw）：
- **OUTCOME NOT CONFIRMED**：sim 判盲基线 transport 掉 2 段，MuJoCo 复现 0/2。
  根因是**滑移率不匹配**：解析累积每步 ~6.2mm、2–3 步越阈；MuJoCo 刚体 ~0.17mm/step
  需 ~90 步——真实 transport 只有 3–8 步，累积 <1.5mm 永不到 15mm。**τ 张力**：无 τ
  同时满足"短程掉件"+"易单元不滑"。
- **MECHANISM CONFIRMED (modest)**：判别带（m/μ>1.1）MuJoCo 滑移盲 **0.23mm** vs
  aware **0.10mm**（~2.3×，盲从未低于 aware）——收缩步确实产生更少的刚体滑移，是
  真实行为，但 **sub-mm**、远低于 15mm 掉件阈。

**结论**：gate(h) 掉件 **outcome 不被 MuJoCo 复现**（现在是滑移率伪差，容量已修、
onset 已验证）；**机制被复现但量级 modest**；h1 的 sim 能力（100%/70%）是最强诚实
正结果。**可声称范围** = "学到 margin-自适应步长收缩（context→action 策略）+ 该
收缩在独立刚性体物里确实减少滑移（机制级）"。

---

## R5 · Phase 7（2026-09-01/02）— 新方向：单一任务小型 VLA（决策 A 挑战）

**目标**（用户决策）：单一任务 VLA，参数 10⁶–10⁷，成功率 > NCA 基线。旧方向
（CerebVLA/MoE-NCA）不再作为主线介绍——NCA 执行器退居"要被打败的锚点"。
HF 不可达、无 SmolVLA 权重 → 决策 A 用 **SigVLA-tiny**（冻结 SigLIP-B16 92.9M +
compact cross-attn 解码器，SmolVLA-faithful 受限替代）。

**本次架构/方法调整**：
- **建立两个评估分布**（`scripts/eval_baseline_hard.py`）：
  ① **硬分布**：hard-frac 0.5 抽样 m/μ∈[1.16,1.26] 判别带 + 感知噪声 σ=3.7mm
  （SigLIP 级，控制器看到加噪 scene，env 用真值判成功）+ transport 首两步间 ±10mm
  中途推扰。
  ② **长程 transport 分布**：记录段连续重放 ×60，让 MuJoCo 刚体滑移累积越 15mm 阈。
  裁决 = sim（obj 落真 place 的 place_tol 内）+ MuJoCo 独立重放。
- **VLA-A 数据**：aware-NCA 闭环 demo 2000 集 → 11,141 个 (image, chunk[8,4],
  state[7])（`scripts/vla_data.py`）；每维归一化（position std ~5mm vs open std 0.4）。

**验证**：

*NCA 锚点*（~10⁵ 参数：6/4 专家 + MoE 路由，盲 114K / aware 118K）：

| agent | 硬 sim | MuJoCo 短程 | 长程×60 掉件 | cycle-to-drop | median slip |
|---|---|---|---|---|---|
| blind | 25/36 (69%) | 36/36 held | 11/15 (73%) | 28 | 15.02mm |
| aware | 36/36 (100%) | 36/36 held | **0/15 (0%)** | none | 7.78mm |

→ **长程 transport 是 MuJoCo 独立刚体物里下第一个干净的 outcome 复现**：盲的大步
（risk 1.3–1.65）让刚体滑移越 15mm 阈；aware 收缩步保持。这是给 VLA 的基准。

*SigVLA-tiny（决策 A，2.44M trainable / 95.4M total）*：

| agent | 硬 sim | MuJoCo 短程 | 长程×60 掉件 | cycle-to-drop |
|---|---|---|---|---|
| VLA-tiny (chunk-8) | 33/36 (92%) | 33/33 held | **14/14 (100%)** | 10 |
| VLA-tiny (chunk-1) | 21/36 (58%) | 33/33 held | **13/13 (100%)** | 10 |

- 短程 sim 92%（3 timeout、0 掉），夹在盲 69% 与 aware 100% 之间；学到弱像素级
  margin→step 收缩（30.6→23.9mm）但不够陡。
- **长程 MuJoCo 14/14 掉、cycle 10——比盲基线（cycle 28）还差 ~3×**；clean 分布
  （σ=0、无推）同样 15/15 掉 → 与扰动无关。
- **机理**：grip 通道正常（Fn≈0.97 F_max），滑移来自**命令运动**——8 步 chunk 开环
  执行 + plant 滞后 → 携带期 max|D| 尖峰 37–52mm（≈2× mean）+ z 振荡（mean|Dz|
  7.6–16.5mm），周期性越过 ~40mm risk 边界。

**对照实验（结构本身是否可用，2026-09-02）**——把"信息上限"与"BC 训练/闭环伪差"分开：
- **C1（感知半）**：冻结 SigLIP mean-pool→margin 回归 **val MAE 0.32N** → 编码器
  通道携带足够载荷物理，**感知端不瓶颈**。
- **C2（匹配闭合）**：chunk-1 per-step 逐点重解码，闭合匹配，回归 7.2mm（好）→
  **长程仍 13/13 掉（cycle 10），sim 58%（反而更差）** → chunk 滞后不是根因。
- **grip 消融**（replay 级）：强置 Fn=F_max 重放 cycle-to-drop 不变 → 非 grip 通道。
- **根因定性 = BC 协变量漂移**：模型在 aware-NCA 状态分布上训练，自驱动后状态偏移 →
  预测噪声反馈放大 → 轨迹退化为 z 振荡 + 步长尖峰；aware 的 per-step slip_risk 反馈 +
  z 恒定 + 陡峭收缩，BC 学不来。

**结论**：端到端 tiny VLA 短程 sim 达 aware 水平，但**不复现 aware 的 MuJoCo 有效
outcome**——失败在行为/目标层（BC 学不到携带纪律），不在编码器信息层。→ **指向
决策 B（分层）**。

---

## R6 · Phase 7 · 决策 B（2026-09-02）— 架构 4.0（终选）：SigLIP 规划器 + aware-NCA 执行器

**目标**：保留 NCA 执行器的安全行为，把两处 oracle 入口都换成像素读出——分层系统
**无任何 env-truth oracle**，与端到端 VLA-A 的对照才公平。

**本次架构调整**：
- tiny **SigLIP 规划器头（0.27M MLP）** 回归 6 向量 `[obĵ_xy, placê_xy, μ̂, m̂]`：
  冻结 SigLIP-B16 mean-pool → LN→768→256→256→6。
- 替换 aware-NCA 闭环里 NCA 基线收 oracle 的两处：
  ① scene → `ctl.reset([obĵ,placê], state)`（路由 + 各技能 xy 目标）；
  ② physics_ctx → 9 维 ctx 用 (μ̂,m̂) 重建（aware 步长收缩的输入）。
- **env/MuJoCo 仍以 TRUE 物理 + TRUE place 裁决** → μ̂/m̂ 低估会让执行器欠收缩 →
  真实掉件。这是无 oracle 的诚实测试。
- 渲染（`planner_data.py`）：静态顶视 256 RGB（绿 place 环 + 物体矩形，边长∝mass
  12–28px、色调∝μ），**无 EEF 点**（规划器是静态场景估计器）；分布 = cells()
  hard-frac 0.5（判别带 [1.16,1.26]，全域 m/μ∈[0.09,1.37]）。

**验证**：
- 训练：12k 场景，6000 步 lr 5e-4 + warmup200 + cosine，val MSE 0.0216。
  per-dim val MAE：obj 4.37/2.90、place 3.35/3.43 → **scene xy MAE 3.51mm**；
  μ/m MAE 0.01；**m/μ rel err：mean 7.2%、hard-band 4.7%**；hard-band 低估(>10%) 6%。
- 闭环（`eval_hierarchical.py`，12 cells × 3 eps，MuJoCo 长程 ×60）：

```
── Oracle-aware（control：TRUE physics ctx，同批 cells）─────────
  sim 36/36 (100%)  |  MuJoCo long-transport 0/15 掉 (median slip 7.06mm)
── Decision-B（planner 从像素读 scene+physics）────────────────
  sim 36/36 (100%)  |  MuJoCo long-transport 0/15 掉 (median slip 9.00mm)
  planner m/μ rel err 8.8% (mean)，under-estimate 31%（都 <10% 伤害阈）
```

- Oracle 控制行把 aware 锚点（36/36、0/15）在脚本内复现 → **harness 对齐**；B 行
  12/12 cells 全 3/3（含最难 m/μ 1.16–1.24、margin 0.6–0.9 cells）。
- 参数量：NCA 118K + 规划器头 266K ≈ **0.38M trainable**（SigLIP 92.9M 冻结推理），
  仍在 10⁶–10⁷ 小参数量级。

**结论（PASS）**：分层系统在**无 oracle** 下复现了 oracle-aware 的 MuJoCo 有效
outcome（36/36 hard sim，0/15 长程）——决策 A 做不到（33/36 但 14/14 掉）。
分层把感知/规划给 tiny VLA、把安全执行留给已持稳 MuJoCo 的 NCA → 两半各尽其长。

---

## R7 · Phase 7 · 决策 B 数据效率（2026-09-02）— 数据预算 → 闭环 outcome

**目标**：把"小模型优势"落到数据轴上——扫规划器训练数据预算 N（固定 ~18 epochs、
共享 1200 张 val；`sweep_planner_data.py` + `report_planner_sweep.py`）。

**验证**（同批 cells，12×3，MuJoCo 长程×60；oracle 控制行三次恒为 36/36 + 0/15
→ harness 确定性）：

| N | 场景MAE | hard m/μ rel | 低估% | hier sim ok | release掉 | timeout | 长程×60 | GPU 训练 |
|---|---|---|---|---|---|---|---|---|
| 1200 | 6.63mm | 7.9% | 16% | 27/36 (75%) | 4 | 5 | 0/15 | ~44s |
| 2400 | 5.18mm | 6.1% | 11% | 32/36 (89%) | 4 | 0 | 0/15 | ~82s |
| 4800 | 4.38mm | 5.2% | 7% | 31/36 (86%) | 4 | 1 | 0/15 | ~156s |
| 10800 | 3.51mm | 4.7% | 6% | 36/36 (100%) | 0 | 0 | 0/15 | ~5min |

**结论（分两条轴）**：
1. **安全轴（MuJoCo 长程）= 数据鲁棒**：低至 1.2k 场景（~44s）就 0/15 不滑——
   aware 执行器的保守收缩覆盖粗糙的 μ̂/m̂（N=1200 低估 56% 也不掉件）。
2. **任务完成轴（sim 成功率）= 需更多数据**：低数据失败是 **release 掉 + timeout**
   （场景错定位尾差 10–28mm 接近 place_tol 20mm），**不是 transport 物理滑移**；
   cliff <2.4k，~100% 到 ~10.8k。
→ 小规划器在 **~2.4–5k 合成场景（1–3 min GPU 训练）即达可用闭环**、MuJoCo 长程
安全从 1.2k 成立——这是与 0.5B 大模型对比时要量化的"数据/算力优势"曲线。

---

## R8 · Track A（2026-09-02/03）— SmolVLA-0.5B 真权重微调对照 = 诚实负面

**目标**：Q1（同等数据/harness 下 0.5B 与 2.44M-tiny 的能力差）+ Q2（决策 A 的
14/14 长程掉件是不是"不够 scale"）。用真 `lerobot/smolvla_base`（450.05M total /
99.88M trainable flow-matching expert）微调到决策 A 的同一 11,141 样本，在 eval_vla
同 harness（σ3.7mm、push ±10mm、no-break-on-drop、MuJoCo 裁决）上打点。
**本次架构调整：无**（这是"加一个规模对照点"的测量，不新增架构）。

**infra + 数据冒烟（pass）**：权重经 hf-mirror 拉全离线可载（`smoke_smolvla.py`
SMOKE_OK，select_action 0.3s）；`train_smolvla.py --smoke` 64 样本 overfit loss
0.12→0.04 → 变异配置（chunk 8 / n_action_steps 8 / 单相机 / state[7] / action[4]）OK。

**全量微调（noise-free，6000 步 batch8，val 0.0170）→ 闭环 NEGATIVE**：
- `eval_smolvla.py`：hard 2-cell + 60 步 trace **全 timeout（0 success）**——停在样本
  质心悬停、从不开合到抓取。
- **固定-state 场景 A/B（image 唯一变量，obj 在分布内）**：SmolVLA 换 obj 位置 →
  chunk 恒定 [0.481,0.455]、open 不闭合；SigVLA-tiny 同场景 → chunk 追踪 object、
  open→0。→ **tiny 从像素定位、SmolVLA 不读图**。
- **因果诊断链**（全做了）：
  1. 冻结 VLM prefix embedding 线性探针：obj/place xy **R² 0.90–0.96**（MAE 5–9mm）
     → 视觉信息在冻结塔里、线性可读。
  2. 强制图像监督（240 合成样本，固定 state）：decode **corr(obj) 0.99、MAE 2–6mm**
     → 架构（flow-matching + frozen VLM）本身完全能提取。
  3. **离线 confound 定量**：chunk0 目标从 state alone 线性 **R² 0.88/0.84/0.89/0.86**
     (x/y/z/open)，full 8×4 chunk 0.535 → 11k 样本的 loss 被"保持姿势"自相关主导，
     image 对 chunk0 的边际解释仅 ~12% → 100M expert 走 **state 捷径**。
  4. **容量解释**：2.44M 直接回归 tiny 受容量限制被迫经 image 泛化；100M flow-
     matching expert 容量足够在 offline 上过拟合 state 捷径而不学图。
- **修正尝试（state-noise 增强，σ_xy 10mm，12k 步 batch16，val 0.0155）**：排除 LR
  衰减卡死（restart val 0.0144 仍不定位）；state-noise 改善 grasp 相位但仍不定位
  （固定-state chunk 依旧 ~0.483 恒值）；闭环 smoke 0/6 全 timeout。

**结论（诚实、限定范围）**：
- 同一 11,141 样本 + 同一 harness：SigVLA-tiny（2.44M 直接回归）33/36 sim 且从像素
  定位 object；SmolVLA-0.5B（99.88M trainable）**0% sim、全 timeout**（两协议）。
  → **Q1：此任务/数据预算下小直接回归模型胜过 0.5B flow-matching BC**。
- 不是"SmolVLA 架构做不到"（合成监督 corr 0.99）；是 **flow-matching 间接监督 +
  over-capacity expert 在 confounded 小数据上取 state 捷径**、不提取转移性视觉 map。
- **Q2**：0.5B 的同一 BC 家族在 basic grasp 即失败 → 决策 A 的失败不是 scale；
  决策 B（0.27M 直接回归 planner + aware-NCA）仍是无 oracle 的胜者。限定：未穷尽
  10× data / VLM 去冻结。

产物：`scripts/{smoke_smolvla,train_smolvla,eval_smolvla}.py`、ckpts/smolvla_{ft,probe,
noisy}（存 norm=None，eval 反归一化复用 vla_tiny best.pt 的 ch/st stats）。

---

## 9. 最终对照：参数量 → 成功率的谱系曲线

决策 A 分布 + 决策 B 分布全部在同一 harness（硬：σ3.7mm + push ±10mm；长程：
MuJoCo ×60 重放）上测：

| 系统 | 可训练参数 | 硬 sim | MuJoCo 长程×60 | 备注 |
|---|---|---|---|---|
| NCA 盲基线 | ~0.11M | 25/36 (69%) | 11/15 掉 | 无物理 ctx |
| NCA aware | ~0.12M | 36/36 (100%) | 0/15 | **需 oracle 物理 ctx** |
| VLA-A（SigVLA-tiny） | 2.44M | 33/36 (92%) | **14/14 掉** | 端到端 BC |
| **决策 B（分层）** | **0.38M** | **36/36 (100%)** | **0/15** | **无 oracle** |
| SmolVLA-0.5B | 99.88M | 0%（全 timeout） | — | 不进入抓取 |

核心经验：**参数量不是单调越好**——0.38M 分层系统是唯一同时达成 sim 100% + 长程
0/15 且不靠 oracle 的配置；0.5B 的同一 BC 家族反而最差。真正的杠杆是**架构分解 +
归纳偏置**（反应式闭环执行器 + 直接回归感知分离 + 物理模态通道），不是缩放。

---

## 10. 跨版本关键工程修复清单（架构 bug → 修复 → 影响）

| # | Round | 修复 | 影响 |
|---|---|---|---|
| 1 | R1 | 回归任务 lr 1e-4→1e-3 | 专家学成 echo，修复后收敛 |
| 2 | R1 | roll-in 重训练 + 稠密 hold-at-goal | 闭环不卡死（echo 病态） |
| 3 | R1 | 完成判定 mean→max | SigLIP 闭环 85%→100% |
| 4 | R1 | completed 过渡约束（完成→only next / 超时→{self,next}） | 链不卡死 |
| 5 | R2 | release 分支先于 F_n<held_req 检查 | 放置不再被误判掉件 |
| 6 | R2 | 位置感知 detach | aware 恢复极端硬单元 |
| 7 | R3/R4 | 解析载荷必须 plant 一致（a=K2·Δx + 矢量和）；τ 0.05 | 诚实物理裁决 |
| 8 | R4 | MuJoCo 容量校准（二分 pad gap → F_n 30N） | 滑移 onset 对齐诚实 risk |
| 9 | R4 | `sqrt(0)` 反传 NaN → +eps | 训练稳定 |
| 10 | R4 | 诚实滑移率仍 40–100× 高估（`(risk−1)·dx` vs 刚体） | outcome 复现失败（诚实负面） |
| 11 | R5 | VLA 数据每维归一化（position ~5mm vs open 0.4） | 避免 open 主导 MSE |
| 12 | R5 | clip 局部变量 / 解码器瘦身到 2.44M | 正确参数量统计 / 小模型对照 |
| 13 | R8 | flow-matching state 捷径（11k 样本 confound） | SmolVLA 0% 闭环（诚实负面） |

---

## 11. 待办 / 未决

1. **决策 B 泛化边界**（可选）：真摄像头多帧闭环（EEF 视野）或扩展抓取扰动分布——
   当前是静态顶视单帧。这是验证"无 oracle 复现"能否延伸到真实感知的关键。
2. **VLA 路线续作**（预算内不划算、未做）：10× data / VLM 去冻结 / 直接回归式辅助
   监督，验证能否打破 SmolVLA 的 state 捷径。
3. **执行器声称定位为 sufficiency（实例够用）、非 necessity（必须）**：保留
   NCA/MoE 架构与叙事（涌现、同质、事件驱动路由是创新载体），但论文把执行器贡献
   写成"这个以自组织 NCA 专家 + 路由实现的反应式闭环执行器（0.12M、吃物理 ctx）
   经 MuJoCo 验证够用"，并加一行边界："未建立 NCA/MoE 是唯一或必要实现——承重的
   是'吃物理 ctx 的反应式闭环执行器'这一角色，该角色以 NCA 实例化"。单一体反应式
   执行器对照是**可选强化项**（两种结果都可用：有 gap → NCA 必要；无 gap → 仍主张
   sufficiency），不在关键路径上。另：路由层的正当性在**多技能组合/泛化**叙事，
   单一任务 benchmark 不承载它（固定链状态机即可），这点写清楚反而护住其创新位置。

# Experiment Log — Soma

## Phase 6 — Coulomb physics modality + implicit action planning (2026-08-28/29)

Goal: give the homogeneous body-graph NCA a physical modality channel
(friction / dynamics / contact) and probe whether it learns *implicit action
planning* (load-adaptive grip/speed) rather than hand-written rules.

Design (see plan `binary-roaming-gem.md`): analytic differentiable Coulomb
model (`soma/physics.py`), per-step `physics_ctx[9] = [μ, m, F_n, slip_risk] +
one-hot(contact 5)` broadcast into the update rule (54→63 dim), and a slip
loss `relu(Σ relu(risk−1)·|Δx|·held / SLIP_THRESH − 0.5)` training the 4 held
skills (grasp/lift/transport/place). F_t = m·(g0·held + |Δx|/τ²) linear
worst-case-aligned (quadrature form makes the inertial term a 2nd-order
correction — unobservable). τ=0.1s calibrated so at full grip the blind
baseline risk ≈ m/μ, i.e. genuine slip onset at m/μ > 1 (margin ≲ 1.1 N).

### Round A — first build + first gate (h)

- Trained `checkpoints_phys/` (uniform feasible (μ,m), w_slip=5.0, 1500 steps).
- **gate (h) v1: blind 63/80 (79%) vs aware 64/80 (80%) — no h1 gap.**
- Diagnosis:
  1. **Release artifact (both agents)**: `_coulomb_step` checked `F_n < held_req`
     (→ drop) *before* `g ≥ 0.5` (→ release). Since g≥0.5 implies F_n ≤ 7.5 N <
     held_req on every hard cell, the release branch was dead code — every
     placement over the target was scored a drop. → **Fix 1**: reorder
     GRASPED/SLIP branches to check the intentional release first.
  2. **Cell sampling misses the slip band**: uniform feasible (μ,m) has mean
     m/μ≈0.6; the blind only genuinely slips at m/μ>1. With the artifact fixed,
     blind 77/80 vs aware 78/80 — the remaining hard cells (margin 2–3 N) don't
     slip for the blind at its ~52 mm steps.
- Round A already showed the core signatures: **h2** aware transport step
  contracts with margin (16.8→23.4 mm) vs blind flat ~53 mm; **h3** morphogen
  ablation null.

### Round B — sim modeling fixes + hard-tail experiment + final

- **Fix 2 — position-aware detach**: `dropped` is no longer unconditional. The
  object detaches wherever support is lost (F_n < held_req or slip past
  threshold); success is decided by *where* obj_final lands (within place_tol
  of the target = placed). This is what let the aware actually recover the
  extreme cells (its release opening passes through the F_n < held_req window
  before g reaches 0.5).
- **hard_frac experiment (discarded)**: retrained with 50% m/μ∈[1.02,1.32]
  hard-tail samples (`checkpoints_phys2/`). Result: globally conservative —
  transport step flat ~20 mm, h2 contraction lost, still dropped the extreme
  cells. Uniform training is strictly better (keeps margin-conditional
  contraction *and* recovers hard cells).
- Tightened eval hard band to m/μ∈[1.08,1.35] so the blind's genuine slip cells
  are densely sampled.
- **Final gate (h)** (uniform experts + position-aware detach, 40 cells × 3 eps):
  - **h1**: aware 119/120 (99%) vs blind 106/120 (88%); dropped 1 vs 14.
    Aware recovers the m/μ>1 hard tail. Note: ctx-zeroed aware also 119/120 —
    the hard-tail capability comes mostly from the slip-loss-trained
    conservative prior, not from inference-time ctx.
  - **h2**: with ctx, transport step contracts 12.2→23.8 mm with margin (~2×,
    monotone); ctx-zeroed aware flat ~19 mm; blind flat ~53 mm. The
    margin-adaptive plan is driven specifically by the modality channel
    (same-expert on/off ctx control).
  - **h3 (negative)**: morphogen warm-start ablation 119→117/120. Since (μ,m)
    is constant per episode and fully present in the ctx every step, the plan
    is a context→action policy re-derived within a single relax — not carried
    in cross-step morphogen memory.
- **Regression**: `eval_closed_loop.py --ckpt-dir checkpoints_phys` — gates
  e/f/g all pass on the Coulomb sim (20/20; robust 20/20; ~944–951 Hz GPU).
  Blind rollin2 unchanged (100% on the easy default cell).

### Conclusion

A homogeneous NCA given a physical modality channel learns an implicit
load-adaptive speed plan (margin→step contraction, ~2×) and thereby recovers
hard cells the blind baseline drops (99% vs 88%). The plan lives in the
context→action policy, not morphogen memory. Keep `checkpoints_phys/` (uniform
training); `checkpoints_phys2/` (hard_frac) is a discarded experiment.

## Level A — MuJoCo 独立物理交叉验证（2026-08-31，诚实负面）

**Goal**: 把 gate (h) 的 outcome 裁决从解析库仑模型（soma/physics.py）换成
MuJoCo 3.11 接触求解器——策略决策（NCA 闭环轨迹）不变，物理完全独立。主张：
盲基线在 m/μ>1 硬单元的 ~52mm transport 步让对象在 MuJoCo 里真实滑出；aware
收缩步持稳。计划诚实性条款：两个方向的失败都是有效科学发现。

**模型（`scripts/eval_level_a.py`）— pseudo-force / co-moving frame**（重写）：
- 尝试 1 废弃：双 pad 力伺服抓取（slide-joint pad + motor ctrl=F_n，hand 运动
  学驱动）。**MuJoCo 切向摩擦对动态 body 是软的**——物体全重压在两动态 pad
  的静摩擦上，solver 以 ~20mm/s 悄悄滑移（world geom 则 ~0.12mm/s 干净保持），
  该蠕动直接淹没 transport slip 信号。
- 最终：**pseudo-force 模型**。pad 是固定 world geom（干净保持）；hand 运动作为
  惯性伪力施加：每子步 `opt.gravity = (k²·Δx, k²·Δy, −g0 + k²·Δz)·(1−plant_f)^u`，
  k = −ln(1−plant_f)/τ = 6.93（plant_f=0.5, τ=0.1）。Galilean 等价使 onset 与真实
  移动抓手一致（对象相对 pad 的滑移在两参考系相同；只省略无关的绝对运动）。

**诚实物理学的两个后果（负面结论的机制）**：
1. **沿 pad 法向（x）的 transport 加速度由接触法向承载**（对象被 capture 在两 pad
   之间），不是摩擦；只有 pad 切向（y, z）分量是摩擦载荷。解析模型把整个标量
   m(g0+a) 记在摩擦上。
2. **plant 真实峰值加速度 = k²·Δx ≈ 48·Δx，不是解析的 a = Δx/τ² = 100·Δx**
   （~2× 高估）。
解析滑移边界 m/μ>1 落在诚实静态边界 m/μ≈1.42–1.53 之下；可行带内 m/μ ≤ 1.376，
故诚实物里下任何单元在 transport 都不滑。

**`--calib` 单步滑移 onset（验证机制）**——硬单元 (0.20, 0.27) m/μ=1.35：
```
D_mm  risk_an  risk_y  risk_x |  mj_y   mj_x
  25    1.11*    0.89    0.88 |  0.02   0.02
  40    1.24*    0.90    0.88 |  0.02   0.02
  55    1.38*    0.91    0.88 |  0.02   0.02
  70    1.51*    0.93    0.88 |  0.03   0.02
```
`risk_an`（sim 的载荷公式）在 D≥25mm 越过 1（* = 解析判滑）；诚实切向 `risk_y`
全程 ≤0.93 < 1；MuJoCo 实测滑移 0.02–0.03mm。

**transport 重放**（3 个 seed：混合带 + 全硬带，25–40 cells × 3–8 eps）：
- **MuJoCo 重放 730 段 transport（aware+blind），0 段滑出**（最大 slip 0.15mm，
  阈 15mm）。
- sim 判盲基线 transport 掉件 **14 段，MuJoCo 独立复现 0/14**（全部 HELD，
  slip ≤ 0.15mm）；aware 的 sim transport 掉件同样全部被 MuJoCo 持稳。

**结论（诚实负面）**: Level A 通过标准"判别单元上 MuJoCo 独立复现
aware-held / blind-dropped"**未达成**。盲基线在 m/μ>1 的 sim 掉件是解析库仑模型
载荷参数化的产物（标量 m(g0+Δx/τ²) + a=100·Δx 的惯性高估），在 MuJoCo 刚性体
物里下不存在——诚实的切向载荷（重量 + 切向惯性的矢量和，峰值 a=48·Δx）在可行带
内始终低于摩擦容量。因此 gate (h) 的 **outcome 主张（aware 收回复盲掉件）不成立**：
诚实物里下盲基线同样持稳。h2（步长随余量收缩）是真实学到的行为，但解决的是一个
诚实物里下不存在的问题。h3（计划在 context→action 策略，非 morphogen 记忆）不受
此影响。

保留 `scripts/eval_level_a.py`（--calib + 重放，可再生）。回归：未改
sim/physics/controller 代码，eval_closed_loop 结果不受影响。

---

## Level A+ — 诚实重建 + MuJoCo 可复现滑移带（2026-09-01）

**目标**（计划 binary-roaming-gem.md）：把解析库仑模型改造成 plant 一致，重标定
τ 到快臂 regime 找到真实、MuJoCo 可复现的滑移带，重训 aware，重跑 gate (h)，
MuJoCo 裁决。

### 诚实重建（`soma/physics.py`）
- τ 0.1→**0.05**；峰值加速度 `a = K2·|Δx|`，`K2=(−ln(1−0.5)/0.05)²≈192`
  （原 `a = dx/τ² = 100·dx` 高估 ~2×）。
- 切向载荷改**矢量和** `F_t = m·sqrt((g0·held)² + a²)`（原标量 `m·(g0·held+a)`）。
- `SLIP_THRESH` 5mm→15mm（对齐 MuJoCo 重放阈）；`HARD` 带 [1.10, 1.35]、
  `EVAL` 带 [1.16, 1.26]。
- **训练中发现的 bug**：`sqrt((g0·held)²+a²)` 在首步 (held=0, a=0) 反传
  `inf·0=nan`（physics_fn 以 prev_reseed==prev_target 播种），每轮 grasp 训练
  CUDA 发散。修复 `+eps` 入 sqrt（3000 步训练干净，loss 0.36）。

### MuJoCo 容量校准（Level A "0 滑移"的真根因）
- MuJoCo 摩擦是干净 Coulomb（有效 μ==mu_pad），但 pad gap 沉降出
  **F_n_total≈75N** 而非意图的 30N：盒-盒软接触刚度 ~165kN/m（假设 65e3 的
  2.5×），且 F_n 随物重缩放。容量 2.5× 高于解析 μ·F_max → 无步滑。
- 修复：按 (mass, Fn) **二分搜索 pad gap** 使沉降 F_n_total=2·F_MAX=30N。
- 校准后 MuJoCo 单步滑移 onset **对齐诚实 risk**（`--calib`）：
  m/μ=1.35 D≥40mm 滑（mj_y 0.06→0.38mm）、m/μ=1.20 40-55mm onset、
  m/μ=0.50 永不滑（mj_y≤0.02mm）。

### 剩余诚实负面——滑移率不匹配
MuJoCo 物理滑移率 **~40-100× 慢于** sim 的解析累积 `slip_disp += (risk−1)·dx`。
risk≈1.12、D=52mm：sim 6.2mm/step vs MuJoCo ~0.17mm/step；累积 15mm 需 2-3 步
vs ~90 步。真实 transport 仅 3-8 步 → MuJoCo 重放盲基线硬带 transport
**<1.5mm 累积 → HELD**。**τ 张力**：τ=0.02 让 MuJoCo 在短程 transport 内掉件，
但 K2≈1200 连易单元（m/μ≈0.6）也滑（全保守化）；无 τ 同时满足"短程掉件"+
"易单元不滑"——判别带对诚实刚体滑移率过窄。

### gate (h) 重跑（`checkpoints_phys_honest`，40 cells × 3 eps，hard-frac 0.5）
- **h1**：aware 120/120（100%，0 掉）vs blind 84/120（70%，36 掉）。盲正好掉
  低余量单元（margin<~1.1N，即 m/μ>~1.05）；aware 全部恢复。诚实模型下硬尾是
  唯一判别区。
- **h2 SPLIT**：**lift** margin 条件收缩 24.8→38.3mm（余量 0.5→7.8N，~1.5×），
  ctx 置零压平到 ~26mm —— 真实 **ctx→action 隐式速度计划**。**transport** 统一
  ~30mm 保守步（vs blind ~53mm 平坦），非 margin 条件：hard-frac 0.5 训练让
  transport 重现旧 checkpoints_phys2 全保守化模式，只有 lift 保留自适应；ctx
  置零 transport 仍 ~30mm → 统一步速是训练先验、非推理期 ctx。
- **h3**：morphogen 热启动消融 120/120 vs 120/120 —— NULL（不变：计划在
  context→action 策略，非 morphogen 记忆）。

### MuJoCo 裁决（20 cells × 3 eps，worst-case yaw）
- **OUTCOME NOT CONFIRMED**：blind transport sim 判掉 **2 段**，MuJoCo 复现
  **0/2**（全 HELD：slip 0.08/0.27mm，min dz −0.08/−0.20mm）。sim 的 blind 掉件
  （gate h 36/120，多数是 **lift 掉件**——55mm 竖直步在硬尾也越过 risk>1）仍是
  解析滑移率伪差。
- **MECHANISM CONFIRMED (modest)**：判别硬带（m/μ>1.1）MuJoCo 滑移盲恒定高于
  aware——blind 均值 **0.23mm** vs aware **0.10mm**（~2.3×），盲从未在带内任何
  单元低于 aware。全单元聚合稀释到 0.13 vs 0.09mm（1.4×）。量级 **sub-mm**：
  短真实 transport 无法累积 15mm 刚体滑移，这正是掉件 outcome 不可复现的原因。

### 结论（诚实）
gate (h) **outcome 主张（aware 收回复盲掉件）不被 MuJoCo 独立刚性体物里复现**
——现在是解析滑移率（~40-100× 过度），而非容量（已修）或 onset（已验证）。
**机制主张被复现但量级 modest**：判别带上 aware 收缩步比盲基线累积更少的刚体
滑移（带均值 0.10 vs 0.23mm，~2.3×），是真实、刚体有效的行为，但短程 transport
下为亚毫米级、远低于 15mm 掉件阈。h3 不受影响。**h1（sim 能力）是最强诚实正
结果**：诚实模型下 aware 硬尾 100% vs 盲 70%。

保留 `checkpoints_phys_honest/`、`scripts/eval_level_a.py`（--calib + 重放）、
`scripts/eval_implicit_planning.py`。回归：eval_closed_loop gates (e) 20/20、
(g) 436 Hz 通过（默认单元 m/μ≈0.25 远低于任何滑移 onset）。一处行为观察：
honest-aware 的 grasp 技能在闭环中不再登记正式 NCA 完成（滑移损失把下降步收缩
→ duration 超时转 lift），但对象仍被功能抓取、episode 20/20 成功——良性行为
偏移，非门禁回归。

---

## Phase 7 — 单一任务小型 VLA 的评估分布 + NCA 基线（2026-09-01）

**新方向**（用户决策）：单一任务 VLA，参数目标 10⁶–10⁷，任务成功率要高于
NCA 基线。两步走：先定基线（本段），后搭 SmolVLA-0.5B 微调基线对照。上一方向
（CerebVLA/MoE-NCA）按用户要求不再作为主线介绍。

**评估分布**（`scripts/eval_baseline_hard.py`，SmolVLA 将测同一分布）：
- **硬分布**（per-episode）：hard-frac 0.5 抽样 m/μ∈[1.16,1.26] 判别带；
  场景感知噪声 σ=3.7mm（SigLIP 级，控制器看到的 scene 加噪，env 用真值判成功）；
  transport 首两步之间 ±10mm 中途推扰（transport 专家从扰动位重规划）。
- **长程 transport 分布**：把记录的 transport 段连续重放 Kmax=60 次，让 MuJoCo
  刚体滑移累积越过 15mm 掉件阈——回答"判别带在刚体物里下是否真实"。

**裁决**：sim（闭环成功，obj 落在真 place 的 place_tol 内）+ MuJoCo 独立重放
（worst-case yaw，FixedPadsReplay 伪力模型）。

### NCA 基线测量（锚点，~10⁵ 参数）
盲基线 114K / aware 118K 参数（6/4 个 body-graph NCA 专家 + MoE 路由器）。

**硬分布**（12 cells × 3 eps；σ=3.7mm，推 ±10mm）：
```
agent              | sim ok   | sim drops | MuJoCo tseg | MuJoCo held | mj slip
blind (no physics) | 25/36 69%|  11       | 36          | 36/36       | 0.08 mm
aware (physics ctx)| 36/36 100%|  0        | 36          | 36/36       | 0.10 mm
```
- sim：aware 100% vs blind 69%；blind 的 11 掉件全在 m/μ≥1.16 单元（8 lift + 3
  transport，lift 掉件为主——55mm 竖直步在硬尾也越过 risk>1）。
- **MuJoCo 短程 transport：0/3 复现 blind 的 sim transport 掉件**（全部 HELD，
  slip 0.08–0.10mm）——已知的诚实负面（解析滑移率伪差）在硬分布下依旧成立：
  blind 的 sim 掉件主要不是刚体真实滑出。

**长程 transport 分布**（band m/μ∈[1.26,1.35]，10 cells × 3 eps，×60 重放）：
```
agent              | dropped by Kmax | median cycle-to-drop | median max slip @Kmax
blind (no physics) | 11/15 (73%)     | 28                   | 15.02 mm
aware (physics ctx)|  0/15 (0%)      | none (< 60)          |  7.78 mm
```
- **MuJoCo 独立刚体物里下第一个干净的 outcome 复现**：长程携带下盲基线的 ~52–80mm
  大步（risk 1.3–1.65，扰动后更大）让刚体滑移累积越过 15mm 阈（73% 段掉件），
  aware 的收缩步（~30mm，risk<1 或略超）保持（0/15，中位 7.78mm < 阈）。
- 率判别：盲中位 cycle-to-drop 28 vs aware >115（~4× 慢）——aware 在硬尾
  (m/μ=1.35) 仍以 ~38mm 步略超 risk=1，弹性蠕滑 ~0.13mm/cycle，非完全免疫，但
  比盲慢一个数量级量级。
- **诚实要点**：aware 的 sim 硬分布优势部分不可 MuJoCo 复现（短程 slip-rate
  伪差）；但长程分布把 outcome 主张救回——延长暴露后盲基线真实刚体滑出，aware
  持稳。这是 SmolVLA 对照的第一个可复现 outcome 基准。

### SigVLA-tiny 基线（决策 A 端到端 VLA，2.44M trainable / 95.4M total）
HF 不可达、无 SmolVLA 权重，改用 **冻结 SigLIP-B16（92.9M，0 训练）+ 紧凑
cross-attn 动作解码器（2.44M 训练）** 作为 SmolVLA-faithful 替代（接口
image+state → 8 步绝对位姿 chunk，后续可换真 SmolVLA-0.5B）。已在 README 中
诚实记录为"受限替代"而非改名。

- 数据：aware-NCA 闭环 demo 2000 集（hard-frac 0.5，band [1.16,1.26]）→
  11,141 个 (image, chunk[8,4], state[7]) 样本（`scripts/vla_data.py`）；
  每维归一化（position std ~5mm vs open std 0.4，避免 open 主导 MSE）。
- 训练：4000 步，AdamW + cosine，val MAE **7.6mm**（<< place_tol 20mm）。
- 架构修过两次：`self.clip` 会被 nn.Module 自动注册（把 110M text tower 也算进
  trainable 139.8M → 改成局部变量）；默认 d=768/3L/8H → d=256/2L/4H + 768→256
  patch 投影，trainable 30M → **2.44M**。

**同一硬/长程分布评估**（seed 0，12 cells × 3 eps；与 NCA 锚点同 harness）：
```
agent                    | sim ok (hard) | MuJoCo 短程 tseg | 长程×60 掉件 | median cycle-to-drop | median slip
blind NCA  (0.11M)       | 25/36 (69%)   | 36/36 held       | 11/15 (73%)  | 28                   | 15.02 mm
aware NCA  (0.12M)       | 36/36 (100%)  | 36/36 held       |  0/15 (0%)   | none                 |  7.78 mm
VLA-tiny   (2.44M train) | 33/36 (92%)   | 33/33 held       | 14/14 (100%) | 10                   | 15.07 mm
```
- **短程 sim**：VLA 92%（3 timeouts，0 掉件）夹在盲 69% 与 aware 100% 之间——
  感知噪声 σ=3.7mm 下仍能近似复现 aware 的短程闭环成功。
- **margin→transport 步**：VLA 有弱单调收缩（高 margin 30.6mm → 低 margin
  23.9mm）——从像素（尺寸→质量、色调→μ）学到了一部分载荷自适应，但收缩太弱、
  不是 aware 的陡峭条件收缩。
- **长程 MuJoCo：VLA 14/14 掉，cycle-to-drop 10——比盲基线（11/15，cycle 28）
  还差 ~3×**。clean 分布（σ=0、无推）同样 15/15 掉（cycle 9）→ 与扰动无关。
- **机理**：诊断记录显示 VLA 携带期 grip 正常（mean open 0.02–0.04，Fn≈0.97
  F_max），但 **max|D| ≈ 37–52mm ≈ 2× mean|D|**（mean 23mm）——8 步 chunk 开环
  执行 + plant 滞后（plant_f=0.5）→ EEF 落后于陈旧 chunk → 峰值命令位移周期性
  越过 m/μ≈1.2 的 ~40mm risk 边界 → 刚体滑移每 cycle 累积。aware NCA 逐步闭环
  重规划 + 每步 slip_risk 反馈保持步长紧致（max≈mean），是持稳的关键。
- 逐步重规划变体（chunk-1，per-step 逐点匹配闭合，本 session 收敛后重测）：
  ```
  VLA-tiny chunk-1 | sim 21/36 (58%) | MuJoCo 短程 33/33 held (slip 1.77mm) |
                    | 长程×60 13/13 掉 (cycle 10, 15.03mm) | margin→step 18→26mm
  ```
  回归精度好（val MAE **7.2mm**，chunk-8 是 7.6mm），但逐点重解码后 sim 反而
  更差（58%，6 timeouts），长程依旧 13/13 掉（cycle 10）——与 chunk-8 相同。

### 对照实验：结构本身是否可用（2026-09-02）
用户问："目前的结构本身是可用的吗，能否在对照实验中证实"。设计了两个控制实验，
把"结构信息上限"与"BC 训练/闭环执行伪差"分开：

**Control 1 — 感知半是否信息受限**（`scripts/probe_margin.py`）：
冻结 SigLIP-B16 特征的 mean-pool → 回归 margin = μ·F_max − m·g0。
- **val MAE 0.32 N**（margin std ~1.9N，判别带 0.6–1.1N）→ 编码器通道携带足够
  的载荷物理信息（质量→尺寸、μ→色调在像素里可读）。**结构在感知端不瓶颈。**

**Control 2 — 匹配闭合下闭环是否变安全**（chunk-1 per-step，本 session）：
数据改为每步渲染/每步预测下一目标，评估也逐点重解码，闭合匹配；模型回归
7.2mm MAE（好），但 **长程仍 13/13 掉（cycle 10），sim 58%（反而更差）**。
→ 匹配闭合没有救回 outcome：chunk 滞后不是根因。

**滑移机理分解**（`/tmp/diag_grip.py`，replay 级 grip 消融）：
- 执行 grip 在携带期保持闭合（open 0.005–0.019，Fn/F_max 0.98–0.995）；把
  Fn 强行为 F_max（全闭合）重放 cycle-to-drop 基本不变（c8↔c8, c8↔c9,
  c11↔c12）→ **不是 grip 通道**。
- 滑移来自命令运动本身：携带期 **maxD 水平尖峰 34–47mm**（m/μ≈1.2 下越过
  ~40mm risk 边界）+ **携带期 z 振荡**（mean |Dz| 7.6–16.5mm；aware 携带期
  保持 z 恒定）。z 下探瞬态松脱夹持、水平尖峰超摩擦容限 → 每 cycle 定向漂移
  ~1.5mm，10 cycle 累计 15mm。
- 机理本质是 **BC 协变量漂移**：模型在 aware-NCA 状态分布上训练，自驱动后状态
  分布偏移 → 预测噪声反馈放大 → 轨迹退化为 z 振荡 + 步长尖峰。aware NCA 的
  per-step slip_risk 反馈 + z 恒定 + 陡峭 margin 收缩，是它持稳的机制，BC 学不来。

**诚实回答用户**：结构可分割地"可用"——感知半读得到 margin（C1，0.32N），
短程回归精确（C2，7.2mm）；但**仅靠 open-loop BC 训练的闭环策略不复现 aware 的
安全行为（z 恒定 + 载荷自适应收缩），匹配闭合后长程依然滑出**。失败在行为/目标
层（BC + 协变量漂移），不在编码器信息层。这进一步支持决策 B（分层：tiny VLA
规划器 + NCA 执行器）——执行器保留安全行为，VLA 只加感知/规划；若坚持决策 A，
需闭环训练（DAgger/自举 rollout）、slip-loss 监督、或改变动作空间（delta 命令 +
grip 纪律）来弥合。

### 结论（决策 A 的基线数据点）
端到端 tiny VLA 短程 sim 成功率可达到 aware 水平（chunk-8 92% vs 100%），
学会了像素级弱载荷自适应，但**没有复现 aware NCA 的 MuJoCo 有效 outcome 主张**
——无论开环 chunk 还是逐点匹配闭合，长程携带都真实滑出（cycle 10，比盲基线
cycle 28 快 ~3×），根因是 BC 学不到 aware 的携带期纪律（z 恒定 + 收缩）。这
**指向决策 B（分层）** 或"端到端 VLA 需闭环训练 + slip 反馈"——A/B 对照中都需被测。

### 待办
1. （可选）VLA 改进迭代：滑动窗口重规划（chunk-2/4 折中）、delta-chunk 平滑目标、
   slip 反馈通道——验证峰值步问题是否可缓解（当前报告的是不调参基线）。
2. SmolVLA-0.5B 数据生成 → 微调 → 同一两分布评估（参数/成功率对照曲线：
   NCA 0.1M / VLA-tiny 2.44M / SmolVLA 0.5B 三个点）。

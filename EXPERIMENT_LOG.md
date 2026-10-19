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
NCA 基线。两步走：先定基线（本段），后搭 SmolVLA-0.5B 微调基线对照。上一方向（自包含的 MoE-NCA
路线）按用户要求不再作为主线介绍。

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

### 决策 B：分层规划器 + aware-NCA 执行器（2026-09-02，PASS）
用户选定决策 B，且规划器**场景 + 物理全部从像素读出**（不剩任何 env-truth oracle，
与端到端 VLA-A 的对照才公平）。架构：tiny SigLIP 规划器头回归 6 向量
`[obĵ_xy, placê_xy, μ̂, m̂]`，注入 aware-NCA 闭环中 NCA 基线收 oracle 的两处：
- scene → `ctl.reset([obĵ,placê], state)`（路由 + 各技能 xy 目标）
- physics_ctx → 9 维 ctx 用 (μ̂,m̂) 重建（aware 步长收缩的输入）

env/MuJoCo 仍以 TRUE 物理 + TRUE place 裁决滑移/掉落/成功 → 若 μ̂/m̂ 低估载荷
（m/μ̂ 过低）→ 执行器欠收缩 → 真实掉件。这是无 oracle 的诚实测试。

**数据**（`scripts/planner_data.py`，12k 场景）：顶视 256 RGB 只画静态场景
（绿 place 环 + 物体矩形，边长∝mass 12–28px、色调∝μ，与 VLA 渲染同编码），
无 EEF 点（规划器是静态场景估计器）。场景分布 = cells()：hard-frac 0.5 落在
m/μ 判别带 [1.16,1.26]，m/μ 全域 [0.09,1.37]。

**训练**（`scripts/train_planner.py`）：冻结 SigLIP-B16 mean-pool → 0.27M MLP 头
（LN→768→256→256→6），6000 步 lr 5e-4 + warmup200 + cosine，val MSE 0.0216。
- val per-dim MAE：obj_x 4.37 / obj_y 2.90 / place_x 3.35 / place_y 3.43 mm
  → **scene xy MAE 3.51 mm**；μ MAE 0.01、mass MAE 0.01
- **m/μ rel err：mean 7.2%，hard-band 4.7%**；hard-band 低估(>10% 低) 6%
  → 远在 ±10% 因果阈值内。

**闭环评估**（`scripts/eval_hierarchical.py`，12 cells × 3 eps，MuJoCo 长程 ×60）：
```
── Oracle-aware（control：TRUE physics ctx，同批 cells）────────────
  sim 36/36 (100%)  |  MuJoCo long-transport 0/15 掉 (median slip 7.06mm)
── Decision-B（planner 从像素读 scene+physics）────────────────────
  sim 36/36 (100%)  |  MuJoCo long-transport 0/15 掉 (median slip 9.00mm)
  planner m/μ rel err 8.8% (mean)，under-estimate 31%（都 <10% 量级）
```
- Oracle 控制行把 aware 锚点（36/36、0/15）在脚本内复现 → harness 对齐。
- B 行 12/12 cells 全 3/3，含最难的 m/μ 1.16–1.24、margin 0.6–0.9 cells；
  各 hard cell μ̂ err 0.003–0.010、m̂ err 0.006–0.019（相对 ~1–6%），31% 的
  低估都在伤害阈值（>10%）之下 → 收缩保持充分，长程 max slip 9.0mm < 15mm。
- B 比 oracle 严格难（多 ~3.5mm 场景回归误差 + 物理误差），仍复现 oracle outcome。

**对照结论**：分层系统（SigLIP 规划器 + aware-NCA 执行器）**在无 oracle 下复现了
oracle-aware 的 MuJoCo 有效 outcome**（36/36 hard sim，0/15 长程）——这正是决策 A
做不到的（VLA-A sim 33/36 92% 但长程 14/14 掉、cycle 10）。参数：NCA 执行器 118K
+ 规划器头 266K ≈ 0.38M trainable（SigLIP-B16 92.9M 冻结推理），仍在 10⁶ 小参数量级。
分层把"感知/规划"给 tiny VLA、把"安全执行"留给已持稳 MuJoCo 的 NCA → 两半各尽其长。

### 决策 B 数据效率：数据预算 → 闭环 outcome（2026-09-02）
小模型优势要落在数据轴上。扫规划器训练数据预算 N（固定 ~18 epochs、共享固定
1200 张 val；`scripts/sweep_planner_data.py` + `scripts/report_planner_sweep.py`）：
```
N      场景MAEmm  obj  place   μ      m     hard m/μ rel  低估%    GPU训练
 1200    6.63    8.44  6.22  0.023 0.015     7.9%        16%      ~44s
 2400    5.18    6.35  5.12  0.019 0.013     6.1%        11%      ~82s
 4800    4.38    5.46  4.27  0.016 0.012     5.2%         7%     ~156s
 9600    3.61    4.48  3.51  0.015 0.011     4.7%         5%     ~290s
10800    3.51    4.37  3.35  0.015 0.011     4.7%         6%      ~5min
```
（首版曲线因报表去归一化 bug 报 30mm 场景误差——模型 valMSE 其实与全量相当，
修 bug 后为上方真值；`report_planner_sweep.py` 重读 ckpt 重算。）

同批 cells 的闭环 outcome（12×3，MuJoCo 长程×60；oracle 控制行三次恒为
36/36 + 0/15 slip 7.06mm → harness 确定性确认）：
```
N      hier sim ok   release掉  timeout   MuJoCo长程×60   median slip
 1200   27/36 (75%)     4        5          0/15           6.58mm
 2400   32/36 (89%)     4        0          0/15           5.75mm
 4800   31/36 (86%)     4        1          0/15           6.54mm
10800   36/36 (100%)    0        0          0/15           9.00mm   (全量 best.pt)
```
**结论分成两条轴**：
1. **安全轴（MuJoCo 长程）= 数据鲁棒**：低至 1.2k 场景（GPU ~44s）就 0/15 不滑
   ——aware 执行器的保守收缩覆盖粗糙的 μ̂/m̂（N=1200 低估 56% 也不掉件）。
2. **任务完成轴（sim 成功率）= 需更多数据**：低数据失败是 **release 掉 + timeout**
   （场景错定位：N=1200 obĵ/placê 尾部 10–28mm，接近 place_tol 20mm → 释放点
   偏、或超时），不是 transport 物理滑移。cliff 在 <2.4k，~100% 要到 ~10.8k。
→ 小规划器在 ~2.4–5k 合成场景（1–3 min GPU 训练）即达可用闭环、MuJoCo 长程
安全从 1.2k 就成立——这是与 0.5B 大模型对比时要量化的"数据/算力优势"曲线。

### A线：SmolVLA-0.5B 真实权重微调 → 同一闭环分布对照（2026-09-02/03，诚实负面）
**目标**：Q1（同等数据/harness 下 0.5B 与 2.44M-tiny 的能力差）+ Q2（A 的 14/14
长程掉件是不是"不够 scale"）。用真 lerobot/smolvla_base 权重微调到决策 A 数据
（11,141 个 chunk-8 样本），在 eval_vla 同 harness 闭环（感知噪声 σ3.7mm、
mid-carry push ±10mm、no-break-on-drop、MuJoCo 裁决）上打点。

**infra + 数据冒烟（pass）**
- 权重（smolvla_base + SmolVLM2-500M-Video-Instruct）经 hf-mirror 拉全、离线可载；
  `smoke_smolvla.py` SMOKE_OK：450.05M total / 99.88M trainable，select_action 0.3s，
  finite [1,6] action。
- `train_smolvla.py` 微调（post-load 变异 cfg：chunk 8 / n_action_steps 8 / 单相机
  camera1[3,256,256] / state[7] / action[4]；flow-matching loss 掩到真实动作维 [:4]；
  数据 npz 图像 [-1,1]→*0.5+0.5 还原 [0,1] 喂入——与 tiny 同分布）。`--smoke`
  64-sample overfit：loss 0.12→~0.04 → 反向/变异配置 OK。

**全量微调（noise-free，6000 步 batch8 ≈4.5 epoch，val 0.0170）→ 闭环 NEGATIVE**
- `eval_smolvla.py`（run_episode_vla/collect/report 从 eval_vla 逐字节继承）：
  hard 分布 2-cell + 60 步 trace：**全部 timeout（0 success / 0 drop）**——机器人
  停在样本质心 ~(0.486,0.455,z0.62) 悬停，从不开合到抓取。
- **固定 state 场景 A/B（image 唯一变量，obj 在分布内）**：
  - SmolVLA：obj 换 (0.40/0.45/0.47,·) → chunk 恒定 [0.481,0.455]、open 不闭合。
  - SigVLA-tiny 同场景：chunk 追踪 object（c6 收敛到 obj xy、open→0）→ **tiny 从
    像素定位、SmolVLA 不读图**。
- **因果诊断链**（都做了）：
  1. 冻结 VLM prefix embedding 线性探针：obj/place xy **R² 0.90–0.96**（MAE 5–9mm）
     → 视觉信息在冻结塔里、线性可读。
  2. 强制图像监督（合成 240 样本：固定 state、chunk=逼近 object、overfit）：
     decode **corr(obj) 0.99、MAE 2–6mm** → 架构（flow-matching + frozen VLM）
     本身完全能提取——不是结构做不到。
  3. 离线 confound 定量：chunk0 目标从 state alone 线性 **R² 0.88/0.84/0.89/0.86**
     (x/y/z/open)，full 8×4 chunk 0.535 → 11k 样本的 loss 被"保持姿势"自相关主导，
     image 对 chunk0 的边际解释仅 ~12%、转移到后段/决策窗 → 100M expert 走
     **state 捷径**拟合训练分布，不学闭环必需的像素→object 图。
  4. 容量解释：2.44M 直接回归 tiny 受容量限制被迫经 image 泛化；100M flow-matching
     expert 容量足够在 offline 上过拟合 state 捷径而不学图。

**修正尝试：state-noise 增强（σ_xy 10mm，12k 步 batch16 ≈19 epoch，val 0.0155）**
- 排除 LR 衰减卡死（probe +2000 步 restart val 0.017→0.0144，仍不定位）。
- state-noise 后 val < noise-free 地板（0.0155<0.017）、grasp 相位改善（open 端位
  0.23–0.49 闭合）——**但仍不定位 object**（固定-state 测试 chunk 依旧 ~0.483 恒值、
  dist(c0,obj) 与 noise-free 相同）。闭环 smoke **0/6、全 timeout**。

**结论（诚实、限定范围）**
- 同一 11,141 样本 + 同一 harness：SigVLA-tiny（2.44M 直接回归）33/36 sim 且从像素
  定位 object；SmolVLA-0.5B（99.88M trainable flow-matching expert-only 微调）无法
  学像素→object 定位 → **0% sim、全 timeout**（含 state-noise 增强）。→ **Q1：
  此任务/数据预算下小直接回归模型胜过 0.5B flow-matching BC**。
- 不是"SmolVLA 架构做不到"（合成强制监督 corr 0.99）；是 **flow-matching 间接监督 +
  over-capacity expert 在 confounded 小数据上取 state 捷径**、无法提取转移性视觉
  map——与小模型直觉一致：大模型在该 small-data synthetic 场景下的容量是负担而非优势。
- **Q2**：0.5B 的同一 BC 家族在 basic grasp 即失败（比 A 的 14/14 长程掉件更早、
  更基础）→ 决策 A 的失败不是 scale；决策 B（0.27M 直接回归 planner + aware-NCA）
  仍是无 oracle 的胜者。给定实现的限定：未穷尽数据规模（10× data）与 VLM 去冻结。
- 文件：`smoke_smolvla.py` / `train_smolvla.py`(--smoke/--init/--state-noise-mm) /
  `eval_smolvla.py`；ckpts/smolvla_ft(0.017) & smolvla_noisy(0.0155)（ckpt 存 norm=None，
  eval 反归一化复用 ckpts/vla_tiny/best.pt 的 ch/st stats——已验证与 npz 同源一致）。

### 待办
1. ~~SmolVLA-0.5B 微调对照~~（完成，诚实负面见上）。可选跟进：10× 数据 / VLM
   去冻结 / 直接回归式辅助监督，验证是否打破 state 捷径（预算内不划算，未做）。
2. （可选）把规划器接成真摄像头闭环（EEF 视野多帧）或扩展抓取扰动分布，
   验证 B 的泛化边界——当前是静态顶视单帧。

---

## 🔶 新突破方向 — 改变 NCA 在 VLA 架构中的形态（2026-10-13）

> 这不是又一个 Round，而是对**研究主线本身**的重新定位：此前所有 Round 都在调
> "NCA 周围的架构"（physics_ctx、路由、感知、对照），从未真正质疑 **NCA 的形态
> 本身**。本节把 NCA 的细胞定义从"语义自由度"改成"同质空间单元"，让 NCA 从
> 一个**每技能一个的动作执行器**，变成一个**跨工作空间的空间规划场**。

### 动机：为什么必须先动形态（探针结论，本次会话实测）

两个探针（`scripts/probe_morphogen.py` 容量探针、`scripts/probe_morph_necessity.py`
2×2 必要性探针）把"h3 NULL（形态素消融无影响）"这件事钉死：

- **容量探针**：aware 专家的形态素**确实写了负载**——per-skill 线性探针
  R²(m/μ) = grasp **0.850** / lift **0.883** / transport **0.797** / place **0.879**；
  R²(mu)=0.530、R²(margin)=0.592、R²(transport mm)=0.910。state(pose) R²≈0.03（姿势
  不含负载，sanity 通过），permuted-label 零基线 R²=0.034，blind 对照全 ~0/负。
  → **形态（form）装得下负载，网络自发地把它写进了形态素。**
- **必要性探针（ctx×morph 2×2）**：四种配置**全部 48/48、0 掉件**，
  transport 步长 ctxON/morphON 30.9mm、ctxON/morphOFF 23.3、ctxOFF/morphON 29.0、
  ctxOFF/morphOFF 25.0mm。
  → **任务对"使用记忆"施加零压力**：一个保守先验（~25mm 步）在没有任何负载信息时
  也能全部解决。所以 h3 NULL 是**任务驱动**的，不是形态缺陷。

**结论**：现行 7 细胞形态是 **sufficiency（够用），不是 necessity（必须）**——
它区别于盲基线的行为（margin-自适应收缩）在当前任务里是**装饰性的**。而且 7 个细胞
是 **异质语义 DOF**（px/py/pz/rx/ry/rz/g，角色由"它是哪个自由度"硬编码），**不是
空间同质单元**——所以"涌现分化 / 自组织 / 规模无关"的叙事**根本没有被支撑**：
角色是预先指定的，不是涌现的。这正是"形态本身不足"的根因。

### 突破口：把细胞重定义为同质空间单元 → NCA = 空间规划场

**核心主张**：当 NCA 的细胞 = **空间位置**（网格/点云上的一个格子）而非语义自由度时，
一个**同质局部规则**获得三个**可证伪**的归纳偏置，使得 0.1M 级网络能对**任意规模**
的工作空间输出规划场：

1. **局部性**：每格只看邻居（参数与问题维度无关）；
2. **权重共享 ⇒ 平移/尺寸等变**：同一规则作用在任意 G×G；
3. **迭代传播 ⇒ 有界参数、无界感受野**：K 步局部传播 = K 跳可达 → **定尺寸网络做
   长程规划**（这正是"规划"这一角色在 VLA 里需要的）。

在 VLA 中的定位：这不再是"每技能一个动作执行器"，而是**规划器**——从当前场景
产生一张空间计划场（下一步往哪走 / 先抓哪个），场的 argmax/梯度读出 = 下一个目标，
可直接喂给（或取代）现在冗余的 MoE 路由器的"下一步做什么"。

### Tier 1 实验设计：杂乱环境避障到达（cluttered reach planning）

- **任务**：G×G 网格（起始 G=32 或 48），随机障碍块，一个目标区。真值 = 到目标的
  **测地距离场**（网格 BFS/Dijkstra，归一化）。
- **监督**：经典求解器（BFS/Dijkstra）→ 纯回归，**无 RL**。
- **形态**：
  - 细胞 = 网格位置 (i,j)；
  - 状态 s_i = [alpha(1) | plan(1) | morphogen(8)]；
  - 外部输入 e_i = [goal_i, obstacle_i, start_i]；
  - 同质规则 F(s_i, {s_j : j∈N₈(i)}, e_i)，**所有格共用同一规则**；
  - 迭代 K 步传播；读出 = plan 通道 = 预测测地场；
  - 动作 = 从 agent 格在场上做梯度下降（或直接读 argmin 路径）。
- **代码布局**：`soma/spatial_nca.py`（bodygraph_nca.py 骨架，邻接从运动链图换成
  网格 8-邻接）+ `scripts/spatial_plan_data.py`（BFS/Dijkstra 真值生成）+
  `scripts/train_spatial_nca.py` + `scripts/eval_spatial_plan.py`。

**三条可证伪的泛化轴**（每条都是"定尺寸网络 vs 固定深度 CNN"的正面交锋）：

| 轴 | 训练 | 测试 | 考验的偏置 |
|---|---|---|---|
| 1 规模 | G=32 | G=64 | 无界感受野（迭代传播） vs 固定深度 |
| 2 复杂度 | 0–3 障碍 | 4–8 障碍 | 规则本身的泛化，非记忆 |
| 3 长程 | 直走廊 | U 形墙（欧氏近/测地远） | 迭代传播 vs 卷积感受野 |

**公平基线**（关键控制变量：**同局部性、固定深度 D、无迭代**）：
- **CNN（固定深度 D）**——THE 核心对照，用来隔离"迭代传播 / 无界感受野"这一优势：
  CNN 局部性相同但感受野被深度锁死；
- Transformer、MLP（大感受野/无局部性对照）；
- 参数量、数据量、训练步数全部对齐。

### Tier 1 结果（2026-10-14 实测，G=32 训练，4 个参数对齐模型 ~16K）

训练分布：family=mixed（50% 杂乱块 / 50% 单缝墙）、G=32、障碍 0–3。
四个模型参数量对齐（NCA 15,929 / CNN 16,321 / Transformer 17,249 / MLP 9,793），
数据量、步数、优化器完全一致，**只差归纳偏置**。NCA 用 random-K 训练
（每 batch 采样 K∈[48,96] 松弛步），迫使规则收敛到**不动点**（否则固定 K 学成积分器，
G=64 会随步数累积漂移）。

评估用**固定归一化** `norm_c=64`（目标 = 测地距离/64；不用 per-scene 最大值，因为后者
依赖一个网络看不见的全局量，导致跨 G 不迁移）。三条泛化轴 + 训练分布，每轴 128 场景。

| 轴 | 指标 | **NCA** | CNN (RF=9) | Transformer | MLP |
|---|---|---|---|---|---|
| in-dist G32 mixed | ρ / MSE / prog | **+1.00** / **0.0002** / **90.5%** | +0.36 / 0.0456 / -0.0% | +0.52 / 0.0304 / 21.1% | +0.06 / 0.0699 / 0.5% |
| scale G64 mixed | ρ / MSE / prog | **+0.95** / **0.0974** / **40.3%** | +0.25 / 0.4894 / 0.4% | +0.49 / 0.6239 / 9.6% | +0.03 / 0.4675 / 0.0% |
| complex G32 o4-8 | ρ / MSE / prog | **+1.00** / **0.0001** / **90.8%** | +0.23 / 0.0288 / 0.9% | +0.41 / 0.0487 / 11.8% | +0.05 / 0.0354 / 0.1% |
| longrange G32 wall | ρ / MSE / prog | **+1.00** / **0.0002** / **85.4%** | +0.56 / 0.0727 / 1.0% | +0.47 / 0.0470 / 6.7% | +0.07 / 0.1055 / 0.0% |

- **ρ（Spearman，pred vs 真值场，自由格）—主指标**：场的秩排序是否等于"到目标距离序"
  （即是否为合法导航函数）。秩基 → 对场的绝对尺度与噪声都鲁棒。
- **MSE**：场回归误差。
- **prog（planning progress）**：贪心下降动作读出，从场景起点沿预测场下降，走完的
  **最优（测地）路径比例**（1.0 = 到达目标）。

**结论：NCA 在四条轴上 ρ / MSE / prog 三项全部第一，且差距巨大。** 关键对照是 CNN：
局部性相同、参数量相同，但感受野被深度锁死在 **9 格**——在长程 U 形墙（欧氏近、测地
~2G）上 prog 仅 1.0%、ρ 0.56；在缩放 G64 上 MSE 0.49（比 NCA 高 5×）。Transformer
（全局 attention、无局部性/无跨迭代权重共享）同样不会迁移。MLP（点式、无传播）ρ≈0.06，
基本不可用。→ **"局部 + 迭代传播"（有界参数、无界感受野）正是空间规划任务需要的归纳偏置**，
假设成立。

**两个实测踩坑（记录，避免复现）**：
1. **二值"到达目标"成功是刀锋指标**：每步场的真实梯度 ≈ 1/路径长度（长路径 0.016），
   与场误差（RMSE ~0.014）同量级 → 一个浅的伪局部极小就把下降困在离目标几格处，
   **所有模型（含 ρ=1.00 的 NCA）二值成功率都 ≈0**。故改用**连续的 prog**（走完比例）
   作为 roll-out 指标——它衡量"下降是否沿最优路径推进"，鲁棒且任务相关。
   NCA 的 ρ≈0.999 但每场景仍有 2–4 个伪极小（goal 非全局最小），这是 1/路径 SNR≈1 的必然。
2. **per-scene 最大值归一化阻碍跨 G 迁移**：G64 中段被系统性高估（bias +0.27…+0.35）。
   改为固定 `norm_c`（dist/64）后目标跨 G 尺度一致，G64 prog 从 0% → 40%。

**待办**：（a）本设计归档（本节，完成）；（b）Tier 1 三条泛化轴对比完成——**NCA 全胜，
满足"必须先超过其他小模型"的门槛**；（c）结果回填 EXPERIMENT_SUMMARY.md；
（d）进入泛化强化（规模轴 + 能力轴，见下节）。

---

### 泛化强化：规模轴 + 能力轴（2026-10-15）

方向调整（用户）：**不做单一任务**——把主线放在**多任务泛化**与**从 NCA 规模入手**，逐步建立
VLA 对物理世界的理解；单一抓放任务对当前架构无信息量（这正是 h3 NULL 的根因：任务对该能力
零压力）。据此两条轴并行推进（本地两张 4090，`scripts/sweep_spatial_scale.py` /
`scripts/probe_multigoal.py`）。

#### 轴 A — 容量扫描（自变量=可训练参数，训练 mixed G32）

| cfg | model | params | in-dist | scale G64 | complex o4-8 | longrange wall |
|---|---|---|---|---|---|---|
| nca_xs | nca | 2,213 | .93/.010/58% | .69/.289/14% | .98/.005/96% | .86/.016/3% |
| nca_s | nca | 4,953 | .92/.017/44% | .68/.388/30% | .94/.003/74% | .84/.037/14% |
| **nca_m** | nca | **15,929** | **1.00/.000/90%** | **.94/.129/31%** | **1.00/.000/87%** | **1.00/.000/90%** |
| nca_l | nca | 58,257 | 1.00/.000/51% | .96/.036/13% | 1.00/.000/90% | 1.00/.000/1% |
| cnn_s | cnn | 1,657 | .22 | .16 | .12 | .36 |
| cnn_m | cnn | 16,321 | .37 | .24 | .29 | .56 |
| cnn_l | cnn | 37,153 | .70 | .42 | .62 | .83 |
| cnn_xl | cnn | 78,817 | .96 | .63 | .93 | 1.00 |
| cnn_xxl | cnn | **162,145** | .97 | **.63** | .95 | 1.00 |

（每格 ρ/MSE/prog；CNN 格只列 ρ，全表见 `checkpoints_spatial/scale_sweep.json`。）

1. **NCA 在 ~16K 参数处过拐点**：nca_m 在 in-dist/complex/longrange 即达 ρ≈1.00；G64 随规模
   0.94→0.96。更小（2K/5K）欠拟合（ρ 不稳、prog 掉）。→ 该归纳偏置**参数效率极高**。
2. **CNN 的 ρ 确实随规模上升**——cnn_xl（79K, RF=33）在 G32 三轴逼近 NCA。所以"CNN 完全不行"
   是错的：给 5–10× 参数它能拟合**训练分布**的场。
3. **但 (a) CNN 的规划始终差**：即使 cnn_xxl（162K，10× nca_m）prog 也只有 22%/4%/42%/1%，
   对 nca_m 的 90%/31%/87%/90%。(b) **G64 上 CNN 永远追不上**：cnn_xl/xxl ρ 卡在 0.63、prog 2–4%。
4. **决定性控制**：cnn_xxl 深度 32 → **感受野 65 ≥ G=64**，即单次前向就能"看到"整张 G64 网格，
   仍然 ρ=0.63 / prog=4%；而同规模 1/10 的 nca_m 是 0.94 / 31%。→ **长程/规模优势来自"迭代共享
   传播"这一归纳偏置本身，不是感受野、也不是容量**。这是对"形态本身有意义"最干净的一次证伪。

诚实说明：prog 在近完美场（ρ≈1）区间**方差大**（如 nca_l 在 longrange ρ=1.00 但 prog=1%——
单个伪极小就卡死在 2G 长路径上），故规模轴以 **ρ/MSE 为主信号**，prog 作粗校验。

#### 轴 B — 多目标能力探针（把"下一个抓哪个"场化）

场景含 k 个目标，真值 = 到**最近目标**的测地场（多源 Dijkstra）。训练 k∈{1,2,3}，测试
k∈{1,3,4,6,8}——**k=4,6,8 是目标数外推**（更多目标，无额外 head、无重训）。

| model | params | k=1 | k=3 | k=4 (OOD) | k=6 (OOD) | k=8 (OOD) |
|---|---|---|---|---|---|---|
| **nca** | 15,929 | **1.00/100%** | **1.00/100%** | **1.00/100%** | **1.00/99%** | **1.00/100%** |
| cnn | 16,321 | .43/3% | .63/9% | .71/8% | .81/31% | .85/55% |
| transformer | 17,249 | .64/24% | .41/35% | .36/39% | .27/40% | .28/21% |
| mlp | 9,793 | .05/−1% | .09/1% | .11/−0% | .13/4% | .15/8% |

NCA 在**全部**目标数上 ρ=1.00、prog≈100%，**包括从未见过的 k=4,6,8**——一个共享迭代规则直接
推广到更多目标（CNN 随 k 略升是因为"就近"更容易，但仍远低于 NCA；k=1 的 100% 是因为此轴只用
family=random，未混 wall）。→ **"场化选哪个"这个能力，NCA 形态天然支持，且目标数可外推。**

**待办**：规模轴 + 能力轴结果回填 EXPERIMENT_SUMMARY.md；继续扩展能力套件（动态/预测场、物理量场）。

#### 轴 C — 物理量场探针：调和（Laplace）势场

场景 = 同款杂乱网格；真值 = ∇²φ=0 稳态（Dirichlet：目标 φ=0、障碍 φ=1；静电势 / 稳态扩散）。
求解它**就是**迭代松弛——NCA 的原生运算。真值用精确稀疏求解得到（与 2 万次松弛校验
max|Δ|=8e-9），训练 `--target harmonic --family mixed`，四轴：同分布 G32 / 高密度 o4-8 /
规模 G64 / 长程 wall。四模型参数匹配（~16K）。

| model | params | in-dist G32 | complex o4-8 | scale G64 | longrange wall |
|---|---|---|---|---|---|
| **nca** | 15,929 | **.61/.007** | **.71/.006** | .39/.082 | **.43/.004** |
| cnn | 16,321 | .32/.110 | .43/.032 | .16/.070 | .21/.044 |
| transformer | 17,249 | .46/.009 | .51/.013 | **.45/.014** | .40/.004 |
| mlp | 9,793 | .06/.129 | .05/.040 | .03/.071 | .08/.051 |

（ρ/MSE。）1. **NCA 对 CNN 四轴全胜**（ρ 与 MSE 都赢）。2. **例外**：G64 规模轴上
transformer 的 ρ/MSE（.45/.014）**超过** nca_m（.39/.082）——全局注意力在大尺度上这次确实
迁移得更好。3. **prog 在此目标上无效**：调和势梯度浅，贪心下降全体卡住（≈0–3%），故本轴只认 ρ。

#### 轴 D — 动态/预测场探针：移动障碍的 time-to-contact 场

场景 = k 个整数速度、撞墙反弹的方块；ext=[obstacle_now, vx, vy]；真值 = 任意方块**首次到达**
每个格子的时间（归一化，1.0=horizon=64 步内永不）。训练 k∈{1,2,3}、G32，测试四轴（含外推）。

| model | params | k=1 G32 | k=3 G32 | k=5 G32* | k=7 G32* | k=2 G64* |
|---|---|---|---|---|---|---|
| nca | 15,929 | .14/.052 | .18/.106 | .20/.142 | .22/.168 | .09/.032 |
| **cnn** | 16,321 | **.17/.044** | **.24/.090** | **.30/.127** | **.35/.158** | **.11/.033** |
| transformer | 17,249 | .10/.048 | .10/.104 | .15/.124 | .16/.130 | .12/.027 |
| mlp | 9,793 | nan/.055 | nan/.113 | nan/.161 | nan/.203 | nan/.039 |

（ρ/MSE；`*`=训练范围外。）**结论是负面的、也是诚实的**：CNN 在**每一轴**都胜 NCA，且全体绝对
水平都弱（ρ≤.35，MLP 塌成常数）。**原假设（"固定深度 CNN 没有时钟所以该失败"）被证伪**。
原因：首次到达时间由**最近的那个接近方块**决定——是**局部**的多源距离变换，而非单源全局波前，
所以"无界迭代传播"这个归纳偏置在这里**不是**瓶颈。→ 提炼出的可证伪陈述：NCA 的优势出现在
**"稀疏源 + 全局约束场 + 需无界深度传播"**（测地/多目标/调和）的场合，**不**出现在局部可解
的场（time-to-contact）。要公平检验"动态"，需让场真正长程（稀疏慢速方块 + 长 horizon，或单方块）。

#### 轴 E — 迭代不动点审计 + Route A 修复（2026-10-16）

起因：调和场 G64 上 Transformer（ρ .461）反超 NCA（.458），怀疑 NCA 的"迭代到不动点"没真成立。
**k 扫描（固定权重，只改推理迭代数 k_eval）证实了更严重的问题**：

| k_eval | 48 | 96 | 160 | 192 | 256 | 384 |
|---|---|---|---|---|---|---|
| 测地 G32 ρ | .966 | **.998** | .996 | .934 | .817(崩) |
| 调和 G32 ρ | .507 | **.664** | — | .311 | −.080 | −.597(崩) |
| 测地 G64 ρ | .867 | .976 | .975 | .973 | .939 |
| 调和 G64 ρ | .424 | .458 | .498 | −.006(崩) | −.479 | −.516 |

→ **"迭代到不动点"实为"稳定窗口"**：峰值后 ~1.5–4× 训练 k 即崩，最终 MSE→1.5。
根因（`soma/spatial_nca.py`）：更新 `plan += dt·dplan`（dt=0.2），零初始化只保起点恒等，
**没有任何训练信号逼 `dplan→0`**——规则学的是"一直往场里推"而非"到位就停"。旧配方 random-K∈[48,96]
窗口仅 2×，太窄。**这是训练配方的 bug，不是架构极限。**

**Route A 修复（三组配方对比）**：

| 配方 | 测地 G32 峰ρ | 测地 G64 | 调和 G32 | 调和 G64 | 稳定 |
|---|---|---|---|---|---|
| 旧（窄 K∈[48,96]） | .998 | .976 | .664 | .458 | ✗ 崩 |
| 宽 K∈[48,384] | .989 | .978 | .47 | .41 | ✓ 宽平台 |
| 固定 K=256 + 残差 λ=1 | — | — | .575@256 | .487@256 | ✗ 只在 256 |
| **宽 K + 残差** | **.999**(至 768) | .998@256 | **.657@384** | **.474@384** | 半 ✓ |

残差损失实现：`SpatialNCA.forward(..., return_delta=True)` 返回末步更新量 `delta`，
训练加 `--fp-lambda·mean(delta²)`（`scripts/train_spatial_nca.py`，非 NCA 模型向后兼容）。

1. **测地场：Route A 纯赢**——宽 K 使迭代成为真收缩映射，G64 ρ .978、k 拉到 512 不塌，**零精度代价**。
2. **调和场：治好不稳定，但没拿到"独一无二"**——精度恢复（G32 .657≈旧峰值 .664），G64 提到 .474
   但**仅在调好的 k=384**（k=256 时只有 .388），且 **MSE 全面输** Transformer（.023–.041 vs .0141）。
   → 与 ttc 并列，**归为"边界"**：NCA 无干净唯一优势的场。
3. **副作用**：残差提升排序精度，却让场值在极端 k/G 下爆掉（测地 G64 k≥512：MSE→28，ρ 仍 .99 = 纯标度漂移）。
4. **净收益**：把"迭代到不动点"从**偶然为真**变成**可训练为真**——这本身是对架构主张的一次加固。
   **采纳宽 random-K 为默认配方。**

### Tier 2（后续）：多物体顺序抓放

场编码"下一个该抓哪个目标"；训练 N=1–3 个物体、测试 N=4–6 → 变物体数的顺序规划。
这直接把空间规划场接回 VLA 闭环：场的 argmax = 下一个 pick 目标。

#### 轴 F — 规划场闭环的归因审计（2026-10-17）：增益真实，但不是 NCA 的

多物体 N=3、双墙、visit 级 SmolVLA（uf, valMSE .0056）+ 固定执行器，n=200、seed 1234、
各臂评估**同一批场景**（场景 RNG 与执行器 RNG 分离）。出口 readout 统一为
`field_path_escape`（严格下降 + 半径 24 内 BFS 逃逸），或经典 Dijkstra 路由。

| 臂 | 全成功 | vs 直线 | McNemar |
|---|---|---|---|
| 直线执行 | 146/200 (73%) | — | — |
| NCA-long 场（2800 步，valMSE .0019） | 178/200 (89%) | +34/−2 | p=1.9e-8 |
| Pointwise MLP 场（wall-blind, valMSE .0043） | 180/200 (90%) | +36/−2 | p=5.4e-9 |
| **Euclid 场（无参数、无学习，直接 ‖goal−cell‖）** | **179/200 (90%)** | +34/−1 | p=2.1e-9 |
| CNN-long 场（valMSE .0019） | 143/200 (72%) | +6/−9 | p=.61 |
| GT 测地场（同一 readout） | 200/200 (100%) | +54/−0 | p=1.1e-16 |
| 经典 Dijkstra 路由（无场） | 200/200 (100%) | +54/−0 | p=1.1e-16 |

**结论（负面且明确）**：73%→90% 的提升真实且显著，但**归因于 readout（单调势场 + 局部逃逸搜索）**，
**不归因于 NCA**。一个零参数欧氏势场与 NCA 打平；wall-blind MLP 亦打平。场本身的贡献只在
"质量达到 GT"时体现（GT 场 100%），而学习场离 GT 有明显距离。

OOD 墙布局（两面垂直带缝墙、缝隙随机不锚定起点，40 例，离线）：
MLP greedy 34/40 escape 37/40；NCA 23/40, 29/40；CNN-short 19/22；CNN-long 12/16；zero 0/0。
**wall-blind 的欧氏势场在 OOD 上反而最好**——逃逸 readout 会替它完成全局绕墙。

方法论结论：**MSE 是误导指标**。CNN-long 场 MSE 与 NCA 同为 .0019，但局部排序违例率 20.6%
（NCA 1.8%、MLP 0.6%），闭环 143/200（零效应）。场的可用性由**局部排序**决定，必须用
闭环成功率或排序违例率评估，不能用目标回归误差。

**对研究问题的含义**：oracle occupancy 下，"NCA 规划改善 VLA"不成立；改善来自把直线执行换成
经典势场路由。若要让 NCA 的主张可证伪地成立，必须换到它不可被闭式势场替代的设定：
(1) 占用需从图像/部分观测推断（场同时承担建图与规划），或 (2) 多约束/多任务场
（如代价地形、可通行性、任务偏好），而不是单目标测地距离。

直接配对检验（同一批 200 场景）：NCA vs Euclid 3 胜/4 负（p=1.0），MLP vs Euclid 2 胜/1 负（p=1.0）
—— 学习场与零参数欧氏场**统计上不可区分**。GT 场 200/200 vs Euclid 179/200 说明场本身能再贡献
21 个成功，但前提是**全局正确**（局部排序违例≈0）；当前学习场（NCA 2.2%、MLP 1.7% 符号违例）
离这个门槛还有距离，且换更大的逃逸半径只是用经典搜索弥补场的误差。

**轴 F 事后审计（2026-10-18）——负结果的作用域与训练侧嫌疑**：
- 过拟合门：8 条固定 leg、27.5K NCA、2000 步，train MSE 卡在 .0016、符号违例 2.4%、strict 2/8；
  加大到 93.6K 参数（hidden 256/morph 16/msg 64）在 400 步时 held viol 仅 1.0%，但未见稳定过拟合。
  → **不是数据量问题；该架构/目标函数连 8 个场景都难以拟合到零违例。**
- 空间诊断：1515 个 GT 路径步中 33 个违例（2.2%）；**94% 的违例在距障碍 ≤1 格、100% ≤2 格**
  （正常步只有 24%/30% 在该区域）。→ 错误集中在墙边边界层，而非全局。
- 机制嫌疑：冻结障碍格 `plan=0` 且其消息进入邻域均值；mean 聚合倾向平滑测地场在墙边的
  尖锐 crease；MSE 不直接优化排序；无硬 Dirichlet 目标边界。
- 修复尝试：屏蔽障碍消息（`mask_obstacle_msgs=True`，已加入 `soma/spatial_nca.py`，默认关）
  1200 步 train viol 2.1%（基线 2.4%）；近墙 4× 损失权重在 800 步 held viol 0.0%、strict 6/8，
  但 2000 步回到 1.6%，不稳定。→ **尚无可靠修复，表示/优化仍是瓶颈。**
- 作用域：现有负面结论应表述为"**oracle occupancy + escape readout + 当前未调好的场**下，
  NCA 无可证实的特异优势"，**不能**推广为"NCA 不适用于 VLA"。要公平检验，需同时满足：
  (a) 墙边/留出违例压到 <0.2%，(b) 任务本身需要学到的场（经典规划不可用或 VLA 决策是瓶颈）。

---

## 方法 2 方案评估 — 把规划场注入 SmolVLA 动作专家（2026-10-18）

轴 F 的结论把方法 1（场路由执行器）判为"readout 增益、非 NCA 特异"。方法 2 不改执行器，
把规划场作为条件信号注入 SmolVLA 的 flow-matching action expert。本日完成成本/可行性
评估，完整报告见 `REPORT_FIELD_INJECTION.md`。

- **低成本路径是条件注入，不是改流匹配**：场是 denoiser 的额外 context（prefix token /
  KV adapter / FiLM），Euler 去噪与 ODE 完全不变。
- **SmolVLA 代码事实**：`modeling_smolvla.py:718` 每个 chunk 只填一次 prefix KV，
  `denoise_step:750` 复用 10 步；默认 `chunk=50`、`n_action_steps=50`、
  `train_expert_only=True`。新增 64 个场 token 的边际算力很小，且只需训练 ~0.1B expert
  与 0.1–0.3M 场编码器。
- **推荐注入**：小 CNN 编码 64×64 场 → 8×8=64 token 拼入 prefix；KV adapter 为信息上限
  消融，FiLM 为参数下限消融，路点 token 仅作退化作对照。
- **成本量级**（2×4090）：SmolVLA 3–5 天（工程 1–3 天 + 单卡训练）；π0/π0.5 1–2 周；
  GR00T N1.x 2–3 周；OpenVLA 自回归离散动作，不匹配。
- **必带对照**：同一编码器下 GT / NCA / Euclid / random / zero 场，加 field dropout 与
  推理时场交换；否则会重演轴 F 的归因问题（增益可能仍来自 Euclid 势场）。
- **风险**：模态坍缩、坐标系错位（工作空间场 vs 相机图）、墙边场质量（当前符号违例
  2.2%，94% 在距墙 ≤1 格）、oracle occupancy 在真机不可得。

**下一步**：E0 用 GT field token 验证注入通道上限（+ zero/random 对照），通过后再做
E1（NCA vs Euclid，参数/预算匹配）。E0 不通则先修通道，不读 NCA 结论。

**E0/E1 完整实验方案（2026-10-19）**：数据、动作标签、注入、训练、评估、预注册判据与
解释矩阵见 `REPORT_E0_E1_PROTOCOL.md`。核心约束：移除 field-routed 固定执行器（场只
进入一次）；所有臂共用 GT-field teacher 的动作标签；E0 通过标准为 GT 比 zero 高 ≥10 pp
且场交换显著降低成功率；E1 主对比 NCA vs Euclid。

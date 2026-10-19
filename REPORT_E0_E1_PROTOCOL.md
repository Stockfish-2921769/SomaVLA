# SomaVLA — E0 / E1 实验方案：场注入通道与来源归因

> 日期：2026-10-19
> 状态：方案定稿，待执行
> 关联：`REPORT_FIELD_INJECTION.md`（成本/可行性评估）、`EXPERIMENT_LOG.md` 轴 F、
> `EXPERIMENT_SUMMARY.md` R10
>
> **一句话**：E0 证明“场→action expert”这条通道可用；E1 证明增益是否必须来自 NCA。
> E0 不通过时 E1 的负面结果不可解释。

---

## 0. 摘要

- **E0（通道有效性）**：用 GT 测地场做上限探针，比较 `gt-plan` vs `zero-plan` /
  `random-plan` / `wrong-goal` / `no-field`；通过标准是成功率提升 ≥10 pp、配对显著、
  且推理时场交换确实降低成功率。E0 不涉及 NCA。
- **E1（来源归因）**：通道成立后，只替换场的来源——GT / NCA / Euclid / MLP / CNN /
  random / zero——其余全部固定。主对比是 **NCA vs 零参数 Euclid 场**。
- **核心设计**：方法 2 必须**移除原来 field-routed 固定执行器**，否则场进入两次；
  低层动作标签统一由 **GT 场路由的 teacher** 生成，所有实验臂共用同一套标签。
- **时间**：E0 约 3–4 天，E1 约 3–4 天，合计约一周（2×RTX 4090）。
- **预先写死判据**：E1 中 NCA 与 Euclid 统计不可区分 → 不能声称 NCA 特异；
  E0 中 GT 不优于 zero → 先修通道，不读 E1 结论。

---

## 1. 背景与设计原则

轴 F 的方法 1 闭环结果（n=200，配对）：

| 臂 | 全成功 |
|---|---|
| 直线执行 | 146/200 (73%) |
| NCA-long 场 | 178/200 (89%) |
| Pointwise MLP 场 | 180/200 (90%) |
| **Euclid 场（零参数）** | **179/200 (90%)** |
| CNN-long 场 | 143/200 (72%) |
| GT 测地场 | 200/200 |
| 经典 Dijkstra 路由 | 200/200 |

三条教训直接决定 E0/E1 的设计：

1. **增益可能来自“单调势场 + escape readout”，不是 NCA**。因此 E1 必须带零参数
   Euclid 对照。
2. **MSE 与闭环可用性脱节**。CNN-long 场 MSE 与 NCA 同级，但排序违例 20.6%、
   闭环只有 143/200。E0/E1 不能用 val MSE 选模型或下结论。
3. **学习场墙边质量不足**。当前 NCA 符号违例 2.2%，94% 在距障碍 ≤1 格。E1 必须
   先报告场的离线质量，避免把“场坏了”误读成“注入没用”。

### 1.1 三条不变量

1. **场只进入一次**：移除方法 1 的 field-routed 固定执行器，低层策略直接输出动作。
2. **只改一个变量**：E0 只改场条件；E1 只改 plan 通道来源。高层、低层结构、数据、
   动作标签、训练预算、评估场景全部固定。
3. **动作标签固定**：所有臂都拟合同一套 GT-field teacher 的动作，避免“场质量”
   与“动作标签质量”混淆。

### 1.2 可证伪性

- 若 E1 中 NCA ≈ Euclid，则“NCA 提供了不可替代的规划信息”这一主张被证伪；
  方法 2 仍可作为“场条件有效”的工程结果，但不能算 NCA 贡献。
- 若 E0 中 GT ≈ zero，则注入通道无效；此时任何 E1 结果都不能用于评价 NCA。

---

## 2. 共同实验基础设施

### 2.1 任务与场景

- 环境：多物体顺序抓放，N=3，两道垂直带缝墙，随机起点/目标；与轴 F 完全同分布。
- 主评估集：200 个固定场景（seed 777），与轴 F 同一批场景，保证与方法 1 可比。
- OOD 集：40 个随机墙布局（轴 F OOD harness），仅用于次要泛化报告。
- 场景级切分：train / val / test 按 scene seed 分开，禁止按样本切分。

### 2.2 高层策略（冻结）

- 使用双墙分布上最强的 visit-level SmolVLA：默认 `ckpts/visit_2w_base/best.pt`；
  执行前先用同一 200 场景复测它与 `ckpts/smolvla_multi_v2/best.pt`，取更强者。
- 每个 visit 在 hover 处查询一次，输出 `[gx,gy,gz,open]`；所有实验臂共用同一
  高层权重与查询节奏。
- 训练数据用 teacher 子目标；推理用高层预测子目标；训练时给子目标加 1–2 cm
  噪声，缓解 exposure bias。
- 次要诊断：GT 臂额外用 teacher 子目标跑 50 个场景，用于分离“高层误差”与
  “低层注入误差”。

### 2.3 低层 action expert

- 基座：`lerobot/smolvla_base`（SmolVLM2-500M + flow-matching expert）。
- config 变异：单相机、state 7、action 7、chunk 8；沿用现有 `train_smolvla.py`
  的配置路径，只把 action dim 从 4 改为 7。
- 输入：当前图像、归一化 state[7]、归一化子目标[4]、场 tokens。
- 输出：H=8 步的绝对 EEF target + 夹爪（7 维）；10 步 Euler 去噪。
- 执行：receding horizon，执行前 4 步后重新观测、重算场、重新预测。
- 训练：只训 expert + 场编码器；VLM 冻结（`train_expert_only=True`,
  `freeze_vision_encoder=True`）。

**动作语义**（必须严格遵守）：

```python
target_pose = [waypoint_x, waypoint_y, waypoint_z,
               state[3], state[4], state[5], grip]
state_next = state + 0.5 * (target_pose - state) + noise
```

动作标签是 plant 的 **target command**，不是 `state[t+1] - state[t]`。否则学习到的
策略不包含 plant 阻尼模型，闭环会系统性滞后。

### 2.4 场表示与注入

- 网格：64×64，世界帧；四通道 `[obstacle, plan, dx/G, dy/G]`。
- `obstacle`：与 GT 场同一膨胀规则的障碍占用；`dx,dy`：当前格到当前子目标的方向
  偏移；`plan`：唯一的实验变量。
- 编码器：4→32→64（stride2）→128（stride2）的三层 CNN，输出 8×8×128，
  展平为 64 tokens，线性投影到 VLM hidden；约 0.2M 参数。
- 注入：在 `SmolVLAFlowMatching.embed_prefix`（`modeling_smolvla.py:534`）末尾拼接
  `[image | language | field | state]`；
  - 64 个场 token 的 `pad_masks` 全 1；
  - prefix 的 `att_masks` 保持 0；
  - `position_ids` 由 `cumsum(pad_masks)-1` 自动顺延；
  - prefix KV 每 chunk 填一次（`sample_actions` :718），10 步去噪复用。
- 场 dropout：训练时 15% 概率把整张场置零（保留 token 数量），可加学习到的
  null token 作为替换。
- 冻结 VLM 下梯度仍应回传到场编码器；训练前必须做梯度检查。
- 首版使用世界帧场 + state 中的 EEF xy；若场交换显示方向错误，改为 ego-centric
  局部裁剪后重跑一个臂诊断。

### 2.5 低层数据生成

**Teacher**：方法 1 的“GT 测地场 + escape readout”控制器，在仿真中逐步执行并
记录 target command。它不是决策 oracle；GT 场在方法 1 中就是管线上限。

每条样本：

```text
image    [3,256,256] uint8
state    [7]         归一化 EEF pose
subgoal  [4]         归一化 [gx,gy,gz,open]（teacher）
action   [8,7]       归一化逐步 target command（GT-field teacher）
field    [4,64,64]   [obstacle, plan, dx/G, dy/G]（float16）
meta     scene seed / episode id / step / visit id
```

- 规模：600 episodes，每 episode 约 100–150 步；chunk H=8、stride 4（与
  receding 对齐）→ 约 15–20K chunk 样本。
- 存储：图像约 3–4 GB；场约 0.5–0.7 GB。
- 场计算：每个 chunk 一份，目标为当前 visit 子目标；不得使用教师未来轨迹、
  最终任务目标或其他未观测位置。
- 数据分布：确保“绕墙步”占比足够；建议 50% 场景强制 detour、25% 长程 U 形墙、
  25% 随机布局，否则场在数据中冗余，E0 会被稀释。

**回放门**：用记录的动作标签开环回放，末端误差 <1 cm；否则动作语义或归一化有错。

### 2.6 训练配方

| 项 | 设置 |
|---|---|
| 初始化 | `lerobot/smolvla_base`，config 变异（单相机 / state 7 / action 7 / chunk 8） |
| 冻结 | VLM + 视觉编码器 |
| 可训练 | ~0.1B expert + ~0.2M 场编码器 |
| 优化 | AdamW；expert lr 1e-4，encoder lr 3e-4；wd 1e-5；grad clip 1.0；bf16 |
| batch | 16（显存允许可 32） |
| steps | 所有臂固定 15K；pilot 先确定步数；**不按 val MSE 早停** |
| seeds | 2 个（0/1）；临界对比补第 3 个 |
| 场 dropout | 15% |
| checkpoint | 固定步数保存 last；若需选模，统一使用一小批闭环验证场景 |

**Pilot 三门（各 1–2 小时）**：

1. **梯度门**：50 步后确认场编码器 `weight.grad` 非零、显存 <24 GB。
2. **过拟合门**：64 个含绕墙步的 chunk 过拟合；GT/zero 动作误差均能下降，且
   数据子集确实包含需要绕行的动作。
3. **回放门**：动作标签开环回放末端误差 <1 cm。

### 2.7 评估协议

- 200 个固定场景；每个场景的初始状态、物体排列、墙布局、flow-matching 噪声种子
  在所有臂间完全一致，保证配对。
- 高层权重、查询节奏、最大步数（1500）完全一致。
- 成功定义：三个物体全部放到颜色匹配目标。
- 主指标：全任务成功率；次要：放置数、碰撞次数、完成步数、路径长度 / GT 路径长度。
- 诊断：
  1. **场交换**：训练完成后，推理时替换场（GT↔random / wrong-goal），记录成功率
     跌幅与首个 action chunk 的归一化 L2 偏差；跌幅 <5 pp 视为模型基本忽略场。
  2. **teacher 子目标 vs 高层预测子目标**：仅在 GT 臂跑 50 个场景，分离高层误差。
  3. **OOD**：40 个随机墙布局，E0 报告但不调参；E1 作为正式次要结果。
- 离线指标：动作 chunk MSE、场梯度与动作方向余弦（只诊断，不选模）。

### 2.8 公平性与防泄漏

- 同一场编码器、同一 token 数、同一训练预算、同一 seed 预算；`no-field` 少 0.2M
  参数，报告时注明。
- 场只由 occupancy + 当前子目标生成；不包含教师未来动作、最终目标、未观测物体。
- 场在每个 chunk 重算，但目标在 visit 内保持不变；跨 visit 由高层更新。
- **oracle occupancy 是仿真设定**：E0/E1 的结论作用域限定为“已知障碍布局”；
  真机/部分观测是 P1，不在本方案内。

---

## 3. E0：场注入通道有效性

### 3.1 目的与假设

**目的**：在完美场条件下，验证场→action expert 的条件通道能否产生闭环增益。

- H0：`gt-plan` 与 `zero-plan` / `random-plan` 无差异。
- H1：`gt-plan` 显著更优，且推理时交换场会改变动作、降低成功率。
- H2：增益不能由“多了 64 个可训练 token”解释；`random-plan` 应接近 `zero-plan`。

### 3.2 实验臂

| 臂 | 场 token | plan 通道 | 作用 |
|---|---|---|---|
| `no-field` | 无 | — | 控制新增 token/编码器本身 |
| `zero-plan` | 有 | 全零 | 控制额外条件容量；保留 obstacle + dx/dy |
| `random-plan` | 有 | 平滑随机势场 | 控制任意结构化信号 |
| `wrong-goal` | 有 | GT 场但目标换成随机空格 | 控制目标对齐是否必要 |
| **`gt-plan`** | 有 | 当前子目标的精确测地场 | **主臂/上限** |

- `random-plan` 用与 GT 场同功率谱的低通高斯随机场（保持平滑度与边缘分布），
  不是逐像素噪声。
- `wrong-goal` 选与真实子目标距离相近的随机空格，避免距离尺度混淆。

### 3.3 三个子门

1. **E0a 信息量探针（小时级）**：固定场，用小 MLP 从“EEF 周围局部场 patch +
   state + subgoal”回归 teacher 的下一步动作/waypoint，留出场景评估。若 MAE
   异常大或方向余弦接近 0，先查坐标、分辨率与归一化，不进闭环。
2. **E0b 闭环主实验**：5 臂 × 2 seeds × 200 episodes。主对比 `gt-plan` vs
   `zero-plan`。
3. **E0c 依赖度**：`gt-plan` 训练后做场交换（GT→random / wrong-goal）；记录动作
   偏差与成功率跌幅，识别模态坍缩。

### 3.4 运行矩阵

| 项目 | 规模 |
|---|---|
| 主矩阵 | 5 臂 × 2 seeds × 200 eps = 2000 episodes |
| 场交换 | `gt-plan` × 2 seeds × 200 eps（random + wrong-goal 各一遍） |
| teacher 子目标诊断 | `gt-plan` × 2 seeds × 50 eps |
| OOD（次要） | `gt-plan` / `zero-plan` / `random-plan` × 1 seed × 40 eps |

### 3.5 预注册判据

**主对比：`gt-plan` vs `zero-plan`。**

- **通过**：成功率提升 ≥10 pp；单侧 McNemar p<0.05；`random-plan − zero-plan`
  <5 pp；场交换成功率跌幅 ≥5 pp。
- **弱通过**：提升 5–10 pp 且显著；可进 E1，但结论标为“弱通道”。
- **不通过**：<5 pp 或 p≥0.05 → 不进 E1，先修通道。

统计：场景级配对；McNemar 精确检验；成功率差用场景级 bootstrap 95% CI；
次要比较 Holm 校正；seed 0 为预注册主分析，seed 1 为复现。

### 3.6 解释矩阵

| GT vs zero | random vs zero | 场交换跌幅 | 解释 | 动作 |
|---|---|---|---|---|
| ≫ | ≈ | 大 | 通道有效，模型在用场的空间内容 | 进 E1 |
| ≈ | ≈ | ≈0 | 模型忽略场/注入无效 | 查 KV、mask、坐标；试 KV adapter、解冻 |
| ≫ | ≫ | 大 | 增益来自额外条件容量而非场 | 以 `no-field` 为基准重估；加强控制 |
| ≫ | ≈ | 小 | 可能只用了 dx/dy 目标方向 | 做只留 dx/dy 的消融；查 plan 梯度 |
| `no-field` ≈ GT | — | — | 场与图像/state 冗余 | 换遮挡/长绕行场景或更难的执行任务 |
| GT(teacher 子目标) ≫ GT(预测子目标) | — | — | 高层误差主导 | 先修高层，不读注入结论 |

### 3.7 调试清单

1. 打印 `prefix_embs.shape`、`pad_masks.sum()`；确认 64 个场 token 在 suffix 前。
2. 确认场编码器 `requires_grad=True` 且梯度非零。
3. 场交换跌幅接近零时，先试 KV adapter 或解冻 VLM 最后 2–4 层，再判“场无用”。
4. 把预测动作方向与场负梯度画在一起；余弦接近 0 时改 ego-centric 场。
5. 统计 teacher 在绕墙步与直行步的动作差异；若数据集几乎全是直行，场自然没用。
6. 用 teacher 子目标做对照评估，分离高层与低层瓶颈。
7. 开环回放动作标签；末端误差 >1 cm 时先修动作语义/归一化。
8. 禁止用 val action MSE 选模型或下结论。

### 3.8 交付物

- `ckpts/e0/{arm}_s{seed}/last.pt`
- `logs/e0/{arm}_s{seed}.json`（逐 episode 结果）
- `reports/E0.md`（主表 + 场交换表 + go/no-go）
- 低层数据生成、场编码器、E0 评估脚本入库

---

## 4. E1：场来源归因

### 4.1 目的与假设

**目的**：E0 证明“场有条件价值”后，判断这份价值是否必须由学习到的 NCA 提供。

- H1：NCA 场显著优于零参数 Euclid 场。
- H0：NCA ≈ Euclid → 与方法 1 相同，增益来自“单调势场 + 局部方向”，不是 NCA。

### 4.2 实验臂

除 `plan` 通道外，其余全部固定：

| 臂 | 场来源 | 参数量/性质 |
|---|---|---|
| **GT** | 精确 8 连通 Dijkstra 测地场 | 上限 |
| **NCA** | 当前最强 NCA-long | 27.5K，学习所得 |
| **Euclid** | `hypot(dx/G, dy/G)` | **0 参数，墙盲** |
| MLP | pointwise MLP 场 | ~9.8K，墙盲 |
| CNN-long | 固定深度 CNN 场 | ~47K，局部 |
| random | 平滑随机场 | 下限 |
| zero | 全零 | 下限 |

- NCA checkpoint：方法 1 审计使用的 `plan_nca_long3r`（27,501 参数，hidden 128 /
  morph 12 / msg 48 / K=64）；`.pt` 不入库，按 `scripts/train_spatial_nca.py`
  复现。仓库内同时保留 `ckpts/multi_plan_nca.pt` 作为可加载备选，选强不选弱。
- MLP / CNN 场以方法 1 审计实际加载的 checkpoint 为准；缺失时按
  `soma/spatial_baselines.py` 重训，参数量与审计口径对齐。
- 所有臂共用同一场编码器、同一 token 数、同一低层 expert 初始化、同一数据/步数/
  seed 预算；Euclid 零参数不是缺陷，它正是“增益是否只需一个势场”的判别器。

### 4.3 控制变量

| 变量 | 设置 |
|---|---|
| 动作标签 | 全部来自 GT-field teacher，所有臂共用 |
| 高层 | 冻结，同一 checkpoint、同一查询节奏 |
| 低层结构 | 同一 SmolVLA expert、同一初始化、同一场编码器 |
| 数据 | 同一 600 episodes / 同一 chunk 划分 |
| 训练 | 同 15K 步、同 batch、同 lr、同 seed 预算 |
| 评估 | 同一 200 场景、同一噪声种子、同一最大步数 |
| 场分辨率 | 统一 64×64（32×32 作为可选扩展） |

### 4.4 场质量前置表

在同一 200 场景上，先离线报告每类场的：

- Spearman ρ、MSE；
- strict greedy 成功率、escape 成功率；
- GT-descent 路径上的符号违例率与 |dp|/|dg| 中位数；
- 墙边（距障碍 ≤1 格）违例占比。

用途：区分“注入通道问题”与“场质量问题”。当前 NCA 违例 2.2%、94% 在墙边；
若 E1 中 NCA 明显落后，先检查这张表。**E1 期间不调场**；改进场是独立实验。

### 4.5 预注册判据

**主对比：NCA vs Euclid。**

- **NCA 特异**：NCA − Euclid ≥5–10 pp，配对 McNemar p<0.05（Holm 校正），
  且在 2–3 个 seed 上方向一致；OOD 不反向。
- **非 NCA 特异**：NCA ≈ Euclid（差异不显著，CI 包含 0）→ 与方法 1 同结论。
- **Euclid 更优**：优先怀疑场质量；做离线排序诊断，不归因给架构。
- **GT ≫ 所有学习场**：场学习器是瓶颈；先改场（墙距输入、min-plus/Bellman 聚合、
  排序损失、硬目标边界），再谈注入。

辅助对比：

- NCA vs MLP / CNN：学习场之间的排序；
- NCA vs random：确认不是任意信号；
- NCA vs GT：剩余 headroom；
- GT vs zero（沿用 E0）：通道是否仍然有效。

### 4.6 解释矩阵

| 结果 | 结论 | 下一步 |
|---|---|---|
| NCA ≫ Euclid，且 swap 跌幅大 | NCA 提供了不可替代的规划信息 | E2（dropout/swap/OOD）+ 规模/感知轴 |
| NCA ≈ Euclid | 增益是场通用的，不是 NCA | 转 P1 感知条件场或 P2 决策瓶颈任务 |
| Euclid > NCA | 场质量或欧氏先验占优 | 离线诊断墙边违例；修场后重跑 E1 |
| GT ≫ 所有 | 场学习器是当前瓶颈 | 先做场学习器改进实验，不扩大 VLA |
| 所有臂 ≈ zero | 注入通道未建立（与 E0 矛盾） | 回 E0 查通道/训练 |

### 4.7 可选扩展（不属于 E1 主判定）

1. **E1-self-teacher**：每个臂用自己场的路由控制器生成动作标签。这测的是
   “场+策略”的端到端性能，包含场质量，不能用于架构归因；仅作工程结果。
2. **场分辨率**：64×64 vs 32×32；检查注入成本和信息损失。
3. **NCA 迭代数 K**：只影响场质量，不参与 E1 主对比；若使用，固定 K 并报告。
4. **注入方式**：prefix token（主）vs KV adapter vs FiLM；只在 E0 未通过时作为
   通道诊断，不与 E1 混合。

### 4.8 交付物

- `ckpts/e1/{field}_s{seed}/last.pt`
- `logs/e1/{field}_s{seed}.json`
- `reports/E1.md`（场质量表 + 闭环主表 + 配对统计 + 结论）
- 归因结论一句话：**增益是否必须来自 NCA；若不是，作用域是什么。**

---

## 5. 统计与报告规范

1. **配对比率**：主分析用场景级配对 McNemar（每场景每臂同初始条件/同噪声种子）。
2. **效应量**：报告成功率差、95% bootstrap CI；不要只报 p 值。
3. **多重比较**：E1 的 NCA vs Euclid 为主对比，其余用 Holm 校正。
4. **seed**：seed 0 为预注册主分析，seed 1 为复现；第 3 个 seed 仅用于临界结果。
5. **报告所有臂**：包括表现差的臂；不得只报最好的。
6. **离线与闭环并列**：每份结果同时给场质量表与闭环表，便于区分瓶颈。
7. **预先写死判据**：不得在看到结果后改阈值或更换主对比。

---

## 6. 风险登记与应急预案

| 风险 | 触发信号 | 应急 |
|---|---|---|
| 模型忽略场 | 场交换跌幅 <5 pp | KV adapter；解冻 VLM 最后 2–4 层；提高 field dropout 之外的对比监督 |
| 场与图像冗余 | `no-field` ≈ `gt-plan` | 换遮挡/长绕行场景；增加需要全局信息的任务 |
| 高层子目标误差主导 | teacher 子目标明显更好 | 先修高层或固定 teacher 子目标做 E0 |
| 动作语义错误 | 开环回放末端误差 >1 cm | 修 target command 记录/归一化 |
| 低层欠拟合 | pilot 训练动作误差不降 | 降 lr、加数据、检查 chunk 对齐 |
| NCA 场质量差 | 离线违例率远高于 GT | 先做场改进；E1 结论标注“受场质量限制” |
| 算力超支 | 训练排队 | 按固定顺序削减：CNN-long → MLP → 减少 OOD；GT/NCA/Euclid/zero/random 必须保留 |
| OOD 反向 | 同分布通过、OOD 失败 | 如实报告；不回调参后声称泛化 |

---

## 7. 时间线（2×RTX 4090）

| 阶段 | 内容 | 估计 |
|---|---|---|
| D1 | 低层数据生成（600 eps，含场与动作标签） | 1–2 h |
| D1 | Pilot 三门（梯度/过拟合/回放） | 2–4 h |
| D2–D3 | E0：5 臂 × 2 seeds 训练 + 200 eps 评估 | 2–3 天 |
| D4 | E0 汇总 + go/no-go | 0.5 天 |
| D5–D7 | E1：7 臂 × 2–3 seeds 训练 + 评估 | 2–3 天 |
| D8 | E1 汇总、场质量表、归因 memo | 0.5–1 天 |

总计约 **7–9 天**；若不跑 CNN/MLP 臂，可压缩到 5–6 天。

---

## 8. 代码改动清单

| 文件 | 内容 |
|---|---|
| `scripts/multi_lowlevel_data.py`（新） | GT-field teacher 回放，记录 `(image,state,subgoal,action,field)` |
| `soma/field_encoder.py`（新） | 4→64 token 的小 CNN 编码器 |
| `scripts/train_smolvla.py`（扩展） | `--field-data`、`--field-arm`、`--field-dropout`、action dim 7 |
| `scripts/eval_multi.py`（扩展） | `--field-arm`、`--field-swap`、低层 action expert 评估模式 |
| `scripts/audit/field_quality_table.py`（新） | E1 前置的离线场质量表 |
| `scripts/audit/field_swap.py`（新） | 推理时场交换与动作偏差统计 |
| `scripts/audit/paired_stats.py`（新） | 场景级 McNemar + bootstrap CI |

工程注意：

- 场 token 的 `pad_masks`/`att_masks`/`position_ids` 必须与 SmolVLA 约定一致；
- 冻结 VLM 时验证场编码器梯度；
- 所有实验臂写同一份 manifest（seed、数据 hash、步数、场源、checkpoint），
  防止不同臂训练条件漂移。

---

## 附录 A：仓库事实与代码锚点

- SmolVLA flow matching：
  - `modeling_smolvla.py:534` `embed_prefix`
  - `modeling_smolvla.py:718` `sample_actions`（每 chunk 填一次 KV）
  - `modeling_smolvla.py:750` `denoise_step`（10 步复用 KV）
  - `configuration_smolvla.py:31-91`：`chunk_size`、`n_action_steps`、`num_steps`、
    `attention_mode`、`train_expert_only`、`freeze_vision_encoder`
- 当前多任务基线：`scripts/multi_vla_data.py`（visit-level subgoal chunk [8,4]，
  state 7，图像 256×256）；`scripts/eval_multi.py`（`SubgoalController` /
  `VisitController` / `NCAWaypoints`）；`scripts/multi_task.py:430` `teacher_rollout`。
- 高层 checkpoint 候选：`ckpts/visit_2w_base/best.pt`、`ckpts/smolvla_multi_v2/best.pt`。
- 场 checkpoint 候选：`ckpts/multi_plan_nca.pt`、`ckpts/multi_plan_cnn.pt`；
  方法 1 审计的 NCA-long 需按 `scripts/train_spatial_nca.py` 复现。
- 轴 F 审计脚本：`scripts/audit/`；场质量与闭环口径见 `scripts/audit/offline_table.py`
  使用的 `field_path` / `field_path_escape`。

## 附录 B：场定义

| 场 | 定义 |
|---|---|
| GT | 8 连通 Dijkstra 距离 / 128；障碍膨胀与训练一致 |
| Euclid | `hypot(dx/G, dy/G)`；墙盲、零参数 |
| NCA | 空间 NCA 的 plan 通道输出；K=64，27.5K 参数 |
| MLP | pointwise 3→96→96→1，逐格输入 `[obstacle, dx/G, dy/G]` |
| CNN-long | 6×3×3 conv/32ch，固定深度 |
| random | 与 GT 同功率谱的低通高斯随机场，重标定到 plan 值域 |
| zero | 全零 |

## 附录 C：预注册检查单

- [ ] E0 主对比、通过阈值、场交换阈值已写死（≥10 pp / p<0.05 / <5 pp / ≥5 pp）
- [ ] E1 主对比固定为 NCA vs Euclid，阈值 ≥5–10 pp
- [ ] 所有臂共用同一动作标签、高层、编码器、数据、训练预算
- [ ] 场只进入一次（固定执行器已移除）
- [ ] 场算到当前子目标，不用最终目标/未来轨迹
- [ ] 不用 val MSE 选模
- [ ] 场质量前置表在 E1 前完成
- [ ] 结果报告包含所有臂、效应量、CI、seed 复现

# VLA-lite 任务书：重建一个可归因的轻量 VLA

> 日期：2026-10-20
> 状态：设计定稿，待执行
> 关联：`reports/E0.md`（注入闭环的负面结果）、`REPORT_E0_E1_PROTOCOL.md`（E0/E1 原始方案）、
> `EXPERIMENT_SUMMARY.md` R11、`EXPERIMENT_LOG.md` E0
>
> **一句话**：E0 说明“往冻结的 SmolVLA 里塞场 token”既没有证明 NCA 的贡献，也把
> 通道本身做坏了；VLA-lite 改为从零搭一个**无 LLM、场在环路内、读取端可训练**的
> 小 VLA，用同一套场景/数据/评估把“NCA 结构是否必要”变成可证伪的对照实验。

---

## 0. 摘要

### 0.1 要回答的问题

> 在多物体顺序抓放、需要绕墙规划的任务上，**NCA 的迭代式局部规则**相比
> 前馈 CNN、零参数 Euclid 势场和“无场”条件，是否能给 VLA 的闭环成功率带来
> 可复现、可归因、在分布外仍成立的增益？

这不是“NCA 能不能拟合测地场”的问题（轴 F 已证明能），而是：

1. **通道问题**：场进入策略后，策略是否真的会用它？（E0 没通过）
2. **来源问题**：增益是否必须来自 NCA 的结构，而不是任意单调势场？
3. **规模问题**：NCA 的固定参数、网格无关规则，是否在 G32→G64、障碍变多时
   比同参数、同感受野的 CNN 更稳？

### 0.2 核心变化（相对 E0）

| 维度 | E0（SmolVLA 注入） | VLA-lite（本任务书） |
|---|---|---|
| 视觉/语言主干 | SmolVLM2-500M，冻结（`train_expert_only`） | SigLIP-B/16 视觉塔冻结；无 LLM，语言用可学习任务槽 |
| 场的来源 | 外部 4×64×64 通道，训练时 GT / 评估时替换 | NCA 在策略内部生成，端到端可微；另设 GT/Euclid/CNN 对照 |
| 场的读取 | 64 个 field token 插入 prefix，冻结 VLM 不更新 | 专用空间适配器 + 动作解码器，读取端全部可训练 |
| 场进入哪一层 | 只进低层 action expert | 可进低层、也可进高层（`NCA-low` / `NCA-both`） |
| 位置编码 | prefix 113→177 token，RoPE 偏移 +57% | 无 LLM prefix；网格 token 走独立 readout，不改 RoPE |
| 可训练参数 | ~100M expert + 0.218M 场编码器 | ~7.4M（视觉塔冻结） |
| 归因 | 场来源与注入实现混在一起 | 场生成器是唯一变量，读取端/数据/预算/评估全部固定 |

### 0.3 预注册判据（详见 §2.3）

- **G0 基础设施**：评估确定性、障碍 IoU ≥ 0.90、GT 场臂 ≥ no-field +10 pp。
- **G1 NCA 通道**：NCA 臂 ≥ no-field +10 pp（配对显著），且推理时置零/交换场
  至少掉 5 pp。
- **G2 NCA 特异性**：分布内 NCA 不差于 CNN/Euclid 5 pp 以上；G32→G64 尺度外推
  NCA 掉点 ≤5 pp 且比 CNN 高 ≥10 pp；障碍数 4–8 的 OOD 上 NCA 比 CNN/Euclid
  高 ≥10 pp（5–10 pp 只算弱证据）。
- **G3 端到端**：NCA-both ≥ VLA-lite no-field +10 pp，且不差于 E0 SmolVLA
  no-field 5 pp 以上；G1/G2 未过则 G3 只能作为工程结果，不能主张 NCA 贡献。

### 0.4 预算

- 2×RTX 4090；Stage A 单次 20K step、batch 64，实测约 8–10 min；全矩阵 2–3 天。
- 数据再生成 1–2 天（CPU），SigLIP 特征预计算约 38 GB 磁盘（低层 9.8 GB +
  高层 28.3 GB）、数十分钟。
- 总周期 3–4 周（含调试与缓冲），里程碑见 §9。

---

## 1. 背景与问题定义

### 1.1 E0 的结论（必须作为设计输入）

E0 在 SmolVLA-0.5B 上做了 5 臂 × 2 seed 的闭环对照（n=200 固定场景，配对）：

| 臂 | s0 | s1 | 合并 |
|---|---:|---:|---:|
| no-field | 87.0% | 88.0% | 87.5% |
| gt-plan | 70.0% | 79.0% | 74.5% |
| zero-plan | 71.0% | 76.5% | 73.75% |
| random-plan | 62.0% | 82.0% | 72.0% |
| wrong-goal | 71.5% | 81.5% | 76.5% |

关键读数：

1. **主对比失败**：gt-plan vs zero-plan 合并 +0.8 pp，95% CI [−5.0, +6.2]，
   单侧 McNemar p=0.4345。
2. **通道本身有害**：no-field 比 zero-plan 合并高 **+13.8 pp**（+96/−41，
   p<0.0001）；比 gt-plan 同 seed 高 17.0/9.0 pp。只多了 217.8K 场编码器和
   64 个 token，却稳定掉 9–17 pp。
3. **教师诊断不是低层无能**：teacher subgoal 下 gt-plan 臂 100%（382/382、
   379/379 visits）；换成预测高层后只有 74%/86%。低层会执行，但高层给的目标
   已经错了。
4. **交换/确定性噪声**：swap 跌幅均 <5 pp，说明策略对场内容不敏感；
   det-noise 再掉 4.8 pp，说明部分差异来自 flow 采样噪声而非场。
5. **OOD 方向反转**：s1 上 gt−zero = +7.5 pp，s0 上 −15.0 pp，不能作为支持证据。

结论：E0 不通过；E1（NCA vs Euclid 来源归因）暂缓。继续在冻结 SmolVLA 上做
“换场来源”的实验，只会重复测同一个注入惩罚。

### 1.2 为什么只改动作头不够

E0 的实现有三个结构性限制，任何一个都足以让“场有没有用”不可识别：

1. **读取端冻结**：`train_expert_only=True` 冻结了整个 VLM，只有 action expert
   和 217.8K 场编码器可训练。场 token 与图像/语言 token 处在同一个双向块
   （`att_mask=0`，按 `make_att_2d_masks` 的 cumsum 语义），冻结的 VLM 没有
   动力也没有能力学会“这是一个规划通道”。
2. **注入方式有损**：field token 插在 state 之前，prefix 从约 113 token 增到 177
   （+57%），state 位置和 suffix 的 RoPE offset 全部后移；`pad_mask=1,
   att_mask=0` 让 64 个新 token 参与双向注意力，等于给预训练分布加了一个
   大扰动。no-field 的 +13.8 pp 就是这笔扰动税。
3. **高层是瓶颈**：E0 只把场喂给低层，高层仍是冻结的 SmolVLA visit policy。
   teacher subgoal 100% vs 预测高层 74%/86% 的差距说明：即使低层完美读取场，
   高层选错对象/目标也会让闭环失败。NCA 的信息必须能同时到达决策层。

### 1.3 为什么重建、为什么去掉 LLM

重建不是目的，**可归因**才是目的。VLA-lite 保留 V/L/A 三要素，但去掉自回归
LLM：

- **V**：冻结的 SigLIP-B/16 视觉塔（约 93M），提供通用视觉特征。
- **L**：可学习任务嵌入 + 可训练投影（语言槽），不是生成式 LLM；冻结句向量
  接口预留，待离线权重可用时接入。
- **A**：空间适配器 + NCA 规划核 + 动作解码器 + 高层决策头，全部可训练。

去掉 LLM 的理由：

1. 当前任务的指令是短模板（“把三个方块放到同色目标，绕开墙”），LLM 的
   开放世界语义/常识推理没有被使用，保留它只会引入 450M 冻结参数、token
   拼接和 RoPE 偏移等混杂因素。
2. LLM 的 prefix 机制让“场放在哪、谁读得到”难以控制；VLA-lite 用独立的
   空间网格和 readout，场的位置和读取路径是显式的。
3. 小模型允许 3 seed × 多臂 × 多 OOD 的完整矩阵在 2×4090 上跑完；E0 的
   单臂训练已经需要数小时，再乘 seed/臂数不现实。
4. NCA 贡献的对照要求“除场生成器外全部固定”；一个 7M 的可训练栈比
   100M 的 expert 更容易做到参数/算力匹配。

**代价与边界**：VLA-lite 不是基础模型，不承诺开放词汇、跨任务语义泛化或
LLM 式推理。所有结论限定在双墙多物体抓放及其受控 OOD 上。若未来需要语言
泛化，可把 VLA-lite 的场/解码器蒸馏回 SmolVLA，或加回冻结 LLM；这是后续
工作，不是本任务书的成功判据。

### 1.4 研究问题与假设

**RQ1（通道）**：在一个读取端可训练的 VLA 里，NCA 场能否稳定提升闭环成功率？

- H1：NCA 臂 ≥ no-field +10 pp（配对，teacher subgoal 与端到端各测一次）。
- H1b：推理时把 NCA 场置零或换成 GT 场，成功率至少掉 5 pp。

**RQ2（来源）**：增益是否来自 NCA 的迭代结构，而不是“任何单调势场”？

- H2a：分布内 NCA 与 GT 场上界差距 ≤5 pp，且不差于 CNN/Euclid 5 pp 以上。
- H2b：G32 训练、G64 推理时，NCA 掉点 ≤5 pp，CNN 掉点 ≥10 pp，NCA 比 CNN
  高 ≥10 pp。NCA 权重与网格尺寸无关；同参数 CNN 的感受野固定在 cell 数上。
- H2c：障碍数从 2 增到 4–8 的零样本 OOD 上，NCA 比 CNN/Euclid 高 ≥10 pp。

**RQ3（端到端价值）**：场同时进入高层和低层，是否把 E0 的高层瓶颈补上？

- H3：NCA-both ≥ NCA-low ≥ no-field（端到端），且 NCA-both 的高层 visit
  成功率显著高于 vision-only 高层。
- H3 只在 G1、G2 通过后才作为“NCA 改善 VLA 性能”的主张；否则只报告
  “场对低层有帮助/无帮助”。

---

## 2. 设计原则与可证伪性

### 2.1 五条不变量

1. **场只从策略内部来**：主臂的场由 NCA 在环路内生成；推理时不读 GT 场、
   不读 GT 障碍。GT 场臂只作为上界诊断。
2. **只改场生成器**：no-field / zero / random / GT / Euclid / CNN / NCA 七臂
   共用视觉塔、适配器、障碍头、动作解码器、高层、数据、动作标签、训练预算。
3. **动作标签固定**：全部臂拟合同一套 GT-field teacher 的绝对目标动作
   （`multi_lowlevel_2w.npz` 的 `actions`），避免“场质量”和“动作标签质量”
   混淆。
4. **场景级切分**：train/val/test 按 scene 切，chunk 和 visit 不跨切分；
   外部评估用 E0 同一批 200 场景（seed 777）。
5. **先写判据再跑**：G0–G3 和停止规则在实验前固定；测试集不用于调参。

### 2.2 主张阶梯

| 层级 | 主张 | 需要的证据 | 不通过时怎么写 |
|---|---|---|---|
| C0 | 场通道在这个 VLA 中有效 | G0 + G1 | 通道仍未修好，不能评价 NCA |
| C1 | NCA 是有效场源 | G1 + 分布内 G2a | NCA 可用，但不优于简单场 |
| C2 | NCA 结构不可替代 | G2b/G2c 至少一项 | “场内容有用，迭代结构非必要” |
| C3 | NCA 改善端到端 VLA | G3 + C2 | 只主张低层机制收益 |
| C4 | 可推广到更大多任务 | 多任务/OOD 复现 | 限定在双墙任务 |

**可证伪点**：如果 NCA ≈ CNN ≈ Euclid（分布内和所有 OOD 都在 ±5 pp 内），
则“NCA 结构贡献”被证伪；仍可保留“场条件有效”的工程结论，但不能写 NCA
贡献。如果 NCA 只在分布内好、OOD 崩，则支持“过拟合到训练场景”，同样不能
主张结构优势。

### 2.3 预注册门（G0–G3）

#### G0：基础设施（不过则不做科学）

- **确定性**：同一 ckpt 连跑两次，200 场景逐场景结果完全一致（含 det-noise）。
- **障碍读取**：验证集障碍 IoU ≥ 0.90；若 <0.90，先修视觉/障碍头，不比较场源。
- **通道上界**：GT 场臂（teacher subgoal）≥ no-field +10 pp，配对单侧
  McNemar p<0.05；否则说明读取端/解码器仍不会用场。
- **基线可比**：VLA-lite no-field（预测高层）在外部 200 场景上落在 E0
  SmolVLA no-field 的 ±5 pp 内（E0：87%/88%）。若明显更低，先查高层容量、
  数据切分和动作解码，不进入 G1。
- **泄漏检查**：训练场景指纹与 val/test/外部 200 场景无交集；同一 scene 的
  chunk/visit 不跨切分。

#### G1：NCA 通道

- 主对比（teacher subgoal，分布内，3 seed 合并）：
  **NCA − no-field ≥ +10 pp**，配对单侧 p<0.05，bootstrap 95% CI 下界 >0。
- 顺序控制：NCA ≥ zero-field、random-field（至少各高 5 pp）。
- 敏感性：推理时把 NCA 场置零 → 掉 ≥5 pp；把 NCA 场换成 GT 场 → 变化
  应小（说明 NCA 场已接近上界）或明确记录方向。
- **不通过即停止**：不跑 G2，先诊断解码器是否忽略场（见 §7.3 探针）。

#### G2：NCA 特异性

- **G2a 分布内非劣**：NCA 与 GT 场上界差 ≤5 pp；NCA 比 CNN、Euclid 都不低
  5 pp 以上。
- **G2b 尺度外推**：训练 G=32，推理 G=64（同一权重，适配器/readout 做
  参数无关插值）。NCA 相对 G32 掉点 ≤5 pp；CNN 掉点 ≥10 pp；G64 上
  NCA − CNN ≥ +10 pp。
- **G2c 障碍 OOD**：训练 2 道墙，零样本评估 4–8 道墙/障碍。NCA 比
  CNN/Euclid 高 ≥10 pp 为通过；5–10 pp 且 3 seed 方向一致记为“弱证据”；
  <5 pp 记为失败。
- G2 通过不要求三项全过；预先指定 **G2b 为主判据**，G2c 为支持判据。
  若 G2b 过、G2c 不过，可写“尺度泛化支持 NCA，障碍复杂度不支持”。

#### G3：端到端

- **G3a**：NCA-both（场进高层+低层）≥ VLA-lite no-field +10 pp（预测高层，
  外部 200 场景，3 seed 合并）。
- **G3b**：NCA-both 不差于 E0 SmolVLA no-field 5 pp 以上（非劣），仅作为
  工程可比性；架构不同，不作为因果证据。
- **G3c**：NCA-both ≥ NCA-low（场只进低层）≥ no-field，且高层 visit 成功率
  有同方向提升。
- G3 只有在 G1 和 G2b 通过后，才能写“NCA 改善 VLA 性能”。

#### 停止/降级规则

| 情况 | 动作 |
|---|---|
| G0 确定性或 IoU 失败 | 停，修 harness/视觉，不产生科学结论 |
| G1 失败 | 停，诊断解码器；不跑 G2/G3 |
| G1 过、G2b 失败 | 报告“场有效但 NCA 结构非必要”；G3 降为工程结果 |
| G3a 失败、G1/G2 过 | 报告“低层机制收益，端到端被高层/容量抵消” |
| 任一 OOD 方向与分布内相反 | 只做描述性报告，不并入主主张 |

---

## 3. 系统架构

### 3.1 数据流总览

```
RGB 256×256 ──native──> SigLIP-B/16 (frozen) ──> 16×16×768 patch tokens
                                                        │
                                              spatial adapter (trainable)
                                                        │
                         ┌──────────────────────────────┴──────────────┐
                         │                                             │
                 32×32×64 vision features                    obstacle logits 32×32
                         │                                             │
                         │                                   p_obs = sigmoid(logits)
                         │                                             │
                         └──────────────┬──────────────────────────────┘
                                        ▼
     state(7) ──┐                 NCA core: K=12, shared 3×3 rule
     subgoal(4)─┼──> ext 77ch ──> state C=32, 32×32  (plan + latent)
     language ──┘                       │
                                        ▼
                             readout: 1×1 + stride-2 → 16×16×256 tokens
                                        │
              state/subgoal/language/time tokens ──> action decoder (4 layers, d=256)
                                        │
                                        ▼
                             flow matching → [B,8,7] absolute targets
                                        │
                             exec first 4, replan every 4 plant steps

高层（每 visit 查询一次）：RGB features + state + language + NCA pooled field
        → [gx,gy,gz,open]，与 E0 的 visit command 同语义
```

### 3.2 视觉前端（冻结）

- **模型**：SigLIP-B/16 视觉塔，约 93M 参数，`requires_grad=False`。
- **输入**：数据为 256×256 RGB，原生分辨率直接输入；归一化 `x/255*2-1`
  （等价均值 0.5、标准差 0.5）。权重来自本地缓存的 open_clip
  `ViT-B-16-SigLIP-256`（webli），离线加载。
- **输出**：16×16×768 patch tokens（256 个）。
- **为什么冻结**：与 SmolVLA 的“预训练视觉先验”保持可比；可训练容量集中在
  适配器/NCA/解码器，避免 100M 级微调掩盖结构差异。另设一个“解冻最后一层”
  的诊断臂（§7.3），但不进主矩阵。
- **领域差距风险**：当前场景是俯视渲染的简单几何体，SigLIP 的域差距可能
  存在；适配器 + 障碍头负责把通用特征映射到任务空间。G0 的 IoU 是硬门槛。

### 3.3 空间适配器与障碍头（可训练）

- **输入**：16×16×768 patch tokens（reshape 成 16×16 特征图）。
- **结构**：
  1. 1×1 conv 768→256；
  2. 双线性插值到 32×32；
  3. 两个 residual block：3×3 conv 256→256 + GroupNorm(8) + GELU；
  4. 1×1 conv 256→64，作为 NCA 的视觉外部输入；
  5. 旁路 1×1 conv 256→1，输出 32×32 障碍 logits。
- **参数**：约 1.39M。
- **障碍监督**：目标为数据集里的膨胀障碍网格（墙 + 未放置物体 + 膨胀半径，
  与 teacher 路由一致），32×32，BCE。推理时 `p_obs=sigmoid(logits)`；
  NCA 用软掩码 `alpha=1−p_obs`（可微），障碍 IoU 用 0.5 阈值报告。
- **关键原则**：主臂**不使用 GT 障碍**。GT 障碍只出现在 GT 场上界臂。
- **坐标一致**：适配器输出与 `xy_to_cell` 的 32×32 网格严格对齐；用单元测试
  检查“世界坐标 → 图像像素 → 网格”的往返误差 ≤1 cell。

### 3.4 NCA 规划核（可训练，核心）

**状态与外部输入**

- 每个 cell 的状态 `s ∈ R^32`：通道 0 是 plan 标量，通道 1–31 是 latent /
  morphogen 工作记忆。
- 外部输入 `ext ∈ R^77`，逐 cell 广播：
  - 1 维障碍概率 `p_obs`；
  - 2 维目标相对偏移 `(gc−j)/G, (gc−i)/G`；
  - 2 维起点相对偏移 `(sc−j)/G, (sc−i)/G`；
  - 64 维适配器视觉特征；
  - 8 维任务/语言嵌入（广播）。
- 目标由高层 subgoal `[gx,gy]` 映射到网格；起点由当前 state `[x,y]` 映射。
  两者都是策略输入，不是 oracle 规划信息。

**更新规则**

```
alpha = 1 - p_obs                      # 障碍 cell 冻结
s_0   = 0                              # plan 与 latent 均从零开始
for k in 1..K:
    u = Conv3x3_{shared}([s_{k-1}, ext])       # hidden 96, GELU
    u = Conv1x1(96→96) + GELU
    d = Conv1x1(96→32)                          # 最后一层零初始化
    s_k = s_{k-1} + dt * tanh(d) * alpha        # dt=0.5, 残差、tanh 有界
plan = s_K[..., 0]
```

- **K（2026-10-30 修订，见 §13）**：训练 random-K∈[16,64]（每步均匀采样），
  推理 K=64（G32）；评估时 K 随网格等比缩放（G64→K=128），保证同一物理传播
  距离。固定 K∈{1,4,12,24} 为消融（`--k-steps`）。
- **参数量**：约 107K；与网格尺寸无关，G32/G64 共用同一组权重。
- **稳定性**：残差 + tanh + 最后一层零初始化，初始为恒等映射；梯度裁剪 1.0；
  可选不动点残差损失 `--fp-weight`（R9 Route A）。
- **为什么是 3×3 conv 而不是原 SpatialNCA 的 8 邻域 MLP**：两者都是局部规则；
  conv 更容易与 CNN 对照做参数匹配。原 `soma/spatial_nca.py` 的
  “共享规则 + 障碍冻结 + 迭代传播”语义保留。
- **感受野（修订）**：原设计 K=12 把 NCA 感受野锁在 25×25、与 12 层 CNN 相同，
  只剩“权重共享”一项差异——但 R9 证明 NCA 的优势恰恰来自**无界深度迭代到
  不动点**（cnn_xxl 感受野 ≥G 仍 ρ=.63），K=12 在 G32 带绕行的场景上传播不完，
  G64 更不可能。因此主臂改为深迭代；感受野/深度的公平性由 CNN 的两档对照承担
  （12 层参数匹配 + ≥G 层全局感受野，见 §6.4）。
- **读出**：plan 通道用于辅助监督；完整 32 通道经 1×1 conv→256 + stride-2
  conv→16×16×256，作为动作解码器的 256 个网格 token。

**场的监督**

- 目标：GT 测地距离场 `d/(2G)`，障碍 cell 不参与 MSE；G32/G64 都用各自的
  `2G` 归一化，保证尺度外推时数值范围一致。
- 障碍 cell 的 plan 不监督，但 alpha 冻结使其不传播。
- 辅助损失权重 `λ_plan=0.1`（主臂）；`λ_plan∈{0,0.1,0.5}` 为消融。
- 离线指标：plan MSE、墙边符号违例、排序违例、固定点残差
  `|s_K−s_{K−1}|`。这些只作诊断，不作为模型选择依据（E0 教训）。

### 3.5 动作解码器（可训练）

- **输入 token**：
  - 256 个 NCA 网格 token（16×16×256，展平）；
  - 1 个 state token：`[state(7), subgoal(4)]` → MLP → 256；
  - 1 个 language token：语言槽输出 → 256；
  - 1 个 flow-time token：正弦嵌入 → MLP → 256；
  - 1 个 phase token（可选，标识 grasp/transport/place 阶段；主臂不加，
    避免额外标签）。
- **结构**：4 层 Transformer encoder，d=256，8 heads，FFN 1024，
  pre-norm，self-attention 同时看网格和上下文 token。
- **输出头**：`Linear(256→256) + SiLU + Linear(256→8×7)`，输出 8 步动作
  chunk；训练用 flow matching，推理用 10 步 Euler 积分。
- **参数**：约 3.39M。
- **动作语义**：与 E0 完全一致——绝对 plant target
  `[x,y,z,rot(3),grip]`，不是 delta；归一化用训练集 mean/std，rot 在本任务
  恒定（训练时置零，推理保留当前 rot）。
- **执行**：chunk=8，执行前 4 步后重规划（与 E0 的 `STRIDE=4` 一致）。
- **关键**：所有臂共享这个解码器；no-field 臂把 256 个网格 token 置零，
  参数完全相同；NCA 的迭代算力差异由 zero-field 臂单独控制。

### 3.6 高层决策（可训练）

- **查询时机**：每个 visit 开始时（hover 处）查询一次，输出该 visit 的终端
  命令 `[gx,gy,gz,open]`，与 E0 的 visit command 同语义。
- **输入**（base 臂，已实现）：SigLIP patch tokens（256×768）、归一化
  state、language token；（NCA-both 臂）另加 NCA 池化场特征。
- **结构**：2 层 Transformer，d=256，8 heads；vision/state/language token
  与 learned query 拼接，取 query 输出接 `Linear(256→4)`。约 1.92M。
- **输出归一化**：用 visit 数据集的 `ch_mean/ch_std`；评估时反归一化。
- **训练**：Stage B，BC 于 teacher visit command；NCA 冻结（来自 Stage A）。
- **两种高层臂**：
  - `HL-vision`：不看场，所有低层臂共用；用于隔离低层归因。
  - `HL-nca`：看 NCA 池化特征；用于 NCA-both 端到端。
  - `HL-gt`：看上界 GT 场，只做诊断。
- **为什么高层也要场**：E0 的 teacher-vs-predicted 差距（100% vs 74%/86%）
  说明对象/目标选择是主要瓶颈。NCA 的场如果只喂低层，高层仍可能选错目标。

### 3.7 语言槽（无 LLM）

- **主臂（已实现）**：`learned task token`——8 维可学习任务嵌入 +
  `Linear(8→256)`；嵌入同时广播进 NCA（`task8`）和解码器/高层（token）。
  当前任务指令恒定，语言槽主要保证接口完整，不应期待语言带来增益。
- **预留消融**：冻结 `all-MiniLM-L6-v2`（约 22M，384 维）+ `Linear(384→256)`。
  当前机器离线且无该权重，故实现为 learned slot；权重可用时再作为消融接入。
- **指令模板**（未来接入句向量时使用）：
  `"place the three blocks on the matching colored targets and avoid the walls"`，
  全场景同一条。
- **language swap**（换成无关指令/任务 id）作为 sanity check，预期无显著影响。
- **边界**：不声称开放词汇或指代理解；若未来做“按语言指定顺序/目标”的
  变体，再引入可区分的指令和语言泛化实验。

### 3.8 参数量与计算

| 模块 | 参数量 | 是否训练 | 备注 |
|---|---:|---|---|
| SigLIP-B/16 视觉塔 | ~92.9M | 冻结 | 16×16×768 |
| 空间适配器 + 障碍头 | 1.39M | 是 | 32×32×64 + logits |
| NCA 核 | 0.107M | 是 | K=12，3×3 共享规则 |
| 动作解码器 | 3.39M | 是 | 4 层 d=256 |
| 高层决策 | 1.92M | 是 | 2 层 d=256（含未启用的 field 投影） |
| 语言投影（learned slot） | 0.002M | 是 | 8→256 |
| **可训练合计** | **~7.4M** | | 低层 5.49M + 高层 1.92M |

- 单次前向：SigLIP 是主要开销（约 17 GFLOPs/图）；NCA 12 次迭代约 1.2 GMAC；
  解码器 4 层 × 260 token 约 1 GMAC/chunk；flow 10 步约 10 GMAC/chunk。
  在 4090 上可忽略，瓶颈在视觉塔和仿真。
- 训练显存：batch 64、32×32 网格、BPTT 12 步，状态/激活 <2 GB；不需要
  gradient checkpointing。若开 K=24 或 G=64，再评估。

---

## 4. 数据

### 4.1 任务与环境

- 任务：多物体顺序抓放，N=3 个彩色方块，每个方块有同色目标；两道垂直带缝墙，
  随机起点/目标；成功 = 3/3 放置。
- 工作空间：`x∈[0.22,0.74] m`，`y∈[0.22,0.60] m`；状态 7 维
  `[x,y,z,rot(3),grip]`；动作 7 维绝对目标。
- 路由网格：64×64（E0 数据）；VLA-lite 主训练用 32×32（2×2 下采样），
  尺度 OOD 用 64×64。
- 环境物理：`MultiEnv` 阻尼 plant，`plant_f=0.5`，确定性抓取，碰撞/撞飞/
  墙阻挡；噪声 `pos 1mm, rot 0.01, open 0.01`。
- 动作标签：GT-field teacher（oracle 最近优先 visit、8 连通 Dijkstra、
  escape readout、pure pursuit、脚本化抓放），与 E0 完全相同。

### 4.2 现有数据与再生成

现有文件：

- `data/multi_lowlevel_2w.npz`：25,004 chunks，4,806 visits，600 episodes；
  `images (25004,3,256,256) uint8`、`states (25004,7)`、
  `subgoals (25004,4)`、`actions (25004,8,7)`、`fields (25004,4,64,64)`
  `[obstacle, GT plan, dx/G, dy/G]`、`plans_wrong/random`、`metas (25004,10)`、
  `detour`、归一化统计。
- `data/multi_visit_2w.npz`：72,000 个 visit 决策样本；
  `images (72000,3,256,256)`、`chunks (72000,8,4)`、`states (72000,7)`、
  `metas (72000,12)`（objs 6 + targets 6）、`ch_mean/ch_std`。

**再生成的原因**：

1. 现有 `metas` 没有显式的 `scene_id/ep_id/visit_id`，无法做严格的场景级
   切分和泄漏检查；
2. VLA-lite 需要 32×32 的障碍/plan 监督、障碍掩码、placed/active 标签；
3. 需要把 split 写进数据，避免训练脚本临时猜。

**新增脚本** `scripts/vla_lite_data.py`（复用 `multi_lowlevel_data.py` 与
`multi_visit_data.py` 的采样/teacher 逻辑）：

- 输出 `data/vla_lite_2w.npz`（低层）与 `data/vla_lite_visit_2w.npz`（高层）。
- 低层新增键：
  - `obstacles [N,32,32] uint8`：膨胀障碍（与 teacher 路由一致）；
  - `plans [N,32,32] float16`：GT 测地场，`d/(2*32)`，障碍 cell 置 0；
  - `scene_id/ep_id/visit_id/t/active_idx [N]`；
  - `placed_mask [N,3] uint8`、`split [N]`（0/1/2）；
  - `st_mean/std, act_mean/std`（只在 train split 上算）。
- 高层新增键：`scene_id/ep_id/visit_id/placed_mask/split`，其余沿用
  `multi_visit_2w.npz` 的 `images/chunks/states/metas/ch_mean/ch_std`。
- 生成时校验：每条 episode 必须成功；失败 episode 重采样并记录；
  600 条全部成功后再写盘。
- 切分：按 `scene_id` 顺序 **450 train / 75 val / 75 test**；不按 chunk 或
  visit 随机切。外部评估仍用 seed 777 的 200 场景，与 seed 0 训练数据
  按构造不相交；脚本额外做场景指纹交集断言。
- 归一化：`st_mean/std`、`act_mean/std` 只从 train split 估计；高层
  `ch_mean/std` 从 train split visits 估计。

**数据量预期**：450/600 ≈ 75% 的低层 chunks（约 18.7K）和 visits（约 3.6K）
进训练；75 个 val 场景用于早停/诊断；75 个 test 场景用于内部复核。外部
200 场景是唯一主报告集，test 场景不用于调参。

### 4.3 泄漏控制

- 场景是随机采样的连续参数；同一场景的所有 chunk/visit 必须同属一个 split。
- 场景指纹：`round(objs,5), round(targets,5), round(walls,5)` 的哈希；
  训练前断言 train ∩ val = train ∩ test = ∅，且与外部 200 场景无交集。
- 视觉泄漏：同一场景的多帧图像高度相似，禁止按帧随机切分。
- 归一化统计不得用 val/test；否则会把测试分布信息带进训练。
- 高层 Stage B 只使用 train scenes 的 visits；val/test visits 只做评估。

### 4.4 OOD 轴

| 轴 | 训练 | 评估 | 目的 | 优先级 |
|---|---|---|---|---|
| O1 尺度 | G=32 | G=64（同一 NCA 权重） | 检验网格无关/固定参数传播 | 主 |
| O2 障碍复杂度 | 2 道墙 | 4–8 道墙/障碍，零样本 | 检验长程绕行与多障碍 | 主 |
| O3 绕行强度 | 混合 detour | 仅高 detour（routed/euclid ≥1.45） | 检验规划需求 | 次 |
| O4 外观 | 标准渲染 | 颜色/光照/纹理扰动 | 检验视觉鲁棒性 | 可选 |

- **O1 实现**：适配器输出双线性插值到 G×G；NCA 的 3×3 conv 与权重无关；
  readout 用 stride-2 把 G×G 映射回 16×16 token（G64→32×32→16×16 需两次
  stride-2，或 adaptive pool 到 16×16）。解码器 token 数保持 256，保证只有
  “网格分辨率/传播”变化。
- **O2 实现**（已实现，`scripts/vla_lite_ood.py::sample_scene_multi_wall`）：
  两墙基底 + 2–6 段随机额外墙段（长 8–16 cm），保留起点净空与
  `_scene_reachable` 检查；其余物理/成功判据不变。训练集只出现 2 道墙；
  O2 是零样本。评估：`--scene-type multi-wall --n-extra-walls 2 6`。
- **O3**：评估时筛选 `detour≥1.45` 的场景（从 200 场景里选，或另生成 200
  个高 detour 场景）。报告样本数和配对统计。
- **O4**：只在时间允许时做；不作为 G2 判据。

---

## 5. 训练

### 5.1 Stage A：低层动作策略（主训练）

**目标**：在 teacher subgoal 条件下，学习“视觉 + 状态 + 场 → 8 步绝对动作”。

- 数据：`vla_lite_2w.npz` train split；每条样本含 image/state/subgoal/
  action/obstacle/plan。
- 视觉塔冻结；适配器、NCA、解码器可训练；高层不参与。
- 损失：
  - `L_action`：flow matching MSE（与 SmolVLA 同参数化）；
  - `L_plan = 0.1 * masked MSE(plan, GT plan)`（自由 cell）；
  - `L_obs = 0.5 * BCEWithLogits(obstacle logits, GT obstacle)`；
  - 总计 `L = L_action + L_plan + L_obs`。
- 关键：NCA 的障碍输入是**预测的** `p_obs`，不是 GT；GT 障碍只用于
  `L_obs` 监督和 GT 上界臂。
- 步数：20,000；batch 64；AdamW；`lr`: adapter 3e-4，NCA 1e-3，
  decoder 1e-4；betas (0.9,0.95)，wd 0.01；warmup 500；cosine 到 10%；
  bf16；梯度裁剪 1.0。
- 固定步数，不按离线 MSE 早停（E0 教训）；如需 checkpoint，选固定
  `step=20000`。
- seed：0/1/2；所有臂同 seed 同初始化策略（NCA/解码器用同一初始化种子）。

### 5.2 Stage B：高层 visit 策略

**目标**：在每个 hover 处输出终端 visit command。

- 数据：`vla_lite_visit_2w.npz` train split；输入 image/state/language；
  NCA-both 臂额外读 Stage A 的冻结 NCA 特征。
- 目标：`[gx,gy,gz,open]` 归一化；损失：
  - `L_xyz = L1(pred_xyz, gt_xyz)`（归一化空间）；
  - `L_open = BCE(pred_open, gt_open)`；
  - 总计 `L = L_xyz + L_open`。
- 步数：8,000；batch 64；lr 1e-4；cosine；bf16；其余同 Stage A。
- NCA/适配器/解码器冻结；只训高层。3 seed。
- 诊断：teacher subgoal vs 预测 subgoal 的差距，直接对标 E0 的 100% vs
  74%/86%。

### 5.3 Stage C：联合微调（可选，默认不做）

只有在 Stage A 的 G1 和 Stage B 的高层成功率都通过后才考虑：

- 解冻 NCA + 适配器 + 高层（解码器可选），lr 2e-5；
- 5,000 步，batch 32；
- 训练时以 50% 概率用预测 subgoal 替换 teacher subgoal（scheduled
  sampling），减少 exposure bias；
- 目的：让高层和 NCA 共同适应闭环误差，而不是只优化单步 BC。
- 若 Stage C 使 G1 的 teacher-subgoal 指标变差，回退到 Stage A+B，不强行
  联合训练。

### 5.4 工程细节

- **预计算 SigLIP 特征**（视觉塔冻结，特征精确）：
  - 低层 25,004×256×768 fp16 ≈ 9.8 GB；
  - 高层 72,000×256×768 fp16 ≈ 28.3 GB；
  - 总计约 38 GB。用 `np.memmap` 或分片 `.npy` 存储；训练时按 index 读取。
  - 注意：`NpzFile` 每次切片都会重新解压整个数组，预计算/训练前须一次性
    载入图像或数组，否则吞吐会掉一个量级（实测 128 图切片约 12 s）。
  - 若磁盘不足，改为在线前向（慢 2–3×）；或只对 train split 预计算。
- **数据加载**：低层样本是 npz 里的 uint8 图像，读取/归一化开销小；预计算
  特征后瓶颈在 NCA BPTT 和解码器。`num_workers=8`，`pin_memory=True`。
- **混合精度**：bf16 前向/反向；NCA 的 tanh 残差在 bf16 下稳定；损失用
  fp32 累加。
- **确定性**：训练 seed 固定；评估 `torch.manual_seed(seed*100003)`，
  flow 噪声按 `(scene_id, query_i)` 哈希，保证换臂/换 seed 可比。
- **日志**：每 100 步记录 action loss、plan loss、obstacle loss、梯度范数；
  每 2,000 步在 val 上算一次 plan MSE/IoU（只诊断，不选模型）。
- **检查点**：只存 `model` 和归一化统计；`ckpts/` 已在 `.gitignore`，
  不入库。

---

## 6. 评估

### 6.1 Harness

直接复用 E0 的批量向量化闭环评估骨架（`scripts/eval_field_expert_vec.py`），
新建 `scripts/eval_vla_lite.py`：

- **场景**：200 个固定场景，`scene seed=777`，与 E0 完全同一批。
- **批量**：`batch_episodes=16`，所有 episode 锁步推进；每场景独立
  `ep_rng = RandomState(777*7919+i)`，保证各臂遇到相同 plant 噪声。
- **episode**：`max_steps=1500`，`visit_cap=180`，`exec_steps=4`，
  `lift_n=10`；成功 = 3/3 放置。
- **高层查询**：每个 visit 一次；主报告用 VLA-lite 预测高层；机制对比用
  `--teacher-subgoal` 的 oracle 命令。
- **低层查询**：每 4 步一个 chunk；chunk=8，执行前 4。
- **flow 噪声**：`--det-noise`，噪声由 `(scene_id, query_i)` 确定；所有臂、
  所有 seed 使用同一噪声流。
- **状态**：严格使用当前环境状态渲染图像和查询，禁止缓存过期状态
  （E0 修过的 bug）。
- **输出**：每场景 `success/placed/wall_hits/bumps/steps/visits_ok/
  visits_total/dev`；JSON + stdout 表格。

### 6.2 指标

**主指标**

- 全成功 3/3（主）。
- placed count（0–3，次级）。
- visit 成功率（visits_ok/visits_total）。
- wall hits、bumps、steps（效率/安全性）。

**场诊断（离线或在线探针）**

- 障碍 IoU（预测 vs GT）。
- plan MSE（自由 cell）。
- 墙边符号违例：距障碍 ≤1 cell 的 free cell 上，plan 是否沿正确方向下降。
- 排序违例：随机采样 free cell 对，比较 plan 差与测地距离差的方向。
- 固定点残差：NCA 最后一步 `|s_K−s_{K−1}|` 的均值；太大说明未收敛。
- 动作误差：teacher subgoal 下，首 chunk 与 teacher 动作的 L2（归一化/
  反归一化都报）。

**高层诊断**

- visit command 的 L1（归一化）与 `open` 准确率。
- teacher subgoal vs 预测 subgoal 的成功率差（对标 E0 的 100% vs 74%/86%）。
- 高层选错对象/目标的比例（用 meta 的 active/placed 判定）。

### 6.3 统计

- **配对**：同一 scene、同一 plant 噪声、同一 det-noise，逐场景配对。
- **主检验**：精确 McNemar（单侧用于预注册方向，双侧用于探索）。
- **区间**：10,000 次 scene bootstrap 的 95% CI。
- **seed**：3 seed 分层合并（先按 seed 算不一致对，再合并）；同时报告
  每个 seed，禁止只报最好 seed。
- **多重比较**：三个主对比（GT vs no-field；NCA vs CNN；NCA vs Euclid）
  用 Holm；其余为次级/描述性。
- **功效**：n=200 时，+10 pp 的功效约 71%；+5 pp 功效不足。OOD 上 5–10 pp
  只能写“弱证据/方向一致”，不能写显著。
- **不调测试集**：val 75 场景用于诊断；外部 200 场景只在最终评估使用。
  若必须选 checkpoint，用固定步数或 val 闭环，不用 test。

### 6.4 实验臂与控制

**Stage A 低层（teacher subgoal，7 臂 × 3 seed）**

| 臂 | 场生成器 | 障碍来源 | 目的 |
|---|---|---|---|
| no-field | 无（网格 token 置零） | — | 主基线；参数匹配，算力由 zero-field 控制 |
| zero-field | NCA 运行（含 plan 监督），但 readout token 置零 | 预测 | 控制“多算了 12 次迭代” |
| random-field | 平滑随机场（离线固定） | 预测 | 控制“任意空间信号” |
| GT-field | GT 测地场 | GT（oracle） | 通道上界 |
| Euclid | 零参数欧氏势场 | 预测 | 强简单基线（E0 方法 1 的教训） |
| CNN | 12 层 3×3 CNN，参数/感受野匹配 | 预测 | 前馈对照 |
| NCA | K=12 共享 3×3 规则 | 预测 | 主臂 |

- 所有臂共享适配器、障碍头、解码器、数据、动作标签、步数、seed。
- CNN 与 NCA 匹配（2026-10-30 修订）：CNN 输出与 NCA 同宽的 **32 通道状态**
  （通道 0 = plan，其余 latent），解码器看到的场带宽一致；原实现 CNN 只输出
  1 通道 plan、其余补零，NCA>CNN 可能只是“32 个学习通道 > 1 个标量”。
  两档 CNN：`cnn`（12 层，参数匹配 ~111K vs NCA ~107K）与 `cnn --cnn-layers 32`
  （感受野 65 ≥ G32，全局感受野对照，对应 R9 的 cnn_xxl）。
- **带宽口径**：`--readout full`（默认）下 NCA/CNN/vis 暴露 32 通道，
  GT/Euclid/random 只有通道 0；因此 NCA vs GT/Euclid 的主比较必须在
  `--readout plan`（所有臂只暴露 plan 标量）下再跑一遍，两种口径都报。
- zero-field 臂的 NCA **参与训练**（进优化器、同样的 plan 监督），与 NCA 臂
  唯一差异是解码器看不到场（原实现 NCA 参数未进优化器，是 bug）。vis 臂的
  1×1 投影同理已加入优化器。
- Euclid：`plan = ||cell−goal||/G`（或归一化到 [0,1]），无障碍感知；
  用同一个 readout 进解码器。它故意“不懂墙”，用来检验“单调势场是否足够”。
- GT-field：oracle 障碍 + oracle 测地场，作为上界；不参与主主张。
- zero-field：NCA 仍运行、仍收视觉 ext、仍算 plan 监督，但 readout token 在
  进解码器前置零；用于区分“NCA 的额外算力”与“场内容”。

**Stage B 高层（3 臂 × 3 seed）**

| 臂 | 输入 | 目的 |
|---|---|---|
| HL-vision | 视觉+state+language | 无场基线 |
| HL-nca | + NCA 池化特征 | NCA-both |
| HL-gt | + GT 场池化 | 上界诊断 |

**Stage C / 端到端组合（评估为主）**

| 组合 | 低层 | 高层 | 目的 |
|---|---|---|---|
| E2E-no-field | no-field | HL-vision | 端到端基线 |
| E2E-NCA-low | NCA | HL-vision | 场只进低层 |
| E2E-NCA-both | NCA | HL-nca | 主端到端臂 |
| E2E-GT-both | GT-field | HL-gt | 上界 |

**外部参照（不重新训练）**

- E0 SmolVLA no-field：87%/88%（s0/s1）。
- E0 SmolVLA gt-plan：70%/79%。
- E0 teacher-subgoal gt-plan：100%。
这些只作为背景刻度，不参与 VLA-lite 的配对检验。

---

## 7. 实验矩阵与消融

### 7.1 主矩阵

**分布内（外部 200 场景，seed 777）**

1. Stage A 七臂 × 3 seed，teacher subgoal：G1 + 分布内 G2a。
2. Stage B 三臂 × 3 seed：高层成功率与命令误差。
3. 端到端四组合：G3。
4. 每臂报告：success、placed、wall hits、bumps、steps、visits、首 chunk L2。

**OOD**

5. O1：G32 训练的七臂，G64 推理（同一权重）；报告掉点与 NCA−CNN 差。
6. O2：4–8 障碍零样本；七臂同协议。
7. O3：高 detour 子集；至少 NCA/CNN/Euclid/no-field 四臂。
8. O4（可选）：外观扰动。

**诊断**

9. Swap：NCA 训练的策略在推理时换成 GT 场；GT-field 训练的策略换成 NCA 场。
10. Zero-at-inference：NCA 场置零，测掉点。
11. Obstacle oracle vs predicted：同一 NCA 臂，障碍换 GT，量化视觉误差代价。
12. Teacher vs predicted subgoal：隔离高层瓶颈。

### 7.2 消融（按优先级）

| 消融 | 取值 | 问题 | 优先级 |
|---|---|---|---|
| K 步数 | 1 / 4 / 12 / 24 | 迭代是否必要？K=1 是否已足够？ | 高 |
| 场监督 | λ_plan 0 / 0.1 / 0.5 | 辅助场损失是否帮助动作？ | 高 |
| 网格 | G16 / 32 / 64 | 分辨率与尺度外推 | 高 |
| 视觉解冻 | 冻结 / 最后一层微调 | 域差距是否限制上限？ | 中 |
| 语言 | learned token / MiniLM（待权重）/ swap | 语言槽是否有任何作用？ | 中 |
| 障碍头 | 预测 / GT | 视觉误差 vs 场误差 | 中 |
| 解码器 | 4 层 / 2 层 / MLP | 容量是否足够读场？ | 中 |
| 高层 | vision / nca / gt | 高层是否用场？ | 高 |
| Stage C | 有 / 无 | 联合微调是否值得？ | 低 |

### 7.3 机制探针（G1 失败时必做）

- **场置零探针**：NCA 臂推理时把网格 token 置零，若成功率不掉，说明解码器
  完全忽略场。
- **线性探针**：用解码器中间层特征预测 plan/obstacle，检查场信息是否进入。
- **注意力探针**：统计解码器对网格 token 的注意力质量；与 no-field 臂对比。
- **动作扰动**：沿 plan 梯度方向扰动场，观察动作是否同方向变化。
- **swap 探针**：NCA 场换 GT 场、GT 场换 NCA 场，区分“场内容错”与“读取端
  不敏感”。

这些探针只用于诊断，不替代 G1 的闭环判据。

---

## 8. 风险与缓解

| 风险 | 表现 | 缓解 |
|---|---|---|
| NCA 训练不稳定 | plan 爆炸/不收敛、固定点残差大 | 残差+tanh+零初始化；K 从 4 预热到 12；梯度裁剪；λ_plan 退火 |
| 障碍头 IoU 低 | 预测障碍漏墙/膨胀过度 | 颜色/光照增强；膨胀标签；G0 硬门槛；oracle 障碍上界诊断 |
| 视觉域差距 | SigLIP 特征分不开墙/物体 | 适配器容量；解冻最后一层诊断；必要时换 DINOv2（需重跑对照） |
| 坐标/长宽比错位 | 场与图像/动作不对齐 | 复用 `xy_to_cell`；往返单测 ≤1 cell；叠加可视化 |
| 高层仍是瓶颈 | teacher 100% vs 预测低 | NCA-both；高层容量/数据；Stage C；报告高层命令误差 |
| 解码器忽略场 | 置零不掉点 | 场 dropout 训练（0.1–0.2）；辅助 readout；探针定位 |
| NCA ≈ Euclid | 分布内无差异 | 预先接受该可能；把主张降到 C1；用 OOD/尺度检验 C2 |
| CNN 对照过强/过弱 | 参数/感受野不匹配 | 严格匹配 12 层/C32/RF25；报告参数量；3 seed |
| 无 LLM 的语言泛化弱 | 换指令无反应 | 限定主张；语言槽消融；未来蒸馏回 SmolVLA |
| 评估过拟合 | 在 200 场景上调参 | val 75 场景；固定步数；测试集只最终用 |
| 数据泄漏 | 同场景跨 split | 场景指纹断言；按 scene 切；归一化只用 train |
| 计算/时间超支 | 矩阵跑不完 | 预计算特征；先跑 G0/G1 四臂；OOD 分阶段；3 seed 优先于 5 seed |
| 与 E0 不可比 | 架构/高层不同 | 明确 G3 为次级；主因果在 VLA-lite 内部；外部数字只作刻度 |

---

## 9. 里程碑与产出

| 里程碑 | 时间 | 内容 | 通过标准 |
|---|---|---|---|
| M0 | D1–D3 | 数据再生成、特征预计算、harness、单测 | 场景切分/指纹/IoU 管线跑通 |
| M1 | D4–D7 | Stage A no-field/GT/NCA pilot | G0 通过；GT ≥ no-field +10 pp |
| M2 | D8–D10 | Stage A 七臂 × 3 seed | G1 判定 |
| M3 | D11–D14 | Stage B 三臂 + 端到端组合 | G3 所需数据齐 |
| M4 | D15–D18 | O1/O2/O3 + swap/探针 | G2 判定 |
| M5 | D19–D21 | 分析、报告、图、提交 | 报告自洽；所有 seed/臂都报告 |
| 缓冲 | D22–D24 | 复跑异常、补 seed | — |

**产出**

- 代码：`soma/vla_lite/`、`scripts/*vla_lite*`；
- 数据：`data/vla_lite_2w.npz`、`data/vla_lite_visit_2w.npz`（不入库）；
- 日志/JSON：`logs/vla_lite/`；
- 报告：`reports/VLA_LITE.md`（结论、矩阵、判据、失败分析、局限）；
- 图：`assets/vla_lite/`（主矩阵、OOD、场可视化、探针）；
- 决策备忘：是否值得把 NCA 场蒸馏回 SmolVLA / 进入下一轮。

**报告结构**

1. 问题与 E0 教训；
2. 架构与参数量；
3. 数据/切分/OOD；
4. G0–G3 逐项判定；
5. 主矩阵与 OOD；
6. 消融与探针；
7. 失败模式与局限；
8. 结论：C0–C4 到哪一级；
9. 下一步（蒸馏/多任务/更大规模）。

---

## 10. 代码与复现

### 10.1 文件结构

```
soma/vla_lite/
  __init__.py
  vision.py          # 冻结 SigLIP 包装 + 特征预计算接口
  adapter.py         # 16×16×768 → 32×32×64，障碍头
  nca.py             # 网格无关 NCA 核（K、dt、FiLM/ext 注入）
  decoder.py         # 4 层动作解码器 + flow matching
  language.py        # learned task token（MiniLM 接口预留）
  high_level.py      # visit 决策头
  policy.py          # 端到端包装（低层+高层）
  baselines.py       # Euclid / CNN / GT / random 场生成器
scripts/
  vla_lite_data.py         # 再生成低层/高层数据（含 split/ID/32×32 场）
  precompute_siglip.py     # 冻结视觉特征 → memmap
  train_vla_lite_low.py    # Stage A
  train_vla_lite_high.py   # Stage B
  finetune_vla_lite_joint.py # Stage C（可选）
  eval_vla_lite.py         # 批量闭环评估
  probe_vla_lite.py        # 置零/线性/注意力/swap 探针
  analyze_vla_lite.py      # 配对统计、功效、Holm
  plot_vla_lite.py         # 主矩阵/OOD/场可视化
reports/VLA_LITE.md
assets/vla_lite/
```

### 10.2 复现命令（草案）

```bash
# 0) 数据（CPU，一次；从 E0 的 multi_*_2w.npz 派生）
python scripts/vla_lite_data.py \
    --out-low data/vla_lite_2w.npz --out-visit data/vla_lite_visit_2w.npz

# 1) 冻结视觉特征（每卡分片）
CUDA_VISIBLE_DEVICES=0 python scripts/precompute_siglip.py \
    --data data/vla_lite_2w.npz --out data/feat_low_fp16.npy
CUDA_VISIBLE_DEVICES=0 python scripts/precompute_siglip.py \
    --data data/vla_lite_visit_2w.npz --out data/feat_visit_fp16.npy

# 2) Stage A：七臂 × 3 seed（示例：NCA s0）
CUDA_VISIBLE_DEVICES=0 python scripts/train_vla_lite_low.py \
    --data data/vla_lite_2w.npz --feat data/feat_low_fp16.npy \
    --arm nca --seed 0 --steps 20000 --batch 64 \
    --out ckpts/vla_lite/nca_s0

# 3) Stage B：高层（示例：base s0；NCA-both 变体待 Stage A 场特征接入）
CUDA_VISIBLE_DEVICES=1 python scripts/train_vla_lite_high.py \
    --data data/vla_lite_visit_2w.npz --feat data/feat_visit_fp16.npy \
    --seed 0 --steps 8000 \
    --out ckpts/vla_lite/high_s0

# 4) 闭环评估（外部 200 场景，teacher 高层，det-noise 默认开启）
CUDA_VISIBLE_DEVICES=0 python scripts/eval_vla_lite.py \
    --ckpt ckpts/vla_lite/nca_s0 \
    --n-episodes 200 --seed 777 --batch-episodes 16 \
    --json-out logs/vla_lite/nca_teacher_s0.json

# 预测高层（主报告用；Stage B ckpt）
CUDA_VISIBLE_DEVICES=0 python scripts/eval_vla_lite.py \
    --ckpt ckpts/vla_lite/nca_s0 --high-ckpt ckpts/vla_lite/high_s0 \
    --n-episodes 200 --seed 777 --batch-episodes 16 \
    --json-out logs/vla_lite/nca_pred_s0.json

# 机制对比：随机 flow 噪声（关掉 det-noise；仅诊断）
CUDA_VISIBLE_DEVICES=0 python scripts/eval_vla_lite.py \
    --ckpt ckpts/vla_lite/nca_s0 --stochastic-noise \
    --n-episodes 200 --seed 777 --json-out logs/vla_lite/nca_stoch_s0.json

# O1 尺度：G64（同一权重；K 自动按 G 等比放大，--no-k-scale 关闭）
CUDA_VISIBLE_DEVICES=0 python scripts/eval_vla_lite.py \
    --ckpt ckpts/vla_lite/nca_s0/last.pt --grid 64 \
    --n-episodes 200 --seed 777 \
    --json-out logs/vla_lite/nca_teacher_g64_s0.json

# O2 障碍复杂度：两墙基底 + 2–6 段额外墙（共 6–10 个墙矩形），零样本
CUDA_VISIBLE_DEVICES=0 python scripts/eval_vla_lite.py \
    --ckpt ckpts/vla_lite/nca_s0/last.pt --scene-type multi-wall \
    --n-extra-walls 2 6 --n-episodes 200 --seed 777 \
    --json-out logs/vla_lite/nca_teacher_o2_s0.json

# 推理期探针：置零 / 只换 plan 通道（zero | plan-zero | plan-gt | plan-random）
CUDA_VISIBLE_DEVICES=0 python scripts/eval_vla_lite.py \
    --ckpt ckpts/vla_lite/nca_s0/last.pt --field-override zero \
    --n-episodes 200 --seed 777 \
    --json-out logs/vla_lite/nca_teacher_zero_s0.json

# 5) 统计与图
python scripts/analyze_vla_lite.py logs/vla_lite/*.json > assets/vla_lite/analysis.txt
python scripts/plot_vla_lite.py --logs logs/vla_lite --out assets/vla_lite
```

### 10.3 单测清单

- `xy_to_cell`/`cell_to_xy` 往返 ≤1 cell（G32/G64）。
- 适配器输出尺寸、障碍 logits 尺寸、NCA 状态尺寸、readout token 数。
- 同 ckpt 两次评估逐场景一致。
- det-noise 在换臂时保持一致。
- 场景指纹切分断言。
- NCA 零初始化时输出恒为 0（恒等残差）。
- K=1 与 K=12 同一权重可运行；G32/G64 同一权重可运行。
- flow matching 动作形状/归一化往返。

---

## 11. 开放决策

以下问题在执行前需要拍板；括号内是默认建议。

1. **视觉塔**：SigLIP-B/16（默认，便于与 SmolVLA 对照）vs DINOv2-S
   （可能更适合空间，但失去对照）。建议先 SigLIP，DINOv2 作为诊断。
2. **语言槽**：learned task token（已实现，默认）vs 冻结 MiniLM（待离线
   权重）。当前指令恒定，两者预计无差；主臂用 learned，MiniLM 作为消融。
3. **高层动作空间**：终端 visit command（默认，与 E0 同语义）vs
   对象/目标离散 ID。建议主臂 command，ID 版作为可解释性消融。
4. **数据**：再生成（默认，含 split/ID/32×32 场）vs 修补现有 npz。
   建议再生成，避免隐式泄漏。
5. **Stage C**：默认不做，只有 A/B 都过且端到端明显受 exposure bias 影响
   时才做。
6. **OOD 障碍生成器**：4–8 道矩形墙（默认）vs 随机物体障碍；建议墙段，
   与训练物理一致。
7. **seed 数**：3（默认）vs 5。先 3；若 G2 处于 5–10 pp 边界，再补 2 seed。
8. **checkpoint 选择**：固定 20K 步（默认）vs val 闭环选择。建议固定，
   避免把评估信号带进训练。

---

## 12. 与 E0 的差异清单（防止重复踩坑）

| E0 的问题 | VLA-lite 的对应设计 | 验证方式 |
|---|---|---|
| 冻结 VLM 不会读场 | 读取端（适配器/NCA/解码器）全部可训练 | 置零探针、G1 |
| field token 插入 prefix、RoPE 偏移 | 无 LLM prefix；独立网格 readout | 结构审查 + 单测 |
| 4 通道场有损 | 32 通道 NCA 状态 + 学到的视觉特征 | 场可视化、plan MSE |
| 场只进低层 | NCA-low / NCA-both 两臂 | G3c |
| 高层冻结且是瓶颈 | 高层重训，可读 NCA | teacher vs predicted |
| 场来源与注入实现混淆 | 七臂只改场生成器 | 参数/算力匹配表 |
| 离线 MSE 与闭环脱节 | 固定步数；闭环为主；MSE 只诊断 | G0–G3 |
| Euclid 可能已足够 | Euclid 必选对照 | G2a |
| 评估噪声/交换解释不清 | det-noise + swap + teacher | 诊断矩阵 |
| 场景切分/泄漏未显式 | 场景级 split + 指纹断言 | G0 |

---

## 附录 A：张量形状

| 张量 | 形状 | 说明 |
|---|---|---|
| image | `[B,3,256,256]` uint8 | 原始渲染（即 SigLIP 输入，无 resize） |
| patch_tokens | `[B,256,768]` | 16×16 |
| adapter_feat | `[B,64,32,32]` | 视觉外部输入 |
| obstacle_logits | `[B,1,32,32]` | BCE 监督 |
| nca_state | `[B,32,32,32]` | plan + latent |
| nca_ext | `[B,77,32,32]` | 障碍/目标/起点/视觉/任务 |
| plan | `[B,32,32]` | 通道 0 |
| readout_tokens | `[B,256,256]` | 16×16×256 展平 |
| decoder_tokens | `[B,260,256]` | 256 网格 + state/lang/time |
| action_chunk | `[B,8,7]` | 绝对目标 |
| visit_cmd | `[B,4]` | `[gx,gy,gz,open]` |

## 附录 B：判据计算与报告口径

- **成功率**：`success = placed==3`；不做“部分成功”折算。
- **配对差**：`Δ_pp = 100 * (mean(NCA) − mean(baseline))`，逐场景配对；
  同时报告不一致对 `(+NCA-only, −baseline-only)`。
- **McNemar**：精确二项检验；预注册方向用单侧，探索用双侧。
- **Bootstrap**：按场景有放回重采样 10,000 次，报 2.5/97.5 分位。
- **seed 合并**：先每个 seed 算不一致对，再合并计数；不先平均成功率。
- **多重比较**：三个主对比 Holm；其余标“次级/描述性”。
- **报告所有结果**：包括失败 seed、崩溃 run、OOD 反向；不删异常点。
- **效应量**：主报告 pp + CI；同时给 placed count 和 visit 成功率，避免
  “全成功”二值化掩盖部分进展。
- **图上标注**：n、seed 数、CI、判据线（+10 pp / ±5 pp）、数据/commit hash。

---

## 附录 C：一页执行清单

- [ ] 再生成低层/高层数据，写 split/ID/32×32 场；跑指纹断言。
- [ ] 预计算 SigLIP 特征；检查磁盘 ~38 GB。
- [ ] 实现 `soma/vla_lite/{vision,adapter,nca,decoder,language,high_level,policy,baselines}.py`。
- [ ] 实现 `train_vla_lite_low.py`，先跑 no-field/GT/NCA 三臂 pilot。
- [ ] G0：确定性、IoU、GT−no-field ≥10 pp、no-field 与 E0 基线 ±5 pp。
- [ ] G1：NCA−no-field ≥10 pp；置零掉 ≥5 pp。
- [ ] 七臂 × 3 seed；分析主矩阵。
- [ ] Stage B 三臂；端到端四组合。
- [ ] O1/O2/O3；swap/探针。
- [ ] 统计、图、`reports/VLA_LITE.md`；提交代码与资产（ckpt 不入库）。
- [ ] 按 C0–C4 写结论；未过判据的部分明确写“未证明”，不升格。

---

## 13. 设计修订（2026-10-30，执行前）

跑 pilot 之前的设计审查发现以下问题会让 G2 结论无法解读，已在代码中修复：

| # | 问题 | 后果 | 修复 |
|---|---|---|---|
| 1 | CNN/GT/Euclid 臂只给解码器 1 个 plan 通道，NCA 臂给 32 通道（含视觉 latent） | NCA>CNN 可能只是带宽差，重演方法 1/E0 的归因问题 | `CNNField` 输出 32 通道状态；新增 `--readout plan`（所有臂只暴露通道 0），主比较两种口径都报 |
| 2 | NCA K=12，与 12 层 CNN 感受野相同 | 去掉了 R9 中 NCA 优势的来源（深迭代到不动点）；G32 绕行传播不完、G64 必败，G2b 失败会被误读为“NCA 结构非必要” | 训练 random-K∈[16,64]，推理 K=64，评估 K∝G；`--fp-weight` 可选；固定 K=12 保留为消融 |
| 3 | zero-field 臂的 NCA 参数不在优化器里 | zero 臂不是“同样训练、只隐藏场”的对照 | NCA 加入 zero 臂优化器并施加相同 plan 监督 |
| 4 | vis 臂的 1×1 投影不在优化器里 | vis 带宽对照的投影层停留在随机初始化 | 加入优化器 |
| 5 | 评估脚本无 `--grid`/K 覆盖/推理期探针/O2 场景 | O1/O2/§7.3 无法执行（`--scene-type random` 是单墙，比训练更简单） | `--grid`、`--k-steps`/`--no-k-scale`、`--field-override {zero,plan-zero,plan-gt,plan-random}`、`--scene-type multi-wall --n-extra-walls LO HI`（`scripts/vla_lite_ood.py`） |

**修订后的 Stage A 主矩阵**：原七臂 + `cnn --cnn-layers 32`（全局感受野对照）；
NCA/CNN/GT/Euclid 四臂在 `--readout plan` 下各补一组。预算优先级：先 3 臂 pilot
（none / gt / nca，full），G0 过后再展开。

**O2 口径更新**：训练场景 = 两道带缝墙（4 个墙矩形）；O2 = 同样两墙基底 + 2–6 段
额外墙段（共 6–10 个矩形），零样本。

**新增单测**（`scripts/test_vla_lite.py`）：NCA/CNN 状态同宽、`plan` 口径只保留
通道 0、K 覆盖真实改变输出、G64 前向、四种推理期 override、O2 场景可达。

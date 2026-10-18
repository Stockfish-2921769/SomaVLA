# SomaVLA — 把 NCA 规划场注入 VLA 动作专家：成本与可行性评估

> 日期：2026-10-18
> 状态：方案评估（尚未做注入实验）
> 关联：`EXPERIMENT_LOG.md` 轴 F（规划场闭环归因审计）、`EXPERIMENT_SUMMARY.md` R10
>
> **结论先行**：可行，且是低侵入改动。flow matching 的条件接口本来就是开放的——规划场
> 作为 denoiser 的额外 context（prefix token / KV / FiLM）进入，不需要改 ODE 求解器。
> 真正的成本不在 NCA 推理，而在 VLA 微调、坐标对齐、数据管线和参数匹配的对照设计。

---

## 0. 结论摘要

- **方法 2 的定义**：保留 SmolVLA 的 action expert 和低层执行器，把 64×64 规划场作为
  条件信号注入其 flow-matching denoiser；场不替换执行器，也不直接产生动作。
- **为什么便宜**：SmolVLA 每个 action chunk 只做一次 prefix VLM 前向并填 KV cache，
  随后 10 步 Euler 去噪全部复用；往 prefix 增加 64 个场 token 的边际算力很小。
  默认 `train_expert_only=True` 时只需训练 ~0.1B 的 expert 和新增的场编码器。
- **成本量级**（本仓库口径，2×RTX 4090）：
  - NCA/场推理：单次约 $10^{10}$ FLOP，毫秒到十毫秒级，按 chunk/visit 调用可忽略。
  - 场编码器：0.1–0.3M 参数。
  - SmolVLA：工程 1–3 天，单卡 expert-only 训练数小时/轮，端到端 **3–5 天**可完成
    E0/E1 两个实验。
  - π0 / π0.5：1–2 周；GR00T N1.x：2–3 周；OpenVLA（自回归离散动作）不匹配。
- **必须带的对照**：同一编码器下分别注入 **GT 场 / NCA 场 / Euclid 场 / random 场 /
  zero 场**，加 field dropout 和推理时场交换测试。没有 Euclid 对照，方法 2 会重演
  轴 F 的归因问题：增益可能仍然来自"单调势场 + 逃逸 readout"，而不是 NCA。
- **首要风险**：模态坍缩（模型忽略场）、坐标系错位（工作空间场 vs 相机图）、场质量
  （墙边符号违例）、以及 oracle occupancy 在真机上不可得。

---

## 1. 背景：为什么需要方法 2

轴 F 已把方法 1（场路由执行器）的贡献拆开（n=200，同一批场景，配对检验）：

| 臂 | 全成功 | vs 直线 |
|---|---|---|
| 直线执行 | 146/200 (73%) | — |
| NCA-long 场 | 178/200 (89%) | +34/−2，p=1.9e-8 |
| Pointwise MLP 场 | 180/200 (90%) | +36/−2，p=5.4e-9 |
| **Euclid 场（零参数）** | **179/200 (90%)** | +34/−1，p=2.1e-9 |
| CNN-long 场 | 143/200 (72%) | +6/−9，p=.61 |
| GT 测地场 | 200/200 (100%) | +54/−0 |

**结论**：73%→90% 的提升真实且显著，但归因于"单调势场 + 局部逃逸 readout"，
不归因于 NCA；NCA 与零参数 Euclid 场统计上不可区分（3 胜/4 负，p=1.0）。学习场
墙边符号违例 2.2%（94% 在距障碍 ≤1 格），尚未达到 GT 场的天花板。

方法 2 的意义不是再证明一次"场能改善闭环"，而是回答：**当规划场以条件信号进入
VLA 的动作生成过程时，它是否提供了动作专家单独学不到的信息，以及这部分信息是否
必须由 NCA 提供。** 这要求保留 action expert、保留端到端可微路径，并且把场质量与
注入通道分开评估。

---

## 2. 为什么注入不改流匹配本身

### 2.1 条件接口

流匹配学习的是速度场

$$
v_\theta(x_t, t, c), \qquad x_{t-\Delta t} = x_t - \Delta t \, v_\theta(x_t, t, c),
$$

其中 $c$ 是任意条件上下文。把规划场 $\phi$ 加进去只是把 $c$ 扩展为 $(c_{\text{img}},
c_{\text{lang}}, c_{\text{state}}, c_{\phi})$；**生成过程（Euler/Heun 求解器、
去噪步数、噪声调度）完全不变**。因此改动集中在 denoiser 的 conditioning 路径，
而不是流匹配算法本身。这也是为什么"注入成本"远低于"改扩散/流匹配结构"。

### 2.2 SmolVLA 的代码事实

在 `lerobot/policies/smolvla/` 当前版本中确认：

- `modeling_smolvla.py:534` `embed_prefix(images, img_masks, lang_tokens,
  lang_masks, state)`：把图像（SigLIP）、语言 token、state 投影拼成 prefix。
- `modeling_smolvla.py:718` `sample_actions`：**每个 action chunk 只做一次** VLM
  前向，`fill_kv_cache=True` 填充 prefix KV。
- `modeling_smolvla.py:750` `denoise_step`：10 步 Euler 去噪复用同一份 KV cache，
  每步只前向 expert 的 suffix（噪声动作 + 时间）。
- `configuration_smolvla.py:31-91`：`chunk_size=50`、`n_action_steps=50`、
  `num_steps=10`、`attention_mode="cross_attn"`、`train_expert_only=True`、
  `freeze_vision_encoder=True`。

两点直接推论：

1. **场 token 的算力是摊薄的**：新增 64 个 prefix token 只影响每 chunk 一次 KV 填充，
   10 步去噪不重复付费；端到端策略延迟的增量通常在几个百分点内。
2. **训练便宜**：默认只训练 expert（~0.1B）与新增编码器；冻结的 VLM 仍可把梯度
   反传到场编码器（权重不更新，但计算图是通的），因此场编码器可以端到端训练。

---

## 3. 注入方案对比

| 方式 | 改哪里 | 工程量 | 每 chunk 额外算力 | 信息保留 | 主要风险 |
|---|---|---|---|---|---|
| **场 token 进 prefix** | `embed_prefix` 末尾拼 8×8=64 token + mask | 1–2 天 | 一次 +64 token VLM 前向（KV 复用） | 高 | 冻结 VLM 未必学会使用新 token |
| 场 token 进 KV（adapter） | expert cross-attn 的 KV 尾部拼接 | 2–3 天 | 同上，但绕过 VLM | 高 | 改 attention 拼接与 mask，易错 |
| FiLM / adaLN | 场→全局向量→expert 各层 scale/shift | ~1 天 | ≈0 | 低–中 | 64×64 几何压成向量，墙边细节丢失 |
| 额外视觉通道 | 场图作为新 view/通道进图像分支 | 1–2 天 | +一次小 CNN/vision 前向 | 高 | 需改预处理；SigLIP 固定 3 通道，实际仍接小 CNN |
| 路点/场梯度 token | `embed_prefix` 追加 1–3 个 token | 0.5 天 | ≈0 | 低 | 退化成方法 1 的子目标 |
| 训练-free guidance | 修改 denoise 的 velocity | 0.5 天 | 小 | 中 | 场是势不是速度方向，容易帮倒忙 |

### 3.1 场 token 进 prefix（推荐主方案）

- 场编码器：3 层小 CNN（如 32/64/128 通道）+ 8×8 池化，输出 64 个 token，线性投影
  到 VLM hidden；参数量约 0.1–0.3M。
- 在 `embed_prefix` 返回前拼接：`embs = [img, lang, field, state]`，
  `pad_masks = [img_mask, lang_mask, ones(64), state_mask]`，
  prefix 的 `att_masks` 保持 0（SmolVLA 约定 prefix=0 / suffix=1）。
- `position_ids` 由 `cumsum(pad_masks)-1` 自动顺延，无需额外改动。
- 优点：改动集中、KV 只填一次、场的空间结构以 token 形式保留。
- 风险：冻结 VLM 没有见过这类 token，可能把它们当噪声；若 E0 显示通道无效，再换
  KV adapter 或解冻 VLM 最后 2–4 层。

### 3.2 KV adapter（次选）

在场编码器输出后，将其作为额外的 K/V 直接拼到每个 cross-attention 层的 prefix KV
尾部，绕过冻结 VLM。信息保留最好，且不改变 VLM 的 token 序列；代价是要改
`forward_cross_attn_layer` 的 KV 拼接与 attention mask，工程量和出错概率都更高。
适合在 3.1 已证明"场有用但 VLM 不消费"之后再做。

### 3.3 FiLM / adaLN

场经全局池化/注意力池化成一个向量，对 expert 每层的 hidden 做 scale/shift。参数最少、
训练最便宜，但 64×64 的墙边几何被压进一个向量，恰好丢掉了当前学习场唯一可能带来
差异的局部排序信息。建议作为消融，不作为主方案。

### 3.4 额外视觉通道

把场渲染成图像，作为额外 view 或额外通道送入视觉分支。实现直观、复用现成视觉编码器，
但 SigLIP 的通道数固定，实际上仍需要一个独立的小 CNN 或复制/改输入层；信息保留与
3.1 等价，工程上反而更绕。若未来要接真机相机，可以把场叠加到相机图上，这条路线
会更自然。

### 3.5 路点/场梯度 token

从场中抽取下一步 waypoint 或局部下降方向，作为 1–3 个 state-like token。成本最低，
但它把"场作为输入"退化成了"场作为 planner 输出"，无法检验场是否给 action expert
提供了新信息。只适合作为 sanity check。

### 3.6 训练-free guidance（不推荐）

在场下降方向上给 velocity 加偏置。场是势函数，不是速度场；势的负梯度方向与动作
chunk 的速度方向没有天然对应，guidance 尺度难以标定，且无法做干净的归因。不建议
作为主路线。

---

## 4. 成本分项

### 4.1 NCA / 场推理

当前 NCA-long：hidden 128、morph 12、msg 48、27,501 参数、K=64、G=64。

- 每 cell 每步约 $5.4\times10^4$ FLOP（perception + update MLP）；
- 全网格每步约 $2.2\times10^8$ FLOP；K=64 约 **$1.4\times10^{10}$ FLOP/场**；
- 4090 上小 kernel 利用率低，实测预计 **毫秒到十毫秒级**；
- 每个 chunk 或每个 visit 调用一次即可（墙与目标在 chunk 内不变），可忽略。
- 若 VLA 输入只需 32×32 场，成本再降 4×。

**结论：NCA 不是成本项。**

### 4.2 场编码器与注入

- 小 CNN + 投影：0.1–0.3M 参数；前向 < 1 ms。
- prefix 增加 64 token：约 +10–30% prefix 长度，但只在每 chunk 填 KV 时计算一次。
- 训练时显存增量主要来自新增 token 的 attention，预计 < 1–2 GB（取决于 batch）。

### 4.3 训练

| 配置 | 可训练参数 | 硬件 | 估计 |
|---|---|---|---|
| expert-only + 场编码器（默认） | ~0.1B + 0.3M | 1×4090 24GB | 数小时/轮 |
| expert-only + LoRA on VLM | +5–10M | 1×4090 | 数小时–1 天/轮 |
| 解冻 VLM 最后数层 | +数十 M | 2–3×4090 | 1–3 天/轮 |
| 全量 SmolVLA | ~0.45B | 2–3×4090 | 2–5 天/轮 |

数据量、序列长度和是否用梯度检查点会显著改变上述估计；这里给的是量级，不是承诺。

### 4.4 数据

- **仿真内**：规划场可由真值 occupancy + 目标直接生成，采集成本为零；只需在现有
  BC 数据里增加一路 `field[64,64]`（或在训练时按场景参数实时生成），数据体积约
  每样本 +16 KB（float32）或 +4 KB（fp16/uint8 量化）。
- **真机/视觉输入**：场必须从感知或已知场景图生成，这是 P1 的工作，不属于本次
  注入工程。不要把 oracle occupancy 当默认假设写进方法叙事。
- 需要保证场与动作 chunk 的时间对齐：场按决策点生成，chunk 内冻结。

### 4.5 评估

- 复用现有 200 episode 闭环 harness + OOD 墙布局；新增的只是"注入场"这一条件。
- 关键成本是**对照臂的实验轮数**：至少 5 个场来源（GT/NCA/Euclid/random/zero）
  × 训练/评估，每次数小时；建议先固定同一编码器与同一训练预算。
- 还应加两项行为学诊断：
  1. **场交换**：推理时把 GT 场换成 random 场，看动作是否显著改变（测依赖度）。
  2. **field dropout**：训练时以 10–20% 概率置零/替换场，抑制模态坍缩。

---

## 5. 分模型可行性

| 模型 | 规划/动作结构 | 可行性 | 主要成本 |
|---|---|---|---|
| **SmolVLA** | SmolVLM2 + ~0.1B flow-matching expert，cross-attn，KV cache | **最高** | 1–3 天工程 + 单卡训练 |
| π0 / π0.5 | PaliGemma-3B + ~300M flow expert | 高 | 1–2 周；3B 微调/LoRA |
| GR00T N1.x | System 2 VLM + DiT flow matching（System 1） | 中 | 2–3 周；数据格式与算力 |
| Hy-Embodied-VLA | MoT 主干 + 双塔 flow-matching expert | 中 | 取决于开源完整度；1–2 周 |
| LingBot-VLA | Qwen2.5-VL + flow-matching action expert | 中 | 同上；1–2 周 |
| OpenVLA | 自回归离散 action token（非流匹配） | 低 | 7B；要改 prompt/词表或换连续头，不推荐 |

说明：

- **SmolVLA**：代码结构、训练脚本、评估 harness 都在本仓库，改动面最小；且
  `train_expert_only` 让首轮实验可以在单卡完成。
- **π0/π0.5**：openpi 的 flow-matching expert 与 SmolVLA 同源（π0 是 SmolVLA 的
  直接参照），注入逻辑可复用；成本主要在 3B 主干的微调与数据格式转换。π0.5 的高层
  语义子任务预测可以作为"场→子目标"的天然接口，但那是方法 1 的变体。
- **GR00T**：DiT 与 System 2 分离清晰，理论上适合作为场注入点；但仓库、数据格式
  （LeRobot/Isaac）和算力要求都更重，适合在 SmolVLA 验证后再迁移。
- **Hy-Embodied / LingBot**：结构上都是"VLM 主干 + flow expert"，可行性类似 π0；
  主要不确定性是开源权重/训练代码的完整度与复现成本，需先核对具体版本。
- **OpenVLA**：动作是 256-bin 离散 token 的自回归生成，没有 denoiser 条件路径；
  场只能通过 prompt 或额外连续头进入，改动大且与本文的条件注入假设不匹配。

---

## 6. 对齐与工程细节

### 6.1 时间节奏

- SmolVLA 默认 `chunk_size=50`、`n_action_steps=50`、`num_steps=10`；
  当前配置是**整 chunk 开环执行**（无 chunk 内 receding）。
- 场在每次策略调用时重算：障碍与目标在 chunk 内不变，场无需逐步更新。
- 若改为 receding horizon（`n_action_steps < chunk_size`），场仍按策略调用频率
  刷新；更新频率与子目标/场景变化频率一致即可，不必跟随 10 步去噪。
- 场编码与 KV 填充每个 chunk 一次；不要把场塞进 `denoise_step` 的每步前向。

### 6.2 坐标系

- 规划场定义在工作空间全局帧（`xy_to_cell` 的 64×64 网格）；SmolVLA 的视觉输入是
  顶视相机图，state 在机器人/末端帧。
- 推荐两种对齐方式：
  1. **目标相对 + 局部裁剪**：以 EEF 为中心裁 64×64 场，或把绝对场转换为
     "相对 EEF 的 dx,dy" 通道；
  2. **场 + 目标坐标**：场保持全局，同时把当前子目标 `(gx,gy)` 归一化后作为额外
     token 传入，让模型自己学习对齐。
- 不要假设相机外参已知；若未来接真机，场应转换到相机/机器人帧后再注入。

### 6.3 注入实现草图

```python
# modeling_smolvla.py, inside embed_prefix (sketch)
field_tokens = self.field_encoder(field)              # [B, 64, vlm_hidden]
embs = torch.cat([img_embs, lang_embs, field_tokens, state_emb], dim=1)
pad_masks = torch.cat([img_masks, lang_masks,
                       torch.ones(B, 64, device=...), state_mask], dim=1)
att_masks = torch.cat([img_att, lang_att,
                       torch.zeros(B, 64, device=...), state_att], dim=1)
```

- `att_masks` 的 0/1 约定沿用 SmolVLA：prefix=0，suffix=1。
- `position_ids` 由 `cumsum(pad_masks) - 1` 自动处理，无需手工编号。
- 若用 KV adapter，则在 `fill_kv_cache=True` 之后、`denoise_step` 之前，把场 token
  的 K/V 拼到每层 cache 尾部，并同步扩展 attention mask。

### 6.4 训练配方建议

1. **Stage 1**：冻结 VLM，训练场编码器 + expert（默认配置），先验证通道有效。
2. **Stage 2**（可选）：解冻 VLM 最后 2–4 层或加 LoRA，缓解冻结主干对新 token
   的分布外问题。
3. **正则**：field dropout 10–20%；对场编码器加轻微权重衰减；保持 expert 的学习率
   不高于原配方，避免遗忘预训练动作先验。
4. **监控**：除 action MSE 外，记录"场交换前后动作差异"和闭环成功率；只看 loss
   无法区分"学会用场"和"学会忽略场"。

---

## 7. 实验阶梯与对照设计

| 阶段 | 注入内容 | 目的 | 成功标准 |
|---|---|---|---|
| **E0** | GT 场 token（+ zero/random 对照） | 验证注入通道上限；确认不是接口问题 | GT 显著优于 zero/random |
| **E1** | NCA 场 vs Euclid 场 vs MLP 场（同编码器、同参数量） | 归因：增益是否必须来自 NCA | NCA 显著优于 Euclid；否则方法 2 的 NCA 主张不成立 |
| **E2** | field dropout + 推理时场交换 | 测依赖度、防模态坍缩 | 换场后动作/成功率显著变化 |
| **E3** | OOD 墙布局 / 部分观测 | 泛化与真实场景前提 | 在 OOD 上保持增益 |

**判定规则（预先写死，避免事后解释）**：

- 若 E0 中 GT 场不优于 zero 场 → 注入通道或训练配方有问题，先修通道，不读 NCA 结论。
- 若 E1 中 NCA 与 Euclid 统计不可区分 → 与方法 1 相同的结论：增益来自势场 + readout，
  不是 NCA；应转向"场必须依赖学习/感知"的任务（P1/P2）。
- 若 E2 中场交换不改变行为 → 模型忽略了场，E1 的"无差异"不能作为 NCA 无效的证据。
- 统计口径沿用轴 F：同一批场景、配对 McNemar、报告效应量与置信区间，不只报成功率。

---

## 8. 风险与失败模式

| 风险 | 表现 | 缓解 |
|---|---|---|
| 模态坍缩 | 模型忽略场，field swap 无影响 | field dropout；场交换测试；Stage 2 解冻 |
| 坐标系错位 | 动作朝错误方向偏置 | ego 帧/目标相对表示；可视化注意力与场叠加 |
| 场质量不足 | 墙边违例沿路径放大 | 先用 GT 场做上限；把违例 <0.2% 作为上场门槛 |
| oracle occupancy | 仿真有效、真机不可得 | 明确 P1 感知路线；不要把 oracle 写进主张 |
| 注意力稀释 | 64 个 token 淹没语言/图像 | 减少到 16–32 token；adapter；FiLM 消融 |
| 灾难性遗忘 | 原有操作能力下降 | 低学习率、LoRA、expert-only 首轮 |
| 训练/推理分布偏移 | 闭环状态超出数据集 | 现有 harness 闭环评估；OOD 臂 |

---

## 9. 建议路径

1. **先做 E0（1–2 天）**：GT field token 注入 + zero/random 对照。这是最便宜、
   信息量最大的一步；如果通道本身不通，后面所有 NCA 实验都没有意义。
2. **再做 E1（2–3 天）**：同一编码器下 NCA vs Euclid vs MLP，参数匹配、预算匹配。
3. **只有 E1 通过**（NCA 显著优于 Euclid）才值得投入 Stage 2、解冻 VLM、OOD 和
   更大模型迁移；否则按轴 F 的结论转 P1（感知条件场）或 P2（决策瓶颈任务）。
4. 首次实现建议选 **场 token 进 prefix**；KV adapter 与 FiLM 分别作为"信息量上限"
   和"参数下限"的消融，而不是并行铺开。

**一句话**：在 SmolVLA 上把 NCA 场注入 action expert 是 3–5 天量级的工作；瓶颈不是
NCA 也不是流匹配，而是微调数据、坐标对齐和参数匹配的对照设计。E0 必须先证明
"场条件通道"本身有效，E1 才能回答"NCA 是否提供了不可替代的信息"。

---

## 附录 A：代码锚点

- `lerobot/policies/smolvla/modeling_smolvla.py:534` — `embed_prefix`
- `lerobot/policies/smolvla/modeling_smolvla.py:718` — `sample_actions`（每 chunk 填 KV）
- `lerobot/policies/smolvla/modeling_smolvla.py:750` — `denoise_step`（10 步复用 KV）
- `lerobot/policies/smolvla/configuration_smolvla.py:31-91` — chunk/denoise/train flags
- `soma/spatial_nca.py` — 空间 NCA（G×G 同质细胞，`grid_neighbors`，`mask_obstacle_msgs`）
- `soma/spatial_baselines.py` — CNN / Transformer / MLP 基线与 `predict_field`
- `scripts/multi_plan_data.py` — 场数据、`field_path` / `field_path_escape` / `field_path_cost`
- `scripts/eval_multi.py` — 闭环 harness 与 Euclid 零参数对照

## 附录 B：估算口径

- NCA FLOP：perception $(13{\to}48)$ 约 $1.2$ KFLOP/cell/step；update
  $(65{\to}128{\to}128{\to}13)$ 约 $52.7$ KFLOP/cell/step；合计约
  $5.4\times10^4$ FLOP/cell/step。G=64 → 4096 cells → $2.2\times10^8$ FLOP/step；
  K=64 → $1.4\times10^{10}$ FLOP/场。未计 gather/索引开销。
- 训练时间按 1×RTX 4090 24GB、bf16、batch 视数据量调整；实际以 profiler 为准。
- 所有成本数字是工程估计，不是实测承诺；E0 落地后应回填实测延迟与显存。

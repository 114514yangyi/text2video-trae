# MILESTONE-1 — PixArt-alpha 新骨架与十臂条件注入矩阵

> 状态：**规格阶段，未实现**（2026-10-22 立，用户指令 m00002）
> 分支：`milestone1-pixart`（从 `main`/`bdf36d1` 切出，已做代码清场）
> 定位：本文件是 milestone-1 的**唯一验收依据**；实现开工前，第 11 节开放问题必须先裁决。

---

## 0. 一句话

把项目从头做法重置为一个**全新项目**：backbone 换成 **PixArt-α** 骨架、用 **x-prediction**、
全局条件 `c` 只喂最简单的 timestep，唯一要扫的东西是**条件怎么进模型**（文本 / 参考帧
frame-0 往哪层、以什么形式注入）以及四个配套因子（motion loss、文本编码器深度与可训练性、
预训练权重），共 10 个实验臂。

---

## 1. 决策摘要

| 项 | 决策 | 来源 |
| --- | --- | --- |
| 评测指标 | **官方标准 FVD 代码与模型**（I3D Inception-v1 Kinetics-400，400-d logits），**24 帧模式** | 用户 m00002 基准 1 |
| 全局条件 `c` | **不叠加任何额外条件**，只保留普通 DiT 里最简单的 **timestep** 嵌入 | 用户 m00002 基准 2 |
| backbone | **PixArt-α**（DiT-XL/2 骨架），目的是能用其预训练权重初始化部分层 | 用户 m00002 基准 3 |
| 扩散目标 | **x-prediction**（预测 x0） | 用户 m00002 基准 4；与 RULES 方法硬约束 3 一致 |
| base 模型 | **1a + 4c + 2a + 3b + 5b** | 用户 m00002 |
| 实验数 | 10 臂（E1–E10），全部相对 base 单因子改动，E9 为双因子交互探针 | 用户 m00002 |
| 旧代码 | 已删除并隔离（208 → 45 个文件），本分支当新项目 | 用户 m00002；提交 `88945f0` |
| 训练/评测 | 本文件只定规格，**不实现、不训练** | 用户 m00002 末句 |
| **latent 空间** | **保持 1024 维 1-token 契约**（TiTok-CVAE 不变；改 PixArt 的 patchify/unpatchify 去适配 latent） | 用户 2026-10-22 裁决（原 Q1） |
| **context 注入语义** | 1b 的 context = **条件 token 拼进 self-attn 序列当前缀**（不走 cross-attn） | 用户 2026-10-22 裁决（原 Q2） |
| **24 帧口径** | **从头抽取 24 帧**（取 frame 0 起连续 24 帧；官方 I3D 模型与代码不变；不足 24 帧的样本跳过并落盘 id 清单） | 用户 2026-10-22 裁决（原 Q5，第二轮修订） |
| **长度** | **本里程碑不做长度预测**；评测时直接用 **GT 视频长度** | 用户 2026-10-22 裁决（原 Q6） |
| **文本→骨架桥接** | mT5 768 → 骨架宽度的投影层 **可训练**（Linear + LayerNorm） | 用户 2026-10-22 裁决（Q3） |
| **4c 的训练范围** | **embedding + encoder.layer[0:3]** 全部可训练（"前 3 层"**含 embedding**） | 用户 2026-10-22 裁决（Q7） |
| **骨架超参** | **PixArt-α 原尺寸**：28 层 / 1152 宽 / 16 head（便于 5a 逐层移植） | 用户 2026-10-22 裁决（Q8） |
| **motion loss 定义** | **沿用** `MultiScaleDeltaLoss`（k∈{1,2,4} 多尺度差分 + warmup） | 用户 2026-10-22 裁决（Q9） |
| **2a/2c 层分配（Q4）** | **间隔放置**：每 4 个 block 为一组，前 3 个 text、第 4 个 ref ⇒ 28 层 = `[T,T,T,R] × 7`，ref 落在 layer **3/7/11/15/19/23/27** | 用户 2026-10-22 裁决（Q4） |

---

## 2. 基准设定（硬设定，10 个臂全部共用）

### 2.1 评测：官方 FVD + 24 帧

- **只用官方 FVD**：I3D Inception-v1（Kinetics-400）的 **400 维 logits** 作为特征，
  short side 224 + center crop 224、像素区间 `[-1,1]`、float64 Fréchet 距离。
  实现走 `tools/eval_fvd.py`（已从 `8454407` 移植到本分支），口径文档
  `FVD_SPEC_20261021.md`。
- **24 帧模式 = 从头抽取 24 帧（2026-10-22 用户裁决，第二轮修订）**：官方 I3D 模型与代码
  不变，取帧规则改为**每条视频从第一帧起取连续 24 帧**（frame 0 → frame 23），
  对应 `tools/eval_fvd.py` 的 `--frame-mode contiguous --num-frames 24`。
  - 与历史 **16 帧均匀** 口径**不可混表**：帧数不同、取帧位置也不同，特征统计不可比。
  - **不要沿用 `--resize-hw 272 224`**（那是旧预处理口径）；缩放仍按官方
    224 口径（short side 224 + center crop 224），除非有明确理由并在文档写清。
  - **不足 24 帧的样本跳过**：这会把评测集从 640 条压到 **~613 条**（旧 24 帧口径
    实测就是 613 条）。因此报表必须写**有效 N**，并把**有效 id 清单落盘复用**，
    10 个臂用**同一份**清单——否则臂间不是同一批样本，比较无效。
    若之后想恢复 N=640，只能改成"末帧补齐到 24 帧"并**重跑全部臂**，两种策略禁止混用。
  - 生成侧帧数 = GT 长度（见 §2.2）；评测时**只取生成视频的前 24 帧**。所以
    "GT 长度 < 24" 与 "生成是否够长" 两件事都要在采样脚本里断言，不够长就报错不要静默补。
  - 真值侧缓存 key 已带帧数（`f<N>`），改口径后 **`--self-check` 与真值本底都要重跑**，
    并把新的本底数值写进本文件（原 16 帧本底 60–75 不再适用）。
  - 10 个臂必须**完全同一口径**，否则表内比较无效。
- 只在 **test split 单侧**报告，采样清单落盘复用（`reports/eval/samples/test_n640_seed*.txt`），
  同采样器 / 同 CFG / 同步数 / 同长度规则 / 同种子。
- 改过 FVD 代码必须跑 `--self-check`（官方权重自检应约 131，判据 100–180），并看
  `real_split_half_fvd` 本底（N≈640 时通常 60–75）；模型分数低于本底要先怀疑管线。
  改了取帧逻辑后 `--self-check` 与真值本底**都要重新确认**。

### 2.2 全局条件 `c`：只有 timestep

普通 DiT 的 adaLN 条件向量是 `c = t_emb + y_emb`（timestep + 类/其他全局条件）。
本里程碑**去掉 `y` 那一半**：`c = t_emb` 一项，不拼长度、不拼池化文本、不拼池化参考帧。

含义要讲清楚，因为它决定了整个设计的骨架：

- **adaLN 只调制"这一步是第几步噪声"**，不携带任何内容信息；
- 内容信息（文本、参考帧）**只能**从注意力通路进模型：要么 cross-attn 的 K/V，
  要么拼进 self-attn 的序列里当 token。这正是因子 1 要扫的东西。
- 副作用：**生成长度没有来源了**。旧项目用"文本→长度 MLP 进 c"提供长度；现在 c 只剩
  timestep，**本里程碑干脆不做长度预测**（2026-10-22 用户裁决）：训练时用样本自身长度，
  评测/生成时直接给 **GT 视频长度**。因此：
  - `t2v/models/length_model.py` 在 milestone-1 **不进入训练图**（保留文件只为将来部署）；
  - 所有臂的长度输入完全一致，**长度不会成为臂间差异的来源**；
  - 报表里要写明"帧数用 GT 长度"，否则读者会以为模型自己决定长度。

### 2.3 backbone：PixArt-α

按 PixArt-α 公开实现，其骨架是 **DiT-XL/2 变体**：patchify + N 个 transformer block
（每块 self-attn + **cross-attn** + MLP，adaLN 调制）+ 最终 adaLN + unpatchify；
文本侧原本用 T5-xxl（4096 维、120 token）经 cross-attn 注入。

换到这个项目要处理三处不适配（这是 M1.1/M1.2 的主要工作量，不是"换一个类名"那么简单）：

1. **latent 空间——已裁决：保持 1024 维 1-token**（2026-10-22 用户）。PixArt-α 原本吃
   SD-VAE 的 4 通道 latent，本项目改成吃冻结 TiTok-CVAE 的 **每帧 1 个 1024 维 token**。
   具体含义：
   - latent 序列 = `[B, T, 1024]`，**每帧就是一个 token**，所以 **patchify 被替换成
     "每帧过一个 Linear(1024→D)"**，`patch_size / 位置编码 / unpatchify` 全部按项目契约重写，
     不存在"2×2 patch 拼回 4 通道"的路径；
   - 参考帧仍以 `ref_features [128,512]` 形态进来（走条件侧，不占 latent 序列）；
   - **`patch_embed` / 位置编码 / `proj_out` 无法从 PixArt-α 预训练权重移植**，
     因子 5（5a）能灌的是 block 内部：self-attn 的 qkv/proj、MLP、adaLN 调制线性层、
     cross-attn 的 qkv/proj。**灌了哪些层必须逐项写进 config**，否则 5a/5b 对照不成立。
   - 好处：冻结 VAE、`data/phoenix_latents/v1` 缓存、`tests/test_data.py`、
     `tests/test_frozen_guard.py` 全部原样有效，不需要重建任何 latent。
2. **文本维度**：本项目文本编码器是 mT5-base（768 维），PixArt 的 cross-attn 期望 4096 维
   T5-xxl。需要一层投影（768→D），并且投影是否可训练要写清（Q3）。
3. **条件 token 形状**：文本 120 token + 参考帧 128 token，谁在前谁在后、mask 怎么给、
   哪些层看见哪些，全部要在配置里显式表达（因子 2 就是这件事）。

### 2.4 扩散目标：x-prediction

预测 x0（不是 ε，不是 velocity）。`t2v/diffusion/gaussian.py` 已原生支持 x0 目标、
cosine/linear 调度、DDIM/DDPM/DPM-Solver++，可以直接复用。
`dynamic_loss`（因子 3 的 motion loss）也要求 x0 目标，与该设定一致。

### 2.5 与旧项目的隔离

本分支当作新项目对待，不复用旧结论、不复用旧架构代码：

- 已删除：旧 DiT（`t2v/models/dit.py`）、`factory.py`、生成管线 `pipelines/generation.py`、
  训练入口 `t2v/train.py`、旧辅助损失与对齐模块（margin/repa/semantic/softalign/
  stage_connector/temporal_align/text_slots/latent_text）、39 个旧实验 config、
  91 个实验驱动/probe/回译/统计脚本、18 个旧架构测试。
- 只保留与新骨架无关、可原样复用的底座：冻结 VAE 适配器、mT5 文本适配器、
  latent 数据集与 collate、扩散数学、条件编码器骨架、EMA/ckpt/采样器等训练基建、
  冻结 VAE 解码器、官方 FVD 评测链。
- **旧的实验数字（FVD 22–26 的 R(2+1)D 口径、旧复现表、回译 BLEU-4）一律不进入
  milestone-1 任何表格**。新表从零开始。

---

## 3. 可变因子定义

### 因子 1 — 条件注入通路

| 取值 | 含义 | 怎么实现 |
| --- | --- | --- |
| **1a** | **只用 cross-attn** 注入条件 | 文本/参考帧的 token 序列只作为 cross-attn 的 K/V；self-attn 序列里只有 latent 帧 token |
| **1b** | **只用 context 注入** | 条件 token 直接**拼进 self-attn 序列**（前缀 token），**不设 cross-attn 层**；条件通过自注意力的全连接图影响每一帧。**（2026-10-22 用户确认该语义）** |
| **1c** | **两者同时** | 同一份条件既当 self-attn 前缀 token，又当 cross-attn 的 K/V（旧项目 `text_prefix` 就是这个语义） |

要点：1a 与 1b 的差别不是"同一个东西换个位置"，而是**信息路由方式不同**——
cross-attn 是"每帧主动查询条件"（Q 来自帧，K/V 来自条件），代价是每层多一组参数；
前缀 token 是"条件被动混进帧序列"（Q/K/V 都是同一序列），不新增参数但会拉长序列、
且条件与帧在同一注意力里互相竞争。1c 是两者叠加，参数量与序列长度都更大。

### 因子 2 — ref 与 text 在 cross-attn 里的分配方式

前提：**1a**（只有 cross-attn）；若某实验是 1b/1c，因子 2 定义的是 cross-attn 那一路怎么分配。

| 取值 | 含义 |
| --- | --- |
| **2a** | **分层注入**：把 cross-attn 层分成两组，一组只注入 ref、一组只注入 text；**text 的层数多于 ref** |
| **2b** | **拼接后注入所有层**：`[ref tokens ; text tokens]` 拼成一个 K/V 序列，喂给**每个** cross-attn 层 |
| **2c** | **分层注入，但 ref 那些层注入的是 [ref 拼 text]**：text 层只看见 text；ref 层看见 ref+text |

这三个不是同一条谱系上的等距点，各自回答不同问题（见第 6 节）。2a/2c 需要写死一张
**逐层注入表**（第几层看见什么）。

**Q4 已裁决：间隔放置，不是把 ref 堆到尾部。** 28 个 block 按 4 个一组切成 7 组，
每组**前 3 个 block 注入 text、第 4 个注入 ref**：

```
组 p = 0..6，block 下标 i = 4p+0 / 4p+1 / 4p+2 → text
              block 下标 i = 4p+3         → ref
⇒ ref 层 = {3, 7, 11, 15, 19, 23, 27}（7 层），text 层 = 其余 21 层
```

为什么是"间隔"而不是"后半段"：ref 与 text 的信息若全挤在相邻的少数层，等于把两种条件
放在同一个局部深度里竞争；每 4 层插一层 ref 让**条件信息在整条深度上反复回到表层**，
也让"某几层专门看参考帧"这件事不至于只有一次机会。代价是每一层的 K/V 来源不同，
实现上必须按层查表，不能默认"所有层共用一份 context"。

- **2a**：text 层 K/V = text tokens；ref 层 K/V = ref tokens。
- **2c**：text 层 K/V = text tokens；ref 层 K/V = **`[ref ; text]` 拼接**（ref 在前，text 在后）。
- **2b**：不看这张表，所有层都用 `[ref ; text]`。
- **1b**（无 cross-attn）：整张表不用，条件走 self-attn 前缀。

### 因子 3 — motion loss

| 取值 | 含义 |
| --- | --- |
| **3a** | **加** motion loss |
| **3b** | **不加** |

**Q9 已裁决：直接沿用保留下来的 `t2v/models/dynamic_loss.py`（`MultiScaleDeltaLoss`）**：
对 `pred_x0` 与真值 latent 做 k 步差分（k ∈ {1,2,4}）的 masked MSE，只在两端都是真帧的
位置计入。作用在 **latent 空间**，不是像素。权重与 warmup 要写成配置项并记录到
run 的 `config.yaml`（否则事后无法复现）；E4/E9 用的权重必须是同一个值。

### 因子 4 — 文本编码器

| 取值 | 含义 | 可训练参数 |
| --- | --- | --- |
| **4a** | **整个冻结的 mT5 encoder**（12 层，不更新） | 0（只有下游投影可训练） |
| **4b** | **可训练的 mT5**（12 层全量微调） | 全量 encoder |
| **4c** | **只取 mT5 前 3 层 encoder，且可训练** | **embedding + layer[0:3]**（Q7 已裁决：含 embedding） |

mT5-base：`d_model=768`、12 层 encoder、padding 到固定 120 token。
"只取前 3 层"= 深度截断，不是宽度截断：输出仍是 `[B,120,768]`，只是经过的层数变少
（容量下降、显存与时延下降）。base 用的是 **4c**，所以 base 的文本侧是**浅且可训练**的。

注意 4a 与 4b/4c 的对照要读对：文本侧参数从"完全冻结 + 只有投影可学"变成了
"encoder 自己也在学"，这会显著改变优化难度（文本侧学习率必须单独设，并写进配置）。

**Q3/Q7 已裁决的两条细节**：

- **文本→骨架的投影可训练**（`Linear(768→1152)` + LayerNorm），它不属于"文本编码器"这个
  因子，**10 个臂全部都有、全部可训练**，不构成臂间差异；
- **4c 的"前 3 层"含 embedding**：训练范围 = `embedding + encoder.layer[0:3]`，
  其余层（layer[3:12]）丢弃不用。若只训练 encoder 而冻结 embedding，梯度路径会断，
  文本侧学不动——这是选"含 embedding"的原因。

### 因子 5 — 预训练权重初始化

| 取值 | 含义 |
| --- | --- |
| **5a** | 用 **PixArt-α 预训练权重**初始化（能对上的层直接灌权重） |
| **5b** | **不用**预训练权重，随机初始化 |

base 是 **5b**（随机初始化）。Q1 已裁决保持 1024 维 1-token，因此 **5a 能移植的只有
transformer blocks 内部**：self-attn 的 qkv/out、cross-attn 的 qkv/out、MLP、
adaLN 调制线性层；而 `patch_embed`（原本把 4 通道 latent 切成 token）与 `proj_out`
（原本 unpatchify 回 4 通道）**形状不对，必须重新初始化**，位置编码同理。
**"灌了哪些层、跳过了哪些层"必须写进 run 的 config 与 EXPERIMENT 备注**，否则 5a/5b
对比不成立；E10 的结论也只能解释成"block 内部权重的初始化收益"。

---

## 4. Base 模型解剖

`base = 1a + 4c + 2a + 3b + 5b`，逐项展开成一份完整配置：

| 维度 | base 取值 | 说明 |
| --- | --- | --- |
| backbone | **PixArt-α 原尺寸**：28 个 block / 1152 宽 / 16 head（Q8 已裁决，写死进 config） | 见 2.3 |
| 扩散目标 | x0 | cosine 调度，v-pred 关闭 |
| 全局条件 c | **只有 timestep** | 无长度、无池化文本、无池化参考帧 |
| 条件注入通路 | **1a：只有 cross-attn** | self-attn 序列里**没有**前缀 token |
| ref/text 分配 | **2a：间隔分层注入**`[T,T,T,R]×7`；ref 层 = 3/7/11/15/19/23/27 | 逐层表见 §3 因子 2 |
| 文本编码器 | **4c：mT5 前 3 层，可训练** | 输出 `[B,120,768]` → 投影到骨架宽度 |
| motion loss | **3b：不加** | 只用扩散主损失 |
| 预训练初始化 | **5b：不用** | 全随机初始化 |

base 的训练成本是最贵的组合之一：随机初始化（5b）+ 全层可训练，没有任何预训练先验，
必须从零学出 latent 分布。这也是 E10（切 5a）值得单独做一个臂的原因。

---

## 5. 实验矩阵（E1–E10）

所有臂与 base 的差别**只有一列**（E9 例外，是两列，用于探交互）。

| # | 名称 | 配置 | 相对 base 的唯一改动 | 想回答什么 |
| --- | --- | --- | --- | --- |
| E1 | **base** | 1a+4c+2a+3b+5b | —（基准） | 这套组合本身就是最强吗 |
| E2 | **base-2b** | 1a+4c+**2b**+3b+5b | cross-attn 改成 ref+text 拼接喂所有层 | 全局可见 vs 分层可见 |
| E3 | **base-2c** | 1a+4c+**2c**+3b+5b | ref 层改成注入 [ref;text] | 让少量层同时看到两路是否更好 |
| E4 | **base-3a** | 1a+4c+2a+**3a**+5b | 加 motion loss | 时序差分监督对 FVD 的增益 |
| E5 | **base-4a** | 1a+**4a**+2a+3b+5b | mT5 全冻 12 层 | 文本侧容量/可训练性收益 |
| E6 | **base-4b** | 1a+**4b**+2a+3b+5b | mT5 12 层全可训练 | 文本表征上限 |
| E7 | **base-1b** | **1b**+4c+2a+3b+5b | 只走 context/前缀注入 | 换注入通路 |
| E8 | **base-1c** | **1c**+4c+2a+3b+5b | 前缀 + cross-attn 同时 | 双通路是否互补 |
| E9 | **base-1c-3a** | **1c**+4c+2a+**3a**+5b | 1c 且加 motion loss | 交互：motion loss 在双通路下是否仍有效 |
| E10 | **base-5a** | 1a+4c+2a+3b+**5a** | 开 PixArt-α 预训练初始化 | 预训练权重带来的绝对收益 |

读取规则的三个要点：

1. **E1 是所有比较的分母**。任何臂的"更好"都只相对 E1 说，且必须同口径（第 7 节）。
2. **E8 vs E9 是唯一允许的双因子臂**：它同时动了因子 1 和因子 3，只用来判断
   "motion loss 的收益是否依赖注入通路"，**不能**用它去替代 E4 或 E8 的单因子结论。
3. **矩阵是"单因子 + 一个交互"**，不要在这 10 臂里偷偷加别的（比如顺手换调度器、
   改 CFG、改步数）。要加就新开编号。

---

## 6. 分组科学问题（为什么是这 10 臂）

- **G1 注入通路之争（E1 / E7 / E8）**：内容信息到底是"每帧主动查询"（cross-attn）好，
  还是"混进同一序列被自注意力看到"（前缀）好，还是两者都要。这是本里程碑的主线。
- **G2 两路条件的可见性（E1 / E2 / E3）**：ref 与 text 是**分开喂**好（2a，各自专业化的层），
  还是**合并可见**好（2b，每层都能同时对齐 ref 与 text），还是折中（2c，少数层看齐全、
  多数层只看 text）。直觉上 2b 参数效率高但会让 ref 与 text 在同一注意力里互相干扰，
  2a 干净但 ref 层看不见文本命令。**E3 是最有信息量的一个臂**：如果它≈E2 且优于 E1，
  说明"ref 层需要文本上下文"是关键，可以直接指导后续设计。
- **G3 时序监督（E4，以及 E8→E9）**：纯扩散损失不显式约束帧间运动是否连贯；
  motion loss 在 latent 上做 k 步差分约束，理论上直接压"抖动/静止化"。E4 测它在 1a 下
  的增益，E9 测它在 1c 下是否还有增益（交互）。
- **G4 文本侧（E5 / E6）**：4c（浅+训）是 base，往两边各推一格：4a（深+冻）测"冻结的
  深表征"是否够用，4b（深+训）测"文本侧放开训练"的上限。三者一起才能判断收益来自
  **深度**还是来自**可训练性**——只做其中一对会得出错误结论。
- **G5 初始化（E10）**：随机初始化 vs PixArt-α 权重，直接回答"换骨架到底图什么"。
  若 E10 没有明显收益，5a 这条路线在本任务上就该降级。

---

## 7. 训练与评测协议（10 臂必须完全一致的部分）

| 项 | 规则 |
| --- | --- |
| 训练数据 | PHOENIX-2014T `train` split（7021 条，256 帧上限过滤后）；冻结 latent 缓存 `data/phoenix_latents/v1` |
| 评测数据 | `test` split 全量 **640/642**（官方 split；采样清单落盘复用） |
| 长度规则 | **不做长度预测**：训练用样本自身长度、评测/生成**直接给 GT 长度**；所有臂相同，不允许按臂调（2026-10-22 裁决） |
| 采样器 | 固定一种（推荐 DPM-Solver++ 或 DDIM，二选一后全臂不变）、步数固定、CFG 固定 |
| 种子 | 训练种子 + 采样种子都写进 config；采样种子全臂相同 |
| 指标 | 官方 FVD（**24 帧从头抽取**，第 2.1 节），唯一主指标 |
| 记录 | 每臂跑完登记 `EXPERIMENT.md` 一行（实验名/来源/做法/变量对照/协议/指标/裁决/去向/备注），`runs/` 与 `reports/` 不入库 |
| 复现 | 每臂的 `config.yaml` 必须包含：五因子取值、注入逐层表、文本侧 lr、motion loss 权重/warmup、预训练权重灌了哪些层 |

**预算口径**：10 臂 × 全量训练 + **~613 条有效样本**（24 帧口径下短片被跳过）× 24 帧生成的 FVD 评测，是本里程碑最大的成本项。
M1.5 先用 1–2 个臂的短程训练跑通全链路（含 FVD 自检），确认无管线错误后再排满矩阵
（RULES 15："少开探针、多跑真实训练"，但也不允许在管线未验证前烧 10 臂）。

---

## 8. 代码基线与分支现状（本次已完成）

- 分支：`milestone1-pixart`，从 `main`（`bdf36d1`，2026-10-18）切出。
  不从 HEAD 切的原因：HEAD 侧 25 个提交是 Wan 骨架与旧复现代码，本次全部不要。
- 提交 1 `44b874c`：把官方 FVD 评测链（`tools/eval_fvd.py`、`fetch_i3d_weights.py`、
  `redecode_fvd_frames.py`、`collect_experiment_table.py`、`tests/test_eval_fvd.py`）与
  `AGENTS.md`/`.gitignore` 的 FVD 口径修订从 `8454407` 单独移植过来（不含 Wan 与旧复现代码）。
- 提交 2 `88945f0`：清场。**208 → 45 个跟踪文件**，删除 163 个文件约 2.16 万行。
- 保留清单（45 个）：冻结 VAE 与 mT5 适配器、latent 数据集/collate/normalizer、
  `diffusion/gaussian.py`、`models/{conditioning,length_model,dynamic_loss}.py`、
  `training_utils.py`、`monitoring.py`、`pipelines/decode.py`、官方 FVD 与队列/解码类 tools、
  `configs/frozen/vae_h02.yaml`、13 个只依赖保留模块的 tests。
- 清场时顺手做的三处收口（都是为了不留悬空依赖）：
  `dataset.py` 去掉 paraphrase 开关、`conditioning.py` 去掉 slots 接口（`text_slots.py` 已删）、
  `training_utils.py` 的旧 DiT checkpoint 适配器改为懒加载 + 明确报错。
- 回滚：`git revert 88945f0` + `git revert 44b874c`，或直接 `git switch codex/signvip-paper-repro-20261022`。
- ⚠ 本分支**尚未跑过任何一个测试**（当前 shell 没有 torch 环境，只有 `/usr/bin/python3`）。
  开展 M1.x 前，必须在训练环境（`onvae`）里跑 `python tests/run_all.py` 确认保留文件全绿。

---

## 9. 阶段计划与验收判据

| 阶段 | 内容 | 验收判据 |
| --- | --- | --- |
| **M1.0** | 本规格文档 + 分支 + 清场 | 文档落 `.trae`、分支两个提交干净、`git status` 无残留（本次已完成） |
| **M1.1** | **latent 契约落地（Q1 已裁决：保持 1024 维 1-token）**：写死 `[B,T,1024]`；把 PixArt 的 patchify 替换为 `Linear(1024→D)` per-frame 投影；位置编码、unpatchify/`proj_out`（`D→1024`）按本项目重写 | 一个能前向跑通的 `latent [B,T,1024] → tokens → latent [B,T,1024]` 往返，形状断言全绿 |
| **M1.2** | **PixArt 骨架 + 条件接口抽象**：`ConditionBag{text_tokens, ref_tokens, masks}`；每层 cross-attn 的**注入计划表**（`layer → {none, text, ref, ref+text}`）成为配置项，默认值 = `[T,T,T,R]×7`（Q4）；mT5 768→**1152** 的**可训练**投影 + LN（Q3）；c 只含 timestep；1b 的"无 cross-attn"路径必须真实可配置 | 五因子能用**配置**表达（不改代码就能在 1a/1b/1c、2a/2b/2c 间切换）；单元测试覆盖各组合的前向形状 |
| **M1.3** | **训练循环**：x0 目标、EMA、原子 ckpt、分桶采样（复用 `training_utils.py`）、文本侧独立 lr、motion loss 开关；**不带长度头**（长度用样本自身长度） | 单臂短程训练（如 200 步）能产出 `config.yaml`/`metrics.jsonl`/ckpt，loss 下降 |
| **M1.4** | **评测链**：按 GT 长度采样（每臂同一批 id）、**从头抽取 24 帧** FVD（`--frame-mode contiguous --num-frames 24`）、有效 id 清单与真值缓存落盘 | `tests/test_eval_fvd.py` 绿 + `--self-check` ≈131 + **24 帧真值本底重新确认并写进文档** |
| **M1.5** | **管线验证**：1 臂（推荐 E1）小规模训练 → 生成 → FVD，全链路跑通 | E1 的 FVD 数字可信（不是 NaN、不低于本底、可复现两次） |
| **M1.6** | **十臂矩阵** | E1–E10 全部有 FVD + `EXPERIMENT.md` 十行 + 一张对照表 + 裁决（哪些因子该保留） |

规则：**M1.1/M1.2 不通过就不要排训练**；每一阶段结束更新 `STRUCTURE.md` 与 `TODO.md`。

---

## 10. 风险与缓解

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| `patchify`/`proj_out`/位置编码无法从 PixArt-α 移植（Q1 已裁决走 1024 维） | 5a 只能灌 block 内部，收益可能被稀释 | 逐层写清"灌了哪些、跳过哪些"；E10 的结论只能解释为"block 权重初始化收益" |
| 随机初始化（5b）+ 无预训练先验 | base 收敛慢，10 臂成本翻倍 | 固定训练步数与算力预算；先跑单臂确认学习率 |
| 24 帧从头抽取会丢样本（不足 24 帧跳过） | 有效 N 从 640 降到 ~613，跨臂不同批就无法比较 | 有效 id 清单落盘、10 臂共用；报表写有效 N；缓存 key 带帧数；改完重跑 `--self-check` 与真值本底 |
| 原尺寸骨架（28 层 / 1152 宽 / 16 head）+ 最多 256 帧 token | 单卡显存吃紧、batch 只能很小 | 小 batch + 梯度累积；显存不够也不缩骨架（Q8 已定），只调 batch 与序列上限 |
| 1b/1c 前缀注入（最多 248 条条件 token + 256 帧 token） | 序列近翻倍、self-attn 成本约 4 倍，臂间速度不可比 | 记录每臂 tokens 数、显存峰值与每步耗时，表里注明成本差 |
| 2a/2c 的逐层分配靠拍脑袋 | 结论可能只是"分配表选得差" | 分配表写进配置，先固定一版；后续可做层分配小消融（但**不在本里程碑 10 臂内**） |
| 前缀注入（1b/1c）拉长序列 | 显存/时延上涨，臂间比较不公平 | 记录每臂的 tokens 数与显存峰值 |
| motion loss 权重未调 | E4/E9 可能测成"噪声" | 权重与 warmup 固定且写进配置；必要时先用短程确认量级 |

---

## 11. 设计问题台账

> 这些是"不裁决就会整轮白跑"的问题（RULES 15）。**已裁决的保留在这里，不要删**
> （决策可追溯）；未裁决的才会阻塞对应阶段。

### 11.1 已裁决（2026-10-22 用户）

| ID | 问题 | **裁决** | 落地位置 |
| --- | --- | --- | --- |
| **Q1** | latent 空间：PixArt-α 的 SD-VAE 4ch vs 本项目 1 token/帧 1024 维 | **保持 1024 维 1-token（TiTok-CVAE 不变）**。改 PixArt 适配本项目：patchify → per-frame `Linear(1024→D)`，位置编码与 `proj_out` 重写；5a 只移植 block 内部权重 | §2.3 第 1 条、§9 M1.1 |
| **Q2** | "context 注入"的确切含义 | **是**：条件 token 拼进 self-attn 序列当前缀，1b 不设 cross-attn 层 | §3 因子 1 表、§9 M1.2 |
| **Q5** | 24 帧 FVD 怎么取帧 | **从头抽取 24 帧**（frame 0→23；官方 I3D 模型与代码不变；不足 24 帧跳过、有效 id 清单落盘）——**2026-10-22 第二轮改判**，原答「均匀抽取」作废 | §2.1、§9 M1.4 |
| **Q6** | 生成长度规则 | **本里程碑不做长度预测**；训练用样本自身长度、评测/生成用 **GT 视频长度** | §2.2、§7 长度规则行 |
| **Q3** | mT5 768 → 骨架宽度的投影是否可训练 | **可训练**（Linear + LayerNorm；10 个臂都有，不构成臂间差异） | §3 因子 4、§9 M1.2 |
| **Q7** | 4c 的"前 3 层"是否含 embedding | **含**：训练范围 = `embedding + encoder.layer[0:3]` | §3 因子 4 |
| **Q8** | 骨架超参 | **PixArt-α 原尺寸**：28 层 / 1152 宽 / 16 head | §4、§9 M1.2 |
| **Q9** | motion loss 定义 | **沿用** `MultiScaleDeltaLoss`（k∈{1,2,4} + warmup） | §3 因子 3 |
| **Q4** | 2a/2c 的层分配 | **间隔放置**：每 4 个 block 前 3 个 text、第 4 个 ref ⇒ `[T,T,T,R]×7`，ref 层 = 3/7/11/15/19/23/27 | §3 因子 2、§4、§9 M1.2 |

### 11.2 未裁决

**无。** Q1–Q9 全部裁决完毕（2026-10-22 两轮）。下一步不是再裁决，而是等用户授权进入
**M1.1**（latent 契约落地）——按 RULES「授权不延伸」，本文件不自动开工。

## 12. 回滚与隔离

- 代码：`git switch codex/signvip-paper-repro-20261022` 回到旧世界；
  或 `git revert 88945f0 44b874c` 在本分支恢复被删文件。
- 知识库：`.trae/` 不在 git 里，切分支不影响；本文件是新增文件，删掉即可。
- 旧实验产物（`runs/`、`reports/`）未动；milestone-1 的产物应写到新的目录前缀
  （建议 `runs/m1_<臂名>/`、`reports/eval/m1_<臂名>/`），与旧表物理隔开。

## 13. 目标机实测补记（2026-10-22，实现期）

24帧有效N不再使用历史~613估计：当前冻结test642中640条num_frames>=24，两条短片跳过；清单 `reports/eval/samples/m1_test_first24_ids.txt`，SHA256=6f96ad9dda2721b877dea1f08dab7bbf17bb4c6bcb9d90becb665824a69110f6。官方I3D-400 logits/24连续帧/frame0起/224²/fulltest640 的真实split-half本底实测 **84.29751411436246**（原16帧本底不适用）。证据 `reports/eval/m1_first24_prepare_20261022/real_floor.json` 与 protocol.json。自检133.908176通过。该补记仅用实测值回填§2.1，不改变取帧规则。

# 思辨会话 — 条件注入方式与 latent token 压缩 · 2026-10-14

> **2026-10-14 迁移**：本文件原在 `docs/critique/sparring-20261014-conditioning-and-token-compression.md`，按 `RULES.md` 知识库流转规范第 8 条
> 整体迁入 `.trae/documents/`；文中出现的旧路径已改写为新路径。

<!-- research-sparring 记录。保留：每条反驳与证据等级、用户回应、我收回过哪些。 -->

- 项目 / 仓库：`/home/huyang/data/trae_projects/text2video`
- 我的主张：第 1 场 C1/C2（第 0 轮）；第 2 场 C3/C4（见第 2 场第 0 轮）
- 强度设定：S2–S3 默认
- 跑了几轮：第 1 场 0–1、第 2 场 0–1（均已发出，用户尚未逐条回应）
- 流程偏差（记录在案）：技能要求在用户确认复述后再进入反驳；本次用户明确要求"现在反驳我"，
  故第 0 轮与第 1 轮合并发出，**立论未经用户确认**。若复述有误，以用户修正为准。

---

## 第 0 轮 · 立论

**CLAIM C1**："cross-attention、AdaLN 调制、joint attention 这三种文本条件注入方式，
都可以在本项目上尝试一下。"

**CLAIM C2**："因为（可能）一个德语文本 token 对应多个视频帧的动作，所以可以把多个 latent
token 压缩为/表示为一个 token 与 text token 关联，再用这个 token 还原多个 latent。"

**支撑**（用户给出的理由）

1. 三种注入方式是当前视频生成的主流做法 → 第三方观察，用户未给论文或数据。
2. 一段文本 token 对应多帧动作 → 直觉提问（"是不是"），无证据。
3. 压缩能减少 text/video token 数不匹配 → 直觉（D）。

**承重假设**

| 编号 | 假设 | 级别 | 若为假会怎样 | 现有支撑 |
|------|------|------|-------------|---------|
| A1 | 现役 DiT 里"注入方式"仍是可分辨的瓶颈，三种各跑一遍能分出高下 | CRITICAL | 改造顺序与预算分配全错 | 部分（仓库已有 cross-attn 基线 vs text-prefix 对照，但结构变化与训练预算同时变化） |
| A2 | 词↔帧是"一个词对多帧"的稳定多对一关系，且该粒度是语义瓶颈 | CRITICAL | C2 的设计前提不存在 | 无；数据侧无时间对齐标注（见第 1 轮 R3） |
| A3 | 压成一个 token 再还原，不会丢掉生成所需的帧级细节（或损失可补偿） | CRITICAL | 生成质量与对齐同时退化 | 无 |
| A4 | 压缩后的 summary token 与 text token 的关联，比现有 cross-attn/prefix 更能提升"文本决定语义" | IMPORTANT | C2 的收益假设落空 | 无；仓库证据指向瓶颈是"文本利用率"而非"粒度" |
| A5 | 现役 SignMbart/32 条 test 口径能分辨这些改动 | IMPORTANT | 实验做了也判不出 | **已被本仓库测量基本否定**（地板 9.32 vs 现役 9.47，ΔB1 +0.16 [−5.03,+5.13]） |
| A6 | 目标压缩比取"文本 token 数"（均值 24）是合适设计 | SECONDARY | 长度依赖的瓶颈，长句退化 | 无 |

**成功标准**：用户未定义。暂定（待用户改）：新结构在 dev 上检索 MRR ≥ 1.5× 随机水平，
且 SignMbart B1/B4 相对地板（9.32 / 0）有配对 CI 不含 0 的提升。

**目标与约束**：冻结 1-token/帧（1024 维）VAE、latent 扩散、x-prediction、纯文本条件、
SignMbart 回译口径（`RULES.md` 五条硬约束）；共享 GPU，训练约 7–8 steps/s（30k ≈ 1 h）。

**复述确认**：未确认（见上"流程偏差"）。

---

## 第 1 轮 · 思想层

**我复述你的主张**：这三种文本条件注入方式都值得在本项目试；并且因为一个文本 token
可能对应多帧动作，可以把多帧 latent 压成一个 token 与文本关联、再展开回多帧。

### 反驳 1 ｜ S1 ｜ 镜头 A1 概念模糊

- 你的说法："现在视频生成主流的方向有 cross-attention，AdaLN 调制，以及 joint attention；
  我觉得这 3 种文本条件注入的方式都可以尝试一下"
- 问题：三者不在同一分类轴上。adaLN 注入的是一个全局向量（所有帧共享同一调制），
  cross-attn 与 joint attention 注入的是 token 序列（前者独立子层零门控，后者直接进
  self-attn）。它们**可以同时存在**，现役配置就是"cross-attn + 池化文本→adaLN"同时开，
  `text_prefix` 再叠加 joint attention。按字面"三种都试"，仓库里目前**没有**
  "text 开着但关掉 cross-attn"的构造路径（`factory.py` 里 `use_text=True` 时
  `context_dim` 恒等于 hidden_size）。真正没试过的只有一种：池化文本 + adaLN、完全去掉
  cross-attn 的裸变体，代价是加一个开关（约 10 行）+ 一次等预算训练。
- 证据：`t2v/models/conditioning.py` 头部契约（池化文本→adaLN 与 cross-attn 并存）、
  `t2v/models/dit.py` 差异清单第 4/6 条、`t2v/models/factory.py:build_bundle`；
  `DESIGN.md` 里程碑"文本条件消融"（cross-attn 基线 vs text-prefix 已跑）；
  `NEXT_BLEU_FVD_REVIEW_20261014.md` §2（32 条 test：prefix-60k 相对错配真视频
  ΔB1 +0.16 [−5.03,+5.13] 不显著；prefix-90k ΔB1 +5.44 [+0.58,+10.55] 显著）。
- 证据等级：B
- 什么能让我收回：你指出的"三种"是另一套划分（例如 self-attn 前/中的门控形式），
  或你要试的第三种配置与我理解的不同；或给出证据说明裸 adaLN 变体可能显著更好。
- **你的回应**：待回应

### 反驳 2 ｜ S3 ｜ 镜头 A3 隐含假设未检验

- 你的说法："我在想能不能变相地将多个 latent token 压缩为或者表示为一个 token
  去和 text token 进行关联"
- 问题：这条思路的核心前提是"text token 数与 video token 数不匹配 ⇒ 对齐学不好"，
  但仓库自己的测量指向另一个原因：**文本影响弱**（换文本只动输出 ~1%；cross-attn 门控
  幅度 0.18，有效贡献约为 MLP 的 3%）。注意力不要求两侧 token 数相等，cross-attn 的
  代价是 O(N_v·N_t)，粒度与"模型愿不愿意用文本"是两件事。压缩 token 数改变的是条件
  的粒度，不改变利用文本的压力。
- 证据：`DESIGN.md` 里程碑（通道解剖：msa 0.30 / xa 0.18 / mlp 0.62，cross-attn 有效贡献
  ≈ MLP 的 3%；基线换文本只动输出 ~1%）；`NEXT_BLEU_FVD_REVIEW_20261014.md` §0.2
  （表示层不是主瓶颈，条件生成才是）。C 级：注意力机制不要求 N_v = N_t。
- 证据等级：B（含 C）
- 什么能让我收回：给出测量——压到 N_text 附近后，文本对输出的影响或 dev 检索 MRR
  显著高于"等容量但不压缩"的对照；或证明当前利用率低确实源于 token 数不匹配
  （例如 attention map 显示文本 token 被稀释/混叠）。
- **你的回应**：待回应

### 反驳 3 ｜ S3 ｜ 镜头 A2 前提错位（数据集事实）

- 你的说法："对于德语文本来说是不是一个 text 的 token 就会对应多个视频帧的动作"
- 问题：
  (a) **数据里没有时间对齐标注**。PHOENIX14T 语料注释的 start/end 字段全部为 -1：
  train 7097 行、dev 520、test 643、complex-annotation 7097 行，非 -1 计时行数 **0**。
  词/gloss 级时间戳不存在，"德语 token k ↔ 帧区间 [a,b]"没有可监督的对应关系。
  (b) 语料是**句级翻译**，且 gloss 也不是德语：一个德语词可对应带修饰的 gloss
  （例：`mancherorts` → `ORT-PLUSPLUS`），德语与 DGS 语序不一致（topic-comment/SOV、
  非手控标记承载否定与疑问）。四舍五入，这条映射最多是启发式，不是事实。
  (c) 我实测的计数：T 均值 115.6（中位 111）；德语词均值 14.0；mT5 token（含特殊符）
  均值 24.4；gloss 均值 7.8。三个候选"目标 token 数"给出的压缩比分别是 1:8 / 1:5 / 1:15。
- 证据：本仓库数据实测（可复现命令见文末）+ 注释文件本身（B）；DGS 语序为领域共识（C，
  文献未逐条核验，故不标 A）。
- 证据等级：B + C
- 什么能让我收回：指出我漏掉的词级时间标注文件；或用现有 latent→文本教师导出的对齐
  统计证明"德语 token ↔ 帧区段"存在稳定单调对应。
- **你的回应**：待回应

### 反驳 4 ｜ S2 ｜ 镜头 A4 过强外推 + A5 失败模式未定义

- 你的说法："然后再使用这一个 token 去还原多个 latent"
- 问题：
  (a) 你不可能在 tokenizer 层做（硬约束 1 冻结 1-token/帧 VAE），只能在 DiT 内部做
  "池化→条件交互→上采样"。已核验的"时间维压缩"主流做法都是前者：CogVideoX 用 3D VAE
  沿空间与时间压缩，再用 expert adaptive LayerNorm 融合两模态；W.A.L.T 用 causal 视频
  VAE 压缩，文本→视频走级联（base latent diffusion + 两级 super-resolution）。
  即：压缩器与重构联合训练、还原由独立生成阶段承担。
  (b) 网络内部瓶颈若不配独立的还原监督，梯度倾向让"池化→上采样"学成恒等或低通，
  此时文本的粒度变了、但"文本决定语义"的压力没变；理解端的压缩先例（Perceiver
  Resampler、Q-Former、TiTok 的 1D tokenizer）目标只是"够用来判断/重建"，与
  "够用来高保真条件生成"的要求不同。
  (c) 设计缺陷：若 summary token 数取文本 token 数（均值 24），瓶颈宽度就变成数据依赖——
  mT5 token 从 4 到 85，压缩比随之浮动 4–85×；固定 k（如 32 个查询）在文献里更稳。
- 证据（身份与摘要已核验）：CogVideoX arXiv:2408.06072（摘要：3D VAE 沿空间与时间压缩、
  expert adaptive LayerNorm 融合两模态）；W.A.L.T arXiv:2312.06662（摘要：causal encoder
  联合压缩图像/视频；文本→视频为级联 + super-resolution）；TiTok arXiv:2406.07550
  （256×256 图像 → 32 个 token）；Flamingo arXiv:2204.14198、BLIP-2 arXiv:2301.12597
  （固定数量查询压缩，用于理解/对齐）。等级 A（身份与结论）+ C（无重构目标的信息瓶颈
  不保证保细节）。
- 证据等级：A + C
- 什么能让我收回：给出同规模对照，证明"内部压缩+还原"在不加额外监督、不改 VAE 的条件下，
  dev 检索 MRR 或 SignMbart 显著优于等容量不压缩的对照；或说明你的实现实为级联生成
  （那样它等于仓库里已有的 E7 时间计划路线，不是新机制）。
- **你的回应**：待回应

### 反驳 5 ｜ S2 ｜ 镜头 A9 指标替代目标 / B5 指标错配

- 你的说法（隐含）：试完用现有 SignMbart 回译判断哪种更好。
- 问题：该指标当前的地板是 B1 9.32（错配真视频），现役生成 9.47，配对 bootstrap
  ΔB1 +0.16 [−5.03,+5.13]，P(>0)=0.57 —— 二者不可区分；32 条上的 CI 约 ±3 B1。
  这两条思路的预期效应都远小于 CI。表示上限 31.57 说明表示层不是主瓶颈；而近免费项
  （DDIM 50→250 使 B4 0→2.77；90k > 60k）还没花完。指标不改，注入方式与瓶颈结构的消融
  都判不出结果。
- 证据：`NEXT_BLEU_FVD_REVIEW_20261014.md` §0、§2（地板 / 上限 / DDIM-NFE / 检索 MRR：
  重建 0.246、prefix-90k 0.218、prefix-60k 0.135、随机 0.127）。
- 证据等级：B
- 什么能让我收回：在统一评测矩阵（全 518 dev、3 个生成 seed、配对噪声/长度/文本）与
  检索指标下证明效应可分辨。
- **你的回应**：待回应

---

（本轮共 5 条，请逐条回应：接受 / 反驳 / 修订 / 存疑）

---

## 本轮实测与核验记录

数据统计（可复现）：

```bash
cd /home/huyang/data/trae_projects/text2video && /data/huyang/miniconda/envs/onvae/bin/python - <<'PY'
import json, statistics as st, csv
base="data/phoenix_latents/v1/"
frames=[json.loads(l)["T"] for l in open(base+"train.jsonl")]
texts=[json.loads(l)["text"] for l in open(base+"train_text.jsonl")]
print("T", st.mean(frames), st.median(frames), min(frames), max(frames))
print("de words", st.mean(len(t.split()) for t in texts))
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained("models/mt5-base")
print("mt5 tokens", st.mean(len(tok(t)["input_ids"]) for t in texts))
p="/data/dataset/phoenix2014-T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv"
gl=[len(r["orth"].split()) for r in csv.DictReader(open(p), delimiter="|")]
print("glosses", st.mean(gl), st.median(gl))
PY
```

注释计时字段检查：

```bash
for f in /data/dataset/phoenix2014-T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/*.csv; do
  awk -F'|' -v f="$f" 'NR>1 && $3!="-1"{n++} END{print f, "nonneg_start="n+0}' "$f"
done
```

已核验论文（仅身份与摘要，未通读全文）：
CogVideoX 2408.06072 · W.A.L.T 2312.06662 · TiTok 2406.07550 · Flamingo 2204.14198 ·
BLIP-2 2301.12597 · 另有仓库既有核验：DiT 2212.09748、SD3 2403.03206、PixArt-α 2310.00426。

未完成的检索：arXiv API 与 Semantic Scholar 均返回 429（限流），
`search_papers.py` 本轮未产出文献表；第 3 轮的对立检索需重跑。

---

## 第 2 场 · 2026-10-14（同日续场；C3/C4）

> **落盘说明**：本场内容原为对话形式（第 1 场的 R1–R5 用户未逐条回应，直接提出 C3/C4；
> 之后是 Q2–Q6 五个问答）。按 `RULES.md` 规范 11，调研/思考类默认不落盘；2026-10-14
> 用户明确要求"先落到文档"，故把对话内容整理补记于此。整理与对话同日进行，措辞可能与
> 实时版本略有出入；证据与数字以本文档为准，且全部来自随后固化的 `tools/` 探针或
> `reports/eval/` 既有产物（命令与关键数字见本场文末）。

### 第 0 轮 · 立论（第 2 场）

**CLAIM C3**："把现在 text 的 cross-attention 的 gate 直接取消掉、设置为 1，能不能让
text 条件更加发挥作用。"

**CLAIM C4**："采用 CogVideoX 中的模型架构——混合的 expert AdaLN 以及拼接 text latent
和 video latent——其中 video latent 再用一个（时间）压缩器在时间维度上再压缩一下。"

**支撑**：C3 —— 通道探针显示 `gate_xa` 均值只有 0.152（用户在场看到
`reports/eval/probe_channels_prefix60k.json`）；C4 —— CogVideoX 是开源视频生成的强架构，
被当作"更强的条件融合方式"的参照。

**承重假设（续编号 A7–A9）**

| 编号 | 假设 | 级别 | 若为假会怎样 | 现有支撑 |
|------|------|------|-------------|---------|
| A7 | gate 是文本的主要限流器：放大 gate 即放大文本的因果作用，且不破坏其他通路 | IMPORTANT | C3 不再是"机制修复"，只剩分布外干预 | 部分：cross 子层对文本敏感（rel 1.09），但其贡献只有 MLP 的 4.6%；gate 是训练学出的值 |
| A8 | CogVideoX 式注入（expert AdaLN + 拼接）在本项目规模上能提高文本利用 | IMPORTANT | C4 失去理由 | 弱：本项目已有"拼接"（text_prefix）与池化→adaLN；expert 分支的净增量≈双流/容量问题（idea-004） |
| A9 | 时间压缩降低"text 要覆盖的内容量"，间接提高文本监督强度 | IMPORTANT | 压缩器只剩省算力动机（而算力侧已被实测否定） | 无：k=4 压力上限 1.22×；文本敏感度与 T/token 覆盖率秩相关 +0.003 |

**成功标准**：用户未定义。沿用第 1 场暂定标准，但第 1 场反驳 5 已指出该口径当前
分辨力不足（32 条 CI ±3 B1、地板与现役不可分）。

**复述确认**：未确认（同样的"现在反驳我/我想用……"式请求，立论与反驳同发）。

### 反驳 6 ｜ S2 ｜ C3 ｜ 缺陷类型：机制误读 + 剂量未定义

- 你的说法："能不能把现在 text 的 cross-attention 的 gate 直接取消掉，设置为 1，
  让 text 条件更加发挥作用。"
- 问题：(a) gate 不是阀门而是**学出来的限流器**——12 个 block 的 `gate_xa` 均值 0.152
  （0.088–0.240），它把 cross 输出乘 ~0.15 再进残差；把它设成 1 等于把占 MLP 约 4.6% 的
  贡献通道一次性放大 ~6.6×（`contrib_cross` 0.039 → ~0.26，相对输出 RMS 1.83 从 ~2% 到 ~14%），
  这是分布外干预，等价于推理期单方面改一条残差支路的缩放。(b) 更关键：cross 子层**本身**
  对文本高度敏感（`rel_correct_shuffled_cross` 均值 1.09，换文本让该子层输出变化 109%），
  所以"文本信息进不来"不成立；被压低的是**幅值**。(c) 因此 C3 值得做的不是直接置 1，
  而是剂量曲线 + 与 CFG 等文本剂量的对照（已立 idea-016）。
- 证据：`reports/eval/probe_channels_prefix60k.json`（checkpoint `runs/dit_b_prefix30k/final_60k.pt`，
  权重=model，t=499）。均值复算：gate_xa 0.1524（0.0877–0.2397）、gate_msa 0.4008、
  gate_mlp 0.7007；contrib_cross 0.0387 vs contrib_mlp 0.8417；rel_correct_shuffled_cross 1.0908、
  rel_correct_shuffled_hidden 0.1159、rel_correct_shuffled_kv 0.9839；输出 RMS 1.83、
  换文本末端输出变化 0.0229（1.25%）。等级 B。
- 什么能让我收回：剂量曲线（`gate_xa`×k，k∈{1,2,4,6.6}）显示 text-swap 相对幅度随 k
  单调显著上升，且与 CFG 等剂量对照下 test-642+train-642 配对指标不劣 → 收回"置 1 是
  分布外干预而非修复"的判断，转为支持直接放大。
- **你的回应**：待回应

### 反驳 7 ｜ S2 ｜ C3 ｜ 缺陷类型：目标替代（免费实现已存在）+ 成本错配

- 你的说法（隐含）：需要结构级手段让文本更起作用。
- 问题："放大文本推力"已有**免训练**的推理期实现：CFG。同 checkpoint（test-32，DPM++ 50）
  上 CFG 1.0 → 3.0 使 B1 8.37 → 14.77、MRR 0.133 → 0.205、FD 54.30 → 54.02（几乎不降）。
  CFG 的代价是 2× NFE；gate 放大若有效，价值在于 1× NFE 就拿到同等文本推力——因此它
  应该与**CFG 等文本剂量的结果**对照，而不是跟 CFG=1.0 的基线比。若同剂量打不过 CFG，
  C3 没有独立价值。
- 证据：`reports/eval/RESULTS_TABLE_20261014.md`（`prefix60k_dpmpp_o2_s50_cfg10` / `cfg30` 行；
  CFG 扫描已有 1.0/1.5/2.5/3.0 四个点）。等级 B。
- 什么能让我收回：gate 剂量曲线与 CFG 在**等同文本剂量**下打平或更好（配对 CI 支持）；
  或证明二者可叠加（CFG=3.0 之上再加 gate 放大仍单调受益）。
- **你的回应**：待回应

### 反驳 8 ｜ S2 ｜ C4 ｜ 缺陷类型：分类轴混乱 + 跨规模外推

- 你的说法："采用 cogvideox 中的模型架构，也就是混合的专家 AdaLN 以及拼接 text latent
  和 video latent。"
- 问题：(a) CogVideoX 的架构是一整套配方（3D causal VAE 时空压缩、patchify、RoPE、
  expert AdaLN、3D full attention），单拎"expert AdaLN + 拼接"移植的前提不成立；论文
  自述 16×16×8 的 patch 收敛困难、采用 RoPE 才收敛更快。(b) "拼接"本项目**已经在跑**：
  `text_prefix: true` 就是把文本 token 拼进自注意力（mmDiT 式），池化文本→adaLN 也在跑；
  真正没试过的净增量是"expert 分支"（文本/视频各自一组调制参数），那属于双流/容量对照
  问题（idea-004，已排队），不是新机制。(c) CogVideoX 在数千万级视频片段上训练，其架构
  结论外推到 PHOENIX（7k 句）规模，证据等级最多 C。
- 证据：CogVideoX arXiv:2408.06072（身份+摘要级结论已核验）；本项目
  `configs/dit_b_text_prefix.yaml`（`text_prefix: true`）、`t2v/models/conditioning.py`
  头部契约、`DESIGN.md`"文本条件消融"。等级 A（论文身份）+ B（本项目现状）。
- 什么能让我收回：等预算下 expert 分支相对容量匹配对照的条件利用增益可被判定
  （统一协议 P、检索 MRR 为门槛）→ 那时 C4 的这部分成立。
- **你的回应**：待回应

### 反驳 9 ｜ S3 ｜ C4（时间压缩器）｜ 缺陷类型：隐含假设未被数据支持 + 收益上限可量化

- 你的说法："video latent 再使用一个时间压缩器在时间维度上再压缩一下。"
- 问题：冻结 1-token/帧 VAE 的硬约束下，压缩只能在 DiT 内部做。三条实测：
  (a) **上限**：k 帧合并的"冗余窗口" r(k=4)=18.2%，对文本的相对压力最多 1/(1−r)=1.22×
  （k=16 才 2.09×）；(b) **墙钟**：训练实际 batch=16 上 k=4 加速 ≈1.0×（4090 初测 0.99×，
  3090 复测 1.14×）；batch≥64 才 ~1.45–1.50×，而同等省算力可由**免费**的 pad-mask 拿到
  （batch 64: 1.59–1.67×），pad-mask 已实测零收益（FAIL-003）；(c) **文本侧**：
  "text 要覆盖几百帧很困难"的直接量化指标 T/token 与文本敏感度的秩相关 ≈0（+0.003），
  分层探针短片 1.52% vs 长片 1.41%（差 +0.11pp ±0.21pp，95% CI 无显著差异）。
  三条都指向：压缩器不解决文本问题，算力收益也只在训练用不到的 batch 上出现。
- 证据：`tools/probe_temporal_redundancy.py`、`tools/probe_text_sensitivity_by_length.py`、
  `tools/bench_sequence_length.py`（产物 `reports/eval/probe_text_sensitivity_by_length_prefix60k.json`、
  `reports/eval/bench_sequence_length_prefix60k.json`）；FAIL-003（route_d 5k 等预算对照）。等级 B。
- 什么能让我收回：池化-解码信息探针显示 k=2/4 合并后同锚点重建退化 ≤0.5 dB 且检索不降
  （即"合并丢掉的东西模型确实用不到"），且后续等预算训练在 P 协议下不劣 → 压缩器值得做。
- **你的回应**：待回应

### 反驳 10 ｜ S2 ｜ C3+C4 ｜ 缺陷类型：量尺错配（同第 1 场 R5）

- 你的说法（隐含）：这两个改动都值得试。
- 问题：能判定结构级改动的量尺很贵且分辨力有限：32 条 test 上 SignMbart B1 的配对 CI
  约 ±3；错配真视频地板 B1 9.32 与现役 9.47 不可区分（ΔB1 +0.16 [−5.03,+5.13]）。
  项目里唯一把指标从量尺里推出来的干预是**改目标**与**改采样/CFG**（B 路线 dev64 MRR 0.171
  vs 同起点续训 control 0.137；CFG 3.0 MRR 0.205）。所以 C3/C4 都应先走"免训练探针筛选
  → 统一评测矩阵（规则 6 / TODO D1–D3）→ 再谈训练级改动"的顺序。
- 证据：`NEXT_BLEU_FVD_REVIEW_20261014.md` §0、§2；`reports/eval/RESULTS_TABLE_20261014.md`。等级 B。
- 什么能让我收回：统一评测矩阵落地并证明结构级效应可分辨 → 顺序不再受此限制。
- **你的回应**：待回应

---

## 本场问答（Q2–Q6）

（以下为用户提问，未主张新机制；按规范 11 原仅在对话回答，此处补记。）

### Q2 · 有 pose 监督就"容易"，只有 text 时视频几乎都是乱的，为什么

证据等级 B，四点：

1. 不是"文本监督天然弱于 pose"，而是**信息与损失结构不同**。x0-MSE 只要求重建得准：
   能用文本的通道走文本，用不上的走图像先验。通道探针：换文本让 cross 子层输出变化 1.09
   （信息到了），末端输出只动 1.25%（幅值被压低）——问题在"文本利用率"。
2. 目标里有一部分信息文本决定不了（手型细节、时长、身份风格）；且 PHOENIX 是**句级**翻译
   语料，没有词/gloss 级时间标注（非 -1 计时行数 0），"哪个词管哪段帧"没有可监督对应。
3. pose 之所以让生成变容易：pose 与视频几乎同构（逐帧、空间对齐），用 pose 监督等于把
   **"文本→姿态"这一步短路**。RefSign 阶梯（SignMbart B1）：真视频 50.39（N=64）/44.55（N=642）
   → gtpose 20.66 → deploy（text→pose→video）12.87。gtpose 与 deploy 差的 7.8 B1
   全部落在"文本→姿态"这一步——恰好就是本项目要从文本直接学的那一步。
4. 含义：无 pose 硬约束下要提升，只有 (a) 增大"必须用文本"的压力（目标层 idea-001、
   推理期 idea-016），(b) 提高文本与目标的对应信息（对齐先验 idea-002，但无标注），
   (c) 换分辨力更高的量尺（已有队列）。

### Q3 · FlashSign 的方法能不能借鉴、代码有没有开源

核验结论（2026-10-14）：

- 论文：FlashSign，arXiv:2603.27915v1。**代码未开源**：对应仓库 README 77 字节、单提交，
  无训练/推理代码可读。
- 任务与设置不同：词级 ASL（WLASL）范式，基于 Wan2.1-1.3B/14B 微调，文本走 UmT5；
  条件侧没有可移植的新机制。
- 指标口径不同：报告 FVD 453 / PSNR 25.74 / SSIM 0.83 / LPIPS 0.07；全文 **BLEU/WER 出现 0 次**，
  与 SignMbart 回译口径不可比；同一对比里 SignGen 的 PSNR 27.72 反而更高。
- 其 Limitations 自认长句受限、需要 pose 先验——与我们的无 pose 硬约束直接冲突。
- 结论：能借鉴的只有工程侧（数据规模、PSNR/SSIM 类视频质量指标写法），不构成方法借鉴；
  不建议以 FlashSign 作为改进目标。

### Q4 · 要不要做 latent 时间维 patchify

结论：**现在不做**（三条实测否决，见反驳 9）。若将来复活，前置是"池化-解码信息探针"：
`tools/eval_vae_recon.py` 加 pooling 模式（k=2/4，同锚点解码），判据"重建退化 ≤0.5 dB 且
检索不降"；通过也仍要按 P 协议做等预算训练对照。登记为 idea-018（parked）。

### Q5 · 前缀位置里为什么这么多 pad

- 事实：`t2v/adapters/text.py:19,80` 用 `padding="max_length", max_length=120`。
  语料统计（mT5 base，含特殊符）：train n=7096 均值 24.4 / p50 23 / p90 38 / p99 53 / max 85；
  test n=642 均值 21.7 / p50 21 / p90 32 / p99 44 / max 52；**截断从未触发**；
  pad 占比 79.6%（train）/81.9%（test）。120 与 PixArt-α §3.1 的 120 一致（推测为其来源，
  未逐字求证）。
- 真正的问题不在垫到 120，而在 prefix 路径把 pad 全当有效 key（`t2v/models/dit.py:639`
  `prefix_valid = torch.ones(...)`）：注意力质量 pad 40.2% / 真实词 12.1%（末层 62.9%/14.6%）。
- 修法已实现并测过：`mask_text_padding` + `truncate_text_prefix`
  （`configs/dit_b_text_prefix_padmask.yaml`）。等预算对照（route_d 5k，dev-64 DPM++ 50）：
  MRR 0.1079 vs 0.1077 持平、B1 8.75 vs 10.04 更低 → 零收益（FAIL-003；D6 降级为
  "省算力选项"）。零样本 test-32 的 MRR 0.135→0.155
  （`reports/eval/retrieval_prefix60k_padmask_zeroshot_s50.json`）样本太小，不算证据。
- 判定：pad 是接口层浪费，不是文本利用瓶颈——这也是 idea-017 的风险来源。

### Q6 · 用 Perceiver 式可学习 query（文本 + 120 可学习 token → 取固定 token 作文本表示）可以吗

- 可做，但要三处修正：(a) 方向反了——真实 token 均值 24、垫到 120 再"取 120"是**上采样**，
  而 Flamingo/BLIP-2/TiTok 都是 N→M<N 压缩，用到的是"固定槽位"这一面；槽数应取 16/32；
  (b) 唯一真实优点是**槽身份稳定**（第 i 槽跨句语义一致；现状第 i 位语义随句浮动、
  且大部分位置是常数 pad 块）；(c) 风险：与 pad-mask 同类"接口清理"干预在项目内历史全零
  （FAIL-003），且前缀位置编码权重会失配——需要等参数对照（truncate+mask、pooled 复制 k 份）
  才能归因。排序：在 idea-001 之后。登记为 idea-017（candidate）。

---

## 本场实测与核验记录（第 2 场）

复现命令：

```bash
# 时间冗余/压力上限（本会话前段固化）
python tools/probe_temporal_redundancy.py --videos 400

# 文本敏感度：长度分层 + token 匹配带（本场固化）
python tools/probe_text_sensitivity_by_length.py --mode both --gpu cuda:7 \
    --out reports/eval/probe_text_sensitivity_by_length_prefix60k.json

# 帧数 × 文本长度 × pad-mask 墙钟矩阵（本场固化）
python tools/bench_sequence_length.py --gpu cuda:7 \
    --out reports/eval/bench_sequence_length_prefix60k.json
```

关键数字：

- 时间冗余（train，400 条）：r(k)=0.0650/0.1822/0.3484/0.5218（k=2/4/8/16）；
  1/(1−r)=1.069/1.223/1.535/2.091；丢弃 RMS 8.81/14.75/20.41/24.97；逐帧方差分解
  clip 间 0.199 / clip 内 0.801；相邻帧余弦 0.9689。
- 文本敏感度（prefix-60k，state=model，t=499，seed 0）：
  - strat：N=96，T 15–230，tokens 8–48；rel 均值 1.55%、中位 1.54%、min 0.00%、max 2.81%；
    Spearman(rel,T)=−0.223、Spearman(rel,tokens)=−0.297、**Spearman(rel,T/tokens)=+0.003**；
    分箱：T∈[0,40) 2.04%（n=5）、[40,80) 1.66%（18）、[80,120) 1.52%（34）、
    [120,180) 1.46%（31）、[180,300) 1.53%（8）；96 条中 2 条 rel<0.5%。
  - matched（token 带 20–32）：短片 n=25（T≈76，tok≈22.1）rel 1.52%±0.47；
    长片 n=48（T≈159，tok≈28.5）rel 1.41%±0.34；差 +0.11pp ±0.21pp（95%）。
  - 口径注意：本探针与 `probe_condition_channels` 统一用原始 `model` 权重；会话中初次
    口头数字用的是 EMA 权重（rel 均值 1.05%），结论不随 state 改变。
- 墙钟矩阵（DiT-B fwd+bwd，含 AdamW step；bfloat16）：
  - 4090（cuda:3，会话初测）：batch16 82.4/80.5/83.0/82.5 ms（k=4 0.99×）；batch64
    150.1/90.1/100.2/83.0 ms（pad-mask 1.67×、k=4 1.50×）；batch128 307.4/184.3/204.8/83.9 ms。
  - 3090（cuda:7，固化工具复测）：batch16 90.3/83.1/79.1/83.8 ms（pad-mask 1.09×、k=4 1.14×、
    k=4+mask 1.08×）；batch64 278.5/174.9/192.2/85.2 ms（1.59×/1.45×/3.27×）；
    batch128 523.8/319.9/354.3/134.2 ms（1.64×/1.48×/3.90×）。
  - 说明：绝对值随 GPU 型号/占用漂移（4090 vs 3090 约 1.9×），**比值是稳定量**；
    两种卡上 batch16 的 k=4 加速都 ≤1.14×。
- 已核验论文（本场新增）：FlashSign arXiv:2603.27915（代码未开源、无 BLEU/WER）；
  既有核验：CogVideoX 2408.06072、DiT 2212.09748、SD3 2403.03206、PixArt-α 2310.00426、
  Flamingo 2204.14198、BLIP-2 2301.12597、TiTok 2406.07550、W.A.L.T 2312.06662。

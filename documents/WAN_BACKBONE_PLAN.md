# Wan2.1-1.3B 骨干迁移：方案评估 + 完整实施方案

- 日期：2026-10-19
- 分支：`wan-backbone`
- 状态：**待裁决**（未做任何训练、未下载任何权重、未改任何既有代码）
- 作者：agent（评估与策划），决策权在用户
- 关联：THINK `idea-045`、TODO `W-N1`…`W-N8`（Q1–Q5 拍板后再登记 D49/D50）

> 本文所有"查证"结论都标了来源；标 `推断` 的地方是尚未读源码确认的，**动手前必须逐条钉死**（见 §8 的 G4）。

---

## 0. 结论速览

| 问题 | 结论 |
|---|---|
| 这个方向值不值得做？ | **值得**，但理由和原方案说的不完全一样：真正的赌注不是"更大的模型"，而是"Wan 的 cross-attn 带过来「会听文本」这个先验"。见 §2.1 |
| 首选骨干对吗？ | **Wan2.1-1.3B（T2V）选对了**，且我们查到一个原方案没提的**决定性加分项**：它的 patchify 时间维不压缩，正好对上我们"每帧 1 token"。见 §2.2 |
| 原方案最大的漏洞 | **文本塔**。原方案把"用 umT5 还是保留 mT5-base"当成二选一的口味问题——不是。换成 mT5-base 会把我们最想要的那个先验直接毁掉。见 §3.3 |
| 原方案缺少的关键设计 | ① latent 尺度对齐（不做就训不动）② **等容量随机初始化对照**（不做就会自欺，项目已经这样栽过 3 次） |
| 原方案里我说不同意的一点 | "换骨干后所有历史数字作废"——**不成立**。评测侧（冻结 VAE + SignMbart + test 642）一字不改，数字可同表；只有当采样器也换了才不可比。见 §2.4 |
| 最可能失败的地方 | 表示错配：Wan 的块学的是"怎么在它自己 16×8×8×4 的 VAE 隐空间里去掉纹理噪声"，我们的 1024 维/帧是手语专用语义码。这份先验能不能过来，**必须先用 2 GPU·h 证伪**。 |
| 建议的第一步 | **不是下载权重，是 Stage 2 的证伪实验**（冻结块 + 只训 I/O vs 随机块等容量对照） |
| 需要你拍板 | 5 个问题，其中 Q1（x0→velocity 要不要改 RULES 第 3 条）最硬。见 §10 |

---

## 1. 事实核查（哪些是查到的）

### 1.1 Wan2.1-T2V-1.3B 官方 config（**原文，非杜撰**）

来源：`https://modelscope.cn/api/v1/models/Wan-AI/Wan2.1-T2V-1.3B/repo?FilePath=config.json`

```json
{
  "_class_name": "WanModel",
  "_diffusers_version": "0.30.0",
  "dim": 1536,
  "eps": 1e-06,
  "ffn_dim": 8960,
  "freq_dim": 256,
  "in_dim": 16,
  "model_type": "t2v",
  "num_heads": 12,
  "num_layers": 30,
  "out_dim": 16,
  "text_len": 512
}
```

推论（算术，不是猜）：`dim/num_heads = 1536/12 = 128` = head_dim；`in_dim = out_dim = 16` = Wan VAE 的 latent 通道数。

### 1.2 仓库文件与体积（ModelScope `repo/files` API 实读）

| 文件 | 体积 | 我们要不要 |
|---|---|---|
| `models_t5_umt5-xxl-enc-bf16.pth` | 11.362 GB | **要**（文本塔，见 §3.3）；只在离线编码用一次 |
| `diffusion_pytorch_model.safetensors` | 5.676 GB | **要**（DiT 本体）；`5.676 GB / 4 bytes = 1419M` 参数 ⇒ **1.42B 实参** |
| `Wan2.1_VAE.pth` | 0.508 GB | **不要**（我们用自己冻结的 1-token/帧 VAE） |
| 合计 | **17.5 GB** | 净下载 **17.0 GB**（可先只下 DiT 的 5.68 GB） |

### 1.3 我们的接口（新骨干必须对齐）

- `t2v/models/dit.py` `DiT.forward(self, x, t, cond=None, frame_mask=None, self_cond=None) -> Tensor`
  - `x: [B, T, 1024]`（`T` 变长）、`t: [B]`、`frame_mask: [B, T]` bool（True = 有效帧）、返回 `[B, T, 1024]`（`learn_sigma` 时 `[B,T,2048]`）
- `t2v/models/conditioning.py:68` `DiTConditioning`：`context [B,L,D]` / `context_mask [B,L]` / `global_condition [B,D]` / `video_condition [B,D]`
- 装配唯一入口 `t2v/models/factory.py` 的 `build_bundle(config, text_dim=768)` → `ModelBundle(model, condition_encoder, context_dim, use_text, use_ref)`
- **现有 DiT-B 参数量实测 181.1M**（hidden 768 / depth 12）⇒ Wan 是它的 **7.8×**

### 1.4 环境与资源现状（实测）

| 项 | 值 | 影响 |
|---|---|---|
| `onvae` env | torch 2.6.0+cu124 / transformers 5.17.0 / accelerate 1.14.0 / einops 0.8.2 / safetensors 0.8.0 | 评测同源环境；torch 够新 |
| `onvae` 缺 | **diffusers 缺失、peft 缺失** | 必须先装 |
| `gfslt` env | torch 2.1.2+cu121 / **transformers 4.32.0** | 太旧，`UMT5EncoderModel` 不在这个版本里 |
| GPU | 8 张：4×RTX 4090(24G) + 4×RTX 3090(24G) | 单卡 24G，全参微调 1.42B 很紧 |
| 磁盘 `/data` | **53 GB 可用（100% 已用）**；`runs/` 190 GB、`reports/` 37 GB | **硬约束，必须先腾** |
| 网络 | HF 直连**不可达**；`hf-mirror.com` **HTTP 200 可用**；ModelScope API **可用**；PyPI **可用**（diffusers ≤ 0.41.0） | 下载走 ModelScope / hf-mirror |

---

## 2. 我对这个方案的评估

### 2.1 方向对，但赌注要重新表述（这是最重要的一条）

原方案的论证是"我们要一份视觉/动作先验"。**这个论证在我们自己的数据上站不住**，理由是：

- 我们的**错配地板**（真视频配错字幕）是 `B1 9.32 / B4 0.00`，而现役最好臂 `B4 5.65`（`route_t_align_20k_ema_k4`；下文数字沿用立项时的 4.66 口径）。也就是说：**回译分的高低几乎不取决于"视频看起来像不像手语"**（真视频配错文本照样 B4=0），而取决于"视频里有没有那句话的内容"。
- 我们自己的两份诊断把失败点钉死在文本通路上：
  - 内容词召回：真视频回译 ~55%，现役最强臂 **5.7%**（86.3% 的样本召回为 0，跟常数高频词表 1.20% 同量级）。
  - 通道解剖：换掉文本后 cross-attn 输出变化 0.89–1.64，但 block 隐状态只变 4–6%、最终输出只变 ~1%；**cross-attn 的有效贡献只有 MLP 的 3–12%**。

⇒ **我们缺的不是"去噪能力"，是"听文本的能力"。**

那么 Wan 能带来什么？**它的 cross-attn 是在 web-scale 图文/视频对上训出来的、真的会跟随 prompt 的注意力层，而且它的 `text_embedding`（Linear 4096→1536）与 cross-attn 的 K/V 是配套的。** 这是唯一一个我们还没试过、且直接打在失败点上的轴（我们已经在自己的小 DiT 上试穷了"改条件接口 / 改生成结构 / 改损失形状"三条轴，9 个机制全部零收益或显著为负）。

**所以本方案的可证伪假设是：**

> **H-Wan：把 Wan 的块（连同它配套的文本通路）当初始化，能把"文本跟随"这个行为迁移到我们的 latent 空间，从而把内容词召回从 ~6% 抬起来，进而把 B4 从 5.65 推向 10。**

它的**杀手判据**很明确（Stage 2，2 GPU·h）：如果冻结 Wan 块、只训新 I/O 层都不能显著优于"等容量随机块 + 同样只训 I/O"，那"预训练先验可迁移"就是假的，**这条路立刻停，一分钱不再投**。

### 2.2 骨干选型：同意 Wan，并补一条原方案没说的决定性理由

原方案给的"视频骨干 → 动作先验对口"这条理由，按 §2.1 的重新表述其实**不是**主要理由（B4 不靠动作像不像）。真正让我们选 Wan 的是两条：

1. **patchify 的时间维不压缩，天然对上我们的 1-token/帧。** Wan 的 `patch_size = (1, 2, 2)`：时间维 patch 是 **1**，压缩全在空间维。我们只要把空间网格设成 `H'=W'=1`，patch 核就退化成 `(1,1,1)`，**每一帧正好一个 token，一行代码都不用改 patchify 的语义**。相比之下 CogVideoX 的 patchify 在时间维是 2（`(2,2,2)`），我们要么丢一半帧，要么改结构。原方案说这是"打折"，实际上 Wan 在这点上**不打折**——这是选它的第一条硬理由。
2. **umT5-xxl 覆盖德语**，且它的文本嵌入空间正是 Wan cross-attn 期待的那个空间（§3.3）。

（顺带：`_class_name: WanModel` + `_diffusers_version: 0.30.0` ⇒ diffusers 侧有 `WanTransformer3DModel`，我们可以直接用它的 block 实现，省掉自己复刻 30 层的风险。）

### 2.3 原方案里我**同意**的部分

- 继承 transformer 块、重接 I/O。✅
- **不要 latent 适配器**（不要在两边 VAE 的隐空间之间硬塞翻译器）。✅ 我再加强一点：Stage 2 的证伪实验本身就是"最朴素的 I/O 重接"，适配器会把这个实验的因果搞浑。
- LoRA + 新层全参、小步前进、走我们自己的评测协议。✅
- "先 PixArt-α 跑通链路"这条保底建议——**我不同意作为第一步**，理由见 §5 的顺序设计：链路能不能通，用 Wan 自己 5.68 GB 的权重 + CPU 冒烟测试半天就能验，不需要为它多跑一条 0.6B 的臂。

### 2.4 原方案里我**不同意**的部分（两处）

**① "换骨干后所有历史数字作废，要开一张新表重算"——不成立。**

我们的指标链条是：`生成 latent → 冻结 VAE 解码 → SignMbart 回译 → BLEU/ROUGE/WER`，再加上 FVD 和检索 MRR。换骨干只改**生成侧**；解码器、回译模型、测试集、长度口径、seed 全都不动。**所以新旧臂可以放同一张表配对比较**，这正是我们想要的（要和 `route_t_align_20k_ema` 比 B4）。

真正会导致"换口径"的是**采样器**：如果我们连训练目标一起换成 flow matching，采样器就从 DPM-Solver++ 变成 `FlowMatchEuler`（或 UniPC），那一步才是口径变更，必须在 EXPERIMENT 里显式标注，并且**另跑一条"我们的 DiT-B + flow matching"把"换目标"和"换骨干"两个变量分开**。

**② "PixArt-α 先跑通链路、别指望涨点"——顺序不对。**

"跑通链路"的正确代价是 **0.5 天 CPU + 0.5 GPU·h**（读源码 + 加载权重 + 形状冒烟测试），不是多养一条 0.6B 的臂。多一条臂就多一份磁盘（我们只剩 53 GB）、多一份评测队列、多一个要维护的 config，而它给出的信息量约等于零。

### 2.5 原方案**漏掉**的两个致命细节（我补上）

**① latent 尺度不对齐 ⇒ 训练直接不动。**

Wan 的 adaLN 门控是按**它自己 VAE latent 的噪声尺度**校准的。我们的 1024 维 latent 范数量级不同（NOTES 记录 PHOENIX latent 范数 108–181，而 WASL 那种外部数据是 410±301）。直接把我们的 `x_t` 喂进去，模型看到的是 OOD 输入，`time_projection` 出来的 6 路 shift/scale/gate 全错位。

修法（干净、不违规）：用缓存里现成的 `data/phoenix_latents/v1/train_stats.safetensors` 做**固定的逐维标准化**（减均值、除标准差），喂入前归一化、输出后反归一化。这是固定仿射变换，**不引入新监督、不改冻结 VAE 契约、不影响 latent 缓存内容**。

**② 没有等容量对照，一定会自欺。**

项目已经**三次**栽在同一个坑里（FAIL-019 dual4 vs dualctl、FAIL-024 slots vs slotsctl、FAIL-034 u vs v）：免训练探针全 PASS，一加等参数对照就归零。FAIL-024 甚至明确写下"免训练敏感度探针不再作放行依据"。
⇒ 本方案把 **"随机初始化等容量块 + 只训 I/O"** 作为 Stage 2 的强制对照组，**没有它就不许进 Stage 3**。

---

## 3. 架构适配设计（精确到层）

### 3.1 形状映射

```
我们的视频 latent        x: [B, T, 1024]          （T 变长，训练 ≤256，test p90 155 / max 242）

喂进 Wan 之前reshape      [B, 1024, T, 1, 1]       （C=1024, T'=T, H'=W'=1）
       │
       ├─ patch_embedding: Conv3d(1024 → 1536, kernel=(1,1,1), stride=(1,1,1))   ← 新建
       ▼
                          [B, 1536, T, 1, 1] → flatten → [B, T, 1536]
       │
       ├─ + RoPE(grid_sizes=(T,1,1))
       ├─ adaLN 调制（t → time_embedding → time_projection → 6 路 shift/scale/gate）
       ├─ 30 × block：self-attn(带 RoPE) → cross-attn(对文本) → FFN
       ▼
       ├─ norm_out（继承）→ head: Linear(1536 → 1024)                            ← 新建
       ▼
我们的 x0 预测            [B, T, 1024] → 冻结 VAE 解码 → 视频
```

对比原模型：`patch_embedding` 原本是 `Conv3d(16 → 1536, (1,2,2), (1,2,2))`，`head` 原本是 `Linear(1536 → 16×1×2×2 = 64)`。我们只是把通道数从 16 换成 1024、空间核从 2×2 换成 1×1。

### 3.2 继承 / 新建清单

| 组件 | 处理 | 参数量 |
|---|---|---|
| `patch_embedding` | **新建**，随机初始化，全参训练 | 1024×1536+1536 ≈ **1.57M** |
| `time_embedding`（freq 256→1536）+ `time_projection`（1536→1536×6） | 继承 | ≈ 15.2M `推断` |
| `text_embedding`（Linear 4096→1536 ×2 + GELU） | **继承**（前提：文本塔用 umT5-xxl，§3.3） | ≈ 12.6M |
| 30 × block（self-attn / cross-attn / FFN 1536→8960→1536 / norm / modulation） | **继承**（LoRA 或全参，见 Q4） | ≈ **1385M** |
| `norm_out` | 继承 | 3K |
| `head` | **新建**，随机初始化，全参训练 | 1536×1024+1024 ≈ **1.57M** |
| RoPE buffer | 重建（`grid_sizes` 变了） | 0（buffer 不占训练参数） |

⇒ **新建参数仅 ≈ 3.1M**，其余 ≈ 1416M 全部继承。

### 3.3 文本塔：**必须用 umT5-xxl**（这是全案成败的关键，不是口味问题）

Wan 的 cross-attn K/V 是在 **umT5-xxl 的 4096 维嵌入空间**上训出来的，`text_embedding` 是配套的 `Linear(4096→1536)`。

如果换成我们的 mT5-base（768 维）再随机初始化一个新投影层：

- 预训练 cross-attn 的**输入分布被整体打乱**；
- 我们最想要的那个先验（"会听文本"）**恰好就在这一步被毁掉**；
- 结果多半是"继承了 1.4B 参数、却只拿到随机初始化的文本通路"——花了钱买了个大号随机模型。

**所以：umT5-xxl，冻结，离线预计算。**

成本控制设计：
- 我们的德语字幕平均 ~14 词，umT5 分词后 ~25–40 token ⇒ **文本 padding 长度定为 64**（不是 Wan 默认的 512）。
- 缓存体积 = 8257 条（train 7096 + eval 519 + test 642）× 64 × 4096 × 2 bytes ≈ **4.3 GB**。
- ⚠️ umT5 的 pad token **有真实 embedding（不是零向量）**。所以两种做法二选一：① 缓存里把 pad 位填真实 pad 向量；② 缓存真实长度、训练时同时传 `context_mask`（我们已有该接口，DiT 用它屏蔽 cross-attn 的 pad key）。**推荐 ②**，省空间也更贴我们现有实现。

### 3.4 RoPE 的已知折扣（说清楚，不藏）

Wan 的 3D RoPE 把 `head_dim = 128` 切成 t/h/w 三段（`推断` 约 44/42/42，**动手前必须读源码确认**）。当 `grid_sizes = (T, 1, 1)` 时：

- h/w 两个网格的位置索引**恒为 0** ⇒ 对应的旋转角 = 0 ⇒ `cos=1, sin=0` ⇒ **旋转矩阵是恒等变换**；
- 于是整条 RoPE **退化成纯一维时间 RoPE**，但**只有约 44/128 ≈ 34% 的旋转子空间在工作**，剩余 66% 恒等。

这不是 bug，因为 Wan 的时间维 patch = 1，它的 t 段本来就肩负"帧序"这件事；但**有效位置编码维度只有 1/3** 是一个真实折扣，记在风险里（§9 R4）。

### 3.5 输出目标：x0 还是 velocity

见 §4。默认 **x0**（合规 + 口径干净）。

---

## 4. 训练口径：x0 vs velocity（RULES 第 3 条的正面冲突）

RULES 方法硬约束第 3 条：**"必须使用 x-prediction（预测 x0）"**。Wan 是 rectified-flow 训出来的，原生目标是**速度** `v`。

三者的数学关系（用户原文已给，我确认无误）：`x_t = (1-t)ε + t·x0`、`v = x0 - ε`，且 `x0 / ε / v` 三者线性等价（知道任意一个，另两个可算）。所以"预测什么"本质是**换坐标**。真正的差别只有三条：

| | 我们的现状 | flow matching |
|---|---|---|
| 监督目标 | x0-MSE | velocity-MSE |
| 路径 | cosine 加噪曲线（弯的） | 直线插值 |
| 采样器 | DPM-Solver++(2M) | FlowMatchEuler / UniPC |

**这是实质差别，不是装饰**：velocity-MSE 对所有 t 等权，x0-MSE 在低 t（噪声大）处权重极大。而 exp-064 实测我们**当前 t≥500 的段占了 loss 权重的 97.94%、t=950 单独 75.35%** —— 说明"用哪个坐标"直接决定模型把力气花在哪。所以这个选择不能顺手改，必须预注册。

三个选项与建议：

| 选项 | 做法 | 采样器 | 与历史臂可比？ | 建议 |
|---|---|---|---|---|
| **W-x0** | 保留我们的 GaussianDiffusion（cosine + DPM++ + x0 目标），**只换主干** | DPM++ 50（不变） | ✅ **完全可比** | **首选第一臂** |
| W-fm | 换成 rectified-flow 目标 + `FlowMatchEuler` | 变了 | ❌ 需同表标注 + 另跑 DiT-B+FM 对照分离变量 | 预注册第二臂 |
| W-fm-x0head | 主干按 FM 语义用，head 吐 x0，采样时 `v=(x0-x_t)/(1-t)` 换算 | DPM++ | ⚠️ 目标合规但采样器与直线路径不匹配 | 不推荐 |

**建议：先跑 W-x0。** 理由：① 字面合规，不用改 RULES；② 采样器不变 ⇒ 与 `route_t_align_20k_ema` 的配对最干净；③ 出数最快。是否允许把 RULES 第 3 条改成"x0 或 velocity 二选一，须在 config 显式声明"，**交你拍板（Q1）**；无论是否改，W-fm 都作为预注册的第二臂，用来分离"换目标"和"换骨干"两个变量。

---

## 5. 分阶段执行路线（含证伪实验与判据）

### Stage 0 · 环境与可行性（0.5 天，0 GPU）

| 步骤 | 产物 | 判据 |
|---|---|---|
| G1 装 `diffusers`(≥0.33) + `peft` | 环境可 import | `WanTransformer3DModel` 可实例化 |
| G2 下载权重：先只下 DiT 5.68 GB，再下 umT5-xxl 11.36 GB | `/data/huyang/models/Wan2.1-T2V-1.3B/` | sha 校验通过 |
| G3 写 `tools/convert_wan_weights.py`：官方 safetensors → diffusers 命名 → **bf16 落盘** | `wan_1_3b_dit_bf16.safetensors`（≈2.84 GB） | 转换后 `load_state_dict` 无 missing/unexpected |
| G4 **读源码钉死三处**：`WanRotaryPosEmbed` 的 t/h/w 维度切分、`WanTransformerBlock` 的 modulation 通路（6 路还是 2 路 ×3）、patchify 的 reshape 顺序 | `.trae/documents/WAN_ARCH_NOTES.md` | 每一处都写清"文件:行 + 原文片段"；**不许凭记忆** |
| G5 写 `tools/cache_umt5_embeddings.py`（冻结 umT5-xxl，64 token，存 bf16 + 真实长度） | `data/text_emb_umt5/v1/{train,eval,test}.pt` ≈ 4.3 GB | 抽样解码回文本与原文一致 |

**门槛**：G4 不完成不许写模型代码。

### Stage 1 · 形状与接线冒烟（0.5 GPU·h）

- 实现 `t2v/models/wan_backbone.py`：`WanBackbone.forward(x, t, cond, frame_mask, self_cond)`——**与 `DiT.forward` 同签名**，这样 `factory.build_bundle` 只多一个分派分支，训练/推理/评测三条链路零改动。
- `configs/*.yaml` 新增 `model.backbone: dit|wan`（默认 `dit`，**保证老配置一字不变**）。
- 加 `&latent_stats` 的标准化/反标准化（§2.5 ①）。
- 单测 `tests/test_wan_backbone.py`：
  - 前向形状 `[B,T,1024] → [B,T,1024]`；
  - **变长 T 不变性**：同一 batch 里 T 不同的样本，单独跑与拼批跑结果一致（容差内）；
  - **`frame_mask` 不变量**：改 pad 区的值，非 pad 位置的输出不变；
  - 参数量 == 1419M ± 1%；新 I/O 层梯度非零、继承块梯度（LoRA 时）非零。

**判据**：这四条全过。**过不了不进 Stage 2。**

### Stage 2 · 【关键证伪】冻结块 + 只训 I/O（2 GPU·h）——**这一步决定整案生死**

| 臂 | 配置 | 说明 |
|---|---|---|
| `wan-io` | Wan 预训练块**冻结**，只训 `patch_embedding`+`head`（3.1M） | 被试 |
| `wan-io-rand` | **等容量随机初始化 30 层**，同样只训 I/O | **强制对照**（§2.5 ②） |

- 预算：各 2k 步、bs 16、lr 1e-4、同一 seed 集合、同一数据顺序。
- 判据（**预注册，不许事后改**）：
  1. `wan-io` 的**验证 x0-MSE 必须显著低于 `wan-io-rand`**（按帧配对，p < 0.01）；
  2. 且相对优势 ≥ 5%（效应量下限）。
- **失败即终止**：两条任一不满足 ⇒ "预训练先验可迁移"证伪，写 FAIL，整条 Wan 路线关闭，**后续预算全部省下**。

> 这个设计的价值：它是整个方案里**唯一能在 2 GPU·h 内杀死它的实验**，所以必须排在所有花钱的事情前面。这也正是项目三次栽跟头（FAIL-019/024/034）本该有而没有的东西。

### Stage 3 · 正式臂：LoRA + 新 I/O，20k 步（8–12 GPU·h）

- 起点：Wan2.1-1.3B 预训练权重 + 新 I/O；LoRA rank 64 挂在 self-attn / cross-attn 的 q,k,v,o 与 FFN（是否连块一起全参微调见 Q4）。
- 预算对齐 `route_t_align_20k`：20k 步、bs 16、lr 新层 1e-4 / LoRA 1e-5、bf16、grad_clip 1.0、EMA 0.9999。
- 文本：umT5-xxl 冻结 + 离线缓存（§3.3）。
- 评测：**标准协议**（test 642 / dpm++(o2) 50 NFE / CFG 2.0 / gate 1.0 / oracle 长度 / seed 42），同时出 **raw 与 EMA** 两档（D46）。
- 判据（预注册）：
  1. 与 `route_t_align_20k_ema` 按 video_id 配对，**ΔB4 > 0 且 p < 0.05**；
  2. **ΔWER 不显著退化**（D20）；
  3. **MRR 不作放行条件**（MRR 涨而回译跌已复现 4 次）。
- 附带必测的诊断（都是现成工具）：内容词召回（`tools/probe_retrieval_caption_overlap.py`）、文本交换 gap、cross-attn 贡献占比。**这三个决定我们要不要继续投 Stage 4**——如果 B4 涨了但召回没涨，说明涨的来路和我们的假设不同，必须先解释清楚。

### Stage 4 · 归因与扩展（视 Stage 3 结果）

- 若有效，拆变量：跑 `wan-mt5`（把文本塔换回 mT5-base + 新投影）——用来验证 §3.3 的判断（"umT5 是必要的"）。
- 若有效，试 W-fm（§4）看直线路径是否再加一档。
- 若无效，把"死在哪一环"写进 FAIL：文本通路没过来？表示错配？RoPE 折扣？还是容量？

---

## 6. 强制对照组设计（缺一不可）

| # | 对照 | 回答什么问题 | 在哪一步 |
|---|---|---|---|
| C1 | 等容量**随机初始化**块 + 训 I/O | 涨点来自"预训练先验"还是"新 I/O 层的容量"？ | Stage 2（**硬门槛**） |
| C2 | 我们的 DiT-B **加宽/加深到 1.4B 等参数量** | 涨点来自"骨干先验"还是"参数变多"？ | Stage 4（可选，但若 Stage 3 有效则必做） |
| C3 | 同配置 **≥3 个生成 seed** | 数字是不是种子噪声？ | Stage 3 起，按 TODO 11 的既有纪律 |

> C2 的注意点：exp-031 已证**盲目加宽/加深是显著亏的**（`route_l_wide1024_30k` B4 0.49、`route_m_deep16_30k` B4 0.88）。所以 C2 的用途**只是"等参数量对照"**，不是"更大更好"的尝试；如果 C2 都训不动，那"涨点来自参数"这条解释也自动被削弱。

---

## 7. 成本预算

| 项 | 量 | 说明 |
|---|---|---|
| 下载 | 17.0 GB（可先 5.68 GB） | 走 ModelScope / hf-mirror |
| 文本嵌入缓存 | 4.3 GB | 64 token 口径 |
| 权重（bf16） | 2.84 GB/份 | 训练中另存 raw + EMA 各一份 |
| 训练显存 | LoRA+新层：1.42B×(2+2) + Adam(仅可训部分) ≈ **3–6 GB + 激活** ⇒ 24 GB 单卡宽裕 | 短序列（≤256 token）激活很小 |
| 训练显存（若全参微调块） | 1.42B × 8 bytes(权重+梯度+Adam) ≈ **11.4 GB + 激活** ⇒ 24 GB **很紧**，需梯度检查点或 ZeRO | 见 Q4 |
| GPU·h | Stage 1 0.5 + Stage 2 2×2 + Stage 3 8–12 + Stage 4 6–8 | **合计 ≈ 20–30 GPU·h**（不含 C2 与 W-fm） |
| 墙钟 | 按 8 卡里 3–4 张可用算 | 2–3 天 |
| **磁盘净需求** | **≥ 30 GB**，现在只有 53 GB 且评测队列还在写 | **必须先清 `runs/`**（190 GB，多数臂已出结论） |

---

## 8. 前置工程清单（可勾选）

- [ ] **W-N1** 腾磁盘：清 `runs/` 里已出结论且未再引用的臂，目标 ≥ 30 GB 净空（等当前 15 条评测队列跑完再做，避免删到正在读的 checkpoint）
- [ ] **W-N2** 装 `diffusers`(≥0.33) + `peft`（环境见 Q2）
- [ ] **W-N3** 下载 Wan DiT（5.68 GB）+ umT5-xxl（11.36 GB）到 `/data/huyang/models/`
- [ ] **W-N4** 读源码钉死 RoPE / modulation / patchify（G4），产出 `WAN_ARCH_NOTES.md`
- [ ] **W-N5** `tools/convert_wan_weights.py` → bf16
- [ ] **W-N6** `tools/cache_umt5_embeddings.py` → 4.3 GB 缓存
- [ ] **W-N7** `t2v/models/wan_backbone.py` + `factory.py` 分派 + `tests/test_wan_backbone.py`
- [ ] **W-N8** `configs/wan_b_1280_20k.yaml`（含 `model.backbone: wan`、`latent_stats` 路径、`text_emb_path`、`prediction_type`）

---

## 9. 风险登记（按严重度排序）

| # | 风险 | 表现（怎么认出来） | 缓解 |
|---|---|---|---|
| **R1** | **表示错配**：Wan 块学的是它自己 VAE 隐空间的去噪 | Stage 2 的 `wan-io` 打不过 `wan-io-rand` | **就是 Stage 2 要杀的**；杀不出来就停 |
| **R2** | **文本塔接错** ⇒ 先验全毁 | cross-attn 输出对换文本几乎不敏感（复用现有通道探针） | 强制用 umT5-xxl，不换 mT5 |
| **R3** | **latent 尺度不匹配** | 训练 loss 一开始就爆 / 完全不降 | `train_stats` 固定标准化（§2.5 ①） |
| **R4** | RoPE 有效子空间只剩 ~34% | 长视频（>128 帧）位置的输出质量明显劣于短片 | 记录在案；必要时只对 t 段重标定 |
| **R5** | 磁盘不足 | 队列 HOLD / 写 checkpoint 失败 | W-N1 先腾 |
| **R6** | 环境冲突（diffusers/peft 缺、transformers 版本） | import 失败 / UMT5 类不存在 | Q2 定环境；建议 onvae（torch 2.6 + transformers 5.17） |
| **R7** | **容量解释**（自欺） | 无 C1 对照就宣布成功 | C1 是硬门槛 |
| **R8** | 口径混淆 | 换了采样器却和历史臂同表比 | W-x0 首选；W-fm 必须显式标注 + 加 DiT-B+FM 对照 |

---

## 10. 需要你拍板的开放问题

| # | 问题 | 我的建议 |
|---|---|---|
| **Q1** | 是否允许把 RULES 第 3 条从"必须 x0"放宽为"x0 或 velocity，须在 config 显式声明"？ | **先不动 RULES，第一臂走 W-x0**；W-fm 作为预注册第二臂，等 W-x0 出数再决定要不要改 |
| **Q2** | 训练落在哪个环境？ | **onvae**（torch 2.6.0 + transformers 5.17 + 评测同源），避免 gfslt 的 transformers 4.32 太旧 |
| **Q3** | 是否授权清理 `runs/`（190 GB）腾出 ≥30 GB？ | **是，但等 15 条评测队列跑完**，且只清已出结论的臂；我给清单你先过目 |
| **Q4** | 块用 LoRA 还是全参微调？ | **Stage 2/3 用 LoRA rank 64**（便宜、快、失败面小）；Stage 4 若有效再试解冻 |
| **Q5** | 首次投入上限？ | **Stage 0 + Stage 1 + Stage 2 ≈ 3 GPU·h**，拿到 Stage 2 判据再决定要不要投 Stage 3 的 8–12 GPU·h |

---

## 11. 与知识库的对接

- 本方案 → THINK `idea-045`（L3，分层：换骨干先验）；Stage 2 的判据写进卡的"最小证伪实验"
- 工程条目 → TODO `W-N1`…`W-N8`
- 若 Q1 同意放宽 → 另立 RULES 第 3 条修订 + 决策登记 `D50`
- 本文件为**过程类文档**，落在 `.trae/documents/`；若最终验证有效，结论进 `NOTES.md`，无效进 `FAIL.md`

---

## 附：一句话总结

> **选 Wan2.1-1.3B 是对的，真正的赌注是"把 Wan 的文本通路先验搬到我们的 latent 空间"；这件事能不能成，2 GPU·h 就能证伪，所以先做证伪，再谈下载 17 GB 和花 30 GPU·h。**

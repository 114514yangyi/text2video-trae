# Wan2.1-1.3B 架构钉死笔记（W-N4 产出）

> 规则：**每一处都必须写清"文件:行 + 原文片段"，禁止凭记忆。**
> 全部证据来自本机已安装的 `diffusers` 源码，路径记为
> `$SP = /data/huyang/miniconda/envs/onvae/lib/python3.12/site-packages/diffusers/models/transformers/transformer_wan.py`
> 与下载下来的权重 `config.json` / `index.json`。
> 版本：diffusers **0.41.0**（`python -c "import diffusers; print(diffusers.__version__)"`）。

---

## 0. 权重配置（实测，非记忆）

`.models/Wan2.1-T2V-1.3B-Diffusers/transformer/config.json`：

| 字段 | 值 | 对我们的意义 |
| --- | --- | --- |
| `patch_size` | `[1, 2, 2]` | **时间维 patch 恒为 1** ⇒ 空间网格设 1×1 后 patch 退化成 (1,1,1)，每帧一个 token |
| `num_attention_heads` | `12` | |
| `attention_head_dim` | `128` | inner_dim = 12×128 = **1536** |
| `num_layers` | `30` | |
| `ffn_dim` | `8960` | |
| `text_dim` | `4096` | umT5-xxl 的 d_model |
| `freq_dim` | `256` | 时间正弦嵌入宽度 |
| `in_channels` / `out_channels` | `16` / `16` | **我们要改成 1024 / 1024**（1 token/帧 × 1024 维） |
| `cross_attn_norm` | `true` | ⇒ `norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True)` |
| `qk_norm` | `"rms_norm_across_heads"` | q/k 在 head 维上做 RMSNorm |
| `eps` | `1e-6` | |
| `rope_max_seq_len` | `1024` | 需要 ≥ 我们最大帧数（256），够用 |
| `image_dim` / `added_kv_proj_dim` | `null` | 纯 T2V，无 I2V 分支 ⇒ **cross-attn 的 K/V 就是 `to_k` / `to_v`**，没有 `add_k_proj` |

权重文件是**分片**的：`transformer/diffusion_pytorch_model-0000X-of-0000Y.safetensors`
+ `diffusion_pytorch_model.safetensors.index.json`（825 个键）。

---

## 1. RoPE 把 head_dim=128 切成 t/h/w 三段的确切维数（G4 问题①）

`WanRotaryPosEmbed.__init__`（`$SP:363-368`）**原文**：

```python
self.attention_head_dim = attention_head_dim
self.patch_size = patch_size
self.max_seq_len = max_seq_len

h_dim = w_dim = 2 * (attention_head_dim // 6)
t_dim = attention_head_dim - h_dim - w_dim

self.t_dim = t_dim
self.h_dim = h_dim
self.w_dim = w_dim
```

代入我们的 `attention_head_dim = 128`：

- `h_dim = w_dim = 2 * (128 // 6) = 2 * 21 = 42`
- `t_dim = 128 − 42 − 42 = 44`

⇒ **三段宽度 `(t, h, w) = (44, 42, 42)`，合计 128**（**不是**均匀三分，也不是 3/8-3/8-2/8）。

每段各自调 `get_1d_rotary_pos_embed(dim, max_seq_len, theta, use_real=True, repeat_interleave_real=True)`
（`$SP:380-389`，`theta=10000.0`），返回宽度恰为 `dim` 的 cos/sin（`repeat_interleave_real=True`
把 `dim//2` 个频率重复插值铺满 `dim`，这样下游 `[..., 0::2]` / `[..., 1::2]` 才能配对），
再 `torch.cat(freqs_cos, dim=1)` 拼成 `[max_seq_len, 128]` 并注册为**非持久化 buffer**
（`persistent=False` ⇒ **不在权重文件里，是现算的**）。

`WanRotaryPosEmbed.forward`（`$SP:396-418`）按 `split_sizes = [t_dim, h_dim, w_dim]` 切回三段，
再各自按 t/h/w 的位置展开：

```python
split_sizes = [self.t_dim, self.h_dim, self.w_dim]
freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
freqs_cos_f = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
...
freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
return freqs_cos, freqs_sin
```

⇒ 返回**元组 `(freqs_cos, freqs_sin)`，形状 `[1, N, 1, 128]`**（N = 帧数 = ppf·pph·ppw）。
第 3 维是 **1** 而不是 head 数 12 —— 这是为了跟 `apply_rotary_emb` 里的
`[B, N, heads, head_dim]` 广播（见第 5 节）。

**对我们的直接后果（本臂的真实"时间位置编码"只有 22 对频率）**：
把空间网格设成 `H=W=1` 后 `pph = ppw = 1`，于是
`freqs_cos_h` / `freqs_cos_w` 各自只取第 0 行并 `expand` 到所有位置 ——
**h/w 两段（42+42 = 84 维）在所有帧上取值完全相同**，等价于对那 84 维做一次
与位置无关的常量旋转。真正随帧号变化的只有 **t 段 44 维 = 22 个旋转对**
（在 `[..., 0::2]` 抽样后是 64 个槽位里的前 22 个）。

这不是缺陷 —— 项目自己的探针（FAIL-031）已显示帧间表示几乎不随时间变化
（cross-attn 帧间余弦 0.9887）；Wan 的时间 RoPE 恰好补的是我们最缺的那一维。
**但它是必须记录在案的架构事实**：本臂的时间位置编码是 22 个频率对，
不是 64 个；这也意味着如果后续想做"帧序"相关消融，可动的旋钮只有 t 段。

---

## 2. `WanTransformerBlock` 的 modulation 通路：6 路 shift/scale/gate 怎么拆（G4 问题②）

`WanTransformerBlock.forward`（`$SP:161-193`）第一行：

```python
shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
    self.scale_shift_table + temb.float()
).chunk(6, dim=1)
```

`scale_shift_table` 是 `nn.Parameter(torch.randn(1, 6, dim))`（`$SP:160`），
`temb` 来自 `time_proj` 的 `dim*6` 输出经 `unflatten(1, (6, -1))`（`$SP:676`）。
拆分顺序**从左到右依次是**：

| 序号 | 名字 | 作用位置 | 用途 |
| --- | --- | --- | --- |
| 1 | `shift_msa` | self-attn 前的 `norm1` | adaLN 平移 |
| 2 | `scale_msa` | self-attn 前的 `norm1` | adaLN 缩放 |
| 3 | `gate_msa` | self-attn 输出 | 残差门控 |
| 4 | `c_shift_msa` | **FFN** 前的 `norm3` | adaLN 平移（名字带 `c_` 但用在 FFN，**不是** cross-attn） |
| 5 | `c_scale_msa` | **FFN** 前的 `norm3` | adaLN 缩放 |
| 6 | `c_gate_msa` | **FFN** 输出 | 残差门控 |

三条前向（`$SP:170-191`）：

```python
# self-attention
norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

# cross-attention（**没有 adaLN**，只有一层 FP32LayerNorm）
norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
hidden_states = hidden_states + self.attn2(norm_hidden_states, encoder_hidden_states, None, None)

# feed-forward
norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
ff_output = self.ffn(norm_hidden_states)
hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
```

**两个必须记住的坑**：

1. **cross-attn 没有门控、也没有 adaLN**：`hidden_states + self.attn2(...)` 是**无门控直连**。
   这与我们自己的 `DiTBlock` 完全不同（那边 cross-attn 有一个**零初始化**的 `gate_xa`，
   所以训练初期 cross-attn 完全不起作用）。Wan 的 cross-attn 一上来就满强度生效 ——
   这正是我们想要的"文本通路先验"，但它也意味着**换骨干后文本信息第一次真正进入主干**，
   若评测结果仍与文本无关，就不能再归因于"通路被门控压住了"。
2. **`attn1` 和 `attn2` 的调用都多传了一个 `None`**：`self.attn1(norm, None, None, rotary_emb)` 的
   第 3 个 `None` 是 `attention_mask`。**stock 代码从不传非空 mask** ⇒ 文本 pad token 与
   帧 padding 都会参与注意力。我们自己的 `_block_forward` 就是在这里补上 mask 的
   （`t2v/models/wan_backbone.py`），这是**有意偏离 stock**的第一处。

---

## 3. patchify 的 reshape 顺序（G4 问题③）

`WanTransformer3DModel.forward`（`$SP:647-663`）：

```python
hidden_states = hidden_states.unsqueeze(2) if hidden_states.ndim == 4 else hidden_states
# ...
p_t, p_h, p_w = self.config.patch_size
post_patch_num_frames = num_frames // p_t
post_patch_height = height // p_h
post_patch_width = width // p_w

rotary_emb = self.rope(hidden_states)
hidden_states = self.patch_embedding(hidden_states)
hidden_states = hidden_states.flatten(2).transpose(1, 2)
```

即：`[B, C, F, H, W]` --Conv3d(patch_size, stride=patch_size)--> `[B, inner, F/p_t, H/p_h, W/p_w]`
--> `flatten(2)` 把后三维压平 --> `transpose(1,2)` 得 **`[B, N, inner]`**，N = 帧×高×宽。

**我们这档**：`patch_size=(1,1,1)`、输入 `[B, 1024, T, 1, 1]`
⇒ `[B, 1536, T, 1, 1]` ⇒ `flatten(2)` = `[B, 1536, T]` ⇒ `transpose` = `[B, T, 1536]`。
**token 顺序就是帧顺序**（T 是唯一的序列维），无需任何 permute。✔

`proj_out` 侧同理（`$SP:686-694`）：

```python
hidden_states = self.proj_out(hidden_states)
# unpatchify
hidden_states = hidden_states.reshape(batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1)
hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
```

我们**不用 stock 的 `proj_out` + unpatchify**：把 `out_channels` 设成 1024 后
`nn.Linear(1536, 1024 * 1*1*1)` 直接输出 `[B, T, 1024]`，unpatchify 对 1×1×1 是恒等，
所以我直接取 `proj_out(hidden)` 的 `[B, T, 1024]`，不做 reshape/permute。

---

## 4. 时间步量纲：为什么传 `σ × 1000`

`WanTimeTextImageEmbedding.__init__`（`$SP:493-499`）：

```python
self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, out_channels=dim)
self.act_fn = nn.SiLU()
self.time_proj = nn.Linear(dim, time_proj_dim)
```

`Timesteps`（`diffusers/models/embeddings.py`）内部调
`get_timestep_embedding(timesteps, num_channels, flip_sin_to_cos=True, downscale_freq_shift=0)`，
其频率是按 **timestep ∈ [0, 1000]** 设计的（`max_period=10000`）。
Wan 官方 pipeline 的时间步也确实落在 [0, 1000]（`scheduler_config.json` 里
`num_train_timesteps: 1000`、`timestep_spacing: "linspace"`）。

⇒ 我们的 σ ∈ [0,1] 必须乘 1000 再送进去：`t = (σ * 1000).to(dtype)`。

**顺序方向**（Wan 官方 `wan/train.py` 的约定，与 `scheduler_config.json` 的
`prediction_type: "flow_prediction"` 一致）：

```
σ = 1 → 纯噪声；σ = 0 → 纯数据
x_σ = σ·ε + (1 − σ)·x₀            （代码里是 latents = t*noise + (1-t)*latents）
target = ε − x₀                   （代码里是 target = noise - latents）
```

⚠️ **`scheduler_config.json` 用的不是最朴素的 Euler**：

```json
{"_class_name": "UniPCMultistepScheduler", "prediction_type": "flow_prediction",
 "use_flow_sigmas": true, "flow_shift": 3.0, "solver_order": 2, "predict_x0": true,
 "num_train_timesteps": 1000, "timestep_spacing": "linspace"}
```

即官方**推理**用 UniPC(2 阶) + **flow-shift = 3.0** 重参数化 sigma 轴。
我们的 `tools/sample_wan_flow.py` 用 **1 阶 Euler + 同样的 shift=3.0**，
并把这个选择写进 `meta.json`（`"sampler": "euler-flow"`, `"flow_shift": 3.0`），
以便后续单独做"采样器"这条消融时口径可追溯。**训练**侧 Wan 官方是 σ ~ U(0,1) 不加 shift，
我们保持一致。

---

## 5. 注意力布局与 mask 广播（写 mask 前必须确认的第三件事）

`WanAttention.forward`（`$SP:273-284`）：

```python
def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None, **kwargs):
    return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, rotary_emb, **kwargs)
```

`WanAttnProcessor.__call__` 里 q/k/v 的构造（`$SP:98-102`）：

```python
query = query.unflatten(2, (attn.heads, -1))
key = key.unflatten(2, (attn.heads, -1))
value = value.unflatten(2, (attn.heads, -1))
```

关注的是 `unflatten(**2**)` —— 说明输入是 `[B, N, inner]`，拆完是 **`[B, N, heads, head_dim]`**
（**不是** SDPA 惯例的 `[B, heads, N, head_dim]`）。这解释了为什么 `freqs_cos` 的形状是
`[1, N, 1, head_dim]`：它要跟 `[B, N, heads, head_dim]` 逐维对齐广播。

`attention_mask` 一路传进 `dispatch_attention_fn(..., attn_mask=attention_mask, ...)`（`$SP:140-150`）。

⇒ 在**我们的自定义 block 前向**里，mask 形状取 **`[B, 1, 1, S]`**（S = key 轴长度）：
第 2/3 维留 1，与 `[B, N, heads, S]` 广播时对齐的是 `N` 与 `heads` 两个位置 ——
**这两个位置恰好都需要广播**（N 相同则等值广播；heads 必须为 1 才能广播）。
`tests/test_wan_backbone.py::test_frame_mask_blocks_padding_leakage` 与
`::test_context_mask_blocks_pad_text` 就是这个假设的实测判据
（改 pad 位置数值，有效位置输出必须逐比特不动）。

---

## 6. `_keep_in_fp32_modules`：为什么 bf16 训练不会精度崩

`$SP:563-568`：

```python
_keep_in_fp32_modules = ["rope", "time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
_skip_layerwise_casting_patterns = ["rope", "patch_embedding", "condition_embedder", "norm"]
```

⇒ adaLN 的调制量表、所有 LayerNorm、时间嵌入在 bf16 训练里**始终以 fp32 参与**，
块内每次用到它们都显式 `.float()` 再 `.type_as(hidden_states)` 转回去。
我们在 `tools/train_wan_flow.py` 里用 **fp32 权重 + `torch.autocast(bf16)`**
（而不是把权重整体转 bf16），与此完全一致，且可训练参数（LoRA + 新 I/O）保持 fp32 主权重，
不需要 GradScaler。

---

## 7. 一句话总结（给未来的自己）

**Wan 的 30 层块可以直接吃我们 `[B, T, 1024]` 的 latent，只需要把
`in/out_channels` 换成 1024、`patch_size` 换成 (1,1,1)、把 σ 乘 1000、
再用 `ε − x₀` 当目标。** 唯一需要我们自己补的是**两张 attention mask**
（文本 pad、帧 pad）——stock 代码根本不给这两处传 mask。

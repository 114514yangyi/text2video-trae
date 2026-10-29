# CONTEXT — 里程碑1 术语表（ubiquitous language）

> 本文件只放**词**，不放计划。计划见 `.trae/documents/MILESTONE1_ARCH_PIXART_20261022.md`。
> 规则：出现新术语先在这里定名，再写进代码与文档；一个词只允许一个意思。
> 生效范围：分支 `milestone1-pixart`（2026-10-22 起）。旧世界的词（text_prefix / expert_adaln /
> as_modulation / slots / REPA / semantic matcher）**不在本表**，也不要再出现在新代码里。

## 实验结构

| 词 | 定义 | 不要混用 |
| --- | --- | --- |
| **臂（arm）** | 一次完整训练+评测的配置，编号 E1–E10 | 「实验」在旧文档里既指 idea 又指 run，本表统一叫臂 |
| **因子（factor）** | 可独立取值的实验维度，共 5 个，编号 1–5 | 不是「超参」：因子是实验要扫的，超参是固定不变的 |
| **base** | 基准臂 E1 = `1a + 4c + 2a + 3b + 5b` | 不是「基线模型权重」，本项目的 base 也是一次训练 |
| **单因子改动** | 某臂相对 base 只改一个因子；唯一例外是 E9（1c+3a，探交互） | — |

## 条件与注入

| 词 | 定义 |
| --- | --- |
| **条件（conditioning）** | 文本（mT5 编码的德语字幕）+ 参考帧 frame 0（`ref_features [128,512]`） |
| **全局条件向量 `c`** | adaLN 用的每样本向量。**里程碑1 里 `c = timestep 嵌入` 一项**，不含长度、不含任何池化条件 |
| **cross-attn 通路** | 条件 token 作为 K/V，帧 token 作为 Q：每帧主动查询条件 |
| **context 通路（前缀注入）** | 条件 token 直接拼进 self-attn 的序列（作为前缀 token），与帧 token 同一条注意力 |
| **注入计划（injection plan）** | 一张逐层表：`layer_index → {none, text, ref, ref+text}`，决定每个 cross-attn 层看见什么。是配置项，不是代码分支 |
| **ConditionBag** | 传给骨架的条件容器：`{text_tokens [B,120,D], text_mask, ref_tokens [B,128,D], ref_mask}`。骨架只认这个接口 |
| **2a / 2c 的层分配** | 注入计划的两个特例：2a 各层只看见一种；2c 的 ref 层看见 `[ref; text]` 拼接 |
| **ref/text 拼接顺序** | 恒为 **text 在前、ref 在后**（与旧 `ConditionEncoder` 一致，mask 定位依赖该顺序） |

## 模型与扩散

| 词 | 定义 |
| --- | --- |
| **x-prediction** | 网络输出干净 latent `x0`，不是噪声 ε、不是速度 v |
| **latent 契约** | 冻结 TiTok-CVAE 产出 `[T,1024]`（每帧 1 token）；参考帧为 `[128,512]`；缓存**不含 frame 0** |
| **PixArt-α 骨架** | DiT-XL/2 形状的 transformer：N×(self-attn + cross-attn + MLP) → unpatchify，adaLN 调制 |
| **per-frame 投影** | 本项目用 `Linear(1024→D)` 把每帧 latent 变成一个 token，**替代** PixArt 原来的 2×2 patchify（latent 已裁决为 1024 维 1-token） |
| **前缀注入（1b 的 context 通路）** | 条件 token 拼在 latent 帧 token 前面，进同一条 self-attn 序列（不是 cross-attn） |
| **注入计划表** | 一个长度 28 的数组，每层取值 ∈ {none, text, ref, ref+text}，决定该层 cross-attn 的 K/V 来自谁；默认 `[T,T,T,R]×7`（Q4） |
| **预训练初始化（5a）** | 把 PixArt-α 权重灌进形状能对上的层；**灌了哪些、跳过哪些必须写进 config** |
| **motion loss** | 在 latent 上对 `pred_x0` 做 k 步差分（k∈{1,2,4}）的 masked MSE，实现 `t2v/models/dynamic_loss.py` |

## 评测

| 词 | 定义 |
| --- | --- |
| **官方 FVD** | I3D Inception-v1（Kinetics-400）**400-d logits** 特征的 Fréchet 距离；实现 = `third_party/video_metric/calculate_fvd.py`（`method='styleganv'`，2026-10-27 起），入口 `tools/eval_video_metric_fvd.py` |
| **24 帧口径** | **从头抽取 24 帧**（2026-10-22 裁决，第二轮改为从头）：每条视频取 frame 0→23，官方 I3D 模型与代码不变；不足 24 帧的样本跳过，**有效 N ≈613** 且 id 清单跨臂共用；与历史 16 帧表**不可混** |
| **有效 N** | 24 帧口径下因短片被跳过而剩下的评测样本数，报表必须写明；标题里的 640 是 16 帧口径的数字 |
| **骨架超参** | PixArt-α 原尺寸：**28 层 / 1152 宽 / 16 head**（2026-10-22 裁决 Q8） |
| **文本桥接投影** | mT5 输出 768 维 → 骨架 1152 维的 `Linear + LayerNorm`，**可训练**，10 个臂共有（Q3） |
| **4c 训练范围** | `mT5 embedding + encoder.layer[0:3]`（"前 3 层"含 embedding，Q7）；其余层丢弃 |
| **长度规则** | 本里程碑**不做长度预测**：训练用样本自身长度，评测/生成用 **GT 长度**（2026-10-22 裁决） |
| **真值本底** | `real_split_half_fvd`：真值自身半分的 FVD，模型分数低于它要先怀疑管线 |

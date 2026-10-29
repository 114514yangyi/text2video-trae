# DESIGN — 目标与设计理念

> ## ⚠️ 2026-10-22 里程碑1 变更（本文件第 1 节以后为历史设计，仅作背景）
>
> 新方向：**backbone 换成 PixArt-α**、**全局条件 c 只保留 timestep**（不再有长度/池化文本/
> 池化参考帧进 c）、**10 个臂只扫条件注入方式**及其四个配套因子。
> 权威规格：`.trae/documents/MILESTONE1_ARCH_PIXART_20261022.md`。
> 北极星指标更新为 **最小化官方 I3D-FVD**（`RULES.md` 研发探索规范第 19 条，
> 回译 BLEU-4 已废止）；本里程碑按 **24 帧**口径评测（与历史 16 帧表不可混）。

## 一句话目标

基于冻结的 1-token/帧 VAE latent（与 RefSign 同一条 TiTok-CVAE），用**严格按官方 DiT
实现**的 Diffusion Transformer，从德语文本**直接**生成 PHOENIX-2014T 手语视频；
本阶段是新方向原型，追求链路正确、可复现、可观察。
**（2026-10-22 更新）**：骨架由 DiT 改为 **PixArt-α**，条件注入方式成为唯一主变量，
详见上方里程碑1 说明。

## 设计理念

- **方法硬约束（2026-10-12 立；2026-10-19 修订第 ④ 条）**：① 生成基于已训练好的 1-token（1024 维/帧）
  冻结 VAE；② diffusion 在 VAE latent 空间进行；③ 扩散目标用 x-prediction（预测 x0）；
  ④ **条件 = 文本 + 参考帧 frame 0**（`conditioning.reference.enabled: true`，即"方法 4"；
  2026-10-19 用户修订，取代原"纯 text→video"），仍不引入 pose/gloss/光流等其它监督；
  部署接口为 `text + frame0 → video`（长度由文本预测）。与本文档其余表述冲突时，以这四条为准。
- **方向切换**：课题目标是 text→video（参考 DiT 等直接生成路线的做法），不是
  text→pose→video；pose 监督版本保留在 `../RefSign`，两个仓库独立演进。
- **DiT 严格对齐官方**：block/初始化/条件嵌入（adaLN-Zero，`c = t_emb + y_emb`）逐条对齐
  `facebookresearch/DiT`；所有面向 1-token/帧与变长视频的改动都在
  `t2v/models/dit.py` 头部列明，绝不悄悄“改良”。
- **条件接口 = 文本的官方类比**：文本 token 进 cross-attention（论文附录变体），
  池化文本进 adaLN-Zero 全局向量；长度作为第二个全局条件；CFG 无条件分支用空串
  （与 RefSign 的文本管线一致，便于共享 mT5 与后续回译评测）。
- **文本条件带宽（2026-10-13）**：在 cross-attn 之外提供 `model.text_prefix` 可选开关
  （mmDiT/SD3 式联合注意力：文本 token 前缀进 self-attn、带 type embedding，
  cross-attn 保留；官方差异清单第 6 条）。实测把"换文本→输出变化"放大 ~2×、
  回译 correct−shuffled 配对差放大 2–8×，FVD 同步变好 → 当前默认部署配置
  （`configs/dit_b_text_prefix.yaml`）。
- **VAE 与 latent 缓存冻结**：训练进程不加载 VAE；只有可视化/生成时才懒加载解码器。
  缓存/统计/VAE checkpoint 任一变化 = 新缓存版本，禁止静默覆盖（继承 RefSign 契约）。
- **每 N 步必须有眼睛看**：训练期固定一小撮视频（默认 eval 前 2 条），
  同 seed 做 DDIM+CFG 采样、解码、与 GT 对齐抽帧拼图，输出对比图/GIF/MP4；
  同时落盘原始 latents（`samples_*.pt`）供离线复核。
- **先闭环，再质量**：跑通“文本→latent→视频”的全链路与可视化之后，再做
  规模/条件消融/评测（FVD、回译）——与 RefSign 的“先闭环再质量”一致。

## 技术栈

### 2026-10-13：动作语义首选路线 A→B

按用户“直接做首选路线”的指令，新增可选 mT5 全量微调与独立语义目标。
主生成 mT5 参与梯度更新（lr 1e-5），DiT/条件模块 lr 1e-4；VAE/缓存/统计始终冻结。
语义教师只在既有 train 文本—视频 latent 配对上训练，dev 检索通过严格等长错配门槛
后冻结；辅助文本端使用原始预训练 mT5 的固定池化特征，与主生成编码器分开。
视频分支不接收句子，在真实预测 x0 上提供语义梯度，不监督 CFG 空文本分支。

实验先做 A 5000 步，再从同一 A 检查点分别做 B 与 A-only 各 5000 步；
同 64 条 dev、seed、oracle 长度下评测绝对 SignMbart BT、正确−打乱配对差、
B−A 等预算配对差及 FVD，并单独检查文本预测长度部署。test 不用于预筛。
主文本权重完整保存恢复，CFG 空串随权重更新，raw 模型/条件模块/文本编码器一起生成。
已有文本长度头保持原始 mT5 特征接口。真实语义提升以独立回译为准，内部损失降低
或教师检索通过不能替代生成效果评测。

- 环境：conda `onvae`（torch 2.6.0+cu124、transformers、omegaconf、wandb、opencv）。
- 数据：PHOENIX-2014T；latent 缓存与文本 jsonl 全部走 RefSign 软链（只读）。
- 生成模型：DiT-S（384/12/6）默认；官方 DiT-B/L/XL 尺寸已在 `DiT_MODELS` 预置。
- 采样：DDIM（eta=0）+ CFG（现役评测口径 50 步 / CFG 2.0）；可选 DPM-Solver++(2M)
  （2026-10-14 加入，20–50 步即可达到 DDIM 250–500 步的指标，见 REVIEW_20261014 §7）；
  DDPM/可变步数已实现备用。

## 里程碑

- [X] 建仓：目录、配置、冻结适配器（VAE/mT5）、单元测试。
- [X] DiT + 扩散 + 数据管线闭环；冒烟训练（60 步）验证对比抽帧产物。
- [X] 正式训练（30k 步，10:40–12:43）与样例观察（loss 1.00→0.631；30×2 张对比抽帧图；
      终模目检见 NOTES：中间帧可辨认人形，多数帧仍噪声，瓶颈在规模/条件而非链路）。
- [X] 用户硬约束落地（2026-10-12）：x-prediction + cosine + logit-normal 修复碎帧；
      DiT-S/DiT-B 30k 重训（x0 配方）+ DiT-B 续训 60k；32 视频过拟合对照验证。
- [X] 文本条件诊断（2026-10-13，口径修正）：16 视频口径作废（SEM≈均值）→ 64 视频 + SEM；
      数据天花板探针（1-NN 检索 gap +0.143）；通道解剖证明基线"文本信息在、影响弱"
      （换文本只动输出 ~1%；门控幅度 msa 0.30 / xa 0.18 / mlp 0.62，cross-attn 有效贡献
      约为 MLP 的 3%，2026-10-13 晚间用修正的分块统计重算，旧值 0.13/0.40/0.63 与「5–10%」
      作废）。判据改为 **correct−shuffled 配对差**（latent gap 仅辅助，两口径可反向）。
- [X] 文本条件消融：CFG/eta/权重扫描（`reports/eval/cfg_sweep_dits_x0/`）+ 敏感度探针
      （`tools/text_sensitivity.py`）+ **text-prefix 联合注意力消融**（30k/60k/90k；
      GFSLT 口径下曾记「90k 不加分 → 部署 60k」，已被 SignMbart 口径推翻，见 README）：
      60k 回译 B1 11.36 / ROUGE-L 11.87 / FVD 51.61，correct−shuffled 增益
      B1 +4.71 / B2 +4.16 / ROUGE-L +6.60（基线 +0.56 / +2.37 / +2.85）。
      → 文本塌缩从"条件≈无效"变为"条件有效、绝对对齐仍有限"（GFSLT 口径 B4≈2.9；
      SignMbart 口径生成 B4 全为 0，GT 上限 29.51）。
- [X] 评测：FVD + 生成视频回译。**现行 FVD 链路 = `video_metric` 原样 `calculate_fvd`**
      （入口 `tools/eval_video_metric_fvd.py`，2026-10-27 起；旧 `tools/eval_fvd.py` 已删除）；
      回译链路为 SignMbart
      （`tools/run_bt_eval_signmbart.sh`，2026-10-13 纪律 #5）；`tools/run_bt_eval.sh`（GFSLT）
      仅作历史口径，不得与新数字混表。
- [X] 长度预测器（`t2v/train_length.py`，文本→长度桶，acc 0.41 / MAE 18.3）替换 oracle 长度
      （预测长度部署生成 + 回译已验证）。
- [X] 审计整改（2026-10-13 晚，`.trae/documents/AUDIT.md`）：训练加留出 `dev/mse`
      （`t2v/dev_eval.py`，RNG 中立、可选点）；通道探针改用 `tools/probe_stats.py` 的
      `chunk_abs_mean` 并重跑 5 个 JSON；评测写帧前清空目录 + BT 链帧数断言；冻结 VAE
      sha 校验默认开启（fail-closed）；README/DESIGN 口径单一化。
- [X] 参考条件开关：**2026-10-19 用户修订约束 4 后启用**（`conditioning.reference.enabled: true`，
  "方法 4"），部署接口 `text + frame0 → video`；见 TODO **D45**。

# NOTES — 已验证结论（note）

> **角色（2026-10-14 起，见 `RULES.md` 知识库流转规范第 9 条）**：本文件是流转链的终点之一——
> 设想在 `THINK.md` → 任务在 `TODO.md` → 产物在 `FINISH.md` → **测试判定有效**后出池记到这里
> （无效则进 `FAIL.md`）。条目按"日期 + 问题/结论 + 证据路径"追加。
> 2026-10-14 之前的章节是探索期的过程记录（当时尚无流转规范），保留原文不动。

## 2026-10-12 建仓：text→video（DiT）新方向

**用户需求**：课题方向修正为“完全 text → video”（参考 DiT / image-transformer 式直接生成），
不需要 pose 监督；pose 监督项目保留在 RefSign。要求：DiT 严格按论文/官方实现、
学习其条件嵌入方式、在 `/home/huyang/data/trae_projects/text2video` 新建仓库、
复用同一条 1-token VAE、训练中每 N 步输出手语视频对比抽帧样例。

**结论/决策**：

1. **官方 DiT 获取**：GitHub 直连超时（`github.com:443` 不可达），经镜像 `ghfast.top`
   克隆 `facebookresearch/DiT` 到 `.trae/reference/DiT`（git 忽略）。该仓库为
   CC BY-NC 4.0，因此**不复制源码**，`t2v/models/dit.py` 为按论文+官方结构独立实现，
   文件头逐条列出与官方的对应/差异。
2. **token 化方式**：沿用 RefSign 的“每帧 1 token、1024 维、只沿 T 轴 self-attention”
   约定（用户要求同一条 VAE）；DiT 输入投影 = `nn.Linear(1024, D)`（官方 PatchEmbed 的
   1-token 退化），位置编码用官方 1D sincos。
3. **条件嵌入**：官方 adaLN-Zero 的 `c = t_emb + y_emb` 直接推广为
   `c = t_emb + 池化文本 + 长度`；文本 token 走 cross-attention（论文附录变体，
   零门控、与官方同风格）。CFG 无条件分支 = 空字符串 mT5 编码（与 RefSign 一致）。
4. **扩散配方**：官方 DiT 用 ADM 的线性 β 调度 + ε 预测 + 均匀 t（不是 RefSign 的
   cosine + x0），本仓库按官方实现；`learn_sigma`（learned_range + vb 项）已实现，
   默认关闭（新方向先求稳，配置一行可开）。
5. **参考帧问题**：冻结 CVAE 是条件 VAE，解码必须给 GT 首帧（decode anchor）。
   为保持“text 直接生成”的语义，DiT 条件默认**不含** reference（配置可开），
   只在解码时把 GT 首帧作为 VAE 的固有输入。
6. **训练期可视化**：与 RefSign“只存 latents、离线解码”不同，本仓库在训练进程内
   懒加载冻结 VAE 解码（首次 ~40s，之后复用；也可 `sample.decode=false` 关闭），
   直接产出对比抽帧图/GIF/MP4，另存 `samples_*.pt`。
7. **验证**：单元测试 3 组全过（含“官方零初始化”与“oracle-ε 的 DDIM 精确回 x0”两个
   强校验）；冒烟训练 60 步（7021 条 train、8 it/s、loss 0.997→0.839）端到端跑通，
   对比图确认“上 GT / 下生成”对齐正确。

**已知偏差/待办**：① 训练为单卡（DiT-S 足够小）；DDP 后续需要再加。
② 长度是 oracle（训练/评测都用 GT 长度），长度预测器未接。
③ FVD/回译评测未接。④ `.trae/reference/DiT` 是完整克隆（含其 diffusion 实现），
仅供比对。

## 2026-10-12 训练配置选择

- DiT-S（hidden 384 / depth 12 / heads 6，46M 参数含条件编码器）：与论文最小档一致，
  单卡 4090 上 8 it/s，30k 步约 1 小时，抽样可视化开销可忽略。
- batch 16、AdamW(1e-4, wd 0)、EMA 0.9999、grad clip 1.0、bf16：
  与官方 DiT 训练脚本一致（官方无 clip，1.0 为稳定性兜底，可设为 0 关闭）。
- `max_num_frames: 256` 沿用 RefSign 读侧契约（丢弃 75 条长视频，7021 条参与训练）。

## 2026-10-12 晚：采样可视化固定 seed（对齐 RefSign 约定）

- 首版 `TrainingVisualizer` 用 `seed + step` 作采样随机种子，导致相邻 step 的对比图
  噪声不同、不利于目检“模型在变好”。已改为固定 `training.seed`（与 RefSign
  `save_samples` 一致）：同一批视频、同一初始噪声，跨 step 的对比图只差在权重。
- 正式训练（30k 步，10:40 启动）用的是旧种子策略，产物仍可用；后续实验用新约定。

## 2026-10-12 夜：首个正式训练完成（DiT-S，30k 步）+ 目检结论

- **训练**：10:40–12:43，单卡 4090，30k 步 7369 s（纯训练 ~7.8 it/s；每 1000 步
  可视化+存盘约 +1 min）。loss 1.00→0.631，~15k 步后进入平台（0.630–0.633）。
  60 张对比抽帧图 + 全部 metrics 已同步 wandb（project `text2video`, run `full_text_dit`）。
- **目检（EMA vs raw、CFG 扫描，视频 6698，seed 42）**：
  - 18k 步时中间帧已能解码出可辨认的人形（黑西装/手部位置），说明“文本→latent→VAE”
    链路确实在学；但多数帧仍是高频噪声（latent 偏出 1-token VAE 的锐利解码域）。
  - 30k 终模：raw 权重 + CFG 4.0、EMA + CFG 2.0 的中间帧最干净；EMA + CFG 4.0 噪声最大。
    → 本阶段 CFG≈2 比官方默认 4 更稳；EMA 未见优势（decay 0.9999、30k 步未足量）。
  - 结论：**质量瓶颈在“数据量/模型规模/条件带宽”，不是链路 bug**；符合“新方向先要
    闭环与可视化”的定位。后续优先级：① 训练更长/换 DiT-B；② 文本条件加强
    （tokens 全序列已用，可试 pooled+length 之外再加 CFG 调参）；③ 参考帧条件
    `conditioning.reference.enabled: true`（解码域匹配，可能显著改善 VAE 解码质量）；
    ④ 评测量化（FVD + 回译）。
- **产物**：`runs/full_text_dit/`（final.pt/last.pt/samples/step_*/）；
  `reports/generations/final_6698_cfg_sweep.png`（EMA/RAW × cfg 四联目检图）。

## 2026-10-12 晚：用户方法硬约束（x-prediction 等四条）+ 旧仓库早期 text-only 经验对照

- **用户指令（方法硬约束，写入 RULES.md / AGENTS.md / DESIGN.md）**：
  ① 生成基于已训练好的 1-token（每帧 1024 维）冻结 VAE；② diffusion 在 latent 上；
  ③ 必须用 **x-prediction（预测 x0）**；④ 只用 text→video，不加任何额外监督/条件
  （部署所需长度也必须由文本预测；VAE 解码锚点=冻结 VAE 固有输入，不算新增条件）。
- **用户观察**：`../RefSign` 早期（2026-09-28）的 text-only 过拟合实验
  （`runs/overfit_text{,_v2}`，32 视频、4×256、x0 + cosine + logit-normal、CFG 2.0）
  能生成"完整连续的视频，只是动作不对"；而本仓库 30k 步 ε + 线性 β + 均匀 t
  + CFG 4.0 的产物明显更碎（大量噪声帧）。需给出对照证据并修正配方。
- **差异清单（待实验判定）**：① 目标函数 ε vs x0（用户已强制 x0）；
  ② 噪声调度 linear-ADM vs cosine；③ t 采样 uniform vs logit-normal；
  ④ CFG 4.0（EMA）vs 2.0（raw）；⑤ 旧实验含 reference+length 条件的时期
  文本条件接近塌缩（NOTES 2026-09-29：模型实际学 p(轨迹|reference,length)），
  本仓库按新约束不含 reference。本轮按 x0 重构配方并做 32 视频过拟合复现对照。

## 2026-10-12 晚二：根因定位与 x0 配方修复（"旧仓库能出完整视频，新仓库全是噪声"）

- **对照证据（第一手）**：用冻结 VAE 解码 `../RefSign/runs/overfit_text_v2/samples_0015000.pt`
  （旧 32 视频 text-only 过拟合，x0+cosine+logit-normal，CFG 2.0）→ 帧帧干净、连续，
  sample_psnr 14.3/14.7（对照 gt_decode 32.4/31.4），即"完整视频、动作不对"；
  产物 `reports/old_baseline/`。
- **根因**：首轮 30k 步用的是官方 DiT 的 ε 预测 + 线性 β + 均匀 t 采样。该组合在本
  1-token VAE 潜空间（1024 维、重尾、逐维标准化）上训练出的 latent 大量落在冻结 VAE
  的锐利解码域之外 → 高频碎帧；旧仓库能出干净视频的组合是 **x0 + cosine + logit-normal**。
- **修复（按用户硬约束 3 落地）**：`t2v/diffusion/gaussian.py` 增加
  `prediction_type: x0`（默认）与 `noise_schedule: cosine`，训练循环增加
  `timestep_sampling: logit_normal`；CFG 改为在预测空间（x0）做（与 RefSign 采样器一致）；
  单测覆盖 x0 oracle-DDIM 精确回推、cosine 调度性质、logit-normal 采样、
  x0 空间 CFG。`configs/full_text_dit.yaml` / `dit_b_text.yaml` / `smoke.yaml`
  全部切到 x0+cosine+logit-normal，训练期样例 CFG 2.0。
- **验证（同 seed、同 VAE、同两条视频）**：
  - 32 视频过拟合（`runs/overfit_text_x0`）：**1k 步**时解码帧已完全干净连续，
    3k–15k 逐步出现手部动作；对照旧 overfit 的"平均姿态冻结"仍属同类形态
    （动作弱/不对），但"碎帧"症状消失。
  - 全量 DiT-S（`runs/full_text_dit_x0`）：1k 步样例即干净（对比：ε 版 30k 步仍大量噪声），
    说明修复与数据规模无关。
- **遗留（下一步测）**：文本条件敏感度探针（`tools/text_sensitivity.py`）在 10k 步
  中间检查点测得 correct vs shuffled 文本的 latent 距离几乎相同（mean text gap ≈ 0.002，
  d_between ≈ 0.003）→ **"文本塌缩"仍在**（与 RefSign 2026-09-29 诊断同类症状；
  历史那轮靠 reference+length 撑起画面）。最终指标以 30k 终模的 CFG 扫描、
  文本敏感度、FVD、GFSLT 回译为准。

## 2026-10-12 晚三：x0 修复终模验证 + 全套评测（CFG / 文本敏感度 / FVD / 回译 / 长度预测）

- **训练**：`runs/full_text_dit_x0`（DiT-S，30k，~7 it/s，loss 1.0→0.017）；
  `runs/dit_b_text_x0`（DiT-B 182M，30k 完成，`final.pt`；续训 60k 进行中，
  `runs/dit_b_text_x0_...` 同目录）；`runs/overfit_text_x0`（32 视频，15k）。
- **CFG/eta/权重扫描**（eval 6698，raw/ema × eta{0,1} × cfg{1,1.5,2,2.5,3}）：
  全部设置均无碎帧（修复后 CFG 主要影响平滑度）；
  动作幅度统计（帧间差）与 GT（3.95/帧）同量级（生成 3.6–4.4），
  EMA 更平滑（pix_std 18–19）而 raw 更接近 GT 纹理（23.4–23.8 vs GT 23.3）。
  部署选择 raw + cfg 2.0；拼图 `reports/eval/cfg_sweep_dits_x0/cfg_sweep.png`。
- **文本条件敏感度**（16 test，cfg 2.0/eta 0，正确 vs 打乱文本、同 seed）：
  - DiT-S raw：mean gap **+0.0025**（7/16 为正），d_between 0.045；
  - DiT-B raw：mean gap **+0.0182**（10/16），d_between 0.154；
  - DiT-B EMA：+0.0143（9/16），d_between 0.039。
  → 文本塌缩大幅缓解但**仍未解除**：模型尺度增大后文本效应开始可测（DiT-B 约 DiT-S 的 7×），
  这与 RefSign 2026-09-29 的历史诊断（文本被忽略）是同一核心难题，后续优先。
- **FVD**（32 test，R2+1D-18，16 帧）：DiT-S 58.4 / DiT-B 57.4（对照 RefSign
  pose 条件部署 36–41；FVD 在本数据主要度量背景/画质，仅作趋势）。
- **GFSLT 回译**（32 test，`tools/run_bt_eval.sh`，RefSign 指标协议）：
  DiT-S BLEU-1 8.42 / BLEU-2 2.16 / ROUGE-L 7.50 / WER 119.6；
  DiT-B BLEU-1 7.93 / BLEU-2 2.57 / BLEU-3 1.27 / ROUGE-L 6.84 / WER 120.2；
  BLEU-4 均 0（对照：RefSign 部署口径 BLEU-4 6.03；口径/条件不同，仅量级参照）。
- **长度预测器**（`t2v/train_length.py`，文本→8 分位桶；`runs/length_model_text`）：
  eval bucket acc 0.409 / top-2 0.666 / MAE 18.3 帧（对照旧 RefSign 长度模型
  0.367 / 0.589 / 22.1）；16 条 test 用**预测长度**生成+回译：
  BLEU-1 9.13 / BLEU-2 2.18（与 oracle 长度同档）→ **纯文本部署链路闭环**。
- **产物**：`reports/eval/{cfg_sweep_dits_x0, text_sens_*, bt_*, recipe_before_after_6698.png}`；
  评测工具 `tools/{eval_t2v, eval_fvd, text_sensitivity, cfg_sweep}.py`、`tools/run_bt_eval.sh`。

## 2026-10-12 深夜：DiT-B 续训 60k 结果（全指标提升，"任务重建"闭环达成）

- **训练**：DiT-B（768/12/12，182M 参数）首段 30k（14:32 启动，wandb `dit_b_text_x0`
  / `korraa8q`）→ 续训至 60k（16:46 启动，wandb `dit_b_text_x0_ext` / `1z43255c`，
  18:12 完成）。loss 轨迹：0.689(50) → 0.019(10k) → 0.016(30k) → 0.015(40–50k)
  → **0.0138(60k)**，仍缓慢下降未饱和。`final.pt` = 60k 权重（2.9GB）。
- **评测（32 test，CFG 2.0，raw 权重；对照 30k 与 DiT-S）**：
  - 文本敏感度（16 test，correct vs shuffled，同 seed）：mean gap **+0.0259**
    （9/16 为正；30k 为 +0.0182，DiT-S 为 +0.0025），d_between 0.314
    （30k 0.154，DiT-S 0.045）→ 文本效应随规模与训练量单调增强（较 DiT-S 已 10×），
    但"文本塌缩"彻底解除尚需更强条件（下一优先级）。
  - FVD（R2+1D-18）：**54.8**（30k 57.4，DiT-S 58.4；RefSign pose 条件 36–41 仅量级参照）。
  - GFSLT 回译（32 test）：BLEU-1 **9.87** / BLEU-2 **5.22** / BLEU-3 **3.67** /
    BLEU-4 **2.85** / ROUGE-L **10.43** / WER **116.5**
    （30k：7.93 / 2.57 / 1.27 / 0 / 6.84 / 120.2；DiT-S：8.42 / 2.16 / – / 0 / 7.50 / 119.6）。
    → BLEU-4 从 0 → 2.85，回译侧首次出现 4-gram 命中。"任务重建"（能生成与文本
    部分对应的完整连续视频）在 60k 档成立。
  - 60k 训练期样例（每 1000 步，6697/6698）经目检：全程干净连续、手部有运动、无碎帧；
    终版拼图 `runs/dit_b_text_x0/samples/step_0060000/*/comparison_sheet.png`。
- **结论与下一步**：① 配方已稳定（x0+cosine+logit-normal）；② 质量随规模/步数单调改善，
  60k 仍在下降 → 继续延长训练或加大模型仍有收益（单卡时间成本已明确为 ~3.5h/30k）；
  ③ 核心开放问题收敛为"文本条件带宽"（塌缩缓解但未解除）：下一步试文本 dropout 调低/
  加强 cross-attn 路径/文本条件池化方式，以及部署默认用 60k 权重；
  ④ 长度预测器维持 30k 档（`runs/length_model_text`）不变。

## 2026-10-12 深夜二：文本条件专项诊断（测量口径修正 + 数据天花板 + 通道解剖 + prefix 消融启动）

- **测量口径修正（重要）**：16 条视频的 text_gap SEM ≈ 0.024，与均值同量级 →
  之前"30k→60k 文本效应提升"的结论**不成立**（在噪声内）。改口径：64 条视频（SEM≈0.009），
  并固化 CFG 扫描协议（25 步 DDIM、raw 权重、same-seed correct vs shuffled）。
- **CFG 扫描（DiT-B 60k raw，64 test，25 步）**：gap = +0.0126(cfg2) / +0.0212(cfg3) /
  +0.0268(cfg4)，d_between = 0.207 / 0.347 / 0.467 → **文本效应随引导系数近似线性放大**
  （部署侧可用杠杆，但有失真代价）。
- **数据侧探针 `tools/probe_text_signal.py`**（3000 train / 600 test，PCA+ridge 与 2 层 MLP）：
  - latent→text（预测冻结 mT5 池化向量）：线性 R²=**0.053**、MLP 天花板 R²=**0.076**
    （cosine 0.816 vs 常数基线 0.799；shuffle 对照 −0.02）→ 冻结 latent 里可直接解码的
    文本信息**很少**；
  - text→latent R²=0.0065 vs **length→latent R²=0.0109** → 单是长度就比整段文本更能解释
    latent 变化；pairwise Spearman(latent~text)≈0（含时序特征口径）。
- **数据侧天花板 `tools/probe_text_ceiling.py`**（与模型同口径的 gap）：用文本 1-NN 检索
  训练集视频的 GT 轨迹作为"条件均值"预测，128 test：**mean gap = +0.143（SEM 0.047，
  72/128 为正）**（比值口径 ≈ +0.06）。→ 数据存在真实但对 GT 而言不强的文本信号；
  模型当前 +0.013~+0.027 距该粗天花板仍有量级空间。
- **通道解剖 `tools/probe_condition_channels.py`**（DiT-B 60k raw，t=500）：
  - gate 全部健康（block 均值 msa **0.30** / xa **0.18** / mlp **0.62**，无塌缩）。
    ⚠️ 2026-10-13 修正：原记的 0.13 / 0.40 / 0.63 来自错误的分块统计
    （把 `[B,6D]` 的 batch 均值当成分块序列切片）；正确值用 `chunk(6)`/`chunk(3)` +
    `mean|·|` 重算，见下「2026-10-13 晚」与 `reports/eval/probe_channels_*.json`；
  - 换文本时 **kv 相对变化 ~1.06–1.17（>100%）**、**cross-attn 输出相对变化 ~0.89–1.64**，
    但 **block 隐状态只变 4–6%、最终输出只变 ~1%**（correct-vs-shuffled 0.0181 ≈
    correct-vs-empty 0.0174）→ 文本通路**信息在、影响弱**：cross-attn 的有效贡献
    （mean|gate| × 子层输出 RMS）只有 MLP 的约 **3%**（按 block 平均比值口径约 4–6%）；
    原记「5–10%」同样是错误分块统计的产物，方向性结论不变，末层进一步衰减。
    这是"文本塌缩"症状的机理定位（此前 2026-09-29 诊断的升级版）。
- **行动（消融实验）**：新增 `model.text_prefix`（mmDiT/SD3 式联合注意力：文本 token 前缀进
  self-attn，带 type embedding；cross-attn 保留；官方差异清单第 6 条已登记；单测 2 个新增）。
  配置 `configs/dit_b_text_prefix.yaml`，DiT-B 30k（`runs/dit_b_prefix30k`，`tools/launch_prefix_train.sh`
  在共享集群轮询空闲 GPU 启动）。判定口径：同 64 视频协议（cfg 2/4）+ 32 视频 GFSLT 回译，
  对照 = DiT-B 30k/60k（注意 30k 权重已被续训覆盖，对照以 60k 为主）。
- **顺带**：`.trae` 结构与新增探针工具已同步（STRUCTURE.md）；`tests/run_all.py` 全过。

## 2026-10-13 凌晨：text-prefix 联合注意力消融——30k 首战（回译侧文本增益显著）

- **训练**：`runs/dit_b_prefix30k`（DiT-B + `model.text_prefix: true`，30k 步完成，
  loss 0.0156 @30k，与基线 30k 的 0.0163 相当；30k 权重另存 `final_30k.pt`，
  续训 60k 进行中）。
- **生成质量/latent 口径**（32 test，CFG 2.0，raw）：FVD **52.96**（基线 30k 57.4 /
  60k 54.8）；帧间差 3.36（GT 4.82，基线 60k 3.59）→ 非静止退化。
- **latent 文本敏感度（64 test，25 步）**：cfg2 gap **+0.0052**（d_between 0.086）/
  cfg4 **+0.0121**（0.275）——**低于**基线 60k（cfg2 +0.0126 / cfg4 +0.0268）。
- **GFSLT 回译（32 test，CFG 2.0）+ 打乱文本对照（关键）**：
  | 运行 | B1 | B2 | B4 | ROUGE-L | WER |
  |---|---|---|---|---|---|
  | 基线 60k correct | 9.87 | 5.22 | 2.85 | 10.43 | 116.5 |
  | 基线 60k **shuffled** | 9.31 | 2.85 | **0.00** | 7.58 | 116.8 |
  | prefix 30k correct | **10.51** | **6.54** | **4.09** | 9.44 | **110.2** |
  | prefix 30k **shuffled** | 7.61 | 1.48 | **0.00** | 7.02 | 121.3 |
  → **正确 vs 打乱 的回译增益：B4 基线 +2.85 / prefix +4.09；B2 基线 +2.37 / prefix +5.06。**
  即 30k 的 prefix 模型在"语义上确实按文本生成"这一维度已超过 60k 基线（B4 差 1.24、
  B2 差 2.69），且 WER 更低（110.2 vs 116.5）。FVD 也更好。
- **注意口径差异**：latent 近邻 gap（对 GT 的接近度）与回译语义增益方向相反
  （prefix 的 latent gap 更小但回译增益更大）——说明"对单条 GT 轨迹的逐点接近"
  与"语义是否正确"是两件事；**今后文本条件判定以"correct vs shuffled 的配对差"为准
  （回译为最终口径）**，latent gap 仅作辅助。
- **shuffled 对照的方法论意义**：基线 60k 的 B1 9.87 vs 9.31（几乎无差）说明其 B1 主要来自
  "泛化手语动作的可读性"而非文本；prefix 的 B1 10.51 vs 7.61 才体现真正的文本依赖。
- **下一步**：prefix 续训 60k（`final_60k.pt`）完成后跑同一套口径（文本敏感度 + FVD +
  回译 correct/shuffled），并做正确 vs 打乱样例的目检拼图（`tools/make_comparison_sheet.py`）。

## 2026-10-13 凌晨二：prefix 续训 60k 收尾——文本依赖度全口径增强（决定性证据齐）

- **训练**：30k→60k 续训完成（`runs/dit_b_prefix30k/final_60k.pt`；同目录 `final_30k.pt`
  保留 30k 快照；60k loss 0.0148 仍缓降）。评测链 `tools/eval_prefix60k_after_extend.sh`
  全跑通（通道探针→64 条敏感度×3→32 条生成→FVD→回译→打乱对照回译）。
- **正确 vs 打乱 回译（32 test，CFG 2.0，50 步，最终口径）**：
  | 运行 | B1 | B2 | B3 | B4 | ROUGE-L | WER |
  |---|---|---|---|---|---|---|
  | 基线 60k correct | 9.87 | 5.22 | 3.67 | 2.85 | 10.43 | 116.5 |
  | 基线 60k shuffled | 9.31 | 2.85 | 0.00 | 0.00 | 7.58 | 116.8 |
  | prefix 30k correct | 10.51 | 6.54 | 4.92 | **4.09** | 9.44 | **110.2** |
  | prefix 30k shuffled | 7.61 | 1.48 | 0.00 | 0.00 | 7.02 | 121.3 |
  | prefix 60k correct | **11.36** | 5.52 | 3.77 | 2.89 | **11.87** | 117.0 |
  | prefix 60k shuffled | 6.65 | 1.36 | 0.00 | 0.00 | 5.27 | 120.5 |
  → **correct−shuffled 增益**：B1 基线 +0.56 / prefix60k **+4.71**；B2 +2.37 / **+4.16**；
  ROUGE-L +2.85 / **+6.60**。文本依赖度（配对差）在 60k 口径再放大；
  绝对分 60k 拿到 B1/ROUGE-L/FVD 最优，30k 拿到 B2/B4/WER 最优（检查点选择的取舍，
  两个快照都保留）。
- **通道探针 60k**（`reports/eval/probe_channels_prefix60k.json`，t=499，12 层均值）：
  文本交换后隐状态相对变化 **11.6%**（基线 5.9%；prefix 30k 12.9%）；
  correct-vs-empty 输出距离 **0.0322**（基线 0.0174，+85%）；
  dshuf/dempty 0.71（基线 1.04 → 基线里"随机文本 ≈ 空文本"，prefix 下已区分）。
- **64 条文本敏感度（latent gap，辅助口径，25 步）**：cfg2 **+0.0055±0.0099**（31/64，
  t=0.56，不显著）；cfg4 +0.0215±0.0175；cfg4+rescale0.7 +0.0152±0.0134；
  d_between 0.208 / 0.465 / 0.422（与基线 60k 的 0.207/0.467 几乎相同）。
  → 再次确认：**latent gap 与语义依赖方向不一致**（prefix 的 latent gap 略低但回译增益
  大幅更高）；latent gap 不能作为文本条件的判据，只能看"换条件后 output 变化"这种
  相对量或回译配对差。rescale 0.7 对 gap 无增益。
- **FVD 51.61**（基线 60k 54.8 / prefix 30k 52.96）；帧间差 3.85（基线 3.66；GT 4.95，
  同 32 条）→ 运动更强、无退化。
- **目检**：`reports/eval/prefix60k_vs_baseline_6704.png`、`..._6603.png`
  （GT / 基线 60k / prefix 60k 三行 8 帧）；训练期样例 `runs/dit_b_prefix30k/samples/step_0060000/`。
- **结论**：`model.text_prefix` 联合注意力在本任务上稳定增强文本条件（机理口径 2×、
  语义配对差 2–8×），且生成的视频质量/FVD 同步变好；"文本塌缩"从"条件≈无效"
  变为"条件有效但绝对对齐仍有限"。**绝对 BLEU 仍低**（B4≈2.9），距数据侧天花板
  （1-NN gap +0.143）仍有量级空间 → 下一优先级：text dropout 比例、更长训练/更大模型、
  以及文本编码器侧（冻结 mT5 池化向量的信息量本身已被探针证伪为次要瓶颈）。

### 追记（2026-10-13 凌晨三）：两个后续实验已挂后台串行跑

- `tools/run_followups_20261013.sh`（setsid 后台；日志 `log_followups_20261013.log`）：
  ① `tools/run_prefix_extend90k.sh`：prefix 60k→**90k** 续训（loss 60k 时仍缓降；
  结束即写 `final_90k.pt`，评测链 `tools/eval_prefix90k_after_extend.sh` 待命）；
  ② `tools/run_prefix_d005_train.sh`：**text dropout 0.05 消融**（30k，
  `configs/dit_b_text_prefix_d005.yaml`，`runs/dit_b_prefix_d005`；动机=绝对对齐仍有限，
  试条件通路强化；p=0.05 仍保留空条件分支以支持 CFG）。
- 两个脚本都沿用"轮询空闲 GPU + --resume"模式，被抢卡会自行续跑；
  90k 评测脚本新增逐步骤选卡+重试（`run_step`，无卡时 sleep 120 轮询）。
- 结果出来后按同口径（64 敏感度 + 32 回译 correct/shuffled + FVD）写入本文件与 README。

## 2026-10-13 凌晨四：prefix 90k 结果——加长训练无额外收益，60k 仍为最佳部署

- **训练**：60k→90k 完成（`runs/dit_b_prefix30k/final_90k.pt`），90k loss 0.0129。
- **回译（32 test，CFG 2.0）+ 打乱对照**：prefix 90k correct **B1 11.05 / B2 3.56 /
  B3 0 / B4 0 / ROUGE-L 10.54 / WER 111.9**；shuffled 8.33 / 1.57 / 0 / 0 / 6.86 / 120.2
  → 文本依赖 **ΔB1 +2.72 / ΔB2 +2.00 / ΔROUGE +3.68**（比 60k 的 +4.71/+4.16/+6.60 小）。
  B3/B4 全 0（逐对生成文本流畅但与参考无 3-gram 命中，不是评分故障——同脚本对 60k 正常）。
- **FVD 53.32**（60k 51.61）；通道探针 **d_shuf 0.0339 / d_empty 0.0377**（比 60k 更大，
  hidden-swap 12.2%）；64 条 latent gap cfg2 **+0.0117±0.0117**（t=1.00）/ cfg4
  **+0.0264±0.0199**（t=1.33），d_between 0.329/0.653（换条件差异变大）。
- **结论**：退化性的"训练越久越好"假设不成立——90k 在 B1/ROUGE/FVD/文本依赖度上全部
  不如 60k（只有 WER 111.9 更好）。**部署维持 prefix-60k**；后续 30k 起即收敛，
  要做增益得换变量（条件通路/dropout/模型规模），而不是纯加步数。
- **口径提醒**：30k/60k/90k 三个快照的 B2/B4 波动大（6.54/4.09 → 5.52/2.89 → 3.56/0），
  32 条评测的方差不可忽略；跨快照比 V 只看配对差方向，**同预算对照以"同 step 配对"为准**
  （唯一干净对照：60k baseline vs 60k prefix）。

## 2026-10-13 凌晨五：text dropout 0.05 消融——负结果（0.10 维持）

- **训练**：`runs/dit_b_prefix_d005`（其他超参与 prefix-30k 完全一致，只把 CFG 文本 dropout
  0.10→0.05），30k 完成，loss 0.0157 ≈ prefix-30k 的 0.0156（训练目标无差别）。
- **回译（32 test，CFG 2.0）**：correct **B1 7.13 / B2 1.38 / B3 0 / B4 0 / ROUGE-L 6.03 /
  WER 122.2**；shuffled 4.59 / 1.65 / 0 / 0 / 3.75 / 121.9 → ΔB1 +2.53 / ΔROUGE +2.28
  （文本依赖方向仍在，但**绝对质量全面劣化**：对 prefix-30k 的 10.51/6.54/4.09 全面下跌）。
- **FVD 56.56**（prefix-30k 52.96，基线 60k 54.8）；64 条 latent gap cfg2 +0.0104±0.0063
  （t=1.65）/ cfg4 +0.0264±0.0122（t=2.17）——**latent gap 反而"更显著"**，
  再次验证该口径与生成质量脱钩（负结果里 gap 上升）。
- **通道探针异常**（`reports/eval/probe_channels_d005.json`）：所有层
  `rel_correct_shuffled_hidden` 80–134%（其余运行 12–13%）→ 隐状态被文本直接主导，
  但输出距离 d_shuf 仅 0.0177（与基线同级）→ 疑似"文本淹没低幅残差流 + 末层压缩"，
  与生成质量崩坏同时出现。机理留待需要时复查。
- **结论**：**dropout 0.10 维持**；降 dropout 不是"强化条件"的正确杠杆
  （很可能因空条件分支训练不足而伤 CFG 质量）。若继续条件调味，方向应改为
  升 dropout（0.15–0.2）或独立调 CFG/sampler，且必须用"回译配对差 + 绝对质量"双口径。

## 2026-10-13 中午：回译模型切换为 SignMbart（纪律第 5 条）+ 复现差异根因与修复

### 1) 纪律变更（用户指令）

- **今后所有回译指标一律用 SignMbart**（HF `hulala/SignLanguage` → `weight/phoneix14t/SLT1/_best_model.pt`，
  PHOENIX14T SLT1；本地 `/data/huyang/models/SignLanguage/SignMbart`），**不再用 GFSLT-VLP**。
- 已写入 `RULES.md`（方法硬约束 #5）与 `AGENTS.md`。链路：
  `tools/run_bt_eval_signmbart.sh <frames-root> <out-root> [gpu] [labels]`
  = 生成帧 → SignMbart（视频→文）→ **原 RefSign 指标协议不变**（raw corpus BLEU / ROUGE-L / jiwer WER）。
- **换模型=换口径**：GFSLT 旧数字只能作为历史，不得与新数字混进同一张对比表。

### 2) 复现差异根因：mBART 提示 embedding 少了 sqrt(d_model) 缩放

首次跑 GT 视频自检时**全线重复循环**（`aber aber aber…`，32 条 WER 381.8%），但官方
checkpoint 自称 test B4 28.0 / ROUGE-L 53.3。定位过程：

1. **先验证视觉前端**：GT 视频 CTC gloss WER ≈ 20–24%，与 checkpoint 自己报的 27.28% 同量级
   → 帧、resize/crop、顺序、PoolFormer/TemporalConv/BiLSTM 全部正确（帧数还与官方
   `phoenix-2014t.*` 注解里的 `num_frames` 逐条对齐）。坏的是 **gloss→text（mBART）这一段**。
2. **逐项排除**：解码配置（fp16/fp32 × beam6/greedy/no_repeat_ngram）与特征缩放都不改善；
   连 teacher-forcing（喂参考文本当解码前缀）的下一词一致率都只有 0.32 → 说明不是搜索问题，
   而是编码器输入分布与训练时不一致。
3. **根因**：官方训练用老版 transformers（`MBartScaledWordEmbedding` 在 *embedding 内部*
   乘 `sqrt(d_model)=32`）。transformers≥4.30 删掉了该类，把缩放移进 encoder/decoder 的
   forward（`embed_tokens(ids) * embed_scale`）——**只对 `input_ids` 生效**。官方代码
   `SLTModel.get_prompt_embeding` 却用 `self.embed_tokens(...)` **直接**构造编码器提示
   `[</s>, de_DE]`，在新版下这个直接调用**静默丢掉 ×32**，编码器提示过小、mBART 退化成
   重复循环（与 `translationPure.py` 里成为死代码的 `self.input_embed_scale` 互为佐证）。
4. **证据（10 条 test 网格，F=视觉特征缩放，P=提示 embedding 缩放）**：
   F×1,P×1 → TF 0.32 / B1 5.5 / B4 0；F×1/32,P×1 → 0.64 / 33.1 / 13.9；
   **F×1,P×32 → 0.73 / 37.3 / 14.6**（最佳）；F×1/32,P×32 → 0.68 / 29.2 / 10.4。
5. **修复**（`tools/signmbart_bt_infer.py`）：忠实还原 `MBartScaledWordEmbedding`
   （forward 内缩放），并把 encoder/decoder 内部的 `embed_scale` 钉成 1.0
   （`_neutralize_double_embed_scale`），保证 token embedding **只被缩放一次**。
6. **验证**（GT 视频，官方自身数字为参照）：64 条 gloss WER 20.1% / B1 47.6 / B2 37.4 /
   B3 30.2 / B4 25.0 / R 44.5（官方 27.28 / 54.2 / 41.6 / 33.5 / 28.0 / 53.3）；全 642 条见 §4。
7. **顺带的性能修复**：官方 CTC 用 `ctcdecode.CTCBeamDecoder(num_processes=10)`，内部是
   内核自旋等待（`sys` 时间 11 分钟/8 条视频 ≈ 85 s/视频）。翻译分支是 `input_type='feature'`，
   **根本不吃 gloss**，因此默认改为 torch 贪婪 CTC（`--ctc max`，仅诊断口径），0.6–0.7 s/视频，
   文本假设逐位相同；需要复现官方 gloss 数字时用 `--ctc beam`。

### 3) 重跑全部现有回译指标（32 条 test，CFG 2.0，raw 权重，50 步 DDIM；SignMbart）

| run | B1 | B2 | B3 | B4 | ROUGE-L | WER | ΔB1 | ΔB2 | ΔROUGE-L | ΔWER(改善) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-60k | 7.42 | 0.00 | 0 | 0 | 5.88 | 114.2 | −0.96 | −1.57 | −0.28 | +1.4 |
| prefix-30k | 6.72 | 2.03 | 0 | 0 | 5.84 | 112.2 | −1.23 | −0.20 | −0.31 | +1.1 |
| prefix-60k | 9.47 | 4.04 | 0 | 0 | 9.57 | 114.2 | +0.65 | +1.77 | +0.78 | +0.3 |
| **prefix-90k** | **14.75** | **6.45** | **2.44** | 0 | **12.75** | **104.0** | **+5.79** | **+3.13** | **+5.10** | **+6.8** |
| d005-30k | 7.92 | 2.73 | 0 | 0 | 6.28 | 107.4 | +1.15 | +1.27 | +1.44 | −0.6 |

（Δ = 正确文本 − 打乱文本，同一批 32 条、参考同源，见 `labels.*.test` 校验。）

- **配对 bootstrap（2000 次重采样，`tools/bt_paired_stats.py`）**：只有 **prefix-90k** 的
  ΔB1 +5.79 [+0.87, +11.26]、ΔROUGE-L +5.10 [+0.73, +10.17]、ΔWER 改善 +6.82 [+0.91, +13.54]
  的 95% 区间不含 0（P(>0)=0.99）；其余四条（baseline-60k / prefix-30k / prefix-60k / d005-30k）
  的区间都跨 0。
- **口径警示（重要）**：换回译模型后**排序结论会变**——GFSLT 口径下 prefix-60k 最好（B1 11.36，
  ΔB1 +4.71），SignMbart 口径下 **prefix-90k 明显最好**（B1 14.75、唯一非零 B3、唯一显著文本依赖）。
  两者都不能"绝对化"：32 条样本 + 两个视频→文模型的分辨率差异；后续要下结论就用
  **同一模型 + 配对 bootstrap**，并优先把评估集从 32 条扩到 64/128 条。

- **复核（2026-10-13 下午，用户要求）**：10 个 run 全部重跑，`hyps.txt` / `score.json` 与首轮
  **逐位一致**；配对 bootstrap（seed 0，2000 次）CI 全部复现（仍只有 prefix-90k 显著）。
  **同 32 条 GT 视频上限**（`reports/eval/gt32_signmbart/`）：B1 50.42 / B2 42.29 / B3 34.88 /
  B4 29.51 / ROUGE-L 48.93 / WER 61.08 → 生成 run 的 B4 全 0 是真实语义差距，不是评分问题。

### 4) 全 642 条 GT 自检（`tools/signmbart_validate.py`）

- 输出：`reports/eval/gt_full_signmbart/`（`gt642.txt` 假设 / `.ref.txt` 参考 / `.gloss.txt` CTC gloss）+ `run.log`。
- 结果（全量 642，CTC 贪婪，`preproc=p0`）：**gloss WER 27.17%（官方自报 27.28%，≈逐点复现）**；
  文本 WER 71.99%；BLEU B1 44.55 / B2 34.53 / B3 27.61 / B4 22.74 / ROUGE-L 42.37
  （官方自报 54.16 / 41.62 / 33.54 / 28.00 / 53.30，达 ~81%）。**文本侧残余差距为已知限制**：
  视觉/CTC 侧已对齐（gloss 差 0.11 点），残余来源疑在文本解码配置/后处理；不影响本仓库跨 run
  比较（10 个 run 同口径重跑）。64 条子集时 gloss WER 20.1 / B4 25.0（子集偏易，勿引用为全量）。
- 交叉校验：同一批 hyps 在 RefSign 指标协议下复评（
  `../RefSign/reports/video_bt/signmbart_views_2026-10-13/test642/score.json`）：BLEU 与上游实现
  逐位一致，ROUGE/WER 因实现细节差 ≤4（43.04 vs 42.37；68.09 vs 71.99）。

### 5) 相关文件

- 新工具：`tools/signmbart_bt_infer.py`（驱动 + 语义还原修复）、`tools/run_bt_eval_signmbart.sh`（单 run 链路）、
  `tools/run_bt_all_signmbart.sh`（10 个 run 批量重跑）、`tools/signmbart_validate.py`（GT 自检 vs 官方数字）、
  `tools/signmbart_probe.py` / `tools/signmbart_trans_debug.py` / `tools/signmbart_gen_ablation.py`（诊断）、
  `tools/bt_paired_stats.py`（配对 bootstrap）。
- `tools/run_bt_eval.sh` 已标注 DEPRECATED（GFSLT 历史口径）。
- 跨仓库：RefSign 四视图（deploy / gtpose / orig64 / test642）同帧重跑见
  `../RefSign/reports/video_bt/signmbart_views_2026-10-13.md`（真视频 +5~8 B4；deploy/gtpose
  基本不动 → 与"部署瓶颈在条件侧"一致）。

## 2026-10-13 下午：全项目审计（read-only）—— 结论 CONDITIONAL，0 Critical / 5 Major / 9 Minor

**问题**：用户要求「仔细检查和思考项目有什么问题、有什么改进方案，并落地为文档」。

**做法**：按 `research-codebase-audit` 技能做七轴审计（正确性/泄漏/模型结构/数据管线/代码与声明/
占位伪造/严谨性）+ 声明核验；通读 `t2v/` 全部 18 个模块、`tools/` 21 个脚本、`configs/`、`tests/`、
`.trae` 文档与 `../RefSign/UPSTREAM.md`；实跑测试、实算 sha、独立复算有疑问的探针。
**全程只读**：未改任何训练/评测代码、配置、缓存或 checkpoint。

**正向核验（已确认正确）**：`tests/run_all.py` 4 个测试全过；冻结 VAE checkpoint sha256
`82f37058…` 与 config sha `0678ee8f…` 与 manifest 一致（实算）；`train_stats.json` 的
`manifest_sha256` = `sha256(train.jsonl)` = `c74d160b…`（统计只拟合 train）；当前上游 VAE 代码加载
冻结 checkpoint 为 missing 0 / unexpected 0；split 无 ID 重叠、无重复、无缺失文本、全部
`T == num_frames-1`；256 帧过滤后 7021/518/642 与文档一致。

**5 个 MAJOR**：
1. **无验证集**：训练循环只有 train 指标，`best_dev_loss` 参数全仓无调用方，`final.pt` 恒为最后一步；
   30k/60k/90k 的部署选择与「文本依赖显著」判定都建立在 32 条 **test** 回译上 → test 调参/选模型。
2. **通道探针统计量定义错误**：`probe_condition_channels.py` 把 `[B,6D]` 输出的 batch 均值当成
   分块序列（`g[:3]` / `g[3:]`），正确应为 `chunk(6)`/`chunk(3)`。文档引用的
   「gate 0.13 / 0.40 / 0.63」「cross-attn 贡献仅 MLP 5–10%」不可复现；本次独立复算得
   `mean|gate_msa|=0.294 / mean|gate_mlp|=0.616 / mean|gate_xa|=0.175`，贡献比 ≈3.4%，
   且同一代码换一次噪声就能把 gate_mlp 从 +0.63 变成 −0.27。**定性结论（文本信息在、影响弱）
   由记录正确的 block_out/输出距离支撑，不受影响且被修正值加强**，但被引用的数字必须重算。
3. **评测帧目录复用**：`tools/eval_t2v.py` 直接覆盖写 `imagesNNNN.png` 且不清理多余旧帧，
   而 BT 标签又按 `basename(out_root)` 派生 → 更短的新结果会带着上一次的高位帧进入 SignMbart/FVD，
   指标文件里看不出来。
4. **冻结 VAE 契约未强制**：`verify_checkpoint_sha256` 默认 False、`strict=False` 只查 unexpected、
   运行期从可变仓库 import。实算发现 VAE repo HEAD = `a312ab7b`，而 `UPSTREAM.md` 声明的契约 commit
   是 `d81ce1e1`（**落后 5 个 commit**，改了 `modeling/titok_cvae.py` / `cvae_blocks.py`）。
   缓解证据：新增项（pixel_refine / extra_blocks / posterior_wide）默认关闭，`tanh(1+gate)` 只作用于新分支，
   logvar clamp 不影响被缓存的 posterior mean，加载覆盖 0/0 → **目前无证据说明码已变，但也没有测试守这条线**。
5. **口径切换后文档未同步**：README 顶部仍写「当前最优 prefix-60k」，而 SignMbart 表里只有 90k 显著；
   「90k 不加分→部署 60k」「B4≈2.9」都是 GFSLT 口径却排在当前区块；差异清单写 5 处而 `dit.py` 列 6 条；
   `DESIGN.md` 里程碑仍把 GFSLT `run_bt_eval.sh` 当现行评测。

**9 个 MINOR**：`sampling.sampler` 死配置；`clip_denoised` 用图像域 ±1 裁剪标准化 latent；
生成/训练样例默认 EMA 而部署是 raw（README 示例未加 `--no-ema`）；长度模型无种子、取最后一轮；
train∩test 有 20 条相同译文（train∩eval 15、eval∩test 9）；resume 后 epoch 重置导致批序重放；
评测脚本硬编码绝对路径与 `gfslt` 环境（BT 的标签/打分仍来自 GFSLT 仓库）；仓库零 commit、签名不含
代码/依赖/VAE commit。

**改进方案（P0/P1/P2）**：P0 = 补 dev 选点 + 清理评测目录与标签断言 + 冻结契约 fail-closed +
文档口径单一化 + 修探针重算；P1 = 评测扩到 128 条并默认报配对 CI、在 dev 上做 CFG∈{2,3,4}×rescale
的 SignMbart 配对实验、训练样例改 raw、补 git/依赖签名并把 BT 协议脚本收进本仓库；
P2 = 条件侧（mT5 取层/双向编码器/条件编码器容量）、dropout 0.15–0.20、`frame_stride=2` 读取侧实验、
VAE 手部上限天花板探针、把配对差口径固化成 `tools/eval_suite.py`。

**结论**：判定 **CONDITIONAL** —— 现有数字可作为初步结果，但在完成 P0 前不宜把 32 条 test 的排序
写成论文级结论；完整报告见 `.trae/documents/AUDIT.md`（含每条发现的 file:line 证据、修复与复测条件、
覆盖表、声明核验表、泄漏评估表、mock 清单与残余不确定性）。本次遵循 RULES「不自行增删 TODO 条目」，
故待办条目未改动，P0/P1/P2 全部记在 AUDIT.md §9。

## 2026-10-13 晚：审计 P0 整改（第一轮，已落地并验证）

**背景**：用户要求「开始修改这些主要的问题」——即 `AUDIT.md` 的 5 个 MAJOR（M1–M5）。
本轮只动 P0 范围，未重训、未改缓存/checkpoint/上游仓库。

**做法（TDD：先写失败测试，再实现）**：新增 4 个红灯测试 → 实现转绿，`python tests/run_all.py`
现为 **8 个测试全过**（新增 `test_dev_eval` / `test_eval_frames` / `test_frozen_guard` / `test_probe_stats`）。

1. **M1 dev 选点**：新增 `t2v/dev_eval.py`（`build_dev_snapshot` + `evaluate_dev_loss`：固定噪声/时间步、
   `frame_mask` 口径、局部 `torch.Generator`、不碰全局 RNG、退出恢复 train 模式）；
   `t2v/train.py` 在训练前用 `eval` split 建 8×batch 快照，每 `dev_every`（默认=log_every）记 `dev/mse`，
   条件用 dropout=0（不扰动训练 RNG）。冒烟验证：`runs/_smoke_dev_eval/metrics.jsonl` 的
   `dev/mse` 从 1.019 → 0.437（60 步）单调下降，`test` 未被触达。
2. **M2 探针修正**：新增 `tools/probe_stats.py`（`chunk_abs_mean`，adaLN 6 块/3 块的单一定义）；
   `tools/probe_condition_channels.py` 改用分块 + `mean|·|`、默认 split 从 `test` 改 `eval`。
   **重跑 5 个 checkpoint（t=500，split=test video 0，与审计口径一致）**，新 `reports/eval/probe_channels_*.json`：

   | run | gate_msa | gate_xa | gate_mlp | cross/mlp 贡献（block 均值比） |
   | --- | --- | --- | --- | --- |
   | 基线 60k | 0.296 | 0.177 | 0.619 | 5.8% |
   | prefix-30k | 0.319 | 0.115 | 0.499 | 6.3% |
   | prefix-60k | 0.401 | 0.152 | 0.701 | 8.7% |
   | prefix-90k | 0.429 | 0.223 | 0.839 | 12.3% |
   | d005-30k | 0.317 | 0.115 | 0.458 | 15.7% |

   基线复算值（0.296/0.177/0.619）与审计独立复算（0.294/0.175/0.616）一致 → 修正正确。
   NOTES 本文件 189/192 行与 `DESIGN.md` 的旧数字已就地标注修正（不删除历史）。
3. **M3 评测目录与标签**：`t2v/pipelines/decode.py::prepare_frame_dir`（rmtree+mkdir）；
   `tools/eval_t2v.py` 写帧前清空、并断言 `解码帧数 == length`；
   `tools/run_bt_eval_signmbart.sh` 增加 `LABEL_TAG`（默认仍取 out-root 名）与
   「每视频帧数 == `meta.json` length」断言，不一致直接失败（不静默进入 SignMbart）。
4. **M4 冻结契约 fail-closed**：`t2v/adapters/vae.py` 默认 `verify_checkpoint_sha256=True` /
   `verify_config_sha256=True`（sha 校验带 `(size,mtime,ctime)` 缓存，改文件即失效，
   可用 `T2V_SKIP_FROZEN_VERIFY=1` 显式跳过）；`assert_full_state_dict_coverage` 让 missing/unexpected
   任一非空即抛错（原先 `strict=False` 只查 unexpected）；`contract` 记录 repo HEAD 与契约 commit 供签名。
5. **M5 文档口径单一化**：`README.md` 新增「当前部署与评测口径（单一来源）」区块（部署 checkpoint 单一定义
   `$CKPT`、SignMbart 口径、n=32、明确「步数选择尚未定论」与「不许用 test 选点」）；
   GFSLT 结论（90k 不加分→部署 60k、B4≈2.9、dropout 消融）就地标注为历史口径；
   差异清单 5→6 处；目录树 `t2v/models/length_model.py`、评测说明改 SignMbart。
   `DESIGN.md` 里程碑把 GFSLT 链路改为历史、新增本轮整改条目。

**顺带修的 MINOR**：N1 `sampling.sampler` 从死配置改为**强制校验**（非 ddim 直接报错，
`resolve_sampler`）；N3 生成/评测默认 **raw**（`tools/generate.py` `--no-ema`→`--ema`、
`tools/eval_t2v.py`/`text_sensitivity.py` `--raw`→`--ema`，旧参数名保留为隐藏别名以免脚本失效），
`TextToLatentPipeline` 请求 EMA 但 checkpoint 无 EMA 时**报错而非静默回退**；
N5 `t2v/train_length.py` 加 `set_seed(--seed)` 并按 dev MAE 选**最优轮**（保存 `best_epoch`/`last_epoch_metrics`）；
N9 `config_signature` 追写 `code`（git HEAD / dirty）、`deps`（torch/transformers/omegaconf/numpy/safetensors）
与 `frozen_contract`（VAE repo HEAD vs 契约 commit、checkpoint/config sha）——当前签名会显式暴露
上游 `a312ab7b` 与契约 `d81ce1e1` 的漂移。

**验证**：`tests/run_all.py` 8/8 通过；冒烟训练 60 步（GPU 4）全链路通过并记录 `dev/mse`；
5 个探针 JSON 重跑；`config_signature` 实打印确认新字段。

**未做（留待用户决策）**：
- 未重训任何正式 run，因此 M1 的「用 dev 重新选点后复跑 30k/60k/90k」尚未执行；
- M3 未逐目录复核历史 `reports/eval/bt_*` 是否真的混入过残留帧（建议抽检 2 个目录）；
- M4 的「冻结解码金标准」像素哈希回归测试未加（需 2 GB checkpoint + GPU，会拖慢无卡机器；
  现以 sha + 全量 key 覆盖 + 签名记录替代）；
- N8（把 GFSLT 仓库的标签/打分脚本收进本仓库、去掉硬编码路径）未做；
- N7（resume 后 epoch 重置导致批序重放）未处理。
- 仓库此前**零 commit**：本轮已产出**首次 commit**（见 `git log`），N9 的「签名绑定代码版本」从这次提交起生效（此前 `git_head` 为 null）。


## 2026-10-13：文本→视频语义监督专项分析（JiT / DiT）

用户明确希望改善文本决定视频语义的能力，指定阅读JiT与DiT，GPU时间无硬限制。本轮检查正式日志、SignMbart结果、模型/条件/损失与探针，核验DiT、JiT、SD3、PixArt原论文及官方实现；没有改训练代码或启动训练。

结论：prefix90k有弱文本依赖，但32条test B1=14.75/B4=0，原视频B1=50.42/B4=29.51，绝对语义差距大。DiT/JiT主结果为类别条件图像生成，不能作为自由文本到视频已解决的证据；adaLN-Zero条件优劣也不能直接推广到句子时序。JiT预测x但计算速度换算loss，当前是cosine VP+直接x0 MSE。prefix共享模态参数，不是完整MMDiT。

纠偏：1-NN“天花板”是检索基线、不是上界；低阶/PCA/稀疏动态探针低R²不证明完整latent缺语义。原视频BT不是冻结VAE重建参考，需单独评测。dev/mse已有代码与冒烟结果，但旧正式run未记录，训练也未自动保存best-dev。

下一轮建议先做dev真实latent解码回译、分噪声段条件依赖、32条纯噪声生成语义过拟合，再单因素测条件通道、现有配对训练的独立语义编码器/辅助loss、噪声采样和x0输出配方；双流/文本时间规划随后。固定VAE/latent/x0/text-only/SignMbart约束不变。完整证据与判据见TEXT_SEMANTIC_ADVISORY_20261013.md；TODO保持不变。

## 2026-10-13：用户要求文本 encoder 可学习 + 直接手语视频文献核验

**用户决定**：文本编码器不再固定冻结，要基于预训练模型参与微调；VAE/latent/x0/text-only/SignMbart 约束继续适用。

**检索结论**：FlashSign 是 pose-free RGB 视频路线，但实验为 WLASL 孤立词、长句仍受限，UmT5 是否更新未明确，作者仓库只有 README；TuneSL 是直接短视频生成（出版商摘要可读），推理有单帧 DDIM inversion 结构引导；2026 Text2Sign 是直接短 RGB 视频但语义 prompt 区分很弱。SignGen 实际融合 pose/flow/depth，SignViP 使用 pose/3D-hand token 中介，不能当作严格同条件基线。Text2SignDiff 输出 pose，句子级对比对齐只作为机制参考。SignMoD 全文未获取，仅列待核实线索。JiT 的可学习条件是类别 embedding，不是微调自然语言 encoder。

**本地设计**：建议预训练 mT5-base 全量微调、独立较低 lr（初始 1e-5，对 DiT 1e-4），保存/恢复文本权重，动态重编码 CFG 空文本，dev/可视化同步 eval 模式，并以一致 raw 状态评测。已发现 no_grad、optimizer 参数集、全组 lr 覆盖、缺文本 checkpoint 与缓存空文本五处必须联动；不能只改 requires_grad。完整文献证据及有限改动设计见 `SIGN_VIDEO_LITERATURE_20261013.md`。

**状态**：完成检索/代码范围检查/设计，尚待 brainstorming 要求的短设计确认；未修改训练代码、未启动训练。TODO 不新增条目。

## 2026-10-13：brainstorming 动作语义提升的多方案比较

**目标**：用户要求多个方法供选择；本轮为探索性建议，不是执行上一轮文本微调设计。语义成功定义为输入句子的内容、关系、顺序、关键细节在生成动作中体现；固定 VAE / latent / x0 / text-only / SignMbart。主文本 encoder 按用户决定可微调，现有代码仍冻结。

**候选**：A 预训练 mT5 全量领域微调（独立较小 lr）；B 现有 train 文本—视频 latent 配对预训练独立时序语义匹配器，再监督生成 x0；C 独立 latent→德语文本 decoder 的 token 重建约束；D 文本/视频模态独立参数的联合注意力或更直接的 cross-attn；E 由文本预测粗时间条件序列/低频 latent 计划，再由 x0 diffusion 细化；F 分噪声段课程、长度匹配难负样本与少量 GT latent 差分约束作为配套。语义匹配器/反向 decoder 不接收原句作视觉输入，需先验证留出正确/错配区分；重复/近义句不作负样本；不能用升高错文本去噪 MSE 来冒充语义改善。干净数据训出的评判器在生成/噪声分布上可能失效，需校准与轻量扰动、关注独立 SignMbart 而非优化内部评分。

**文献核验**：[Text2SignDiff](https://arxiv.org/html/2509.10845v1) 的跨模态对齐/语义监督、[TMR](https://openaccess.thecvf.com/content/ICCV2023/html/Petrovich_TMR_Text-to-Motion_Retrieval_Using_Contrastive_3D_Human_Motion_Synthesis_ICCV_2023_paper.html) 的对比检索提供 B 的机制依据，二者为 pose/motion 研究，不声称为 RGB 视频直接基线；[SD3](https://arxiv.org/html/2403.03206v1) 支持 D 的模态独立参数设计；[ReAlign](https://arxiv.org/html/2511.19217v1) 提醒时步/噪声分布下的语义评分偏移。C/E 为本项目可检验的适配设想，不声称现有 1-token 手语实验已验证。

**排序**：推荐按 A→B 单因素推进；若仍存在关键词遗漏/关系错误，独立测 C；若文本作用依然弱，测 D；若主要错误是顺序/时间分配，再测 E。F 是配套，不独立视为语义方案。预算只给相对等级；先用同初始化/有效 batch 的 5k–10k 预筛，再根据 dev 结果决定长训，最终候选重复 3 个 seed。

**统一判据**：先看冻结真实 latent 解码的语义保真参考；dev 上固定 seed/解码锚点/长度、正确与长度匹配错配文本配对评测，并核验预测长度部署；绝对 SignMbart BT 与配对差一起改善，FVD/多样性作为保护指标。难文本反事实对用于测试/负样本，不能把改词后的句子与原视频作为正对训练。test 不用于方案选择。仅追加思考记录，未实施候选方案，TODO 保持不变。

## 2026-10-13：首选路线 A→B 实施与实验启动

**用户授权**：“请你开始直接做这个首选路线”。在当前仓库执行，保留既有文档改动，未改共享 VAE/缓存/统计，未添加 TODO 条目，未提交或推送。

**A 已实现**：预训练 mT5 全量微调，独立 lr 1e-5 对 DiT/条件模块 1e-4；梯度、优化器、lr 调度、裁剪、文本 checkpoint、CFG 空文本重编码、dev/可视化 eval 状态完整联动。生成还原主文本权重，微调配置缺文本权重直接拒绝；采用一致 raw 权重。已有长度头单独使用原始 mT5 空间，避免主编码器微调导致输入漂移。新 checkpoint 原子替换，写失败保留旧断点。配置 `configs/dit_b_text_finetune.yaml`。

**B 已实现**：独立匹配器在 train7021 文本/视频 latent 配对上训练。有效视频重采样32时间位置，加 latent 差分，2层128维 Transformer 提取128维归一化表示；文本用独立原始 mT5 固定池化特征，经 train 特征统计标准化和可学习 MLP 对齐。重复文本按多正样本处理。视频分支看不到原句；教师冻结后梯度从真实预测 x0 回传 DiT，损失 `1−cosine`，权重 .05/1000步渐增，CFG 丢文本样本不受语义监督。配置 `configs/dit_b_text_semantic.yaml`。

**教师验证**：40 epoch 完成，选 epoch10；128条随机固定 dev（排除与 train 相同文本）R@1=24.21875%。最邻近长度错配逐对准确率88.28125%，gap=.28593、95% bootstrap CI [.25713,.31410]。进一步严格要求完全相同原始长度、不同文本：74条视频满足，逐对准确率91.346%，gap CI下界 .24590，通过 v2 门槛。正确/打乱/倒序平均相似度 .36560/.01518/.21508，倒序R@1=7.8125%。文件 `reports/verification/semantic_matcher_v2.json`；新教师 `runs/semantic_matcher_v1/validated_v2.pt`。这些是 dev 教师检索结果，不是生成视频的回译改善。

**真实验证**：mT5 12步→恢复续到14步，实际 norm 参数最大变化 .00010222；恢复文本及 CFG 空串完全一致，固定seed生成有限且重复一致 [8,1024]。语义6步真实训练、保存与VAE解码成功，产出 `runs/_smoke_text_semantic_v2/samples/step_0000006/`。首次GPU1语义smoke在另一进程抢占显存后OOM，换空闲GPU0成功，没有干预其他进程。最新全套13个独立测试文件通过，含语义梯度、长度捷径拒绝、文本状态保留、原子checkpoint及打乱生成仍使用原始回译标签。

**正式状态**：A 从 prefix60k 同训5000步已启动（GPU4，`runs/route_a_mt5_5k`），已通过1000步checkpoint和GT/生成PNG/GIF/MP4；2000步dev/mse=.02140，去噪损失波动，不能解释为语义改善。后台调度 `tools/run_preferred_followup.py` 已启动：A终点后同初始化分别 B 与 A-only 继续各5000步；同64条dev评正确/打乱文本 SignMbart、配对bootstrap、B−等预算A bootstrap、FVD；另核验B文本预测长度。状态 `runs/preferred_route/status.json`，阶段日志与最终summary同目录。此次尚无正式生成语义增益结论；旧部署不替换，test不用于本轮预筛。

**独立审查修复**：原最近长度 baseline 的排除自身口径不够严格，补完全等长控制与纯长度合成失败用例，重新验证教师；`.gitignore` 原 `models/`/`data/` 意外忽略包内源码，改为根目录模式；恢复自定义文本后再冻结保存也必须保留该权重，增加回归。仍存在预先选5k端点而非自动best-dev、近义文本假负样本、真实latent教师在生成分布可能被利用等限制，下一步以独立 SignMbart 与画质指标判断。

## 2026-10-13：A/B 完成后的 BLEU/FVD 下一轮实验组合

**用户请求**：生成多种后续实验方案与论文启发，重点提高BLEU、降低FVD。本轮为研究建议，未启动新训练或修改训练代码。

**最新事实**：后台A5k、B5k、等预算A-only5k及64条dev全部评测完成。A-only/B的BLEU-1=11.647/12.177、BLEU-4=2.161/1.792、ROUGE-L=10.272/11.249、FD=40.655/39.307；B−A的BLEU-1/ROUGE置信区间都包含0。B不是全指标优胜，正确−打乱差不能直接证明其相对A的文本依赖增强显著。预测长度B的BLEU-4=0，需独立长度误差分组分析。

**纠正上一轮口头判断**：B正式语义loss在1–1000/2001–3000/4001–5000步平均.44264/.36802/.32939，仍下降；约.9是smoke量级，不能据此宣称正式模型在3k平台。dev MSE上升提示目标取舍或过拟合假设，但FD反而降低，尚不能证明画质冲突。延长训练的收益未被排除，需B和A同时等预算续训比较。

**评测校准**：本地`tools/eval_fvd.py`是R(2+1)D-18/Kinetics400特征+整段均匀16帧，非标准I3D-FVD，不与论文绝对值横比。eval缓存519条，按max_num_frames=256过滤后518条；64条用于预筛，候选全dev和多seed确认。补BLEU-4配对统计、VAE真实latent重建参考、时间扰动敏感性；不把VAE重建称严格上限。

**组合方案**：E1低权重/教师可靠噪声区间的语义损失（新增λ=.005/.01/.02，与现有0/.05对照）；E2静态CFG和有限噪声区间CFG；E3借TM2T逆向约束，只用train配对训练独立latent→text逐词教师；E4借MDM的动态一致性思想匹配latent多尺度差分；E5借SD3模态独立参数做部分双流；E6 Min-SNR的x0正确权重；E7文本生成粗latent时间计划作为长句后备。主mT5可训练，冻结VAE/latent/x0/text-only/SignMbart不变。

**推荐顺序**：先诊断和E1/E2，再验证E3教师可否真正读懂GT latent；BLEU路线优先E3，FVD路线优先E4，结构升级E5/E7依赖失败分析。REPA原版外部视觉教师不直接迁移；FlashSign空间tile/孤立词实验不能直接替换当前每帧1token句子模型。SignMbart只做独立评测，不进入训练奖励。完整来源、超参、对照、预算、停止规则见 `NEXT_BLEU_FVD_EXPERIMENTS_20261013.md`。TODO未新增条目。

## 2026-10-13：BLEU/FVD 路线开始实施

按用户授权的路线先落地评测和低风险单因素实验，未改冻结 VAE、latent 缓存、x0 或纯文本约束。

- CFG 支持 `guidance_min_t/guidance_max_t`，只在指定扩散时间区间启用预测空间 CFG；`tools/cfg_sweep.py`、`tools/eval_t2v.py`、`tools/generate.py` 和训练可视化都会把区间写入请求/元数据。单样例 smoke 已完成，产物为 `reports/eval/cfg_sweep_route_b_interval/cfg_sweep.png`。
- 语义权重命令行覆盖和可复现实验调度已加入 `tools/run_semantic_weight_ablation.sh`，固定从 `runs/route_a_mt5_5k/final.pt` warm start，依次跑 λ=.005/.01/.02 的 5k 等预算分支；运行日志在 `runs/semantic_weight_ablation_logs/`。
- 独立 latent→词序列教师 `tools/train_latent_text_teacher.py` 已完成 25 epoch 全量 train 训练。固定 128 条 eval 门控：正确 NLL=2.9657、打乱=3.3786、置零=3.3802，自由解码 token accuracy=.0875，满足预注册的 0.9 NLL 比值和 .05 accuracy 门槛，导出 `runs/latent_text_teacher_v1/validated.pt`。这只证明 GT latent 携带可读文本信息，不等同于生成视频 BLEU 提升。
- 通过门控的教师已封装为 `LatentTextObjective`，冻结教师参数、仅让梯度经过预测 x0 回到 DiT；`configs/dit_b_text_inverse.yaml` 和 `runs/route_c_inverse_5k` 为独立 C 分支。由于 cuDNN GRU 在 eval 模式不能反传，教师只将无 dropout 的 GRU 保持 train mode，Transformer 仍 eval；2 步 GPU smoke 已成功。

当前后台实验：语义 λ 消融串行使用 GPU4，C 分支 5k 使用 GPU0；完成后必须用同一 SignMbart 协议、同一 dev 子集和本地 FD-R2Plus1D 口径比较 BLEU-1/BLEU-4/ROUGE/WER/FVD，再决定是否延长或接入更复杂的时间计划模块。

## 2026-10-14：计划审阅 + 给指标定标（地板/上限）+ 视频侧对照

**用户请求**：审阅 `NEXT_BLEU_FVD_EXPERIMENTS_20261013.md` 的漏洞，并给出更好的方法。
本轮未改训练代码、未改缓存/checkpoint；新增三个只读诊断脚本与四组测量。

**实测（同一 32 条 test、同一 SignMbart / R(2+1)D 协议）**：
- 冻结 VAE 用真实 latent 重建：PSNR 26.37 dB，回译 **B1 31.57 / B2 22.23 / B3 17.51 /
  B4 14.93 / ROUGE 29.32 / WER 91.8 / FVD 28.37**。
- 错配真视频对照（另一条 GT 视频配本句参考文本）：**B1 9.32 / B4 0 / ROUGE 9.22 / WER 121.0**。
- 现役 prefix-60k 生成 9.47 / 0 / 9.57 / 114.2 / FVD 51.61；
  配对 bootstrap 生成 vs 错配真视频 **ΔB1 +0.16 [−5.03, +5.13]，P=0.57**（= 无差别）。
- prefix-90k 则 **ΔB1 +5.44 [+0.58, +10.55]，P=0.99**、ΔWER +17.05 [+3.71,+32.23] → 真有视频依赖。
- DDIM 100 步：9.17 / 0 / 9.42 / FVD 51.42，与 50 步在噪声内相同 → 采样步数不是瓶颈。
- latent 检索（教师 NLL 排名，随机 MRR=0.127 @N=32）：重建 0.246、prefix-90k 0.218、
  prefix-60k 0.135、rolled/zero 对照 ≈ 随机。

**结论**：指标地板 9.3，表示上限 31.6，现役生成贴在地板上；丢分主因在条件生成而非
冻结 VAE。计划里 ±0.5~1 B1 的晋级判据在该指标分辨力以下；建议改用检索 MRR 做筛选、
SignMbart 只做汇报，并先把评测矩阵统一到全 dev + 多生成 seed。完整审阅见
`.trae/documents/archive/NEXT_BLEU_FVD_REVIEW_20261014.md`。TODO 未新增条目。

## 2026-10-14 下午：求解器/VFE 阶梯、回译标签回归修复、外部数据探测

**用户裁决（本轮输入）**：① 不能碰 VAE → 接受上限 B1≈31.6 / FD≈28.4；② 接受把检索 MRR 作
筛选主指标、SignMbart 仅汇报；③ 外部手语数据"先试一下"。

**先修正上一条记录的一个结论**：DDIM 50→100 相同并不代表"采样步数不是瓶颈"。补齐阶梯
（同 prefix-60k、同 32 test、raw、CFG 2.0）后：50 → B4 0.00；100 → 0.00；**250 → 2.77**；
500 → 2.59。B4 的 0 → 2.6~2.8 是一次性地板跳变，之后不再涨；FVD 51.61→51.27 缓慢下降。
所以：**用 50 步做 B4 筛选是无效的**（历史 5k 分支的 B4 比较全部作废），但**样本质量瓶颈
也不在采样步数**——(a) 换二阶求解器、（b）模型/条件本身才是剩余 12 个 B4 点的去向。

**新增求解器 DPM-Solver++(2M)**（`t2v/diffusion/gaussian.py`，`sampling.sampler`/
`sampling.order`，两条采样路径统一走 `sample_latents`）：一阶在代数上等价于 DDIM
（`tests/test_dpm_solver.py` 逐位验证），二阶用 scipy `solve_ivp(rtol=1e-12)` 积分
`dx/dλ = -α²x + α·x0` 作独立参照验证收敛更快。实测（同 checkpoint/协议/32 test）：
DPM++ 2M 20 步 B4 2.02 / MRR 0.137；50 步 **ROUGE 10.12 / WER 110.5 / FVD 51.13 / MRR 0.153**；
100 步 **MRR 0.160**、FVD 51.22。全部优于 DDIM 50/250/500（ROUGE ≤9.89、WER ≥114.2、
MRR 0.135），而 NFE 只要 1/10 → **建议部署与筛选采样器切到 DPM++ 2M 50 步**。
注意换求解器=换采样口径，新旧数字不得同表混用。B4 在 n=32 上本身抖动（2.02/0.00/0.00），
确认不适合做主判据。

**回译标签回归（重要流程教训）**：`tools/run_bt_eval_signmbart.sh` 不传标签时回退到 GFSLT
`make_eval_labels_from_frames.py`，该脚本按固定 test 清单匹配；对 eval 划分的生成匹配到
**0 条却仍退出 0**，写出 `videos: 0` 的全 0 `score.json`（帧数守卫还打印 "ok (64 videos)"），
导致 `reval_dev64_steps250.sh` 的前两个结果作废。修复：新增 `tools/make_eval_labels.py`
（由 meta.json + `<split>_text.jsonl` 生成 `labels.<split>.pkl.gz`），所有启动脚本显式传标签；
`tools/fix_bt_labels.sh` 修复历史空结果；`tools/check_label_path_equivalence.sh` 校验新旧
标签路径产出一致；`tests/test_eval_labels_cli.py` 固化纪律。`reval_dev64_steps250.sh` 改为
分阶段幂等（帧/标签/BT/FVD/检索各自按产物判断），可安全重启。

**dev-64、250 步复评**（SignMbart/SignMbart 同协议）：A5k（`route_a_mt5_5k`）
B1 8.12 / B4 1.64 / ROUGE 7.73 / FVD 44.22；A-only（`route_a_control_5k`）
**10.40 / 2.51 / 9.18 / 40.24** → 排序与 50 步一致（等预算 A-only 优于 A5k），
这是 H3"排序保持"的第一次抽查通过。B 分支与 C 分支的 250 步复评在跑。

**外部数据探测**（`tools/probe_external_data_feasibility.py`，12 条 WLASL + 4 条 PHOENIX
对照，同 VAE/预处理/解码）：PHOENIX 对照编码→解码 PSNR **25.11 dB**，WLASL 只有
**7.49 dB**；WLASL latent 逐帧范数 410±301（PHOENIX 108~181），**38.7%** 的帧落在
PHOENIX p01–p99 之外。工程成本不是瓶颈（编码 ~250 fps/卡，全量 ~1.2 GPU 小时、~5 GB latent），
但**冻结 VAE 在 WLASL 上不成立**，灌进 DiT 会让模型拟合解码器画不出的分布 → 不建议。
本机没有 How2Sign/CSL-Daily；域相近的外部语料必须先做同款探测再谈下载。替代方案：
文本侧增强（德语天气播报同义改写，不碰 VAE，直接针对已确诊的条件利用率低）。

**产物**：`reports/eval/RESULTS_TABLE_20261014.md`（`tools/collect_results_table.py` 统一
出表，所有评测目录按 (checkpoint, split, 采样) 归因）、`reports/eval/external_probe_wlasl.json`、
`reports/eval/prefix60k_s{100,250,500}`、`reports/eval/prefix60k_dpmpp_o2_s{20,50,100}`。
文档：审阅文件新增 §7（`NEXT_BLEU_FVD_REVIEW_20261014.md`）与用户裁决记录（§5）。
TODO 未新增条目。


## 2026-10-14：思辨（research-sparring）— 条件注入方式与 latent token 压缩

**用户主张**：① cross-attn / AdaLN / joint attention 三种文本条件注入都值得试；
② 因为一个德语 text token 可能对应多帧动作，想把多个 latent token 压成一个与文本关联、
再还原回多个 latent。

**本轮反驳（5 条，等待用户逐条回应）**：
1. 三者不在同一分类轴且两种已并存（cross-attn + 池化文本→adaLN 同时开，`text_prefix` 叠加
   joint attention）；真正没试的只有"关掉 cross-attn 的裸 adaLN"变体，约 10 行代码。
2. "token 数不匹配 ⇒ 对齐差"未验证；仓库探针指向利用率（换文本动输出 ~1%、
   cross-attn 有效贡献 ≈ MLP 3%），注意力不要求两侧数量相等。
3. PHOENIX14T 无时间对齐：全部注释 start/end = -1（train 7097 / dev 520 / test 643 /
   complex 7097，非 -1 行数 0）；德语是句级翻译、gloss 均值 7.8 vs 德语词 14.0 vs
   mT5 token 24.4、T 均值 115.6。故"词↔帧"对应无监督信号也无事实依据。
4. 时间维压缩主流在 tokenizer 层（CogVideoX 3D VAE + expert adaLN；W.A.L.T causal VAE +
   级联 SR，身份与摘要已核验）；DiT 内部瓶颈在无重构监督时易退化，且 k 取文本 token 数
   会让压缩比随句长浮动（4–85×）。
5. 现役 SignMbart/32 条 test 口径量不出这类改动（地板 9.32 vs 现役 9.47，
   ΔB1 +0.16 [−5.03,+5.13]）；筛选应先换检索 MRR + 全 dev + 多 seed。

**产物**：`.trae/documents/archive/SPARRING_CONDITIONING_TOKEN_COMPRESSION_20261014.md`、
`.trae/documents/archive/POSITION_CONDITIONING_TOKEN_COMPRESSION_20261014.md`。未改代码/训练/缓存，
TODO 未新增条目。


## 2026-10-14 傍晚：采样口径切换、评测标定、两个新漏洞与一次结论更正

用户三项裁决（不能碰 VAE / 接受检索 MRR 作筛选主指标 / 外部数据先试）之后的执行本轮：

**1. 采样口径（不重训的增益，已定）**
- DDIM 50/100/250/500 阶梯 + DPM-Solver++(2M) 20/50/100 对照：**DPM++ 50 NFE 全面优于
  DDIM 250–500**（ROUGE/WER/FVD/MRR），推理便宜 10 倍 → 新默认采样器 = `dpm-solver++` o2 / 50 步。
- CFG 扫描（prefix-60k、test-32、DPM++ 50）：1.0/1.5/2.0/2.5/**3.0** 单调向上，
  B1 8.37→**14.77**、MRR 0.133→**0.205**、WER 122.7→104.8；FD 从 51.1 升到 54.0（仍远低于静止对照 160.8）。
  → **部署默认 CFG 2.0 是从 RefSign 继承的、没调够**；dev-64 的 2.5/3.0/3.5 扫描在跑
  （`tools/run_cfg_sweep_dev64.sh`）。
- 纪律：换求解器/CFG 都是换口径，旧 DDIM 数字只作历史，不混表。

**2. 评测标定（计划里完全没有的一层）**
- 新建 `tools/eval_vae_recon.py --mode {static,reverse,zero}` + `tools/run_anchor_controls.sh`：
  同一冻结 VAE、同一 **GT 首帧解码锚点**、同一解码路径，只换 latent（32 条 test，正确文本）：
  | 控制 | B1 | B4 | ROUGE | WER | FD | MRR |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | 真实 latent 正放 | 31.56 | 14.93 | 29.32 | 91.8 | 28.37 | 0.246 |
  | 静止（第 0 帧重复） | 7.55 | 1.88 | 8.17 | 101.4 | 160.8 | 0.123 |
  | **逆序（真实 latent 倒放）** | **14.97** | 2.79 | 12.80 | 111.9 | **32.07** | **0.241** |
  | 全零 latent | 4.66 | 0.00 | 4.62 | 99.4 | 709.2 | 0.127 |
  | 现役 prefix-60k 生成 | 9.47 | 0.00 | 9.57 | 114.2 | 51.61 | 0.135 |
  两条结论：(a) 现役生成只比"静止真首帧"高 1.9 B1，与错配真视频地板（9.32）不可分；
  (b) **把真视频倒放仍比现役生成好**，且 FD 只涨 3.7、MRR 只掉 0.005 → **FVD 与检索 MRR
  基本对时间顺序不敏感**，BLEU-1 掉一半、B4 掉到 1/5。B4/ROUGE 想测的顺序信息，
  正是这套指标最看不见的。
- dev-64 检索上限新算：真实 latent MRR **0.1986**（随机 0.074）→ prefix-90k 在 test-32 已达上限的 90%。

**3. 两个新漏洞（审阅文件 §1 H14/H15）**
- H14：解码锚点是 GT 首帧 ⇒ 所有数字都含 oracle 信息，表格从未标注；已补静态帧地板。
- H15：文本前缀 120 token 里 ~97 个是 padding，且全部参与自注意力（`tools/probe_prefix_attention.py`：
  latent query 注意力质量 真实词 12.1% / pad 40.2% / latent 47.7%，末层 pad 62.9%）。
  已实现 `model.mask_text_padding` / `truncate_text_prefix`（`tests/test_text_pad_mask.py`
  断言"开 flag 后输出对 pad 内容逐位不变"）。**等预算对照（`runs/route_d_padmask_5k` vs
  `route_d_control_5k`，dev-64、DPM++ 50）证伪了它的收益**：MRR 0.1079 vs 0.1077 持平、
  B1 8.75 vs 10.04 更差、只有 WER/FD 略好 → 机制真实但**不是瓶颈**，降级为省算力选项。

**4. 噪声标定（判断方法前必须先看）**
- 同 checkpoint、同协议、同 seed 两次独立运行：latent 最大差 1.8e-4，B1 11.15 vs 11.77
  （Δ0.62）、ROUGE Δ0.28；叠加回译重打分噪声（σB1 0.27）⇒ **64 条 dev 同配置重跑 B1 波动 ≈0.5–0.6**。
  E1 的 λ 效应（Δ0.62）正好在这个量级 → 判据必须配对或明显大于 0.6。

**5. 结论更正：训练预算阶梯在 dev-64 重做后推翻 §0.3**
- dev-64 / DPM++ 50：prefix-30k B1 7.92 / MRR 0.098；**60k B1 11.77 / MRR 0.112**；
  90k B1 **8.08** / MRR 0.144（配对所有 64 条、1000 次重采样：ΔB1 −3.69 [−6.64,−0.58]，P(>0)=0.01）。
- 即：**BLEU 口径下 60k > 90k（显著），MRR/WER 口径下 90k > 60k**，两个指标排序相反。
  "继续训练到 90k/120k 有用"**不能**当已证结论；30k→60k 是两个口径唯一一致的一步，
  60k 是当前唯一两个口径都支持的预算点。需 ≥3 生成 seed 的配对检验才能定性。

**产物**：`reports/eval/{dev64_dpmpp50,dev64_steps250,dev64_prefix_ladder,anchor_control_*,vae_recon_dev64,
gt64_rotated,prefix60k_dpmpp_o2_s*,prefix90k_*}`、`reports/eval/RESULTS_TABLE_20261014.md`（含退化控制段）、
`runs/{route_d_padmask_5k,route_d_control_5k}`。审阅文件新增 §8.1–8.7。
结构改动已同步 `STRUCTURE.md`；**TODO 未新增条目**。

## 2026-10-14：idea-generation — 突破"文本条件监督不强"的方案组合

**用户请求**：找方法突破文本条件监督不强的问题。未改训练代码、未启动训练、未动 VAE/缓存。

**归因诊断（读现有全部证据后）**：主瓶颈 **E 监督信号错配**——x0-MSE 对"用文本"没有边际压力。
三条独立证据指向同一结论：① A（mT5 全量微调）把 MRR 从等预算对照的 0.137 压到 **0.078**（加文本能力无用）；
② B（句向量语义损失）把 MRR 拉到 **0.171**、λ=0.10 时 B4 2.46（加目标函数有用）；
③ text dropout 0.10→0.05 全面劣化 + CFG 1.0→3.0 把 B1 从 8.37 拉到 14.77（信号在但弱）。
并列闸门是 **F 评估**（地板 9.32 ≈ 现役 9.47；逆序真视频 14.97 优于所有生成模型；
FVD/MRR 对时间顺序几乎不敏感，正是 B4/ROUGE 想测的东西）；**D 数据**是放大器（7021 条、
句级翻译、无时间对齐、一对多 → 条件均值塌缩）；**B 表示**设硬上限（31.6/28.4，VAE 冻结）。

**分层方案**（26 条候选 → 8 张卡片，已迁入 `.trae/documents/THINK.md`）：
- L3：`idea-001` 显式文本依赖目标（差分/对比损失 + 条件残差分解）、`idea-002` 无标注时序对齐压力
  （单调注意力偏置/对齐似然 + 顺带产出顺序敏感指标）、`idea-003` 文本多样性对抗条件均值塌缩。
  三条各配半天证伪实验（32 句文本交换敏感度 / 真实 latent 块置换敏感性 / 同长度互检索混淆率）。
- L2：`idea-004` 条件路径重构（含容量匹配对照）、`idea-005` 语义目标升级（λ(t)+序列级教师）、
  `idea-006` 文本侧 LoRA（诊断 A 的负结果）、`idea-007` 域内 REPA + 多尺度动态损失。
- L1：`idea-008` 止血包（采样/CFG 收口、统一 dev 评测矩阵、顺序敏感指标、self-conditioning、
  Min-SNR、长度头）。

**路线图**：先做 C02/C03 量尺（无训练）→ 两个半天证伪实验 → 通过的 L3 与 idea-005/C04 并行 20k。
判据：先过"MRR ≥ 1.5× 随机 + 不输逆序对照"，才允许上 SignMbart/FVD。

**限制**：本轮 arXiv/S2 全部不可达（SSL/429），文献表未生成；已核验的按"我自己核验/仓库既有记录/
待核验线索"三类标注，L3 的新颖性判定暂记待核验。

**未决（需用户裁决）**：train 标注的 gloss（orth）能否作为**文本侧**中间表示——硬约束只写"不引入
pose/reference 等额外监督"，gloss 是否算额外监督需用户定。

**产物**：`.trae/documents/THINK.md`（原 `docs/ideas/IDEA-BANK.md` + `idea-001..008` 卡片，2026-10-14 并入）、
`.trae/documents/archive/BRAINSTORM_TEXT_CONDITIONING_20261014.md`（原 `docs/ideas/brainstorm-20261014-text-conditioning.md`）。
TODO 未新增条目。

## 2026-10-14 晚（用户指令：评测规范 + T1/T2 执行）

### 问题 / 指令

1. 用户新增两条规范（写入 `RULES.md` 评测规范第 6/7 条与 `AGENTS.md`）：
   - **双集配对评测**：任何指标对比必须同时给 test 与 train（采样到同 N、同协议）；
     采样清单落盘复用；**暂时不用 dev 集筛选**。
   - **决策入 TODO**：所有定下来的结论登记到 `.trae/todo/TODO.md`（agent 获授权新增条目）。
2. 用户裁决 T1、T2 都可执行（此前 T0 已完成；T3 表示层/外部数据已关闭）。

### 结论 / 产物

- 新增规范落地工具：`tools/make_eval_sample.py`（固定种子采样清单，
  `reports/eval/samples/{test,train}_n642_seed42.txt`，test 642 = 全量 test，train 7021 抽 642）、
  `tools/eval_matrix.sh`（一次跑 test+train 两侧：gen→labels→SignMbart→FVD→检索 MRR，
  两侧协议完全一致）、`tools/prune_frames.py` + `tools/decode_latents_dump.py`
  （评测后删帧省盘，`latents.pt` 保留、可确定性重建）。
- 协议要点：train 侧 MRR 用的是在 train 上拟合的 latent→文本教师（`latent_text_teacher_v1`），
  读 train 数字时要记住这一项偏向乐观；其余指标（BT/FVD）口径两侧一致。
- **T1.5 self-conditioning 已实现并测试**：`model.self_conditioning` 把上一步 x0 估计拼进输入
  （`x_embedder: Linear(2·latent_dim, hidden)`），新的一半零初始化；
  `expand_self_conditioning_state_dict` 让 60k 冻结文本 checkpoint 可直接热启动
  （零输入时逐位复现父模型）；训练按 batch 50% 概率走两遍前向（`use_self_cond`）；
  DDIM 与 DPM-Solver++ 都支持两遍采样；`tests/test_self_conditioning.py` 覆盖
  「零半复现父模型 / self_cond=None 等于零 / 两个采样器真的换了样本 / loss 路径 / 非 SC 模型不受影响」。
- **T2.9 文本压力课程已实现并测试**：`conditioning.text.dropout_schedule{start,end,warmup_steps}`
  （句级 CFG dropout 0.30→0.05）与 `word_dropout_probability`（词级 dropout，
  实现方式是"按采样出的位置把词换成 pad 并重编码"，不是事后清零，
  所以被丢词的上下文不会残留）；`tests/test_text_word_dropout.py` 断言
  「可见 token 全是 special / 掩码不新增 key / 不出现空句 / 给定 rng 可复现 /
  等价于手工重编码的结果（无泄漏）」。
- **T2.10 文本侧 lr**：新增 `configs/dit_b_text_prefix_textlr.yaml`（text lr 1e-5 → 5e-5）。
- 新配置：`dit_b_text_prefix_sc.yaml`（T1.5）、`dit_b_text_prefix_curriculum.yaml`（T2.9）、
  `dit_b_text_prefix_textlr.yaml`（T2.10）；三者均 60k 热启动、20k 步，冒烟 12 步通过。
- **GPU 队列调度**：`tools/gpu_queue.py`（队列文件即控制面，可运行时追加；每张卡一个作业，
  显存阈值判空闲，失败重试）。队列 = 3 个 20k 分支 → 60k→120k 续训（T2.14）→
  各分支 test/train 配对评测 → prefix90k 的 train 侧 → CFG 3.0 与区间 CFG 的 test/train 配对。
  所有评测 `PRUNE_FRAMES=1`（删帧留 latents）。
- 评测规范带来的口径变更：**dev-64 的一切筛选结论作废**（含此前排队未跑的 dev64 CFG 扫描，
  已取消），全部改在新矩阵上重做。
- 磁盘：`/data` 只剩 ~7 GB → 清理一次性 `runs/_smoke_*`（17 GB）与已被取代的旧评测帧
  （12.6 GB，`latents.pt` 保留可重建）。

## 2026-10-14 深夜（T1/T2 执行续：REPA、长度头、可达性、队列与磁盘）

### 问题 / 指令
用户重申两条规范（已落 `RULES.md` 第 6/7 条与 `AGENTS.md`）：**双集配对评测**（test + train
采样同 N 同协议，暂不用 dev）与**决策入 TODO**；并裁决 **T1、T2 都可执行**。

### 结论 / 产物

- **T2.11 表示对齐（REPA）实现完成**：`t2v/models/repa.py`（`REPAObjective` + `BlockFeatureRecorder`）
  + `DiT._last_prefix_len`（文本前缀模型切帧 token 用）+ `train.py` 接线（`repa.*` 配置）；
  teacher 固定为本项目冻结 latent→文本编码器（`runs/latent_text_teacher_v1/validated.pt`，
  3.77 M 参数），经 `LatentTextTeacher.encode_bins` 取每 bin（64）隐藏态作为对齐目标，同模态同步长、
  无外部数据、不新增条件；投影头只在训练期存在（不进 checkpoint、不改变生成路径，见 D12）。
  `tests/test_repa.py` 覆盖：encode_bins 与 encode_video 一致、teacher 冻结、
  梯度只到学生特征与投影头、优化后对齐损失下降、hook 只返回帧 token（前缀被切掉）、
  hook 可移除。12 步冒烟训练通过（train/repa 已进 metrics）。
- **T2.13 语义可达性复核（已完成，结论见 D11）**：新增 `tools/probe_semantic_reachability.py`，
  在 test 32 条上量冻结 matcher（`validated_v2`）的 paired gap（同长度负例协议与门控一致）：
  - 真实 GT latent：gap **+0.327**（CI low +0.262）、r1 0.438、pair_acc 0.875；
  - 加噪 t=100/250/500/750：+0.328 / +0.327 / +0.313 / **+0.228**（r1 在 t=750 掉到 0.219）
    → 语义信号在整个前 3/4 轨迹上几乎不衰减；
  - 生成 latent（prefix-60k / DPM++ 50 / CFG 2.0，同 32 条）：gap **+0.205**（CI low +0.143）、
    r1 0.219。
  → 语义目标在生成 latent 上**仍有信号**（约 GT 的 63%），瓶颈在 matcher 自身判别力
  （GT gap 只有 0.33），**继续加语义权重不是解法**；这与 λ 扫描（0.005/0.01/0.02 无增益）一致。
  产物：`reports/eval/semantic_reachability_prefix60k.json`。
- **T1.7 长度头分布化**：`predict_length` 增加 `expectation`（softmax 加权桶中位）readout，
  训练脚本 epoch 选择改为 **train 内部固定种子 15% holdout**（不再用 dev，D10），
  并在 test（642）与 train 采样（642）上并列汇报两种 readout 的 MAE。
（8/32 桶 × median/expectation 四组对照数字见本节末尾「长度头对照」。）
- **T1.6 采样侧**：区间 CFG 与 rescale 已排入队列（全程 3.0 / t∈[200,999] / t∈[0,200] /
  rescale 0.7，均为 test+train 配对）。
- **运维（D13）**：队列加磁盘守卫（可用 <12 GiB 暂停起新作业）；本次清理 17.2 GiB
  （重复 checkpoint 改硬链、已完成 run 的 `last.pt`、可重建的评测帧），可用空间 27 → 44 GiB。
  三条全量 642 矩阵（prefix60k test/train、prefix90k test）在跑，生成速率约 6 条/分钟
  （≈1.8 h/侧），跑完自动删帧。

### 长度头对照（T1.7，test 642 / train 采样 642，30 epoch，train-holdout 选点）

| 桶数 | 选点 readout | test MAE (median 回读) | test MAE (expectation 回读) | train 采样 MAE (median / expectation) | test 桶准确率 |
| --- | --- | ---: | ---: | ---: | ---: |
| 8 | median | 17.26 | **16.37** | 16.48 / 15.85 | 0.422 |
| 8 | expectation | 17.48 | 16.44 | 18.42 / 16.22 | 0.452 |
| 32 | median | 18.04 | 16.27 | 16.36 / 16.07 | 0.111 |
| 32 | expectation | 18.76 | 16.31 | 16.55 / 14.76 | 0.143 |

读法：**readout 比桶数重要** —— 同一模型下 `expectation` 回读比 `median` 回读在 test 上低
0.9–2.4 帧（旧 head 的 dev MAE 是 18.29，口径不同不可直接比）；8 桶与 32 桶在 `expectation`
下差 0.1–0.2 帧（噪声内）。→ 部署取 8 桶 + expectation（D14）。长度绝对误差仍大
（平均句长 ~110 帧、MAE ~16 帧），所以 predicted-length 回译列才是真正的判据（已排队）。

## 2026-10-14 深夜二（首次全量 642 双集配对矩阵 + 队列/磁盘/工具三处修复）

### 问题 / 指令
用户重申两条规范后裁决 T1、T2 都可执行；随后追加"训练不用每几百步存 checkpoint，
只在最后存一次"。本轮把双集配对评测第一次真正跑到全量 642 双侧。

### 结论一：双集配对矩阵（test 642 / train 采样 642，同协议）

| run | 指标 | test | train | Δ (train−test) |
| --- | --- | ---: | ---: | ---: |
| prefix60k 60k | BLEU-1 | 10.88 | 12.07 | +1.19 |
| prefix60k 60k | BLEU-4 | 2.14 | 2.03 | −0.11 |
| prefix60k 60k | ROUGE-L | 10.32 | 11.20 | +0.87 |
| prefix60k 60k | WER ↓ | 106.3 | 103.3 | −3.0 |
| prefix60k 60k | FD(R2+1D) ↓ | 22.44 | 21.64 | −0.79 |
| prefix60k 60k | 检索 MRR | 0.0165 | 0.0153 | −0.0012 |
| prefix90k 90k | BLEU-1 | 11.63 | —（已排队） | — |
| prefix90k 90k | BLEU-4 | 1.70 | — | — |
| prefix90k 90k | ROUGE-L | 10.28 | — | — |
| prefix90k 90k | WER ↓ | 106.5 | — | — |
| prefix90k 90k | FD(R2+1D) ↓ | 20.62 | — | — |
| prefix90k 90k | 检索 MRR | 0.0306 | — | — |

- 协议：DPM-Solver++(2M) 50 NFE、CFG 2.0、oracle 长度、SignMbart 回译、R(2+1)D FVD、
  冻结 latent→文本教师检索；两侧同一份固定种子采样清单（`reports/eval/samples/`）。
- **过拟合信号温和**：BLEU-1 训练侧比测试侧高 1.19、ROUGE-L 高 0.87，FVD 几乎相同，
  检索 MRR 在训练侧**没有**更高（0.0153 vs 0.0165）。原先担心的"检索教师拟合在
  train latent 上因此 train MRR 偏乐观"在这一档上量级很小（对照地板两侧都是 0.0110）。
- **FVD 随 N 变化极大**：全量 642 上现役生成 FVD 20.6–22.4，低于历史 32 条口径下
  "冻结 VAE 重建"的 28.37。→ 参照锚点必须同 N 重测（TODO 14），否则"离上限多远"读不出来。
- **60k vs 90k 仍然互相矛盾**：642 test 上 90k 的 B1 更高（11.63 vs 10.88）、FVD 更低
  （20.62 vs 22.44）、检索 MRR 高 1.85×（0.0306 vs 0.0165），但 B4 更低（1.70 vs 2.14）。
  dev-64 上"90k 的 B1 显著更低"的结论在 642 全量上不成立。D4（部署预算）仍需 ≥3 生成 seed
  的配对检验（TODO 11），本轮不做裁决。
- **MRR 必须按 N 归一**：随机水平 ≈ (ln N+0.577)/N，N=642 时地板 = 0.0110。因此
  D3 门槛改写为"相对该侧零信息对照的倍数"：60k = 1.5× 地板（正好压线）、
  90k = 2.8×（当前唯一有实质检索能力的行）。

### 结论二：三处运维/工具修复（都已生效，见 TODO D13）
1. **队列解释器 bug**：`bash -lc` 下裸 `python` 落到 conda base（无 torch），8 条训练作业
   全部会静默失败（route_e/route_f 已经 GIVEUP 两次）。修法：队列把行首裸 `python`
   重写成绝对解释器，并新增 `#done=<path>` 指令让重启可跳过已完成的训练。
2. **checkpoint 刷盘**：`checkpoint_every: 1000` 让三条并行训练每 ~3 分钟各写 2.9–6.2 GB
   （route_g 带文本侧优化器状态 = 6.2 GB），/data 被打到 100%、可用 40→19 GiB，还拖慢共盘
   评测读帧。按用户指令**改成只在结束时写 `final.pt`**（`--checkpoint-every 0`），
   配置文件默认同步为 0；清掉重启遗留的 3 个 `last.pt` 释放 11.2 GiB。
3. **FVD 真视频特征缓存**：`tools/eval_fvd.py` 加 `--ref-cache/--fill-ref-cache/--ids-file/
   --load-workers`。真视频侧只取决于（split, 采样清单, 帧数, backbone），首次算完写
   `reports/eval/fvd_ref/<split>_<idhash>_f<N>_r2plus1d18.npz`，后续复用；键是 id 列表哈希
   且加载时复核 id。已用 smoke8 验证命中/换样本两条路径，数值与旧实现逐位一致
   （FVD 90.38007680026121 不变）。动机：单侧 642 条真视频读 PNG ≈ 20 分钟，
   队列里还有 ~25 次双侧 FVD。
4. **踩坑记录**：`tools/eval_matrix.sh` 在 17:18 起的三条作业运行期间被改动过（17:34），
   bash 边读边执行导致换行错位，两条 test 侧跑完但**跳过删帧、也没打印收尾行**，
   train 侧那一条的检索重定向文件名也被错位写成 `_train`。教训：**long-running shell
   脚本在作业跑完前不要重写**（要改就改副本或用 `bash < <(cat ...)` 一次性读入）。

## 2026-10-15：T1/T2 训练臂与 CFG 扫描的配对裁决（首轮出池）

**问题**：2026-10-14 晚队列的 8 条训练 + 27 个双侧配对评测全部跑完后，各臂的差值是否超过
噪声、哪些设想可以出池。裁决分析全文见 `.trae/documents/archive/T1T2_ARM_VERDICTS_20261015.md`；
读数表见 `reports/eval/MATRIX_TABLE.md`。

**工具（本次新增，已自检）**：

- `tools/paired_arm_stats.py`：两臂在**同一批 642 条视频**上做逐视频配对 bootstrap。
  BLEU 用 sacrebleu 的可加 n-gram 统计量、WER 用可加的编辑数/参考词数、ROUGE 用逐视频均值；
  四个指标都能精确复现官方 `score.json`（prefix60k test：B1 10.88 / B4 2.14 / ROUGE 10.32 /
  WER 106.32）。**口径坑**：官方 bleu1/bleu4 是分别以 1-gram / 4-gram 为最大阶数各自算的，
  直接取 4-gram BLEU 的 `precisions` 是另一个数。
- `tools/paired_mrr_stats.py` + `probe_latent_retrieval.py --dump-ranks`：落盘每条视频自己
  caption 的排名，使**检索 MRR 也能给配对 CI**。13 条臂的 rank 在 `reports/eval/ranks/`，
  逐条复现 `retrieval_matrix_*.json`。

**结论一：idea-006 的 text_lr 臂有效（本轮最高性价比）。**
`route_g_textlr5e5_20k`（text_lr 1e-5→5e-5，从 60k 分叉再训 20k）四项全显著超基线：
ΔB1 **+2.07** [+0.94,+3.25]、ΔB4 +1.29、ΔROUGE +2.06、ΔWER（改善）+5.21、
ΔMRR **+0.0127** [+0.0079,+0.0187]（0.0165→0.0292，2.66× 地板）。FVD 20.29 为全部生成臂最低。
用 1/3 额外预算打平 120k（ΔB1 +0.39 n.s.），FVD 更好。

**结论二：idea-011 的预算结论被改写。**
`prefix_120k` 相对 60k 显著提升：ΔB1 **+1.68** [+0.54,+2.82]、ΔMRR +0.0171
[+0.0106,+0.0248]（0.0336，3.06× 地板，全场最高）；相对 90k 在 B4 +1.49、ROUGE +2.31、
WER +5.98 上显著更好（B1/MRR 不显著）。**旧的"60k 是最佳部署预算点"（32 条 dev/ddim 口径）
不成立**。FVD 上 90k（20.62）优于 120k（24.62），但 FVD 只作汇报。
TODO 11 的 ≥3 生成 seed 配对检验仍未做，单点排序不算定论。

**结论三：CFG 扫描有效，低噪声段单独加 CFG 有害。**
同一份 prefix60k 权重上：全程 CFG 3.0 → ΔB1 +1.40、MRR 1.96× 地板；
**仅高噪声段 t∈[200,999] → ΔB1 +1.84** [+0.98,+2.75]、MRR 2.00× 地板，是解码侧最好的一格；
仅低噪声段 t∈[0,200] → B1 −0.77 n.s. 但 **B4 −1.59 / ROUGE −1.80 / WER −2.90 显著亏**，否决。
区间化相对全程 CFG 3.0 只把 WER 拉显著（+1.76 [+0.01,+3.67]），B1 与 MRR 未显著更好。

**结论四：长度闭环成立（部署扣分项解除）。**
predicted-length 列（部署用文本预测长度替代真值长度）ΔB1 +0.69、ΔB4 −0.24、
ΔMRR −0.0018，**均不显著**，即部署不需要真值长度。

**结论五（交叉观察，后续选点必读）**：
① 两个正向结论都是"便宜"的改动（一个超参、一次加预算）；四个机制改造（双流、课程、
REPA、动态损失）全部不显著或显著为负。
② self-conditioning 的 MRR 涨而 B1/B4/ROUGE 显著跌——**MRR 作唯一门槛时必须同时报回译三项**。
③ 所有方法臂都只在 CFG 2.0 下评过，而 CFG 区间化白拿 +1.84 B1；后续选部署点应做
臂 × CFG 的 2×2，不能拿现在的单列排名直接定部署。

**FVD 口径提醒（本次核实）**：矩阵里的 FVD 随 N 变化极大，N=642 时现役生成 20–27，
历史 32 条口径下"冻结 VAE 重建"是 28.37 —— **跨 N 不可比**，同 N 锚点重测仍待 TODO 14。

## 2026-10-16：方法学教训 —— 对照臂卫生与 MRR 门槛（源自 FAIL-023~026）

> 本节只留**可推广的方法学结论**；四条 idea 的具体裁决与失败链路在 `FAIL.md`
> FAIL-023~026，指标明细在 `EXPERIMENT.md` exp-036~040。按规范 9，被证伪的 idea 本身写在 FAIL。

D28 队列（idea-001 margin / idea-002 softalign / idea-003 paraphrase / idea-017 固定槽接口）
四条免训练前置探针全部 PASS，训练后**全部"无提高"**。除了记 FAIL，从中能提炼两条以后每次开臂
都要先过的检查。

**教训一（对照臂必须干净，否则结论无法归因）。**
TODO 原本给 margin/softalign/paraphrase 三条声明的等预算对照是 `route_g_textlr5e5_20k`，
但该臂是 idea-006 的**干预臂**（开了文本微调 5e-5），带自己的变量，不能当纯 recipe 对照；
项目里**从未跑过"60k 分叉 + 20k + 无文本微调 + 无额外 loss"的纯续训臂**。因此这三条本轮只能按
**vs prefix60k** 报告，该对比把"续训 20k 本身的净效果"与"干预效果"混在一起，读差值要扣掉这个
未知量；补跑纯续训对照已登记 TODO **N5**。教训落到纪律上：**宣布对照关系之前，先打开那条臂的
配置确认它是干净的**（这是 D28 自己写下的规矩，却在记录时挑错了臂）。

**教训二（MRR 单门槛漏洞第三次复现）。**
idea-006 的 text_lr 5e-5 仍是当前唯一显著正向的续训干预（exp-005）。本轮 margin 与 paraphrase
都呈现同一模式：**回译三项显著亏、只有 MRR 显著正**（margin ΔB1 −1.36 / ΔB4 −0.96 /
ΔROUGE −1.70，MRR +0.0050；paraphrase ΔB4 −1.13 / ΔROUGE −1.47，MRR +0.0036）。这与
`FAIL-021`（self-conditioning）、`FAIL-019`（课程臂）是同一模式的第 3、4 个实例，再次确认
**D20：MRR 不能作唯一门槛，必须同时报回译三项**。softalign 是唯一"四项回译 + MRR 全部 n.s."
的臂（最接近中性，但仍无提高）；固定槽接口则被等参数对照判死——处理臂与对照臂四指标 n.s.，
"槽身份稳定"不构成收益来源。

**教训三（累计证据：改"条件使用方式"这条轴已饱和）。**
到本轮为止，非预算、非超参的机制类改动累计九条（双流、文本压力课程、REPA、动态损失、
self-conditioning、margin、softalign、保义改写、固定槽接口），**全部零收益或显著为负**。
可复用的正向结论仍只有两条便宜路线：文本分支 lr 5e-5（exp-005）与加训练预算（exp-004）。
新提案若仍落在这条轴上，应先假定它大概率无效，并用干净的等预算对照先证伪。

## 2026-10-17：route_b 重判为正常实验（用户撤销 D30 / 新增 D34）

**用户指令**：文本编码器微调重新允许；`route_b`（`trainable: true`）按正常实验记录，不再当"违规历史产物"。

**规则层改动**：`RULES.md` 方法硬约束第 6 条由"新臂默认冻结文本编码器"改写为"文本编码器可微调、
但必须显式声明 `trainable` 与文本侧 lr"；TODO **D30** 标注被 **D34** 取代（原文保留）。
`FAIL-006` / `FAIL-028` 保留各自实测结论，只加补记说明其"禁止微调"的解读已失效。

**用什么做的**：`route_b_semantic_5k` 在 5k 步上同时做三件事——全量微调 mT5 文本塔（lr 1e-5）、
加一条句向量语义辅助损失（冻结 matcher `runs/semantic_matcher_v1/validated_v2.pt`，权重 0.05，
1000 步 warmup）、DiT 与文本塔同时更新。起点是 `runs/route_a_mt5_5k/final.pt`
（route A 全量文本微调 5k 的产物），因此它是"在已微调文本塔之上继续 5k"的臂。

**为什么这算正常实验**：项目硬约束（冻结 VAE、latent 扩散、x0 预测、纯 text→video、SignMbart）
里从来没有"文本塔必须冻结"这一条；D30 是 2026-10-16 临时加的实验纪律，目的是避免"文本侧到底
训没训"成为混淆项。用户撤销 D30 后，判据回到正常流程：只要**声明清楚文本侧状态**，
文本微调就是一个普通变量。

**test 642 单侧结果（DPM++(2M)50 / CFG2.0 / oracle 长度，配对 bootstrap）**：

| 臂 | B1 | B4 | ROUGE-L | WER | FVD | MRR | ×地板 |
|---|---|---|---|---|---|---|---|
| prefix60k（部署点基线） | 10.88 | 2.14 | 10.32 | 106.32 | 22.44 | 0.0165 | 1.50× |
| route_b_semantic_5k | **14.12** | **3.24** | **12.55** | 106.60 | **19.43** | **0.0336** | **3.05×** |

- vs prefix60k（部署点）：ΔB1 **+3.24 [+2.03,+4.46] 显著**、ΔB4 **+1.10 [+0.23,+2.03] 显著**、
  ΔROUGE **+2.23 [+1.05,+3.44] 显著**、ΔWER −0.31 n.s.、ΔMRR **+0.0171 [+0.0103,+0.0247] 显著**。
  即相对部署点，回译三项与检索同时显著提升、WER 不退化。
- vs prefix30k（同 5k 但 from scratch 的参考）：ΔB1 +5.41、ΔB4 +2.11、ΔROUGE +4.38 全显著，
  ΔMRR +0.0160 显著；说明增益不是"多跑了 5k"这么简单（30k 基线本身更弱）。
- FVD 19.43 也是现役臂里最好的，但 FVD 的种子方差很大（见 TODO 11），只作参考不作门槛。

**与 idea-016 gate4 的关系**：`prefix60k_gate4`（零训练，推理期把 cross-attn gate ×4）
同样拿到 B1 12.75 / MRR 0.0313。两者配对比较：B1 −1.38 n.s.、B4 +0.31 n.s.、
ROUGE +0.04 n.s.、**WER +4.85 显著更优**（gate4 更好）、MRR −0.0023 n.s.。
即"零训练 gate 缩放"与"文本微调 + 语义损失"在质量上大体打平，gate4 的 WER 更好，
而 route_b 的 B1/MRR 点估计略高但未达显著。

**同预算单变量对照（2026-10-17 补跑，结论有反转）**：`route_a_control_5k` 与 route_b
同起点（都从 `runs/route_a_mt5_5k/final.pt` 续训）、同 5k、同文本侧 lr 1e-5、都开文本微调，
**唯一差别是有无语义损失项**。它的 test 642 结果是
B1 12.21 / B4 3.37 / ROUGE 11.27 / **WER 104.17** / FVD 19.80 / MRR 0.0209。

| 对比 | ΔB1 | ΔB4 | ΔROUGE | ΔWER | ΔMRR |
|------|----:|----:|-------:|-----:|-----:|
| route_b vs `route_a_control_5k`（干净等预算） | **+1.92 [+0.72,+3.16] 显著** | −0.12 n.s. | **+1.29 [+0.03,+2.58] 显著** | **−2.42 [−4.50,−0.28] 显著退化** | **+0.0126 [+0.0070,+0.0186] 显著** |
| `route_a_control_5k` vs prefix60k | +1.32 显著 | +1.22 显著 | +0.94 n.s. | +2.11 n.s. | +0.0044 显著 |

**这条对照把结论改了**：语义损失并不是"四项全赢"。它用**显著的 WER 代价**换来了
B1/ROUGE/MRR 的显著提升（ΔB4 不显著）。按 **D20**（MRR 过门槛但回译任一项显著退化 →
不晋级），**route_b 不能以"全面优于部署点"晋级**；它相对 prefix60k 的"WER 不退化"是因为
对照双方都还在同一水平线上，而真正拆出语义损失那一项后，代价就露出来了。
反过来看，`route_a_control_5k`（只做文本微调续训、不加语义损失）相对 prefix60k 是
温和正向但 ROUGE/WER 不显著的组合——**说明"文本微调续训"本身贡献了 B1/B4/MRR 的大部分，
语义损失主要是在同一水平线上重新分配 B1/ROUGE/MRR 与 WER**。

**因此当前定性**：route_b 是**有效的正常实验**（用户 D34 已确认），但它**不构成一个可晋级的
净提升**——增益与 WER 代价并存，是否接受这个取舍由用户裁决（TODO **N9**）。
按 D20 纪律记账：**不晋级、不写 FAIL**（因为它确有显著正向项，不是"无提高"），
状态 = 待用户裁决的取舍点。

**登记**：`EXPERIMENT.md` exp-049（指标行）与 exp-053（对照臂）；`FINISH.md` F-002 出池；
决策 `TODO D34`；旧 dev-64 数字（exp-017，MRR 0.171 vs 0.137）只作溯源，不与上表混比。

## 2026-10-17：cross-attention 时序结构与训练预算的关系（idea-019 前置的副产品）

**背景**：为裁决 idea-019（文本自生成布局监督注意力图）跑了只读前置探针
（`tools/probe_cross_attn_temporal_structure.py`，TODO **P1** / `FAIL.md` **FAIL-031**）。
探针本体结论是否决——但它顺带测出了一个值得单独记住的量。

**测的是什么**：对每个 DiT block，取 latent 帧作为 query 时落在文本 token 上的
cross-attention 分布，算不同帧之间这个分布的**平均两两余弦相似度**。
相似度接近 1 表示"每一帧看文本的方式几乎一模一样"（没有时间分工）；
明显低于 1 表示不同时间段在看文本的不同部分（有时间分工）。

**实测（同一批 10–12 句、80 帧、t=400）**：

| 权重 | 余弦相似度（全体 block 均值） | 范围 | 观察 |
|---|---|---|---|
| `prefix60k`（部署点） | **0.9887** | 0.963–0.998 | 各帧几乎完全相同，无时序结构 |
| `prefix_120k` | **0.9370** | 0.817–0.998 | 中间层（1/3/4）降到 0.82–0.90；末层 9–11 仍 >0.99 |

**怎么读**：训练预算从 60k 加到 120k，**中间层的 cross-attention 开始出现时间分工**
（不同帧看文本的不同部分），但**末层始终没有**（>0.99）。这与"文本条件利用率随预算上升"
的图像一致，也和 120k 在 B1/MRR 上继续变好（TODO 11）方向一致——但目前只是**相关观测**，
没有做因果实验，因此**不构成任何晋级依据**，也不改变 idea-019 被否决的结论
（部署点 60k 上确实没有结构可对齐）。

**用途**：这条给了 idea-019 一个明确的复活入口——若日后要在 **≥120k 预算**或
**只监督中间层**的设定下重提该方向，先按 `FAIL-031` 的复活条件重跑探针。
另外它提示"注意力时序结构"本身可以作为一个**诊断量**跟踪模型的条件利用率，
但按 D27，不建议为它单独开训练臂。

## 2026-10-17 晚：语义目标权重 λ 的消融 —— λ=0.01 是去掉 WER 代价的甜点

**背景**：route_b（句向量语义辅助目标，λ=0.05）相对部署点 prefix60k 全正向，但它的
**同预算单变量对照** `route_a_control_5k`（同起点、同 5k、同文本微调 1e-5，只差语义损失）
显示语义损失是**用显著 WER 退化（ΔWER −2.42）换 B1/ROUGE/MRR**（TODO N8 / exp-053）。
按 D20 这不能晋级。于是补跑 TODO N9 选项③：项目里已有 λ=0.005/0.01/0.02 三条同起点、
同预算、同文本微调的臂，只缺 test 642 评测（纯推理，不动权重）。

**测的是什么**：五条臂只差一个量——语义损失权重 λ；其余（起点 `route_a_mt5_5k/final.pt`、
5k 步、文本侧 `trainable: true` 且 lr 1e-5、同采样协议）完全一致。全部 test 642，
DPM++(2M) 50 NFE、CFG 2.0、oracle 长度、SignMbart 回译、同一检索教师。

| 臂 | λ | B1 | B4 | ROUGE-L | WER | FVD | MRR | MRR×地板 |
|---|---|---|---|---|---|---|---|---|
| `route_a_control_5k`（无语义损失） | 0 | 12.21 | 3.37 | 11.27 | **104.17** | 19.80 | 0.0209 | 1.91× |
| `route_b_lambda005_5k` | 0.005 | 11.37 | 1.88 | 9.26 | 108.69 | 20.44 | 0.0199 | 1.81× |
| `route_b_lambda010_5k` | 0.01 | 13.28 | 3.61 | 11.77 | **104.30** | 20.46 | 0.0312 | 2.85× |
| `route_b_lambda020_5k` | 0.02 | 14.30 | 3.41 | 12.25 | 106.15 | 20.04 | 0.0333 | 3.04× |
| `route_b_semantic_5k` | 0.05 | 14.12 | 3.24 | 12.55 | 106.60 | 19.43 | 0.0336 | 3.06× |

**配对 bootstrap 判读（都对无语义损失对照）**：

- **λ=0.005**：ΔB1 −0.85 n.s.、**ΔB4 −1.49 显著亏**、**ΔROUGE −2.02 显著亏**、
  **ΔWER −4.52 显著退化**、ΔMRR −0.0010 n.s. → **全面不成立**。λ 太小不是"减轻代价"，
  而是让语义项变成了有害噪声。
- **λ=0.01**：**ΔB1 +1.06 显著**、ΔB4 +0.24 n.s.、ΔROUGE +0.50 n.s.、
  **ΔWER −0.14 n.s.（WER 代价消失）**、**ΔMRR +0.0103 显著（2.85× 地板）**。
  相对 λ=0.05 的唯一显著差异是 **WER 好 +2.28**，B1/ROUGE 的差都不显著。
- **λ=0.02**：ΔB1 **+2.09 显著**、ΔMRR **+0.0124 显著**，但 **ΔWER −1.99 显著退化**；
  与 λ=0.05 四项全 n.s.（两者等价）。

**结论**：WER 代价随 λ **单调**（λ=0.01 处消失，λ≥0.02 起显著），而检索/B1 增益在
λ≥0.01 就基本到位。**λ=0.01 是"保住 MRR 显著增益、又彻底去掉 WER 代价"的甜点**，
也是本项目**首个满足 D20 晋级口径的语义臂**（MRR 显著正 + 回译无显著退化）。
待用户裁决是否采纳 λ=0.01 为语义目标默认权重（TODO **N9**）。

**口径提醒**：只有同起点同预算同文本微调状态才可比；`route_a_control_5k` 的 FVD 19.80、
λ 臂 20.0–20.5，FVD 差异在种子方差内，不作排序依据。

**补记（2026-10-18，用户裁决回填）**：用户对 TODO **N9** 的裁决为**采纳 λ=0.02 为语义目标
默认权重**（登记 **D35**，取代 λ=0.05 的 `route_b_semantic_5k`）：理由更看重 B1/MRR 的绝对
数值（B1 14.30 / MRR 0.0333 为全表最高），接受显著 WER 代价（ΔWER −1.99 vs 无语义损失）；
λ=0.01 作为"零 WER 代价"备选保留、未采纳也未否决。按 D20 该臂属"用户显式权衡后采纳"，
汇报时必须带 WER 代价。本段上文"待用户裁决是否采纳 λ=0.01"的记录保持原文（当时状态），
以本补记为准。

## 2026-10-18：idea-016 gate 缩放 k=4 采纳为部署默认（用户裁决 TODO **N6**）

**这是什么**：DiT 每一层的 cross-attention（读取文本的注意力）都带一个训练学出来的"闸门"
`gate_xa`（adaLN-Zero 的输出门，逐特征乘在 cross-attn 输出上），它决定文本信息以多大幅度
进入该层。idea-016 不改任何权重，只在**采样时**把 `gate_xa` 统一乘以常数 k——相当于推理期
把"文本推力"调到训练学出强度的 k 倍。k=1 就是模型本来的函数；k 越大文本影响越强。

**为什么值得做**：2026-10-14 的通道探针（exp-021）发现，cross-attn 子层本身对文本**高度
敏感**（换文本相对变化 rel 1.09），但它的实际贡献被 gate 压到 MLP 的 4.6%——即"通道在，
但闸门开得小"。若把闸门开大就能拿到更多文本推力，那是零训练、零 NFE 的纯收益。

**预注册判据（前置探针，exp-048 / TODO N6）**：文本交换敏感度必须随 k 单调上升；否则关闭。
实测 **PASS**：correct vs 打乱文本的输出 gap 随 k 单调上升 0.043→0.060→0.086→0.105
（k=1/2/4/6.6），同噪两点距离 0.318→1.372——机制真实、剂量可控。

**正式评测（exp-047，test 642，同协议 dpm++(2M)50·CFG2.0·oracle 长度，配对 bootstrap）**：

| 臂 | B1 | B4 | ROUGE-L | WER | MRR | MRR×地板 |
|---|---|---|---|---|---|---|
| `prefix60k`（k=1，部署点） | 10.88 | 2.14 | 10.32 | 106.32 | 0.0165 | 1.50× |
| `prefix60k_gate2` | 11.94 | 2.87 | 11.39 | 104.59 | 0.0227 | 2.07× |
| **`prefix60k_gate4`（采纳）** | **12.75** | **3.55** | **12.58** | **101.75** | **0.0313** | **2.85×** |
| `prefix60k_gate66` | 12.48 | 3.20 | 12.35 | 101.59 | 0.0286 | 2.61× |

- k=4 相对 k=1：ΔB1 **+1.87 [+0.84,+2.92] 显著**、ΔB4 **+1.41 [+0.69,+2.13] 显著**、
  ΔROUGE **+2.27 [+1.19,+3.36] 显著**、ΔWER **+4.55 [+2.54,+6.68] 显著改善**、
  ΔMRR **+0.0148 [+0.0085,+0.0220] 显著**（0.0165→0.0313，2.85× 地板）。
- **四项回译 + 检索全部显著为正，且 NFE 不增加**（gate 只在原前向里乘一个系数）。k=6.6 也全
  显著，但 B1/B4/ROUGE/MRR 都低于 k=4（无收益、无必要），故选 k=4。
- **与等文本剂量的 CFG 方案对比（exp-052）**：同为约 1.5× 文本剂量，`prefix60k_cfg30i200`
  （CFG=3.0 仅高噪声段 t∈[200,999]，2× NFE）vs `gate4`（CFG=2.0 + gate×4，1× NFE）——
  回译四项全部 n.s.（ΔB1 +0.03、ΔB4 +0.18、ΔROUGE +0.10、ΔWER +1.13），但 gate4 的
  **MRR 显著更高**（0.0313 vs 0.0219，Δ+0.0093 [+0.0034,+0.0161]）。即 gate 用一半的
  采样成本换到同等回译、更好的检索。

**用户裁决（2026-10-18，TODO N6 选项①）**：**采纳 k=4 为部署默认**，登记决策 **D37**。
落地：`configs/dit_b_text_prefix.yaml` 的 `sampling.cross_gate_scale: 4.0`；
`tools/generate.py` 与 `tools/eval_t2v.py` 的 `--cross-gate-scale` 默认改读 config
（显式传参仍可覆盖，`--cross-gate-scale 1.0` 可复现历史 k=1 数字）。

**口径提醒**：
- 历史表（README 部署口径、第 5 节矩阵的既有行）都是 **k=1** 口径；新评测若沿用部署 config
  会自动带 k=4，**引用旧数字时必须注明 k=1，不能与新数字混比**。
- D17 的"CFG=3.0 仅高噪声段"仍是有效杠杆（exp-007），不撤销；只是部署默认选择更省 NFE 的
  gate k=4。**k=4 与 CFG3.0[200,999] 叠加没测过**，需要时另跑同协议评测再定。

## 2026-10-18：route_q 改判"回译侧有效"、idea-002 复活（用户裁决 TODO **N7**）

**背景**：`route_q` 是 idea-002（窗口化软文本–帧对齐损失）的训练臂，从 `final_60k.pt`
分叉续训 20k。它最初按"四项回译 + MRR 全部 n.s."判 **FAIL-025**——但那次对照用的是
**含续训漂移混杂**的 prefix60k（对照臂卫生问题，见 NOTES 2026-10-16 教训一）。
补跑干净等预算对照 `route_s_cont20k`（exp-042）后重读，结论反转：

| 对比（test 642，配对 bootstrap） | ΔB1 | ΔB4 | ΔROUGE | ΔWER | ΔMRR |
|---|---|---|---|---|---|
| route_q vs `route_s_cont20k`（干净对照，B1 10.57 / B4 1.57 / ROUGE 9.54 / WER 104.57 / MRR 0.0195） | +0.81 n.s. | **+1.07 [+0.40,+1.79] 显著正** | **+1.29 [+0.15,+2.51] 显著正** | −1.15 n.s.（方向偏好） | +0.0001 n.s. |

**要裁的矛盾点**：项目晋级门槛（D3）是检索 MRR/top-1。以往失败模式都是"MRR 涨、回译亏"
（D20 因此要求回译不得退化）；route_q 恰好相反——**回译显著涨、MRR 平**。这种组合没有
预注册过：不能照 D20 判活跃，也不能拿 MRR 判死。且 P9 诊断显示 **MRR 尺子已饱和**
（最好臂达冻结 VAE 重建上限 97%；检索教师把序列压到 64 bin，看不见细时间结构），
"MRR 不动"的证据力本身存疑，但替代尺子（idea-029）尚未裁决采纳。

**用户裁决（2026-10-18，TODO N7 选项②）**：**改判"回译侧有效"**——认可 B4/ROUGE 的显著
正收益；**原 FAIL-025 裁决撤销**（保留条目与完整链路，加补记二）；**idea-002 复活**，
重新进入 `THINK.md` 活跃池（登记决策 **D38**）。后续要不要以更大预算/更强实现重跑、
以及在 D3 门槛与回译信号之间如何定新判据，由用户下一步决定。

**边界（必须随结论一起引用）**：
- 该结论只证明**这个实现**（窗口化软单调对齐 + 该损失权重、20k、文本冻结）在 **B4/ROUGE**
  上有显著正收益；它不是 D20 式的全面晋级，MRR 无显著变化、B1/WER 方向偏好但不显著。
- "回译显著正但 MRR 不动"这一组合的直接后果是**动摇了 MRR 单帽子门槛在当前尺度的
  判别力**，与 idea-029（细时间结构尺子）互为印证；裁决时须一并考虑。

## 2026-10-19：部署口径切换为 EMA 影子权重（全臂，D46）+ 本轮对齐跟进收尾

### 一、EMA 影子权重作为全臂部署默认（有效，用户裁决 TODO **P26** → 决策 **D46**）

**背景**：训练脚本每步维护 EMA 影子权重（decay 0.9999）并写进 checkpoint，但此前**全部 53 条
test-642 臂的 `meta.json` 都是 `use_ema=false`**——即历史数字全是**原始权重（raw）口径**，
EMA 从未被评过。exp-076 先证明两套权重确实不是同一个函数（prefix60k 全局相对 L2 0.1046，
逐张量最大 `blocks.3.cross_adaLN_modulation.1.weight` rel 0.3722），故"没差别"这个默认假设未经验证。

**实测（exp-078，test 642，同臂同协议同种子，只切 `--ema`，配对 bootstrap）**：

| 臂 | raw B1/B4/ROUGE/WER/MRR | EMA B1/B4/ROUGE/WER/MRR | 判读 |
|---|---|---|---|
| `route_t_align_20k`（对齐，k=1） | 12.18 / 3.91 / 12.66 / 100.78 / 0.0232 | **13.51 / 4.66 / 13.32 / 101.45 / 0.0312** | ΔB1 +1.33 显著、ΔB4 +0.75 显著、ΔMRR +0.0080 显著，WER 无显著退化 |
| `prefix60k_gate4`（未对齐，k=4） | 12.75 / 3.55 / 12.58 / 101.75 / 0.0313 | 13.38 / 3.71 / 12.29 / 105.05 / 0.0336 | B1/B4 涨但 **ΔWER −3.30 显著退化** |
| `prefix60k`（未对齐，k=1） | 10.88 / 2.14 / 10.32 / 106.32 / 0.0165 | 10.92 / 1.63 / 9.30 / 108.47 / 0.0184 | 回译持平/略降，仅 ΔMRR 显著涨 |

**结论（用户裁决，D46）**：**所有臂**的部署/评测默认改用 EMA 影子权重。`route_t_align_20k_ema`
（B1 13.51 / B4 4.66 / MRR 0.0312）成为**当时代价最低的回译最强臂**（该头衔已于 2026-10-20 被 `route_t_align_20k_ema_k4` 的 **B4 5.65** 取代，见文末 §「2026-10-20」）。落地：`tools/eval_t2v.py` /
`tools/generate.py` 的 CLI 默认 `use_ema=True`，`--raw` / `--no-ema` 回退；`meta.json.use_ema`
记录实际口径。

**必须随行注明的代价与口径限定**：
- EMA **不是无条件免费**：未对齐的 `prefix60k`（尤其 k=4 档）有**显著 WER 代价**（101.75→105.05）；
  用户显式接受该权衡后采用（与 **D35** 同类的"权衡后采纳"，不是自动晋级）。
- **不得混表**：历史全部 53 条 test-642 数字是 **raw 口径**，新数字是 **EMA 口径**；引用时必须标注。
- 文本微调过的臂（checkpoint 含 `text_encoder` 键）**不支持 EMA**（pipeline 显式报错），本条不覆盖。

### 二、R1（idea-024）跟进与收尾（负结果，出路见 FAIL）

本轮把"R1 能否更强 / 施力方式能否改进"这条线一次走完，**三个方向全部无效**，均已在
`FAIL.md` 立案、`EXPERIMENT.md` 留行（此处只给结论指针）：

- **P20 的三条单变量跟进臂**（exp-081/084）：W（叠语义 λ=0.02）**四项回译全显著退化**；
  X（开文本微调 5e-5）只换来 WER 显著退化；Y（权重 0.1→0.3）**四项全显著退化**。
  → **R1 的正增益自足于 align w=0.1 这一档**，不能靠叠语义项 / 开文本微调 / 加大权重加强。
- **idea-033/034（改施力噪声窗 / 时间分辨率）**：FAIL-036/037，显式改这两处都显著退化。
- **idea-037（改 warp 预算带宽，P24 后续臂）**：FAIL-040，窄 0.05 打平、宽 0.5 显著亏。

**读法**：R1 的增益机制仍未完全拆开（不是"直接监督速度"——见 exp-073 的 route_j；也不是
"噪声窗"或"时间分辨率"或"warp 预算"这些单变量），但它**作为一个整体配方的正增益是稳的**
（四项回译显著为正），且**当前没有任何一个单变量改动能让它更好**。后续若要再动，须换一个
**交互假设**（如 max_frames × 带宽、或与 ref 条件的交互）并重新预注册。

## 2026-10-20：Wan 臂的 mask 极性 bug（作废 20k 结果）与修复

**症状**：`wan_flow_lora_v1`（Wan2.1-1.3B 冻结主干 + LoRA + 新 I/O，20k）在 test-642 上
B1 8.01 / B4 0.81 / MRR 1.02× 地板 / FVD 44.51（raw 档 B1 9.42，低于错配真视频地板 9.72），
即「生成内容与文本基本无关」。

**根因**：`t2v/models/wan_backbone.py` 的 `_block_forward` 把两个 mask 取了反
（`~frame_mask` / `~context_mask`，commit 5fa6293 引入，注释断言「本项目 True=padding」是错的）。实测证据：
1. diffusers `attention_mask` 语义是 **True = 参与**：用 `[T,T,F,F]` 时的输出与「只看前两个 token」
   完全一致（max diff 0.00000），全 False 等价于无 mask；
2. 本项目 `frame_mask`（collate.py 的 `mask[index,:length]=True`）与 `context_mask`
   （`context.abs().sum(-1)>0`）都是 **True = 真**；umT5 缓存实测 64 行里只有 13 行为真、其余精确为 0；
3. 取反后 cross-attn 只看得见 51 个零向量 pad → 文本条件在每个 batch 都被掐断。

**修复**：新增唯一转换点 `to_keep_mask(mask) -> mask.bool()`（不取反），`_block_forward` 改回同向；
训练态加「每行至少一个真」断言；`tests/test_wan_masks.py` 用四个 CPU 单测锁死契约
（极性 / mask 路由 / pad 内容不变性 / L_motion 数学）。

**处置**：`wan_flow_lora_v1{,_raw}` 的 20k 数字标注为「bug 口径、作废、留档」，不进任何对比表。

## 2026-10-20：单臂设计（mlp4096 + L_motion），不设探针

按用户指令，把下列改动**一次性合并**到一条 20k 臂（`wan_flow_mlp4096_motion_v2`）：
1. mask 修复（上文）；
2. 有监督跨 VAE 桥：Wan VAE 编码（256x256 帧 -> [16, F_lat, 32, 32]）→ 每帧取 `chunk=min((t+1)//4, F_lat-1)`、
   空间象限 `t%4` 的 16x16x16 展平（4096 维）作为目标，训 `LatentBridge(1024->4096)`；
3. 两端适配器：输入 `LayerNorm(4096) -> Linear(4096,1536) -> GELU -> Linear(1536,1536)`，
   输出 `LayerNorm(1536) -> Linear(1536,1024) -> GELU -> Linear(1024,1024)`（末层 zero-init，保持第 0 步零流场）；
4. `L_motion = mean_t ||（x0_hat[t+1]-x0_hat[t]) - (x0[t+1]-x0[t])||^2`，x0_hat 由 v 反解
   （`x0 = x_sigma - sigma*v`），weight 0.1 / warmup 2000；
5. 训练仍用 flow matching（velocity 目标）、umT5 缓存、LoRA r64、bs16、20k 步。

**预登记风险**：L_motion 与已失败的 route_j（多尺度帧间差分，FAIL-020）同族，可能零收益；
本臂不设对照臂（用户要求），因此若指标不涨，只能记「合并臂无增益」，不能归因到具体机制。

## 2026-10-20：三条零训练评测的裁决 —— 新最强臂 B4 5.65、容差曲线、旗舰 v2 证伪

### 1. exp-091 免费杠杆叠加臂 → **B4 5.65 为项目历史最高**（部署默认建议改为 `ema + gate k=4`）

在 `route_t_align_20k` 的 EMA 权重上叠推理杠杆，三档零训练（同 checkpoint / 同 config / 同采样器 / 同 seed 42）：

| 档 | B1 | B4 | ROUGE | WER | MRR |
| --- | --- | --- | --- | --- | --- |
| EMA · k=1（原最强） | 13.51 | 4.66 | 13.32 | 101.45 | 0.0312 |
| **EMA · k=4** | **15.92** | **5.65** | **15.42** | **98.68** | **0.0341** |
| EMA · CFG3.0[200,999] | 14.89 | 5.26 | 14.55 | 101.24 | 0.0332 |
| EMA · k=4 + CFG | 14.68 | 5.12 | 14.88 | 98.64 | 0.0332 |

**结论**：① k=4 单独最优（ΔB4 **+0.99** / ΔB1 +2.41 / ΔWER 改善 2.77）；② **k=4 与区间 CFG 不可叠加**
（叠加 5.12 < 5.65 单独），与 exp-069 在 raw 口径上的「饱和」结论**跨口径一致**；③ 5.65 / 13.58（冻结 VAE
重建上限）= **41.6%**，到北极星 10 的缺口 **1.77×**。
**口径与不确定度**：EMA（D46）、单 seed 42；跨种子 B4 极差在 1.0–1.7 量级（TODO 11），
**5.65 的不确定度约 ±0.5，晋级前必须补 seed 43/44**。**不得与 raw 臂混表。**

### 2. exp-092 解码容差曲线 → **BLEU 只对方向误差敏感，对整体幅度缩放几乎免疫**

给 GT latent 注入已知量级扰动，走同一条冻结 VAE 解码 + SignMbart 链（自检档 σ=0 逐位复现 exp-051 上限，
故后续读数有效）：

| 扰动 | rel_err | B1 | B4 | B4/上限 |
| --- | --- | --- | --- | --- |
| 无（自检） | 0.00 | 32.36 | 13.58 | 100% |
| 各向同性噪声 σ=0.15 | 0.15 | 30.01 | 11.83 | 87% |
| 各向同性噪声 σ=0.30 | 0.30 | 24.66 | 8.48 | 62% |
| 各向同性噪声 σ=0.45 | 0.45 | 17.55 | 4.87 | 36% |
| 纯幅度缩放 ×0.85 | 0.15 | 32.41 | 13.45 | 99% |
| 纯幅度缩放 ×0.70 | 0.30 | 32.19 | 13.39 | 99% |
| 纯幅度缩放 ×0.55 | 0.45 | 32.39 | 13.52 | 100% |

**三条结论**：① **幅度缩到 0.55（rel_err 0.45）B4 仍 13.52，几乎无损** → exp-088 派生的
「治欠弥散」方向被推翻；② 同 rel_err 0.45 时各向同性噪声只剩 4.87，**差 2.8 倍** → **BLEU 由方向/结构决定**；
③ 插值出拐点 `r* ≈ 0.37`（B4 掉到上限一半处），而现役各臂 rel_err 0.44–0.51 **全在拐点之外** →
按预注册判据「保真度这条轴已是墙」。
**必须记住的反例**：`_ema_k4` 的 rel_err 0.4703 却拿到 5.65，高于同 rel_err 各向同性噪声的 4.87
→ **误差的结构比误差的大小更重要**；「把 rel_err 压下去」不是可靠的目标。

### 3. exp-093 旗舰 v2（ref-as-modulation）→ **证伪，v2 出池**

| 臂 | 预算 | B1 | B4 | ROUGE | WER | FVD | MRR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `flagship_ref_v1`（ref 进 cross-attn context） | 60k | 11.47 | 3.04 | 10.53 | 106.1 | 19.01 | 0.0222 |
| `flagship_ref_v1_k4` | 60k | 9.76 | 2.76 | 9.70 | 102.3 | 19.11 | 0.0189 |
| `flagship_ref_v2`（ref 只调制视频 token） | 20k | 6.56 | 0.70 | 5.96 | 106.0 | 21.72 | 0.0138 |
| `flagship_ref_v2_k4` | 20k | 6.38 | 0.58 | 6.00 | 104.4 | 22.42 | 0.0128 |

**结论**：v2 两档都远差于 v1（0.70 vs 3.04；0.58 vs 2.76），命中 P29 预注册的证伪线 →
**`as_modulation` 这一改写被证伪，旗舰线上 v2 出池**。
**两条必须随行注明的限定**：
① **预算不等（20k vs 60k）**，严格说只能作「不值得继续投入」的判据，不能作「机制有害」的定论；
而同 step dev/mse 反而是 v2 略优（Δ −0.0003，噪声内）→ **dev/mse 与回译指标再次分叉**。
② v1 自己的 k=4（2.76）**低于** k=1（3.04）→ 在冷启动旗舰臂上 gate k=4 **不是免费的**，
与对齐臂上的 ΔB4 +0.99 **方向相反** → 免费杠杆的增益依赖"臂已经学到的东西"，不能跨臂外推。
**EMA 缺档的工程原因**：v1/v2 的文本塔 `trainable=true`，而 EMA 包里不含文本权重，
`TextToLatentPipeline.from_checkpoint` 直接抛
`fine-tuned text runs require consistent raw weights; EMA bundle unavailable`
→ D46「全臂 EMA 默认」的限定③对旗舰系不适用，两档 `_ema` 作业已取消（占位空目录已删除）。

**统一口径提醒**：`_ema_k4` 的 5.65 是 **EMA**；`flagship_*` 与全部历史 test-642 数字是 **raw**。两张表不可混比。

## 2026-10-21 FVD 口径修复：旧 22–26 不是 FVD，官方 I3D 口径下基线为 132–185

**结论**：导师指出旧 FVD（22–26）异常，核查成立。官方 FVD（arXiv:1812.01717、
`google-research/frechet_video_distance`）= Kinetics-400 预训练 **I3D（Inception-v1）的
400 维 logits** 特征分布的 Fréchet 距离；16 帧、短边 resize 224 + center crop 224、
像素 [-1,1]、float64 精确 sqrtm。旧实现用 torchvision **R(2+1)D-18 的 512 维
average-pool** + ImageNet 归一化（有效 112²），特征空间完全不同，数字无换算关系，
整体作废。修复后自检 133.91（官方 0-vs-255 参考约 131），四条基线在 test 642 单侧为
`prefix60k` **178.48**、`route_t_align_20k` **184.67**、`_ema` **172.43**、
`_ema_k4` **132.30**；真值有限样本本底 `real_split_half_fvd = 74.17`。

**证据/落地**：`tools/eval_fvd.py`（重写，报告含 `protocol` 与 `real_split_half_fvd`）、
`tools/fetch_i3d_weights.py`（sha256 `bec6519f…a5e2`）、`tools/redecode_fvd_frames.py`；规范 `.trae/documents/FVD_SPEC_20261021.md`；
`EXPERIMENT.md` 4.0 基线区 + 第 5 节自动区（`FVD↓(I3D)`，旧值标 `⚠️R(2+1)D`）；
`RULES.md` 与 `AGENTS.md` 已改为"唯一主对比指标 = 官方 FVD、回译不再要求"。

**口径提醒**：ΔFVD < 50 在论文经验里不足以断言感知差异；本表 132–185 只在本协议内可比，
不要与 SignViP 的 1025（协议未写全）直接相减。

## 2026-10-21 prefix60k 的 SignViP 对齐口径：24 帧 159.38、32 帧 239.90、全帧 221.82

**做法**：按 SignViP 官方代码（`reports/eval/signvip_ref/`）对齐——I3D-K400 400-d logits、
短边 224 + center crop、`(x−0.5)×2`；帧取法改成"从第 1 帧起连续 N 帧"或"全部帧"，
分辨率先把帧 resize 到 SignViP 的 272×224（`--resize-hw 272 224`）。真值侧同样跳过条件帧 0。

**结果**（test 642 单侧，prefix60k）：24 帧 **159.38**（本底 68.56）、32 帧 **239.90**
（本底 84.20）、全帧 **221.82**（本底 60.34）；原 16 帧均匀口径为 178.48（本底 74.17）。

**结论**：① 24/32 帧只看视频开头，数值低于全序列 16 帧，两套口径不可混表；② 32 帧 > 24 帧，
后段漂移更严重；③ 全帧把整段做 I3D 时间平均，与固定窗口不是同一统计量，只能同口径比较；
④ SignViP 的 1025.45 仍不可与本表直接相减（片段是否含条件帧、训练/推理协议不同）。

**工程**：修掉 TorchScript profiling executor 在共享 GPU 上第二次 forward 报
`CUDA error: out of memory` 的问题（`torch._C._jit_set_profiling_executor(False)` /
`_jit_set_profiling_mode(False)`，自检与多次 forward 均正常）。

## 2026-10-21 我们与 SignViP 的 FVD 实现逐位一致（同输入相对差 5×10⁻⁹）

**做法**：从 SignViP 仓库取官方 `metrics/fvd/styleganv/fvd.py` 与 `calculate_fvd.py`，
与本仓库 `tools/eval_fvd.py` 在**同一批 20 条 prefix60k 视频**上对算：32 帧、272×224、
I3D-K400 400-d logits、`only_final=True`、同一份 `i3d_torchscript.pt` 权重
（sha256 `bec6519f…a5e2`），预处理同为短边 224 + center crop + `(x−0.5)×2`。

**结果**：SignViP 官方代码 **715.2227465**，我们的工具 **715.2227501**，
绝对差 3.6×10⁻⁶、相对差 **5.1×10⁻⁹** —— 仅浮点求和顺序差异，实现等价。

**含义**：我们的 FVD 与 SignViP 论文表格是同一度量，可以直接比较；残余差异只能来自
模型/条件/帧数协议，不是指标实现。等价性脚本存 `reports/eval/signvip_ref/verify_equiv.py`。

**工程备注**：官方 `calculate_fvd.py` 的 `load_i3d_pretrained` 用 `DataParallel` 会强制
`cuda:0`；共享卡上还需先关闭 JIT profiling executor，否则第二次 forward 会报假 OOM。

**全量复核（2026-10-21 22:14）**：用 SignViP 官方代码在 test 642 条、32 帧、272×224 上完整跑一遍
（`reports/eval/signvip_ref/run_signvip_fvd_prefix60k.py`，chunk=16、bs=2），FVD = **239.9040**、
real split-half 本底 83.94；我们工具同协议为 **239.9002**，差 0.0038（相对 1.6×10⁻⁵），
差异只来自分批与浮点累加顺序。结论：prefix60k 的官方口径 FVD 就是 **≈239.9**。

**SignViP 端到端复现（2026-10-22 00:42）**：用其官方发布权重在本机搭出完整 text→pose→video 链路
（`tools/signvip/gen_rwth.py`：t2gpt 自回归生成 12 token/帧姿态码 → `VQConditionEncoder.preprocess` →
`SignViPPipeline`，20 步、CFG 3.5、context 24/overlap 2、seed 42、参考帧 = 真值 frame 0、最多 32 帧），
test 642 条全部生成（`reports/eval/signvip_gen_642/`：601 条 32 帧、中位数 32、7 条 < 10 帧）。
用我们的官方口径工具评测（`--frame-mode contiguous --num-frames 32 --resize-hw 272 224`，即 SignViP
`metrics/fvd/styleganv` 协议）：**FVD = 259.00**（num_videos 634，7 条 <10 帧被跳过），
real split-half 本底 **86.31**，mean_feature_l2 35.72。
对照论文（arXiv:2506.15980 Table 3，RWTH-2014T）：SignViP **1025.45**、SignGAN 1431.38、SignGen 1715.32；
Table 4 扩散类 275.22–556.63。判读：**实现侧已证明一致**（同输入下官方代码 239.9040 vs 我们 239.9002，
相对差 1.6×10⁻⁵），故 259 vs 1025 的差距不来自 FVD 代码；候选来源 = 论文评测子集/样本量
（小 N 本底可达数百上千：我们 N=59 时本底 553）、条件来源（本机只能用文本→姿态，官方训练期验证用 GT 姿态）、
或论文数字取自训练期小批量验证。定性检查：生成帧与真值像素差 43–48（非拷贝）、同文本同种子逐位一致、
换文本平均差 0.17（文本条件有效）。
**结论（更正为可证明的版本）：在 272×224、同一帧清单、同一预处理下，我们的 FVD 工具与 SignViP 官方
`fvd.py` **逐元素一致**（`align_audit.py`：输入张量 max|Δ|=0、400-d 特征 max|Δ|=0、FVD 相同）；
在 256×256 等其他几何下两者有可测但很小的差（输入 max|Δ|=0.0376、特征 cos=0.9998、FVD 差 0.53%）。
论文 1025.45 在本机协议下复现不出（我们更低），差距主要来自样本量与协议，不是代码实现。**

**SignViP 复现值与论文差 4 倍的原因：样本量偏差，不是实现/模型问题（2026-10-22 诊断）**：
同一批 642 条生成帧、同一份真值特征，只变抽样子集大小：N=20 → **907.1**（本底 1488.1）、
N=40 → 660.8（823.7）、N=80 → 457.0（469.1）、N=160 → 344.6（332.8）、N=320 → 284.6（163.9）、
N=634 → **259.0**（86.3）；更旧的 prefix60k 同曲线（N=20 → 727.1、N=634 → 239.9）。
一阶偏差模型 `F(N) ≈ 233 + 1.65×10⁴/N`。N=150 退化基线：生成 343.0、时间乱序 901.1、
冻结首帧 1156.4、真值 test vs train 229.0 —— 论文 1025 在 N≈150 口径下相当于"画面冻结"，
不像 SOTA 模型成绩。官方仓库唯一的 FVD 调用点 `train_stage2_multi_cond.py::log_valid`
用 `batch_size=1`、`max_eval_batches=10`、GT pose、dev 集，即**≤10 条**；论文正文未写 FVD
的 N/帧数/分辨率。结论：259 vs 1025 **口径不同，不可直接相减**；FVD 实现等价已另行证明
（同输入 239.9040 vs 239.9002，相对差 1.6×10⁻⁵）。规则已写入 `FVD_SPEC_20261021.md` §3.2/§5.7。

**样本量敏感性（2026-10-22；更正：不能据此断言论文 N=10）**：同一特征池把 N 扫到 10，得
N=10 → **1005.16**、15 → 1005.00、20 → 969.00、25 → 867.22、30 → 790.15、40 → 656.46、
60 → 531.32。论文 1025.45 与 N≈10 的量级相容（差约 2%），但论文正文未公开 FVD 的 N/帧数/分辨率；
仓库唯一的 FVD 调用点 `train_stage2_multi_cond.py::log_valid`（`batch_size=1`、`max_eval_batches=10`、
GT pose、dev 集）只能证明**训练期验证至多 10 条**，不等于论文表格的评测协议。因此正确表述是：
小 N 会把 FVD 抬到 ~1000 量级，259（N=634）与 1025.45 **口径不同、不可直接相减**；
"论文数字来自 N=10" 只是与证据相容的假设，不是已证明的事实。

**SignViP 复现集 24 帧 / 256×256 官方口径（2026-10-22）**：用 SignViP 官方 `fvd.py` 代码
（`get_fvd_feats` + `frechet_distance(feats_fake, feats_real)`）对 `reports/eval/signvip_gen_642`
全测试集评测：**FVD = 196.76**（N=613，24 帧连续、先 resize 到 256×256、再短边 224+crop、400-d logits），
真值 split-half 本底 88.31；同协议本仓库工具 194.19（N=635，含 16–23 帧短片），差 1.3%。
对照：32f/272×224 本仓库工具为 259.00（N=634）。脚本 `reports/eval/signvip_ref/run_signvip_fvd_generic.py`。

**GT-pose 条件近似复现（2026-10-22）**：论文 Table 4 的 GT-pose 设定（Sign Video Diffusion
Model 275.22）在本机做近似复现：条件 = RefSign DWPose 134 关键点经 SignViP `draw_pose`
渲染的 272×224 骨架 + 2D 手部关键点（**无 HAMER 3D 手部 mesh**，这是近似点）；dense
`sk_images`/`hamer_images` 直接进 `SignViPPipeline`（与官方训练验证同一通路）。642 条
生成视频，contiguous 32 帧、272×224、I3D-400 logits → **FVD = 158.86**（N=642，本底 84.20，
mean_l2 25.33；按官方真值窗口帧 0..T-1 重算为 **156.27**，N=642）。对照：文本→姿态→视频
259.00（N=634）；GT 条件使 FVD 降约 100，方向正确。**不可与论文 275.22 直接比较**：手部条件是
2D 关键点而非 HAMER 3D mesh、生成最多 32 帧（官方验证 32 帧窗口）、N 与协议未公开。产物
`reports/eval/matrix/signvip_gtpose/test_n642/fvd_i3d400.json`，脚本
`tools/signvip/gen_rwth_gtpose.py`，规范 `FVD_SPEC_20261021.md` §3.3。

**审计回合：FVD 协议对齐实验（2026-10-22）**：按审计意见逐项做了可证伪实验。
① **预处理实现**：`tools/eval_fvd.py` 新增 `--preprocess {pil_aa,signvip_tensor}`；`signvip_tensor` 复刻
SignViP `preprocess_single`（张量双线性、无抗锯齿）。`align_audit.py`（20 条 × 32 帧）显示
272×224 下两条路径的输入张量、400-d 特征、FVD 完全相同（max|Δ|=0）；256×256 下输入 max|Δ|=0.0376、
特征 cos=0.9998、FVD 差 0.53%（1072.23 vs 1077.86）。
② **帧 0 对齐**：从官方数据集代码取到直接证据——`signdatasets/sign_lang_video.py:255-260` 验证集
`random_sample=False, start_idx=0`，`batch_ids = linspace(0, clip_len-1, num_frames)`，即真值窗口是
**帧 0..T-1**（不是 1..T）；`ref_frame` 是同一视频随机帧（`:352`）。工具新增 `--include-frame0`。
重算：文本→姿态 32f/272×224 **259.92**（N=635，旧跳过帧 0 为 259.00，+0.35%）；GT-pose 近似
32f/272×224 **156.27**（N=642，旧 158.86，−1.63%）。帧 0 偏移是 ~1% 量级，不是 4 倍差距来源。
③ **32 帧截断**：同一 20 条文本→姿态样本，max_frames=32 vs 48 生成后都取前 32 帧评测：
844.50 vs 857.09（+1.49%，N=20，含采样 RNG 漂移，属上界）——截断效应 ~1.5%，不是 4 倍来源。
④ 因此 259 vs 1025.45 的差距**不能**归因于预处理、帧 0 偏移或 32 帧截断（三项合计 <3%）；
主因仍是样本量/协议未公开（见上条样本量敏感性）。**更正此前"在 N=10 上两者一致"的说法**：
那只是同量级相容，不是同协议等价。

**官方代码含帧 0 复核（2026-10-22）**：`reports/eval/signvip_ref/run_signvip_fvd_generic.py` 加
`INCLUDE_FRAME0=1`（真值取 `images0001..images00NN`，即源帧 0..T-1），24 帧 / 256×256 / 官方 fvd.py：
**FVD = 197.55**（N=613，本底 88.20）；帧 1 口径为 196.76（本底 88.31）。差 +0.4%，再次说明帧 0
对齐只影响 ~1% 量级。日志 `reports/eval/logs/signvip_official_gen642_24f_256_f0.json`。

## 2026-10-23 D57 · SignViP 50步完整生成重跑已完成

2026-10-22 22:37:36（Asia/Shanghai）生成642/642及评测完成，所有进程退出0；代码2319729，产物 `reports/eval/signvip_paper_repro_20261022_v2`。原始210×260官方test与字幕、DDIM50/CFG3.5、完整EOS长度、context24/overlap2、fp16、参考帧0、官方视频编码及styleganv FVD函数。

预声明四协议：前32帧短片末帧补齐 N642 **290.291187**；严格足32帧子集 N597 **308.710916**；全视频均匀16帧 N642 **73.669359**；GT起点后移1帧 N642 **289.375657**。生成短片35、真实短片17、合并45；严格子集差异包含换样本效应，不能全归因补帧。台账新增exp-100至103。

10-23只读核验：五组特征均[642,400]、float64、有限；ID唯一且顺序匹配manifest；清单和代码SHA吻合。基于既有特征用独立对称PSD特征值公式重算四项，与JSON误差小于1e-8。论文Table3 **1025.45未复现**，不证明本机模型更强。论文完整FVD片段/样本/短片策略和最终发布权重版本对应关系尚未确认；不得推断论文N=10，也不得将小样本碰巧接近当成复现成功。

## 2026-10-22 · 目标机 conda 环境验证

已配置独立 `text2video` 环境。依赖一致性、四卡CUDA/bf16、真实缓存→本地mT5→DiT前向、冻结VAE校验及单帧解码通过；128 passed / 9 skipped（Wan权重未打包）。I3D常量输入自检133.908176。迁移949332文件仅.git/index摘要不同，其余匹配。未启动训练、未实测batch16反向峰值显存。证据和启动命令：`ENVIRONMENT_SETUP_20261022.md`；决策D63。

## 2026-10-24 · 里程碑1 E1 base 训练完成：FVD 126.93（first24/N=640，本底 84.30，两次逐位一致）

**结论**：里程碑1 的 base 臂 E1（PixArt-α 骨架 28×1152/16、1a cross-attn 分层 `[T,T,T,R]×7`、
4c mT5 前3层+embedding 可训、2a、3b 无 motion、5b from scratch）训练 30k 步后，按
first24 协议（contiguous 24 帧含 frame0、官方 I3D-K400 400d logits、224²、DDIM50/CFG2、
GT 长度、seed42、raw 权重、test 单侧）测出 **FVD = 126.92693857176755**（N=640/642；
真值半分本底 `real_split_half_fvd` = 84.29751411436246；mean_feature_l2 32.32±6.53）。
同协议第二次评测**逐位相同**（FVD、ids SHA256 均一致），说明整条生成/计分链路确定可复现，
M1.5 判据（非 NaN、不低于真值本底、可复现两次）通过。

**读法**：这是 from-scratch、30k 步、有效 batch12、无预训练先验的基准数，约为真值本底的
**1.51×**——有信号但离真值分布仍有明显差距，不能当作方法有效性证据；也不能与旧 16 帧
R(2+1)D 口径（22–26）或 SignViP 32 帧/272×224 口径（156–260）数字混表。它是 M1 十臂矩阵
**唯一分母**（E2–E10 均相对它单因子改动）。

**产物/证据**：`/data1/huyang/text2video/runs/m1_E1_base/`（final.pt/last.pt 13.68 GiB，
30 组视觉回调，metrics）；`reports/eval/m1_E1_base/{fvd,protocol}.json` 与
`reports/eval/m1_E1_base_repeat/`；执行记录 `MILESTONE1_EXECUTION_20261022.md`；
台账 `EXPERIMENT.md` exp-099。

**限制**：只有 30k 终点一个评估点（无中间 ckpt 的 FVD 曲线，P3 清理后可低成本补）；
只测了 raw 权重，EMA 口径未测（用户待裁决）；batch 因用户 GPU≤3 约束由 16 改为 12（D65），
与十臂后续必须保持同一 batch。

## 2026-10-24 D66 · 里程碑1 E7（1b 前缀注入）30k 完成：first24 官方 I3D-FVD = 124.0733

训练：`configs/milestone1/E7.yaml`（base + factor1=1b，其余因子 4c+2a+3b+5b）从 0 训 30k，三卡 0/4/5（1×RTX4090 + 2×RTX3090），micro1×accum4×world3 = effective batch12，seed42，bf16，EMA0.9999；10-23 14:11 → 10-24 13:00，82157 s（22.8 h），末段 2.81 s/步，峰值 13.58 GiB；loss 5k 0.0218 / 10k 0.0203 / 15k 0.0191 / 20k 0.0188 / 25k 0.0139 / 30k 0.0190，末 200 步均值 0.0166。

评测：`tools/eval_milestone.py --config runs/m1_E7/config.yaml --checkpoint runs/m1_E7/final.pt --out reports/eval/m1_E7_first24_20261024 --gpu cuda:0`，13:01 开始、约 35 分钟完成 640/640 生成与 FVD。生成用 raw 权重（`ema_used=false`）、DDIM50、CFG2.0、GT 长度、seed42；FVD 用官方 I3D-K400 400d logits、24 帧连续起始、224²、float64 Fréchet，权重 sha256 `bec6519f…a5e2`。

结果：**FVD 124.0732936749423**（N=640，`skipped_short: []`）；同批真值半分本底 `real_split_half_fvd` = **84.29751447172157**（与本机 `m1_first24_prepare_20261022/real_floor.json` 一致）；生成特征 L2 32.114±6.701。

口径注意：① E1 base 的 30k 在 `/home/wxy` 主机上跑（`MILESTONE1_E1_LAUNCH*.json` 记录），本机无同口径 FVD，**不得**把 E7 与本机旧口径数字（R(2+1)D-18 / 240 prompts 等）横比；② 本 run 为 4090+2×3090 混卡，E1 为 3×3090，对照表须标卡型差异；③ 评测期间 `/data` 只剩 5.9 GiB，已删除与 `final.pt` 同为 step30000 的 `runs/m1_E7/last.pt`（12.3 GB）腾空间，`final.pt` 保留可复现（如需 resume 点，可由 `final.pt` 复制恢复）。

## 2026-10-24 D67 · 强制SignViP原样FVD后E7重测与匹配对照

用户要求正式FVD直接用SignViP源码。本轮把原始`metrics/fvd/styleganv/fvd.py`逐字保存到`third_party/signvip/fvd.py`（SHAc9a1acbf...fda37），入口`tools/eval_signvip_fvd.py`只负责IO/first24人口/统一源几何，预处理/特征提取/距离均调用原函数；权重SHA bec6519f...a5e2。`eval_milestone.py`和旧`eval_fvd.py`CLI统一走新入口。代码a7756ed，分支codex/signvip-first24-fvd-20261024。95项目测试通过，19项相关测试通过；官方常量视频自检133.6239。第三方源自身含一行尾空格，为保持逐字一致未改。

既有E7 raw PNG首24、GT首24均256方形；新评测无抗锯齿（官方函数）、224中心裁剪。E7全640 **126.84746316**，真值半分84.88998481；旧本地实现124.07329367保留为历史值，当前正式SignViP实现以126.84746316为准。

完整SignViP发布权重生成中有23条在E7清单内但不足24帧，不补帧，匹配交集617。两者用同GT、相同617ID、先256方形（SignViP输出由272×224转为256方形；E7/GT原本256方形不改变）、同官方预处理、同I3D和统计：**E7 128.47619769 vs SignViP 218.50745041**，共同真值半分89.08661494。之前无这一整套完全匹配的结果；旧613条24帧结果不是本轮对照。

评测条件统一，不代表生成配方统一：E7 DDIM50/CFG2.0/GT长度/raw/PNG；SignViP DDIM50/CFG3.5/文本预测长度/官方MP4，参考首帧均有。故只报告此共享FVD协议下的实测，不能宣称论文1025.45已复现或所有生成变量完全受控。

原始结果`reports/eval/m1_E7_signvip_first24_20261024/fvd.json`，全部特征/ID保存在同名npz，command/comparison_setup/code_version记录可追溯。已复读这些特征重新调用官方距离函数、核对640及617唯一ID与mask顺序、源码及wrapper SHA，一致。未训练、未重新生成、未删除旧产物。

## 2026-10-24 · E1 按 SignViP 原样入口重测：FVD 129.7222（canonical）

**背景**：用户 2026-10-24 指令要求所有 FVD 测试改用 SignViP 官方脚本。正式入口固定为
`tools/eval_signvip_fvd.py` → `third_party/signvip/fvd.py`（SHA256 `c9a1ac…da37`，I3D 权重
`bec6519f…0a5e2`），`tools/eval_milestone.py` 固定调用该入口；旧 `tools/eval_fvd.py` 只作历史/诊断。

**做法**：用同一批已生成的 E1 首 24 帧（frame0 + 23 预测帧）与 GT 首 24 帧（源帧 0..23），
统一 256×256 源几何后交给 SignViP 原函数 `preprocess_single`（张量双线性、**无抗锯齿**、
短边 224 + center crop、[-1,1]）与 `get_fvd_feats`，再用原 `frechet_distance` 计算；
N=640，raw 权重，DDIM50/CFG2/GT 长度/seed42 均不变。

**结果**：**FVD = 129.7222078577488**（N=640），真值半分本底 **84.87336765689959**，
生成特征 L2 32.441±6.552。对照：旧本地实现（PIL 抗锯齿路径）同帧同协议为 **126.92693857176755**
（两次逐位一致）——差异 +2.795（约 +2.2%）来自官方 `preprocess_single` 的无抗锯齿预处理，
属于脚本口径差异，不是模型或数据差异。两套独立封装（本地草稿入口与 canonical 入口）在同一
输入上给出**完全相同**的 FVD（129.7222078577488），可作为入口正确性的交叉校验。

**判读**：M1.6 十臂对比的分母从本日起使用 **129.7222（canonical SignViP 入口，floor 84.8734）**；
126.9269 仅作为历史口径保留、不得与 canonical 数字混表。注意 E1 训练于 3×RTX3090、E7 于
RTX4090+2×3090 混卡，跨臂比较仍须标注硬件差异（D66/D67）。

**产物**：`reports/eval/m1_E1_base_signvip/{fvd.json,fvd.npz}`；`reports/eval/m1_E1_base{,_repeat}/`
旧报告保留为历史；`MILESTONE1_EXECUTION_20261022.md`。

## 2026-10-24 D72 · E1 远程正式 SignViP FVD 重测完成

E1 base 的新口径结果为 **FVD 129.7222078577488**，test有效N=640，真值半分本底 **84.87336765689959**。使用远程 `tools/eval_signvip_fvd.py`，上游 `third_party/signvip/fvd.py` SHA256 c9a1acbf811c3050d92b5289f4a4789ba6209a71a5bafbeba10df5576ffbda37；wrapper SHA256 8e4311937dc31b7380dc0da4fdb736bdd7551e51176b64acd4cc1231f08413f5，主仓提交3f0afd4。

复用E1已有640条首24帧PNG；全部相关配置与远程E1一致（模型、条件、数据、VAE、扩散、采样、评测），固定清单SHA256 6f96ad9dda2721b877dea1f08dab7bbf17bb4c6bcb9d90becb665824a69110f6。原生成raw/DDIM50/CFG2/GT长度/seed42；训练文本前3层+embedding可训、text lr1e-5。源统一256方形，经官方no-AA预处理到224中心裁剪，I3D-K400 400d logits，首24帧含frame0、不补帧。所有GPU均被占用，本次CPU计分；没有重训或重新生成。

原任务PID3719637已经在运行，发现后终止本轮重复PID3720666，继续跟踪并独立核验原任务产物。真值和生成特征均为640×400且有限，ID与清单完全一致，源码/权重SHA锁定。保存特征重新调用官方距离函数得到129.72220785883525（与报告差约1.1e-9）；本底复算84.87336669963007（差约9.6e-7，数值库计算误差）。旧126.9269留作历史口径，不能将口径变化解释为模型质量变化；与其他设备上的分数比较时须注明CPU/GPU差异。

正式结果和特征：`reports/eval/m1_E1_base_signvip/fvd.json`、`fvd.npz`；日志 `/tmp/m1_sv_canonical_e1.log`；本轮配置快照、命令与核验证据 `reports/eval/m1_E1_remote_signvip_20261024/`。台账exp-109。

## 2026-10-25 · 里程碑1 E2（2b 全层 [ref;text]）完成：FVD 142.32，劣于 E1 的 129.75

**做法**：E2 = E1 唯一改动因子 2 `2a`→`2b`（28 层 cross-attn 全部注入 `[ref;text]`），其余配方
完全同 E1（1a、4c、3b、5b、effective batch12、lr1e-4/text lr1e-5、seed42、30k、DDIM50/CFG2、raw）。
训练 2026-10-24 11:54 → 10-25 23:14，共 **127,248.8 s = 35.35 h**（2×RTX3090，world2×accum6=batch12，
末 3k 步 loss 0.0171，峰值显存 16.64 GiB，30 组视觉回调）。

**评测**（canonical SignViP 原样入口，first24/N=640、GT 长度、frame0、raw、256²→官方 224²、
batch8/GPU0）：**FVD = 142.32016432887428**（N=640，`skipped_short=[]`，真值半分 **84.89697227815674**）。
E1 同协议重算（同一入口、batch8/GPU）为 **129.74927595301466**（floor 84.89697227815674，与 E2
逐位一致），因此 **Δ(E2−E1) = +12.5709（相对 +9.69%）→ E2 更差**。

**读法**：在相同预算/帧协议/真值/入口下，把 ref+text 拼成一份 K/V 喂给所有层（2b）比 E1 的
"text 层只看 text、ref 层只看 ref"（2a，`[T,T,T,R]×7`）明显更差。可能的机制：全局拼接使 ref 与
text 在每一层互相竞争、削弱分层专业化；但这只是一个 seed、且 E1 为 3 卡/E2 为 2 卡（同 effective
batch12、同为 3090，rank 划分/RNG 不同），不能当作强因果结论，需 M1.6 的条件臂一起读。

**参照（同 canonical 入口）**：E7（1b 前缀注入，4090+2×3090 混卡训练）126.8475 < E1 129.7493 <
E2 142.3202；E1/E2 与 E7 的硬件不同，跨臂表须标注卡型。E2 的旧本地实现数字不单独保留
（该 run 只做过 canonical 评测）。

**产物**：`/data1/huyang/text2video/runs/m1_E2/`（final.pt 13.68 GiB、30 组回调）；
`reports/eval/m1_E2/{fvd.json,fvd.npz,protocol.json}`；自动评测链日志 `/tmp/m1_E2_chain.log`。

## 2026-10-27 · 里程碑1 E4（3a motion loss）完成：FVD 130.85，与 E1 129.75 基本持平

**做法**：E4 = E1 唯一改动因子 3 `3b`→`3a`：`MultiScaleDeltaLoss`（scales 1/2/4、weight 0.1、
warmup 2000）作用在 `pred_x0` 与 GT latent 的逐视频多尺度时间差分 MSE 上；其余配方完全同 E1
（1a、4c、2a、5b、effective batch12、lr1e-4/text lr1e-5、seed42、30k、DDIM50/CFG2、raw）。
训练 2026-10-25 23:53 → 10-27 05:29，共 **106,535.8 s = 29.59 h**（3×RTX3090，world3×accum4=batch12，
末段 mse 0.0201 + 0.1×motion 0.0195 = loss 0.0220，grad 0.056，峰值 16.59 GiB，30 组回调）。

**评测**（canonical SignViP 原样入口、first24/N=640、GT 长度、frame0、raw、256²→官方 224²、
batch8/GPU0，05:29→06:00）：**FVD = 130.84863100287862**（N=640，`skipped_short=[]`，
真值半分 **84.89697227815674**，与 E1/E2 逐位一致 → 协议完全相同）。

**对照**（同入口/同协议）：E1（2a，无 motion）**129.7493**；E4（2a + motion）**130.8486** →
**Δ +1.0994（+0.85%）**；E2（2b）142.3202；E7（1b，混卡训练）126.8475。
即 **motion loss 在当前配方下没有带来可测的 FVD 改善**：E4 与 E1 基本持平（略差 0.85%），
该差值在单 seed 的训练/数值噪声量级内，不能据此断言 motion loss 有害。E9（1c+3a）是交互探针，
不能替代 E4 对"1a 下 motion 是否有效"的回答；若要做功效更强的判断，需要多 seed 或先看
E4 与 E1 在其它诊断量（如运动幅度/帧间差分）上的差异。

**产物**：`/data1/huyang/text2video/runs/m1_E4/`（final.pt 13.68 GiB、30 组回调）；
`reports/eval/m1_E4/{fvd.json,fvd.npz,protocol.json}`；链日志 `/tmp/m1_E4_chain.log`。


## 2026-10-27 · D75 案例4-2教学靶场复现完成（非模型实验）

用户授权在本服务器复现所附DVWA反射型/存储型XSS案例并交付报告。采用官方tag2.3（commit 34a10d4166dcdc31f9641cf36ba5bd0fc1d4c59c），191个上游原文件与源码包逐字一致；只新增本地配置。独立MySQL5.7端口13342、PHP7.4端口18042均只监听127.0.0.1，不操作现有数据库、不占GPU、不改模型代码。

最终一轮`reference/dvwa-case42-20261027/results/20261027_193731/summary.json`：反射型16+存储型18，共34/34符合预期；提交、GET刷新、新会话的弹框内容经独立检查；2/2 Cookie补充核验通过，无外部请求。impossible使用有效令牌验证HTML编码，错误/缺失令牌另做对照；实际源码公共输出函数会编码旧留言，已据实纠正初始假设。前两轮调试记录保留，最终结果仅取本轮。

产物：`documents/CASE42_复现报告_20261027/`，含20页PDF、DOCX、独立HTML、Markdown与完整ZIP（352个条目，15,127,861字节，逐项SHA256校验通过）。ZIP含脚本、锁定依赖、官方源码包、小型PHP扩展包、原始证据、19个独立实验库逻辑备份及七张原始教材图。启动/停止/重启后22个本轮及调试练习库数量一致；保留本机主服务，18043/18044临时服务已停止。主项目Git状态与任务前一致（原有未跟踪tools/make_milestone_compare_video.py未动）。

本条为独立教学任务，不参与手语模型晋级或FVD比较；台账exp-112另列。


## 2026-10-27 · FVD 实现切换为 video_metric 原样计算（用户指令）

**用户指令**：以后所有 FVD 测试不再使用原先的任何脚本，改用
`/home/huyang/data/trae_projects/video_metric` 中的 FVD 原样计算方式；把以前的 FVD 计算脚本
与计算约束都删除，并把两个 `.git`（主仓库 + 知识库）同步到 GitHub。

**结论**：

1. **实现**：正式 FVD 只调用 `video_metric` 的
   `calculate_fvd(videos1, videos2, device, method='styleganv', only_final=True)`。
   逐字节副本落在 `third_party/video_metric/`（上游 `JunyaoHu/common_metrics_on_video_quality`）；
   IO 壳 `tools/eval_video_metric_fvd.py` 只负责 test 固定清单、first24 连续帧、
   256² 源几何、`[B,T,C,H,W]` 像素 `[0,1]` 张量与报告，不含任何指标数学；
   `tools/eval_milestone.py` 生成后自动调用它。
2. **数值不变（已核验）**：旧 `third_party/signvip/fvd.py` 与
   `video_metric/fvd/styleganv/fvd.py` **SHA256 完全相同**
   （`c9a1acbf…da37`），I3D 权重相同（`bec6519f…0a5e2`），数据层协议相同
   （test 640 / first24 / 不补帧 / 256² → 上游 224²）。所以 E1 129.7493、E2 142.3202、
   E4 130.8486 等历史 FVD 的**数值**在相同输入下不会被这次切换改变；本轮是
   **实现治理与脚本清理**，不是口径变更（口径仍是官方 I3D-K400 400-d logits）。
3. **删除（用户明确授权）**：`tools/eval_fvd.py`、`tools/eval_signvip_fvd.py`、
   `third_party/signvip/`、`tools/redecode_fvd_frames.py`、`tools/fetch_i3d_weights.py`、
   `tests/test_eval_fvd.py`、`tests/test_signvip_fvd.py`，以及 `tests/test_milestone_eval.py`
   中两个旧 FVD 测试；旧 `FVD_SPEC_20261021.md` 及其中 SignViP 强制入口/SHA 锁定/
   `--self-check≈131`/617-640 双 N 等约束全部作废；新唯一规范 = `.trae/documents/FVD_SPEC.md`。
   历史 FVD 报告产物（`reports/eval/**/fvd*.json/npz/lock`、`fvd_command.json`、
   `real_floor.json`、`reports/eval/fvd_ref/`）也按用户选择删除；生成帧/`ids.txt`/`protocol.json` 保留。
4. **来源说明**：用户给的 `/home/huyang/data/trae_projects/video_metric` 在评测机上不存在
   （`/home/huyang` 整个目录没有）；本轮从本机同源副本
   `/home/hulala/gsw/SLP2025/VideoVAET/video_metric`（与 `VideoVQVAE` 下副本逐字节相同）
   拷贝，全部 SHA256 记在 `FVD_SPEC.md` §3 与 `third_party/video_metric/PROVENANCE.md`。
   若之后拿到用户路径下的副本，先按同一张表复核等价再替换。

**后续影响**：E6 训完后的自动评测链会直接走新入口；新结果写入新的 `fvd.json`（含
`protocol.upstream_sha256`）。旧台账数字保留作历史引用，不再当可复算证据。
决策登记 TODO **D76**。

### 新入口对已跑完臂的复算验证（2026-10-27）

用新入口 `tools/eval_video_metric_fvd.py` 对三个已跑完臂的**既有生成帧**重算（不重新生成、
不改任何帧），直接检验“切换不影响数值”：

| 臂 | 旧入口（SignViP 壳，batch8/GPU） | 新入口（video_metric 原样，上游 bs=10） | Δ |
|---|---:|---:|---:|
| E1 | 129.7493 | **129.7520** | +0.0027（+0.002%） |
| E2 | 142.3202 | **142.3279** | +0.0077（+0.005%） |
| E4 | 130.8486 | **130.8517** | +0.0031（+0.002%） |
| 真值半分 floor | 84.8970 | **84.8977** | +0.0007 |

**结论**：差异 ≤0.008 FVD（≤0.006%），落在 batch 大小（旧入口 8 / 新入口走上游硬编码 10）
与 GPU 浮点归约噪声的量级内；同一批帧、同一 400-d 特征空间、同一协议，三臂的相对排序与
结论（E1 最好、E2 差约 9.7%、E4 与 E1 基本持平）全部不变。产物
`reports/eval/m1_E{1,2,4}_video_metric_20261027/fvd.json`；台账 exp-113/114/115。

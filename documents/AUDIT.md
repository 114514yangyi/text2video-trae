# AUDIT — text2video 全链路审计与改进方案

| 项 | 内容 |
| --- | --- |
| 审计日期 | 2026-10-13 |
| 范围 | text→latent DiT 训练/生成/可视化链路；评测链路（SignMbart 回译、FVD、文本敏感度、天花板/通道探针）；冻结契约适配；文档与代码声明一致性 |
| 方法 | `research-codebase-audit` 技能：七轴（正确性/数据泄漏/模型结构/数据管线/代码与声明/占位与伪造/严谨性）+ 声明核验 + 严重度分级 |
| 只读声明 | 本次审计未修改任何训练/评测代码、配置、缓存或 checkpoint。唯一新增：本文件、`NOTES.md` 记录、`STRUCTURE.md` 索引 |
| 证据 | 通读 `t2v/` 全部 18 个模块、`tools/` 21 个脚本（9 个精读）、`configs/` 6 个、`tests/` 4 个、`.trae/documents/*`、`../RefSign/UPSTREAM.md`；抽读 `runs/`、`reports/` 产物；实测命令见 §11 |
| 语言 | 中文（与仓库知识库一致） |

## 1. 审计结论

**总体判定：CONDITIONAL** —— 未发现伪造结果、硬编码指标或训练/测试串数据；主要风险集中在「用 test 选模型与选步数」「探针统计量定义错误且被文档引用」「评测帧目录复用可能污染」「冻结 VAE 契约默认不校验、且上游代码已漂移 5 个 commit」「SignMbart 口径切换后文档未同步、部署结论自相矛盾」。

发现数：**0 CRITICAL / 5 MAJOR / 9 MINOR**。

一句话危害排序：

1. M1：DiT 训练没有验证集，30k/60k/90k 的部署选择与「文本依赖显著」判定都落在 32 条 test 上；
2. M2：通道探针的 gate/contrib 数字不是模型的门控幅度，`NOTES.md:189,192` / `DESIGN.md:53` 引用的 0.13/0.40/0.63 与「5–10%」不可复现；
3. M3：`eval_t2v.py` 复用输出目录且 BT 标签按目录名派生，陈旧帧可以静默进入回译与 FVD；
4. M4：冻结 VAE 契约在工程上没有强制（checkpoint sha 默认不校验、repo 未锁 commit、missing keys 静默），实际代码已在契约 commit 之后前进 5 个 commit；
5. M5：换 SignMbart 口径后 README/DESIGN 未回改，部署建议与「当前结论」相互矛盾。

本次实测的**正向核验**（这些是「已确认正确」而非「未检查」）：

| 检查 | 结果 |
| --- | --- |
| `python tests/run_all.py`（4 个测试） | 全部通过（`test_conditioning/data/diffusion/dit`） |
| 冻结 VAE checkpoint sha256 | `82f37058…e7725`，与 manifest 声明一致（本次实算，2m13s） |
| 冻结 VAE config sha256 | `0678ee8f…f7865`，与 manifest 声明一致 |
| latent 统计口径 | `train_stats.json` 的 `manifest_sha256` = `sha256(train.jsonl)` = `c74d160b…`，一致（统计确实只拟合 train） |
| VAE checkpoint 在当前上游代码下的加载覆盖 | missing 0 / unexpected 0（225.5M 参数全部命中，无随机初始化权重混入） |
| 数据切分完整性 | train/eval/test = 7096/519/642，video_id 无重叠、无重复、无缺失文本、空文本 0，全部 `T == num_frames-1` |
| 256 帧过滤口径 | train 7021/7096、eval 518/519、test 642/642，与文档一致 |
| 归一化统计 | 1024 维 mean/std 全有限，无 NaN/Inf |

## 2. 覆盖范围

| 区域 | 文件 | 状态 |
| --- | --- | --- |
| 模型 | `t2v/models/{dit,conditioning,factory,length_model}.py` | 全部通读 |
| 扩散 | `t2v/diffusion/gaussian.py` | 通读（损失/掩码/采样/CFG 段精读） |
| 数据 | `t2v/data/{dataset,collate,normalizer}.py` | 通读 + 缓存实测校验 |
| 适配器 | `t2v/adapters/{vae,text}.py` | 通读 + 实算 sha/加载覆盖 |
| 管线与训练 | `t2v/{train,train_length,training_utils,monitoring}.py`、`t2v/pipelines/*` | 全部通读 |
| 配置 | `configs/*.yaml`、`configs/frozen/vae_h02.yaml` | 全部通读 |
| 评测工具 | `tools/{eval_t2v,eval_fvd,text_sensitivity,cfg_sweep,probe_*,bt_paired_stats}.py`、`tools/run_bt_eval_signmbart.sh` | 通读（脚本与 4 个探针精读） |
| SignMbart 链路 | `tools/{signmbart_bt_infer,validate,probe,trans_debug,gen_ablation}.py` | 采样通读（本文结论不依赖其内部数值，已由 `signmbart_validate.py` 全量 642 自检背书） |
| 调度脚本 | `tools/{launch,run_*,eval_*}.sh` | 采样（重点看输出目录与依赖路径） |
| 测试 | `tests/*` | 通读 + 实跑 |
| 产物 | `runs/`、`reports/eval/*` | 抽读（验证声明数字来源，不逐文件审计） |
| 上游契约 | `../RefSign/UPSTREAM.md`、`README.md`、`data/phoenix_latents/v1/manifest.json` | 通读 + 与本地实测对照 |

## 3. MAJOR 发现

### M1. 没有验证集：部署模型与步数是用 test 回译指标选出来的

- 轴：G（严谨性）/ B（选择泄漏）
- 位置：`t2v/train.py:303-391`；`t2v/training_utils.py:91-116`；`README.md:85-93`
- 证据：
  - 训练循环只有 `train/loss`、`train/mse`、`train/lr`（`t2v/train.py:348-361`），全仓没有 dev/val 评估；
    `rg -n "best_dev_loss"` 只命中 `training_utils.py:99,114-115` 的定义与写盘分支，`train.py` 从不传入；
    `load_split(config,"eval")` 只被 `t2v/train_length.py:68` 和 `tools/cfg_sweep.py:37` 使用。
  - `final.pt` 恒为最后一步（`t2v/train.py:382-384`），没有按指标选点。
  - 部署结论来自 32 条 test 的回译表：`README.md:85-93` 用 test 上的 ΔB1/ΔROUGE/WER 比较 30k/60k/90k，并据此写「部署」。
- 影响：`test` 既是选点集又是报告集，32 条 test 上的排序与显著性带乐观偏差；「prefix 显著增强文本条件」目前是 test 结论，不是留出集结论。这不是伪造，而是纪律问题——一旦写成论文数字，审稿人复算会把「模型选择」算作 test 调参。
- 状态：CONFIRMED
- 修复：在训练循环内加 dev（`eval` split）评估——至少每 `log_every`×k 记 `dev/mse`（同一扩散损失、`frame_mask` 口径），并在 dev 上跑小样本回译/FVD 作为选点指标；`test` 只在最终一次评测触达，且每个快照只评一次。
- 复测：用 dev 选点后重跑 prefix-30k/60k/90k 与基线的 test 评测，检查排序与显著性是否变化；若变，则 M5 的部署建议一并重写。

### M2. 通道探针的 gate/contrib 统计量定义错误，文档引用的数字不成立

- 轴：A（计算错误）/ E（声明不一致）
- 位置：`tools/probe_condition_channels.py:108-118`（hook）、`tools/probe_condition_channels.py:132-151`（聚合）
- 证据：hook 记录的是 `adaLN_modulation[1]` 输出 `[B, 6D]` 在 batch 上的均值 → 一维 `[6D]`；聚合处却把它当成分块序列：
  `entry["gate_msa"] = float(g[:3].mean())`、`entry["gate_mlp"] = float(g[3:].mean())`，
  而正确写法是 `g.chunk(6, dim=-1)`（gate 分别是第 3、第 6 块），交叉注意力同理应为 `gx.chunk(3, dim=-1)` 的第 3 块。
- 实测（本次审计，独立复算 `runs/dit_b_text_x0/final.pt`，t=500，abs 统计）：
  | 量 | 仓库报告值（`reports/eval/probe_channels_ditb60k.json`） | 正确统计量 |
  | --- | --- | --- |
  | gate_msa | 0.1314（实为 `shift_msa` 前 3 维的均值） | `abs().mean()` = **0.294**（门控幅度） |
  | gate_mlp | 0.6281（实为 `g[3:]` 的带符号均值） | `abs().mean()` = **0.616**（门控幅度） |
  | gate_xa | 0.3975（shift/scale/gate 三块混合） | `abs().mean()` = **0.175**（门控幅度） |
  | cross/mlp 贡献比 | 7.6%（= gate×rms） | 修正后约 **3.4%** |
  并且同一份代码换一次随机噪声即可把 `gate_mlp` 从 +0.63 变成 −0.27（本次复算实测），说明该统计量本身无定义。
- 影响：被引用到 `.trae/documents/NOTES.md:189,192`「gate 全部健康（msa 0.13 / xa 0.40 / mlp 0.63）」与「cross-attn 贡献只有 MLP 的 5–10%」，以及 `.trae/documents/DESIGN.md:53` 同一句话。**定性结论不受影响且被修正值加强**（门控确实非零：0.29/0.62/0.17；cross-attn 相对 MLP 更小，约 3%），因为「文本信息在、影响弱」的主要证据来自记录正确的 `rel_correct_shuffled_hidden / block_out / 最终输出距离`。
- 状态：CONFIRMED
- 修复：改为 `chunk` 取块 + 用 `abs().mean()` 作为门控幅度；同时输出每块的 rms 与残差贡献，重跑 5 个 JSON 并回改 `NOTES.md:189,192` 与 `DESIGN.md:53` 的数字。
- 复测：重跑 `tools/probe_condition_channels.py`（5 个 checkpoint），确认 gate 幅度与贡献比新值稳定、且换噪声不跳变。
- 注：按审计技能约定（read-only，修复另开一轮），本次**未**直接修改 `NOTES.md` 历史条目、`DESIGN.md` 与 `README.md` 的数字，只在本文件登记待改位置。

### M3. 评测帧目录复用 + BT 标签按输出目录名派生 → 陈旧帧可静默进入指标

- 轴：A（正确性）/ D（数据管线）
- 位置：`tools/eval_t2v.py:161-166`；`tools/run_bt_eval_signmbart.sh:20,26-33`
- 证据：
  - `video_dir.mkdir(parents=True, exist_ok=True)` 后按 `images{index:04d}.png` 覆盖写，**不清理目录内多余旧帧**。若同一 `--out` 二次运行且新结果更短（例如 `--length-mode predicted` 或换 checkpoint），上一次的高位帧会残留。
  - BT 链的标签路径由 `TAG=$(basename "$OUT_ROOT")` 决定（`run_bt_eval_signmbart.sh:20`），label 又由 frames-root 自动生成（`:29-33`），因此「帧集合」与「参考文本」始终跟着目录内容走，指标文件里看不出多/少了哪些帧。
  - FVD 同理按目录内 `images*.png` 全量读取（`tools/eval_fvd.py:97`）。
- 影响：回译与 FVD 可能对「上一次的残留帧」评分，且无法从 `score.json` 察觉；同一 out 目录被两个 run 复用时，数字不可归因。当前已有多组历史产物共享 `reports/eval/bt_*` 命名（`bt_prefix60k`、`bt_prefix90k`…），属于易踩区。
- 状态：CONFIRMED（代码层面）；已产出数字是否受影响未逐目录复核（见 §10）
- 修复：写入前 `shutil.rmtree(video_dir, ignore_errors=True)`；或输出到带 run tag 的子目录（`frames/<tag>/<split>/<id>`）并在 `meta.json` 记 tag；BT 链显式传入 `--label-tag` 并断言 `帧数 == meta.length`，不一致直接失败。
- 复测：同一 out 目录先跑 32 条 oracle、再跑 predicted，比较帧数与 FVD/BT 是否随清理而变化；对现有 `reports/eval/bt_*` 抽 2 个目录核对 `score.json.per_video` 数量 = 32。

### M4. 冻结 VAE 契约默认不校验，且上游代码已漂移 5 个 commit

- 轴：G（可复现性）/ C（模型结构前提）
- 位置：`t2v/adapters/vae.py:60-68`（运行期 `sys.path` 插入可变仓库并 import）、`:76-82`（`verify_checkpoint_sha256` 默认 `False`）、`:90-98`（`strict=False`，只有 unexpected 才报错）；`../RefSign/UPSTREAM.md:10`（契约 commit 行）
- 证据：
  - `RULES.md` 第 1/6 条与 `UPSTREAM.md` 声明 VAE 代码 commit = `d81ce1e1e3989c82aeb699015c32e0ad9e9a363e`；实测 `git -C /data/huyang/trae_projects/TiTok_base/main rev-parse HEAD` = **`a312ab7b64bbcd369fa08c8078fc73b307218db7`**，落后契约 5 个 commit，且工作区还有 5 处改动。
  - 这 5 个 commit 改了 `modeling/titok_cvae.py`、`modeling/modules/cvae_blocks.py`、`modeling/modules/cvae_losses.py` 与 `data/*.py` —— 其中 `modeling.titok_cvae` 正是解码器 import 的模块。
  - 工作区也不干净（10 项）：删除 4 个根目录 md、修改 `scripts/build_frame_pair_csv.py`（数据管线）、新增 `.trae/*`、`configs/.../titok_cvae_h07.yaml`、`audit-report.md`。`UPSTREAM.md:10` 写的「代码目录未改动」已与现状不符。
  - 缓解证据（本次实测）：当前代码加载冻结 checkpoint 为 `missing 0 / unexpected 0`；新增项均默认关闭（`pixel_refine=False`、`decoder_extra_blocks=0`、`posterior_wide=False`、`aggregator_extra_blocks=0`），`tanh(1+gate)` 改动只作用于新分支，`posterior_parameters` 的 `logvar.clamp(-30,20)` 不影响被缓存的 posterior mean。因此**目前无证据表明解码数值已改变**，但也没有任何自动化测试在守这条线。
- 影响：所有像素侧数字（FVD、回译、对比抽帧图）依赖「解码器行为不变」这一未经校验的前提。契约文档与实际代码状态已经不一致（声明冲突），且每次上游提交都可能静默改变解码结果。
- 状态：CONFIRMED（漂移与默认不校验）；解码数值是否变化：LIKELY 不变，未验证
- 修复：默认 `verify_checkpoint_sha256=True`；把 VAE repo commit、config sha、checkpoint sha 一起写进运行签名并在加载时比对；`missing_keys` 非空即 `raise`；新增一条「冻结解码金标准」回归测试（固定 latent + 固定 ref → 固定像素哈希），任何上游改动立刻暴露。
- 复测：改动 checkpoint 中任意一个张量后跑 decode，应报错；同一 latent 在契约 commit 与当前 HEAD 下解码输出应逐位一致（或明确记录差异）。

### M5. SignMbart 口径切换后文档未同步，部署结论自相矛盾

- 轴：E（代码与声明）
- 位置：`README.md:56-60`、`README.md:85-93`、`README.md:103-138`、`README.md:14`、`README.md:157-159`；`.trae/documents/DESIGN.md` 里程碑段；`t2v/models/dit.py:18-40`
- 证据（均为同一文件内互相冲突）：
  - `README.md:56-60` 把「当前最优」写成 `prefix-60k`；`README.md:85-93` 的 SignMbart 表里**只有 90k 显著**（ΔB1 +5.79 [+0.87,+11.26]），90k 的 B1/ROUGE/WER 全面最好。
  - `README.md:118-122` 的「训练长度消融结论：90k 不加分 → 部署维持 prefix-60k」来自 GFSLT 口径，却排在历史表之后的正文位置，读者会当成当前结论。
  - `README.md:135` 「绝对语义对齐仍有限（B4≈2.9）」是 GFSLT 数字；SignMbart 口径下所有生成 run 的 B4 均为 0（`README.md:87-98` 自己写明 GT 上限 B4 29.51）。
  - `README.md:14` 写「为适配 1-token/帧而做的 5 处最小改动」，`t2v/models/dit.py:18-40` 实际列 6 条（含 `text_prefix`）。
  - `README.md:157` 的目录树把长度模型写成 `models/length_model.py`，实际在 `t2v/models/length_model.py`；`README.md:159` 仍写 `tools/eval_t2v.py` 是「GFSLT 回译可直接消费」。
  - `DESIGN.md` 里程碑仍把「生成视频回译（`tools/run_bt_eval.sh`，GFSLT 链路）」列为已完成评测项。
- 影响：违反 RULES 第 5 条「换模型=换口径，历史表整表重算」的核心纪律：任何引用 README 当前区块的人都会把 GFSLT 数字与 SignMbart 数字混用，或采用与最新证据相反的部署 checkpoint。
- 状态：CONFIRMED
- 修复：README 顶部只保留一个「当前部署 + 当前口径」区块（SignMbart、32 条 test、明确标注样本量），历史 GFSLT 全部下沉到单一历史小节并注明「不可与前表混用」；部署 checkpoint 只在一处定义；差异清单改 6 条；修正目录树路径。
- 复测：全文 `rg -n "GFSLT"`、`rg -n "60k|90k"` 逐条确认口径标注；新读者只看顶部区块即可复现部署。

## 4. MINOR 发现

| 编号 | 问题 | 位置 | 影响与建议 |
| --- | --- | --- | --- |
| N1 | `sampling.sampler: ddim` 是死配置：全仓无读取点，DDPM 无法通过配置切换（`ddpm_sample_loop` 仅被实现、未接线） | `configs/*.yaml`；`rg "sampling.sampler"` 命中 0 | 配置暗示了一个不存在的能力；要么接线，要么删键并在 README 注明仅支持 DDIM |
| N2 | `clip_denoised` 用图像域 `clamp(-1,1)`，但作用对象是逐维标准化 latent（无 ±1 界） | `t2v/diffusion/gaussian.py:282-283,343-344`；`t2v/data/normalizer.py:52-56` | 默认关闭故当前无影响；一旦打开会截断分布。建议改为按维度分位数裁剪或直接删除该开关 |
| N3 | 生成默认走 EMA，而所有报告数字与部署示例都是 raw | `tools/generate.py:92`、`t2v/pipelines/generation.py:62-69`（无 EMA 时静默回退）；`README.md:57-60` 示例未加 `--no-ema` | 照 README 复现会得到 EMA 权重。建议默认 raw，EMA 显式开启，且缺 EMA 时报错而非回退 |
| N4 | 训练期样例用 EMA（`sample.use_ema: true`），与部署权重不一致 | `configs/dit_b_text_prefix.yaml:training.sample.use_ema`；`t2v/pipelines/visualize.py` | 训练曲线上的对比图不能直接代表部署模型；建议与部署保持一致或同时输出两行 |
| N5 | 长度模型不设 torch 种子、取最后一轮而非最优轮；指标报在 dev（口径可接受但不可复现） | `t2v/train_length.py:60-121,145-158` | 加 `set_seed`、按 dev MAE 选轮；测试集仍未触达，符合纪律 |
| N6 | 文本重复：train∩test 相同译文 20 条（train∩eval 15、eval∩test 9） | 实测 `*_text.jsonl` | 输入句子在 train/test 间重复会让「文本→视频」存在记忆通道，也会让回译参考串重复；建议按文本哈希切分或在报告中披露这 20 条 |
| N7 | resume 后 `epoch` 重置为 0，BucketedBatchSampler 按 epoch 0 重放批序 | `t2v/train.py:269,304`；`t2v/training_utils.py:62-74` | 续训段批序与首段重复（不影响正确性，影响随机性声明）；建议把 epoch 存进 checkpoint |
| N8 | 评测脚本硬编码绝对路径、GPU、外部仓库位置 | `t2v/adapters/vae.py:29-38`；`tools/run_bt_eval_signmbart.sh:22-27`（依赖 `gfslt_vlp` 仓库与 `envs/gfslt` 解释器） | 换机器/清理旧仓库即失效。SignMbart 主链路的**标签与打分**仍复用 GFSLT 仓库实现，建议把 `make_eval_labels_from_frames.py` 与 `score_refsign_protocol.py` 收进本仓库 |
| N9 | 仓库零 commit（`git log` 报 no commits yet），运行签名不含代码与依赖版本 | `git log`；`t2v/training_utils.py:77-88` | 结果无法绑定代码版本；建议首次提交后把 `git rev-parse HEAD`、torch/transformers 版本、mT5 文件哈希写进 `signature` |

## 5. 声明核验（claims ledger）

| # | 声明（来源） | 证据 | 状态 |
| --- | --- | --- | --- |
| 1 | DiT 严格对齐官方 `facebookresearch/DiT`，差异全部登记 | `t2v/models/dit.py:18-40` 列 6 条；块结构/初始化对齐官方写法 | PARTIAL（README 仍写 5 处） |
| 2 | x-prediction + cosine + logit-normal 修复「碎帧」 | `configs/*.yaml` 默认 x0/cosine/logit_normal；`tests/test_diffusion.py` 覆盖 x0 精确回推 | CONFIRMED |
| 3 | 冻结 1-token VAE + latent 缓存只读、不改缓存 | 软链只读；本次实算 checkpoint/config sha 与 manifest 一致 | CONFIRMED（工程强制不足，见 M4） |
| 4 | 所有回译一律 SignMbart，不与 GFSLT 混表 | 链路 `tools/run_bt_eval_signmbart.sh` 已切换 | PARTIAL（依赖 GFSLT 仓库标签/打分脚本与解释器；DESIGN 里程碑仍写 GFSLT） |
| 5 | prefix 联合注意力显著增强文本条件 | SignMbart 32 条下仅 90k 显著；GFSLT 下 60k 最好 | PARTIAL（口径依赖；test 选点，见 M1） |
| 6 | 「90k 不加分 → 部署维持 prefix-60k」 | 该结论来自 GFSLT；SignMbart 表结论相反 | CONTRADICTED |
| 7 | 「绝对语义对齐仍有限（B4≈2.9）」 | SignMbart 生成 run B4 全为 0，GT 上限 29.51 | CONTRADICTED（作为当前口径陈述） |
| 8 | 长度为文本预测（部署闭环），acc 0.409 / MAE 18.3 | `t2v/train_length.py` 实现；`tools/eval_t2v.py:103-134` 支持 `--length-mode predicted` | CONFIRMED（指标报在 dev；无种子/无选轮见 N5） |
| 9 | 通道解剖：文本信息在、影响弱；cross-attn 贡献仅 MLP 5–10% | 定性成立；数字错误（应为 ~3%） | PARTIAL（见 M2） |
| 10 | 训练期每 N 步产出 GT/生成对比抽帧图 | `configs/*.yaml: sample_every`；`t2v/pipelines/visualize.py`；`runs/*/samples/` 产物存在 | CONFIRMED |
| 11 | 全量 642 条 GT 自检复现 SignMbart 官方数字（gloss WER 27.17 vs 27.28） | `tools/signmbart_validate.py`；`reports/eval/gt_full_signmbart/` | CONFIRMED（属评估模型自检，不由本仓库复核） |
| 12 | 「10 个 run 重跑逐位一致」 | `reports/eval/bt_*/bt_signmbart/score.json` 存在 | UNVERIFIABLE（本次未重跑；且 M3 提示需先确认帧目录干净） |

## 6. 泄漏与污染评估

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| 目标泄漏（latent 由标签派生） | 无 | latent 来自帧，文本来自独立 corpus jsonl |
| 训练/测试 video_id 重叠 | 0 | 本次实测三对 split 交集 |
| 同一文本跨 split | 20 条 train∩test（另 15/9） | 见 N6 |
| 归一化统计拟合范围 | 仅 train | `train_stats.json` 的 `manifest_sha256` = `sha256(train.jsonl)`，实算一致 |
| 词表/编码器泄漏 | 无 | mT5 为外部冻结模型，未在本数据上训练 |
| 测试集用于选点/选步数 | **是** | 见 M1 |
| 测试集用于超参搜索 | 是（CFG/dropout/步数均以 test BT 决策） | `README.md:85-93,103-138`；`NOTES.md` |
| 探针反复在 test 上评估 | 是（敏感度/天花板/信号探针） | `tools/probe_*.py` 默认 `--split test` | 
| 评测帧目录陈旧残留 | 可能 | 见 M3 |
| 缓存陈旧 | 未发现 | manifest/config/checkpoint sha 一致（§1） |
| 外部数据污染 | 无 | 仅 PHOENIX-2014T 与冻结 mT5/SignMbart |

## 7. mock / 占位 / 伪造清单

红线扫描（`reports/audit/2026-10-13/red_flags.json`，138 个源文件）后的逐条判定：

| 位置 | 类型 | 仅测试？ | 影响结果？ | 处置 |
| --- | --- | --- | --- | --- |
| `tools/signmbart_bt_infer.py:60-68` `class SummaryWriter` | import-time stub（缺 tensorboard 时补空实现） | 否 | 否（只影响日志写入） | 保留，但建议加一条显式注释与依赖检查 |
| `tests/test_dit.py:69-73` `try/except ValueError: return` | 扫描器误报（异常未抛出时下面会 `raise AssertionError`） | 是 | 否 | 无需处理 |
| `t2v/adapters/text.py:7` / `signmbart_bt_infer.py:25` 注释里的 "precomputed/hardcoded" | 文字命中 | — | 否 | 无需处理 |
| 训练日志 `log_*.log` 中的 `metric_literal` 159 处 | 日志行 | — | 否 | 无需处理 |
| 硬编码指标 / 假基线 / 伪造产物 | 未发现：无 `Mock()`、无 `if debug: return`、无按需返回固定分数的分支 | — | — | — |

## 8. 严谨性与可复现性

| 检查 | 结果 | 证据/说明 |
| --- | --- | --- |
| 种子 | python/numpy/torch/cuda 全设（`training_utils.py:17-21`），文本 dropout RNG 固定 | 未设 `cudnn.deterministic`；DataLoader 无 `worker_init_fn`（PyTorch 默认由父进程种子派生，当前数据管线无随机增广，风险低） |
| cudnn 确定性 | 未设置 | TF32 打开（`t2v/train.py:30-31`）→ 跨机不完全逐位可复现 |
| 每次运行捕获配置 | 是（`runs/<name>/config.yaml` + checkpoint 内 `config`/`signature`） | `training_utils.py:77-116` |
| 捕获代码版本 | 否 | 仓库无 commit（N9） |
| 捕获依赖版本 | 否 | `requirements.txt` 只有下界（`>=`） |
| 捕获冻结契约版本 | 部分 | cache manifest hash 进入签名；VAE commit/sha 未进入，且默认不校验（M4） |
| checkpoint 选择准则 | 无（恒取最后一步） | M1 |
| 测试集纪律 | 破坏 | M1 |
| 指标方向与聚合 | 正确 | `_masked_mean` 只统计有效帧；回译协议与 RefSign 一致 |
| 误差棒 | 已有配对 bootstrap（`tools/bt_paired_stats.py`），但仅 32 条 | 建议扩样本 |
| 基线可比性 | 同数据、同 VAE、同步数对照（60k baseline vs 60k prefix） | 唯一干净对照，NOTES 已注明 |
| 手工步骤 | 有：GPU 轮询脚本、外部仓库标签脚本、后台 setsid 调度 | 建议脚本化并写入产物元数据 |

## 9. 改进方案

### P0 — 先修纪律与污染面（不训练也能做）

1. **加 dev 验证与选点**（M1）：训练循环内每 N 步记 `dev/mse`；新建 `tools/eval_dev.py`，在 dev split（519/518 条）上做小样本（32–64）回译 + FVD，作为 checkpoint/步数选择依据；`test` 只在最终评测触达一次。
2. **清理评测目录 + 显式标签**（M3）：`eval_t2v.py` 写帧前 `rmtree`；BT 链增加 `--label-tag` 与「帧数 == meta.length」断言；FVD 读取时校验每条视频帧数与 `meta.json` 一致。
3. **冻结契约 fail-closed**（M4）：默认校验 checkpoint/config sha；把 VAE commit 写进 `signature` 并在加载时比对；`missing_keys` 非空即报错；加一条「冻结解码金标准」回归测试。
4. **文档口径单一化**（M5）：README 顶部只留一个「当前部署 + 当前口径」区块（SignMbart / 32 条 / 样本量 / checkpoint 路径），GFSLT 与 60k-vs-90k 的旧结论统一下沉到「历史口径」小节并标注不可混表。
5. **修探针并重算**（M2）：`chunk(6)`/`chunk(3)` + `abs().mean()`；重跑 5 个 JSON；回改 `NOTES.md`/`DESIGN.md` 中的 gate 与贡献比数字。

### P1 — 把评估做扎实（1 周内）

6. **评测集扩到 128 条**（含 GT 上限），默认输出配对 bootstrap 95% CI 与 `P(Δ>0)`；32 条结论一律降级为「初步」。
7. **CFG 扫描补 SignMbart 口径**：目前只有 CFG 2.0 被系统回译过，而 latent gap 显示 CFG 4 文本依赖更强；在 dev 上做 CFG∈{2,3,4}×rescale∈{0,0.7} 的 BT 配对实验，再用 test 复核最佳组合。
8. **训练期样例与部署一致**（N3/N4）：`sample.use_ema: false`（或同时输出 raw/EMA 两行），`tools/generate.py` 默认 raw。
9. **补齐可追溯性**（N8/N9）：首次 `git commit`；`signature` 里加 `git rev-parse HEAD`、依赖版本、mT5 文件哈希；把 GFSLT 仓库的标签/打分脚本收进本仓库 `tools/bt_protocol/`。

### P2 — 方法层下一步（有明确假设再投入）

10. **条件侧**：mT5 取层消融（当前用最后一层 hidden + 池化）、token 级 vs 池化权重、换成双向编码器（mBART/LaBSE）作为冻结文本塔；条件编码器容量与预训练量（当前随 DiT 从零训）。
11. **文本 dropout 与 CFG 的联合调参**：已排除 0.05（负结果），可试 0.15–0.20；但必须与 P1 的扩样本评估一起做，否则读不出差异。
12. **时长/帧率读取侧实验**：`frame_stride=2` 是缓存契约允许的读取侧改动（输出 fps 12.5），可显著降低序列长度、提升训练效率，代价是时间分辨率。
13. **天花板实验**：把「1-token VAE 手部细节上限」作为独立探针（解码 GT latent 的 PSNR/手部区域指标），避免把 VAE 上限误判成 DiT 条件问题。
14. **配对差口径固化**：把「correct vs shuffled 回译配对差 + 绝对质量」写成一个命令（`tools/eval_suite.py`），所有实验一键产出同格式报告，杜绝手工拼表。

## 10. 残余不确定性与覆盖缺口

- M3 只证明「代码允许污染」，未逐目录复核历史 `reports/eval/bt_*` 是否真的混入过残留帧；建议按 §3 复测项抽检。
- M4 未做跨 commit 的解码数值等价测试（会改动相邻仓库工作区，未执行）；当前仅有「加载覆盖 0/0 + 新增分支默认关闭」的间接证据。
- SignMbart 链路内部（`signmbart_bt_infer.py` 的语义还原修复）未独立复核，采信 `signmbart_validate.py` 的全量 642 自检结论。
- 32 条 test 的配对 bootstrap 只覆盖采样不确定性，不覆盖「32 条样本选取」这一层不确定性；扩样本前不宜外推。
- FVD 在 32 条规模下 Frechet 估计噪声大（工具自身注释已声明），只作趋势。
- 本次未重跑训练；所有「训练是否可复现」结论来自代码与日志一致性，不含重训对照。

## 11. 附录：本次运行的命令与证据

```bash
python tests/run_all.py                                   # 4 个测试全过
python /home/huyang/.agents/skills/research-codebase-audit/scripts/scan_red_flags.py \
    --root . --exclude runs,reports,data,models,reference \
    --out reports/audit/2026-10-13/red_flags.json          # 138 文件，0 跳过
sha256sum /data/huyang/trae_projects/TiTok_base/main/titok_cvae_run_gpu2_main_phoenix14t_main/psnr-best.pt
sha256sum configs/frozen/vae_h02.yaml data/phoenix_latents/v1/train.jsonl
git -C /data/huyang/trae_projects/TiTok_base/main rev-parse HEAD
# 通道探针复算（scratch，不入库）：/tmp/probe_gate_abs.py  → gate_msa|0.294| gate_mlp|0.616| gate_xa|0.175
```

证据文件：`reports/audit/2026-10-13/red_flags.json`（线索清单）、`reports/eval/probe_channels_*.json`（原始探针输出）、`runs/*/final*.pt`（checkpoint）、`data/phoenix_latents/v1/{manifest.json,train_stats.json}`。

## 12. 整改状态（2026-10-13 晚，第一轮 P0）

用户指令「开始修改这些主要的问题」后，本轮只动 P0 范围（不改缓存/checkpoint/上游仓库、不重训）。
过程与证据见 `NOTES.md`「2026-10-13 晚：审计 P0 整改」。

| 发现 | 状态 | 落地位置 | 验证 |
| --- | --- | --- | --- |
| M1 无验证集 / 用 test 选点 | **部分修复** | `t2v/dev_eval.py`（新增）、`t2v/train.py` 记 `dev/mse`、`tools/probe_condition_channels.py` 默认 split 改 `eval` | `tests/test_dev_eval.py` 通过；冒烟 60 步 `dev/mse` 1.019→0.437；**尚未用 dev 重新选点后重跑 30k/60k/90k** |
| M2 探针统计量错误 | **已修复并重算** | `tools/probe_stats.py`（新增 `chunk_abs_mean`）、`tools/probe_condition_channels.py` | 5 个 JSON 重跑；基线复算 0.296/0.177/0.619 ≈ 审计独立值 0.294/0.175/0.616；`tests/test_probe_stats.py` 通过；`NOTES.md:189,192`/`DESIGN.md` 数字已标注修正 |
| M3 评测目录复用 / 标签歧义 | **已修复** | `t2v/pipelines/decode.py::prepare_frame_dir`、`tools/eval_t2v.py`（清空 + 帧数断言）、`tools/run_bt_eval_signmbart.sh`（`LABEL_TAG` + 帧数==meta 断言） | `tests/test_eval_frames.py` 通过；脚本 `bash -n` 通过；**历史 `bt_*` 目录未逐目录复核** |
| M4 冻结契约未强制 | **已修复（工程侧）** | `t2v/adapters/vae.py`（默认 sha 校验 + 全量 key 覆盖 + 契约记录）、`t2v/training_utils.py`（签名写 `frozen_contract`） | `tests/test_frozen_guard.py` 通过；签名实打印暴露上游 `a312ab7b` vs 契约 `d81ce1e1`；**未做像素级金标准回归** |
| M5 口径文档未同步 | **已修复** | `README.md`（单一口径区块 + GFSLT 下沉）、`.trae/documents/DESIGN.md`、`STRUCTURE.md` | `rg -n "GFSLT|B4≈2.9|5 处" README.md` 逐条确认标注 |
| N1 死配置 `sampling.sampler` | 已修复 | `t2v/diffusion/gaussian.py::resolve_sampler` + visualize/generation 调用 | `test_diffusion.py::test_sampler_config_is_enforced` |
| N2 `clip_denoised` 语义 | 已记录（未改行为） | `t2v/diffusion/gaussian.py` 两处注释 | 默认全关 |
| N3 生成默认 EMA ≠ 部署 raw | 已修复 | `tools/generate.py`、`tools/eval_t2v.py`、`tools/text_sensitivity.py`、`t2v/pipelines/generation.py` | 默认 raw；请求 EMA 但缺 EMA 时报错 |
| N5 长度模型无种子/取末轮 | 已修复 | `t2v/train_length.py`（`--seed` + 按 dev MAE 选最优轮） | 语法/编译检查；重训待用户决定 |
| N9 签名无代码/依赖/契约版本 | 已修复 | `t2v/training_utils.py::config_signature`（`code`/`deps`/`frozen_contract`） | 实打印 smoke 配置签名确认 |
| N7 resume 后 epoch 重置 | 未处理 | `t2v/train.py` / `BucketedBatchSampler` | — |
| N8 硬编码路径 / BT 协议脚本外置 | 未处理 | `tools/run_bt_eval_signmbart.sh`、`t2v/adapters/vae.py` | — |
| N6 文本跨 split 重复 20 条 | 未处理（已在 §6 披露） | 数据侧 | — |
| 仓库零 commit | 已修复 | git | 本轮产出**首次 commit**（代码+文档，`runs/reports/data/models` 已忽略）；此后 `config_signature().code.git_head` 可绑定代码版本 |

结论：5 个 MAJOR 全部进入「已修复或部分修复」，其中 M1 的复测（用 dev 重新选点再复跑三种步长）
需要重训，属下一轮；M3/M4 的残余项已在表中明确。

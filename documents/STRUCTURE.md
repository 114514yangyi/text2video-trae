# STRUCTURE — 项目结构

## 2026-10-22 当前实现（优先阅读）

当前 `milestone1-pixart` 从 c20c4a8 开始完成 M1 实现。以下旧版“45 文件/待实现”是历史状态。

- `t2v/models/pixart.py`：PixArt adaLN-single、ConditionBag、无全局内容的 token 投影、cross/prefix/both、逐层注入、严格 block-only 预训练载入；生产尺寸28/1152/16，latent1024。
- `t2v/models/factory.py`：训练/推理共用组装。
- `t2v/adapters/text.py`：新增物理截断 num_layers；4c 保留 embedding+3block+finalnorm 并训练。
- `t2v/train.py`：单卡/torchrun DDP，分桶有效batch、梯度累积、文本独立LR、可选motion、完整三模块CPU EMA与精确续训。
- `t2v/milestone_runtime.py`：配置继承、本地路径重定位、CPU状态快照、原子存档、RNG状态。
- `t2v/pipelines/milestone.py`：GT长度采样、首帧一次、固定前24帧解码、训练可视化，完整raw/EMA一致载入。
- `tools/eval_milestone.py`：固定清单、显式pilot、生成后固定调用正式 FVD 入口 `tools/eval_video_metric_fvd.py` 计分。
- `tools/eval_video_metric_fvd.py`：FVD 的 IO 壳（test 清单/帧协议 + 真值半分本底 + protocol/SHA256）；所有指标数学调用 `third_party/video_metric` 的 `calculate_fvd`，壳内不含任何本地实现。
- `third_party/video_metric/`：上游 `JunyaoHu/common_metrics_on_video_quality` 的逐字节副本（`calculate_fvd.py` + `fvd/styleganv` + `fvd/videogpt` 的 py；I3D 权重为本地 git-ignore 资产），见其 `PROVENANCE.md`。
- `tools/make_milestone_compare_video.py`：固定 test 视频的 GT-vs-生成对比视频，两种长度：`--mode first24`（评测计分的前 24 帧）与 `--mode full`（GT 长度完整片，用 `latents.pt` 的完整 latent 轨迹经冻结 VAE 在 CPU 重建，不载模型权重）；每臂一个 side-by-side + 全臂 2×2 网格 + contact sheet + manifest；只读 `reports/eval/*/frames` 与 `fvd.json`，不占 GPU。
- `configs/milestone1/{base,E1..E10}.yaml`：全部十臂仅用配置切换；本轮只启动 E1。
- 新测试：test_pixart/test_milestone_text/test_milestone_training/test_milestone_eval/test_video_metric_fvd。

实际24帧清单为 test640/642（两条不足24），`reports/eval/samples/m1_test_first24_ids.txt`；24帧从frame0开始的官方I3D-400本底84.297514。旧 FVD 报告产物（含当时的 `real_floor.json`）已于 2026-10-27 按用户指令删除，本底需用 `tools/eval_video_metric_fvd.py --fill-ref-cache` 重算；与旧16帧表不混。


> 权威文档：根目录 `README.md`（目标/命令/产物/差异表）。本文件随代码结构变动同步维护。
> 审计与改进方案：`.trae/documents/AUDIT.md`（2026-10-13 全链路审计，含发现/证据/修复/复测条件）。
> **2026-10-12 建仓**：由 `../RefSign` 的 pose→video 路线派生出的新方向——
> 直接 text→video（DiT），无 pose；冻结 VAE/latent 栈与 RefSign 共享（软链只读）。

> ## ⚠️ 2026-10-22 里程碑1 重置（读本节优先，第 1 节以后为历史）
>
> 分支 `milestone1-pixart`（从 `main`/`bdf36d1` 切出）把项目当**全新项目**重建：
> 旧 DiT / Wan / SignViP / 全部探针与消融代码已删除（208 → **45 个跟踪文件**，
> 提交 `88945f0`；同分支 `44b874c` 单独移植了官方 FVD 评测链）。
> 规格见 `.trae/documents/MILESTONE1_ARCH_PIXART_20261022.md`。
> **本文件第 1 节「目录结构」及以后描述的是旧世界（`main`/HEAD 分支）**，在本分支上
> 那些文件已不存在；查旧模块职责时仍可读，但不要据此判断本分支有什么。
>
> ### 本分支当前结构（45 个跟踪文件）
>
> ```text
> t2v/
> ├── adapters/{vae,text}.py                  # 冻结 TiTok-CVAE + mT5-base（未改，契约不变）
> ├── data/{dataset,collate,normalizer}.py    # latent 缓存读取 / padding+mask / 归一化
> ├── diffusion/gaussian.py                   # x0/ε/v 目标、cosine、DDIM/DDPM/DPM-Solver++、CFG
> ├── models/{conditioning,dynamic_loss,length_model}.py
> │                                          # 条件编码器骨架 / motion loss / 长度头（待新骨架改写）
> ├── pipelines/decode.py                     # 冻结 VAE 解码 + 帧/视频写出
> ├── training_utils.py                       # 种子 / EMA / 原子 ckpt / BucketedBatchSampler
> └── monitoring.py                           # metrics.jsonl / wandb
> tools/  eval_milestone.py eval_video_metric_fvd.py collect_experiment_table.py
>         eval_vae_recon.py make_eval_sample.py prune_frames.py gpu_queue.py queue_status.py
> configs/frozen/vae_h02.yaml                # 冻结 VAE 契约（→ ../RefSign）
> tests/  run_all + 13 个只依赖上述模块的测试
> ```
>
> ### 已删除（不在本分支，需要时去旧分支看）
>
> `t2v/models/dit.py`（旧 DiT 1299 行）、`factory.py`、`pipelines/{generation,visualize}.py`、
> `t2v/{train,dev_eval,train_length}.py`、`models/{margin,repa,semantic,softalign,stage_connector,
> temporal_align,text_slots,latent_text}.py`、`data/paraphrase.py`、39 个旧实验 config、
> 91 个实验驱动/probe/回译/统计 tools、18 个旧架构 tests。
>
> ### 待实现（M1.1–M1.4）
>
> PixArt-α 骨架本体、`ConditionBag` 条件接口 + 逐层注入计划表、训练入口、采样入口、
> 24 帧 uniform FVD 路径（规格 Q5）。

## 目录结构

```text
text2video/
├── t2v/                              # 核心 Python 包
│   ├── models/
│   │   ├── dit.py                    # DiT（官方结构：adaLN-Zero/sincos/FinalLayer + 1-token/帧适配
│   │   │                             #  + 可选 text_prefix：文本 token 前缀进 self-attn，2026-10-12）
│   │   │                             #  + set_cross_gate_scale(k)：idea-016 推理期 gate 缩放，不改权重
│   │   │                             #  + expert_adaln：idea-041 文本/帧分开 adaLN；MonotonicTextGrounding：
│   │   │                             #    idea-040 帧↔词软单调窗（零门控挂到帧流）
│   │   ├── conditioning.py           # text(cross-attn+池化)/length/可选 reference → DiTConditioning
│   │   │                             #  （DiTConditioning 带 text_len：MTG 窗口用；reference 条件由约束 4 启用）
│   │   ├── latent_text.py             # 独立 latent→词序列教师（仅门控诊断，不进入生成条件；
│   │   │                             #  encode_bins 兼作 T2.11 REPA 的对齐目标，2026-10-14 晚）
│   │   ├── repa.py                    # T2.11 表示对齐：冻结 latent 编码器特征 + 训练期投影头
│   │   │                             #  + BlockFeatureRecorder（hook 取中间层，无第二次前向）
│   │   ├── dynamic_loss.py           # T2.12 多尺度时间差分损失 ΔP_k(x0̂) vs ΔP_k(z)
│   │   ├── margin.py                 # idea-001 正确 vs 长度匹配错配的条件 margin 铰链（wrong 分支 no_grad）
│   │   ├── text_slots.py             # idea-017 固定槽接口：M 个可学习 query 读真实 token 出定长槽
│   │   ├── softalign.py              # idea-002 窗口化软文本-帧对齐损失（作用在 DiT 可见文本 context）
│   │   └── factory.py                # build_bundle（DiT+条件编码器）/ build_diffusion
│   │   └── semantic.py               # 独立时序 latent/文本匹配器、检索门槛、冻结语义目标
│   ├── diffusion/
│   │   └── gaussian.py               # x0/ε 预测、linear/cosine 调度、q_sample、vb 损失、DDIM/DDPM、
│   │                                 #  DPM-Solver++(2M)（2026-10-14）、sample_latents 分派、CFG、
│   │                                 #  respaced_timesteps 的 uniform/logsnr 阶梯（idea-032，2026-10-18）
│   ├── models/length_model.py        # 文本→长度桶分类器（部署长度由文本预测）；
│   │                                 # predict_length 支持 median / expectation 两种 readout（T1.7）
│   ├── data/
│   │   ├── dataset.py                # 冻结 latent 缓存读取（preload 或 LRU shard；text/meta）
│   │   │                             #  + 可选 idea-003 保义改写增强（paraphrase_probability）
│   │   ├── paraphrase.py             # idea-003 改写表 + 槽位审核，**审计与训练共用同一份实现**
│   │   ├── collate.py                # 变长 padding + frame_mask
│   │   └── normalizer.py             # 逐维 mean/std（与 RefSign 缓存兼容）
│   ├── adapters/
│   │   ├── vae.py                    # 冻结 TiTok-CVAE（编码参考帧/后验均值/解码）
│   │   └── text.py                   # 预训练 mT5，可选全量微调；当前空串=CFG 无条件
│   ├── pipelines/
│   │   ├── generation.py             # TextToLatentPipeline（checkpoint → 文本→latent）
│   │   │                             #  + GenerationRequest.cross_gate_scale（idea-016 推理期剂量）
│   │   ├── decode.py                 # LatentVideoDecoder + gif/mp4/对比抽帧图 writer
│   │   └── visualize.py              # TrainingVisualizer（每 N 步采样→解码→对比样例）
│   ├── train.py                      # 训练入口（bf16、EMA、AdamW、log/ckpt/sample 调度）
│   ├── dev_eval.py                   # eval split 留出评估：确定性快照 + dev/mse（审计 M1）
│   ├── monitoring.py                 # metrics.jsonl + wandb
│   └── training_utils.py             # 种子/EMA/checkpoint/BucketedBatchSampler/签名
│                                     #  （签名含 git HEAD/依赖版本/冻结 VAE 契约，审计 N9）
├── configs/
│   ├── flagship_ref_v1.yaml          # ★旗舰 V1：从零 + ref 条件 + Expert AdaLN + MTG + 对齐损失
│   │                                 #   + 长度加权 + 文本微调（branch flagship-ref-grounded，2026-10-19）
│   ├── full_text_dit.yaml            # 主配置（DiT-S、30k 步、sample_every 1000）
│   ├── dit_b_text.yaml               # DiT-B 60k 基线（文本仅 cross-attn；x0+cosine+logit-normal）
│   ├── dit_b_text_prefix.yaml        # text-prefix 联合注意力消融（2026-10-12 深夜）
│   ├── dit_b_text_prefix_padmask.yaml # + mask_text_padding/truncate_text_prefix（2026-10-14 审查 H15 修复）
│   ├── dit_b_text_prefix_sc.yaml     # T1.5 self-conditioning 分支（2026-10-14）
│   ├── dit_b_text_prefix_align.yaml  # ★idea-024 R1：+可微 soft-DTW 时间对齐损失（prefix60k→20k，文本冻结）
│   ├── dit_b_text_prefix_curriculum.yaml # T2.9 文本压力课程 + 词级 dropout
│   ├── dit_b_text_prefix_textlr.yaml # T2.10 文本侧 lr 5e-5
│   ├── dit_b_text_prefix_dual4.yaml  # T2.8 前 4 层模态独立 QKV/FFN（双流）
│   ├── dit_b_text_prefix_dualctl4.yaml # T2.8 容量匹配对照（加宽 MLP，无模态分离）
│   ├── dit_b_text_prefix_dynamic.yaml # T2.12 多尺度时间差分损失 λ=0.5
│   ├── dit_b_text_prefix_repa.yaml   # T2.11 REPA 对齐（block 6 → 冻结 latent 编码器）
│   ├── dit_b_text_finetune.yaml      # A：mT5 全量微调，独立 1e-5 lr
│   ├── dit_b_text_semantic.yaml      # B：A + 已验证冻结语义目标（x0，weight .05）
│   ├── dit_b_text_inverse.yaml       # C：A + 已门控 latent→词序列辅助目标（x0）
│   ├── smoke.yaml                    # 冒烟配置（60 步、1 条样例）
│   └── frozen/vae_h02.yaml           # → ../RefSign 冻结 VAE 配置（软链，单一来源）
├── t2v/train_length.py               # 训练文本→长度分类器（无 oracle 长度部署）；epoch 选择改用 train 内部
│                                     #  固定种子 holdout；readout/桶数在 test 642 上裁决（D10/D26）
├── tools/generate.py                 # 离线文本→视频（缓存 video-id 或自由文本）
├── tools/eval_t2v.py                 # 批量生成 test 帧（回译输入）+ latents/meta 落盘
├── tools/eval_fvd.py                 # FVD / 特征 L2（r2plus1d_18）
├── tools/eval_vae_recon.py           # 真实 latent 经冻结 VAE 解码（回译/表示上限参照）
├── tools/probe_latent_retrieval.py   # latent→文本教师检索诊断（correct/rolled/zero；--dump-ranks 落逐视频排名）
├── tools/run_prefix60k_nfe_sweep.sh  # prefix-60k 的 DDIM 步数阶梯（50/100/250/500）同协议对照
├── tools/run_prefix60k_solver_sweep.sh # 同协议但换求解器（DPM-Solver++ 2M 20/50/100）
├── tools/run_solver_followups.sh     # 求解器后续：dev-64 对照 + 新求解器下的 CFG 扫描
├── tools/reval_dev64_steps250.sh     # dev-64 A/B/C 分支在 DDIM 250 步下的复评（标签显式传入）
├── tools/reval_dev64_dpmpp50.sh      # 同上但换新默认求解器 DPM-Solver++ 2M / 50 NFE（新口径整表重算）
├── tools/reval_prefix_ladder_dev64.sh # 训练预算阶梯（prefix 30k/60k/90k）在 dev-64 + 新求解器下复评
├── tools/eval_branch_dev64_dpmpp50.sh # 任意 run/<tag> 在 dev-64 上的一站式评测（BT/FVD/检索）
├── tools/probe_prefix_attention.py   # 文本前缀注意力质量探针（真实词 / pad / latent 三类 key）
├── tools/run_anchor_controls.sh      # 退化 latent 控制：静止 / 逆序 / 全零（解码锚点 oracle 标定）
├── tools/make_eval_labels.py         # 由 meta.json 生成 labels.<split>.pkl.gz（BT 必须显式传标签）
├── tools/fix_bt_labels.sh            # 修复 videos=0 的空回译结果（旧回译标签回退导致的回归）
├── tools/check_label_path_equivalence.sh # 校验显式标签与旧标签路径产出完全一致
├── tools/collect_results_table.py    # 汇总所有评测目录 → Markdown/JSON 结果表
├── tools/make_eval_sample.py         # 新评测规范：固定种子 test/train 采样清单（2026-10-14）
├── tools/eval_matrix.sh              # ★新评测矩阵：一次跑 test+train 两侧同协议全链（2026-10-14）
├── tools/queue_flagship_eval.sh      # ★旗舰 V1 收尾守卫：等 runs/flagship_ref_v1/final.pt 出现后
│                                     #   自动跑 test-642 评测（默认 + gate k=4 两档，2026-10-19）
├── tools/gpu_queue.py                # GPU 作业队列（空闲卡自动起活，可运行时追加作业）
├── tools/prune_frames.py             # 评测后删帧省盘（latents.pt 保留）
├── tools/collect_matrix_table.py     # ★评测结果表（2026-10-15 起 test 单侧）：matrix/<tag><suffix>/<split>_n<N> → Markdown
├── tools/collect_experiment_table.py # ★实验台账指标区：matrix 产物 → EXPERIMENT.md（2026-10-15）
├── tools/paired_arm_stats.py         # ★双臂逐视频配对 bootstrap（BLEU/B4/ROUGE/WER 的 95% CI）
├── tools/paired_mrr_stats.py         # ★双臂 MRR 配对 bootstrap（配 probe 的 --dump-ranks）
├── tools/paired_arm_stats_by_length.py # ★长度分层配对 bootstrap（复用 paired_arm_stats 的可加分解；2026-10-18）
├── tools/probe_semantic_reachability.py # T2.13 语义可达性：GT/加噪/生成 latent 的 paired gap
├── tools/decode_latents_dump.py      # 由 latents.pt 重建帧（无需重采样）
├── tools/probe_external_data_feasibility.py # 外部语料可行性探测（编码/保真/分布）
├── tools/train_semantic_matcher.py   # train 视频/文本配对预训练，dev 门槛后导出
├── tools/train_latent_text_teacher.py # 独立 latent→文本教师训练与 correct/shuffled/zero 门控
├── tools/eval_semantic_matcher.py    # 相同长度错配、倒序验证，v2 门槛重新导出
├── tools/run_preferred_followup.py   # A→B 与等预算 A 对照，dev SignMbart/FVD/bootstrap
├── tools/verify_text_finetuning.py   # 真实文本参数更新、checkpoint 恢复、推理验证
├── tools/text_sensitivity.py         # 正确 vs 打乱文本条件敏感度探针（--cross-gate-scale=k，idea-016）
├── tools/probe_cross_attn_temporal_structure.py # ★idea-019 前置：各帧 cross-attn 分布的时序结构（余弦/JS → verdict）
├── tools/probe_alignment_loss_properties.py # ★idea-024 前置：轻微变速/倒序/静止下的 DTW 损失性质（纯 CPU）
├── tools/probe_align_band_budget.py  # ★idea-037 前置：Sakoe-Chiba 带宽对「轻微变速 vs 退化」分离度扫描（纯 CPU）
├── tools/probe_discrete_code_content.py # ★idea-027 前置A：EMA 码本的码-文本互信息（置换去偏，纯 CPU）
├── tools/probe_gate_scale_sensitivity.sh # ★idea-016 前置：k∈{1,2,4,6.6} 的文本交换敏感度扫描
├── tools/probe_latent_dispersion.py  # ★P9 诊断：逐帧运动幅度/整体 std vs GT（判欠弥散，纯 CPU）
├── tools/probe_motion_amplify.py     # ★P10 前置证伪：z->z0+s(z-z0) 放大后回译/FVD 是否变好（含 --gt 有效性对照）
├── tools/probe_order_sensitivity.py  # ★idea-029 前置②：倒序/块置换/静止的 soft-DTW 顺序缺口 vs GT（零训练）
├── tools/dump_ranks_arm.sh           # 通用逐视频 rank dump 入口（已存在产物即跳过；供新臂配对 MRR 用）
├── tools/probe_error_by_timestep.py  # ★P17：x0 误差按噪声阶段分箱 + 该步预测的平坦度（idea-031 前置，纯推理）
├── tools/queue_ranks_after_eval_r1r2.sh # ★R1/R2 收尾：等 latents.pt 出现后 dump rank 并跑配对 MRR
├── tools/queue_diagnostics_after_eval_r1r2.sh # ★R1/R2 收尾：欠弥散 + 顺序敏感度诊断（等 latents.pt）
├── tools/probe_text_response_by_timestep.py # ★idea-026 前置：文本交换相对响应随噪声阶段的曲线（纯推理）
├── tools/probe_stage_connector_alpha.py # ★idea-026/R2 机制检查：训练后动态臂的 α(t) 是否真的随阶段变化
├── tools/probe_text_to_coarse_plan.py # ★idea-010 前置：文本→粗时间计划（K 块均值）相对长度先验的可预测性（纯 CPU+编码）
├── t2v/models/temporal_align.py     # ★idea-024：可微 soft-DTW 局部时间对齐损失（批内对角向量化，R1 臂用）
├── t2v/models/stage_connector.py    # ★idea-026：阶段相关文本连接器（dynamic/static 等参数配对，R2 臂用）
├── tools/queue_lambda_ranks_after_eval.sh # ★TODO N9：等 λ 消融臂 latents.pt 出现后再 dump 逐视频 rank（守卫防早跑）
├── tools/dump_ranks_gate_scale.sh    # ★idea-016 收口：gate 臂的逐视频 rank dump（配对 MRR 用）
├── tools/dump_ranks_anchor.sh        # ★TODO14：N=642 锚点（重建上限）逐视频 rank dump
├── tools/probe_condition_margin.py   # ★idea-001 前置：correct/长度匹配 wrong/empty 的 x0 误差 + 梯度分配
├── tools/probe_text_interface.py     # ★idea-017 前置：固定槽/仅真实 token/pad 前缀的检索 + 交换敏感度
├── tools/probe_temporal_order.py     # ★idea-002 前置：各变体自加噪重建误差（顺序可见性）
├── tools/paraphrase_audit.py         # ★idea-003 前置：全量字幕改写审核（槽位漂移 + 覆盖率）
├── tools/queue_status.py             # 队列进度速查（完成/在跑/待跑）
├── tools/cfg_sweep.py                # CFG/eta/权重扫描目检图
├── tools/run_semantic_weight_ablation.sh # λ=.005/.01/.02 同初始化等预算消融
├── tools/eval_semantic_followups.py  # 新分支统一生成、SignMbart、配对统计与 FVD
├── tools/run_bt_eval_signmbart.sh    # ★回译主链路：帧 → SignMbart → RefSign 指标协议
├── tools/run_bt_all_signmbart.sh     # 10 个现役 run 的 SignMbart 回译批量重跑
├── tools/signmbart_bt_infer.py       # SignMbart（PHOENIX14T SLT1）驱动 + 训练语义还原修复
├── tools/signmbart_validate.py       # GT 全量自检：复现官方 gloss WER / BLEU / ROUGE
├── tools/signmbart_probe.py          # 逐条诊断：CTC gloss WER vs teacher-forcing 一致率
├── tools/signmbart_trans_debug.py    # 翻译段诊断（多解码配置 + teacher forcing）
├── tools/signmbart_gen_ablation.py   # 解码消融（fp16/fp32 × beam/greedy/nrn）
├── tools/bt_paired_stats.py          # 回译配对 bootstrap（正确 vs 打乱，32 条口径）
├── tools/run_bt_eval.sh              # DEPRECATED：GFSLT 历史口径（不与新数字混表）
├── tools/probe_text_signal.py        # 探针：latent↔文本信息量（线性+MLP 天花板、长度对照）
├── tools/probe_text_ceiling.py       # 探针：文本 1-NN 检索的 gap 天花板（模型同口径）
├── tools/probe_condition_channels.py # 探针：训练态 DiT 各条件通道（gate/贡献/文本敏感度）
├── tools/probe_stats.py              # adaLN 调制分块的单一定义（chunk_abs_mean，审计 M2）
├── tools/probe_temporal_redundancy.py # 探针：latent 时间冗余 r(k) 与 patchify 压力上限（2026-10-14）
├── tools/probe_text_sensitivity_by_length.py # 探针：换文本相对幅度 × 长度分层 / token 匹配带（2026-10-14）
├── tools/bench_sequence_length.py    # 墙钟：帧数 × 文本长度 × pad-mask（含 k=4 等价算例）（2026-10-14）
├── tools/launch_prefix_train.sh      # 共享集群上轮询空闲 GPU 后启动 prefix 训练
├── tools/run_prefix_resume.sh        # prefix 训练崩溃/被抢卡后的续跑（轮询 GPU + --resume）
├── tools/run_prefix_extend60k.sh     # prefix 30k→60k 续训（轮询 GPU，写 final_60k.pt）
├── tools/eval_prefix_after_train.sh  # prefix 30k 终模整套评测链（探针/敏感度/FVD/回译）
├── tools/eval_prefix60k_after_extend.sh # prefix 60k 终模整套评测链（同上 + shuffled 对照）
├── tools/run_shuffled_controls.sh    # 打乱文本对照生成 + 回译（correct vs shuffled 口径）
├── tools/make_comparison_sheet.py    # 多运行 GT/生成对比抽帧拼图（8 帧）
├── tools/run_prefix_extend90k.sh     # prefix 60k→90k 续训（轮询 GPU，写 final_90k.pt）
├── tools/run_prefix_d005_train.sh    # text dropout 0.05 消融训练（30k，轮询 GPU）
├── tools/run_followups_20261013.sh   # 串行调度：90k 续训 → d005 消融（后台 setsid）
├── tools/eval_prefix90k_after_extend.sh # 90k 评测链（逐步骤选卡 + 重试）
├── tools/eval_prefix_d005_after_train.sh # d005 消融评测链（同 90k 模式）
├── tests/                            # test_dit / test_diffusion / test_data / test_conditioning
│                                     # + test_dev_eval / test_eval_frames / test_frozen_guard
│                                     # + test_probe_stats；入口 run_all.py
├── data/                             # 软链（只读）：phoenix_latents、phoenix_frames
├── models/mt5-base                   # 软链（只读）：预训练文本初始化；微调产物保存于 runs
├── .trae/                            # 知识库（布局见下节）；.trae/reference/DiT = 官方 DiT 克隆（git 忽略）
└── runs/                             # 训练输出（config/metrics/ckpt/samples）
```

## 知识库布局（2026-10-14：四态流转；2026-10-16：过程文档归档到 documents/archive/）

规范见 `RULES.md` 知识库流转规范（第 8–10 条）。所有"过程类"文档只写 `.trae/documents/`，
`docs/` 已作废（原内容 2026-10-14 整体迁入）。

```text
.trae/
├── documents/
│   ├── STRUCTURE.md / DESIGN.md / RULES.md / AUDIT.md   # 长期规范、设计、审计
│   ├── THINK.md        # think：设想池（idea-NNN 卡片；只留未裁决）
│   ├── FINISH.md       # finish：已完成待测试的产物（只留未裁决）
│   ├── FAIL.md         # fail：被证伪路线（think→todo→finish→测试 的完整失败链路）
│   ├── NOTES.md        # note：已验证/已裁决的结论（含 2026-10-14 前的历史过程记录）
│   ├── EXPERIMENT.md   # experiment：实验结果台账（exp-NNN，一行一个 run×协议；指标区自动生成）
│   └── archive/        # 历史过程文档（调研/评审/思辨/计划；被 NOTES/FAIL/THINK 当证据引用，故归档不删）
│                       #   BRAINSTORM / NEXT_* / SPARRING / POSITION / PREFERRED_ROUTE_PLAN /
│                       #   SIGN_VIDEO_LITERATURE / TEXT_SEMANTIC_ADVISORY / T1T2_ARM_VERDICTS / THINK_TRIAGE
├── todo/TODO.md        # 待办：只保留未完成条目 + 决策登记（完成 → FINISH.md）
└── reference/DiT/      # 官方 DiT 克隆（git 忽略，仅参考）
```

流转：`THINK.md`（设想）→ 用户筛选 → `TODO.md`（要做的事）→ AI 执行完 → 从 TODO 剔除并登记
`FINISH.md` → 用户测试裁决 → 有效写 `NOTES.md`、无效写 `FAIL.md`（并从 THINK/FINISH 删除）。

横切台账：`EXPERIMENT.md`（`RULES.md` 第 13 条 / TODO **D25**）不参与上面这条流转，它把每次实验的
名称 / 来源 idea / 做法 / 协议 / 指标 / 裁决 / 备注记成一行（`exp-NNN`，行只增不删）。实验跑完即登记，
idea 从 FINISH 出池时回填终值与去向；第 5 节指标区由 `tools/collect_experiment_table.py` 自动生成。

## 关键契约

- **latent**：`[T, 1024]`（每帧 1 token，T = 参考帧之后的帧数），来源 `data/phoenix_latents/v1`
  （train 7021 条可用 / max_num_frames=256 过滤后；eval 519 / test 642）。
- **参考帧**：latent 缓存不含 frame 0；`ref_features [128,512]` 由缓存提供，解码时必须传 GT 首帧图
  （CVAE 条件）。**2026-10-19 起**（约束 4 修订，"方法 4"）：参考帧也作为 **DiT 条件**
  （`conditioning.reference.enabled: true`，进 cross-attn context + adaLN 全局向量）。
- **条件**：text（mT5，CFG=空串）+ length（log 归一化，训练用 oracle）+ reference（可选；旗舰臂开启）。
- **扩散（2026-10-12 用户硬约束）**：x-prediction（预测 x0）；调度默认 cosine
  （linear ADM 可选）；t 采样默认 logit-normal（uniform 可选）；DDIM/DDPM
  （官方 respacing）+ CFG（在预测空间，默认 2.0）；`learn_sigma` 可选（默认 off）。
- **产物**：`runs/<name>/{config.yaml, metrics.jsonl, last.pt, final.pt, samples_*.pt, samples/step_*/}`；
  `metrics.jsonl` 每 `log_every` 步记 `train/*` 与留出 `dev/mse`（选点依据，`test` 不参与选点）。
- **冻结契约强制（2026-10-13，审计 M4）**：`t2v/adapters/vae.py` 默认校验 checkpoint/config
  sha256（`82f37058…` / `0678ee8f…`），state_dict 必须 missing 0 / unexpected 0；契约 repo commit
  只作记录（当前上游 HEAD `a312ab7b` 落后契约 `d81ce1e1` 5 个 commit，已写入运行签名）。
  跳过校验需显式 `T2V_SKIP_FROZEN_VERIFY=1`。
- **采样与写帧防护**：`sampling.sampler` 支持 ddim 与 dpm-solver++（`sampling.order` 1/2，其他值
  直接报错，审计 N1）；**`sampling.timestep_spacing` 支持 `uniform`（默认，历史口径）与
  `logsnr`（idea-032，2026-10-18；CLI `--timestep-spacing`）**，未知值直接报错，
  `uniform` 逐位复现全部历史数字；两条采样路径统一走 `sample_latents`，训练可视化与评测不会各走一套；
  一阶 DPM-Solver++ 与 DDIM 在代数上等价（`tests/test_dpm_solver.py` 逐位校验）；
  `model.mask_text_padding` / `model.truncate_text_prefix`（2026-10-14，审查 H15）：文本适配器把每句
  pad 到 `max_length=120` 后拼进自注意力，默认路径把这 120 个位置全部当有效 key，实测 latent
  query 只有 ~12% 注意力质量落在真实词上、~40% 落在 pad 上（`tools/probe_prefix_attention.py`）；
  两个 flag 用 `context_mask` 屏蔽 pad key 并把前缀截到 batch 内最长真句
  （`tests/test_text_pad_mask.py` 断言"开 flag 后输出对 pad 内容逐位不变"）；评测写帧前
  清空该视频目录（`prepare_frame_dir`），BT 链断言帧数 == `meta.json` 长度（审计 M3）。
- **文本微调**：`conditioning.text.trainable` 默认 false，A/B 配置 true；较小文本 lr
  按 param-group 比例调度，文本 state 随 checkpoint 保存；微调生成仅允许一致权重（EMA 阴影权重
  仅覆盖 model/条件模块，**含 `text_encoder` 的微调 checkpoint 不支持 EMA**，pipeline 显式报错；
  部署默认 EMA 见 TODO **D46**）。
  `length_text_adapter()` 单独复用原始 mT5 供已有长度头，避免微调后的特征漂移。
- **语义目标**：视频 latent 有效序列重采样为 32 个时间位置并加差分，Transformer
  编码后与固定原始 mT5 池化特征对齐；重复文本是多正样本，不作为假负样本。
  dev 门槛必须包括完全等长的不同句子配对；使用 cache manifest/stats 指纹核验兼容性。
  教师冻结，梯度仅穿过视频分支进入预测 x0；CFG 丢文本分支不受句子语义监督。

- `tools/signvip/reproduce.py`：2026-10-22 SignViP发布权重论文条件重跑（原始test、50步完整生成、官方视频/FVD、固定清单及校验）；测试 `tests/test_signvip_reproduce.py`。

- 2026-10-24 D67：正式FVD入口 `tools/eval_signvip_fvd.py`（只做数据IO/固定first24人口/调用）；原样第三方源码 `third_party/signvip/fvd.py` 校验SHA，预处理/特征/距离不再本地实现。`eval_milestone.py`和旧`eval_fvd.py` CLI统一转入此入口。（**历史记录**：2026-10-27 已按用户指令删除该入口与 `third_party/signvip/`，改由 `tools/eval_video_metric_fvd.py` + `third_party/video_metric/` 承担，见 `FVD_SPEC.md`。）

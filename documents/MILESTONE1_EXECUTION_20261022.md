# Milestone1 实现与 E1 执行记录（2026-10-22）

授权：用户要求切至c20c4a8a774f2b376811538c085e6ea01fcff6f6，完成milestone实现并开始第一个base实验。当前在milestone1-pixart，保留既有迁移文件与冻结配置实体化差异，未恢复旧架构。

代码产物：730b3a6（物理三层文本）、da264fc（PixArt）、8733c10（训练与十臂配置）、c26dc71/d71b23d（审查修复）、131390b（采样评测）。独立review见TASK1/2/3/4_REPORT/REVIEW。计划见MILESTONE1_IMPLEMENTATION_PLAN_20261022.md。

## 验证
- 原目标分支55测试通过；截至最终集成83passed/6warnings，9.47s，/tmp/m1-final-suite.log。
- CPU小模型4连续步 vs2+恢复2：三模块raw/EMA全部逐位一致；已保存优化器和随机源/批次游标。
- 实际两卡小模型：6连续步 vs4+恢复2，模型/投影/文本raw/EMA全部逐位一致；/tmp/m1-ddp-continuous.log和m1-ddp-resume.log。
- 小模型200真实训练步：runs/m1_pipeline_validation_20261022；前20步平均loss1.9317→末20步1.5466。它是小尺寸管线验证，绝不是正式E1或质量对照。
- 同一小模型完整GT长度→first24解码→官方I3D：4条显式pilot，FVD1532.3668975，split-half964.5567362；重复计分完全相同。reports/eval/m1_pipeline_validation_20261022/{fvd,fvd_repeat,protocol}.json。不进入正式E1表。
- 活训练模块回调实测：CPU冻结VAE解码、两例MP4/GIF/对比图/完整latent写出，模块train/eval状态还原通过；runs/m1_pipeline_validation_20261022/samples/step_0000200。
- test24协议：640/642（两个短视频排除），first24含frame0、官方224/I3D400，真实本底84.2975141；reports/eval/m1_first24_prepare_20261022/real_floor.json，清单reports/eval/samples/m1_test_first24_ids.txt。I3D自检133.908176通过。

## 评测修复与验收（2026-10-22 22:33 之后）
- 提交 `85952a6`（仅 tools/eval_milestone.py、tools/eval_fvd.py、tests/test_milestone_eval.py）：入口在生成前比较 `data.cache/stats/frame_root` 与完整 `vae` 契约（绝对化后 resolve，允许别名、不要求 out_dir 等训练设置一致）；FVD 先按真实片段帧数筛除短片再校验生成长度，`skipped_short` 写入 prepare 与 scoring 两种报告。
- Lead 独立复跑（`OMP_NUM_THREADS=4`，text2video 环境）：`pytest tests -q` → **91 passed, 6 warnings, 10.90s**，`/tmp/m1-lead-final-suite-20261022.log`；聚焦 `tests/test_milestone_eval.py tests/test_eval_fvd.py` → **22 passed, 4 warnings, 3.96s**，`/tmp/m1-lead-eval-focused-20261022.log`。此前 Task3 报告的 91 passed 得到独立复现。
- 十臂配置矩阵用 `load_config` 逐臂复核：E1–E10 的五因子取值、`injection`、`[T,T,T,R]×7`/全 ref/无 cross 三种注入表、28 层/1152 宽/16 head、30k 步、effective batch 16 全部与规格 §5 一致；E2 为全层 `[ref;text]`，E3 为 ref 层 `[ref;text]`，E5/E6 文本深度 `full`（冻结/全可训），E10 预训练开启。
- 独立 scoped 复审（子代理，只读）见 `MILESTONE1_TASK3_REREVIEW_20261022.md`：**通过，两条 P2 均关闭、无新 P1/P2**，另列 4 条 P3 残留（不可达的第二处长度检查、`--include-frame0` 关闭时的资格口径、契约只比路径字符串/真值缓存无源指纹、4 处测试补强），登记为 TODO 的「M1-评测 P3 残留」，不阻塞 E1。
- Lead 真实资产负向验证：复制 pilot run 的 resolved config、仅把 `data.cache` 改成不存在的 `data/phoenix_latents/v1_contract_probe`，用真实 `runs/m1_pipeline_validation_20261022/final.pt` 运行 `tools/eval_milestone.py --gpu cpu --out /tmp/m1_contract_out` → 在加载 dataset / 创建输出目录之前抛出 `ValueError: Requested source/decoder contract differs from checkpoint config`（exit 1，输出目录未创建），确认 P2-1 是 fail-fast。
- Lead 修复后**正向全链路复验**（2 条显式 pilot，非正式 E1）：`tools/eval_milestone.py --config runs/m1_pipeline_validation_20261022/config.yaml --checkpoint runs/m1_pipeline_validation_20261022/final.pt --pilot post-fix-recheck --max-videos 2 --out /tmp/m1_eval_postfix_recheck --gpu cuda:0` → 契约通过、共享清单 640/642 复用（两条短片列入 `skipped_short`）、GT 长度生成、first24/frame0 解码、官方 I3D-K400 计分全部跑通（N=2，FVD 1878.10，`real_split_half_fvd=null`，exit 0）；protocol 落盘 `source_decoder_contract`。该数字只证明入口可用，不作任何质量/效果结论。

## 训练启动（正式 E1，4 卡首启——已中止，见下节 3 卡重启）

> ⚠️ 本节记录 2026-10-22 22:32 的 4 卡启动。用户随后要求**最多使用 3 张 GPU**，该 job 于 23:35 在 step1110 正常停止，产物归档；当前正式训练是下节「E1 3 卡重启」。
- 命令：`torch.distributed.run --standalone --nproc_per_node=4 -m t2v.train --config configs/milestone1/E1.yaml`；4 卡 RTX3090（GPU 0–3），launcher PID 3446445，四个 rank PID 3446455–3446458，out_dir `runs/m1_E1_base`，启动时 HEAD `63f946f`（评测修复仅动 tools，不影响训练代码）。启动记录 `MILESTONE1_E1_LAUNCH_20261022.json`。
- 实测：7021 条训练视频；backbone **607,113,088** + 条件投影 **1,481,472** + 文本 **213,325,440** 参数（全尺寸，非验证小模型）；microbatch1×accumulation4×world4=effective16。
- 运行观测：step1 loss 1.12718/grad 0.39844/峰值13.73GiB；step10 loss 1.01020、3.373s/步、峰值16.60GiB；step160 loss **0.13229**；step1000 loss **0.03839**、grad 0.4147；step1030 loss **0.03077**、~3.45–3.47s/步；全程 grad 有限（0.15–1.20），峰值显存 16.60 GiB/24 GiB。状态 running，无报错。
- checkpoint_every=1000、sample_every=1000 → 首次正式 `last.pt` 与视觉回调已在 **step1000（23:28）** 产出并完成独立校验，见下节「E1 产物验收」；磁盘在 13.68 GiB checkpoint 写入后仍余 42 GiB（下次原子替换需 ~14 GiB 临时空间，余量足够）。

## E1 产物验收（4 卡首启 step1000，2026-10-22 23:28——产物已归档保留）
- `runs/m1_E1_base/last.pt = 14,688,154,023 B（13.68 GiB）` 于 23:28:38 原子写出，日志 `[ckpt] last.pt step=1000`；随后视觉回调在约 20 秒内写出 `samples/step_0001000/<id>/{images0000..0023.png, comparison.gif, comparison.mp4, comparison_sheet.png, latents.pt, meta.json}`，两例合计 61 文件 / 8.2 MiB；训练在回调后自动继续（23:30 已到 step1030）。
- **CPU 独立校验 checkpoint**（脚本 `/tmp/m1_verify_ckpt2.py`、`/tmp/m1_verify_pipe2.py`）：
  - 顶层键 `version/step/epoch/batch_cursor/config/model/condition_encoder/text_encoder/optimizer/ema/rng_states` 齐全；`version=t2v/milestone1`；`step=1000`、`epoch=2`、`batch_cursor=124`、`rng_states=4`（与原 world size 4 一致）。载入耗时 6.1 s、峰值 RSS 14.1 GiB。
  - 内嵌 config 与运行目录 `config.yaml` **逐字段相等**：arm=E1、五因子 `1a/2a/3b/4c/5b`、sampling DDIM50/CFG2/seed42、runtime world4/effective16/micro1/accum4、git `63f946f`。
  - 三模块 raw 与 EMA 键集合完全一致：backbone 431 tensors / **607,113,088** 参数；condition_encoder 8 / **1,481,472**；text_encoder 31 keys / 405,411,456（T5 tied embedding 在 state_dict 里出现两次，唯一参数 **213,325,440**，3 个 encoder block）。抽查张量全部有限；EMA 已偏离 raw（`x_embedder` mean|raw−ema|=**0.00429**，raw std 0.03117），符合 decay 0.9999 的早期状态。
  - optimizer 2 个参数组（backbone+投影 / 文本），lr **1e-4 / 1e-5**，469 个 state 条目。
  - **评测入口真实全尺寸载入成功**：`MilestonePipeline.from_checkpoint(last.pt,'cpu',use_ema=False)` 16.6 s、`use_ema=True` 13.5 s，均 `step=1000`、weights 标记正确、backbone **607,113,088**、文本 3 blocks / **213,325,440**、注入表 `TTTR×7`；CPU 峰值 RSS 17.5 GiB。即 step1000 checkpoint 可直接用于 first24 评测，无需等到训练结束再验证加载路径。
- 样例元数据（`samples/step_0001000/.../meta.json`）：`length_rule=gt`、`full_latent_frames=27`、`full_num_frames=28`、`decoded_num_frames=24`（frame0 + 前 23 个预测帧）、`weights=raw`、`checkpoint_step=1000`、`video_seed=3100465367500681010`、sampling DDIM50/CFG2/seed42。

## E1 3 卡重启（用户约束：最多 3 GPU，2026-10-22 23:35）
- 用户指令：不得四卡并行，最多使用 3 张 GPU。原 4 卡 job 于 step1110 收到 SIGTERM 正常退出（训练进程与显存全部释放）；4 卡产物整体归档到 `runs/m1_E1_base_4gpu_aborted_20261022/`（含已验收的 step1000 `last.pt` 13.68 GiB 与回调样本，未删除）。
- effective batch16 无法被 `micro1×world3` 整除，经用户选择：**改为 3 卡、microbatch1 × accumulation4 × world3 = effective batch12**（写入 `configs/milestone1/base.yaml`，十臂共用；lr1e-4、text lr1e-5、seed42、30k 步及其余配方不变）。这是对实施计划"默认 batch16"的修订，登记 TODO **D65**；README、十臂配置测试与启动说明已同步，提交 `97b9e9c`（只动配置/测试/README，训练代码与 `85952a6` 相同；run config 记录的 `git_commit=85952a6` 是启动时的 HEAD）。
- 重启命令：`CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --standalone --nproc_per_node=3 -m t2v.train --config configs/milestone1/E1.yaml`；launcher PID **3456546**、rank0 PID **3456574**，启动于 23:36；out_dir 仍为 `runs/m1_E1_base`（从 0 重训：4 卡恢复点要求 world size 4，不能跨 world size 恢复）。启动记录 `MILESTONE1_E1_LAUNCH_V2_20261022.json`。
- 实测：dataset=7021、world3 / accum4 / effective batch12；step1 loss 0.90315 / grad 0.3451 / 峰值13.73 GiB；step10 loss 0.85899、**3.77 s/步**、峰值16.59 GiB；step20 loss 0.69588、grad 有限；GPU0/1/2 各约 18.8 GiB、**GPU3 空闲（26MiB）**，符合 ≤3 卡约束。按 ~3.77s/步 预计 30k ≈ **31 h**（约 2026-10-24 07:00 前后），每 1000 步（约 63 min）存 `last.pt` 并出视觉回调。
- 磁盘：4 卡归档占 14 GiB 后仍余 42 GiB；新 run `last.pt` 约 13.7 GiB，原子替换瞬时需额外 ~14 GiB，最紧时（末次 `final.pt`）余量约 14 GiB，可运行。若要回收空间，可在确认后删除 4 卡归档的 `last.pt`（保留样本/日志/metrics）。
- E2–E10 仅实现配置与支持，本轮不排作业；完整十臂结果不属于“软件已实现”的声明，后续按 TODO 与 EXPERIMENT 更新。

## E1 训练完成（2026-10-24 07:40）
- 正式 run（3 卡 / effective batch12）跑到 **step 30000**，`status.json` 转 `complete`，总耗时 **115,309.58 s = 32 h 01 m**（2026-10-22 23:38 → 2026-10-24 07:40）；训练进程已退出，GPU 全部释放。
- 产物：`runs/m1_E1_base/{final.pt,last.pt}` 各 14,688,143,110 / 14,688,140,689 B（13.68 GiB）；30 组视觉回调（step1000…step30000）；`metrics.jsonl` 3001 条日志。
- loss 曲线（每 3000 步均值）：0.0657 → 0.0266 → 0.0253 → 0.0225 → 0.0210 → 0.0199 → 0.0189 → 0.0184 → 0.0182 → **0.0176**；末 1000 步 3.84 s/步；全程无 NaN/发散，峰值显存 16.60 GiB。
- `final.pt` CPU 独立校验：`version=t2v/milestone1`、`step=30000`、`epoch=51`、`batch_cursor=165`、3 份 rank RNG；三模块 raw 与 EMA 键一致（backbone 607,113,088；condition_encoder 1,481,472；text_encoder 唯一 213,325,440 / state_dict 405,411,456 含 tied 重复）；optimizer 2 组 lr 1e-4/1e-5。可直接用于评测。

## 正式 E1 评测（first24 test640，2026-10-24 09:12 起）
- 先做 2 条 pilot 验证真实 checkpoint 全链路：`tools/eval_milestone.py --config runs/m1_E1_base/config.yaml --checkpoint runs/m1_E1_base/final.pt --pilot real-e1-pilot --max-videos 2 --out /tmp/m1_E1_pilot_20261024 --gpu cuda:0` → 30 s（含加载）、FVD 1558.57（N=2 无统计意义，仅证明入口可用）、协议/契约检查通过。
- 全量评测已起（**只用 1 张 GPU**，符合 ≤3 卡约束）：`CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 tools/eval_milestone.py --config runs/m1_E1_base/config.yaml --checkpoint runs/m1_E1_base/final.pt --out reports/eval/m1_E1_base --gpu cuda:0 --load-workers 8`；进程 PID 3673448，日志 `/tmp/m1_E1_eval.log`；raw 权重、DDIM50/CFG2/GT 长度/seed42、共享 640 id 清单、官方 I3D-400 计分；实测约 3 s/条 → 预计 ~30–35 min 出 `reports/eval/m1_E1_base/fvd.json`。
- **结果（2026-10-24 09:43，raw 权重，N=640）**：**FVD = 126.92693857176755**；真值半分本底 `real_split_half_fvd` = **84.29751411436246**（与 prepare 阶段逐位一致，确认真值特征缓存复用正确）；`mean_feature_l2 = 32.3205 ± 6.5283`；`skipped_short = []`（共享清单已预筛 640/642）；协议 = first24 contiguous + frame0、官方 I3D-K400 400d / 224²、DDIM50 / CFG2 / GT 长度 / seed42、float64 Fréchet；全流程 09:12:34→09:43:12 约 **30.5 min**。模型 FVD 约为真值本底的 **1.51×**，是 from-scratch 30k 的基准数（有信号、但离真值分布仍有明显差距）；不可与旧口径（R(2+1)D 16 帧、SignViP 32 帧 272×224 等）数字混表。
- 按规格 M1.5 判据还需**同协议可复现两次**：第二次评测已启动（`reports/eval/m1_E1_base_repeat`，同 config/ckpt/seed），两次结果与差值待落盘后追加。

## 根盘 ENOSPC 与权重迁移到 /data1/huyang（2026-10-24 10:20–10:35）
- **事件**：2026-10-24 09:50 第二次复现性评测在 43/640 处因根盘 `/` 被写满（PIL 保存 PNG 报 ENOSPC）中止。根盘为共享机（当时 7 个登录用户），可用空间从 13 GiB 被其他进程吃到 0，随后他人释放约 83 GiB；与本仓库训练/eval 的产物体积无关。第一次正式评测（FVD 126.9269）在写满前已完整落盘，产物未损坏。
- **用户新指令（2026-10-24）**：后续所有权重放 `/data1/huyang`（`/dev/sda`，3.6T、余 882 GiB）。
- **迁移**：`runs/m1_E1_base`（28 GiB）与 `runs/m1_E1_base_4gpu_aborted_20261022`（14 GiB）已复制到 `/data1/huyang/text2video/runs/`；逐文件 `find` 列表+大小比对完全一致（1747/64 个文件），`final.pt` 经软链重新 `torch.load` 校验 `step=30000` 通过。随后删除根盘原目录并在仓库 `runs/` 下保留同名**软链**，根盘可用空间 140 GiB。
- **配置**：`configs/milestone1/{base,E1..E10}.yaml` 的 `training.out_dir` 全部改为 `/data1/huyang/text2video/runs/m1_*`；README 同步。登记 TODO **D66**。测试仍 **91 passed**。
- **复现评测重启**：删除 09:50 的残缺输出目录，第二次评测重新启动（同 config/ckpt/seed、raw、1 GPU、输出 `reports/eval/m1_E1_base_repeat`），2026-10-24 11:06 完成。
- **复现结果**：第二次 **FVD = 126.92693857176755**、`real_split_half_fvd = 84.29751411436246`、`mean_feature_l2 = 32.32046025294281`，与第一次**逐位相同**；两次 `ids.txt` SHA256 均为 `6f96ad9d…10f6`，`protocol.json` 的权重口径/采样/契约一致。**M1.5 判据通过**（非 NaN、不低于真值本底、可复现两次）。

## E2（base-2b）实验启动（2026-10-24 11:54）
- 用户指令：继续跑 **E2**。E2 = E1 唯一改动 factor2 `2a`→`2b`：28 层 cross-attn 全部注入 `[ref;text]`；其余配方与 E1 完全一致（1a、4c、3b、5b、batch12、lr1e-4/text lr1e-5、seed42、30k、raw、DDIM50/CFG2）。
- 资源：GPU2/3 被其他人的作业占用（各 ~15 GiB、~50% util），本次使用**空闲的 GPU0/1（2 卡，符合 ≤3 卡约束）**：`CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m t2v.train --config configs/milestone1/E2.yaml`；micro1×accum6×world2 = **effective batch12**（与 E1 等价）；launcher PID 3692697、rank0 PID 3692722。
- 输出：`/data1/huyang/text2video/runs/m1_E2/`（用户重申所有权重放 `/data1/huyang`；arm 配置 `training.out_dir` 已是绝对路径）。实测 step1 loss 0.90315、step10 0.86243、step20 0.70056、~4.1 s/步、峰值 16.64 GiB、参数 821.92M；预计 30k ≈ 34h（约 2026-10-25 22:00 前后）。
- **自动评测链**：`/tmp/m1_E2_chain.sh`（nohup，当前 PID 3719846）轮询 `status.json`，训练 `complete` 后自动挑一张空闲 GPU、按 first24/test640 协议评测并写 `reports/eval/m1_E2/`；失败会写 `/tmp/m1_E2_chain.log`。评测入口已固定为 SignViP 原样实现（D70/D71），不再传 `--fvd-tool`。结果登记 `EXPERIMENT.md` **exp-108**（已预登记为进行中）。

## FVD 入口统一为 SignViP 原样实现 + E1 正式重测（2026-10-24 15:00）
- **用户指令**：以后所有 FVD 测试必须使用 SignViP 的官方脚本。规则已写入 `RULES.md`（方法约束 5 / 评测 6.1 / 探索 19）、`AGENTS.md`（方法约束 5）、`FVD_SPEC` §0.0、`STRUCTURE.md`；决策 TODO **D70/D71**。
- **canonical 实现**：`third_party/signvip/fvd.py`（SignViP 仓库 `metrics/fvd/styleganv/fvd.py` 的逐字节副本，SHA256 `c9a1ac…da37`）+ `tools/eval_signvip_fvd.py`（只做数据 IO / first24 人口 / 统一 256² 源几何，预处理/特征/距离全部调用原函数）；`tools/eval_milestone.py` 固定调用该入口，旧 `tools/eval_fvd.py` CLI 委托同入口（旧 helpers 仅供历史测试）；I3D 权重 `bec6519f…0a5e2` 每次校验。并行开发中的本地草稿重复实现（`tools/eval_fvd_signvip.py`、`third_party/signvip_fvd/`）已删除，仓库只保留一个正式 FVD 实现（提交 `3f0afd4`，95 tests passed）。
- **E1 正式重测**（同一批已生成帧，不重新生成）：first24/N=640、raw、统一 256² 源几何 → SignViP 原函数 224² 预处理下 **FVD = 129.7222078577488**，真值半分本底 **84.87336765689959**，mean_feature_l2 32.441±6.552。旧本地实现同帧为 126.92693857176755（历史值，不用于新对比；两者差约 +2.2%，来源是无抗锯齿预处理 + PIL 抗锯齿路径差异）。生成侧两次评测的 15,360 张 PNG 逐字节相同，分数确定可复现；两个独立封装入口在同一输入上给出完全相同的 129.7222078577488，作为入口正确性交叉校验。
- **产物**：`reports/eval/m1_E1_base_signvip/{fvd.json,fvd.npz}`（正式）；`reports/eval/m1_E1_base{,_repeat}/` 旧报告保留为历史；台账 `EXPERIMENT.md` exp-099 已回填正式值、exp-108 为 E2。

## E2 训练与正式评测完成（2026-10-25）
- 训练：2026-10-24 11:54 → 10-25 23:14，**127,248.8 s = 35 h 21 m**，step 30,000 正常结束；`final.pt`/`last.pt` 各 13.68 GiB、30 组视觉回调；末 3k 步 loss 0.0171、峰值显存 16.64 GiB。
- 评测：自动链在训练 `complete` 后于 23:14:48 选择空闲 GPU0、用 **canonical SignViP 原样入口**（`tools/eval_signvip_fvd.py`）跑 first24/test640（raw、DDIM50/CFG2、GT 长度、seed42、batch8/GPU），23:46 完成。**FVD = 142.32016432887428**（N=640，`skipped_short=[]`，真值半分 84.89697227815674）。
- 对照：E1 用同一入口、同一 batch/设备重算为 **129.74927595301466**（floor 与 E2 逐位一致 84.89697227815674），故 **E2(2b) 比 E1(2a) 差 +12.5709（+9.69%）**。E2 训练用 2 卡、E1 用 3 卡（同 effective batch12、同为 3090），单一 seed，结论待 M1.6 其它臂复核；同入口下 E7(1b)=126.8475（混卡训练）更低。
- 产物：`/data1/huyang/text2video/runs/m1_E2/`；`reports/eval/m1_E2/{fvd.json,fvd.npz,protocol.json}`；台账 exp-108、NOTES 2026-10-25。

## E4（base-3a，加 motion loss）启动（2026-10-25 23:53）
- 用户指令：继续跑 **E4**。E4 = E1 唯一改动 factor3 `3b`→`3a`：`MultiScaleDeltaLoss`（scales 1/2/4、weight 0.1、warmup 2000，逐视频裁剪 padding）作用在 pred_x0 与 GT latent；其余同 E1（1a/4c/2a/5b、batch12、lr、seed42、30k、DDIM50/CFG2、raw）。
- 资源：GPU0/1/2（3×RTX3090，GPU3 空闲，符合 ≤3 卡约束）；`CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 -m t2v.train --config configs/milestone1/E4.yaml`；world3×accum4=batch12；launcher PID 3973858、rank0 PID 3973883。
- 实测：参数 821.92M；step1 loss 0.90317 / mse 0.90315 / **motion 0.3442**；step10 loss 0.85909 / motion 0.3686；step20 loss 0.69610 / motion 0.2896；~3.75 s/步、峰值 16.59 GiB、无 NaN。motion 项按 warmup 线性加权（前 2000 步从 0→0.1），因此早期 loss-mse 差很小属预期。
- 输出：`/data1/huyang/text2video/runs/m1_E4/`；自动评测链 `/tmp/m1_E4_chain.sh`（等 `complete` → 空闲 GPU → canonical SignViP 入口，输出 `reports/eval/m1_E4`）；台账 `EXPERIMENT.md` exp-110。预计 30k ≈ 31h（约 2026-10-27 07:00）。

## E4 训练与正式评测完成（2026-10-27）
- 训练：2026-10-25 23:53 → 10-27 05:29，**106,535.8 s = 29 h 36 m**，step 30,000 正常结束；`final.pt`/`last.pt` 各 13.68 GiB、30 组视觉回调；末段 mse 0.0201 + 0.1×motion 0.0195 = loss 0.0220、grad 0.056、峰值 16.59 GiB。
- 评测：自动链在 `complete` 后于 05:29:25 选 GPU0、用 canonical SignViP 入口跑 first24/test640（raw、DDIM50/CFG2、GT 长度、seed42、batch8/GPU0），06:00:30 完成。**FVD = 130.84863100287862**（N=640，`skipped_short=[]`，真值半分 84.89697227815674，与 E1/E2 逐位一致）。
- 对照：E1 129.7493（Δ **+1.0994，+0.85%**，基本持平/略差）；E2 142.3202；E7 126.8475（混卡）。结论：motion loss 在该配方下无可测 FVD 改善（单 seed，E9 交互臂不可替代）。
- 产物：`/data1/huyang/text2video/runs/m1_E4/`；`reports/eval/m1_E4/{fvd.json,fvd.npz,protocol.json}`；台账 exp-110、NOTES 2026-10-27。

## E6（base-4b，全 12 层文本可训）启动（2026-10-27 10:26）
- 指令消歧：用户先发"开始 E5"、紧接"开始 E6"，经确认按后一条为准：**只跑 E6，E5 暂不启动**。
- E6 = E1 唯一改动 factor4 `4c`→`4b`：文本编码器保留完整 12 层 mT5 encoder（含 embedding）且全部可训，text lr 1e-5；其余同 E1（1a/2a/3b/5b、batch12、lr、seed42、30k、DDIM50/CFG2、raw）。
- 资源：GPU0/1/2（3×3090，GPU3 空闲，符合 ≤3 卡约束）；`CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 -m t2v.train --config configs/milestone1/E6.yaml`；world3×accum4=batch12；launcher PID 25580、rank0 PID 25606。
- 实测：参数 backbone **607,113,088** + cond 1,481,472 + text **277,040,256**（4c 是 213,325,440，多 63.7M）；step1 loss 0.90315、step10 0.85280、step20 0.70819、grad 0.92 有限；~4.1 s/步；峰值显存 **17.93 GiB/24 GiB**（全 12 层文本可训带来的额外权重/梯度/Adam 状态约 +1.3 GiB，仍在预算内）。
- 输出：`/data1/huyang/text2video/runs/m1_E6/`；自动评测链 `/tmp/m1_E6_chain.sh`（等 `complete` → 空闲 GPU → canonical SignViP 入口 → `reports/eval/m1_E6`）；台账 `EXPERIMENT.md` exp-111。预计 30k ≈ 33–34h（约 2026-10-28 19:30）。
- 注意：E5（4a 全 12 层冻结）未跑，M1.6 的"文本深度 vs 可训练性"分离缺一只对照臂；表格与结论须注明。

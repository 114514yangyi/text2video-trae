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

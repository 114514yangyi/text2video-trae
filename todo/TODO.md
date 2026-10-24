# TODO

> 本文件 = **待办队列** + **决策登记**。规则见 `RULES.md` 评测规范第 7 条 / 知识库流转规范第 8–10 条。
> - 待办只列**未完成**条目；完成即从队列移除、产物入 `.trae/documents/FINISH.md`、裁决入 `NOTES.md` / `FAIL.md`。
> - 决策（D-）只增不删；**ID 不复用**，被取代时保留 ID 并标注"已被 Dxx 取代"。
> - 2026-10-16 整理：完成项与决策的展开叙述已压缩为指针，全文在 `NOTES.md` / `FAIL.md` / `EXPERIMENT.md`。
> - **2026-10-19 精简（用户指令）**：配合 `THINK.md` 的 idea 清理（活跃卡 17 → 6）。本文件
>   当前**未完成条目只有 2 条**：**V1**（旗舰收尾与归因）与 **P15**（route_q 证据加固）；
>   其余 `[x]` 条目为按规范第 7 条保留的历史记录（`(X)` 保留、不删）。空的"第三优先"章节已移除。

## 背景

- 项目：text→video（DiT）手语生成；无 pose；冻结 1-token VAE（与 `../RefSign` 共享 latent 缓存）。权威文档：`README.md`。
- 训练：`python t2v/train.py --config <cfg> --gpu cuda:N`（冒烟 `configs/smoke.yaml`）；生成：`python tools/generate.py`。
- 测试：`python tests/run_all.py`。评测：`bash tools/eval_matrix.sh <tag> <cfg> <ckpt> <GPU> 642 test`。

---

## 当前待办队列（按执行顺序）

### 第一优先 · 在飞 / 已排（队列 `reports/queue/jobs_light_20261016.txt`，GPU 见 D33）

> **🟢 2026-10-19 14:21 —— 已启动，队列运行中：`bash tools/run_pending_evals.sh`**
> 把「已备好但从未跑」的 **15 条评测（0 训练）** 一次性排完，含 V1(c)、P27、P28、P29 全部内容：
> flagship v2 消融 5 条 + 免费杠杆叠加 3 条 + 解码容差曲线 7 条。
> 两阶段：先单独跑 `tol_sigma000` 自检档并强制复现 exp-051 重建上限（B1 32.36 / B4 13.58），
> 复现不出即中止（后 6 档容差读数会作废）；过了再放其余 14 条。
> 无卡环境可先 `bash tools/run_pending_evals.sh --dry-run` 预检。
> 日志：`reports/queue/logs_pending_all/driver.log`（分支日志在 `selfcheck/` 与 `main/`）。
>
> **⚠️ 运维事实更正（2026-10-19 14:21，重要）**：此前记录的"agent 沙箱看不见 GPU"**只对默认沙箱模式成立**。
> 默认模式下 bash 工具运行在一个**过滤过的 `/dev`** 里（无 `/dev/nvidia0`/`nvidiactl`、无 `/dev/nvidia-caps`
> 内容、`CapEff=0`），因此 `nvidia-smi`/torch 全部报错——**这是沙箱视图隔离，不是容器没有 GPU**。
> 用 `sandbox_permissions: danger-full-access` 重试同一条命令即可看到全部 8 张卡
> （4× RTX 4090 @0-3 + 4× RTX 3090 @4-7，宿主驱动 535.230.02）。
> **推论**：以后要"真跑"训练/评测，必须带 `danger-full-access` 起命令；只读监控看工作区内的日志文件即可。
>
> 前置条件：**磁盘需 ≥30 GiB**（起跑时 58 GiB，挂起阈值 25 GiB）。

> **2026-10-16 19:25 并行提速（D33）**：主队列最初按旧的保守安排固定在 `--gpus 6 --max-parallel 1`。
> 新规则（D33/RULES 14 条）允许使用任何空闲卡后，实测该时刻 0/4/7 三张卡显存 ≈0、利用率 0，
> 于是加开**伴随队列** `reports/queue/jobs_light_20261016_par.txt`
> （`--gpus 0,4,7 --max-parallel 3`，日志 `reports/queue/logs_par/`），只放主队列尾段的
> 同 N 锚点与 idea-016 收口，让它们不用等前面十几条串行作业。
> 主队列保持原样不动（其 19 条作业已读入内存，且每条各阶段都有"产物存在即跳过"的守卫，
> 重复执行已完成作业几乎是无操作）；**不抢占在跑的卡**（1/2/3/5 有他人作业，始终未使用）。

- [x] **V1 · 旗舰臂的收尾与归因（2026-10-20 关闭：训练 + 4 档评测 + 入台账 + 部分归因均已完成；剩余归因转入 `THINK.md` idea-044）**
  - **(a) 训练已完成**：`runs/flagship_ref_v1/final.pt` 于 2026-10-19 00:00 落盘，60k（DiT-B，
    从零）。组合：ref 条件（D45）+ Expert AdaLN + MTG 帧↔词接地 + soft-DTW 对齐损失（idea-024）
    + 长度加权损失 + 文本微调 5e-5（D18）。配置 `configs/flagship_ref_v1.yaml`。
  - **(b) 已有两档 raw 评测（D46 之前的口径）**：`flagship_ref_v1` test-642 =
    B1 11.467 / B4 **3.040** / ROUGE 10.527 / WER 106.09 / FVD 19.012 / MRR 0.0222（≈2.02× 地板）；
    `flagship_ref_v1_s40k` = B1 9.723 / B4 1.994 / MRR 0.0169（≈1.54×）。
    `_k4` 档已于 2026-10-20 补齐（B1 9.76 / B4 2.76 / ROUGE 9.70 / WER 102.30，raw 口径）。
  - **(c) 待办 1 —— EMA 复评**：两档的 `meta.json` 都是 `use_ema=false`，而 **D46 已把 EMA 定为
    全臂部署/评测默认**。必须重跑两档（+ 补齐 k4 档），否则与 v2 的对比会混进 `raw vs EMA` 这个
    已知 +0.76 的口径差。
    **2026-10-20 裁决：本条已关闭，不是“待跑”而是“结构上不可行”。** 三点实测：
    - k4 档已补齐（`flagship_ref_v1_k4` = B1 9.76 / B4 2.76 / ROUGE 9.70 / WER 102.30，raw），**比 raw k=1 的 11.47/3.04 更差**。
    - **EMA 档跑不通**：`tools/eval_matrix.sh ... OUT_SUFFIX=_ema` 连试 3 次全部 `ValueError:
      fine-tuned text runs require consistent raw weights; EMA bundle unavailable`
      （守卫在 `t2v/pipelines/generation.py:231-233`）。原因是 flagship 臂 `conditioning.text.trainable: true`，
      EMA 影子权重不含文本编码器，混用会口径不一致 → **含文本微调的臂只能以 raw 口径汇报**（登记 **D53**）。
    - s40k 档无法复评：其 `step_040000.pt` 在 2026-10-19 清盘时被删，`runs/flagship_ref_v1/` 现只剩 `final.pt`。
    ⇒ 旗舰臂与 v2 的对比**永久只能 raw vs raw**；v2 也是 raw，口径一致，无需补 EMA。
  - **(d) 已做（2026-10-19 12:45）—— 入台账**：`EXPERIMENT.md` 已补 **exp-086 / exp-087** 两行、
    第 5 节自动区用 `tools/collect_experiment_table.py --inplace` 重生成（三条旗舰行入表）、
    第 6 节加了口径备注（raw 权重 / 属 **D46 之前** / 不得与 EMA 臂混表）。
    **已完成（2026-10-20）**：结论已写入 `NOTES.md` §「2026-10-20」第 3 节（四档 v1 / v1_k4 / v2 / v2_k4 含 FVD 与 MRR）。
  - **(e) 待办 3 —— 归因**：规则见 **THINK `idea-044`**（原 idea-039/040/041 合并）。
    关键已知量：旗舰 60k（B4 3.040）**低于**只用对齐一个机制、只训 1/3 步数的
    `route_t_align_20k`（raw 3.906 / EMA 4.661）——组合臂是净亏的，但归因缺失。
    （2026-10-20 更新：现役最强已到 `_ema_k4` **B4 5.65**，旗舰 60k 只有它的 54%。）
    **（2026-10-20 部分归因已落地）**：exp-093 证明 `as_modulation` 这一**改写本身**有害
    （v2 0.70 vs v1 3.04、v2_k4 0.58 vs v1_k4 2.76，命中 P29 预注册证伪线）；
    其余三机制（Expert AdaLN / MTG 帧↔词接地 / 长度加权损失）**仍无法拆开**，留 `idea-044`。
    另一条已确定的：旗舰臂上 gate k=4 反而更差（2.76 < 3.04），与对齐臂 ΔB4 +0.99 **方向相反**。

- [x] **11 · 部署预算点复核（已跑完）** — 60k/90k/120k × seed 42/43/44 全部落盘。
  三种子均值：60k B1 10.53 / MRR 0.018；90k 11.89 / 0.028；120k 12.56 / 0.032。
  **排序稳定**（120k > 90k > 60k 在 B1/MRR 上一致），D19 得到三 seed 支撑；FVD 种子方差很大
  （60k 17.4–29.7），不宜作排序依据。`EXPERIMENT.md` exp-046。
- [x] **13 · CFG 补扫（已跑完，待归档）** — 中段 t∈[250,600]：B1 9.63 / MRR 0.0149，**明显劣化**；
  仅高噪声 t∈[600,999]：B1 12.46 / MRR 0.0223，与 D17 的 [200,999]（12.72/0.0219）接近。`EXPERIMENT.md` exp-043/044。
- [x] **14 · 同 N 锚点** — **已完成**（2026-10-17 主体；2026-10-18 补齐 rank dump）：
  - 重建上限 `reports/eval/anchor_real_test_n642`：PSNR 26.67 dB、B1 **32.36** / B4 13.58 /
    ROUGE 30.43 / WER 86.67、FVD **14.72**；
  - 错配地板 `reports/eval/anchor_rotated_test_n642`（`+1` 错配真视频）：B1 9.72 / B4 0.67 /
    ROUGE 8.18 / WER 119.46、FVD 0.72。
  - 修掉的 bug：`tools/eval_fvd.py` 在 `--ids-file` 分支漏定义 `generated_root`（`FINISH.md` F-013）。
  - **已完成（2026-10-18 核实）**：重建上限的逐视频 rank dump 已在
    `reports/eval/ranks/anchor_real_test_n642.json`，错配地板的 rank 等价物是
    `reports/eval/retrieval_anchor_floor_test_n642.json` 的 `rolled_latents`（rolled MRR 0.0107
    ≈ 零信息地板 0.0110）。两条锚点线的检索 CI 所需输入齐备，本条结清。
  - 读法：现役生成臂 test 642 的 B1 在 10.5–14.1、MRR 0.016–0.034，距重建上限（B1 32.4 /
    MRR 0.0343）仍很远，但已远离错配地板（B1 9.7）。
- [x] **15 · 长度头直接回归对照（已跑完）** — B1 10.92 / B4 1.88 / ROUGE 9.91 / MRR 0.0165，
  略低于 8 桶 expectation（11.58/1.91/10.57/0.0176）→ 直接回归无优势，建议维持 D14。
  `EXPERIMENT.md` exp-045。

- [x] **P27 · 免费杠杆叠加臂（2026-10-20 已跑完：k=4 单独 B4 5.65 最优；三档结果见 EXPERIMENT.md exp-091）** · 免费杠杆叠加臂：EMA 之上的 k=4 × 区间 CFG（2026-10-19 立项，0 训练，作业文件已备好）**
  - **动机**：现役最强臂 `route_t_align_20k_ema`（EMA · k=1 · CFG2.0）= B1 13.51 / B4 4.66。
    三个"免费杠杆"各自单独测过：区间 CFG3.0[200,999] ΔB1 +1.84、gate k=4 ΔB1 +1.87、EMA ΔB1 +1.33。
    但 **exp-069 / exp-077 已经在 raw 权重上测过 k=4 与 k=4+CFG 的叠加**，结论是「对齐臂自身的头部
    空间已被吃掉 → 叠加饱和（align+gate4 四项回译全 n.s.）、再加 CFG 显著更差（ΔB1 −1.39 显著退化）」。
    **从未测过的那一格是 EMA 之上的叠加**：EMA 是把同一臂推到新工作点的另一个机制，
    它的头部空间是否已被 k=4 预先占用，只能实测 —— 这是本条目存在的唯一理由，
    不是"三个杠杆没同框过"。
  - **做法**：同一 checkpoint（`runs/route_t_align_20k/final.pt`）/ 同一 config / 同一采样清单 /
    同一 seed 42，只改推理条件，三档单变量：`_ema_k4`、`_ema_cfg30i200`、`_ema_k4cfg30i200`；
    对照 = 已存在的 `route_t_align_20k_ema`。三档一起出才能把"EMA 改变了交互"与"抽样噪声"分开。
  - **作业文件**：`reports/queue/jobs_free_levers_ema_20261019.txt`（3 条，`#done=` 已指向 score.json）。
  - **成本**：0 训练，~1.5–2 GPU·h。**判据**：三档各自与对照做配对 bootstrap
    （`tools/paired_arm_stats.py`）；只有"叠加臂显著优于对照"**且**方向与 raw 臂相反，
    才支持"EMA 改变了杠杆交互"。若三档全部 n.s. 或显著更差 → 这条轴（推理期免费杠杆）正式关闭。
  - **⚠️ 起跑前先看磁盘**：`/data` 当前仅剩 **68 GB（100% 已用）**，队列请设 `--min-free-gib 25`。

- [x] **P28 · 解码容差曲线前置探针（2026-10-20：6/7 档完成，自检档完美复现上限；tol_scale055 因 CUDA OOM 失败已重排；见 exp-092）** · 解码容差曲线前置探针（2026-10-19 立项，0 训练，作业文件已备好）**
  - **问题**：冻结 VAE 能容忍多大的 latent 误差？现役臂落在容忍曲线的哪一段？
    这决定了下一步该压"保真度"还是继续动"条件使用方式"（后者已被 9 条机制类改动证伪）。
  - **(a) 已完成的一半（纯 CPU，本轮实测）**：新工具 `tools/probe_latent_error_percentiles.py`
    给出 11 条臂的 `rel_err = ||ẑ−z*||_F / ||z*||_F` 分位数，并拆成方向/幅度两半
    （`cos` / `dir_err` / `amp_ratio`），结果落 `reports/eval/probe_latent_error.json`
    （登记为 EXPERIMENT.md **exp-088**）。关键读数：
    - **臂的 rel_err 全在 0.44–0.51，而 `GT-static`（把该视频时间平均帧复制满整段、零运动）
      只有 0.4147** —— 纯 L2 意义上，生成 latent 比"完全不动"的预测器离真值还远；
    - 但臂的 `cos` 高达 **0.88–0.91**，说明**方向大体正确**，误差主要是**欠弥散**
      （`amp_ratio` 0.69–0.87；R1 对齐臂把 0.686→0.767、EMA 再推到 0.805）；
    - `route_t_align_20k_ema` 是全场最保真（rel_err 0.4408 / cos 0.9134），
      与它 B4 最高（4.66）方向一致。
  - **(b) 待跑的一半（GPU）**：给 `tools/eval_vae_recon.py` 新增 `--noise-sigma` / `--latent-scale`，
    在**同一解码路径**（同冻结 VAE、同 GT frame-0 锚点）上注入两类量级已知的扰动再回译：
    `z* + σ·rms(z*)·ε`（各向同性噪声，σ ∈ 0.15/0.30/0.45，其中 0.45 正落在现役臂误差水平）与
    `s·z*`（纯幅度缩放、方向完美，s ∈ 0.85/0.70/0.55 → 同一刻度 0.15/0.30/0.45）。
    两族在相同 rel_err 下的差 = "方向错"与"只是幅度缩了"各自的回译代价。
  - **自检档**：`tol_sigma000` 必须复现 exp-051 的冻结 VAE 重建上限
    （B1 32.36 / B4 13.58 / ROUGE 30.43 / WER 86.67，N=642）；**复现不出则后面 6 档读数全部作废**。
  - **作业文件**：`reports/queue/jobs_tolerance_curve_20261019.txt`（7 条）。链路复用
    `tools/eval_matrix.sh`：`eval_vae_recon.py` 先把 `meta.json` 写进 `reports/eval/matrix/<tag>/test_n642/`，
    eval_matrix 的"生成阶段"守卫（meta.json 存在即跳过）随即跳过采样，直接跑 labels→SignMbart→FVD→检索→删帧，
    因此口径与所有历史臂逐项一致，**零新增评测代码**。
  - **成本**：0 训练，~1.5–2 GPU·h（每档峰值磁盘 ~4 GB，跑完自动删帧）。
  - **判据**：曲线给出"B4 掉到重建上限 50% 时的 rel_err"记作 `r*`。
    若 `r* ≫ 0.44`（现役臂误差）→ 臂落在平坦段，**瓶颈不在保真度**，评测/之前的诊断成立；
    若 `r* ≪ 0.44` → 臂早已越过拐点，**保真度就是墙**，训练目标必须直接压 latent 误差
    （这条判据同时决定 idea-042 DVO 与"幅度修正类"方向的优先级）。

- [x] **P29 · flagship_ref_v2 的评测（2026-10-20 已跑完：B4 0.70 / k4 0.58；⚠️预算 20k vs v1 60k 不可比，见 exp-093）** · flagship_ref_v2 的评测（2026-10-19 立项，0 训练，作业文件已备好）**
  - **问题**：`runs/flagship_ref_v2` 已于 13:22 训完（20000 步 / 10606 s / final.pt 6.93 GB），
    但 **`reports/eval/matrix/` 下没有任何 v2 目录，评测一行都没跑过** ——
    `reports/queue/jobs_flagship_ref_v2_20261019.txt` 里当时只写了训练，没写评测。
  - **要回答什么**：v1 与 v2 的**唯一架构差异** = `conditioning.reference.as_modulation`
    （v1 false：参考帧 128 token 拼进 cross-attention context；v2 true：参考帧不进 context，
    参考池化只加到视频 token 的调制基底）。两种接线的**参数名完全相同**，
    checkpoint 错配时 `load_state_dict` 不报错、只会静默丢参考帧，所以必须单变量对照。
  - **作业文件**：`reports/queue/jobs_flagship_v2_eval_20261019.txt`（5 条，协议 P / test 642）：
    ① v2 raw（对齐 exp-086 的 v1 raw 3.040）② v2 EMA（D46 部署默认）
    ③ v1 EMA（补 V1(c)，与 ② 配对才能同口径比 v1/v2）④ v1 gate k=4（补档，目录一直为空）
    ⑤ v2 gate k=4（与 ④ 配对）。每档独立 `OUT_SUFFIX`，`#done=` 守真实产物 `score.json`。
  - **成本**：0 训练，~3.3 GPU·h（`--max-parallel 2`）。**磁盘仍需 `--min-free-gib 25`**。
  - **判据**：v2 两档都 ≤ v1 → `as_modulation` 这一改写被证伪，旗舰线上 v2 出池；
    v2 EMA 若超过**现役最强** `route_t_align_20k_ema_k4`（B4 **5.65**）→ 旗舰组合臂**首次超过单机制臂**，
    才值得继续在旗舰线上加机制。
  - **⚠️ 起跑前先修的两件事（本轮已修其一）**：
    1. ✅ `tools/eval_matrix.sh` 的 7 个阶段日志原先只用 `${TAG}_${SPLIT}` 命名、**不含 OUT_SUFFIX**，
       同 tag 不同档会互相覆盖失败现场（v1 的 k=4 档就是这样把失败现场弄丢的）—— 已修（commit `d411029`）。
    2. ⛔ **不要再用 `tools/queue_flagship_eval.sh`**：它只有 `set -uo pipefail`（缺 `-e`），
       子命令失败会被吞掉、日志照样打 `[guard] ALL DONE` → v1 的 k=4 档静默漏跑成一个空目录。

### 第零优先 · 【待用户拍板】Wan2.1-1.3B 骨干迁移（idea-045，2026-10-19 新立）

> 来源：2026-10-19 用户指定"使用 Wan 模型"方向。方案全文 `.trae/documents/WAN_BACKBONE_PLAN.md`。
> **这 8 条全部未开始**（未下载任何权重、未改任何既有代码、未建 config）。
> **卡口 = WAN_BACKBONE_PLAN.md §10 的五个开放问题**（Q1 x0/velocity 是否放宽 RULES 第 3 条、
> Q2 训练环境、Q3 是否授权清 runs/、Q4 LoRA vs 全参、Q5 首次投入上限）。**Q1 不拍板不写模型代码。**
> 顺序原则：**先做能在 2 GPU·h 内杀死整案的证伪实验（W-N7 的 Stage 2），再花大钱。**

- [x] **W-N1 · 【已完成 2026-10-19】腾磁盘（硬前置）**：清 `runs/` 里已出结论且不再被引用的臂 腾磁盘（硬前置）**：清 `runs/` 里已出结论且不再被引用的臂，目标 **≥30 GB 净空**
  （方案 §7 估需 ≥30 GB；起跑时 53 GB，但当前 15 条评测队列仍在写入）。
  **等队列跑完再做**，且只清已出结论的臂——清单先给用户过目。风险：删到正在读的 checkpoint。
- [x] **W-N2 · 【已完成 2026-10-19】装依赖** 装依赖**：`diffusers`（≥0.33，含 `WanTransformer3DModel`）+ `peft`。
  环境见 Q2（建议 `onvae`：torch 2.6.0+cu124 / transformers 5.17.0 / 评测同源；`gfslt` 的
  transformers 4.32 太旧、无 `UMT5EncoderModel`）。PyPI 实测可用（diffusers ≤0.41.0）。
- [x] **W-N3 · 【已完成 2026-10-19】下载权重** 下载权重**：Wan2.1-T2V-1.3B 的 **DiT（5.676 GB）先下**，umT5-xxl（11.362 GB）后下；
  **不下 VAE（0.508 GB，我们不用）**。来源：HF 直连**不可达**；`hf-mirror.com` 实测 HTTP 200、
  ModelScope API 实测可用。落到 `/data/huyang/models/Wan2.1-T2V-1.3B/`，sha 校验。
- [x] **W-N4 · 【已完成 2026-10-19】读源码钉死三处** 读源码钉死三处（G4，不完成不许写模型代码）**：① `WanRotaryPosEmbed` 把
  `head_dim=128` 切成 t/h/w 三段的**确切维数**；② `WanTransformerBlock` 的 modulation 通路
  （6 路 shift/scale/gate 的具体拆分）；③ patchify 的 reshape 顺序。
  产出 `.trae/documents/WAN_ARCH_NOTES.md`，每处写清"文件:行 + 原文片段"，**禁止凭记忆**。
- [x] **W-N5 · 【作废】权重转换**：直接用了 diffusers 官方格式，不需要转换 权重转换**：`tools/convert_wan_weights.py`，官方 safetensors → diffusers 命名 →
  **bf16 落盘**（≈2.84 GB）。判据：`load_state_dict` 无 missing / unexpected。
- [x] **W-N6 · 【已完成 2026-10-19】文本嵌入缓存** 文本嵌入缓存**：`tools/cache_umt5_embeddings.py`，冻结 umT5-xxl、**文本长度截到 64**
  （不是 Wan 默认的 512；我们德语字幕 ~14 词 → ~25–40 token），存 bf16 + 真实长度。
  8257 条 × 64 × 4096 × 2B ≈ **4.3 GB** → `data/text_emb_umt5/v1/`。
  ⚠️ umT5 的 pad token **有真实 embedding**；推荐"缓存真实长度 + 训练时传 `context_mask`"
  （复用现有 `DiTConditioning.context_mask`），不要靠补零。
- [x] **W-N7 · 【被 2026-10-19 用户第十轮拍板推翻】模型重接 + Stage 2** 【关键证伪】模型重接 + Stage 2**：`t2v/models/wan_backbone.py`（**与
  `DiT.forward(x, t, cond, frame_mask, self_cond)` 同签名**，`factory.build_bundle` 只加分派分支，
  config 加 `model.backbone: dit|wan`，默认 `dit` 保证老配置一字不变）+ latent 固定标准化
  （用 `train_learn...`/`train_stats.safetensors`，仿射变换，不改冻结 VAE 契约）+ `tests/test_wan_backbone.py`
  （前向形状 / 变长 T 不变性 / `frame_mask` 不变量 / 参数量 1419M±1% / 新 I/O 层梯度非零）。
  **然后立刻跑 Stage 2**：`wan-io`（Wan 块**冻结**、只训新 I/O ≈3.1M）vs `wan-io-rand`
  （**等容量随机初始化** 30 层、同样只训 I/O），各 2k 步、bs 16、同 seed、同数据顺序。
  **预注册判据**：① `wan-io` 验证 x0-MSE 显著低于对照（按帧配对，**p<0.01**）；② 相对优势 **≥5%**。
  **两条任一不满足 ⇒ 整条 Wan 路线关闭并写 FAIL.md**（这是全案唯一能在 2 GPU·h 内杀死它的实验）。
- [x] **W-N8 · 【已被 exp-090 取代】正式臂** 正式臂（仅在 W-N7 过关后）**：`configs/wan_b_1280_20k.yaml` —— 20k 步、bs 16、
  lr 新层 1e-4 / LoRA(r64) 1e-5、bf16、grad_clip 1.0、EMA 0.9999；**W-x0 口径**（保留我们自己的
  GaussianDiffusion，只换主干 ⇒ 采样器 DPM++ 50 不变 ⇒ 与历史臂**完全可比**）。评测出 raw + EMA 两档，
  与 `route_t_align_20k_ema` 按 video_id 配对。判据：**ΔB4>0 且 p<0.05，ΔWER 不显著退化（D20）；
  MRR 不作放行条件**。**强制附带三项诊断**（内容词召回 / 文本交换 gap / cross-attn 贡献占比）——
  若 B4 涨而召回没涨，必须解释清楚来路再进 Stage 4。另需 **≥3 个生成 seed**（TODO 11 纪律）。

- [x] **W-N9 · Wan `motion_v2` 评测与裁决（2026-10-20 已跑完并裁决）**
  > **结果（2026-10-20 21:5x，EMA 口味，test 642）**：B1 9.16 / B4 **1.13** / ROUGE 9.12 /
  > **WER 120.62** / **FVD 66.92** / MRR 0.0260（2.37× 地板）；内容词召回 **3.0%**。
  > **裁决 = 失败**：四项回译全项差于 `route_t_align_20k_ema_k4`（15.92/5.65/15.42/98.68），
  > WER 反而高于错配真视频地板 119.46，FVD 是重建上限的 4.6 倍。`_raw` 档目录存在但**未跑**
  > （无 meta.json），不影响裁决（EMA 已是该臂的较强设定）。已登记 **exp-094**，
  > 第 5 节自动区已重生成（107 行）。⚠️ **MRR 0.0260 = 2.37× 地板，会通过 D3 闸门，
  > 而 B4 贴在 1.13** —— 再一次证明 MRR 闸门在头部区间无效。
  > **待用户确认**：是否关闭整条 Wan 路线（写 `FAIL.md` + THINK `idea-045` 出池）。
  > 注：W-N9 预注册的判死线是「B4 ≤ 1.0」，实测 1.13 **未严格触发**，故按规矩不自行关闭。
  - **臂全名** `wan_flow_mlp4096_motion_v2`，训练产物 `runs/wan_flow_mlp4096_motion_v2/`，
    评测目录 `reports/eval/matrix/wan_flow_mlp4096_motion_v2/test_n642/`。
  - **架构**：Wan2.1-1.3B 的 30 个 block **全冻结** + LoRA(r64) 挂
    `to_q/to_k/to_v/to_out.0/ffn.net.0.proj/ffn.net.2` + MLP-4096 adapter +
    **训练过的跨 VAE 桥** `runs/latent_bridge_v1/bridge.pt`（把冻结 VAE 的 1024 维映射到
    Wan 的 16 维）+ motion loss（λ 0.1 / warmup 2000）。flow-matching，目标 `ε − x₀`，
    20k 步、bs 16、bf16、seed 42。
  - **训练曲线（2026-10-20 18:46 完成，20925 s）**：`train/loss` 1.02997、`dev/mse` **1.02159**、
    `baseline_zero_pred` 2.1936 → **只解释约 53% 方差**（对比 `wan_flow_lora_v1` 的
    dev/mse 0.3493 ≈ 82%，但两者目标空间不同，不可直比）。`motion_ratio` 0.0727
    （motion 项约占总 loss 7.3%）、`gt_delta_rms` 0.5264。
  - **桥的质量**：`val_mse 4.11644`、`val_rmse 2.029`、`within_chunk_cos_pred **0.9968**`
    对随机配对 `cos_gap 0.4226` → 区分度很大，桥本身确实学到了东西
    （不像 `wan_flow_lora_v1` 那次的零初始化线性探针，cos 仅 0.586）。
  - **评测进度**：18:47 自动接上，GPU 7，0.07 视频/秒；20:04 已到 **316/642**；
    采样预计 ~21:20 完成，之后 labels→SignMbart→FVD→检索 约 1 h，**预计 22:20 前后出最终数字**。
  - **裁决口径（预注册）**：
    - 主对照 `route_t_align_20k_ema_k4`（B1 15.92 / B4 **5.65** / ROUGE 15.42 / WER 98.68，历史最强）；
    - 上限 `anchor_real_test_n642`（32.36 / **13.58** / 30.43 / 86.67 / FVD 14.72）；
    - 地板 `anchor_rotated_test_n642`（B4 0.67）；
    - 内部参照 `wan_flow_lora_v1`（8.01 / **0.81** / 7.74 / 103.19 / FVD 44.51）。
    - **判据**：B4 是否 **> 1.0**（越过上代 Wan 臂且明显脱离地板）；FVD 是否 **< 44.51**；
      并强制附两项诊断——**内容词召回**（现役最强臂仅 5.7%，检索式上限 55.4%）与
      **文本交换 gap**。
  - **裁决分支**：
    - B4 ≥ 2.0 且 FVD 改善 ⇒ 有信号，继续投（下一步加参考帧条件、补 R1 软-DTW）；
    - B4 在 1.0–2.0 ⇒ 边缘，需与同预算 DiT 臂对照才能判；
    - **B4 ≤ 1.0 ⇒ 写 `FAIL.md` 关闭整条 Wan 路线**，理由写成机制级
      （"冻结的视频扩散先验在 1-token/帧 的抽象 latent 上没有可迁移的内容信号"），
      并同步登记 `EXPERIMENT.md`。
  - **台账要求**：出数后立刻补 `EXPERIMENT.md` **exp-094** + 重生成第 5 节自动区；
    若判死，`THINK.md` 的 **idea-045 出池**并写 `FAIL.md`。

### 第二优先 · 探针/收尾（已排队但排在 N6/N7 之后）

- [x] `prefix60k`/`route_b_semantic_5k`/`route_c_inverse_5k` 的 test 复评与 rank dump —— **已完成**
  （2026-10-17）：三臂 test 642 数字与 `reports/eval/ranks/*_test_n642.json` 均已落盘；
  route_b 转正常实验（`NOTES.md` 2026-10-17 节），route_c 出池（`FAIL-030`）。
- [x] **N8 · route_b 同预算单变量对照臂 `route_a_control_5k` 的 test 642 评测** —— **已完成**
  （2026-10-17，纯推理）。结果：B1 12.21 / B4 3.37 / ROUGE 11.27 / WER 104.17 / FVD 19.80 /
  MRR 0.0209。**它改写了 route_b 的结论**：
  - route_b vs 本臂（同起点、同 5k、同文本 lr，唯一差别是有无语义损失）：ΔB1 **+1.92 显著**、
    ΔB4 −0.12 n.s.、ΔROUGE **+1.29 显著**、**ΔWER −2.42 显著退化**、ΔMRR **+0.0126 显著**。
  - 即语义损失是**用显著 WER 代价换 B1/ROUGE/MRR**；按 **D20** route_b **不晋级**，
    状态 = 待用户裁决取舍（不是 FAIL，它有显著正向项）。
  - 副产品：`route_a_control_5k`（只有文本微调续训、无语义损失）vs prefix60k 是
    ΔB1 +1.32 显著 / ΔB4 +1.22 显著 / ΔROUGE +0.94 n.s. / ΔWER +2.11 n.s. / ΔMRR +0.0044 显著
    → "文本微调续训"本身贡献了大部分增益。证据 `EXPERIMENT.md` exp-053、
    `reports/eval/paired/route_b_semantic_5k_vs_route_a_control_5k_*.log`。
- [x] **N9 · route_b 语义损失取舍 —— 2026-10-17 用户裁决：采纳 λ=0.02**（λ 消融已完成，2026-10-17 21:1x）
  选项③（调低 λ）已跑完，λ≈0.01 出现明确甜点，**推荐采纳 λ=0.01**：
  - 三条 λ 臂 test 642 全指标（同起点 route_a_mt5_5k、同 5k、同文本微调，只差语义权）：
    | 臂 | λ | B1 | B4 | ROUGE | WER | MRR |
    | --- | --- | --- | --- | --- | --- | --- |
    | `route_a_control_5k`（无语义损失） | 0 | 12.21 | 3.37 | 11.27 | **104.17** | 0.0209 |
    | `route_b_lambda005_5k` | 0.005 | 11.37 | 1.88 | 9.26 | 108.69 | 0.0199 |
    | `route_b_lambda010_5k` | 0.01 | 13.28 | 3.61 | 11.77 | **104.30** | 0.0312 |
    | `route_b_lambda020_5k` | 0.02 | 14.30 | 3.41 | 12.25 | 106.15 | 0.0333 |
    | `route_b_semantic_5k` | 0.05 | 14.12 | 3.24 | 12.55 | 106.60 | 0.0336 |
  - **配对判读**（vs 无语义损失对照）：λ=0.005 **全面退化**（B4/ROUGE/WER 三项显著亏）→ 不是减弱而是破坏了训练；
    λ=0.01 **ΔB1 +1.06 显著、ΔMRR +0.0103 显著，ΔB4/ΔROUGE/ΔWER 全 n.s.**；
    λ=0.02/0.05 **ΔWER 显著退化（−1.99/−2.42）** 换更大的 B1/MRR。
  - **结论**：WER 代价随 λ 单调（0.01 处消失、0.02 起显著），检索增益在 λ≥0.01 就到位，
    故 **λ=0.01 是保住 MRR 显著增益、去掉 WER 代价的甜点**；按 **D20**（MRR 显著正 + 回译无显著退化）
    是本项目首个满足晋级口径的语义臂。
  - **用户裁决（2026-10-17）**：采纳 **λ=0.02** 为语义目标默认权重 —— 更看重 B1/MRR 的绝对数值
    （B1 14.30 / MRR 0.0333，3.04× 地板，均为全表最高），接受显著 WER 代价（−1.99，WER 106.15）。
    λ=0.01 作为"零 WER 代价"备选保留（其 WER 104.30 与无语义损失对照 104.17 持平），未采纳但未否决。
    登记为决策 **D35**；旧 λ=0.05 臂 `route_b_semantic_5k` 被 λ=0.02 取代为语义目标默认配置。
  - 证据：`EXPERIMENT.md` exp-060/061/062、`reports/eval/paired/route_b_lambda*_vs_*_test.log`、`reports/eval/matrix/route_b_lambda*/test_n642`。
- [x] **P1 · idea-019 前置：cross-attention 是否有时序结构** —— **2026-10-17 已跑完**，
  入口 `tools/probe_cross_attn_temporal_structure.py`（新写，已登记 STRUCTURE.md）。
  - **prefix60k（部署点）→ verdict fail**：12/12 个 block 的平均两两余弦相似度都 > 0.95
    （整体均值 **0.9887**，范围 0.963–0.998；JS 散度仅 0.005 bits）→ 按预注册判据
    **前置不成立，不该开 idea-019 的训练臂**。
  - **稳健性复核**：t=100（0.9924）与 t=800（0.9757）同样接近"各帧看文本方式几乎一样"；
    只有 t=800 勉强落到 0.95 线下，不足以翻案。
  - **一个额外发现（需记入 NOTES）**：最强的 `prefix_120k` 权重上 cos 均值 **0.9370**
    （范围 0.817–0.998），中间层 1/3/4 降到 0.82–0.90，而末层 9–11 仍 >0.99。
    即"注意力近乎无时序结构"这个现象**随训练预算增长会部分缓解，且集中在中间层**。
    这不改变 idea-019 的否决（部署点 60k 上确实没有结构可对齐），但给出了一个可追踪的
    "条件利用率随预算上升"的观测，值不值得单独立卡由用户定。
- [x] **P2 · idea-027 前置探针 A：离散码是否有内容** —— **2026-10-17 已跑完**，
  入口 `tools/probe_discrete_code_content.py`（新写，纯 CPU）。
  - 做法：对 K∈{4,8,16} 帧窗做均值池化 + EMA 码本（256 码）量化，先量码本健康度，
    再用**置换零假设去偏**的互信息比较"码序列↔句子"与"长度桶↔句子"。
  - **结果（400 条 train）→ verdict FAIL**：去偏后码的**超长信息几乎为零**
    （K=4 −0.005、K=8 +0.026、K=16 +0.007 bits，判据线 +0.05）——即码序列基本
    只复述了长度；同时码本严重坍缩（256 码只激活 4–7%、perplexity 仅 2.7–3.9）。
  - **关键坑（记录下来）**：朴素经验互信息有严重小样本上偏——不去偏时"码↔文本"
    看着有 +0.86~+2.99 bits，置换后全部归零。**以后凡用 MI 必须先做置换去偏。**
  - **限制**：这否的是**当前这个朴素实现**（均值池化窗 + 单层 EMA 码本），不是
    "离散语义 token"整个方向；若换更强的量化器（如论文的双辅助监督 + 更大数据）仍可重测，
    但按卡上判据，**当前形态下 idea-027 的码不是内容载体 → 不排训练臂**。
- [x] **P3 · idea-027 前置探针 B：文本→码可预测性 —— 已随前置 A 否决而作废（2026-10-17）**：
  前置 A（P2）已证明码序列基本只承载长度信息，且码本严重坍缩。按卡上"两个探针任一失败即关闭"
  的规则，**不必再跑探针 B**（即便规划器能预测这串码，也只是把长度头重写一遍）。
- [x] **P4 · idea-020 前置：奖励可利用性 —— 已由现有证据结清（2026-10-17）**：
  `reports/eval/teacher_reliability_prefix60k_test.json`（642 条生成 latent）显示现有冻结教师
  **prefers_correct = true**（正确文本 NLL 显著更低，paired gap 0.193 [0.053, 0.328]）、
  **rejects_static = true**（静止惩罚 +0.50~0.64），但 **rejects_reverse = false**
  （倒序 −0.0003~+0.003）、**rejects_block_permute = false**。
  → 按 idea-020 卡上的预注册判据（"若奖励上升的样本包含更多倒放/块置换这类时序退化，
  则判定该奖励不可直接用于后训练，关闭路线 (a)"），**奖励式后训练路线 (a) 的前置不成立**，
  不排训练。偏好式路线 (b) 不依赖该奖励，仍需用户裁决后另做前置（可记为新条目）。
  结论与 **D24** 一致，不新增冲突记录。
- [x] **P5 · idea-005 前置：教师可靠性两关 —— 主体已跑（2026-10-17 结清）**：
  `reports/eval/teacher_reliability_prefix60k_test.json`：prefers_correct=true、
  rejects_static=true、**rejects_reverse=false、rejects_block_permute=false**。
  → 现有教师**可作语义奖励、不可作顺序奖励**（= D24）。idea-005 若要推进，必须**先换/校准教师**
  并重跑同一脚本；在那之前不排训练臂。
- [x] **P6 · idea-024 前置：受限时间对齐损失的性质检查** —— **2026-10-17 已跑完**，
  入口 `tools/probe_alignment_loss_properties.py`（新写，纯 CPU、不占 GPU；已登记 STRUCTURE.md）。
  - 构造 correct / 轻微变速（×1.15、×0.85）/ 倒序 / 静止 五类，用可 warp 但不可 cross 的
    对称 DTW 残差（按"远距离帧对"的典型距离归一成 `dtw_norm`，correct=0）打分。
  - **结果（200 条 train，≤64 帧）→ verdict PASS**：
    轻微变速 **0.077 / 0.082**（宽容，远低于 0.10 线）、倒序 **1.436**、静止 **13.69**
    （都远高于 0.50 线）。即"对齐损失宽容合理节奏偏移、惩罚倒序/静止"这个前提**成立**。
  - **注意**：这是损失**性质**检查，不是效果证据；PASS 只说明 idea-024 的损失形状值得
    进一个小规模等预算训练臂，**不晋级**。是否开臂仍待用户裁决（D27/D31）。
- [x] **P7 · idea-026 前置：各噪声阶段文本干预响应** —— **2026-10-17 已跑完**，
  入口 `tools/probe_text_response_by_timestep.py`（新写，纯推理；已登记 STRUCTURE.md）。
  - 做法：同一 `x_t` 下用正确文本与邻居文本各跑一次去噪前向，量逐视频相对输出变化
    `rel(t)=RMS(Δ)/RMS(out)`，扫 t∈{100,…,900}×3 noise seed。
  - **结果（prefix60k·test 8 条×60 帧）→ verdict PASS**：rel 随噪声阶段**单调上升**
    0.127(t=100)→0.229(t=900)，max/min **1.80×**，两端 bootstrap CI 不重叠。
    即句子条件在不同噪声阶段被利用的程度确实不同，阶段相关文本连接器有可利用结构。
  - **注意**：这是设计依据，不是效果证据；卡片预注册的正式对照是**等参数静态 vs 动态**，
    是否开臂仍待用户裁决（D27/D31 下不自动开训练）。

**B. 已有半成品、只需补最后一步**

- [x] **P8 · idea-010 前置：文本→粗时间计划可预测性** —— **2026-10-17 已跑完**，
  入口 `tools/probe_text_to_coarse_plan.py`（新写，只编码+ridge；已登记 STRUCTURE.md）。
  - 做法：K=4 块均值当粗时间计划目标，文本池化嵌入 vs 长度标量各拟合 ridge，评 held-out R²，
    看文本是否提供长度先验之外的计划信息（文本 PCA 64 / 计划 PCA 32，避免高维目标过拟合）。
  - **结果（train 1500 拟合 / 400 held-out）→ verdict FAIL**：r2(text→plan) **0.021** vs
    r2(length→plan) **0.008**，差仅 **+0.013**（判据线 +0.05）；打乱文本 −0.035。
    即"粗时间计划"基本只反映长度，文本预测不出额外结构。
  - **限制**：这是线性 ridge 读出的**线性可预测性**，不排除更强的非线性规划器；但按卡上预注册判据
    （无优势即维持 parked），**idea-010 维持 parked、不排训练臂**。

**C. 明确暂缓（工程量大、且不是当前瓶颈）**

- `idea-021`（分块异步去噪）：模型 + 采样器大改，接口检查不构成效果证据 → 用户选出后再排。
- `idea-022`（可复用动作单元）、`idea-023`（草稿→扩散修正）：两阶段研发成本高，
  且依赖"先有可用草稿/单元"——当前无此产物 → 暂缓。
- `idea-025`（语义覆盖状态）：配套候选，依赖生成过程分解 → 暂缓。

**执行顺序建议（2026-10-17 更新，P1–P8 已全部跑完）**：免训练前置全部结清——
P1/P2 否决 idea-019/027，P4/P5 由现有证据结清 idea-020/005，P6 PASS（idea-024）、
P7 PASS（idea-026）、P8 FAIL（idea-010 维持 parked）。因此**现在只剩两个"前置已过、可开小臂"的
idea：024 与 026**；其余或需用户裁决（016/020b/005），或属大工程暂缓（021/022/023/025）。

**当前 GPU 状态（2026-10-17 12:0x）**：0/1/3/4/5 有他人作业；2/6/7 正在跑本轮三条
λ 消融评测（各留约 20 GiB 余量）。**没有整张空闲卡**，故本轮先不新起训练，等 λ 评测结束
（预计 ~1 h）腾出 3 张卡再排下表的训练臂。

**体检结论：剩下真正"未验证 + 前置已过"的 idea 只有两条，都建议先做等预算小臂**
（各遵守 D34：显式声明 `trainable` 与文本侧 lr；对照用同起点同预算，不用 train 侧做晋级）：

- [x] **R1 · idea-024 等预算臂 —— 2026-10-18 已跑完，四项回译全部显著为正（首个训练侧正向杠杆）**。
  - 实现：`t2v/models/temporal_align.py`（可微 soft-DTW，**对角向量化**；把 torch 的 loss
    从 per-cell Python 循环改成 anti-diagonal 批内向量化，T=64/B=16 由 **1.2 s → 0.20 s**/call）；
    `train.py` 接线 `train/align` 指标；配置 `configs/dit_b_text_prefix_align.yaml`。
  - **必须记录的两个坑**：① 训练在 bf16 autocast 下，DP 必须在 loss 内部
    `torch.autocast(enabled=False)` + 全程 fp32，否则 `+inf` 掩码被降精度成 NaN；
    ② `logsumexp` 遇到三个前驱全 `+inf` 时反向会出 `-inf - -inf = NaN`，
    需先把不可达前驱 clamp 成有限大数（BIG=1e30）再 soft-min。
  - **自检**：与独立写的 per-sample 参考实现逐格对齐（20×20、17×24 误差 <1e-6）；
    真实 latent 40 条复现前置判据（变速 0.026/0.010 ≤0.10、倒序 0.98、静止 12.07 ≥0.50）；
    smoke 50 步无 NaN、`train/align` 正常下降。
  - 臂设定：起点 `runs/dit_b_prefix30k/final_60k.pt` → 20k，**文本冻结（trainable 未开，与对照一致，按 D34 声明）**，
    唯一变量 = `align_loss`（weight 0.1、warmup 2000、gamma 0.1、max_frames 64）。
    对照 = `route_s_cont20k`（同起点同预算纯续训）。
  - 状态（2026-10-17 晚）：**gpu0 训练中**（~3.2 step/s，20k 约 1.7 h），队列
    `reports/queue/jobs_route_t_20261017.txt`（`--gpus 0,4,5,6,7 --max-parallel 1`），
    训练完自动接 test 642 评测。判据：D3 检索门槛 + D20 回译三项不得显著退化。
  - **结果（2026-10-18，test 642，k=1 口径，DPM++(2M) 50 NFE，CFG 2.0，oracle 长度）**：

    | 臂 | B1 | B4 | ROUGE | WER | FVD | MRR |
    |---|---|---|---|---|---|---|
    | `route_t_align_20k`（align w=0.1） | **12.18** | **3.91** | **12.66** | **100.78** | 26.00 | 0.0232 |
    | `route_s_cont20k`（同起点同预算纯续训） | 10.57 | 1.57 | 9.54 | 104.57 | 26.38 | 0.0195 |

  - **配对判读（vs route_s_cont20k，642 条逐视频 bootstrap）**：ΔB1 **+1.59 [+0.24,+2.94] 显著**、
    ΔB4 **+2.34 [+1.32,+3.36] 显著**、ΔROUGE **+3.10 [+1.67,+4.56] 显著**、
    ΔWER **+3.78 [+1.87,+5.66] 显著**（WER 越低越好，故为正改善）；ΔMRR **+0.0036 [−0.0011,+0.0088] n.s.**
    （P(>0)=0.93，方向为正但未过 95% 线）。
  - **结论**：按 **D3 + D20**，本臂**四项回译全部显著为正**、检索 MRR 不显著退化（方向为正）
    → **是本项目第一个通过晋级口径的训练侧杠杆**（此前的训练侧增益都伴随显著回译退化）。
    与 P11 顺序敏感度互证：`route_t_align_20k` 的 reversed gap **1.1263**、block4 **1.0112**，
    是全部生成臂中最接近 GT 上限（1.2818 / 1.1452）的，说明它确实把细时间结构做回来了。
  - **注意**：MRR 的绝对水平（0.0232）低于 λ=0.02 语义臂（0.0333），但那是**不同轴**的指标；
    本臂的意义是"回译显著变好且不牺牲检索"，而语义臂是"检索/B1 高但 WER 显著退化"。
    两者机制不同（轨迹形状 vs 句子语义压力），故已排 P20 测叠加。
  - 产物：`runs/route_t_align_20k/final.pt`、`reports/eval/matrix/route_t_align_20k/test_n642/`、
    `reports/eval/ranks/route_t_align_20k_test_n642.json`、`reports/eval/dispersion_route_t_align_20k.json`、
    `reports/eval/order_sensitivity_r1r2.json`。EXPERIMENT 待登记。
- [x] **R2 · idea-026 等参数静态 vs 动态对照臂 —— 2026-10-18 已跑完，机制不成立（动态不优于等参数静态）**
  （文本响应随噪声阶段 1×→9× 单调上升，ratio 1.80）。卡片预注册判据：**动态必须显著优于同参数静态**，
  否则收益只能归因于新增容量（教训 FAIL-019 双流 / FAIL-024 槽位）。
  - 实现：`t2v/models/stage_connector.py`（`StageTextConnector`）。做法 = 在文本侧对
    "整句摘要"（token 的 masked mean）与"局部 token"做**逐特征混合**：
    `alpha = sigmoid(t_proj(stage_embed + t_embed))`、`blend = (1-alpha)*context + alpha*pooled`、
    `context' = context + mix_out(norm(blend))`。`mix_out` 零初始化 → step 0 与父模型**逐位相等**。
    residual 按 `context_mask` 屏蔽，pad 位置保持父表示不变。
  - **等参数对照的做法**：两臂结构/初始化/参数量完全相同（各 **+1,774,080**），都持有同一个
    零初始化 `stage_embed` 并走同一个 `t_proj`；**唯一差别** = dynamic 额外把 `t_embed` 加上去，
    static 不加（于是 static 只能学一个与阶段无关的常数混合）。任何差异因此只能归因于**阶段依赖本身**。
  - **必须记录的两个坑**：① `DiT.initialize_weights()` 用 Xavier 重刷所有 `nn.Linear`，
    会覆盖连接器的零初始化 → 必须在 `initialize_weights()` **之后**调 `reset_to_identity()`，
    否则热启动不再逐位等于父模型（实测 max|diff| 0.00037 → 修好后 **0.0**）；
    ② 连接器 residual 必须按 mask 置零，否则 pad token 也被扰动（测试抓到）。
  - 自检：`tests/test_stage_connector.py`（热启动两臂逐位精等、参数量相等、只有 dynamic 吃 timestep、
    pad 不外泄、CFG 双批前向可用、父模型不带连接器）；真 checkpoint 热启动实测 max|diff| **0.0**；
    两臂各 30 步 smoke 全绿；`tests/run_all.py` 全绿。
  - 臂设定：起点 `runs/dit_b_prefix30k/final_60k.pt` → 20k，**文本冻结（未开 trainable，按 D34 声明）**，
    与干净续训对照 `route_s_cont20k` 同起点同预算；**不启用语义目标**（避免第二个机制混淆单变量对照）。
    config `configs/dit_b_text_prefix_stageconn.yaml`（dynamic）/ `..._ctl.yaml`（static）。
  - 状态（2026-10-17 晚）：**gpu5 / gpu6 并行训练中**（~6 step/s，20k 约 55 min），队列
    `reports/queue/jobs_route_uv_20261017.txt`（`--gpus 5,6,7 --max-parallel 2`，日志
    `reports/queue/logs_route_uv/`），训练完自动接 test 642 评测。
  - 判据：D3 检索门槛 + D20（MRR 过门槛但回译任一项显著退化则不晋级）；机制成立须 **u 显著优于 v**。
  - **结果（2026-10-18，test 642，k=1 口径）**：

    | 臂 | B1 | B4 | ROUGE | WER | FVD | MRR |
    |---|---|---|---|---|---|---|
    | `route_u_stageconn_20k`（dynamic，吃 timestep） | 10.05 | 1.27 | 9.06 | 105.59 | 26.45 | 0.0194 |
    | `route_v_stageconnctl_20k`（static，等参数对照） | 10.09 | 1.13 | 8.88 | 105.89 | 28.00 | 0.0204 |
    | `route_s_cont20k`（干净续训） | 10.57 | 1.57 | 9.54 | 104.57 | 26.38 | 0.0195 |

  - **配对判读**：u vs v（**机制对照**）ΔB1 −0.05 n.s.、ΔB4 +0.14 n.s.、ΔROUGE +0.18 n.s.、
    ΔWER +0.30 n.s.，**ΔMRR −0.0010 [−0.0041,+0.0008] n.s.**；
    u vs cont20k ΔMRR −0.0001 n.s.；v vs cont20k ΔMRR +0.0009 n.s.。
  - **结论**：卡片预注册判据是"**动态必须显著优于同参数静态**，否则收益只能归因于新增容量"
    （教训 FAIL-019 双流 / FAIL-024 槽位）。实测 **u 与 v 在所有指标上都无显著差异**，
    且两臂都不优于干净续训 → **idea-026 机制不成立**，按规范 9 出池（FAIL 待写）。
  - 副产品（P11 顺序敏感度，`reports/eval/order_sensitivity_r1r2.json`）：u reversed +1.1115 / block4 +1.03497、
    v +1.1211 / +1.05095 —— 两者都**比老最好臂更接近 GT 上限**（λ=0.02 是 +1.0054/+0.9172），
    但彼此只差 ~0.01，再次说明该尺子不能给顶部队列排序（与 P16 结论一致）。
  - 产物：`reports/eval/matrix/{route_u_stageconn_20k,route_v_stageconnctl_20k}/test_n642/`、
    `reports/eval/paired/*_mrr.log`、`reports/eval/dispersion_route_{u,v}*.json`。

**暂缓（需用户裁决或工程量大，不在本轮排期）**：
- ~~`idea-016`：只等用户裁决 → TODO N6~~ → **2026-10-18 已裁决采纳 k=4，见 D37**（部署默认已改）。
- ~~`idea-002`：等用户裁决 → TODO N7~~ → **2026-10-18 已裁决复活，见 D38**；下一步方向待选，见 **P15**。
- `idea-020`：奖励式路线 (a) 前置失败（D24）；偏好式路线 (b) 不依赖该奖励，需用户先裁决方向。
- `idea-005`：现有教师可作语义奖励、不可作顺序奖励；推进须先换/校准教师并重跑可靠性脚本。
- `idea-021/022/023/025`：模型/采样器或两阶段研发成本高，等用户选出再排（D27）。

### 第四优先 · 新方向（2026-10-17 诊断驱动）

- [x] **P9 · 前沿诊断：检索门槛是否已饱和 + 幅度是否就是 BLEU 差距的来源** —— **2026-10-17 已跑完**。
  - 做法：① `tools/probe_latent_dispersion.py`（新写）在归一化 latent 空间量每条臂的
    整体 std / 逐帧差 RMS / 每段平均 |Δ|，与同批 GT latent 对比；② 检索尺子的读法复核
    （`LatentTextTeacher` 把每段**线性重采样到 64 bin** 再编码）。
  - **发现一（门槛饱和）**：最好的臂 MRR 已达**冻结 VAE 重建上限的 97%**
    （`route_b_lambda020_5k` 0.0333 vs 上限 0.0343；`prefix_120k` 97.9%、λ=0.05 97.7%），
    而有 7 条臂 ≥90%。**D3 检索门槛在当前最好臂上已经没有区分度**——继续用 MRR 选点等于
    在噪声里选。且检索教师输入被压到 64 bin，**结构性看不见细时间结构**，这解释了
    "检索饱和、回译远未饱和"（B1 14.30 vs 上限 32.36）并存。
  - **发现二（幅度不足）**：所有臂的逐帧运动幅度只有 GT 的 **27%–63%**
    （delta_rms 比：`route_b_lambda020_5k` 0.480、`prefix60k` 0.366、`route_k_repa` 0.276）。
    整体 std 只能到 GT 的 60–78%，即生成轨迹**系统性偏"平"**。这是 x0-MSE 在宽噪声分布下
    向条件均值回归的典型表现（与"过度平滑/欠弥散"一致）。
  - **登记**：`EXPERIMENT.md` 待补；`reports/eval/dispersion_*.json`；工具入 `STRUCTURE.md`。
  - **限制**：幅度与 B1 的跨臂相关只有 r=0.29，单一数量不能解释全部差距；**幅度是否是
    *因果*瓶颈，靠 P10 的放大实验判定**，不靠这条相关。
- [x] **P10 · 前置证伪：把生成的运动幅度放大，回译会不会变好** —— **2026-10-18 已跑完**（本条目是执行过程中的旧叙述，**结清结论以下方 `[x]` 的 P10 条目为准**；保留原文不删）。
  - 首轮（`jobs_amplify_20261017.txt`）两条臂**都因脚本 bug 失败**：① 配置里的 `data.cache` 是
    `str`，脚本直接做 `config.data.cache / "shards"` 抛 `TypeError`；② arm 分支在 `video_ids`
    定义之前就引用它，抛 `UnboundLocalError`。另修两个接线错误：回译脚本要的是 `frames` 的
    **父目录**、FVD 要的是同一层，原写法多嵌了一层 `test/test`。另外把摘要输出改为跟随
    `--out-root`（冒烟测试曾把 6 条样本的摘要写进正式产物路径，使队列误判"已完成"）。
  - 修好后端到端冒烟通过（6 条样本走通 解码→帧数守卫→SignMbart→FVD）。**第三轮又抓到一个 bug**：
    GT 分支把缓存里的**原始** latent 又做了一次逆归一化（缓存里本来就是 raw，`eval_vae_recon.py`
    也是直接解码 raw），导致 s=1 的 B1 只有 **4.95**（重建上限是 32.36）——这正是预注册判据②
    "GT 线在 s=1 不是最优就该作废探针"要抓的情形。修好后 8 条冒烟复测 s=1 B1 **26.39**，
    与上限同量级。λ=0.02 臂不受影响（其 s=1 给 15.06，与全量 14.30 一致）。当前两臂在跑：
    `jobs_amplify2_20261018.txt`（GT 对照，gpu4）+ `jobs_amplify_b_20261018.txt`（λ=0.02 臂，gpu7），
    各 192 条 × 4 档 s∈{1,1.5,2,2.5}。**已改为每档出完 FVD 即删帧**（/data 已 99%）。
  - 做法：取已完成臂的 `latents.pt`，按 `z_t -> z_0 + s*(z_t - z_0)`（首帧锚点不动）放大后，
    走**同一冻结 VAE 解码 + 同一 SignMbart 回译 + 同 FVD**；`s=1` 复现原样。
  - **预注册判据不变**：① 某条臂 s>1 明显变好 ⇒ 幅度是瓶颈、训练侧可修（立/推进 idea-028）；
    ② GT 重建线在 s=1 不是最优 ⇒ 探针无效、结论作废；③ 放大不变或变差 ⇒ 否决该方向。
  - **注意**：这条探针**本身不是可部署方法**（用了目标 latent 的幅度，属事后处理），
    只用来判定"训练侧该不该往弥散/幅度上使劲"；可部署版本见 P13 的 idea-030。
  - 产物：`reports/eval/motion_amplify_route_b_lambda020_5k.json`、`..._gt_recon.json`。

- [x] **P11 · idea-029 前置②：细时间结构还在不在（顺序敏感度，零训练）** —— **2026-10-18 已跑完**，
  入口 `tools/probe_order_sensitivity.py`。
  - 做法：对同一段 latent 造 原序 / 倒序 / 块置换(block4) / 静止 四个变体，用 idea-024 的
    `TemporalAlignLoss`（soft-DTW）读它们的残差缺口——**不经过 64-bin 重采样**，所以是 D3
    检索尺子缺的细粒度读出。GT 重建上限 + 5 条臂（prefix60k / prefix_120k / λ=0.02 / λ=0.05 /
    route_a_control）。
  - **结果（test 642，全量）**：

    | 臂 | reversed | block4 | static |
    |---|---|---|---|
    | GT 重建上限 | **+1.2818** | **+1.1452** | **+16.6543** |
    | `prefix_120k`（MRR 0.0336，97.9% 上限） | +1.0184 | +0.9161 | +1.8975 |
    | `route_b_lambda020_5k`（MRR 0.0333） | +1.0054 | +0.9172 | +3.5243 |
    | `route_b_semantic_5k`（MRR 0.0336） | +0.9993 | +0.9138 | +3.1725 |
    | `prefix60k`（MRR 0.0165） | +0.9731 | +0.8783 | +2.7042 |

  - **判读**：细时间结构**确实被削弱且与预算相关**——GT 的 reversed/block4 缺口（+1.28/+1.15）
    高于所有训练臂（+0.97~+1.02 / +0.88~+0.92），static 差距更大（+16.7 vs +1.9~+3.5）。
    且**缺口随预算单调上升**（prefix60k +0.97 → prefix_120k +1.02），与"训练越久时序结构越回来"
    的 P1 观察一致。→ 按卡片预注册判据（GT 与最好臂之间要留出显著间隔）**PASS**。
  - **关键限制（重要）**：四条臂**彼此之间**的差距只有 ~0.05（+0.97~+1.02），远小于 GT 与它们的
    0.26 差距。所以这把尺子**证明"生成整体丢了细时间结构"，但不能证明"它能在顶部几条臂之间排序"**
    ——而 idea-029 作为**晋级门槛**需要的是后者。结合 P16（MRR↔B1 跨 44 臂 +0.781，
    但顶部 5 条挤在 0.0305–0.0336），**idea-029 若要替代 D3，必须先证明顶部排序能力**，
    否则只适合作并列诊断量。裁决要按这条收紧后的判据来。
  - **边界**：这是**判据可用性**证据，不是模型增益；idea-029 是否采纳为并列门槛**待用户裁决**。

- [x] **P12 · D37 遗留缺口：gate k=4 与 CFG3.0[200,999] 叠加 —— 2026-10-18 已跑完，两者饱和、不叠加**。
  - 依据：`NOTES.md` D37 落地时明确写"k=4 与 CFG3.0[200,999] 叠加没测过，需要时另跑同协议
    评测再定"。两者是本项目目前最强的两条推理期杠杆（各显著为正、机制不同：gate 放大每步
    cross-attn 文本推力且 **1× NFE**；CFG 在高噪声段加倍前向、**2× NFE**）。
  - 臂：`prefix60k_gate4cfg30i200`（k=4 + CFG3.0[200,999]），对照 `prefix60k_gate4`（0.0313 MRR /
    B1 12.75）与 `prefix60k_cfg30i200`（0.0219 / 12.72）。入口
    `reports/queue/jobs_gate4_cfg_20261018.txt`（等卡；/data 99% 下 `--min-free-gib 20`）。
  - 判读：若叠加 ≈ 单杠杆 ⇒ **已饱和**，部署保持 k=4（1× NFE 更省）；若显著更好 ⇒ 记入部署口径；
    若回译涨而 MRR 掉 ⇒ 按 D20 不晋级。
  - **结果（2026-10-18，test 642，k 口径见下）**：

    | 臂 | B1 | B4 | ROUGE | WER | FVD | MRR |
    |---|---|---|---|---|---|---|
    | `prefix60k_gate4cfg30i200`（k=4 + CFG3.0[200,999]） | 12.67 | 3.34 | 12.54 | **101.25** | 23.45 | **0.0310** |
    | `prefix60k_gate4`（只 k=4） | **12.75** | **3.55** | **12.58** | 101.75 | 21.89 | 0.0313 |
    | `prefix60k_cfg30i200`（只 CFG3.0[200,999]） | 12.72 | 3.37 | 12.49 | 102.89 | 22.89 | 0.0219 |

  - **判读**：叠加臂在 B1/B4/ROUGE/MRR 上都**没有超过**只用 k=4 的那条（B4 12.75→3.34 反而更低、
    MRR 0.0313→0.0310 基本持平），只在 WER 上略好（101.25 vs 101.75，属噪声量级）。
    → **两条杠杆饱和**：CFG3.0[200,999] 在 k=4 之上不再提供增量，而它还要多花 **1× NFE**。
    **部署保持 k=4**（D37 不变），本条不需要修改部署口径。
  - 产物：`reports/eval/matrix/prefix60k_gate4cfg30i200/test_n642/`、
    `reports/eval/retrieval_matrix_prefix60k_gate4cfg30i200_test.json`。

- [x] **P10 · 前置证伪：把生成的运动幅度放大，回译会不会变好** —— **2026-10-18 已跑完，判据③成立 → 否决"事后放大"方向**。
  - **实测（192 条 test，预注册三档 s∈{1,1.5,2}）**：

    | 臂 | s | B1 | B4 | ROUGE | WER | FVD |
    |---|---|---|---|---|---|---|
    | `route_b_lambda020_5k` | 1.0 | **15.06** | 3.97 | 13.04 | 104.04 | 29.96 |
    | `route_b_lambda020_5k` | 1.5 | 10.47 | 1.07 | 8.08 | 121.68 | 205.65 |
    | `route_b_lambda020_5k` | 2.0 | 3.55 | 0.00 | 4.00 | 104.55 | 332.50 |
    | **GT 重建对照** | 1.0 | **32.33** | 11.68 | 30.49 | 89.22 | 19.52 |
    | **GT 重建对照** | 1.5 | 6.40 | 1.01 | 6.55 | 101.29 | 333.37 |

  - **判读**：判据②**不成立**（GT 线在 s=1 就是最优，32.33 ≈ 上限 32.36，探针本身有效——
    这正是首轮那个"原始 latent 被二次逆归一化"bug 修好后的读数，见本条历史）；判据①**不成立**
    （放大后回译单调变差、FVD 爆炸到 200–330）。故按预注册判据③ **否决"事后放大"**：
    **运动幅度不足不是"整体缩放"能修的瓶颈**——放大把轨迹推出冻结 VAE 的可解码流形
    （连 GT 自己 s=1.5 都掉到 B1 6.40、FVD 333），不是"内容对了只是太小"。
  - **连带结论**：**idea-028（弥散校准的 x0 目标）按本条不立项**（TODO P14 结清）；
    **idea-031（min-SNR 加权）**另有 P17 的独立证据削弱（前提只成立一半）。
  - **重要限定（必须一起引用）**：本条否的是**离流形的整体缩放**。P9 测到的欠弥散
    （逐帧幅度仅 GT 的 27%–63%）是**真现象**，但它不能靠"乘一个系数"修。
    **沿噪声表的采样期注入（idea-030）与阶梯重排（idea-032）是不同干预**，不继承此否决，
    各自按新预注册判据独立评测——已分别在 `jobs_eta_sde_20261018.txt` 与
    `jobs_logsnr_spacing_20261018.txt` 开跑。
  - 产物：`reports/eval/motion_amplify/route_b_lambda020_5k/s{1,1.5,2}/`、
    `reports/eval/motion_amplify/gt_recon/s{1,1.5}/`。
  - **运行事故（已处理）**：`jobs_amplify2` 与 `jobs_amplify_gt` 两个队列**同时**把同一个 GT
    对照写进 `reports/eval/motion_amplify/gt_recon/`（两条命令等价、`#done=` 守卫相同），
    会让两个进程互相 `prepare_frame_dir` 删帧。2026-10-18 02:0x 已 kill 重复的
    `jobs_amplify_gt` runner 与其子进程，只留 `jobs_amplify2` 那条；已完成的 s=1/s=1.5 产物
    是在冲突前写完的（分数自洽：GT s=1 = 32.33 ≈ 上限），s=2 目录只剩 frames 半成品、
    无分数，**不作为证据**。教训：**同一输出目录同一时刻只能有一个 runner**。

- [x] **P18 · idea-032（采样阶梯 log-SNR 均匀）执行与判读** —— **2026-10-18 已实现、已开跑；2026-10-19 判读：无效（FAIL-038）**。
  - 依据：现役 50 步阶梯是 t 索引均匀，但求解器积分变量是半 log-SNR λ。实测该阶梯下
    **第一步走掉 40.3% 的 λ 行程**、t≥850 占 55.1%、训练最密的 t∈[300,700) 只占 4.8%、
    λ 步长比 **92×**；DPM++(2M) 的 `ratio_clip=(0.5,2.0)` 在这条阶梯上有 2 处被截断
    （二阶退化成近似一阶）。λ 均匀后步长比降到 3.96×、截断步 2→1。
  - 实现：`t2v/diffusion/gaussian.py:respaced_timesteps(steps, spacing)` 新增 `"logsnr"`；
    新增 `resolve_timestep_spacing(config)` 读 `sampling.timestep_spacing`（默认 `uniform`，
    未知值**报错不静默回退**）；`sample_latents` / `ddim_sample_loop` /
    `dpm_solver_pp_sample_loop` / `GenerationRequest` / `tools/eval_t2v.py` /
    `tools/generate.py` 全部接线，`meta.json` 记录 `timestep_spacing`。
    **默认路径逐位不变**（`spacing="uniform"`），所以全部历史数字不受影响。
  - 自检：`tests/test_dpm_solver.py` 新增两条（λ 均匀性 + 单调性 + 默认不变 + 未知值报错；
    两档 solver 的输出确实改变 + config resolver 语义）。`tests/run_all.py` 全绿。
    注意：**仿射于 λ 的 head 对两种阶梯给出相同结果**（本来就该如此），所以敏感性用
    依赖 x 的非线性 head 测，否则测的是错的东西。
  - 队列 `reports/queue/jobs_logsnr_spacing_20261018.txt`（gpu5）：先 `uniform` 对照
    （自跑、不复用历史 prefix60k），再 `logsnr` 两档 solver。全部显式
    `--cross-gate-scale 1.0` 回到 k=1 口径（config 现默认 k=4，见 D37）。
  - 判据（预注册）：① 逐帧幅度比显著上升；② B1/WER 至少一项显著改善、另一项不显著退化；
    ③ MRR 不得显著退化（D20）。三条全过才通过。
   - 读法提醒：本条与 P10 是**不同干预**（P10 是离流形整体缩放，本条是采样路径重排），
     不继承 P10 的否决。
   - **判读（2026-10-19 收尾）**：**无效，出池（FAIL-038）**。uniform 对照 B1 10.88/B4 2.14/ROUGE 10.32/WER 106.32/MRR 0.0165、幅度比 0.3671；
     DPM++ logsnr 11.24/2.27/10.69/106.32/0.0172/幅度比 **0.3870**；DDIM logsnr 11.16/1.95/10.50/107.60/0.0167/0.3878。
     判据①成立（幅度比 0.3665→0.387，+5.6%）、判据②不成立（回译四项全 n.s.、MRR 也 n.s.）→ 机制真实但幅度不是这条路径的可用杠杆。见 `EXPERIMENT.md` exp-080。

- [x] **P19 · idea-030（DDIM η 注入）执行与判读** —— **2026-10-18 已跑完，判据③成立 → 否决该方向（FAIL-035）**。
  - 队列 `reports/queue/jobs_eta_sde_20261018.txt`（gpu4 + gpu7 并行）：先补 **DDIM η=0**
    对照（隔离"换采样器"与"加噪声"两个变量），再跑 **η=0.3**。全部显式
    `--cross-gate-scale 1.0` 回到 k=1 口径。
  - 口径说明：现状部署是 DPM++(2M) 纯确定性（无 η 通道），DDIM 路径已支持 η>0；
    故本档必须自跑 DDIM η=0 对照，**不能拿 DPM++ 的历史 prefix60k 当对照**。
  - 判据（预注册）：① 幅度比显著上升且 B1 不降 → 再补 η∈{0.2,0.5}；② η>0 档 WER/FVD
    明显变差（CFG 会把注入噪声一起放大）→ 按 D20 不晋级、不加 η；③ 幅度不动 → 否决该方向。
  - 与 P10 的关系同上：不继承 P10 否决（离流形缩放 ≠ 沿噪声表注入）。

  - **判读（2026-10-18 已完成）**：两档 test-642（DDIM 50 NFE、CFG 2.0、oracle、k=1）：

    | 臂 | B1 | B4 | ROUGE | WER | FVD | MRR | delta_rms 比 |
    |---|---|---|---|---|---|---|---|
    | `prefix60k_ddim_eta000`（对照） | 10.99 | 1.84 | 9.95 | 107.02 | 22.58 | 0.0164 | 0.3612 |
    | `prefix60k_ddim_eta030`（η=0.3） | 11.01 | 1.97 | 10.26 | 105.23 | 22.41 | 0.0174 | 0.3572 |

    配对 η=0.3 vs η=0：ΔB1 +0.02 [−0.86,+0.92] **n.s.**、ΔB4 +0.12 n.s.、ΔROUGE +0.30 n.s.、
    ΔWER +1.74 n.s.。→ 判据①不成立（回译全 n.s.）、判据②不成立（WER/FVD 没变差）、
    **判据③成立**（幅度比 0.3612→0.3572 **没升**）。故 **idea-030 出池**，写入 `FAIL.md` **FAIL-035**。
    机制解读：η 只改每步注入的随机量，不改模型学到的 x0 条件期望，而轨迹幅度由条件期望决定。
    **不连带 idea-032**（改阶梯离散化、非注入随机量，见 P18）。证据见 `EXPERIMENT.md` exp-071。

- [x] **P13 · 新立 idea-030/031：欠弥散的两条修法** —— **2026-10-18 结清**：
  P10 判"事后放大"无效（见上），但**不等于**两条修法都死。逐条结清：
  **idea-030（采样期随机注入）→ 保留并开跑**（见 **P19**）——它是**沿噪声表**加方差，
  与"离流形整体缩放"不是同一干预；**idea-031（min-SNR 加权）→ 由 P17 独立削弱**
  （前提只成立一半：loss 确实被高噪声步主导，但那些步的预测**并不更平**，见 P17）。
  另新立 **idea-032（阶梯 log-SNR 均匀）**并开跑（见 **P18**）——它比 030/031 都便宜
  （零训练、NFE 不变、与 η 正交）。
  - 背景：P9 实测所有臂逐帧幅度只有 GT 的 27%–63%；P10 正在判定"幅度是不是**因果**瓶颈"。
    两条新卡都指向同一机制，但**成本/可部署性不同**：
    - **idea-030 · 采样期随机注入（SDE/DDIM η）**：**零训练、可部署**。现状 DPM++(2M) 采样
      完全确定性（`t2v/diffusion/gaussian.py:dpm_solver_pp_sample_loop` 无 η 通道），而 DDIM
      路径已支持 `eta>0`（`ddim_sample_loop` 有 σ 注入）。做法 = 换带随机项的更新，
      η∈{0.2,0.5,0.8} 同 test-642 协议评测。**这是 P10"事后放大"的可部署版本**
      （P10 用目标 latent 的幅度、不能部署；随机注入是纯采样期改动）。
    - **idea-031 · 噪声阶段加权的 x0 目标（min-SNR-γ）**：**改训练**。现状对 t 等权 MSE；
      改成 `w(t)=min(SNR,γ)/SNR`（γ≈5），避免高噪声步的"条件均值"解吃掉训练预算。
      20k 分叉臂约 1 h；对照 `route_s_cont20k`；γ∈{5,20} 两档。与 idea-028 同族、按成本择一。
  - **开档条件**：P10 判"幅度是瓶颈" ⇒ 030 先跑（零训练成本、且可部署）；判"不是瓶颈" ⇒ 两条
    一起搁置（同一个因果假设被证伪）。判定前不占训练资源。

- [x] **P17 · idea-031 前置：x0 误差是否被高噪声步主导、那些步是否更平** —— **2026-10-18 全量已跑完**
  （32 视频 × 11 个 t × 8 seed，`reports/eval/error_by_timestep_prefix60k.json`）**，入口 `tools/probe_error_by_timestep.py`（新写，纯推理）。
  - 做法：在真实 clip 上按 t 分箱，量 ① 每步 x0 MSE；② 该步在 **logit_normal 实际权重**下占
    总 loss 的百分比；③ 预测的帧间差 RMS / GT 的帧间差 RMS——单次抽样版（`single`）与
    **多次抽样取均值**版（`mean`，即条件均值 E[x0|x_t,c] 的估计）。
  - **冒烟结果（prefix60k，train 6 条 × t∈{100,500,900} × 4 noise seed）**：
    - **前提一成立**：loss 被高噪声步**压倒性主导**——t≥500 占 **97.3%**、t=900 单独占 **94.5%**；
      t=100/500 各只占 ~2.7%。等权 MSE 实际上几乎只在训练"接近纯噪声"的步。
    - **前提二不成立**：那些步的预测**并不更平**——条件均值版的帧间差比是 t=100 **0.999**、
      t=500 **0.991**、t=900 **1.034**（≥1 说明还偏高，很可能是残余噪声把帧间差抬起来了，
      取 4 个 seed 均值后仍未完全抵消）。即"高噪声步的 x0 预测被拉平成条件均值"这个**直接机制
      没有在单步预测上观察到**。
  - **判读**：idea-031 的**前提只成立一半**。loss 权重确实极度倾斜（这条支持"重新配权"），
    但"高噪声步的预测更平"没被观测到——而 P9 测到的欠弥散是在**采样后的完整轨迹**上。
    两者合起来更像"**每步预测没问题、但沿采样链累积/被 CFG 平均成平轨迹**"，
    这**更支持 idea-030（采样侧）而不是 idea-031（训练权重）**。
  - **限制（必须一起引用）**：① 冒烟只有 6 条视频、3 个 t、4 个 seed，**不足以定论**；
    ② `mean` 列在 seed 数少时仍有残余噪声上偏；③ 这是单步预测诊断，**不能替代**采样轨迹上的
    P9/P10 证据。**下一步**：等卡跑全量（32 条 × 11 个 t × 8 seed），再决定 031 是否降级。
  - 产物：`reports/eval/error_by_timestep_prefix60k_smoke.json`（冒烟）。

- [x] **P14 · P10 判读后推进 idea-028** —— **2026-10-18 结清：idea-028 不立项**。
  P10 判"事后放大"无效（GT 自己 s=1.5 就崩、放大臂单调变差），说明幅度不足**不是整体缩放
  能修的瓶颈**。028 的三种候选实现都是"让目标不再惩罚偏离条件均值的波动"，与"放大已有轨迹"
  共享同一个已被否的因果假设 → **不排训练臂**。若将来要复活，须先给出**新的**、与"整体缩放"
  不同的因果证据（例如证明欠弥散来自训练目标而非采样路径），并重新预注册。

- [x] **P15 · 【2026-10-20 用户指令取消，不做】idea-002（route_q 软对齐）的证据加固（2026-10-19 改写：三个选项收敛为一个）**
  —— **未开跑**。
  - 依据 **D38**：`route_q` 在干净等预算对照上 ΔB4 **+1.07 显著**、ΔROUGE **+1.29 显著**、
    ΔB1/ΔWER 方向偏好不显著、ΔMRR +0.0001 不显著 → FAIL-025 撤销、idea-002 复活。
  - **唯一保留的动作**：**在多生成种子下复现这条 +1.07**。理由：实测跨种子 B4 极差为
    prefix60k **1.70** / prefix90k **1.00** / prefix_120k **1.06**；项目自己的 TODO 11
    （60k/90k/120k × seed 42/43/44）也已因 FVD 种子方差过大而声明"FVD 不宜作排序依据"——
    **但同样的纪律没有施加到 B4 上**。若 +1.07 在 ≥3 个种子下守不住，则 idea-002 应再次出池。
  - **已作废的两个原选项**：② "更强对齐形式（soft-DTW）"——已由 R1/idea-024 在另一轴上做完，
    且其后续臂 W/X/Y 已全部出池（exp-081/084）；③ "换细粒度尺子重读"——idea-029 已按
    `THINK.md` §0 出池（评测口径不占 idea 编号），其结论"尺子不能在顶部队列内排序"已入
    `EXPERIMENT.md` exp-065。
  - **不做的事**：不在没有新预注册的情况下重跑同一配置；不复用已撤销的 FAIL 编号。
  - 相关证据：`reports/eval/paired/route_q_softalign_20k_vs_cont20k_test.log`、
    `..._mrr_test.log`；`FAIL.md` FAIL-025 补记二；`NOTES.md` 2026-10-18。

- [x] **P20 · R1 的三个单变量跟进臂（2026-10-18 已实现、已开跑；2026-10-19 判读：三条全部失败）** —— R1 是首个四项回译全部显著为正的
  训练侧杠杆（见上），本条只回答"它还能不能更好 / 能不能叠加"：
  - 队列 `reports/queue/jobs_align_followups_20261018.txt`（`--gpus 0,4,5,6,7 --max-parallel 2`，
    日志 `reports/queue/logs_align_followups/`）。三条臂全部从 `runs/dit_b_prefix30k/final_60k.pt`
    热启动 20k，**每条只动一个变量**，文本侧状态按 **D34** 显式声明：
    - **W · `route_w_align_sem02_20k`**（config `dit_b_text_prefix_align_sem.yaml`）：
      align(w=0.1) + 语义 λ=0.02（D35 默认），**文本冻结**。对照 `route_t_align_20k`，唯一变量 = 语义项。
      问：轨迹形状增益与句级语义增益是否可叠加（λ=0.02 是 B1/MRR 最高但有显著 WER 代价；R1 是无代价的
      回译增益）。
    - **X · `route_x_align_textlr5e5_20k`**（config `..._align_textlr.yaml`）：align(w=0.1) +
      **文本微调 5e-5**（D18 已验证档），无语义项。对照 `route_t_align_20k`，唯一变量 = 文本侧
      `trainable`/lr。问：对齐损失能否叠在文本微调上（文本微调本身也贡献了显著增益，见 N8 副产品）。
    - **Y · `route_y_align_w030_20k`**（config `..._align_w030.yaml`）：align 权重 **0.1 → 0.3**，
      文本冻结。对照 `route_t_align_20k`，唯一变量 = 权重。问剂量反应：0.1 已显著为正，是否还有空间。
  - 自检：三条 config 的 30 步 / 10 步 smoke 均通过（无 NaN、`train/align` 正常、语义 scorer 加载正常）。
   - 判据（预注册）：D3 检索门槛 + D20 回译不退化；每条相对 `route_t_align_20k` 走 `tools/paired_arm_stats.py`
     配对 CI。**注意**：W 预期继承 λ=0.02 的 WER 代价（那是用户 2026-10-17 显式采纳的权衡），
     汇报时必须带该代价，不能当作新退化。
   - **判读（2026-10-19 收尾）**：**三条跟进臂全部失败**（`EXPERIMENT.md` exp-081/084）：
     **W**（align + 语义 λ=0.02）**四项回译全显著退化**（ΔB1 −1.84 / ΔB4 −2.37 / ΔROUGE −2.75 / ΔWER −7.36）；
     **X**（align + 文本微调 5e-5）只换来 **ΔWER −2.04 显著退化**、其余 n.s.；
     **Y**（align 权重 0.1→0.3）**四项全显著退化**（ΔB1 −2.50 / ΔB4 −2.62 / ΔROUGE −3.86 / ΔWER −1.98）。
     → R1 的正增益**不能靠叠语义项、开文本微调或加大权重来加强**，它自足于 align w=0.1 这一档；不再追此方向。

- [x] **P21 · 部署口径叠加：新最好训练臂 × gate k=4 —— 2026-10-18 已跑完，叠加饱和** ——
  `route_t_align_20k` 是新的回译最强臂，而 **D37** 的 gate k=4 是当前部署默认（对 prefix60k
  四项回译 + MRR 全显著为正、1× NFE）。两者机制不同（轨迹形状 vs 文本条件剂量），
  部署时必然同时生效，所以必须实测叠加是否**增益可加**还是**互相饱和/干扰**。
  - 队列 `reports/queue/jobs_align_stack_20261018.txt`（`--gpus 0,4,5,6,7 --max-parallel 2`，
    日志 `reports/queue/logs_align_stack/`），零训练：
    ① `route_t_align_20k_gate4`（k=4）；② `route_t_align_20k_gate4cfg30i200`（k=4 + CFG3.0[200,999]）。
  - 对照：`route_t_align_20k`（k=1，已有）、`prefix60k_gate4`（k=4，已有）。
  - 判据：D3 + D20；若叠加 ≈ 单杠杆 ⇒ 饱和（部署取 1× NFE 的 k=4）；若显著更好 ⇒ 更新部署口径；
    若回译涨而 MRR 显著掉 ⇒ 按 D20 不晋级。
  - **结果（2026-10-18，test 642，配对 bootstrap，EXPERIMENT exp-077）**：

    | 臂 | B1 | B4 | ROUGE | WER | FVD | MRR |
    |---|---|---|---|---|---|---|
    | `route_t_align_20k`（k=1，对照） | 12.18 | 3.91 | 12.66 | 100.78 | 26.00 | 0.0232 |
    | `route_t_align_20k_gate4`（k=4） | 12.03 | 3.59 | 12.09 | 100.12 | 26.92 | 0.0240 |
    | `route_t_align_20k_gate4cfg30i200`（k=4+CFG） | 10.62 | 3.08 | 10.61 | 100.14 | 31.26 | 0.0225 |

    - align+gate4 vs align k=1：ΔB1 **−0.13 [−1.24,+0.94] n.s.**、ΔB4 −0.32 n.s.、ΔROUGE −0.56 n.s.、
      ΔWER +0.67 n.s.、ΔMRR +0.0007 n.s. → **完全打平**。
    - align+gate4+CFG vs align+gate4：ΔB1 **−1.39 [−2.35,−0.46] 显著退化**、ΔROUGE **−1.47 显著退化**、
      ΔB4 −0.50 n.s.、ΔWER −0.02 n.s. → **加 CFG 反而显著更差**。
    - 长度分层（`tools/paired_arm_stats_by_length.py`）：叠加在短片 −0.82 n.s.、长片 +1.18 n.s.，
      两端都没拿到东西。
  - **结论**：**叠加饱和，且部署口径要分开**——`route_t_align_20k` 是当前最强训练臂，**它自己已经吃掉了
    头部空间**，再叠 gate k=4 无增益、再叠 CFG 显著退化；而 k=4 只对**未对齐的 prefix60k** 有效
    （D37 结论不变，但适用范围要写明）。部署建议：**对齐臂用 k=1，未对齐的 prefix60k 用 k=4**。
  - 这条与 **TODO P25** 的长度分层诊断互相印证：两个机制不同的杠杆都在短片拿增益、都在长片失效，
    叠加时自然抢同一份预算。

- [x] **P16 · 跨臂相关性体检：MRR / B1 / FVD 三者谁在解释谁（零成本，2026-10-18 已算）**
  - 做法：把已完成的 **44 条** test-642 臂的 corpus B1、检索 MRR、FVD 拉到一个表里算 Pearson。
  - **结果**：MRR↔B1 **+0.781**（强正，检索确实与回译同向）；FVD↔B1 **−0.648**；
    FVD↔MRR **−0.492**（FVD 越低越好，故都是"更好同向"）。
  - **读法**：这说明 **D3 门槛并非无信息**——跨 44 条臂它和 B1 强相关。但**在最好的几条臂之间**
    它会饱和（前 4 条臂 MRR 0.0305–0.0336 挤在一起，同时 B1 从 11.44 拉到 14.30）。
    所以 P9 的"门槛饱和"要限定为**顶部队列内部**，不能推广成"MRR 与质量无关"（否则会与这条
    +0.781 冲突）。**idea-029 的判据应据此收紧**：新尺子要证明的是"在顶部 5 条臂之间还能分开"，
    而不是"整体上比 MRR 更相关"。
  - 产物：无独立文件（一次聚合查询，可复算）；本条结论供 NOTES/idea-029 裁决引用。

- [x] **P22 · idea-035（噪声调度 cross-attn gate）执行与判读** —— **2026-10-18 已实现、已开跑；2026-10-19 判读：无效（FAIL-039）**。
  - 依据：**D37** 的常数 k=4 是当前最强最省的推理期杠杆（四项回译 + MRR 全显著为正、1× NFE），
    **P12** 已证明它与 CFG3.0[200,999] 叠加饱和（两条杠杆在 k=4 之上不再互补）。于是"同一根旋钮
    还能不能再榨"只剩**形状**这一个自由度。P7 探针（`reports/eval/probe_text_response_by_timestep_prefix60k.json`）
    实测同一 x_t 下换文本的相对输出变化**随噪声阶段单调上升**（rel 0.127@t=100 → 0.229@t=900，
    max/min **1.80×**、两端 bootstrap CI 不重叠）→ 模型在高噪声步对文本更敏感，常数 k 在两端
    施加同样推力并非最优。
  - 实现：`GenerationRequest.cross_gate_schedule=(k_lo,k_hi)`；`TextToLatentPipeline.generate` 在
    每次前向前把 `k(t)=k_lo+(k_hi-k_lo)·t/(T-1)` 写进 DiT 的 `cross_gate_scale` 属性
    （**无新参数、不进 checkpoint**）；`--cross-gate-schedule K_LO K_HI` 接到 `tools/eval_t2v.py`
    与 `tools/generate.py`，`meta.json` 记录调度。**不给调度时逐位退回常数路径**（历史数字不变）。
  - 自检：`tests/test_cross_gate_schedule.py`（端点精确、单调、常数路径不变、gate 改动确实改变前向）；
    `tests/run_all.py` 全绿；8 步端到端冒烟跑通、`meta.json` 正确记录 `[1.0, 8.0]`。
  - 队列 `reports/queue/jobs_gate_schedule_20261018.txt`（`--max-parallel 1`）：**S1** k=(2,6)
    （缓坡，平均 ≈4，与 k=4 **同平均剂量**，是关键对照）与 **S2** k=(1,8)（陡坡，平均 ≈4.5）。
  - 判据（预注册）：协议 P + **D3/D20**；**必须显著优于同平均剂量的常数 k=4**（对照
    `prefix60k_gate4`：B1 12.75 / B4 3.55 / ROUGE 12.58 / WER 101.75 / MRR 0.0313）才算通过；
     若只打平 ⇒ 调度自由度无增益，部署维持常数 k=4（D37 不变）。须报告 k_lo/k_hi 与平均 k。
   - **判读（2026-10-19 收尾）**：**无效，出池（FAIL-039）**（`EXPERIMENT.md` exp-079）。S1(2,6)（与 k=4 同平均剂量）
     B1 12.38/B4 3.21/ROUGE 12.26/WER 102.18/MRR 0.0313；S2(1,8) 12.26/3.21/12.22/100.97/0.0304。
     S1 vs k=4 与 S2 vs k=4 全部 n.s. 且方向为轻微负 → 调度形状无增益，部署维持常数 k=4（D37 不变）。

- [x] **idea-033（对齐损失的噪声阶段门控，Z/Z2）—— 2026-10-19 已跑完，verdict：无效（FAIL-036）**。
  Z（t≤900）四项回译**全显著亏**（ΔB1 −2.19 / ΔB4 −2.77 / ΔROUGE −3.55 / ΔWER −3.53）；
  Z2（t≤700）四项全 n.s.（打平）。改 R1 的施力噪声窗无收益。产物
  `reports/eval/matrix/route_z_align_gate900_20k/test_n642`、`.../route_z2_align_gate700_20k/test_n642`；
  见 `FAIL.md` FAIL-036、`EXPERIMENT.md` exp-082。

- [x] **P23 · idea-034（原生帧率对齐窗口）执行与判读** —— **2026-10-19 已跑完，verdict：无效（FAIL-037）**。
  **结果**：`route_aa_align_window_20k` B1 10.09 / B4 1.13 / ROUGE 9.01 / WER 104.72 / FVD 27.59 /
  MRR 0.0206（1.87×）；vs R1 四项回译**全显著亏**（ΔB1 −2.09 / ΔB4 −2.78 / ΔROUGE −3.65 / ΔWER −3.95）。
  原生窗口未修好长片、整体显著退化。见 `FAIL.md` FAIL-037、`EXPERIMENT.md` exp-083。
  - 依据：exp-070 长度分层——R1（idea-024）的增益**全部来自短片**（≤128 帧 ΔB1 **+2.78 显著**；
    >128 帧 ΔB1 −0.44 n.s.）。机制：`TemporalAlignLoss(max_frames=64)` 对长 clip 做**均匀跨步**
    降到 64 帧（84.9% 的 train clip 受影响，平均 1.96×、最大 3.98×）→ 一个 4 帧手势被压成
    1 个 token，对齐损失看到的轨迹被**时域混叠**抹平。
  - 实现：`TemporalAlignLoss(window=True)` —— 长 clip 取一段**连续的原生帧率窗口**（长度
    `max_frames`）而非跨步采样；窗口起点由**私有 `torch.Generator`** 抽取，**不消耗全局 RNG**，
    因此不会移动对照臂 `route_t_align_20k` 的噪声流、单变量对照保持干净。配置键
    `align_loss.window`；`window=False` 为历史跨步路径、**逐位不变**。
  - 自检：`tests/test_temporal_align_window.py`（默认不变、短片两路一致、窗口路径值不同、
    **私有 RNG 不扰动全局流**、起点范围与随机性、`max_frames=0` 仍有限）；`tests/run_all.py` 全绿；
    30 步冒烟跑通（`window=True` 生效、无 NaN、`train/align` 正常）。
  - 臂 `route_aa_align_window_20k`：从 `final_60k.pt` 热启动 20k、同权重 0.1、文本冻结（D34 声明），
    已并入 `reports/queue/jobs_align_followups_20261018.txt`（复合命令：训练完自动接 test-642 评测）。
  - 判据（预注册）：对照 `route_t_align_20k`（同起点、同预算、同权重，唯一变量 = window）；
    **必须在长度分层上判读**——长片（>128 帧）ΔB1 显著为正、短片不显著退化；全局 D3 + D20 照旧。
  - 读法提醒：若长片不涨，说明长片瓶颈不是对齐分辨率（候选解释：train 里 >128 帧占 34.0%、
    test 只占 22.9%，是覆盖率问题）→ 应转到数据侧，而不是继续调损失。

- [x] **P24 · idea-037 前置：对齐损失的 warp 预算性质探针** —— **2026-10-18 已跑完，verdict PASS**。
  - 依据：exp-073 的交叉诊断留了一个**未解释的读数**——显式帧间差损失（route_j，FAIL-020）
    **没能**把幅度学回来（delta_rms 比 0.3056 < 基线 0.3665），而软-DTW 对齐（R1）**抬到 0.4041**。
    两者都"比较帧间结构"，差别只在**软-DTW 允许 warp**（时间伸缩）→ 立 idea-037 测这个自由度。
  - 做法：先用**只读探针**（`tools/probe_align_band_budget.py`）在真实 train latent 上扫
    `band_ratio∈{0.05,0.1,0.25,0.5}`，对 correct / speed±15% / reversed / static 五变体读
    `TemporalAlignLoss` 的残差，算 separation = min(退化) / max(轻微)。
  - **结果**（train 48 条）：0.05 → mild 0.0612 / separation **24.5×**；0.1 → 0.0284 / **52.2×**；
    0.25 → 0.0274 / **53.8×**；0.5 → 0.0274 / 53.8×。**verdict PASS**——窄档对轻微变速的惩罚
    是宽档的 **2.2×**，而退化残差在所有档都保持高位（separation 全 >24×）。
  - 实现：`banded_soft_dtw(..., band_ratio)` + `TemporalAlignLoss(band_ratio)` + 配置键
    `align_loss.band_ratio`；`band_ratio=0.25` **逐位复现历史带宽**（`// 4 + 2`）。
    `tests/test_temporal_align_band.py` 覆盖默认不变、band 外为 +inf、非法值报错、宽档对 warp 更宽容；
    `tests/run_all.py` 全绿。产物 `reports/eval/align_band_budget.json`、`EXPERIMENT.md` exp-074。
   - **后续臂与判读（2026-10-19 收尾）**：两档等预算臂已跑完（窄 0.05 / 宽 0.5，对照 `route_t_align_20k`）：
     窄(0.05) B1 12.01/B4 3.73/ROUGE 12.46/WER 101.46/MRR 0.0235/幅度比 0.4176（**四项全 n.s.**）；
     宽(0.5) 10.06/1.33/9.12/103.98/0.0218/0.2349（**四项全显著亏**）。**无效，出池（FAIL-040）**：
     窄档打平、宽档显著退化，且幅度比 0.05→0.25→0.5 = 0.4176→0.4041→0.2349 **非单调** → warp 预算不是
     R1 的活性成分。见 `EXPERIMENT.md` exp-085。


- [x] **P25 · 长度分层的共性诊断：两个杠杆都在长片上失效 —— 2026-10-18 已跑完**
  - 做法：用新工具 `tools/paired_arm_stats_by_length.py`（复用 `paired_arm_stats.py` 的可加分解，
    只改重采样总体）把 test-642 按 oracle 长度切成 `<=128`（495 条）/`>128`（147 条）两层，
    在**每层内部**做配对 bootstrap。工具已用 exp-070 复算自检（+2.77 vs 记录 +2.78，CI 一致）。
  - **结果一（R1 / idea-024 对齐损失 vs 等预算续训）**：`<=128` **ΔB1 +2.77 [+1.01,+4.57] 显著**、
    `>128` **−0.45 [−2.33,+1.31] n.s.**；绝对水平 短片 10.97→13.75、长片 9.48→9.03。
  - **结果二（gate k=4 / idea-016 vs k=1，本轮新增）**：`<=128` **ΔB1 +2.81 [+1.44,+4.23] 显著**、
    `>128` **+0.83 [−0.74,+2.50] n.s.**；短片 11.25→14.05、长片 9.09→9.92。
    阈值稳健性：th=96 时 +3.15 / +1.54；th=160 时 +2.00 / +1.13——**长片档在所有阈值下都显著更弱**。
  - **结果三（反例对照：语义损失 λ=0.02 vs 同起点无语义损失）**：`<=128` +1.09、
    `>128` **+3.14** —— **长片反而增益更大**。即"长片学不动"**不是**所有杠杆的通病。
  - **读法**：两个**机制完全不同**的杠杆（轨迹形状 vs 文本条件剂量）在长片上同时失效，
    而 GT 重建上限在两层上几乎持平（32.54 / 31.29）→ 长片差距**不是表示上限**。
    结合帧数分布实算（train 去掉 >256 后 7021 条：`>128` 占 **35.0% 样本**但占 **50.4% 帧数**），
    指向"**按片段平均的损失把长片梯度权重压低了**"这一候选解释 → 立 **idea-039**。
  - 产物：无独立文件（可复算）；证据行供 idea-039 与 NOTES 引用。

- [x] **P26 · idea-038（EMA 影子权重读数）执行与判读** —— **2026-10-18 已实现、2026-10-19 用户裁决：全臂部署默认切 EMA（D46）**。
  - 依据（本轮只读实算）：每个 checkpoint 都存 `model` + `ema` 两套 state_dict；
    两套权重**确实不同**——prefix60k 全局相对 L2 **0.1046**（逐张量最大
    `blocks.3.cross_adaLN_modulation.1.weight` rel **0.3722**）、route_t_align_20k 0.1025、
    route_s_cont20k 0.1035；而 `reports/eval/matrix` 下 **53 条 test-642 臂的 `meta.json`
    全部 `use_ema=false`** → 历史数字全是原始权重口径，EMA **从未被评过**。
  - 队列 `reports/queue/jobs_ema_zerotrain_20261018.txt`（`--gpus 0,4,5,6,7 --max-parallel 1`，
    日志 `reports/queue/logs_ema/`），零训练、零 NFE，只切 `--ema`：
    ① `prefix60k_ema`（k=1）；② `prefix60k_gate4_ema`（k=4）；③ `route_t_align_20k_ema`（k=1）。
  - 判据（预注册）：**同臂**配对 bootstrap（`tools/paired_arm_stats.py` 回译四项 +
    `tools/paired_mrr_stats.py` MRR）；至少一项显著改善且无任一项显著退化 → 免费杠杆成立、
    部署默认切 EMA；否则记"EMA 与原始权重等价"入 NOTES（**不是 FAIL**，它没否掉任何机制假设）。
  - **限制**：文本微调过的臂（checkpoint 含 `text_encoder` 键）不支持 EMA（pipeline 显式报错），
    本卡只覆盖文本冻结的臂。
   - **口径提醒**：EMA 与原始权重**不得混表**（换权重=换读数），引用历史数字时须注明是原始权重口径。
   - **判读（2026-10-19 用户裁决）**：**采纳为全臂部署默认——切 EMA 影子权重（D46）**（`EXPERIMENT.md` exp-078）。
     对齐臂 `route_t_align_20k` → EMA **B1 12.18→13.51 / B4 3.91→4.66 / MRR 0.0232→0.0312**（ΔB1/ΔB4/ΔMRR 显著，WER 无显著退化），
     成新回译最强臂；未对齐 `prefix60k` k=4 → EMA B1 12.75→13.38 / B4 3.55→3.71 但 **ΔWER −3.30 显著退化**；k=1 回译持平、仅 MRR 显著涨。
     用户明确对**所有臂**切 EMA，接受未对齐臂的 WER 代价（与 D35 同类"显式权衡后采纳"）。落地：`tools/eval_t2v.py` / `tools/generate.py` 默认
     `use_ema=True`，`--raw`/`--no-ema` 回退；**历史 53 条 test-642 数字全是 raw 口径，引用须标权重口径**。

---

## 决策登记（D，只增不删；ID 不复用）

> 每条给"结论 + 出处"。历史数字与完整论证见所引文件；本表不再重复叙述。

- **D42 idea-038（EMA 影子权重读数）立项（2026-10-18，由本轮只读实算驱动）**：
  训练脚本每步都维护 EMA 影子权重并写进 checkpoint（decay 0.9999），评测脚本有 `--ema` 开关，
  但 **53 条历史 test-642 臂全部用原始权重**（`meta.json` 全为 `use_ema=false`）。
  本轮只读实算证明两套权重不是同一个函数：prefix60k 全局相对 L2 **0.1046**、逐张量最大
  `blocks.3.cross_adaLN_modulation.1.weight` **rel 0.3722**（文本闸门本身差 37%）；
  route_t_align_20k 0.1025、route_s_cont20k 0.1035。故"没差别"这个默认假设**未经验证**，
  值得花 1.5h/臂直接读。**零训练、零 NFE**，队列 `jobs_ema_zerotrain_20261018.txt` 已开跑
  （3 条臂）。判据见 TODO **P26**；**不是**新机制，是口径补全。

- **D43 idea-039（按帧数加权的训练损失）立项（2026-10-18，由长度分层共性诊断驱动）**：
  本轮零成本诊断（TODO **P25**）发现两个**机制完全不同**的杠杆——R1 软-DTW 对齐（idea-024）
  与 gate k=4（idea-016）——增益**都几乎只在短片**：`<=128` ΔB1 +2.77 / +2.81（均显著），
  `>128` −0.45 / +0.83（均 n.s.）；阈值 th=96/128/160 三档结论一致。而 GT 重建上限在两层上
  几乎持平（32.54 / 31.29）→ 长片差距**不是表示上限**。反例对照：语义损失 λ=0.02 在长片上
  **反而增益更大**（+3.14 vs +1.09），说明这不是所有杠杆的通病。
  帧数实算（train 去掉 >256 后 7021 条）显示 `>128` 的片占 **35.0% 样本**却占 **50.4% 帧数**，
  而现役 x0-MSE 是**按片段平均**（`_masked_mean` 先帧维平均、再 batch 平均）→ 长片的梯度权重
  被系统性压低。据此立 **idea-039**（改损失归一化，唯一变量；对照 `route_s_cont20k` 等预算版本）。
  **未排训练**（用户要求省资源），待用户选出；判据见 `THINK.md` idea-039 卡（协议 P + D3/D20 +
  **必须长度分层**）。

- **D44 研发探索规范（2026-10-19 用户指令）**：确立六条执行纪律 ——（1）**少开探针、多跑真实
  训练/评测任务**（探针只在"不先验证前提就会整轮白跑"时立，能靠小训练臂/现有产物评测回答的就
  不立探针）；（2）**允许从头训练、不强制热启动**（热启动降为省成本可选项）；（3）**鼓励高收益/
  创新路线**（可提 L2/L3 大胆设想，由用户筛选）；（4）**大改动走 git 分支、分支内可放手重构**；
  （5）**北极星指标 = SignMbart 回译 BLEU-4 ≥ 10**（test 642 单侧；当前最好生成臂 B4≈4.7，
  重建上限 13.58，GT 上限约 22.7）；⑥ agent 执行 idea/todo 时**兼管 git tree**（主题分支 +
  意图单一的 commit + `runs/reports` 不入库 + 未采纳分支保留不删）。
  规范正文 `RULES.md` 研发探索规范第 15–20 条；与 D27（探针优先/训练暂缓）取向不同处以本条为准。

- **D45 约束 4 修订：正式开放参考帧条件（"方法 4"）（2026-10-19 用户指令）**：生成条件由
  "纯文本"改为 **文本 + 参考帧 frame 0**（`conditioning.reference.enabled: true`，`ref_features
  [128,512]` 进 cross-attn context + adaLN 全局向量）。部署接口变为 **`text + frame0 → video`**；
  原"纯 text→video"降级为可选消融口径，历史数字不混表。仍禁止 pose/gloss/光流。规范正文
  `RULES.md` 方法硬约束第 4 条（2026-10-19 改写）；`AGENTS.md`/`DESIGN.md`/`STRUCTURE.md`/`README.md`/
  `THINK.md` §0H 已同步。**动机**：症状 C（生成与原始视频差很远）的主因是 CVAE 解码需要锚点、
  而 DiT 看不到锚点导致 `(z, ref)` 不自洽。

- **D46 部署口径切换为 EMA 影子权重（2026-10-19 用户指令，全臂）**：用户裁决**所有臂**的部署/评测默认
  改用训练脚本每步维护的 EMA 影子权重（decay 0.9999），取代此前"默认原始权重（raw）"的口径。
  实现：`tools/eval_t2v.py` / `tools/generate.py` 的 CLI 默认 `use_ema=True`（`--raw` / `--no-ema` 回退到 raw），
  `meta.json.use_ema` 记录实际口径；库层 `TextToLatentPipeline` 构造器默认不动。
  依据 exp-078（test 642、同臂同协议、配对 bootstrap）：对齐臂 `route_t_align_20k` → EMA **B1 12.18→13.51、
  B4 3.91→4.66、MRR 0.0232→0.0312**（ΔB1/ΔB4/ΔMRR 均显著，WER 无显著退化），成新回译最强臂；
  未对齐 `prefix60k` k=4 → EMA B1 12.75→13.38、B4 3.55→3.71，但 **ΔWER 101.75→105.05 显著退化**；k=1 → 回译持平、仅 MRR 显著涨。

- **D47 下一步押注切换到"latent 保真度"轴（2026-10-19 立项，由本轮只读实测驱动）**：
  9 条"改条件使用方式"的机制类改动已全部零收益或显著为负（NOTES D 节），三个推理期免费杠杆
  也已饱和（exp-069 / exp-077），因此下一问必须换轴。本轮只读实测（`tools/probe_latent_error_percentiles.py`，
  登记 exp-088）给出换轴的**直接证据**：11 条臂的 latent 相对误差 `||ẑ−z*||_F/||z*||_F` 全部落在
  **0.44–0.51 的窄带**内，且**比"把时间平均帧复制满整段"的零运动预测器（0.4147）还差**；
  同时臂的 `cos` 高达 0.88–0.91、`amp_ratio` 只有 0.69–0.87 → **误差的主体是欠弥散（幅度缩了），
  方向大体是对的**。这解释了"BLEU 上不去"为什么和"条件用不狠"无关：条件用法的改动改变的是方向，
  而当前误差的主要成分在幅度。**代价**：这条轴要求先把"误差多大才致命"测出来（否则压保真度可能白压），
  故先立 P28 容差曲线做前置；判据与成本见 P28。**未经验证的部分**：容差曲线的拐点位置仍是未知，
  且 B4 与 rel_err 只在 11 条臂上呈现**方向一致但非严格单调**的关系（例：`prefix_120k` rel_err 0.4865
  却 B4 3.18 > `route_t_align_20k_gate4` 的 3.59 时 rel_err 0.4967），因此本决策是**押注**不是结论。

- **D48 EMA 与其它推理杠杆的交互必须实测、不得外推（2026-10-19 记录）**：
  exp-077 在 **raw** 对齐臂上得到"k=4 叠加饱和、k=4+CFG 显著更差"，是对齐臂内部的结论；
  EMA 改变的是同一个臂的权重工作点（exp-078：ΔB1 +1.33 / ΔB4 +0.75 显著），
  因此"EMA 之上再叠 k=4/CFG"**不能**由 exp-077 外推得到，也不能由"三个杠杆各自为正"外推得到
  （加法外推与实测的饱和现象直接冲突，exp-069 就是加法外推被实测否掉的先例）。
  据此立 P27 做三档单变量实测；本决策本身不含新数字，只规定**外推禁令**：
  任何"合起来应该更好"的论断在得到配对实测之前不得写进 NOTES。
  **必须随行注明的代价与限定**：① EMA **不是无条件免费**，未对齐 k=4 档有显著 WER 代价，用户显式接受
  （同类 D35"权衡后采纳"）；② **历史全部 test-642 数字（53 条臂）都是 raw 口径**，与 EMA 数字**不得混表**，
  引用历史值须标"raw 口径"、引用新值须标"EMA 口径"；③ 文本微调过的臂（checkpoint 含 `text_encoder`）不支持 EMA。
  来源：TODO **P26**、`EXPERIMENT.md` exp-078、`NOTES.md` 2026-10-19。

- **D50 默认只出方案，不动手（2026-10-19 用户指令，约束 agent 自身，优先级最高）**：
  **未被明确授权时，agent 只输出设计方案供用户审阅**，不得改 `t2v/` `tools/` `configs/` `tests/`
  下代码、不得新建脚本、不得启动训练/评测作业、不得下载大体积权重、不得删除已有产物。
  - **方案模式（默认）**：触发词 = 分析 / 评估 / 看看 / 调研 / 设想 / 讨论 / 你觉得 / 怎么改 / 提 idea。
    只读（读代码、读日志、读指标、跑不写仓库且不占 GPU 的只读探针）+ 只写 `.trae/documents` 下的方案文档；
    交付 = 改动点（文件:行 + 张量形状）/ 判据 / 成本（GPU·h）/ 风险 / 替代方案，**到此停下**。
  - **执行模式**：触发词 = 写 / 改 / 实现 / 跑 / 训练 / 执行 / 直接做 / 按这个做。
  - **授权不延伸**：「评估 X」≠「实现 X」≠「训练 X」；一个任务里拿到的授权只在那件事上有效，
    需要下一段时**明确问一句**，不默认继承、不以"上次做过""这是显然的下一步"作依据。
  - **执行中修自己刚引入的 bug 不算越权**（属同一任务）。
  - **破坏性操作额外一道闸**：删文件、清 `runs/`、改 `RULES.md`/`AGENTS.md`、改冻结契约、覆盖已有产物，
    **即使已授权**也要先报「动什么 / 为什么 / 影响范围 / 回滚方式」再动。
  - 触发本条的起因：2026-10-19 用户只要求"评估 Wan2.1-1.3B 做骨干的方案"，agent 自行推进到
    改代码、下 27 GB 权重、起 2 万步训练。产物未浪费，但判断权被抢走。
  - 条文全文 = `.trae/documents/RULES.md` 开头「授权边界规范」；`AGENTS.md` 第一条同步写入。

- **D51 参考接线硬校验支持 dict config + 旗舰系只有 raw 口径（2026-10-19，由评测队列失败现场驱动）**：
  两处修正：① `t2v/pipelines/generation.py::_ref_as_modulation` 原先只走属性访问，读 checkpoint 里存下来的
  **纯 dict** config 时静默退化成 `False`，于是 `flagship_ref_v2`（`reference.as_modulation: true`）的 raw/k4 全部
  被 `_check_ref_wiring_matches` 硬校验拦下、**从未成功加载过**；改为 dict 与 OmegaConf 都能读（已复核 v2 → True、v1 → False，两侧校验均 PASS）。
  ② 与 D46 自身限定③一致：**文本微调过的臂（checkpoint 含 `text_encoder`）没有 EMA 口径**，旗舰 v1/v2 的 `_ema`
  评测作业取消；旗舰系的 v1↔v2 单变量对照只能在 **raw** 口径下做，且 v1/v2 的 k=4 档也固定 `--raw`。
  受影响产物：`reports/queue/jobs_flagship_v2_eval_20261019.txt`（两条 `_ema` 取消、两条 k=4 加 `--raw`）；
  `EXPERIMENT.md` §6 已加口径行；`t2v/pipelines/generation.py` 的修复已入 git。

- **D49 预测目标放宽为 x0 或 velocity，二选一且必须显式声明（2026-10-19 用户修订第 3 条）**：
  原第 3 条硬约束"必须 x0-prediction"改为**允许 flow-matching / rectified-flow**
  （`prediction_type: velocity`），默认仍 x0。理由：exp-064 测得 t≥500 段占 loss 权重 97.94%、
  t=950 单独就占 75.35%，说明高噪段信号极稀，x0 目标在该段梯度被稀释；velocity 目标在
  flow 参数化下对高噪段是良态的。**代价与限定**：① **换目标 = 换口径**，velocity 臂的 loss
  不得与 x0 臂的 loss 横比，采样器也不同（velocity 用 FlowMatchEuler/UniPC，不得与 DPM-Solver++ 臂混表直比）；
  ② 跨口径比较必须另跑等预算"同骨干 + x0"对照；③ latent 仍是冻结 VAE 的 **1024 维/帧**，不放宽；
  ④ 每条臂的 `prediction_type` 必须写进 config 并在 EXPERIMENT.md 备注列注明。
  本次改动局限在 `wan-backbone` 分支，未在主干合并前不改变历史臂的可比性。

- **D1 采样口径**：默认 DPM-Solver++(2M) / 50 NFE，替换 DDIM；旧表按新口径整表重算。
- **D2 评测矩阵**：对比只在 **test 侧**、默认全量 642、固定种子；**不用 dev 筛选**。（2026-10-15 由 D26 修订，取消双集配对。）
- **D3 筛选主指标**：检索 **MRR / top-1** 是晋级门槛，SignMbart 与 FVD 只作汇报。门槛写成"相对该侧零信息对照的倍数"：≥1.5× 逃出地板、≥2.5× 有实质检索能力。
- **D4 部署预算**：原定 prefix-60k → **已被 D19 取代**（保留 ID）。
- **D5 表示层**：不触碰冻结 VAE，接受表示上限（B1≈31.6 / FD≈28.4）；WLASL 绕行路线关闭。
- **D6 pad-mask**：真实病理，但等预算对照无收益 → 只作省算力选项，不再投入（`configs/dit_b_text_prefix_padmask.yaml` 保留）。
- **D7 解码锚点**：所有生成数字含 GT 首帧解码锚点；不同锚点不得混表。
- **D8 知识库四态流转**：中间产物入 `.trae/documents/`；THINK → TODO → FINISH → NOTES/FAIL。
- **D9 评测执行口径**：新评测走 `tools/eval_matrix.sh`，固定清单 `reports/eval/samples/{test,train}_n642_seed42.txt`；dev 只留历史、不作选点。
- **D10 长度头选择**：改用 train 内部固定种子 holdout（15%）选点；部署 readout 在 test 642 裁决。
- **D11 语义目标可达性**：冻结 matcher 上语义信号确实存在（生成 latent gap +0.205）但受 matcher 判别力限制 → **不再加语义权重**。
- **D12 REPA 口径**：teacher 固定为项目内冻结 `runs/latent_text_teacher_v1/validated.pt`；projector 仅训练期、不进 checkpoint。
- **D13 运维**：队列磁盘守卫（输出盘 <12 GiB 暂停起新作业）；评测帧跑完即删；队列用 `setsid` 起；训练 `--checkpoint-every 0`（中途不存 ckpt，只写 `final.pt`）；FVD 真视频特征缓存 `--ref-cache`。
- **D14 长度头部署**：默认 head = **8 桶 + `expectation` 回读**（`runs/length_model_text/length_head.pt`）；predicted-length 列已复核有效（闭环成立）。
- **D15 调研/思考类提问默认不落盘**：只在对话回答；要留存须先问用户。
- **D16 回答可读性**：平实用词、术语先解释、"是什么→为什么→怎么起作用"，解释深度与复杂度成正比。
- **D17 采样侧 CFG 部署口径**：默认 **CFG=3.0、仅高噪声段 t∈[200,999]**；**禁止低噪声段单独加 CFG**（FAIL-022）。
- **D18 训练侧 text_lr**：**5e-5** 为验证有效配置（route_g）；1e-4 已测、不采用（WER 显著更差）。
- **D19 部署预算点**：**prefix-120k 为当前最强训练结果**（B1 12.55 / MRR 0.0336）；route_g（80k 等效）以 1/3 成本打平。取代 D4 的"60k 最佳"。多 seed 配对（TODO 11）仍未做。
- **D20 MRR 门槛的使用限制**：MRR 必须**同时报回译三项**；MRR 过门槛但回译任一项显著退化 → 不晋级、记 FAIL（教训 FAIL-021）。
- **D21 容量对照预算**：idea-011 宽/深臂改为从零训 30k、共用 `runs/dit_b_prefix30k/final_30k.pt` 参照；等步数不等 FLOPs。已出结论 FAIL-027。
- **D22 text_lr 上限试验口径**：idea-006/任务 7 的 1e-4 臂从 `final_60k.pt` 分叉 20k；mT5 中层臂改从零训 30k。
- **D23 同 N 锚点口径**：参照线一律在 `test_n642` 清单上重建（上限=重建缓存解码全链；地板=`+1` 错配真视频）；入口 `tools/run_anchor_controls_n642.sh`。
- **D24 教师可靠性前置**：教师升级须先过 `tools/probe_teacher_reliability.py` 两关；实测该教师偏好正确、拒绝静止，但**不能区分倒放/块置换** → 可作语义奖励，不可作顺序奖励。
- **D25 实验结果台账**：`EXPERIMENT.md` 一行一个 run×协议，**跑完即登记、行只增不删**；指标区由 `tools/collect_experiment_table.py` 自动生成、禁止手改。
- **D26 评测改为测试集单侧**：取消双集配对规范；只跑 test、不留 train 对比列（已跑的 train 产物只作过拟合诊断）。取代 D2 的双集部分。
- **D27 执行优先级**：活跃队列只服务未验证 idea；探针/复核类暂缓并标 `[暂缓]`（不删不判死）。
- **D28 队列 v5**：四个免训练前置探针通过后，队列重写为 5 训 + 5 test 评测（`route_o/q/r`、`route_p/slotsctl`）。**2026-10-16 已全部出结论**（FAIL-023~026）。
- **D29 SignMbart 可作训练奖励**：不在硬约束第 4 条列举之列故允许；但 pose 派生监督仍禁止；用其训练须标注同源自证风险，晋级仍以 D3 检索门槛为准。
- **D30 新臂默认冻结文本编码器**：新开训练臂 `conditioning.text` 一律冻结；只读换层不算微调；例外须用户单独放行。**⚠️ 已于 2026-10-17 被 D34 取代（用户撤销该限制）；条目原文保留不改。**
- **D31 idea-016 收口口径**：仅剩的免训练推理期杠杆作本轮收口，**不再新开任何训练臂**。
- **D32 idea-009 直接取消**：用户指令取消，不验证、不保留候选；链路入 FAIL-029。复活须重新立卡、不复用编号。
- **D33 所有空闲 GPU 均可使用**：取消按"属于谁"划的 GPU 白名单，改为**按实时空闲**分配；不抢占别人在跑的作业；留够自身显存。规范正文 `RULES.md` 资源使用规范第 14 条。
- **D34 文本编码器微调重新允许（2026-10-17 用户指令，取代 D30）**：撤销"新臂默认冻结文本编码器"；新建训练臂可自由选择 `conditioning.text` 冻结或微调（全量微调/LoRA/部分解冻均可），文本微调按**正常实验变量**参与对照与晋级。唯一硬性要求 = 每条臂必须**显式声明**文本侧是否可训练（`trainable` 取值）与文本侧学习率；两侧状态不同的臂在没有声明该差异时不得互为对照。规范正文 `RULES.md` 方法硬约束第 6 条（2026-10-17 改写）。**连带重判**：`route_b_semantic_5k`（`trainable: true`）不再是"违规历史产物"，转为**正常实验**，结果入 `NOTES.md`（F-002 出池）。
- **D35 语义目标默认权重 λ=0.02（2026-10-17 用户指令）**：句向量语义辅助目标的默认权重定为 **λ=0.02**
  （取代 λ=0.05 的 `route_b_semantic_5k`）。依据：λ 消融（EXPERIMENT exp-060~062）——λ=0.005 全面退化；
  λ=0.01 无 WER 代价但 B1/MRR 较低；**λ=0.02 与 λ=0.05 四项统计等价**（全 n.s.），而 λ=0.02 的
  B1 **14.30** / MRR **0.0333**（3.04× 地板）为全表最高，WER 106.15 比 λ=0.05 的 106.60 还好。
  用户明确选择接受 λ=0.02 的显著 WER 代价（ΔWER −1.99 vs 无语义损失）以换取最高 B1/MRR。
  **注意**：按 D20 该臂有显著回译退化，属"用户显式权衡后采纳"，不是自动晋级；汇报时必须带 WER 代价。

- **D36 λ=0.02 落地 + idea-026 排上（2026-10-17 用户指令）**：① 采纳 **D35** 的 λ=0.02
  作语义目标默认权重（取代 λ=0.05 的 `route_b_semantic_5k`）；② 把"下一个 idea" = **idea-026**
  （去噪阶段相关文本条件选择）排上执行，登记为 TODO **R2**。R2 按卡片预注册的**等参数配对**开两臂
  （u=`dynamic` / v=`static`，起点 60k、20k 步、文本冻结、各 +1,774,080 参数），
  不启用语义目标以保持单变量对照；机制成立须 u 显著优于 v。实现与自检见 TODO R2。

- **D37 idea-016 gate 缩放 k=4 采纳为部署默认（2026-10-18 用户指令，裁决 TODO N6 选项①）**：
  `prefix60k` 上 k∈{2,4,6.6} 全测（exp-047/048），**k=4 最优**——相对 k=1 四项回译 + MRR
  全部显著为正（B1 12.75 / B4 3.55 / ROUGE 12.58 / WER 101.75 / MRR 0.0313，2.85× 地板），
  **1× NFE**（gate 只在原前向里乘系数）；与等剂量 CFG3.0[200,999]（2× NFE）回译打平、
  MRR 显著更高（+0.0093，exp-052）。落地：`configs/dit_b_text_prefix.yaml` 的
  `sampling.cross_gate_scale: 4.0`；`tools/generate.py` / `tools/eval_t2v.py` 的
  `--cross-gate-scale` 默认改读 config（显式传参可覆盖，传 1.0 复现历史数字）。
  **口径**：历史数字全部是 k=1 口径，引用时须注明；D17（CFG3.0 仅高噪声段）不撤销、
  降为备选；**k=4 与 CFG3.0[200,999] 叠加未测**。依据 `NOTES.md` 2026-10-18、`FINISH.md` F-012 出池。

- **D38 route_q 改判"回译侧有效"、idea-002 复活（2026-10-18 用户指令，裁决 TODO N7 选项②）**：
  干净等预算对照 `route_s_cont20k` 下 route_q **ΔB4 +1.07 / ΔROUGE +1.29 显著正**，
  ΔB1/ΔWER 方向偏好不显著，ΔMRR +0.0001 不显著（`FAIL.md` FAIL-025 补记）。
  **FAIL-025 的否决结论撤销**（条目与链路保留存档，加补记二）；**idea-002 复活**，
  重回 `THINK.md` 活跃池（索引+卡片已重建），后续重启/扩预算/更强实现按协议 P 重新预注册；
  与 idea-029（细时间结构尺子）一并解读"回译正、MRR 不动"。依据 `NOTES.md` 2026-10-18、
  `EXPERIMENT.md` exp-037/042。

- **D39 idea-028 不立项 / idea-031 降级 / 两条新推理杠杆保留（2026-10-18，由 P10 + P17 判读）**：
  P10（事后整体放大，192 条 × s∈{1,1.5,2}）判据③成立 → **否决"事后放大"**：
  λ=0.02 臂 B1 15.06(s=1) → 10.47(s=1.5) → 3.55(s=2)，FVD 29.96 → 205.65 → 332.50；
  **GT 重建对照 s=1 给 32.33（≈ 上限 32.36）、s=1.5 崩到 6.40 / FVD 333.37** →
  判据②不成立、探针本身有效。结论：**运动幅度不足不能靠"乘系数"修**——放大把轨迹推出冻结 VAE
  可解码流形。连带：**idea-028（弥散校准 x0 目标）不立项**（P14 结清）。
  另由 P17（32 视频 × 11 个 t × 8 seed）：loss 确实被高噪声步压倒性主导（t≥500 占 97.94%、
  t=950 单独占 75.35%），但那些步的**单步预测并不更平**（条件均值版 delta_ratio 在 t=950 为
  0.984、t=900 为 1.016）→ **idea-031（min-SNR 加权）前提只成立一半、降级为候选**（不再优先排）。
  **不继承此否决的两条**（各自独立预注册、已开跑）：**idea-030**（DDIM η 注入，沿噪声表加方差，
  见 TODO **P19**）与 **idea-032**（采样阶梯按 log-SNR 均匀重排，见 TODO **P18**）——
  两者都不是"离流形整体缩放"。依据 `reports/eval/motion_amplify/`、`reports/eval/error_by_timestep_prefix60k.json`。

- **D40 idea-032 采样阶梯 log-SNR 均匀（2026-10-18 新立，零训练可部署）**：
  现役 50 步阶梯按 **t 索引**均匀（ADM respacing），但 DDIM / DPM-Solver++ 积分的变量是
  **半 log-SNR λ=log(α/σ)**。实测该阶梯在 λ 上极不均匀：**第一步 t=999→979 一步走掉 40.3%
  的 λ 行程**，t≥850 的 8 步占 55.1%，训练分布最密的 t∈[300,700)（logit-normal 下约 60% 样本）
  只占 4.8%，λ 步长最大/最小 **92×**；DPM++(2M) 的 `ratio_clip=(0.5,2.0)` 在这条阶梯上有
  **2 处**被截断（二阶退化为近似一阶）。改为 λ 均匀后步长比 3.96×、截断步 2→1。
  落地：`respaced_timesteps(steps, spacing)` + `sampling.timestep_spacing` +
  `--timestep-spacing`；**默认 `uniform` 逐位不变**（历史数字不受影响），未知值报错不静默回退。
  依据 `tests/test_dpm_solver.py` 新增两条单测、TODO **P18**、`THINK.md` idea-032。

- **D41 同一输出目录同一时刻只能有一个 runner（2026-10-18 运行事故后立的规范）**：
  `jobs_amplify2` 与 `jobs_amplify_gt` 两个队列把**同一条** GT 对照命令写进**同一个**输出目录
  （`reports/eval/motion_amplify/gt_recon/`），两条 `#done=` 守卫也相同 → 两个进程并行跑同一臂，
  而 `prepare_frame_dir` 会 **`shutil.rmtree` 目标目录**，会互相删帧、使 FVD/BT 读到不完整帧。
  处置：kill 重复 runner 与其子进程，只留一条；已写完的 s=1/s=1.5 分数自洽可用，半成品 s=2 弃用。
  **规范**：起队列前必须核对"同一 `#done=` 路径 / 同一 `--out` 目录"是否已被在跑的队列占用；
  不同队列不得写同一产物路径。参考 `tools/gpu_queue.py` 的 `#done=` 语义。

---

- **D53 · 文本微调臂只能以 raw 口径汇报，EMA 复评结构上不可行（2026-10-20）**：
  `t2v/pipelines/generation.py:231-233` 有硬守卫 —— 当 `conditioning.text.trainable: true` 时，
  `use_ema=True` 直接 `raise ValueError('fine-tuned text runs require consistent raw weights; EMA bundle unavailable')`。
  原因：EMA 影子权重只覆盖 DiT，不含文本编码器；用 EMA 的 DiT 配 raw 的文本编码器属于混口径。
  **后果**：`flagship_ref_v1` / `flagship_ref_v2`（两者都 `text.trainable: true`）**永久只有 raw 数字**，
  V1 的「EMA 复评」一项据此关闭；今后凡声明文本微调的臂，表中必须标 `raw`，不得与 EMA 臂混表。
  实测：`OUT_SUFFIX=_ema` 连试 3 次同一报错，见 `reports/queue/logs_pending2/console.log`。

- **D54 · `.trae/` 整个目录不进入 git 管理（2026-10-20 用户指令）**：
  理由：知识库记录方法演进与判断依据，必须**跨方法、跨分支全局唯一可查**。跟着 git 走有三个坏结果：
  ① 切分支时 `.trae` 被检出成旧版本，当前分支的 `exp-NNN` 看着凭空消失；
  ② 多分支并行时 `EXPERIMENT.md` / `TODO.md` 几乎每行都冲突；
  ③ commit 历史被纯文档改动污染，真正的方法改动被埋掉。
  落地（已执行并校验）：① `.gitignore` 由 `.trae/reference/` 改为 **`.trae/`**（整体忽略）；
  ② `git rm -r --cached .trae`，23 个文件退出跟踪、**磁盘内容不变**；
  ③ `git filter-branch --index-filter 'git rm -r --cached --ignore-unmatch .trae' --prune-empty -- --all`
  重写全部 4 个分支，**commit 39 → 26**（13 个纯 `.trae` 文档提交被剪除）；
  ④ `rm -rf .git/refs/original` + `git reflog expire --expire=now --all` + `git gc --prune=now`。
  **校验**：`git log --all -- .trae` = 0 行；`git ls-files .trae` = 0 个；遍历全部 commit 的 tree
  含 `.trae/` 者 = **0 个**；工作树 0 脏文件；`.trae` 的 12 个文档 + 1 个 TODO 仍在磁盘且内容未变。
  **备份与回滚**：`/tmp/text2video-pre-trae-purge-20261020.bundle`（`git bundle` 全量，含清理前 4 分支完整历史），
  `git clone <bundle> <dir>` 即可恢复。规范条文写在 `RULES.md` 第 0 节 + `AGENTS.md`「仓库边界」。
  ⚠️ 后续约束：改 `.trae` **不要** `git add`、不要为它单独 commit；追溯靠台账自身的 `exp-NNN`/`D-NNN`，
  备份靠仓库外副本（打包到 `/tmp` 或项目外），不与代码仓库共享 commit 图。

- **D55 · FVD 口径修复 + 此后只比 FVD（2026-10-21 用户指令）**：
  导师指出旧 FVD（22–26）异常；核查确认**不是官方 FVD**。官方口径（arXiv:1812.01717 +
  `google-research/frechet_video_distance`）= **Kinetics-400 I3D（Inception-v1）的 400 维
  logits**、16 帧、短边 resize 224 + center crop 224、像素 [-1,1]、float64 Fréchet；
  旧实现用 torchvision R(2+1)D-18 的 512 维 average-pool + ImageNet 归一化（有效 112²），
  换骨干=换特征空间，数字不可比、整体作废。文献手语域（PHOENIX14T）FVD 为几百
  （SignViP 1025.45 / SignGAN 1431.38 / SignGen 1715.32；FlashSign 453–597）。
  **落地**：① 重写 `tools/eval_fvd.py`（400-d logits，`--self-check` 实测 133.91 ≈ 官方 131，
  报告写 `protocol` 与 `real_split_half_fvd`）；② 新增 `tools/fetch_i3d_weights.py`
  （hf-mirror，sha256 `bec6519f…a5e2`）与 `tools/redecode_fvd_frames.py`（从 `latents.pt`
  只解码 16 个等距帧，CPU 可跑）；③ 新规范 `.trae/documents/FVD_SPEC_20261021.md` 生效；
  ④ 改写 `RULES.md` 方法硬约束 5 / 评测 6 & 6.1 / 研发 19，与 `AGENTS.md` 第 5/6/13 条：
  **此后所有实验唯一主对比指标 = 官方 FVD，回译不再要求、不再作为晋级指标**；
  ⑤ `EXPERIMENT.md` 新增 4.0：`prefix60k`(c9f28ea)、`route_t_align_20k`(ed05ea0)、
  `_ema`(ed05ea0)、`_ema_k4`(ed05ea0) 四条为**被认可基线**，其余全部 exp-* 归**探索实验**。
  **状态（2026-10-21 已完成）**：本 shell 无 `/dev/nvidia*`，基线 FVD 已在 CPU 上按官方口径
  全量重算并回填：`prefix60k` **178.48**、`route_t_align_20k` **184.68**、
  `route_t_align_20k_ema` **172.43**、`route_t_align_20k_ema_k4` **132.30**
  （`real_split_half_fvd = 74.17`，N=642）；`collect_experiment_table.py` 的 FD 来源已切到
  `fvd_i3d400.json`（旧 `fvd.json` 标 `⚠️R(2+1)D`）；旧 R(2+1)D 数字只留追溯。
  **后续（不阻塞本任务）**：GPU 恢复后抽一条臂用 CUDA 复跑，确认 CPU/CUDA 数值一致；
  全量 642×4 的 CPU 复现脚本与耗时已写进 `FVD_SPEC_20261021.md` 第 6–7 节。

- **D56 · SignViP 端到端复现与 FVD 脚本正确性（2026-10-22）**：
  用户 m01447/m01638 要求用 SignViP 官方权重跑测试集、看 FVD 是否与论文一致，以验证我们脚本。
  已做：①官方代码 vs 我们工具同输入等价（239.9040 vs 239.9002，相对差 1.6e-5）；
  ②本机搭出 text→pose→video 全链（`tools/signvip/gen_rwth.py`），test 642 全生成；
  ③官方协议（32 帧、272×224、I3D-400 logits）FVD = **259.00**（N=634，本底 86.31）。
  结论：**脚本正确性已证**；论文 1025.45 **未复现**（我们低约 4×），差距不在 FVD 代码，
  候选原因 = 论文评测子集/样本量、条件来源（GT pose vs 生成 pose）或训练期小批量验证。
  后续：①需向作者确认论文 FVD 的样本量与条件；②`tools/eval_fvd.py` 新增 `--min-frames`（默认 10）已随代码提交。

## 历史条目（已完成，仅留指针，不删）

> 编号不复用。完整叙述见所引文件。

- **(X)1** 新建 text→video 项目（DiT 落地 + 冻结 VAE + x0 配方 + 全链路 + 30k/60k 训练）→ `NOTES.md` 2026-10-12 各条。
- **(X)2** CFG 区间/动态调度 + rescale → 结论入 `NOTES.md` 2026-10-15；低噪声段单加 CFG 为 FAIL-022。
- **(X)3** self-conditioning → FAIL-021。
- **(X)4** 长度头分布化 → 有效，转 `NOTES.md`；mT5 直接回归拆为任务 15。
- **(X)5** 条件路径重构（双流 + 容量对照）→ FAIL-019。
- **(X)6** 文本压力课程 → FAIL-019。
- **(X)7** 文本分支优化器/特征层（text_lr 5e-5 有效转 NOTES；1e-4、mT5 中层）→ FAIL-028；idea-006 关闭。
- **(X)8** REPA/iREPA 域内对齐 → FAIL-020。
- **(X)9** 多尺度 latent 动态损失 → FAIL-020。
- **(X)10** 120k 续训（有效转 NOTES，推翻"60k 最佳"）+ 容量对照 → FAIL-027；idea-011 关闭。
- **(X)12** 配对汇总表（`tools/collect_matrix_table.py`、`paired_arm_stats.py`、`paired_mrr_stats.py`）→ `NOTES.md` 2026-10-15。
- **(X)N5** 纯续训等预算对照臂 `route_s_cont20k`（20k，test 全 n.s.，MRR +0.0030 显著为正）；后果见 N7。
- **N1–N4**（margin / softalign / paraphrase / slots）→ FAIL-023~026，出池。
- **idea-009 / 018 / 012 / 006 / 011** → 见 `THINK.md` §5 出池表与 `FAIL.md`。

- **D52 单臂实验：mask 修复 + mlp4096 跨 VAE 桥 + 两端适配器 + L_motion（2026-10-20 用户指令）**：
  一条 20k 臂（`wan_flow_mlp4096_motion_v2`）同时包含四项改动：① mask 极性修复（`to_keep_mask`，
  `tests/test_wan_masks.py` 锁死）；② 有监督桥 1024→4096（Wan VAE 每帧 16x16x16 摘要，
  目标切分 `chunk=min((t+1)//4, F_lat-1)`、象限 `t%4`）；③ 两端 LayerNorm+MLP 适配器；
  ④ `L_motion`（weight 0.1 / warmup 2000）。**用户明确不设探针、不设对照臂**，
  故若指标不涨只能记「合并臂无增益」。`wan_flow_lora_v1{,_raw}` 因 mask bug 作废（见 NOTES 2026-10-20）。
  **状态（2026-10-20 11:45）**：链式流水线已起（`tools/run_stage2_chain.sh`），当前在跑 Wan VAE 编码（eval 全量 + train 每 8 条取 1），随后自动训桥、起 20k 训练、挂评测看守；`EXPERIMENT.md` 已预填 exp-090（进行中）。

- [ ] **D57 · SignViP 论文条件重新复现（2026-10-22 用户授权）**：使用原始 210×260 官方 test642 数据和字幕，50步/CFG3.5、完整EOS长度、官方视频编解码与FVD函数；独立产物 `reports/eval/signvip_paper_repro_20261022_v2`。主报告固定32帧、短片末帧补齐全642，同时预先声明严格32帧子集、全视频均匀16帧及GT首帧偏移敏感性；未知论文窗口/N不作推断，不通过改变N匹配1025.45。分支 `codex/signvip-paper-repro-20261022`，实现提交 `2319729`。已完成原始642条视频准备与首例冒烟（24帧/18.5s）；GPU1/4两分片全量生成已启动，supervisor PID3970442，完成后自动官方FVD；状态见产物目录 `pipeline_status.json`。执行中。

---

## 里程碑1（2026-10-22 立，全新项目；与上面条目互不相关）

> 规格（唯一验收依据）：`.trae/documents/MILESTONE1_ARCH_PIXART_20261022.md`
> 分支：`milestone1-pixart`（从 `main`/`bdf36d1` 切出）。旧台账条目（上面 D-xx）属旧世界，
> 不影响本里程碑；milestone-1 自己的产物写 `runs/m1_*`、`reports/eval/m1_*`。

- **D58 · 里程碑1 立项与代码清场（2026-10-22 用户指令）**：backbone 换 PixArt-α、x-prediction、
  c 只留 timestep、官方 FVD 24 帧口径；10 臂矩阵 = E1(base=1a+4c+2a+3b+5b) / E2(2b) / E3(2c) /
  E4(3a) / E5(4a) / E6(4b) / E7(1b) / E8(1c) / E9(1c+3a) / E10(5a)。分支已建，代码已清场
  208→45 个跟踪文件（提交 `44b874c` 移植官方 FVD 链、`88945f0` 删除旧架构/探针/Wan/SignViP）。
  **本轮只出规格与清场，不实现、不训练。**

- **D59 · 里程碑1 四项设计裁决（2026-10-22 用户第二轮）**：
  **Q1 → 保持 1024 维 1-token latent 契约**（改 PixArt：patchify 换成 per-frame
  `Linear(1024→D)`，位置编码与 `proj_out` 重写；5a 只移植 block 内部权重，逐层写清单）；
  **Q2 → context 注入 = 条件 token 拼进 self-attn 序列当前缀**（1b 不设 cross-attn 层）；
  **Q5 → 24 帧 FVD 取帧口径**（首轮曾定"均匀抽取"，**第二轮已改判为"从头抽取"，见 D60**；
  官方 I3D 模型与代码不变）；**Q6 → 本里程碑不做长度预测**，训练用样本自身长度、
  评测/生成用 **GT 视频长度**。已写入规格 §1/§2.1/§2.2/§2.3/§7/§9/§11.1 与 `CONTEXT.md`。

- **D60 · 里程碑1 第二轮裁决 + Q5 改判（2026-10-22）**：
  **Q5 改判**：24 帧口径从"均匀抽取"改为**从头抽取**（取 frame 0→23，`--frame-mode contiguous
  --num-frames 24`）；不足 24 帧的样本跳过，**有效 N ≈613**，id 清单落盘且 10 臂共用，报表写有效 N；
  不要沿用 `--resize-hw 272 224`（旧 SignViP 预处理），缩放保持官方 224。
  **Q3 → mT5 768→骨架宽度的投影可训练**（Linear+LN，10 臂共有）；
  **Q7 → 4c 的"前 3 层"含 embedding**（训练 embedding + layer[0:3]）；
  **Q8 → 骨架用 PixArt-α 原尺寸 28 层 / 1152 宽 / 16 head**；
  **Q9 → motion loss 沿用 `MultiScaleDeltaLoss`**（k∈{1,2,4} + warmup）。
  已写入规格 §1/§2.1/§3/§4/§9/§10/§11 与 `CONTEXT.md`。

- **D61 · 里程碑1 Q4 注入计划裁决（2026-10-22 用户）**：2a/2c 的层分配改为**间隔放置**——
  28 个 block 每 4 个一组，**前 3 个注入 text、第 4 个注入 ref**，即 `[T,T,T,R]×7`，
  ref 层 = layer **3/7/11/15/19/23/27**（7 层），text 层 = 其余 21 层。
  2a：text 层只看 text、ref 层只看 ref；2c：ref 层看 `[ref ; text]` 拼接。
  写进配置的 `layer→{none,text,ref,ref+text}` 映射表，10 臂内不再改。
  **至此 Q1–Q9 全部裁决完毕**，下一步等授权进入 **M1.1**（latent 契约落地）。

- **D62 · 磁盘清理（2026-10-22 用户授权 A+B+C+附带项）**：删除 Wan 检查点 60 个 .pt（46.09G）、
  `.models/Wan2.1-T2V-1.3B-Diffusers`（27G）、旧 DiT 的 11 个 final/last（26.62G）、
  `data/text_emb_umt5`（4.1G）、`data/wan_vae_latents`（1.3G）、`.tmp_adaln`（169M），
  仓库 201G → **96G**（回收 105G）。明细与回滚方式见 `.trae/documents/CLEANUP_20261022.md`。
  **未删**：`samples_*.pt`（0.48G）、`reports/eval/matrix`（37G 证据链）、`reports` 内旧生成视频（24.1G）。

- **D64 · 里程碑1 实现完成 + E1 base 启动（2026-10-22 用户授权"继续完成实现并开始跑第一个 base 实验"）**：
  实现链 `730b3a6`/`da264fc`/`8733c10`/`c26dc71`/`d71b23d`/`131390b` + 评测修复 `85952a6`（独立复审通过，
  见 `MILESTONE1_TASK3_REREVIEW_20261022.md`）；M1.1–M1.4 实现完成，M1.5 小尺寸全链路
  （200 步训练→GT 长度生成→first24 官方 FVD pilot）已跑通但只作管线证据；全仓 **91 passed**
  （Lead 22:43 独立复跑，`/tmp/m1-lead-final-suite-20261022.log`），十臂配置矩阵复核与规格 §5 一致。
  **E1 已启动并完成首 checkpoint 验收**：4×RTX3090、effective batch16、`runs/m1_E1_base`、启动时 HEAD
  `63f946f`、launcher PID 3446445；step1030 loss 0.031、~3.45s/步、峰值显存 16.6 GiB；
  **step1000 `last.pt`（13.68 GiB）三模块 raw/EMA、优化器、RNG、config 全部校验通过，视觉回调 2 例 61 文件正常，
  raw/EMA 均能经 `MilestonePipeline.from_checkpoint` 全尺寸载入**（证据见
  `MILESTONE1_EXECUTION_20261022.md`「E1 产物验收」）。
  **注意：该 4 卡启动随后按用户 GPU≤3 约束中止并按 D65 以 3 卡/batch12 从 0 重训**；4 卡产物归档保留。
  E2–E10 只备配置、不排作业。
- **D68 · M1.5 通过：E1 base 正式数字 FVD 126.9269（2026-10-24）**：3 卡/batch12 从 0 训 30k 步
  （32h01m，final.pt 13.68 GiB）后，按 first24 test640 协议评测（raw 权重、DDIM50/CFG2/GT 长度/seed42、
  官方 I3D-K400 400d/224²、本底 84.2975）：**FVD = 126.92693857176755**（N=640/642，
  mean_feature_l2 32.32±6.53）；同协议第二次评测**逐位相同**（ids SHA256 一致）→ M1.5 判据
  （非 NaN、不低于本底、可复现两次）**通过**。产物 `/data1/huyang/text2video/runs/m1_E1_base/`
  （仓库 `runs/m1_E1_base` 为软链）与 `reports/eval/m1_E1_base{,_repeat}/`；已登记
  `EXPERIMENT.md` exp-099、`NOTES.md` 2026-10-24、规格 §13。**M1.6 十臂需用户明确授权**；
  raw/EMA 口径与 P3 清理仍待裁决/后续。
- **D66 · 权重改存 /data1/huyang（2026-10-24 用户指令）**：后续所有权重/run 输出放 `/data1/huyang`。
  起因：根盘 `/` 为共享机，2026-10-24 09:50 被他进程写满导致第二次复现评测在 43/640 因 ENOSPC 中止
  （第一次评测已完整落盘）。处置：`runs/m1_E1_base`（28 GiB）与 4 卡归档（14 GiB）复制到
  `/data1/huyang/text2video/runs/`（逐文件列表+大小一致，ckpt 重载校验通过）后删除根盘原目录，
  仓库 `runs/` 下保留同名软链；`configs/milestone1/{base,E1..E10}.yaml` 的 `training.out_dir`
  全部改绝对路径 `/data1/huyang/text2video/runs/m1_*`；README 同步。根盘可用 140 GiB。
  复现评测在迁移后重跑，2026-10-24 11:06 完成且与第一次逐位相同（见 D68）。
- **D67 · 分支改名（2026-10-24 用户侧操作，非本 agent 发起）**：`milestone1-pixart` 分支被重命名为
  `main`（reflog: "Branch: renamed refs/heads/milestone1-pixart to refs/heads/main"），随后新增
  `be30f83`（忽略迁移包/vendor 等）、`24dad45`（冻结 VAE 配置实体化入库）、`4fab4d1`（迁移包入库）、
  `4dd02e2`（多臂对比图/对照渲染工具入库，已在 `origin/main`）；旧 `milestone1-pixart` 本地指针留在
  `97b9e9c`。本 agent 的权重路径提交 `77e145b` 落在 `main`、**领先 origin/main 1 个提交（未 push）**。
  后续里程碑工作以 `main` 为准；文档里旧的 `milestone1-pixart` 字样按本条理解为同一分支。
- **D69 · E2（base-2b）实验启动（2026-10-24 用户指令）**：用户授权继续跑 **E2**（十臂矩阵第 2 臂，
  非全套十臂）。E2 = E1 唯一改动 factor2 `2a`→`2b`：28 层 cross-attn 全部注入 `[ref;text]`；其余
  配方完全同 E1（1a/4c/3b/5b、batch12、lr、seed42、30k）。2026-10-24 11:54 启动，launcher PID
  **3692697**、rank0 PID 3692722，因 GPU2/3 被他人占用，使用**空闲的 GPU0/1（2 卡，符合 ≤3 约束）**，
  micro1×accum6×world2 = effective batch12（与 E1 等价）；参数 821.92M。权重/输出写
  **`/data1/huyang/text2video/runs/m1_E2/`**（用户重申所有权重放 /data1/huyang，arm 配置已是绝对路径）。
  已挂**自动评测链** `/tmp/m1_E2_chain.sh`（nohup）：等训练 `complete` → 选一张空闲 GPU → 按 first24
  test640 协议评测 `reports/eval/m1_E2`；结果登记 exp-100。预计 30k ≈ 34h（约 10-25 22:00 前后）。
- **D70 · 所有 FVD 测试改用 SignViP 官方脚本（2026-10-24 用户指令）**：以后所有 FVD 测试
  **必须使用 SignViP 的官方脚本**。落地：vendor `third_party/signvip_fvd/fvd.py`
  （SignViP 仓库 `metrics/fvd/styleganv/fvd.py` 血统；SHA256 `c9a1ac…da37`）+ `tools/eval_fvd_signvip.py`
  调用其 `get_fvd_feats`/`frechet_distance`；`tools/eval_milestone.py` 默认 `--fvd-tool signvip`；
  旧 `tools/eval_fvd.py` 只用于历史复算/等价性诊断。帧协议不变（first24 含 frame0、GT 长度、
  test640、DDIM50/CFG2、raw）；预处理改为 SignViP `preprocess_single`（无抗锯齿）。
  规则写入 `RULES.md` 方法约束 5/评测 6.1/探索 19、`AGENTS.md` 方法约束 5、`FVD_SPEC` §0.0、
  `STRUCTURE.md`；校验 = 脚本/权重 SHA256 + 同输入特征逐位一致（max|Δ|=0，20 条 FVD 差 1.4e-6）。
  E1 已用新脚本重算、E2 自动评测链已改用新脚本。
  ⚠️ 原 SignViP repo 副本在本机迁移时丢失（旧机 reports 未迁移）且本机无外网；本次用的是本机两处
  SLP2025 项目里的同源字节副本。若用户能提供 SignViP 官方 repo 副本，应替换并复核 SHA256/等价性。
- [ ] **M1.6 十臂矩阵**：M1.5 通过且用户明确授权后，再按 E2–E10 排作业（同一份 640 id、同采样器/CFG/
  步数/长度口径/种子）；本轮不排。
- [ ] **M1-评测 P3 残留（不阻塞 E1）**：① `tools/eval_fvd.py:398-400` 在 contiguous/24 下不可达，
  仅在 `num_frames < min_frames` 的矛盾参数下误标长片，应改显式报错或删除；② `:393` 资格判断用偏移后
  帧数，关 `--include-frame0` 时与"源 `num_frames>=24`"不一致（milestone wrapper 不受影响）；
  ③ 契约只比路径字符串、FVD 真值缓存 key 无 frame_root/源指纹，换源时须 `--no-ref-cache`；
  ④ 补测 vae/stats/frame_root 别名、fill-ref-cache 的 `skipped_short`、sampling 不一致拒绝。
  复审见 `MILESTONE1_TASK3_REREVIEW_20261022.md`；清理后需重跑 `tests/test_eval_fvd.py`、`--self-check`
  并确认共享真值本底不受影响。
- **D65 · E1 改用 3 卡 + effective batch12（2026-10-22 用户约束）**：用户指令"不要四个 GPU 一起使用，
  最多只能使用三个 GPU"。原 4 卡 E1 于 step1110 正常停止，产物归档 `runs/m1_E1_base_4gpu_aborted_20261022/`
  （step1000 ckpt/回调已验收，保留）。因 effective batch16 不能被 `micro1×world3` 整除，用户选择
  **3 卡 + microbatch1×accumulation4×world3 = effective batch12**；`configs/milestone1/base.yaml` 已改，
  十臂共用；lr1e-4 / text lr1e-5 / seed42 / 30k / 其余配方不变。E1 从 0 重训（跨 world size 不能 resume），
  启动于 23:36，launcher PID 3456546、rank0 PID 3456574，`runs/m1_E1_base`；
  启动记录 `MILESTONE1_E1_LAUNCH_V2_20261022.json`。GPU0/1/2 各 ~18.8 GiB、GPU3 空闲。
  实测 step1 loss 0.90315、step20 loss 0.69588、~3.77s/步、峰值16.59 GiB。配方改动提交 `97b9e9c`（README/base.yaml/十臂测试，训练代码不变），README 与实施计划已同步。


- **D62 · 完整训练迁移包（2026-10-22 用户授权）**：保留全部项目代码/.git/.trae、冻结VAE项目代码和匹配psnr-best权重、离线mT5 PyTorch权重、完整v1缓存/文本/统计、当前PHOENIX train/dev/test帧。迁移入口在内存重定位路径，不修改冻结契约和源缓存；不打包历史runs/reports、Wan/SignViP大模型。打包产物 `/var/tmp/text2video_migration_20261022.zip`，包内 `migration/README.md` 记录安装和验证范围；源现存帧抽查256×256。对应F-015；目标机新环境安装/GPU训练仍需迁移后实测。

- **D63 · 目标机训练环境配置与验证（2026-10-22 用户授权）**：已新建 conda `text2video`（Python3.12.15、torch2.6.0+cu124），安装迁移依赖和pytest。pip check通过；四张RTX3090 CUDA/bf16、真实缓存/mT5/DiT前向及冻结VAE单帧解码通过；128测试通过、9项Wan缺权重跳过；I3D自检133.91。实际可运行的是迁移DiT入口，PixArt里程碑未实现；未启动训练。环境与启动说明见 `ENVIRONMENT_SETUP_20261022.md`，包版本见 `ENVIRONMENT_PACKAGES_20261022.txt`。

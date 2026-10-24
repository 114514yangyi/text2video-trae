# FINISH — 已完成待测试的产物（finish）

> **用途**：AI 按 `TODO.md` 条目执行完毕、**产物与复现命令齐备**后，从 TODO **剔除**该条目并登记在此，
> 等待用户/评测给出裁决。裁决后出池：有效 → `NOTES.md`（note）；无效 → `FAIL.md`（fail）。
> 本文件只保留"做完但还没有结论"的东西；未做完的留在 TODO，未筛选的留在 `THINK.md`。
> **建立**：2026-10-14 用户指令（`RULES.md` 知识库流转规范第 9 条）。

## 登记要求（每条必填）

| 字段 | 说明 |
|------|------|
| TODO 编号 / think 卡 | 从哪条任务来、对应哪个设想（`idea-NNN`） |
| 产物 | `runs/…` 权重、`reports/eval/…` 结果、新增/改动的 `tools/…`、`configs/…` |
| 复现命令 | 一行即可复跑（脚本入口优先） |
| 判据 | 按 `RULES.md` 评测规范（2026-10-15 修订）：**test 642 单侧**同协议；主判据 = 检索 MRR / top-1（TODO D3、D26） |
| 现状数字 | 已完成评测的中间数字，注明 split/采样器/CFG/长度口径（口径不同不能直接比） |
| 状态 | `待测试` / `测试中`（裁决后整条出池） |
| 日期 | 登记日期 |

---

## 出池记录

| 编号 | 裁决 | 去向 | 日期 |
|------|------|------|------|
| F-001 text-prefix 预算阶梯 30k/60k/90k | 已裁决（旧"60k 最佳"被推翻） | `NOTES.md`「2026-10-15」结论二 + TODO **D19**；≥3 seed 复核仍留 TODO 11 | 2026-10-15 |
| F-002 route B：句向量语义目标（含 λ 消融） | **有效但不可晋级**（增益与 WER 代价并存，待用户裁决取舍） | `NOTES.md`「2026-10-17：route_b 重判为正常实验」+ `EXPERIMENT.md` exp-049/053；用户撤销 D30、新增 **D34**；裁决入口 TODO **N9** | 2026-10-17 |
| F-003 route C：逐词 latent→文本教师辅助目标 | 无效（弱于同族 route_b，ROUGE/WER 退化） | `FAIL.md` **FAIL-030** + `EXPERIMENT.md` exp-050 | 2026-10-17 |
| F-004 评测基建：双集配对矩阵 + SignMbart 回译链 | 有效（35 作业双侧跑通） | `NOTES.md`「2026-10-14 深夜二」「2026-10-15」；`reports/eval/MATRIX_TABLE.md` | 2026-10-15 |
| F-005 T2.13 语义目标可达性复核 | 有效（结论已在 D11） | `NOTES.md`「2026-10-14 深夜」；`reports/eval/semantic_reachability_prefix60k.json` | 2026-10-15 |
| F-006 T1.7 长度头分布化 + predicted-length 列 | 有效（部署闭环成立） | `NOTES.md`「2026-10-14 深夜」「2026-10-15」结论四 + TODO **D14** | 2026-10-15 |
| F-007 T1/T2 训练臂与 CFG 扫描的双侧配对矩阵 | 部分有效 | `NOTES.md`「2026-10-15」+ `FAIL.md` FAIL-019/020/021/022 | 2026-10-15 |
| F-008 D28 四条 idea（margin/softalign/paraphrase/slots）训练臂与评测 | 无效（四条全部证伪）；**其中 FAIL-025（softalign）于 2026-10-18 被用户裁决撤销**（改判"回译侧有效"，idea-002 复活，见 `NOTES.md` 2026-10-18 / TODO **D38**） | `FAIL.md` **FAIL-023/024/025/026** + `EXPERIMENT.md` exp-036~040 | 2026-10-16 |
| F-009 容量对照：宽（hidden 1024）/ 深（depth 16）各 30k | 无效（同预算两臂都显著劣于基线） | `FAIL.md` **FAIL-027** + `EXPERIMENT.md` exp-031 | 2026-10-16 |
| F-010 mT5 中间层条件臂 route_n | 无效（同预算回译三项显著退化） | `FAIL.md` **FAIL-028** + `EXPERIMENT.md` exp-041 | 2026-10-16 |
| F-011 N5 纯续训等预算对照臂 route_s_cont20k | 对照成立（本身不追增益） | `EXPERIMENT.md` exp-042 + `FAIL.md` FAIL-025 补记 | 2026-10-16 |
| F-013 锚点作业 FVD 崩溃修复 | 已修（`--ids-file` 分支未定义 `generated_root`） | `tools/eval_fvd.py`：`--ids-file` 路径下 `generated_root` 未定义；补跑队列 `jobs_anchor_fix_20261017.txt` | 2026-10-17 |
| F-014 idea-032 采样阶梯 log-SNR 均匀（P18） | 无效（幅度升、回译/MRR 全 n.s.） | `FAIL.md` **FAIL-038** + `EXPERIMENT.md` exp-080 | 2026-10-19 |
| F-015 idea-035 gate 噪声调度 S1/S2（P22） | 无效（不优于同剂量常数 k=4） | `FAIL.md` **FAIL-039** + `EXPERIMENT.md` exp-079 | 2026-10-19 |
| F-016 idea-037 对齐损失 warp 预算两档（P24 后续） | 无效（窄打平、宽显著亏） | `FAIL.md` **FAIL-040** + `EXPERIMENT.md` exp-085 | 2026-10-19 |
| F-017 idea-033/034 对齐损失改施力方式（Z/Z2/AA） | 无效（显式改噪声窗/时间分辨率均退化） | `FAIL.md` **FAIL-036/037** + `EXPERIMENT.md` exp-082/083 | 2026-10-19 |
| F-018 P20 · R1 三个单变量跟进臂（W/X/Y） | 无效（语义叠加 / 文本微调 / 加大权重三条全失败） | `EXPERIMENT.md` exp-081/084 + `NOTES.md` 2026-10-19 | 2026-10-19 |
| F-019 idea-038 EMA 影子权重（P26） | 有效（用户裁决全臂部署默认切 EMA） | `NOTES.md` 2026-10-19 + TODO **D46** + `EXPERIMENT.md` exp-078 | 2026-10-19 |

> 出池时把整条从这里删除，并在 `NOTES.md`（有效）或 `FAIL.md`（无效）写入完整链路。

## F-014 · SignViP 论文条件复现执行器（2026-10-22）

来源D57，用户授权原始test642重跑。产物 `tools/signvip/reproduce.py` 与 `run_reproduction.py`，提交2319729；4个unittest、既有FVD测试与官方编码器roundtrip通过。完整首例24帧/18.5秒，无提前截断。全量运行目录 `reports/eval/signvip_paper_repro_20261022_v2`；2026-10-22 22:37生成642/642及评测全部退出0。最终前32补齐290.291187(N642)，严格32为308.710916(N597)，uniform16为73.669359(N642)，GT偏移1为289.375657(N642)。结果已登记NOTES 2026-10-23 D57及exp-100至103。论文完整协议对应关系仍待确认，不宣称复现成功。

## F-015 · 完整训练迁移包（2026-10-22）

来源D62。产物 `/var/tmp/text2video_migration_20261022.zip`，包含当前可训练DiT工作区2319729、完整Git历史/.trae、冻结TiTok-CVAE项目及匹配权重、离线mT5、完整v1数据和PHOENIX现存帧。迁移入口独立置于包内migration，不修改现有代码或冻结数据。源环境CPU检查已通过数据读取、文本编码、条件DiT前向、VAE冻结SHA和完整权重加载、帧路径重定位。目标机新环境安装及GPU训练待测试；源机包源SSL故障未能验证全新依赖安装，迁移requirements修正OpenCV/NumPy依赖声明冲突。包内README含详细边界。

## F-016 · 里程碑1 PixArt 实现（M1.1–M1.4）与 E1 base 启动（2026-10-22）

- 来源：TODO **D64**（用户授权"继续完成 milestone 实现并开始跑第一个 base 实验"）。
- 产物：分支 `milestone1-pixart` 实现提交 `730b3a6`/`da264fc`/`8733c10`/`c26dc71`/`d71b23d`/`131390b` + 评测修复 `85952a6`；十臂配置 `configs/milestone1/{base,E1..E10}.yaml`；训练 `t2v/train.py`；采样/评测 `t2v/pipelines/milestone.py`、`tools/eval_milestone.py`；E1 运行目录 `runs/m1_E1_base/`（运行中）。
- 复现命令：训练（用户约束 ≤3 GPU，D65 配方 batch12）`CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --standalone --nproc_per_node=3 -m t2v.train --config configs/milestone1/E1.yaml`；评测 `python tools/eval_milestone.py --config runs/m1_E1_base/config.yaml --checkpoint runs/m1_E1_base/final.pt --out reports/eval/m1_E1_base --gpu cuda:0`。
- 判据：first24 test 单侧官方 I3D-FVD（N=640/642、真值本底 84.2975），非 NaN、可复现两次；本轮无十臂对照结论。
- 现状数字：全仓 **91 passed**（Lead 独立复跑 `/tmp/m1-lead-final-suite-20261022.log`）；小尺寸 pilot FVD 1532.37 仅管线证据，不进入正式表。E1 首启为 4 卡/batch16（step1110 时按用户 GPU≤3 约束中止并归档 `runs/m1_E1_base_4gpu_aborted_20261022/`，其 step1000 checkpoint 三模块 raw/EMA、优化器、RNG、config 与视觉回调均已验收）；**正式 run 为 3 卡/batch12 从 0 重训，已于 2026-10-24 07:40 训完 30k 步（32h01m，final.pt 13.68 GiB 已校验）**；正式 first24 test640 **FVD = 126.9269**（N=640、raw 权重、真值本底 84.2975、mean_feature_l2 32.32）；同协议第二次评测（第一次在 43/640 时因共享机根盘 ENOSPC 中止，改存 `/data1/huyang` 后重跑）**逐位相同 = 126.92693857176755** → M1.5"可复现两次"判据通过。权重与 run 目录已迁至 `/data1/huyang/text2video/runs/`（仓库内软链）。
- 状态：**M1.5 完成**（训练+评测+两次复现均通过）；M1.6 十臂效果对比待用户授权。日期：2026-10-22（训练/评测完成 2026-10-24）。

## F-017 · 里程碑1 E7（base + 前缀注入）30k 训练与 first24 官方 I3D-FVD（2026-10-24）

- 来源：TODO **D66**（用户授权"先走 E7，用 3 个 GPU，测 FVD，记录结果"）。
- 产物：`runs/m1_E7/`（`config.yaml`、`final.pt` 12.3 GB、`metrics.jsonl`、`samples/`、`status.json`）、`runs/m1_E7_report.py`；`reports/eval/m1_E7_first24_20261024/`（`ids.txt`、`protocol.json`、`fvd.json`、`frames/`）；共享 ids `reports/eval/samples/m1_test_first24_ids.txt`；本机真值本底 `reports/eval/m1_first24_prepare_20261022/real_floor.json`。
- 复现命令：训练 `CUDA_VISIBLE_DEVICES=0,4,5 python -m torch.distributed.run --standalone --nproc_per_node=3 -m t2v.train --config configs/milestone1/E7.yaml`（本机实例实际写作 `--out-dir runs/m1_E7`，日志 `runs/m1_E7.console.log`）；评测 `python tools/eval_milestone.py --config runs/m1_E7/config.yaml --checkpoint runs/m1_E7/final.pt --out reports/eval/m1_E7_first24_20261024 --gpu cuda:0`；自动链 `runs/m1_E7_chain.sh`（等 `final.pt` 出现即评测）。
- 判据/数字：30k 步 22.8 h（末段 2.81 s/步，峰值 13.58 GiB，3 卡 effective batch12，RTX4090+2×RTX3090 混卡）；**FVD 124.073294**（N=640），真值半分本底 84.297514，相对本底 **+47.2%**；生成特征 L2 32.114±6.701。
- 状态：**已完成待裁决**——E1 base 同口径 FVD 未在本机产出（E1 运行在 `/home/wxy` 主机），本轮只有单点基准，不能与 E1 比较，也不能据此判断"换注入通路更优/更差"。
- 日期：2026-10-24。

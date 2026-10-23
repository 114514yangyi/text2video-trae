# Milestone 1 全分支最终审查（2026-10-22）

范围：固定差异 `c20c4a8..63f946f`，依据 `MILESTONE1_ARCH_PIXART_20261022.md` 与实施计划。审查模型、文本、训练、恢复、采样、评测及十臂配置；结合分任务审查与 `MILESTONE1_EXECUTION_20261022.md`。本次只读代码、日志、报告和既有产物，不重跑测试、不起作业、不修改源码。

**结论：当前 E1 配置没有发现训练启动阻塞项，可以进入正式全尺寸训练的首次运行验证。整个软件交付仍有两项 P2 评测问题需修复，修复前不能宣称评测入口已全部验收。** 两项均不改变 E1 的训练梯度，也不影响现有相同资产配置下的验证产物。

## Actionable findings

1. **P2 — 生成和计分允许混用不同的数据/解码资产。** `tools/eval_milestone.py:130–134` 只核对 `sampling`，但 dataset、清单和真实计分视频来自命令行 config，而采样时反归一化统计、VAE 和参考图像路径来自 checkpoint config（`t2v/pipelines/milestone.py:102–103`、`:111–115`、`:169–170`）。两份 config 的 sampling 一样而 `data.cache`、`data.stats`、`data.frame_root` 或 `vae` 不同时仍能运行；相同 ID/帧数无法排除参考特征、参考图与真值视频来源混用。修复应在生成前比较解析后的资产路径或可信身份，或统一使用 checkpoint 的有效资产配置，并在协议中记录。用“sampling 相同、资产不同”的回归用例验证拒绝。不必比较无关的输出目录等训练设置。**影响：阻止评测入口整体验收；不阻止当前 E1 训练启动。**

2. **P2 — 对短真值样本先验证生成长度，违反应跳过的规则。** `tools/eval_fvd.py:390–395` 在检查 `min_frames` 前调用 `paired_paths`，后者对 contiguous24 无条件要求生成至少 24 帧（`:214–215`）。若直接官方 CLI 收到完整 test 树，其中真实视频及正确生成视频均只有 23 帧，会报错而非跳过。milestone wrapper 预筛清单掩盖此情况，但公共评测入口仍允许这种输入。先按真值帧数筛除不合格视频，再对合格视频检查生成长度；保留“GT≥24 而生成<24 必须失败”。分别覆盖这两个分支，保留被跳过 ID。**影响：公共评测 CLI 的短片规则；不阻止当前预筛 640 清单及 E1 启动。**

本次独立读全链时得到上述结论，随后与 Task 3 审查一致；不新增重复问题编号。未发现 P0/P1。Standards 维度没有额外硬规则违规；Spec/行为正确性维度保留上述两项。

## 已确认的生产路径

- **真实 E1 配置尺寸正确。** `configs/milestone1/base.yaml:16–23` 明确 1024 latent、1152 hidden、28 blocks、16 heads，E1 继承而不缩小。训练上限 256 源帧、模型容量 512 latent 帧；测试集不套训练长度过滤。E1 为 cross、`[T,T,T,R]×7`、三层可训练文本、无 motion、无 PixArt 预训练。十臂只调整约定因子，共用配方；E4/E9 motion 权重和 warmup 相同。
- **模型与条件符合契约。** Task 1 静态审查已核对官方 adaLN-single 公式、timestep-only 调制、线性逐帧输入输出、padding、prefix 无 cross 参数、分层路由和严格 block/t_block 初始化。没有长度头、pose/gloss/光流通路。
- **文本梯度所有权正确。** 文本物理截断在 GPU 搬运前完成；embedding、前三层与最终 norm 保留且可训练。`MilestoneSystem` 注册实际 encoder，模型/投影/text 同属一个 DDP wrapper；优化器文本独立学习率，参数没有被加入两次。冻结全文本臂正确保持 eval，不进优化器。
- **DDP 累积实现匹配有效 batch。** 相同的全局分桶批次分给各 rank，局部不重叠；`no_sync` 包裹前后向，最后一次同步；损失除以累积次数，DDP 再按 rank 平均。main loss 先按每视频有效帧归约；motion 修复为逐视频裁掉 padding 后计算，独立于微批次组合。
- **恢复信息完整。** 保存三模块 raw/EMA、优化器、epoch/batch cursor 和每 rank 的 Python/NumPy/CPU/current-CUDA RNG。恢复要求原 world size 和实验配置；采样器用局部 RNG 复建顺序。E10 恢复禁用初始化源重载并保留审计清单；保存采用临时文件和原子替换，单写者锁防止并发训练覆盖。
- **采样使用完整 GT 轨迹。** `generate` 实际采 `[T,1024]` 完整 latent，然后仅解码前 23 个预测帧、在前面加入一次 frame0，形成 first24 展示/计分视频。完整 latent 仍保存；不是仅生成 23 个 latent 冒充 GT 长度。回调使用活模块与 CPU VAE，保存并恢复模型模式，训练调用方恢复 RNG；无需复制一套 XL 模型。
- **官方评测正常路径一致。** first24 包括 frame0；GT 少于 24 的清单记录并排除；有效 ID 清单不可静默改写；pilot 显式命名；官方 I3D K400 的 400 维 logits、224 官方预处理、float64 Fréchet，权重 hash 纳入缓存检查。各臂应复用同一份 640 ID。

## 已阅读的验证证据

- `/tmp/m1-final-suite.log`：**83 passed，6 warnings，9.47s**。未由本审查重跑。
- `/tmp/m1-ddp-continuous.log` 与 `/tmp/m1-ddp-resume.log`：实际两 GPU 小模型的第 5/6 步 loss 对齐，均成功保存；执行记录另报告三模块 raw/EMA 逐位相同。单进程连续/恢复和 200 步真实数据小模型验证也有记录。上述 DDP 实测是 **150240 参数 backbone 的小模型**，不能作为 XL 显存或吞吐量证明。
- 官方 first24 真值报告实际为 **640/642**，`real_split_half_fvd=84.29751411436246`；规格末尾已补记实测值。I3D 自检 133.908176 通过。
- 4 条显式 pilot 的原始与重复报告均为 **FVD=1532.3668975285498**；协议为 I3D logits400、first24、frame0、224，结果逐值相同。它证明该采样/计分路径可执行和复现，不是正式 E1 成绩或方法有效性证据。
- 活训练模块的 CPU VAE 回调已实际产出 MP4/GIF/对比图与完整 latent，并恢复 train/eval 模式；本审查不重复该耗时验证。

## E1 启动边界与资源风险

**可以启动，不代表已经验证 XL 长程运行。** 全尺寸模型、三层 mT5、Adam 状态、梯度与 DDP 缓冲需在实际空闲卡上确认；采用配置中 microbatch=1 和满足有效 batch16 的累积，显存不足先调整卡数/累积，不缩小骨架。

`t2v/milestone_runtime.py:76–85` 每步在 rank0 同步搬运全部状态做 CPU EMA，其他 rank 在下一次同步等它；这里会有每步数 GB 量级 GPU→CPU 流量和主存运算，属于明确吞吐风险，尚无全尺寸实测可判其严重性。checkpoint 保存还会复制 raw 与优化器到 CPU，各 rank 又分别预加载数据，主存峰值也需观察。实现没有显示的 EMA 数值错误，不能仅因这些成本断言无法启动。首次全尺寸更新应记录实测 loss/梯度有限性、每步耗时、GPU/主存峰值，并实际确认 checkpoint 和回调能够继续运行；这属于已授权生产启动的验证，不要求另开研究探针。

后续完整 30k 训练、正式 640 条 E1 FVD、十臂效果比较仍未完成，属于实验结果待产出，不能用当前软件测试或 pilot 替代，也不构成本轮 E1 启动的代码阻塞项。

## 正式 E1 启动证据补记

主控随后启动正式 E1。本审查只读核对 `runs/m1_E1_base/train.log` 与 `status.json`：4 GPU，7021 训练视频，microbatch1 × accumulation4 × world4 = effective16；backbone **607113088**、条件投影 **1481472**、文本 **213325440** 参数，符合全尺寸模型而非验证小模型。第 1 步 loss **1.12718284**、grad norm **0.39844468**、峰值显存 **13.731 GiB**；第 10 步汇总 loss **1.01019645**、grad norm **0.35861245**、区间耗时 **3.373 秒/步**、峰值显存 **16.595 GiB**，状态为 running。

这消除了“尚无全尺寸首步运行证据”的限制：当前硬件上 CPU EMA 与 DDP 可以完成连续更新。无新增训练阻塞项；首个正式 checkpoint/回调及长程稳定性仍需由运行产物确认。Task 3 修复的 focused 测试据主控消息已通过，待固定修复提交和完整测试证据到位后另作范围内复审，不能把消息本身当作已审过的差异。

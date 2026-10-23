# Milestone 1 Task 1 审查（2026-10-22）

审查范围：固定差异 `730b3a6..da264fc`，仅 `t2v/models/pixart.py`、`t2v/models/factory.py`、`tests/test_pixart.py`。检查时这三个工作区文件相对 `da264fc` 无额外差异。依据为 Task 1 brief、`MILESTONE1_ARCH_PIXART_20261022.md` 与实施计划。本次按委托执行静态审查，不修改实现、不重跑测试、不启动训练。

## Standards / 代码质量

**未发现需要修复的明确问题（0 项）。**

- 模型、条件投影、工厂职责清楚；条件包只有 token 和 mask，没有夹带 pooled/global 条件。冻结 1024 维契约显式检查，原始缓存不在修改范围内。
- mask 在输入投影前处理，避免无效输入污染投影梯度；空文本/参考帧 dropout 使用有效零 token；SDPA 使用 True=valid，block 及最终输出清除 padded 帧。
- `pixart.py:249` 直接把当前 block 与显式参数交给非重入 checkpoint，没有循环闭包晚绑定问题；条件投影与 modulation 保留梯度路径。
- `pixart.py:268` 先验证全部目标 block/t_block key 与 shape，再执行加载；只选择允许移植的参数并输出排序后的 loaded/skipped 清单。缺失与形状不匹配会报错，没有静默退化成部分随机初始化。

## Spec / 规格符合性

**未发现需要修复的明确问题（0 项）。**

| 要求 | 静态证据 |
| --- | --- |
| 官方 adaLN-single block 与最终调制 | `pixart.py:151`：共享 timestep modulation 加逐层表，分别调制并门控 self-attn/MLP；cross residual 不额外归一化、不门控；`pixart.py:171` 保留最终两行调制表 |
| 原尺寸、逐帧输入输出、时间位置 | `pixart.py:177` 默认 28/1152/16；输入输出维度均为每帧 1024；固定时间 sin/cos；生产尺寸测试使用 meta device |
| 全局条件只有 timestep | `pixart.py:237` 的 timestep→t_block 与内容隔离；条件编码器接受但忽略 lengths，没有长度/文本/参考帧池化通路 |
| 1a/1b/1c 与层路由 | `pixart.py:187` 默认 `[T,T,T,R]×7`；`pixart.py:201` 的 prefix 模式完全不创建 cross 模块；`pixart.py:241` 为 prefix/both 拼接 ref+text，cross 路由逐层选择 |
| both 中 cross 只查询视频帧 | `pixart.py:156` 在 cross 路径排除前缀，前缀仍经 self-attn 与 MLP 演化，符合视频查询条件的规格定义 |
| block-only 本地初始化 | `pixart.py:256` 接受原始或 state_dict 包装的本地检查点；排除输入/输出/位置/文本桥等参数；无网络下载路径 |

公式与参数命名已对照官方源码：[PixArt.py](https://raw.githubusercontent.com/PixArt-alpha/PixArt-alpha/master/diffusion/model/nets/PixArt.py)、[PixArt_blocks.py](https://raw.githubusercontent.com/PixArt-alpha/PixArt-alpha/master/diffusion/model/nets/PixArt_blocks.py)。原图像 patch 与 xformers 替换为逐帧投影和原生 SDPA，属于规格要求的适配。

## 测试证据与结论边界

实施报告 `.superpowers/sdd/MILESTONE1_IMPLEMENTATION_PLAN_20261022/task-1-report.md` 记录：实现前缺失模块的 RED；专项 pytest **10 passed in 2.39s**；修复并行任务入口后的完整 `tests/run_all.py` **all tests passed，exit 0**；owned-path diff whitespace check 通过。本次阅读测试实现与报告，**未独立重跑这些命令**。

测试覆盖九种注入/路由组合、真实条件敏感度、padding 不变性、空条件安全性、checkpoint 与普通前后向全部参数梯度对照、显式残差方程以及合成官方命名检查点的完整加载/拒绝。测试通过不能等同于已验证真实生产规模 GPU 训练、真实官方下载权重或实验效果；这些属于后续集成/运行阶段。

结论：**Task 1 在本次范围内通过静态规格与代码质量审查，无阻塞项。** Standards 0 项，Spec 0 项；不对训练入口、实验配置、评测流程或最终 FVD 作本任务范围外的通过声明。

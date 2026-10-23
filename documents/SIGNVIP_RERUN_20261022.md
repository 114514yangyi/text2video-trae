# SignViP 论文条件重跑（用户授权，2026-10-22）

目标：使用公开权重完整生成原始test642，判断是否能复现Table3 FVD1025.45。论文/tmp/signvip.pdf（v2）正文确认50步、CFG3.5、272×224。公开CLI默认20步，本轮以论文50步为准。

数据：只读/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T原始210×260及官方字幕。按作者recipe编码libx264 crf0 yuv420p24fps，参考帧从MP4解码，PIL默认resize224×272。与本项目256×256缓存隔离。

生成：tools/signvip/reproduce.py调用现有发布权重加载器和官方pipeline，完整EOS长度，不在扩散前截32。DDIM50、CFG3.5、context24/overlap2、fp16，seed42+排序样本序号，分片不影响单样本噪声。官方utils.save_video双阶段编码，输出完整MP4与tokens，固定清单与指纹。

评测：直接官方styleganv/fvd.py，400维logits；原始视频同PIL resize到272×224，官方tensor缩放/中心裁剪到224²。主报告固定首32帧、短视频重复末帧、全642（本地声明协议，不冒称论文完整协议）；同时报告严格足32帧子集、完整视频均匀16帧、GT跳首帧敏感性，均预先声明，不通过选择子集N去凑1025.45。所有特征重新提取；统计与ID一起保存。

边界：论文未公开完整FVD窗口/N/短片策略，最终只能报告公开权重+所列协议是否接近论文，不能反推作者协议。公开GPT示例head_layers1与权重兼容的2存在差异，按严格权重加载成功的2执行。

产物：reports/eval/signvip_paper_repro_20261022_v2。前一准备目录仅含同批原始编码真值，v2通过软链复用并逐视频校验。无旧实验删除或覆盖。

验证：4个unittest通过；既有tests/test_eval_fvd.py通过；独立审查后增加分片拓扑冻结/排他锁及代码、权重、GT校验。GPU启动前动态查空闲，预计若干GPU小时，实测首例后更新。

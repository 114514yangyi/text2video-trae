# 直接文本→手语视频文献核验与文本微调设计

日期：2026-10-13。用户明确要求文本编码器参与微调、不冻结；要求查找和学习直接 text→手语视频论文。研究目标仍为 PHOENIX14T 句子级语义对齐，固定 1-token/帧 VAE、latent 扩散、x0 输出、纯文本条件、SignMbart 评测。

## 1. 结论与证据范围

存在直接文本条件的手语 RGB 视频生成研究，但“端到端”“无 gloss”“pose-free”不能互换。最贴近无 pose 路线的是 FlashSign、TuneSL 和 2026 Text2Sign；它们的长句语义、额外输入、预训练依赖、文本训练策略各有限制。没有检索到可直接证明“在我们的冻结 1-token latent 上，全量微调 mT5 必定改善句子语义”的证据。

已核验论文题名、作者或正式来源，读 FlashSign / Text2Sign / Text2SignDiff 的公开正文、TuneSL 的出版商摘要，核对 JiT 官方模型及当前本地代码。SignGen 的 51 MB PDF 超过网页工具内容限制，方法证据来自 ECVA 官方 PDF 的可检索段落（第 3/5 页）与作者仓库，未声称通读全文。SignMoD 只核对了作者单位公告与发表摘要，不能核实完整方法中的条件约束。未克隆外部代码、未下载 checkpoint、未改训练实现、未启动训练。

## 2. 直接 RGB 视频候选

### FlashSign

[FlashSign: Pose-Free Guidance for Efficient Sign Language Video Generation](https://arxiv.org/html/2603.27915v1)，Zhang et al.，2026；[作者仓库](https://github.com/AIGeeksGroup/FlashSign)。

- 正文 II-B：基于视频生成骨干，使用 3D VAE、预训练 UmT5 与视频 DiT，绕过 pose 中介；II-C 使用随时间步变化的稀疏 tile attention。
- 实验 III-A 是 WLASL 的孤立词；IV 明确承认长句、多 sign 连贯生成仍受限。
- 文中只写预训练 UmT5，没有明确给出其是否更新、学习率或冻结消融，不能说论文证实了文本全量微调。
- 仓库截至本次访问仅列 README，没有可用训练实现供核对。
- 对本项目的启发：pose-free 是成立的研究方向；条件语义和长句时间建模应单独验证。tile attention 的空间划分不直接适配我们的每帧 1-token，不能原样移植。不能更换冻结 VAE 或照搬其预测目标。

### TuneSL

[Towards End-to-End Text to Sign Language Video with Diffusion Models](https://link.springer.com/chapter/10.1007/978-3-032-25314-9_16)，Andreas Nikolaidis、Sotirios Chatzis；ECML PKDD 2024 workshop 论文集，出版商标注 2026-05-08 上线。

- 出版商摘要明确描述直接视频路线：预训练 T2I/T2V pipeline 在 How2Sign 上微调，输出 20 帧。
- 推理将一张图重复 20 次，并使用 DDIM inversion 得到结构引导的初始 latent。
- 所以无 pose 不等于只有文本；其结构引导不能作为本项目新增条件使用。
- 仅摘要可访问，不能确认它是否微调文本 encoder，也不能捏造底层 backbone/训练细节。
- 可学习的思路是对预训练生成能力做领域适配；但其现成骨干/VAE 与我们不匹配，不能直接加载至 `[T,1024]` DiT。

### Text2Sign（2026，非旧 GAN 同名工作）

[Text2Sign: A Single-GPU Diffusion Baseline for Text-to-Sign Language Video Generation](https://arxiv.org/html/2607.13164v1)，Ruize Xia，2026；DOI 10.1109/ACCESS.2026.3686260；[作者代码](https://github.com/xiaruize0911/text2sign)。

- 直接文本→短 RGB 视频，How2Sign，冻结 CLIP 文本 encoder，3D UNet + Transformer/factorized attention。
- 正文对照了冻结预训练 CLIP 与从头联合训练的自定义文本 encoder；这不是“同一预训练 encoder 冻结 vs 微调”的公平对照，不能用于否定预训练 mT5 微调。
- 作者报告正确与打乱 prompt 在留出诊断上区分很弱；最后两层 CLIP 解冻的短续训也未给出可支撑正向结论的证据。
- 对本项目最直接的启发：画面变好、去噪 loss 降低，仍可能缺乏语义控制。因此微调必须看 SignMbart 正确/错配配对差和绝对回译，而非仅 train loss。
- 其短片段裁剪、低分辨率和评价设置与本项目不一致，不混表比较。

## 3. 容易误归类的工作

### SignGen（ECCV 2024）

[SignGen: End-to-End Sign Language Video Generation with Latent Diffusion，ECVA 正文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06988.pdf)；[作者仓库](https://github.com/mingtiannihao/SignGen)。

官方正文第 3 页明确列出 motion 条件包括 optical flow、pose、depth；第 5 页图 2 将 text 与 motion 条件共同送入生成模型，文本经 CLIP。因此不能当作与本项目严格 text-only 同条件的直接基线。仓库 README 的 training 部分仍写 coming soon，虽然已有若干模块，不能称训练复现链路完整。文本 CLIP 更新策略未从现有证据确认。

值得学习的是非手部信号/动作时序与条件融合的重要性，不迁移 pose、光流、深度监督。

### Text2SignDiff（DICTA 2025）

[Text2Sign Diffusion: A Generative Approach for Gloss-Free Sign Language Production](https://arxiv.org/html/2509.10845v1)，Feng et al.；DOI 10.1109/DICTA68720.2025.11302423。

它是文本→pose 序列而非 RGB 视频。可借鉴 III-B 的句子级跨模态对比对齐和 III-C 的语义指导思路，但只作为机制启发，不能称直接视频基线。若后续适配，只用现有视频 latent 替代 pose 数据，冻结 VAE 不变。该适配效果尚未验证。

### SignMoD（2026，后续阅读线索）

[作者单位公告](https://ccs.imu.edu.cn/info/1053/2808.htm)、[发表摘要](https://pubmed.ncbi.nlm.nih.gov/42213566/)，DOI 10.1109/TPAMI.2026.3698334。

摘要描述多语言文本编码、SSM 时间建模和混合扩散；全文获取失败，是否严格 text-only、是否微调文本 encoder 均未确认。先列为待核实线索，不列为已验证的纯文本参考。

## 4. JiT 的具体纠正

[JiT 官方 model_jit.py](https://github.com/LTH14/JiT/blob/main/model_jit.py) 中 `LabelEmbedder`（约第 74–84 行）是 `nn.Embedding(num_classes+1, hidden_size)`；`JiT.forward` 接收类别 y，并把 `y_emb` 与时间向量相加。类别 embedding 可学习，但它不是 mT5/CLIP 自然语言文本 encoder。原论文不能作为“可微调文本 encoder 改善 text→视频”的实验证据。

这个纠正不影响用户的新决定：冻结 mT5 只是旧实现选择，不是 VAE 冻结契约的一部分；可以从现有预训练权重微调文本 encoder。

## 5. 已查明的本地改动范围

以下是现有代码事实，尚未修改：

- `t2v/adapters/text.py:42–46` 把 encoder `.eval().requires_grad_(False)`；`:55`、`:71` 两层 `no_grad` 阻断梯度。只改 requires_grad 不会训练文本模型。
- `t2v/train.py:240` optimizer 只包含 DiT 与 ConditionEncoder，没有文本 encoder。
- `t2v/train.py:388–390` 每步把所有参数组 lr 覆盖成同一值；新增文本参数组必须同步保留 lr 比例，否则独立文本 lr 会被覆盖。
- `t2v/training_utils.py:156–199` checkpoint 不含文本权重；`generation.py:52–60` 每次重新加载共享预训练文本模型。只解冻而不保存，会导致推理丢掉训练结果。
- `t2v/pipelines/visualize.py:195–198` 用初始化时预计算的空文本 embedding。微调后它会过期，CFG 必须重新编码空字符串。
- dev/可视化要把文本 encoder 一起切换到 eval，并在退出时恢复模式。去掉 no_grad 后也要将文本 forward 放入合适的 bf16/autocast 范围。
- 现有 EMA 只跟踪 DiT，条件 encoder 使用 raw。新文本微调实验先使用一致的 raw 全链路；不把 EMA DiT 与不匹配的文本状态混用。若未来支持 EMA 生成，需要跟踪完整一致的条件状态。
- 文本→长度模型目前独立使用原始预训练 mT5；这次先保留其自身约定，不把它的特征静默换成微调后的文本 encoder。自由文本视频仍由该模型预测长度。

## 6. 建议的有限改动设计（待用户确认）

采用 **预训练 mT5-base 全量微调 + 现有 DiT-B/prefix 联合优化**。不从随机初始化重训语言模型；LoRA 可作为未来省显存消融，但本次默认全部文本 encoder 参数可学习，落实“不冻结”的要求。

初始配方：文本 lr=1e-5，DiT/条件投影 lr=1e-4；文本 lr 是本项目起始假设，不冒充任何论文最佳值。开启梯度 checkpointing，必要时减小 microbatch 配合累积保持有效 batch 16。新增独立配置与 run 目录，旧配置/历史冻结 checkpoint 保持可复现。主评测与样例使用 raw，一致保存/恢复 DiT、ConditionEncoder、mT5 和 optimizer。

兼容策略：旧 checkpoint 以旧冻结文本路径加载；新微调 checkpoint 必须包含文本状态，缺失即报错；从旧 run warm-start 可以初始化 DiT/条件模块，文本从共享原始预训练权重开始，optimizer 重新建立；新 run resume 必须恢复文本状态与 optimizer。共享模型软链只读，新权重写入当前 run。

验证必须包括真实的非零文本梯度/optimizer 后参数变化、独立学习率不会被覆盖、checkpoint 恢复后 eval 文本输出一致、CFG 空文本随权重更新、dev/可视化不改变训练模式或污染 RNG，以及旧冻结路径兼容。通过小模型测试再做真实 mT5 短冒烟；正式训练尚未启动。

第一轮仅改变文本微调，保留 x0/cosine/prefix/CFG 配方，以 dev 的绝对 SignMbart 回译与正确−错配增益判断是否改善。对比 frozen/full 的相同初始化与相同训练预算，不能把 train MSE 的下降当作句子语义进步。

更换 encoder 为其他多语言模型、额外跨模态语义损失、双流 backbone 均留在单独消融中。GPU 时间增加无法自动补足语言—动作对齐数据。

## 7. 研究覆盖与剩余不确定性

查询围绕直接 text→RGB、pose-free、文本 encoder 微调、条件弱敏感度负结果，以及 SignGen/FlashSign/TuneSL 的正式方法。检索到的技术结论只采用原论文、出版商摘要、作者代码或作者单位发布的来源；二手网页仅用于发现题名。

已覆盖标准方法、较近文本基线、同类失败、可迁移对齐机制、负结果与代码可用性。仍不确定：SignGen CLIP 更新策略、FlashSign UmT5 是否训练、TuneSL 完整训练设置、SignMoD 完整条件。因此没有提出“解冻一定有效”或将不同口径论文数字与 SignMbart 混表的结论。

本轮仅添加研究记录。brainstorming 的有限改动设计确认环节尚未完成，训练代码仍是冻结 mT5。

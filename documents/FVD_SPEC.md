# FVD 规范（FVD_SPEC，2026-10-27 生效）

> 状态：**生效**（2026-10-27，用户指令）。本文件是仓库内 FVD 口径的**唯一来源**，
> **取代并删除**旧文件 `FVD_SPEC_20261021.md`（2026-10-21 版 + 2026-10-24 SignViP 强制版）。
> `RULES.md` 方法硬约束第 5 条、评测规范第 6/6.1 条、研发探索规范第 19 条与本文件绑定；
> 任何 FVD 数字必须能追溯到本文件的协议块，否则不得进对比表、不得作为晋级依据。
> 历史文档里对 `FVD_SPEC_20261021.md` 的引用只作历史考据，不再是有效规则。

## 0. 一句话规则

**所有 FVD 测试只使用 `video_metric` 包的 `calculate_fvd` 原样计算**
（`method='styleganv'`、`only_final=True`）；仓库内**不再保留任何旧的 FVD 计算脚本、
旧实现副本、旧封装和旧计算约束**（2026-10-27 用户指令，旧的已全部删除）。

“原样”是指：FVD 的每一步数学（取帧后的空间预处理、I3D 特征、特征统计、Fréchet 距离）
都由 `video_metric` 的上游代码执行，本仓库只负责“把哪些帧、以什么形状喂进去”和“把结果写成报告”。

## 1. 实现（唯一允许的 FVD 计算）

- **上游**：`JunyaoHu/common_metrics_on_video_quality`（下称 `video_metric`）。
  逐字节副本固定在 `third_party/video_metric/`，SHA256 见 §3。
- **调用**：`calculate_fvd(videos1, videos2, device, method='styleganv', only_final=True)`，
  FVD 值 = `result['value'][0]`。内部全链路 =
  `fvd/styleganv/fvd.py` 的 `get_fvd_feats`（含 `preprocess_single`）与 `frechet_distance`。
- **禁止**在仓库内重写 FVD 的预处理 / 特征统计 / 距离 / 特征提取。判断标准：仓库代码中
  不得出现 `frechet_distance`、`compute_stats`、`preprocess_single`、`get_fvd_feats`
  的本地实现，也不得用 `np.cov` / `sqrtm` 等自行拼距离。`tools/eval_video_metric_fvd.py`
  只做数据 IO 和报告，不含任何指标数学。
- **加载方式**：入口先在 `sys.modules` 里预置父包 `video_metric`，再用 `importlib`
  导入 `video_metric.calculate_fvd`。这样**不执行**上游的 `video_metric/__init__.py`
  （它会经 `metric.py` 引入本项目环境没装的 `pytorch_msssim`），但真正被导入的
  `calculate_fvd.py` 与 `fvd/**` 仍是逐字节原样。
- **方法固定 `styleganv`**：`videogpt` 与 TensorFlow 版不进入正式口径。`fvd/videogpt/*.py`
  只是因为上游 `fvd/__init__.py` 会 import 它才随包存在，其权重不落盘、不参与评测。
- **I3D 权重**：`third_party/video_metric/fvd/styleganv/i3d_torchscript.pt`，
  SHA256 `bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2`；
  与 `models/i3d/i3d_torchscript.pt` 是同一个文件（本地资产，`*.pt` 被 git 忽略）。
- **来源说明**：用户指定的 `/home/huyang/data/trae_projects/video_metric` 在评测机上不可用，
  本轮从本机同源副本 `/home/hulala/gsw/SLP2025/VideoVAET/video_metric` 逐字节拷贝。
  若之后拿到用户路径下的副本，先用 §3 的 SHA 表复核等价，再决定是否替换。

## 2. 数据层协议（2026-10-27 保持不变）

| 项目 | 取值 |
|------|------|
| 数据集 | PHOENIX test 单侧；固定 ID 清单（当前 **640** 条，清单落盘复用） |
| 取帧 | 源 **frame0..23 连续 24 帧**（first24）；不补帧、不跨帧间隔 |
| 源几何 | 统一 **256×256**（PIL bilinear、RGB）；E7/现有 GT 本身就是 256 方形 |
| 张量 | `[B, T, C, H, W]`、float32、像素 **[0,1]**（即 `calculate_fvd` 的入参；内部转 BCTHW） |
| 上游预处理 | `preprocess_single`：张量双线性**无抗锯齿**、短边 224、center crop 224、`[-1,1]` |
| 特征 | I3D-K400 的 **400 维 logits**（不经 softmax） |
| 距离 | 上游 `frechet_distance` |
| 真值本底 | `real_split_half_fvd`：真值按 ID 顺序对半分（各 `N//2` 条，N 为奇数时丢弃最后一条），同样走 verbatim `calculate_fvd` 调用 |
| 短片 | 源不足 24 帧的 ID **跳过**并记入 `skipped_short`；生成不足 24 帧 **直接报错**，禁止补帧 |
| 生成侧 | GT 长度、raw/EMA 权重、采样器/CFG/步数/种子各自声明，跨臂必须一致 |
| 可比性 | 跨模型比较必须共用同一真值、同一 ID 清单、同一取帧、同一尺寸、同一代码；**不得混 N** |

`calculate_fvd` 的 `bs`（内部 `get_feats` 默认 10）与 CLI 的 `--batch-size` 只影响显存/速度，
**不影响数值**，因此不写进协议块。`only_final=True` 表示只算整段 24 帧的 FVD，
不使用 `only_final=False` 的 10..T 逐长度曲线。

## 3. 版本固定与校验

每次运行前逐文件 SHA256 校验，不匹配即拒绝运行；报告里记录整张表。

| 文件（相对 `third_party/video_metric/`） | SHA256 |
|---|---|
| `__init__.py` | `58997fdc7845bd897b81956e040f851fd0ac8283362ab54dbdd51e6d6a6ec4f0` |
| `calculate_fvd.py` | `ad36575e9377c0c97f2dbb37a4b4454a2df53e850c3fd32e13756ad53d77c751` |
| `fvd/__init__.py` | `ba54c88d1db05946bca93deed0e4ad4f22f593b3e31aabf4ca834666c2c3a314` |
| `fvd/styleganv/__init__.py` | `274caa9c7d7bc753361dcceb7c799a94321d4d2f503b862943f172b19b1408a0` |
| `fvd/styleganv/fvd.py` | `c9a1acbf811c3050d92b5289f4a4789ba6209a71a5bafbeba10df5576ffbda37` |
| `fvd/styleganv/i3d_torchscript.pt` | `bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2` |
| `fvd/videogpt/__init__.py` | `a65e1cf5446c0b861e6c7a7b4d316a7798155a8a7df2f0199819e9381ddc6aa3` |
| `fvd/videogpt/fvd.py` | `a1fb54c15461c84761988bb7307e7d1b13feafceb6beb9b64c1a07bacb81f651` |
| `fvd/videogpt/pytorch_i3d.py` | `e4b241d82ae0432cd19410ce4fb08ab1003c09721aca8b907cdb8a21e15086f8` |

- 不得手改 `third_party/video_metric/` 下任何原样文件。需要升级 = 用户指令 + 换副本 +
  重算 SHA + 更新本表 + 记 TODO，并说明旧结果是否仍可比。
- 来源、拷贝命令与“为什么不执行 `__init__.py`”写在 `third_party/video_metric/PROVENANCE.md`。

## 4. 运行入口

- **正式入口（唯一）**：`tools/eval_video_metric_fvd.py`（IO 壳）。
- **里程碑链**：`tools/eval_milestone.py` 生成后固定调用该入口
  （`--fvd-tool video_metric` 只是兼容参数，没有第二种工具）。
- 典型命令（test 固定清单、first24、raw 权重）：

```bash
# 1) 真值本底（只算 real split-half，不评分）
python tools/eval_video_metric_fvd.py \
  --ids-file reports/eval/<run>/ids.txt \
  --frame-root data/phoenix_frames --cache-dir data/phoenix_latents/v1 \
  --fill-ref-cache --gpu cuda:0 --out reports/eval/<run>/real_floor.json

# 2) 生成臂评分（--generated 指向 <run>/frames，内部结构 frames/test/<video_id>/images*.png）
python tools/eval_video_metric_fvd.py \
  --ids-file reports/eval/<run>/ids.txt \
  --frame-root data/phoenix_frames --cache-dir data/phoenix_latents/v1 \
  --generated reports/eval/<run>/frames --gpu cuda:0 --out reports/eval/<run>/fvd.json
```

**GPU 可见性（上游原样行为，必须注意）**：`calculate_fvd` 内部调用的上游
`load_i3d_pretrained` 会把 I3D 包进 `torch.nn.DataParallel`（没有 `device_ids`），
因此**会用到当前所有可见 GPU**，并要求模型在第一个可见卡上。入口已做防护：
`--gpu` 给 `cuda:N` 且未设 `CUDA_VISIBLE_DEVICES` 时，先把 `CUDA_VISIBLE_DEVICES` 钉成 `N`
再用 `cuda:0`；若外部已设 `CUDA_VISIBLE_DEVICES`，只接受 `--gpu cuda:0`（否则显式报错）。
自动评测链的既有写法 `CUDA_VISIBLE_DEVICES=<空闲卡> ... --gpu cuda:0` 不受影响。
这样既保持上游计算原样，又不会把别人的训练卡拉进来。

## 5. 报告格式

`fvd.json` 至少包含：

- `num_videos`、`video_ids`、`skipped_short`
- `fvd`（生成侧评分时）与 `real_split_half_fvd`（本底）
- `ids_file_sha256`
- `protocol` 块：`engine`（video_metric 原样 calculate_fvd）、`vendored_root`、
  `upstream_sha256`（§3 整张表）、`weights_sha256`、`feature`、`feature_dim`、
  `frame_mode`、`num_frames`、`real_frame_offset`、`padding`、`source_resize_hw`、
  `source_resize`、`preprocess`、`distance`、`wrapper_sha256`

只写一个 “FVD” 而不注明 backbone / 取帧 / 分辨率 / N / 实现 SHA 的数字视为无效；
`real_split_half_fvd` 缺失的评测不得用作晋级依据（先怀疑管线）。

## 6. 已删除的旧内容（2026-10-27 用户指令，不可恢复）

用户原话：“以后所有的 FVD 测试不再使用原先的任何脚本了，而是使用 video_metric 中的 FVD
原样计算方式……把以前的 FVD 计算脚本以及计算约束都删除。”

本轮删除：

- **旧计算脚本**：`tools/eval_fvd.py`、`tools/eval_signvip_fvd.py`、
  `tools/redecode_fvd_frames.py`、`tools/fetch_i3d_weights.py`
- **旧实现副本**：`third_party/signvip/`（含 README）
- **旧测试**：`tests/test_eval_fvd.py`、`tests/test_signvip_fvd.py`，
  以及 `tests/test_milestone_eval.py` 中两个只测旧脚本的 FVD 测试
- **旧计算约束**（随 `FVD_SPEC_20261021.md` 一并删除）：SignViP 强制入口、SignViP 源码/
  I3D 的 SHA 锁定、`--self-check≈131` 自检、E7 全 640 与共同 617 的双 N 规则、
  “旧 helpers 仅供历史测试/诊断”、`--preprocess pil_aa` 路径等。这些条款不再有效，
  由本文件 §1–§5 取代。
- **历史 FVD 报告产物**：`reports/eval/**/fvd*.json`、`fvd*.npz`、`fvd*.lock`、
  `fvd_command.json`、`real_floor.json`，以及 `reports/eval/fvd_ref/`。
  生成帧、`ids.txt`、`protocol.json` 等**非 FVD 产物保留**。

`NOTES.md` / `EXPERIMENT.md` 里的历史数字（129.7493、142.3202、130.8486 等）保留为历史记录，
但已没有对应证据文件可复算；引用时必须注明“历史数字，报告产物已删”。

## 7. 为什么切换后数值不变

2026-10-24 用的 `third_party/signvip/fvd.py` 与 `video_metric` 的 `fvd/styleganv/fvd.py`
**SHA256 完全相同**（`c9a1acbf…da37`），I3D 权重逐字节相同（`bec6519f…0a5e2`），
数据层协议也相同（first24 / 256² / 不补帧 / 同 ID 清单）。所以本次切换是
**实现治理与脚本清理**，不是口径变更：同一批帧喂进新入口，FVD 数值与旧入口一致。

## 8. 变更历史

- **2026-10-21**：建立官方 I3D-FVD 口径，旧 R(2+1)D-18 / 16 帧数字作废。
- **2026-10-24**：改为 SignViP 原样实现（`FVD_SPEC_20261021.md` 的 2026-10-24 版）。
- **2026-10-27**：改为 `video_metric` 原样计算；删除旧脚本、旧实现、旧约束与历史 FVD
  报告产物（本文件）。决策登记 TODO **D76**。

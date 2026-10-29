# FVD 规范（FVD_SPEC）

> 状态：**生效**（2026-10-29，用户指令）。本文件是仓库内 FVD 口径的**唯一来源**；
> `AGENTS.md` 方法硬约束第 5 条、`RULES.md` 评测规范第 5/6.1 条、研发探索第 19 条与本文件绑定。
> 任何 FVD 数字必须能追溯到本文件的协议块（`fvd.json` 的 `protocol` + `floor`），
> 否则不得进对比表、不得作为晋级依据。
> 取代 `FVD_SPEC_20261021.md`（16 帧 `tools/eval_fvd.py` 口径，已归档到
> `archive/FVD_SPEC_20261021.md`）与 2026-10-27 的 `video_metric/styleganv` 口径。

## 0. 一句话

正式 FVD = **`videovaet_fvd`** 的 `fvd_mean`：把每条生成视频与同一条真值视频各自送进
VideoVAET 的评测变换与 `video_metric.calculate_fvd(method='videogpt', only_final=True)`，
得到**逐 clip** 的一个数，再对所有 clip 取**算术平均**。

它**不是** pooled Fréchet（把整批 400 维特征拼起来估均值+协方差算一次）。两者数量级不同
（同一批 SignViP 特征：per-clip 均值 1067.68 / pooled 191.41），**不可混表、不可换算**。

## 1. 工具与代码身份

| 项目 | 取值 |
|------|------|
| 实现包 | **`third_party/videovaet_fvd/`**（仓库内唯一副本，2026-10-29 用户指令随主仓库发布；`VIDEOVAET_FVD_ROOT` 可指向别的 checkout 做交叉验证） |
| 仓库内入口 | `tools/eval_videovaet_fvd.py`（只在本地做帧的发现/校验/视图与落盘，指标全在包里） |
| 里程碑接线 | `tools/eval_milestone.py` 生成完成后固定调用上面这个入口 |
| 参考实现 | `/data/huyang/VideoVAET`（SLP/SLRT 副本），`seq_scriptsT.py::seq_eval` → `video_metric/metric.py::compute_all` |
| 逐字节 vendored 文件 | `video_metric/calculate_fvd.py`、`fvd/videogpt/{fvd.py,pytorch_i3d.py}`、`i3d_pretrained_400.pt`(50.9 MB)、`dataset/video_augmentation.py`；sha256 见包内 `videovaet_fvd/vendor/MANIFEST.sha256`，每次加载校验 |
| 自检 | `third_party/videovaet_fvd/bin/videovaet-fvd verify`（manifest OK、无绝对路径字面量）、`bin/videovaet-fvd selftest`（本机 2026-10-29 实测 `fvd_mean = 0.5691307187080383`；包里记录的 0.5694066286087036 是同口径下不同 OpenCV mp4 编码器的结果，差异 3e-4，容器级噪声，不是代码差异） |

**禁止**：在本仓库保存该包副本、复制 vendored 文件、另写预处理/特征统计/距离。
旧实现（`tools/eval_fvd.py` R(2+1)D、`tools/eval_signvip_fvd.py`、
`tools/eval_video_metric_fvd.py` styleganv 分支、`tools/redecode_fvd_frames.py`、
`tools/fetch_i3d_weights.py` 及对应测试）已于 2026-10-25 / 2026-10-29 删除，不得恢复。

## 2. 协议（包内 `PROTOCOL.md` 的摘要）

| 项目 | 取值 |
|------|------|
| 骨干 | I3D Inception-v1，Kinetics-400，torch `i3d_pretrained_400.pt` |
| 特征 | **400 维 Kinetics-400 logits**（softmax 前） |
| 变换 | `Compose([Resize(224), ToTensor(), FixedSample(52)])`；`ToTensor` = `x/128-1` → [-1,1] |
| 帧采样 | `l > 52` → `np.linspace(0, l-1, 52)` 全片段均匀；`l <= 52` → `l=(l//4)*4` 保留**前** l 帧（本仓库两侧都先裁到同一 N≤52，见 §3） |
| 几何 | 解码后 **stretch 到 224×224**（`cv2.INTER_LANCZOS4`），**不保持长宽比、不 center crop** |
| 距离 | `calculate_fvd(..., method='videogpt', only_final=True)`；`test_batch_size=1` ⇒ N=1 ⇒ `frechet_distance` 只取均值项，协方差项被跳过 |
| 聚合 | `np.mean(per-clip FVD)` |
| 环境 | 本机 `torch 2.6.0+cu124` / `numpy 1.26.4` / RTX 4090；换成 CPU 也应有相同数值（浮点舍入内） |

上面的 `N=1` 是关键：**这条指标度量的是"每一对（生成,真值）clip 的 400 维 logits 均值之差的平方"，
再对 clip 取平均**，所以它的量纲比 pooled FVD 大一个数量级，且 N 越大越稳。

## 3. 数据层（本仓库固定，不随工具改变）

| 项目 | 取值 |
|------|------|
| 划分 | **test 单侧**；清单 = **全部 642 条源**（`reports/eval/samples/m1_test_first52_ids.txt`），采样后 T<10 的片段由包自己跳过并写进 `skipped` |
| 每条的帧数（窗口） | **N = min(52, 源帧数)**。源 ≥52 帧 → 用满 52；源 17..51 帧 → 用满它自己的帧数 |
| 真值 | `data/phoenix_frames/test/<id>/images0001..N`（video frame 0..N-1，**frame0 = 参考帧，参与评测**），经 N 帧 symlink 视图 `data/phoenix_frames_first52/` 送入 |
| 生成 | `<eval>/frames/test/<id>/images0000..N-1`（`save_sample` 写 frame0 = 参考帧 + 解码帧；`--decode-frames 52` 且按 GT 长度封顶） |
| 补帧 | **禁止**。生成帧数必须**恰好等于**计划的 N，否则入口直接报错（防止把 24 帧旧产物当 52 帧评分） |
| 编码 | 两侧都直接读 PNG；生成侧不经 mp4 重编码 |
| 本机实测分布（642 条源） | 源帧数 min 17 / p10 45 / median 99 / p90 155 / max 242；**556 条（86.6%）用满 52 帧**，其余是 48(14) / 44(15) / 40(13) / 36(13) / 32(14) / 28(6) / 24(9) / 20(1) / 16(1)；**没有一条**落到 T<10，故本数据 0 条被跳过 |

为什么必须做 N 帧视图：包的默认行为是"读整段真值 → 均匀取 52 帧"。整段真值常 70–140 帧，
若直接用整段，`FixedSample(52)` 会把 52 帧铺满整个视频，而生成侧只有前 N 帧——不是同一时间窗。
视图先按 N 裁好，两侧再走 `FixedSample(52)`：N ≤ 52 时它是恒等（N 是 4 的倍数）或对两侧做**同样的**
向下取整到 4 的倍数（N 不是 4 的倍数时，例如源 45 帧 → 两侧都取前 44 帧），因此 T 恒相等。

## 4. 本底（floor）

包的协议没有"本底"，本仓库要求同时报告，定义 = **同一 verbatim 管线跑真值互配**：
clip i 与 clip i+N/2（N=640 → 偏移 320）按名字配对，仍然逐 clip `calculate_fvd` 后取均值，
写进 `fvd.json` 的 `floor.fvd_mean`。

含义：两条**互不相关**的真实视频在这个特征空间里的平均距离。`fvd_mean` 若明显低于 `floor`，
说明生成结果在内容上确实贴近自己的真值；若接近 `floor`，说明生成与真值的关系不比"另一条视频"更强。

## 5. 报告与引用规则（强制）

1. 写全称：`FVD (videovaet_fvd videogpt I3D-K400 logits, per-clip window min(52, 源帧数), 224² stretch, per-clip mean, N=642)`；
   只写 "FVD" 时必须能在同段或 `fvd.json` 里读到这几项（`protocol`、`window_histogram`）。
2. 必须同时报 `n`、`skipped` 与 `floor.fvd_mean`；缺任一项视为无效。
3. **禁止混表**：R(2+1)D（22–26、51.6–58.4）、16 帧 `eval_fvd.py`（132–185）、
   32 帧 SignViP、`video_metric/styleganv`（pooled 估计量）、2026-10-29 上午的 24 帧过渡口径
   一律不得与新数字并列或相减；这些历史数字只作追溯。
4. **不跨 N 比较**：`fvd_mean` 是逐 clip 平均，N 变化会让分布尾部权重变化；对比必须同清单。
5. 跨数据集/跨帧数/跨分辨率/跨骨干的数字一律标注"口径不可比"。
6. 文献数字多为 pooled FVD（如 RWTH-2014T 1025.45），与本表 `fvd_mean` 量纲不同，
   只能作量级参照，不能相减。
7. 回译（SignMbart）不再作为对比/晋级指标；如作诊断，不得与本表混排（TODO **D76**）。

## 6. 复现命令

```bash
PY=/data/huyang/miniconda/envs/onvae/bin/python

# 0) 工具自检（改代码/换机器后必跑）
third_party/videovaet_fvd/bin/videovaet-fvd verify
PYTHON=$PY third_party/videovaet_fvd/bin/videovaet-fvd selftest

# 1) 对一个已有生成目录算 FVD（含本底）
CUDA_VISIBLE_DEVICES=0 $PY tools/eval_videovaet_fvd.py \
  --ids-file reports/eval/m1_E5_first24_20261026/ids.txt \
  --generated reports/eval/m1_E5_first24_20261026/frames \
  --device cuda:0 --floor \
  --out reports/eval/m1_E5_first24_videovaet_20261029/fvd.json

# 2) 只算真值本底（eval_milestone --prepare 走这条）
CUDA_VISIBLE_DEVICES=0 $PY tools/eval_videovaet_fvd.py \
  --ids-file <ids.txt> --floor-only --device cuda:0 --out <dir>/real_floor.json

# 3) 端到端（生成 + FVD，52 帧上限）
$PY tools/eval_milestone.py --config runs/m1_E5/config.yaml \
  --checkpoint runs/m1_E5/final.pt \
  --out reports/eval/m1_E5_first52_videovaet_<date> --gpu cuda:0 \
  --decode-frames 52 --min-source-frames 12 \
  --ids-file reports/eval/samples/m1_test_first52_ids.txt
```

耗时（本机，3 臂并行时相互拖慢）：生成 642 条约 50–80 min；FVD 两遍（生成侧 + 本底）
约 9–15 min/遍。串行单卡约 45 min 生成 + 20 min 评分。

## 7. 里程碑三臂（2026-10-29 52 帧口径，当前唯一有效表）

| 臂 | 因子 | N | fvd_mean | median | std | floor | 评分卡 |
|---|---|---:|---:|---:|---:|---:|---|
| E5 | 4a（文本侧全冻 12 层） | 642 | **969.43** | 888.78 | 395.36 | 1503.97 | 4090 |
| E3 | 2c（ref 层看 [ref;text]） | 642 | **985.06** | 930.90 | 383.74 | 1503.97 | 4090 |
| E7 | 1b（前缀/context 注入） | 642 | **993.37** | 920.81 | 388.89 | 1503.84 | 3090 |

- 三臂两两差 15.6 / 8.3 / 23.9（相对 1.6% / 0.8% / 2.4%），排序 **E5 < E3 < E7**；
  单 seed，仍属"同一量级、不可强裁决"，但比 24 帧口径的 ≤0.94% 分得更开。
- 三者都约为本底（≈1504，"两条互不相关真值"）的 64–66%。
- E7 的评分卡是 3090（其余 4090），实测跨卡相对差 ~9e-5（floor 1503.84 vs 1503.97），
  折算到 `fvd_mean` 约 ±0.1，远小于臂间差。
- 逐 clip 窗口：556 条（86.6%）用满 52 帧，其余用满自己长度（48…16）；`skipped = 0`
  （本数据最短 17 帧 → T=16 ≥ 10）。
- E8 没有 `final.pt`，未参与；E1/E2/E4 的帧已在 2026-10-27 清盘删除，需各自 checkpoint
  重新生成后才能补表。

报告与日志：`reports/eval/m1_{E3,E5,E7}_first52_videovaet_20261029/fvd_videovaet.json`、
`log_eval_{E3,E5,E7}_first52.log`；台账 `EXPERIMENT.md` exp-117/118/119。

> **历史（已作废）**：同日更早的 24 帧过渡口径给 E3 1107.53 / E7 1112.43 / E5 1117.90
> （floor ≈1625，N=640），只作追溯，不得与本表并列。

## 8. 已知限制

- 逐 clip N=1 的 `frechet_distance` 不含协方差项，**不是**标准 FVD 估计量；它比 pooled FVD
  更依赖单视频的 logits 均值，也更噪声；`fvd.json` 的 `fvd_std/median/min/max` 一并给出，
  判断"是否真的更好"要看分布而不只看均值。
- `FixedSample(52)` 只对「源 ≥52 帧」的 clip 是均匀抽样；本仓库先把两侧裁到 N=min(52, 源帧数)，
  所以实际跑的是 N 帧（86.6% 的 clip 用满 52；短片段用满自己长度）。换窗口时两侧必须同步改，
  并重跑本底。
- 真值视图是 symlink，重建/删除都无损；`--rebuild-gt-view` 可强制重建。

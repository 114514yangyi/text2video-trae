# 清理执行记录（2026-10-29）

授权：B1–B5 全做，删除前先 `tar` 归档；里程碑 `final.pt` 与 52 帧评分帧都保留。
归档位置：**`/data/huyang/t2v_archive_20261029/`**（每个批次/每个 run 一个 `.tar` + `.paths.txt` 清单 + `.json` 元数据）。

> 事故记录：最初把归档写在 `/home/huyang/t2v_archive_20261029`，但 `/home`（`/dev/sda4`，659 G）已用 100%，
> 写 11 GB 的 B3 大包时报 `No space left on device`。**删除发生在 tar 校验之后**，所以源文件未被破坏；
> 已改为写到 `/data/huyang/t2v_archive_20261029`，并删掉那份截断的 tar。`/home` 现回到 11 G 空闲。

## 已执行

| 批次 | 内容 | 归档包 | 回收 |
|---|---|---|---:|
| B1 | `models/.hf_cache`（98 MB）、8 个 `__pycache__`、`.pytest_cache`、2 个 `*.bak`、26 个 `bt/tmp_*.txt` | `B1-caches-and-temp.tar` | 99.1 MB |
| B2 | `reports/eval/m1_E5_first24_20261026`、`reports/eval/_discarded`、`data/phoenix_frames_first24` | `B2-void-protocol-artifacts.tar` | 2.1 GB |
| B3 | 35 个旧 run 的 `samples*/` + `wandb/` + `samples_*.pt`（保留 `config.yaml`/`metrics.jsonl`/`status.json`/`*.pt`） | `B3-<run>.tar` × 35 | 8.7 GB |
| B4 | `reports/eval/{gt64_rotated,gt64_rotated_v2,dev64_dpmpp50,dev64_steps250,dev64_prefix_ladder,semantic_followups,preferred_route,bt_prefix,vae_recon_dev64,vae_recon_test32,cfg_sweep_dits_x0,probe_shuffled_final,prefix60k_dev64_*}` | `B4-old-eval-dumps.tar` | 2.1 GB |
| B5 | `runs/m1_E{3,5,7}/samples/step_*` 只留最后 3 个（28k/29k/30k） | `B5-milestone-sample-prune.tar` | 0.49 GB |

（B3 合计 11.1 GB 中，前 2.4 GB 是在 `/home` 那一轮完成的，日志见 `/tmp/opencode/slim_B3.log`。）

## 未执行（用户未选 / 计划外）

- B6：`.dsh/`（33 MB）、根目录 80 个 `log_*.log`（1.1 MB）归档到 `reports/logs/`、`git gc`。
- `runs/route_j_dynamic_20k/failed_attempts_20261014`（489 MB）：不在 B3 定义内，保留。
- 里程碑 `final.pt`（38 GB）与 `reports/eval/m1_E*_first52_videovaet_20261029/frames`（7.4 GB）：按授权保留。

## 结果

| 指标 | 前 | 后 |
|---|---:|---:|
| 仓库总体积 | 63 GB | **47 GB** |
| `runs/` | 51 GB | 39 GB |
| `reports/` | 13 GB | 8.1 GB |
| `data/` | 191 MB | 129 MB |
| `models/` | 98 MB | 4 KB（只剩 mt5 软链） |
| 归档 | — | 16 GB @ `/data/huyang/t2v_archive_20261029/` |

> 注意：归档与仓库在**同一块 15 T 盘**上，所以 `df` 可用空间基本没变（274 GB）。
> 这批清理的收益是**仓库目录变干净**；如果也要腾空间，确认无误后直接删归档目录即可。

## 验证

- `pytest tests` → **92 passed**（代码未改动）。
- `tools/eval_videovaet_fvd.py` 用 4 条 clip 冒烟：`fvd_mean = 999.3019`，与清理前同输入同值 → 52 帧视图、vendored 包、评分链路均正常。
- 保留项核对：`data/phoenix_frames_first52` 在；`reports/eval/m1_E*_first52_videovaet_20261029/{frames,ids.txt,protocol.json,fvd_videovaet.json}` 在；`runs/m1_E{3,5,7}/{final.pt,config.yaml,metrics.jsonl,status.json}` + 各 3 个 sample step 在；`reports/eval/matrix` 649 个 json 在。

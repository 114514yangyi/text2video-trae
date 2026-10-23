# 里程碑1 评测修复的范围内复审（2026-10-22 22:35–23:00）

复审对象：`85952a6`（"Validate evaluation source contracts and skip short real clips first"），只改
`tools/eval_milestone.py`、`tools/eval_fvd.py`、`tests/test_milestone_eval.py`。目标是核验
`MILESTONE1_FINAL_REVIEW_20261022.md` 的两条 P2 是否真正关闭，尤其"fixed 提交 + 完整测试证据"这一验收缺口。
本轮只读代码与产物，不修改源码、不启动训练/GPU 评测。

## 结论

**通过。两条 P2 均已关闭，未发现新的 P1/P2；只留 4 条 P3（下面逐条列出，均不阻塞 E1）。**

## 已核验的 P2 关闭证据

1. **P2-1 评测源/解码契约混用 → 已关闭。** `evaluation_source_contract`（`tools/eval_milestone.py:71-79`）
   把 `data.cache/stats/frame_root` 与 `vae.{config_path,checkpoint_path,repo_path}` 绝对化后 `resolve()`
   再比较；`main` 在 **创建 manifest、输出目录、加载 dataset 之前**（`:125-132`）调用
   `validate_evaluation_contract`，同时保留 sampling 一致性检查，且不要求 `training.out_dir`、`evaluation.ids_file`
   等无关设置一致。生成侧的反归一化统计、VAE、参考帧路径确实全部取自 checkpoint config
   （`t2v/pipelines/milestone.py:102-103/:111-115/:169-170`）；checkpoint 保存的是 resolved 纯 dict
   （`t2v/milestone_runtime.py:88-93`），加载后被 `OmegaConf.create` 包回（`milestone.py:85`），属性访问安全。
   - Lead 额外做了**真实资产端到端负向验证**：复制 pilot run 的 config、只把 `data.cache` 改成不存在的
     `data/phoenix_latents/v1_contract_probe`，用真实 `runs/m1_pipeline_validation_20261022/final.pt` 跑
     `tools/eval_milestone.py --gpu cpu --out /tmp/m1_contract_out` → 在 `:129` 抛
     `ValueError: Requested source/decoder contract differs from checkpoint config`，**exit 1 且输出目录未创建**，
     证明是 fail-fast 而非生成后才报错。

2. **P2-2 先判生成长度再判源片段 → 已关闭。** `tools/eval_fvd.py:389-401` 现在先按**真实片段**帧数
   与 `min_frames` 比较并 `continue`（`skipped_short` 记录 id 与真实帧数），之后才 `paired_paths` 校验
   生成长度；`>=24` 源的生成片段不足 24 仍会在 `:214-215` 抛错；`--fill-ref-cache` 分支走同一套跳过逻辑；
   `skipped_short` 同时写入 prepare 报告（`:480`）与 scoring 报告（`:514`）。

## Lead 独立复跑的测试证据

- `OMP_NUM_THREADS=4 /home/wxy/anaconda3/envs/text2video/bin/python -m pytest tests -q`
  → **91 passed, 6 warnings, 10.90s**，日志 `/tmp/m1-lead-final-suite-20261022.log`。
- `... -m pytest tests/test_milestone_eval.py tests/test_eval_fvd.py -q`
  → **22 passed, 4 warnings, 3.96s**，日志 `/tmp/m1-lead-eval-focused-20261022.log`。
- 十臂配置矩阵：用 `load_config` 逐臂读 E1–E10，五因子、`injection`、注入表（`[T,T,T,R]×7` / 全 ref / 无 cross）、
  28 层/1152 宽/16 head、30k 步、effective batch 16 与规格 §5 全部一致。

## P3 残留（不阻塞 E1，登记为后续清理）

1. `tools/eval_fvd.py:398-400` 的第二处长度检查在 contiguous/24 下不可达，只在
   `--num-frames < --min-frames` 这种自相矛盾的参数组合下被触发，并把"合格长片段"误标成"短片"。
   建议 CLI 入口显式要求 `num_frames >= min_frames`，或删掉该分支。
2. `tools/eval_fvd.py:393` 的资格判断用的是"偏移后"真实帧数；`--include-frame0` 关闭时，
   24 源帧会变成 23 而被跳过，与 `prepare_manifest` 的"源 `num_frames>=24`"规则表述不一致。
   milestone wrapper 固定传 `--include-frame0 --min-frames 24`，因此当前协议不受影响。
3. 契约相等是**路径字符串身份**：同一路径内容被替换不会被发现；FVD 真值特征缓存的 key
   （`eval_fvd.py:422-428`）不含 `frame_root`/源内容指纹，理论上可复用异源缓存。
   milestone 全流程使用固定 `frame_root` + 共享清单，风险低，但换源时应强制 `--no-ref-cache` 或加源指纹。
4. 测试覆盖小缺口：别名归一化只测了 `data.cache`（未测 vae/stats/frame_root）、
   `--fill-ref-cache` 分支的 `skipped_short` 未测、sampling 不一致的拒绝未单测、
   plain-dict→OmegaConf 包装被 mock 掉。属补强项，可并到第 1 条一起改。

## 处置

- 修复提交 `85952a6` 即为当前 HEAD，工作区无未提交源码改动；E1 训练启动于 `63f946f`，评测修复只动 tools，
  **不影响正在跑的训练梯度**。
- P3 不修改已验收提交，登记到 `TODO.md` 的里程碑1条目，等 E1 评测前统一清理（若涉及 FVD 公共 CLI，清理后
  需重跑 `tests/test_eval_fvd.py` 与 `--self-check`，并确认共享真值缓存未受影响的再算一次 floor）。

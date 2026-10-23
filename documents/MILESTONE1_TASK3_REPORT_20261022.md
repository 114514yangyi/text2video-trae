# Task3 实现报告（2026-10-22）

## 范围与接口

仅修改 `t2v/pipelines/milestone.py`、`tools/eval_milestone.py`、`tests/test_milestone_eval.py` 和所需的 `tools/eval_fvd.py` 窄修复；未启动训练、下载权重或改冻结缓存。

- `MilestonePipeline.from_checkpoint(path, device='cuda', use_ema=False)`：检查 `t2v/milestone1` 版本，完整严格载入 model / condition_encoder / text_encoder 的 raw 或 EMA；使用副本禁用初始化权重导入，不丢失原始初始化记录。
- `MilestonePipeline.from_modules(config, modules, device)`：复用训练中的三组模块，只重建本地 tokenizer；不重建模型、不读取 checkpoint；每次以当前文本权重编码空提示。默认 `training.sample_decode_device=cpu`，采样仍在训练设备上。
- `pipeline.generate(sample, decode_frames=24)`：输入 dataset 样本；返回 `latents`（完整 T）、`raw_latents`、`frames`（frame0 加前 23 个解码帧）和 `metadata`。`decode_frames=None` 才解码完整 T+1 帧。视频随机种子由基础 seed 与 video_id 的 SHA256 派生，与处理顺序无关。文本 CFG 两支保留参考帧。模块原有 train/eval 状态会恢复。
- `render_training_samples(config, modules, step, out_dir, device)`：固定 test 前 sample_count 个有效 id，写 `out_dir/samples/step_XXXXXXX/<video_id>/` 下全长 latent、24 PNG、对比 sheet/GIF/MP4 与 metadata。回调结束释放 VAE。
- `save_sample(pipeline,sample,output,decode_frames=24,visuals=False)`：只写全新单视频目录，防止旧高帧号混入；缺帧、短生成和非有限值直接报错。

## 命令

准备共享完整有效清单、真值特征和本底报告：

```sh
OMP_NUM_THREADS=4 /home/wxy/anaconda3/envs/text2video/bin/python tools/eval_milestone.py --config configs/milestone1/base.yaml --prepare --out reports/eval/m1_first24_prepare_20261022 --gpu cuda:2
```

小规模链路验证（必须明确 pilot 标签，不能覆盖完整清单）：

```sh
OMP_NUM_THREADS=4 /home/wxy/anaconda3/envs/text2video/bin/python tools/eval_milestone.py --config runs/m1_pipeline_validation_20261022/config.yaml --checkpoint runs/m1_pipeline_validation_20261022/final.pt --pilot pipeline-validation --max-videos 4 --visuals --out reports/eval/m1_pipeline_validation_20261022 --gpu cuda:3
```

完整评测去掉 `--pilot`、`--max-videos` 即可；`--use-ema` 为显式三模块 EMA 选择。所有 `--out` 都必须是新目录。resolved config 和普通带 base_config 的 arm config 均可读取。生成配置必须与 checkpoint 中的采样配置相同。

## 固定评测协议

清单按 `num_frames>=24`（而非 latent T>=24）筛选全量 test；实际 640/642，由 root 只读盘点确认。共享清单以独占创建写入，已有清单必须逐字一致。subset 清单仅落新 output。输出目录既是单 writer 锁，也是不可覆盖的协议记录。

FVD 新增 `--frame-root`、`--cache-dir`；源路径按 split/video basename 重定位，缓存不变。包含 frame0 时保留完整 T+1，不再截掉最后一帧；exact24 视频不再被错误跳过。生成前24不足会抛错，不会通过截短缩小 N。真值特征缓存额外验证 I3D 权重 SHA256；`--fill-ref-cache --out` 现在写完整 protocol、ids、N、cache 路径与 real_split_half_fvd。保持官方224预处理，不传旧272resize。

## RED / GREEN

1. 新增5个行为测试首先全部失败：管线、CLI、FVD路径/配对接口尚不存在。
2. 初版实现后 `tests/test_milestone_eval.py tests/test_eval_fvd.py`：12 passed。
3. 再增加 CPU 解码/活模块状态恢复和完整 raw/EMA/初始化记录测试：2 failed，分别暴露缺少 CPU decoder 默认和初始化元数据被修改。
4. 修复后同范围：14 passed（最终复跑 3.38s）。已有 scipy sqrtm(disp=False) 弃用警告4条，无失败。
5. `git diff --check` 通过；`tools/eval_milestone.py --help` 通过。
6. 全项目 `OMP_NUM_THREADS=4 /home/wxy/anaconda3/envs/text2video/bin/python tests/run_all.py`：exit 0，最终输出 `all tests passed`；完整日志 `/tmp/m1-task3-full-tests.log`。包括真实本地文本编码器相关测试，已有 transformers checkpoint 类型提示、lm_head 非 encoder 权重提示和 pytest rewrite 警告，无失败。
7. 提交：`131390b`，仅上述4个文件。

GPU自检、640条真值本底与实际200步checkpoint链路验证由 root 执行；此报告不把 CPU 单测当作真实 GPU 结果。

## 评审修复（2026-10-22，两个 P2）

评审来源：`MILESTONE1_TASK3_REVIEW_20261022.md`。检查代码后确认两项均可复现。

1. 入口先加载 checkpoint 管线并验证评测协议、采样设置与源数据/解码契约；契约包含 `data.cache`、`data.stats`、`data.frame_root` 及完整 `vae` 配置，其中路径经项目根解析再 `resolve()`。指向同一位置的路径别名可以通过，不要求 out_dir 等无关设置一致。不兼容时，在创建 shared manifest、output 和加载 dataset 前报错。输出 protocol 记录规范化契约。
2. FVD 先检查真实片段是否达到 min_frames，再校验生成长度；真实23/生成23会按规则排除，真实24/生成23依然抛错。prepare 和 scoring 报告均增加 `skipped_short` 的 id 与真实帧数。

RED：新增六个数据/VAE路径不兼容的入口测试、一个路径别名/无关配置测试和一个真实 CLI 短片回归，共 8 failed / 7 passed。短片回归实际执行 main 的样本选择和报告输出，仅替换外部 I3D 特征读取，随后删除一个有效视频的生成末帧，验证 strict shortage 仍有效。

GREEN：`OMP_NUM_THREADS=4 /home/wxy/anaconda3/envs/text2video/bin/python -m pytest tests/test_milestone_eval.py tests/test_eval_fvd.py -q` → 22 passed（3.69s）；仅已有 scipy 弃用警告4条。三个修改文件的 `git diff --check` 通过。全仓库 diff 检查会报告非本任务的 `configs/frozen/vae_h02.yaml:18` 现存空白，此文件未修改。

全项目 `OMP_NUM_THREADS=4 /home/wxy/anaconda3/envs/text2video/bin/python -m pytest tests -q` → **91 passed, 6 warnings in 10.87s**，exit 0，日志 `/tmp/m1-task3-review-full.log`。未重复 GPU 本底、采样或训练。

# 本机 conda 环境与训练准备检查（2026-10-22）

用户授权：检查项目、配置 conda 环境、熟悉训练所需条件。本轮未启动训练或全量指标评测，未改 t2v/tools/configs/tests、冻结契约、数据或模型权重。

## 已配置

- conda 环境：`text2video`，路径 `/home/wxy/anaconda3/envs/text2video`；Python 3.12.15。
- PyTorch 2.6.0+cu124 / torchvision 0.21.0+cu124；其余训练与 VAE 依赖按 `migration/requirements.txt` 安装，另装 pytest 9.1.1。
- 完整已安装包版本见 `ENVIRONMENT_PACKAGES_20261022.txt`；`python -m pip check`：No broken requirements found。
- 本机 NVIDIA 驱动 535.54.03；4×RTX 3090 24GiB。沙箱内看不到 NVIDIA 设备，沙箱外实测四卡 CUDA/bf16 矩阵计算通过，未修改驱动。

## 项目和输入

实际工作区 HEAD 为 2319729（detached HEAD），含完整旧 DiT 实现。知识库的 PixArt milestone1 规格描述另一分支，其训练入口尚未实现，不能与迁移 DiT 混为一谈。本轮保留现有工作树改动，不切分支、不提交。

当前可用训练入口为 `migration/train.sh` → `migration/train.py` → `migration.runtime.install()` → `t2v.train`。runtime 在内存中重定位原机器的帧路径及 VAE 路径，不修改冻结缓存。历史直接入口和配置中有旧绝对路径。

- 训练缓存 7096 条，max_num_frames=256 过滤后 7021；eval 519 条、过滤后 518；test 642 条、过滤后仍 642。
- 三套缓存对应的本地帧目录全部存在。真实样本 latent 为 [T,1024]，参考特征 [128,512]。
- mT5 模型与 tokenizer 本地齐全；冻结 VAE 源码及 psnr-best.pt 在 vendor/TiTok_base/main；官方 I3D 在 models/i3d/i3d_torchscript.pt。
- 949332 个迁移文件逐一 SHA256 校验，累计 60.16GiB，165秒；仅 `.git/index` 与打包时不同（Git 本地索引），其余全部匹配。

## 本机新验证

1. 离线训练入口 --help 成功，未进入训练。
2. GPU0–3 的 bf16 矩阵计算全部通过。
3. 三套真实缓存读取、归一化和迁移帧路径检查通过。
4. 本地 mT5 + 真实训练样本 + DiT-S 的 GPU bf16 扩散前向通过，输出 [1,52,1024]，loss 有限。未反向传播或更新权重。
5. 冻结 VAE 的配置/权重 SHA256 校验、完整加载（missing_keys=[]）通过；真实缓存解码单帧 [1,3,256,256] 且数值有限。
6. `PYTHONPATH=$PWD HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 python -m pytest tests -q --disable-warnings`：128 passed, 9 skipped, 6 warnings，8.40秒。9项跳过对应未打包的 Wan 模型权重；日志 `/tmp/text2video-pytest-20261022.log`。
7. `python tools/eval_fvd.py --self-check --gpu cpu`：133.908176，ok=true；I3D-K400 400维 logits，16帧、224²、float64 Fréchet，仅常量输入自检，不是生成质量评测。SciPy 对常量输入的奇异协方差给出警告，自检返回成功。

mT5 使用现有 T5EncoderModel 适配器，会提示 mt5/t5 类型及多余 lm_head 权重；编码前向已实测通过。此轮不改变现有适配器行为。

## 启动训练（仅命令，尚未执行）

在有 GPU 设备访问权限的普通终端运行：

```bash
source /home/wxy/anaconda3/etc/profile.d/conda.sh
conda activate text2video
cd /home/wxy/text2video
bash migration/train.sh --gpu cuda:0 --out-dir runs/migration_baseline_20261022 --set conditioning.text.trainable=false conditioning.text.learning_rate=0.0
```

配置：DiT-S 12层/384宽/6头，x0、cosine、logit-normal；文本+首帧参考+长度条件；mT5 冻结（文本学习率0），VAE 冻结；30k步、batch16、lr1e-4、bf16、EMA0.9999。每1000步覆盖 last.pt 并采样2条固定eval样例；完成保存 final.pt。可视化和当前生成流程使用真实长度，本迁移包未提供可直接部署的训练好长度预测器。eval/mse 默认开启，仅作诊断，不应据此筛选；正式方法对比按项目固定 test 协议执行。

恢复同一运行：重复以上命令追加 `--resume`，读取该 out-dir/last.pt。已有输出目录需先核对，避免复用覆盖。脚本关闭 wandb 并强制离线读取 Hugging Face 模型。

本机安装后磁盘约余56GiB；本轮未测完整 batch16 反向传播的峰值显存、优化器保存恢复、周期完整视频可视化及训练吞吐，不能把本次前向检查当成完整训练验收。GPU训练、调整batch或里程碑实现仍需要对应授权。

# SO-101 与 π0.5 社区资源核查

核查日期：2026-11-07。核查对象为仓库公开源码和作者发布的指南，采用静态阅读核对入口、数据映射与硬件配置。

## ljt228/pi05-so101-finetune

[仓库](https://github.com/ljt228/pi05-so101-finetune) 提供改版 OpenPI，存在可核对的 SO-101 数据变换与 π0.5 双专家 LoRA 训练配置。作者报告使用单张 NVIDIA L20，并公开关联数据集、15k 步检查点及训练日志；README 提供实机演示，git tree 中存在 `assets/demo_pi05_so101.mp4`，页面未给出标准化成功率统计。演示素材与作者报告用于说明项目实践，效果判断应结合下载资源和复现实测。

[实际训练配置](https://github.com/ljt228/pi05-so101-finetune/blob/main/src/openpi/training/config.py) 注册 `pi05_so101_lora_finetune`，开启 `pi05=True`、`gemma_2b_lora` 和 `gemma_300m_lora`，从 `pi05_base` 加载权重。SO-101 数据变换默认使用绝对动作，映射实际为 `observation.images.front`、`observation.images.side`、`observation.state` 和 `action`，与 README 概述的 global/wrist 相机名称不同。[输出变换](https://github.com/ljt228/pi05-so101-finetune/blob/main/src/openpi/policies/so101_policy.py) 保留动作前 6 维。

源码核对后的入口形式如下，实际数据集和资源路径需另行配置：

```bash
uv run scripts/compute_norm_stats.py pi05_so101_lora_finetune
uv run scripts/train.py pi05_so101_lora_finetune --exp-name so101_lora_v1
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_so101_lora_finetune \
  --policy.dir=/absolute/path/to/checkpoint
```

[统计脚本](https://github.com/ljt228/pi05-so101-finetune/blob/main/scripts/compute_norm_stats.py) 接收配置名称，README 写成数据集 ID；[服务脚本](https://github.com/ljt228/pi05-so101-finetune/blob/main/scripts/serve_policy.py) 使用 `policy` 联合类型和 `Checkpoint.config/dir`，README 的 `--config/--checkpoint` 形式与其字段不匹配。README 链接的 `scripts/train.sh` 返回 404。训练入口存在，使用时应以源码参数为准。

该方案采用 LeRobot v2.1 数据，结构接近本项目。其单卡 LoRA 路线和多卡完整微调不同，作者列举的 L20/A100/4090 不构成本机 8 GB 显存可运行的证据。

## Project-IRA/interactive-robotic-arm

[完整指南](https://github.com/Project-IRA/interactive-robotic-arm/blob/main/docs/SO101_PIPELINE.md) 涵盖双臂、双相机采集、数据合并、π0.5 完整微调及异步推理。指南指定较新的 Python 3.12 和 LeRobot 0.5.1，并要求 π0.5/异步部分从 LeRobot GitHub 安装；依赖版本与本项目采集环境不同，适合独立环境部署。

[训练脚本](https://github.com/Project-IRA/interactive-robotic-arm/blob/main/scripts/bash/Finetuning/finetune_pi05.sbatch) 确实包含 `--policy.type=pi05`、`lerobot/pi05_base`、float32、30k 步及 6 张 L40S；[FSDP 配置](https://github.com/Project-IRA/interactive-robotic-arm/blob/main/scripts/bash/Finetuning/fsdp_config.yaml) 使用 FULL_SHARD 和 6 个进程。这是完整微调方案的实测配置，不是所有 π0.5 微调方式的最低硬件要求。

[异步服务脚本](https://github.com/Project-IRA/interactive-robotic-arm/blob/main/scripts/bash/Inference/leRobotAsyncInferenceScriptServer.sh) 调用 `lerobot.async_inference.policy_server`。指南详细列出已发布模型、检查点选择和机器人客户端步骤。作者报告后期检查点出现明显过拟合，掷骰子任务表现不稳定；这些是具体任务与运行结果，使用其他任务需重新测试。

## 与本项目的衔接

本项目数据集 [info.json](../data/datasets/so101_task/meta/info.json) 记录 v2.1、50 条示范、16435 帧、30 Hz、6 维状态和动作，以及 front/handeye 双相机。动作顺序为 shoulder_pan、shoulder_lift、elbow_flex、wrist_flex、wrist_roll、gripper。采集使用归一化关节位置，训练和部署应保持相同数值定义；不能仅凭 6 维形状相同就替换动作单位。

采用 ljt228 路线时，先将实际 `side` 映射改为本项目 `handeye`，替换数据集 ID 和任务描述，计算新归一化统计量，并验证数据读取。采用 Project-IRA 路线时，先核对新版数据加载器对 v2.1 的支持及转换流程，再映射其 wrist_left/desk_view 命名。原始数据保留，转换结果写入新目录。

本机约 8 GB 显存适合继续 ACT 基准实验。上述社区配置对应更大显存单卡或多卡服务器，π0.5 的训练与推理硬件应按最终选定实现独立安排。首先采用现有抓方块数据验证机器人适配，再针对新环境、新物体和新任务测试泛化效果。

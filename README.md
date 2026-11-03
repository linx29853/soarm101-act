# SO-ARM101 ACT 复现

本项目为 SO-ARM101 机械臂提供 Action Chunking Transformer（ACT）的数据采集、训练、检查与实机部署流程。算法实现固定在 LeRobot `0.3.4` 的指定提交，本仓库负责版本锁定、SO-101 适配补丁、实验配置和可重复执行的命令入口。

ACT 论文：*Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*

论文地址：https://huggingface.co/papers/2304.13705

参考实现：https://github.com/tonyzhaozh/act

## 仓库结构

```text
.
├── configs/
│   └── act_so101.json          # ACT 超参数、数据路径和训练参数
├── data/
│   └── datasets/               # 本地 LeRobot 数据集，不提交到 Git
├── docs/
│   ├── reproduction.md         # 数据采集、训练与实机评估流程
│   └── results/                # 实验记录和结果表
├── outputs/                    # 日志、检查点和实机评估数据，不提交到 Git
├── patches/
│   └── 0001-so101-adapter.patch
├── scripts/
│   ├── bootstrap.sh            # 固定 LeRobot 版本并应用适配补丁
│   └── smoke_act.py            # 合成数据上的 ACT 前向、反向和推理测试
├── src/
│   ├── lerobot/                # 本地上游源码，忽略提交
│   └── soarm101_act/           # 配置校验和命令构造
└── tests/                      # 配置与命令单元测试
```

## 固定版本

| 组件 | 版本或提交 |
|---|---|
| LeRobot | `0.3.4` |
| 上游提交 | `882c80d446a63a44868c67ae535467af32ce0e80` |
| Python | 3.10 或更高 |
| 训练设备 | 支持 CUDA 的 NVIDIA GPU；CPU 可运行烟雾测试 |
| 数据集格式 | LeRobot `v2.1` |
| 机器人适配 | `so101_follower`、`so101_leader`，补丁同时包含双 SO-101 支持 |

`scripts/bootstrap.sh` 负责检出固定提交并应用 `patches/0001-so101-adapter.patch`。上游源码、数据集、检查点和课程资料均不进入 Git。

## 安装

建议在独立 Python 环境中执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
bash scripts/bootstrap.sh --install
```

仅准备源码、不安装依赖时执行：

```bash
bash scripts/bootstrap.sh
```

安装完成后检查环境：

```bash
python3 -m soarm101_act.cli check --config configs/act_so101.json
```

数据集尚不存在时，`check` 会给出版本、路径和 CUDA 参数的检查结果；完成数据采集后，该命令还会检查 `meta/info.json`。

## ACT 配置

`configs/act_so101.json` 使用论文中的核心参数：

| 参数 | 值 | 作用 |
|---|---:|---|
| `chunk_size` | 100 | 每次预测 100 个动作 |
| `n_action_steps` | 1 | 时间集成模式下每步重新推理 |
| `temporal_ensemble_coeff` | 0.01 | 时间集成指数权重 |
| `dim_model` | 512 | Transformer 隐维 |
| `n_heads` | 8 | 多头注意力头数 |
| `dim_feedforward` | 3200 | 前馈层维度 |
| `n_encoder_layers` | 4 | Transformer 编码器层数 |
| `n_decoder_layers` | 1 | 与原始实现的有效层数一致 |
| `latent_dim` | 32 | VAE 隐变量维度 |
| `kl_weight` | 10.0 | KL 损失权重 |
| `optimizer_lr` | 1e-5 | 主学习率 |
| `optimizer_lr_backbone` | 1e-5 | ResNet 主干学习率 |
| `batch_size` | 8 | 训练批大小 |
| `steps` | 100000 | 训练更新步数 |

默认数据集位置为 `data/datasets/so101_task`，默认输出目录为 `outputs/train/act_so101`。不同任务应建立独立数据集目录，并在配置中同步修改 `dataset.repo_id` 和 `dataset.root`。

## 数据采集

串口和相机参数通过命令行或环境变量提供：

```bash
export SOARM_FOLLOWER_PORT=/dev/ttyACM0
export SOARM_LEADER_PORT=/dev/ttyACM1
export SOARM_CAMERAS_JSON='{"front":{"type":"opencv","index_or_path":"/dev/video0","width":640,"height":480,"fps":30}}'

python3 -m soarm101_act.cli record \
  --config configs/act_so101.json \
  --episodes 50
```

采集前需完成主从臂 ID 设置、从臂校准和相机检查。数据默认保存在本地，不自动上传 Hugging Face Hub。详细操作见 [docs/reproduction.md](docs/reproduction.md)。

## 训练

```bash
python3 -m soarm101_act.cli train \
  --config configs/act_so101.json \
  --device cuda \
  --batch-size 8
```

命令会生成实际使用的 LeRobot 训练配置：

```text
outputs/configs/act_so101_train.json
```

检查点保存在：

```text
outputs/train/act_so101/checkpoints/
```

恢复训练时使用 LeRobot 的原始配置入口：

```bash
lerobot-train \
  --config_path=outputs/train/act_so101/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

## 训练前验证

安装依赖后运行合成数据烟雾测试。该测试不读取数据集，覆盖 ACT 前向损失、反向梯度、动作维度与推理输出：

```bash
python3 -m soarm101_act.cli train-smoke --config configs/act_so101.json
```

不依赖 PyTorch 的配置和命令测试：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## 实机评估

在机械臂周围预留动作范围，首次运行使用较低的 `max_relative_target`。训练完成后执行：

```bash
export SOARM_FOLLOWER_PORT=/dev/ttyACM0
export SOARM_CAMERAS_JSON='{"front":{"type":"opencv","index_or_path":"/dev/video0","width":640,"height":480,"fps":30}}'

python3 -m soarm101_act.cli rollout \
  --config configs/act_so101.json \
  --policy-path outputs/train/act_so101/checkpoints/last/pretrained_model \
  --episodes 10
```

评估数据默认写入 `data/datasets/so101_eval`。成功率和视频文件记录到 `docs/results/`，并保留对应的有效配置和检查点提交号。

## 引用

```bibtex
@article{zhao2023learning,
  title={Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
  author={Zhao, Tony Z and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},
  journal={arXiv preprint arXiv:2304.13705},
  year={2023}
}
```

## 许可证

本项目以 Apache License 2.0 发布。LeRobot 与 ACT 参考实现分别遵循其上游许可证。

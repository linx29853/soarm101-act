# ACT 复现实验流程

## 主实验：200 条示范的训练与部署

主实验采用双相机、180 条训练示范、20 条验证示范、batch size 4 和 100k 训练步数。以下步骤在仓库根目录、安装完成后的 Python 环境中执行；本地数据集 `data/datasets/blue_cube_200` 需包含完整的 LeRobot v2.1 元数据、关节表与双相机视频。

### 写入训练划分

`configs/blue_cube_200_split.json` 保存训练和验证 episode 编号。项目训练封装生成的基础配置包含数据集路径，主实验需将 `train` 编号显式写入上游 LeRobot 的 `dataset.episodes` 字段。直接运行封装的 `train` 命令会使用整个数据集。

```bash
python3 - <<'PYCONFIG'
import json
from pathlib import Path
from soarm101_act.config import load_config, train_config_dict

config = train_config_dict(load_config("configs/act_blue_cube_180train.json"))
split = json.loads(Path("configs/blue_cube_200_split.json").read_text())
train, validation = split["train"], split["validation"]
assert len(train) == len(set(train)) == 180
assert len(validation) == len(set(validation)) == 20
assert set(train).isdisjoint(validation)
assert set(train) | set(validation) == set(range(200))
config["dataset"]["episodes"] = train
output = Path("outputs/configs/act_blue_cube_180train_train.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(config, indent=2) + "\n")
print(output)
PYCONFIG

lerobot-train --config_path=outputs/configs/act_blue_cube_180train_train.json
```

`dataset.episodes` 接口见固定版本的 [LeRobot DatasetConfig](https://github.com/huggingface/lerobot/blob/882c80d446a63a44868c67ae535467af32ce0e80/src/lerobot/configs/default.py)。训练产物写入 `outputs/train/act_blue_cube_180train_100k`。保留有效配置与检查点内的 `train_config.json`，核对训练编号与划分文件一致。

### 验证、检查点与部署

[验证记录](results/chart_data/validation.jsonl) 比较 20k、40k、60k、80k、100k 检查点，每条验证示范抽取 10 个观测，评估完整动作块并排除填充。60k 的前五关节 MAE 为 6.254，在五个检查点中最低。验证集的 20 条示范随后用于 9,447 帧离线回放，具体方法、历史评估入口与产物见 [诊断报告](results/200条数据模型离线诊断.md#复现与产物)。

原始检查点路径为 `outputs/train/act_blue_cube_180train_100k/checkpoints/060000/pretrained_model`。历史部署目录 `outputs/models/act_blue_cube_180train_best60k_gripper1614` 使用相同学习权重，六个夹爪归一化统计缓冲区按校准映射调整。部署时保留对应校准文件与统计，确认相机键为 `front`、`handeye`，分辨率为 640 × 480。README 给出显式选择原始 60k 检查点的部署命令；历史部署版本对应关系见诊断报告。

### 图表重绘

```bash
python scripts/plot_readme_results.py
```

该入口依赖 NumPy、Matplotlib，读取仓库中的 `results/chart_data` 归档数据并生成五组 PNG / SVG 图表。原始模型离线评估与归档数据重绘的入口分别列在 [诊断报告](results/200条数据模型离线诊断.md) 与 [图表说明](results/图表说明.md)。

## 通用基线

以下流程使用 `act_so101.json`、batch size 8 和单相机示例，数据目录为 `data/datasets/so101_task`；检查点使用该次训练的 `last`。

## 1. 目标与基线

复现任务使用单臂 SO-ARM101 和至少一个场景相机。ACT 将当前状态与多相机图像编码为 Transformer 输入，一次输出长度为 100 的动作序列；推理阶段使用系数为 `0.01` 的时间集成，将重叠动作块的预测按指数权重融合。

复现基线固定为：

- LeRobot 提交：`882c80d446a63a44868c67ae535467af32ce0e80`
- 数据集格式：LeRobot `v2.1`
- 配置：`configs/act_so101.json`
- 训练步数：100000
- 批大小：8
- 主学习率：`1e-5`
- 随机种子：1000

## 2. 硬件与软件前置条件

硬件：

- 一台 SO-ARM101 从臂，具备 6 个关节位置控制通道
- 一台 SO-ARM101 主臂，用于遥操作采集
- 至少一个固定视角相机
- 两个可被系统识别的串口设备

软件：

- Ubuntu 22.04 或等价 Linux 环境
- Python 3.10 或更高
- NVIDIA 驱动与可用的 CUDA 环境
- Feetech 驱动依赖
- 可容纳数据集和检查点的磁盘空间

执行安装与硬件检查：

```bash
source .venv/bin/activate
bash scripts/bootstrap.sh --install
python3 -m soarm101_act.cli check --config configs/act_so101.json
lerobot-find-port
lerobot-find-cameras opencv
```

## 3. 数据采集

固定相机位置、光照、物体初始区域和任务文本。每条 episode 包含完整初始状态、接近、抓取、搬运、放置和退出动作，动作完成后保持稳定姿态约 1 秒。

参数要求：

| 参数 | 推荐值 |
|---|---:|
| 控制频率 | 30 Hz |
| episode 时长 | 20 至 40 s |
| 重置时长 | 30 s |
| episode 数量 | 50 至 100 |
| 相机数量 | 1 至 3 |
| 分辨率 | 640 x 480 |

采集命令：

```bash
export SOARM_FOLLOWER_PORT=/dev/ttyACM0
export SOARM_LEADER_PORT=/dev/ttyACM1
export SOARM_CAMERAS_JSON='{"front":{"type":"opencv","index_or_path":"/dev/video0","width":640,"height":480,"fps":30}}'

python3 -m soarm101_act.cli record \
  --config configs/act_so101.json \
  --cameras "$SOARM_CAMERAS_JSON" \
  --episodes 50
```

数据写入 `data/datasets/so101_task`。采集后检查：

```bash
find data/datasets/so101_task -maxdepth 2 -type f | sort
python3 -m soarm101_act.cli check --config configs/act_so101.json --strict
```

`meta/info.json` 中的 `fps`、相机键、状态维度和动作维度应与配置及硬件一致。多相机训练时，各任务会话保持相同相机键和分辨率。

## 4. 训练

```bash
python3 -m soarm101_act.cli train \
  --config configs/act_so101.json \
  --device cuda
```

显存不足时先降低批大小：

```bash
python3 -m soarm101_act.cli train \
  --config configs/act_so101.json \
  --batch-size 4
```

每次运行保留以下文件：

- `outputs/configs/act_so101_train.json`
- `outputs/train/act_so101/checkpoints/last/pretrained_model/config.json`
- `outputs/train/act_so101/checkpoints/last/pretrained_model/train_config.json`
- 训练日志和最终检查点

有效配置、代码提交、数据集名称、检查点步数和训练日志共同构成复现记录。

## 5. 离线验证

离线验证按独立验证 episode 比较以下指标：

- 预测动作与专家动作的 L1 误差
- 每个关节的动作平滑度
- 夹爪开合时序
- 时间集成开启与关闭时的动作偏差

真机任务成功率由下一节的逐次实机试验记录汇总。主实验按验证集前五关节 MAE 选择 60k 检查点，L1 损失用于排查训练收敛和归一化问题。

## 6. 实机评估

首次评估设置较低的 `max_relative_target`，确认方向、关节限位和夹爪开合范围。评估命令：

```bash
export SOARM_FOLLOWER_PORT=/dev/ttyACM0
export SOARM_CAMERAS_JSON='{"front":{"type":"opencv","index_or_path":"/dev/video0","width":640,"height":480,"fps":30}}'

python3 -m soarm101_act.cli rollout \
  --config configs/act_so101.json \
  --cameras "$SOARM_CAMERAS_JSON" \
  --policy-path outputs/train/act_so101/checkpoints/last/pretrained_model \
  --episodes 10
```

记录每次实验：

| 字段 | 内容 |
|---|---|
| 日期 | 实验执行日期 |
| 代码提交 | 本仓库提交号与 LeRobot 提交号 |
| 数据集 | `repo_id`、episode 数量、帧数 |
| 模型 | 检查点步数、配置文件 |
| 硬件 | 相机数量、串口、机械臂编号 |
| 评估 | episode 数量、成功次数、成功率 |
| 产物 | 日志、视频、结果表路径 |

结果保存在 `docs/results/`。表内数据由实际评估填写，视频和检查点保存在被 Git 忽略的数据目录或外部存储。

## 7. 复现检查清单

- [ ] LeRobot 提交与 `scripts/bootstrap.sh` 一致
- [ ] SO-101 适配补丁无冲突
- [ ] `dataset.root/meta/info.json` 存在
- [ ] 数据集相机键与采集命令一致
- [ ] 有效训练配置已保存
- [ ] 检查点包含 `config.json`、`model.safetensors` 和 `train_config.json`
- [ ] 实机评估记录包含 episode 数、成功数和视频路径

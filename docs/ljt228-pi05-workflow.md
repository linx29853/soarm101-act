# ljt228 的 SO-101 π0.5 工作流拆解

核查日期：2026-11-07。仓库 HEAD：`2b7befa55afecb0ca81c34c2bfd7c172c0c716b2`。以下按公开源码组织实现步骤。

## 1. 项目由什么构成

[仓库](https://github.com/ljt228/pi05-so101-finetune) 在 OpenPI 上新增 SO-101 双相机与 6 维关节动作适配，使用官方 π0.5 预训练参数继续训练。核心文件如下：

| 文件 | 功能 |
|---|---|
| [training/config.py](https://github.com/ljt228/pi05-so101-finetune/blob/main/src/openpi/training/config.py) | 注册数据映射、任务文字、预训练权重与训练参数 |
| [so101_policy.py](https://github.com/ljt228/pi05-so101-finetune/blob/main/src/openpi/policies/so101_policy.py) | 将双相机与关节状态变为模型输入，并截取输出前 6 维 |
| [compute_norm_stats.py](https://github.com/ljt228/pi05-so101-finetune/blob/main/scripts/compute_norm_stats.py) | 计算新数据集的状态和动作统计量 |
| [train.py](https://github.com/ljt228/pi05-so101-finetune/blob/main/scripts/train.py) | JAX/Flax 训练与检查点保存 |
| [serve_policy.py](https://github.com/ljt228/pi05-so101-finetune/blob/main/scripts/serve_policy.py) | 加载检查点并提供 WebSocket 推理服务 |
| [so101_breakfast.py](https://github.com/ljt228/pi05-so101-finetune/blob/main/scripts/so101_breakfast.py) | 读取真实相机、连接从臂、执行模型动作 |

仓库包含真实实机客户端，路径为 `scripts/so101_breakfast.py`；`examples/so101` 并非该版本的目录。`config/pi05_so101_lora_finetune.py` 是参考代码片段，真正被训练程序查找的是 `src/openpi/training/config.py` 中已注册的配置。

## 2. 数据怎样进入模型

`LeRobotSO101DataConfig` 把 `observation.images.front` 映射到 `observation/image`，把 `observation.images.side` 映射到 `observation/wrist_image`，并读取 `observation.state` 和 `action`。本项目第二相机名为 `handeye`，应在映射中替换 `side`，原始数据无需改名。

`SO101Inputs` 将两图放入 `base_0_rgb`、`left_wrist_0_rgb`；右腕相机用零图补位并对 π0.5 关闭其有效掩码。浮点图像转 uint8，CHW 图像转 HWC。随后执行归一化、224×224 图像缩放、提示词分词，以及把 6 维状态/动作补到模型要求的 32 维。

π0.5 默认将状态编码到离散文字 token 中；其状态仍来自机械臂关节，不是仅凭语言控制。任务文字由当前配置的 `default_prompt` 提供；该数据配置未开启 `prompt_from_task`，扩展多任务时应明确从每条示范任务加载文字，避免全部套用同一早餐指令。[模型配置](https://github.com/ljt228/pi05-so101-finetune/blob/main/src/openpi/models/pi0_config.py)、[数据加载器](https://github.com/ljt228/pi05-so101-finetune/blob/main/src/openpi/training/data_loader.py)。

## 3. 动作、归一化与时间

默认 `use_delta_joint_actions=False`，学习的是数据里的绝对关节目标。若开启相对动作，掩码为前 5 个关节相对化、夹爪保留绝对值，输出端再还原。当前配置应按默认绝对动作理解。

π0.5 分支使用分位数归一化，统计量由新数据计算，并保存在检查点 assets 中。部署加载同一统计量，反归一化后截取前 6 维。关节顺序、归一化位置单位、夹爪范围必须与采集一致。[归一化与反变换](https://github.com/ljt228/pi05-so101-finetune/blob/main/src/openpi/policies/policy_config.py)。

模型默认一次预测 50 个时间步、32 维动作；30 Hz 数据中覆盖约 1.67 秒。实机客户端默认每次执行 25 步后重新获取观测、请求推理，执行覆盖约 0.83 秒。模型预测长度和客户端实际执行长度是两个参数。

## 4. 实际训练超参数

SO-101 配置只覆盖部分参数，其余继承 `TrainConfig`、`Pi0Config` 与优化器默认值：

| 参数 | 源码值 |
|---|---|
| 初始权重 | `gs://openpi-assets/checkpoints/pi05_base/params` |
| 视觉语言专家 | `gemma_2b_lora`，注意力/FFN LoRA rank 16、alpha 16 |
| 动作专家 | `gemma_300m_lora`，注意力/FFN LoRA rank 32、alpha 32 |
| 模型 dtype | bfloat16；训练脚本另外把冻结参数转为 bfloat16 |
| 动作维度／预测长度 | 32／50 |
| token 上限 | 200 |
| batch size／数据进程 | 32／2 |
| 总步数／随机种子 | 30000／42 |
| 优化器 | AdamW，β1=0.9、β2=0.95、eps=1e-8 |
| 权重衰减／梯度裁剪 | 1e-10／全局范数 1.0 |
| 学习率 | warmup 1000 步，峰值 2.5e-5，30000 步余弦衰减到 2.5e-6 |
| EMA | 关闭 |
| 日志／保存／长期保留 | 100 步／1000 步／5000 步 |
| FSDP devices | 1 |

来源：[模型与冻结过滤器](https://github.com/ljt228/pi05-so101-finetune/blob/main/src/openpi/models/pi0_config.py)、[LoRA rank](https://github.com/ljt228/pi05-so101-finetune/blob/main/src/openpi/models/gemma.py)、[优化器](https://github.com/ljt228/pi05-so101-finetune/blob/main/src/openpi/training/optimizer.py)、[训练配置](https://github.com/ljt228/pi05-so101-finetune/blob/main/src/openpi/training/config.py)。README 提到作者 L20 运行时批大小约 16、每 5k 保存；表格是源码默认，实际运行可由启动参数覆盖。

冻结过滤器匹配 `.*llm.*` 中非 LoRA 的参数，训练集合取其补集。这个写法冻结语言/动作 Transformer 的基础权重，但不等价于全模型仅训练 LoRA：图像编码器及动作投影等位于该匹配范围外的可训练参数仍进入更新集合。显存估算应以实际可训练参数为准。

## 5. 从零执行的依赖顺序

1. 在独立 Python 环境配置该仓库依赖和 GPU，保留原 ACT 环境。
2. 新建抓方块训练配置，替换数据集 ID、asset ID、任务文字和 handeye 相机映射。
3. 用一个样本验证图像、状态顺序和动作单位，再计算统计量。
4. 加载官方 π0.5 权重，少量步数验证训练，再进行完整微调。
5. 加载多个检查点做实机测试，以成功率选择模型。

入口形式如下，其中配置需先按本项目数据修改：

```bash
uv run scripts/compute_norm_stats.py pi05_so101_lora_finetune
uv run scripts/train.py pi05_so101_lora_finetune --exp-name so101_cube_v1
uv run scripts/serve_policy.py --port=10093 policy:checkpoint \
  --policy.config=pi05_so101_lora_finetune \
  --policy.dir=/absolute/path/checkpoints/pi05_so101_lora_finetune/so101_cube_v1/15000
```

统计脚本接收配置名称，不是 repo ID。服务脚本使用 `policy:checkpoint` 子类型和 `--policy.config/--policy.dir`，与 README 展示的简写不同。README 链接的 `scripts/train.sh` 在核查版本返回 404。

## 6. 保存的权重是什么

[CheckpointWeightLoader](https://github.com/ljt228/pi05-so101-finetune/blob/main/src/openpi/training/weight_loaders.py) 先加载官方基础参数，再补入初始化的 LoRA 参数。这里的 merge 表示合并参数字典，不是把低秩矩阵乘积折叠进原权重。

[检查点实现](https://github.com/ljt228/pi05-so101-finetune/blob/main/src/openpi/training/checkpoints.py) 保存 `params`、`train_state` 和 `assets`。关闭 EMA 时，推理参数来自完整 `state.params`，包含基础参数与 LoRA 参数，并非只导出小型 adapter。加载推理时使用同样带 LoRA 结构的模型配置。源码保存链路没有 LoRA 矩阵折叠步骤，因此不能将其解释为已合并的普通 π0.5 权重。

检查点目录为 `checkpoints/<配置名>/<实验名>/<步数>/`，服务端接收步数目录，再读取其下 params 和 assets。保留策略为最新检查点加 5000 步倍数节点。

## 7. 实机客户端怎样工作

`so101_breakfast.py` 的流程是：连接真实从臂并读取校准 → 打开两个相机 → 获取 6 维关节状态和两张 RGB 图 → 发送文字与观测 → 接收动作块 → 逐步写入 6 个关节目标 → 下一次推理。

客户端硬编码了作者的串口、校准 ID、相机路径、早餐任务，以及 `127.0.0.1:10093`。服务脚本默认端口是 8000，启动时需显式改为 10093 或同步修改客户端。启动入口是 `main(Args())`，没有调用命令行参数解析器，因此其 dataclass 字段不能直接假定支持 `--host` 等参数。采用远程服务器时应修改配置或补参数解析，并对齐服务端端口。

客户端依赖 `lerobot.robots.so_follower` 的 SOFollower，而本项目采集使用旧版 `so101_follower`；需要在独立客户端环境验证 API，或将硬件层换成本项目已有接口。`ActionChunkBroker` 按块等待推理，属于同步分块执行，不能当成后台异步推理。每个相机一次观测连续读取三帧，以及串口前后状态打印，也会影响实际执行速率；30 Hz 是目标上限，不是已测闭环频率。

客户端没有自动成功判定，`is_episode_complete()` 始终为 False；默认最多 100000 个执行步。投入本项目时宜补明确的单次运行时限、停止入口、动作范围和单步变化限制，先验证预测值，再运行完整抓放。

## 8. 可以直接借鉴的部分

本项目已有 50 条 v2.1、6 维动作、front/handeye 双相机示范，适合复用该仓库的数据变换设计和配置组织方式。优先借鉴双相机映射、6→32 维补齐与 32→6 输出、统计量随检查点保存，以及服务端和硬件客户端分离。

现有硬件接口、校准、相机传输设置继续沿用；作者的串口、相机 ID、早餐指令、默认 batch size 和执行时长应替换为本项目参数。训练使用更大显存服务器；本机约 8 GB 显存保留 ACT 训练和机器人采集职责。

仓库 git tree 中存在 [demo_pi05_so101.mp4](https://github.com/ljt228/pi05-so101-finetune/blob/main/assets/demo_pi05_so101.mp4)，README 也提供实机演示链接；核查时网页抓取曾显示上传占位文字，查看演示应结合 README 实际附件链接及资产文件。实机演示和作者训练结果表明该路线具有可参考实现；本项目效果以自身固定测试位置、成功率和运行延迟评估。

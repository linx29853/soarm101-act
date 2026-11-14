# ACT 开源数据选择与训练资料调研

调研日期：2026-11-09。对象为原始 ACT/ALOHA 论文与作者代码、Hugging Face LeRobot 官方文档与源码。在线 `main` 按调研日期核对；本项目训练环境使用提交 `882c80d446a63a44868c67ae535467af32ce0e80`，运行参数应以该提交为准。

## 数据如何选

LeRobot 官方实机教程给出的起点非常具体：至少采集 50 条示范，每个物体位置约 10 条；相机固定、抓取方式一致、目标在相机内清楚可见。采集人员只看相机画面也应能完成任务。先建立可靠抓取，再逐步增加位置、抓法或视角变化。[官方实机教程：Tips for gathering data](https://huggingface.co/docs/lerobot/en/il_robots#tips-for-gathering-data)

原始论文中，6 个实机任务通常各用 50 条示范，Thread Velcro 使用 100 条；示范控制频率 50 Hz，单条约 8–14 秒。论文对两个仿真任务的脚本示范和人工示范明确写明各记录 50 条成功示范。该段的“成功示范”描述直接对应仿真设置，实机数据段没有逐条成功标签或失败剔除清单。[原始论文，第 V-B 节](https://arxiv.org/html/2304.13705v1#S5.SS2)

作者项目页介绍实机展示任务时，说明训练与测试都沿 15 cm 参考线随机化物体初始位置。数据价值来自目标范围内的覆盖与动作质量，而非仅累计示范条数。[ALOHA 项目页](https://tonyzhaozh.github.io/aloha/)

“论文使用成功示范”与“录制脚本自动过滤成功”需要分别判断。作者公开的 `record_sim_episodes.py` 会打印成功或失败、累计成功次数，但保存 HDF5 的路径没有 success 条件；直接执行生成命令得到的数据仍需检查。训练加载器同样没有根据成功标签筛选轨迹的逻辑。[仿真采集代码](https://github.com/tonyzhaozh/act/blob/main/record_sim_episodes.py)、[数据加载代码](https://github.com/tonyzhaozh/act/blob/main/utils.py)

ACT 的标准目标是模仿示范动作；失败轨迹若直接放入数据，其动作也会成为监督目标。保留可恢复的偏差应以“出现偏差后成功纠正并完成任务”为采集目标；抓空后结束、掉落后结束等轨迹适合单独保存为失败分析与评估素材。这是依据损失机制提出的项目数据规则。[原始 ACT 损失实现](https://github.com/tonyzhaozh/act/blob/main/policy.py)

LeRobot 官方数据文章还强调图像质量、完整 episode、状态与动作维度、特征名称和任务标注的一致性。筛选宜同时记录成功结果、相机可见性、动作完整性与异常原因。[LeRobot 官方数据质量文章](https://huggingface.co/blog/lerobot-datasets)

## 训练如何组织

### 原始 ACT

输入为当前多视角 RGB、关节状态；监督为未来一段动作。CVAE 在训练时编码动作序列，推理时 latent 设为零；损失为动作 L1 加 KL 正则，标准训练不需要 reward。[原始 ACT 策略源码](https://github.com/tonyzhaozh/act/blob/main/policy.py)

作者加载器按整条 episode 随机分为 80% 训练、20% 验证；每次访问一条 episode 随机抽时间起点，因此一次 epoch 并非遍历全部帧。RGB 转为 0–1，状态与动作使用 mean/std，后续图像采用 ImageNet 归一化；末尾动作补齐并提供 padding mask。原实现统计使用全部 episode，项目独立留出验证应改用训练子集统计。[作者数据加载器](https://github.com/tonyzhaozh/act/blob/main/utils.py)、[图像与损失预处理](https://github.com/tonyzhaozh/act/blob/main/policy.py)

论文表 III 给出学习率 `1e-5`、batch `8`、隐藏维度 `512`、8 个头、4 层编码器、7 层解码器、FFN `3200`、chunk `100`、KL 权重 `10`。项目页的实机演示则注明 chunk `90`、50 Hz，时间跨度约 1.8 秒。参数选择应按任务和控制频率解释。[论文附录 D](https://arxiv.org/html/2304.13705v1#A4)、[实机展示配置](https://tonyzhaozh.github.io/aloha/)

作者 README 的仿真示例使用 `num_epochs=2000`。训练循环每个 epoch 计算验证 loss，并保存最低验证 loss 的 `policy_best.ckpt`，另有周期与最后检查点。epoch 的含义取决于该加载器，不能与 LeRobot 的 optimizer steps 等同。[作者训练示例](https://github.com/tonyzhaozh/act#simulated-experiments)、[训练与选检查点代码](https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py)

作者还明确指出：loss 平台期之后，继续训练仍可能提升成功率与平滑度。验证 loss 可用于候选筛选，固定任务条件下的闭环成功率用于最终比较。[ACT 官方调参提示入口](https://github.com/tonyzhaozh/act)

### LeRobot ACT

LeRobot 官方推荐先用默认参数，batch 从 8 起，示例训练规模为 100k steps；训练时间依 GPU、batch 和数据读取而变化。[LeRobot ACT 文档](https://huggingface.co/docs/lerobot/en/act)

| 项目 | 调研时 LeRobot main 配置 |
|---|---|
| 预测与执行 | `chunk_size=100`、`n_action_steps=100` |
| 时间融合 | 默认 `temporal_ensemble_coeff=None` |
| 优化 | ACT 学习率 `1e-5`、weight decay `1e-4` |
| CVAE | `use_vae=True`、KL 权重 `10` |
| 模型结构 | 4 层编码器、1 层解码器，隐藏维度 512 |
| 输入和动作 | mean/std 归一化 |

来源：[LeRobot ACT 配置源码](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/configuration_act.py)。原作者配置声明 7 层解码器，但 Transformer 返回各层中间结果，`DETRVAE.forward` 取 `[0]`，动作头实际使用第一层结果；LeRobot 配置为 1 层与该有效路径对应。复现实验应同时核对声明配置与实际前向路径。[原作者 Transformer](https://github.com/tonyzhaozh/act/blob/main/detr/models/transformer.py)、[动作头取值](https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py)

在线 main 已提供 `dataset.episodes`、`exclude_episodes` 与 `dataset.eval_split`；`eval_split` 默认 0。训练配置的 `eval_steps` 默认 0，配合正的留出比例才启用离线验证；`env_eval_freq` 则控制仿真环境闭环评估。周期保存检查点和自动按成功率选最优模型也是不同操作。[数据配置](https://github.com/huggingface/lerobot/blob/main/src/lerobot/configs/default.py)、[训练配置](https://github.com/huggingface/lerobot/blob/main/src/lerobot/configs/train.py)

项目固定提交的接口与在线 main 有差异。本项目已经通过 `scripts/train_with_validation.py` 实现训练子集与验证，继续实验可沿用该版本，避免直接复制新版验证参数。[项目已有 ACT 流程核对](ACT官方流程对照.md)

动作块的实际时长为 `chunk_size / fps`；100 步在 30 Hz 下约 3.33 秒。`n_action_steps` 决定重看画面前连续执行多久，时间融合则提供另一种逐帧更新方式。训练 chunk、执行长度和采集频率应一起记录，并按固定条件分别比较。

## 本项目的数据与实验建议

项目已有 200 条数据的 180/20 留出实验，训练至 100k steps，部署候选为 60k；另有从 200 条筛到 83 条、27,208 帧、`front+handeye` 的子集。筛选报告提供保留与剔除索引及完整性校验，83 条的任务成功率和具体剔除原因需要通过逐条标注形成记录。[已有训练与部署核对](ACT官方流程对照.md)

83 条数据已有前 50 条训练、后 33 条留出的回放实验；编号 75 之后场景与腕部起始姿态发生变化，例如编号 76 腕部起始值约 14.9，而训练起始范围约 34.7～66.7。该顺序切分同时包含条件外推，应将同条件验证与新条件测试分别统计。[新模型留出集离线验证](新模型留出集离线验证.md)

建议先完成以下对照：

1. 固定任务成功标准，逐条标注 200 条轨迹的成功/失败、初始位置、遮挡、停顿、掉落及剔除原因；保留成功纠偏示范，归档终止失败轨迹。
2. 检查 83 条子集是否保留各个初始位置的覆盖。按 5 个位置、每位置约 10 条成功示范建立第一轮平衡基线，再按失败集中位置补数据。
3. 按物体位置与机器人起始姿态分层，将完整 episode 分为训练、验证和实机测试条件；同一条轨迹的相邻帧放在同一分组，归一化只由训练子集计算。
4. 比较全量集与筛选集时固定验证集、随机种子、相机、优化器和部署策略，并同时记录更新步数与样本曝光量。固定 steps 会让小数据集被重复读取更多次。
5. 保存 20k、40k、60k、80k、100k 候选，结合验证动作误差选取实机候选；使用相同初始位置与任务成功标准统计闭环成功次数。
6. 数据、训练参数与部署执行长度分别做对照；同时改变多项会降低结果归因的清晰度。

上述建议属于依据官方机制与本项目记录形成的实验方案，具体成功率由新一轮固定条件实测记录。

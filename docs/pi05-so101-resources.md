# SO-101 与 π0.5 开源资源

检索日期：2026-11-07。

## 当前项目条件

本项目使用 LeRobot 0.3.4 固定源码、LeRobot v2.1 数据、六维关节状态与动作，以及 front、handeye 两个相机。现有数据包含 50 条示范。机械臂配置使用 use_degrees=false。训练设备为约 8 GB 显存的 RTX 5070 Laptop。

## 单任务微调与部署

- [ljt228/pi05-so101-finetune](https://github.com/ljt228/pi05-so101-finetune)：作者介绍了 SO-101 双相机、v2.1 数据、OpenPI/JAX 和单卡 L20 LoRA 微调流程。具体训练与部署入口见[源码核查记录](pi05-so101-source-check.md)。
- [twu3202/soarm101-vla](https://github.com/twu3202/soarm101-vla)：包含 π0.5 LoRA 抓方块任务、双相机数据与远程部署代码。作者使用 RTX 6000 Ada 48 GB；README 报告离线误差与相机消融结果，闭环表现需结合实机记录评估。部署客户端读取角度值，接入本项目时需要对齐关节单位。来源：[README](https://github.com/twu3202/soarm101-vla)、[部署客户端](https://github.com/twu3202/soarm101-vla/blob/main/vla_sort/so101_vla_deploy.py)。

## SO-101 领域适配权重

[jinnymo/so101-pi05-base](https://github.com/jinnymo/so101-pi05-base) 提供 SO-101/SO-100 数据整合、全量微调、容器、权重和推理说明。作者报告基础适配训练使用 8 张 A100 80 GB，实机推理使用 RTX 3090 Ti 24 GB。该训练规模用于构建领域适配基础模型；任务级微调应参考独立 LoRA 配置。

[LoRA 指南](https://github.com/jinnymo/so101-pi05-base/blob/main/docs/07-lora-finetuning.md) 使用 v2.1 数据，重点说明相机槽位、动作坐标和归一化统计量。作者已报告的 LoRA 实验基于原始 pi05_base，基于新发布领域权重的 LoRA 实验列为后续验证任务。

## 完整流程与官方入口

- [Project-IRA 完整指南](https://github.com/Project-IRA/interactive-robotic-arm/blob/main/docs/SO101_PIPELINE.md)：组装、采集、数据合并、π0.5 微调和推理；适合逐阶段核对流程。
- [LeRobot 官方 π0.5 指南](https://huggingface.co/docs/lerobot/pi05)：提供 PyTorch 训练配置、权重加载、统计量与部署入口。当前页面默认量化分位数归一化需要 q01/q99，旧数据可重新计算统计量或明确采用对应的均值方差归一化方案。
- [Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi)：原始模型与微调实现。

## 接入顺序

1. 保留 ACT 环境与原始数据，在独立目录和环境固定所选 π0.5 仓库版本。
2. 优先核对单任务 LoRA 方案的数据加载和远程推理接口。
3. 使用数据副本校验相机顺序、六维关节顺序、单位及动作约定。
4. 在服务器完成数据读入和少量训练步测试，检查检查点。
5. 先读取预测动作并检查范围，再进行受控的实机测试。

当前 8 GB GPU 已完成 ACT 试训练。上述社区案例使用的 π0.5 训练 GPU 更大，服务器规格应按选定配置实测确定。

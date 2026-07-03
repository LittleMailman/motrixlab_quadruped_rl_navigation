# MotrixLab Quadruped RL Navigation

基于 MotrixLab 的四足机器人强化学习导航实践项目。

本仓库记录了从 ANYmal C 最小环境搭建、ANYmal C 寻点导航训练，到 VBot Section01 越障导航任务完成的完整实践过程。项目重点放在 MotrixLab 工程实现、强化学习训练流程、奖励函数设计、问题排查和最终结果验证。

## Learning Route

```text
ANYmal C Minimal Environment
    ↓
ANYmal C Point Navigation
    ↓
VBot Section01 Obstacle Navigation
```

## Project Stages

## Environment Matrix

| Stage | Task | MotrixLab Source | Branch / Version | Purpose |
|---|---|---|---|---|
| 01 | ANYmal C Minimal Environment | MotrixLab mainline | main / default | 学习环境注册、模型加载、观测和动作空间 |
| 02 | ANYmal C Point Navigation | MotrixLab mainline | main / default | 跑通 PPO 寻点导航训练链路 |
| 03 | VBot Section01 Navigation | MotrixLab arena branch | MotrixArena-S1 | 完成 MotrixArena Section01 越障导航任务 |

## Highlights

* 在 MotrixLab 中搭建自定义四足机器人导航环境
* 完成 ANYmal C 模型加载、环境注册、观测空间和动作空间验证
* 将 ANYmal C minimal 环境扩展为 point navigation 强化学习任务
* 使用 PPO 训练四足机器人寻点导航策略
* 设计 VBot Section01 waypoint 越障导航路线
* 设计 progress、tracking、anti-stall、reach-goal 等奖励项
* 调试动作缩放、PD 控制参数、速度命令和地形分段限速
* 解决 quaternion reset、sensor view panic、contact sensor、reset shape mismatch、后腿交叉、坡脚前栽等问题
* 最终 VBot 策略能够完成 Section01 第一阶段任务并到达 2026 平台

## Repository Structure

```text
.
├── docs/
│   ├── setup.md
│   ├── motrixlab_workflow.md
│   ├── training_and_evaluation.md
│   ├── troubleshooting.md
│   └── results.md
│
├── stages/
│   ├── 01_anymal_c_minimal/
│   ├── 02_anymal_c_point_navigation/
│   └── 03_vbot_section01_navigation/
│
├── assets/
│   ├── images/
│   ├── gifs/
│   └── videos/
│
├── README.md
├── .gitignore
└── LICENSE
```

## Final Result

最终训练得到的 VBot 策略能够从 Section01 起点出发，沿 waypoint 路线通过崎岖区域和坡道衔接区域，并最终到达 2026 平台。

## Notes

This repository does not include the full official MotrixLab source code, large checkpoints, or full training logs. It only contains custom environment files, training configuration, scripts, documentation, and selected results.

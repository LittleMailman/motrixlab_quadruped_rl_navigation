# MotrixLab Quadruped RL Navigation

基于 MotrixLab 的四足机器人强化学习导航实践项目。

本仓库记录了两个 MotrixLab 相关实践：

1. 基于 MotrixLab 主线环境完成 ANYmal C 最小环境搭建与寻点导航训练；
2. 基于 MotrixArena-S1 分支完成 VBot Section01 越障导航任务。

该项目的测试系统环境为 Ubuntu 22.04。

## Official Documentation

MotrixLab 官方文档：

[MotrixLab Documentation](https://motrixlab.readthedocs.io/zh-cn/latest/index.html#)

## Learning Route

```text
ANYmal C Minimal Environment
    ↓
ANYmal C Point Navigation
    ↓
VBot Section01 Obstacle Navigation
```

## Project Stages

| Stage | Project                      | Goal                                     |
| ----- | ---------------------------- | ---------------------------------------- |
| 01    | ANYmal C Minimal Environment | 搭建最小可运行的 ANYmal C 四足机器人环境                |
| 02    | ANYmal C Point Navigation    | 跑通 PPO 寻点导航训练链路                          |
| 03    | VBot Section01 Navigation    | 完成 MotrixArena Section01 越障导航并到达 2026 平台 |

## Environment Matrix

| Stage | Task                         | MotrixLab Source       | Branch / Version | Purpose                         |
| ----- | ---------------------------- | ---------------------- | ---------------- | ------------------------------- |
| 01    | ANYmal C Minimal Environment | MotrixLab mainline     | main / default   | 学习环境注册、模型加载、观测和动作空间             |
| 02    | ANYmal C Point Navigation    | MotrixLab mainline     | main / default   | 跑通 PPO 寻点导航训练链路                 |
| 03    | VBot Section01 Navigation    | MotrixLab arena branch | MotrixArena-S1   | 完成 MotrixArena Section01 越障导航任务 |

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
├── README.md
├── .gitignore
└── LICENSE
```

## Directory Guide

### `docs/`

项目文档目录，用于说明环境准备、运行流程、训练命令、问题排查和最终结果。

| File                              | Description                                            |
| --------------------------------- | ------------------------------------------------------ |
| `docs/setup.md`                   | 说明如何准备两个不同的 MotrixLab 运行环境：主线环境和 `MotrixArena-S1` 分支环境 |
| `docs/motrixlab_workflow.md`      | 说明 MotrixLab 的基本工作流，包括环境层、训练层和脚本入口                     |
| `docs/training_and_evaluation.md` | 记录三个阶段的训练、推理和 TensorBoard 命令                           |
| `docs/troubleshooting.md`         | 整理训练和环境搭建过程中遇到的主要问题与解决方式                               |
| `docs/results.md`                 | 汇总最终训练配置、推理结果和关键观察                                     |

### `stages/`

项目分阶段实践目录。每个阶段对应一个相对独立的 MotrixLab 实践任务。

| Directory                              | Description                               |
| -------------------------------------- | ----------------------------------------- |
| `stages/01_anymal_c_minimal/`          | ANYmal C 最小环境搭建阶段，用于理解模型加载、环境注册、观测空间和动作空间 |
| `stages/02_anymal_c_point_navigation/` | ANYmal C 寻点导航阶段，用于跑通 PPO 训练链路             |
| `stages/03_vbot_section01_navigation/` | VBot Section01 越障导航阶段，是本仓库的综合实战项目         |

### Root Files

| File         | Description                                |
| ------------ | ------------------------------------------ |
| `README.md`  | 仓库首页，说明项目目标、阶段划分、环境关系和最终结果                 |
| `.gitignore` | 忽略 Python 缓存、虚拟环境、训练日志、checkpoint 等不应上传的文件 |
| `LICENSE`    | 项目许可证文件                                    |

## Main Environments

| Stage | Environment Name              | Runtime               |
| ----- | ----------------------------- | --------------------- |
| 01    | `anymal_c_navigation_minimal` | MotrixLab mainline    |
| 02    | `anymal_c_navigation_point`   | MotrixLab mainline    |
| 03    | `vbot_navigation_section01`   | MotrixArena-S1 branch |

## Method Overview

```text
Observation
    ↓
Policy Network
    ↓
12D Joint Position Target
    ↓
PD Controller
    ↓
MotrixSim Physics
    ↓
Reward / Termination / Reset
```

## Final Result

最终训练得到的 VBot 策略能够从 Section01 起点出发，沿 waypoint 路线通过崎岖区域和坡道衔接区域，并最终到达 2026 平台。

## Notes

This repository does not include the full official MotrixLab source code, large checkpoints, or full training logs.

It only contains custom environment files, training configuration, scripts, documentation, and selected results.

机器人学基础理论，例如坐标系、齐次变换、轨迹规划、PD 控制和强化学习基本概念，将在博客中单独整理。本仓库只保留理解代码和复现实验所必需的内容。
- Blog: [小邮差不会敲代码](https://www.cnblogs.com/littlemailman)

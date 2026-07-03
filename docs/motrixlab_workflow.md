# MotrixLab Workflow

本文件记录本项目中使用到的 MotrixLab 基本工作流。

## Repository Layers

本项目中将 MotrixLab 理解为三层：

```text
motrix_envs/    # 环境层：物理仿真、任务定义、观测、奖励、终止条件
motrix_rl/      # 训练层：PPO 配置、RL 框架适配、训练参数
scripts/        # 应用层：训练、推理、可视化入口
```

## Main Scripts

| Script | Usage |
|---|---|
| `scripts/view.py` | 加载环境并可视化 |
| `scripts/train.py` | 启动训练 |
| `scripts/play.py` | 加载 checkpoint 并推理 |
| `tensorboard` | 查看训练曲线 |

## Main Environments

| Stage | Environment | Runtime |
|---|---|---|
| 01 | `anymal_c_navigation_minimal` | MotrixLab mainline |
| 02 | `anymal_c_navigation_point` | MotrixLab mainline |
| 03 | `vbot_navigation_section01` | MotrixArena-S1 |

## Environment Registration

自定义环境通常需要完成以下步骤：

1. 编写环境配置文件；
2. 编写环境实现文件；
3. 在 `__init__.py` 中导入模块触发注册；
4. 在训练配置中绑定 PPO 参数；
5. 使用 `view.py`、`train.py`、`play.py` 验证环境是否可运行。

## Notes

本仓库只保留自定义扩展文件、运行命令、训练配置和结果说明，不复制完整 MotrixLab 官方代码。

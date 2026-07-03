# 03 VBot Section01 Obstacle Navigation

本阶段是本仓库的综合实战项目。

目标是在 MotrixLab / MotrixSim 中训练 VBot 机器人完成 MotrixArena Section01 越障导航任务，并最终到达 2026 平台。

## Runtime Environment

| Item | Value |
|---|---|
| MotrixLab source | MotrixArena-S1 branch |
| Branch | `MotrixArena-S1` |
| Environment name | `vbot_navigation_section01` |
| Purpose | 完成 MotrixArena Section01 越障导航任务 |

## Goal

- 配置 VBot 机器人和 Section01 地形；
- 使用 waypoint 分阶段导航；
- 设计奖励函数引导机器人稳定前进；
- 使用 PPO 训练越障导航策略；
- 通过崎岖区域、坡脚衔接区域和坡道区域；
- 最终到达 2026 平台。

## Space

| Space | Dimension |
|---|---|
| Action Space | 12 |
| Observation Space | 68 |

## Waypoints

```text
(0.0, -0.60)
(0.0, 1.20)
(0.0, 2.25)
(-0.7, 3.8)
(-1.7, 5.8)
(-2.7, 8.0)
```

## Control Parameters

| Item | Value |
|---|---|
| action_scale | 0.45 |
| stiffness | 80.0 |
| damping | 6.0 |

## PPO Configuration

| Item | Value |
|---|---|
| num_envs | 4096 |
| rollouts | 48 |
| learning_epochs | 6 |
| mini_batches | 32 |
| learning_rate | 3e-4 |
| discount_factor | 0.99 |
| lambda | 0.95 |

## Reward Design

主要奖励项包括：

- `tracking_lin_vel`
- `tracking_ang_vel`
- `tracking_goal_vel`
- `tracking_yaw`
- `forward_progress`
- `target_progress`
- `reach_goal`
- `reach_all_goal`
- `feet_air_time`
- `anti_stall`

## Run

Train:

```bash
uv run scripts/train.py \
  --env vbot_navigation_section01 \
  --num-envs 4096 \
  --train-backend torch \
  --seed 42
```

Play:

```bash
uv run scripts/play.py \
  --env vbot_navigation_section01 \
  --sim-backend np \
  --num-envs 1 \
  --policy <policy_path>
```

TensorBoard:

```bash
uv run tensorboard --logdir runs --port 6006
```

## Result

最终策略能够完成 Section01 第一阶段越障导航任务，并到达 2026 平台。

训练结果中，回合最大存活步数稳定达到 4000 步上限，reward 曲线随训练推进持续上升。

# 02 ANYmal C Point Navigation

本阶段在 ANYmal C minimal 环境基础上扩展为寻点导航任务。

## Runtime Environment

| Item | Value |
|---|---|
| MotrixLab source | MotrixLab mainline |
| Branch | main / default |
| Environment name | `anymal_c_navigation_point` |
| Purpose | 跑通 PPO 寻点导航训练链路 |

## Goal

- 加入随机目标点；
- 根据目标方向生成速度命令；
- 扩展 observation space；
- 设计 point navigation 奖励函数；
- 编写 PPO 训练配置；
- 使用 TensorBoard 观察训练过程；
- 使用 `play.py` 验证策略效果。

## Space

| Space | Dimension |
|---|---|
| Action Space | 12 |
| Observation Space | 54 |

## Run

Train:

```bash
uv run scripts/train.py \
  --env anymal_c_navigation_point \
  --sim-backend np
```

Play:

```bash
uv run scripts/play.py \
  --env anymal_c_navigation_point \
  --sim-backend np \
  --num-envs 1 \
  --rllib skrl \
  --policy <policy_path>
```

TensorBoard:

```bash
uv run tensorboard --logdir runs
```

## Role

该阶段用于从“环境能运行”过渡到“策略能训练”，为后续 VBot 越障导航任务做准备。

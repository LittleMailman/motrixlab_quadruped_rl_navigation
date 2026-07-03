# Results

本文件记录最终训练结果。

## Final Task

最终任务环境：

```text
vbot_navigation_section01
```

任务目标：

- 从 Section01 起点出发；
- 通过崎岖区域；
- 通过坡脚衔接区域；
- 沿坡道继续前进；
- 到达 2026 平台。

## Final Control Parameters

| Item | Value |
|---|---|
| `action_scale` | `0.45` |
| `stiffness` | `80.0` |
| `damping` | `6.0` |

## Final PPO Configuration

| Item | Value |
|---|---|
| `num_envs` | `4096` |
| `rollouts` | `48` |
| `learning_epochs` | `6` |
| `mini_batches` | `32` |
| `learning_rate` | `3e-4` |
| `discount_factor` | `0.99` |
| `lambda` | `0.95` |
| `entropy_loss_scale` | `0.0` |
| `value_loss_scale` | `2.0` |
| `policy_hidden_layer_sizes` | `512, 256, 128` |
| `value_hidden_layer_sizes` | `512, 256, 128` |

## Result Summary

最终训练得到的策略能够完成 MotrixArena Section01 越障导航第一阶段。

推理测试中，机器人能够：

1. 从 Section01 起点出发；
2. 沿 waypoint 路线向前移动；
3. 通过崎岖区域；
4. 在坡脚前保持稳定；
5. 沿坡道继续前进；
6. 到达 2026 平台。

## Key Observations

- 回合最大存活步数稳定达到 4000 步上限；
- reward 曲线随训练推进持续上升；
- 最终 checkpoint 能够在 play 阶段复现到达平台的行为；
- 动作仍存在小碎步和蠕动现象，但已满足任务完成标准。

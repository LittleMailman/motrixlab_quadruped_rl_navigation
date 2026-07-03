# 01 ANYmal C Minimal Environment

本阶段目标是在 MotrixLab 中搭建最小可运行的 ANYmal C 四足机器人环境。

## Runtime Environment

| Item | Value |
|---|---|
| MotrixLab source | MotrixLab mainline |
| Branch | main / default |
| Environment name | `anymal_c_navigation_minimal` |
| Purpose | 学习自定义环境注册、模型加载、观测空间和动作空间 |

## Goal

- 引入 ANYmal C MJCF 模型；
- 创建 `navigation/anymal_c/` 自定义环境目录；
- 编写环境配置文件 `cfg.py`；
- 编写环境实现文件 `env.py`；
- 使用 `view.py` 成功加载模型；
- 验证 observation space 和 action space。

## Space

| Space | Dimension |
|---|---|
| Action Space | 12 |
| Observation Space | 45 |

## Run

```bash
uv run scripts/view.py \
  --env anymal_c_navigation_minimal \
  --sim-backend np \
  --num-envs 1
```

## Role

该阶段用于理解 MotrixLab 自定义环境的最小组成，包括模型文件、配置类、环境类、观测空间和动作空间。

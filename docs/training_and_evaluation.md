# Training and Evaluation

本文件记录三个阶段的训练、推理和 TensorBoard 命令。

## 01 ANYmal C Minimal Environment

查看环境：

```bash
uv run scripts/view.py \
  --env anymal_c_navigation_minimal \
  --sim-backend np \
  --num-envs 1
```

## 02 ANYmal C Point Navigation

训练：

```bash
uv run scripts/train.py \
  --env anymal_c_navigation_point \
  --sim-backend np
```

推理：

```bash
uv run scripts/play.py \
  --env anymal_c_navigation_point \
  --sim-backend np \
  --num-envs 1 \
  --rllib skrl \
  --policy <policy_path>
```

## 03 VBot Section01 Navigation

训练：

```bash
uv run scripts/train.py \
  --env vbot_navigation_section01 \
  --num-envs 4096 \
  --train-backend torch \
  --seed 42
```

推理：

```bash
uv run scripts/play.py \
  --env vbot_navigation_section01 \
  --sim-backend np \
  --num-envs 1 \
  --policy <policy_path>
```

TensorBoard：

```bash
uv run tensorboard --logdir runs --port 6006
```

## Checkpoint

训练完成后的 checkpoint 通常位于：

```text
runs/<env_name>/<run_name>/checkpoints/best_agent.pt
```

本仓库不直接上传大型 checkpoint 文件。

# Troubleshooting

本文件记录项目中遇到的主要工程问题、现象、原因和解决方式。

## 1. Quaternion Reset Error

### Symptom

训练时出现类似错误：

```text
The dof pos at index 25 is invalid, it should be a normalized quaternion.
```

### Cause

环境中除了机器人 base 外，还可能包含 arrow 等 freejoint body。如果错误地写入全局 `dof_pos`，可能污染 freejoint 的 quaternion，导致四元数非法。

### Fix

- 不直接向全局 `dof_pos` 的最后几位写入关节角；
- 对 freejoint quaternion 做合法化处理；
- 单独设置机器人关节角；
- 对可视化 body 显式设置单位四元数。

## 2. Sensor View Panic

### Symptom

读取部分 sensor 时出现底层 sensor view 相关 panic。

### Cause

当前 XML 中 sensor 名称或 sensor view 与环境读取逻辑不完全匹配。

### Fix

- 对 sensor 读取增加安全包装；
- 读取失败后返回默认值；
- 必要时通过 pose 差分估计速度和 yaw rate；
- 避免训练链路因单个 sensor 异常崩溃。

## 3. Contact Sensor Issue

### Symptom

足端接触状态读取不稳定，可能导致训练中断。

### Cause

部分 contact sensor 名称或底层 view 与当前环境不完全一致。

### Fix

- 使用安全读取函数；
- 对读取失败的 sensor 做缓存；
- 将多个足端 sensor 聚合为每条腿的 contact proxy；
- 保证观测维度稳定。

## 4. Reset Shape Mismatch

### Symptom

训练中出现 batch shape 不一致，例如：

```text
operands could not be broadcast together with shapes (255,) () (256,)
```

### Cause

部分环境 done 后，MotrixLab 可能只 reset 子集环境，而不是完整 `num_envs`。如果 reward 或 termination 中固定使用 `self._num_envs`，就会导致 batch shape 不一致。

### Fix

在 reward 和 termination 中统一使用当前 batch 大小：

```python
num_envs = data.shape[0]
```

并对 `info` 中的数组进行 shape 检查。

## 5. Rear Leg Crossing

### Symptom

机器人能够前进，但后腿出现交叉、互相绊腿。

### Cause

原始奖励主要关注整体前进，没有明确约束后腿左右髋关节的相对位置，也没有要求后腿参与推进。

### Fix

- 加入 `rear_cross_penalty`；
- 加入 `rear_hip_l2`；
- 加入 `rear_drive_reward`；
- 保持保守权重，避免后腿通过原地抖动刷奖励。

## 6. Falling Before Slope

### Symptom

机器人通过崎岖区后，在坡脚前或上坡前容易前栽。

### Cause

- 崎岖区出口速度过快；
- 前腿接近坡面时后腿没有跟上；
- 坡脚姿态变化大，速度命令过激。

### Fix

- 对 drop exit 和 approach slope 区域进行分段限速；
- 加入 pitch penalty；
- 坡道段保持低速持续前进；
- 适当放宽终止条件，让机器人有机会恢复姿态。

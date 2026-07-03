# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0.
# ==============================================================================

import os
from dataclasses import dataclass, field

from motrix_envs import registry
from motrix_envs.base import EnvCfg

ANYMAL_C_NAVIGATION_POINT_ENV_NAME = "anymal_c_navigation_point"

def _default_model_file() -> str:
    """Return the custom ANYmal C scene file."""
    return os.path.join(
        os.path.dirname(__file__),
        "xmls",
        "anybotics_anymal_c",
        "scene.xml",
    )

@dataclass
class NoiseConfig:
    level: float = 1.0
    scale_joint_angle: float = 0.03
    scale_joint_vel: float = 1.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.1

@dataclass
class ControlConfig:
    """Action decoding parameters.

    Position control:
        target_joint_angle = default_angle + action * action_scale
    """

    # 提交增强版：保持 0.15。
    # 如果后续发现动作仍然偏小，可以尝试 0.17；
    # 但不建议直接上 0.20，容易导致步态变乱。
    action_scale: float = 0.15

@dataclass
class InitState:
    """Initial robot state."""

    pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.50])

    # 机器人初始位置分布范围。
    # 目标点是相对机器人初始位置采样，所以这里不需要太大。
    pos_randomization_range: list[float] = field(
        default_factory=lambda: [-1.5, -1.5, 1.5, 1.5]
    )

    default_joint_angles: dict[str, float] = field(
        default_factory=lambda: {
            "LF_HAA": 0.0,
            "RF_HAA": 0.0,
            "LH_HAA": 0.0,
            "RH_HAA": 0.0,
            "LF_HFE": 0.4,
            "RF_HFE": 0.4,
            "LH_HFE": -0.4,
            "RH_HFE": -0.4,
            "LF_KFE": -0.8,
            "RF_KFE": -0.8,
            "LH_KFE": 0.8,
            "RH_KFE": 0.8,
        }
    )

@dataclass
class Commands:
    """Navigation command settings.

    提交增强版：
        目标方向：全方向
        目标半径：1.2m ~ 5.0m

    pose_command_range 主要用于提供 yaw_min / yaw_max。
    reset() 中实际目标采样使用极坐标：
        radius in [target_radius_min, target_radius_max]
        angle  in [pose_command_range[2], pose_command_range[5]]
    """

    # [dx_min, dy_min, yaw_min, dx_max, dy_max, yaw_max]
    pose_command_range: list[float] = field(
        default_factory=lambda: [-5.0, -5.0, -3.14, 5.0, 5.0, 3.14]
    )

    # 大范围演示时放宽到达判定。
    # 5m 目标下，0.55m 范围内可认为已经完成寻点。
    position_threshold: float = 0.55
    heading_threshold: float = 0.55
    yaw_deadband: float = 0.20

    # 极坐标目标采样半径。
    target_radius_min: float = 1.2
    target_radius_max: float = 5.0

@dataclass
class Normalization:
    lin_vel: float = 2.0
    ang_vel: float = 0.25
    dof_pos: float = 1.0
    dof_vel: float = 0.05

@dataclass
class Asset:
    body_name: str = "base"

    # 你的 scene.xml 里地面 geom 名称是 floor。
    ground_name: str = "floor"

    foot_names: list[str] = field(
        default_factory=lambda: ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]
    )

    terminate_after_contacts_on: list[str] = field(
        default_factory=lambda: ["base"]
    )

    # 目标点和方向箭头可视化 body。
    target_marker_body: str = "target_marker"
    robot_heading_arrow_body: str = "robot_heading_arrow"
    desired_heading_arrow_body: str = "desired_heading_arrow"

@dataclass
class Sensor:
    """Sensor names.

    保持为空，避免 MotrixSim sensor view panic。
    base velocity 和 yaw rate 由 pose 差分估计。
    """

    base_linvel: str = ""
    base_gyro: str = ""

@dataclass
class RewardConfig:
    """Reward parameters for point navigation."""

    termination_penalty: float = -20.0

    # 鼓励跟踪目标方向速度。
    tracking_lin_vel_scale: float = 3.0
    tracking_ang_vel_scale: float = 0.2

    # 鼓励距离下降。
    approach_reward_scale: float = 8.0

    # 到达与停止奖励。
    arrival_bonus: float = 15.0
    stop_reward_scale: float = 2.0
    zero_yaw_bonus: float = 4.0

    # 物理稳定性惩罚。
    lin_vel_z_penalty_scale: float = 2.0
    ang_vel_xy_penalty_scale: float = 0.05

    # 动作惩罚保持较低，避免抑制步态。
    torque_penalty_scale: float = 0.000001
    action_rate_penalty_scale: float = 0.0002

    # 防止远离目标时站着不动。
    distance_penalty_scale: float = 0.08
    low_speed_far_penalty_scale: float = 0.30
    no_progress_penalty_scale: float = 0.20

@dataclass
class DebugConfig:
    # 准备作业阶段建议打开，方便观察 distance / speed。
    # 如果终端输出太多，改成 False。
    enable: bool = True
    print_every: int = 1000

@registry.envcfg(ANYMAL_C_NAVIGATION_POINT_ENV_NAME)
@dataclass
class AnymalCNavigationPointCfg(EnvCfg):
    """Custom ANYmal C point-navigation environment.

    Registered task:
        anymal_c_navigation_point
    """

    model_file: str = field(default_factory=_default_model_file)

    reset_noise_scale: float = 0.01
    max_episode_seconds: float = 12.0
    sim_dt: float = 0.01
    ctrl_dt: float = 0.01
    reset_yaw_scale: float = 0.1
    max_dof_vel: float = 100.0

    env_name: str = ANYMAL_C_NAVIGATION_POINT_ENV_NAME
    task_name: str = ANYMAL_C_NAVIGATION_POINT_ENV_NAME
    task_type: str = "navigation"
    robot_name: str = "anymal_c"
    is_custom_task: bool = True

    num_actions: int = 12
    num_observations: int = 54

    action_dim: int = 12
    observation_dim: int = 54

    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    normalization: Normalization = field(default_factory=Normalization)
    asset: Asset = field(default_factory=Asset)
    sensor: Sensor = field(default_factory=Sensor)
    debug: DebugConfig = field(default_factory=DebugConfig)

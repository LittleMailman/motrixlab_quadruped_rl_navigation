# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0.
# ==============================================================================

import os
from dataclasses import dataclass, field

from motrix_envs import registry
from motrix_envs.base import EnvCfg

# ==============================================================================
# Registered MotrixLab task name
# ==============================================================================
#
# View:
#   uv run scripts/view.py --env anymal_c_navigation_minimal --sim-backend np --num-envs 1
#
# Train, later:
#   uv run scripts/train.py --env anymal_c_navigation_minimal --sim-backend np --num-envs 4
#
ANYMAL_C_NAVIGATION_MINIMAL_ENV_NAME = "anymal_c_navigation_minimal"

def _default_model_file() -> str:
    """Return the default MuJoCo scene file for ANYmal C."""
    return os.path.join(
        os.path.dirname(__file__),
        "xmls",
        "anybotics_anymal_c",
        "scene.xml",
    )

@dataclass
class InitStateCfg:
    """Initial robot state."""

    # Root base position.
    # If the robot appears slightly above/below the floor, tune z between 0.52 and 0.62.
    pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.56])

    # Default standing joint angles.
    # These names must match actuator / joint names in anymal_c.xml.
    default_joint_angles: dict[str, float] = field(
        default_factory=lambda: {
            "LF_HAA": 0.0,
            "LF_HFE": 0.4,
            "LF_KFE": -0.8,
            "RF_HAA": 0.0,
            "RF_HFE": 0.4,
            "RF_KFE": -0.8,
            "LH_HAA": 0.0,
            "LH_HFE": -0.4,
            "LH_KFE": 0.8,
            "RH_HAA": 0.0,
            "RH_HFE": -0.4,
            "RH_KFE": 0.8,
        }
    )

@dataclass
class ControlCfg:
    """Action decoding parameters."""

    # Normalized action [-1, 1] is mapped to joint position offset.
    #
    # For position actuators, do not start with a large action scale.
    # A large random position target can make the robot jump or fall immediately.
    action_scale: float = 0.06

@dataclass
class AssetCfg:
    """Names of bodies and geoms used by the environment."""

    # Root body name in anymal_c.xml.
    body_name: str = "base"

    # Ground geom name in scene.xml.
    ground_name: str = "ground"

    # If any of these geoms collide with the ground, the episode terminates.
    terminate_after_contacts_on: list[str] = field(default_factory=lambda: ["base"])

@dataclass
class SensorCfg:
    """Sensor names used in observation."""

    # Recommended XML sensor:
    # <framelinvel name="base_linvel" objtype="body" objname="base"/>
    base_linvel: str = "base_linvel"

    # Recommended XML sensor:
    # <gyro name="base_gyro" site="imu_site"/>
    base_gyro: str = "base_gyro"

@dataclass
class RewardCfg:
    """Minimal reward parameters."""

    upright_reward_scale: float = 1.0
    action_penalty_scale: float = 0.001

@dataclass
class TerminationCfg:
    """Minimal termination parameters."""

    min_base_height: float = 0.25
    max_tilt_xy_norm: float = 0.9

@registry.envcfg(ANYMAL_C_NAVIGATION_MINIMAL_ENV_NAME)
@dataclass
class AnymalCNavigationMinimalCfg(EnvCfg):
    """Minimal ANYmal C navigation environment configuration.

    Current registered task:
        anymal_c_navigation_minimal

    Current stage:
        Minimal runnable model-loading and visualization environment.

    Later full navigation extension:
        Add target command, position error, heading error, distance,
        reached flag and stop-ready flag to expand observation from 45D to 54D.
    """

    # --------------------------------------------------------------------------
    # MotrixLab / MotrixSim base configuration
    # --------------------------------------------------------------------------
    model_file: str = field(default_factory=_default_model_file)

    # Simulation timestep.
    sim_dt: float = 0.01

    # Control timestep.
    # For the minimal viewer test, keep sim_dt and ctrl_dt equal first.
    # Later you can tune to sim_dt=0.002, ctrl_dt=0.02.
    ctrl_dt: float = 0.01

    max_episode_seconds: float = 5.0
    render_spacing: float = 1.0

    # --------------------------------------------------------------------------
    # Task metadata
    # --------------------------------------------------------------------------
    env_name: str = ANYMAL_C_NAVIGATION_MINIMAL_ENV_NAME
    task_name: str = ANYMAL_C_NAVIGATION_MINIMAL_ENV_NAME
    task_type: str = "navigation"
    robot_name: str = "anymal_c"
    is_custom_task: bool = True

    # --------------------------------------------------------------------------
    # Space definition
    # --------------------------------------------------------------------------
    # 4 legs * 3 actuators = 12
    num_actions: int = 12

    # linvel(3) + gyro(3) + projected_gravity(3)
    # + joint_pos_error(12) + joint_vel(12) + last_action(12) = 45
    num_observations: int = 45

    # Aliases for readability.
    action_dim: int = 12
    observation_dim: int = 45

    # --------------------------------------------------------------------------
    # Sub-configs
    # --------------------------------------------------------------------------
    init_state: InitStateCfg = field(default_factory=InitStateCfg)
    control: ControlCfg = field(default_factory=ControlCfg)
    asset: AssetCfg = field(default_factory=AssetCfg)
    sensor: SensorCfg = field(default_factory=SensorCfg)
    reward: RewardCfg = field(default_factory=RewardCfg)
    termination: TerminationCfg = field(default_factory=TerminationCfg)

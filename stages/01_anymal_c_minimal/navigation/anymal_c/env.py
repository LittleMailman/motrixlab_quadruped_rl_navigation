# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0.
# ==============================================================================

import gymnasium as gym
import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.math import quaternion
from motrix_envs.np.env import NpEnv, NpEnvState

from .cfg import (
    ANYMAL_C_NAVIGATION_MINIMAL_ENV_NAME,
    AnymalCNavigationMinimalCfg,
)

@registry.env(ANYMAL_C_NAVIGATION_MINIMAL_ENV_NAME, "np")
class AnymalCNavigationMinimalEnv(NpEnv):
    """Minimal ANYmal C navigation environment.

    Registered MotrixLab task:
        anymal_c_navigation_minimal

    Observation space:
        45D = base_linvel(3)
            + base_gyro(3)
            + projected_gravity(3)
            + joint_pos_error(12)
            + joint_vel(12)
            + last_action(12)

    Action space:
        12D normalized position target offsets.
    """

    _cfg: AnymalCNavigationMinimalCfg

    def __init__(self, cfg: AnymalCNavigationMinimalCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)

        self._body = self._model.get_body(cfg.asset.body_name)

        self._num_action = self._model.num_actuators
        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel

        if self._num_action != cfg.num_actions:
            raise ValueError(
                f"ANYmal C actuator number mismatch: "
                f"model.num_actuators={self._num_action}, "
                f"cfg.num_actions={cfg.num_actions}"
            )

        self._action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._num_action,),
            dtype=np.float32,
        )

        self._observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(cfg.num_observations,),
            dtype=np.float32,
        )

        self.gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        self.default_angles = self._build_default_angles()

        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_dof_vel = np.zeros((self._num_dof_vel,), dtype=np.float32)

        self._apply_init_state()
        self._init_termination_contact()

    # --------------------------------------------------------------------------
    # Spaces
    # --------------------------------------------------------------------------
    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    # --------------------------------------------------------------------------
    # Initialization helpers
    # --------------------------------------------------------------------------
    def _build_default_angles(self) -> np.ndarray:
        """Build default joint angles following actuator order."""
        default_angles = np.zeros((self._num_action,), dtype=np.float32)

        for i, actuator_name in enumerate(self._model.actuator_names):
            matched = False

            for joint_name, angle in self._cfg.init_state.default_joint_angles.items():
                if joint_name in actuator_name:
                    default_angles[i] = angle
                    matched = True
                    break

            if not matched:
                raise ValueError(
                    f"Cannot find default joint angle for actuator '{actuator_name}'. "
                    f"Please check cfg.init_state.default_joint_angles."
                )

        return default_angles

    def _apply_init_state(self) -> None:
        """Write root position and default joint angles into init dof position."""
        init_pos = np.asarray(self._cfg.init_state.pos, dtype=np.float32)

        if init_pos.shape != (3,):
            raise ValueError(f"cfg.init_state.pos must be 3D, got {init_pos.shape}")

        # For a floating-base robot, dof_pos is usually:
        # [root_xyz(3), root_quat(4), joint_pos(12)].
        if self._init_dof_pos.shape[0] >= 7 + self._num_action:
            self._init_dof_pos[0:3] = init_pos

        self._init_dof_pos[-self._num_action:] = self.default_angles

    def _init_termination_contact(self) -> None:
        """Initialize base-vs-ground contact pairs for termination."""
        try:
            ground_index = self._model.get_geom_index(self._cfg.asset.ground_name)
        except Exception:
            ground_index = None

        body_indices = []

        if ground_index is not None:
            for geom_name in self._cfg.asset.terminate_after_contacts_on:
                try:
                    geom_index = self._model.get_geom_index(geom_name)
                except Exception:
                    geom_index = None

                if geom_index is not None:
                    body_indices.append(geom_index)

        if ground_index is None or len(body_indices) == 0:
            # Keep the environment runnable even if geom names differ.
            # Height and tilt termination still work.
            self.termination_contact = None
            self.num_termination_check = 0
            return

        self.termination_contact = np.array(
            [[idx, ground_index] for idx in body_indices],
            dtype=np.uint32,
        )
        self.num_termination_check = self.termination_contact.shape[0]

    # --------------------------------------------------------------------------
    # State extraction
    # --------------------------------------------------------------------------
    def get_dof_pos(self, data: mtx.SceneData) -> np.ndarray:
        """Return 12D joint positions."""
        joint_pos = self._body.get_joint_dof_pos(data)

        if joint_pos.shape[-1] != self._num_action:
            joint_pos = joint_pos[:, -self._num_action:]

        return joint_pos.astype(np.float32)

    def get_dof_vel(self, data: mtx.SceneData) -> np.ndarray:
        """Return 12D joint velocities."""
        joint_vel = self._body.get_joint_dof_vel(data)

        if joint_vel.shape[-1] != self._num_action:
            joint_vel = joint_vel[:, -self._num_action:]

        return joint_vel.astype(np.float32)

    def _get_sensor_or_zeros(
        self,
        sensor_name: str,
        data: mtx.SceneData,
        dim: int,
    ) -> np.ndarray:
        """Read a sensor if it is safely available, otherwise return zeros.

        MotrixSim may raise a PyO3/Rust PanicException when trying to read
        a sensor that does not exist or has no valid view. Therefore, for the
        minimal visualization stage, this function must avoid crashing the
        viewer and safely fall back to zeros.
        """
        zeros = np.zeros((data.shape[0], dim), dtype=np.float32)

        if sensor_name is None or sensor_name == "":
            return zeros

        sensor_names = getattr(self._model, "sensor_names", None)

        if sensor_names is not None:
            if sensor_name not in sensor_names:
                return zeros

        try:
            value = self._model.get_sensor_value(sensor_name, data)

            if value is None:
                return zeros

            value = np.asarray(value, dtype=np.float32)

            if value.shape[-1] != dim:
                return zeros

            return value

        except BaseException:
            return zeros

    # --------------------------------------------------------------------------
    # Observation
    # --------------------------------------------------------------------------
    def _get_obs(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        """Build 45D observation."""
        pose = self._body.get_pose(data)

        base_quat = pose[:, 3:7]

        base_linvel = self._get_sensor_or_zeros(
            self._cfg.sensor.base_linvel,
            data,
            dim=3,
        )

        base_gyro = self._get_sensor_or_zeros(
            self._cfg.sensor.base_gyro,
            data,
            dim=3,
        )

        projected_gravity = quaternion.rotate_inverse(base_quat, self.gravity_vec)

        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)

        joint_pos_error = joint_pos - self.default_angles

        last_actions = info.get(
            "last_actions",
            np.zeros((data.shape[0], self._num_action), dtype=np.float32),
        )

        obs = np.concatenate(
            [
                base_linvel,        # 3
                base_gyro,          # 3
                projected_gravity,  # 3
                joint_pos_error,    # 12
                joint_vel,          # 12
                last_actions,       # 12
            ],
            axis=-1,
        ).astype(np.float32)

        expected_shape = (data.shape[0], self._cfg.num_observations)
        if obs.shape != expected_shape:
            raise RuntimeError(
                f"Observation shape mismatch: expected {expected_shape}, got {obs.shape}"
            )

        return obs

    # --------------------------------------------------------------------------
    # RL loop
    # --------------------------------------------------------------------------
    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> NpEnvState:
        """Apply normalized actions to position actuators.

        Action decoding:
            target_joint_angle = default_angle + action * action_scale
        """
        actions = np.asarray(actions, dtype=np.float32)

        if actions.ndim == 1:
            actions = np.tile(actions, (self._num_envs, 1))

        if actions.shape != (self._num_envs, self._num_action):
            raise ValueError(
                f"Expected actions shape {(self._num_envs, self._num_action)}, "
                f"got {actions.shape}"
            )

        actions = np.clip(actions, self.action_space.low, self.action_space.high)

        if "current_actions" not in state.info:
            state.info["current_actions"] = np.zeros_like(actions)

        state.info["last_actions"] = state.info["current_actions"].copy()
        state.info["current_actions"] = actions.copy()

        target_joint_pos = (
            self.default_angles[None, :]
            + actions * self._cfg.control.action_scale
        ).astype(np.float32)

        state.data.actuator_ctrls = target_joint_pos

        return state

    def update_state(self, state: NpEnvState) -> NpEnvState:
        """Update observation, reward and termination after simulation step."""
        obs = self._get_obs(state.data, state.info)
        terminated = self._compute_terminated(state.data)
        reward = self._compute_reward(state.data, state.info, terminated)

        return state.replace(
            obs=obs,
            reward=reward,
            terminated=terminated,
        )

    def reset(self, data: mtx.SceneData) -> tuple[np.ndarray, dict]:
        """Reset environments and return initial observation."""
        num_envs = data.shape[0]

        dof_pos = np.tile(
            self._init_dof_pos,
            (*data.shape, 1),
        ).astype(np.float32)

        dof_vel = np.tile(
            self._init_dof_vel,
            (*data.shape, 1),
        ).astype(np.float32)

        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)

        info = {
            "last_actions": np.zeros(
                (num_envs, self._num_action),
                dtype=np.float32,
            ),
            "current_actions": np.zeros(
                (num_envs, self._num_action),
                dtype=np.float32,
            ),
        }

        obs = self._get_obs(data, info)

        return obs, info

    # --------------------------------------------------------------------------
    # Reward / termination
    # --------------------------------------------------------------------------
    def _compute_terminated(self, data: mtx.SceneData) -> np.ndarray:
        """Minimal termination: low body, large tilt, or base contact."""
        pose = self._body.get_pose(data)

        base_height = pose[:, 2]
        base_quat = pose[:, 3:7]

        projected_gravity = quaternion.rotate_inverse(base_quat, self.gravity_vec)
        tilt_xy_norm = np.linalg.norm(projected_gravity[:, :2], axis=1)

        terminated = np.zeros((data.shape[0],), dtype=bool)

        terminated = np.logical_or(
            terminated,
            base_height < self._cfg.termination.min_base_height,
        )

        terminated = np.logical_or(
            terminated,
            tilt_xy_norm > self._cfg.termination.max_tilt_xy_norm,
        )

        terminated = np.logical_or(
            terminated,
            ~np.isfinite(base_height),
        )

        terminated = np.logical_or(
            terminated,
            ~np.isfinite(projected_gravity).all(axis=1),
        )

        if self.termination_contact is not None and self.num_termination_check > 0:
            contact_query = self._model.get_contact_query(data)
            base_contact = contact_query.is_colliding(self.termination_contact)
            base_contact = base_contact.reshape(
                (data.shape[0], self.num_termination_check)
            ).any(axis=1)

            terminated = np.logical_or(terminated, base_contact)

        return terminated

    def _compute_reward(
        self,
        data: mtx.SceneData,
        info: dict,
        terminated: np.ndarray,
    ) -> np.ndarray:
        """Minimal reward: upright reward minus action penalty."""
        pose = self._body.get_pose(data)
        projected_gravity = quaternion.rotate_inverse(pose[:, 3:7], self.gravity_vec)

        # If upright, projected_gravity[:, 2] should be close to -1.
        upright_reward = np.clip(-projected_gravity[:, 2], 0.0, 1.0)

        current_actions = info.get(
            "current_actions",
            np.zeros((data.shape[0], self._num_action), dtype=np.float32),
        )

        action_penalty = np.sum(np.square(current_actions), axis=1)

        reward = (
            self._cfg.reward.upright_reward_scale * upright_reward
            - self._cfg.reward.action_penalty_scale * action_penalty
        )

        reward = np.where(terminated, 0.0, reward)

        return reward.astype(np.float32)

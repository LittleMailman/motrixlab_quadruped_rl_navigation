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

from .cfg_point import (
    ANYMAL_C_NAVIGATION_POINT_ENV_NAME,
    AnymalCNavigationPointCfg,
)

@registry.env(ANYMAL_C_NAVIGATION_POINT_ENV_NAME, "np")
class AnymalCNavigationPointEnv(NpEnv):
    """Custom ANYmal C point-navigation environment."""

    _cfg: AnymalCNavigationPointCfg

    def __init__(self, cfg: AnymalCNavigationPointCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)

        self._body = self._model.get_body(cfg.asset.body_name)

        self._action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32,
        )

        self._observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(54,),
            dtype=np.float32,
        )

        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel
        self._num_action = self._model.num_actuators

        if self._num_action != 12:
            raise ValueError(f"Expected 12 actuators, got {self._num_action}")

        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_dof_vel = np.zeros(
            (self._model.num_dof_vel,),
            dtype=np.float32,
        )

        self.gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        self._init_contact_geometry()
        self._init_buffer()
        self._init_visualization_bodies()

        self._debug_step = 0

    # --------------------------------------------------------------------------
    # Spaces
    # --------------------------------------------------------------------------
    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------
    def _init_buffer(self):
        cfg = self._cfg

        self.default_angles = np.zeros(self._num_action, dtype=np.float32)

        self.commands_scale = np.array(
            [
                cfg.normalization.lin_vel,
                cfg.normalization.lin_vel,
                cfg.normalization.ang_vel,
            ],
            dtype=np.float32,
        )

        for i in range(self._model.num_actuators):
            matched = False
            actuator_name = self._model.actuator_names[i]

            for name, angle in cfg.init_state.default_joint_angles.items():
                if name in actuator_name:
                    self.default_angles[i] = angle
                    matched = True
                    break

            if not matched:
                raise ValueError(
                    f"Cannot find default joint angle for actuator '{actuator_name}'"
                )

        if self._init_dof_pos.shape[0] >= 7 + self._num_action:
            self._init_dof_pos[0:3] = np.asarray(cfg.init_state.pos, dtype=np.float32)

        self._init_dof_pos[-self._num_action:] = self.default_angles

    def _init_contact_geometry(self):
        cfg = self._cfg

        try:
            self.ground_index = self._model.get_geom_index(cfg.asset.ground_name)
        except Exception:
            self.ground_index = None

        self._init_termination_contact()
        self._init_foot_contact()

    def _init_termination_contact(self):
        cfg = self._cfg
        base_indices = []

        if self.ground_index is not None:
            for base_name in cfg.asset.terminate_after_contacts_on:
                try:
                    base_idx = self._model.get_geom_index(base_name)
                    if base_idx is not None:
                        base_indices.append(base_idx)
                except Exception:
                    pass

        if base_indices and self.ground_index is not None:
            self.termination_contact = np.array(
                [[idx, self.ground_index] for idx in base_indices],
                dtype=np.uint32,
            )
            self.num_termination_check = self.termination_contact.shape[0]
        else:
            self.termination_contact = np.zeros((0, 2), dtype=np.uint32)
            self.num_termination_check = 0

    def _init_foot_contact(self):
        cfg = self._cfg
        foot_indices = []

        if self.ground_index is not None:
            for foot_name in cfg.asset.foot_names:
                try:
                    foot_idx = self._model.get_geom_index(foot_name)
                    if foot_idx is not None:
                        foot_indices.append(foot_idx)
                except Exception:
                    pass

        if foot_indices and self.ground_index is not None:
            self.foot_contact_check = np.array(
                [[idx, self.ground_index] for idx in foot_indices],
                dtype=np.uint32,
            )
            self.num_foot_check = self.foot_contact_check.shape[0]
        else:
            self.foot_contact_check = np.zeros((0, 2), dtype=np.uint32)
            self.num_foot_check = 0

    def _init_visualization_bodies(self):
        cfg = self._cfg.asset

        self._target_marker_body = self._safe_get_body(cfg.target_marker_body)
        self._robot_heading_arrow_body = self._safe_get_body(
            cfg.robot_heading_arrow_body
        )
        self._desired_heading_arrow_body = self._safe_get_body(
            cfg.desired_heading_arrow_body
        )

    def _safe_get_body(self, body_name: str):
        try:
            return self._model.get_body(body_name)
        except Exception:
            return None

    # --------------------------------------------------------------------------
    # State helpers
    # --------------------------------------------------------------------------
    def get_dof_pos(self, data: mtx.SceneData):
        joint_pos = self._body.get_joint_dof_pos(data)
        if joint_pos.shape[-1] != self._num_action:
            joint_pos = joint_pos[:, -self._num_action:]
        return joint_pos.astype(np.float32)

    def get_dof_vel(self, data: mtx.SceneData):
        joint_vel = self._body.get_joint_dof_vel(data)
        if joint_vel.shape[-1] != self._num_action:
            joint_vel = joint_vel[:, -self._num_action:]
        return joint_vel.astype(np.float32)

    @staticmethod
    def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def _compute_projected_gravity(self, quat: np.ndarray) -> np.ndarray:
        return quaternion.rotate_inverse(quat, self.gravity_vec)

    def _safe_get_sensor_or_none(self, sensor_name: str, data: mtx.SceneData):
        if sensor_name is None or sensor_name == "":
            return None

        sensor_names = getattr(self._model, "sensor_names", None)
        if sensor_names is not None and sensor_name not in sensor_names:
            return None

        try:
            value = self._model.get_sensor_value(sensor_name, data)
            value = np.asarray(value, dtype=np.float32)
            return value
        except BaseException:
            return None

    def _extract_root_state(self, data: mtx.SceneData, info: dict):
        pose = self._body.get_pose(data)
        root_pos = pose[:, :3].astype(np.float32)
        root_quat = pose[:, 3:7]
        root_yaw = quaternion.get_yaw(root_quat).astype(np.float32)

        sensor_linvel = self._safe_get_sensor_or_none(
            self._cfg.sensor.base_linvel,
            data,
        )

        sensor_gyro = self._safe_get_sensor_or_none(
            self._cfg.sensor.base_gyro,
            data,
        )

        if sensor_linvel is not None and sensor_linvel.shape[-1] >= 3:
            root_linvel = sensor_linvel[:, :3].astype(np.float32)
        else:
            prev_root_pos = info.get("prev_root_pos", root_pos.copy())
            if prev_root_pos.shape[0] != data.shape[0]:
                prev_root_pos = root_pos.copy()

            dt = max(float(self._cfg.ctrl_dt), 1e-6)
            root_linvel = ((root_pos - prev_root_pos) / dt).astype(np.float32)

        if sensor_gyro is not None and sensor_gyro.shape[-1] >= 3:
            gyro = sensor_gyro[:, :3].astype(np.float32)
        else:
            prev_root_yaw = info.get("prev_root_yaw", root_yaw.copy())
            if prev_root_yaw.shape[0] != data.shape[0]:
                prev_root_yaw = root_yaw.copy()

            dt = max(float(self._cfg.ctrl_dt), 1e-6)
            yaw_rate = self._wrap_to_pi(root_yaw - prev_root_yaw) / dt
            gyro = np.zeros((data.shape[0], 3), dtype=np.float32)
            gyro[:, 2] = yaw_rate.astype(np.float32)

        info["prev_root_pos"] = root_pos.copy()
        info["prev_root_yaw"] = root_yaw.copy()
        info["base_linvel"] = root_linvel.copy()
        info["base_gyro"] = gyro.copy()

        return root_pos, root_quat, root_linvel, gyro

    # --------------------------------------------------------------------------
    # RL API
    # --------------------------------------------------------------------------
    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        actions = np.asarray(actions, dtype=np.float32)
        num_envs = state.data.shape[0]

        if actions.ndim == 1:
            actions = np.tile(actions, (num_envs, 1))

        actions = np.clip(actions, -1.0, 1.0)

        if "current_actions" not in state.info:
            state.info["current_actions"] = np.zeros_like(actions)

        if state.info["current_actions"].shape != actions.shape:
            state.info["current_actions"] = np.zeros_like(actions)

        state.info["last_actions"] = state.info["current_actions"].copy()
        state.info["current_actions"] = actions.copy()

        actions_scaled = actions * self._cfg.control_config.action_scale
        state.data.actuator_ctrls = (
            self.default_angles[None, :] + actions_scaled
        ).astype(np.float32)

        return state

    def update_state(self, state: NpEnvState):
        data = state.data
        info = state.info
        num_envs = data.shape[0]

        root_pos, root_quat, base_lin_vel, gyro = self._extract_root_state(data, info)

        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles

        projected_gravity = self._compute_projected_gravity(root_quat)

        pose_commands = info["pose_commands"]
        robot_position = root_pos[:, :2]
        robot_heading = quaternion.get_yaw(root_quat)

        target_position = pose_commands[:, :2]
        target_heading = pose_commands[:, 2]

        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)

        reached_position = distance_to_target < self._cfg.commands.position_threshold

        # 大范围目标速度命令：
        #   direction = 指向目标的单位向量
        #   speed     = 0.40 ~ 1.15 m/s
        distance_safe = np.maximum(distance_to_target, 1e-6)
        target_direction_xy = position_error / distance_safe[:, None]

        target_speed = np.clip(
            distance_to_target,
            0.40,
            1.15,
        )

        target_speed = np.where(
            reached_position,
            0.0,
            target_speed,
        )

        desired_vel_xy = target_direction_xy * target_speed[:, None]

        heading_diff = self._wrap_to_pi(target_heading - robot_heading)
        reached_heading = np.abs(heading_diff) < self._cfg.commands.heading_threshold
        reached_all = np.logical_and(reached_position, reached_heading)

        desired_yaw_rate = np.clip(heading_diff * 1.0, -1.0, 1.0)
        desired_yaw_rate = np.where(
            np.abs(heading_diff) < self._cfg.commands.yaw_deadband,
            0.0,
            desired_yaw_rate,
        )

        desired_yaw_rate = np.where(reached_all, 0.0, desired_yaw_rate)
        desired_vel_xy = np.where(reached_all[:, None], 0.0, desired_vel_xy)

        velocity_commands = np.concatenate(
            [desired_vel_xy, desired_yaw_rate[:, None]],
            axis=-1,
        ).astype(np.float32)

        info["velocity_commands"] = velocity_commands
        info["desired_vel_xy"] = desired_vel_xy
        info["distance_to_target"] = distance_to_target.astype(np.float32)
        info["heading_diff"] = heading_diff.astype(np.float32)
        info["reached_all"] = reached_all

        noisy_linvel = base_lin_vel * self._cfg.normalization.lin_vel
        noisy_gyro = gyro * self._cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * self._cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * self._cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale

        position_error_normalized = position_error / 5.0
        heading_error_normalized = heading_diff / np.pi
        distance_normalized = np.clip(distance_to_target / 5.0, 0.0, 1.0)

        stop_ready = np.logical_and(reached_all, np.abs(gyro[:, 2]) < 5e-2)

        reached_flag = reached_all.astype(np.float32)
        stop_ready_flag = stop_ready.astype(np.float32)

        last_actions = info.get(
            "current_actions",
            np.zeros((num_envs, self._num_action), dtype=np.float32),
        )

        if last_actions.shape != (num_envs, self._num_action):
            last_actions = np.zeros((num_envs, self._num_action), dtype=np.float32)

        obs = np.concatenate(
            [
                noisy_linvel,
                noisy_gyro,
                projected_gravity,
                noisy_joint_angle,
                noisy_joint_vel,
                last_actions,
                command_normalized,
                position_error_normalized,
                heading_error_normalized[:, None],
                distance_normalized[:, None],
                reached_flag[:, None],
                stop_ready_flag[:, None],
            ],
            axis=-1,
        ).astype(np.float32)

        if obs.shape != (num_envs, 54):
            raise RuntimeError(f"Observation shape mismatch: {obs.shape}")

        if not np.isfinite(obs).all():
            raise RuntimeError("Observation contains NaN or Inf.")

        self._update_target_marker(data, pose_commands)
        self._update_heading_arrows(data, root_pos, desired_vel_xy, base_lin_vel[:, :2])

        reward = self._compute_reward(data, info, velocity_commands)
        terminated = self._compute_terminated(data)

        self._debug_step += 1
        self._debug_print(data, info, obs, reward, terminated)

        return state.replace(
            obs=obs,
            reward=reward,
            terminated=terminated,
        )

    def reset(self, data: mtx.SceneData) -> tuple[np.ndarray, dict]:
        cfg = self._cfg
        num_envs = data.shape[0]

        pos_range = cfg.init_state.pos_randomization_range

        robot_init_x = np.random.uniform(
            pos_range[0],
            pos_range[2],
            num_envs,
        ).astype(np.float32)

        robot_init_y = np.random.uniform(
            pos_range[1],
            pos_range[3],
            num_envs,
        ).astype(np.float32)

        robot_init_pos = np.stack([robot_init_x, robot_init_y], axis=1)

        pose_range = cfg.commands.pose_command_range

        # 全方向目标采样：极坐标。
        # radius: [target_radius_min, target_radius_max]
        # angle:  [yaw_min, yaw_max]
        radius_min = float(getattr(cfg.commands, "target_radius_min", 1.2))
        radius_max = float(getattr(cfg.commands, "target_radius_max", 5.0))

        target_radius = np.random.uniform(
            radius_min,
            radius_max,
            size=(num_envs,),
        ).astype(np.float32)

        target_angle = np.random.uniform(
            pose_range[2],
            pose_range[5],
            size=(num_envs,),
        ).astype(np.float32)

        target_offset = np.stack(
            [
                target_radius * np.cos(target_angle),
                target_radius * np.sin(target_angle),
            ],
            axis=1,
        ).astype(np.float32)

        target_positions = robot_init_pos + target_offset

        # 目标朝向指向目标点方向，便于 marker 可视化。
        target_headings = target_angle.astype(np.float32)

        pose_commands = np.concatenate(
            [target_positions, target_headings[:, None]],
            axis=1,
        ).astype(np.float32)

        init_dof_pos = np.tile(
            self._init_dof_pos,
            (*data.shape, 1),
        ).astype(np.float32)

        init_dof_vel = np.tile(
            self._init_dof_vel,
            (*data.shape, 1),
        ).astype(np.float32)

        if init_dof_pos.shape[-1] >= 7 + self._num_action:
            init_dof_pos[:, 0] = robot_init_x
            init_dof_pos[:, 1] = robot_init_y
            init_dof_pos[:, 2] = cfg.init_state.pos[2]

        data.reset(self._model)
        data.set_dof_vel(init_dof_vel)
        data.set_dof_pos(init_dof_pos, self._model)
        self._model.forward_kinematic(data)

        pose = self._body.get_pose(data)
        root_pos = pose[:, :3].astype(np.float32)
        root_quat = pose[:, 3:7]
        root_yaw = quaternion.get_yaw(root_quat).astype(np.float32)

        distance_to_target = np.linalg.norm(
            target_positions - root_pos[:, :2],
            axis=1,
        ).astype(np.float32)

        info = {
            "pose_commands": pose_commands,
            "last_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "current_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "ever_reached": np.zeros(num_envs, dtype=bool),
            "min_distance": distance_to_target.copy(),
            "prev_root_pos": root_pos.copy(),
            "prev_root_yaw": root_yaw.copy(),
            "base_linvel": np.zeros((num_envs, 3), dtype=np.float32),
            "base_gyro": np.zeros((num_envs, 3), dtype=np.float32),
        }

        self._update_target_marker(data, pose_commands)

        obs_state = NpEnvState(
            data=data,
            obs=np.zeros((num_envs, 54), dtype=np.float32),
            reward=np.zeros((num_envs,), dtype=np.float32),
            terminated=np.zeros((num_envs,), dtype=bool),
            truncated=np.zeros((num_envs,), dtype=bool),
            info=info,
        )

        obs_state = self.update_state(obs_state)

        return obs_state.obs, obs_state.info

    # --------------------------------------------------------------------------
    # Reward / termination
    # --------------------------------------------------------------------------
    def _compute_reward(
        self,
        data: mtx.SceneData,
        info: dict,
        velocity_commands: np.ndarray,
    ) -> np.ndarray:
        cfg = self._cfg
        rew_cfg = cfg.reward_config
        num_envs = data.shape[0]

        termination_penalty = np.zeros((num_envs,), dtype=np.float32)

        dof_vel = self.get_dof_vel(data)
        vel_max = np.abs(dof_vel).max(axis=1)
        vel_overflow = vel_max > cfg.max_dof_vel
        vel_extreme = (
            np.isnan(dof_vel).any(axis=1)
            | np.isinf(dof_vel).any(axis=1)
            | (vel_max > 1e6)
        )

        termination_penalty = np.where(
            vel_overflow | vel_extreme,
            rew_cfg.termination_penalty,
            termination_penalty,
        )

        if self.num_termination_check > 0:
            cquerys = self._model.get_contact_query(data)
            termination_check = cquerys.is_colliding(self.termination_contact)
            termination_check = termination_check.reshape(
                (num_envs, self.num_termination_check)
            )
            base_contact = termination_check.any(axis=1)
            termination_penalty = np.where(
                base_contact,
                rew_cfg.termination_penalty,
                termination_penalty,
            )

        pose = self._body.get_pose(data)
        root_quat = pose[:, 3:7]
        proj_g = self._compute_projected_gravity(root_quat)
        gxy = np.linalg.norm(proj_g[:, :2], axis=1)
        gz = proj_g[:, 2]
        tilt_angle = np.arctan2(gxy, np.abs(gz))
        side_flip_mask = tilt_angle > np.deg2rad(75)
        termination_penalty = np.where(
            side_flip_mask,
            rew_cfg.termination_penalty,
            termination_penalty,
        )

        base_lin_vel = info.get(
            "base_linvel",
            np.zeros((num_envs, 3), dtype=np.float32),
        )

        gyro = info.get(
            "base_gyro",
            np.zeros((num_envs, 3), dtype=np.float32),
        )

        if base_lin_vel.shape[0] != num_envs:
            base_lin_vel = np.zeros((num_envs, 3), dtype=np.float32)

        if gyro.shape[0] != num_envs:
            gyro = np.zeros((num_envs, 3), dtype=np.float32)

        lin_vel_error = np.sum(
            np.square(velocity_commands[:, :2] - base_lin_vel[:, :2]),
            axis=1,
        )

        tracking_lin_vel = np.exp(-lin_vel_error / 0.25)

        ang_vel_error = np.square(velocity_commands[:, 2] - gyro[:, 2])
        tracking_ang_vel = np.exp(-ang_vel_error / 0.25)

        distance_to_target = info["distance_to_target"]
        reached_all = info["reached_all"]

        if "ever_reached" not in info or info["ever_reached"].shape[0] != num_envs:
            info["ever_reached"] = np.zeros(num_envs, dtype=bool)

        first_time_reach = np.logical_and(reached_all, ~info["ever_reached"])
        info["ever_reached"] = np.logical_or(info["ever_reached"], reached_all)

        arrival_bonus = np.where(
            first_time_reach,
            rew_cfg.arrival_bonus,
            0.0,
        )

        if "min_distance" not in info or info["min_distance"].shape[0] != num_envs:
            info["min_distance"] = distance_to_target.copy()

        distance_improvement = info["min_distance"] - distance_to_target
        info["min_distance"] = np.minimum(info["min_distance"], distance_to_target)

        approach_reward = np.clip(
            distance_improvement * rew_cfg.approach_reward_scale,
            -1.0,
            1.0,
        )

        speed_xy = np.linalg.norm(base_lin_vel[:, :2], axis=1)

        zero_yaw_mask = np.abs(gyro[:, 2]) < 0.05

        zero_yaw_bonus = np.where(
            np.logical_and(reached_all, zero_yaw_mask),
            rew_cfg.zero_yaw_bonus,
            0.0,
        )

        stop_base = rew_cfg.stop_reward_scale * (
            0.8 * np.exp(-((speed_xy / 0.2) ** 2))
            + 1.2 * np.exp(-((np.abs(gyro[:, 2]) / 0.1) ** 4))
        )

        stop_bonus = np.where(
            reached_all,
            stop_base + zero_yaw_bonus,
            0.0,
        )

        lin_vel_z_penalty = np.square(base_lin_vel[:, 2])
        ang_vel_xy_penalty = np.sum(np.square(gyro[:, :2]), axis=1)

        actuator_ctrls = np.asarray(data.actuator_ctrls, dtype=np.float32)
        if actuator_ctrls.shape[0] != num_envs:
            actuator_ctrls = np.zeros((num_envs, self._num_action), dtype=np.float32)

        torque_penalty = np.sum(np.square(actuator_ctrls), axis=1)

        current_actions = info.get(
            "current_actions",
            np.zeros((num_envs, self._num_action), dtype=np.float32),
        )

        last_actions = info.get(
            "last_actions",
            np.zeros((num_envs, self._num_action), dtype=np.float32),
        )

        if current_actions.shape[0] != num_envs:
            current_actions = np.zeros((num_envs, self._num_action), dtype=np.float32)

        if last_actions.shape[0] != num_envs:
            last_actions = np.zeros((num_envs, self._num_action), dtype=np.float32)

        action_diff = current_actions - last_actions
        action_rate_penalty = np.sum(np.square(action_diff), axis=1)

        distance_penalty = rew_cfg.distance_penalty_scale * np.clip(
            distance_to_target,
            0.0,
            5.0,
        )

        low_speed_far = np.logical_and(
            ~reached_all,
            np.logical_and(distance_to_target > 1.0, speed_xy < 0.05),
        )

        low_speed_far_penalty = np.where(
            low_speed_far,
            rew_cfg.low_speed_far_penalty_scale,
            0.0,
        )

        no_progress = np.logical_and(
            ~reached_all,
            distance_improvement < 1e-4,
        )

        no_progress_penalty = np.where(
            no_progress,
            rew_cfg.no_progress_penalty_scale,
            0.0,
        )

        reward_not_reached = (
            rew_cfg.tracking_lin_vel_scale * tracking_lin_vel
            + rew_cfg.tracking_ang_vel_scale * tracking_ang_vel
            + approach_reward
            - distance_penalty
            - low_speed_far_penalty
            - no_progress_penalty
            - rew_cfg.lin_vel_z_penalty_scale * lin_vel_z_penalty
            - rew_cfg.ang_vel_xy_penalty_scale * ang_vel_xy_penalty
            - rew_cfg.torque_penalty_scale * torque_penalty
            - rew_cfg.action_rate_penalty_scale * action_rate_penalty
            + termination_penalty
        )

        reward_reached = (
            arrival_bonus
            + stop_bonus
            - rew_cfg.lin_vel_z_penalty_scale * lin_vel_z_penalty
            - rew_cfg.ang_vel_xy_penalty_scale * ang_vel_xy_penalty
            - rew_cfg.torque_penalty_scale * torque_penalty
            - rew_cfg.action_rate_penalty_scale * action_rate_penalty
            + termination_penalty
        )

        reward = np.where(
            reached_all,
            reward_reached,
            reward_not_reached,
        )

        return reward.astype(np.float32)

    def _compute_terminated(self, data: mtx.SceneData) -> np.ndarray:
        num_envs = data.shape[0]
        terminated = np.zeros((num_envs,), dtype=bool)

        dof_vel = self.get_dof_vel(data)
        vel_max = np.abs(dof_vel).max(axis=1)
        vel_overflow = vel_max > self._cfg.max_dof_vel
        vel_extreme = (
            np.isnan(dof_vel).any(axis=1)
            | np.isinf(dof_vel).any(axis=1)
            | (vel_max > 1e6)
        )

        terminated = np.logical_or(terminated, vel_overflow)
        terminated = np.logical_or(terminated, vel_extreme)

        if self.num_termination_check > 0:
            cquerys = self._model.get_contact_query(data)
            termination_check = cquerys.is_colliding(self.termination_contact)
            termination_check = termination_check.reshape(
                (num_envs, self.num_termination_check)
            )
            base_contact = termination_check.any(axis=1)
            terminated = np.logical_or(terminated, base_contact)

        pose = self._body.get_pose(data)
        root_quat = pose[:, 3:7]
        root_height = pose[:, 2]

        proj_g = self._compute_projected_gravity(root_quat)
        gxy = np.linalg.norm(proj_g[:, :2], axis=1)
        gz = proj_g[:, 2]
        tilt_angle = np.arctan2(gxy, np.abs(gz))

        side_flip_mask = tilt_angle > np.deg2rad(75)
        low_height_mask = root_height < 0.20

        terminated = np.logical_or(terminated, side_flip_mask)
        terminated = np.logical_or(terminated, low_height_mask)

        return terminated

    # --------------------------------------------------------------------------
    # Visualization
    # --------------------------------------------------------------------------
    def _update_target_marker(self, data: mtx.SceneData, pose_commands: np.ndarray):
        if self._target_marker_body is None:
            return

        try:
            num_envs = data.shape[0]

            marker_pos = np.column_stack(
                [
                    pose_commands[:, 0],
                    pose_commands[:, 1],
                    np.full((num_envs,), 0.5, dtype=np.float32),
                ]
            ).astype(np.float32)

            marker_quat = quaternion.from_euler(
                0,
                0,
                pose_commands[:, 2],
            ).astype(np.float32)

            mocap = self._target_marker_body.mocap
            mocap.set_pose(data, np.concatenate([marker_pos, marker_quat], axis=1))
        except Exception:
            return

    def _update_heading_arrows(
        self,
        data: mtx.SceneData,
        robot_pos: np.ndarray,
        desired_vel_xy: np.ndarray,
        base_lin_vel_xy: np.ndarray,
    ):
        arrow_height = 0.76

        robot_arrow_body = self._robot_heading_arrow_body
        desired_arrow_body = self._desired_heading_arrow_body

        if robot_arrow_body is None and desired_arrow_body is None:
            return

        robot_arrow_pos = robot_pos.copy()
        robot_arrow_pos[:, 2] = arrow_height

        if robot_arrow_body is not None:
            try:
                actual_yaw = np.where(
                    np.linalg.norm(base_lin_vel_xy, axis=1) > 1e-3,
                    np.arctan2(base_lin_vel_xy[:, 1], base_lin_vel_xy[:, 0]),
                    0.0,
                )
                actual_quat = quaternion.from_euler(0, 0, actual_yaw).astype(np.float32)
                robot_arrow_body.mocap.set_pose(
                    data,
                    np.concatenate([robot_arrow_pos, actual_quat], axis=1),
                )
            except Exception:
                pass

        if desired_arrow_body is not None:
            try:
                desired_yaw = np.where(
                    np.linalg.norm(desired_vel_xy, axis=1) > 1e-6,
                    np.arctan2(desired_vel_xy[:, 1], desired_vel_xy[:, 0]),
                    0.0,
                )
                desired_quat = quaternion.from_euler(
                    0,
                    0,
                    desired_yaw,
                ).astype(np.float32)
                desired_arrow_body.mocap.set_pose(
                    data,
                    np.concatenate([robot_arrow_pos, desired_quat], axis=1),
                )
            except Exception:
                pass

    # --------------------------------------------------------------------------
    # Debug
    # --------------------------------------------------------------------------
    def _debug_print(
        self,
        data: mtx.SceneData,
        info: dict,
        obs: np.ndarray,
        reward: np.ndarray,
        terminated: np.ndarray,
    ):
        if not self._cfg.debug.enable:
            return

        if self._cfg.debug.print_every <= 0:
            return

        if self._debug_step % self._cfg.debug.print_every != 0:
            return

        pose = self._body.get_pose(data)
        base_pos = pose[:, :3]

        distance = info.get(
            "distance_to_target",
            np.zeros((data.shape[0],), dtype=np.float32),
        )

        base_linvel = info.get(
            "base_linvel",
            np.zeros((data.shape[0], 3), dtype=np.float32),
        )

        current_actions = info.get(
            "current_actions",
            np.zeros((data.shape[0], self._num_action), dtype=np.float32),
        )

        velocity_commands = info.get(
            "velocity_commands",
            np.zeros((data.shape[0], 3), dtype=np.float32),
        )

        print(
            "\n"
            f"[ANYmalC point debug] step={self._debug_step}\n"
            f"  distance_mean={distance.mean():.3f}, "
            f"distance_min={distance.min():.3f}, distance_max={distance.max():.3f}\n"
            f"  base_xy_mean=({base_pos[:, 0].mean():.3f}, {base_pos[:, 1].mean():.3f}), "
            f"base_height_mean={base_pos[:, 2].mean():.3f}\n"
            f"  cmd_xy_abs_mean={np.abs(velocity_commands[:, :2]).mean():.4f}, "
            f"base_linvel_xy_abs_mean={np.abs(base_linvel[:, :2]).mean():.4f}\n"
            f"  action_abs_mean={np.abs(current_actions).mean():.4f}, "
            f"action_abs_max={np.abs(current_actions).max():.4f}\n"
            f"  reward_mean={reward.mean():.4f}, terminated_ratio={terminated.mean():.4f}\n"
            f"  obs finite={np.isfinite(obs).all()}, reward finite={np.isfinite(reward).all()}"
        )

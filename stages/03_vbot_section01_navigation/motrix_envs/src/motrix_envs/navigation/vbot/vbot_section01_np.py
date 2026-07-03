# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import motrixsim as mtx
import gymnasium as gym

from motrix_envs import registry
from motrix_envs.np.env import NpEnv, NpEnvState
from motrix_envs.math.quaternion import Quaternion

from .cfg import VBotSection01EnvCfg


def generate_repeating_array(num_period, num_reset, period_counter):
    """
    生成重复数组，用于在固定位置中循环选择
    num_period: 位置总数
    num_reset: 需要重置的环境数
    period_counter: 当前计数器
    """
    idx = []
    for i in range(num_reset):
        idx.append((period_counter + i) % num_period)
    return np.array(idx)


@registry.env("vbot_navigation_section01", "np")
class VBotSection01Env(NpEnv):
    """
    VBot在Section01地形上的导航任务
    继承自NpEnv，使用VBotSection01EnvCfg配置
    """
    _cfg: VBotSection01EnvCfg

    # 类常量
    OBS_DIM = 68                                    # 观测维度
    ACTION_SCALE = 0.45                             # 动作幅度缩放系数
    KP = 80.0                                       # PD 控制器位置增益
    KD = 6.0                                        # PD 控制器速度增益
    TRACKING_SIGMA = 0.2                            # 速度跟踪误差的高斯衰减系数
    FEET_AIR_TIME_TARGET = 0.45                     # 期望抬脚空中时间（秒）
    REACH_THRESHOLD = 0.35                          # waypoint 到达距离阈值
    COMMAND_SMOOTH_TAU = 0.25                       # 命令一阶低通滤波时间常数（秒）
    ROUGH_Y_RANGE = (-1.65, 1.85)                   # 崎岖区在 Y 轴的范围
    TERRAIN_SCAN_OFFSETS = np.asarray(              # 前方地形扫描距离序列（米）
        [0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.40, 1.60],
        dtype=np.float32,
    )
    GOAL_SUCCESS_WINDOW_SIZE = 200                  # 目标成功判定连续步数窗口

    def __init__(self, cfg: VBotSection01EnvCfg, num_envs: int = 1):
        # 调用父类NpEnv初始化
        super().__init__(cfg, num_envs=num_envs)
        
        # 初始化机器人body和接触
        self._body = self._model.get_body(cfg.asset.body_name)
        self._init_contact_geometry()
        
        # 获取目标标记的body
        self._target_marker_body = self._model.get_body("target_marker")
        
        # 获取箭头body（用于可视化，不影响物理）
        try:
            self._robot_arrow_body = self._model.get_body("robot_heading_arrow")
            self._desired_arrow_body = self._model.get_body("desired_heading_arrow")
        except Exception:
            self._robot_arrow_body = None
            self._desired_arrow_body = None
        
        # 动作和观测空间
        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        # 观测空间：68维
        self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(68,), dtype=np.float32)
        
        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel
        self._num_action = self._model.num_actuators
        
        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_dof_vel = np.zeros((self._model.num_dof_vel,), dtype=np.float32)
        
        # 查找target_marker的DOF索引
        self._find_target_marker_dof_indices()
        
        # 查找箭头的DOF索引
        if self._robot_arrow_body is not None and self._desired_arrow_body is not None:
            self._find_arrow_dof_indices()
        
        # 初始化缓存
        self._init_buffer()
        
        # 初始位置生成参数：从配置文件读取
        self.spawn_center = np.array(cfg.init_state.pos, dtype=np.float32)  # 从配置读取
        self.spawn_range = 0.1  # 随机生成范围：±0.1m（0.2m×0.2m区域）
    
        # 导航统计计数器
        self.navigation_stats_step = 0

        # waypoint 统一从 cfg 读取，避免 cfg.py 和环境内部不一致
        self.waypoint_targets = np.asarray(cfg.commands.waypoint_targets, dtype=np.float32)
        self.waypoint_threshold = cfg.commands.waypoint_reach_threshold
        # final_goal_threshold 已从结构中移除；使用 waypoint_reach_threshold 兼作最终目标判断
        self.final_goal_threshold = self.waypoint_threshold

        # 传感器读取保护；真实 foot sensor 不可靠时使用 proxy contact
        self._missing_contact_sensors = set()
    
    def _init_buffer(self):
        """初始化缓存和参数"""
        cfg = self._cfg
        self.default_angles = np.zeros(self._num_action, dtype=np.float32)
        
        # 归一化系数
        self.commands_scale = np.array(
            [cfg.normalization.lin_vel, cfg.normalization.lin_vel, cfg.normalization.ang_vel],
            dtype=np.float32
        )
        
        # 设置默认关节角度
        for i in range(self._model.num_actuators):
            for name, angle in cfg.init_state.default_joint_angles.items():
                if name in self._model.actuator_names[i]:
                    self.default_angles[i] = angle
        
        # 不要写全局 dof_pos 的最后 12 位。
        # scene_section01.xml 里最后一段包含 arrow freejoint quaternion，
        # 写入 default_angles 会污染 quaternion。
        # 关节角应在 reset() 中通过 self._body.set_dof_pos() 设置。
        # self._init_dof_pos[-self._num_action:] = self.default_angles
        self.action_filter_alpha = 0.3
    
    def _find_target_marker_dof_indices(self):
        """查找target_marker在dof_pos中的索引位置"""
        self._target_marker_dof_start = 0
        self._target_marker_dof_end = 3
        self._init_dof_pos[0:3] = [0.0, 0.0, 0.0]
        self._base_quat_start = 6
        self._base_quat_end = 10
    
    def _find_arrow_dof_indices(self):
        """查找箭头在dof_pos中的索引位置"""
        self._robot_arrow_dof_start = 22
        self._robot_arrow_dof_end = 29
        self._desired_arrow_dof_start = 29
        self._desired_arrow_dof_end = 36
        
        arrow_init_height = self._cfg.init_state.pos[2] + 0.5 
        if self._robot_arrow_dof_end <= len(self._init_dof_pos):
            self._init_dof_pos[self._robot_arrow_dof_start:self._robot_arrow_dof_end] = [0.0, 0.0, arrow_init_height, 0.0, 0.0, 0.0, 1.0]
        if self._desired_arrow_dof_end <= len(self._init_dof_pos):
            self._init_dof_pos[self._desired_arrow_dof_start:self._desired_arrow_dof_end] = [0.0, 0.0, arrow_init_height, 0.0, 0.0, 0.0, 1.0]
    
    def _init_contact_geometry(self):
        """初始化接触检测所需的几何体索引"""
        self._init_termination_contact()
        self._init_foot_contact()
    
    def _init_termination_contact(self):
        """初始化终止接触检测：基座geom与地面geom的碰撞"""
        termination_contact_names = self._cfg.asset.terminate_after_contacts_on
        
        # 获取所有地面geom（遍历所有geom，找到包含ground_subtree名称的）
        ground_geoms = []
        ground_prefixes = self._cfg.asset.ground_subtree
        if isinstance(ground_prefixes, str):
            ground_prefixes = (ground_prefixes,)

        for geom_name in self._model.geom_names:
            if geom_name is None:
                continue
            if any(prefix in geom_name for prefix in ground_prefixes):
                ground_geoms.append(self._model.get_geom_index(geom_name))
        
        # if len(ground_geoms) == 0:
        #     print(f"[Warning] 未找到以 '{ground_prefix}' 开头的地面geom！")
        #     self.termination_contact = np.zeros((0, 2), dtype=np.uint32)
        #     self.num_termination_check = 0
        #     return
        
        # 构建碰撞对：每个基座geom × 每个地面geom
        termination_contact_list = []
        for base_geom_name in termination_contact_names:
            try:
                base_geom_idx = self._model.get_geom_index(base_geom_name)
                for ground_idx in ground_geoms:
                    termination_contact_list.append([base_geom_idx, ground_idx])
            except Exception as e:
                print(f"[Warning] 无法找到基座geom '{base_geom_name}': {e}")
        
        if len(termination_contact_list) > 0:
            self.termination_contact = np.array(termination_contact_list, dtype=np.uint32)
            self.num_termination_check = len(termination_contact_list)
            print(f"[Info] 初始化终止接触检测: {len(termination_contact_names)}个基座geom × {len(ground_geoms)}个地面geom = {self.num_termination_check}个检测对")
        else:
            self.termination_contact = np.zeros((0, 2), dtype=np.uint32)
            self.num_termination_check = 0
            print("[Warning] 未找到任何终止接触geom，基座接触检测将被禁用！")
    
    def _init_foot_contact(self):
        self.foot_contact_check = np.zeros((0, 2), dtype=np.uint32)
        self.num_foot_check = 4  
    
    def get_dof_pos(self, data: mtx.SceneData):
        return self._body.get_joint_dof_pos(data)
    
    def get_dof_vel(self, data: mtx.SceneData):
        return self._body.get_joint_dof_vel(data)
    
    def _extract_root_state(self, data):
        """从self._body中提取根节点状态"""
        pose = self._body.get_pose(data)
        root_pos = pose[:, :3]
        root_quat = pose[:, 3:7]
        root_linvel = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data)
        return root_pos, root_quat, root_linvel
    
    @property
    def observation_space(self):
        return self._observation_space
    
    @property
    def action_space(self):
        return self._action_space
    
    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        # 保存上一步的关节速度（用于计算加速度）
        state.info["last_dof_vel"] = self.get_dof_vel(state.data)
        
        state.info["last_actions"] = state.info["current_actions"]
        
        if "filtered_actions" not in state.info:
            state.info["filtered_actions"] = actions
        else:
            state.info["filtered_actions"] = (
                self.action_filter_alpha * actions + 
                (1.0 - self.action_filter_alpha) * state.info["filtered_actions"]
            )
        
        state.info["current_actions"] = state.info["filtered_actions"]

        state.data.actuator_ctrls = self._compute_torques(state.info["filtered_actions"], state.data)
        
        return state
    
    def _compute_torques(self, actions, data):
        """计算PD控制力矩（VBot使用motor执行器，需要力矩控制）"""
        action_scaled = actions * self._cfg.control_config.action_scale
        target_pos = self.default_angles + action_scaled
        
        # 获取当前关节状态
        current_pos = self.get_dof_pos(data)  # [num_envs, 12]
        current_vel = self.get_dof_vel(data)  # [num_envs, 12]
        
        # PD控制器：tau = kp * (target - current) - kv * vel
        kp = 80.0   # 位置增益
        kv = 6.0    # 速度增益
        
        pos_error = target_pos - current_pos
        torques = kp * pos_error - kv * current_vel
        
        # 限制力矩范围（与XML中的forcerange一致）
        # hip/thigh: ±17 N·m, calf: ±34 N·m
        torque_limits = np.array([17, 17, 34] * 4, dtype=np.float32)  # FR, FL, RR, RL
        torques = np.clip(torques, -torque_limits, torque_limits)
        
        return torques
    
    def _compute_projected_gravity(self, root_quat: np.ndarray) -> np.ndarray:
        """计算机器人坐标系中的重力向量"""
        gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        gravity_vec = np.tile(gravity_vec, (root_quat.shape[0], 1))
        return Quaternion.rotate_inverse(root_quat, gravity_vec)
    
    def _get_heading_from_quat(self, quat: np.ndarray) -> np.ndarray:
        """从四元数计算yaw角（朝向）"""
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        heading = np.arctan2(siny_cosp, cosy_cosp)
        return heading

    def _wrap_to_pi(self, angle: np.ndarray) -> np.ndarray:
        """把角度归一化到 [-pi, pi]。"""
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _quat_from_yaw(yaw: np.ndarray) -> np.ndarray:
        half = 0.5 * yaw
        quat = np.zeros((yaw.shape[0], 4), dtype=np.float32)
        quat[:, 2] = np.sin(half)
        quat[:, 3] = np.cos(half)
        return quat

    @staticmethod
    def _quat_to_yaw(quat_xyzw: np.ndarray) -> np.ndarray:
        x = quat_xyzw[:, 0]
        y = quat_xyzw[:, 1]
        z = quat_xyzw[:, 2]
        w = quat_xyzw[:, 3]
        return np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        ).astype(np.float32)

    def _terrain_masks(self, root_pos: np.ndarray) -> dict[str, np.ndarray]:
        """Section01 地形分区 mask。"""
        y = root_pos[:, 1]
        tc = self._cfg.terrain_config

        return {
            "rough": (y >= tc.rough_y_min) & (y <= tc.rough_y_max),
            "drop": (y > tc.drop_y_min) & (y <= tc.drop_y_max),
            "slope": (y > tc.slope_y_min) & (y <= tc.slope_y_max),
            "true_slope": (y > tc.true_slope_y_min) & (y <= tc.true_slope_y_max),
            "platform": y >= tc.platform_y_min,
        }

    def _command_limits(self, root_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        分段速度限制。
        目的：
        1. 崎岖区允许通过，不能太慢；
        2. 崎岖出口 / 落差出口减速，避免后腿跟不上；
        3. 上坡前减速，避免坡脚前栽；
        4. 坡道上保持低速持续前进。
        """
        num_envs = root_pos.shape[0]
        cfg = self._cfg

        # 普通区速度限制（来自 cfg.commands.vel_limit）
        normal = np.asarray(cfg.commands.vel_limit, dtype=np.float32).reshape(2, 3)
        # 崎岖区速度限制（来自 cfg.commands.rough_vel_limit）
        rough_limit = np.asarray(cfg.commands.rough_vel_limit, dtype=np.float32).reshape(2, 3)

        # 初始化为普通区限制，再按分区覆盖
        low = np.tile(normal[0], (num_envs, 1))
        high = np.tile(normal[1], (num_envs, 1))

        y = root_pos[:, 1]

        # 崎岖区（ROUGH_Y_RANGE）：使用 rough limit，不能太慢否则卡住
        rough = self._rough_zone_mask(root_pos[:, :2])
        low[rough] = rough_limit[0]
        high[rough] = rough_limit[1]

        # drop_exit（1.25 < y <= 1.85）：崎岖出口/落差出口，后腿容易跟不上，vx 限 0.35
        drop_exit = (y > 1.25) & (y <= 1.85)
        high[drop_exit, 0] = np.minimum(high[drop_exit, 0], 0.35)
        high[drop_exit, 1] = np.minimum(high[drop_exit, 1], 0.16)
        low[drop_exit, 1] = np.maximum(low[drop_exit, 1], -0.16)
        high[drop_exit, 2] = np.minimum(high[drop_exit, 2], 0.55)
        low[drop_exit, 2] = np.maximum(low[drop_exit, 2], -0.55)

        # approach_slope（1.75 < y <= 2.60）：坡脚前最危险前栽区，vx 限 0.38
        approach_slope = (y > 1.75) & (y <= 2.60)
        high[approach_slope, 0] = np.minimum(high[approach_slope, 0], 0.38)
        high[approach_slope, 1] = np.minimum(high[approach_slope, 1], 0.16)
        low[approach_slope, 1] = np.maximum(low[approach_slope, 1], -0.16)
        high[approach_slope, 2] = np.minimum(high[approach_slope, 2], 0.55)
        low[approach_slope, 2] = np.maximum(low[approach_slope, 2], -0.55)

        # slope（2.60 < y <= 6.90）：坡道段，低速持续上坡，vx 限 0.50
        slope = (y > 2.60) & (y <= 6.90)
        high[slope, 0] = np.minimum(high[slope, 0], 0.50)
        high[slope, 1] = np.minimum(high[slope, 1], 0.18)
        low[slope, 1] = np.maximum(low[slope, 1], -0.18)
        high[slope, 2] = np.minimum(high[slope, 2], 0.60)
        low[slope, 2] = np.maximum(low[slope, 2], -0.60)

        return low.astype(np.float32), high.astype(np.float32)

    def _update_waypoint_command(self, root_pos, root_quat, root_vel, info):
        """
        waypoint 导航命令：
        - waypoint 是世界坐标；
        - command 是机身坐标 [vx_body, vy_body, yaw_rate]；
        - 分段限速；
        - 低通平滑；
        - 最终目标附近自动减速。
        """
        cfg = self._cfg
        num_envs = root_pos.shape[0]

        waypoint_targets = self.waypoint_targets

        waypoint_index = info.get(
            "waypoint_index",
            np.zeros(num_envs, dtype=np.int32),
        )
        waypoint_index = np.clip(waypoint_index, 0, len(waypoint_targets) - 1)

        robot_xy = root_pos[:, :2]
        target_xy = waypoint_targets[waypoint_index]

        position_error = target_xy - robot_xy
        distance_to_target = np.linalg.norm(position_error, axis=1)

        is_last_waypoint = waypoint_index >= (len(waypoint_targets) - 1)
        reached_waypoint = distance_to_target < self.waypoint_threshold

        # 防止卡在崎岖区 waypoint 附近不切换
        y = root_pos[:, 1]
        forced_reach = (
            ((waypoint_index == 1) & (y > 1.05))
            | ((waypoint_index == 2) & (y > 2.05))
        )

        reached_waypoint = reached_waypoint | forced_reach

        advance = reached_waypoint & (~is_last_waypoint)
        waypoint_index = np.where(
            advance,
            np.minimum(waypoint_index + 1, len(waypoint_targets) - 1),
            waypoint_index,
        ).astype(np.int32)

        target_xy = waypoint_targets[waypoint_index]
        position_error = target_xy - robot_xy
        distance_to_target = np.linalg.norm(position_error, axis=1)

        is_last_waypoint = waypoint_index >= (len(waypoint_targets) - 1)
        final_done = is_last_waypoint & (distance_to_target < self.final_goal_threshold)

        # 世界系误差转机身系误差
        delta_world = np.zeros((num_envs, 3), dtype=np.float32)
        delta_world[:, :2] = position_error
        delta_body = Quaternion.rotate_inverse(root_quat, delta_world)[:, :2]

        robot_heading = self._get_heading_from_quat(root_quat)
        desired_heading = np.arctan2(position_error[:, 1], position_error[:, 0])
        heading_error = self._wrap_to_pi(desired_heading - robot_heading)

        raw_command = np.zeros((num_envs, 3), dtype=np.float32)
        raw_command[:, 0] = cfg.commands.waypoint_lin_kp * delta_body[:, 0]
        raw_command[:, 1] = cfg.commands.waypoint_lin_kp * delta_body[:, 1]
        raw_command[:, 2] = cfg.commands.waypoint_ang_kp * heading_error

        low, high = self._command_limits(root_pos)
        raw_command = np.clip(raw_command, low, high).astype(np.float32)

        # 终点附近按距离减速，避免冲上平台后摔
        final_slow_zone = is_last_waypoint & (distance_to_target < 1.0)
        if np.any(final_slow_zone):
            slow_scale = np.clip(distance_to_target[final_slow_zone] / 1.0, 0.15, 1.0)
            raw_command[final_slow_zone, :2] *= slow_scale[:, None]
            raw_command[final_slow_zone, 2] *= slow_scale

        raw_command[final_done] = 0.0

        # 一阶低通平滑
        last_command = info.get("velocity_commands", np.zeros_like(raw_command))
        tau = max(float(cfg.commands.waypoint_cmd_smooth_tau), 1e-6)
        alpha = float(cfg.ctrl_dt) / (tau + float(cfg.ctrl_dt))
        velocity_commands = alpha * raw_command + (1.0 - alpha) * last_command
        velocity_commands = np.clip(velocity_commands, low, high)

        pose_commands = info["pose_commands"].copy()
        pose_commands[:, :2] = target_xy
        pose_commands[:, 2] = 0.0

        info["waypoint_index"] = waypoint_index
        info["waypoint_reached"] = advance
        info["final_done"] = final_done
        info["pose_commands"] = pose_commands
        info["velocity_commands"] = velocity_commands
        info["target_distance"] = distance_to_target
        info["target_xy"] = target_xy
        info["heading_error"] = heading_error

        return velocity_commands, pose_commands, position_error, distance_to_target, heading_error, final_done

    def _estimate_foot_positions_world(self, root_pos: np.ndarray, root_quat: np.ndarray, joint_pos: np.ndarray) -> np.ndarray:
        """
        足端位置 proxy。
        不依赖真实 foot geom / sensor，也不依赖 Quaternion.rotate。
        只用于 reward shaping，不用于物理仿真。

        这里主要使用足端相对高度 foot_z 来构造：
        - contacts proxy
        - swing_foot_height
        - per_leg_swing
        """
        num_envs = root_pos.shape[0]
        pcfg = self._cfg.contact_proxy_config

        hip_offsets_body = np.array(
            [
                [ 0.18, -0.11, 0.0],  # FR
                [ 0.18,  0.11, 0.0],  # FL
                [-0.18, -0.11, 0.0],  # RR
                [-0.18,  0.11, 0.0],  # RL
            ],
            dtype=np.float32,
        )

        thigh_l = pcfg.thigh_length
        calf_l = pcfg.calf_length

        foot_pos = np.zeros((num_envs, 4, 3), dtype=np.float32)

        for leg in range(4):
            thigh = joint_pos[:, leg * 3 + 1]
            calf = joint_pos[:, leg * 3 + 2]

            # 简化二维腿模型：重点估计足端高度
            z_rel = -(
                thigh_l * np.cos(thigh)
                + calf_l * np.cos(thigh + calf)
            )

            x_rel = (
                thigh_l * np.sin(thigh)
                + calf_l * np.sin(thigh + calf)
            )

            foot_pos[:, leg, 0] = root_pos[:, 0] + hip_offsets_body[leg, 0] + x_rel
            foot_pos[:, leg, 1] = root_pos[:, 1] + hip_offsets_body[leg, 1]
            foot_pos[:, leg, 2] = root_pos[:, 2] + hip_offsets_body[leg, 2] + z_rel

        return foot_pos.astype(np.float32)

    def _get_contacts_proxy(self, foot_pos: np.ndarray) -> np.ndarray:
        """
        以每个 env 的最低脚为接触参考。
        最低脚附近认为触地，其他脚认为摆动。
        """
        threshold = self._cfg.contact_proxy_config.contact_height_threshold
        foot_z = foot_pos[:, :, 2]
        min_z = np.min(foot_z, axis=1, keepdims=True)
        return (foot_z <= (min_z + threshold)).astype(bool)

    def _update_contact_state(self, data: mtx.SceneData, info: dict, root_pos, root_quat, joint_pos):
        """
        更新 contacts / first_contact / feet_air_time / foot_pos。
        """
        foot_pos = self._estimate_foot_positions_world(root_pos, root_quat, joint_pos)
        contacts = self._get_contacts_proxy(foot_pos)

        old_air_time = info.get(
            "feet_air_time",
            np.zeros((data.shape[0], 4), dtype=np.float32),
        ).copy()

        first_contact = np.logical_and(old_air_time > 0.0, contacts)

        info["air_time_before_contact"] = old_air_time
        info["first_contact"] = first_contact
        info["feet_air_time"] = np.where(
            contacts,
            0.0,
            old_air_time + self._cfg.ctrl_dt,
        )
        info["contacts"] = contacts
        info["foot_pos"] = foot_pos

    def _rough_zone_mask(self, base_xy: np.ndarray) -> np.ndarray:
        return np.logical_and(
            base_xy[:, 1] >= self.ROUGH_Y_RANGE[0],
            base_xy[:, 1] <= self.ROUGH_Y_RANGE[1],
        )

    def _get_terrain_scan(self, base_xy: np.ndarray) -> np.ndarray:
        sample_y = base_xy[:, 1:2] + self.TERRAIN_SCAN_OFFSETS.reshape(1, -1)
        y_min, y_max = self.ROUGH_Y_RANGE

        inside = np.logical_and(sample_y >= y_min, sample_y <= y_max)
        entry_edge = np.clip(1.0 - np.abs(sample_y - y_min) / 0.6, 0.0, 1.0)
        exit_edge = np.clip(1.0 - np.abs(sample_y - y_max) / 0.6, 0.0, 1.0)
        edge_signal = 0.5 * np.maximum(entry_edge, exit_edge)

        scan = np.where(inside, 1.0, edge_signal)
        return scan.astype(np.float32)

    def _sample_forward_terrain(self, root_pos: np.ndarray, root_quat: np.ndarray) -> np.ndarray:
        """
        8 维前方地形风险观测。
        不是精确 mesh 高度，但提供：崎岖、落差、坡道、平台接近度。
        """
        num_envs = root_pos.shape[0]
        sample_dists = np.array(
            [0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.40, 1.60],
            dtype=np.float32,
        )

        heading = self._get_heading_from_quat(root_quat)
        forward_x = np.cos(heading)[:, None]
        forward_y = np.sin(heading)[:, None]

        sample_y = root_pos[:, 1:2] + forward_y * sample_dists[None, :]
        sample_x = root_pos[:, 0:1] + forward_x * sample_dists[None, :]

        tc = self._cfg.terrain_config

        terrain = np.zeros((num_envs, 8), dtype=np.float32)

        rough = (sample_y >= tc.rough_y_min) & (sample_y <= tc.rough_y_max)
        drop = (sample_y > tc.drop_y_min) & (sample_y <= tc.drop_y_max)
        slope = (sample_y > tc.slope_y_min) & (sample_y <= tc.slope_y_max)
        platform = sample_y >= tc.platform_y_min

        # 基础阶段风险
        terrain += rough.astype(np.float32) * 0.35
        terrain += drop.astype(np.float32) * 0.70

        # 坡道随 y 增加而增加
        slope_progress = np.clip(
            (sample_y - tc.slope_y_min) / max(tc.slope_y_max - tc.slope_y_min, 1e-6),
            0.0,
            1.0,
        )
        terrain += slope.astype(np.float32) * (0.45 + 0.45 * slope_progress.astype(np.float32))

        # 平台风险降低，但提示需要减速
        terrain += platform.astype(np.float32) * 0.25

        # 横向偏离惩罚提示：离中心线越远风险越高
        lateral_risk = np.clip(np.abs(sample_x) / 1.0, 0.0, 1.0)
        terrain = np.maximum(terrain, 0.25 * lateral_risk.astype(np.float32))

        return np.clip(terrain, 0.0, 1.0).astype(np.float32)

    def _get_feet_contact_obs(self, data: mtx.SceneData) -> np.ndarray:
        """
        安全版足端接触观测，占位 12 维。

        当前 MotrixSim 版本中，直接调用 get_sensor_value(foot_name, data)
        会因为 sensor 名称或 sensor view 不匹配触发 Rust panic，
        普通 try/except 无法安全捕获。

        为了先跑通 68 维观测训练，这里先返回 4 足 × 3 维 = 12 维零向量。
        后续如果确认了正确的 foot force sensor 名称，再替换成真实接触力。
        """
        return np.zeros((data.shape[0], 12), dtype=np.float32)

    def _get_feet_proxy_obs(self, info: dict, num_envs: int) -> np.ndarray:
        """
        12维足端 proxy 观测：
        4维 contacts + 4维 foot_clearance + 4维 feet_air_time
        替代真实足端接触力，避免 sensor panic，同时给策略足端状态。
        """
        contacts = info.get(
            "contacts",
            np.zeros((num_envs, 4), dtype=bool),
        ).astype(np.float32)

        foot_pos = info.get(
            "foot_pos",
            np.zeros((num_envs, 4, 3), dtype=np.float32),
        )
        foot_z = foot_pos[:, :, 2]
        foot_clearance = foot_z - np.min(foot_z, axis=1, keepdims=True)
        foot_clearance = np.clip(foot_clearance / 0.20, 0.0, 1.0)

        feet_air_time = info.get(
            "feet_air_time",
            np.zeros((num_envs, 4), dtype=np.float32),
        )
        feet_air_time = np.clip(feet_air_time / self._cfg.reward_config.feet_air_time_target, 0.0, 1.0)

        return np.concatenate(
            [
                contacts,
                foot_clearance.astype(np.float32),
                feet_air_time.astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)

    def _safe_get_sensor_value(self, sensor_name: str, data: mtx.SceneData):
        if sensor_name in self._missing_contact_sensors:
            return None
        try:
            return self._model.get_sensor_value(sensor_name, data)
        except BaseException:
            self._missing_contact_sensors.add(sensor_name)
            return None

    @staticmethod
    def _sensor_value_to_contact(sensor_value, num_envs: int, threshold: float = 0.01) -> np.ndarray:
        if sensor_value is None:
            return np.zeros(num_envs, dtype=bool)

        value = np.asarray(sensor_value, dtype=np.float32)

        if value.ndim == 0:
            return np.full(num_envs, np.abs(float(value)) > threshold, dtype=bool)

        if value.shape[0] != num_envs:
            flat = value.reshape(-1)
            hit = np.max(np.abs(flat)) > threshold if flat.size > 0 else False
            return np.full(num_envs, hit, dtype=bool)

        if value.ndim == 1:
            return np.abs(value) > threshold

        return np.linalg.norm(value, axis=1) > threshold

    def _aggregate_contact_group(self, sensor_names: list[str], data: mtx.SceneData, threshold: float = 0.01):
        contact = np.zeros((data.shape[0],), dtype=bool)
        for sensor_name in sensor_names:
            sensor_value = self._safe_get_sensor_value(sensor_name, data)
            contact = np.logical_or(
                contact,
                self._sensor_value_to_contact(sensor_value, data.shape[0], threshold),
            )
        return contact

    def _get_foot_contacts(self, data: mtx.SceneData) -> np.ndarray:
        return np.stack(
            [self._aggregate_contact_group(group, data) for group in self._cfg.sensor.foot_contact_sensor_groups],
            axis=1,
        )

    def _sanitize_freejoint_quaternions(self, dof_pos: np.ndarray) -> np.ndarray:
        """
        修复所有已知 freejoint quaternion 块，避免 set_dof_pos 时 quaternion 非归一化 panic。
        当前 scene_section01.xml 中：
        - base quaternion: 6:10
        - robot_heading_arrow quaternion: 25:29
        - desired_heading_arrow quaternion: 32:36
        """
        quat_blocks = [
            (6, 10),
            (25, 29),
            (32, 36),
        ]

        for start, end in quat_blocks:
            if dof_pos.shape[1] < end:
                continue

            quat = dof_pos[:, start:end]
            norm = np.linalg.norm(quat, axis=1, keepdims=True)
            valid = norm[:, 0] > 1e-6

            quat_fixed = np.zeros_like(quat, dtype=np.float32)
            quat_fixed[:, 3] = 1.0
            quat_fixed[valid] = quat[valid] / norm[valid]

            dof_pos[:, start:end] = quat_fixed

        return dof_pos

    def _sensor_value_to_force_vec(self, sensor_value, num_envs: int) -> np.ndarray:
        if sensor_value is None:
            return np.zeros((num_envs, 3), dtype=np.float32)

        value = np.asarray(sensor_value, dtype=np.float32)

        if value.ndim == 0:
            return np.tile(np.array([[0.0, 0.0, float(value)]], dtype=np.float32), (num_envs, 1))

        if value.ndim == 1:
            if value.shape[0] == num_envs:
                return np.stack(
                    [np.zeros_like(value), np.zeros_like(value), value],
                    axis=1,
                ).astype(np.float32)

            if num_envs == 1 and value.shape[0] >= 3:
                return value[:3].reshape(1, 3).astype(np.float32)

            scalar = float(value.reshape(-1)[0]) if value.size > 0 else 0.0
            return np.tile(np.array([[0.0, 0.0, scalar]], dtype=np.float32), (num_envs, 1))

        if value.shape[0] != num_envs:
            flat = value.reshape(-1)
            sample = np.zeros(3, dtype=np.float32)
            if flat.size >= 3:
                sample = flat[:3].astype(np.float32)
            elif flat.size > 0:
                sample[2] = float(flat[0])
            return np.tile(sample.reshape(1, 3), (num_envs, 1))

        if value.shape[1] >= 3:
            return value[:, :3].astype(np.float32)

        if value.shape[1] == 1:
            return np.concatenate(
                [np.zeros((num_envs, 2), dtype=np.float32), value],
                axis=1,
            ).astype(np.float32)

        return np.zeros((num_envs, 3), dtype=np.float32)

    def _get_foot_contact_force(self, data: mtx.SceneData, root_quat: np.ndarray) -> np.ndarray:
        force_list = []

        for sensor_group in self._cfg.sensor.foot_contact_sensor_groups:
            group_force = np.zeros((data.shape[0], 3), dtype=np.float32)
            group_norm = np.zeros((data.shape[0],), dtype=np.float32)

            for sensor_name in sensor_group:
                force_vec = self._sensor_value_to_force_vec(
                    self._safe_get_sensor_value(sensor_name, data),
                    data.shape[0],
                )
                force_norm = np.linalg.norm(force_vec, axis=1)
                take = force_norm > group_norm

                if np.any(take):
                    group_force[take] = force_vec[take]
                    group_norm[take] = force_norm[take]

            force_list.append(group_force)

        contact_force_world = np.concatenate(force_list[:4], axis=1).astype(np.float32)
        reshaped = contact_force_world.reshape(data.shape[0], 4, 3)
        rotated = [Quaternion.rotate_inverse(root_quat, reshaped[:, i, :]) for i in range(4)]

        return np.concatenate(rotated, axis=1).astype(np.float32)

    def _update_target_marker(self, data: mtx.SceneData, pose_commands: np.ndarray):
        """更新目标位置标记的位置和朝向"""
        num_envs = data.shape[0]
        all_dof_pos = data.dof_pos.copy()
        
        for env_idx in range(num_envs):
            target_x = float(pose_commands[env_idx, 0])
            target_y = float(pose_commands[env_idx, 1])
            target_yaw = float(pose_commands[env_idx, 2])
            all_dof_pos[env_idx, self._target_marker_dof_start:self._target_marker_dof_end] = [
                target_x, target_y, target_yaw
            ]
        
        data.set_dof_pos(all_dof_pos, self._model)
        self._model.forward_kinematic(data)
    
    def _update_heading_arrows(self, data: mtx.SceneData, robot_pos: np.ndarray, desired_vel_xy: np.ndarray, base_lin_vel_xy: np.ndarray):
        """更新箭头位置（使用DOF控制freejoint，不影响物理）"""
        if self._robot_arrow_body is None or self._desired_arrow_body is None:
            return
        
        num_envs = data.shape[0]
        arrow_offset = 0.5  # 箭头相对于机器人的高度偏移
        all_dof_pos = data.dof_pos.copy()
        
        for env_idx in range(num_envs):
            # 算箭头高度 = 机器人当前高度 + 偏移
            arrow_height = robot_pos[env_idx, 2] + arrow_offset
            
            # 当前运动方向箭头
            cur_v = base_lin_vel_xy[env_idx]
            if np.linalg.norm(cur_v) > 1e-3:
                cur_yaw = np.arctan2(cur_v[1], cur_v[0])
            else:
                cur_yaw = 0.0
            robot_arrow_pos = np.array([robot_pos[env_idx, 0], robot_pos[env_idx, 1], arrow_height], dtype=np.float32)
            robot_arrow_quat = self._euler_to_quat(0, 0, cur_yaw)
            quat_norm = np.linalg.norm(robot_arrow_quat)
            if quat_norm > 1e-6:
                robot_arrow_quat = robot_arrow_quat / quat_norm
            else:
                robot_arrow_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            all_dof_pos[env_idx, self._robot_arrow_dof_start:self._robot_arrow_dof_end] = np.concatenate([
                robot_arrow_pos, robot_arrow_quat
            ])
            
            # 期望运动方向箭头
            des_v = desired_vel_xy[env_idx]
            if np.linalg.norm(des_v) > 1e-3:
                des_yaw = np.arctan2(des_v[1], des_v[0])
            else:
                des_yaw = 0.0
            desired_arrow_pos = np.array([robot_pos[env_idx, 0], robot_pos[env_idx, 1], arrow_height], dtype=np.float32)
            desired_arrow_quat = self._euler_to_quat(0, 0, des_yaw)
            quat_norm = np.linalg.norm(desired_arrow_quat)
            if quat_norm > 1e-6:
                desired_arrow_quat = desired_arrow_quat / quat_norm
            else:
                desired_arrow_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            all_dof_pos[env_idx, self._desired_arrow_dof_start:self._desired_arrow_dof_end] = np.concatenate([
                desired_arrow_pos, desired_arrow_quat
            ])
        
        data.set_dof_pos(all_dof_pos, self._model)
        self._model.forward_kinematic(data)
    
    def _euler_to_quat(self, roll, pitch, yaw):
        """欧拉角转四元数 [qx, qy, qz, qw] - Motrix格式"""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return np.array([qx, qy, qz, qw], dtype=np.float32)
    
    def update_state(self, state: NpEnvState) -> NpEnvState:
        # —— 1. 状态提取 ——
        data = state.data
        cfg = self._cfg

        root_pos, root_quat, root_vel = self._extract_root_state(data)
        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles

        # 世界系速度 → 机身系速度
        local_lin_vel = Quaternion.rotate_inverse(root_quat, root_vel)
        gyro = self._model.get_sensor_value(cfg.sensor.base_gyro, data)
        projected_gravity = self._compute_projected_gravity(root_quat)

        # —— 2. 真实足端接触检测（替代 proxy contact）——
        # _get_foot_contacts() 读取 foot force sensor
        # 更新 first_contact / air_time_before_contact / feet_air_time / contacts
        # （不再使用 _update_contact_state() 及其依赖的 contact_proxy_config）
        contacts = self._get_foot_contacts(data)

        air_time_before_contact = state.info.get(
            "feet_air_time",
            np.zeros((data.shape[0], 4), dtype=np.float32),
        ).copy()

        state.info["first_contact"] = np.logical_and(air_time_before_contact > 0.0, contacts)
        state.info["air_time_before_contact"] = air_time_before_contact
        state.info["feet_air_time"] = np.where(
            contacts,
            0.0,
            air_time_before_contact + self._cfg.ctrl_dt,
        )
        state.info["contacts"] = contacts

        # —— 3. waypoint 命令更新 ——
        # _update_waypoint_command()：
        #   - 检测 waypoint 是否到达，更新 waypoint_index
        #   - 计算机身系速度命令 velocity_commands [vx, vy, wz]
        #   - 分段限速（_command_limits）
        #   - 命令平滑（低通滤波）
        #   - 终点减速（final_slow_zone）
        velocity_commands, pose_commands, position_error, distance_to_target, heading_diff, reached_all = (
            self._update_waypoint_command(root_pos, root_quat, root_vel, state.info)
        )

        robot_position = root_pos[:, :2]
        target_position = pose_commands[:, :2]
        target_heading = pose_commands[:, 2]

        # 用于箭头显示
        desired_vel_xy = velocity_commands[:, :2]

        # —— 4. 构造观测（68维）——
        noisy_linvel = local_lin_vel * cfg.normalization.lin_vel
        noisy_gyro = gyro * cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale
        last_actions = state.info["current_actions"]

        # base_obs（48维）：noisy_linvel(机身系), noisy_gyro, projected_gravity,
        #                   noisy_joint_angle, noisy_joint_vel,
        #                   last_actions, command_normalized
        base_obs = np.concatenate(
            [
                noisy_linvel,       # 3
                noisy_gyro,         # 3
                projected_gravity,  # 3
                noisy_joint_angle,  # 12
                noisy_joint_vel,    # 12
                last_actions,       # 12
                command_normalized, # 3
            ],
            axis=-1,
        )

        # feet_contact_obs（12维）：_get_feet_proxy_obs()
        # terrain_obs（8维）：_get_terrain_scan() 前方地形风险信号
        feet_contact_obs = self._get_feet_proxy_obs(state.info, data.shape[0])
        terrain_obs = self._get_terrain_scan(root_pos[:, :2])

        obs = np.concatenate(
            [
                base_obs,
                feet_contact_obs,
                terrain_obs,
            ],
            axis=-1,
        ).astype(np.float32)

        assert obs.shape == (data.shape[0], 68), obs.shape

        # —— 5. 目标标记和箭头更新（仅可视化，不影响物理）——
        self._update_target_marker(data, pose_commands)
        local_lin_vel_xy = local_lin_vel[:, :2]
        self._update_heading_arrows(data, root_pos, desired_vel_xy, local_lin_vel_xy)

        # —— 6. 奖励计算 ——
        # _compute_reward()：详见其内部注释
        reward = self._compute_reward(data, state.info, velocity_commands)

        # —— 7. 终止条件计算 ——
        # _compute_terminated()：摔倒/超时/关节超速/速度异常等
        terminated_state = self._compute_terminated(state)
        terminated = terminated_state.terminated

        # —— 8. 更新 state ——
        state.obs = obs
        state.reward = reward
        state.terminated = terminated

        return state
    
    def _compute_base_contact(self, data: mtx.SceneData) -> np.ndarray:
        """
        安全版机身接触检测。

        当前 MotrixSim 版本中的 ContactQuery.is_colliding()
        不支持传入 geom_a, geom_b 两个参数。
        因此这里先不使用 query-based pair 检测，避免 view/train 每一步刷 warning。

        后续如果需要更精确的机身-地形接触检测，再根据 MotrixSim 当前 API 单独适配。
        """
        return np.zeros(self._num_envs, dtype=bool)


    def _compute_terminated(self, state: NpEnvState) -> NpEnvState:
        data = state.data
        info = state.info
        cfg = self._cfg

        root_pos, root_quat, root_vel = self._extract_root_state(data)
        joint_vel = self.get_dof_vel(data)
        projected_gravity = self._compute_projected_gravity(root_quat)

        final_done = info.get("final_done", np.zeros(self._num_envs, dtype=bool))

        base_low = root_pos[:, 2] < 0.18
        orientation_bad = np.linalg.norm(projected_gravity[:, :2], axis=1) > 1.15
        fall_bad = base_low | orientation_bad

        numeric_bad = (
            np.any(~np.isfinite(root_pos), axis=1)
            | np.any(~np.isfinite(root_vel), axis=1)
            | np.any(~np.isfinite(joint_vel), axis=1)
        )

        dof_vel_bad = np.any(np.abs(joint_vel) > cfg.max_dof_vel, axis=1)

        planar_speed = np.linalg.norm(root_vel[:, :2], axis=1)
        speed_bad = planar_speed > 8.0

        terminated = (
            final_done
            | base_low
            | orientation_bad
            | numeric_bad
            | dof_vel_bad
            | speed_bad
        )

        info["goal_done"] = final_done
        info["base_low"] = base_low
        info["orientation_bad"] = orientation_bad
        info["fall_bad"] = fall_bad
        info["numeric_bad"] = numeric_bad

        return state.replace(terminated=terminated)
    
    def _compute_reward(self, data: mtx.SceneData, info: dict, velocity_commands: np.ndarray) -> np.ndarray:
        # —— 状态提取 ——
        cfg = self._cfg
        scales = cfg.reward_config.scales
        num_envs = data.shape[0]

        root_pos, root_quat, root_vel = self._extract_root_state(data)
        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)
        gyro = self._model.get_sensor_value(cfg.sensor.base_gyro, data)
        projected_gravity = self._compute_projected_gravity(root_quat)

        # 世界系速度 → 机身系速度（用于 reward 计算）
        local_lin_vel = Quaternion.rotate_inverse(root_quat, root_vel)

        commands = velocity_commands
        command_xy = commands[:, :2]
        command_speed_xy = np.linalg.norm(command_xy, axis=1)
        body_speed_xy = np.linalg.norm(local_lin_vel[:, :2], axis=1)
        active_move = command_speed_xy > 0.05

        # —— 速度跟踪奖励 ——
        # tracking_lin_vel：跟踪机身坐标系线速度
        lin_vel_error = np.sum((commands[:, :2] - local_lin_vel[:, :2]) ** 2, axis=1)
        tracking_lin_vel = np.exp(-lin_vel_error / cfg.reward_config.tracking_sigma)

        # tracking_ang_vel：跟踪机身坐标系角速度
        ang_vel_error = (commands[:, 2] - gyro[:, 2]) ** 2
        tracking_ang_vel = np.exp(-ang_vel_error / cfg.reward_config.tracking_sigma)

        # forward_progress：沿命令方向前进的奖励
        cmd_dir = command_xy / np.maximum(command_speed_xy[:, None], 1e-6)
        forward_speed = np.sum(local_lin_vel[:, :2] * cmd_dir, axis=1)
        forward_progress = np.clip(forward_speed, 0.0, 1.5) * active_move

        # target_progress：靠近 waypoint 目标的奖励
        target_xy = info.get("target_xy", info["pose_commands"][:, :2])
        target_rel_world = np.zeros((num_envs, 3), dtype=np.float32)
        target_rel_world[:, :2] = target_xy - root_pos[:, :2]
        distance_to_waypoint = np.linalg.norm(target_rel_world[:, :2], axis=1)

        prev_distance = info.get("prev_distance_to_waypoint", distance_to_waypoint)
        target_progress = np.clip(prev_distance - distance_to_waypoint, -0.2, 0.2)
        info["prev_distance_to_waypoint"] = distance_to_waypoint.copy()

        # —— 目标方向跟踪奖励 ——
        # tracking_goal_vel：沿目标方向持续推进的奖励
        target_rel_body = Quaternion.rotate_inverse(root_quat, target_rel_world)[:, :2]
        target_dir_body = target_rel_body / (
            np.linalg.norm(target_rel_body, axis=1, keepdims=True) + 1e-6
        )
        tracking_goal_vel = np.sum(target_dir_body * local_lin_vel[:, :2], axis=1)
        tracking_goal_vel = np.clip(tracking_goal_vel, -1.0, 1.0) * active_move

        # tracking_yaw：朝目标 yaw 角看齐的奖励
        current_yaw = self._quat_to_yaw(root_quat)
        desired_yaw = np.arctan2(target_rel_world[:, 1], target_rel_world[:, 0]).astype(np.float32)
        tracking_yaw = np.exp(-np.abs(self._wrap_to_pi(desired_yaw - current_yaw)))

        # —— 抗停滞：有命令但身体速度不足 ——
        speed_deficit = np.clip(command_speed_xy - body_speed_xy, 0.0, None)
        anti_stall = speed_deficit * active_move

        # —— 足端接触奖励 ——
        contacts = info.get("contacts", np.zeros((num_envs, 4), dtype=bool))
        first_contact = info.get("first_contact", np.zeros((num_envs, 4), dtype=bool))
        air_time_before_contact = info.get(
            "air_time_before_contact",
            np.zeros((num_envs, 4), dtype=np.float32),
        )
        feet_air_time_reward = np.sum(
            (air_time_before_contact - cfg.reward_config.feet_air_time_target) * first_contact,
            axis=1,
        )
        feet_air_time_reward *= active_move

        # —— 后腿防交叉（保护项）：防止后腿向中线内收过多 ——
        # 关节顺序：FR(0-2), FL(3-5), RR(6-8), RL(9-11)，每条腿 hip/thigh/calf
        # RR hip index = 6, RL hip index = 9
        rr_hip = joint_pos[:, 6]
        rl_hip = joint_pos[:, 9]

        # gap 太小时认为后腿有交叉/贴近风险
        rear_hip_gap = rl_hip - rr_hip
        rear_cross_penalty = np.maximum(0.10 - rear_hip_gap, 0.0) ** 2

        # 防止后髋大幅扫腿
        rear_hip_l2 = rr_hip ** 2 + rl_hip ** 2

        # —— 后腿参与推进（保护项）：防止只用前腿走路 ——
        # RR = joint 6:9, RL = joint 9:12
        rear_joint_vel = joint_vel[:, 6:12]
        rear_motion = np.mean(np.abs(rear_joint_vel), axis=1)

        command_speed_for_rear = np.linalg.norm(velocity_commands[:, :2], axis=1)
        active_move_for_rear = command_speed_for_rear > 0.05

        # 只在有移动命令时鼓励后腿适度运动，避免原地抖腿刷分
        rear_drive_reward = (
            np.clip(rear_motion, 0.0, 2.0) / 2.0
            * active_move_for_rear.astype(np.float32)
        )

        # —— 坡道抗前栽（保护项） ——
        y = root_pos[:, 1]

        approach_slope_zone = (y > 1.75) & (y <= 2.60)
        slope_zone = (y > 2.60) & (y <= 6.90)

        # projected_gravity[:, 0] 近似对应 pitch 倾斜程度
        pitch_penalty = projected_gravity[:, 0] ** 2

        approach_pitch_penalty = pitch_penalty * approach_slope_zone.astype(np.float32)
        slope_pitch_penalty = pitch_penalty * slope_zone.astype(np.float32)

        # 坡道上鼓励低速持续前进，而不是冲坡
        slope_forward = (
            np.clip(local_lin_vel[:, 0], 0.0, 0.55)
            * slope_zone.astype(np.float32)
        )

        # —— 正则化惩罚 ——
        orientation = np.sum(projected_gravity[:, :2] ** 2, axis=1)
        lin_vel_z = local_lin_vel[:, 2] ** 2
        ang_vel_xy = np.sum(gyro[:, :2] ** 2, axis=1)
        dof_vel = np.sum(joint_vel ** 2, axis=1)

        current_actions = info.get(
            "current_actions",
            np.zeros((num_envs, self._num_action), dtype=np.float32),
        )
        last_actions = info.get(
            "last_actions",
            np.zeros_like(current_actions),
        )
        action_rate = np.sum((current_actions - last_actions) ** 2, axis=1)

        # torques：关节力矩能量惩罚
        torques = np.clip(
            np.nan_to_num(data.actuator_ctrls, nan=0.0, posinf=0.0, neginf=0.0),
            -200.0,
            200.0,
        )
        torques_term = np.sum(torques.astype(np.float64) ** 2, axis=1).astype(np.float32)

        # dof_acc：关节加速度惩罚（抑制急促抖动）
        dt = max(cfg.ctrl_dt, 1e-6)
        last_dof_vel = info.get("last_dof_vel", np.zeros_like(joint_vel))
        dof_acc = np.sum(((joint_vel - last_dof_vel) / dt) ** 2, axis=1)

        waypoint_reached = info.get("waypoint_reached", np.zeros(num_envs, dtype=bool))
        final_done = info.get("final_done", np.zeros(num_envs, dtype=bool))

        # —— reward 汇总 ——
        reward = np.zeros(num_envs, dtype=np.float32)

        reward += scales.get("tracking_lin_vel", 0.0) * tracking_lin_vel
        reward += scales.get("tracking_ang_vel", 0.0) * tracking_ang_vel
        reward += scales.get("tracking_goal_vel", 0.0) * tracking_goal_vel
        reward += scales.get("tracking_yaw", 0.0) * tracking_yaw
        reward += scales.get("forward_progress", 0.0) * forward_progress
        reward += scales.get("target_progress", 0.0) * target_progress
        reward += scales.get("reach_goal", 0.0) * waypoint_reached.astype(np.float32)
        reward += scales.get("reach_all_goal", 0.0) * final_done.astype(np.float32)

        reward += scales.get("feet_air_time", 0.0) * feet_air_time_reward
        reward += scales.get("anti_stall", 0.0) * anti_stall

        reward += -2.0 * rear_cross_penalty
        reward += -0.05 * rear_hip_l2
        reward += 0.4 * rear_drive_reward

        reward += -1.5 * approach_pitch_penalty
        reward += -1.0 * slope_pitch_penalty
        reward += 1.2 * slope_forward

        reward += scales.get("orientation", 0.0) * orientation
        reward += scales.get("lin_vel_z", 0.0) * lin_vel_z
        reward += scales.get("ang_vel_xy", 0.0) * ang_vel_xy
        reward += scales.get("torques", 0.0) * torques_term
        reward += scales.get("dof_vel", 0.0) * dof_vel
        reward += scales.get("dof_acc", 0.0) * dof_acc
        reward += scales.get("action_rate", 0.0) * action_rate

        reward += -0.01  # 时间惩罚，防止磨蹭

        # —— TensorBoard 日志 ——
        info["Reward/tracking_lin_vel"] = tracking_lin_vel
        info["Reward/tracking_ang_vel"] = tracking_ang_vel
        info["Reward/tracking_goal_vel"] = tracking_goal_vel
        info["Reward/tracking_yaw"] = tracking_yaw
        info["Reward/forward_progress"] = forward_progress
        info["Reward/target_progress"] = target_progress
        info["Reward/feet_air_time"] = feet_air_time_reward
        info["Reward/anti_stall"] = anti_stall
        info["Reward/rear_cross_penalty"] = rear_cross_penalty
        info["Reward/rear_drive"] = rear_drive_reward
        info["Reward/approach_pitch_penalty"] = approach_pitch_penalty
        info["Reward/slope_pitch_penalty"] = slope_pitch_penalty
        info["Reward/slope_forward"] = slope_forward

        info["Metrics/rear_hip_gap"] = rear_hip_gap
        info["Metrics/rr_hip"] = rr_hip
        info["Metrics/rl_hip"] = rl_hip
        info["Metrics/rear_motion"] = rear_motion

        info["Metrics/target_distance"] = distance_to_waypoint
        info["Metrics/waypoint_index"] = info.get("waypoint_index", np.zeros(num_envs))
        info["Metrics/body_speed_xy"] = body_speed_xy
        info["Metrics/command_speed_xy"] = command_speed_xy
        info["Metrics/base_height"] = root_pos[:, 2]
        info["Metrics/contact_FR"] = contacts[:, 0].astype(np.float32)
        info["Metrics/contact_FL"] = contacts[:, 1].astype(np.float32)
        info["Metrics/contact_RR"] = contacts[:, 2].astype(np.float32)
        info["Metrics/contact_RL"] = contacts[:, 3].astype(np.float32)

        return np.clip(reward, -100.0, 1000.0).astype(np.float32)

    def reset(self, data: mtx.SceneData, done: np.ndarray = None) -> tuple[np.ndarray, dict]:
        # —— 1. 随机化起始位置 ——
        cfg: VBotSection01EnvCfg = self._cfg
        num_envs = data.shape[0]

        # 在 spawn_center 周围 ±spawn_range 范围内随机生成 robot 位置
        # 高度使用配置的高度（spawn_center[2]）
        random_xy = np.random.uniform(
            low=-self.spawn_range,
            high=self.spawn_range,
            size=(num_envs, 2)
        )
        robot_init_xy = self.spawn_center[:2] + random_xy  # [num_envs, 2]
        terrain_heights = np.full(num_envs, self.spawn_center[2], dtype=np.float32)

        # 组合 XYZ 坐标
        robot_init_pos = robot_init_xy  # [num_envs, 2]
        robot_init_xyz = np.column_stack([robot_init_xy, terrain_heights])  # [num_envs, 3]

        # —— 2. 初始化 DOF 状态 ——
        dof_pos = np.tile(self._init_dof_pos, (num_envs, 1))
        dof_vel = np.tile(self._init_dof_vel, (num_envs, 1))

        # 设置 base 的 XYZ 位置（DOF 3-5）
        dof_pos[:, 3:6] = robot_init_xyz

        # 随机目标位置占位（后续被 waypoint 覆盖）
        target_offset = np.random.uniform(
            low=cfg.commands.pose_command_range[:2],
            high=cfg.commands.pose_command_range[3:5],
            size=(num_envs, 2)
        )
        target_positions = robot_init_pos + target_offset

        target_headings = np.random.uniform(
            low=cfg.commands.pose_command_range[2],
            high=cfg.commands.pose_command_range[5],
            size=(num_envs, 1)
        )

        pose_commands = np.concatenate([target_positions, target_headings], axis=1)

        # waypoint 初始化：第一个目标使用世界坐标 waypoint[0]
        waypoint_targets = np.asarray(cfg.commands.waypoint_targets, dtype=np.float32)
        waypoint_index = np.zeros(num_envs, dtype=np.int32)

        pose_commands[:, :2] = waypoint_targets[waypoint_index]
        pose_commands[:, 2] = 0.0

        # 强制初始朝向 +Y（yaw=π/2，+15%随机扰动）
        yaw = np.full((num_envs,), 0.5 * np.pi, dtype=np.float32)
        yaw += np.random.uniform(-0.15, 0.15, size=(num_envs,)).astype(np.float32)
        dof_pos[:, 6:10] = self._quat_from_yaw(yaw)

        # arrow freejoint quaternion 同上（不影响物理，仅可视化）
        marker_yaw = np.full((num_envs,), 0.5 * np.pi, dtype=np.float32)

        if dof_pos.shape[1] >= 29:
            dof_pos[:, 25:29] = self._quat_from_yaw(marker_yaw)

        if dof_pos.shape[1] >= 36:
            dof_pos[:, 32:36] = self._quat_from_yaw(marker_yaw)

        # —— 3. 设置关节角度（避免污染 quaternion）——
        data.reset(self._model)
        data.set_dof_vel(dof_vel)

        dof_pos = self._sanitize_freejoint_quaternions(dof_pos)

        data.set_dof_pos(dof_pos, self._model)

        # 通过 body API 设置关节角（include_floatingbase=False）
        # 而非直接写入 dof_pos 数组，避免 arrow quaternion 被覆盖
        # 添加 ±0.03 rad 随机关节噪声
        joint_noise = np.random.uniform(
            low=-0.03,
            high=0.03,
            size=(num_envs, self._num_action),
        ).astype(np.float32)

        joint_dof_pos = self.default_angles + joint_noise
        joint_dof_vel = np.zeros((num_envs, self._num_action), dtype=np.float32)

        self._body.set_dof_pos(data, joint_dof_pos, include_floatingbase=False)
        self._body.set_dof_vel(data, joint_dof_vel, include_floatingbase=False)

        self._model.forward_kinematic(data)

        # 更新目标位置标记（可视化了不影响物理）
        self._update_target_marker(data, pose_commands)

        # —— 4. 状态提取 ——
        root_pos, root_quat, root_vel = self._extract_root_state(data)

        joint_pos = self.get_dof_pos(data)
        joint_vel = self.get_dof_vel(data)
        joint_pos_rel = joint_pos - self.default_angles

        # 世界系速度 → 机身系速度
        local_lin_vel = Quaternion.rotate_inverse(root_quat, root_vel)
        gyro = self._model.get_sensor_value(self._cfg.sensor.base_gyro, data)
        projected_gravity = self._compute_projected_gravity(root_quat)

        # —— 5. 初始化 waypoint 命令 ——
        # 调用 _update_waypoint_command() 产生 velocity_commands
        # 保证 reset obs 和 update_state obs 一致
        (velocity_commands, pose_commands, position_error,
         distance_to_target, heading_diff, reached_all) = self._update_waypoint_command(
            root_pos, root_quat, root_vel, {
                "waypoint_index": waypoint_index,
                "pose_commands": pose_commands,
                "velocity_commands": np.zeros((num_envs, 3), dtype=np.float32),
            }
        )

        # —— 6. 构造观测（68维）——
        noisy_linvel = local_lin_vel * self._cfg.normalization.lin_vel
        noisy_gyro = gyro * self._cfg.normalization.ang_vel
        noisy_joint_angle = joint_pos_rel * self._cfg.normalization.dof_pos
        noisy_joint_vel = joint_vel * self._cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale
        last_actions = np.zeros((num_envs, self._num_action), dtype=np.float32)

        # base_obs（48维）：noisy_linvel, noisy_gyro, projected_gravity,
        #                   noisy_joint_angle, noisy_joint_vel,
        #                   last_actions, command_normalized
        base_obs = np.concatenate(
            [
                noisy_linvel,       # 3
                noisy_gyro,         # 3
                projected_gravity,  # 3
                noisy_joint_angle,  # 12
                noisy_joint_vel,    # 12
                last_actions,       # 12
                command_normalized, # 3
            ],
            axis=-1,
        )

        # feet_contact_obs（12维）：全零占位（真实 foot sensor 不稳定）
        feet_contact_obs = np.zeros((num_envs, 12), dtype=np.float32)
        # terrain_obs（8维）：_get_terrain_scan() 前方地形风险信号
        terrain_obs = self._get_terrain_scan(root_pos[:, :2])

        obs = np.concatenate(
            [
                base_obs,
                feet_contact_obs,
                terrain_obs,
            ],
            axis=-1,
        ).astype(np.float32)

        assert obs.shape == (num_envs, 68), obs.shape

        # —— 7. info 字典初始化 ——
        # waypoint 状态、接触状态、anti-stall 状态等全部清零
        # prev_distance_to_waypoint 填入初始距离
        info = {
            "pose_commands": pose_commands,
            "last_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "steps": np.zeros(num_envs, dtype=np.int32),
            "current_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "filtered_actions": np.zeros((num_envs, self._num_action), dtype=np.float32),
            "ever_reached": np.zeros(num_envs, dtype=bool),
            "prev_distance_to_waypoint": distance_to_target.copy(),
            "last_distance": distance_to_target.copy(),
            "waypoint_index": waypoint_index,
            "waypoint_reached": np.zeros(num_envs, dtype=bool),
            "final_done": np.zeros(num_envs, dtype=bool),
            "velocity_commands": velocity_commands.copy(),
            "target_xy": pose_commands[:, :2].copy(),
            "target_distance": distance_to_target.copy(),
            # 足端接触状态
            "contacts": np.zeros((num_envs, 4), dtype=bool),
            "first_contact": np.zeros((num_envs, 4), dtype=bool),
            "feet_air_time": np.zeros((num_envs, 4), dtype=np.float32),
            "air_time_before_contact": np.zeros((num_envs, 4), dtype=np.float32),
            "foot_pos": np.zeros((num_envs, 4, 3), dtype=np.float32),
            "heading_error": np.zeros(num_envs, dtype=np.float32),
            # anti-stall 状态
            "last_y": root_pos[:, 1].copy(),
            "stall_counter": np.zeros(num_envs, dtype=np.int32),
        }

        return obs, info
    

# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0.
# ==============================================================================

from dataclasses import dataclass

from motrix_rl.registry import rlcfg
from motrix_rl.rslrl.cfg import RslrlCfg
from motrix_rl.skrl.config import SkrlCfg

class skrl:
    @rlcfg("anymal_c_navigation_point")
    @dataclass
    class MyAnymalPointPPO(SkrlCfg):
        """Custom ANYmal C point-navigation SKRL PPO configuration.

        This config keeps the custom task name:
            anymal_c_navigation_point

        Compared with the official anymal_c_navigation_flat config:
            official: 2048 envs, 48 rollouts, 32 mini-batches
            custom:    256 envs, 48 rollouts,  4 mini-batches

        Both keep approximately the same mini-batch size:
            2048 * 48 / 32 = 3072
             256 * 48 /  4 = 3072

        This is much more stable than the earlier 64-env config:
             64 * 48 /  8 = 384
        """

        def __post_init__(self):
            # Practical middle ground:
            # - lighter than official 2048 envs
            # - much stronger than the earlier 64 envs
            # - mini-batch size aligned with the official setting
            self.num_envs = 1024
            self.play_num_envs = 16

            runner = self.runner
            models = runner.models
            agent = runner.agent
            trainer = runner.trainer

            # ===== Basic Training Parameters =====
            runner.seed = 42

            # ===== Network Architecture =====
            models.policy.hiddens = [256, 128, 64]
            models.value.hiddens = [256, 128, 64]

            # ===== PPO Core Parameters =====
            agent.rollouts = 48
            agent.learning_epochs = 6

            # 256 envs * 48 rollouts = 12288 samples/update
            # 12288 / 4 = 3072 samples/mini-batch
            agent.mini_batches = 16

            agent.learning_rate = 3e-4
            agent.discount_factor = 0.99
            agent.lam = 0.95
            agent.grad_norm_clip = 1.0

            # ===== PPO Clipping Parameters =====
            agent.ratio_clip = 0.2
            agent.value_clip = 0.2
            agent.clip_predicted_values = True

            # ===== Training Budget =====
            # The previous 64-env + 48000 setting was too small for learning locomotion.
            # This is a first serious training run for the custom point-navigation task.
            trainer.timesteps = 48000

class rslrl:
    @rlcfg("anymal_c_navigation_point")
    @dataclass
    class MyAnymalPointPpoRslrl(RslrlCfg):
        """Custom ANYmal C point-navigation RSLRL PPO configuration."""

        def __post_init__(self):
            self.num_envs = 256
            self.play_num_envs =4

            runner = self.runner
            algo = runner.algorithm

            # ===== Basic Training Parameters =====
            runner.seed = 42
            runner.num_steps_per_env = 48
            runner.experiment_name = "anymal_c_navigation_point"

            # A moderate first RSLRL run.
            # Official config uses 1017 iterations with 2048 envs.
            # Here we keep it smaller for local testing.
            runner.max_iterations = 500

            # ===== Network Architecture =====
            runner.actor.hidden_dims = [256, 128, 64]
            runner.critic.hidden_dims = [256, 128, 64]

            # ===== Algorithm Parameters =====
            algo.learning_rate = 3e-4
            algo.num_learning_epochs = 6

            # 256 * 48 / 4 = 3072 samples/mini-batch
            algo.num_mini_batches = 4

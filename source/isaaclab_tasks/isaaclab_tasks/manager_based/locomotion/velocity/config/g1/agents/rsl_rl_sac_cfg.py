# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOffPolicyRunnerCfg,
    RslRlSacActorModelCfg,
    RslRlSacAlgorithmCfg,
    RslRlSacCriticModelCfg,
)


@configclass
class G1RoughSACRunnerCfg(RslRlOffPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    log_interval = 1
    start_training = 1
    experiment_name = "g1_rough"
    actor = RslRlSacActorModelCfg(
        hidden_dims=[1024, 512, 256],
        activation="swish",
        obs_normalization=True,
        layer_norm=False,
        init_noise_std=0.15,
        log_std_min=-20.0,
        log_std_max=2.0,
    )
    critic = RslRlSacCriticModelCfg(
        hidden_dims=[1024, 512, 256],
        activation="swish",
        obs_normalization=True,
        layer_norm=True,
    )
    algorithm = RslRlSacAlgorithmCfg(
        replay_buffer_size=int(5.0e6),
        num_learning_epochs=1,
        num_mini_batches=12,       # UTD ≈ 1: matches paper's per-step update rate
        mini_batch_size=8192,
        actor_learning_rate=3.0e-4,
        critic_learning_rate=3.0e-4,
        alpha_learning_rate=3.0e-4,
        gamma=0.97,
        tau=0.005,
        alpha=0.01,
        auto_alpha=True,
        target_entropy_scale=1.0,  # paper default: target = -action_dim = -23
        max_grad_norm=1.0,
        policy_frequency=1,
        n_steps=1,
    )


@configclass
class G1FlatSACRunnerCfg(G1RoughSACRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 600
        self.experiment_name = "g1_flat"
        self.actor.hidden_dims = [512, 256, 128]
        self.critic.hidden_dims = [512, 256, 128]
        self.algorithm.gamma = 0.96

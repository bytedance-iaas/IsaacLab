# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlFlashSacRunnerCfg


@configclass
class G1RoughFlashSACRunnerCfg(RslRlFlashSacRunnerCfg):
    """FlashSAC configuration for the rough G1 task.

    The algorithm defaults are the FlashSAC Isaac Lab benchmark settings (2 gradient steps of 2048 transitions per
    environment step, so 48 per 24-step iteration). The replay buffer is halved from 10M to 5M transitions because
    rsl_rl stores both observations and next observations (310-dim on rough terrain).
    """

    num_steps_per_env = 24
    max_iterations = 800
    save_interval = 100
    log_interval = 1
    start_training = 1
    experiment_name = "g1_rough"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.replay_buffer_size = int(5.0e6)


@configclass
class G1FlatFlashSACRunnerCfg(RslRlFlashSacRunnerCfg):
    """FlashSAC configuration for the flat G1 task (FlashSAC Isaac Lab benchmark settings)."""

    num_steps_per_env = 24
    max_iterations = 800
    save_interval = 100
    log_interval = 1
    start_training = 1
    experiment_name = "g1_flat"

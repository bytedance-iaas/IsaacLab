# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SO-101 pick-and-place: put a red cube into a bin standing on the table.

The scene was authored in the Isaac Sim GUI and translated into config here;
poses and dimensions come from measuring the real assets. See pnp_env_cfg.

Registration is picked up automatically: tasks/__init__.py runs import_packages
over this directory, so adding this package is enough to expose the task ids.
"""

import gymnasium as gym

##
# Register Gym environments.
##

# Reuses the lift task's PPO runner config -- the network and hyperparameters
# carry over unchanged, only the scene and the goal differ.
_AGENT = "isaac_so_arm101.tasks.lift.agents.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg"

gym.register(
    id="Isaac-SO-ARM101-PnP-Bin-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pnp_env_cfg:SoArm101PnPEnvCfg",
        "rsl_rl_cfg_entry_point": _AGENT,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-SO-ARM101-PnP-Bin-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pnp_env_cfg:SoArm101PnPEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": _AGENT,
    },
    disable_env_checker=True,
)

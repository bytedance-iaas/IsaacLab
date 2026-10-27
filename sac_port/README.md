# SAC and FlashSAC for rsl-rl-lib 5.4.1

Upstream rsl-rl-lib has no SAC. `rsl_rl_sac.patch` adds the SAC algorithm, actor/critic models, replay buffer and `OffPolicyRunner`, ported from `release_v1`'s `rsl_rl_port`. `rsl_rl_flashsac.patch`, applied on top, adds FlashSAC ([paper](https://arxiv.org/abs/2604.04539), [code](https://github.com/Holiday-Robot/FlashSAC), MIT, see `LICENSE-FlashSAC`), which reuses the replay buffer and `OffPolicyRunner`.

```
pip install rsl-rl-lib==5.4.1
./sac_port/apply.sh
```

`apply.sh` skips a patch that is already applied, so it also adds FlashSAC to an environment that only has SAC.

Train with `--agent rsl_rl_sac_cfg_entry_point` (SAC) or `--agent rsl_rl_flashsac_cfg_entry_point` (FlashSAC) on `Isaac-Velocity-Flat-G1` / `Isaac-Velocity-Rough-G1`.

FlashSAC differs from the reference implementation in these places:

- No torch.compile, mixed precision or fused kernels, so it runs on CPU, CUDA and NPU devices alike.
- The noise repeat length is drawn per environment instead of once for all environments, so env shards
  (`remote_env/async_runner.py`) see the same exploration as a single env.
- Collection starts with the initial policy instead of uniform random actions.
- Updates run once per 24-step rollout (48 gradient steps by default) instead of 2 steps after every environment step.

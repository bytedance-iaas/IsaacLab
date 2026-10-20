# SAC for rsl-rl-lib 5.4.1

Upstream rsl-rl-lib has no SAC. `rsl_rl_sac.patch` adds the SAC algorithm, actor/critic models, replay buffer and `OffPolicyRunner`, ported from `release_v1`'s `rsl_rl_port`.

```
pip install rsl-rl-lib==5.4.1
./sac_port/apply.sh
```

Train with `--agent rsl_rl_sac_cfg_entry_point` on `Isaac-Velocity-Flat-G1` / `Isaac-Velocity-Rough-G1`.

# training-service

A config-driven training layer for Isaac Lab. The user (or the frontend) fills in a single YAML
request — or a form that produces one — and never touches the command line or Python.

**Task-agnostic**: `launcher.py` and `schema.py` are generic; each task contributes one
`profiles/<task>.yaml`. Adding a task needs no code change.

## Layout

```
training-service/
├── schema.py            # request schema + validation (common base + task-specific section)
├── launcher.py          # read request -> validate -> build Hydra command -> run (supports --dry-run)
├── profiles/            # one per task: task-id mapping, tunable knobs, presets, DR switches
│   ├── reach.yaml
│   └── lift.yaml
└── example_request.yaml # example request (what the frontend ultimately produces)
```

## Two-layer design

- **Common base** (identical for every task): `model` / `task` / `training_budget`
  (quick/standard/thorough) / `sim2real_robustness` / `record_video` / `seed` / `output_name` /
  `behavior`
- **Task-specific section** (described by the profile): `goal` — spatial ranges such as the reach
  goal area, or the lift object and goal areas

## Usage

```bash
# inside the image, at /workspace/isaaclab/training-service
python launcher.py --config request.yaml            # validate and start training
python launcher.py --config request.yaml --dry-run  # print the command only
```

A backend can also call `launcher.build_command(request_dict)` directly, or validate first with
`schema.validate(req)`, which raises `ValidationError` with a user-readable message.

## Knobs to Hydra overrides

Every knob resolves to a Hydra override of an existing config value, which is why new tasks need no
code:

- goal ranges → `env.commands.*.ranges.*` / `env.events.reset_*.params.pose_range.*`
- behavior preset → `env.rewards.*.weight`
- training scale → `--num_envs` / `--max_iterations` (mapped from `training_budget`)
- sim2real off → rewrite the DR event parameters to no-ops (see `dr_off_overrides` in each profile)

## sim2real robustness (physics domain randomization)

DR events are built into the soarm101 reach and lift `EventCfg` (actuator gains and joint friction;
lift additionally randomizes object mass and friction), modeled on isaaclab
`manipulation/deploy/reach`. They are active when `sim2real_robustness: true`; when `false`, the
launcher neutralizes them via `dr_off_overrides`. Tasks such as locomotion ship their own DR.

## Adding a task

1. Confirm the task is manager-based — reward and DR tuning require it; direct tasks only support
   the common base.
2. Add `profiles/<task>.yaml` with `task_ids`, `goal_zones`, `behavior_presets`, and optionally
   `dr_off_overrides`.
3. If the task has no physics DR, add randomization events to its `EventCfg` (see `deploy/reach` or
   the velocity tasks for reference).

## Calibrating the budget tiers (TODO)

The `num_envs` / `max_iterations` values in `schema.py:BUDGET_PRESETS` are estimates. They should be
calibrated against measured wall-clock time on the target GPU so that quick/standard/thorough map to
a duration that can be promised to users.

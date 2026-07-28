# training-service

A config-driven training layer for Isaac Lab. The user, or the frontend on their behalf, fills in a
single YAML request and never touches the command line or Python.

**Robot- and task-agnostic**: adding a robot or a task only touches `profiles/`; `schema.py` and
`launcher.py` stay unchanged.

## Layout

```
training-service/
├── schema.py            # request schema + validation
├── launcher.py          # read request -> validate -> build Hydra command -> run (supports --dry-run)
├── profiles/
│   ├── robots.yaml      # robot catalog: robot -> supported tasks (task ids) + DR capability (true/false/builtin)
│   └── tasks/           # task-type profiles (robot-agnostic)
│       ├── reach.yaml   #   goal_zones / behavior_presets / dr_off_overrides
│       ├── lift.yaml
│       └── velocity.yaml
└── example_request.yaml # example request (what the frontend produces)
```

- **robots.yaml** answers "which robot can do which tasks, and does it have domain randomization".
- **tasks/&lt;task&gt;.yaml** answers "what can be tuned for this task" — spatial ranges, behavior
  presets, and how to neutralize DR.
- Robots and tasks are decoupled: one reach profile serves SO-ARM, Franka and UR; only the task id
  differs.

## Currently supported

| Robot | Tasks | DR | Status |
|-------|-------|----|--------|
| SO-ARM100 / 101 | reach, lift | switchable (added by us) | ✅ training verified |
| Franka Panda | reach, lift | none | reach verified; lift is structurally identical |
| UR10 | reach | none | same structure as Franka reach |
| Anymal-C | velocity (locomotion) | built in | ✅ verified |

## Usage

```bash
# inside the image, at /workspace/isaaclab/training-service
../isaaclab.sh -p launcher.py --config request.yaml            # validate and start training
../isaaclab.sh -p launcher.py --config request.yaml --dry-run  # print the command only
```

A backend can also call `launcher.build_command(req)` directly, or validate first with
`schema.validate(req)`, which raises `ValidationError` carrying a user-readable message.

## Worked example: launching a training run

Say the goal is a precise SO-ARM101 pick-and-place policy, with real-robot robustness on, at the
standard budget.

**Step 1 — get the request YAML.** Either configure it in the console and copy the YAML, or write it
by hand:

```yaml
# my_pick.yaml
robot: so101
task: lift
training_budget: standard      # quick | standard | thorough
behavior: precise              # balanced | precise | smooth
sim2real_robustness: true
seed: 42
output_name: pick_v1
goal:
  object_start_zone: {x: [-0.08, 0.08], y: [-0.15, 0.15]}
  target_zone:       {x: [0.15, 0.30], y: [-0.15, 0.15], z: [0.10, 0.25]}
```

**Step 2 — start training:**

```bash
kubectl exec -it isaaclab -n default -- bash
cd /workspace/isaaclab/training-service
../isaaclab.sh -p launcher.py --config my_pick.yaml            # run it
# ../isaaclab.sh -p launcher.py --config my_pick.yaml --dry-run  # inspect the command first
```

The launcher validates the request (for example, whether the object range leaves the workspace),
looks up `robots.yaml` and `tasks/lift.yaml`, assembles the full command — task id
`Isaac-SO-ARM101-Lift-Cube-v0`, `--num_envs 4096`,
`env.rewards.object_goal_tracking_fine_grained.weight=10.0`, the range overrides — and runs it. **The
user never sees a task id, a Hydra path, or a reward term name.**

**Step 3 — artifacts:** checkpoints land in `logs/rsl_rl/lift/<timestamp>/model_*.pt`.

### With and without training-service

| | Without | With |
|---|---|---|
| What the user writes | a long command carrying task ids, Hydra paths and reward names | one plain-language YAML |
| On mistakes | a wrong path silently trains the wrong thing | out-of-range values and unsupported combinations are **rejected up front** with an explanation |
| Switching robots | edit many parameters | edit one line, `robot:` |

> Note: this runs inside the pod, so training is tied to the exec session (foreground). With the API
> in front of it, a run can be started from the browser as a background K8s Job.

## DR / sim2real behavior

- `dr: true` (SO-ARM) — the sim2real switch is effective; when turned off the launcher neutralizes
  the DR terms using the task profile's `dr_off_overrides`.
- `dr: false` (Franka, UR) — the task has no DR wired up, so the switch is ignored and no
  `dr_off` overrides are emitted.
- `dr: builtin` (Anymal velocity) — the task ships its own DR and it is always on.

## Adding a robot or a task

- **Add a robot**: another entry in `robots.yaml` (name / group / dr / tasks).
- **Add a task type**: another profile under `tasks/` (goal_zones / behavior_presets /
  dr_off_overrides). If the task is manager-based and needs a sim2real switch, also add
  randomization events to its `EventCfg` — see `deploy/reach` or the velocity tasks for reference.

## Calibrating the budget tiers (TODO)

The `num_envs` / `max_iterations` values in `schema.py:BUDGET_PRESETS` are estimates. They should be
calibrated against measured wall-clock time on the target GPU.

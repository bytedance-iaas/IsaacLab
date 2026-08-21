# training-service

Scripts that run alongside training: publishing the live view, turning a finished run into a
readable report, archiving what it produced, and measuring an asset before you write a task around
it. Each is a standalone command; nothing here orchestrates anything.

```
training-service/
├── livekit_stream.py   # --stream on train.py publishes the training view here
├── make_report.py      # finished run -> report.md (all hyperparameters + derived metrics)
├── upload_tos.py       # upload a run directory to TOS, preserving structure
└── inspect_usd.py      # measure a USD and check whether it carries physics
```

## livekit_stream.py

Not run by hand. `train.py --stream` imports it, adds a camera that follows the robots, and
publishes one video track to LiveKit for the browser viewer (`viewer/`). Credentials and the
server address come from the environment, which the Helm chart fills in; without them training
runs normally and simply publishes nothing.

## make_report.py

```bash
./isaaclab.sh -p training-service/make_report.py --latest
```

Reads the TensorBoard log and the saved `params/` of a run and writes `report.md` into the run
directory: every hyperparameter that was actually in effect, plus derived quantities (reward per
step, update-to-data ratio, the trend over the last stretch). It states measurements and draws no
conclusions, which is deliberate — the report is meant to be pasted whole into an AI assistant and
asked what looks wrong.

`--latest` picks the most recent run, so there is no timestamp to copy.

## upload_tos.py

```bash
./isaaclab.sh -p training-service/upload_tos.py --local-dir <run 目录> --prefix <前缀>
```

Walks the directory and uploads it to TOS with the structure intact, over the internal endpoint so
it costs no public traffic. Credentials come from the pod environment; without them it prints one
line and skips rather than failing the job.

⚠️ `--local-dir` is not optional in practice: without it the default is the whole `logs/` tree,
which means uploading every run this pod has ever done.

## inspect_usd.py

```bash
_isaac_sim/kit/python/bin/python3 training-service/inspect_usd.py <path.usd>
```

One second, no Isaac Sim. Reports the measured bounding box, whether the asset carries a rigid
body, collision geometry and mass, and whether the scale looks like it is in the wrong unit.

Useful when writing a task by hand: the bounding box is where the poses and the geometry-derived
reward thresholds come from. In the lift task, for instance, `minimal_height: 0.025` follows from
a 3 cm cube resting with its centre at 1.5 cm — swap the object and that number no longer means
"lifted", so the measurement has to come first.

Runs against a standalone usd-core in `/opt/usd-core`, kept out of the training environment's own
pxr.

# api — training console backend

Makes the frontend's "start training" button actually launch training on a cluster GPU as a
Kubernetes Job, with checkpoints uploaded to TOS. Validation and command building are reused from
`../training-service`, which stays the single source of truth.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/catalog` | Robot catalog, task profiles and budget tiers (the frontend renders from this) |
| POST | `/api/validate` | Validate a request and return the command that would run (dry-run) |
| POST | `/api/train` | Validate, create a ConfigMap and a GPU Job, return the job name |
| GET | `/api/jobs/{name}` | Job status and recent logs |
| GET | `/` | The frontend page |

## Running the MVP locally

```bash
cd api
python3 -m pip install -r requirements.txt
# a local kubeconfig with cluster access is required for /train to create jobs
uvicorn server:app --port 8000
# then open http://localhost:8000
```

- `/api/catalog` and `/api/validate` are read-only, never touch the cluster, and work immediately.
- `/api/train` creates the job in the cluster using the local kubeconfig.

## Environment variables

| Variable | Default | Notes |
|----------|---------|-------|
| `TRAIN_IMAGE` | last verified image tag | **Point this at the new tag once CI publishes an image with the TOS SDK.** |
| `K8S_NAMESPACE` | default | Namespace the job is created in |
| `TOS_SECRET` | tos-creds | Name of the TOS credential Secret mounted into the job (envFrom, optional) |

## TOS credential Secret (required for checkpoint upload)

Create it from an IAM sub-account key, not the root account:

```bash
kubectl create secret generic tos-creds -n default \
  --from-literal=TOS_ACCESS_KEY=<your-AK> \
  --from-literal=TOS_SECRET_KEY=<your-SK> \
  --from-literal=TOS_BUCKET=isaaclab-ckpt-test \
  --from-literal=TOS_ENDPOINT=tos-cn-beijing.ivolces.com \
  --from-literal=TOS_REGION=cn-beijing
```

Training works without the Secret: `upload_tos.py` skips the upload when no credentials are present,
and the artifacts stay in the container.

## What the job does

The job created by the API requests one GPU and lands on a free node, then:

1. mounts the request ConfigMap at `/config/request.yaml`
2. runs `launcher.py --config /config/request.yaml` to train
3. runs `upload_tos.py --prefix <output_name>` to push `logs/` to TOS

## Moving into the cluster (later)

Package this directory plus `training-service` into an image (python:slim base) and run it
in-cluster. `server.py` already prefers `load_incluster_config()`; give it a ServiceAccount with RBAC
to create Jobs and ConfigMaps and to read pod logs.

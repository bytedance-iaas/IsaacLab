"""Backend API for the training console.

Endpoints:
  - GET  /api/catalog          -> robot catalog + task profiles + budget tiers (single source of truth for the frontend)
  - POST /api/validate         -> validate a request and return the command that would run (launcher dry-run)
  - POST /api/train            -> validate, create a ConfigMap and a GPU K8s Job, return the job name
  - GET  /api/jobs/{name}      -> job status and recent logs
  - GET  /                     -> the frontend page, served same-origin to avoid CORS

MVP deployment: run uvicorn locally and create jobs with the local kubeconfig. It can later be
containerized and run in-cluster (switching to the in-cluster config). Validation and command
building are reused from training-service so there is one source of truth.
"""
from __future__ import annotations

import os
import sys
import uuid

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

# Reuse training-service's validation and command-building logic
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "training-service"))
import schema  # noqa: E402
import launcher  # noqa: E402

NAMESPACE = os.environ.get("K8S_NAMESPACE", "default")
# Training image: point this environment variable at a new CI tag; defaults to the last verified image
TRAIN_IMAGE = os.environ.get(
    "TRAIN_IMAGE",
    "iaas-us-cn-beijing.cr.volces.com/physicalai/isaaclab:abd9ed344b3e44108a62d0e4dfd6ff0d",
)
# Name of the TOS credential Secret mounted into the job (envFrom); without it training still runs, it just skips the upload
TOS_SECRET = os.environ.get("TOS_SECRET", "tos-creds")

app = FastAPI(title="Robot Training Console API")


# ---------------- Kubernetes ----------------
def _k8s():
    from kubernetes import client, config

    try:
        config.load_incluster_config()  # when running inside the cluster
    except Exception:
        config.load_kube_config()  # local development
    return client.BatchV1Api(), client.CoreV1Api(), client


def _job_manifest(k8s, name: str, req: dict) -> "object":
    """Build the training job: mount the request ConfigMap, request one GPU, run the launcher and
    then upload to TOS."""
    prefix = req.get("output_name") or name
    # The report is a convenience, so it must not be able to cost the user their checkpoint: it runs
    # under `|| true` so that a reporting bug still leaves the upload to run.
    cmd = (
        "set -e; cd /workspace/isaaclab/training-service; "
        "../isaaclab.sh -p launcher.py --config /config/request.yaml; "
        "../isaaclab.sh -p make_report.py --latest || echo '[report] skipped'; "
        f"../isaaclab.sh -p upload_tos.py --prefix {prefix}"
    )
    container = k8s.V1Container(
        name="train",
        image=TRAIN_IMAGE,
        command=["/bin/bash", "-lc", cmd],
        resources=k8s.V1ResourceRequirements(limits={"nvidia.com/gpu": "1"}),
        volume_mounts=[
            k8s.V1VolumeMount(name="cfg", mount_path="/config"),
            k8s.V1VolumeMount(name="dshm", mount_path="/dev/shm"),
        ],
        # TOS credentials: injected when the Secret exists; without it training is unaffected (upload is skipped)
        env_from=[k8s.V1EnvFromSource(secret_ref=k8s.V1SecretEnvSource(name=TOS_SECRET, optional=True))],
    )
    pod_spec = k8s.V1PodSpec(
        restart_policy="Never",
        containers=[container],
        volumes=[
            k8s.V1Volume(name="cfg", config_map=k8s.V1ConfigMapVolumeSource(name=name)),
            k8s.V1Volume(name="dshm", empty_dir=k8s.V1EmptyDirVolumeSource(medium="Memory", size_limit="16Gi")),
        ],
    )
    return k8s.V1Job(
        metadata=k8s.V1ObjectMeta(name=name, labels={"app": "robot-training", "task": req["task"]}),
        spec=k8s.V1JobSpec(
            backoff_limit=0,
            ttl_seconds_after_finished=86400,
            template=k8s.V1PodTemplateSpec(
                metadata=k8s.V1ObjectMeta(labels={"app": "robot-training", "job": name}),
                spec=pod_spec,
            ),
        ),
    )


# ---------------- API ----------------
@app.get("/api/catalog")
def catalog():
    robots = schema.load_robots()
    task_types = {tk for r in robots.values() for tk in r["tasks"]}
    tasks = {t: schema.load_task(t) for t in task_types}
    return {"robots": robots, "tasks": tasks, "budgets": schema.BUDGET_PRESETS}


@app.post("/api/validate")
def validate(req: dict):
    try:
        cmd = launcher.build_command(req)
        return {"ok": True, "command": " ".join(cmd)}
    except schema.ValidationError as e:
        return {"ok": False, "errors": str(e)}


@app.post("/api/train")
def train(req: dict):
    try:
        r = schema.validate(req)
        launcher.build_command(r)  # confirm a command can actually be built
    except schema.ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    name = f"train-{r['task']}-{uuid.uuid4().hex[:6]}"
    try:
        batch, core, k8s = _k8s()
        # The request is written to a ConfigMap and mounted into the job
        core.create_namespaced_config_map(
            NAMESPACE,
            k8s.V1ConfigMap(metadata=k8s.V1ObjectMeta(name=name), data={"request.yaml": yaml.safe_dump(r, allow_unicode=True)}),
        )
        batch.create_namespaced_job(NAMESPACE, _job_manifest(k8s, name, r))
    except Exception as e:  # cluster or job-creation failure -> return readable JSON, not a bare 500
        raise HTTPException(status_code=502, detail=f"failed to create the job (usually a flaky cluster connection, please retry): {type(e).__name__}: {str(e)[:200]}")
    return {"job": name, "task_id": schema.load_robots()[r["robot"]]["tasks"][r["task"]]}


def _render_job_manifest(k8s, name: str, task_id: str, prefix: str):
    """Preview-render job: run render_preview to produce an image and upload it to TOS for the
    frontend (skipped without credentials). Requests one GPU."""
    cmd = (
        "set -e; cd /workspace/isaaclab/training-service; mkdir -p /tmp/preview_out; "
        f"../isaaclab.sh -p render_preview.py --task {task_id} --template /config/template.yaml --out /tmp/preview_out/preview.png; "
        f"../isaaclab.sh -p upload_tos.py --local-dir /tmp/preview_out --prefix {prefix} || true"
    )
    container = k8s.V1Container(
        name="render", image=TRAIN_IMAGE, command=["/bin/bash", "-lc", cmd],
        resources=k8s.V1ResourceRequirements(limits={"nvidia.com/gpu": "1"}),
        volume_mounts=[k8s.V1VolumeMount(name="cfg", mount_path="/config")],
        env_from=[k8s.V1EnvFromSource(secret_ref=k8s.V1SecretEnvSource(name=TOS_SECRET, optional=True))],
    )
    pod_spec = k8s.V1PodSpec(
        restart_policy="Never", containers=[container],
        volumes=[k8s.V1Volume(name="cfg", config_map=k8s.V1ConfigMapVolumeSource(name=name))],
    )
    return k8s.V1Job(
        metadata=k8s.V1ObjectMeta(name=name, labels={"app": "robot-render"}),
        spec=k8s.V1JobSpec(
            backoff_limit=0, ttl_seconds_after_finished=3600,
            template=k8s.V1PodTemplateSpec(
                metadata=k8s.V1ObjectMeta(labels={"app": "robot-render", "job": name}), spec=pod_spec),
        ),
    )


@app.post("/api/render")
def render(req: dict):
    """Render a preview of the current (templated) scene so the user can confirm it looks right."""
    try:
        r = schema.validate(req)
    except schema.ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    name = f"render-{r['task']}-{uuid.uuid4().hex[:6]}"
    task_id = schema.load_robots()[r["robot"]]["tasks"][r["task"]]
    template = r.get("template", {"scene": {"assets": []}})
    prefix = f"previews/{r.get('output_name') or name}"
    try:
        batch, core, k8s = _k8s()
        core.create_namespaced_config_map(
            NAMESPACE,
            k8s.V1ConfigMap(metadata=k8s.V1ObjectMeta(name=name),
                            data={"template.yaml": yaml.safe_dump(template, allow_unicode=True)}),
        )
        batch.create_namespaced_job(NAMESPACE, _render_job_manifest(k8s, name, task_id, prefix))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"failed to create the render job (cluster connection?): {type(e).__name__}: {str(e)[:200]}")
    # Once rendering finishes the frontend fetches prefix/preview.png from TOS
    return {"job": name, "preview_key": f"{prefix}/preview.png"}


@app.get("/api/jobs/{name}")
def job_status(name: str):
    try:
        batch, core, _ = _k8s()
        job = batch.read_namespaced_job_status(name, NAMESPACE)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"failed to query the job (flaky cluster connection, please retry): {type(e).__name__}")
    s = job.status
    phase = "running"
    if s.succeeded:
        phase = "succeeded"
    elif s.failed:
        phase = "failed"
    elif not s.active:
        phase = "pending"
    # recent logs
    logs = ""
    try:
        pods = core.list_namespaced_pod(NAMESPACE, label_selector=f"job={name}")
        if pods.items:
            logs = core.read_namespaced_pod_log(pods.items[0].metadata.name, NAMESPACE, tail_lines=40)
    except Exception:
        pass
    return {"job": name, "phase": phase, "active": s.active, "succeeded": s.succeeded, "failed": s.failed, "logs": logs}


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(HERE, "index.html"), encoding="utf-8") as f:
        return f.read()

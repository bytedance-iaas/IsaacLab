# isaaclab chart

Helm packaging of the Isaac Lab image, which serves two different jobs. One value picks which:

| `mode` | What it is | What it renders |
|---|---|---|
| `train` | Isaac Lab training. Nothing exposed — exec in and launch a run. | StatefulSet, headless Service, PVC |
| `render` | An Isaac Sim workstation streamed over WebRTC: build scenes, simulate, watch it render. | The above, plus a load balancer, RBAC and an init container |

Both get a GPU, a persistent disk and a large `/dev/shm`. Training idles on `sleep infinity`;
rendering starts Isaac Sim itself, in the foreground, so its log is the container log.

The load balancer is **Volcengine-specific**. `mode=train` renders none of it and installs on any
cluster.

## Install

```bash
helm install isaaclab-train ./docker/charts -n isaaclab --create-namespace \
  -f docker/charts/examples/values-train.yaml \
  --set image.tag=<tag>
```

```bash
helm install isaaclab-render ./docker/charts -n isaaclab --create-namespace \
  -f docker/charts/examples/values-render-new-clb.yaml \
  --set image.tag=<tag>
```

`image.tag` and `persistence.storageClass` have no defaults the chart could guess, so it fails at
render time rather than deploying something broken. The examples in `examples/` carry real values
for the Volcengine test cluster.

## Hardware

The two modes do **not** accept the same machines. Training needs compute. Rendering additionally
needs a GPU with a **display engine** and the **NVENC encoder** the WebRTC stream feeds off — on a
GPU without them Isaac Sim starts, the log looks healthy, and the stream never produces a picture.

| Mode | Instance types |
|---|---|
| `render` | `ecs.gna3c.7xlarge` (L20) — verified end to end |
| `train` | *to be confirmed* |

> ⚠️ **This table is provisional.** Only the L20 entry has actually been exercised. `hardware.renderInstanceTypes`
> and `hardware.trainInstanceTypes` in `values.yaml` ship **empty**, which disables the check
> entirely; fill them in as types are validated.

### Doing both on one node

Some GPUs can train *and* render. If a release is meant to do both, the instance type has to be in
**both** lists — the intersection is the only hardware that satisfies both, and picking from either
list alone gets you a machine that fails at the other job.

The chart checks this and **prints a warning in the install notes** when `nodeSelector` pins a type
that is not listed for the mode:

```
⚠️ HARDWARE WARNING
  nodeSelector pins node.kubernetes.io/instance-type=ecs.g3i.large, which is not in
  hardware.renderInstanceTypes: ecs.gna3c.7xlarge
  Rendering needs a GPU with a display engine and an NVENC encoder. [...]
```

**It never fails the install.** The lists are maintained by hand and a stale one must not block a
valid deployment — so this is advisory, and an empty list means no check at all.

## Why rendering needs a public address

Worth understanding before touching the streaming values, because getting it wrong fails as a
black screen rather than an error.

Isaac Sim streams over WebRTC. Signalling (TCP 49100) only exchanges an SDP; the video travels over
**UDP**. The server has to advertise a **reachable** address inside that SDP as an ICE candidate,
and on its own it only knows the address its socket is bound to — a private pod IP. A client handed
such a candidate can never connect: signalling succeeds, the logs look fine, and ICE sits in
`checking` forever.

No proxy solves this:

- **An API gateway cannot.** Advertising an address is an application-level act, and a transparent
  proxy has no business rewriting the SDP body. (The thing that does that is a media gateway/SBC.)
  An HTTP gateway additionally cannot carry the UDP media at all.
- **STUN/TURN cannot**, because Isaac Sim exposes no settings for them.
- **NVIDIA does not avoid it either.** Their own Kubernetes streaming stack takes the same shape —
  a load balancer per stream, TCP signalling plus UDP media — and merely automates the address.

So the chart supplies the address: the load balancer has a stable VIP, an init container reads that
VIP from the Kubernetes API, and the entrypoint hands it to Isaac Sim. **Nobody has to know an IP
at deploy time**, and because the VIP belongs to the Service rather than the pod, it survives pod
replacement and upgrades.

Set `streaming.publicEndpoint` to skip the lookup — and the RBAC with it — when clients reach the
pod through something the cluster knows nothing about, such as a DNAT rule.

## Load balancer — two ways

### A. Provision a new CLB

```yaml
streaming:
  loadBalancer:
    create: true
    addressType: PUBLIC
```

Created with the release, address assigned in 10–30s, and the init container waits for it.

> ⚠️ **`helm uninstall` deletes it, and the address goes too.** A reinstall comes back on a
> different one. Once anybody has that address, switch to B.

### B. Bind to an existing CLB

```yaml
streaming:
  loadBalancer:
    create: false
    existingId: clb-13g25lmrk0um83n6nu4lgus8l
```

The release binds to that instance instead of creating one, so the address outlives the release.
Find the id of a CLB an earlier release provisioned:

```bash
kubectl get svc <release>-stream -n <namespace> \
  -o jsonpath='{.metadata.labels.service\.beta\.kubernetes\.io/volcengine-loadbalancer-id}'
```

> ⚠️ Do not point two releases at one CLB **on the same ports** — the second one's listeners
> collide with the first. Sharing is only safe when the port sets are distinct.

## Defaults worth knowing

- **Ports map 1:1** (`port == targetPort`). Isaac Sim writes the port numbers into the SDP next to
  the public address, so a rewritten port makes the advertised endpoint wrong. This is also why
  `publicEndpointPort` never has to be set.
- **The UDP media port is pinned** (`fixedHostPort`) so the CLB exposes one port instead of the
  whole 47998–48020 range Isaac Sim would otherwise pick from at session setup.
- **`externalTrafficPolicy` stays `Cluster`.** Measured on Volcengine CLB: a single five-tuple
  lands consistently on one backend node, so the source address ICE sees does not move mid-session.
  The cost is SNAT — Isaac Sim sees a node's internal IP, not the real client. `Local` removes the
  SNAT and a hop but changes how health checks pick backends; the default is what has been proven.
- **The PVC is reused, not replaced.** The claim is a `volumeClaimTemplate` named
  `data-<release>-0`, bound if it already exists, so a reinstall keeps its contents and
  `helm uninstall` deliberately leaves the disk behind.
- **`args`, never `command`.** The image entrypoint is `tini`; a pod-level `command:` would replace
  it and leave nothing reaping what Isaac Sim re-parents.

## Things that will bite you

**Mounting the volume over the Isaac Lab install.** `/workspace/isaaclab` holds Isaac Lab *and* the
`_isaac_sim` symlink to the Isaac Sim runtime, in the image. A volume mounted there hides both and
nothing starts. The chart rejects this outright — keep `persistence.mountPath` on a path the image
does not populate, e.g. `/data`.

**Expecting the UDP port to be listening.** Isaac Sim binds the media port lazily, at session setup
— it is absent from `/proc/net/udp` until a client actually connects. That is not a fault. To check
it once connected (the image has no `netstat`, `ss` or `lsof`):

```bash
kubectl exec <release>-0 -- sh -c 'grep -i ":BB7E " /proc/net/udp'   # BB7E = 47998
```

**A black screen with healthy logs.** Two usual causes, in order: the advertised address does not
actually reach the pod (`kubectl exec <release>-0 -- cat /shared/public_ip`), or the browser —
Chrome and Safari have a known issue with Isaac Sim in a container, so try the native streaming
client or Firefox before suspecting the network.

**The init container waiting forever.** If its log says it cannot read the Service, that is RBAC.
Otherwise the CLB never got an address — `kubectl describe svc <release>-stream`, and a
`subnetId` outside this cluster's VPC is the usual cause.

**Renaming a release.** The PVC name derives from it, so a rename strands the old disk and starts
on an empty one.

**`nvidia.com/gpu` requests and limits must be equal.** It is non-compressible; the API server
rejects the pod otherwise.

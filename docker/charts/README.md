# isaaclab chart

Helm packaging of the Isaac Lab image, which serves two different jobs. One value picks which:

| `mode` | What it is | What it renders |
|---|---|---|
| `train` | Isaac Lab training. Nothing exposed — exec in and launch a run. | StatefulSet, headless Service, PVC |
| `render` | An Isaac Sim workstation streamed over WebRTC: build scenes, simulate, watch it render. | The above, plus a load balancer, RBAC and an init container |

Training runs can additionally publish a live view to **LiveKit** for browser viewing
(`train.py --stream`) — see [Watching a training run](#watching-a-training-run-livekit). That path
is outbound-only, so it costs the training pod no exposure at all.

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

## Exposure and hardening

**Isaac Sim's WebRTC endpoint authenticates nobody.** There is no setting for it anywhere in the
livestream extension — the design assumes a trusted network. Anyone who can reach the signalling
port can drive that session: build scenes, run the simulation, watch it. There is no password step
to fail, so a leaked address is full access.

That makes network reachability the only access control there is, which is why
`streaming.loadBalancer.addressType` defaults to **`PRIVATE`**.

What each path exposes:

| Path | Exposed | Authenticated |
|---|---|---|
| `mode=train` | nothing | — |
| `mode=render` | signalling 49100/TCP, media 47998/UDP | **no** |
| `mode=render` + `exposeNvcfPort` | the above, plus 8011/TCP | **no**, and far worse — see below |
| LiveKit dependency | 7880/7881 TCP, 7882/UDP | **yes** — JWT signed with the API key; without one the server answers `401 no permissions to access the room` |

### Port 8011 is off, and should stay off

`streaming.exposeNvcfPort` defaults to `false`. That port serves the `omni.services.livestream.nvcf`
extension's HTTP API: **15 unauthenticated endpoints**, confirmed by fetching `/openapi.json` from
the VIP with no credentials. Among them:

```
POST /convert/asset/process     import_path / output_path / archive_path
POST /convert/cad/process       import_path / output_path / config_path
POST /v1/streaming/endsession
GET  /metrics
```

`import_path` is documented as "Full path to source asset" and `output_path` is an arbitrary
output path. The container runs as **root**. None of the routes declare an auth dependency. So
publishing this port is not "someone can watch your scene" — it is arbitrary-path file access and
the ability to end sessions at will.

It is off because the streaming client does not appear to need it: Isaac Lab's own
`--livestream 1` sets only `port=49100` and never mentions 8011, which is the NVCF platform's
orchestration API rather than part of the client protocol. **That is inference, not proof.** If a
client stops connecting after upgrading, set `streaming.exposeNvcfPort=true` — and say so, because
it means the inference was wrong and this default needs revisiting.

### If you need PUBLIC

The chart cannot restrict access — a `LoadBalancer` Service has no allowlist to render. Do it at
the network layer, and do it *before* the address is handed out:

1. **Restrict by source address.** Put an ACL on the CLB, or a security group rule on its backends,
   allowing only the offices, VPN egress addresses or bastion hosts that should reach it. This is
   the only control that actually stands between the internet and the session.
2. **Publish the narrowest port set.** Leave `exposeNvcfPort=false`. Signalling and media are
   enough for a client to connect.
3. **Treat the address as a credential**, since in effect it is the only one. Do not put it in
   tickets, chat or dashboards that outlive the deployment.
4. **Take it down when it is not in use.** A workstation left running over a weekend is an
   unauthenticated session on the internet for two days. `helm uninstall`, or scale the
   StatefulSet to zero.

The safer shape, when it fits, is `PRIVATE` plus a bastion or VPN: the same access, but reaching
it requires an account somewhere that *does* authenticate.

> ⚠️ Anything deployed from the hand-written manifests this chart replaces was public. Installing
> the chart over it lands on `PRIVATE`, which re-provisions the CLB — **the address changes**. Set
> `streaming.loadBalancer.addressType=PUBLIC` explicitly to keep a public one, ideally along with
> the ACL that should have been there.

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

### Bandwidth — the default cannot carry video

**The platform default is 1 Mbps.** A stream on that link connects, completes ICE, and then
delivers a slideshow or stalls outright — which reads as a broken stream rather than a capped
link, so it costs an afternoon to diagnose.

`streaming.loadBalancer.bandwidthMbps` defaults to **10**, a working floor for a single 1080p
session. Raise it for higher resolution, a higher frame rate, or concurrent viewers.

```yaml
streaming:
  loadBalancer:
    bandwidthMbps: 10
    eipBillingType: ""     # empty = platform default; by-bandwidth and by-traffic differ a lot in cost
```

> ⚠️ **Verify it in the console after deploying.** The CCM echoes these annotations back verbatim
> whether or not it understands them — measured on this cluster: a probe Service carrying two
> different spellings had *both* returned in
> `system-volcengine-loadbalancer-last-applied-annotations`, with no warning either way. A wrong
> key therefore looks identical to a working one from `kubectl`. The chart sends both documented
> spellings for that reason, but the authoritative reading is the EIP's bandwidth in the console.

Only applies when the chart **creates** the CLB. Binding to an existing one inherits whatever
bandwidth that CLB already has.

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

## Watching a training run (LiveKit)

`train.py --stream` publishes the training view to LiveKit, where people watch it in a browser.
Note the direction, because it is the whole point:

**The training pod connects OUT to LiveKit and never accepts a connection.** No load balancer, no
public address, no RBAC on the training side. A hundred training runs share one public entry
point instead of needing a hundred — which is exactly what `mode=render` cannot do, and why the
two paths are different.

It is **one-way video**. Viewers watch; they cannot touch the simulation. Interactive work is what
`mode=render` is for, and no amount of LiveKit replaces it — LiveKit is a video relay, Isaac Sim's
own WebRTC is a remote desktop protocol carrying input back. Setting `livekit.enabled=true` on a
`mode=render` release is rejected for that reason.

### The server comes from a dependency

The LiveKit server is the `livekit` chart from the same OCI registry, declared in `Chart.yaml`
and gated on `livekit.enabled`. It is not reimplemented here: it already solves the same problems
this chart would have had to (advertising the CLB's VIP to clients, provisioning the CLB) and
additionally keeps the CLB on uninstall — which matters, because that public IP is baked into
every viewer's URL.

```bash
helm dependency update ./docker/charts     # required before install or package
```

`docker/publish_charts.sh` does this itself. The fetched tarballs under `charts/` are gitignored;
`Chart.lock` pins the version and digest.

Everything under the `livekit:` key belongs to that chart — see its own `values.yaml` for the full
set. `nameOverride` is pinned to `livekit` here rather than left to default, for two reasons: the
dependency names its objects `<name>-ip`, which would otherwise collide with this chart's own
ServiceAccount, and a shared server should have a stable address that does not change with
whichever release happens to own it. You get `livekit-clb`, `livekit-auth`, and so on.

### Deploy it once, point everything else at it

```yaml
# The one release that owns the server
livekit:
  enabled: true
  service:
    subnetId: subnet-xxxxxxxx      # required, must be in this cluster's VPC
    keepOnUninstall: true
```
```bash
--set livekit.auth.keys="devkey: $(openssl rand -hex 32)"
```

Every other training release brings up nothing and just publishes to it:

```yaml
livekit:
  enabled: false
stream:
  url: ws://livekit-clb.<namespace>.svc.cluster.local:7880
  existingSecret: livekit-auth
```

> ⚠️ Turning `enabled: true` on per training run gives you N servers and N load balancers, which
> is **worse than not using LiveKit at all**. Uninstalling the owning release stops the server for
> everyone still publishing to it — though with `keepOnUninstall` the CLB and its address survive.

Prefer the **in-cluster** URL over the public VIP: the publisher runs inside the cluster, so going
out to the public address and back is a hairpin that pays for a round trip and egress for nothing.

### One Secret, both ends

The dependency creates a Secret (`livekit-auth`) holding LiveKit's own credential format:

```
keys: "api_key: api_secret"
```

The publisher reads **that same Secret**, as `LIVEKIT_KEYS`. `livekit_stream.py` accepts this form
in addition to the split `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` precisely so one credential can
serve both ends — a second copy in another shape is free to drift, and the failure mode is a
connection that authenticates and then fails, which looks nothing like a configuration mistake.

> ⚠️ `livekit.auth.keys` goes into the Helm release history **verbatim**, readable by anyone who
> can run `helm get values`. The dependency takes it that way deliberately: it is installed from
> the VKE console's Helm page, which has no shell, so "create the Secret first" is not something
> anyone can do there. Use a dedicated key, not one shared with anything else.

### The server has the same address problem

LiveKit must advertise an address its clients can reach, and inside a pod it only knows a private
one — the identical problem Isaac Sim has, and the dependency solves it the identical way: an init
container reads the CLB's VIP from the Kubernetes API and passes it as `--node-ip`. Setting
`livekit.nodeIp` explicitly skips the lookup and drops that RBAC with it. (LiveKit can also
discover this itself via STUN, which is not used because it needs a reachable public STUN server.)

### `--stream` needs a rendering GPU

It builds a camera sensor and reads frames back, so a run using it needs a GPU that can **render**,
not merely compute. Such a run is the "both jobs on one node" case from the hardware section: its
instance type must be in **both** lists.

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

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

**Training**, on its own. LiveKit is not involved — most runs never call `--stream`:

```bash
helm install isaaclab-train ./docker/charts -n isaaclab --create-namespace \
  --set image.tag=<tag> --set persistence.storageClass=<class>
```

**Training with a browser viewer.** This is the one thing that needs LiveKit, and it needs both
values:

```bash
helm install isaaclab-train ./docker/charts -n isaaclab --create-namespace \
  --set image.tag=<tag> --set persistence.storageClass=<class> \
  --set stream.url=ws://livekit-clb.isaaclab.svc.cluster.local:7880 \
  --set stream.existingSecret=livekit-auth \
  --set viewer.enabled=true \
  --set viewer.publicLivekitUrl=ws://<livekit-clb-public-ip>:7880
```

**An Isaac Sim workstation**, streamed over WebRTC on a CLB of its own:

```bash
helm install isaaclab-render ./docker/charts -n isaaclab --create-namespace \
  --set image.tag=<tag> --set persistence.storageClass=<class> \
  --set mode=render --set streaming.loadBalancer.addressType=PUBLIC
```

`image.tag` and `persistence.storageClass` have no defaults the chart could guess, so it fails at
render time rather than deploying something broken.

**Train mode additionally requires a decision about LiveKit** — see
[Watching a training run](#watching-a-training-run-livekit). And any release that deploys LiveKit
or the viewer needs Secrets that exist *before* the install, because no chart here will create
one:

```bash
kubectl create secret generic livekit-auth -n isaaclab \
  --from-literal=keys="devkey: $(openssl rand -hex 32)"

kubectl create secret generic isaaclab-viewer-auth -n isaaclab \
  --from-literal=username=<user> --from-literal=password='<password>'
```

## Hardware

**The chart does not check this.** There is no hardware validation in the templates and no
`hardware:` values block — getting the node pool right is a deployment decision, and a list of
instance types maintained inside a chart goes stale without anyone noticing. What follows is the
requirement; put machines that meet it in the pool, and pin them with `nodeSelector`.

The two modes do **not** accept the same machines. Training needs compute. Rendering additionally
needs a GPU with a **display engine** and the **NVENC encoder** the WebRTC stream feeds off.

> ⚠️ On a GPU without them, **Isaac Sim starts, the log looks healthy, and the stream stays
> black.** There is no error to find — this is the single most expensive way to get the node wrong,
> which is why it is the first thing to check when a render release produces no picture.

| Mode | Needs | Verified |
|---|---|---|
| `render` | display engine + NVENC | `ecs.gna3c.7xlarge` (L20), end to end |
| `train` | compute only — any GPU node | — |

`train.py --stream` is the exception that catches people: it **renders** a camera view, so a
training run using it needs render-capable hardware even though the release is `mode=train`. A
node that satisfies both rows above is the only kind that can host such a release.

Pin it either way:

```yaml
nodeSelector: { node.kubernetes.io/instance-type: ecs.gna3c.7xlarge }   # any node of this type
nodeSelector: { kubernetes.io/hostname: 192.168.3.32 }                  # this exact node
```

Prefer the instance type unless you specifically need one machine — a hostname pin survives
neither the node being replaced nor the cluster being rebuilt.

> ⚠️ Worth pinning once a PVC exists regardless: `ebs-*` volumes are **zonal**, so a pod
> rescheduled into another zone can never attach the disk it already has.

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
| LiveKit (separate release) | 7880/7881 TCP, 7882/UDP | **yes** — JWT signed with the API key; without one the server answers `401 no permissions to access the room` |

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

**Confirmed working**: a release requesting 10 came out as 10 Mbps on the EIP in the console.

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

> ⚠️ Binding to a CLB the **CCM itself created** leaves this Service at `<pending>` forever — the
> data path works, but the CCM will not adopt a load balancer it made, so it never reports the
> address and the render pod's init container would wait for one that is never coming. For the
> render stream, unlike the viewer, that is fatal: it needs the address to advertise in the SDP.
> **Use a console-created CLB here**, or let the release provision its own. See
> [How many CLBs](#how-many-clbs-a-deployment-needs).

> ⚠️ Do not point two releases at one CLB **on the same ports** — the second one's listeners
> collide with the first. Sharing is only safe when the port sets are distinct, and that collision
> is also invisible from Kubernetes.

## Watching a training run (LiveKit)

`train.py --stream` publishes the training view to a LiveKit server, where people watch it in a
browser. Note the direction, because it is the whole point:

**The training pod connects OUT to LiveKit and never accepts a connection.** No load balancer, no
public address, no RBAC on the training side. A hundred training runs share one public entry point
instead of needing a hundred — which is exactly what `mode=render` cannot do, and why the two
paths are different.

It is **one-way video**. Viewers watch; they cannot touch the simulation. Interactive work is what
`mode=render` is for, and no amount of LiveKit replaces it — LiveKit is a video relay, Isaac Sim's
own WebRTC is a remote desktop protocol carrying input back.

### This chart does not deploy LiveKit

Because **one server serves the whole cluster.** Bundling it here would make the server a property
of a training release, which argues for exactly the arrangement that defeats the point — a server,
and a load balancer, per training run. It would also mean that reading one release's values told
you nothing about where the others publish.

So LiveKit is installed once, separately, from its own chart:

```bash
kubectl create secret generic livekit-auth -n <namespace> \
  --from-literal=keys="devkey: $(openssl rand -hex 32)"

helm install livekit oci://ai-containers-cn-beijing.cr.volces.com/physicalai/livekit \
  -n <namespace> --set nameOverride=livekit \
  --set auth.existingSecret=livekit-auth \
  --set service.subnetId=<subnet-in-this-cluster-vpc>
```

and every Isaac Lab release is merely **told where it is**.

### Only the viewer needs LiveKit

Training does not. `train.py --stream` is opt-in and most runs never use it, so a release that
names no server installs without complaint — nothing is wired into the pod, and `--stream` would
fall back to whatever address `livekit_stream.py` was built against.

**The viewer is what makes it mandatory.** It lists the rooms on a LiveKit server and signs each
visitor a token with that server's own Secret, so `viewer.enabled=true` requires both:

```yaml
stream:
  url: ws://livekit-clb.<namespace>.svc.cluster.local:7880   # IN-CLUSTER; the BACKEND uses this
  existingSecret: livekit-auth                                # the SAME Secret object the server reads
viewer:
  enabled: true
  publicLivekitUrl: ws://<livekit-clb-public-ip>:7880         # the BROWSER uses this
```

Set `stream.url` and `stream.existingSecret` **both or neither** — one alone is a half-configured
publisher, and each half looks configured while nothing works. `stream.url` must carry a port: the
chart reads it back out for the viewer's page, so there is no second value to disagree with it.

> ⚠️ Secrets do not cross namespaces. If LiveKit runs in another one, copy its Secret into this
> release's namespace — the publisher must read the **same object** the server does, not a copy of
> the values.

### One Secret, both ends

**You create the Secret; no chart here does.** It holds LiveKit's own credential format:

```bash
kubectl create secret generic livekit-auth -n <namespace> \
  --from-literal=keys="devkey: $(openssl rand -hex 32)"
```

The LiveKit server reads it, and the publisher reads **that same object**, as `LIVEKIT_KEYS`.
`livekit_stream.py` accepts this form in addition to the split `LIVEKIT_API_KEY` /
`LIVEKIT_API_SECRET` precisely so one credential can serve both ends — a second copy in another
shape is free to drift, and the failure mode is a connection that authenticates and then fails,
which looks nothing like a configuration mistake.

> ⚠️ Neither chart will take the key through Helm values, and the livekit chart **refuses** one
> rather than ignoring it. Helm stores values verbatim in the release history, so a key passed
> that way stays readable by anyone who can run `helm get values`, on every revision that ever
> carried it — and this key mints an access token for *any* room on that server.

### The viewer, and where it is published

The viewer (`viewer/` in this repository) lists the rooms currently being published to and plays
one in the browser. Viewers never handle a LiveKit key: the backend holds the credential and signs
a short-lived, read-only, single-room token for each page. It is published with **mandatory** HTTP
Basic auth — a page that enumerates every running training and plays it must not be open, and
there is deliberately no switch to turn that off.

**It binds to LiveKit's CLB by default** — port 80 fits alongside 7880/7881/7882 — with one
consequence to accept, described under [How many CLBs](#how-many-clbs-a-deployment-needs).

```yaml
viewer:
  enabled: true
  auth:
    existingSecret: isaaclab-viewer-auth
  publicLivekitUrl: ws://<livekit-clb-public-ip>:7880
  service:
    existingId: clb-xxxxxxxxxxxxxxxx   # LiveKit's
```

Set `service.create: true` (with `subnetId` and `bandwidthMbps`) to give the page its own CLB
instead.

The page needs **two different addresses**, and both are inputs:

| value | who uses it | why it cannot be the other one |
| --- | --- | --- |
| `stream.url` | the **backend**, in-cluster | going out to the public address and back is a hairpin |
| `viewer.publicLivekitUrl` | the **browser** | a browser cannot resolve a Service name |

`publicLivekitUrl` is **required**, and nothing in the release can derive it. LiveKit sits on its
own load balancer, which this release neither creates nor may bind to — see the ⚠️ above. An
earlier version read the address off the viewer's *own* Service, which was correct only while the
two shared one CLB; since that arrangement is not allowed, the lookup would now return the wrong
address with full confidence, so it was removed along with the init container and its RBAC.

```yaml
viewer:
  publicLivekitUrl: ws://<livekit-clb-public-ip>:7880
```

> ⚠️ Open the page over **http**, not https, until LiveKit terminates TLS. An https page is not
> allowed to open a plaintext `ws://` connection, so it loads and connects to nothing.

### `--stream` needs a rendering GPU

It builds a camera sensor and reads frames back, so a run using it needs a GPU that can **render**,
not merely compute — the case called out under [Hardware](#hardware). A `mode=train` release whose
runs use `--stream` must therefore sit on render-capable hardware, which nothing in the chart will
tell you: the run starts and the picture never arrives.

## How many CLBs a deployment needs

Left to itself a full deployment reaches three load balancers — the render stream, LiveKit, and
the viewer. Their ports are disjoint (49100/47998, 7880/7881/7882, 80), so two of them collapse
into one: **the viewer binds to LiveKit's CLB by default**, and a train deployment is one CLB, not
two.

### Binding to a CCM-created CLB works, but the Service never gets adopted

This is the part worth understanding before you rely on it.

```
Warning  EnsureLoadBalancerFailed  service-controller
  can not reuse clb clb-xxxx which is created by vke
```

The CCM **builds the data path correctly** — the listener is created, the backend server group
gets the Service's NodePort, and the page is reachable on the shared address. What it refuses is
to record *ownership* of a load balancer it made. So:

- `.status.loadBalancer.ingress` is never written. `kubectl get svc` shows **`EXTERNAL-IP
  <pending>` forever**, and the reconcile fails again every ~30s for the life of the release.
  Anything watching Service health will flag it. It is not broken; it is **unadopted**.
- The backend server group is therefore almost certainly **not maintained** — every reconcile ends
  in that error before it could update anything. `externalTrafficPolicy` is `Cluster`, so any node
  in the group forwards to the pod wherever it runs; but nodes added or removed later will not be
  reflected. **Unverified**: nobody has watched it across a node change.

> ⚠️ Nothing in the chart can wait on that address, which is why `viewer.publicLivekitUrl` is a
> required input rather than something discovered at startup. An earlier version had an init
> container poll the Service for it — against a CCM-created CLB that polls forever, and against a
> self-provisioned one it returns the viewer's own address instead of LiveKit's. It was removed.

Neither problem exists for a CLB created **outside** Kubernetes — console, API, Terraform. The CCM
adopts those normally, writes the address, and keeps the backend group current. That is the clean
way to share one load balancer, and the only way to share one across *many* releases.

`viewer.service.create: true` gives the page a CLB of its own if you would rather have a Service
that reports its address than one fewer EIP.

### The render stream keeps its own regardless

- `mode=render` publishes an **unauthenticated** Isaac Sim session — network reachability is the
  only access control there is. A CLB is public or private *as a whole*, so putting the stream on
  the same public instance as the password-protected viewer hands it exposure the viewer can
  afford and it cannot. Keeping it separate is what lets it stay `PRIVATE`, or carry its own ACL.
- Its ports are chosen **per release** (`streaming.signallingPort`, `streaming.mediaPort`). On a
  dedicated CLB that is a local decision; on a shared one, two render releases both defaulting to
  49100/47998 collide — on the load balancer rather than in Kubernetes, so nothing warns you.

> ⚠️ **Size a shared CLB's bandwidth for the sum.** One instance carrying signalling, every
> subscriber's media *and* the page caps all of them at once, and a capped link reads as a broken
> stream rather than a capped link. 10 Mbps is a floor for a **single** 1080p session. Set it where
> the CLB is made: a release binding to an existing instance sends no bandwidth annotation.

## Finding these values in your cluster

Everything the chart asks for that is specific to your environment, and where it comes from. Run
these in the namespace you are installing into.

> ⚠️ These commands are written out but were **not executed against a live cluster** while this
> was written. Check the output looks like the example before pasting it into a values file.

### The LiveKit server — `stream.url`, `stream.existingSecret`

```bash
kubectl get svc -n <namespace> -l app=livekit
```

```
NAME          TYPE           CLUSTER-IP     EXTERNAL-IP       PORT(S)                                       AGE
livekit-clb   LoadBalancer   172.16.1.23    115.190.190.201   7880:31234/TCP,7881:31235/TCP,7882:30987/UDP  3d
```

The Service **name** is the host and the first port is signalling, so:

```yaml
stream:
  url: ws://livekit-clb.<namespace>.svc.cluster.local:7880
```

The Secret is whatever that server was told to read. Ask the LiveKit release rather than assuming
`livekit-auth` — reading a *different* object than the server does is the one mistake that fails
after appearing to work:

```bash
helm get values livekit -n <namespace> -a | grep -A3 '^auth:'
```

```
auth:
  existingSecret: livekit-auth
  keysKey: keys
```

If there is no `livekit` release, the cluster has no server yet — see
[This chart does not deploy LiveKit](#this-chart-does-not-deploy-livekit).

### The CLB to publish the viewer on — `viewer.service.existingId`

The instance LiveKit is already using. The CCM writes its id onto the Service as a label:

```bash
kubectl get svc livekit-clb -n <namespace> --show-labels
```

Look for `service.beta.kubernetes.io/volcengine-loadbalancer-id=clb-…`. To take just that value:

```bash
kubectl get svc livekit-clb -n <namespace> \
  -o jsonpath='{.metadata.labels.service\.beta\.kubernetes\.io/volcengine-loadbalancer-id}'
```

> ⚠️ Empty output means the CLB has not been provisioned yet — it takes 10–30s after the LiveKit
> install, and the label does not exist until it is. Wait and re-run; do not fill in a guess.

The address it will publish on, which is also where the viewer page comes up:

```bash
kubectl get svc livekit-clb -n <namespace> \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

### Storage — `persistence.storageClass`

```bash
kubectl get storageclass
```

Pick one that supports `ReadWriteOnce` block storage; on Volcengine that is an `ebs-*` class. The
chart has no default because an empty `storageClassName` means "disable dynamic provisioning" to
Kubernetes, so the claim would silently never bind.

> ⚠️ `ebs-*` volumes are **zonal**. A pod rescheduled into another zone can never attach the disk
> it already has, which is a reason to pin `nodeSelector` once a PVC exists.

### Nodes and GPUs — `nodeSelector`, `resources`

Which nodes exist, and what type they are:

```bash
kubectl get nodes -L node.kubernetes.io/instance-type
```

How much is actually **free** on one — this is the number that matters, not the node's capacity:

```bash
kubectl describe node <node-name>
```

Read the `Allocatable` block for the ceiling and `Allocated resources` for what is already
claimed. Set `resources.requests` from the difference, and keep `limits.memory` **under**
`Allocatable` — a limit above it lets one pod drive the whole node into memory pressure instead of
being OOM-killed alone.

Pin a node either way:

```yaml
nodeSelector: { node.kubernetes.io/instance-type: <type> }   # any node of this type
nodeSelector: { kubernetes.io/hostname: <node-name> }        # this exact node
```

> ⚠️ `nvidia.com/gpu` requests and limits must be **equal**; it is non-compressible and the API
> server rejects the pod otherwise.

### The subnet for a new CLB — `streaming.loadBalancer.subnetId`

Render mode only, and only when provisioning. It must be in **this cluster's** VPC — a subnet from
elsewhere is the usual reason a CLB never gets an address and the init container waits it out.

The easiest source is a Service that already has a working CLB, since its annotation is known-good
for this cluster:

```bash
kubectl get svc -A -o yaml | grep -B2 'volcengine-loadbalancer-subnet-id'
```

Otherwise take it from the VPC console, matching the cluster's VPC.

### An existing CLB for the render stream — `streaming.loadBalancer.existingId`

Same label as the viewer's, on the stream Service of a release that provisioned one. **Read it
before uninstalling that release** — the id goes with it:

```bash
kubectl get svc <release>-stream -n <namespace> --show-labels
```

### Checking the Secrets exist before installing

Both must exist first; a missing one holds the pod in `CreateContainerConfigError` rather than
starting something without credentials.

```bash
kubectl get secret livekit-auth isaaclab-viewer-auth -n <namespace>
```

To see which **keys** a Secret holds, without printing the values:

```bash
kubectl describe secret livekit-auth -n <namespace>
```

`livekit-auth` must have a `keys` entry; `isaaclab-viewer-auth` must have `username` and
`password`. A Secret that exists with the wrong key name fails exactly like a missing one.

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


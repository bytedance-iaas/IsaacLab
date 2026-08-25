{{/*
Resource name. Comes from the release name so that two installs never collide;
nameOverride pins it only when you deliberately want a fixed name.
*/}}
{{- define "isaaclab.fullname" -}}
{{- default .Release.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "isaaclab.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Selector labels. Deliberately just `app`: this lands in the StatefulSet's IMMUTABLE
.spec.selector, so keeping it minimal leaves room to change the descriptive labels later
without having to delete and recreate the workload.
*/}}
{{- define "isaaclab.selectorLabels" -}}
app: {{ include "isaaclab.fullname" . }}
{{- end }}

{{- define "isaaclab.labels" -}}
{{ include "isaaclab.selectorLabels" . }}
app.kubernetes.io/name: {{ include "isaaclab.fullname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/component: {{ .Values.mode }}
helm.sh/chart: {{ include "isaaclab.chart" . }}
{{- end }}

{{/*
True when this release runs the streaming workstation. Every render-only object keys off this,
so there is one definition of what "render mode" means.
*/}}
{{- define "isaaclab.isRender" -}}
{{- eq .Values.mode "render" -}}
{{- end }}

{{/*
Name of the ServiceAccount the pod runs as. Only render mode that has to DISCOVER its address
needs one; training mode reads nothing, and so does render mode when the address is supplied
explicitly. Must stay in step with the condition guarding rbac.yaml, or the pod would reference
a ServiceAccount that was never created.
*/}}
{{- define "isaaclab.serviceAccountName" -}}
{{- if and (eq (include "isaaclab.isRender" .) "true") (not .Values.streaming.publicEndpoint) }}
{{- printf "%s-ip" (include "isaaclab.fullname" .) }}
{{- else }}
{{- "default" }}
{{- end }}
{{- end }}

{{/*
The load balancer Service that carries the stream. Named apart from the headless Service so
both can exist; the init container looks this one up by name.
*/}}
{{- define "isaaclab.streamServiceName" -}}
{{- printf "%s-stream" (include "isaaclab.fullname" .) }}
{{- end }}
{{/*
The LiveKit signalling port, read back out of stream.url so there is no second value to disagree
with it. stream.url is validated to carry a port, so `last` is a port and not a hostname.
*/}}
{{- define "isaaclab.stream.port" -}}
{{- .Values.stream.url | splitList ":" | last }}
{{- end }}

{{/*
Where the publisher sends its video, and the Secret it authenticates with. Empty when streaming is
off, which is what drops LIVEKIT_URL and LIVEKIT_KEYS from the pod.
*/}}
{{- define "isaaclab.stream.url" -}}
{{- .Values.stream.url }}
{{- end }}

{{- define "isaaclab.stream.secretName" -}}
{{- .Values.stream.existingSecret }}
{{- end }}

{{- define "isaaclab.stream.room" -}}
{{- default (printf "train-%s" (include "isaaclab.fullname" .)) .Values.stream.room }}
{{- end }}

{{/*
LiveKit is OPTIONAL for training, and required only by the viewer.

A training run reaches LiveKit through `train.py --stream`, which most runs do not use -- so a
train release that names no server is a perfectly ordinary headless deployment and installs
without complaint. Name one and the address and credentials are wired into the pod; leave it empty
and they are not, and `--stream` falls back to whatever livekit_stream.py was built against.

The viewer is where it stops being optional: it lists the rooms on a LiveKit server and signs each
page a token with that server's own Secret, so it has nothing to show and nothing to sign with
unless a server is named. That check lives in isaaclab.viewer.validate.
*/}}
{{- define "isaaclab.stream.validate" -}}

{{/* LiveKit is the TRAINING view; the render path streams over its own WebRTC endpoint and never
     reads these, so a value here would be read by nobody while looking deliberate. */}}
{{- if eq .Values.mode "render" }}
{{- if or .Values.stream.url .Values.stream.existingSecret }}
{{- fail "stream.url / stream.existingSecret do not apply when mode=render, and are rejected rather than ignored: the render path streams over its own WebRTC endpoint (streaming.loadBalancer) and never reads them. LiveKit carries the TRAINING view — it relays one-way video and cannot carry input back, so the two are not interchangeable either way round." }}
{{- end }}
{{- end }}

{{/* Both or neither. One alone is a half-configured publisher: an address with no credentials
     authenticates against nothing, and credentials with no address go to the compiled-in
     default -- each looks configured and neither works. */}}
{{- if and .Values.stream.url (not .Values.stream.existingSecret) }}
{{- fail "stream.url is set but stream.existingSecret is empty. The publisher needs both: the address to reach LiveKit, and the Secret to authenticate with — which must be the SAME object the server itself reads, or the two ends authenticate against different keys and every connection fails after appearing to succeed." }}
{{- end }}
{{- if and .Values.stream.existingSecret (not .Values.stream.url) }}
{{- fail "stream.existingSecret is set but stream.url is empty. Without an address the publisher falls back to the one compiled into livekit_stream.py, so the credentials here would be used against a server nobody chose. Set stream.url to the in-cluster address of the LiveKit server, e.g. ws://livekit-clb.<namespace>.svc.cluster.local:7880." }}
{{- end }}

{{/* Validated because the chart reads the PORT back out of it for the viewer's page: a URL
     without one yields a hostname where a port belongs, and the page loads and never connects. */}}
{{- if .Values.stream.url }}
{{- if not (regexMatch "^wss?://[^/:]+:[0-9]+$" .Values.stream.url) }}
{{- fail (printf "stream.url must be ws://host:port or wss://host:port, got %q. The port is not optional: the chart reads it back out of this value to tell the viewer's page which port to reach LiveKit on. Use the in-cluster address, e.g. ws://livekit-clb.%s.svc.cluster.local:7880." .Values.stream.url .Release.Namespace) }}
{{- end }}
{{- end }}

{{- end }}

{{/*
The chart ships no default image tag, so catch it here: an empty tag would otherwise render as
"repo:" and surface much later as a confusing ImagePullBackOff.
*/}}
{{- define "isaaclab.validate" -}}
{{- if not .Values.image.tag }}
{{- fail "image.tag is required: the chart does not track the image version. Pass --set image.tag=<tag>." }}
{{- end }}
{{- if not .Values.image.repository }}
{{- fail "image.repository is required." }}
{{- end }}
{{- if not .Values.persistence.storageClass }}
{{- fail "persistence.storageClass is required: an empty storageClassName means 'disable dynamic provisioning' to Kubernetes, so the claim would never bind. Pass --set persistence.storageClass=<class> (kubectl get storageclass)." }}
{{- end }}
{{- if not (has .Values.mode (list "train" "render")) }}
{{- fail (printf "mode must be either \"train\" or \"render\", got %q." .Values.mode) }}
{{- end }}

{{/*
Mounting over the Isaac Lab install is the one persistence mistake that produces a baffling
failure -- the whole framework simply vanishes from the image -- so it is rejected outright.
*/}}
{{- $mount := .Values.persistence.mountPath | default "" }}
{{- if not $mount }}
{{- fail "persistence.mountPath is required." }}
{{- end }}
{{- if or (eq $mount "/workspace") (eq $mount "/workspace/isaaclab") (hasPrefix "/workspace/isaaclab/" $mount) (eq $mount "/isaac-sim") }}
{{- fail (printf "persistence.mountPath %q would hide the Isaac Lab installation that ships in the image: /workspace/isaaclab holds Isaac Lab and the _isaac_sim symlink to the Isaac Sim runtime, and a volume mounted over it makes both disappear. Pick a path the image does not populate, e.g. /data." $mount) }}
{{- end }}

{{- include "isaaclab.streaming.validate" . }}
{{- include "isaaclab.stream.validate" . }}
{{- include "isaaclab.viewer.validate" . }}
{{- end }}

{{/*
The viewer is published to the internet by default, so its password is not optional and neither
is the Secret holding it. Everything here fails rendering rather than starting an open page that
lists every running training and plays it.
*/}}
{{- define "isaaclab.viewer.validate" -}}
{{- if .Values.viewer.enabled }}
{{- if eq .Values.mode "render" }}
{{- fail "viewer.enabled=true does not apply when mode=render: the viewer watches TRAINING runs published to LiveKit. A render release streams its own interactive session over streaming.loadBalancer, and there is nothing for the page to list." }}
{{- end }}
{{- if or (not .Values.stream.url) (not .Values.stream.existingSecret) }}
{{- fail (printf "viewer.enabled=true requires stream.url and stream.existingSecret — this is the ONLY thing in the chart that makes LiveKit mandatory. The page lists the rooms on that server and signs each visitor a token with that server's own Secret, so without both it has nothing to show and nothing to sign with:\n\n  stream:\n    url: ws://livekit-clb.%s.svc.cluster.local:7880   # IN-CLUSTER; the backend uses this\n    existingSecret: livekit-auth                        # the SAME Secret the server reads\n\nThis chart does not deploy LiveKit. See \"Finding these values in your cluster\" in the chart README." .Release.Namespace) }}
{{- end }}
{{/*
No check on viewer.image.tag: empty means "follow image.tag", which is the common case since both
images come out of the same build. image.tag itself is already required, so an empty pair cannot
get through -- and the viewer keeps its own key for the times the two need to be pinned apart.
*/}}
{{- if not .Values.viewer.auth.existingSecret }}
{{- fail "viewer.auth.existingSecret is required. The viewer is published with HTTP Basic authentication in front of it and there is no way to turn that off -- a page that enumerates every running training and plays it must not be open. Create the Secret first, in the release namespace:\n  kubectl create secret generic isaaclab-viewer-auth -n <namespace> --from-literal=username=<user> --from-literal=password='<password>'\nThe chart deliberately cannot create it from values, because Helm keeps values in the release history where the password would stay readable." }}
{{- end }}
{{- if or (not .Values.viewer.auth.usernameKey) (not .Values.viewer.auth.passwordKey) }}
{{- fail "viewer.auth.usernameKey and viewer.auth.passwordKey are both required." }}
{{- end }}

{{/*
Refuse a credential passed through values instead of ignoring it. The template reads only
existingSecret, so a --set viewer.auth.password=... would otherwise take effect nowhere while
looking like it had -- and the operator would believe a password was in place that never was.
Failing here is also the only moment anyone is looking: by the time the page is up, an ignored
value is indistinguishable from a working one until someone tries the wrong password and gets in.
*/}}
{{- range $key := (list "username" "password" "passwd" "secret" "credentials" "htpasswd") }}
{{- if hasKey $.Values.viewer.auth $key }}
{{- fail (printf "viewer.auth.%s is not a supported value: credentials are only ever read from an existing Secret, never passed through values. Helm stores values verbatim in the release history, so anyone able to run `helm get values` would read the password back. Create the Secret first and name it in viewer.auth.existingSecret:\n  kubectl create secret generic isaaclab-viewer-auth -n <namespace> --from-literal=username=<user> --from-literal=password='<password>'" $key) }}
{{- end }}
{{- end }}
{{- if eq .Values.viewer.auth.usernameKey .Values.viewer.auth.passwordKey }}
{{- fail "viewer.auth.usernameKey and viewer.auth.passwordKey must be different." }}
{{- end }}
{{- if not (has .Values.viewer.service.type (list "LoadBalancer" "ClusterIP")) }}
{{- fail (printf "viewer.service.type must be \"LoadBalancer\" or \"ClusterIP\", got %q." .Values.viewer.service.type) }}
{{- end }}
{{- if eq .Values.viewer.service.type "LoadBalancer" }}
{{/*
The same pair, and the same two mistakes, as streaming.loadBalancer.

⚠️ Binding to a CLB the CCM itself created works in the DATA PATH -- listener and backend group
are built, the page is reachable -- but the CCM will not record ownership of it, so the Service
never gets .status.loadBalancer.ingress and reconcile fails every ~30s forever. That is a
deliberate trade, documented in values.yaml, not a fault. Nothing here can check an id's
provenance anyway; this only catches the shape.
*/}}
{{- if .Values.viewer.service.create }}
{{- if .Values.viewer.service.existingId }}
{{- fail "viewer.service.existingId must be empty when viewer.service.create=true. Set create=false to bind to a CLB created in the console or through the API, or leave existingId empty to provision one." }}
{{- end }}
{{- else }}
{{- if not .Values.viewer.service.existingId }}
{{- fail "viewer.service.existingId is required when viewer.service.create=false. Use the LiveKit server's CLB — port 80 does not collide with its 7880/7881/7882, so the page fits on it:\n\n  kubectl get svc <livekit-service> -n <livekit-namespace> --show-labels\n\nand read service.beta.kubernetes.io/volcengine-loadbalancer-id. Or set create=true to give the page a load balancer of its own." }}
{{- end }}
{{- end }}
{{- if not (has .Values.viewer.service.addressType (list "PUBLIC" "PRIVATE")) }}
{{- fail (printf "viewer.service.addressType must be \"PUBLIC\" or \"PRIVATE\", got %q." .Values.viewer.service.addressType) }}
{{- end }}
{{- end }}
{{/*
REQUIRED, always. The page has to be handed an address a BROWSER can reach, and only the operator
knows it: LiveKit sits on its own load balancer, which this release neither creates nor can be
pointed at (the CCM refuses to bind a Service to a CLB it created). Discovering it from the
viewer's own Service was possible only while the two shared one instance, and that arrangement
turned out not to be allowed -- so the lookup would now confidently return the WRONG address and
the page would fail to connect with nothing to read.
*/}}
{{- if not .Values.viewer.publicLivekitUrl }}
{{- fail "viewer.publicLivekitUrl is required when viewer.enabled=true: it is the address a BROWSER uses to reach LiveKit, and nothing in this release can derive it. The backend reaches LiveKit in-cluster through stream.url, but a browser cannot resolve a Service name -- it needs the LiveKit CLB's public address.\n\n  viewer:\n    publicLivekitUrl: ws://<livekit-clb-public-ip>:7880\n\nRead it from the LiveKit Service (see the chart README, \"Finding these values in your cluster\"):\n  kubectl get svc <livekit-service> -n <livekit-namespace> -o jsonpath='{.status.loadBalancer.ingress[0].ip}'\n\n⚠️ ws://, not https:// -- and the page must then be opened over http, since an https page may not open a plaintext ws:// connection." }}
{{- end }}
{{- if hasPrefix "https://" .Values.viewer.publicLivekitUrl }}
{{- fail "viewer.publicLivekitUrl is a LiveKit signalling URL, not a web address: use ws:// or wss://." }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Fail fast on load balancer settings that cannot work, so the error names the missing value
instead of surfacing as an init container that waits forever for a VIP.
*/}}
{{- define "isaaclab.streaming.validate" -}}
{{- if eq (include "isaaclab.isRender" .) "true" }}
{{- $lb := .Values.streaming.loadBalancer }}
{{- if $lb.create }}
{{- if $lb.existingId }}
{{- fail "streaming.loadBalancer.existingId must be empty when create=true. Set create=false to reuse an existing CLB, or leave existingId empty to provision a new one." }}
{{- end }}
{{- else }}
{{- if not $lb.existingId }}
{{- fail "streaming.loadBalancer.existingId is required when streaming.loadBalancer.create=false: set it to the CLB instance id (clb-...), or set create=true to provision a new one." }}
{{- end }}
{{- end }}
{{- if not (has $lb.addressType (list "PUBLIC" "PRIVATE")) }}
{{- fail (printf "streaming.loadBalancer.addressType must be \"PUBLIC\" or \"PRIVATE\", got %q." $lb.addressType) }}
{{- end }}
{{/*
Compared as strings on purpose: values from --set arrive as int64 while those from values.yaml
are int, and uniq compares with DeepEqual -- so 49100 from one source does not match 49100 from
the other, and the duplicate would slip through.
*/}}
{{- $ports := list (toString .Values.streaming.signallingPort) (toString .Values.streaming.nvcfPort) (toString .Values.streaming.mediaPort) }}
{{- if ne (len (uniq $ports)) 3 }}
{{- fail (printf "streaming.signallingPort, streaming.nvcfPort and streaming.mediaPort must all differ, got %s." (join ", " $ports)) }}
{{- end }}
{{- end }}
{{- end }}

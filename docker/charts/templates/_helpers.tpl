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

{{/*
Advisory hardware check. Returns warning text when nodeSelector pins an instance type that is
not listed for this mode, and nothing otherwise. NEVER fails: the lists ship empty and are
maintained by hand, so treating them as authoritative would block valid deployments.

Rendering needs a GPU with a display engine and the NVENC encoder the WebRTC stream feeds off;
training needs neither. A release that has to do both must sit on an instance type present in
both lists.
*/}}
{{- define "isaaclab.hardware.warning" -}}
{{- $key := .Values.hardware.instanceTypeKey }}
{{- $selected := "" }}
{{- with .Values.nodeSelector }}
{{- $selected = default "" (index . $key) }}
{{- end }}
{{- if $selected }}
{{- $render := eq (include "isaaclab.isRender" $) "true" }}
{{- $list := ternary .Values.hardware.renderInstanceTypes .Values.hardware.trainInstanceTypes $render }}
{{- if and $list (not (has $selected $list)) }}
nodeSelector pins {{ $key }}={{ $selected }}, which is not in hardware.{{ if $render }}render{{ else }}train{{ end }}InstanceTypes:
    {{ join ", " $list }}
  {{- if $render }}
  Rendering needs a GPU with a display engine and an NVENC encoder. If this instance type has
  neither, Isaac Sim starts but the stream never produces a picture.
  {{- else }}
  This instance type is not on the validated training list.
  {{- end }}
  This is advisory only — nothing was blocked. Update hardware.{{ if $render }}render{{ else }}train{{ end }}InstanceTypes
  if the list is out of date.
{{- end }}
{{- end }}
{{- end }}

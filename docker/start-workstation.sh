#!/usr/bin/env bash
# Start Isaac Sim as a WebRTC-streamed workstation, in the foreground.
#
# This is the container's main process in render mode: Isaac Sim is exec'd at the end so it
# inherits PID handling from tini, its logs go to stdout where `kubectl logs` picks them up, and
# a crash is visible to the orchestrator instead of leaving an idle container behind.
#
# WHY A PUBLIC ADDRESS IS REQUIRED
#
# WebRTC signalling only exchanges an SDP; the video itself travels over UDP. The server has to
# advertise a reachable address in that SDP as an ICE candidate, and all it knows on its own is
# the address its socket is bound to -- inside a container, a private pod IP. A client that
# receives such a candidate can never connect: signalling succeeds, the logs look healthy, and
# the picture never arrives because ICE stays in checking.
#
# So the address has to be supplied from outside. There is no way around this at the proxy
# layer: advertising an address is an application-level act, and a transparent proxy cannot
# rewrite the SDP body. Isaac Sim exposes no STUN/TURN settings either.
#
# Supply it with either:
#   ISAACSIM_PUBLIC_ENDPOINT       the address itself, or
#   ISAACSIM_PUBLIC_ENDPOINT_FILE  a file holding it (default /shared/public_ip)
#
# Under the Helm chart in docker/charts, an init container looks the load balancer's VIP up from
# the Kubernetes API and writes that file, so nothing has to be configured by hand.

set -euo pipefail

ISAACLAB_PATH="${ISAACLAB_PATH:-/workspace/isaaclab}"
ISAACSIM_ROOT="${ISAACSIM_ROOT:-${ISAACLAB_PATH}/_isaac_sim}"

# WebRTC signalling (TCP). What the streaming client connects to.
ISAACSIM_SIGNALLING_PORT="${ISAACSIM_SIGNALLING_PORT:-49100}"
# WebRTC media (UDP). Pinned rather than left to the default, which picks a port anywhere in
# 47998-48020 at session setup -- a range a load balancer would have to expose in full.
ISAACSIM_MEDIA_PORT="${ISAACSIM_MEDIA_PORT:-47998}"

ISAACSIM_PUBLIC_ENDPOINT="${ISAACSIM_PUBLIC_ENDPOINT:-}"
ISAACSIM_PUBLIC_ENDPOINT_FILE="${ISAACSIM_PUBLIC_ENDPOINT_FILE:-/shared/public_ip}"

log() { printf '[start-workstation] %s\n' "$*" >&2; }
die() { printf '[start-workstation] error: %s\n' "$*" >&2; exit 1; }

if [[ -z "${ISAACSIM_PUBLIC_ENDPOINT}" && -r "${ISAACSIM_PUBLIC_ENDPOINT_FILE}" ]]; then
    ISAACSIM_PUBLIC_ENDPOINT="$(tr -d '[:space:]' < "${ISAACSIM_PUBLIC_ENDPOINT_FILE}")"
    log "public endpoint ${ISAACSIM_PUBLIC_ENDPOINT} (from ${ISAACSIM_PUBLIC_ENDPOINT_FILE})"
else
    log "public endpoint ${ISAACSIM_PUBLIC_ENDPOINT:-<unset>} (from the environment)"
fi

if [[ -z "${ISAACSIM_PUBLIC_ENDPOINT}" ]]; then
    die "no public endpoint. Set ISAACSIM_PUBLIC_ENDPOINT, or provide it in
       ${ISAACSIM_PUBLIC_ENDPOINT_FILE} (ISAACSIM_PUBLIC_ENDPOINT_FILE).
       Under Helm this file is written by the wait-clb-ip init container -- check its log:
         kubectl logs <pod> -c wait-clb-ip"
fi

[[ -x "${ISAACSIM_ROOT}/isaac-sim.sh" ]] ||
    die "${ISAACSIM_ROOT}/isaac-sim.sh not found or not executable. Set ISAACSIM_ROOT."

# The container runs as root and Kit refuses to start that way without this.
export OMNI_KIT_ALLOW_ROOT=1

cd "${ISAACSIM_ROOT}"

log "starting Isaac Sim: signalling ${ISAACSIM_SIGNALLING_PORT}/tcp, media ${ISAACSIM_MEDIA_PORT}/udp"

# isaac-sim.streaming.sh is deliberately not used: its Kit app enables only the nvcf variant and
# opens no port. The extension is enabled here on top of the ordinary entrypoint instead.
#
# Note the UDP media port is bound lazily, at session setup rather than at startup -- an empty
# /proc/net/udp before the first client connects is expected, not a fault.
exec ./isaac-sim.sh \
    --no-window \
    --allow-root \
    --no-ros-env \
    --/app/livestream/publicEndpointAddress="${ISAACSIM_PUBLIC_ENDPOINT}" \
    --/app/livestream/port="${ISAACSIM_SIGNALLING_PORT}" \
    --/app/livestream/fixedHostPort="${ISAACSIM_MEDIA_PORT}" \
    --/app/livestream/allowDynamicResize=true \
    --enable omni.services.livestream.nvcf \
    "$@"

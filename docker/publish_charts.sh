#!/usr/bin/env bash
# Package the Isaac Lab Helm chart and push it to the Volcengine OCI registry.
#
#   bash docker/publish_charts.sh            # package, log in, push
#   DRY_RUN=1 bash docker/publish_charts.sh  # package only
#
# Everything deployment-specific comes from the environment, so this file holds no
# registry coordinates and no credentials:
#   HELM_REGISTRY_HOST       registry hostname, e.g. example.cr.volces.com
#   HELM_REGISTRY_NAMESPACE  namespace the chart is pushed under
#   HELM_REGISTRY_USERNAME   robot account      (not needed for DRY_RUN)
#   HELM_REGISTRY_PASSWORD   its password       (not needed for DRY_RUN)
#
# Optional:
#   HELM_VERSION             helm to fetch when the runner has none (default: latest)
#
# Nothing here needs root or a package manager. The CI image runs as nobody on a
# release old enough that its package sources are gone, so the only things
# assumed present are bash, curl, tar and diff.

set -euo pipefail

# Checked before anything else: a missing coordinate is a configuration error, and
# failing here costs nothing, whereas failing after the helm download wastes the
# whole setup. DRY_RUN needs these too, since it reports the push target.
: "${HELM_REGISTRY_HOST:?Set HELM_REGISTRY_HOST, the registry hostname (e.g. example.cr.volces.com).}"
: "${HELM_REGISTRY_NAMESPACE:?Set HELM_REGISTRY_NAMESPACE, the namespace the chart is pushed under.}"
registry="${HELM_REGISTRY_HOST}"
namespace="${HELM_REGISTRY_NAMESPACE}"

# Resolve from the script's own location so the working directory does not matter.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
chart_dir="docker/charts"

echo "=== environment ==="
echo "user: $(id -un 2>/dev/null || echo unknown) (uid $(id -u))"
echo "os:   $(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-unknown}" || uname -s) $(uname -m)"
echo "helm: $(command -v helm >/dev/null 2>&1 && helm version --short 2>&1 || echo MISSING)"
echo "curl: $(command -v curl || echo MISSING)"
echo "tar:  $(command -v tar || echo MISSING)"
echo "diff: $(command -v diff || echo MISSING)"
echo "==================="

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

# helm ships a statically linked binary, so unpacking it into the work directory
# needs no privileges — the only option available here, since the runner is
# nobody, sudo cannot elevate, and there is no usable package source.
if ! command -v helm >/dev/null 2>&1; then
  case "$(uname -m)" in
    x86_64 | amd64) helm_arch=amd64 ;;
    aarch64 | arm64) helm_arch=arm64 ;;
    *) echo "error: unsupported architecture $(uname -m)" >&2; exit 1 ;;
  esac
  helm_os="$(uname -s | tr '[:upper:]' '[:lower:]')"
  helm_version="${HELM_VERSION:-$(curl -fsSL --max-time 30 https://get.helm.sh/helm-latest-version 2>/dev/null || echo v4.2.4)}"
  helm_url="https://get.helm.sh/helm-${helm_version}-${helm_os}-${helm_arch}.tar.gz"

  echo "helm missing, fetching ${helm_url}"
  mkdir -p "${work}/helm"
  curl -fsSL --max-time 300 "${helm_url}" | tar -xz -C "${work}/helm" ||
    { echo "error: could not download helm from ${helm_url}" >&2; exit 1; }
  export PATH="${work}/helm/${helm_os}-${helm_arch}:${PATH}"

  command -v helm >/dev/null 2>&1 ||
    { echo "error: helm still not on PATH after unpacking" >&2; exit 1; }
  echo "helm ready: $(helm version --short 2>&1)"
fi

# Name and version come from Chart.yaml, which is also where helm reads them, so
# the package is simply whatever lands in this otherwise empty directory. That
# keeps the chart's identity in one place and this script out of the business of
# parsing YAML.
mkdir -p "${work}/pkg"
lint_args=(--set image.tag=ci-lint --set persistence.storageClass=ci-lint)

# `helm lint` checks the chart's shape. It does NOT fail on a template error -- helm 4 logs
# `funcMap fail` at INFO and still exits 0 -- so it cannot be the only gate here, or a chart that
# renders nothing at all would ship a green build.
helm lint "${chart_dir}" "${lint_args[@]}" --set mode=train

# `helm template` DOES fail, so this is the real check. Every shape the chart supports renders a
# different set of objects, and a template error in any one of them would otherwise ship
# unnoticed: render mode alone brings up the stream load balancer, its RBAC and an init
# container; the viewer brings up a second Deployment, a second Service and RBAC of its own;
# train mode without streaming leaves all of it out.
helm template lint-check "${chart_dir}" "${lint_args[@]}" >/dev/null
helm template lint-check "${chart_dir}" "${lint_args[@]}" \
  --set stream.url=ws://livekit-clb.default.svc.cluster.local:7880 \
  --set stream.existingSecret=livekit-auth >/dev/null
helm template lint-check "${chart_dir}" "${lint_args[@]}" \
  --set stream.url=ws://livekit-clb.default.svc.cluster.local:7880 \
  --set stream.existingSecret=livekit-auth \
  --set viewer.enabled=true --set viewer.publicLivekitUrl=ws://10.0.0.1:7880 >/dev/null
# Binding an existing CLB renders a different annotation set from provisioning one.
helm template lint-check "${chart_dir}" "${lint_args[@]}" \
  --set stream.url=ws://livekit-clb.default.svc.cluster.local:7880 \
  --set stream.existingSecret=livekit-auth \
  --set viewer.enabled=true --set viewer.publicLivekitUrl=ws://10.0.0.1:7880 \
  --set viewer.service.create=false --set viewer.service.existingId=clb-lint >/dev/null
helm template lint-check "${chart_dir}" "${lint_args[@]}" --set mode=render >/dev/null

helm package "${chart_dir}" --destination "${work}/pkg"
package="$(echo "${work}/pkg"/*.tgz)"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "dry run: would push ${package##*/} to oci://${registry}/${namespace}"
  exit 0
fi

: "${HELM_REGISTRY_USERNAME:?Set HELM_REGISTRY_USERNAME (registry robot account).}"
: "${HELM_REGISTRY_PASSWORD:?Set HELM_REGISTRY_PASSWORD.}"

# Keep the login in a throwaway config: the default ~/.config/helm would leave
# the robot credentials on a reused CI runner for the next job to find.
config="${work}/registry.json"
printf '%s' "${HELM_REGISTRY_PASSWORD}" |
  helm registry login "${registry}" --username "${HELM_REGISTRY_USERNAME}" \
    --password-stdin --registry-config "${config}"

# A published version is immutable: nothing here overwrites one, and there is no flag that lets
# it. helm push would replace an existing tag without complaint, leaving two different charts
# answering to one number with no way to tell them apart afterwards.
#
# The check is on content rather than on presence, because republishing the SAME chart under the
# SAME version is what CI does on every commit that did not touch the chart, and that has to stay
# a no-op.
#
# Name and version come from the package helm just built, so they cannot drift from what is
# about to be pushed.
chart_meta="$(helm show chart "${package}")"
name="$(awk '$1 == "name:" { print $2; exit }' <<< "${chart_meta}")"
version="$(awk '$1 == "version:" { print $2; exit }' <<< "${chart_meta}")"
ref="oci://${registry}/${namespace}/${name}"

if helm show chart "${ref}" --version "${version}" --registry-config "${config}" >/dev/null 2>&1; then
  # Without diff there is no way to tell a rebuild from a real conflict, and guessing either way
  # is worse than saying so: guessing "identical" silently drops a changed chart, guessing
  # "different" fails every rebuild.
  command -v diff >/dev/null 2>&1 || {
    echo "error: ${name} ${version} is already published and diff is missing, so its content" >&2
    echo "       cannot be compared. Install diffutils on the runner." >&2
    exit 1
  }

  # Compare extracted trees rather than digests: `helm package` records file mtimes, so
  # repackaging the same source yields a different byte stream and a different digest.
  cmp_dir="${work}/cmp"
  mkdir -p "${cmp_dir}/remote" "${cmp_dir}/local"
  if helm pull "${ref}" --version "${version}" -d "${cmp_dir}/remote" \
       --registry-config "${config}" >/dev/null 2>&1 &&
     tar -xzf "${cmp_dir}/remote"/*.tgz -C "${cmp_dir}/remote" &&
     tar -xzf "${package}" -C "${cmp_dir}/local" &&
     diff -r "${cmp_dir}/remote/${name}" "${cmp_dir}/local/${name}" >/dev/null 2>&1; then
    echo "${name} ${version} is already published and identical - nothing to do."
    exit 0
  fi

  echo "error: ${name} ${version} is already in the registry, with DIFFERENT content." >&2
  echo "       A published version is never replaced. Bump version: in" >&2
  echo "       ${chart_dir}/Chart.yaml and push that instead." >&2
  exit 1
fi

# helm push reports the pushed reference and its digest on success.
helm push "${package}" "oci://${registry}/${namespace}" --registry-config "${config}"

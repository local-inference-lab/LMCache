#!/usr/bin/env bash
# Build a source-locked LMCache wheel for the CUDA 13.4 serving runtime.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tool_dir="${repo_root}/ci/lil_wheels"
lock_path="${tool_dir}/runtime.lock"
output_dir=${1:-"${repo_root}/dist/lil-lmcache-wheel"}

lock_value() {
  local key=$1
  awk -F= -v key="${key}" '$1 == key {sub(/^[^=]*=/, ""); print; found=1} END {exit !found}' \
    "${lock_path}"
}

source_commit=$(git -C "${repo_root}" rev-parse HEAD)
source_tree=$(git -C "${repo_root}" rev-parse 'HEAD^{tree}')
source_date_epoch=$(git -C "${repo_root}" show -s --format=%ct HEAD)
repository=${GITHUB_REPOSITORY:-local-inference-lab/LMCache}
release_tag=${LMCACHE_RELEASE_TAG:-"lmcache-cu134-sm120-beta-${source_commit}"}
builder=$(lock_value buildx.builder)
test -z "$(git -C "${repo_root}" status --porcelain)"

mkdir -p "$(dirname "${output_dir}")"
if ! mkdir "${output_dir}"; then
  printf 'Output path already exists: %s\n' "${output_dir}" >&2
  exit 1
fi
mkdir -p "${output_dir}/raw" "${output_dir}/bundle/wheels"

docker buildx build \
  --builder "${builder}" \
  --file "${tool_dir}/Dockerfile" \
  --build-arg "BUILDER_IMAGE=$(lock_value builder.image)" \
  --build-arg "UV_IMAGE=$(lock_value uv.image)" \
  --build-arg "UV_SHA256=$(lock_value uv.sha256)" \
  --build-arg "SOURCE_COMMIT=${source_commit}" \
  --build-arg "SOURCE_DATE_EPOCH=${source_date_epoch}" \
  --target export \
  --output "type=local,dest=${output_dir}/raw" \
  "${repo_root}"
cp -a "${output_dir}/raw/wheels/." "${output_dir}/bundle/wheels/"

wheel=$(find "${output_dir}/bundle/wheels" -maxdepth 1 -name 'lmcache-*.whl' -print -quit)
test -n "${wheel}"
metadata=$(unzip -p "${wheel}" '*/METADATA')
package_name=$(awk -F': ' '$1 == "Name" {print $2; exit}' <<<"${metadata}")
package_version=$(awk -F': ' '$1 == "Version" {print $2; exit}' <<<"${metadata}")
test "${package_name}" = lmcache
wheel_listing=$(unzip -Z1 "${wheel}")
grep -Eq '^lmcache/cuda_ops\.[^/]*\.so$' <<<"${wheel_listing}"
digest=$(sha256sum "${wheel}" | awk '{print $1}')
file=$(basename "${wheel}")
url="https://github.com/${repository}/releases/download/${release_tag}/${file}"
printf '%s @ %s --hash=sha256:%s\n' "${package_name}" "${url}" "${digest}" \
  > "${output_dir}/bundle/requirements-github.txt"

jq -n \
  --arg status research-only \
  --arg repository "https://github.com/${repository}.git" \
  --arg commit "${source_commit}" \
  --arg tree "${source_tree}" \
  --arg package_version "${package_version}" \
  --arg release_tag "${release_tag}" \
  --arg file "${file}" \
  --arg sha256 "${digest}" \
  --arg url "${url}" \
  --arg builder_image "$(lock_value builder.image)" \
  --arg python "$(lock_value python.version)" \
  --arg cuda "$(lock_value cuda.version)" \
  --arg pytorch "$(lock_value pytorch.version)" \
  --arg pytorch_commit "$(lock_value pytorch.commit)" \
  --arg cuda_arch_list "$(lock_value cuda.arch-list)" \
  '{schema: "local-inference-lmcache-wheel-release/v1", status: $status,
    scope: "LMCache Python sources and native CUDA extensions for the declared ABI",
    source: {repository: $repository, commit: $commit, tree: $tree},
    package_version: $package_version, release_tag: $release_tag,
    runtime: {builder_image: $builder_image, python: $python, cuda: $cuda,
      pytorch: $pytorch, pytorch_commit: $pytorch_commit,
      cuda_arch_list: $cuda_arch_list},
    packages: [{name: "lmcache", version: $package_version, file: $file,
      sha256: $sha256, url: $url}]}' \
  > "${output_dir}/bundle/manifest.json"

cp "${lock_path}" "${tool_dir}/install.sh" "${output_dir}/bundle/"
chmod 0755 "${output_dir}/bundle/install.sh"
(
  cd "${output_dir}/bundle"
  find wheels -maxdepth 1 -name '*.whl' -print0 | sort -z | xargs -0 sha256sum
  sha256sum manifest.json requirements-github.txt runtime.lock install.sh
) > "${output_dir}/bundle/SHA256SUMS"

archive="${output_dir}/lmcache-cu134-sm120-${source_commit}.tar.zst"
tar --sort=name --mtime="@${source_date_epoch}" --owner=0 --group=0 \
  --numeric-owner --zstd -C "${output_dir}/bundle" -cf "${archive}" .
(cd "${output_dir}" && sha256sum "$(basename "${archive}")") \
  > "${archive}.sha256"
cat > "${output_dir}/release-notes.md" <<EOF
Status: **research-only**

This release contains LMCache ${package_version} from source commit
\`${source_commit}\` for the declared Python 3.12, CUDA 13.4.1, NVIDIA PyTorch
26.08, and C++11 ABI runtime. CUDA, PyTorch, and external dependencies are not
included.
EOF

printf '%s\n' "${output_dir}/bundle"

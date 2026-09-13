#!/usr/bin/env bash
# Install the source-locked LMCache wheel into a compatible existing venv.
set -euo pipefail

bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_path=${1:?Pass the destination venv path}
uv_binary=${UV_BIN:-uv}

(cd "${bundle_dir}" && sha256sum --check SHA256SUMS)
"${uv_binary}" pip install \
  --python "${venv_path}/bin/python" \
  --no-deps \
  --require-hashes \
  -r "${bundle_dir}/requirements-github.txt"
"${venv_path}/bin/python" -c 'import lmcache; import lmcache.cuda_ops; print("lmcache_install=PASS")'

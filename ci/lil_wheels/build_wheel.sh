#!/usr/bin/env bash
# Build and normalize LMCache native extensions for the declared foundation.
set -euo pipefail

mkdir -p /wheelhouse
base_version=$(/build-venv/bin/python -m setuptools_scm | tr -d '[:space:]')
base_version=${base_version%%+*}
package_version="${base_version}+lil.cu134.g${SOURCE_COMMIT:0:12}"

env -u PYTHONPATH \
  SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE="${package_version}" \
  uv build \
    --wheel \
    --no-build-isolation \
    --python /build-venv/bin/python \
    --out-dir /wheelhouse \
    /src/lmcache

test "$(find /wheelhouse -maxdepth 1 -name 'lmcache-*.whl' | wc -l)" -eq 1
/build-venv/bin/python ci/lil_wheels/normalize_wheel.py \
  --wheel /wheelhouse/lmcache-*.whl \
  --torch-version 2.14.0a0+4fdf77b940.nv26.8.63802676 \
  --source-date-epoch "${SOURCE_DATE_EPOCH:?}"

/build-venv/bin/python - <<'PY'
from pathlib import Path
from zipfile import ZipFile

wheel = next(Path("/wheelhouse").glob("lmcache-*.whl"))
with ZipFile(wheel) as archive:
    cuda_extensions = [
        name
        for name in archive.namelist()
        if name.startswith("lmcache/cuda_ops.") and name.endswith(".so")
    ]
if len(cuda_extensions) != 1:
    raise RuntimeError(
        "LMCache wheel must contain exactly one lmcache.cuda_ops extension"
    )
print(f"lmcache_cuda_extension={cuda_extensions[0]}")
PY

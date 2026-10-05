# syntax=docker/dockerfile:1.7

ARG UV_IMAGE
ARG BUILDER_IMAGE
FROM ${UV_IMAGE} AS uv
FROM ${BUILDER_IMAGE} AS wheels

ARG BUILD_JOBS=12
ARG SOURCE_COMMIT
ARG SOURCE_DATE_EPOCH
ARG UV_SHA256
ARG TORCH_CUDA_ARCH_LIST=12.0a
ARG CMAKE_CUDA_ARCHITECTURES=120a
ARG BUILD_REQUIREMENTS=ci/lil_wheels/build-requirements.lock
# Empty for linux/amd64 so its existing caches stay valid.
ARG CACHE_PLATFORM=
ENV PATH=/build-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} \
    BUILD_WITH_CUDA=1 \
    LMCACHE_CUDA_MAJOR=13 \
    ENABLE_CXX11_ABI=1 \
    MAX_JOBS=${BUILD_JOBS} \
    NVCC_THREADS=1

COPY --from=uv /uv /usr/local/bin/uv
COPY ${BUILD_REQUIREMENTS} /tmp/build-requirements.lock
RUN test "$(sha256sum /usr/local/bin/uv | awk '{print $1}')" = "${UV_SHA256}" \
    && uv venv --python /usr/bin/python --system-site-packages /build-venv \
    && uv pip install --python /build-venv/bin/python \
      --require-hashes --no-deps -r /tmp/build-requirements.lock

COPY . /src/lmcache
WORKDIR /src/lmcache
RUN /build-venv/bin/python - <<'PY'
import importlib.metadata
import torch

assert importlib.metadata.version("torch") == (
    "2.14.0a0+4fdf77b940.nv26.8.63802676"
)
assert torch.version.cuda == "13.4"
assert torch._C._GLIBCXX_USE_CXX11_ABI
PY
RUN --mount=type=cache,id=lil-lmcache-native-cu134-torch214-sm120${CACHE_PLATFORM},target=/src/lmcache/build,sharing=locked \
    --mount=type=cache,id=lil-wheel-uv,target=/root/.cache/uv,sharing=locked \
    SOURCE_COMMIT="${SOURCE_COMMIT}" \
    SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH}" \
    ci/lil_wheels/build_wheel.sh

FROM scratch AS export
COPY --from=wheels /wheelhouse/ /wheels/

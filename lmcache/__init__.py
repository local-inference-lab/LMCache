# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import site
import sys

# This source overlay carries patched Python modules but not wheel-built native
# extensions. Keep source modules first and use the matching installed package
# only as a fallback for submodules such as lmcache.c_ops.
_LMCACHE_BINARY_PACKAGE_PATHS = tuple(
    str(Path(site_dir) / "lmcache")
    for site_dir in site.getsitepackages()
    if (Path(site_dir) / "lmcache").is_dir()
)
for binary_package_path in _LMCACHE_BINARY_PACKAGE_PATHS:
    if binary_package_path not in __path__:
        __path__.append(binary_package_path)

# First Party
from lmcache.logging import init_logger

# --------------------------
# Backend instance & Device detection
# --------------------------
from lmcache.v1.platform import get_backend
from lmcache.v1.platform import torch_dev as torch_dev
from lmcache.v1.platform import torch_device_type as torch_device_type

try:
    # First Party
    from lmcache._version import __version__
except ImportError:
    __version__ = "unknown"

logger = init_logger(__name__)

__all__ = ["__version__", "torch_dev", "torch_device_type"]

_ops = get_backend(torch_device_type)
if _ops is not None:
    # Override lmcache.c_ops with merged module,
    # in which:
    #     python_ops_fallback as base,
    #     use backend implementation if exists
    sys.modules["lmcache.c_ops"] = _ops
else:
    logger.warning(
        "No compute backend loaded; CLI-only mode (torch/numba not installed)"
    )

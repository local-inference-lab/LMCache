# SPDX-License-Identifier: Apache-2.0
"""Pre-import bootstrap for the ``lmcache`` console script."""

# Standard
import os
import sys


def _reexec_cpu_only_server() -> None:
    """Hide CUDA before importing the LMCache package for a CPU-only server."""
    if (
        sys.argv[1:2] == ["server"]
        and "--cpu-only" in sys.argv[2:]
        and os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    ):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        os.execvpe(
            sys.executable,
            [sys.executable, *sys.argv],
            env,
        )
        raise RuntimeError("CPU-only server preflight re-exec returned unexpectedly")


def main() -> None:
    """Run the LMCache CLI after applying pre-import process settings."""
    _reexec_cpu_only_server()

    # Import only after the CPU-only preflight: importing ``lmcache`` performs
    # accelerator detection in the package initializer.
    # First Party
    from lmcache.cli.main import main as lmcache_main

    lmcache_main()

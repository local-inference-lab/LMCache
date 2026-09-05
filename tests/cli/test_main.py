# SPDX-License-Identifier: Apache-2.0
"""Process-order tests for the top-level LMCache CLI."""

# Standard
from unittest.mock import patch
import os
import subprocess
import sys

# First Party
from lmcache_cli_bootstrap import _reexec_cpu_only_server


def test_cpu_only_reexec_precedes_lmcache_package_import() -> None:
    script = r"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.argv = [
    "lmcache",
    "server",
    "--cpu-only",
    "--supported-transfer-mode",
    "engine_driven",
    "--l1-size-gb",
    "1",
    "--eviction-policy",
    "LRU",
]


def fake_execvpe(executable, argv, env):
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert "lmcache" not in sys.modules
    raise SystemExit(73)


os.execvpe = fake_execvpe
import lmcache_cli_bootstrap

try:
    lmcache_cli_bootstrap.main()
except SystemExit as exc:
    if exc.code != 73:
        raise
else:
    raise AssertionError("CPU-only preflight did not request re-exec")
print("cpu-only preflight ran before LMCache package import")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "cpu-only preflight ran before LMCache package import" in result.stdout


def test_cpu_only_preflight_is_idempotent_when_cuda_is_hidden(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["lmcache", "server", "--cpu-only"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    with patch("lmcache_cli_bootstrap.os.execvpe") as mock_exec:
        _reexec_cpu_only_server()

    mock_exec.assert_not_called()


def test_cpu_only_preflight_ignores_non_server_commands(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["lmcache", "status", "--cpu-only"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    with patch("lmcache_cli_bootstrap.os.execvpe") as mock_exec:
        _reexec_cpu_only_server()

    mock_exec.assert_not_called()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"

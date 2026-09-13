# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for incomplete community wheel payloads."""

# Standard
from pathlib import Path
import hashlib
import zipfile

# Third Party
import pytest

# First Party
from ci.lil_wheels.verify_community_source import verify_python_payload


@pytest.mark.parametrize("failure", [None, "missing", "modified", "native"])
def test_wheel_preserves_committed_modules(tmp_path: Path, failure: str | None) -> None:
    module = "lmcache/v1/multiprocess/checkpoint_storage.py"
    payload = b"checkpoint_contract = True\n"
    receipt = {"files": {module: hashlib.sha256(payload).hexdigest()}}
    wheel = tmp_path / "lmcache.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        if failure != "missing":
            archive.writestr(module, b"wrong" if failure == "modified" else payload)
        if failure != "native":
            archive.writestr("lmcache/cuda_ops.cpython-312-x86_64-linux-gnu.so", b"ELF")
    if failure is None:
        assert verify_python_payload(wheel, receipt) == 1
    else:
        with pytest.raises(ValueError, match="wheel (omits|changes)"):
            verify_python_payload(wheel, receipt)

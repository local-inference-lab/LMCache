# SPDX-License-Identifier: Apache-2.0
"""Tests for immutable LMCache release verification."""

# Standard
from pathlib import Path
import hashlib
import json

# Third Party
import pytest

# First Party
from ci.lil_wheels.verify_release_assets import verify_release


def make_release(directory: Path) -> None:
    wheel = directory / "lmcache-1-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "local-inference-lmcache-wheel-release/v1",
                "source": {"commit": "abc"},
                "release_tag": "beta-abc",
                "packages": [
                    {
                        "file": wheel.name,
                        "sha256": hashlib.sha256(b"wheel").hexdigest(),
                    }
                ],
            }
        )
    )


def test_verify_release_accepts_matching_assets(tmp_path: Path) -> None:
    make_release(tmp_path)
    verify_release(tmp_path, "abc", "beta-abc")


def test_verify_release_rejects_modified_wheel(tmp_path: Path) -> None:
    make_release(tmp_path)
    (tmp_path / "lmcache-1-py3-none-any.whl").write_bytes(b"modified")
    with pytest.raises(ValueError, match="wheel digest mismatch"):
        verify_release(tmp_path, "abc", "beta-abc")

"""Tests for the CUDA 13.4 LMCache wheel normalizer."""

from email.parser import BytesParser
from email.policy import compat32

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from ci.lil_wheels.normalize_wheel import rewrite_requirements


def test_rewrite_requirements_pins_torch() -> None:
    metadata = (
        b"Metadata-Version: 2.4\n"
        b"Name: lmcache\n"
        b"Requires-Dist: torch\n"
        b"Requires-Dist: aiohttp\n\n"
    )
    rewritten = rewrite_requirements(metadata, torch_version="2.14.0a0+nv")
    message = BytesParser(policy=compat32).parsebytes(rewritten)
    requirements = {
        canonicalize_name(req.name): str(req)
        for req in map(Requirement, message.get_all("Requires-Dist", []))
    }
    assert requirements["torch"] == "torch==2.14.0a0+nv"
    assert requirements["aiohttp"] == "aiohttp"
    assert message["X-Local-Inference-Runtime"] == "jovian-cu134-torch214-cxx11"


def test_rewrite_requirements_rejects_missing_torch() -> None:
    metadata = b"Metadata-Version: 2.4\nName: lmcache\nRequires-Dist: aiohttp\n\n"
    try:
        rewrite_requirements(metadata, torch_version="2.14.0a0+nv")
    except ValueError as error:
        assert "torch is missing" in str(error)
    else:
        raise AssertionError("missing torch dependency was accepted")

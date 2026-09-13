#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bind an LMCache wheel to the CUDA 13.4 foundation ABI."""

# Future
from __future__ import annotations

# Standard
from email.parser import BytesParser
from email.policy import compat32
from pathlib import Path
import argparse
import binascii
import csv
import hashlib
import stat
import subprocess
import tempfile
import time
import zipfile

# Third Party
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def rewrite_requirements(metadata: bytes, *, torch_version: str) -> bytes:
    """Pin the native extension to the exact Torch distribution ABI."""
    message = BytesParser(policy=compat32).parsebytes(metadata)
    requirements = message.get_all("Requires-Dist", [])
    rewritten: list[str] = []
    found_torch = False
    for value in requirements:
        requirement = Requirement(value)
        if canonicalize_name(requirement.name) == "torch":
            rewritten.append(f"torch=={torch_version}")
            found_torch = True
        else:
            rewritten.append(value)
    if not found_torch:
        raise ValueError("LMCache dependency contract changed: torch is missing")

    del message["Requires-Dist"]
    for requirement in rewritten:
        message["Requires-Dist"] = requirement
    message["X-Local-Inference-Runtime"] = "jovian-cu134-torch214-cxx11"
    return message.as_bytes(policy=compat32.clone(max_line_length=0))


def portable_rpath(relative_path: Path) -> str:
    """Resolve Torch, CUDA, and NCCL from the destination site-packages."""
    levels = len(relative_path.parent.parts)
    site_packages = "/".join(".." for _ in range(levels))
    prefix = "$ORIGIN" if not site_packages else f"$ORIGIN/{site_packages}"
    return ":".join(
        (
            "$ORIGIN",
            f"{prefix}/torch/lib",
            f"{prefix}/nvidia/cu13/lib",
            f"{prefix}/nvidia/cudnn/lib",
            f"{prefix}/nvidia/nvshmem/lib",
            f"{prefix}/local_inference_nccl/lib",
        )
    )


def patch_elf_runpaths(root: Path) -> dict[str, str]:
    """Replace builder paths on every packaged LMCache ELF object."""
    runpaths: dict[str, str] = {}
    for path in sorted((root / "lmcache").rglob("*")):
        if not path.is_file():
            continue
        with path.open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                continue
        relative = path.relative_to(root)
        rpath = portable_rpath(relative)
        subprocess.run(["patchelf", "--set-rpath", rpath, str(path)], check=True)
        actual = subprocess.run(
            ["patchelf", "--print-rpath", str(path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if actual != rpath or "/usr/local" in actual:
            raise ValueError(f"non-portable RPATH for {relative}: {actual}")
        runpaths[str(relative)] = actual
    if not runpaths:
        raise ValueError("LMCache wheel contains no native ELF payload")
    return runpaths


def record_digest(path: Path) -> str:
    """Return a URL-safe wheel RECORD digest."""
    digest = hashlib.sha256(path.read_bytes()).digest()
    encoded = binascii.b2a_base64(digest, newline=False)
    encoded = encoded.rstrip(b"=").replace(b"+", b"-").replace(b"/", b"_")
    return "sha256=" + encoded.decode()


def write_record(root: Path, record: Path) -> None:
    """Regenerate RECORD after metadata and ELF normalization."""
    rows: list[tuple[str, str, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != record:
            rows.append(
                (
                    path.relative_to(root).as_posix(),
                    record_digest(path),
                    str(path.stat().st_size),
                )
            )
    rows.append((record.relative_to(root).as_posix(), "", ""))
    with record.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)


def pack_wheel(root: Path, output: Path, source_date_epoch: int) -> None:
    """Write a deterministic wheel archive."""
    timestamp = time.gmtime(max(source_date_epoch, 315532800))[:6]
    temporary = output.with_suffix(".normalized.whl")
    with zipfile.ZipFile(
        temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(path.relative_to(root).as_posix(), timestamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (
                stat.S_IMODE(path.stat().st_mode) | stat.S_IFREG
            ) << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    temporary.replace(output)


def normalize_wheel(
    wheel: Path, *, torch_version: str, source_date_epoch: int
) -> dict[str, str]:
    """Normalize one LMCache wheel in place and return its ELF runpaths."""
    with tempfile.TemporaryDirectory(prefix="lmcache-wheel-normalize-") as directory:
        root = Path(directory)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(root)
        metadata_files = list(root.glob("*.dist-info/METADATA"))
        record_files = list(root.glob("*.dist-info/RECORD"))
        if len(metadata_files) != 1 or len(record_files) != 1:
            raise ValueError("wheel must contain exactly one METADATA and RECORD")
        metadata_files[0].write_bytes(
            rewrite_requirements(
                metadata_files[0].read_bytes(), torch_version=torch_version
            )
        )
        runpaths = patch_elf_runpaths(root)
        write_record(root, record_files[0])
        pack_wheel(root, wheel, source_date_epoch)
    return runpaths


def main() -> None:
    """Parse the foundation contract and normalize one wheel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--torch-version", required=True)
    parser.add_argument("--source-date-epoch", type=int, required=True)
    args = parser.parse_args()
    runpaths = normalize_wheel(
        args.wheel,
        torch_version=args.torch_version,
        source_date_epoch=args.source_date_epoch,
    )
    for path, rpath in runpaths.items():
        print(f"lmcache_elf={path} rpath={rpath}")


if __name__ == "__main__":
    main()

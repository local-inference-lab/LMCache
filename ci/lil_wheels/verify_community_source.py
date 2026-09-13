# SPDX-License-Identifier: Apache-2.0
"""Verify community ancestry and byte-complete Python wheel packaging."""

# Standard
from pathlib import Path
from typing import Any
import argparse
import hashlib
import json
import subprocess
import zipfile


def git(root: Path, *args: str) -> bytes:
    """Run a read-only Git query and return its output, failing on errors."""
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True
    ).stdout


def source_receipt(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    """Require attributed review ancestry and record committed source hashes.

    Hash equality proves packaging fidelity, not behavioral correctness of
    changes after the reference commit. Runtime contract tests remain required.
    """
    if contract.get("schema") != "local-inference-lmcache-community-source/v1":
        raise ValueError("unknown LMCache community source contract")
    for number, commit in contract["required_reviews"].items():
        result = subprocess.run(
            ["git", "-C", str(root), "merge-base", "--is-ancestor", commit, "HEAD"],
            capture_output=True,
        )
        if result.returncode:
            raise ValueError(f"community source is missing attributed PR #{number}")
    names = git(
        root, "ls-tree", "-rz", "--name-only", "HEAD", "lmcache", "csrc", "rust"
    )
    files = [name.decode() for name in names.split(b"\0") if name]
    missing = set(contract["required_files"]) - set(files)
    if missing:
        raise ValueError(
            f"community source is missing required files: {sorted(missing)}"
        )
    hashes = {
        name: hashlib.sha256(git(root, "show", f"HEAD:{name}")).hexdigest()
        for name in files
    }
    return {
        "schema": "local-inference-lmcache-source-receipt/v1",
        "source_commit": git(root, "rev-parse", "HEAD").decode().strip(),
        "source_tree": git(root, "rev-parse", "HEAD^{tree}").decode().strip(),
        "required_reviews": contract["required_reviews"],
        "required_files": contract["required_files"],
        "files": hashes,
    }


def verify_python_payload(wheel: Path, receipt: dict[str, Any]) -> int:
    """Require every tracked LMCache Python module unchanged inside the wheel."""
    expected = {
        name: digest
        for name, digest in receipt["files"].items()
        if name.startswith("lmcache/") and name.endswith(".py")
    }
    if not expected:
        raise ValueError("source receipt contains no Python modules")
    with zipfile.ZipFile(wheel) as archive:
        for name, digest in expected.items():
            if name not in archive.namelist():
                raise ValueError(f"wheel omits committed Python module: {name}")
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"wheel changes committed Python module: {name}")
        if not any(
            name.startswith("lmcache/cuda_ops.") and name.endswith(".so")
            for name in archive.namelist()
        ):
            raise ValueError("wheel omits the native CUDA extension")
    return len(expected)


def main() -> None:
    """Produce a source receipt and optionally validate a compiled wheel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    receipt = source_receipt(args.root, json.loads(args.contract.read_text()))
    if args.wheel:
        receipt["verified_python_modules"] = verify_python_payload(args.wheel, receipt)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

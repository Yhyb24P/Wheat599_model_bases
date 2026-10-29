#!/usr/bin/env python3
"""Fetch the pinned BGLR wheat.RData and verify its SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/gdlc/BGLR-R/"
    "7e92aa2d835c6fef13c5f0e062863204964605f1/data/wheat.RData"
)
SHA256 = "8710523389007dd8b7c88a5deb74ee16aea0bae158496bc11f0626fb43b5ea21"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/raw"),
        help="directory for wheat.RData (default: data/raw)",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = parser.parse_args()

    target = args.out_dir / "wheat.RData"
    if target.exists() and not args.force:
        actual = sha256_of(target)
        if actual == SHA256:
            print(f"OK {target} sha256={actual}")
            return 0
        print(
            f"Refusing to overwrite {target}: sha256={actual} != {SHA256}. "
            "Use --force to replace.",
            file=sys.stderr,
        )
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL, timeout=120) as response:
        payload = response.read()

    target.write_bytes(payload)
    actual = sha256_of(target)
    if actual != SHA256:
        print(f"SHA-256 mismatch: expected {SHA256}, got {actual}", file=sys.stderr)
        return 1

    (args.out_dir / "SHA256SUMS").write_text(f"{actual}  wheat.RData\n")
    print(f"OK {target} ({len(payload)} bytes) sha256={actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

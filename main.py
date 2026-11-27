#!/usr/bin/env python3
"""Copy bundled entry.png to the output directory and print its SHA-256."""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path


def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    src_path = base_dir / "entry.png"

    if not src_path.is_file():
        print(f"Error: Input file not found: {src_path}", file=sys.stderr)
        sys.exit(1)

    out_dir_str = os.environ.get("MAKEFUN_OUTPUT_DIR")
    out_dir = Path(out_dir_str) if out_dir_str else Path("/output")
    out_dir.mkdir(parents=True, exist_ok=True)

    dst_path = out_dir / "entry.png"
    shutil.copyfile(src_path, dst_path)

    digest = compute_sha256(dst_path)
    print(f"Copied to {dst_path} (sha256: {digest})")


if __name__ == "__main__":
    main()

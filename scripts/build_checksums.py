#!/usr/bin/env python3
"""Generate deterministic SHA-256 checksums for release files."""

from __future__ import annotations

import hashlib
from pathlib import Path


root = Path(__file__).resolve().parent.parent
destination = root / "SHA256SUMS"
paths = sorted(
    path for path in root.rglob("*")
    if path.is_file() and path != destination and ".git" not in path.parts
)
lines = []
for path in paths:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {len(lines)} checksums to {destination}")

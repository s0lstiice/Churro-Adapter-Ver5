#!/usr/bin/env python3
"""Build a resumable, portable CHURRO OCR shard on an external drive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from universal_progress_monitor.progress_client import ProgressTask


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_space(path: Path, reserve_mb: int, incoming: int = 0) -> None:
    free = shutil.disk_usage(path).free - incoming
    if free < reserve_mb * 1024 * 1024:
        raise RuntimeError(
            f"D: reserve reached: {max(0, free) // (1024 * 1024)} MiB would remain; "
            f"required reserve is {reserve_mb} MiB"
        )


def copy_atomic(source: Path, destination: Path, reserve_mb: int) -> None:
    if destination.is_file() and destination.stat().st_size == source.stat().st_size:
        if sha256(destination) == sha256(source):
            return
    destination.parent.mkdir(parents=True, exist_ok=True)
    require_space(destination.parent, reserve_mb, source.stat().st_size)
    temporary = destination.with_suffix(destination.suffix + ".part")
    shutil.copy2(source, temporary)
    if sha256(temporary) != sha256(source):
        temporary.unlink(missing_ok=True)
        raise IOError(f"copy verification failed: {source}")
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--pages", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--canonical-adapter-label", required=True)
    parser.add_argument("--minimum-free-mb", type=int, default=20480)
    args = parser.parse_args()

    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    require_space(destination, args.minimum_free_mb)
    runtime_destination = destination / "runtime"
    shutil.copytree(args.runtime.resolve(), runtime_destination, dirs_exist_ok=True)
    shutil.copy2(Path(__file__).resolve().parent / "merge_portable_region_predictions.py", runtime_destination / "scripts" / "merge_portable_region_predictions.py")
    shutil.copy2(
        Path(__file__).resolve().parent / "REMOTE_SECOND_COMPUTER_INSTRUCTIONS.md",
        destination / "SECOND_COMPUTER_INSTRUCTIONS.md",
    )

    regions = read_jsonl(args.regions)
    portable = []
    unique_images: dict[str, Path] = {}
    for row in regions:
        source = Path(str(row["image"])).resolve()
        key = sha256(source)
        suffix = source.suffix.lower() or ".jpg"
        relative = Path("data") / "images" / f"{key[:24]}{suffix}"
        unique_images[key] = source
        portable.append({**row, "inference_image": relative.as_posix()})

    with ProgressTask(
        "Copy remote LOC OCR shard to D",
        total=len(unique_images),
        unit="images",
        task_id="loc-two-host-copy-remote-shard",
        output_dir=destination,
        metadata={"minimum_free_mb": args.minimum_free_mb},
    ) as task:
        for index, (key, source) in enumerate(sorted(unique_images.items()), start=1):
            target = destination / "data" / "images" / f"{key[:24]}{source.suffix.lower() or '.jpg'}"
            copy_atomic(source, target, args.minimum_free_mb)
            task.update(index, message=source.name, metrics={"free_mb": shutil.disk_usage(destination).free // (1024 * 1024)})

    write_jsonl(destination / "data" / "regions.jsonl", portable)
    shutil.copy2(args.regions, destination / "data" / "canonical_regions.jsonl")
    shutil.copy2(args.pages, destination / "data" / "pages.jsonl")
    launch = """#!/usr/bin/env bash
set -euo pipefail
ROOT=\"$(cd \"$(dirname \"$0\")\" && pwd)\"
PYTHON=\"${PYTHON:-/home/vgu/aiproj/.venv/bin/python}\"
cd \"$ROOT\"
\"$PYTHON\" runtime/scripts/evaluate_churro_fullpage_qlora.py \\
  --manifest data/regions.jsonl \\
  --model stanford-oval/churro-3B \\
  --adapter runtime/adapter \\
  --adapter-label __CANONICAL_ADAPTER__ \\
  --output output/region_recognition \\
  --decode-profile grounded-faithful \\
  --max-pixels 1605632 \\
  --max-new-tokens 3072 \\
  --continuous-page-budget \\
  --selective-incomplete-retry \\
  --max-incomplete-retries 1
\"$PYTHON\" runtime/scripts/merge_portable_region_predictions.py \\
  --regions data/regions.jsonl \\
  --predictions output/region_recognition/predictions.jsonl \\
  --output output/page_drafts.jsonl
""".replace("__CANONICAL_ADAPTER__", args.canonical_adapter_label)
    (destination / "RUN_REMOTE_WSL.sh").write_text(launch, encoding="utf-8", newline="\n")
    powershell = r'''$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$linuxRoot = (& wsl.exe -d Ubuntu-24.04 wslpath -a $root)
if (-not $linuxRoot) { throw "Could not convert the transfer-folder path for WSL." }
& wsl.exe -d Ubuntu-24.04 --cd $linuxRoot bash RUN_REMOTE_WSL.sh
exit $LASTEXITCODE
'''
    (destination / "RUN_REMOTE.ps1").write_text(powershell, encoding="utf-8-sig")
    readme = f"""# Remote half of the LOC Epoch 22 transcription run

This folder contains {len(regions):,} provenance-bound OCR regions plus the Epoch 22
adapter and runtime. The CHURRO base weights are not redistributed; the first run
downloads `stanford-oval/churro-3B` from Hugging Face.

From PowerShell run:

```powershell
.\\RUN_REMOTE.ps1
```

Progress is resumable in `output/region_recognition/predictions.jsonl`. After it
finishes, copy that file back to the primary computer. `output/page_drafts.jsonl`
is provided for human viewing; the primary computer performs the final strict
provenance merge. See `SECOND_COMPUTER_INSTRUCTIONS.md` for environment setup,
GitHub fallback, monitoring, resuming, and the exact result file to return.
"""
    (destination / "README.md").write_text(readme, encoding="utf-8")
    summary = {
        "regions": len(regions),
        "unique_images": len(unique_images),
        "destination": str(destination),
        "minimum_free_mb": args.minimum_free_mb,
        "canonical_adapter_label": args.canonical_adapter_label,
    }
    (destination / "PACKAGE_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run layout normalization, region OCR, and page-order reconstruction."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BASE_MODEL = "stanford-oval/churro-3B"


def run(*parts: object) -> None:
    command = [sys.executable, *(str(part) for part in parts)]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--backend", choices=("doctr", "geometry"), default="doctr")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-pixels", type=int, default=1_605_632)
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()

    scripts = Path(__file__).resolve().parent
    bundle = scripts.parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    adapter = (args.adapter or bundle / "adapter").resolve()
    manifest = args.manifest.resolve()

    normalized = output / "normalized_pages.jsonl"
    image_dir = output / "normalized_images"
    audit = output / "layout_audit.jsonl"
    regions = output / "ocr_regions.jsonl"
    recognition = output / "region_recognition"
    merged = output / "predictions.jsonl"

    normalize_command: list[object] = [
        scripts / "normalize_document_layouts.py",
        "--manifest", manifest,
        "--output-manifest", normalized,
        "--image-dir", image_dir,
        "--audit", audit,
        "--backend", args.backend,
        "--batch-size", args.batch_size,
    ]
    if args.limit is not None:
        normalize_command.extend(("--limit", args.limit))
    if args.skip_missing:
        normalize_command.append("--skip-missing")
    run(*normalize_command)

    run(
        scripts / "expand_layout_regions_manifest.py",
        "--manifest", normalized,
        "--output", regions,
        "--skip-nonmeaningful",
    )
    run(
        scripts / "evaluate_churro_fullpage_qlora.py",
        "--manifest", regions,
        "--output", recognition,
        "--model", args.model,
        "--adapter", adapter,
        "--decode-profile", "grounded-faithful",
        "--max-pixels", args.max_pixels,
        "--max-new-tokens", args.max_new_tokens,
        "--continuous-page-budget",
        "--selective-incomplete-retry",
        "--max-incomplete-retries", 1,
    )
    run(
        scripts / "merge_layout_region_predictions.py",
        "--page-manifest", normalized,
        "--region-manifest", regions,
        "--region-predictions", recognition / "predictions.jsonl",
        "--output", merged,
        "--expected-model", args.model,
        "--expected-adapter", adapter,
        "--expected-decode-profile", "grounded-faithful",
    )
    print(f"Finished: {merged}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Expand layout-normalized pages into provenance-bound OCR-region rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from universal_progress_monitor.progress_client import ProgressTask


PROVENANCE_SCHEMA = "layout-region-provenance-v1"


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row is not an object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_fingerprint(path: Path, cache: dict[str, str] | None = None) -> str:
    resolved = str(path.expanduser().resolve())
    if cache is not None and resolved in cache:
        return cache[resolved]
    digest = hashlib.sha256()
    try:
        with Path(resolved).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"cannot fingerprint layout source {resolved!r}: {exc}") from exc
    value = digest.hexdigest()
    if cache is not None:
        cache[resolved] = value
    return value


def identity(row: dict) -> str:
    value = row.get("page_id") or row.get("id")
    if value is None and row.get("image"):
        value = Path(str(row["image"])).stem
    if value is None or not str(value).strip():
        raise ValueError("manifest row has no non-empty page identity")
    return str(value)


def assert_unique(values: list[str], label: str) -> None:
    duplicates = sorted(key for key, count in Counter(values).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate {label}: {duplicates[:20]}")


def source_path(page: dict) -> Path:
    value = (
        page.get("root_original_image")
        or page.get("original_image")
        or page.get("layout_input_image")
        or page.get("image")
    )
    if not value:
        raise ValueError(f"page {identity(page)!r} has no source image")
    return Path(str(value))


def ordered_regions(page: dict) -> list[dict]:
    parent_id = identity(page)
    supplied = page.get("layout_ocr_regions") or []
    if not isinstance(supplied, list):
        raise ValueError(f"page {parent_id!r} layout_ocr_regions is not a list")
    regions = list(supplied)
    if not regions:
        regions = [
            {
                "id": parent_id,
                "parent_id": parent_id,
                "reading_order": 0,
                "region": "full",
                "image": page["image"],
                "meaningful_content": True,
            }
        ]
    for index, region in enumerate(regions):
        if not isinstance(region, dict):
            raise ValueError(f"page {parent_id!r} region {index} is not an object")
        if not region.get("image"):
            raise ValueError(f"page {parent_id!r} region {index} has no image")
        if region.get("parent_id") is not None and str(region["parent_id"]) != parent_id:
            raise ValueError(
                f"page {parent_id!r} region {index} has mismatched parent_id "
                f"{region['parent_id']!r}"
            )
    indexed = sorted(
        enumerate(regions),
        key=lambda pair: (int(pair[1].get("reading_order", 0)), pair[0]),
    )
    prepared: list[dict] = []
    for sorted_index, (_, region) in enumerate(indexed):
        value = dict(region)
        logical_id = str(
            region.get("id") or f"{parent_id}__layout_region_{sorted_index:02d}"
        )
        if not logical_id.strip():
            raise ValueError(f"page {parent_id!r} region {sorted_index} has an empty ID")
        value["_logical_id"] = logical_id
        prepared.append(value)
    assert_unique(
        [str(region["_logical_id"]) for region in prepared],
        f"logical layout-region IDs on page {parent_id!r}",
    )
    return prepared


def page_fingerprints(
    page: dict,
    regions: list[dict],
    file_cache: dict[str, str] | None = None,
) -> tuple[str, str]:
    parent_id = identity(page)
    source = source_path(page)
    source_digest = file_fingerprint(source, file_cache)
    region_layout = []
    for region in regions:
        image = Path(str(region["image"]))
        region_layout.append(
            {
                "logical_id": str(region["_logical_id"]),
                "reading_order": int(region.get("reading_order", 0)),
                "region": region.get("region"),
                "source_box": region.get("source_box"),
                "output_box": region.get("output_box"),
                "rotation_ccw": region.get("rotation_ccw"),
                "meaningful_content": bool(region.get("meaningful_content", True)),
                "image": str(image.expanduser().resolve()),
                "image_sha256": file_fingerprint(image, file_cache),
            }
        )
    layout_digest = canonical_hash(
        {
            "schema": PROVENANCE_SCHEMA,
            "parent_id": parent_id,
            "source_image": str(source.expanduser().resolve()),
            "source_sha256": source_digest,
            "normalization_version": page.get("layout_normalization_version"),
            "normalization": page.get("layout_normalization"),
            "regions": region_layout,
        }
    )
    return source_digest, layout_digest


def expansion_provenance(
    page: dict,
    include_nonmeaningful: bool,
    file_cache: dict[str, str] | None = None,
) -> tuple[list[dict], dict]:
    regions = ordered_regions(page)
    selected = [
        region
        for region in regions
        if include_nonmeaningful or bool(region.get("meaningful_content", True))
    ]
    # Never silently turn a source page into zero OCR inputs.
    if not selected:
        selected = regions
    source_digest, layout_digest = page_fingerprints(page, regions, file_cache)
    selection = [str(region["_logical_id"]) for region in selected]
    config = {
        "schema": PROVENANCE_SCHEMA,
        "include_nonmeaningful": include_nonmeaningful,
        "fallback_to_all_when_filter_empty": True,
        "selected_logical_region_ids": selection,
    }
    expansion_digest = canonical_hash(
        {
            "parent_id": identity(page),
            "source_sha256": source_digest,
            "layout_sha256": layout_digest,
            "config": config,
        }
    )
    return selected, {
        "schema": PROVENANCE_SCHEMA,
        "parent_id": identity(page),
        "source_sha256": source_digest,
        "layout_sha256": layout_digest,
        "expansion_sha256": expansion_digest,
        "config": config,
    }


def expand(
    rows: list[dict],
    include_nonmeaningful: bool = True,
    progress: ProgressTask | None = None,
) -> list[dict]:
    assert_unique([identity(page) for page in rows], "source page IDs")
    logical_ids = [
        str(region["_logical_id"])
        for page in rows
        for region in ordered_regions(page)
    ]
    assert_unique(logical_ids, "logical layout-region IDs across pages")
    expanded: list[dict] = []
    file_cache: dict[str, str] = {}
    for page_number, page in enumerate(rows, start=1):
        parent_id = identity(page)
        selected, page_provenance = expansion_provenance(
            page, include_nonmeaningful, file_cache
        )
        for region_index, region in enumerate(selected):
            row = dict(page)
            logical_id = str(region["_logical_id"])
            clean_region = {
                key: value for key, value in region.items() if key != "_logical_id"
            }
            region_digest = canonical_hash(
                {
                    "page": page_provenance,
                    "logical_id": logical_id,
                    "reading_order": int(region.get("reading_order", 0)),
                    "region_image_sha256": file_fingerprint(
                        Path(str(region["image"])), file_cache
                    ),
                }
            )
            region_id = f"{logical_id}__provenance_{region_digest[:20]}"
            provenance = {
                **page_provenance,
                "logical_region_id": logical_id,
                "region_sha256": region_digest,
                "bound_region_id": region_id,
            }
            for key in (
                "generation_max_new_tokens",
                "generation_max_incomplete_retries",
                "generation_limit_reason",
            ):
                row.pop(key, None)
            row.update(
                {
                    "id": region_id,
                    "page_id": region_id,
                    "layout_parent_id": parent_id,
                    "layout_region_logical_id": logical_id,
                    "layout_region_index": region_index,
                    "layout_region_count": len(selected),
                    "layout_region": clean_region,
                    "layout_region_provenance": provenance,
                    "image": str(region["image"]),
                    "text": "",
                    "reference_available": False,
                    "official_transcript_available": False,
                    "eligible_for_supervised_training": False,
                }
            )
            for key in (
                "generation_max_new_tokens",
                "generation_max_incomplete_retries",
                "generation_limit_reason",
            ):
                if region.get(key) is not None:
                    row[key] = region[key]
            expanded.append(row)
        if progress is not None:
            progress.update(
                page_number,
                message=parent_id,
                metrics={"ocr_regions": len(expanded)},
            )
    assert_unique([str(row["id"]) for row in expanded], "layout-region IDs")
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-nonmeaningful",
        action="store_true",
        help="Skip dark/blank regions, while retaining at least one region per page.",
    )
    args = parser.parse_args()
    pages = read_jsonl(args.manifest)
    with ProgressTask(
        "Bind layout regions to source provenance",
        total=len(pages),
        unit="pages",
        task_id=f"expand-layout-regions-{args.output.stem}",
        output_dir=args.output.parent,
    ) as task:
        rows = expand(
            pages,
            include_nonmeaningful=not args.skip_nonmeaningful,
            progress=task,
        )
    write_jsonl(args.output, rows)
    summary = {
        "provenance_schema": PROVENANCE_SCHEMA,
        "pages": len(pages),
        "ocr_regions": len(rows),
        "extra_region_inferences": len(rows) - len(pages),
        "skip_nonmeaningful": args.skip_nonmeaningful,
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

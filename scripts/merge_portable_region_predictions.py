#!/usr/bin/env python3
"""Create readable page drafts from a portable region OCR shard."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    regions = read_jsonl(args.regions)
    predictions = read_jsonl(args.predictions)
    expected = {str(row["id"]): row for row in regions}
    if len(expected) != len(regions):
        raise ValueError("duplicate region IDs")
    by_id = {str(row["id"]): row for row in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("duplicate prediction IDs")
    unknown = sorted(set(by_id) - set(expected))
    if unknown:
        raise ValueError(f"predictions contain unknown region IDs: {unknown[:10]}")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for region in regions:
        prediction = by_id.get(str(region["id"]))
        if prediction is None:
            continue
        for key in ("layout_parent_id", "layout_region_index", "layout_region_provenance", "image"):
            if prediction.get(key) != region.get(key):
                raise ValueError(f"prediction {region['id']} changed bound field {key}")
        grouped[str(region["layout_parent_id"])].append(prediction)

    pages = []
    for parent, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row.get("layout_region_index", 0)))
        pages.append(
            {
                "id": parent,
                "page_id": parent,
                "prediction": "\n".join(
                    str(row.get("prediction") or "").strip() for row in rows
                    if str(row.get("prediction") or "").strip()
                ),
                "regions_completed": len(rows),
                "regions_expected": int(rows[0].get("layout_region_count") or len(rows)),
                "model_prediction_is_not_ground_truth": True,
                "human_review_required": True,
            }
        )
    write_jsonl(args.output, pages)
    print(f"Wrote {len(pages)} partial/complete page drafts to {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge provenance-checked layout-region OCR results into source-page order."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from expand_layout_regions_manifest import (
    PROVENANCE_SCHEMA,
    assert_unique,
    canonical_hash,
    expand,
    identity,
)
from universal_progress_monitor.progress_client import ProgressTask


CONFIG_KEYS = ("model", "adapter", "decode_profile", "generation_settings")
MANIFEST_BOUND_KEYS = (
    "id",
    "page_id",
    "layout_parent_id",
    "layout_region_logical_id",
    "layout_region_index",
    "layout_region_count",
    "layout_region",
    "layout_region_provenance",
    "image",
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
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


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prediction_identity(row: dict) -> str:
    value = row.get("id") or row.get("page_id")
    if value is None or not str(value).strip():
        raise ValueError("prediction row has no non-empty id/page_id")
    return str(value)


def expected_region_ids(page: dict) -> list[str]:
    """Legacy helper for callers that do not have a bound region manifest."""
    parent_id = identity(page)
    regions = list(page.get("layout_ocr_regions") or [])
    if not regions:
        return [parent_id]
    return [
        str(region.get("id") or f"{parent_id}__layout_region_{index:02d}")
        for index, region in enumerate(
            sorted(regions, key=lambda value: int(value.get("reading_order", 0)))
        )
    ]


def prediction_has_meaningful_text(prediction: dict, expected: dict | None) -> bool:
    meaningful = True
    if expected is not None:
        region = expected.get("layout_region") or {}
        meaningful = bool(region.get("meaningful_content", True))
    return not meaningful or bool(str(prediction.get("prediction") or "").strip())


def merge_one_page(
    page: dict,
    region_ids: list[str],
    by_id: dict[str, dict],
    expected_rows_by_id: dict[str, dict] | None = None,
) -> dict | None:
    if not region_ids:
        raise ValueError(f"page {identity(page)} has no expected OCR regions")
    if any(region_id not in by_id for region_id in region_ids):
        return None
    region_rows = [by_id[region_id] for region_id in region_ids]
    expected_rows = expected_rows_by_id or {}
    if any(
        not prediction_has_meaningful_text(row, expected_rows.get(region_id))
        for region_id, row in zip(region_ids, region_rows)
    ):
        return None
    parent_id = identity(page)
    output = dict(page)
    texts = [str(row.get("prediction") or "").strip() for row in region_rows]
    output.update(
        {
            "id": parent_id,
            "page_id": parent_id,
            "prediction": "\n".join(text for text in texts if text),
            "raw_prediction": None,
            "raw_prediction_regions": [row.get("raw_prediction") for row in region_rows],
            "model": region_rows[0].get("model"),
            "adapter": region_rows[0].get("adapter"),
            "decode_profile": region_rows[0].get("decode_profile"),
            "generation_incomplete": any(
                bool(row.get("generation_incomplete")) for row in region_rows
            ),
            "generation_incomplete_reasons": sorted(
                {
                    str(reason)
                    for row in region_rows
                    for reason in row.get("generation_incomplete_reasons") or []
                }
            ),
            "layout_region_results": region_rows,
            "layout_regions_expected": len(region_ids),
            "layout_regions_completed": len(region_rows),
            "layout_region_provenance_schema": PROVENANCE_SCHEMA,
            "recognition_strategy": "orientation_normalize_region_ocr_reading_order_merge",
            "metrics": None,
            "model_prediction_is_not_ground_truth": True,
            "human_review_required": True,
        }
    )
    return output


def merge_complete_pages(
    pages: list[dict],
    predictions: list[dict],
    expected_by_parent: dict[str, list[str]] | None = None,
    expected_rows_by_id: dict[str, dict] | None = None,
) -> tuple[list[dict], int]:
    prediction_ids = [prediction_identity(row) for row in predictions]
    assert_unique(prediction_ids, "prediction IDs")
    by_id = dict(zip(prediction_ids, predictions))
    merged: list[dict] = []
    expected_total = 0
    for page in pages:
        parent_id = identity(page)
        region_ids = (
            expected_by_parent.get(parent_id, [])
            if expected_by_parent is not None
            else expected_region_ids(page)
        )
        expected_total += len(region_ids)
        output = merge_one_page(page, region_ids, by_id, expected_rows_by_id)
        if output is not None:
            merged.append(output)
    return merged, expected_total


def inference_config(row: dict) -> dict:
    missing = [key for key in CONFIG_KEYS if key not in row]
    if missing:
        raise ValueError(
            f"prediction {prediction_identity(row)!r} lacks inference config fields: {missing}"
        )
    if not str(row.get("model") or "").strip():
        raise ValueError(f"prediction {prediction_identity(row)!r} has an empty model")
    if not str(row.get("decode_profile") or "").strip():
        raise ValueError(
            f"prediction {prediction_identity(row)!r} has an empty decode_profile"
        )
    if not isinstance(row.get("generation_settings"), dict):
        raise ValueError(
            f"prediction {prediction_identity(row)!r} generation_settings is not an object"
        )
    return {key: row.get(key) for key in CONFIG_KEYS}


def load_bound_region_manifest(
    pages: list[dict], region_manifest: Path
) -> tuple[dict[str, list[str]], dict[str, dict]]:
    assert_unique([identity(page) for page in pages], "source page IDs")
    supplied = read_jsonl(region_manifest)
    supplied_ids = [prediction_identity(row) for row in supplied]
    assert_unique(supplied_ids, "region manifest IDs")
    if not supplied:
        raise ValueError("region manifest is empty")
    include_values: set[bool] = set()
    for row in supplied:
        provenance = row.get("layout_region_provenance")
        if not isinstance(provenance, dict):
            raise ValueError(
                f"region {prediction_identity(row)!r} lacks layout_region_provenance"
            )
        if provenance.get("schema") != PROVENANCE_SCHEMA:
            raise ValueError(
                f"region {prediction_identity(row)!r} has unsupported provenance schema"
            )
        config = provenance.get("config")
        if not isinstance(config, dict) or "include_nonmeaningful" not in config:
            raise ValueError(
                f"region {prediction_identity(row)!r} lacks expansion configuration"
            )
        include_values.add(bool(config["include_nonmeaningful"]))
    if len(include_values) != 1:
        raise ValueError("region manifest mixes expansion configurations")
    include_nonmeaningful = include_values.pop()
    regenerated = expand(pages, include_nonmeaningful=include_nonmeaningful)
    regenerated_by_id = {str(row["id"]): row for row in regenerated}
    if set(supplied_ids) != set(regenerated_by_id):
        missing = sorted(set(regenerated_by_id) - set(supplied_ids))[:10]
        extra = sorted(set(supplied_ids) - set(regenerated_by_id))[:10]
        raise ValueError(
            "region manifest does not match current source/layout/config; "
            f"missing={missing}, extra={extra}"
        )
    by_parent: dict[str, list[dict]] = defaultdict(list)
    supplied_by_id = dict(zip(supplied_ids, supplied))
    for region_id, expected in regenerated_by_id.items():
        actual = supplied_by_id[region_id]
        actual_bound = {key: actual.get(key) for key in MANIFEST_BOUND_KEYS}
        expected_bound = {key: expected.get(key) for key in MANIFEST_BOUND_KEYS}
        if canonical_hash(actual_bound) != canonical_hash(expected_bound):
            raise ValueError(
                f"region manifest provenance/config mismatch for {region_id!r}"
            )
        by_parent[str(actual["layout_parent_id"])].append(actual)
    page_ids = {identity(page) for page in pages}
    if set(by_parent) != page_ids:
        raise ValueError("region manifest parent IDs do not match the page manifest")
    expected_by_parent: dict[str, list[str]] = {}
    for parent_id, rows in by_parent.items():
        ordered = sorted(rows, key=lambda row: int(row["layout_region_index"]))
        indices = [int(row["layout_region_index"]) for row in ordered]
        if indices != list(range(len(ordered))):
            raise ValueError(f"page {parent_id!r} has invalid region indices {indices}")
        if any(int(row["layout_region_count"]) != len(ordered) for row in ordered):
            raise ValueError(f"page {parent_id!r} has inconsistent region counts")
        expected_by_parent[parent_id] = [str(row["id"]) for row in ordered]
    return expected_by_parent, supplied_by_id


class IncrementalJSONLReader:
    """Read only appended complete JSONL records and retain a partial tail."""

    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.pending = b""
        self.file_key: tuple[int, int] | None = None
        self.anchor_start = 0
        self.anchor_digest: str | None = None

    def _anchor(self, handle: Any, offset: int) -> tuple[int, str]:
        start = max(0, offset - 4096)
        handle.seek(start)
        return start, hashlib.sha256(handle.read(offset - start)).hexdigest()

    def _reset_needed(self, handle: Any, stat: os.stat_result) -> bool:
        key = (int(stat.st_dev), int(stat.st_ino))
        if self.file_key is not None and key != self.file_key:
            return True
        if stat.st_size < self.offset:
            return True
        if self.anchor_digest is not None and self.offset:
            handle.seek(self.anchor_start)
            current = hashlib.sha256(
                handle.read(self.offset - self.anchor_start)
            ).hexdigest()
            if current != self.anchor_digest:
                return True
        return False

    def read_new(self, final: bool = False) -> tuple[list[dict], bool]:
        if not self.path.is_file():
            if final:
                raise ValueError(f"prediction file does not exist: {self.path}")
            return [], False
        reset = False
        with self.path.open("rb") as handle:
            stat = os.fstat(handle.fileno())
            if self._reset_needed(handle, stat):
                self.offset = 0
                self.pending = b""
                self.anchor_digest = None
                reset = True
            self.file_key = (int(stat.st_dev), int(stat.st_ino))
            handle.seek(self.offset)
            chunk = handle.read()
            self.offset += len(chunk)
            self.anchor_start, self.anchor_digest = self._anchor(handle, self.offset)
        data = self.pending + chunk
        lines = data.splitlines(keepends=True)
        self.pending = b""
        if lines and not lines[-1].endswith((b"\n", b"\r")):
            self.pending = lines.pop()
        if final and self.pending.strip():
            lines.append(self.pending)
            self.pending = b""
        rows: list[dict] = []
        for raw in lines:
            if not raw.strip():
                continue
            try:
                value = json.loads(raw.decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid complete JSONL row in {self.path}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row in {self.path} is not an object")
            rows.append(value)
        return rows, reset


class PredictionAccumulator:
    def __init__(
        self,
        pages: list[dict],
        expected_by_parent: dict[str, list[str]],
        expected_rows_by_id: dict[str, dict],
        expected_model: str | None = None,
        expected_adapter: str | None = None,
        expected_decode_profile: str | None = None,
    ):
        self.pages = pages
        self.pages_by_id = {identity(page): page for page in pages}
        self.expected_by_parent = expected_by_parent
        self.expected_rows_by_id = expected_rows_by_id
        self.region_parent = {
            region_id: parent_id
            for parent_id, region_ids in expected_by_parent.items()
            for region_id in region_ids
        }
        self.expected_model = expected_model
        self.expected_adapter = expected_adapter
        self.expected_decode_profile = expected_decode_profile
        self.by_id: dict[str, dict] = {}
        self.merged_by_parent: dict[str, dict] = {}
        self.config: dict | None = None

    def clear(self) -> None:
        self.by_id.clear()
        self.merged_by_parent.clear()
        self.config = None

    def _validate_prediction(self, row: dict) -> str:
        region_id = prediction_identity(row)
        if region_id in self.by_id:
            raise ValueError(f"duplicate prediction ID: {region_id!r}")
        if region_id not in self.expected_rows_by_id:
            raise ValueError(f"unexpected/stale prediction ID: {region_id!r}")
        expected = self.expected_rows_by_id[region_id]
        actual_provenance = row.get("layout_region_provenance")
        if canonical_hash(actual_provenance) != canonical_hash(
            expected["layout_region_provenance"]
        ):
            raise ValueError(f"prediction provenance mismatch for {region_id!r}")
        for key in (
            "layout_parent_id",
            "layout_region_logical_id",
            "layout_region_index",
            "layout_region_count",
            "image",
        ):
            if row.get(key) != expected.get(key):
                raise ValueError(
                    f"prediction manifest field {key!r} mismatch for {region_id!r}"
                )
        config = inference_config(row)
        if self.config is None:
            self.config = config
        elif canonical_hash(config) != canonical_hash(self.config):
            raise ValueError(f"prediction inference config mismatch for {region_id!r}")
        expectations = {
            "model": self.expected_model,
            "adapter": self.expected_adapter,
            "decode_profile": self.expected_decode_profile,
        }
        for key, wanted in expectations.items():
            if wanted is not None and config[key] != wanted:
                raise ValueError(
                    f"prediction {key} mismatch for {region_id!r}: "
                    f"expected {wanted!r}, got {config[key]!r}"
                )
        return region_id

    def add(self, row: dict) -> bool:
        region_id = self._validate_prediction(row)
        self.by_id[region_id] = row
        parent_id = self.region_parent[region_id]
        merged = merge_one_page(
            self.pages_by_id[parent_id],
            self.expected_by_parent[parent_id],
            self.by_id,
            self.expected_rows_by_id,
        )
        previously_complete = parent_id in self.merged_by_parent
        if merged is None:
            self.merged_by_parent.pop(parent_id, None)
        else:
            self.merged_by_parent[parent_id] = merged
        return previously_complete != (merged is not None)

    def ordered_merged(self) -> list[dict]:
        return [
            self.merged_by_parent[parent_id]
            for parent_id in (identity(page) for page in self.pages)
            if parent_id in self.merged_by_parent
        ]

    def incomplete_details(self) -> tuple[list[str], list[str]]:
        missing: list[str] = []
        empty_meaningful: list[str] = []
        for region_id, expected in self.expected_rows_by_id.items():
            if region_id not in self.by_id:
                missing.append(region_id)
            elif not prediction_has_meaningful_text(self.by_id[region_id], expected):
                empty_meaningful.append(region_id)
        return missing, empty_meaningful


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-manifest", type=Path, required=True)
    parser.add_argument(
        "--region-manifest",
        type=Path,
        required=True,
        help="Expanded region manifest used to verify source/layout provenance.",
    )
    parser.add_argument("--region-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=3.0)
    parser.add_argument("--done-file", type=Path)
    parser.add_argument(
        "--ready-file",
        type=Path,
        help="Atomically publish after the initial prediction scan and output rewrite.",
    )
    parser.add_argument("--expected-model")
    parser.add_argument("--expected-adapter")
    parser.add_argument("--expected-decode-profile")
    args = parser.parse_args()
    if args.done_file is not None and not args.watch:
        parser.error("--done-file requires --watch")
    pages = read_jsonl(args.page_manifest)
    expected_by_parent, expected_rows_by_id = load_bound_region_manifest(
        pages, args.region_manifest
    )
    accumulator = PredictionAccumulator(
        pages,
        expected_by_parent,
        expected_rows_by_id,
        expected_model=args.expected_model,
        expected_adapter=args.expected_adapter,
        expected_decode_profile=args.expected_decode_profile,
    )
    reader = IncrementalJSONLReader(args.region_predictions)
    task_id = f"merge-layout-regions-{args.output.parent.name}"
    predictions_seen = 0
    merged: list[dict] = []
    expected_regions = len(expected_rows_by_id)
    output_initialized = False
    first_iteration = True
    with ProgressTask(
        "Merge layout-region OCR into pages",
        total=len(pages),
        unit="pages",
        task_id=task_id,
        output_dir=args.output.parent,
    ) as task:
        while True:
            producer_done = bool(args.done_file and args.done_file.is_file())
            new_rows, reset = reader.read_new(final=not args.watch or producer_done)
            if reset:
                accumulator.clear()
                predictions_seen = 0
            changed = reset or first_iteration
            for row in new_rows:
                predictions_seen += 1
                changed = accumulator.add(row) or changed
            current = accumulator.ordered_merged()
            # The merged file is derived from this exact bound manifest.  Clear
            # a stale page-level result on the first validated snapshot, even
            # when no region has completed yet.
            if changed or not output_initialized:
                write_jsonl(args.output, current)
                output_initialized = True
            if first_iteration and args.ready_file is not None:
                write_json(
                    args.ready_file,
                    {
                        "pid": os.getpid(),
                        "region_predictions": predictions_seen,
                        "merged_pages": len(current),
                    },
                )
            first_iteration = False
            merged = current
            missing, empty_meaningful = accumulator.incomplete_details()
            task.update(
                len(merged),
                message=(
                    "all pages merged"
                    if len(merged) == len(pages)
                    else "waiting for valid OCR regions"
                ),
                metrics={
                    "region_predictions": predictions_seen,
                    "expected_regions": expected_regions,
                    "missing_regions": len(missing),
                    "empty_meaningful_regions": len(empty_meaningful),
                },
            )
            if len(merged) == len(pages) or not args.watch:
                break
            if producer_done:
                raise RuntimeError(
                    "OCR producer finished before merge completion; "
                    f"missing={missing[:10]}, empty_meaningful={empty_meaningful[:10]}"
                )
            time.sleep(max(0.2, args.poll_seconds))
    summary = {
        "provenance_schema": PROVENANCE_SCHEMA,
        "pages": len(pages),
        "merged_pages": len(merged),
        "region_predictions": predictions_seen,
        "expected_regions": expected_regions,
        "complete": len(merged) == len(pages),
        "output": str(args.output.resolve()),
    }
    write_json(args.output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

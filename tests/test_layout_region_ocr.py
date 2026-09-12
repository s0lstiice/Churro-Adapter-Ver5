from __future__ import annotations

import tempfile
from pathlib import Path

from expand_layout_regions_manifest import expand
from merge_layout_region_predictions import merge_complete_pages


def page(root: Path | None = None) -> dict:
    root = root or Path("/tmp")
    composite = root / "composite.jpg"
    left = root / "left.jpg"
    right = root / "right.jpg"
    if root != Path("/tmp"):
        composite.write_bytes(b"composite-image")
        left.write_bytes(b"left-image")
        right.write_bytes(b"right-image")
    return {
        "id": "page-1",
        "page_id": "page-1",
        "image": str(composite),
        "layout_ocr_regions": [
            {
                "id": "page-1__layout_region_000",
                "image": str(left),
                "reading_order": 0,
                "meaningful_content": True,
            },
            {
                "id": "page-1__layout_region_001",
                "image": str(right),
                "reading_order": 1,
                "meaningful_content": False,
                "generation_max_new_tokens": 128,
                "generation_max_incomplete_retries": 0,
            },
        ],
    }


def test_expand_preserves_region_reading_order_and_parent() -> None:
    with tempfile.TemporaryDirectory() as directory:
        source = page(Path(directory))
        rows = expand([source])
        assert rows[0]["id"].startswith("page-1__layout_region_000__provenance_")
        assert rows[1]["id"].startswith("page-1__layout_region_001__provenance_")
        assert all(row["layout_parent_id"] == "page-1" for row in rows)
        assert [row["image"] for row in rows] == [
            source["layout_ocr_regions"][0]["image"],
            source["layout_ocr_regions"][1]["image"],
        ]
        assert rows[1]["generation_max_new_tokens"] == 128
        assert rows[1]["generation_max_incomplete_retries"] == 0


def test_merge_waits_for_every_region() -> None:
    prediction = {
        "id": "page-1__layout_region_000",
        "prediction": "first half",
    }
    merged, expected = merge_complete_pages([page()], [prediction])
    assert expected == 2
    assert merged == []


def test_merge_concatenates_regions_in_page_order() -> None:
    predictions = [
        {
            "id": "page-1__layout_region_001",
            "prediction": "second half",
            "model": "model",
            "adapter": "adapter",
            "decode_profile": "faithful",
            "generation_incomplete": False,
        },
        {
            "id": "page-1__layout_region_000",
            "prediction": "first half",
            "model": "model",
            "adapter": "adapter",
            "decode_profile": "faithful",
            "generation_incomplete": False,
        },
    ]
    merged, expected = merge_complete_pages([page()], predictions)
    assert expected == 2
    assert len(merged) == 1
    assert merged[0]["prediction"] == "first half\nsecond half"
    assert merged[0]["recognition_strategy"] == "orientation_normalize_region_ocr_reading_order_merge"

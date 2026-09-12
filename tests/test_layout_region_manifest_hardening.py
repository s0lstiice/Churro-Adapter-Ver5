import json
import tempfile
import unittest
from pathlib import Path

from expand_layout_regions_manifest import PROVENANCE_SCHEMA, expand
from merge_layout_region_predictions import (
    IncrementalJSONLReader,
    PredictionAccumulator,
    load_bound_region_manifest,
    merge_complete_pages,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def page_fixture(tmp_path: Path) -> dict:
    source = tmp_path / "source.jpg"
    left = tmp_path / "left.jpg"
    right = tmp_path / "right.jpg"
    source.write_bytes(b"source-v1")
    left.write_bytes(b"left-v1")
    right.write_bytes(b"right-v1")
    return {
        "id": "page-1",
        "page_id": "page-1",
        "image": str(source),
        "root_original_image": str(source),
        "layout_normalization_version": "test-v1",
        "layout_normalization": {"transform": "split"},
        "layout_ocr_regions": [
            {
                "id": "page-1-left",
                "parent_id": "page-1",
                "reading_order": 0,
                "region": "left",
                "source_box": [0, 0, 10, 10],
                "rotation_ccw": 0,
                "image": str(left),
                "meaningful_content": True,
            },
            {
                "id": "page-1-right",
                "parent_id": "page-1",
                "reading_order": 1,
                "region": "right",
                "source_box": [10, 0, 20, 10],
                "rotation_ccw": 180,
                "image": str(right),
                "meaningful_content": False,
            },
        ],
    }


def prediction(region: dict, text: str = "transcript") -> dict:
    return {
        **region,
        "model": "model-v1",
        "adapter": "/adapter/v1",
        "decode_profile": "grounded-faithful",
        "generation_settings": {"do_sample": False},
        "prediction": text,
        "raw_prediction": text,
    }


def accumulator(page: dict, regions: list[dict]) -> PredictionAccumulator:
    expected = {str(region["id"]): region for region in regions}
    by_parent = {"page-1": [str(region["id"]) for region in regions]}
    return PredictionAccumulator(
        [page],
        by_parent,
        expected,
        expected_model="model-v1",
        expected_adapter="/adapter/v1",
        expected_decode_profile="grounded-faithful",
    )


class LayoutRegionHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.tmp_path = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_expand_binds_ids_to_source_and_layout(self) -> None:
        page = page_fixture(self.tmp_path)
        first = expand([page])
        self.assertTrue(all("__provenance_" in row["id"] for row in first))
        self.assertTrue(
            all(
                row["layout_region_provenance"]["schema"] == PROVENANCE_SCHEMA
                for row in first
            )
        )

        Path(page["root_original_image"]).write_bytes(b"source-v2")
        source_changed = expand([page])
        self.assertNotEqual(
            [row["id"] for row in source_changed], [row["id"] for row in first]
        )

        page["layout_ocr_regions"][0]["source_box"] = [1, 0, 10, 10]
        layout_changed = expand([page])
        self.assertNotEqual(
            [row["id"] for row in layout_changed],
            [row["id"] for row in source_changed],
        )

    def test_expand_rejects_duplicate_page_and_logical_region_ids(self) -> None:
        page = page_fixture(self.tmp_path)
        with self.assertRaisesRegex(ValueError, "duplicate source page IDs"):
            expand([page, dict(page)])
        page["layout_ocr_regions"][1]["id"] = page["layout_ocr_regions"][0]["id"]
        with self.assertRaisesRegex(ValueError, "duplicate logical layout-region IDs"):
            expand([page])

    def test_expand_rejects_cross_page_region_id_and_parent_mismatch(self) -> None:
        first = page_fixture(self.tmp_path)
        second_dir = self.tmp_path / "second"
        second_dir.mkdir()
        second = page_fixture(second_dir)
        second["id"] = second["page_id"] = "page-2"
        for region in second["layout_ocr_regions"]:
            region["parent_id"] = "page-2"
        with self.assertRaisesRegex(ValueError, "across pages"):
            expand([first, second])

        first["layout_ocr_regions"][0]["parent_id"] = "wrong-page"
        with self.assertRaisesRegex(ValueError, "mismatched parent_id"):
            expand([first])

    def test_bound_manifest_rejects_changed_source_and_config(self) -> None:
        page = page_fixture(self.tmp_path)
        region_path = self.tmp_path / "regions.jsonl"
        write_jsonl(region_path, expand([page]))
        load_bound_region_manifest([page], region_path)

        Path(page["root_original_image"]).write_bytes(b"changed-after-expansion")
        with self.assertRaisesRegex(ValueError, "source/layout/config"):
            load_bound_region_manifest([page], region_path)

        page = page_fixture(self.tmp_path)
        tampered = expand([page])
        tampered[0]["layout_region_provenance"]["config"][
            "fallback_to_all_when_filter_empty"
        ] = False
        write_jsonl(region_path, tampered)
        with self.assertRaisesRegex(ValueError, "provenance/config mismatch"):
            load_bound_region_manifest([page], region_path)

    def test_meaningful_empty_prediction_is_not_complete(self) -> None:
        page = page_fixture(self.tmp_path)
        regions = expand([page])
        state = accumulator(page, regions)
        state.add(prediction(regions[0], text="  "))
        state.add(prediction(regions[1], text=""))
        self.assertEqual(state.ordered_merged(), [])
        missing, empty = state.incomplete_details()
        self.assertEqual(missing, [])
        self.assertEqual(empty, [regions[0]["id"]])

    def test_nonmeaningful_empty_prediction_can_complete(self) -> None:
        page = page_fixture(self.tmp_path)
        regions = expand([page])
        state = accumulator(page, regions)
        state.add(prediction(regions[0], text="left text"))
        state.add(prediction(regions[1], text=""))
        self.assertEqual(state.ordered_merged()[0]["prediction"], "left text")

    def test_prediction_duplicate_provenance_and_config_are_rejected(self) -> None:
        page = page_fixture(self.tmp_path)
        regions = expand([page])
        state = accumulator(page, regions)
        first = prediction(regions[0])
        state.add(first)
        with self.assertRaisesRegex(ValueError, "duplicate prediction ID"):
            state.add(first)

        state = accumulator(page, regions)
        stale = prediction(regions[0])
        stale["layout_region_provenance"] = {
            **stale["layout_region_provenance"],
            "source_sha256": "0" * 64,
        }
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            state.add(stale)

        state = accumulator(page, regions)
        state.add(prediction(regions[0]))
        wrong_config = prediction(regions[1])
        wrong_config["decode_profile"] = "legacy"
        with self.assertRaisesRegex(ValueError, "inference config mismatch"):
            state.add(wrong_config)

    def test_merge_complete_pages_rejects_duplicate_predictions(self) -> None:
        page = page_fixture(self.tmp_path)
        regions = expand([page])
        row = prediction(regions[0])
        with self.assertRaisesRegex(ValueError, "duplicate prediction IDs"):
            merge_complete_pages([page], [row, dict(row)])

    def test_incremental_reader_buffers_partial_tail_and_only_reads_appends(self) -> None:
        path = self.tmp_path / "predictions.jsonl"
        encoded = json.dumps({"id": "one"}).encode("utf-8")
        path.write_bytes(encoded[:5])
        reader = IncrementalJSONLReader(path)
        rows, reset = reader.read_new()
        self.assertEqual(rows, [])
        self.assertFalse(reset)

        with path.open("ab") as handle:
            handle.write(encoded[5:] + b"\n")
        rows, reset = reader.read_new()
        self.assertEqual(rows, [{"id": "one"}])
        self.assertFalse(reset)

        with path.open("ab") as handle:
            handle.write(json.dumps({"id": "two"}).encode("utf-8") + b"\n")
        rows, reset = reader.read_new()
        self.assertEqual(rows, [{"id": "two"}])
        self.assertFalse(reset)

    def test_incremental_reader_restarts_after_truncation(self) -> None:
        path = self.tmp_path / "predictions.jsonl"
        path.write_text('{"id":"old-long-value"}\n', encoding="utf-8")
        reader = IncrementalJSONLReader(path)
        self.assertEqual(reader.read_new()[0], [{"id": "old-long-value"}])
        path.write_text('{"id":"new"}\n', encoding="utf-8")
        rows, reset = reader.read_new()
        self.assertTrue(reset)
        self.assertEqual(rows, [{"id": "new"}])


if __name__ == "__main__":
    unittest.main()

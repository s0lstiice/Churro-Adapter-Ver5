from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from normalize_document_layouts import (
    analyze,
    compose_vertical,
    learn_item_layout_priors,
    recommended_generation_limits,
    region_content_metrics,
    render_doctr_normalization,
    vertical_gutter_evidence,
)


def synthetic_page() -> Image.Image:
    image = Image.new("L", (800, 1100), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=32)
    for index in range(12):
        y = 100 + index * 72
        draw.text((80, y), f"This is handwritten line number {index}", fill="black", font=font)
        draw.line((80, y + 36, 650, y + 36), fill=100, width=2)
    return image.convert("RGB")


def test_rotation_axis_equivariance() -> None:
    page = synthetic_page()
    upright = analyze(page)
    sideways = analyze(page.rotate(90, expand=True, fillcolor="white"))
    assert upright["selected_rotation_ccw"] in {0, 180}
    assert sideways["selected_rotation_ccw"] in {90, 270}


def test_blank_page_is_unchanged_and_flagged() -> None:
    result = analyze(Image.fromarray(np.full((700, 500, 3), 255, dtype=np.uint8)))
    assert result["selected_rotation_ccw"] == 0
    assert "sparse_or_blank" in result["flags"]


def test_region_composition_preserves_reading_order() -> None:
    first = Image.new("RGB", (20, 10), (255, 0, 0))
    second = Image.new("RGB", (10, 15), (0, 0, 255))
    result = compose_vertical([first, second], separator=3)
    assert result.size == (20, 28)
    assert result.getpixel((10, 2)) == (255, 0, 0)
    assert result.getpixel((10, 20)) == (0, 0, 255)


def learned_result(angle: int, confidence: float, meaningful: bool = True) -> dict:
    return {
        "class": 0,
        "correction_rotation_ccw": angle,
        "confidence": confidence,
        "meaningful_content": meaningful,
    }


def wide_inspection(left: dict, right: dict, full_angle: int = 90, full_confidence: float = 0.9) -> dict:
    return {
        "geometry": {"flags": ["wide_spread_or_landscape"]},
        "wide": True,
        "gutter_x": 400,
        "doctr_full_page": learned_result(full_angle, full_confidence),
        "region_predictions": [
            {"region": "left", **left},
            {"region": "right", **right},
        ],
    }


def test_dark_cover_is_not_treated_as_paper_content() -> None:
    dark = Image.new("RGB", (400, 400), (20, 20, 20))
    paper = Image.new("RGB", (400, 400), "white")
    draw = ImageDraw.Draw(paper)
    for y in range(40, 360, 35):
        draw.line((35, y, 365, y), fill="black", width=4)
    assert region_content_metrics(dark)["meaningful_content"] is False
    assert region_content_metrics(paper)["meaningful_content"] is True


def test_conflicting_wide_regions_do_not_use_unsafe_full_rotation() -> None:
    image = Image.new("RGB", (800, 400), "white")
    inspection = wide_inspection(
        learned_result(90, 0.8),
        learned_result(0, 0.8),
    )
    normalized, audit = render_doctr_normalization(image, inspection)
    assert normalized.size == image.size
    assert audit["transform"] == "unchanged_conflicting_wide_layout"
    assert audit["ambiguous"] is True


def test_dark_cover_cannot_trigger_low_confidence_opposed_split() -> None:
    image = Image.new("RGB", (800, 400), "white")
    inspection = wide_inspection(
        learned_result(270, 0.77, meaningful=False),
        learned_result(90, 0.41, meaningful=True),
    )
    normalized, audit = render_doctr_normalization(image, inspection)
    assert normalized.size == image.size
    assert audit["transform"] == "unchanged_single_region_uncertain"


def test_item_prior_resolves_repeated_opposed_spread() -> None:
    image = Image.new("RGB", (800, 400), "white")
    inspection = wide_inspection(
        learned_result(90, 0.69),
        learned_result(90, 0.39),
    )
    prior = {
        "left_rotation_ccw": 90,
        "right_rotation_ccw": 270,
        "supporting_pages": 7,
        "opposed_evidence_pages": 7,
        "support_share": 1.0,
    }
    normalized, audit = render_doctr_normalization(image, inspection, item_prior=prior)
    assert normalized.height > image.height
    assert audit["transform"] == "item_prior_opposed_regions_then_vertical_reading_order"
    assert audit["item_layout_prior_applied"] is True


def test_item_prior_requires_repeated_dominant_pattern() -> None:
    rows = [
        {"item_id": "volume-a", "page_id": f"page-{index}", "image": f"page-{index}.jpg"}
        for index in range(6)
    ]
    strong = wide_inspection(learned_result(90, 0.7), learned_result(270, 0.6))
    reverse = wide_inspection(learned_result(270, 0.8), learned_result(90, 0.8))
    priors = learn_item_layout_priors(rows, [strong, strong, strong, strong, strong, reverse])
    assert priors["volume-a"]["left_rotation_ccw"] == 90
    assert priors["volume-a"]["right_rotation_ccw"] == 270
    assert priors["volume-a"]["opposed_support_share"] == 5 / 6


def test_duplicate_rows_cannot_create_item_prior() -> None:
    duplicate = {"item_id": "volume-a", "page_id": "same-page", "image": "same.jpg"}
    strong = wide_inspection(learned_result(90, 0.7), learned_result(270, 0.6))
    assert learn_item_layout_priors([duplicate] * 8, [strong] * 8) == {}


def test_item_prior_never_overrides_confident_reverse_pair() -> None:
    image = Image.new("RGB", (800, 400), "white")
    reverse = wide_inspection(
        learned_result(270, 0.85),
        learned_result(90, 0.85),
    )
    prior = {"left_rotation_ccw": 90, "right_rotation_ccw": 270}
    _, audit = render_doctr_normalization(image, reverse, item_prior=prior)
    assert audit["transform"] == "split_opposed_regions_then_vertical_reading_order"
    assert audit["item_layout_prior_applied"] is False


def test_item_prior_never_overrides_confident_same_angle_pair() -> None:
    image = Image.new("RGB", (800, 400), "white")
    same = wide_inspection(
        learned_result(90, 0.85),
        learned_result(90, 0.85),
    )
    prior = {"left_rotation_ccw": 90, "right_rotation_ccw": 270}
    _, audit = render_doctr_normalization(image, same, item_prior=prior)
    assert audit["transform"] == "region_agreed_whole_page_rotation"
    assert audit["item_layout_prior_applied"] is False


def test_nonwide_mixed_layout_abstains_from_global_rotation() -> None:
    image = Image.new("RGB", (400, 800), "white")
    inspection = {
        "geometry": {"flags": ["mixed_orientation_regions"]},
        "wide": False,
        "gutter_x": None,
        "doctr_full_page": learned_result(90, 0.99),
        "region_predictions": [],
    }
    normalized, audit = render_doctr_normalization(image, inspection)
    assert normalized.size == image.size
    assert audit["transform"] == "unchanged_mixed_nonwide_layout"


def test_generation_limits_bound_blank_and_dark_mount_regions() -> None:
    assert recommended_generation_limits({"meaningful_content": False})[
        "generation_max_new_tokens"
    ] == 128
    dark_mount = recommended_generation_limits(
        {
            "meaningful_content": True,
            "median_luminance": 28.0,
            "bright_fraction_160": 0.35,
        }
    )
    assert dark_mount["generation_max_new_tokens"] == 384
    assert dark_mount["generation_max_incomplete_retries"] == 0
    assert recommended_generation_limits(
        {
            "meaningful_content": True,
            "median_luminance": 180.0,
            "bright_fraction_160": 0.8,
        }
    ) == {}


def test_wide_landscape_without_central_seam_is_not_partitioned() -> None:
    image = Image.new("RGB", (1000, 500), "white")
    draw = ImageDraw.Draw(image)
    for y in range(50, 450, 45):
        draw.line((40, y, 960, y), fill="black", width=7)
    evidence = vertical_gutter_evidence(image)
    assert evidence["valid"] is False

    inspection = {
        "geometry": {"flags": ["wide_spread_or_landscape"]},
        "wide": True,
        "gutter_x": None,
        "gutter_candidate": evidence,
        "doctr_full_page": learned_result(90, 0.99),
        "region_predictions": [],
    }
    normalized, audit = render_doctr_normalization(image, inspection)
    assert normalized.size == image.size
    assert audit["transform"] == "unchanged_unvalidated_wide_partition"


def test_item_prior_abstains_when_local_conflict_dominates() -> None:
    image = Image.new("RGB", (800, 400), "white")
    inspection = wide_inspection(
        learned_result(270, 0.74),
        learned_result(270, 0.40),
    )
    prior = {"left_rotation_ccw": 90, "right_rotation_ccw": 270}
    _, audit = render_doctr_normalization(image, inspection, item_prior=prior)
    assert audit["transform"] == "unchanged_item_prior_conflict"
    assert audit["item_layout_prior_applied"] is False

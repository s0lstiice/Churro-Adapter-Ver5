from __future__ import annotations

from pathlib import Path

from evaluate_churro_fullpage_qlora import counterfactual_negative_for


def row(identifier: str, item: str, image: Path) -> dict:
    return {
        "id": identifier,
        "page_id": identifier,
        "item_id": item,
        "task_type": "page",
        "image": str(image),
    }


def test_missing_counterfactual_images_are_ignored(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    source.touch()
    missing = tmp_path / "not-downloaded-yet.jpg"
    current = row("current", "item-a", source)
    assert counterfactual_negative_for(
        current,
        [current, row("missing", "item-b", missing)],
    ) is None


def test_existing_unrelated_counterfactual_is_selected(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    source.touch()
    negative = tmp_path / "negative.jpg"
    negative.touch()
    current = row("current", "item-a", source)
    selected = counterfactual_negative_for(
        current,
        [current, row("z", "item-b", negative)],
    )
    assert selected is not None
    assert selected["id"] == "z"

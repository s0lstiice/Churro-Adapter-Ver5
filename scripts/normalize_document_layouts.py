#!/usr/bin/env python3
"""Create transcript-free, layout-normalized images for document OCR.

The original image is never modified.  A learned page-orientation model handles
ordinary pages.  Wide bound spreads are analyzed regionally: confidently
opposed halves are rotated independently, and a repeated per-item layout pattern
can resolve weaker pages from the same volume.  Low-confidence or conflicting
layouts remain unchanged and are explicitly flagged instead of trusting a weak
whole-page rotation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

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


def safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_.") or "page"


def source_fingerprint(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:10]


def foreground(gray: np.ndarray, maximum_side: int = 1200) -> np.ndarray:
    height, width = gray.shape
    scale = min(1.0, maximum_side / max(height, width))
    if scale < 1.0:
        gray = cv2.resize(gray, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    # Remove slow paper/illumination changes while retaining thin handwriting.
    sigma = max(7.0, min(gray.shape) / 45.0)
    background = cv2.GaussianBlur(gray, (0, 0), sigma)
    normalized = cv2.divide(gray, np.maximum(background, 1), scale=235)
    _, ink = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = ink > 0
    border_y = max(2, round(0.01 * ink.shape[0]))
    border_x = max(2, round(0.01 * ink.shape[1]))
    ink[:border_y] = False
    ink[-border_y:] = False
    ink[:, :border_x] = False
    ink[:, -border_x:] = False
    return ink


def smooth_profile(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    sigma = max(1.0, len(values) / 500.0)
    return cv2.GaussianBlur(values.reshape(-1, 1), (1, 0), sigmaX=0, sigmaY=sigma).ravel()


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.flatnonzero(np.diff(padded))
    return [(int(a), int(b)) for a, b in changes.reshape(-1, 2)]


def band_score(profile: np.ndarray) -> tuple[float, int, float]:
    profile = smooth_profile(profile)
    mean = float(profile.mean())
    if mean <= 1e-6 or float(profile.max()) <= 1e-6:
        return -10.0, 0, 0.0
    cv = float(profile.std() / (mean + 1e-6))
    q65 = float(np.quantile(profile, 0.65))
    threshold = max(mean * 0.75, q65 * 0.70)
    active = profile > threshold
    active_runs = runs(active)
    plausible = [pair for pair in active_runs if 1 <= pair[1] - pair[0] <= max(3, len(profile) // 5)]
    gap_fraction = float((profile < mean * 0.45).mean())
    score = math.log1p(max(1, len(plausible))) + 1.25 * cv + 0.75 * gap_fraction
    return score, len(plausible), gap_fraction


def upright_score(ink: np.ndarray) -> tuple[float, int]:
    profile = smooth_profile(ink.mean(axis=1))
    if float(profile.max()) <= 1e-6:
        return 0.0, 0
    threshold = max(float(profile.mean()) * 0.65, float(np.quantile(profile, 0.58)) * 0.65)
    raw_bands = runs(profile > threshold)
    bands: list[tuple[int, int]] = []
    merge_gap = max(2, round(len(profile) * 0.008))
    for start, end in raw_bands:
        if bands and start - bands[-1][1] <= merge_gap:
            bands[-1] = (bands[-1][0], end)
        else:
            bands.append((start, end))
    evidence: list[tuple[float, float]] = []
    for start, end in bands:
        height = end - start
        if height < 4 or height > max(12, round(0.22 * ink.shape[0])):
            continue
        local = ink[start:end].sum(axis=1).astype(np.float32)
        total = float(local.sum())
        if total < 20:
            continue
        coordinates = (np.arange(height, dtype=np.float32) + 0.5) / height
        centroid = float((local * coordinates).sum() / total)
        peak = float((int(np.argmax(smooth_profile(local))) + 0.5) / height)
        # Upright Latin handwriting tends to place its dense body/baseline below
        # the vertical center; ascenders outnumber deep descenders.
        asymmetry = 0.65 * (centroid - 0.5) + 0.35 * (peak - 0.5)
        evidence.append((asymmetry, min(total, 2000.0)))
    if not evidence:
        return 0.0, 0
    values = np.asarray([value for value, _ in evidence], dtype=np.float32)
    weights = np.asarray([weight for _, weight in evidence], dtype=np.float32)
    return float(np.average(values, weights=weights)), len(evidence)


def local_axis_votes(ink: np.ndarray) -> list[int]:
    votes: list[int] = []
    height, width = ink.shape
    for y0, y1, x0, x1 in (
        (0, height, 0, width // 2),
        (0, height, width // 2, width),
        (0, height // 2, 0, width),
        (height // 2, height, 0, width),
    ):
        tile = ink[y0:y1, x0:x1]
        if tile.size == 0 or tile.mean() < 0.001:
            continue
        horizontal = band_score(tile.mean(axis=1))[0]
        vertical = band_score(tile.mean(axis=0))[0]
        margin = horizontal - vertical
        if abs(margin) >= 0.12:
            votes.append(0 if margin > 0 else 90)
    return votes


def analyze(image: Image.Image) -> dict:
    image = ImageOps.exif_transpose(image).convert("L")
    ink = foreground(np.asarray(image))
    ink_fraction = float(ink.mean())
    horizontal_score, horizontal_bands, horizontal_gaps = band_score(ink.mean(axis=1))
    vertical_score, vertical_bands, vertical_gaps = band_score(ink.mean(axis=0))
    axis_margin = horizontal_score - vertical_score
    base_angle = 0 if axis_margin >= 0 else 90

    candidates: list[dict] = []
    for angle in (base_angle, (base_angle + 180) % 360):
        rotated = np.rot90(ink, k=(angle // 90) % 4)
        upright, line_bands = upright_score(rotated)
        candidates.append({"angle": angle, "upright_score": upright, "line_bands": line_bands})
    candidates.sort(key=lambda row: row["upright_score"], reverse=True)
    selected = int(candidates[0]["angle"])
    direction_margin = float(candidates[0]["upright_score"] - candidates[1]["upright_score"])

    votes = local_axis_votes(ink)
    mixed = bool(votes and 0 in votes and 90 in votes)
    sparse = ink_fraction < 0.002
    axis_confidence = abs(axis_margin) / max(0.25, abs(horizontal_score) + abs(vertical_score))
    direction_confidence = min(1.0, direction_margin / 0.045)
    confidence = float(min(1.0, 0.75 * axis_confidence + 0.25 * direction_confidence))
    ambiguous = sparse or mixed or axis_confidence < 0.035 or direction_confidence < 0.12
    if sparse:
        selected = 0
    flags: list[str] = []
    if sparse:
        flags.append("sparse_or_blank")
    if mixed:
        flags.append("mixed_orientation_regions")
    if image.width > 1.45 * image.height:
        flags.append("wide_spread_or_landscape")
    if image.height > 2.3 * image.width:
        flags.append("tall_strip")
    if ambiguous:
        flags.append("orientation_ambiguous")
    return {
        "selected_rotation_ccw": selected,
        "confidence": confidence,
        "axis_confidence": axis_confidence,
        "direction_confidence": direction_confidence,
        "horizontal_band_score": horizontal_score,
        "vertical_band_score": vertical_score,
        "horizontal_bands": horizontal_bands,
        "vertical_bands": vertical_bands,
        "horizontal_gap_fraction": horizontal_gaps,
        "vertical_gap_fraction": vertical_gaps,
        "ink_fraction": ink_fraction,
        "local_axis_votes": votes,
        "ambiguous": ambiguous,
        "flags": flags,
        "candidate_directions": candidates,
    }


def content_fraction(image: Image.Image) -> float:
    return float(foreground(np.asarray(ImageOps.exif_transpose(image).convert("L"))).mean())


def vertical_gutter_evidence(image: Image.Image) -> dict:
    """Locate and validate a central low-ink partition.

    A wide aspect ratio alone is not evidence that an image contains two bound
    pages.  The regional orientation path is enabled only when the central
    search band contains a narrow, materially lower-ink seam.  This keeps a
    normal landscape page from being cut through its writing.
    """

    gray = np.asarray(ImageOps.exif_transpose(image).convert("L"))
    height, width = gray.shape
    scale = min(1.0, 900.0 / max(height, width))
    if scale < 1.0:
        gray = cv2.resize(
            gray,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    # Edge density is deliberately used instead of the general foreground
    # mask.  Archival paper texture can look like foreground everywhere, while
    # a binding/blank partition still has far fewer structural edges than the
    # document-bearing outer regions.
    profile = smooth_profile((cv2.Canny(gray, 40, 100) > 0).mean(axis=0))
    left = round(0.32 * len(profile))
    right = round(0.68 * len(profile))
    central = profile[left:right]
    outer = np.concatenate(
        [
            profile[round(0.05 * len(profile)) : round(0.28 * len(profile))],
            profile[round(0.72 * len(profile)) : round(0.95 * len(profile))],
        ]
    )
    reference = float(np.quantile(outer, 0.50)) if outer.size else 0.0
    central_low = float(np.quantile(central, 0.20)) if central.size else reference
    depth_ratio = central_low / max(reference, 1e-6)
    threshold = reference * 0.25
    low_runs = runs(central <= threshold)
    if low_runs:
        # Prefer a broad blank partition.  Distance to image center breaks ties
        # without assuming that the photographed binding is perfectly centered.
        best_start, best_end = max(
            low_runs,
            key=lambda pair: (
                pair[1] - pair[0],
                -abs((left + (pair[0] + pair[1]) / 2) - len(profile) / 2),
            ),
        )
        run_left = left + best_start
        run_right = left + best_end
        index = (run_left + run_right) // 2
    else:
        index = left + int(np.argmin(central))
        run_left = index
        run_right = index + 1
    minimum = float(profile[index])
    width_fraction = float(run_right - run_left) / max(1, len(profile))
    valid = bool(
        reference >= 0.005
        and depth_ratio <= 0.25
        and width_fraction >= 0.02
    )
    return {
        "x": round(index * image.width / len(profile)),
        "run_left_x": round(run_left * image.width / len(profile)),
        "run_right_x": round(run_right * image.width / len(profile)),
        "valid": valid,
        "minimum_ink": minimum,
        "reference_ink": reference,
        "central_low_ink": central_low,
        "depth_ratio": depth_ratio,
        "width_fraction": width_fraction,
        "search_fraction": [0.32, 0.68],
    }


def vertical_gutter(image: Image.Image) -> int:
    """Compatibility helper returning the best candidate partition."""

    return int(vertical_gutter_evidence(image)["x"])


def doctr_predictions(predictor, images: list[Image.Image]) -> list[dict]:
    arrays = [np.asarray(ImageOps.exif_transpose(image).convert("RGB")) for image in images]
    classes, angles, confidences = predictor(arrays)
    return [
        {
            "class": int(label),
            "correction_rotation_ccw": int(angle) % 360,
            "confidence": float(confidence),
        }
        for label, angle, confidence in zip(classes, angles, confidences)
    ]


def region_content_metrics(image: Image.Image) -> dict:
    """Measure whether a region resembles light paper carrying visible marks.

    Raw foreground fraction alone treats a nearly black album cover as dense
    writing.  Archival manuscript pages are normally dark ink on appreciably
    lighter paper, so luminance is an important independent guard.
    """

    gray = np.asarray(ImageOps.exif_transpose(image).convert("L"))
    height, width = gray.shape
    scale = min(1.0, 900.0 / max(height, width))
    if scale < 1.0:
        gray = cv2.resize(
            gray,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ink_fraction = float(foreground(gray, maximum_side=900).mean())
    median_luminance = float(np.median(gray))
    bright_fraction_160 = float((gray >= 160).mean())
    bright_fraction_200 = float((gray >= 200).mean())
    p05, p95 = np.quantile(gray, [0.05, 0.95])
    meaningful = bool(
        ink_fraction >= 0.002
        and (median_luminance >= 80.0 or bright_fraction_160 >= 0.18)
        and float(p95 - p05) >= 12.0
    )
    return {
        "ink_fraction": ink_fraction,
        "median_luminance": median_luminance,
        "bright_fraction_160": bright_fraction_160,
        "bright_fraction_200": bright_fraction_200,
        "luminance_p05": float(p05),
        "luminance_p95": float(p95),
        "meaningful_content": meaningful,
    }


def recommended_generation_limits(metrics: dict) -> dict:
    """Bound obvious blank/dark-mount failure cases without skipping the scan."""

    if not bool(metrics.get("meaningful_content")):
        return {
            "generation_max_new_tokens": 128,
            "generation_max_incomplete_retries": 0,
            "generation_limit_reason": "low_visual_content_region",
        }
    if (
        float(metrics.get("median_luminance", 255.0)) < 60.0
        and 0.10 <= float(metrics.get("bright_fraction_160", 1.0)) < 0.65
    ):
        return {
            "generation_max_new_tokens": 384,
            "generation_max_incomplete_retries": 0,
            "generation_limit_reason": "dark_mount_with_local_light_content",
        }
    return {}


def inspect_with_doctr(image: Image.Image, predictor) -> dict:
    """Return learned and geometric layout evidence without changing pixels."""

    original = ImageOps.exif_transpose(image).convert("RGB")
    geometry = analyze(original)
    wide = original.width > 1.35 * original.height
    gutter = vertical_gutter_evidence(original) if wide else None
    split_x = int(gutter["x"]) if gutter and gutter["valid"] else None
    regions = [original]
    metric_regions = [original]
    region_names = ["full"]
    probe_boxes: dict[str, list[int]] = {
        "full": [0, 0, original.width, original.height]
    }
    if split_x is not None:
        left_region = original.crop((0, 0, split_x, original.height))
        right_region = original.crop((split_x, 0, original.width, original.height))
        # Classify orientation on the content-facing sides of a broad blank
        # seam.  Feeding half a page of whitespace to the orientation network
        # makes otherwise clear sideways text needlessly uncertain.
        probe_left_x = max(1, int(gutter["run_left_x"]))
        probe_right_x = min(original.width - 1, int(gutter["run_right_x"]))
        left_probe = original.crop((0, 0, probe_left_x, original.height))
        right_probe = original.crop((probe_right_x, 0, original.width, original.height))
        regions.extend([left_probe, right_probe])
        metric_regions.extend([left_region, right_region])
        region_names.extend(["left", "right"])
        probe_boxes["left"] = [0, 0, probe_left_x, original.height]
        probe_boxes["right"] = [probe_right_x, 0, original.width, original.height]
    predictions = doctr_predictions(predictor, regions)
    by_name = dict(zip(region_names, predictions))
    region_audit = []
    for name, region in zip(region_names[1:], metric_regions[1:]):
        region_audit.append(
            {
                "region": name,
                **by_name[name],
                **region_content_metrics(region),
                "orientation_probe_box": probe_boxes[name],
            }
        )
    return {
        "geometry": geometry,
        "wide": wide,
        "gutter_x": split_x,
        "gutter_candidate": gutter,
        "doctr_full_page": by_name["full"],
        "full_content_metrics": region_content_metrics(original),
        "region_predictions": region_audit,
    }


def item_identity(row: dict) -> str:
    # Collection-wide fallbacks are too broad: unrelated volumes can have
    # different binding conventions.  A prior is learned only for an explicit
    # item identity.
    return str(row.get("item_id") or "")


def stable_page_identity(row: dict) -> tuple[str, str]:
    identifier = str(row.get("page_id") or row.get("id") or "")
    image = str(Path(str(row.get("image") or "")).resolve())
    return identifier, image


def learn_item_layout_priors(rows: list[dict], inspections: list[dict]) -> dict[str, dict]:
    """Learn a repeated opposed-spread convention from strong pages in an item."""

    evidence: dict[str, dict[tuple[int, int], list[tuple[float, str]]]] = {}
    eligible_wide_pages: dict[str, set[tuple[str, str]]] = {}
    seen: dict[str, set[tuple[str, str]]] = {}
    for row, inspection in zip(rows, inspections):
        item = item_identity(row)
        regions = inspection.get("region_predictions") or []
        if not item or len(regions) != 2:
            continue
        page_key = stable_page_identity(row)
        if page_key in seen.setdefault(item, set()):
            continue
        seen[item].add(page_key)
        left, right = regions
        if not left.get("meaningful_content") or not right.get("meaningful_content"):
            continue
        eligible_wide_pages.setdefault(item, set()).add(page_key)
        left_angle = int(left["correction_rotation_ccw"]) % 360
        right_angle = int(right["correction_rotation_ccw"]) % 360
        pair = (left_angle, right_angle)
        if pair not in {(90, 270), (270, 90)}:
            continue
        minimum_confidence = min(float(left["confidence"]), float(right["confidence"]))
        if minimum_confidence < 0.40:
            continue
        page_name = page_key[0] or page_key[1]
        evidence.setdefault(item, {}).setdefault(pair, []).append(
            (minimum_confidence, page_name)
        )

    priors: dict[str, dict] = {}
    for item, pairs in evidence.items():
        counts = {pair: len(values) for pair, values in pairs.items()}
        best_pair = max(counts, key=lambda pair: (counts[pair], pair))
        best_count = counts[best_pair]
        opposed_total = sum(counts.values())
        opposed_share = best_count / opposed_total
        eligible_total = len(eligible_wide_pages.get(item, set()))
        eligible_share = best_count / max(1, eligible_total)
        if best_count >= 5 and opposed_share >= 0.80 and eligible_share >= 0.25:
            selected_evidence = pairs[best_pair]
            priors[item] = {
                "left_rotation_ccw": best_pair[0],
                "right_rotation_ccw": best_pair[1],
                "supporting_pages": best_count,
                "opposed_evidence_pages": opposed_total,
                "eligible_wide_pages": eligible_total,
                "opposed_support_share": opposed_share,
                "eligible_support_share": eligible_share,
                "minimum_support_confidence": min(value for value, _ in selected_evidence),
                "supporting_page_ids": [page for _, page in selected_evidence],
            }
    return priors


def compose_vertical_with_placements(
    regions: list[Image.Image], separator: int = 28
) -> tuple[Image.Image, list[list[int]]]:
    width = max(region.width for region in regions)
    height = sum(region.height for region in regions) + separator * (len(regions) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    placements: list[list[int]] = []
    y = 0
    for index, region in enumerate(regions):
        x = (width - region.width) // 2
        canvas.paste(region, (x, y))
        placements.append([x, y, x + region.width, y + region.height])
        y += region.height
        if index + 1 < len(regions):
            y += separator
    return canvas, placements


def compose_vertical(regions: list[Image.Image], separator: int = 28) -> Image.Image:
    return compose_vertical_with_placements(regions, separator=separator)[0]


def render_doctr_normalization(
    image: Image.Image,
    inspection: dict,
    item_prior: dict | None = None,
) -> tuple[Image.Image, dict]:
    original = ImageOps.exif_transpose(image).convert("RGB")
    geometry = inspection["geometry"]
    wide = bool(inspection["wide"])
    split_x = inspection.get("gutter_x")
    full = inspection["doctr_full_page"]
    flags = list(geometry["flags"])
    transform = "unchanged"
    selected_angle = 0
    normalized = original.copy()
    applied_regions = [
        {
            "region": "full",
            "source_box": [0, 0, original.width, original.height],
            "rotation_ccw": 0,
            "output_box": [0, 0, original.width, original.height],
        }
    ]
    transform_confidence = float(full["confidence"])
    region_audit = list(inspection.get("region_predictions") or [])
    prior_applied = False

    if split_x is not None:
        left = original.crop((0, 0, split_x, original.height))
        right = original.crop((split_x, 0, original.width, original.height))
        left_result, right_result = region_audit
        left_angle = int(left_result["correction_rotation_ccw"]) % 360
        right_angle = int(right_result["correction_rotation_ccw"]) % 360
        left_content = bool(left_result.get("meaningful_content"))
        right_content = bool(right_result.get("meaningful_content"))
        both_content = left_content and right_content
        one_content = left_content != right_content
        confident_regions = (
            float(left_result["confidence"]) >= 0.40
            and float(right_result["confidence"]) >= 0.40
        )
        opposed = (left_angle, right_angle) in {(90, 270), (270, 90)}
        if confident_regions and opposed and both_content:
            # Strong page-local evidence is authoritative, even when a volume
            # prior exists in the opposite direction.
            left_fixed = left.rotate(left_angle, expand=True, fillcolor="white") if left_angle else left
            right_fixed = right.rotate(right_angle, expand=True, fillcolor="white") if right_angle else right
            normalized, placements = compose_vertical_with_placements([left_fixed, right_fixed])
            applied_regions = [
                {
                    "region": "left",
                    "source_box": [0, 0, split_x, original.height],
                    "rotation_ccw": left_angle,
                    "output_box": placements[0],
                },
                {
                    "region": "right",
                    "source_box": [split_x, 0, original.width, original.height],
                    "rotation_ccw": right_angle,
                    "output_box": placements[1],
                },
            ]
            transform_confidence = min(
                float(left_result["confidence"]), float(right_result["confidence"])
            )
            transform = "split_opposed_regions_then_vertical_reading_order"
            if "mixed_orientation_regions" not in flags:
                flags.append("mixed_orientation_regions")
            flags = [flag for flag in flags if flag != "orientation_ambiguous"]
            flags.append("orientation_resolved_by_regional_agreement")
        elif both_content and left_angle == right_angle and min(
            float(left_result["confidence"]), float(right_result["confidence"])
        ) >= 0.70:
            same_confidence = min(
                float(left_result["confidence"]), float(right_result["confidence"])
            )
            safe_180 = left_angle != 180 or same_confidence >= 0.90
            if not safe_180:
                transform = "unchanged_unsafe_180"
                if "orientation_ambiguous" not in flags:
                    flags.append("orientation_ambiguous")
            elif left_angle == 0:
                transform = "unchanged_region_agreement"
                flags = [flag for flag in flags if flag != "orientation_ambiguous"]
            else:
                selected_angle = left_angle
                normalized = original.rotate(selected_angle, expand=True, fillcolor="white")
                applied_regions = [
                    {
                        "region": "full",
                        "source_box": [0, 0, original.width, original.height],
                        "rotation_ccw": selected_angle,
                        "output_box": [0, 0, normalized.width, normalized.height],
                    }
                ]
                transform_confidence = same_confidence
                transform = "region_agreed_whole_page_rotation"
                flags = [flag for flag in flags if flag != "orientation_ambiguous"]
                flags.append("orientation_resolved_by_regional_agreement")
        elif item_prior is not None and (left_content or right_content):
            prior_left = int(item_prior["left_rotation_ccw"]) % 360
            prior_right = int(item_prior["right_rotation_ccw"]) % 360
            local_support = sum(
                float(candidate["confidence"])
                for candidate, angle, expected, content in (
                    (left_result, left_angle, prior_left, left_content),
                    (right_result, right_angle, prior_right, right_content),
                )
                if content and angle == expected and float(candidate["confidence"]) >= 0.40
            )
            local_conflict = sum(
                float(candidate["confidence"])
                for candidate, angle, expected, content in (
                    (left_result, left_angle, prior_left, left_content),
                    (right_result, right_angle, prior_right, right_content),
                )
                if content and angle != expected and float(candidate["confidence"]) >= 0.40
            )
            # Use a continuous dominance margin instead of a brittle .74/.75
            # cutoff.  A weak conflict can be corrected by a strong volume
            # prior; a clearly dominant local vote forces an abstention.
            strong_conflict = local_conflict >= 0.65 and local_conflict > local_support + 0.15
            compatible_local_support = (
                left_content
                and left_angle == prior_left
                and float(left_result["confidence"]) >= 0.40
            ) or (
                right_content
                and right_angle == prior_right
                and float(right_result["confidence"]) >= 0.40
            )
            full_support = (
                int(full["correction_rotation_ccw"]) % 360 in {prior_left, prior_right}
                and float(full["confidence"]) >= 0.50
            )
            if not strong_conflict and (compatible_local_support or full_support):
                # Keep both source regions even when one looks blank/dark.  The
                # light-paper heuristic controls confidence, not data deletion.
                left_fixed = left.rotate(prior_left, expand=True, fillcolor="white")
                right_fixed = right.rotate(prior_right, expand=True, fillcolor="white")
                normalized, placements = compose_vertical_with_placements([left_fixed, right_fixed])
                applied_regions = [
                    {
                        "region": "left",
                        "source_box": [0, 0, split_x, original.height],
                        "rotation_ccw": prior_left,
                        "output_box": placements[0],
                    },
                    {
                        "region": "right",
                        "source_box": [split_x, 0, original.width, original.height],
                        "rotation_ccw": prior_right,
                        "output_box": placements[1],
                    },
                ]
                transform_confidence = float(item_prior.get("minimum_support_confidence", 0.0))
                transform = "item_prior_opposed_regions_then_vertical_reading_order"
                prior_applied = True
                if "mixed_orientation_regions" not in flags:
                    flags.append("mixed_orientation_regions")
                flags = [flag for flag in flags if flag != "orientation_ambiguous"]
                flags.extend(["item_layout_prior_applied", "orientation_resolved_by_item_layout_prior"])
            else:
                transform = "unchanged_item_prior_conflict"
                if "orientation_ambiguous" not in flags:
                    flags.append("orientation_ambiguous")
        elif one_content:
            # A luminance heuristic is not sufficient authority to discard half
            # of an archival scan.  Preserve the original and expose the weak
            # result in the audit for an optional multi-view rescue.
            transform = "unchanged_single_region_uncertain"
            if "orientation_ambiguous" not in flags:
                flags.append("orientation_ambiguous")
        else:
            # On a spread, a full-page vote is unsafe whenever meaningful
            # regions disagree: one global correction can invert half a book.
            transform = "unchanged_conflicting_wide_layout"
            if "orientation_ambiguous" not in flags:
                flags.append("orientation_ambiguous")
    elif wide:
        # A wide page without a validated central seam must not be globally
        # rotated: a two-view spread can contain opposing orientations.
        transform = "unchanged_unvalidated_wide_partition"
        if "unvalidated_wide_partition" not in flags:
            flags.append("unvalidated_wide_partition")
        if "orientation_ambiguous" not in flags:
            flags.append("orientation_ambiguous")
    else:
        full_angle = int(full["correction_rotation_ccw"]) % 360
        confidence_needed = 0.90 if full_angle == 180 else 0.50
        mixed_geometry = "mixed_orientation_regions" in geometry.get("flags", [])
        if full_angle and mixed_geometry:
            transform = "unchanged_mixed_nonwide_layout"
            if "orientation_ambiguous" not in flags:
                flags.append("orientation_ambiguous")
        elif full_angle and float(full["confidence"]) >= confidence_needed:
            selected_angle = full_angle
            normalized = original.rotate(selected_angle, expand=True, fillcolor="white")
            applied_regions = [
                {
                    "region": "full",
                    "source_box": [0, 0, original.width, original.height],
                    "rotation_ccw": selected_angle,
                    "output_box": [0, 0, normalized.width, normalized.height],
                }
            ]
            transform = "whole_page_rotation"
        elif full_angle:
            transform = "unchanged_low_confidence"
            if "orientation_ambiguous" not in flags:
                flags.append("orientation_ambiguous")

    audit = {
        **geometry,
        "backend": "doctr_mobilenet_v3_small_page_orientation",
        "doctr_full_page": full,
        "full_content_metrics": inspection.get("full_content_metrics"),
        "region_predictions": region_audit,
        "gutter_x": split_x,
        "gutter_candidate": inspection.get("gutter_candidate"),
        "transform": transform,
        "selected_rotation_ccw": selected_angle,
        "confidence": float(full["confidence"]),
        "transform_confidence": transform_confidence,
        "applied_regions": applied_regions,
        "output_size": [normalized.width, normalized.height],
        "item_layout_prior": item_prior,
        "item_layout_prior_applied": prior_applied,
        "flags": flags,
        "ambiguous": "orientation_ambiguous" in flags,
    }
    return normalized, audit


def normalize_with_doctr(
    image: Image.Image,
    predictor,
    item_prior: dict | None = None,
) -> tuple[Image.Image, dict]:
    """Compatibility wrapper for callers that normalize one page at a time."""

    inspection = inspect_with_doctr(image, predictor)
    return render_doctr_normalization(image, inspection, item_prior=item_prior)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--backend", choices=("doctr", "geometry"), default="doctr")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--source-field",
        default="image",
        help="Manifest field containing the source image (use root_original_image to re-audit outputs).",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip manifest rows whose downloaded image is not present.",
    )
    args = parser.parse_args()
    rows = read_jsonl(args.manifest)
    if args.source_field != "image":
        remapped_rows: list[dict] = []
        for row in rows:
            source_value = row.get(args.source_field)
            if not source_value:
                raise ValueError(
                    f"row {row.get('page_id') or row.get('id')} lacks --source-field "
                    f"{args.source_field!r}"
                )
            remapped = dict(row)
            remapped["image"] = source_value
            remapped_rows.append(remapped)
        rows = remapped_rows
    if args.limit is not None:
        rows = rows[: args.limit]
    row_ids = [
        str(row.get("page_id") or row.get("id") or Path(str(row.get("image") or "")).stem)
        for row in rows
    ]
    duplicates = sorted(identifier for identifier, count in Counter(row_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate manifest page IDs: {duplicates[:20]}")
    output_rows: list[dict] = []
    audit_rows: list[dict] = []
    counts = {
        "transformed": 0,
        "unchanged": 0,
        "ambiguous": 0,
        "mixed": 0,
        "missing": 0,
        "item_prior_applied": 0,
    }
    predictor = None
    if args.backend == "doctr":
        from doctr.models import page_orientation_predictor

        predictor = page_orientation_predictor(pretrained=True, batch_size=args.batch_size)
    counts["split_opposed"] = 0
    prepared: list[tuple[dict, dict | None]] = []
    if predictor is not None:
        with ProgressTask(
            "Analyze unusual document layouts",
            total=len(rows),
            unit="pages",
            task_id=f"layout-analyze-{args.output_manifest.stem}",
            output_dir=args.output_manifest.parent,
        ) as task:
            for index, row in enumerate(rows, start=1):
                source = Path(row["image"])
                if not source.is_file():
                    if not args.skip_missing:
                        raise FileNotFoundError(source)
                    counts["missing"] += 1
                    audit_rows.append(
                        {
                            "id": row.get("page_id") or row.get("id"),
                            "original_image": str(source.resolve()),
                            "status": "missing_input_image",
                        }
                    )
                    task.update(index, message=f"missing: {row.get('page_id') or row.get('id')}", metrics=counts)
                    continue
                with Image.open(source) as opened:
                    inspection = inspect_with_doctr(opened, predictor)
                prepared.append((row, inspection))
                task.update(index, message=str(row.get("page_id") or row.get("id")), metrics=counts)
        priors = learn_item_layout_priors(
            [row for row, _ in prepared],
            [inspection for _, inspection in prepared if inspection is not None],
        )
    else:
        priors = {}
        for row in rows:
            source = Path(row["image"])
            if not source.is_file():
                if not args.skip_missing:
                    raise FileNotFoundError(source)
                counts["missing"] += 1
                audit_rows.append(
                    {
                        "id": row.get("page_id") or row.get("id"),
                        "original_image": str(source.resolve()),
                        "status": "missing_input_image",
                    }
                )
                continue
            prepared.append((row, None))

    with ProgressTask(
        "Normalize unusual document layouts",
        total=len(prepared),
        unit="pages",
        task_id=f"layout-normalize-{args.output_manifest.stem}",
        output_dir=args.output_manifest.parent,
    ) as task:
        for index, (row, inspection) in enumerate(prepared, start=1):
            source = Path(row["image"])
            with Image.open(source) as opened:
                original = ImageOps.exif_transpose(opened).convert("RGB")
                if predictor is not None:
                    normalized, result = render_doctr_normalization(
                        original,
                        inspection,
                        item_prior=priors.get(item_identity(row)),
                    )
                else:
                    result = analyze(original)
                    angle = 0 if result["ambiguous"] else int(result["selected_rotation_ccw"])
                    normalized = original.rotate(angle, expand=True, fillcolor="white") if angle else original.copy()
                    result["backend"] = "projection_geometry"
                    result["transform"] = (
                        "whole_page_rotation"
                        if angle
                        else "unchanged_geometry_ambiguous"
                        if result["ambiguous"]
                        else "unchanged"
                    )
                    result["selected_rotation_ccw"] = angle
                angle = int(result["selected_rotation_ccw"])
                changed = not str(result["transform"]).startswith("unchanged")
                if changed:
                    identifier = row.get("page_id") or row.get("id") or source.stem
                    fingerprint = source_fingerprint(source)
                    transform_tag = safe_name(result["transform"])[:42]
                    destination = args.image_dir / (
                        f"{safe_name(identifier)}__{fingerprint}__{transform_tag}.jpg"
                    )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_name(
                        destination.stem + ".part" + destination.suffix
                    )
                    normalized.save(temporary, quality=92, optimize=True)
                    os.replace(temporary, destination)
                    counts["transformed"] += 1
                else:
                    destination = source
                    counts["unchanged"] += 1
                identity = str(row.get("page_id") or row.get("id") or source.stem)
                ocr_regions: list[dict] = []
                applied = list(result.get("applied_regions") or [])
                regional_evidence = {
                    str(candidate.get("region")): candidate
                    for candidate in result.get("region_predictions") or []
                }
                if len(applied) <= 1:
                    full_metrics = result.get("full_content_metrics") or region_content_metrics(original)
                    region_record = {
                        "id": identity,
                        "parent_id": identity,
                        "reading_order": 0,
                        "region": "full",
                        "image": str(destination.resolve()),
                        "source_box": [0, 0, original.width, original.height],
                        "rotation_ccw": angle,
                        "meaningful_content": bool(full_metrics.get("meaningful_content")),
                        "content_metrics": full_metrics,
                    }
                    region_record.update(recommended_generation_limits(full_metrics))
                    ocr_regions.append(region_record)
                else:
                    for region_index, region in enumerate(applied):
                        output_box = tuple(int(value) for value in region["output_box"])
                        region_image = normalized.crop(output_box)
                        region_name = str(region["region"])
                        region_destination = args.image_dir / "regions" / (
                            f"{safe_name(identity)}__{source_fingerprint(source)}__"
                            f"r{region_index:02d}_{safe_name(region_name)}.jpg"
                        )
                        region_destination.parent.mkdir(parents=True, exist_ok=True)
                        temporary = region_destination.with_name(
                            region_destination.stem + ".part" + region_destination.suffix
                        )
                        region_image.save(temporary, quality=92, optimize=True)
                        os.replace(temporary, region_destination)
                        evidence = regional_evidence.get(region_name, {})
                        region_record = {
                            "id": f"{identity}__layout_region_{region_index:02d}",
                            "parent_id": identity,
                            "reading_order": region_index,
                            "region": region_name,
                            "image": str(region_destination.resolve()),
                            "source_box": region["source_box"],
                            "rotation_ccw": region["rotation_ccw"],
                            "meaningful_content": bool(
                                evidence.get("meaningful_content", True)
                            ),
                            "orientation_confidence": evidence.get("confidence"),
                            "content_metrics": {
                                key: value
                                for key, value in evidence.items()
                                if key
                                in {
                                    "ink_fraction",
                                    "median_luminance",
                                    "bright_fraction_160",
                                    "bright_fraction_200",
                                    "luminance_p05",
                                    "luminance_p95",
                                    "meaningful_content",
                                }
                            },
                        }
                        region_record.update(
                            recommended_generation_limits(region_record["content_metrics"])
                        )
                        ocr_regions.append(region_record)
            counts["ambiguous"] += int(result["ambiguous"])
            counts["mixed"] += int("mixed_orientation_regions" in result["flags"])
            counts["split_opposed"] += int("opposed_regions_then_vertical_reading_order" in result["transform"])
            counts["item_prior_applied"] += int(result.get("item_layout_prior_applied", False))
            enriched = dict(row)
            previous_normalization = row.get("layout_normalization")
            history = list(row.get("layout_normalization_history") or [])
            if previous_normalization:
                history.append(previous_normalization)
            enriched["root_original_image"] = str(
                row.get("root_original_image")
                or row.get("original_image")
                or source.resolve()
            )
            enriched["layout_input_image"] = str(source.resolve())
            enriched.setdefault("original_image", str(source.resolve()))
            enriched["image"] = str(destination.resolve())
            enriched["layout_normalization"] = result
            enriched["layout_normalization_history"] = history
            enriched["layout_ocr_regions"] = ocr_regions
            enriched["layout_normalization_version"] = "doctr-orientation-item-layout-prior-v4" if predictor is not None else "projection-bands-upright-asymmetry-v1"
            output_rows.append(enriched)
            audit_rows.append({"id": row.get("page_id") or row.get("id"), "original_image": str(source.resolve()), "inference_image": str(destination.resolve()), **result})
            task.update(index, message=str(row.get("page_id") or row.get("id")), metrics=counts)
    if len(output_rows) + counts["missing"] != len(rows):
        raise RuntimeError("layout reconciliation failed: processed + missing != manifest rows")
    if counts["transformed"] + counts["unchanged"] != len(output_rows):
        raise RuntimeError("layout reconciliation failed: transformed + unchanged != pages")
    destinations = [str(row["image"]) for row in output_rows]
    if len(destinations) != len(set(destinations)):
        raise RuntimeError("layout normalization produced duplicate destination paths")
    write_jsonl(args.output_manifest, output_rows)
    write_jsonl(args.audit, audit_rows)
    summary = {
        "manifest_rows": len(rows),
        "pages": len(output_rows),
        "learned_item_layout_priors": priors,
        **counts,
        "output_manifest": str(args.output_manifest.resolve()),
        "audit": str(args.audit.resolve()),
    }
    args.audit.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

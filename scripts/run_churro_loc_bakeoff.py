#!/usr/bin/env python3
"""Run a resumable CHURRO bake-off on held-out LOC manuscript pages."""

from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

try:
    from rapidfuzz.distance import Levenshtein as RapidLevenshtein
except ImportError:  # Keep exported bundles usable without RapidFuzz.
    RapidLevenshtein = None

from universal_progress_monitor.progress_client import ProgressTask


MODEL_ID = "stanford-oval/churro-3B"
PAGE_SYSTEM_PROMPT = "Transcribe the entirety of this historical document to XML format."
LINE_SYSTEM_PROMPT = (
    "Transcribe every visible handwritten character in this historical document line. "
    "Return only the transcription, without commentary or markup."
)


def parse_pages(value: str) -> list[int]:
    pages: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(piece) for piece in part.split("-", 1))
            pages.extend(range(start, end + 1))
        else:
            pages.append(int(part))
    return sorted(set(pages))


def jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def normalize_for_scoring(text: str) -> str:
    without_punctuation = "".join(
        character
        for character in unicodedata.normalize("NFKC", text).casefold()
        if not unicodedata.category(character).startswith("P")
    )
    return " ".join(without_punctuation.split())


def edit_distance(left, right) -> int:
    if RapidLevenshtein is not None:
        return int(RapidLevenshtein.distance(left, right))
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def visible_text(raw: str) -> str:
    value = raw.strip()
    value = re.sub(r"^```(?:xml|markdown|text)?\s*", "", value, flags=re.I)
    value = re.sub(r"\s*```$", "", value)
    # CHURRO emits useful descriptive metadata, but it is not document text and
    # must not be counted as a transcription hallucination.
    value = re.sub(r"<Metadata\b[^>]*>.*?</Metadata>\s*", "", value, flags=re.I | re.S)
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.I)
    value = re.sub(r"</(?:Line|Header|Paragraph|TextBlock|Page)>\s*", "\n", value, flags=re.I)
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    return "\n".join(" ".join(line.split()) for line in value.splitlines() if line.split())


def score(target: str, prediction: str) -> dict:
    target_norm = normalize_for_scoring(target)
    prediction_norm = normalize_for_scoring(prediction)
    char_edits = edit_distance(target_norm, prediction_norm)
    target_words = target_norm.split()
    prediction_words = prediction_norm.split()
    word_edits = edit_distance(target_words, prediction_words)
    return {
        "target_normalized": target_norm,
        "prediction_normalized": prediction_norm,
        "character_edits": char_edits,
        "target_characters": len(target_norm),
        "word_edits": word_edits,
        "target_words": len(target_words),
        "prediction_characters": len(prediction_norm),
        "prediction_words": len(prediction_words),
        "cer": char_edits / max(1, len(target_norm)),
        "wer": word_edits / max(1, len(target_words)),
        "exact": target_norm == prediction_norm,
    }


def build_page_items(data_root: Path, pages: list[int], condition: str, support_page: int) -> list[dict]:
    items = []
    support_image = data_root / "pages" / "mss184240342" / f"page_{support_page:03d}.jpg"
    support_text = (data_root / "page_transcripts" / f"page_{support_page:03d}.txt").read_text(
        encoding="utf-8"
    )
    for page in pages:
        items.append(
            {
                "id": f"page_{page:03d}",
                "page": page,
                "image": str(data_root / "pages" / "mss184240342" / f"page_{page:03d}.jpg"),
                "target": (data_root / "page_transcripts" / f"page_{page:03d}.txt").read_text(
                    encoding="utf-8"
                ),
                "condition": condition,
                "support_image": str(support_image) if condition == "support_page" else None,
                "support_text": support_text if condition == "support_page" else None,
            }
        )
    return items


def build_line_items(lines_manifest: Path, pages: list[int]) -> list[dict]:
    wanted = {f"mss184240342_page_{page:03d}" for page in pages}
    rows = [row for row in jsonl(lines_manifest) if row.get("page_id") in wanted]
    rows.sort(
        key=lambda row: (
            row.get("page_id", ""),
            int(row.get("column_id", 0)),
            int(row.get("line_in_column", row.get("line_id", 0))),
            int(row.get("line_id", 0)),
        )
    )
    items = []
    seen = set()
    for row in rows:
        key = (row.get("page_id"), row.get("line_id"), row.get("image"))
        if key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "id": f"{row['page_id']}_line_{int(row['line_id']):04d}",
                "page": int(str(row["page_id"]).rsplit("_", 1)[-1]),
                "line_id": int(row["line_id"]),
                "column_id": int(row.get("column_id", 0)),
                "line_in_column": int(row.get("line_in_column", row["line_id"])),
                "image": row["image"],
                "target": row.get("matched_text") or row.get("text", ""),
                "condition": "polygon_lines",
            }
        )
    return items


def content_image(path: str, max_pixels: int) -> dict:
    return {
        "type": "image",
        "image": f"file://{Path(path).resolve()}",
        "min_pixels": 256 * 28 * 28,
        "max_pixels": max_pixels,
    }


def make_messages(item: dict, page_max_pixels: int, line_max_pixels: int) -> list[dict]:
    condition = item["condition"]
    if condition == "polygon_lines":
        return [
            {"role": "system", "content": [{"type": "text", "text": LINE_SYSTEM_PROMPT}]},
            {"role": "user", "content": [content_image(item["image"], line_max_pixels)]},
        ]
    if condition == "support_page":
        support_instruction = (
            "The first image is a reference page written in the same hand. Its exact transcription is:\n"
            f"<ReferenceTranscript>\n{item['support_text']}\n</ReferenceTranscript>\n"
            "Use it only to learn this writer's letter forms, spacing, and line cadence. "
            "The second image is the target. Transcribe the entire target image in reading order to XML. "
            "Do not copy text from the reference unless it is visibly present in the target."
        )
        return [
            {"role": "system", "content": [{"type": "text", "text": PAGE_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    content_image(item["support_image"], page_max_pixels),
                    {"type": "text", "text": support_instruction},
                    content_image(item["image"], page_max_pixels),
                ],
            },
        ]
    return [
        {"role": "system", "content": [{"type": "text", "text": PAGE_SYSTEM_PROMPT}]},
        {"role": "user", "content": [content_image(item["image"], page_max_pixels)]},
    ]


def load_model(model_id: str):
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quantization,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return processor, model


@torch.inference_mode()
def infer(processor, model, messages: list[dict], max_new_tokens: int) -> str:
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    batch = processor(
        text=[rendered],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    batch = batch.to(model.device)
    generated = model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        no_repeat_ngram_size=8,
        use_cache=True,
    )
    trimmed = [output[len(input_ids) :] for input_ids, output in zip(batch.input_ids, generated)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def aggregate_summary(rows: list[dict]) -> dict:
    totals = defaultdict(int)
    for row in rows:
        metrics = row["metrics"]
        for key in (
            "character_edits",
            "target_characters",
            "word_edits",
            "target_words",
            "prediction_characters",
            "prediction_words",
        ):
            totals[key] += metrics[key]
        totals["exact"] += int(metrics["exact"])
    count = len(rows)
    result = {"count": count, **totals}
    result["cer"] = totals["character_edits"] / max(1, totals["target_characters"])
    result["wer"] = totals["word_edits"] / max(1, totals["target_words"])
    result["exact_rate"] = totals["exact"] / max(1, count)
    result["output_character_ratio"] = totals["prediction_characters"] / max(
        1, totals["target_characters"]
    )
    result["output_word_ratio"] = totals["prediction_words"] / max(1, totals["target_words"])
    return result


def write_page_reassemblies(rows: list[dict], output: Path) -> None:
    by_page: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_page[int(row["page"])].append(row)
    assembled = []
    for page, page_rows in sorted(by_page.items()):
        page_rows.sort(
            key=lambda row: (
                int(row.get("column_id", 0)),
                int(row.get("line_in_column", row.get("line_id", 0))),
                int(row.get("line_id", 0)),
            )
        )
        target = "\n".join(row["target"] for row in page_rows)
        prediction = "\n".join(row["prediction_text"] for row in page_rows)
        assembled.append(
            {"page": page, "target": target, "prediction": prediction, "metrics": score(target, prediction)}
        )
    with (output / "page_reassemblies.jsonl").open("w", encoding="utf-8") as handle:
        for row in assembled:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "page_reassembly_summary.json").write_text(
        json.dumps(aggregate_summary(assembled), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=("full_page", "polygon_lines", "support_page"), required=True)
    parser.add_argument("--pages", default="61-65")
    parser.add_argument("--support-page", type=int, default=60)
    parser.add_argument("--data-root", type=Path, default=Path("loc_full_pages_v1_unseen_mss184240342"))
    parser.add_argument("--lines-manifest", type=Path, default=Path("loc_mss184240342_v4_polygon_words_v1/lines.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--page-max-pixels", type=int, default=1_605_632)
    parser.add_argument("--line-max-pixels", type=int, default=401_408)
    parser.add_argument("--page-max-new-tokens", type=int, default=768)
    parser.add_argument("--line-max-new-tokens", type=int, default=192)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    pages = parse_pages(args.pages)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.condition == "polygon_lines":
        items = build_line_items(args.lines_manifest, pages)
    else:
        items = build_page_items(args.data_root, pages, args.condition, args.support_page)
    if args.limit is not None:
        items = items[: args.limit]

    predictions_path = args.output / "predictions.jsonl"
    existing = list(jsonl(predictions_path)) if predictions_path.exists() else []
    done = {row["id"] for row in existing}
    pending = [item for item in items if item["id"] not in done]
    task = ProgressTask(
        f"CHURRO LOC {args.condition}",
        total=len(items),
        unit="items",
        task_id=f"churro-loc-{args.condition}",
        output_dir=args.output,
        metadata={"model": args.model, "pages": pages, "resume_count": len(existing)},
    )
    task.update(len(existing), message="loading 4-bit CHURRO model")
    try:
        processor, model = load_model(args.model)
        with predictions_path.open("a", encoding="utf-8") as handle:
            for item in pending:
                messages = make_messages(item, args.page_max_pixels, args.line_max_pixels)
                max_tokens = (
                    args.line_max_new_tokens if args.condition == "polygon_lines" else args.page_max_new_tokens
                )
                raw = infer(processor, model, messages, max_tokens)
                prediction = visible_text(raw)
                row = {
                    **{key: value for key, value in item.items() if key != "support_text"},
                    "model": args.model,
                    "quantization": "bnb_nf4_4bit",
                    "raw_prediction": raw,
                    "prediction_text": prediction,
                    "metrics": score(item["target"], prediction),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                existing.append(row)
                task.update(
                    len(existing),
                    message=f"{args.condition}: {item['id']}",
                    metrics={"cer": row["metrics"]["cer"], "wer": row["metrics"]["wer"]},
                )
                print(
                    json.dumps(
                        {
                            "current": len(existing),
                            "total": len(items),
                            "item": item["id"],
                            "cer": row["metrics"]["cer"],
                            "wer": row["metrics"]["wer"],
                        }
                    ),
                    flush=True,
                )
        summary = aggregate_summary(existing)
        summary.update(
            {
                "condition": args.condition,
                "model": args.model,
                "pages": pages,
                "support_page": args.support_page if args.condition == "support_page" else None,
                "scoring": "NFKC + casefold + Unicode punctuation removal + whitespace collapse",
                "evaluation_scope": (
                    "transcript-aligned polygon line subset"
                    if args.condition == "polygon_lines"
                    else "official LOC page transcript"
                ),
            }
        )
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if args.condition == "polygon_lines":
            write_page_reassemblies(existing, args.output)
        task.finish("CHURRO bake-off condition complete", metrics={"cer": summary["cer"], "wer": summary["wer"]})
    except BaseException as error:
        task.fail(f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()

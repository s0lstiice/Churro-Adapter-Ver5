#!/usr/bin/env python3
"""Transcribe page or line images with CHURRO plus the bundled LoRA adapter."""

from __future__ import annotations

import argparse
import html
import json
import re
from contextlib import nullcontext
from pathlib import Path

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


DEFAULT_BASE_MODEL = "stanford-oval/churro-3B"
PAGE_PROMPTS = (
    "Transcribe the entirety of this historical document to XML format.",
    (
        "Transcribe every visible word in this entire historical document to XML. "
        "Preserve reading order, spelling, capitalization, and punctuation. Never summarize, "
        "shorten, skip, or replace visible text with comments or placeholders. Continue through "
        "the bottom of the page and close the XML only after all visible text is transcribed."
    ),
    (
        "Read this entire historical document from top to bottom and return only the complete "
        "plain-text transcription. Transcribe every visible line in reading order. Never summarize, "
        "abbreviate, skip content, or add comments. If one word is unreadable, mark only that word "
        "as [unclear] and continue through the bottom of the page."
    ),
)
LINE_PROMPT = (
    "Transcribe this single handwritten line exactly. Preserve visible spelling, capitalization, "
    "and punctuation. Return only the transcription."
)
OMISSION_PATTERNS = (
    re.compile(r"omitted\s+for\s+brevity", re.I),
    re.compile(r"(?:rest|remainder).{0,80}(?:omitted|not\s+transcribed)", re.I | re.S),
    re.compile(r"(?:page|document|content).{0,80}(?:omitted|not\s+transcribed)", re.I | re.S),
    re.compile(
        r"<!--(?:(?!-->).)*(?:\.{3,}|…+|omit(?:ted|s|ting)?|brevity|rest|remainder|continues?)"
        r"(?:(?!-->).)*-->",
        re.I | re.S,
    ),
)
IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def visible_text(raw: str) -> str:
    value = raw.strip()
    value = re.sub(r"^```(?:xml|markdown|text)?\s*", "", value, flags=re.I)
    value = re.sub(r"\s*```$", "", value)
    value = re.sub(r"<Metadata\b[^>]*>.*?</Metadata>\s*", "", value, flags=re.I | re.S)
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.I)
    value = re.sub(r"</(?:Line|Header|Paragraph|TextBlock|Page)>\s*", "\n", value, flags=re.I)
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    return "\n".join(" ".join(line.split()) for line in value.splitlines() if line.split())


def incomplete(raw: str, task: str, attempt: int) -> list[str]:
    if task == "line":
        return []
    reasons = []
    if attempt < 2 and "</HistoricalDocument>" not in raw:
        reasons.append("unclosed_historical_document")
    if any(pattern.search(raw) for pattern in OMISSION_PATTERNS):
        reasons.append("explicit_omission_placeholder")
    return reasons


def messages(image: Path, task: str, max_pixels: int, attempt: int) -> list[dict]:
    prompt = LINE_PROMPT if task == "line" else PAGE_PROMPTS[min(attempt, 2)]
    return [
        {"role": "system", "content": [{"type": "text", "text": prompt}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{image.resolve()}",
                    "min_pixels": 256 * 28 * 28,
                    "max_pixels": max_pixels,
                }
            ],
        },
    ]


def load_model(base_model: str, adapter: Path, full_precision: bool):
    processor = AutoProcessor.from_pretrained(adapter, use_fast=True)
    kwargs = {"low_cpu_mem_usage": True}
    if full_precision:
        kwargs.update(device_map="auto", dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit loading requires a CUDA GPU. Use --full-precision for CPU loading.")
        kwargs.update(
            device_map={"": 0},
            dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ),
        )
    model = AutoModelForImageTextToText.from_pretrained(base_model, **kwargs)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return processor, model


@torch.inference_mode()
def generate(processor, model, image: Path, task: str, max_pixels: int, max_new_tokens: int, attempt: int) -> str:
    conversation = messages(image, task, max_pixels, attempt)
    rendered = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(conversation)
    batch = processor(
        text=[rendered],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    generated = model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        no_repeat_ngram_size=8,
        use_cache=True,
    )
    trimmed = [output[len(source) :] for source, output in zip(batch.input_ids, generated)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def input_images(values: list[str], recursive: bool) -> list[Path]:
    found: list[Path] = []
    for value in values:
        path = Path(value).expanduser()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            found.append(path.resolve())
        elif path.is_dir():
            iterator = path.rglob("*") if recursive else path.glob("*")
            found.extend(item.resolve() for item in iterator if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
        else:
            raise FileNotFoundError(f"No supported image or directory found: {path}")
    return sorted(dict.fromkeys(found))


def progress_context(total: int):
    try:
        from universal_progress_monitor.progress_client import ProgressTask

        return ProgressTask("churro-loc-adapter-transcription", total=total, unit="images")
    except ImportError:
        return nullcontext()


def update_progress(progress, current: int, total: int, message: str) -> None:
    if progress is not None and hasattr(progress, "update"):
        progress.update(current=current, total=total, message=message)
    print(f"[{current}/{total}] {message}", flush=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Image file(s) or directories")
    parser.add_argument("--adapter", type=Path, default=root / "adapter")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output", type=Path, default=Path("churro_transcriptions"))
    parser.add_argument("--task", choices=("page", "line"), default="page")
    parser.add_argument("--max-pixels", type=int, default=1_605_632)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--retries", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--full-precision", action="store_true", help="Disable 4-bit loading")
    args = parser.parse_args()

    images = input_images(args.inputs, args.recursive)
    if not images:
        raise SystemExit("No supported images were found.")
    args.output.mkdir(parents=True, exist_ok=True)
    processor, model = load_model(args.base_model, args.adapter.resolve(), args.full_precision)
    results_path = args.output / "predictions.jsonl"

    with progress_context(len(images)) as progress:
        with results_path.open("w", encoding="utf-8") as results:
            for index, image in enumerate(images, start=1):
                attempts = []
                maximum_attempt = 0 if args.task == "line" else args.retries
                for attempt in range(maximum_attempt + 1):
                    raw = generate(
                        processor, model, image, args.task, args.max_pixels, args.max_new_tokens, attempt
                    )
                    text = visible_text(raw)
                    reasons = incomplete(raw, args.task, attempt)
                    attempts.append({"attempt": attempt, "raw": raw, "text": text, "reasons": reasons})
                    if not reasons:
                        break

                eligible = [item for item in attempts if not item["reasons"]] or attempts
                chosen = max(eligible, key=lambda item: len(item["text"]))
                stem = f"{index:05d}_{image.stem}"
                (args.output / f"{stem}.txt").write_text(chosen["text"] + "\n", encoding="utf-8")
                (args.output / f"{stem}.xml").write_text(chosen["raw"] + "\n", encoding="utf-8")
                record = {
                    "id": stem,
                    "image": str(image),
                    "task": args.task,
                    "base_model": args.base_model,
                    "adapter": str(args.adapter.resolve()),
                    "selected_attempt": chosen["attempt"],
                    "attempt_count": len(attempts),
                    "raw_prediction": chosen["raw"],
                    "prediction": chosen["text"],
                    "attempt_diagnostics": [
                        {"attempt": item["attempt"], "reasons": item["reasons"], "characters": len(item["text"])}
                        for item in attempts
                    ],
                }
                results.write(json.dumps(record, ensure_ascii=False) + "\n")
                results.flush()
                update_progress(progress, index, len(images), image.name)

    print(f"Saved {len(images)} transcription(s) to {args.output.resolve()}")


if __name__ == "__main__":
    main()

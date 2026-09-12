#!/usr/bin/env python3
"""Evaluate base CHURRO or a CHURRO QLoRA adapter on page manifests."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    StoppingCriteriaList,
)

from churro_decode_processors import (
    RepetitiveXmlTailStoppingCriteria,
    faithful_logits_processors,
)
from churro_visual_close_gate import (
    VisualCoverageBodyCloseGate,
    load_visual_line_evidence,
)
from churro_counterfactual_grounding import counterfactual_grounded_tail_guard
from churro_lora_layer_scaling import (
    apply_language_lora_layer_scales,
    parse_layer_scale_specs,
)
from churro_lora_component_scaling import (
    apply_language_lora_component_scales,
    parse_lora_component_scale_specs,
)
from churro_attention_head_scaling import (
    install_attention_head_scales,
    parse_attention_head_scale_specs,
)
from churro_interlayer_visual_contrast import install_interlayer_visual_contrast
from churro_par_qwen25 import PARConfig, install_par, parse_layers
from run_churro_loc_bakeoff import score, visible_text
from universal_progress_monitor.progress_client import ProgressTask


MODEL_ID = "stanford-oval/churro-3B"
DECODE_PROFILES = (
    "legacy",
    "churro-native",
    "faithful",
    "grounded-faithful",
    "coverage-audit-faithful",
    "grounded-coverage-audit-faithful",
    "coverage-gated-faithful",
    "grounded-coverage-faithful",
    "faithful-beam2",
    "par-pp",
    "par-full",
    "spin-visual",
)
SYSTEM_PROMPT = "Transcribe the entirety of this historical document to XML format."
RETRY_SYSTEM_PROMPT = (
    "Transcribe every visible word in this entire historical document to XML. "
    "Preserve reading order, spelling, capitalization, and punctuation. Never summarize, "
    "shorten, skip, or replace visible text with comments or placeholders such as "
    "'omitted for brevity'. Continue through the bottom of the page and close the XML only "
    "after all visible text has been transcribed."
    " This is an exhaustive-transcription retry because a prior response skipped content. "
    "Start again from the top. Output every visible line. If an individual word is unreadable, "
    "mark only that word as uncertain; never omit a region, paragraph, or remainder of the page."
)
PLAIN_RETRY_SYSTEM_PROMPT = (
    "Read this entire historical document from top to bottom and return only the complete plain-text "
    "transcription. Transcribe every visible line in reading order. Never summarize, abbreviate, "
    "skip content, add XML, or write comments/placeholders such as 'omitted for brevity'. If one "
    "word is unreadable, mark only that word as [unclear] and continue through the bottom of the page."
)
LINE_SYSTEM_PROMPT = (
    "Transcribe this single handwritten line exactly. Preserve the visible spelling, "
    "capitalization, and punctuation. Return only the transcription."
)


def task_type(row: dict) -> str:
    value = str(row.get("task_type") or row.get("granularity") or "page").strip().lower()
    if value in {"full_page", "full-page", "document"}:
        value = "page"
    if value not in {"page", "line"}:
        raise ValueError(f"{row.get('id', '<unknown>')} has unsupported task_type={value!r}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def messages_for(row: dict, max_pixels: int, retry_variant: int = 0) -> list[dict]:
    if task_type(row) == "line":
        prompt = LINE_SYSTEM_PROMPT
    elif retry_variant == 1:
        prompt = RETRY_SYSTEM_PROMPT
    elif retry_variant == 2:
        prompt = PLAIN_RETRY_SYSTEM_PROMPT
    else:
        prompt = SYSTEM_PROMPT
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": prompt}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{inference_image_path(row).resolve()}",
                    "min_pixels": 256 * 28 * 28,
                    "max_pixels": max_pixels,
                }
            ],
        },
    ]


def inference_image_path(row: dict) -> Path:
    """Return a portable local image override without changing bound provenance."""

    value = row.get("inference_image") or row.get("image")
    if not value:
        raise ValueError("manifest row has no image or inference_image")
    return Path(str(value))


def load_model(model_id: str, adapter: Path | None):
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    processor_source = str(adapter) if adapter else model_id
    processor = AutoProcessor.from_pretrained(processor_source, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map={"": 0},
        quantization_config=quantization,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return processor, model


def generation_settings(decode_profile: str) -> dict:
    """Return deterministic generation settings for a named decode profile.

    ``legacy`` preserves this project's historical evaluator exactly.  The
    other profiles remove the global 8-gram ban, which is not present in the
    released CHURRO generation configuration and can force valid repeated
    phrases to be substituted or omitted.  CHURRO's mild repetition penalty is
    retained because fully neutral greedy decoding can enter XML line loops.
    """

    if decode_profile not in DECODE_PROFILES:
        raise ValueError(f"unknown decode profile {decode_profile!r}")
    faithful = decode_profile in {
        "faithful",
        "grounded-faithful",
        "coverage-audit-faithful",
        "grounded-coverage-audit-faithful",
        "coverage-gated-faithful",
        "grounded-coverage-faithful",
        "faithful-beam2",
        "par-pp",
        "par-full",
        "spin-visual",
    }
    settings = {
        "do_sample": False,
        "repetition_penalty": 1.0 if faithful else 1.05,
        "no_repeat_ngram_size": (
            8 if decode_profile == "legacy" else (32 if faithful else 0)
        ),
        "use_cache": True,
        "response_only_repetition_penalty": 1.01 if faithful else None,
        "targeted_loop_guard": (
            "visual_grounding_plus_repetitive_xml_tail"
            if decode_profile == "grounded-faithful"
            else (
                "visual_grounding_plus_semantic_visual_close_gate"
                if decode_profile in {
                    "grounded-coverage-audit-faithful",
                    "grounded-coverage-faithful",
                }
                else (
                "audit_semantic_body_close_with_visual_lines"
                if decode_profile == "coverage-audit-faithful"
                else (
                    "gate_semantic_body_close_with_visual_lines"
                    if decode_profile == "coverage-gated-faithful"
                    else ("no_repeat_ngram_size_32" if faithful else None)
                )
                )
            )
        ),
    }
    if decode_profile == "faithful-beam2":
        # Beam width two is a bounded delayed-commitment diagnostic: a recent
        # alternative can survive long enough for later image-conditioned
        # evidence to select it.  It is deliberately not a default because it
        # can be slower and can strengthen the language prior on some pages.
        settings.update(
            {
                "num_beams": 2,
                "num_return_sequences": 1,
                "early_stopping": False,
                "length_penalty": 1.0,
            }
        )
    return settings


def model_generation_settings(decode_profile: str) -> dict:
    """Strip audit-only keys before calling ``model.generate``."""

    return {
        key: value
        for key, value in generation_settings(decode_profile).items()
        if key not in {"response_only_repetition_penalty", "targeted_loop_guard"}
    }


@torch.inference_mode()
def infer(
    processor,
    model,
    row: dict,
    max_pixels: int,
    max_new_tokens: int,
    retry_variant: int = 0,
    decode_profile: str = "legacy",
    par_controller=None,
    visual_close_evidence: list[dict] | None = None,
    visual_close_options: dict | None = None,
    generation_audit: dict | None = None,
    continuous_loop_stop: bool = False,
) -> str:
    messages = messages_for(row, max_pixels, retry_variant=retry_variant)
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    batch = processor(
        text=[rendered],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    if par_controller is not None:
        par_controller.begin_page(batch.input_ids)
    generate_kwargs = model_generation_settings(decode_profile)
    visual_close_gate = None
    if decode_profile in {
        "coverage-audit-faithful",
        "grounded-coverage-audit-faithful",
        "coverage-gated-faithful",
        "grounded-coverage-faithful",
    }:
        options = dict(visual_close_options or {})
        options["audit_only"] = decode_profile in {
            "coverage-audit-faithful",
            "grounded-coverage-audit-faithful",
        }
        visual_close_gate = VisualCoverageBodyCloseGate(
            processor.tokenizer,
            prompt_length=int(batch.input_ids.shape[-1]),
            visual_lines=visual_close_evidence or [],
            **options,
        )
    if decode_profile in {
        "faithful",
        "grounded-faithful",
        "coverage-audit-faithful",
        "grounded-coverage-audit-faithful",
        "coverage-gated-faithful",
        "grounded-coverage-faithful",
        "faithful-beam2",
        "par-pp",
        "par-full",
        "spin-visual",
    }:
        generate_kwargs["logits_processor"] = faithful_logits_processors(
            prompt_length=int(batch.input_ids.shape[-1]),
            extra_processors=[visual_close_gate] if visual_close_gate is not None else [],
        )
    loop_stopper = None
    if continuous_loop_stop:
        loop_stopper = RepetitiveXmlTailStoppingCriteria(
            processor.tokenizer,
            prompt_length=int(batch.input_ids.shape[-1]),
        )
        generate_kwargs["stopping_criteria"] = StoppingCriteriaList([loop_stopper])
    generated = model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        **generate_kwargs,
    )
    if visual_close_gate is not None:
        visual_close_gate.finalize(generated)
        if generation_audit is not None:
            generation_audit["visual_close_gate"] = visual_close_gate.audit()
    if loop_stopper is not None and generation_audit is not None:
        generation_audit["single_pass_loop_stop"] = loop_stopper.audit
    trimmed = [output[len(input_ids) :] for input_ids, output in zip(batch.input_ids, generated)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


@torch.inference_mode()
def infer_batch(
    processor,
    model,
    rows: list[dict],
    max_pixels: int,
    max_new_tokens: int,
    retry_variant: int = 0,
    decode_profile: str = "legacy",
) -> list[str]:
    """Generate an initial pass for multiple independent OCR regions.

    This deliberately excludes PAR and semantic close-gate controllers, whose
    state is page-specific. Grounded-faithful counterfactual checks remain a
    separate per-result step after generation.
    """

    if not rows:
        return []
    conversations = [
        messages_for(row, max_pixels, retry_variant=retry_variant) for row in rows
    ]
    rendered = [
        processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        for messages in conversations
    ]
    image_inputs, video_inputs = process_vision_info(conversations)
    tokenizer = processor.tokenizer
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        batch = processor(
            text=rendered,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
    finally:
        tokenizer.padding_side = previous_padding_side
    generate_kwargs = model_generation_settings(decode_profile)
    if decode_profile in {
        "faithful",
        "grounded-faithful",
        "faithful-beam2",
    }:
        generate_kwargs["logits_processor"] = faithful_logits_processors(
            prompt_length=int(batch.input_ids.shape[-1])
        )
    generated = model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        **generate_kwargs,
    )
    prompt_width = int(batch.input_ids.shape[-1])
    trimmed = [output[prompt_width:] for output in generated]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )


def counterfactual_negative_for(row: dict, candidates: list[dict]) -> dict | None:
    """Pick a deterministic unrelated page for visual counterfactual scoring."""

    source_image = str(row.get("image") or "")
    source_item = str(row.get("item_id") or row.get("source") or "")
    eligible = [
        candidate
        for candidate in candidates
        if task_type(candidate) == task_type(row)
        and str(candidate.get("image") or "") != source_image
        and str(candidate.get("item_id") or candidate.get("source") or "") != source_item
        # Manifests used by resumable download queues can legitimately contain
        # rows whose image has not arrived yet.  Such a row is not a usable
        # visual counterfactual and must never make an otherwise valid OCR run
        # fail with FileNotFoundError.
        and inference_image_path(candidate).is_file()
    ]
    if not eligible:
        return None
    return sorted(eligible, key=lambda candidate: str(candidate.get("id") or candidate.get("page_id")))[0]


def visual_evidence_for(row: dict, by_page: dict[str, list[dict]]) -> list[dict]:
    """Resolve common page identities without silently combining two pages."""

    candidates = []
    for value in (
        row.get("id"),
        row.get("page_id"),
        Path(str(row.get("image") or "")).stem,
    ):
        key = str(value or "")
        if key and key not in candidates:
            candidates.append(key)
    matches = [(key, by_page[key]) for key in candidates if key in by_page]
    if len(matches) > 1 and any(rows != matches[0][1] for _, rows in matches[1:]):
        raise ValueError(f"ambiguous visual line evidence for {candidates}")
    return matches[0][1] if matches else []


def apply_counterfactual_grounding_guard(
    processor,
    model,
    row: dict,
    negative: dict | None,
    raw: str,
    max_pixels: int,
    retry_variant: int,
) -> tuple[str, dict]:
    if negative is None:
        return raw, {
            "schema_version": "churro_counterfactual_grounding.v1",
            "applied": False,
            "reason": "no_unrelated_counterfactual_page_available",
            "counterfactual_forward_passes": 0,
        }
    negative_row = dict(row)
    negative_row["image"] = negative["image"]
    if negative.get("inference_image"):
        negative_row["inference_image"] = negative["inference_image"]
    else:
        negative_row.pop("inference_image", None)
    guarded, audit = counterfactual_grounded_tail_guard(
        processor,
        model,
        messages_for(row, max_pixels, retry_variant=retry_variant),
        messages_for(negative_row, max_pixels, retry_variant=retry_variant),
        raw,
    )
    audit["counterfactual_page_id"] = str(negative.get("id") or negative.get("page_id"))
    return guarded, audit


OMISSION_PATTERNS = (
    re.compile(r"omitted\s+for\s+brevity", re.IGNORECASE),
    # Models sometimes hide an omission in an XML comment.  A closed XML
    # document is not complete when its body contains ``<!-- ... -->``.
    re.compile(
        r"<!--(?:(?!-->).)*(?:\.{3,}|…+|omit(?:ted|s|ting)?|brevity|remainder|continues?)(?:(?!-->).)*-->",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(?:the\s+)?(?:rest|remainder)\s+of\s+(?:the\s+)?"
        r"(?:page|document|text|content|letter|letter\s+body|body).{0,100}?"
        r"(?:is\s+)?(?:omitted|not\s+transcribed|not\s+included)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(?:page|document|text|content|letter|letter\s+body|body).{0,80}?"
        r"(?:omitted|not\s+transcribed)\s+(?:for|due\s+to)\s+",
        re.IGNORECASE | re.DOTALL,
    ),
)


def incomplete_generation_reasons(raw: str, row: dict) -> list[str]:
    if task_type(row) != "page":
        return []
    reasons = []
    if "</HistoricalDocument>" not in raw:
        reasons.append("unclosed_historical_document")
    if any(pattern.search(raw) for pattern in OMISSION_PATTERNS):
        reasons.append("explicit_omission_placeholder")
    return reasons


def summarize(rows: list[dict]) -> dict:
    totals = defaultdict(int)
    scored_rows = [
        row for row in rows
        if isinstance(row.get("metrics"), dict) and str(row.get("text") or "").strip()
    ]
    for row in scored_rows:
        for key in (
            "character_edits",
            "target_characters",
            "word_edits",
            "target_words",
            "prediction_characters",
            "prediction_words",
        ):
            totals[key] += row["metrics"][key]
    result = {
        "examples": len(rows),
        "scored_examples": len(scored_rows),
        "unscored_examples": len(rows) - len(scored_rows),
        "pages": sum(task_type(row) == "page" for row in rows),
        "lines": sum(task_type(row) == "line" for row in rows),
        **totals,
        "cer": (
            totals["character_edits"] / max(1, totals["target_characters"])
            if scored_rows else None
        ),
        "wer": (
            totals["word_edits"] / max(1, totals["target_words"])
            if scored_rows else None
        ),
        "output_character_ratio": (
            totals["prediction_characters"] / max(1, totals["target_characters"])
            if scored_rows else None
        ),
        "generation_truncated_examples": sum(bool(row.get("generation_truncated")) for row in rows),
        "generation_truncation_rate": sum(bool(row.get("generation_truncated")) for row in rows)
        / max(1, len(rows)),
        "explicit_omission_examples": sum(
            "explicit_omission_placeholder" in row.get("generation_incomplete_reasons", [])
            for row in rows
        ),
        "generation_incomplete_examples": sum(bool(row.get("generation_incomplete")) for row in rows),
        "retry_attempted_examples": sum(bool(row.get("generation_retry_used")) for row in rows),
        "retry_succeeded_examples": sum(bool(row.get("generation_retry_succeeded")) for row in rows),
        "visual_close_gate_examples": sum(
            isinstance(row.get("visual_close_gate"), dict) for row in rows
        ),
        "visual_close_gate_interventions": sum(
            int((row.get("visual_close_gate") or {}).get("interventions") or 0)
            for row in rows
        ),
        "visual_close_gate_candidate_boundaries": sum(
            sum(
                event.get("evidence_reason") == "strong_lower_visual_suffix"
                for event in (row.get("visual_close_gate") or {}).get("events", [])
            )
            for row in rows
        ),
    }
    by_task = {}
    for name in ("line", "page"):
        selected = [row for row in rows if task_type(row) == name]
        if selected:
            by_task[name] = summarize(selected) if len(selected) < len(rows) else {
                "examples": len(selected),
                "cer": result["cer"],
                "wer": result["wer"],
            }
    if len(by_task) > 1:
        result["by_task"] = by_task
    return result


def clean_text_path(output: Path, row_id: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", row_id).strip("_.") or "prediction"
    return output / "clean_text" / f"{safe_id}.txt"


def write_clean_text(output: Path, row: dict) -> None:
    destination = clean_text_path(output, str(row.get("id") or row.get("page_id") or "prediction"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(str(row.get("prediction") or "").rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("churro_loc_fullpage_dataset_v1/test.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument(
        "--adapter-label",
        help=(
            "Portable output label for the adapter. The model still loads from --adapter; "
            "use this only when a transferred shard must preserve canonical provenance."
        ),
    )
    parser.add_argument("--max-pixels", type=int, default=1605632)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument(
        "--continuous-page-budget",
        action="store_true",
        help=(
            "Use --max-new-tokens as one continuous page-generation ceiling instead "
            "of honoring a smaller per-row ceiling. Page decoding still stops normally "
            "when the model emits EOS, so the larger ceiling costs time only when the "
            "page continues. This mode disables full-page regeneration retries."
        ),
    )
    parser.add_argument(
        "--selective-incomplete-retry",
        action="store_true",
        help=(
            "With --continuous-page-budget, permit at most one recovery generation only "
            "when the first page is structurally incomplete and the streaming loop guard "
            "did not fire. Repetitive continuations are stopped rather than regenerated."
        ),
    )
    parser.add_argument(
        "--initial-batch-size",
        type=int,
        choices=(1, 2),
        default=1,
        help=(
            "Generate compatible first attempts in pairs. Retries and visual "
            "counterfactual checks remain independent."
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--decode-profile",
        choices=DECODE_PROFILES,
        default="legacy",
        help=(
            "legacy keeps the prior 8-gram ban; churro-native removes it; faithful also "
            "limits repetition handling to generated-text loops; grounded-faithful removes "
            "only visually unsupported repetitive XML tails; coverage-audit-faithful records "
            "independent polygon-line evidence at body closes without changing output; "
            "grounded-coverage-audit-faithful adds the same repetitive-tail grounding used "
            "by the active combined profile, permitting an apples-to-apples gate audit; "
            "coverage-gated-faithful softly delays a close only when strong lower visual lines "
            "remain; grounded-coverage-faithful combines that close gate with the promoted "
            "repetitive-tail grounding guard; par-pp/par-full add the Qwen2.5-VL "
            "OCR grounding intervention; spin-visual dynamically attenuates the heads least "
            "attentive to image tokens"
        ),
    )
    parser.add_argument(
        "--visual-line-predictions",
        type=Path,
        help=(
            "Ordered transcript-blind polygon-line recognition JSONL used by coverage-* "
            "profiles. The line text is evidence only and is never inserted into page output."
        ),
    )
    parser.add_argument(
        "--visual-line-verifier-predictions",
        type=Path,
        help="Optional second-view line reads; matching views must agree before gating.",
    )
    parser.add_argument("--visual-close-penalty", type=float, default=2.5)
    parser.add_argument("--visual-close-maximum-penalty", type=float, default=6.0)
    parser.add_argument("--visual-close-maximum-margin", type=float, default=4.0)
    parser.add_argument("--visual-close-required-continue-margin", type=float, default=0.25)
    parser.add_argument("--visual-close-anchor-similarity", type=float, default=0.70)
    parser.add_argument("--visual-close-minimum-remaining-lines", type=int, default=2)
    parser.add_argument("--visual-close-minimum-view-consensus", type=float, default=0.82)
    parser.add_argument(
        "--visual-close-require-verifier",
        action="store_true",
        help="Fail closed unless a second crop/view independently supports each residual line.",
    )
    parser.add_argument(
        "--visual-close-require-page-evidence",
        action="store_true",
        help="Refuse the run if any selected page has no joined visual-line evidence.",
    )
    parser.add_argument("--visual-close-maximum-interventions", type=int, default=2)
    parser.add_argument(
        "--par-layers",
        default="0,1,2,3",
        help="Comma-separated language-model layers for PAR profiles.",
    )
    parser.add_argument("--par-seed", type=int, default=1729)
    parser.add_argument(
        "--spin-keep-fraction",
        type=float,
        default=0.90,
        help="For spin-visual, fraction of the most image-attentive heads kept intact.",
    )
    parser.add_argument(
        "--spin-suppression-scale",
        type=float,
        default=0.50,
        help="For spin-visual, output scale for dynamically language-dominant heads.",
    )
    parser.add_argument(
        "--lora-layer-scale",
        action="append",
        default=[],
        help=(
            "Experimental, opt-in language LoRA scaling as START-END:SCALE; "
            "repeat for disjoint ranges. Defaults remain unchanged."
        ),
    )
    parser.add_argument(
        "--lora-component-scale",
        action="append",
        default=[],
        help=(
            "Experimental, opt-in LoRA projection scaling as "
            "LAYER:PROJECTION:SCALE, e.g. 27:o_proj:0.5."
        ),
    )
    parser.add_argument(
        "--attention-head-scale",
        action="append",
        default=[],
        help=(
            "Experimental, opt-in language attention-head output scaling as "
            "LAYER:HEAD:SCALE; repeat for multiple heads. Defaults remain unchanged."
        ),
    )
    parser.add_argument(
        "--interlayer-contrast-layer",
        type=int,
        help=(
            "Experimental, opt-in iTaD-style intermediate decoder layer. "
            "Disabled by default and does not modify stored weights."
        ),
    )
    parser.add_argument(
        "--interlayer-contrast-strength",
        type=float,
        default=1.0,
        help="Premature log-probability subtraction strength in [0,1].",
    )
    parser.add_argument(
        "--interlayer-contrast-cutoff",
        type=float,
        default=0.03,
        help="Keep tokens whose mature probability is this fraction of the maximum.",
    )
    parser.add_argument("--id", action="append", default=[], help="Evaluate only these row IDs/page IDs.")
    parser.add_argument(
        "--max-incomplete-retries",
        type=int,
        default=2,
        help="Retry page generations that truncate or explicitly omit visible content.",
    )
    parser.add_argument(
        "--force-page-retries",
        action="store_true",
        help=(
            "Run the configured retry attempts for every page even when the initial XML is "
            "structurally complete. This is opt-in because silent omissions cannot be detected "
            "from XML health alone and forced retries increase inference cost."
        ),
    )
    args = parser.parse_args()
    if args.adapter_label and not args.adapter:
        parser.error("--adapter-label requires --adapter")
    adapter_record = args.adapter_label if args.adapter_label is not None else (
        str(args.adapter) if args.adapter else None
    )
    if args.initial_batch_size > 1 and args.decode_profile not in {
        "legacy",
        "churro-native",
        "faithful",
        "grounded-faithful",
    }:
        parser.error("batch-2 initial generation is not supported by this decode profile")
    if args.initial_batch_size > 1 and (
        args.lora_layer_scale
        or args.lora_component_scale
        or args.attention_head_scale
        or args.interlayer_contrast_layer is not None
    ):
        parser.error("batch-2 is disabled for experimental layer/head interventions")
    if args.continuous_page_budget and args.initial_batch_size > 1:
        parser.error("continuous page generation requires batch size one")
    if args.selective_incomplete_retry and not args.continuous_page_budget:
        parser.error("--selective-incomplete-retry requires --continuous-page-budget")
    lora_layer_scales = parse_layer_scale_specs(args.lora_layer_scale)
    lora_component_scales = parse_lora_component_scale_specs(args.lora_component_scale)
    attention_head_scales = parse_attention_head_scale_specs(args.attention_head_scale)
    if (lora_layer_scales or lora_component_scales) and not args.adapter:
        parser.error("LoRA scaling requires --adapter")
    coverage_profile = args.decode_profile in {
        "coverage-audit-faithful",
        "grounded-coverage-audit-faithful",
        "coverage-gated-faithful",
        "grounded-coverage-faithful",
    }
    if coverage_profile and args.visual_line_predictions is None:
        parser.error(f"--decode-profile {args.decode_profile} requires --visual-line-predictions")
    if args.visual_close_require_verifier and args.visual_line_verifier_predictions is None:
        parser.error("--visual-close-require-verifier needs --visual-line-verifier-predictions")
    if args.visual_close_require_page_evidence and not coverage_profile:
        parser.error("--visual-close-require-page-evidence requires a coverage decode profile")
    close_numeric = (
        args.visual_close_penalty,
        args.visual_close_maximum_penalty,
        args.visual_close_maximum_margin,
        args.visual_close_required_continue_margin,
        args.visual_close_anchor_similarity,
        args.visual_close_minimum_view_consensus,
    )
    if any(not math.isfinite(float(value)) for value in close_numeric):
        parser.error("visual-close numeric arguments must be finite")
    if args.visual_close_penalty < 0 or args.visual_close_maximum_penalty < args.visual_close_penalty:
        parser.error("invalid visual-close penalty range")
    if args.visual_close_maximum_margin < 0 or args.visual_close_required_continue_margin < 0:
        parser.error("visual-close margins must be non-negative")
    if (
        args.visual_close_maximum_penalty
        < args.visual_close_maximum_margin + args.visual_close_required_continue_margin
    ):
        parser.error(
            "--visual-close-maximum-penalty must cover maximum-margin + required-continue-margin"
        )
    if not 0.0 <= args.visual_close_anchor_similarity <= 1.0:
        parser.error("--visual-close-anchor-similarity must be in [0, 1]")
    if not 0.0 <= args.visual_close_minimum_view_consensus <= 1.0:
        parser.error("--visual-close-minimum-view-consensus must be in [0, 1]")
    visual_evidence_by_page: dict[str, list[dict]] = {}
    visual_evidence_summary = None
    if args.visual_line_predictions is not None:
        visual_evidence_by_page, visual_evidence_summary = load_visual_line_evidence(
            args.visual_line_predictions,
            args.visual_line_verifier_predictions,
        )
    visual_close_options = {
        "penalty": args.visual_close_penalty,
        "maximum_penalty": args.visual_close_maximum_penalty,
        "maximum_close_margin": args.visual_close_maximum_margin,
        "required_continue_margin": args.visual_close_required_continue_margin,
        "minimum_anchor_similarity": args.visual_close_anchor_similarity,
        "minimum_remaining_lines": args.visual_close_minimum_remaining_lines,
        "minimum_view_consensus": args.visual_close_minimum_view_consensus,
        "require_verifier": args.visual_close_require_verifier,
        "maximum_interventions": args.visual_close_maximum_interventions,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    all_manifest_rows = read_jsonl(args.manifest)
    # LOC alignment manifests commonly use ``page_id`` without duplicating it
    # into ``id``.  Normalize the identity once so filtering, resume state,
    # output naming, and inference all use the same stable key.
    for row in all_manifest_rows:
        if not row.get("id") and row.get("page_id"):
            row["id"] = str(row["page_id"])
    rows = list(all_manifest_rows)
    if args.id:
        requested = set(args.id)
        rows = [row for row in rows if str(row.get("id") or row.get("page_id")) in requested]
        missing = requested - {str(row.get("id") or row.get("page_id")) for row in rows}
        if missing:
            raise KeyError(f"requested IDs not found in manifest: {sorted(missing)}")
    if args.limit:
        rows = rows[: args.limit]
    visual_evidence_join = None
    if coverage_profile:
        matched_ids = []
        missing_ids = []
        for row in rows:
            identity = str(row.get("id") or row.get("page_id"))
            if visual_evidence_for(row, visual_evidence_by_page):
                matched_ids.append(identity)
            else:
                missing_ids.append(identity)
        visual_evidence_join = {
            "selected_pages": len(rows),
            "matched_pages": len(matched_ids),
            "missing_pages": len(missing_ids),
            "missing_page_ids": missing_ids[:100],
        }
        if args.visual_close_require_page_evidence and missing_ids:
            parser.error(
                "missing visual-line evidence for selected pages: "
                + ", ".join(missing_ids[:10])
                + (" ..." if len(missing_ids) > 10 else "")
            )
    predictions_path = args.output / "predictions.jsonl"
    existing = read_jsonl(predictions_path) if predictions_path.exists() else []
    existing_changed = False
    for result in existing:
        refreshed_reasons = incomplete_generation_reasons(
            str(result.get("raw_prediction") or ""), result
        )
        if result.get("generation_incomplete_reasons") != refreshed_reasons:
            result["generation_incomplete_reasons"] = refreshed_reasons
            result["generation_incomplete"] = bool(refreshed_reasons)
            existing_changed = True
        if not str(result.get("text") or "").strip() and result.get("metrics") is not None:
            result["metrics"] = None
            existing_changed = True
        write_clean_text(args.output, result)
    if existing_changed:
        with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
            for result in existing:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    done = {row["id"] for row in existing}
    pending = [row for row in rows if row["id"] not in done]
    run_name = "adapter" if args.adapter else "base"
    task_counts = {name: sum(task_type(row) == name for row in rows) for name in ("line", "page")}
    mixed = all(task_counts.values())
    task = ProgressTask(
        f"CHURRO {'mixed-scale' if mixed else ('line' if task_counts['line'] else 'full-page')} {run_name} evaluation",
        total=len(rows),
        unit="examples",
        task_id=f"churro-{'mixed' if mixed else ('line' if task_counts['line'] else 'fullpage')}-{run_name}-eval-{args.output.name}",
        output_dir=args.output,
        metadata={
            "model": args.model,
            "adapter": adapter_record,
            "task_counts": task_counts,
            "decode_profile": args.decode_profile,
            "visual_line_evidence": visual_evidence_summary,
            "visual_line_evidence_join": visual_evidence_join,
            "visual_close_options": visual_close_options if coverage_profile else None,
            "lora_layer_scales": lora_layer_scales,
            "lora_component_scales": {
                f"{layer}:{projection}": scale
                for (layer, projection), scale in sorted(lora_component_scales.items())
            },
            "attention_head_scales": {
                f"{layer}:{head}": scale
                for (layer, head), scale in sorted(attention_head_scales.items())
            },
            "interlayer_visual_contrast": (
                {
                    "layer": args.interlayer_contrast_layer,
                    "strength": args.interlayer_contrast_strength,
                    "plausibility_cutoff": args.interlayer_contrast_cutoff,
                }
                if args.interlayer_contrast_layer is not None else None
            ),
        },
    )
    try:
        task.update(len(existing), message="loading model")
        processor = model = par_controller = attention_head_controller = None
        interlayer_contrast_controller = None
        if pending:
            processor, model = load_model(args.model, args.adapter)
            lora_scaling_audit = (
                apply_language_lora_layer_scales(model, lora_layer_scales)
                if lora_layer_scales else None
            )
            lora_component_scaling_audit = (
                apply_language_lora_component_scales(model, lora_component_scales)
                if lora_component_scales else None
            )
            attention_head_controller = (
                install_attention_head_scales(model, attention_head_scales)
                if attention_head_scales else None
            )
            interlayer_contrast_controller = (
                install_interlayer_visual_contrast(
                    model,
                    layer=args.interlayer_contrast_layer,
                    strength=args.interlayer_contrast_strength,
                    plausibility_cutoff=args.interlayer_contrast_cutoff,
                )
                if args.interlayer_contrast_layer is not None else None
            )
            if args.decode_profile in {"par-pp", "par-full", "spin-visual"}:
                par_controller = install_par(
                    model,
                    PARConfig(
                        pp_enabled=args.decode_profile in {"par-pp", "par-full"},
                        far_enabled=args.decode_profile == "par-full",
                        spin_enabled=args.decode_profile == "spin-visual",
                        target_layers=parse_layers(args.par_layers),
                        seed=args.par_seed,
                        spin_keep_fraction=args.spin_keep_fraction,
                        spin_suppression_scale=args.spin_suppression_scale,
                    ),
                )
        with predictions_path.open("a", encoding="utf-8") as handle:
            def token_limit_for(value: dict) -> int:
                if args.continuous_page_budget and task_type(value) == "page":
                    # Preserve explicit safety caps assigned by layout analysis
                    # to dark/blank/nonmeaningful regions.  The continuous
                    # ceiling is for normal text-bearing pages, not covers.
                    if value.get("generation_limit_reason") and value.get(
                        "generation_max_new_tokens"
                    ) is not None:
                        return max(
                            32,
                            min(
                                args.max_new_tokens,
                                int(value["generation_max_new_tokens"]),
                            ),
                        )
                    return max(32, args.max_new_tokens)
                return max(
                    32,
                    min(
                        args.max_new_tokens,
                        int(value.get("generation_max_new_tokens") or args.max_new_tokens),
                    ),
                )

            def initial_generation_stream():
                if args.initial_batch_size == 1:
                    for value in pending:
                        yield value, None
                    return
                index = 0
                while index < len(pending):
                    first = pending[index]
                    limit = token_limit_for(first)
                    group = [first]
                    index += 1
                    while (
                        index < len(pending)
                        and len(group) < args.initial_batch_size
                        and token_limit_for(pending[index]) == limit
                    ):
                        group.append(pending[index])
                        index += 1
                    if len(group) == 1:
                        generated_group = [
                            infer(
                                processor,
                                model,
                                group[0],
                                args.max_pixels,
                                limit,
                                decode_profile=args.decode_profile,
                            )
                        ]
                    else:
                        generated_group = infer_batch(
                            processor,
                            model,
                            group,
                            args.max_pixels,
                            limit,
                            decode_profile=args.decode_profile,
                        )
                    yield from zip(group, generated_group)

            for row, precomputed_first_raw in initial_generation_stream():
                counterfactual_negative = counterfactual_negative_for(row, all_manifest_rows)
                row_visual_evidence = visual_evidence_for(row, visual_evidence_by_page)
                row_max_new_tokens = token_limit_for(row)
                row_max_incomplete_retries = max(
                    0,
                    min(
                        args.max_incomplete_retries,
                        int(
                            row.get("generation_max_incomplete_retries")
                            if row.get("generation_max_incomplete_retries") is not None
                            else args.max_incomplete_retries
                        ),
                    ),
                )
                if args.continuous_page_budget and task_type(row) == "page":
                    row_max_incomplete_retries = (
                        min(1, row_max_incomplete_retries)
                        if args.selective_incomplete_retry
                        else 0
                    )
                grounding_attempts = []
                close_gate_attempts = []
                first_generation_audit: dict = {}
                first_raw = precomputed_first_raw
                if first_raw is None:
                    first_raw = infer(
                        processor,
                        model,
                        row,
                        args.max_pixels,
                        row_max_new_tokens,
                        decode_profile=args.decode_profile,
                        par_controller=par_controller,
                        visual_close_evidence=row_visual_evidence,
                        visual_close_options=visual_close_options,
                        generation_audit=first_generation_audit,
                        continuous_loop_stop=args.continuous_page_budget,
                    )
                close_gate_attempts.append(first_generation_audit.get("visual_close_gate"))
                first_audit = {}
                if args.decode_profile in {
                    "grounded-faithful",
                    "grounded-coverage-audit-faithful",
                    "grounded-coverage-faithful",
                }:
                    guarded, first_audit = apply_counterfactual_grounding_guard(
                        processor,
                        model,
                        row,
                        counterfactual_negative,
                        first_raw,
                        args.max_pixels,
                        retry_variant=0,
                    )
                    if first_audit.get("applied"):
                        first_audit["_raw_prediction_before_guard"] = first_raw
                    first_raw = guarded
                grounding_attempts.append(first_audit)
                raw_attempts = [
                    first_raw
                ]
                reason_attempts = [incomplete_generation_reasons(raw_attempts[0], row)]
                initial_incomplete = bool(reason_attempts[0])
                first_loop_stop = first_generation_audit.get("single_pass_loop_stop")
                selective_recovery_eligible = bool(
                    args.selective_incomplete_retry
                    and initial_incomplete
                    and first_loop_stop is None
                )
                retry_count = (
                    row_max_incomplete_retries
                    if (
                        (initial_incomplete and not args.continuous_page_budget)
                        or selective_recovery_eligible
                        or (args.force_page_retries and task_type(row) == "page")
                    )
                    else 0
                )
                generation_audits = [first_generation_audit]
                for retry_index in range(retry_count):
                    retry_variant = 1 + (retry_index % 2)
                    retry_generation_audit: dict = {}
                    retry_raw = infer(
                        processor,
                        model,
                        row,
                        args.max_pixels,
                        row_max_new_tokens,
                        retry_variant=retry_variant,
                        decode_profile=args.decode_profile,
                        par_controller=par_controller,
                        visual_close_evidence=row_visual_evidence,
                        visual_close_options=visual_close_options,
                        generation_audit=retry_generation_audit,
                        continuous_loop_stop=args.continuous_page_budget,
                    )
                    generation_audits.append(retry_generation_audit)
                    close_gate_attempts.append(
                        retry_generation_audit.get("visual_close_gate")
                    )
                    retry_audit = {}
                    if args.decode_profile in {
                        "grounded-faithful",
                        "grounded-coverage-audit-faithful",
                        "grounded-coverage-faithful",
                    }:
                        guarded, retry_audit = apply_counterfactual_grounding_guard(
                            processor,
                            model,
                            row,
                            counterfactual_negative,
                            retry_raw,
                            args.max_pixels,
                            retry_variant=retry_variant,
                        )
                        if retry_audit.get("applied"):
                            retry_audit["_raw_prediction_before_guard"] = retry_raw
                        retry_raw = guarded
                    raw_attempts.append(retry_raw)
                    grounding_attempts.append(retry_audit)
                    reason_attempts.append(incomplete_generation_reasons(raw_attempts[-1], row))
                # Prefer a complete candidate. If every attempt is incomplete,
                # retain the one with the most visible text rather than hiding
                # the failure by stripping its placeholder.
                complete_indices = [index for index, reasons in enumerate(reason_attempts) if not reasons]
                complete_xml_indices = [
                    index
                    for index in complete_indices
                    if "</HistoricalDocument>" in raw_attempts[index]
                ]
                if complete_xml_indices:
                    # Prefer a complete structured transcription. A longer
                    # plain-text retry can be verbose or hallucinatory, so it
                    # is only a fallback when exhaustive XML also fails.
                    selected_attempt = max(
                        complete_xml_indices,
                        key=lambda index: len(visible_text(raw_attempts[index])),
                    )
                elif complete_indices:
                    selected_attempt = max(
                        complete_indices,
                        key=lambda index: len(visible_text(raw_attempts[index])),
                    )
                else:
                    selected_attempt = max(
                        range(len(raw_attempts)),
                        key=lambda index: len(visible_text(raw_attempts[index])),
                    )
                raw = raw_attempts[selected_attempt]
                selected_close_gate_audit = close_gate_attempts[selected_attempt]
                selected_grounding_audit = grounding_attempts[selected_attempt]
                raw_before_grounding_guard = selected_grounding_audit.pop(
                    "_raw_prediction_before_guard", None
                )
                prediction = visible_text(raw)
                incomplete_reasons = reason_attempts[selected_attempt]
                result = {
                    **row,
                    "model": args.model,
                    "adapter": adapter_record,
                    "decode_profile": args.decode_profile,
                    "generation_settings": generation_settings(args.decode_profile),
                    "lora_layer_scaling": lora_scaling_audit,
                    "lora_component_scaling": lora_component_scaling_audit,
                    "attention_head_scaling": (
                        attention_head_controller.audit
                        if attention_head_controller is not None
                        else None
                    ),
                    "interlayer_visual_contrast": (
                        interlayer_contrast_controller.audit()
                        if interlayer_contrast_controller is not None
                        else None
                    ),
                    "par_audit": par_controller.audit() if par_controller is not None else None,
                    "grounding_guard": selected_grounding_audit or None,
                    "visual_close_gate": selected_close_gate_audit,
                    "visual_close_gate_attempts": close_gate_attempts,
                    "raw_prediction_before_grounding_guard": raw_before_grounding_guard,
                    "raw_prediction": raw,
                    "prediction": prediction,
                    "generation_max_new_tokens": row_max_new_tokens,
                    "generation_initial_batch_size": args.initial_batch_size,
                    "generation_max_incomplete_retries": row_max_incomplete_retries,
                    "generation_continuous_page_budget": bool(
                        args.continuous_page_budget and task_type(row) == "page"
                    ),
                    "generation_selective_incomplete_retry": bool(
                        args.selective_incomplete_retry and task_type(row) == "page"
                    ),
                    "generation_single_pass_loop_stop": first_generation_audit.get(
                        "single_pass_loop_stop"
                    ),
                    "generation_loop_stop_attempts": [
                        audit.get("single_pass_loop_stop") for audit in generation_audits
                    ],
                    "generation_truncated": task_type(row) == "page"
                    and "</HistoricalDocument>" not in raw,
                    "generation_incomplete": bool(incomplete_reasons),
                    "generation_incomplete_reasons": incomplete_reasons,
                    "generation_attempts": len(raw_attempts),
                    "generation_selected_attempt": selected_attempt + 1,
                    "generation_retry_used": len(raw_attempts) > 1,
                    "generation_retry_forced": bool(args.force_page_retries and not initial_incomplete),
                    "generation_retry_succeeded": bool(reason_attempts[0]) and not incomplete_reasons,
                    "generation_attempt_audit": [
                        {
                            "attempt": index + 1,
                            "reasons": reasons,
                            "visible_characters": len(visible_text(candidate)),
                            "format": "xml" if "</HistoricalDocument>" in candidate else "plain_or_unclosed",
                            "grounding_guard_applied": bool(
                                grounding_attempts[index].get("applied")
                            ),
                            "visual_close_gate_interventions": int(
                                (close_gate_attempts[index] or {}).get("interventions") or 0
                            ),
                        }
                        for index, (candidate, reasons) in enumerate(zip(raw_attempts, reason_attempts))
                    ],
                    "metrics": (
                        score(row["text"], prediction)
                        if str(row.get("text") or "").strip() else None
                    ),
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                write_clean_text(args.output, result)
                existing.append(result)
                latest_metrics = result.get("metrics") or {}
                task.update(
                    len(existing),
                    message=row["id"],
                    metrics=(
                        {"latest_cer": latest_metrics["cer"], "latest_wer": latest_metrics["wer"]}
                        if latest_metrics else {"reference_available": False}
                    ),
                )
        summary = summarize(existing)
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        finish_metrics = {"scored_examples": summary["scored_examples"]}
        if summary["cer"] is not None:
            finish_metrics.update({"cer": summary["cer"], "wer": summary["wer"]})
        task.finish("evaluation complete", metrics=finish_metrics)
    except Exception as error:
        task.fail(f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Counterfactual visual verification for suspicious CHURRO page tails.

Attention to an image token is not proof that a generated word is visible.
This module asks a stronger question: does the exact generated suffix receive a
better teacher-forced likelihood from its real page than from an unrelated
page?  It only performs the two extra forward passes after a textual repetition
screen fires, so ordinary pages retain normal one-pass inference speed.
"""

from __future__ import annotations

import html
from statistics import median
from typing import Sequence

import torch
from qwen_vl_utils import process_vision_info

from churro_grounded_tail_guard import (
    LINE_RE,
    substantial_repetition_pairs,
    xml_line_token_spans,
)


@torch.inference_mode()
def assistant_token_nll(processor, model, prompt_messages: list[dict], raw_answer: str):
    """Return assistant token IDs and per-token teacher-forced NLL on one image."""

    full_messages = list(prompt_messages) + [
        {"role": "assistant", "content": [{"type": "text", "text": raw_answer}]}
    ]
    prompt_text = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full_text = processor.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )
    image_inputs, video_inputs = process_vision_info(full_messages)
    full = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    prompt = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    prompt_length = int(prompt.input_ids.shape[1])
    if not torch.equal(full.input_ids[:, :prompt_length].cpu(), prompt.input_ids):
        raise RuntimeError("counterfactual grounding chat-template prefix mismatch")
    outputs = model(**full, use_cache=False)
    assistant_ids = full.input_ids[0, prompt_length:].detach()
    prediction_logits = outputs.logits[0, prompt_length - 1 : -1]
    if prediction_logits.shape[0] != assistant_ids.shape[0]:
        raise RuntimeError("counterfactual grounding target/logit length mismatch")
    nll_chunks = []
    for start in range(0, int(assistant_ids.shape[0]), 64):
        logits = prediction_logits[start : start + 64].float()
        targets = assistant_ids[start : start + 64]
        nll_chunks.append(
            torch.logsumexp(logits, dim=-1)
            - logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        )
    nll = torch.cat(nll_chunks).detach().cpu()
    ids = assistant_ids.detach().cpu()
    del outputs, full, prompt, prediction_logits
    return ids, nll


def _line_medians(values: torch.Tensor, spans: Sequence[tuple[int, int]]) -> list[float | None]:
    medians = []
    for start, end in spans:
        selected = values[max(0, start) : min(int(values.shape[0]), end)]
        medians.append(float(selected.median()) if selected.numel() else None)
    return medians


def accepted_tail_gain_threshold(
    prior_median: float,
    maximum_tail_gain_ratio: float,
    maximum_absolute_tail_gain: float,
) -> float:
    """Require both a relative collapse and near-zero absolute image advantage."""

    return min(
        maximum_absolute_tail_gain,
        prior_median * maximum_tail_gain_ratio,
    )


def counterfactual_grounded_tail_guard(
    processor,
    model,
    real_prompt_messages: list[dict],
    wrong_prompt_messages: list[dict],
    raw_xml: str,
    *,
    minimum_repetition_pairs: int = 2,
    maximum_tail_gain_ratio: float = 0.40,
    minimum_prior_visual_gain: float = 0.02,
    maximum_absolute_tail_gain: float = 0.02,
) -> tuple[str, dict]:
    """Remove a repetitive tail only when its words do not prefer the real page."""

    matches = list(LINE_RE.finditer(raw_xml))
    texts = [html.unescape(match.group(1)).strip() for match in matches]
    pairs = substantial_repetition_pairs(texts)
    audit = {
        "schema_version": "churro_counterfactual_grounding.v1",
        "applied": False,
        "reason": "insufficient_repetition_evidence",
        "line_count": len(texts),
        "minimum_repetition_pairs": minimum_repetition_pairs,
        "repetition_pairs": [
            {"left": left, "right": right, "similarity": similarity}
            for left, right, similarity in pairs
        ],
        "counterfactual_forward_passes": 0,
    }
    if len(pairs) < minimum_repetition_pairs:
        return raw_xml, audit

    real_ids, real_nll = assistant_token_nll(
        processor, model, real_prompt_messages, raw_xml
    )
    wrong_ids, wrong_nll = assistant_token_nll(
        processor, model, wrong_prompt_messages, raw_xml
    )
    audit["counterfactual_forward_passes"] = 2
    if not torch.equal(real_ids, wrong_ids):
        audit["reason"] = "assistant_tokenization_changed_across_images"
        return raw_xml, audit
    spans = xml_line_token_spans(
        raw_xml, real_ids.tolist(), processor.tokenizer, content_only=True
    )
    token_gain = wrong_nll - real_nll
    line_gains = _line_medians(token_gain, spans)
    candidate = min(left for left, _, _ in pairs)
    audit.update({"candidate_cutoff": candidate, "line_visual_nll_gains": line_gains})
    if candidate < 4:
        audit["reason"] = "repetition_begins_before_visual_baseline"
        return raw_xml, audit
    prior = [value for value in line_gains[max(0, candidate - 10) : candidate] if value is not None]
    tail = [value for value in line_gains[candidate:] if value is not None]
    if len(prior) < 3 or len(tail) < 3:
        audit["reason"] = "insufficient_counterfactual_line_scores"
        return raw_xml, audit
    prior_median = float(median(prior))
    tail_median = float(median(tail))
    # Both gates are mandatory.  A tail can be less image-dependent than an
    # unusually visual preceding section while still receiving strong absolute
    # support from the real page.  Using ``min`` prevents that visible text
    # from being removed merely because the relative ratio fell.
    threshold = accepted_tail_gain_threshold(
        prior_median,
        maximum_tail_gain_ratio,
        maximum_absolute_tail_gain,
    )
    audit.update(
        {
            "prior_visual_nll_gain_median": prior_median,
            "tail_visual_nll_gain_median": tail_median,
            "maximum_accepted_tail_gain": threshold,
            "maximum_tail_gain_ratio": maximum_tail_gain_ratio,
        }
    )
    if prior_median < minimum_prior_visual_gain:
        audit["reason"] = "real_page_has_no_reliable_prior_visual_advantage"
        return raw_xml, audit
    if tail_median > threshold:
        audit["reason"] = "repetitive_tail_still_prefers_real_page"
        return raw_xml, audit

    cutoff = candidate
    for earlier in range(candidate - 1, max(1, candidate - 5), -1):
        value = line_gains[earlier]
        if value is None or value > threshold:
            break
        cutoff = earlier
    if cutoff < max(3, int(0.45 * len(texts))) or len(texts) - cutoff < 5:
        audit["reason"] = "candidate_tail_is_not_a_safe_page_suffix"
        return raw_xml, audit

    removed = raw_xml[matches[cutoff].start() : matches[-1].end()]
    guarded = raw_xml[: matches[cutoff].start()] + raw_xml[matches[-1].end() :]
    audit.update(
        {
            "applied": True,
            "reason": "counterfactually_unsupported_repetitive_tail",
            "cutoff": cutoff,
            "removed_line_count": len(matches) - cutoff,
            "removed_characters": len(removed),
            "removed_text": "\n".join(texts[cutoff:]),
        }
    )
    return guarded, audit

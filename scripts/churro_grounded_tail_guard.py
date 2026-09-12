#!/usr/bin/env python3
"""Conservative visual-grounding guard for repetitive CHURRO page tails.

The guard is deliberately post-generation and fail-closed.  It never changes
individual words.  It may remove a suffix of complete XML ``Line`` elements,
but only when two independent signals agree:

* multiple substantial lines in the suffix repeat earlier suffix lines; and
* normalized image-vs-generated-text attention has collapsed relative to the
  preceding grounded lines.

This preserves ordinary repeated words while catching the common failure mode
where a page ends mid-sentence and the autoregressive decoder continues its own
invented prose.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from statistics import median
from typing import Sequence

import torch


LINE_RE = re.compile(r"<Line>(.*?)</Line>", re.IGNORECASE | re.DOTALL)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")


@dataclass(frozen=True)
class LineGrounding:
    index: int
    text: str
    token_start: int
    token_end: int
    visual_grounding: float | None


def _words(text: str) -> list[str]:
    return [word.casefold() for word in WORD_RE.findall(html.unescape(text))]


def _substantial_similarity(left: str, right: str) -> float:
    """Return repetition similarity while ignoring tiny generic fragments."""

    a = _words(left)
    b = _words(right)
    if min(len(a), len(b)) < 4:
        return 0.0
    shared = len(set(a) & set(b))
    if shared < 3:
        return 0.0
    sequence = SequenceMatcher(None, a, b, autojunk=False).ratio()
    containment = shared / max(1, min(len(set(a)), len(set(b))))
    a_ngrams = {tuple(a[index : index + 4]) for index in range(len(a) - 3)}
    b_ngrams = {tuple(b[index : index + 4]) for index in range(len(b) - 3)}
    four_gram = 1.0 if a_ngrams & b_ngrams else 0.0
    return max(sequence, 0.82 * containment, four_gram)


def _decode_prefix_length(tokenizer, token_ids: Sequence[int], count: int) -> int:
    return len(
        tokenizer.decode(
            list(token_ids[:count]),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    )


def _token_index_for_character(
    tokenizer,
    token_ids: Sequence[int],
    character_offset: int,
    cache: dict[int, int],
) -> int:
    """Find the first decoded-token boundary at or beyond a character offset."""

    low, high = 0, len(token_ids)
    while low < high:
        middle = (low + high) // 2
        length = cache.get(middle)
        if length is None:
            length = _decode_prefix_length(tokenizer, token_ids, middle)
            cache[middle] = length
        if length < character_offset:
            low = middle + 1
        else:
            high = middle
    return low


def xml_line_token_spans(
    raw_xml: str,
    generated_token_ids: Sequence[int],
    tokenizer,
    *,
    content_only: bool = False,
) -> list[tuple[int, int]]:
    """Map decoded XML line or line-content spans onto generated token IDs."""

    prefix_cache = {0: 0}
    spans = []
    for match in LINE_RE.finditer(raw_xml):
        character_start = match.start(1) if content_only else match.start()
        character_end = match.end(1) if content_only else match.end()
        spans.append(
            (
                _token_index_for_character(
                    tokenizer, generated_token_ids, character_start, prefix_cache
                ),
                _token_index_for_character(
                    tokenizer, generated_token_ids, character_end, prefix_cache
                ),
            )
        )
    return spans


def substantial_repetition_pairs(
    texts: Sequence[str],
    *,
    minimum_similarity: float = 0.68,
    maximum_pair_distance: int = 10,
) -> list[tuple[int, int, float]]:
    pairs = []
    for right in range(len(texts)):
        best = None
        for left in range(max(0, right - maximum_pair_distance), right):
            similarity = _substantial_similarity(texts[left], texts[right])
            if similarity >= minimum_similarity and (best is None or similarity > best[2]):
                best = (left, right, similarity)
        if best is not None:
            pairs.append(best)
    return pairs


def line_grounding_from_trace(
    raw_xml: str,
    generated_token_ids: Sequence[int],
    visual_trace: torch.Tensor | Sequence[float],
    tokenizer,
) -> list[LineGrounding]:
    """Associate per-decode-step grounding values with generated XML lines.

    Qwen's prefill predicts the first generated token without a cached
    one-token decode step.  Therefore trace element zero supports generated
    token one; token zero intentionally has no score.
    """

    if isinstance(visual_trace, torch.Tensor):
        trace = visual_trace.detach().float().cpu().flatten().tolist()
    else:
        trace = [float(value) for value in visual_trace]
    token_support: list[float | None] = [None] * len(generated_token_ids)
    for trace_index, value in enumerate(trace):
        token_index = trace_index + 1
        if token_index >= len(token_support):
            break
        token_support[token_index] = value

    spans = xml_line_token_spans(raw_xml, generated_token_ids, tokenizer)
    lines = []
    for index, (match, span) in enumerate(zip(LINE_RE.finditer(raw_xml), spans)):
        token_start, token_end = span
        values = [
            value
            for value in token_support[token_start:token_end]
            if value is not None
        ]
        lines.append(
            LineGrounding(
                index=index,
                text=html.unescape(match.group(1)).strip(),
                token_start=token_start,
                token_end=token_end,
                visual_grounding=float(median(values)) if values else None,
            )
        )
    return lines


def detect_unsupported_repetitive_tail(
    lines: Sequence[LineGrounding],
    *,
    minimum_repetition_pairs: int = 2,
    minimum_similarity: float = 0.68,
    maximum_pair_distance: int = 10,
    maximum_tail_support_ratio: float = 0.62,
) -> tuple[int | None, dict]:
    """Return a conservative line cutoff and an explainable audit record."""

    pairs = substantial_repetition_pairs(
        [line.text for line in lines],
        minimum_similarity=minimum_similarity,
        maximum_pair_distance=maximum_pair_distance,
    )

    audit = {
        "schema_version": "churro_grounded_tail_guard.v1",
        "applied": False,
        "reason": "insufficient_repetition_evidence",
        "line_count": len(lines),
        "repetition_pairs": [
            {"left": left, "right": right, "similarity": similarity}
            for left, right, similarity in pairs
        ],
        "minimum_repetition_pairs": minimum_repetition_pairs,
        "maximum_tail_support_ratio": maximum_tail_support_ratio,
    }
    if len(pairs) < minimum_repetition_pairs:
        return None, audit

    first_repeated_origin = min(left for left, _, _ in pairs)
    # The first repeated origin can itself be invented.  Backtrack over at
    # most four immediately preceding low-grounding lines after establishing
    # a baseline from the earlier page.
    candidate = first_repeated_origin
    if candidate < 4:
        audit["reason"] = "repetition_begins_before_grounding_baseline"
        return None, audit
    prior_values = [
        line.visual_grounding
        for line in lines[max(0, candidate - 10) : candidate]
        if line.visual_grounding is not None
    ]
    tail_values = [
        line.visual_grounding
        for line in lines[candidate:]
        if line.visual_grounding is not None
    ]
    if len(prior_values) < 3 or len(tail_values) < 3:
        audit["reason"] = "insufficient_visual_grounding_samples"
        return None, audit
    prior_median = float(median(prior_values))
    tail_median = float(median(tail_values))
    support_ratio = tail_median / max(prior_median, 1e-9)
    audit.update(
        {
            "candidate_cutoff": candidate,
            "prior_grounding_median": prior_median,
            "tail_grounding_median": tail_median,
            "tail_support_ratio": support_ratio,
        }
    )
    if support_ratio > maximum_tail_support_ratio:
        audit["reason"] = "tail_remains_visually_grounded"
        return None, audit

    low_threshold = prior_median * maximum_tail_support_ratio
    cutoff = candidate
    for earlier in range(candidate - 1, max(1, candidate - 5), -1):
        value = lines[earlier].visual_grounding
        if value is None or value > low_threshold:
            break
        cutoff = earlier

    # Never remove most of a page and require a meaningful repetitive suffix.
    if cutoff < max(3, int(0.45 * len(lines))) or len(lines) - cutoff < 5:
        audit["reason"] = "candidate_tail_is_not_a_safe_page_suffix"
        return None, audit
    audit.update({"applied": True, "reason": "unsupported_repetitive_tail", "cutoff": cutoff})
    return cutoff, audit


def guard_xml_repetitive_tail(
    raw_xml: str,
    generated_token_ids: Sequence[int],
    visual_trace: torch.Tensor | Sequence[float],
    tokenizer,
) -> tuple[str, dict]:
    """Remove an unsupported repetitive XML-line suffix, or return unchanged."""

    matches = list(LINE_RE.finditer(raw_xml))
    lines = line_grounding_from_trace(raw_xml, generated_token_ids, visual_trace, tokenizer)
    cutoff, audit = detect_unsupported_repetitive_tail(lines)
    audit["lines"] = [
        {
            "index": line.index,
            "text": line.text,
            "visual_grounding": line.visual_grounding,
        }
        for line in lines
    ]
    if cutoff is None or cutoff >= len(matches):
        return raw_xml, audit
    start = matches[cutoff].start()
    end = matches[-1].end()
    removed = raw_xml[start:end]
    guarded = raw_xml[:start] + raw_xml[end:]
    audit.update(
        {
            "removed_line_count": len(matches) - cutoff,
            "removed_characters": len(removed),
            "removed_text": "\n".join(line.text for line in lines[cutoff:]),
        }
    )
    return guarded, audit

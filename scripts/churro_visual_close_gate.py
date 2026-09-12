#!/usr/bin/env python3
"""Visual-evidence gate for CHURRO's semantic document-close decision.

CHURRO does not normally stop a page by choosing EOS directly.  The runtime
gate waits until a paragraph has closed, then distinguishes opening another
``<Paragraph>`` from closing ``</Body>``.  This avoids suppressing legitimate
paragraph breaks.  The branch is changed only when independently cropped
visual line reads provide strong evidence below an aligned anchor.

The gate never inserts OCR text and never edits an already generated line.  It
only gives the full-page recognizer one more opportunity to look at the image.
"""

from __future__ import annotations

import html
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import torch
from transformers import LogitsProcessor


LINE_TAG = re.compile(r"<Line(?:\s+[^>]*)?>(.*?)</Line>", re.IGNORECASE | re.DOTALL)
TAG = re.compile(r"<[^>]+>")
SPACE = re.compile(r"\s+")
WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)


def _encode(tokenizer, text: str) -> tuple[int, ...]:
    return tuple(int(value) for value in tokenizer.encode(text, add_special_tokens=False))


@dataclass(frozen=True)
class ParagraphBranchTokens:
    """Tokenizer-specific representation of CHURRO's close/continue fork."""

    boundary_suffix: tuple[int, ...]
    continue_token_id: int
    close_token_id: int
    continue_route: tuple[int, ...]
    close_route: tuple[int, ...]


def paragraph_branch_tokens(tokenizer) -> ParagraphBranchTokens:
    """Derive a context-stable nonfinal-line XML branch.

    Qwen's tokenizer can merge the ``</`` opening with punctuation immediately
    before it (for example ``.</``).  ``Line>\n`` is the stable tail shared by
    those contextual variants, so matching the standalone ``</Line>`` encoding
    would silently miss most training boundaries.
    """

    continue_route = _encode(tokenizer, "Line>\n        <Line>")
    close_route = _encode(tokenizer, "Line>\n      </Paragraph>")
    shared = 0
    for left, right in zip(continue_route, close_route):
        if left != right:
            break
        shared += 1
    if shared < 1 or shared >= min(len(continue_route), len(close_route)):
        raise ValueError(
            "Tokenizer does not expose a stable CHURRO Line> continue/close branch"
        )
    branch = ParagraphBranchTokens(
        boundary_suffix=continue_route[:shared],
        continue_token_id=continue_route[shared],
        close_token_id=close_route[shared],
        continue_route=continue_route[shared:],
        close_route=close_route[shared:],
    )
    if branch.continue_token_id == branch.close_token_id:
        raise ValueError("CHURRO continuation and paragraph-close tokens are identical")
    return branch


def body_close_branch_tokens(tokenizer) -> ParagraphBranchTokens:
    """Derive the safe post-paragraph fork: another paragraph vs body close."""

    continue_route = _encode(tokenizer, "</Paragraph>\n    <Paragraph>")
    close_route = _encode(tokenizer, "</Paragraph>\n    </Body>")
    shared = 0
    for left, right in zip(continue_route, close_route):
        if left != right:
            break
        shared += 1
    if shared < 1 or shared >= min(len(continue_route), len(close_route)):
        raise ValueError(
            "Tokenizer does not expose a unique CHURRO paragraph-continue/body-close branch"
        )
    return ParagraphBranchTokens(
        boundary_suffix=continue_route[:shared],
        continue_token_id=continue_route[shared],
        close_token_id=close_route[shared],
        continue_route=continue_route[shared:],
        close_route=close_route[shared:],
    )


def clean_text(value: object) -> str:
    return SPACE.sub(" ", html.unescape(TAG.sub(" ", str(value or "")))).strip()


def normalized_words(value: object) -> list[str]:
    return [
        "".join(character.casefold() for character in match.group(0) if character.isalnum())
        for match in WORD.finditer(clean_text(value))
    ]


def line_similarity(left: str, right: str) -> float:
    a, b = normalized_words(left), normalized_words(right)
    if not a or not b:
        return 0.0
    words = SequenceMatcher(None, a, b, autojunk=False).ratio()
    characters = SequenceMatcher(None, " ".join(a), " ".join(b), autojunk=False).ratio()
    return 0.72 * words + 0.28 * characters


def has_repetitive_loop(words: list[str]) -> bool:
    if len(words) < 8:
        return False
    for width in (2, 3, 4):
        grams = [tuple(words[index : index + width]) for index in range(len(words) - width + 1)]
        if grams and max(grams.count(gram) for gram in set(grams)) >= 3:
            return True
    return False


def generated_line_texts(raw_xml: str) -> list[str]:
    return [clean_text(value) for value in LINE_TAG.findall(raw_xml) if clean_text(value)]


def align_lines(page_lines: list[str], visual_lines: list[str]) -> list[dict]:
    """Monotonic global alignment with honest gaps for unrelated lines."""

    n, m = len(page_lines), len(visual_lines)
    infinity = float("inf")
    cost = [[infinity] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int, str, float] | None]] = [
        [None] * (m + 1) for _ in range(n + 1)
    ]
    cost[0][0] = 0.0
    for i in range(1, n + 1):
        cost[i][0] = cost[i - 1][0] + 0.52
        back[i][0] = (i - 1, 0, "page_only", 0.0)
    for j in range(1, m + 1):
        cost[0][j] = cost[0][j - 1] + 0.48
        back[0][j] = (0, j - 1, "visual_only", 0.0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            similarity = line_similarity(page_lines[i - 1], visual_lines[j - 1])
            diagonal = 1.0 - similarity if similarity >= 0.28 else 1.05
            choices = (
                (cost[i - 1][j - 1] + diagonal, i - 1, j - 1, "match", similarity),
                (cost[i - 1][j] + 0.52, i - 1, j, "page_only", 0.0),
                (cost[i][j - 1] + 0.48, i, j - 1, "visual_only", 0.0),
            )
            best = min(choices, key=lambda item: item[0])
            cost[i][j] = best[0]
            back[i][j] = (best[1], best[2], best[3], best[4])
    operations = []
    i, j = n, m
    while i or j:
        step = back[i][j]
        if step is None:
            break
        previous_i, previous_j, operation, similarity = step
        operations.append(
            {
                "operation": operation,
                "page_index": i - 1 if operation in {"match", "page_only"} else None,
                "visual_index": j - 1 if operation in {"match", "visual_only"} else None,
                "similarity": similarity,
            }
        )
        i, j = previous_i, previous_j
    operations.reverse()
    return operations


def _page_id(row: dict) -> str:
    return str(row.get("parent_page_id") or row.get("page_id") or row.get("id") or "")


def _line_key(row: dict) -> tuple[str, str]:
    page = _page_id(row)
    index = row.get("line_index")
    if index is not None:
        return page, f"index:{int(index):08d}"
    return page, f"id:{row.get('line_id') or row.get('id') or row.get('image') or ''}"


def _bbox(row: dict) -> tuple[float, float, float, float] | None:
    value = row.get("bbox")
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        candidate = value[:4]
    else:
        candidate = [row.get(name) for name in ("x0", "y0", "x1", "y1")]
    try:
        x0, y0, x1, y1 = (float(item) for item in candidate)
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_visual_line_evidence(
    primary_manifest: Path,
    verifier_manifest: Path | None = None,
) -> tuple[dict[str, list[dict]], dict]:
    """Load spatially ordered line reads and optional second-view consensus."""

    primary = read_jsonl(primary_manifest)
    verifier_rows = read_jsonl(verifier_manifest) if verifier_manifest is not None else []
    verifier_by_key: dict[tuple[str, str], dict] = {}
    for row in verifier_rows:
        key = _line_key(row)
        if key in verifier_by_key:
            raise ValueError(f"duplicate verifier visual-line key: {key}")
        verifier_by_key[key] = row
    grouped: dict[str, list[dict]] = defaultdict(list)
    matched_verifiers = 0
    primary_keys: set[tuple[str, str]] = set()
    for position, row in enumerate(primary):
        page = _page_id(row)
        if not page:
            continue
        key = _line_key(row)
        if key in primary_keys:
            raise ValueError(f"duplicate primary visual-line key: {key}")
        primary_keys.add(key)
        raw_prediction = str(row.get("prediction") or row.get("raw_prediction") or "").strip()
        text = clean_text(raw_prediction)
        verifier = verifier_by_key.get(key)
        verifier_text = clean_text(
            str(verifier.get("prediction") or verifier.get("raw_prediction") or "")
        ) if verifier is not None else ""
        consensus = line_similarity(text, verifier_text) if verifier is not None else None
        matched_verifiers += int(verifier is not None)
        grouped[page].append(
            {
                **row,
                "visual_text": text,
                "visual_raw_prediction": raw_prediction,
                "visual_bbox": _bbox(row),
                "verifier_text": verifier_text,
                "view_consensus_similarity": consensus,
                "_source_position": position,
            }
        )
    for rows in grouped.values():
        rows.sort(
            key=lambda row: (
                int(row.get("line_index") or 0),
                (row.get("visual_bbox") or (0, 0, 0, 0))[1],
                (row.get("visual_bbox") or (0, 0, 0, 0))[0],
                row["_source_position"],
            )
        )
    return dict(grouped), {
        "primary_manifest": str(primary_manifest.resolve()),
        "verifier_manifest": str(verifier_manifest.resolve()) if verifier_manifest else None,
        "primary_rows": len(primary),
        "verifier_rows": len(verifier_rows),
        "pages": len(grouped),
        "matched_verifier_rows": matched_verifiers,
        "unmatched_primary_rows": len(primary_keys) - matched_verifiers,
        "unmatched_verifier_rows": len(set(verifier_by_key) - primary_keys),
    }


class VisualCoverageBodyCloseGate(LogitsProcessor):
    """Suppress a premature ``</Body>`` branch when visual lines remain.

    This is deliberately conservative.  A close is changed only when:

    * generation is exactly at the legal post-``</Paragraph>`` XML fork;
    * an emitted line strongly anchors to an ordered visual line read;
    * at least ``minimum_remaining_lines`` good spatially lower reads remain;
    * optional second-view reads agree; and
    * the model's close logit is not overwhelmingly above continuation.
    """

    def __init__(
        self,
        tokenizer,
        prompt_length: int,
        visual_lines: Iterable[dict],
        *,
        penalty: float = 2.5,
        maximum_penalty: float = 6.0,
        maximum_close_margin: float = 4.0,
        required_continue_margin: float = 0.25,
        minimum_anchor_similarity: float = 0.70,
        minimum_remaining_lines: int = 2,
        minimum_words: int = 3,
        maximum_words: int = 18,
        minimum_view_consensus: float = 0.82,
        require_verifier: bool = False,
        maximum_interventions: int = 2,
        audit_only: bool = False,
    ) -> None:
        if prompt_length < 0:
            raise ValueError("prompt_length must be non-negative")
        numeric_parameters = {
            "penalty": penalty,
            "maximum_penalty": maximum_penalty,
            "maximum_close_margin": maximum_close_margin,
            "required_continue_margin": required_continue_margin,
            "minimum_anchor_similarity": minimum_anchor_similarity,
            "minimum_view_consensus": minimum_view_consensus,
        }
        if any(not math.isfinite(float(value)) for value in numeric_parameters.values()):
            raise ValueError("visual-close numeric parameters must be finite")
        if penalty < 0 or maximum_penalty < penalty:
            raise ValueError("invalid visual-close penalty range")
        if maximum_close_margin < 0 or required_continue_margin < 0:
            raise ValueError("visual-close logit margins must be non-negative")
        if maximum_penalty < maximum_close_margin + required_continue_margin:
            raise ValueError(
                "maximum_penalty must cover maximum_close_margin + required_continue_margin"
            )
        if not 0.0 <= minimum_anchor_similarity <= 1.0:
            raise ValueError("minimum_anchor_similarity must be in [0, 1]")
        if not 0.0 <= minimum_view_consensus <= 1.0:
            raise ValueError("minimum_view_consensus must be in [0, 1]")
        if minimum_remaining_lines < 1 or maximum_interventions < 1:
            raise ValueError("visual-close count thresholds must be positive")
        self.tokenizer = tokenizer
        self.prompt_length = int(prompt_length)
        self.visual_lines = [dict(row) for row in visual_lines]
        self.branch = body_close_branch_tokens(tokenizer)
        self.penalty = float(penalty)
        self.maximum_penalty = float(maximum_penalty)
        self.maximum_close_margin = float(maximum_close_margin)
        self.required_continue_margin = float(required_continue_margin)
        self.minimum_anchor_similarity = float(minimum_anchor_similarity)
        self.minimum_remaining_lines = int(minimum_remaining_lines)
        self.minimum_words = int(minimum_words)
        self.maximum_words = int(maximum_words)
        self.minimum_view_consensus = float(minimum_view_consensus)
        self.require_verifier = bool(require_verifier)
        self.maximum_interventions = int(maximum_interventions)
        self.audit_only = bool(audit_only)
        self.boundaries_seen = 0
        self.interventions = 0
        self._intervened_line_counts: set[int] = set()
        self.events: list[dict] = []

    def _eligible(self, row: dict) -> bool:
        text = str(row.get("visual_text") or "")
        raw = str(row.get("visual_raw_prediction") or text)
        words = normalized_words(text)
        if not text or "\n" in raw or "\r" in raw:
            return False
        if row.get("generation_truncated") or row.get("generation_incomplete"):
            return False
        if not self.minimum_words <= len(words) <= self.maximum_words:
            return False
        if has_repetitive_loop(words) or row.get("visual_bbox") is None:
            return False
        consensus = row.get("view_consensus_similarity")
        if self.require_verifier and consensus is None:
            return False
        if consensus is not None and float(consensus) < self.minimum_view_consensus:
            return False
        return True

    def _remaining_evidence(self, raw_xml: str) -> dict:
        page_lines = generated_line_texts(raw_xml)
        eligible_rows = [row for row in self.visual_lines if self._eligible(row)]
        visual_texts = [str(row["visual_text"]) for row in eligible_rows]
        result = {
            "generated_lines": len(page_lines),
            "visual_lines": len(self.visual_lines),
            "eligible_visual_lines": len(eligible_rows),
            "anchor_similarity": 0.0,
            "anchor_page_index": None,
            "anchor_visual_index": None,
            "remaining_visual_lines": 0,
            "remaining_text": [],
            "evidence_reason": None,
        }
        if not page_lines or not eligible_rows:
            result["evidence_reason"] = "missing_generated_or_eligible_visual_lines"
            return result
        operations = align_lines(page_lines, visual_texts)
        anchors = [
            operation
            for operation in operations
            if operation["operation"] == "match"
            and operation["similarity"] >= self.minimum_anchor_similarity
        ]
        if not anchors:
            result["evidence_reason"] = "no_strong_monotonic_anchor"
            return result
        anchor = max(anchors, key=lambda item: (item["page_index"], item["visual_index"]))
        result.update(
            {
                "anchor_similarity": float(anchor["similarity"]),
                "anchor_page_index": int(anchor["page_index"]),
                "anchor_visual_index": int(anchor["visual_index"]),
            }
        )
        # A remote early anchor is not enough to overrule a close after a long,
        # possibly hallucinated suffix.  The anchor must touch the generated tail.
        if int(anchor["page_index"]) < len(page_lines) - 2:
            result["evidence_reason"] = "strong_anchor_not_near_generated_tail"
            return result
        anchor_row = eligible_rows[int(anchor["visual_index"])]
        anchor_bbox = anchor_row["visual_bbox"]
        anchor_center = (float(anchor_bbox[1]) + float(anchor_bbox[3])) / 2.0
        anchor_height = max(1.0, float(anchor_bbox[3]) - float(anchor_bbox[1]))
        remaining = []
        for visual_index, row in enumerate(eligible_rows):
            if visual_index <= int(anchor["visual_index"]):
                continue
            # Do not call a badly aligned duplicate "remaining" if it already
            # resembles any emitted line.
            if max((line_similarity(str(row["visual_text"]), line) for line in page_lines), default=0.0) >= 0.58:
                continue
            bbox = row["visual_bbox"]
            center = (float(bbox[1]) + float(bbox[3])) / 2.0
            if center <= anchor_center + 0.35 * anchor_height:
                continue
            remaining.append(row)
        result["remaining_visual_lines"] = len(remaining)
        result["remaining_text"] = [str(row["visual_text"]) for row in remaining[:4]]
        result["evidence_reason"] = (
            "strong_lower_visual_suffix"
            if len(remaining) >= self.minimum_remaining_lines
            else "insufficient_distinct_lower_visual_lines"
        )
        return result

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if int(input_ids.shape[0]) != 1 or int(scores.shape[0]) != 1:
            raise RuntimeError(
                "visual close gating currently supports one greedy sequence at a time"
            )
        suffix = self.branch.boundary_suffix
        for batch_index in range(int(input_ids.shape[0])):
            sequence = input_ids[batch_index]
            if sequence.numel() < len(suffix):
                continue
            actual = tuple(int(value) for value in sequence[-len(suffix) :].detach().cpu().tolist())
            if actual != suffix:
                continue
            self.boundaries_seen += 1
            generated = sequence[self.prompt_length :]
            raw_xml = self.tokenizer.decode(
                generated.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            evidence = self._remaining_evidence(raw_xml)
            line_count = int(evidence["generated_lines"])
            event = {
                "boundary": self.boundaries_seen,
                "response_next_token_offset": int(sequence.numel()) - self.prompt_length,
                **evidence,
                "applied": False,
                "audit_only": self.audit_only,
            }
            if evidence["evidence_reason"] != "strong_lower_visual_suffix":
                self.events.append(event)
                continue
            if line_count in self._intervened_line_counts:
                event["decision_reason"] = "already_intervened_at_generated_line_count"
                self.events.append(event)
                continue
            if self.interventions >= self.maximum_interventions:
                event["decision_reason"] = "maximum_interventions_reached"
                self.events.append(event)
                continue
            continue_logit = float(scores[batch_index, self.branch.continue_token_id].item())
            close_logit = float(scores[batch_index, self.branch.close_token_id].item())
            close_margin = close_logit - continue_logit
            event.update(
                {
                    "continue_logit": continue_logit,
                    "close_logit": close_logit,
                    "close_minus_continue_margin": close_margin,
                }
            )
            if close_margin < -self.required_continue_margin:
                event["decision_reason"] = "continuation_already_preferred"
                self.events.append(event)
                continue
            if close_margin > self.maximum_close_margin:
                event["decision_reason"] = "close_margin_exceeds_safe_bound"
                self.events.append(event)
                continue
            applied_penalty = min(
                self.maximum_penalty,
                max(self.penalty, close_margin + self.required_continue_margin),
            )
            event["applied_penalty"] = applied_penalty
            event["decision_reason"] = "audit_only" if self.audit_only else "visual_suffix_gate"
            if not self.audit_only:
                scores[batch_index, self.branch.close_token_id] -= applied_penalty
                self.interventions += 1
                self._intervened_line_counts.add(line_count)
                event["applied"] = True
                event["post_close_minus_continue_margin"] = close_margin - applied_penalty
            self.events.append(event)
        return scores

    def finalize(self, generated_ids: torch.Tensor) -> None:
        """Record which branch token generation selected after each intervention."""

        # Generation is single-example/greedy for the coverage-gated profile.
        sequence = generated_ids[0] if generated_ids.ndim == 2 else generated_ids
        response = sequence[self.prompt_length :].detach().cpu().tolist()
        for event in self.events:
            if not event.get("applied"):
                continue
            offset = int(event["response_next_token_offset"])
            chosen = int(response[offset]) if offset < len(response) else None
            event["selected_next_token_id"] = chosen
            event["selected_route"] = (
                "continue"
                if chosen == self.branch.continue_token_id
                else ("close" if chosen == self.branch.close_token_id else "other_or_missing")
            )
            event["intervention_succeeded"] = chosen == self.branch.continue_token_id
            event["final_generated_lines"] = len(
                generated_line_texts(
                    self.tokenizer.decode(
                        response,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                )
            )

    def audit(self) -> dict:
        return {
            "schema_version": "churro_visual_body_close_gate.v1",
            "policy": "semantic_body_close_only_with_monotonic_visual_suffix",
            "audit_only": self.audit_only,
            "branch": {
                "boundary_suffix": list(self.branch.boundary_suffix),
                "continue_token_id": self.branch.continue_token_id,
                "close_token_id": self.branch.close_token_id,
            },
            "visual_lines": len(self.visual_lines),
            "boundaries_seen": self.boundaries_seen,
            "interventions": self.interventions,
            "events": self.events,
        }


# Backward-compatible import name for the first development revision.  The
# implementation now gates only the safer post-paragraph </Body> decision.
VisualCoverageParagraphCloseGate = VisualCoverageBodyCloseGate

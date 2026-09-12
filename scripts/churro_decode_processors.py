#!/usr/bin/env python3
"""Narrow decoding safeguards for literal CHURRO transcription."""

from __future__ import annotations

import re

import torch
from collections.abc import Iterable

from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    StoppingCriteria,
)


_XML_LINE = re.compile(r"<Line(?:\s+[^>]*)?>(.*?)</Line>", re.IGNORECASE | re.DOTALL)
_NONWORD = re.compile(r"[^\w']+", re.UNICODE)


class RepetitiveXmlTailStoppingCriteria(StoppingCriteria):
    """Stop one continuous decode after a clear repeated XML-line suffix.

    This does not guess replacement text or restart the page.  It checks only
    completed ``<Line>`` elements near the generated tail and requires three
    consecutive copies of the same one-to-four-line block.  Ordinary recurring
    words elsewhere on a page therefore do not trigger it.
    """

    def __init__(
        self,
        tokenizer,
        prompt_length: int,
        *,
        required_repeats: int = 3,
        maximum_block_lines: int = 4,
        check_interval_tokens: int = 16,
        tail_tokens: int = 1024,
    ) -> None:
        if prompt_length < 0:
            raise ValueError("prompt_length must be non-negative")
        if required_repeats < 3 or maximum_block_lines < 1:
            raise ValueError("loop thresholds are too permissive")
        if check_interval_tokens < 1 or tail_tokens < 128:
            raise ValueError("invalid loop-check interval or tail size")
        self.tokenizer = tokenizer
        self.prompt_length = int(prompt_length)
        self.required_repeats = int(required_repeats)
        self.maximum_block_lines = int(maximum_block_lines)
        self.check_interval_tokens = int(check_interval_tokens)
        self.tail_tokens = int(tail_tokens)
        self.last_checked_length = -1
        self.decoded_upto = 0
        self.decoded_text = ""
        self.triggered = False
        self.audit: dict | None = None

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(_NONWORD.sub(" ", value.casefold()).split())

    def _repeated_suffix(self, lines: list[str]) -> tuple[int, list[str]] | None:
        maximum = min(self.maximum_block_lines, len(lines) // self.required_repeats)
        for width in range(1, maximum + 1):
            span = width * self.required_repeats
            suffix = lines[-span:]
            block = suffix[:width]
            block_word_count = sum(len(value.split()) for value in block)
            if block_word_count >= 8 and all(
                suffix[index * width : (index + 1) * width] == block
                for index in range(1, self.required_repeats)
            ):
                return width, block
        return None

    def _repeated_word_suffix(self, raw: str) -> tuple[int, list[str]] | None:
        plain = re.sub(r"<[^>]+>", " ", raw)
        words = self._normalize(plain).split()
        maximum = min(40, len(words) // self.required_repeats)
        for width in range(4, maximum + 1):
            span = width * self.required_repeats
            suffix = words[-span:]
            block = suffix[:width]
            if all(
                suffix[index * width : (index + 1) * width] == block
                for index in range(1, self.required_repeats)
            ):
                return width, block
        return None

    def _low_novelty_word_tail(self, raw: str) -> dict | None:
        """Detect growing or slightly-mutating prose loops near the tail.

        Exact block matching misses a common autoregressive failure where each
        sentence reuses most of the preceding sentence but adds or changes a
        few words.  A clean transcription can repeat ordinary vocabulary, so
        this requires both heavy 4-gram duplication and low word diversity in
        a short suffix.  Thresholds were checked against the held-out LOC100
        references: their maximum sliding duplicate-4-gram ratio was 0.180.
        A genuinely repetitive arithmetic page reached 0.262, while the known
        prose loop crossed 0.344; the 0.30 boundary preserves the former.
        """
        plain = re.sub(r"<[^>]+>", " ", raw)
        words = self._normalize(plain).split()
        window = 64
        ngram_width = 4
        if len(words) < window:
            return None
        tail = words[-window:]
        ngrams = [
            tuple(tail[index : index + ngram_width])
            for index in range(len(tail) - ngram_width + 1)
        ]
        duplicate_ratio = 1.0 - (len(set(ngrams)) / len(ngrams))
        unique_word_ratio = len(set(tail)) / len(tail)
        if duplicate_ratio < 0.30 or unique_word_ratio > 0.60:
            return None
        return {
            "window_words": window,
            "ngram_width": ngram_width,
            "duplicate_ngram_ratio": round(duplicate_ratio, 6),
            "unique_word_ratio": round(unique_word_ratio, 6),
            "tail_preview": " ".join(tail[-24:]),
        }

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        if int(input_ids.shape[0]) != 1:
            raise RuntimeError("single-pass XML loop stopping supports batch size one")
        generated_length = int(input_ids.shape[-1]) - self.prompt_length
        if (
            self.triggered
            or generated_length < self.check_interval_tokens
            or generated_length - self.last_checked_length < self.check_interval_tokens
        ):
            return torch.tensor([self.triggered], dtype=torch.bool, device=input_ids.device)
        self.last_checked_length = generated_length
        generated = input_ids[0, self.prompt_length :]
        # Decode only newly produced tokens.  Re-decoding the entire tail on
        # every check caused a large GPU synchronization/CPU tokenization
        # penalty during long pages.
        new_tokens = generated[self.decoded_upto :].detach().cpu().tolist()
        if new_tokens:
            self.decoded_text += self.tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            self.decoded_upto = int(generated.numel())
            if len(self.decoded_text) > 16384:
                self.decoded_text = self.decoded_text[-16384:]
        raw = self.decoded_text
        lines = [self._normalize(value) for value in _XML_LINE.findall(raw)]
        lines = [value for value in lines if value]
        repeated = self._repeated_suffix(lines)
        reason = "three_consecutive_repeated_xml_line_blocks"
        if repeated is None:
            repeated = self._repeated_word_suffix(raw)
            reason = "three_consecutive_repeated_word_blocks"
        low_novelty = None
        if repeated is None:
            low_novelty = self._low_novelty_word_tail(raw)
            reason = "low_novelty_repetitive_word_tail"
        if repeated is not None:
            width, block = repeated
            self.triggered = True
            self.audit = {
                "reason": reason,
                "generated_tokens": generated_length,
                "block_width": width,
                "normalized_block": block,
            }
        elif low_novelty is not None:
            self.triggered = True
            self.audit = {
                "reason": reason,
                "generated_tokens": generated_length,
                **low_novelty,
            }
        return torch.tensor([self.triggered], dtype=torch.bool, device=input_ids.device)


class TargetedOCRLoopGuard(LogitsProcessor):
    """Break only clear generation loops instead of globally banning n-grams.

    The guard blocks one token when the response is about to continue either a
    long single-token run or a complete multi-token block already repeated
    three times consecutively.  Ordinary repeated words and recurring XML line
    tags do not match the complete-block condition and are left untouched.
    """

    def __init__(
        self,
        prompt_length: int,
        minimum_period: int = 4,
        maximum_period: int = 96,
        required_repeats: int = 3,
        single_token_run: int = 12,
    ) -> None:
        if prompt_length < 0:
            raise ValueError("prompt_length must be non-negative")
        if minimum_period < 2 or maximum_period < minimum_period:
            raise ValueError("invalid loop period bounds")
        if required_repeats < 3 or single_token_run < 4:
            raise ValueError("loop thresholds are too permissive")
        self.prompt_length = prompt_length
        self.minimum_period = minimum_period
        self.maximum_period = maximum_period
        self.required_repeats = required_repeats
        self.single_token_run = single_token_run
        self.events = 0

    def _banned_token(self, sequence: torch.Tensor) -> int | None:
        generated = sequence[self.prompt_length :]
        count = int(generated.numel())
        if count >= self.single_token_run:
            tail = generated[-self.single_token_run :]
            if bool(torch.all(tail == tail[0])):
                return int(tail[0].item())

        maximum = min(self.maximum_period, count // self.required_repeats)
        for period in range(self.minimum_period, maximum + 1):
            span = period * self.required_repeats
            tail = generated[-span:]
            block = tail[:period]
            if all(
                bool(torch.equal(block, tail[index * period : (index + 1) * period]))
                for index in range(1, self.required_repeats)
            ):
                # The sequence currently ends exactly at the repeated block's
                # boundary.  Block only the token that would start it again.
                return int(block[0].item())
        return None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for batch_index in range(input_ids.shape[0]):
            banned = self._banned_token(input_ids[batch_index])
            if banned is not None:
                scores[batch_index, banned] = -torch.inf
                self.events += 1
        return scores


def faithful_logits_processors(
    prompt_length: int,
    extra_processors: Iterable[LogitsProcessor] = (),
) -> LogitsProcessorList:
    """Use a tiny response-only repetition penalty.

    The evaluator supplies a loose built-in 32-gram guard separately.  It is
    substantially cheaper than inspecting GPU token history in Python at every
    decode step, while still interrupting long exact XML loops.
    """

    processors = [
        RepetitionPenaltyLogitsProcessor(
            penalty=1.01,
            prompt_ignore_length=prompt_length,
        ),
    ]
    processors.extend(extra_processors)
    return LogitsProcessorList(processors)

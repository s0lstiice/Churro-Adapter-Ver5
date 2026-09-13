# Modification notice

This repository is not an unmodified copy of CHURRO or Qwen2.5-VL.

## Visual-grounding changes

- Adds image/text contrast training, including wrong-image evidence, to punish
  plausible text that is not supported by the supplied scan.
- Adds first-occurrence token weighting and corrupted-prefix examples to reduce
  learned repetition and premature continuation behavior.
- Makes `grounded-faithful` the production decoding profile. Its transcript-free
  counterfactual image check can remove a suspicious repetitive XML tail only
  when the continuation lacks visual support.
- Does not use reference transcripts at inference, insert wording from another
  recognizer, or silently rewrite ordinary substitutions.
- Adds an opt-in continuous page budget with streaming loop detection and a
  single selective recovery pass for structurally incomplete, non-looping
  pages. The packaged launchers use this instead of blanket repeat generation.

On the frozen 100-page/100-item LOC benchmark, Version 5 with legacy decoding
measured 28.75% CER and 35.42% WER. The same adapter with grounded-faithful
decoding measured 21.12% CER and 27.90% WER, reductions of 26.6% and 21.2%
relative to the legacy-decoding result.

## Ver5 adapter

- Contains independently trained rank-8 LoRA parameters for
  `stanford-oval/churro-3B`; upstream base weights are not redistributed.
- Continues the LOC visual-grounding direction described above.
- Preserves the upstream tokenizer vocabulary. Processor and tokenizer files
  are included only so the adapter loads reproducibly.

## Layout-robust inference

- Adds transcript-free orientation classification, item-level layout priors,
  validated book-gutter detection, per-region rotation, and safe abstention.
- Runs OCR only on provenance-bound regions and reconstructs their outputs in
  deterministic reading order.
- Adds SHA-256 binding for sources, normalized regions, layout decisions, and
  region-generation configuration.
- Adds strict rejection of duplicate, stale, or mismatched region predictions.
- Feeds the grounded recognizer upright, provenance-bound regions without
  changing their visible content.

Exact model and audit metrics are retained in `adapter/` and `evaluation/`.

# Visually Grounded, Layout-Robust CHURRO LOC Adapter — Epoch 22

This upload's central improvement is **visual grounding**: the recognizer was
trained and decoded to rely more strongly on visible handwriting and less on
unsupported autoregressive continuation. It combines the newest Epoch 22
full-page adapter with the layout front end used around it. The system
preserves the original scan, detects sideways/upside-down pages and book
spreads, transcribes safe page regions with CHURRO, and reconstructs one
page-ordered transcript.

The adapter is a PEFT/LoRA delta for `stanford-oval/churro-3B`; CHURRO/Qwen
base weights are **not** included. This is an independent research artifact,
not an official Stanford OVAL, Qwen, or Library of Congress release.

## Main improvement: visual grounding

Earlier versions could produce plausible continuations that were insufficiently
supported by the page, repeat text, or stop before all visible writing had been
covered. This release attacks that failure mode at both training and inference:

- **Visual-contrast training:** Epoch 22 includes image/text contrast examples,
  including deliberately mismatched visual evidence, so the adapter is
  penalized for accepting text that does not belong to the supplied page.
- **First-occurrence emphasis:** the objective gives extra weight to the first
  genuine occurrence of page text, discouraging a repeated continuation from
  becoming an easy substitute for reading new visible content.
- **Grounded-faithful decoding:** the production decoder uses a transcript-free
  counterfactual image check when it detects a suspicious repetitive XML tail.
  It removes a tail only when that continuation lacks visual support. It never
  supplies replacement wording from a reference transcript or another OCR
  model.
- **Selective recovery:** the current decoder gives each page one continuous
  generation budget and stops high-confidence repetitive tails in-stream. It
  permits one recovery pass only when the first result is structurally
  incomplete and no loop was detected; it no longer gives every flagged page
  two full-page retries.

Visual grounding is distinct from layout normalization. The layout layer makes
sure the model receives upright, sensibly ordered page regions; visual grounding
then constrains what the recognizer says about those pixels. The bundled
`run_layout_robust.py` enables both and uses `grounded-faithful` by default.

## What is included

- `adapter/`: Epoch 22 adapter, processor/tokenizer configuration, and metrics.
- `scripts/run_layout_robust.py`: one-command layout-aware inference.
- `scripts/transcribe.py`: a legacy-compatible convenience path for images
  already known to be upright, single-page scans.
- `scripts/normalize_document_layouts.py`: orientation, gutter, region, and
  safe-abstention logic.
- `scripts/expand_layout_regions_manifest.py` and
  `scripts/merge_layout_region_predictions.py`: provenance-bound region OCR
  and deterministic reconstruction.
- `evaluation/`: compact metric and audit reports; no evaluation images or
  reference transcripts are redistributed.
- `tests/`: 30 focused tests for layout handling, provenance, incremental
  merging, and counterfactual-negative behavior.

## Install

Python 3.11 or 3.12 and an NVIDIA GPU are recommended. A 6 GB GPU normally
requires 4-bit loading, which is the default.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads `stanford-oval/churro-3B` and docTR orientation
weights. Authentication with Hugging Face may improve download reliability.

## Input manifest

Create UTF-8 JSONL with one object per physical scan. Use absolute image paths.
`id` or `page_id` must be unique. `text` is optional and is used only for
evaluation; it is never needed for inference.

```json
{"page_id":"scan_0001","image":"/absolute/path/scan_0001.jpg"}
{"page_id":"scan_0002","image":"/absolute/path/scan_0002.jpg"}
```

## Run the newest system

```bash
python scripts/run_layout_robust.py \
  --manifest pages.jsonl \
  --output output/layout_robust_epoch22
```

Final page-level OCR is written to `output/layout_robust_epoch22/predictions.jsonl`.
The directory also retains normalized images, layout decisions, region
manifests, region predictions, summaries, and provenance hashes.

For a known-upright single page, a shorter legacy-compatible path is available.
It does not include the production layout normalization and grounded-tail guard:

```bash
python scripts/transcribe.py page.jpg --output output/direct_epoch22
```

To open the progress dashboard from PowerShell:

```powershell
python scripts/universal_progress_monitor/progress_dashboard.py
```

## Evaluation results

The frozen benchmark contains 100 reviewed Library of Congress pages from 100
items. These pages received no gradient updates and were not used for method
development. Metrics use the same official page references for each system.

| System | CER | WER | Output/target characters |
|---|---:|---:|---:|
| Base CHURRO | 34.68% | 42.16% | 0.927 |
| Epoch 22, legacy decoding | 28.75% | 35.42% | 1.031 |
| Epoch 22, grounded-faithful decoding | **21.12%** | **27.90%** | 1.008 |

Against base CHURRO, Epoch 22 with the production grounded-faithful inference
path reduced aggregate CER by 39.1% and WER by 33.8%. The legacy-decoding row
is the closer adapter-only comparison: it reduced CER by 17.1% and WER by
16.0%. With the **same Epoch 22 adapter**, turning on grounded-faithful decoding
reduced CER by a further 26.6% and WER by 21.2% relative to its legacy-decoding
result. This makes the visual-grounding contribution visible instead of
crediting the whole improvement to LoRA training. The production comparison
still represents a complete system, while the legacy row is the cleaner
weights-only ablation.

A separate no-retry ablation finished in 4,351.7 seconds versus 4,972.1
seconds for the retry baseline, saving 12.5%, but its CER regressed to 24.29%.
The promoted selective policy recovers only structurally incomplete,
non-looping pages. It selected 3/100 pages rather than running 16 extra
generations. The executed hybrid result was 21.86% CER and 28.11% WER, close
to the older retry baseline's 21.12% CER and 27.90% WER while substantially
reducing repeat generation. The focused execution check is included in the
evaluation audit.

Epoch 22 was trained on 2,498 full-page examples; two overlength pages were
skipped. Its final validation loss was 1.2022 under the newer
first-occurrence/visual-contrast objective. Loss values from older objectives
are not directly comparable.

The layout smoke audit used 24 difficult Thornton scans: 20 required a
transform, 22 were judged upright/usable, and two ambiguous scans safely
abstained. All 20 generated transforms retained the visible text in visual
inspection. This is a geometric audit, not an OCR-accuracy benchmark.

See the JSON reports in `evaluation/` for exact counts. CER/WER can exceed
100%; they are error rates, not accuracy percentages.

## Important limitations

- OCR remains imperfect and must be human-reviewed before archival use.
- Five of 100 production-evaluation pages ended incomplete/truncated after the
  configured retry policy.
- The 45-page omission-focused challenge set slightly favored Epoch 21 over
  Epoch 22 (38.67% vs. 38.92% CER), so Epoch 22 is not uniformly better on
  every subset.
- Orientation and gutter detection intentionally abstain when evidence is
  ambiguous. Original files are never overwritten.
- Generated OCR is not an official LOC transcript.

## Tests

```bash
pip install -r requirements-dev.txt
PYTHONPATH=scripts python -m pytest -q tests
```

The 30 focused tests above passed in the packaging workspace. That statement
does not claim that every unrelated historical test in the larger development
workspace passes.

## License and citation

This repository is distributed subject to the included Qwen Research License.
It is for research use under those terms and does not grant commercial rights.
The base-model license and usage restrictions also apply. See `LICENSE`,
`NOTICE`, `MODIFICATIONS.md`, and `CITATIONS.bib`.

CHURRO paper: <https://aclanthology.org/2025.emnlp-main.1763/>

Library of Congress resources used for evaluation remain credited to the
Library of Congress and their source collections. No scan or official
transcript is included in this upload.

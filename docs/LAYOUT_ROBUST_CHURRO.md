# Layout-robust CHURRO inference

This stage changes image presentation, not CHURRO's language weights. It is
designed for scans that are sideways, upside down, two-page spreads, mounted
scrapbook pages, or mixtures of those layouts.

## Pipeline

1. `normalize_document_layouts.py` preserves the source scan, predicts page and
   regional orientation, and records a complete transform audit.
2. A wide aspect ratio is not enough to split an image. The splitter requires
   a real, broad, low-detail partition near the center; otherwise it preserves
   the scan unchanged. Orientation is estimated from content-facing probes so
   a large blank gutter does not drown out the writing.
3. Wide scans are never globally rotated when their meaningful regions
   conflict. Confidently opposed regions are corrected independently.
4. If several unique pages from one explicit LOC item establish the same
   opposed-spread convention, that convention may resolve weak compatible
   pages. Dominant current-page evidence wins, and conflicts remain unchanged
   for review.
5. Dark album covers are not mistaken for dense writing, but no source region
   is deleted solely from that heuristic.
6. `expand_layout_regions_manifest.py` sends independently corrected regions to
   CHURRO at full inference resolution.
7. Region IDs are bound to hashes of the source pixels, normalized pixels,
   crop/rotation geometry, normalization version, and selection policy. Stale
   OCR from a prior layout run cannot silently enter a new page result.
8. `merge_layout_region_predictions.py` waits until every expected meaningful
   region is complete, rejects duplicate or mixed-run records, then joins text
   in physical reading order under the original page ID.
9. Blank/dark cover regions still receive an OCR pass, but with a small bounded
   token budget and no automatic retry. This prevents one non-text scan from
   stalling the queue for many minutes without deleting it from the audit.

The original image path, normalized composite, per-region images, source crop
boxes, rotations, output placements, model confidences, and item-prior evidence
remain in the manifests/audits.

## Current LOC queue

`run_loc_cutoff_through_hamlin_transcription.sh` performs download, layout
analysis, region OCR with the epoch-22 grounded-faithful adapter, live page
merging, and paired transcript export. Progress for every long stage appears in
the universal progress monitor.

The runner uses a single-run lock, supervises its OCR/merge/export workers, and
uses an explicit producer-complete marker. A partial JSONL write, stale output,
or failed worker is treated as an error rather than a complete transcription.

## Validation snapshot

On the 24-page Anna Maria Brodeau Thornton layout audit, the current policy
learned the repeated opposing-half convention from 9 of 20 content-bearing wide
pages. It corrected 20 scans, left 4 unchanged, and explicitly flagged 3 as
ambiguous. The 22 wide scans all passed the new center-partition check. A
separate ordinary-page smoke set previously kept all 20 upright Lincoln and
Dickinson pages at 0 degrees. These are layout checks, not OCR accuracy scores.

## Important limit

Correct orientation and full-resolution regions remove a major presentation
failure, but they do not make illegible or out-of-domain text accurate. Printed
almanac tables and very faint writing can still produce OCR errors and remain
marked for human review.

# Run the remote LOC OCR shard on the second computer

This transfer contains the remote half of the LOC transcription workload. It
contains the page-region images, Epoch 22 adapter, inference code, manifests,
and provenance records. It does **not** redistribute the CHURRO base weights;
the first run downloads `stanford-oval/churro-3B` from Hugging Face.

Do not start OCR until the copy to this drive has completed and the following
files exist beside this document:

- `RUN_REMOTE_WSL.sh`
- `RUN_REMOTE.ps1`
- `data/regions.jsonl`
- `runtime/adapter/adapter_model.safetensors`

## Fast path: the other computer has the same WSL environment

From PowerShell in this folder:

```powershell
.\RUN_REMOTE.ps1
```

The launcher expects Ubuntu 24.04 and defaults to
`/home/vgu/aiproj/.venv/bin/python`. The run is resumable: running the command
again skips completed region IDs already stored in the predictions file.

## New-computer setup or GitHub fallback

The bundled `runtime` directory is the same code and adapter intended for the
GitHub release, so GitHub is not required when that directory is present. If
the runtime or adapter is missing, download/clone the Epoch 22 visually
grounded, layout-robust release from your repository and place its contents in
this folder as `runtime/`, with the adapter at `runtime/adapter/`.

Then open Ubuntu 24.04 through WSL, change to this transfer folder, and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r runtime/requirements.txt
PYTHON="$PWD/.venv/bin/python" bash RUN_REMOTE_WSL.sh
```

If the drive appears under a different WSL mount, locate it first with:

```bash
findmnt -t 9p,drvfs
```

## Monitor progress

From PowerShell in this folder, this shows how many of the 6,510 assigned OCR
regions have finished:

```powershell
if (Test-Path .\output\region_recognition\predictions.jsonl) {
    (Get-Content .\output\region_recognition\predictions.jsonl | Measure-Object -Line).Lines
} else {
    0
}
```

The universal progress state is also written under `.progress_tasks` in the
working directory used by the launcher.

The launcher uses selective recovery: each region receives one continuous
visual pass, obvious repetitive continuations are stopped in-stream, and at
most one additional pass is allowed only when the result is structurally
incomplete and the loop guard did not fire. It does not use the older blanket
two-retry policy.

## Files to bring back

After OCR finishes, safely return the drive to the primary computer. The
required result is:

```text
output/region_recognition/predictions.jsonl
```

Also retain `output/page_drafts.jsonl` for convenient human inspection. The
primary computer will strictly validate provenance, combine this remote result
with the local half, and reconstruct the original page order.

Do not rename or edit `data/regions.jsonl`, the IDs inside the predictions
file, or any provenance fields. Human corrections should be made only after
the two shards have been merged.

# Universal Progress Monitor

This local dashboard shows long-running training, evaluation, crop generation,
downloads, mining, and other batch jobs in one place.

It supports three progress levels:

1. **Exact reporter** — a script explicitly reports its current and total work.
2. **Reported/inferred** — the monitor reads an existing `progress.json` or an
   epoch-based `metrics.jsonl` next to the process output.
3. **Activity only** — the process is alive, but it does not expose a reliable
   denominator. The dashboard shows runtime and resource use without inventing
   a percentage.

## Start the application

From PowerShell in `training_data`, use the top-level launcher:

```powershell
.\OPEN_PROGRESS_MONITOR.ps1
```

The browser opens to <http://127.0.0.1:8765>. Leave the PowerShell window open;
press `Ctrl+C` there to stop the dashboard.

From WSL instead:

```bash
bash universal_progress_monitor/launch_progress_monitor.sh
```

Then open <http://127.0.0.1:8765>.

The Windows launcher reads all exact reporter files in `.progress_tasks` and
does not require WSL. The WSL launcher additionally discovers active WSL Python
and shell jobs. Long-running project jobs should use `ProgressTask` or
`run_tracked.py` so both launch modes show exact progress.

## Track any future command

The wrapper preserves the command's normal console output and exit code while
creating a dashboard record. It recognizes JSON progress, `epoch 2/6`,
`page 20/100`, `20/100`, and percentage output.

```bash
python3 universal_progress_monitor/run_tracked.py \
  --name "Polygon line training" \
  --log runs/polygon-line.log \
  -- python3 train_polygon_line_mask_student.py --epochs 6 --out runs/polygon-line
```

If the program prints the current count but not its total, declare it:

```bash
python3 universal_progress_monitor/run_tracked.py \
  --name "Crop 1173 pages" \
  --total 1173 \
  --unit pages \
  -- python3 crop_pages.py
```

## Add exact progress to Python code

Import the reporter when accurate item-level progress and ETA matter:

```python
from universal_progress_monitor.progress_client import ProgressTask

with ProgressTask("Build word crops", total=len(pages), unit="pages") as progress:
    for index, page in enumerate(pages, 1):
        result = process_page(page)
        progress.update(
            index,
            message=f"finished {page.name}",
            metrics={"accepted_crops": result.accepted},
        )
```

The context manager records successful completion or the exception that stopped
the job. State is written atomically to `training_data/.progress_tasks`.

For scripts outside this workspace, point all reporters and the dashboard at the
same directory:

```bash
export PROGRESS_STATE_DIR=/path/to/shared/.progress_tasks
```

## One-shot status for commands or automation

```bash
python3 universal_progress_monitor/progress_dashboard.py --snapshot
```

This prints the same structured JSON used by the web application.

## What the application can and cannot know

The monitor never fabricates progress. A live process that provides no total is
shown as active with an indeterminate bar. Use `run_tracked.py` or
`ProgressTask` for exact percentages and useful ETAs on future work. Passive
epoch estimates represent completed epochs; work within the current epoch is
not guessed.

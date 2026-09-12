#!/usr/bin/env python3
"""Run any command and expose its printed progress to the dashboard."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any, Optional

try:
    from .progress_client import ProgressTask
except ImportError:  # Direct execution: python3 run_tracked.py ...
    from progress_client import ProgressTask


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
RATIO_PATTERNS = (
    re.compile(r"\bepoch\s*[:#]?\s*(\d+)\s*(?:/|of)\s*(\d+)\b", re.I),
    re.compile(r"\b(?:item|page|batch|step|file)s?\s*[:#]?\s*(\d+)\s*(?:/|of)\s*(\d+)\b", re.I),
    re.compile(r"(?<![\d.])(\d+)\s*/\s*(\d+)(?![\d.])"),
)
PERCENT_PATTERN = re.compile(r"(?<![\d.])(100|\d{1,2})(?:\.\d+)?\s*%")


def parse_progress(text: str) -> tuple[Optional[float], Optional[float], dict[str, Any]]:
    """Return current, total, and useful metrics from one output record."""
    clean = ANSI.sub("", text).strip()
    metrics: dict[str, Any] = {}
    try:
        value = json.loads(clean)
    except (ValueError, TypeError):
        value = None
    if isinstance(value, dict):
        for key in ("loss", "train_loss", "validation_loss", "validation_iou", "iou", "f1", "cer", "wer", "accuracy"):
            if isinstance(value.get(key), (int, float)):
                metrics[key] = value[key]
        pairs = (
            ("current", "total"),
            ("processed", "total"),
            ("completed", "total"),
            ("epoch", "epochs"),
            ("page", "pages"),
            ("page", "pages_total"),
            ("item", "items_total"),
        )
        for current_key, total_key in pairs:
            current, total = value.get(current_key), value.get(total_key)
            if isinstance(current, (int, float)) and isinstance(total, (int, float)) and total > 0:
                return float(current), float(total), metrics
    for pattern in RATIO_PATTERNS:
        match = pattern.search(clean)
        if match and float(match.group(2)) > 0:
            return float(match.group(1)), float(match.group(2)), metrics
    match = PERCENT_PATTERN.search(clean)
    if match:
        return float(match.group(1)), 100.0, metrics
    return None, None, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Human-readable task name")
    parser.add_argument("--task-id", help="Stable dashboard id")
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--total", type=float, help="Known total when output only prints a current count")
    parser.add_argument("--unit", default="items")
    parser.add_argument("--log", type=Path, help="Combined stdout/stderr log destination")
    parser.add_argument("--quiet", action="store_true", help="Do not mirror child output to stdout")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("a command is required after --")

    cwd = (args.cwd or Path.cwd()).resolve()
    log_path = args.log.resolve() if args.log else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    command_text = shlex.join(command)
    task = ProgressTask(
        args.name,
        args.total,
        unit=args.unit,
        task_id=args.task_id,
        state_dir=args.state_dir,
        command=command_text,
        output_dir=cwd,
        metadata={"wrapper_pid": os.getpid(), "log": str(log_path) if log_path else None},
    )
    task.message = "launching command"
    task.write()

    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    task.metadata["child_pid"] = process.pid
    task.message = "running"
    task.write()
    recent = deque(maxlen=30)
    log_handle = log_path.open("a", encoding="utf-8") if log_path else None

    def forward_signal(signum, _frame) -> None:
        if process.poll() is None:
            process.send_signal(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, forward_signal)

    try:
        assert process.stdout is not None
        for raw_line in process.stdout:
            if not args.quiet:
                print(raw_line, end="", flush=True)
            if log_handle:
                log_handle.write(raw_line)
                log_handle.flush()
            clean = ANSI.sub("", raw_line).strip()
            if not clean:
                continue
            recent.append(clean[-500:])
            current, parsed_total, metrics = parse_progress(clean)
            total = parsed_total if parsed_total is not None else args.total
            if current is not None:
                task.update(current, total=total, message=clean[-240:], metrics=metrics)
            else:
                task.message = clean[-240:]
                task.metrics.update(metrics)
                task.metadata["recent_output"] = list(recent)
                task.write()
        return_code = process.wait()
        task.metadata["recent_output"] = list(recent)
        if return_code == 0:
            task.finish("command completed successfully")
        else:
            task.fail(f"command exited with code {return_code}", return_code=return_code)
        return return_code
    except BaseException as error:
        if process.poll() is None:
            process.terminate()
        task.fail(f"wrapper error: {type(error).__name__}: {error}")
        raise
    finally:
        if log_handle:
            log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())

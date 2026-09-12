#!/usr/bin/env python3
"""Tiny, dependency-free progress reporter used by the dashboard.

Example:
    from universal_progress_monitor.progress_client import ProgressTask

    with ProgressTask("prepare crops", total=len(pages), unit="pages") as task:
        for index, page in enumerate(pages, 1):
            process(page)
            task.update(index, message=page.name)
"""

from __future__ import annotations

import json
import os
import re
import socket
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_state_dir() -> Path:
    configured = os.environ.get("PROGRESS_STATE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parent.parent / ".progress_tasks"


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-.")
    return cleaned[:80] or f"task-{uuid.uuid4().hex[:10]}"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    temporary.write_text(serialized, encoding="utf-8")
    # Antivirus, Explorer, and the Windows dashboard can briefly hold a read
    # handle that blocks os.replace on an NTFS path mounted in WSL. Progress
    # reporting must never terminate the workload it is observing.
    for attempt in range(50):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            time.sleep(min(0.25, 0.01 * (attempt + 1)))
    try:
        path.write_text(serialized, encoding="utf-8")
    except OSError:
        pass
    try:
        temporary.unlink(missing_ok=True)
    except OSError:
        pass


class ProgressTask:
    """Write an exact, machine-readable progress sidecar for a long-running job."""

    def __init__(
        self,
        name: str,
        total: Optional[float] = None,
        *,
        unit: str = "items",
        task_id: Optional[str] = None,
        state_dir: Optional[Path | str] = None,
        command: Optional[str] = None,
        output_dir: Optional[Path | str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        unique = task_id or f"{_safe_id(name)}-{uuid.uuid4().hex[:8]}"
        self.task_id = _safe_id(unique)
        self.name = name
        self.total = float(total) if total is not None else None
        self.current = 0.0
        self.unit = unit
        self.state_dir = Path(state_dir).resolve() if state_dir else _default_state_dir()
        self.path = self.state_dir / f"{self.task_id}.json"
        self.started_monotonic = time.monotonic()
        self.started_at = _utc_now()
        self.status = "running"
        self.message = "starting"
        self.command = command or " ".join(sys.argv)
        self.output_dir = str(Path(output_dir).resolve()) if output_dir else None
        self.metadata = dict(metadata or {})
        self.metrics: dict[str, Any] = {}
        self.write()

    def _payload(self) -> dict[str, Any]:
        elapsed = max(0.0, time.monotonic() - self.started_monotonic)
        percent = None
        eta_seconds = None
        if self.total and self.total > 0:
            percent = max(0.0, min(100.0, 100.0 * self.current / self.total))
            if self.current > 0 and self.current < self.total:
                eta_seconds = elapsed * (self.total - self.current) / self.current
            elif self.current >= self.total:
                eta_seconds = 0.0
        return {
            "schema_version": 1,
            "id": self.task_id,
            "name": self.name,
            "status": self.status,
            "source": "exact reporter",
            "confidence": "exact",
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "command": self.command,
            "output_dir": self.output_dir,
            "current": self.current,
            "total": self.total,
            "unit": self.unit,
            "percent": percent,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta_seconds,
            "message": self.message,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "started_at": self.started_at,
            "updated_at": _utc_now(),
        }

    def write(self) -> None:
        _atomic_json(self.path, self._payload())

    def update(
        self,
        current: Optional[float] = None,
        *,
        advance: Optional[float] = None,
        total: Optional[float] = None,
        message: Optional[str] = None,
        metrics: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if current is not None:
            self.current = float(current)
        if advance is not None:
            self.current += float(advance)
        if total is not None:
            self.total = float(total)
        if message is not None:
            self.message = message
        if metrics:
            self.metrics.update(metrics)
        self.write()

    def finish(self, message: str = "complete", metrics: Optional[Mapping[str, Any]] = None) -> None:
        if self.total is not None:
            self.current = self.total
        if metrics:
            self.metrics.update(metrics)
        self.status = "complete"
        self.message = message
        self.write()

    def fail(self, message: str, *, return_code: Optional[int] = None) -> None:
        self.status = "failed"
        self.message = message
        if return_code is not None:
            self.metadata["return_code"] = return_code
        self.write()

    def __enter__(self) -> "ProgressTask":
        return self

    def __exit__(self, exception_type, exception, traceback) -> bool:
        if exception is None:
            self.finish()
        else:
            self.fail(f"{exception_type.__name__}: {exception}")
        return False

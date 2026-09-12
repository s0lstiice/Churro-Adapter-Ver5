#!/usr/bin/env python3
"""Universal local progress dashboard for training, mining, and batch jobs.

The dashboard combines exact reporter sidecars with passive WSL/Linux process
discovery. It uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import urlparse


# Windows PowerShell commonly exposes a legacy console encoding. Keep snapshot
# and diagnostic output robust when task metadata contains Unicode text.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


OUTPUT_ARGUMENTS = (
    "--out",
    "--output",
    "--output-dir",
    "--output_dir",
    "--out-root",
    "--out_root",
    "--save-dir",
    "--save_dir",
    "--run-dir",
    "--run_dir",
)
METRIC_KEYS = (
    "train_loss",
    "validation_loss",
    "val_loss",
    "loss",
    "validation_iou",
    "val_iou",
    "iou",
    "validation_f1",
    "f1",
    "precision",
    "validation_precision",
    "recall",
    "validation_recall",
    "accuracy",
    "cer",
    "wer",
)
IGNORE_PROGRAMS = {"tee", "sleep", "grep", "find", "tail", "cat", "sed", "ps", "nvidia-smi"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def safe_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def json_safe(value: Any) -> Any:
    """Return JSON-safe task data without allowing NaN to break the API."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def human_name(argv: list[str]) -> str:
    for token in argv[1:]:
        if token.endswith((".py", ".sh", ".ps1")):
            return Path(token).stem.replace("_", " ")
    return Path(argv[0]).name.replace("_", " ") if argv else "unknown job"


def argument_value(argv: list[str], names: Iterable[str]) -> Optional[str]:
    wanted = set(names)
    for index, token in enumerate(argv):
        if token in wanted and index + 1 < len(argv):
            return argv[index + 1]
        for name in wanted:
            prefix = name + "="
            if token.startswith(prefix):
                return token[len(prefix) :]
    return None


def resolve_process_path(value: Optional[str], cwd: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and cwd:
        path = Path(cwd) / path
    try:
        return path.resolve()
    except OSError:
        return path


def read_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, UnicodeError):
        return None


def read_last_jsonl(path: Path) -> Optional[dict[str, Any]]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 131072))
            data = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(data.splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def useful_metrics(value: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key in METRIC_KEYS:
        number = safe_float(value.get(key))
        if number is not None:
            metrics[key] = number
    return metrics


def process_exists(pid: Any) -> bool:
    try:
        numeric = int(pid)
    except (TypeError, ValueError):
        return False
    if os.name == "posix":
        return Path(f"/proc/{numeric}").exists()
    try:
        os.kill(numeric, 0)
        return True
    except OSError:
        return False


class ProgressMonitor:
    def __init__(self, roots: list[Path], state_dir: Path, completed_hours: float = 72.0) -> None:
        self.roots = roots
        self.state_dir = state_dir
        self.completed_seconds = completed_hours * 3600.0
        self.started = time.monotonic()
        self._gpu_cache: tuple[float, list[dict[str, Any]]] = (0.0, [])
        self._lock = threading.Lock()

    def _explicit_tasks(self) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        self.state_dir.mkdir(parents=True, exist_ok=True)
        now = time.time()
        for path in self.state_dir.glob("*.json"):
            value = read_json(path)
            if not value:
                continue
            try:
                age = now - path.stat().st_mtime
            except OSError:
                age = 0.0
            status = str(value.get("status", "unknown"))
            pid = value.get("metadata", {}).get("child_pid") if isinstance(value.get("metadata"), dict) else None
            pid = pid or value.get("pid")
            if status == "running":
                # A native Windows dashboard cannot inspect a Linux PID reported
                # by WSL. Fresh sidecar activity is authoritative in that mode.
                interrupted = age > 900.0 if os.name == "nt" else not process_exists(pid)
                if interrupted:
                    value["status"] = "interrupted"
                    value["message"] = "reporting process is no longer updating"
            if value.get("status") != "running" and age > self.completed_seconds:
                continue
            value.setdefault("id", path.stem)
            value.setdefault("name", path.stem.replace("-", " "))
            value.setdefault("source", "exact reporter")
            value.setdefault("confidence", "exact")
            value["state_file"] = str(path)
            value["age_seconds"] = age
            tasks.append(value)
        return tasks

    @staticmethod
    def _linux_processes() -> list[dict[str, Any]]:
        if os.name != "posix" or not hasattr(os, "sysconf"):
            return []
        proc = Path("/proc")
        if not proc.exists():
            return []
        try:
            uptime = float(Path("/proc/uptime").read_text().split()[0])
            ticks = float(os.sysconf(os.sysconf_names["SC_CLK_TCK"]))
            page_size = float(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, KeyError):
            return []
        found: list[dict[str, Any]] = []
        own_pid = os.getpid()
        for entry in proc.iterdir():
            if not entry.name.isdigit() or int(entry.name) == own_pid:
                continue
            try:
                raw = (entry / "cmdline").read_bytes()
                argv = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
                if not argv:
                    continue
                executable = Path(argv[0]).name
                has_script = any(token.endswith((".py", ".sh", ".ps1")) for token in argv[1:])
                interesting = has_script and (
                    "python" in executable.lower()
                    or executable in {"bash", "sh", "zsh", "pwsh", "powershell"}
                )
                if not interesting or executable in IGNORE_PROGRAMS:
                    continue
                joined = " ".join(argv)
                if "progress_dashboard.py" in joined:
                    continue
                stat_text = (entry / "stat").read_text(encoding="utf-8")
                remainder = stat_text[stat_text.rfind(")") + 2 :].split()
                state = remainder[0]
                parent_pid = int(remainder[1])
                cpu_seconds = (float(remainder[11]) + float(remainder[12])) / ticks
                start_ticks = float(remainder[19]) / ticks
                elapsed = max(0.0, uptime - start_ticks)
                rss_pages = float((entry / "statm").read_text().split()[1])
                try:
                    cwd = os.readlink(entry / "cwd")
                except OSError:
                    cwd = None
            except (OSError, ValueError, IndexError, UnicodeError):
                continue
            found.append(
                {
                    "pid": int(entry.name),
                    "parent_pid": parent_pid,
                    "argv": argv,
                    "command": joined,
                    "state_code": state,
                    "cwd": cwd,
                    "elapsed_seconds": elapsed,
                    "cpu_percent_lifetime": 100.0 * cpu_seconds / elapsed if elapsed > 0 else 0.0,
                    "memory_mb": rss_pages * page_size / (1024.0 * 1024.0),
                }
            )
        return found

    def _progress_from_output(self, output_dir: Optional[Path], argv: list[str]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "current": None,
            "total": None,
            "percent": None,
            "unit": None,
            "message": "active; exact percentage is not exposed",
            "metrics": {},
            "updated_at": None,
            "confidence": "activity-only",
        }
        if not output_dir:
            return result
        candidates = [output_dir / "progress.json"]
        for candidate in candidates:
            progress = read_json(candidate)
            if not progress:
                continue
            pairs = (
                ("current", "total"),
                ("processed", "total"),
                ("completed", "total"),
                ("page", "pages_total"),
                ("item", "items_total"),
            )
            for current_key, total_key in pairs:
                current = safe_float(progress.get(current_key))
                total = safe_float(progress.get(total_key))
                if current is not None and total and total > 0:
                    result.update(
                        current=current,
                        total=total,
                        percent=clamp(100.0 * current / total, 0.0, 100.0),
                        unit=progress.get("unit", "items"),
                        message=progress.get("message", f"{current:g}/{total:g}"),
                        confidence="reported",
                        metrics=useful_metrics(progress),
                        updated_at=datetime.fromtimestamp(candidate.stat().st_mtime, timezone.utc).isoformat(),
                    )
                    return result

        metrics_path = output_dir / "metrics.jsonl"
        latest = read_last_jsonl(metrics_path)
        if latest:
            current = safe_float(latest.get("epoch"))
            total_value = argument_value(argv, ("--epochs", "--num-epochs", "--num_epochs"))
            try:
                total = safe_float(float(total_value)) if total_value else None
            except (TypeError, ValueError):
                total = None
            result["metrics"] = useful_metrics(latest)
            try:
                result["updated_at"] = datetime.fromtimestamp(metrics_path.stat().st_mtime, timezone.utc).isoformat()
            except OSError:
                pass
            if current is not None and total and total > 0:
                result.update(
                    current=current,
                    total=total,
                    percent=clamp(100.0 * current / total, 0.0, 100.0),
                    unit="epochs",
                    message=f"epoch {current:g}/{total:g} complete; next epoch is active",
                    confidence="inferred from metrics",
                )
            else:
                result.update(message="metrics are updating; total is unknown", confidence="inferred from metrics")
        return result

    def _passive_tasks(self, ignored_pids: set[int]) -> list[dict[str, Any]]:
        processes = self._linux_processes()
        groups: dict[str, list[dict[str, Any]]] = {}
        for process in processes:
            if process["pid"] in ignored_pids:
                continue
            groups.setdefault(process["command"], []).append(process)
        tasks: list[dict[str, Any]] = []
        for command, members in groups.items():
            members.sort(key=lambda item: item["elapsed_seconds"], reverse=True)
            primary = members[0]
            argv = primary["argv"]
            output_value = argument_value(argv, OUTPUT_ARGUMENTS)
            output_dir = resolve_process_path(output_value, primary.get("cwd"))
            progress = self._progress_from_output(output_dir, argv)
            elapsed = primary["elapsed_seconds"]
            eta = None
            current, total = progress.get("current"), progress.get("total")
            if current and total and 0 < current < total:
                eta = elapsed * (total - current) / current
            states = {member["state_code"] for member in members}
            if "T" in states:
                status, status_message = "paused", "process is suspended"
            elif states <= {"S", "I"}:
                status, status_message = "waiting", progress["message"]
            else:
                status, status_message = "running", progress["message"]
            tasks.append(
                {
                    "id": f"proc-{primary['pid']}",
                    "name": human_name(argv),
                    "status": status,
                    "source": "passive WSL process discovery",
                    "confidence": progress["confidence"],
                    "pid": primary["pid"],
                    "worker_pids": [member["pid"] for member in members],
                    "workers": len(members),
                    "command": command,
                    "cwd": primary.get("cwd"),
                    "output_dir": str(output_dir) if output_dir else None,
                    "current": current,
                    "total": total,
                    "unit": progress.get("unit"),
                    "percent": progress.get("percent"),
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                    "message": status_message,
                    "metrics": progress.get("metrics", {}),
                    "cpu_percent_lifetime": sum(member["cpu_percent_lifetime"] for member in members),
                    "memory_mb": sum(member["memory_mb"] for member in members),
                    "updated_at": progress.get("updated_at") or utc_now(),
                }
            )
        return tasks

    def _gpu(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        with self._lock:
            if now - self._gpu_cache[0] < 3.0:
                return self._gpu_cache[1]
        command = [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=3, check=False)
            rows = []
            for line in completed.stdout.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) != 7:
                    continue
                rows.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "utilization": safe_float(float(parts[2])),
                        "memory_used_mb": safe_float(float(parts[3])),
                        "memory_total_mb": safe_float(float(parts[4])),
                        "temperature_c": safe_float(float(parts[5])),
                        "power_w": safe_float(float(parts[6])),
                    }
                )
        except (OSError, subprocess.SubprocessError, ValueError):
            rows = []
        with self._lock:
            self._gpu_cache = (now, rows)
        return rows

    @staticmethod
    def _host() -> dict[str, Any]:
        result: dict[str, Any] = {"hostname": socket.gethostname(), "platform": sys.platform}
        try:
            load = os.getloadavg()
            result["load_average"] = list(load)
        except (AttributeError, OSError):
            pass
        try:
            values: dict[str, float] = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, raw = line.split(":", 1)
                values[key] = float(raw.strip().split()[0]) / 1024.0
            result["memory_total_mb"] = values.get("MemTotal")
            result["memory_available_mb"] = values.get("MemAvailable")
        except (OSError, ValueError, IndexError):
            pass
        return result

    def snapshot(self) -> dict[str, Any]:
        explicit = self._explicit_tasks()
        active = [task for task in explicit if task.get("status") in {"running", "waiting", "paused"}]
        terminal = [task for task in explicit if task.get("status") not in {"running", "waiting", "paused"}]
        terminal.sort(key=lambda task: float(task.get("age_seconds", 0.0)))
        # Keep the dashboard responsive even after a workspace has accumulated
        # hundreds of historical sidecars. Active jobs are never capped.
        explicit = active + terminal[:75]
        ignored: set[int] = set()
        for task in explicit:
            for value in (task.get("pid"), task.get("metadata", {}).get("child_pid") if isinstance(task.get("metadata"), dict) else None):
                try:
                    ignored.add(int(value))
                except (TypeError, ValueError):
                    pass
        tasks = explicit + self._passive_tasks(ignored)
        order = {"running": 0, "waiting": 1, "paused": 2, "failed": 3, "interrupted": 4, "complete": 5}
        tasks.sort(key=lambda task: (order.get(str(task.get("status")), 9), str(task.get("name", ""))))
        return {
            "generated_at": utc_now(),
            "monitor_uptime_seconds": time.monotonic() - self.started,
            "roots": [str(root) for root in self.roots],
            "state_dir": str(self.state_dir),
            "host": self._host(),
            "gpus": self._gpu(),
            "tasks": tasks,
        }


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Progress Monitor</title>
<style>
:root{color-scheme:dark;--bg:#0b0f14;--panel:#121922;--panel2:#17212d;--line:#263547;--text:#edf4fb;--muted:#92a4b8;--blue:#50a7ff;--green:#55d68b;--amber:#ffbf5b;--red:#ff6b73;--purple:#b28cff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% -10%,#152a40 0,transparent 36%),var(--bg);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--text)}
main{max-width:1200px;margin:auto;padding:28px 20px 60px}.top{display:flex;justify-content:space-between;gap:20px;align-items:flex-end;margin-bottom:22px}h1{font-size:27px;margin:0 0 4px;letter-spacing:-.02em}.sub,.muted{color:var(--muted)}.live{display:flex;align-items:center;gap:8px;color:var(--green)}.dot{width:9px;height:9px;border-radius:50%;background:currentColor;box-shadow:0 0 14px currentColor}
.hardware{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin-bottom:22px}.hardware>div,.card{background:linear-gradient(145deg,var(--panel),#101720);border:1px solid var(--line);border-radius:14px;box-shadow:0 12px 35px #0004}.hardware>div{padding:14px 16px}.hw-title{display:flex;justify-content:space-between;margin-bottom:9px}.track{height:8px;border-radius:8px;background:#26313d;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,var(--blue),var(--purple));transition:width .4s ease}.fill.green{background:linear-gradient(90deg,#35bc7a,var(--green))}
.controls{display:flex;justify-content:space-between;align-items:center;margin:24px 0 12px}h2{font-size:17px;margin:0}.cards{display:grid;gap:13px}.card{padding:17px}.cardhead{display:flex;align-items:flex-start;justify-content:space-between;gap:16px}.name{font-size:17px;font-weight:700}.badges{display:flex;gap:7px;flex-wrap:wrap;justify-content:flex-end}.badge{font-size:11px;text-transform:uppercase;letter-spacing:.06em;border:1px solid var(--line);border-radius:999px;padding:4px 8px;color:var(--muted)}.badge.running{color:var(--green);border-color:#2b6d4a}.badge.waiting,.badge.paused{color:var(--amber);border-color:#70582e}.badge.failed,.badge.interrupted{color:var(--red);border-color:#723940}.badge.exact{color:var(--blue);border-color:#315e89}
.message{margin:7px 0 14px;color:#c5d2df}.progress-row{display:grid;grid-template-columns:1fr auto;gap:14px;align-items:center}.bigtrack{height:11px;background:#26313d;border-radius:10px;overflow:hidden}.bigfill{height:100%;background:linear-gradient(90deg,var(--blue),var(--purple));transition:width .6s ease}.pct{font-variant-numeric:tabular-nums;font-weight:750;min-width:58px;text-align:right}.stats{display:flex;flex-wrap:wrap;gap:8px 18px;margin-top:12px;color:var(--muted)}.stats b{color:var(--text);font-weight:600}.metrics{display:flex;gap:8px;flex-wrap:wrap;margin-top:13px}.metric{padding:6px 9px;border-radius:8px;background:var(--panel2);border:1px solid var(--line);font-variant-numeric:tabular-nums}.metric span{color:var(--muted);margin-right:6px}
details{margin-top:12px;color:var(--muted)}summary{cursor:pointer;user-select:none}.command{white-space:pre-wrap;word-break:break-all;background:#091018;border:1px solid var(--line);padding:10px;border-radius:9px;margin-top:8px;font:12px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace}.empty{padding:32px;text-align:center;color:var(--muted);border:1px dashed var(--line);border-radius:14px}.error{color:var(--red)}@media(max-width:650px){.top,.cardhead{align-items:flex-start;flex-direction:column}.badges{justify-content:flex-start}.progress-row{grid-template-columns:1fr}.pct{text-align:left}}
</style></head><body><main>
<div class="top"><div><h1>Universal Progress Monitor</h1><div class="sub">Training, mining, downloads, evaluation, and batch processing</div></div><div class="live"><span class="dot"></span><span id="connection">connecting</span></div></div>
<div id="hardware" class="hardware"></div><div class="controls"><h2>Jobs</h2><div class="muted" id="updated"></div></div><div id="cards" class="cards"><div class="empty">Looking for active jobs…</div></div>
</main><script>
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const duration=s=>{if(s==null||!isFinite(s))return 'unknown';s=Math.max(0,Math.round(s));const d=Math.floor(s/86400),h=Math.floor(s%86400/3600),m=Math.floor(s%3600/60),x=s%60;return [d&&d+'d',h&&h+'h',m&&m+'m',(!d&&!h)&&x+'s'].filter(Boolean).join(' ')};
const num=value=>{const x=Number(value);return !Number.isFinite(x)?'—':(Math.abs(x)>=100?x.toFixed(1):Math.abs(x)>=1?x.toFixed(3):x.toFixed(5)).replace(/0+$/,'').replace(/\.$/,'')};
function hardware(data){const parts=[];(data.gpus||[]).forEach(g=>{const mp=g.memory_total_mb?100*g.memory_used_mb/g.memory_total_mb:0;parts.push(`<div><div class="hw-title"><b>GPU ${g.index}: ${esc(g.name)}</b><span>${num(g.utilization)}%</span></div><div class="track"><div class="fill" style="width:${g.utilization||0}%"></div></div><div class="stats"><span>VRAM <b>${(g.memory_used_mb/1024).toFixed(2)} / ${(g.memory_total_mb/1024).toFixed(2)} GB</b></span><span><b>${num(g.temperature_c)}°C</b></span><span><b>${num(g.power_w)} W</b></span></div></div>`)});const h=data.host||{};if(h.memory_total_mb){const used=100*(1-h.memory_available_mb/h.memory_total_mb);parts.push(`<div><div class="hw-title"><b>System memory</b><span>${num(used)}%</span></div><div class="track"><div class="fill green" style="width:${used}%"></div></div><div class="stats"><span>Available <b>${(h.memory_available_mb/1024).toFixed(2)} GB</b></span><span>Load <b>${(h.load_average||[]).map(num).join(' / ')}</b></span></div></div>`)}document.getElementById('hardware').innerHTML=parts.join('')}
function card(t){const p=t.percent==null?null:Math.max(0,Math.min(100,t.percent));const confidence=String(t.confidence||'activity-only');const metrics=Object.entries(t.metrics||{}).slice(0,9).map(([k,v])=>`<div class="metric"><span>${esc(k.replace(/^validation_/,'val ').replaceAll('_',' '))}</span>${num(v)}</div>`).join('');const count=t.current!=null&&t.total!=null?`${num(t.current)} / ${num(t.total)} ${esc(t.unit||'')}`:'percentage unavailable';return `<article class="card"><div class="cardhead"><div><div class="name">${esc(t.name)}</div><div class="muted">${esc(t.source||'')}</div></div><div class="badges"><span class="badge ${esc(t.status)}">${esc(t.status)}</span><span class="badge ${confidence==='exact'?'exact':''}">${esc(confidence)}</span></div></div><div class="message">${esc(t.message||'')}</div><div class="progress-row"><div class="bigtrack"><div class="bigfill" style="width:${p??0}%"></div></div><div class="pct">${p==null?'—':p.toFixed(1)+'%'}</div></div><div class="stats"><span>Progress <b>${count}</b></span><span>Elapsed <b>${duration(t.elapsed_seconds)}</b></span><span>ETA <b>${duration(t.eta_seconds)}</b></span>${t.workers?`<span>Processes <b>${t.workers}</b></span>`:''}${t.memory_mb?`<span>RAM <b>${(t.memory_mb/1024).toFixed(2)} GB</b></span>`:''}</div>${metrics?`<div class="metrics">${metrics}</div>`:''}<details><summary>Paths and command</summary>${t.output_dir?`<div class="command">Output: ${esc(t.output_dir)}</div>`:''}<div class="command">${esc(t.command||'No command recorded')}</div></details></article>`}
async function refresh(){try{const r=await fetch('/api/status',{cache:'no-store'});if(!r.ok)throw Error(r.statusText);const data=await r.json();hardware(data);const tasks=data.tasks||[];document.getElementById('cards').innerHTML=tasks.length?tasks.map(card).join(''):'<div class="empty">No tracked or discoverable jobs are currently running.</div>';document.getElementById('connection').textContent='live';document.getElementById('connection').className='';document.getElementById('updated').textContent='Updated '+new Date(data.generated_at).toLocaleTimeString()}catch(e){document.getElementById('connection').textContent='disconnected';document.getElementById('connection').className='error'}}refresh();setInterval(refresh,3000);
</script></body></html>'''


class DashboardHandler(BaseHTTPRequestHandler):
    monitor: ProgressMonitor

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/api/status":
            body = json.dumps(
                json_safe(self.monitor.snapshot()), ensure_ascii=False, allow_nan=False
            ).encode("utf-8")
            content_type = "application/json; charset=utf-8"
        elif route in {"/", "/index.html"}:
            body = HTML.encode("utf-8")
            content_type = "text/html; charset=utf-8"
        elif route == "/api/health":
            body = b'{"ok":true}\n'
            content_type = "application/json; charset=utf-8"
        elif route == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        else:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string: str, *args: Any) -> None:
        if self.path != "/api/status":
            super().log_message(format_string, *args)


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", help="Workspace root; may be repeated")
    parser.add_argument("--state-dir", type=Path, default=here.parent / ".progress_tasks")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--completed-hours", type=float, default=72.0)
    parser.add_argument("--snapshot", action="store_true", help="Print one JSON snapshot and exit")
    args = parser.parse_args()
    roots = [root.expanduser().resolve() for root in (args.root or [here.parent])]
    monitor = ProgressMonitor(roots, args.state_dir.expanduser().resolve(), args.completed_hours)
    if args.snapshot:
        print(json.dumps(monitor.snapshot(), indent=2, ensure_ascii=False))
        return 0
    DashboardHandler.monitor = monitor
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Progress dashboard: http://{args.host}:{args.port}", flush=True)
    print(f"State directory: {monitor.state_dir}", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nStopping dashboard.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

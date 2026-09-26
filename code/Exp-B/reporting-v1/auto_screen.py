#!/usr/bin/env python3
"""Non-destructive eight-GPU scheduler for the Exp-B early-training screen."""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path("./Code/Exp-B/reporting-v1")
STATE, LOG, RUNNER = ROOT / "state.json", ROOT / "autopilot.log", ROOT / "run_trial.sh"
ANALYZER = ROOT / "analyze_screen.py"
MATRIX = [(h, seed) for seed in (13, 29, 47) for h in ("0.0", "0.25", "0.5", "0.75", "1.0")]


def record(message: str) -> None:
    line = f"[{time.strftime('%F %T')}] {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle: handle.write(line + "\n")


def idle() -> bool:
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"], text=True).strip()
    return not output or all(int(row.strip()) <= 1024 for row in output.splitlines())


def name(h: str, seed: int) -> str: return f"screen_q3x3x4_h{h.replace('.', 'p')}_seed{seed}"


def prune_completed_checkpoints(state: dict) -> None:
    """Retain auditable metrics/logs but release disposable screening checkpoints."""
    ledger = ROOT / "checkpoint_pruning.jsonl"
    for trial in state["completed"]:
        checkpoint = ROOT / "runs" / trial / "checkpoints"
        if not checkpoint.exists():
            continue
        files = [str(path.relative_to(checkpoint)) for path in checkpoint.rglob("*") if path.is_file()]
        size = sum(path.stat().st_size for path in checkpoint.rglob("*") if path.is_file())
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"trial": trial, "pruned_at": time.strftime("%FT%T"), "bytes": size, "files": files}) + "\n")
        shutil.rmtree(checkpoint)
        record(f"pruned disposable checkpoint for {trial}: {size} bytes")


def update_analysis() -> None:
    if ANALYZER.exists():
        subprocess.run(["./external/miniconda3/envs/Mem-TV2/bin/python", str(ANALYZER)], check=True)


def main() -> None:
    state = json.loads(STATE.read_text()) if STATE.exists() else {"completed": [], "running": None}
    prune_completed_checkpoints(state)
    running = state.get("running")
    if running:
        run = ROOT / "runs" / running
        if (run / "__SUCCESS__").exists():
            state["completed"].append(running); state["running"] = None; state.pop("pid", None)
            STATE.write_text(json.dumps(state, indent=2) + "\n"); update_analysis(); record(f"completed {running}")
        elif "Traceback" in (run / "logs/train.log").read_text(encoding="utf-8", errors="ignore") if (run / "logs/train.log").exists() else False:
            record(f"failed {running}; retaining state for diagnosis"); return
        elif Path(f"/proc/{state.get('pid', -1)}").exists(): record(f"running {running}"); return
        else: record(f"launcher ended without success: {running}"); return
    for h, seed in MATRIX:
        trial = name(h, seed)
        if trial not in state["completed"]:
            if not idle(): record("GPUs occupied; waiting"); return
            process = subprocess.Popen([str(RUNNER), h, str(seed)], start_new_session=True)
            state["running"], state["pid"] = trial, process.pid
            STATE.write_text(json.dumps(state, indent=2) + "\n"); record(f"launched {trial} pid={process.pid}"); return
    record("screen matrix complete")


if __name__ == "__main__": main()

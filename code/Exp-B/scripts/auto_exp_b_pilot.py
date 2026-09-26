#!/usr/bin/env python3
"""Non-destructive controller for the time-boxed Exp-B pilot."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path('./Code/Exp-B')
PILOT = ROOT / 'pilot'
RUNNER = ROOT / 'scripts' / 'run_exp_b_pilot.sh'
STATE = PILOT / 'state.json'
LOG = PILOT / 'autopilot.log'
MATRIX = [('0.0', 13), ('0.5', 13), ('1.0', 13)]


def record(message: str) -> None:
    line = f'[{time.strftime("%F %T")}] {message}'
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as handle:
        handle.write(line + '\n')


def gpu_idle() -> bool:
    output = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
        text=True,
    ).strip()
    return not output or all(int(row.split(',')[1].strip()) <= 1024 for row in output.splitlines())


def trial_name(h: str, seed: int) -> str:
    return f'pilot_q3x3x4_h{h.replace(".", "p")}_seed{seed}'


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding='utf-8'))
    return {'completed': [], 'running': None}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def main() -> None:
    state = load_state()
    running = state.get('running')
    if running:
        run_dir = PILOT / 'runs' / running
        if (run_dir / '__SUCCESS__').exists():
            state['completed'].append(running)
            state['running'] = None
            state.pop('pid', None)
            save_state(state)
            record(f'completed {running}')
        elif not Path(f'/proc/{state.get("pid", -1)}').exists():
            record(f'pilot launcher ended without success: {running}')
            # Preserve the failed state for diagnosis; do not overwrite logs by retrying
            # the same condition automatically.
            return
        else:
            record(f'pilot still running: {running}')
            return
    for h, seed in MATRIX:
        trial = trial_name(h, seed)
        if trial not in state['completed']:
            if not gpu_idle():
                record('GPUs busy; waiting')
                return
            process = subprocess.Popen([str(RUNNER), h, str(seed)], start_new_session=True)
            state.update(running=trial, pid=process.pid)
            save_state(state)
            record(f'launched {trial}, pid={process.pid}')
            return
    record('pilot matrix complete')


if __name__ == '__main__':
    main()

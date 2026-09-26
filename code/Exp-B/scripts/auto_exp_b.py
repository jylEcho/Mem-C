#!/usr/bin/env python3
"""Non-destructive scheduler for the isolated Exp-B experiment matrix."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path('./Code/Exp-B')
RUNNER = ROOT / 'scripts' / 'run_exp_b_trial.sh'
STATE = ROOT / 'exp_b_state.json'
LOG = ROOT / 'logs' / 'autopilot.log'
MATRIX = [(h, seed) for h in ('0.0', '0.25', '0.5', '0.75', '1.0') for seed in (13, 29, 47)]


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
    if not output:
        return True
    for row in output.splitlines():
        fields = [item.strip() for item in row.split(',')]
        if len(fields) == 2 and int(fields[1]) > 1024:
            return False
    return True


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding='utf-8'))
    return {'completed': [], 'running': None}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def name(h: str, seed: int) -> str:
    return f'expb_q3x3x4_h{h.replace(".", "p")}_seed{seed}'


def main() -> None:
    state = load_state()
    if state['running']:
        run_dir = ROOT / 'runs' / state['running']
        if (run_dir / '__SUCCESS__').exists():
            state['completed'].append(state['running'])
            state['running'] = None
            save_state(state)
            record('trial completed')
        elif (run_dir / 'logs' / 'train.log').exists() and 'Traceback' in (run_dir / 'logs' / 'train.log').read_text(encoding='utf-8', errors='ignore'):
            record(f'trial failed: {state["running"]}')
            return
        elif not Path(f'/proc/{state.get("pid", -1)}').exists():
            record(f'stale launcher state cleared: {state["running"]}')
            state['running'] = None
            state.pop('pid', None)
            save_state(state)
        else:
            record(f'trial still running: {state["running"]}')
            return
    for h, seed in MATRIX:
        trial = name(h, seed)
        if trial not in state['completed']:
            if not gpu_idle():
                record('GPUs busy; waiting')
                return
            process = subprocess.Popen([str(RUNNER), h, str(seed)], start_new_session=True)
            state['running'] = trial
            state['pid'] = process.pid
            save_state(state)
            record(f'launched {trial} pid={process.pid}')
            return
    record('Exp-B matrix complete')


if __name__ == '__main__':
    main()

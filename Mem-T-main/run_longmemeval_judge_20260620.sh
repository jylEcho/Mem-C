#!/usr/bin/env bash
set -euo pipefail

ROOT=./Mem-T-main
PY=./bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
INPUT="$ROOT/traj/longmemeval_oracle_k16_merged_20260612.jsonl"
OUTPUT="$ROOT/traj/longmemeval_oracle_judged_${STAMP}.jsonl"
LOG="$ROOT/logs/longmemeval_oracle_judge_${STAMP}.out"

cd "$ROOT"

if [[ ! -f "$INPUT" ]]; then
  echo "missing input: $INPUT" >&2
  exit 1
fi

nohup bash -lc "
  cd '$ROOT' && '$PY' - <<'PY'
import runpy
import sys
import types

fake_vllm = types.ModuleType('vllm')
class _Dummy: pass
fake_vllm.LLM = _Dummy
fake_vllm.SamplingParams = _Dummy
sys.modules['vllm'] = fake_vllm

sys.argv = [
    'llm_judge.py',
    '--input', '$INPUT',
    '--output', '$OUTPUT',
    '--benchmark', 'longmemeval',
    '--model', 'gpt-5-mini-2025-08-07',
    '--workers', '8',
]
runpy.run_module('llm_judge', run_name='__main__')
PY
" >"$LOG" 2>&1 &

echo "pid=$! log=$LOG output=$OUTPUT"

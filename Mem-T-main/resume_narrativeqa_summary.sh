#!/usr/bin/env bash
set -euo pipefail

ROOT=./Mem-T-main
PY=./bin/python
STAMP=$(date +%Y%m%d_%H%M%S)

cd "$ROOT"
mkdir -p logs

declare -a STARTS=(372 344 423 443)

echo "resume_stamp=$STAMP"

for shard in 0 1 2 3; do
  gpu=$((shard + 4))
  start="${STARTS[$shard]}"
  out="logs/bg_narrativeqa_summary_resume_shard${shard}of4_start${start}_${STAMP}.out"
  nohup bash -lc "cd '$ROOT' && CUDA_VISIBLE_DEVICES=$gpu '$PY' run_benchmark_local.py --data_name narrativeqa --narrativeqa_source summary --device cuda:0 --num_shards 4 --shard_id $shard --start_index $start > '$out' 2>&1" >/dev/null 2>&1 &
  echo "narrativeqa_summary_resume shard=$shard gpu=$gpu start=$start pid=$! log=$out"
done

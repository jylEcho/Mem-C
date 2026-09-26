#!/usr/bin/env bash
set -euo pipefail

ROOT=./Mem-T-main
PY=./bin/python
STAMP=$(date +%Y%m%d_%H%M%S)

cd "$ROOT"
mkdir -p logs

echo "launch_stamp=$STAMP"

for shard in 0 1 2 3; do
  gpu=$shard
  out="logs/bg_hotpotqa_shard${shard}of4_${STAMP}.out"
  nohup bash -lc "cd '$ROOT' && CUDA_VISIBLE_DEVICES=$gpu '$PY' run_benchmark_local.py --data_name hotpotqa --device cuda:0 --num_shards 4 --shard_id $shard > '$out' 2>&1" >/dev/null 2>&1 &
  echo "hotpotqa shard=$shard gpu=$gpu pid=$! log=$out"
done

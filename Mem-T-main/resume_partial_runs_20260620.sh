#!/usr/bin/env bash
set -euo pipefail

ROOT=./Mem-T-main
PY=./bin/python
STAMP=$(date +%Y%m%d_%H%M%S)

cd "$ROOT"
mkdir -p logs

# Current finished counts per NarrativeQA shard:
# shard0 = 372 + 824 = 1196
# shard1 = 344 + 670 = 1014
# shard2 = 423 + 788 = 1211
# shard3 = 443 + 736 = 1179
declare -a NQ_STARTS=(1196 1014 1211 1179)

# Current finished counts per HotpotQA shard.
declare -a HP_STARTS=(1 1 0 2)

echo "resume_stamp=$STAMP"

for shard in 0 1 2 3; do
  gpu=$((shard + 4))
  start="${NQ_STARTS[$shard]}"
  out="logs/bg_narrativeqa_summary_resume2_shard${shard}of4_start${start}_${STAMP}.out"
  nohup bash -lc "cd '$ROOT' && CUDA_VISIBLE_DEVICES=$gpu '$PY' run_benchmark_local.py --data_name narrativeqa --narrativeqa_source summary --device cuda:0 --num_shards 4 --shard_id $shard --start_index $start > '$out' 2>&1" >/dev/null 2>&1 &
  echo "narrativeqa_summary_resume2 shard=$shard gpu=$gpu start=$start pid=$! log=$out"
done

for shard in 0 1 2 3; do
  gpu=$shard
  start="${HP_STARTS[$shard]}"
  out="logs/bg_hotpotqa_resume_shard${shard}of4_start${start}_${STAMP}.out"
  nohup bash -lc "cd '$ROOT' && CUDA_VISIBLE_DEVICES=$gpu '$PY' run_benchmark_local.py --data_name hotpotqa --device cuda:0 --num_shards 4 --shard_id $shard --start_index $start > '$out' 2>&1" >/dev/null 2>&1 &
  echo "hotpotqa_resume shard=$shard gpu=$gpu start=$start pid=$! log=$out"
done

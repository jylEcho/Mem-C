#!/usr/bin/env bash
set -euo pipefail
cd ./Mem-T-main
pkill -f 'run_benchmark_local.py --data_name locomo --limit 1 --device cuda:0' || true
stamp=$(date +%Y%m%d_%H%M%S)
master="logs/locomo_ideav4_sharded8_${stamp}.master.log"
mkdir -p logs pids
{
  echo "[$(date '+%F %T')] launch ideav4 sharded8"
  for gpu in 0 1 2 3 4 5 6 7; do
    log="logs/locomo_local_ideav4_shard${gpu}of8_gpu${gpu}_${stamp}.out"
    pidfile="pids/locomo_local_ideav4_shard${gpu}of8_gpu${gpu}_${stamp}.pid"
    echo "[$(date '+%F %T')] start shard=${gpu} gpu=${gpu} log=${log} pidfile=${pidfile}"
    nohup bash -lc "cd ./Mem-T-main && CUDA_VISIBLE_DEVICES=${gpu} ./bin/python run_benchmark_local.py --data_name locomo --device cuda:0 --shard_id ${gpu} --num_shards 8" > "$log" 2>&1 &
    echo $! > "$pidfile"
    sleep 1
  done
  echo "MASTER_LOG=$master"
  echo "STAMP=$stamp"
} | tee "$master"

#!/usr/bin/env bash
set -euo pipefail

# All experiment outputs are isolated under Code/Exp-B/reporting-v1.
ROOT=./external/Mem-T
EXP_ROOT=$ROOT/Code/Exp-B/reporting-v1
WORKSPACE=$ROOT/Code/Exp-B/workspace
ENV_DIR=./external/miniconda3/envs/Mem-TV2
BASE_MODEL=$ROOT/Mem-T-main/models/Mem-T-4B
RETRIEVAL_SERVER=$ROOT/Code/ChatMem-main/reproduction/tools/memory_retrieval_server_locomo_persistent.py

H=${1:?usage: run_trial.sh COVERAGE_TARGET SEED}
SEED=${2:?usage: run_trial.sh COVERAGE_TARGET SEED}
TAG=${H/./p}
NAME=screen_q3x3x4_h${TAG}_seed${SEED}
RUN_DIR=$EXP_ROOT/runs/$NAME
LOG_DIR=$RUN_DIR/logs
PORT=$((20000 + SEED))

mkdir -p "$LOG_DIR" "$RUN_DIR/checkpoints" "$RUN_DIR/trajectories"
source ./external/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_DIR"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 WG_BACKEND=ray VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONHASHSEED=$SEED HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export EXP_B_TRAJ_DIR=$RUN_DIR/trajectories RAY_ADDRESS=${EXP_B_RAY_ADDRESS:-SERVICE_ADDRESS_NOT_CONFIGURED}
unset http_proxy https_proxy

if ss -ltn "sport = :$PORT" | grep -q LISTEN; then echo "retrieval port in use" >&2; exit 21; fi
cleanup() { [[ -n "${RETR_PID:-}" ]] && kill -0 "$RETR_PID" 2>/dev/null && kill "$RETR_PID" || true; }
trap cleanup EXIT INT TERM

# The shim chooses the complete local BGE-M3 checkpoint; shared Ray is connect-only.
PYTHONPATH=$ROOT/Code/Exp-B/pilot/runtime:$ROOT/Code/ChatMem-main MEMT_RETRIEVAL_PORT=$PORT \
  "$ENV_DIR/bin/python" "$RETRIEVAL_SERVER" > "$LOG_DIR/retrieval.log" 2>&1 &
RETR_PID=$!
sleep 8
kill -0 "$RETR_PID"
ray status --address="$RAY_ADDRESS" > "$LOG_DIR/ray-status-before.log" 2>&1

cat > "$RUN_DIR/manifest.json" <<EOF
{"study":"Exp-B early-training screening","coverage_target":$H,"seed":$SEED,"q":{"m":3,"n":3,"l":4,"k":1},"train_rows":16,"valid_rows":16,"rl_steps":1,"pretrain_validation":false}
EOF

cd "$WORKSPACE"
python -m verl.trainer.main_ppo_memory_ts \
  data.train_files="$EXP_ROOT/data/train.parquet" data.val_files="$EXP_ROOT/data/valid.parquet" \
  data.train_batch_size=16 data.val_batch_size=16 data.max_prompt_length=40960 data.max_response_length=1024 \
  data.max_start_length=15360 data.max_obs_length=20480 data.shuffle_train_dataloader=false algorithm.adv_estimator=memory_tree \
  actor_rollout_ref.model.path="$BASE_MODEL" actor_rollout_ref.model.enable_gradient_checkpointing=true actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.policy_loss=grpo actor_rollout_ref.actor.optim.lr=5e-6 actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
  actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.ppo_mini_batch_size=8 actor_rollout_ref.actor.ppo_micro_batch_size=8 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true actor_rollout_ref.actor.fsdp_config.grad_offload=true actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=8 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 +actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.tree_search=true actor_rollout_ref.rollout.ts_m=3 actor_rollout_ref.rollout.ts_n=3 actor_rollout_ref.rollout.ts_l=4 actor_rollout_ref.rollout.ts_k=1 \
  actor_rollout_ref.rollout.reward_mode=memory actor_rollout_ref.rollout.expand_mode=coverage actor_rollout_ref.rollout.coverage_target="$H" actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  +actor_rollout_ref.rollout.disable_log_stats=true +actor_rollout_ref.rollout.enable_chunked_prefill=true actor_rollout_ref.ref.log_prob_micro_batch_size=8 actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl algorithm.no_think_rl=false algorithm.use_kl_in_reward=false \
  actor_rollout_ref.rollout.n_agent=1 actor_rollout_ref.rollout.temperature=1 actor_rollout_ref.actor.state_masking=true trainer.logger="['console']" \
  trainer.n_gpus_per_node=8 trainer.nnodes=1 trainer.seed="$SEED" +trainer.val_before_train=false trainer.save_freq=0 trainer.test_freq=999999 \
  trainer.total_epochs=1 trainer.total_training_steps=1 trainer.default_hdfs_dir=null trainer.default_local_dir="$RUN_DIR/checkpoints" \
  trainer.project_name=MemTExpBScreen trainer.experiment_name="$NAME" reward_model.structure_format_score=0.2 reward_model.final_format_score=0.1 \
  reward_model.retrieval_score=0 +reward_model.evidence_alpha=0.5 +reward_model.format_score=0.0 max_turns=6 retriever.url="API_ENDPOINT_NOT_CONFIGURED" retriever.topk=5 \
  2>&1 | tee "$LOG_DIR/train.log"
touch "$RUN_DIR/__SUCCESS__"

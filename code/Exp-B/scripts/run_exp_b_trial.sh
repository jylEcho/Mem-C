#!/usr/bin/env bash
set -euo pipefail

# Isolated Exp-B launcher. It never writes into ChatMem-main or Mem-T-main.
ROOT=./external/Mem-T
EXP_ROOT=$ROOT/Code/Exp-B
WORKSPACE=$EXP_ROOT/workspace
ENV_DIR=./external/miniconda3/envs/Mem-TV2
DATA_DIR=$ROOT/Code/ChatMem-main/reproduction/runs/20260822_full_reproduction/paper_protocol/locomo_v1/tree_grpo
RETRIEVAL_SERVER=$ROOT/Code/ChatMem-main/reproduction/tools/memory_retrieval_server_locomo_persistent.py
BASE_MODEL=$ROOT/Mem-T-main/models/Mem-T-4B

H=${1:?usage: run_exp_b_trial.sh COVERAGE_TARGET SEED}
SEED=${2:?usage: run_exp_b_trial.sh COVERAGE_TARGET SEED}
H_TAG=${H/./p}
EXP_NAME=expb_q3x3x4_h${H_TAG}_seed${SEED}
RUN_DIR=$EXP_ROOT/runs/$EXP_NAME
LOG_DIR=$RUN_DIR/logs
CKPT_DIR=$RUN_DIR/checkpoints
TRAJ_DIR=$RUN_DIR/trajectories
PORT_RETR=$((19000 + SEED))
RAY_PORT=$((6410 + SEED))

mkdir -p "$LOG_DIR" "$CKPT_DIR" "$TRAJ_DIR" "$RUN_DIR/ray"
source ./external/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_DIR"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WG_BACKEND=ray
export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_TMPDIR=$RUN_DIR/ray
export EXP_B_TRAJ_DIR=$TRAJ_DIR
export PYTHONHASHSEED=$SEED
export no_proxy=localhost,127.0.0.1
unset http_proxy https_proxy

if ss -ltn "sport = :$PORT_RETR" | grep -q LISTEN; then
  echo "Retrieval port $PORT_RETR is already in use." >&2
  exit 21
fi

cleanup() {
  if [[ -n "${RETR_PID:-}" ]] && kill -0 "$RETR_PID" 2>/dev/null; then
    kill "$RETR_PID" 2>/dev/null || true
  fi
  if [[ "${OWN_RAY:-0}" == 1 ]]; then
    ray stop --force > "$LOG_DIR/ray-stop.log" 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

PYTHONPATH=$ROOT/Code/ChatMem-main MEMT_RETRIEVAL_PORT=$PORT_RETR \
  "$ENV_DIR/bin/python" "$RETRIEVAL_SERVER" > "$LOG_DIR/retrieval.log" 2>&1 &
RETR_PID=$!
sleep 8
kill -0 "$RETR_PID"

if pgrep -af 'raylet|gcs_server' >/dev/null; then
  # This cluster predates Exp-B. Connect only when its GPU resources are idle;
  # never stop or reconfigure a shared cluster.
  RAY_ADDRESS=${EXP_B_RAY_ADDRESS:-SERVICE_ADDRESS_NOT_CONFIGURED}
  ray status --address="$RAY_ADDRESS" > "$LOG_DIR/ray-status-before.log" 2>&1
  export RAY_ADDRESS
  OWN_RAY=0
else
  ray start --head --port="$RAY_PORT" --temp-dir="$RAY_TMPDIR" --disable-usage-stats --num-gpus=8 \
    > "$LOG_DIR/ray-start.log" 2>&1
  OWN_RAY=1
fi

cat > "$RUN_DIR/manifest.json" <<EOF
{"experiment":"$EXP_NAME","coverage_target":$H,"seed":$SEED,"q":{"m":3,"n":3,"l":4,"k":1},"model":"$BASE_MODEL","data":"$DATA_DIR"}
EOF

cd "$WORKSPACE"
python -m verl.trainer.main_ppo_memory_ts \
  data.train_files="$DATA_DIR/train.parquet" \
  data.val_files="$DATA_DIR/valid.parquet" \
  data.train_batch_size=32 data.val_batch_size=32 \
  data.max_prompt_length=40960 data.max_response_length=1024 \
  data.max_start_length=15360 data.max_obs_length=20480 \
  data.shuffle_train_dataloader=true algorithm.adv_estimator=memory_tree \
  actor_rollout_ref.model.path="$BASE_MODEL" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=true actor_rollout_ref.actor.policy_loss=grpo \
  actor_rollout_ref.actor.optim.lr=5e-6 actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
  actor_rollout_ref.actor.use_kl_loss=true actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size=8 actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.grad_offload=true actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=8 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 +actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.tree_search=true \
  actor_rollout_ref.rollout.ts_m=3 actor_rollout_ref.rollout.ts_n=3 actor_rollout_ref.rollout.ts_l=4 actor_rollout_ref.rollout.ts_k=1 \
  actor_rollout_ref.rollout.reward_mode=memory actor_rollout_ref.rollout.expand_mode=coverage \
  actor_rollout_ref.rollout.coverage_target="$H" actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  +actor_rollout_ref.rollout.disable_log_stats=true +actor_rollout_ref.rollout.enable_chunked_prefill=true \
  actor_rollout_ref.ref.log_prob_micro_batch_size=8 actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  algorithm.no_think_rl=false algorithm.use_kl_in_reward=false actor_rollout_ref.rollout.n_agent=1 \
  actor_rollout_ref.rollout.temperature=1 actor_rollout_ref.actor.state_masking=true \
  trainer.logger="['console']" trainer.n_gpus_per_node=8 trainer.nnodes=1 trainer.seed="$SEED" \
  trainer.save_freq=50 trainer.test_freq=50 trainer.total_epochs=20 trainer.total_training_steps=200 \
  trainer.default_hdfs_dir=null trainer.default_local_dir="$CKPT_DIR" trainer.project_name=MemTExpB \
  trainer.experiment_name="$EXP_NAME" reward_model.structure_format_score=0.2 \
  reward_model.final_format_score=0.1 reward_model.retrieval_score=0 +reward_model.evidence_alpha=0.5 \
  +reward_model.format_score=0.0 max_turns=6 retriever.url="API_ENDPOINT_NOT_CONFIGURED" retriever.topk=5 \
  2>&1 | tee "$LOG_DIR/train.log"

touch "$RUN_DIR/__SUCCESS__"

#!/usr/bin/env bash
set -euo pipefail

# ==============================
# Conda
# ==============================

source ./external/miniconda3/etc/profile.d/conda.sh
conda activate ./external/miniconda3/envs/Mem-TV2


# ==============================
# Args
# ==============================

: "${EXP_NAME:?Please set EXP_NAME, e.g. EXP_NAME=locomo_rl_paper_8gpu_m1_n1_l1_k1 bash merge_model.sh}"


# ==============================
# Paths
# ==============================

ROOT=./Code/ChatMem-main
TREE=$ROOT/Tree-GRPO

CKPT=$ROOT/reproduction/runs/20260822_full_reproduction/checkpoints/$EXP_NAME/actor/global_step_50

HF_CONFIG=$CKPT/huggingface

OUTPUT=./Code/Exp-A/merged_models/$EXP_NAME


# ==============================
# Check
# ==============================

if [[ ! -d "$CKPT" ]]; then
    echo "ERROR: checkpoint not found:"
    echo "$CKPT"
    exit 1
fi

if [[ ! -f "$HF_CONFIG/config.json" ]]; then
    echo "ERROR: HuggingFace config not found:"
    echo "$HF_CONFIG/config.json"
    exit 1
fi

if [[ ! -f "$TREE/scripts/merge_ckpt/model_merger.py" ]]; then
    echo "ERROR: model merger not found:"
    echo "$TREE/scripts/merge_ckpt/model_merger.py"
    exit 1
fi


# ==============================
# Merge
# ==============================

cd "$TREE"

export PYTHONPATH="$TREE:${PYTHONPATH:-}"

mkdir -p "$(dirname "$OUTPUT")"

echo
echo "======================================"
echo "Experiment : $EXP_NAME"
echo "Python     : $(which python)"
echo "VERL       : $(python -c 'import verl; print(verl.__file__)')"
echo "Checkpoint : $CKPT"
echo "HF config  : $HF_CONFIG"
echo "Output     : $OUTPUT"
echo "======================================"

python scripts/merge_ckpt/model_merger.py merge \
    --backend fsdp \
    --local_dir "$CKPT" \
    --hf_model_path "$HF_CONFIG" \
    --target_dir "$OUTPUT"

echo
echo "======================================"
echo "Merge completed"
echo "Output: $OUTPUT"
echo "======================================"
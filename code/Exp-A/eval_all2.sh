#!/usr/bin/env bash
set -euo pipefail


# ============================================================
# Config
# ============================================================

SCRIPT=./external/Mem-T-main/run_benchmark_local_exp_a.py

# ------------------------------------------------------------
# GPUs
# 只使用物理 GPU 1 和 GPU 3
# ------------------------------------------------------------
GPU_IDS=(1 3)
NUM_GPUS=${#GPU_IDS[@]}

MODEL_ROOT=./external/Exp-A/merged_models


RUN_NAMES=(
    "locomo_rl_paper_8gpu_m1_n1_l1_k1"
    "locomo_rl_paper_8gpu_m2_n1_l1_k1"
    "locomo_rl_paper_8gpu_m3_n1_l1_k1"
    "locomo_rl_paper_8gpu_m4_n1_l1_k1"
    "locomo_rl_paper_8gpu_m5_n1_l1_k1"
    "locomo_rl_paper_8gpu_m4_n2_l1_k1"
    "locomo_rl_paper_8gpu_m4_n3_l1_k1"
    "locomo_rl_paper_8gpu_m4_n4_l1_k1"
)


DATASETS=(
    # "locomo"
    "hotpotqa"
    # "longmemeval"
    # "narrativeqa"
)


LONGMEMEVAL_VARIANT=${LONGMEMEVAL_VARIANT:-longmemeval_s_cleaned}
NARRATIVEQA_SOURCE=${NARRATIVEQA_SOURCE:-summary}


# ============================================================
# Print GPU config
# ============================================================

echo "============================================================"
echo "GPU config"
echo "============================================================"
echo "Physical GPUs : ${GPU_IDS[*]}"
echo "Num GPUs      : ${NUM_GPUS}"
echo "============================================================"


# ============================================================
# Run one dataset
# ============================================================

run_dataset() {

    local RUN_NAME=$1
    local MODEL_PATH=$2
    local DATA_NAME=$3

    local CONSOLE_LOG_ROOT="./external/Exp-A/logs/${RUN_NAME}/console/${DATA_NAME}"

    mkdir -p "${CONSOLE_LOG_ROOT}"

    echo
    echo "============================================================"
    echo "Run name : ${RUN_NAME}"
    echo "Dataset  : ${DATA_NAME}"
    echo "Model    : ${MODEL_PATH}"
    echo "GPUs     : ${GPU_IDS[*]}"
    echo "Shards   : ${NUM_GPUS}"
    echo "============================================================"


    PIDS=()


    # ========================================================
    # Launch one process per GPU
    # ========================================================

    for SHARD_ID in $(seq 0 $((NUM_GPUS - 1))); do

        # SHARD_ID 是逻辑编号：0, 1, ...
        # GPU_ID 是实际物理 GPU：1, 3, ...
        GPU_ID=${GPU_IDS[$SHARD_ID]}

        CONSOLE_LOG="${CONSOLE_LOG_ROOT}/shard_${SHARD_ID}.log"

        EXTRA_ARGS=()


        # ----------------------------------------------------
        # Dataset-specific args
        # ----------------------------------------------------

        if [[ "${DATA_NAME}" == "longmemeval" ]]; then
            EXTRA_ARGS+=(
                --longmemeval_variant "${LONGMEMEVAL_VARIANT}"
            )
        fi

        if [[ "${DATA_NAME}" == "narrativeqa" ]]; then
            EXTRA_ARGS+=(
                --narrativeqa_source "${NARRATIVEQA_SOURCE}"
            )
        fi


        echo "[Launch] ${RUN_NAME} | ${DATA_NAME} | shard=${SHARD_ID}/${NUM_GPUS} | physical GPU=${GPU_ID}"


        # ----------------------------------------------------
        # Each process only sees one physical GPU.
        #
        # Example:
        #
        # shard 0:
        # CUDA_VISIBLE_DEVICES=1
        # cuda:0 -> physical GPU 1
        #
        # shard 1:
        # CUDA_VISIBLE_DEVICES=3
        # cuda:0 -> physical GPU 3
        # ----------------------------------------------------

        CUDA_VISIBLE_DEVICES=${GPU_ID} \
        python "${SCRIPT}" \
            --data_name "${DATA_NAME}" \
            --model_path "${MODEL_PATH}" \
            --run_name "${RUN_NAME}" \
            --device "cuda:0" \
            --shard_id "${SHARD_ID}" \
            --num_shards "${NUM_GPUS}" \
            "${EXTRA_ARGS[@]}" \
            > "${CONSOLE_LOG}" 2>&1 &


        PIDS+=($!)

    done


    # ========================================================
    # Wait all shards
    # ========================================================

    FAILED=0

    for i in "${!PIDS[@]}"; do

        PID=${PIDS[$i]}
        GPU_ID=${GPU_IDS[$i]}

        if wait "${PID}"; then

            echo "[Done] ${RUN_NAME} | ${DATA_NAME} | shard=${i} | GPU=${GPU_ID}"

        else

            echo "[FAILED] ${RUN_NAME} | ${DATA_NAME} | shard=${i} | GPU=${GPU_ID}"
            FAILED=1

        fi

    done


    # ========================================================
    # Check failures
    # ========================================================

    if [[ ${FAILED} -ne 0 ]]; then

        echo
        echo "============================================================"
        echo "ERROR"
        echo "============================================================"
        echo "Run     : ${RUN_NAME}"
        echo "Dataset : ${DATA_NAME}"
        echo "Logs    : ${CONSOLE_LOG_ROOT}"
        echo "============================================================"

        exit 1

    fi


    echo
    echo "[Finished] ${RUN_NAME} | ${DATA_NAME}"
}


# ============================================================
# Main
# ============================================================

for RUN_NAME in "${RUN_NAMES[@]}"; do

    MODEL_PATH="${MODEL_ROOT}/${RUN_NAME}"

    echo
    echo
    echo "############################################################"
    echo "Start experiment"
    echo "############################################################"
    echo "RUN_NAME   : ${RUN_NAME}"
    echo "MODEL_PATH : ${MODEL_PATH}"
    echo "GPUs       : ${GPU_IDS[*]}"
    echo "############################################################"


    # ========================================================
    # Check model
    # ========================================================

    if [[ ! -d "${MODEL_PATH}" ]]; then

        echo "ERROR: model directory does not exist:"
        echo "${MODEL_PATH}"

        exit 1

    fi


    # ========================================================
    # Datasets sequentially
    # ========================================================

    for DATA_NAME in "${DATASETS[@]}"; do

        run_dataset \
            "${RUN_NAME}" \
            "${MODEL_PATH}" \
            "${DATA_NAME}"

    done


    echo
    echo "############################################################"
    echo "Finished experiment: ${RUN_NAME}"
    echo "############################################################"

done


echo
echo
echo "============================================================"
echo "All models and datasets finished."
echo "============================================================"
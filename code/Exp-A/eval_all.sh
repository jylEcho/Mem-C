#!/usr/bin/env bash
set -euo pipefail


# ============================================================
# Config
# ============================================================

SCRIPT=./Mem-T-main/run_benchmark_local_exp_a.py

NUM_GPUS=${NUM_GPUS:-8}

MODEL_ROOT=./Code/Exp-A/merged_models


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
# Run one dataset
# ============================================================

run_dataset() {

    local RUN_NAME=$1
    local MODEL_PATH=$2
    local DATA_NAME=$3

    local CONSOLE_LOG_ROOT="./Code/Exp-A/logs/${RUN_NAME}/console/${DATA_NAME}"

    mkdir -p "${CONSOLE_LOG_ROOT}"

    echo
    echo "============================================================"
    echo "Run name : ${RUN_NAME}"
    echo "Dataset  : ${DATA_NAME}"
    echo "Model    : ${MODEL_PATH}"
    echo "============================================================"

    PIDS=()

    for SHARD_ID in $(seq 0 $((NUM_GPUS - 1))); do

        GPU_ID=${SHARD_ID}

        CONSOLE_LOG="${CONSOLE_LOG_ROOT}/shard_${SHARD_ID}.log"

        EXTRA_ARGS=()

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


        echo "[Launch] ${RUN_NAME} | ${DATA_NAME} | shard=${SHARD_ID} | GPU=${GPU_ID}"

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

        if wait "${PID}"; then
            echo "[Done] ${RUN_NAME} | ${DATA_NAME} | shard=${i}"
        else
            echo "[FAILED] ${RUN_NAME} | ${DATA_NAME} | shard=${i}"
            FAILED=1
        fi

    done


    if [[ ${FAILED} -ne 0 ]]; then
        echo
        echo "ERROR:"
        echo "Run     : ${RUN_NAME}"
        echo "Dataset : ${DATA_NAME}"
        echo "Logs    : ${CONSOLE_LOG_ROOT}"
        exit 1
    fi


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
    echo "############################################################"


    # Check model
    if [[ ! -d "${MODEL_PATH}" ]]; then
        echo "ERROR: model directory does not exist:"
        echo "${MODEL_PATH}"
        exit 1
    fi


    # --------------------------------------------------------
    # Four datasets sequentially
    # --------------------------------------------------------

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
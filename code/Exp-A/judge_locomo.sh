#!/bin/bash

set -e

TRAJ_ROOT="./Code/Exp-A/traj_0922"

# 评测脚本
EVAL_SCRIPT="./Mem-T-main/llm_judge.py"

EXPS=(
    "locomo_rl_paper_8gpu_m1_n1_l1_k1"
    "locomo_rl_paper_8gpu_m2_n1_l1_k1"
    "locomo_rl_paper_8gpu_m3_n1_l1_k1"
    "locomo_rl_paper_8gpu_m4_n1_l1_k1"
    "locomo_rl_paper_8gpu_m5_n1_l1_k1"
    "locomo_rl_paper_8gpu_m4_n2_l1_k1"
    "locomo_rl_paper_8gpu_m4_n3_l1_k1"
    "locomo_rl_paper_8gpu_m4_n5_l1_k1"
    "locomo_rl_paper_8gpu_m4_n4_l1_k1"
)

for EXP_NAME in "${EXPS[@]}"; do
    EXP_DIR="${TRAJ_ROOT}/${EXP_NAME}"

    echo
    echo "============================================================"
    echo "Experiment: ${EXP_NAME}"
    echo "============================================================"

    if [ ! -d "${EXP_DIR}" ]; then
        echo "[SKIP] Directory not found: ${EXP_DIR}"
        continue
    fi

    # 找最新一次运行产生的 qa_trajectories.jsonl
    INPUT=$(find "${EXP_DIR}" \
        -type f \
        -path "*/hotpotqa_local_*/qa_trajectories.jsonl" \
        | sort \
        | tail -n 1)

    if [ -z "${INPUT}" ]; then
        echo "[SKIP] qa_trajectories.jsonl not found"
        continue
    fi

    RUN_DIR=$(dirname "${INPUT}")
    OUTPUT="${RUN_DIR}/qa_trajectories_judged.jsonl"

    echo "Input : ${INPUT}"
    echo "Output: ${OUTPUT}"

    # --------------------------------------------------------
    # 如果已经评测过，则跳过
    # -s: 文件存在且大小 > 0
    # --------------------------------------------------------
    if [ -s "${OUTPUT}" ]; then
        echo "[SKIP] Judged result already exists."
        continue
    fi

    python "${EVAL_SCRIPT}" \
        --input "${INPUT}" \
        --output "${OUTPUT}" \
        --model "gpt-5.6-sol" \
        --benchmark locomo \
        --workers 8

    echo "[DONE] ${EXP_NAME}"
done

echo
echo "============================================================"
echo "All experiments finished."
echo "============================================================"
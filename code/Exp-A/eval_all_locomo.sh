#!/bin/bash

PYTHON_SCRIPT="./Mem-T-main/run_locomo_local_exp_a.py"

RUN_NAMES=(
    # "locomo_rl_paper_8gpu_m1_n1_l1_k1"
    # "locomo_rl_paper_8gpu_m2_n1_l1_k1"
    # "locomo_rl_paper_8gpu_m3_n1_l1_k1"
    # "locomo_rl_paper_8gpu_m4_n1_l1_k1"
    # "locomo_rl_paper_8gpu_m5_n1_l1_k1"
    # "locomo_rl_paper_8gpu_m4_n2_l1_k1"
    # "locomo_rl_paper_8gpu_m4_n3_l1_k1"
    "locomo_rl_paper_8gpu_m4_n5_l1_k1"
    # "locomo_rl_paper_8gpu_m4_n4_l1_k1"
)

WORKERS=8
LIMIT=0

for RUN_NAME in "${RUN_NAMES[@]}"; do
    echo "======================================================"
    echo "Evaluating: ${RUN_NAME}"
    echo "======================================================"

    python "${PYTHON_SCRIPT}" \
        --name "${RUN_NAME}" \
        --workers "${WORKERS}" \
        --limit "${LIMIT}"

    EXIT_CODE=$?

    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "❌ Evaluation failed: ${RUN_NAME}, exit code=${EXIT_CODE}"
    else
        echo "✅ Evaluation finished: ${RUN_NAME}"
    fi

    echo
done

echo "======================================================"
echo "All evaluations finished."
echo "======================================================"
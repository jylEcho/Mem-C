#!/usr/bin/env bash
set -euo pipefail

MERGE_SCRIPT=./Code/Exp-A/merge_model.sh

EXP_NAMES=(
    # "locomo_rl_paper_8gpu_m1_n1_l1_k1"
    # "locomo_rl_paper_8gpu_m2_n1_l1_k1"
    "locomo_rl_paper_8gpu_m3_n1_l1_k1"
    "locomo_rl_paper_8gpu_m4_n1_l1_k1"
    "locomo_rl_paper_8gpu_m5_n1_l1_k1"
    "locomo_rl_paper_8gpu_m4_n2_l1_k1"
    "locomo_rl_paper_8gpu_m4_n3_l1_k1"
    "locomo_rl_paper_8gpu_m4_n4_l1_k1"
)

for EXP_NAME in "${EXP_NAMES[@]}"; do
    echo
    echo "======================================================"
    echo "Merging: $EXP_NAME"
    echo "======================================================"

    EXP_NAME="$EXP_NAME" bash "$MERGE_SCRIPT"

    echo "Finished: $EXP_NAME"
done

echo
echo "======================================================"
echo "All models merged."
echo "======================================================"
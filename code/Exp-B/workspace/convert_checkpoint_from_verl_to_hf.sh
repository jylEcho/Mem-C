export PYTHONPATH=$(pwd):$PYTHONPATH
python scripts/merge_ckpt/model_merger.py merge \
    --backend fsdp \
    --hf_model_path ./external/guibin/yyw/ChatMem/qwen3-4b-treegrpo \
    --local_dir ./external/guibin/yyw/ChatMem/Tree-GRPO/verl_checkpoints/memory-treesearch-qwen3-4b/actor/global_step_120 \
    --target_dir ./external/guibin/yyw/ChatMem/qwen3-4b-treegrpo-2
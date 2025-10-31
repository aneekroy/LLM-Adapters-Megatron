#!/bin/bash

# Wait for process with PID 1126500 to finish
TARGET_PID=1126500
echo "⏳ Waiting for PID $TARGET_PID to complete..."
while kill -0 "$TARGET_PID" 2>/dev/null; do
  sleep 1
done
echo "✅ PID $TARGET_PID has finished. Starting runs..."

# Activate conda env if not already
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate sparsegpt

# Export WandB project
export WANDB_PROJECT=huggingface
export WANDB_MODE=offline  # or "offline" if needed

# Set common prefix for binary
TORCHRUN="$CONDA_PREFIX/bin/torchrun"

# ---------- First Run ----------
# WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
# "$TORCHRUN" --nproc_per_node=1 --master_port=3301 active_learning.py \
#   --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.33" \
#   --data_path /home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part1_of_3.json \
#   --output_dir ./trained_models/Qwen3_Sparse-Ensemble-Split-33/Qwen3-8B-Sparse-0.33-Math14k-part1-al50 \
#   --rounds 3 \
#   --init_frac 0.1 \
#   --acq_frac 0.2 \
#   --wandb_run_name Qwen3-8B-Sparse-0.33-Math14k-part1-al50

# # ---------- Second Run ----------
# WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
# "$TORCHRUN" --nproc_per_node=1 --master_port=3302 active_learning.py \
#   --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.33" \
#   --data_path /home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part2_of_3.json \
#   --output_dir ./trained_models/Qwen3_Sparse-Ensemble-Split-33/Qwen3-8B-Sparse-0.33-Math14k-part2-al50 \
#   --rounds 3 \
#   --init_frac 0.1 \
#   --acq_frac 0.2 \
#   --wandb_run_name Qwen3-8B-Sparse-0.33-Math14k-part2-al50


WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3401 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.50"  \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/split_50/math_14k_part1_of_2.json\
    --output_dir ./trained_models/Qwen3_Sparse-Ensemble-Split-50/Qwen3-8B-Sparse-0.50-Math14k-part1-rand50 \
    --rounds 1 \
    --init_frac 0.5 \
    --acq_frac 0.2 \
    --wandb_run_name Qwen3-8B-Sparse-0.50-Math14k-part1-rand50


WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3402 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.50"  \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/split_50/math_14k_part2_of_2.json \
    --output_dir ./trained_models/Qwen3_Sparse-Ensemble-Split-50/Qwen3-8B-Sparse-0.50-Math14k-part2-al50 \
    --rounds 3 \
    --init_frac 0.1 \
    --acq_frac 0.2 \
    --wandb_run_name Qwen3-8B-Sparse-0.50-Math14k-part2-al50


echo "✅ All runs completed."
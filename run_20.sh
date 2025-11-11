# #!/usr/bin/env bash
# set -euo pipefail

# # ===== GPU & allocator =====
# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=${GPU:-1}             # override with: GPU=1 ./run_all.sh
# export TORCHINDUCTOR_DISABLE_CUDAGRAPHS=1
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
# # export WANDB_MODE=offline   # uncomment if you want offline W&B

# # ===== Paths =====
# DATA_FULL="/home/aneek/LLM-Adapters/ft-training_set/math_14k.json"
# [[ -f "$DATA_FULL" ]] || { echo "[FATAL] Missing dataset: $DATA_FULL" >&2; exit 1; }

# BASE_ROOT="/home/models/nvidia-sparse"
# STEM="OpenReasoning-Nemotron-14B-Sparse"

# # Output root (kept common across runs)
# OUT_ROOT="./trained_models/${STEM}-Ensemble_fullMath14k"
# LOG_DIR="${OUT_ROOT}/logs"
# mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

# # ===== Common hyperparams =====
# BATCH=4
# MICRO=1
# EPOCHS=3
# LR=3e-5
# CUT=256
# VAL=120
# SCORE_BS=8

# ts(){ date +"%Y-%m-%d_%H-%M-%S"; }

# # ===== Sparsity grid =====
# for SP in 0.20 ; do
#   BASE_MODEL="${BASE_ROOT}/${STEM}-${SP}"
#   TOKENIZER_PATH="${BASE_MODEL}"    # sparse ckpts ship full tokenizer
#   [[ -d "${BASE_MODEL}" ]] || { echo "[FATAL] Missing base model dir: ${BASE_MODEL}" >&2; exit 1; }

#   RUN_STEM="${STEM}-${SP}"
#   echo "==================== ${RUN_STEM} ===================="

# #   # ---------- (A) FULL FINETUNE on all 14k ----------
# #   OUT_FT="${OUT_ROOT}/${RUN_STEM}-Math14k-fullFT"
# #   RUN_FT="FT-${RUN_STEM}-Math14k-full"
# #   echo "[FT] ${OUT_FT}"

# #   python /home/aneek/LLM-Adapters/finetune.py \
# #     --base_model "${BASE_MODEL}" \
# #     --data_path "${DATA_FULL}" \
# #     --output_dir "${OUT_FT}" \
# #     --batch_size ${BATCH} \
# #     --micro_batch_size ${MICRO} \
# #     --num_epochs ${EPOCHS} \
# #     --learning_rate ${LR} \
# #     --cutoff_len ${CUT} \
# #     --val_set_size ${VAL} \
# #     # --wandb_run_name "${RUN_FT}" \
# #     2>&1 | tee "${LOG_DIR}/$(ts)_fullFT_${SP}.log"

#   # ---------- (B) RAND50: one-shot 50% label (0.5) + dummy acq 0.1 ----------
#   OUT_RAND50="${OUT_ROOT}/${RUN_STEM}-Math14k-rand50"
#   RUN_RAND50="AL-rand50-${RUN_STEM}-Math14k"
#   echo "[AL-rand50] ${OUT_RAND50}"

#   python /home/aneek/LLM-Adapters/active_learning_fast.py \
#     --base_model "${BASE_MODEL}" \
#     --tokenizer_path "${TOKENIZER_PATH}" \
#     --data_path "${DATA_FULL}" \
#     --output_dir "${OUT_RAND50}" \
#     --rounds 1 \
#     --init_frac 0.5 \
#     --acq_frac 0.1 \
#     --uncertainty logppl \
#     --cutoff_len ${CUT} \
#     --scoring_batch_size ${SCORE_BS} \
#     --num_epochs ${EPOCHS} \
#     --learning_rate ${LR} \
#     --per_device_train_batch_size ${BATCH} \
#     --micro_batch_size ${MICRO} \
#     --val_set_size ${VAL} \
#     --wandb_run_name "${RUN_RAND50}" \
#     2>&1 | tee "${LOG_DIR}/$(ts)_rand50_${SP}.log"

#   # ---------- (C) AL50: ~50% via 10% init + 2x20% acquisitions ----------
#   OUT_AL50="${OUT_ROOT}/${RUN_STEM}-Math14k-al50"
#   RUN_AL50="AL-al50-${RUN_STEM}-Math14k"
#   echo "[AL-al50] ${OUT_AL50}"

#   python /home/aneek/LLM-Adapters/active_learning_fast.py \
#     --base_model "${BASE_MODEL}" \
#     --tokenizer_path "${TOKENIZER_PATH}" \
#     --data_path "${DATA_FULL}" \
#     --output_dir "${OUT_AL50}" \
#     --rounds 3 \
#     --init_frac 0.1 \
#     --acq_frac 0.2 \
#     --uncertainty logppl \
#     --cutoff_len ${CUT} \
#     --scoring_batch_size ${SCORE_BS} \
#     --num_epochs ${EPOCHS} \
#     --learning_rate ${LR} \
#     --per_device_train_batch_size ${BATCH} \
#     --micro_batch_size ${MICRO} \
#     --val_set_size ${VAL} \
#     --wandb_run_name "${RUN_AL50}" \
#     2>&1 | tee "${LOG_DIR}/$(ts)_al50_${SP}.log"

# done

# echo "[DONE] All sparsities (.20/.25/.33/.50) finished for full Math14k: ${OUT_ROOT}"


# WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 \
# "$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3201 active_learning.py \
#     --base_model "/home/models/nvidia-sparse/OpenReasoning-Nemotron-14B-Sparse-0.67" \
#     --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part1_of_3.json"   \
#     --output_dir ./trained_models/Nemotron-14B-Sparse/OpenReasoning-Nemotron-14B-Sparse-0.67-Ensemble_split_33/OpenReasoning-Nemotron-14B-Sparse-0.67-Math14k-part1-al50 \
#     --rounds 3 \
#     --init_frac 0.1 \
#     --acq_frac 0.2 

# WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 \
# "$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3201 active_learning.py \
#     --base_model "/home/models/nvidia-sparse/OpenReasoning-Nemotron-14B-Sparse-0.67" \
#     --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part2_of_3.json"   \
#     --output_dir ./trained_models/Nemotron-14B-Sparse/OpenReasoning-Nemotron-14B-Sparse-0.67-Ensemble_split_33/OpenReasoning-Nemotron-14B-Sparse-0.67-Math14k-part2-al50 \
#     --rounds 3 \
#     --init_frac 0.1 \
#     --acq_frac 0.2 

# WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 \
# "$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3201 active_learning.py \
#     --base_model "/home/models/nvidia-sparse/OpenReasoning-Nemotron-14B-Sparse-0.67" \
#     --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part3_of_3.json"   \
#     --output_dir ./trained_models/Nemotron-14B-Sparse/OpenReasoning-Nemotron-14B-Sparse-0.67-Ensemble_split_33/OpenReasoning-Nemotron-14B-Sparse-0.67-Math14k-part3-al50 \
#     --rounds 3 \
#     --init_frac 0.1 \
#     --acq_frac 0.2  


# WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 \
# "$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3201 active_learning.py \
#     --base_model "/home/models/nvidia-sparse/OpenReasoning-Nemotron-14B-Sparse-0.67" \
#     --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part1_of_3.json"   \
#     --output_dir ./trained_models/Nemotron-14B-Sparse/OpenReasoning-Nemotron-14B-Sparse-0.67-Ensemble_split_33/OpenReasoning-Nemotron-14B-Sparse-0.67-Math14k-part1-rand50 \
#     --rounds 1 \
#     --init_frac 0.5

# WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 \
# "$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3201 active_learning.py \
#     --base_model "/home/models/nvidia-sparse/OpenReasoning-Nemotron-14B-Sparse-0.67" \
#     --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part2_of_3.json"   \
#     --output_dir ./trained_models/Nemotron-14B-Sparse/OpenReasoning-Nemotron-14B-Sparse-0.67-Ensemble_split_33/OpenReasoning-Nemotron-14B-Sparse-0.67-Math14k-part2-rand50 \
#     --rounds 1 \
#     --init_frac 0.5 

# WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 \
# "$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3201 active_learning.py \
#     --base_model "/home/models/nvidia-sparse/OpenReasoning-Nemotron-14B-Sparse-0.67" \
#     --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part3_of_3.json"   \
#     --output_dir ./trained_models/Nemotron-14B-Sparse/OpenReasoning-Nemotron-14B-Sparse-0.67-Ensemble_split_33/OpenReasoning-Nemotron-14B-Sparse-0.67-Math14k-part3-rand50 \
#     --rounds 1 \
#     --init_frac 0.5 

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.9"


# WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 \
# python finetune_fast.py \
#   --base_model "/home/models/nvidia-sparse/OpenReasoning-Nemotron-14B-Sparse-0.75" \
#   --data_path /home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part1_of_4.json \
#   --output_dir ./trained_models/Nemotron-14B-Sparse/OpenReasoning-Nemotron-14B-Sparse-0.75-Ensemble_split_25/OpenReasoning-Nemotron-14B-Sparse-0.75-Math14k-part1 \
#   --batch_size 2 --micro_batch_size 1 \
#   --num_epochs 3 --learning_rate 3e-5 \
#   --cutoff_len 192 --val_set_size 120 \
#   --speed_preset safe 


#   WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 \
# python finetune_fast.py \
#   --base_model "/home/models/nvidia-sparse/OpenReasoning-Nemotron-14B-Sparse-0.75" \
#   --data_path /home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part2_of_4.json \
#   --output_dir ./trained_models/Nemotron-14B-Sparse/OpenReasoning-Nemotron-14B-Sparse-0.75-Ensemble_split_25/OpenReasoning-Nemotron-14B-Sparse-0.75-Math14k-part2 \
#   --batch_size 2 --micro_batch_size 1 \
#   --num_epochs 3 --learning_rate 3e-5 \
#   --cutoff_len 192 --val_set_size 120 \
#   --speed_preset safe        



#   WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 \
# python finetune_fast.py \
#   --base_model "/home/models/nvidia-sparse/OpenReasoning-Nemotron-14B-Sparse-0.75" \
#   --data_path /home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part3_of_4.json \
#   --output_dir ./trained_models/Nemotron-14B-Sparse/OpenReasoning-Nemotron-14B-Sparse-0.75-Ensemble_split_25/OpenReasoning-Nemotron-14B-Sparse-0.75-Math14k-part3 \
#   --batch_size 2 --micro_batch_size 1 \
#   --num_epochs 3 --learning_rate 3e-5 \
#   --cutoff_len 192 --val_set_size 120 \
#   --speed_preset safe 

env  WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 \
python finetune_fast.py \
  --base_model "/home/models/nvidia-sparse/OpenReasoning-Nemotron-14B-Sparse-0.75" \
  --data_path /home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part4_of_4.json \
  --output_dir ./trained_models/Nemotron-14B-Sparse/OpenReasoning-Nemotron-14B-Sparse-0.75-Ensemble_split_25/OpenReasoning-Nemotron-14B-Sparse-0.75-Math14k-part4 \
  --batch_size 2 --micro_batch_size 1 \
  --num_epochs 3 --learning_rate 3e-5 \
  --cutoff_len 192 --val_set_size 120 \
  --speed_preset safe

# echo "[DONE] All splits finished for Math14k Ensemble: ./trained_models/Nemotron-14B-Sparse/OpenReasoning-Nemotron-14B-Sparse-0.75-Ensemble_split_25/"




env WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 \
python finetune_fast.py \
  --base_model "/home/models/nvidia-sparse/OpenReasoning-Nemotron-14B-Sparse-0.80" \
  --data_path /home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part1_of_5.json \
  --output_dir ./trained_models/Nemotron-14B-Sparse/OpenReasoning-Nemotron-14B-Sparse-0.80-Ensemble_split_20/OpenReasoning-Nemotron-14B-Sparse-0.80-Math14k-part1 \
  --batch_size 4 --micro_batch_size 2 \
  --num_epochs 3 --learning_rate 3e-5 \
  --cutoff_len 192 --val_set_size 120 \
  --speed_preset safe




WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 \
python finetune_fast.py \
  --base_model "/home/models/nvidia-sparse/OpenReasoning-Nemotron-14B-Sparse-0.80" \
  --data_path /home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part2_of_5.json \
  --output_dir ./trained_models/Nemotron-14B-Sparse/OpenReasoning-Nemotron-14B-Sparse-0.80-Ensemble_split_20/OpenReasoning-Nemotron-14B-Sparse-0.80-Math14k-part2 \
  --batch_size 4 --micro_batch_size 2 \
  --num_epochs 3 --learning_rate 3e-5 \
  --cutoff_len 192 --val_set_size 120 \
  --speed_preset safe


  WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 \
python finetune_fast.py \
  --base_model "/home/models/nvidia-sparse/OpenReasoning-Nemotron-14B-Sparse-0.80" \
  --data_path /home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part3_of_5.json \
  --output_dir ./trained_models/Nemotron-14B-Sparse/OpenReasoning-Nemotron-14B-Sparse-0.80-Ensemble_split_20/OpenReasoning-Nemotron-14B-Sparse-0.80-Math14k-part3 \
  --batch_size 4 --micro_batch_size 2 \
  --num_epochs 3 --learning_rate 3e-5 \
  --cutoff_len 192 --val_set_size 120 \
  --speed_preset safe

  WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 \
python finetune_fast.py \
  --base_model "/home/models/nvidia-sparse/OpenReasoning-Nemotron-14B-Sparse-0.80" \
  --data_path /home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part4_of_5.json \
  --output_dir ./trained_models/Nemotron-14B-Sparse/OpenReasoning-Nemotron-14B-Sparse-0.80-Ensemble_split_20/OpenReasoning-Nemotron-14B-Sparse-0.80-Math14k-part4 \
  --batch_size 4 --micro_batch_size 2 \
  --num_epochs 3 --learning_rate 3e-5 \
  --cutoff_len 192 --val_set_size 120 \
  --speed_preset safe

  WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 \
python finetune_fast.py \
  --base_model "/home/models/nvidia-sparse/OpenReasoning-Nemotron-14B-Sparse-0.80" \
  --data_path /home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part5_of_5.json \
  --output_dir ./trained_models/Nemotron-14B-Sparse/OpenReasoning-Nemotron-14B-Sparse-0.80-Ensemble_split_20/OpenReasoning-Nemotron-14B-Sparse-0.80-Math14k-part5 \
  --batch_size 4 --micro_batch_size 2 \
  --num_epochs 3 --learning_rate 3e-5 \
  --cutoff_len 192 --val_set_size 120 \
  --speed_preset safe

echo "[DONE] All splits finished for Math14k Ensemble: ./trained_models/Nemotron-14B-Sparse/OpenReasoning-Nemotron-14B-Sparse-0.80-Ensemble_split_33/"




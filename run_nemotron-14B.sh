# #!/usr/bin/env bash
# set -euo pipefail

# # ========= User paths =========
# DATA_JSON="/home/aneek/LLM-Adapters/ft-training_set/math_14k.json"
# DENSE_MODEL_ROOT="/home/models/nvidia"
# SPARSE_MODEL_ROOT="/home/models/nvidia-sparse"
# TRAINED_ROOT="/home/aneek/LLM-Adapters/trained_models/Nemotron-14B-Sparse"

# # ========= GPUs / Dist =========
# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,0}"
# WORLD_SIZE=2

# # ========= W&B =========
# export WANDB_MODE="${WANDB_MODE:-offline}"
# export WANDB_PROJECT="${WANDB_PROJECT:-huggingface}"
# # export WANDB_ENTITY="your-entity"

# # ========= Torchrun resolver =========
# TORCHRUN="${CONDA_PREFIX:+$CONDA_PREFIX/bin/torchrun}"
# if [[ -z "${TORCHRUN}" || ! -x "${TORCHRUN}" ]]; then
#   TORCHRUN="$(command -v torchrun)"
# fi

# # ========= Models to run =========
# # Dense (exact names you provided)
# # DENSE_MODELS=(
# #   "OpenReasoning-Nemotron-7B"
# #   "OpenReasoning-Nemotron-14B"
# # )

# # Sparse (exact names you provided)
# SPARSE_MODELS=(
#   "OpenReasoning-Nemotron-14B-Sparse-0.80"
#   "OpenReasoning-Nemotron-14B-Sparse-0.75"
#   "OpenReasoning-Nemotron-14B-Sparse-0.67"
  
# )

# # Build list with full paths (and keep a printable slug)
# declare -a MODEL_PATHS=()
# declare -a MODEL_SLUGS=()

# # for m in "${DENSE_MODELS[@]}"; do
# #   MODEL_PATHS+=("${DENSE_MODEL_ROOT}/${m}")
# #   MODEL_SLUGS+=("${m}")
# # done
# for m in "${SPARSE_MODELS[@]}"; do
#   MODEL_PATHS+=("${SPARSE_MODEL_ROOT}/${m}")
#   MODEL_SLUGS+=("${m}")
# done

# # ========= Training hyperparameters =========
# BATCH_SIZE=4
# MICRO_BATCH_SIZE=1
# NUM_EPOCHS=3
# LR=3e-5
# CUTOFF_LEN=256
# VAL_SIZE=120
# SCORING_BS=8

# # ========= Active Learning configs =========
# # rand50 => 1 round, seed 50% (acq frac unused but kept)
# AL_RAND_ROUNDS=1
# AL_RAND_INIT=0.5
# AL_RAND_ACQ=0.1

# # al50 => 3 rounds: 10% + 20% + 20%
# AL_AL50_ROUNDS=3
# AL_AL50_INIT=0.1
# AL_AL50_ACQ=0.2

# # Warm-start AL from the dense FT checkpoint (set 0 to start from base)
# AL_WARM_START=0

# # ========= Utility =========
# ts(){ date +"%Y-%m-%d_%H-%M-%S"; }

# # Ports per regime (offset + model index)
# PORT_BASE_FT=5200
# PORT_BASE_RAND=5300
# PORT_BASE_AL50=5400

# # ========= Sanity checks =========
# [[ -f "${DATA_JSON}" ]] || { echo "[ERROR] Data json not found: ${DATA_JSON}" >&2; exit 1; }
# mkdir -p "${TRAINED_ROOT}"

# # ========= Main loop =========
# for i in "${!MODEL_PATHS[@]}"; do
#   BASE_MODEL="${MODEL_PATHS[$i]}"
#   MODEL_SLUG="${MODEL_SLUGS[$i]}"

#   if [[ ! -d "${BASE_MODEL}" ]]; then
#     echo "[WARN] Skipping (missing model dir): ${BASE_MODEL}"
#     continue
#   fi

#   OUT_ROOT="${TRAINED_ROOT}/${MODEL_SLUG}-Math14k"
#   LOG_DIR="${OUT_ROOT}/logs"
#   mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

#   echo "================ MODEL: ${MODEL_SLUG} ================="
#   echo "Base: ${BASE_MODEL}"
#   echo "Data: ${DATA_JSON}"
#   echo "Out : ${OUT_ROOT}"

#   # -------- (1) Full finetune on all of Math14k --------
#   FT_OUT="${OUT_ROOT}/${MODEL_SLUG}-full"
#   FT_RUN="FT-${MODEL_SLUG}-Math14k"
#   echo "[STEP] Finetune -> ${FT_OUT}"

#   python finetune.py \
#     --base_model "${BASE_MODEL}" \
#     --data_path "${DATA_JSON}" \
#     --output_dir "${FT_OUT}" \
#     --batch_size ${BATCH_SIZE} \
#     --micro_batch_size ${MICRO_BATCH_SIZE} \
#     --num_epochs ${NUM_EPOCHS} \
#     --learning_rate ${LR} \
#     --cutoff_len ${CUTOFF_LEN} \
#     --val_set_size ${VAL_SIZE} \
#     --wandb_run_name "${FT_RUN}" \
#     2>&1 | tee "${LOG_DIR}/$(ts)_finetune_${MODEL_SLUG}.log"

#   # Decide AL starting point
#   AL_BASE="${BASE_MODEL}"
#   if (( AL_WARM_START )); then
#     AL_BASE="${FT_OUT}"
#   fi

#   # -------- (2) Active Learning: rand50 --------
#   AL_RAND_OUT="${OUT_ROOT}/${MODEL_SLUG}-rand50"
#   AL_RAND_RUN="AL-rand50-${MODEL_SLUG}-Math14k"
#   PORT=$((PORT_BASE_RAND + i))
#   echo "[STEP] AL rand50 -> ${AL_RAND_OUT} (port ${PORT})"

#   PYTHONNOUSERSITE=1 "${TORCHRUN}" --nproc_per_node=${WORLD_SIZE} --master_port=${PORT} active_learning_fast.py \
#     --base_model "${AL_BASE}" \
#     --data_path "${DATA_JSON}" \
#     --output_dir "${AL_RAND_OUT}" \
#     --rounds ${AL_RAND_ROUNDS} \
#     --init_frac ${AL_RAND_INIT} \
#     --acq_frac ${AL_RAND_ACQ} \
#     --uncertainty logppl \
#     --cutoff_len ${CUTOFF_LEN} \
#     --scoring_batch_size ${SCORING_BS} \
#     --num_epochs ${NUM_EPOCHS} \
#     --learning_rate ${LR} \
#     --per_device_train_batch_size ${BATCH_SIZE} \
#     --micro_batch_size ${MICRO_BATCH_SIZE} \
#     --val_set_size ${VAL_SIZE} \
#     --wandb_run_name "${AL_RAND_RUN}" \
#     2>&1 | tee "${LOG_DIR}/$(ts)_al_rand50_${MODEL_SLUG}.log"

#   # -------- (3) Active Learning: al50 (10% + 20% + 20%) --------
#   AL_AL50_OUT="${OUT_ROOT}/${MODEL_SLUG}-al50"
#   AL_AL50_RUN="AL-al50-${MODEL_SLUG}-Math14k"
#   PORT=$((PORT_BASE_AL50 + i))
#   echo "[STEP] AL al50 -> ${AL_AL50_OUT} (port ${PORT})"

#   PYTHONNOUSERSITE=1 "${TORCHRUN}" --nproc_per_node=${WORLD_SIZE} --master_port=${PORT} active_learning_fast.py \
#     --base_model "${AL_BASE}" \
#     --data_path "${DATA_JSON}" \
#     --output_dir "${AL_AL50_OUT}" \
#     --rounds ${AL_AL50_ROUNDS} \
#     --init_frac ${AL_AL50_INIT} \
#     --acq_frac ${AL_AL50_ACQ} \
#     --uncertainty logppl \
#     --cutoff_len ${CUTOFF_LEN} \
#     --scoring_batch_size ${SCORING_BS} \
#     --num_epochs ${NUM_EPOCHS} \
#     --learning_rate ${LR} \
#     --per_device_train_batch_size ${BATCH_SIZE} \
#     --micro_batch_size ${MICRO_BATCH_SIZE} \
#     --val_set_size ${VAL_SIZE} \
#     --wandb_run_name "${AL_AL50_RUN}" \
#     2>&1 | tee "${LOG_DIR}/$(ts)_al_al50_${MODEL_SLUG}.log"

#   echo "======== Completed: ${MODEL_SLUG} ========"
# done

# echo "[DONE] Dense 14B-sparse (0.67/0.75/0.80) over Math14k: full, rand50, al50."



# ---- env ----
# env
# export WORLD_SIZE=1
# export CUDA_VISIBLE_DEVICES=2
# export PYTHONNOUSERSITE=1
# TORCHRUN="$CONDA_PREFIX/bin/torchrun"
# SCRIPT="active_learning_fast.py"   # use the updated file I gave you

# # common tokenizer path (dense sibling of the sparse model)
# TOK="/home/models/nvidia/OpenReasoning-Nemotron-14B"
# BASE="/home/models/nvidia-sparse/OpenReasoning-Nemotron-14B-Sparse-0.67"

# # ===== Active Learning (AL 10% seed, acquire 20%, 3 rounds) =====
# $TORCHRUN --nproc_per_node=1 --master_port=3201 "$SCRIPT" \
#   --base_model "$BASE" \
#   --tokenizer_path "$TOK" \
#   --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part1_of_3.json" \
#   --output_dir "./trained_models/Nemotron-14B-Sparse/OpenReasoning-Nemotron-14B-Sparse-0.67-Ensemble_split_33/OpenReasoning-Nemotron-14B-Sparse-0.67-Math14k-part1-al50" \
#   --rounds 3 --init_frac 0.1 --acq_frac 0.2

# $TORCHRUN --nproc_per_node=1 --master_port=3201 "$SCRIPT" \
#   --base_model "$BASE" \
#   --tokenizer_path "$TOK" \
#   --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part2_of_3.json" \
#   --output_dir "./trained_models/Nemotron-14B-Sparse/OpenReasoning-Nemotron-14B-Sparse-0.67-Ensemble_split_33/OpenReasoning-Nemotron-14B-Sparse-0.67-Math14k-part2-al50" \
#   --rounds 3 --init_frac 0.1 --acq_frac 0.2

# $TORCHRUN --nproc_per_node=1 --master_port=3201 "$SCRIPT" \
#   --base_model "$BASE" \
#   --tokenizer_path "$TOK" \
#   --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part3_of_3.json" \
#   --output_dir "./trained_models/Nemotron-14B-Sparse/OpenReasoning-Nemotron-14B-Sparse-0.67-Ensemble_split_33/OpenReasoning-Nemotron-14B-Sparse-0.67-Math14k-part3-al50" \
#   --rounds 3 --init_frac 0.1 --acq_frac 0.2

# # ===== Random 50% baseline (single round) =====
# $TORCHRUN --nproc_per_node=1 --master_port=3201 "$SCRIPT" \
#   --base_model "$BASE" \
#   --tokenizer_path "$TOK" \
#   --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part1_of_3.json" \
#   --output_dir "./trained_models/Nemotron-14B-Sparse/OpenReasoning-Nemotron-14B-Sparse-0.67-Ensemble_split_33/OpenReasoning-Nemotron-14B-Sparse-0.67-Math14k-part1-rand50" \
#   --rounds 1 --init_frac 0.5

# $TORCHRUN --nproc_per_node=1 --master_port=3201 "$SCRIPT" \
#   --base_model "$BASE" \
#   --tokenizer_path "$TOK" \
#   --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part2_of_3.json" \
#   --output_dir "./trained_models/Nemotron-14B-Sparse/OpenReasoning-Nemotron-14B-Sparse-0.67-Ensemble_split_33/OpenReasoning-Nemotron-14B-Sparse-0.67-Math14k-part2-rand50" \
#   --rounds 1 --init_frac 0.5

# $TORCHRUN --nproc_per_node=1 --master_port=3201 "$SCRIPT" \
#   --base_model "$BASE" \
#   --tokenizer_path "$TOK" \
#   --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part3_of_3.json" \
#   --output_dir "./trained_models/Nemotron-14B-Sparse/OpenReasoning-Nemotron-14B-Sparse-0.67-Ensemble_split_33/OpenReasoning-Nemotron-14B-Sparse-0.67-Math14k-part3-rand50" \
#   --rounds 1 --init_frac 0.5




export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=2
export PYTHONNOUSERSITE=1
TORCHRUN="$CONDA_PREFIX/bin/torchrun"
SCRIPT="active_learning_fast.py"   # use the updated file you gave me

# tokenizer stays the dense sibling
TOK="/home/models/nvidia/OpenReasoning-Nemotron-14B"
BASE="/home/models/nvidia-sparse/OpenReasoning-Nemotron-14B-Sparse-0.75"

# ===== Active Learning (AL 10% seed, acquire 20%, 3 rounds) =====
$TORCHRUN --nproc_per_node=1 --master_port=3201 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part1_of_4.json" \
  --output_dir "./trained_models/Nemotron-14B-Sparse/OpenReasoning-Nemotron-14B-Sparse-0.75-Ensemble_split_25/OpenReasoning-Nemotron-14B-Sparse-0.75-Math14k-part1-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3201 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part2_of_4.json" \
  --output_dir "./trained_models/Nemotron-14B-Sparse/OpenReasoning-Nemotron-14B-Sparse-0.75-Ensemble_split_25/OpenReasoning-Nemotron-14B-Sparse-0.75-Math14k-part2-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3201 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part3_of_4.json" \
  --output_dir "./trained_models/Nemotron-14B-Sparse/OpenReasoning-Nemotron-14B-Sparse-0.75-Ensemble_split_25/OpenReasoning-Nemotron-14B-Sparse-0.75-Math14k-part3-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3201 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part4_of_4.json" \
  --output_dir "./trained_models/Nemotron-14B-Sparse/OpenReasoning-Nemotron-14B-Sparse-0.75-Ensemble_split_25/OpenReasoning-Nemotron-14B-Sparse-0.75-Math14k-part4-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

# ===== Random 50% baseline (single round) =====
$TORCHRUN --nproc_per_node=1 --master_port=3201 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part1_of_4.json" \
  --output_dir "./trained_models/Nemotron-14B-Sparse/OpenReasoning-Nemotron-14B-Sparse-0.75-Ensemble_split_25/OpenReasoning-Nemotron-14B-Sparse-0.75-Math14k-part1-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3201 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part2_of_4.json" \
  --output_dir "./trained_models/Nemotron-14B-Sparse/OpenReasoning-Nemotron-14B-Sparse-0.75-Ensemble_split_25/OpenReasoning-Nemotron-14B-Sparse-0.75-Math14k-part2-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3201 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part3_of_4.json" \
  --output_dir "./trained_models/Nemotron-14B-Sparse/OpenReasoning-Nemotron-14B-Sparse-0.75-Ensemble_split_25/OpenReasoning-Nemotron-14B-Sparse-0.75-Math14k-part3-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3201 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part4_of_4.json" \
  --output_dir "./trained_models/Nemotron-14B-Sparse/OpenReasoning-Nemotron-14B-Sparse-0.75-Ensemble_split_25/OpenReasoning-Nemotron-14B-Sparse-0.75-Math14k-part4-rand50" \
  --rounds 1 --init_frac 0.5
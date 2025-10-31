#!/usr/bin/env bash
set -euo pipefail

# ========= User paths =========
DATA_JSON="/home/aneek/LLM-Adapters/ft-training_set/math_14k.json"
DENSE_MODEL_ROOT="/home/models/nvidia"
SPARSE_MODEL_ROOT="/home/models/Qwen_Sparse"
TRAINED_ROOT="/home/aneek/LLM-Adapters/trained_models/Qwen3-4B-Sparse/"

# ========= GPUs / Dist =========
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
WORLD_SIZE=1

# ========= W&B =========
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-huggingface}"
# export WANDB_ENTITY="your-entity"

# ========= Torchrun resolver =========
TORCHRUN="${CONDA_PREFIX:+$CONDA_PREFIX/bin/torchrun}"
if [[ -z "${TORCHRUN}" || ! -x "${TORCHRUN}" ]]; then
  TORCHRUN="$(command -v torchrun)"
fi

# ========= Models to run =========
# Dense (exact names you provided)
# DENSE_MODELS=(
#   "OpenReasoning-Nemotron-7B"
#   "OpenReasoning-Nemotron-14B"
# )

# Sparse (exact names you provided)
SPARSE_MODELS=(
  "Qwen3-4B-Sparse-0.20"
  # "Qwen3-4B-Sparse-0.25"
  # "Qwen3-4B-Sparse-0.33"
  # "Qwen3-4B-Sparse-0.50"
)

# Build list with full paths (and keep a printable slug)
declare -a MODEL_PATHS=()
declare -a MODEL_SLUGS=()

# for m in "${DENSE_MODELS[@]}"; do
#   MODEL_PATHS+=("${DENSE_MODEL_ROOT}/${m}")
#   MODEL_SLUGS+=("${m}")
# done
for m in "${SPARSE_MODELS[@]}"; do
  MODEL_PATHS+=("${SPARSE_MODEL_ROOT}/${m}")
  MODEL_SLUGS+=("${m}")
done

# ========= Training hyperparameters =========
BATCH_SIZE=4
MICRO_BATCH_SIZE=1
NUM_EPOCHS=3
LR=3e-5
CUTOFF_LEN=256
VAL_SIZE=120
SCORING_BS=8

# ========= Active Learning configs =========
# rand50 => 1 round, seed 50% (acq frac unused but kept)
AL_RAND_ROUNDS=1
AL_RAND_INIT=0.5
AL_RAND_ACQ=0.1

# al50 => 3 rounds: 10% + 20% + 20%
AL_AL50_ROUNDS=3
AL_AL50_INIT=0.1
AL_AL50_ACQ=0.2

# Warm-start AL from the dense FT checkpoint (set 0 to start from base)
AL_WARM_START=0

# ========= Utility =========
ts(){ date +"%Y-%m-%d_%H-%M-%S"; }

# Ports per regime (offset + model index)
PORT_BASE_FT=3200
PORT_BASE_RAND=3300
PORT_BASE_AL50=3400

# ========= Sanity checks =========
[[ -f "${DATA_JSON}" ]] || { echo "[ERROR] Data json not found: ${DATA_JSON}" >&2; exit 1; }
mkdir -p "${TRAINED_ROOT}"

# ========= Main loop =========
for i in "${!MODEL_PATHS[@]}"; do
  BASE_MODEL="${MODEL_PATHS[$i]}"
  MODEL_SLUG="${MODEL_SLUGS[$i]}"

  if [[ ! -d "${BASE_MODEL}" ]]; then
    echo "[WARN] Skipping (missing model dir): ${BASE_MODEL}"
    continue
  fi

  OUT_ROOT="${TRAINED_ROOT}/${MODEL_SLUG}-Math14k"
  LOG_DIR="${OUT_ROOT}/logs"
  mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

  echo "================ MODEL: ${MODEL_SLUG} ================="
  echo "Base: ${BASE_MODEL}"
  echo "Data: ${DATA_JSON}"
  echo "Out : ${OUT_ROOT}"

  # -------- (1) Full finetune on all of Math14k --------
  # FT_OUT="${OUT_ROOT}/${MODEL_SLUG}-full"
  # FT_RUN="FT-${MODEL_SLUG}-Math14k"
  # echo "[STEP] Finetune -> ${FT_OUT}"

  # python finetune.py \
  #   --base_model "${BASE_MODEL}" \
  #   --data_path "${DATA_JSON}" \
  #   --output_dir "${FT_OUT}" \
  #   --batch_size ${BATCH_SIZE} \
  #   --micro_batch_size ${MICRO_BATCH_SIZE} \
  #   --num_epochs ${NUM_EPOCHS} \
  #   --learning_rate ${LR} \
  #   --cutoff_len ${CUTOFF_LEN} \
  #   --val_set_size ${VAL_SIZE} \
  #   --wandb_run_name "${FT_RUN}" \
  #   2>&1 | tee "${LOG_DIR}/$(ts)_finetune_${MODEL_SLUG}.log"

  # Decide AL starting point
  AL_BASE="${BASE_MODEL}"
  if (( AL_WARM_START )); then
    AL_BASE="${FT_OUT}"
  fi

  # # -------- (2) Active Learning: rand50 --------
  # AL_RAND_OUT="${OUT_ROOT}/${MODEL_SLUG}-rand50"
  # AL_RAND_RUN="AL-rand50-${MODEL_SLUG}-Math14k"
  # PORT=$((PORT_BASE_RAND + i))
  # echo "[STEP] AL rand50 -> ${AL_RAND_OUT} (port ${PORT})"

  # PYTHONNOUSERSITE=1 "${TORCHRUN}" --nproc_per_node=${WORLD_SIZE} --master_port=${PORT} active_learning_nop.py \
  #   --base_model "${AL_BASE}" \
  #   --data_path "${DATA_JSON}" \
  #   --output_dir "${AL_RAND_OUT}" \
  #   --rounds ${AL_RAND_ROUNDS} \
  #   --init_frac ${AL_RAND_INIT} \
  #   --acq_frac ${AL_RAND_ACQ} \
  #   --uncertainty logppl \
  #   --cutoff_len ${CUTOFF_LEN} \
  #   --scoring_batch_size ${SCORING_BS} \
  #   --num_epochs ${NUM_EPOCHS} \
  #   --learning_rate ${LR} \
  #   --per_device_train_batch_size ${BATCH_SIZE} \
  #   --micro_batch_size ${MICRO_BATCH_SIZE} \
  #   --val_set_size ${VAL_SIZE} \
  #   --wandb_run_name "${AL_RAND_RUN}" \
  #   2>&1 | tee "${LOG_DIR}/$(ts)_al_rand50_${MODEL_SLUG}.log"

  # -------- (3) Active Learning: al50 (10% + 20% + 20%) --------
  AL_AL50_OUT="${OUT_ROOT}/${MODEL_SLUG}-al50"
  AL_AL50_RUN="AL-al50-${MODEL_SLUG}-Math14k"
  PORT=$((PORT_BASE_AL50 + i))
  echo "[STEP] AL al50 -> ${AL_AL50_OUT} (port ${PORT})"

  PYTHONNOUSERSITE=1 "${TORCHRUN}" --nproc_per_node=${WORLD_SIZE} --master_port=${PORT} active_learning_nop.py \
    --base_model "${AL_BASE}" \
    --data_path "${DATA_JSON}" \
    --output_dir "${AL_AL50_OUT}" \
    --rounds ${AL_AL50_ROUNDS} \
    --init_frac ${AL_AL50_INIT} \
    --acq_frac ${AL_AL50_ACQ} \
    --uncertainty logppl \
    --cutoff_len ${CUTOFF_LEN} \
    --scoring_batch_size ${SCORING_BS} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LR} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --micro_batch_size ${MICRO_BATCH_SIZE} \
    --val_set_size ${VAL_SIZE} \
    --wandb_run_name "${AL_AL50_RUN}" \
    2>&1 | tee "${LOG_DIR}/$(ts)_al_al50_${MODEL_SLUG}.log"

  echo "======== Completed: ${MODEL_SLUG} ========"
done

echo "[DONE] Dense 7B-sparse (0.20/0.25/0.33/0.50) over Math14k: full, rand50, al50."
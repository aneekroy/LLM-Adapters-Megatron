#!/usr/bin/env bash
set -euo pipefail

# ===== GPU selection =====
export CUDA_VISIBLE_DEVICES=1
WORLD_SIZE=1

# ===== WANDB =====
export WANDB_MODE=offline
export WANDB_PROJECT="huggingface"     # change if you prefer
# export WANDB_ENTITY="your-entity"     # optional

# ===== Torchrun resolver =====
TORCHRUN="${CONDA_PREFIX:+$CONDA_PREFIX/bin/torchrun}"
if [[ -z "${TORCHRUN}" || ! -x "${TORCHRUN}" ]]; then
  TORCHRUN="$(command -v torchrun)"
fi

# ===== Models (dense) =====
MODEL_ROOT="/home/models/nvidia"
DENSE_MODELS=(
  "OpenReasoning-Nemotron-7B"
)

# ===== Data splits (dir => num_parts) =====
FT_ROOT="/home/aneek/LLM-Adapters/ft-training_set"
declare -A SPLITS
SPLITS["split_20"]=5
SPLITS["split_25"]=4
SPLITS["split_33"]=3
SPLITS["split_50"]=2

# ===== Output + logs root (separate per model + split) =====
TRAINED_ROOT="/home/aneek/LLM-Adapters/trained_models"

# ===== Finetune hyperparams (tune to your GPU) =====
BATCH_SIZE=4
MICRO_BATCH_SIZE=1
NUM_EPOCHS=3
LR=3e-5
CUTOFF_LEN=256
VAL_SIZE=120

# ===== Active Learning configs =====
# rand50 => label 50% in one shot (1 round, init=0.5; acq unused but kept)
AL_RAND_ROUNDS=1
AL_RAND_INIT=0.5
AL_RAND_ACQ=0.1

# al50 => 10% seed + 20% + 20% (3 rounds total)
AL_AL50_ROUNDS=3
AL_AL50_INIT=0.1
AL_AL50_ACQ=0.2

ORDERED_SPLITS=("split_20" "split_25" "split_33" "split_50")

ts(){ date +"%Y-%m-%d_%H-%M-%S"; }

for MODEL_NAME in "${DENSE_MODELS[@]}"; do
  BASE_MODEL="${MODEL_ROOT}/${MODEL_NAME}"
  MODEL_SLUG="$(basename "${BASE_MODEL}")"

  for SPLIT in "${ORDERED_SPLITS[@]}"; do
    PARTS="${SPLITS[$SPLIT]}"
    OUT_ROOT="${TRAINED_ROOT}/${MODEL_SLUG}-Ensemble_${SPLIT/./}"
    LOG_DIR="${OUT_ROOT}/logs"
    mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

    echo "====== MODEL=${MODEL_SLUG} | SPLIT=${SPLIT} (parts=${PARTS}) ======"

    # Give each (model,split,part) a stable, non-overlapping port range
    # Base per split, then bump by model index to avoid conflicts
    case "${SPLIT}" in
      split_20) PORT_BASE=3400 ;;
      split_25) PORT_BASE=3500 ;;
      split_33) PORT_BASE=3600 ;;
      split_50) PORT_BASE=3700 ;;
      *)        PORT_BASE=3800 ;;
    esac

    # Index of model in list for port disambiguation
    MIDX=0
    for i in "${!DENSE_MODELS[@]}"; do
      [[ "${DENSE_MODELS[$i]}" == "${MODEL_NAME}" ]] && MIDX="$i"
    done

    for ((p=1; p<=PARTS; p++)); do
      PART_JSON="${FT_ROOT}/${SPLIT}/math_14k_part${p}_of_${PARTS}.json"
      [[ -f "${PART_JSON}" ]] || { echo "Missing split file: ${PART_JSON}" >&2; exit 1; }
      PART_TAG="part${p}"

      echo "--------------------- ${MODEL_SLUG} :: ${SPLIT} :: ${PART_TAG} ---------------------"

      # (1) Finetune 100% on this split
      # FT_OUT="${OUT_ROOT}/${MODEL_SLUG}-Math14k-${PART_TAG}"
      # FT_RUN="FT-${MODEL_SLUG}-${SPLIT}-${PART_TAG}"
      # echo "[STEP] Finetune -> ${FT_OUT}"

      # python finetune.py \
      #   --base_model "${BASE_MODEL}" \
      #   --data_path "${PART_JSON}" \
      #   --output_dir "${FT_OUT}" \
      #   --batch_size ${BATCH_SIZE} \
      #   --micro_batch_size ${MICRO_BATCH_SIZE} \
      #   --num_epochs ${NUM_EPOCHS} \
      #   --learning_rate ${LR} \
      #   --cutoff_len ${CUTOFF_LEN} \
      #   --val_set_size ${VAL_SIZE} \
      #   --wandb_run_name "${FT_RUN}" \
      #   2>&1 | tee "${LOG_DIR}/$(ts)_finetune_${MODEL_SLUG}_${SPLIT}_${PART_TAG}.log"

      # # (2) Active Learning: rand50
      # AL_RAND_OUT="${OUT_ROOT}/${MODEL_SLUG}-Math14k-${PART_TAG}-rand50"
      # AL_RAND_RUN="AL-rand50-${MODEL_SLUG}-${SPLIT}-${PART_TAG}"
      # PORT=$((PORT_BASE + 10*MIDX + p))
      # echo "[STEP] AL rand50 -> ${AL_RAND_OUT} (port ${PORT})"

      # PYTHONNOUSERSITE=1 "${TORCHRUN}" --nproc_per_node=1 --master_port=${PORT} active_learning_nop.py \
      #   --base_model "${BASE_MODEL}" \
      #   --data_path "${PART_JSON}" \
      #   --output_dir "${AL_RAND_OUT}" \
      #   --rounds ${AL_RAND_ROUNDS} \
      #   --init_frac ${AL_RAND_INIT} \
      #   --acq_frac ${AL_RAND_ACQ} \
      #   --uncertainty logppl --cutoff_len ${CUTOFF_LEN} --scoring_batch_size 8 \
      #   --num_epochs ${NUM_EPOCHS} --learning_rate ${LR} \
      #   --per_device_train_batch_size ${BATCH_SIZE} --micro_batch_size ${MICRO_BATCH_SIZE} --val_set_size ${VAL_SIZE} \
      #   --wandb_run_name "${AL_RAND_RUN}" \
      #   2>&1 | tee "${LOG_DIR}/$(ts)_al_rand50_${MODEL_SLUG}_${SPLIT}_${PART_TAG}.log"

      # # (3) Active Learning: al50 (10% + 20% + 20%)
      AL_AL50_OUT="${OUT_ROOT}/${MODEL_SLUG}-Math14k-${PART_TAG}-al50"
      AL_AL50_RUN="AL-al50-${MODEL_SLUG}-${SPLIT}-${PART_TAG}"
      PORT=$((PORT_BASE + 100 + 10*MIDX + p))
      echo "[STEP] AL al50 -> ${AL_AL50_OUT} (port ${PORT})"

      PYTHONNOUSERSITE=1 "${TORCHRUN}" --nproc_per_node=1 --master_port=${PORT} active_learning_nop.py \
        --base_model "${BASE_MODEL}" \
        --data_path "${PART_JSON}" \
        --output_dir "${AL_AL50_OUT}" \
        --rounds ${AL_AL50_ROUNDS} \
        --init_frac ${AL_AL50_INIT} \
        --acq_frac ${AL_AL50_ACQ} \
        --uncertainty logppl --cutoff_len ${CUTOFF_LEN} --scoring_batch_size 8 \
        --num_epochs ${NUM_EPOCHS} --learning_rate ${LR} \
        --per_device_train_batch_size ${BATCH_SIZE} --micro_batch_size ${MICRO_BATCH_SIZE} --val_set_size ${VAL_SIZE} \
        --wandb_run_name "${AL_AL50_RUN}" \
        2>&1 | tee "${LOG_DIR}/$(ts)_al_al50_${MODEL_SLUG}_${SPLIT}_${PART_TAG}.log"

    done
  done
done

echo "[DONE] Dense Nemotron runs complete."
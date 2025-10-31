#!/usr/bin/env bash
set -euo pipefail

# ===== GPU & stability =====
export TORCHINDUCTOR_DISABLE_CUDAGRAPHS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=offline   # uncomment for offline W&B

# ===== CONFIG =====
BASE_MODEL="/home/models/Qwen_Sparse/Qwen3-4B-Sparse-0.50"
SPLIT="50"
SPLIT_DIR="/home/aneek/LLM-Adapters/ft-training_set/split_${SPLIT}"
OUT_ROOT="./trained_models/Qwen_Sparse/Qwen3-4B-Sparse-0.50-Ensemble_split_${SPLIT}"
RUN_STEM="Qwen_Sparse/Qwen3-4B-Sparse-0.50"

LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

ts(){ date +"%Y-%m-%d_%H-%M-%S"; }

# ===== Run AL (al50: 10% init + 2x20% acq) over 3 parts =====
for p in 1 2 ; do
  PART_JSON="${SPLIT_DIR}/math_14k_part${p}_of_2.json"
  [[ -f "${PART_JSON}" ]] || { echo "[FATAL] Missing: ${PART_JSON}" >&2; exit 1; }

    OUT_DIR="${OUT_ROOT}/${RUN_STEM}-Math14k-part${p}"
    RUN_NAME="FT-${RUN_STEM}-split_${SPLIT}-part${p}"
    echo "==== part${p} -> ${OUT_DIR} ===="

    python /home/aneek/LLM-Adapters/finetune.py \
    --base_model "${BASE_MODEL}" \
    --data_path "${PART_JSON}" \
    --output_dir "${OUT_DIR}" \
    --batch_size 4 \
    --micro_batch_size 2 \
    --num_epochs 3 \
    --learning_rate 3e-5 \
    --cutoff_len 256 \
    --val_set_size 120 \
    --wandb_run_name "${RUN_NAME}" \
    2>&1 | tee "${LOG_DIR}/$(ts)_fullFT_${p}.log"

  OUT_DIR="${OUT_ROOT}/${RUN_STEM}-Math14k-part${p}-rand50"
  RUN_NAME="AL-rnd50-${RUN_STEM}-split_${SPLIT}-part${p}"
  echo "==== part${p} -> ${OUT_DIR} ===="

  python active_learning_fast.py \
    --base_model "${BASE_MODEL}" \
    --data_path "${PART_JSON}" \
    --output_dir "${OUT_DIR}" \
    --rounds 1 \
    --init_frac 0.5 \
    --acq_frac 0.2 \
    --uncertainty logppl \
    --cutoff_len 256 \
    --scoring_batch_size 8 \
    --num_epochs 3 \
    --learning_rate 3e-5 \
    --per_device_train_batch_size 4 \
    --micro_batch_size 1 \
    --val_set_size 120 \
    --wandb_run_name "${RUN_NAME}" \
    2>&1 | tee "${LOG_DIR}/$(ts)_al_rand50_part${p}.log"

  OUT_DIR="${OUT_ROOT}/${RUN_STEM}-Math14k-part${p}-al50"
  RUN_NAME="AL-al50-${RUN_STEM}-split_${SPLIT}-part${p}"
  echo "==== part${p} -> ${OUT_DIR} ===="

  python active_learning_fast.py \
    --base_model "${BASE_MODEL}" \
    --data_path "${PART_JSON}" \
    --output_dir "${OUT_DIR}" \
    --rounds 3 \
    --init_frac 0.1 \
    --acq_frac 0.2 \
    --uncertainty logppl \
    --cutoff_len 256 \
    --scoring_batch_size 8 \
    --num_epochs 3 \
    --learning_rate 3e-5 \
    --per_device_train_batch_size 4 \
    --micro_batch_size 1 \
    --val_set_size 120 \
    --wandb_run_name "${RUN_NAME}" \
    2>&1 | tee "${LOG_DIR}/$(ts)_al_al50_part${p}.log"
done

echo "[DONE] al50 runs for parts 1..2 under: ${OUT_ROOT}"
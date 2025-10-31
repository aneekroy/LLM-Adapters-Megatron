#!/usr/bin/env bash
set -euo pipefail

# ===== GPU & allocator =====
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2          # override with: GPU=1 ./run_all.sh
export TORCHINDUCTOR_DISABLE_CUDAGRAPHS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export WANDB_MODE=offline   # uncomment if you want offline W&B

# ===== Paths =====
DATA_FULL="/home/aneek/LLM-Adapters/ft-training_set/math_14k.json"
[[ -f "$DATA_FULL" ]] || { echo "[FATAL] Missing dataset: $DATA_FULL" >&2; exit 1; }

BASE_ROOT="/home/models/nvidia-sparse"
STEM="OpenReasoning-Nemotron-14B-Sparse"

# Output root (kept common across runs)
OUT_ROOT="./trained_models/${STEM}-New-Ensemble_fullMath14k"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

# ===== Common hyperparams =====
BATCH=4
MICRO=1
EPOCHS=3
LR=3e-5
CUT=256
VAL=120
SCORE_BS=8

ts(){ date +"%Y-%m-%d_%H-%M-%S"; }

# ===== Sparsity grid =====
for SP in 0.67 0.75 0.80; do
  BASE_MODEL="${BASE_ROOT}/${STEM}-${SP}"
  TOKENIZER_PATH="${BASE_MODEL}"    # sparse ckpts ship full tokenizer
  [[ -d "${BASE_MODEL}" ]] || { echo "[FATAL] Missing base model dir: ${BASE_MODEL}" >&2; exit 1; }

  RUN_STEM="${STEM}-${SP}"
  echo "==================== ${RUN_STEM} ===================="

  # ---------- (A) FULL FINETUNE on all 14k ----------
  # OUT_FT="${OUT_ROOT}/${RUN_STEM}-Math14k-fullFT"
  # RUN_FT="FT-${RUN_STEM}-Math14k-full"
  # echo "[FT] ${OUT_FT}"

  # python /home/aneek/LLM-Adapters/finetune.py \
  #   --base_model "${BASE_MODEL}" \
  #   --data_path "${DATA_FULL}" \
  #   --output_dir "${OUT_FT}" \
  #   --batch_size ${BATCH} \
  #   --micro_batch_size ${MICRO} \
  #   --num_epochs ${EPOCHS} \
  #   --learning_rate ${LR} \
  #   --cutoff_len ${CUT} \
  #   --val_set_size ${VAL} \
  #   # --wandb_run_name "${RUN_FT}" \
  #   2>&1 | tee "${LOG_DIR}/$(ts)_fullFT_${SP}.log"

  # ---------- (B) RAND50: one-shot 50% label (0.5) + dummy acq 0.1 ----------
  OUT_RAND50="${OUT_ROOT}/${RUN_STEM}-Math14k-rand50"
  RUN_RAND50="AL-rand50-${RUN_STEM}-Math14k"
  echo "[AL-rand50] ${OUT_RAND50}"

  python /home/aneek/LLM-Adapters/active_learning_fast.py \
    --base_model "${BASE_MODEL}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --data_path "${DATA_FULL}" \
    --output_dir "${OUT_RAND50}" \
    --rounds 1 \
    --init_frac 0.5 \
    --acq_frac 0.1 \
    --uncertainty logppl \
    --cutoff_len ${CUT} \
    --scoring_batch_size ${SCORE_BS} \
    --num_epochs ${EPOCHS} \
    --learning_rate ${LR} \
    --per_device_train_batch_size ${BATCH} \
    --micro_batch_size ${MICRO} \
    --val_set_size ${VAL} \
    --wandb_run_name "${RUN_RAND50}" \
    2>&1 | tee "${LOG_DIR}/$(ts)_rand50_${SP}.log"

  # ---------- (C) AL50: ~50% via 10% init + 2x20% acquisitions ----------
  OUT_AL50="${OUT_ROOT}/${RUN_STEM}-Math14k-al50"
  RUN_AL50="AL-al50-${RUN_STEM}-Math14k"
  echo "[AL-al50] ${OUT_AL50}"

  python /home/aneek/LLM-Adapters/active_learning_fast.py \
    --base_model "${BASE_MODEL}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --data_path "${DATA_FULL}" \
    --output_dir "${OUT_AL50}" \
    --rounds 3 \
    --init_frac 0.1 \
    --acq_frac 0.2 \
    --uncertainty logppl \
    --cutoff_len ${CUT} \
    --scoring_batch_size ${SCORE_BS} \
    --num_epochs ${EPOCHS} \
    --learning_rate ${LR} \
    --per_device_train_batch_size ${BATCH} \
    --micro_batch_size ${MICRO} \
    --val_set_size ${VAL} \
    --wandb_run_name "${RUN_AL50}" \
    2>&1 | tee "${LOG_DIR}/$(ts)_al50_${SP}.log"

done

echo "[DONE] All sparsities (.20/.25/.33/.50) finished for full Math14k: ${OUT_ROOT}"
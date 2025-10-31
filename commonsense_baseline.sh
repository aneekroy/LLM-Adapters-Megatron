#!/usr/bin/env bash
set -euo pipefail

# ========= Config =========
GPU="${GPU:-1}"
PY="${PY:-python}"
EVAL_PY="${EVAL_PY:-/home/aneek/LLM-Adapters/eval_commonsense.py}"
ENGINE="${ENGINE:-vllm}"
TP_SIZE="${TP_SIZE:-1}"
SEED="${SEED:-0}"
DO_SAMPLE="${DO_SAMPLE:-false}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
OUTDIR="${OUTDIR:-./eval_logs/commonsense-baseline/Qwen-8B}"
DRY_RUN="${DRY_RUN:-false}"

# --- Choose ONE model block ---

# (A) Qwen3-4B-Sparse baseline
# MODEL_NAME="Qwen3-4B-Sparse"
# BASE_MODEL="/home/models/Qwen_Sparse/Qwen3-4B-Sparse-0.50"
# TOKENIZER_PATH="/home/models/Qwen_Sparse/Qwen3-4B-Sparse-0.50"

# (B) Qwen3-4B-Instruct baseline
# MODEL_NAME="Qwen3-4B-Instruct"
# BASE_MODEL="/home/models/Qwen/Qwen3-4B-Instruct-2507"
# TOKENIZER_PATH="/home/models/Qwen/Qwen3-4B-Instruct-2507"

# (B) Qwen3-8B baseline
# MODEL_NAME="Qwen3-8B-Instruct"
# BASE_MODEL="/home/models/Qwen/Qwen3-8B"
# TOKENIZER_PATH="/home/models/Qwen/Qwen3-8B"

# (A) Qwen3-4B-Sparse baseline
MODEL_NAME="Qwen3-8B-Sparse"
BASE_MODEL="/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.50"
TOKENIZER_PATH="/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.50"


# (C) Llama-3.2-11B-Vision-Instruct baseline
# MODEL_NAME="Llama-3.2-11B-Vision-Instruct"    
# BASE_MODEL="/home/models/Llama-3.2-11B-Vision-Instruct"
# TOKENIZER_PATH="/home/models/Llama-3.2-11B-Vision-Instruct"
# OUTDIR="${OUTDIR:-./eval_logs/commonsense-baseline/Llama-3.2-11B-Vision-Instruct}"
# DRY_RUN="true"
# GPU="0,1"


# /home/models/Llama-3.2-11B-Vision-Instruct/

# All commonsense datasets supported by eval_commonsense.py
DATASETS=(boolq piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa)

mkdir -p "$OUTDIR"
ts(){ date +"%Y-%m-%d_%H-%M-%S"; }

run_one() {
  local ds="$1"
  local subdir="${OUTDIR}/${MODEL_NAME}/${ds}"
  mkdir -p "$subdir"
  local log="${subdir}/${MODEL_NAME}_${ds}_$(ts).log"

  echo "[${MODEL_NAME}] dataset=${ds}"

  local cmd=(
    "${PY}" "${EVAL_PY}"
      --engine "${ENGINE}"
      --dataset "${ds}"
      --model "${MODEL_NAME}"
      --base_model "${BASE_MODEL}"
      --tokenizer_path "${TOKENIZER_PATH}"
      --adapter none
      --output_dir "${OUTDIR}"
      --tp_size "${TP_SIZE}"
      --seed "${SEED}"
      --do_sample "${DO_SAMPLE}"
      --max_new_tokens "${MAX_NEW_TOKENS}"
  )

  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "CUDA_VISIBLE_DEVICES=${GPU} ${cmd[*]}"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${GPU}" "${cmd[@]}" 2>&1 | tee "${log}"
}

for ds in "${DATASETS[@]}"; do
  run_one "${ds}"
done

echo "All done. Logs in: ${OUTDIR}"
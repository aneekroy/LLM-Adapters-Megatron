#!/usr/bin/env bash
set -euo pipefail

# ====== Config ======
GPU="${GPU:-0}"  # override with: GPU=0 ./run_eval_math.sh
PY="${PY:-python}"
EVAL_PY="/home/aneek/LLM-Adapters/eval_vllm_ensemble_math_b.py"

MODEL_NAME="${MODEL_NAME:-Qwen3-4B-Sparse}"
BASE_MODEL="${BASE_MODEL:-/home/models/Qwen_Sparse/Qwen3-4B-Sparse-0.50}"

# Math datasets you’ve been using
DATASETS=(AddSub MultiArith SingleEq gsm8k AQuA SVAMP)

OUTDIR="${OUTDIR:-./eval_logs/Qwen3-4B-Sparse-0.50-FT-Math14k-rand50-MathEval}"
DRY_RUN="${DRY_RUN:-false}"   # set DRY_RUN=true to print commands only

mkdir -p "$OUTDIR"
ts(){ date +"%Y-%m-%d_%H-%M-%S"; }

run_one() {
  local ds="$1"
  echo "[${MODEL_NAME}] [$(ts)] dataset=${ds}"
  local log="${OUTDIR}/${MODEL_NAME}_${ds}_$(ts).log"

  if [[ "$DRY_RUN" == "true" ]]; then
    echo "CUDA_VISIBLE_DEVICES=${GPU} ${PY} ${EVAL_PY} --dataset ${ds} --model ${MODEL_NAME} --base_model \"${BASE_MODEL}\""
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${GPU}" \
    "${PY}" "${EVAL_PY}" \
      --dataset "${ds}" \
      --model "${MODEL_NAME}" \
      --base_model "${BASE_MODEL}" \
        --lora_weights "part1=/home/aneek/LLM-Adapters/trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.50-Math14k/Qwen3-4B-Sparse-0.50-rand50/round_0" \
        --tp_size 1 \
        --batch_size 16 \
        --gpu_memory_utilization 0.95 \
        --max_loras 3 --max_cpu_loras 3 --max_lora_rank 32 \
      2>&1 | tee "${log}"
}

for ds in "${DATASETS[@]}"; do
  run_one "${ds}"
done

echo "All done. Logs in: ${OUTDIR}"
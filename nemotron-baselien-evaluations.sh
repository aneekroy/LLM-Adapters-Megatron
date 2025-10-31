#!/usr/bin/env bash
set -euo pipefail

########################
# Config (edit as needed)
########################
GPU=3
PYTHON_BIN="/home/aneek/miniconda3/envs/eval_gpu/bin/python"   # <- use the env that has torch+vLLM
EVAL_PY="/home/aneek/LLM-Adapters/eval_llm.py"

# Roots where your fine-tuned LoRA adapters live
MODEL_ROOTS=(
  "/home/aneek/LLM-Adapters/trained_models/Qwen3"
  "/home/aneek/LLM-Adapters/trained_models/Qwen3_Sparse"
)

# Base HF model locations (frozen bases)
BASE_QWEN="/home/models/Qwen"
BASE_QWEN_SPARSE="/home/models/Qwen_Sparse"

DATASETS=(AddSub MultiArith SingleEq gsm8k AQuA SVAMP)

BATCH_SIZE_GEN=16
TP_SIZE=1
MAX_LEN=2048
GPU_MEM_UTIL=0.95

LOG_DIR="/home/aneek/LLM-Adapters/eval_logs_offline/Qwen3_all"
mkdir -p "$LOG_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ts(){ date +"%Y-%m-%d_%H-%M-%S"; }

echo "== Qwen3 Baseline Evaluations =="
echo "GPU=$GPU  PYTHON_BIN=$PYTHON_BIN  EVAL_PY=$EVAL_PY"
echo "Logging to: $LOG_DIR"
echo

# --- quick sanity check on the chosen interpreter ---
"$PYTHON_BIN" - <<'PY'
import sys
ok = True
try:
    import torch
    import vllm
except Exception as e:
    ok = False
    print("ERROR: Missing deps in this interpreter:", e, file=sys.stderr)
print("python:", sys.executable)
print("torch_ok:", 'torch' in sys.modules)
sys.exit(0 if ok else 1)
PY

for ROOT in "${MODEL_ROOTS[@]}"; do
  if [[ ! -d "$ROOT" ]]; then
    echo "WARN: Model root not found: $ROOT" >&2
    continue
  fi

  # Iterate subdirectories (each is one adapter checkpoint dir)
  while IFS= read -r -d '' MODEL_DIR; do
    MODEL_NAME="$(basename "$MODEL_DIR")"

    # Map adapter folder -> (--model, --base_model)
    MODEL_ID=""
    BASE_PATH=""

    case "$MODEL_NAME" in
      Qwen3-8B-Sparse-0.*)
        MODEL_ID="Qwen3-8B-Sparse"
        SPARSITY="$(sed -E 's/^Qwen3-8B-Sparse-0\.([0-9]{2}).*/0.\1/' <<<"$MODEL_NAME")"
        BASE_PATH="${BASE_QWEN_SPARSE}/Qwen3-8B-Sparse-${SPARSITY}"
        ;;
      Qwen3-8B-*)
        MODEL_ID="Qwen3-8B"
        BASE_PATH="${BASE_QWEN}/Qwen3-8B"
        ;;
      Qwen3-4B-*)
        MODEL_ID="Qwen3-4B"
        BASE_PATH="${BASE_QWEN}/Qwen3-4B-Instruct-2507"
        ;;
      *)
        echo "WARN: Unknown model family for '$MODEL_NAME' (skipping)"; continue ;;
    esac

    if [[ ! -d "$BASE_PATH" ]]; then
      echo "WARN: Base model path not found: $BASE_PATH (for $MODEL_NAME) — skipping." >&2
      continue
    fi

    MODEL_LOG_DIR="$LOG_DIR/$MODEL_NAME"
    mkdir -p "$MODEL_LOG_DIR"

    echo "---- Evaluating model: $MODEL_NAME"
    echo "     --model       : $MODEL_ID"
    echo "     --base_model  : $BASE_PATH"
    echo "     --lora_weights: $MODEL_DIR"
    for DS in "${DATASETS[@]}"; do
      RUN_TS="$(ts)"
      LOG_FILE="$MODEL_LOG_DIR/${DS}_${RUN_TS}.log"

      echo "  -> Dataset: $DS  | log: $LOG_FILE"
      CUDA_VISIBLE_DEVICES="$GPU" \
      PYTHONNOUSERSITE=1 \
      "$PYTHON_BIN" "$EVAL_PY" \
        --dataset "$DS" \
        --model "$MODEL_ID" \
        --base_model "$BASE_PATH" \
        --adapter "LoRA" \
        --lora_weights "$MODEL_DIR" \
        --batch_size_gen "$BATCH_SIZE_GEN" \
        --tp_size "$TP_SIZE" \
        --max_model_len "$MAX_LEN" \
        --gpu_mem_util "$GPU_MEM_UTIL" \
        2>&1 | tee "$LOG_FILE"
    done
    echo
  done < <(find "$ROOT" -mindepth 1 -maxdepth 1 -type d -print0)
done

echo "[DONE] All baseline evaluations completed."
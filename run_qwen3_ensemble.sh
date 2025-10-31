#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# ================== Toggles ==================
RUN_BASELINES=true
RUN_AL50=true
RUN_RAND50=true

# ================== Python / CUDA ==================
GPU=2
PY="/home/aneek/miniconda3/envs/sparsegpt/bin/python"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ================== Paths ==================
LOG_DIR="/home/aneek/LLM-Adapters/eval_logs/Qwen3_ensemble"
mkdir -p "$LOG_DIR"
DATASETS=(AddSub MultiArith SingleEq gsm8k AQuA SVAMP)

ENS_PY="/home/aneek/LLM-Adapters/eval_vllm_ensemble_math_b.py"
BATCH_SIZE=16
TP_SIZE=1
GPU_MEM_UTIL=0.95
MAX_LORA_RANK=32

# Base models
BASE_4B="/home/models/Qwen/Qwen3-4B-Instruct-2507"
MODEL_ID_4B="Qwen3-4B-Instruct"

SPARSE_BASE_ROOT="/home/models/Qwen_Sparse"
MODEL_ID_8B_SPARSE="Qwen3-8B-Sparse"

# Adapter roots
ENS_4B_ROOT="/home/aneek/LLM-Adapters/trained_models"
ENS_8B_SPARSE_ROOT="/home/aneek/LLM-Adapters/trained_models"

# Split → parts mapping
declare -A SPLIT_PARTS=( [25]=4 [33]=3 [50]=2)

# STRICT mapping: SplitXX -> 0.XX sparsity
declare -A SPLIT_TO_SPARSE=( [25]="0.25" [33]="0.33" [50]="0.50" )

# Utility: timestamp
ts(){ date +"%Y-%m-%d_%H-%M-%S"; }

# ---------- Preferred round mapping (per suffix) ----------
preferred_round_for_suffix() {
  case "$1" in
    -al50)   echo "2" ;;   # force round_2
    -rand50) echo "0" ;;   # force round_0
    *)       echo ""  ;;   # baseline: no preference
  esac
}

# ---------- Adapter resolver ----------
resolve_adapter_dir() {
  local root="$1"   # candidate dir that may contain adapter_config.json or round_*/
  local pref="$2"   # preferred round number

  if [[ -n "$pref" ]] && [[ -f "$root/round_${pref}/adapter_config.json" ]]; then
    echo "$root/round_${pref}"; return 0
  fi
  if [[ -f "$root/adapter_config.json" ]]; then
    [[ -n "$pref" ]] && echo "WARN: preferred round ${pref} not found for $root; using root" >&2
    echo "$root"; return 0
  fi

  local best="" best_n=-1 d n
  for d in "$root"/round_*; do
    [[ -d "$d" && -f "$d/adapter_config.json" ]] || continue
    n="${d##*/round_}"; [[ "$n" =~ ^[0-9]+$ ]] || n=-1
    if (( n > best_n )); then best="$d"; best_n=$n; fi
  done
  if [[ -n "$best" ]]; then
    [[ -n "$pref" ]] && echo "WARN: preferred round ${pref} not found for $root; using latest round_${best_n}" >&2
    echo "$best"; return 0
  fi

  if [[ -f "$root/adapter_model.bin" ]]; then
    [[ -n "$pref" ]] && echo "WARN: preferred round ${pref} not found for $root; no adapter_config.json; using root (bin only)" >&2
    echo "$root"; return 0
  fi

  return 1
}

# ---------- Build LoRA CSV ----------
# $1: dash pattern (printf-able) e.g. "...-part%d"
# $2: suffix ("" | "-al50" | "-rand50")
# $3: parts count
# $4: underscore fallback pattern (optional; used only when suffix is empty)
build_lora_list() {
  local pattern_dash="$1"
  local suffix="$2"
  local parts="$3"
  local fallback_pattern_underscore="${4:-}"
  local loras=()
  local pref_round; pref_round="$(preferred_round_for_suffix "$suffix")"

  local i p1 p2 resolved
  for ((i=1; i<=parts; i++)); do
    printf -v p1 "$pattern_dash" "$i"
    p1="${p1}${suffix}"

    if [[ -d "$p1" ]]; then
      if resolved="$(resolve_adapter_dir "$p1" "$pref_round")"; then
        loras+=("part${i}=${resolved}")
        continue
      fi
    fi

    if [[ -z "$suffix" && -n "$fallback_pattern_underscore" ]]; then
      printf -v p2 "$fallback_pattern_underscore" "$i"
      if [[ -d "$p2" ]]; then
        if resolved="$(resolve_adapter_dir "$p2" "")"; then
          loras+=("part${i}=${resolved}")
          continue
        fi
      fi
    fi

    echo "WARN: missing adapter for part $i (tried: $p1${fallback_pattern_underscore:+ , $(printf "$fallback_pattern_underscore" "$i")})" >&2
  done

  (IFS=,; echo "${loras[*]-}")
}

# ---------- Runner ----------
run_eval() {
  local model_id="$1"
  local base_path="$2"
  local loras_csv="$3"
  local dataset="$4"
  local log_file="$5"

  if [[ -z "$loras_csv" ]]; then
    echo "SKIP: no adapters for dataset=${dataset}; not invoking $ENS_PY (it requires --lora_weights)." | tee -a "$log_file"
    return 0
  fi

  "$PY" "$ENS_PY" \
    --dataset "$dataset" \
    --model "$model_id" \
    --base_model "$base_path" \
    --lora_weights "$loras_csv" \
    --batch_size "$BATCH_SIZE" \
    --tp_size "$TP_SIZE" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" \
    --max_loras 64 --max_cpu_loras 64 --max_lora_rank "$MAX_LORA_RANK" \
    2>&1 | tee "$log_file"
}

echo "== Qwen3 Ensemble/Baseline Evaluations =="
echo "Using Python: $PY"
$PY - <<'PY'
import sys
print("python:", sys.executable)
PY
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Logs -> $LOG_DIR"
echo

# ───────── 4B DENSE (ENSEMBLE) ─────────
# for SPLIT in 20 25 33 50; do
#   PARTS=${SPLIT_PARTS[$SPLIT]}
#   D_SPLIT="${ENS_4B_ROOT}/Qwen3_4B-Ensemble_Split${SPLIT}"   # dash root: base + al50 + rand50
#   U_SPLIT="${ENS_4B_ROOT}/Qwen3_4B_Ensemble_Split${SPLIT}"   # underscore root: plain parts

#   PAT_DASH="${D_SPLIT}/Qwen3-4B-2507-Math14k-part%d"
#   PAT_UND="${U_SPLIT}/Qwen3-4B-2507-Math14k-part%d"

#   # Baseline (skip if no adapters)
#   if [[ "$RUN_BASELINES" == "true" ]]; then
#     LORA_SPEC_BASE="$(build_lora_list "$PAT_DASH" "" "$PARTS" "$PAT_UND")"
#     if [[ -n "$LORA_SPEC_BASE" ]]; then
#       for DS in "${DATASETS[@]}"; do
#         RUN_TS="$(ts)"; LOG="$LOG_DIR/4B_baseline_split${SPLIT}_${DS}_${RUN_TS}.log"
#         echo "4B BASELINE | split=$SPLIT parts=$PARTS | dataset=$DS"
#         echo "  base=$BASE_4B"
#         echo "  loras=$LORA_SPEC_BASE"
#         run_eval "$MODEL_ID_4B" "$BASE_4B" "$LORA_SPEC_BASE" "$DS" "$LOG"
#         echo
#       done
#     else
#       echo "SKIP 4B BASELINE split=${SPLIT}: no adapters found."
#     fi
#   fi

#   # AL50 (force round_2)
#   if [[ "$RUN_AL50" == "true" ]]; then
#     LORA_SPEC_AL50="$(build_lora_list "$PAT_DASH" "-al50" "$PARTS")"
#     if [[ -n "$LORA_SPEC_AL50" ]]; then
#       for DS in "${DATASETS[@]}"; do
#         RUN_TS="$(ts)"; LOG="$LOG_DIR/4B_al50_split${SPLIT}_${DS}_${RUN_TS}.log"
#         echo "4B AL50 | split=$SPLIT parts=$PARTS | dataset=$DS"
#         echo "  base=$BASE_4B"
#         echo "  loras=$LORA_SPEC_AL50"
#         run_eval "$MODEL_ID_4B" "$BASE_4B" "$LORA_SPEC_AL50" "$DS" "$LOG"
#         echo
#       done
#     else
#       echo "SKIP 4B AL50 split=${SPLIT}: no adapters found."
#     fi
#   fi

#   # RAND50 (force round_0)
#   if [[ "$RUN_RAND50" == "true" ]]; then
#     LORA_SPEC_R50="$(build_lora_list "$PAT_DASH" "-rand50" "$PARTS")"
#     if [[ -n "$LORA_SPEC_R50" ]]; then
#       for DS in "${DATASETS[@]}"; do
#         RUN_TS="$(ts)"; LOG="$LOG_DIR/4B_rand50_split${SPLIT}_${DS}_${RUN_TS}.log"
#         echo "4B RAND50 | split=$SPLIT parts=$PARTS | dataset=$DS"
#         echo "  base=$BASE_4B"
#         echo "  loras=$LORA_SPEC_R50"
#         run_eval "$MODEL_ID_4B" "$BASE_4B" "$LORA_SPEC_R50" "$DS" "$LOG"
#         echo
#       done
#     else
#       echo "SKIP 4B RAND50 split=${SPLIT}: no adapters found."
#     fi
#   fi
# done

# ───────── 8B SPARSE (ENSEMBLE) ─────────
# Only run the sparsity that matches the split, e.g., Split25 -> 0.25
for SPLIT in 25 33 50; do
  PARTS=${SPLIT_PARTS[$SPLIT]}
  S="${SPLIT_TO_SPARSE[$SPLIT]}"

  BASE_8B_SPARSE="${SPARSE_BASE_ROOT}/Qwen3-8B-Sparse-${S}"
  if [[ ! -d "$BASE_8B_SPARSE" ]]; then
    echo "WARN: missing sparse base for split=${SPLIT} (expected ${BASE_8B_SPARSE}) — skipping this split."
    continue
  fi

  D_SPLIT="${ENS_8B_SPARSE_ROOT}/Qwen3_Sparse-Ensemble-Split-${SPLIT}" # al50/rand50
  U_SPLIT="${ENS_8B_SPARSE_ROOT}/Qwen3_Sparse_Ensemble_Split${SPLIT}"  # baseline

  PAT_DASH="${D_SPLIT}/Qwen3-8B-Sparse-${S}-Math14k-part%d"
  PAT_UND="${U_SPLIT}/Qwen3-8B-Sparse-${S}-Math14k-part%d"

  # Baseline (skip if no adapters)
  if [[ "$RUN_BASELINES" == "true" ]]; then
    LORA_SPEC_BASE="$(build_lora_list "$PAT_DASH" "" "$PARTS" "$PAT_UND")"
    if [[ -n "$LORA_SPEC_BASE" ]]; then
      for DS in "${DATASETS[@]}"; do
        RUN_TS="$(ts)"; LOG="$LOG_DIR/8B_sparse${S}_baseline_split${SPLIT}_${DS}_${RUN_TS}.log"
        echo "8B-SPARSE ${S} BASELINE | split=$SPLIT parts=$PARTS | dataset=$DS"
        echo "  base=$BASE_8B_SPARSE"
        echo "  loras=$LORA_SPEC_BASE"
        run_eval "$MODEL_ID_8B_SPARSE" "$BASE_8B_SPARSE" "$LORA_SPEC_BASE" "$DS" "$LOG"
        echo
      done
    else
      echo "SKIP 8B-SPARSE ${S} BASELINE split=${SPLIT}: no adapters found."
    fi
  fi

  # AL50 (force round_2)
  if [[ "$RUN_AL50" == "true" ]]; then
    LORA_SPEC_AL50="$(build_lora_list "$PAT_DASH" "-al50" "$PARTS")"
    if [[ -n "$LORA_SPEC_AL50" ]]; then
      for DS in "${DATASETS[@]}"; do
        RUN_TS="$(ts)"; LOG="$LOG_DIR/8B_sparse${S}_al50_split${SPLIT}_${DS}_${RUN_TS}.log"
        echo "8B-SPARSE ${S} AL50 | split=$SPLIT parts=$PARTS | dataset=$DS"
        echo "  base=$BASE_8B_SPARSE"
        echo "  loras=$LORA_SPEC_AL50"
        run_eval "$MODEL_ID_8B_SPARSE" "$BASE_8B_SPARSE" "$LORA_SPEC_AL50" "$DS" "$LOG"
        echo
      done
    else
      echo "SKIP 8B-SPARSE ${S} AL50 split=${SPLIT}: no adapters found."
    fi
  fi

  # RAND50 (force round_0)
  if [[ "$RUN_RAND50" == "true" ]]; then
    LORA_SPEC_R50="$(build_lora_list "$PAT_DASH" "-rand50" "$PARTS")"
    if [[ -n "$LORA_SPEC_R50" ]]; then
      for DS in "${DATASETS[@]}"; do
        RUN_TS="$(ts)"; LOG="$LOG_DIR/8B_sparse${S}_rand50_split${SPLIT}_${DS}_${RUN_TS}.log"
        echo "8B-SPARSE ${S} RAND50 | split=$SPLIT parts=$PARTS | dataset=$DS"
        echo "  base=$BASE_8B_SPARSE"
        echo "  loras=$LORA_SPEC_R50"
        run_eval "$MODEL_ID_8B_SPARSE" "$BASE_8B_SPARSE" "$LORA_SPEC_R50" "$DS" "$LOG"
        echo
      done
    else
      echo "SKIP 8B-SPARSE ${S} RAND50 split=${SPLIT}: no adapters found."
    fi
  fi
done

echo "[DONE] All requested evaluations dispatched."
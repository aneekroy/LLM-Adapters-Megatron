#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# ================== Toggles ==================
RUN_14B_SPLIT_AL50=false
RUN_14B_SPLIT_RAND50=false
RUN_14B_FULL_AL50=false
RUN_14B_FULL_RAND50=false

RUN_7B_SPLIT_AL50=true
RUN_7B_SPLIT_RAND50=true

RUN_BASELINES=false  # part* (no -al50/-rand50) are full-FT checkpoints -> not LoRA

# ================== Python / CUDA ==================
GPU=2
PY="/home/aneek/miniconda3/envs/sparsegpt/bin/python"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ================== Eval ==================
ENS_PY="/home/aneek/LLM-Adapters/eval_vllm_ensemble_math_b.py"
DATASETS=(AddSub MultiArith SingleEq gsm8k AQuA SVAMP)
BATCH_SIZE=16
TP_SIZE=1
GPU_MEM_UTIL=0.98
MAX_LORA_RANK=32

LOG_DIR="/home/aneek/LLM-Adapters/eval_logs/Nemotron_ensemble"
mkdir -p "$LOG_DIR"

# ================== Bases / IDs ==================
# 14B Sparse bases (e.g., /home/models/nvidia-sparse/OpenReasoning-Nemotron-14B-Sparse-0.25)
SPARSE_BASE_ROOT="/home/models/nvidia-sparse"
MODEL_ID_14B_SPARSE="Nemotron-14B-Sparse"

# 7B Dense base (from your earlier runs)
BASE_7B="/home/models/nvidia/OpenReasoning-Nemotron-7B"
MODEL_ID_7B="Nemotron-7B"

# ================== Trained adapters roots ==================
TRAINED_ROOT="/home/aneek/LLM-Adapters/trained_models"

# Split -> parts
declare -A SPLIT_PARTS=( [20]=5 [25]=4 [33]=3 [50]=2 )

# Split -> sparsity (for 14B-Sparse only)
declare -A SPLIT_TO_SPARSE=( [20]="0.20" [25]="0.25" [33]="0.33" [50]="0.50" )

# Timestamp
ts(){ date +"%Y-%m-%d_%H-%M-%S"; }

# ---------- Round preference ----------
preferred_round_for_suffix() {
  case "$1" in
    -al50)   echo "2" ;;   # AL50 -> round_2
    -rand50) echo "0" ;;   # RAND50 -> round_0
    *)       echo ""  ;;
  esac
}

# ---------- Adapter resolver ----------
resolve_adapter_dir() {
  local root="$1"   # containing adapter_config.json or round_*/
  local pref="$2"   # preferred round number

  if [[ -n "$pref" ]] && [[ -f "$root/round_${pref}/adapter_config.json" ]]; then
    echo "$root/round_${pref}"; return 0
  fi
  if [[ -f "$root/adapter_config.json" ]]; then
    [[ -n "$pref" ]] && echo "WARN: preferred round ${pref} not found at $root; using root" >&2
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

# ---------- Build LoRA CSV from split parts ----------
# $1: printf pattern like "...-part%d"
# $2: suffix: "" | "-al50" | "-rand50"
# $3: parts count
build_lora_list() {
  local pattern_dash="$1" suffix="$2" parts="$3"
  local loras=()
  local pref_round; pref_round="$(preferred_round_for_suffix "$suffix")"

  local i p resolved
  for ((i=1; i<=parts; i++)); do
    printf -v p "$pattern_dash" "$i"
    p="${p}${suffix}"
    if [[ -d "$p" ]]; then
      if resolved="$(resolve_adapter_dir "$p" "$pref_round")"; then
        loras+=("part${i}=${resolved}")
        continue
      fi
    fi
    echo "WARN: missing adapter for part $i (tried: $p)" >&2
  done
  (IFS=,; echo "${loras[*]-}")
}

# ---------- Build LoRA CSV for full-dataset single adapter ----------
# $1: absolute dir (…-al50 or …-rand50), $2: suffix (for round preference)
build_full_lora() {
  local root="$1" suffix="$2"
  local pref_round; pref_round="$(preferred_round_for_suffix "$suffix")"
  if resolved="$(resolve_adapter_dir "$root" "$pref_round")"; then
    echo "part1=${resolved}"
  else
    echo ""
  fi
}

# ---------- Runner ----------
run_eval() {
  local model_id="$1" base_path="$2" loras_csv="$3" dataset="$4" log_file="$5"

  if [[ -z "$loras_csv" ]]; then
    echo "SKIP: no adapters for dataset=${dataset}; not invoking $ENS_PY (requires --lora_weights)." | tee -a "$log_file"
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

echo "== Nemotron Ensemble Evaluations (14B-Sparse + 7B-Dense) =="
$PY - <<'PY'
import sys; print("python:", sys.executable)
PY
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Logs -> $LOG_DIR"
echo

# ───────── 14B SPARSE: split ensembles ─────────
# for SPLIT in 20 25 33 50; do
#   PARTS=${SPLIT_PARTS[$SPLIT]}
#   S="${SPLIT_TO_SPARSE[$SPLIT]}"

#   BASE_14B_SPARSE="${SPARSE_BASE_ROOT}/OpenReasoning-Nemotron-14B-Sparse-${S}"
#   if [[ ! -d "$BASE_14B_SPARSE" ]]; then
#     echo "WARN: missing base ${BASE_14B_SPARSE} — skip split ${SPLIT}."
#     continue
#   fi

#   SPLIT_ROOT_14B="${TRAINED_ROOT}/OpenReasoning-Nemotron-14B-Sparse-${S}-Ensemble_split_${SPLIT}"
#   PAT_14B="${SPLIT_ROOT_14B}/OpenReasoning-Nemotron-14B-Sparse-${S}-Math14k-part%d"

#   if [[ "$RUN_BASELINES" == "true" ]]; then
#     LORA_SPEC_BASE="$(build_lora_list "$PAT_14B" "" "$PARTS")"
#     if [[ -n "$LORA_SPEC_BASE" ]]; then
#       for DS in "${DATASETS[@]}"; do
#         RUN_TS="$(ts)"; LOG="$LOG_DIR/14B_sparse${S}_BASE_split${SPLIT}_${DS}_${RUN_TS}.log"
#         echo "14B-SPARSE ${S} BASE | split=$SPLIT parts=$PARTS | dataset=$DS"
#         run_eval "$MODEL_ID_14B_SPARSE" "$BASE_14B_SPARSE" "$LORA_SPEC_BASE" "$DS" "$LOG"; echo
#       done
#     else
#       echo "SKIP 14B BASE split=${SPLIT}: no LoRA adapters."
#     fi
#   fi

#   if [[ "$RUN_14B_SPLIT_AL50" == "true" ]]; then
#     LORA_SPEC_AL50="$(build_lora_list "$PAT_14B" "-al50" "$PARTS")"
#     if [[ -n "$LORA_SPEC_AL50" ]]; then
#       for DS in "${DATASETS[@]}"; do
#         RUN_TS="$(ts)"; LOG="$LOG_DIR/14B_sparse${S}_AL50_split${SPLIT}_${DS}_${RUN_TS}.log"
#         echo "14B-SPARSE ${S} AL50 | split=$SPLIT parts=$PARTS | dataset=$DS"
#         run_eval "$MODEL_ID_14B_SPARSE" "$BASE_14B_SPARSE" "$LORA_SPEC_AL50" "$DS" "$LOG"; echo
#       done
#     else
#       echo "SKIP 14B AL50 split=${SPLIT}: no adapters."
#     fi
#   fi

#   if [[ "$RUN_14B_SPLIT_RAND50" == "true" ]]; then
#     LORA_SPEC_R50="$(build_lora_list "$PAT_14B" "-rand50" "$PARTS")"
#     if [[ -n "$LORA_SPEC_R50" ]]; then
#       for DS in "${DATASETS[@]}"; do
#         RUN_TS="$(ts)"; LOG="$LOG_DIR/14B_sparse${S}_RAND50_split${SPLIT}_${DS}_${RUN_TS}.log"
#         echo "14B-SPARSE ${S} RAND50 | split=$SPLIT parts=$PARTS | dataset=$DS"
#         run_eval "$MODEL_ID_14B_SPARSE" "$BASE_14B_SPARSE" "$LORA_SPEC_R50" "$DS" "$LOG"; echo
#       done
#     else
#       echo "SKIP 14B RAND50 split=${SPLIT}: no adapters."
#     fi
#   fi
# done

# ───────── 14B SPARSE: full-dataset single-adapter runs ─────────
FULL_ROOT_14B="${TRAINED_ROOT}/OpenReasoning-Nemotron-14B-Sparse-Ensemble_fullMath14k"
for SPLIT in 20 25 33 50; do
  S="${SPLIT_TO_SPARSE[$SPLIT]}"
  BASE_14B_SPARSE="${SPARSE_BASE_ROOT}/OpenReasoning-Nemotron-14B-Sparse-${S}"
  [[ -d "$BASE_14B_SPARSE" ]] || { echo "WARN: missing base ${BASE_14B_SPARSE} — skip full ${S}."; continue; }

  FULL_AL50_DIR="${FULL_ROOT_14B}/OpenReasoning-Nemotron-14B-Sparse-${S}-Math14k-al50"
  FULL_R50_DIR="${FULL_ROOT_14B}/OpenReasoning-Nemotron-14B-Sparse-${S}-Math14k-rand50"

  if [[ "$RUN_14B_FULL_AL50" == "true" ]]; then
    LORA_FULL_AL50="$(build_full_lora "$FULL_AL50_DIR" "-al50")"
    if [[ -n "$LORA_FULL_AL50" ]]; then
      for DS in "${DATASETS[@]}"; do
        RUN_TS="$(ts)"; LOG="$LOG_DIR/14B_sparse${S}_FULL_AL50_${DS}_${RUN_TS}.log"
        echo "14B-SPARSE ${S} FULL AL50 | dataset=$DS"
        run_eval "$MODEL_ID_14B_SPARSE" "$BASE_14B_SPARSE" "$LORA_FULL_AL50" "$DS" "$LOG"; echo
      done
    else
      echo "SKIP 14B FULL AL50 ${S}: adapter not found (or no round_2)."
    fi
  fi

  if [[ "$RUN_14B_FULL_RAND50" == "true" ]]; then
    LORA_FULL_R50="$(build_full_lora "$FULL_R50_DIR" "-rand50")"
    if [[ -n "$LORA_FULL_R50" ]]; then
      for DS in "${DATASETS[@]}"; do
        RUN_TS="$(ts)"; LOG="$LOG_DIR/14B_sparse${S}_FULL_RAND50_${DS}_${RUN_TS}.log"
        echo "14B-SPARSE ${S} FULL RAND50 | dataset=$DS"
        run_eval "$MODEL_ID_14B_SPARSE" "$BASE_14B_SPARSE" "$LORA_FULL_R50" "$DS" "$LOG"; echo
      done
    else
      echo "SKIP 14B FULL RAND50 ${S}: adapter not found (or no round_0)."
    fi
  fi
done

# ───────── 7B DENSE: split ensembles ─────────
# Layout:
# trained_models/OpenReasoning-Nemotron-7B-Ensemble_split_${SPLIT}/OpenReasoning-Nemotron-7B-Math14k-partN-{al50,rand50}/round_{0,1,2}
if [[ -d "$BASE_7B" ]]; then
  for SPLIT in 20 25 33 50; do
    PARTS=${SPLIT_PARTS[$SPLIT]}
    SPLIT_ROOT_7B="${TRAINED_ROOT}/OpenReasoning-Nemotron-7B-Ensemble_split_${SPLIT}"
    PAT_7B="${SPLIT_ROOT_7B}/OpenReasoning-Nemotron-7B-Math14k-part%d"

    if [[ "$RUN_BASELINES" == "true" ]]; then
      LORA_SPEC_BASE="$(build_lora_list "$PAT_7B" "" "$PARTS")"
      if [[ -n "$LORA_SPEC_BASE" ]]; then
        for DS in "${DATASETS[@]}"; do
          RUN_TS="$(ts)"; LOG="$LOG_DIR/7B_BASE_split${SPLIT}_${DS}_${RUN_TS}.log"
          echo "7B BASE | split=$SPLIT parts=$PARTS | dataset=$DS"
          run_eval "$MODEL_ID_7B" "$BASE_7B" "$LORA_SPEC_BASE" "$DS" "$LOG"; echo
        done
      else
        echo "SKIP 7B BASE split=${SPLIT}: no LoRA adapters (likely full-FT checkpoints only)."
      fi
    fi

    if [[ "$RUN_7B_SPLIT_AL50" == "true" ]]; then
      LORA_SPEC_AL50="$(build_lora_list "$PAT_7B" "-al50" "$PARTS")"
      if [[ -n "$LORA_SPEC_AL50" ]]; then
        for DS in "${DATASETS[@]}"; do
          RUN_TS="$(ts)"; LOG="$LOG_DIR/7B_AL50_split${SPLIT}_${DS}_${RUN_TS}.log"
          echo "7B AL50 | split=$SPLIT parts=$PARTS | dataset=$DS"
          run_eval "$MODEL_ID_7B" "$BASE_7B" "$LORA_SPEC_AL50" "$DS" "$LOG"; echo
        done
      else
        echo "SKIP 7B AL50 split=${SPLIT}: no adapters."
      fi
    fi

    if [[ "$RUN_7B_SPLIT_RAND50" == "true" ]]; then
      LORA_SPEC_R50="$(build_lora_list "$PAT_7B" "-rand50" "$PARTS")"
      if [[ -n "$LORA_SPEC_R50" ]]; then
        for DS in "${DATASETS[@]}"; do
          RUN_TS="$(ts)"; LOG="$LOG_DIR/7B_RAND50_split${SPLIT}_${DS}_${RUN_TS}.log"
          echo "7B RAND50 | split=$SPLIT parts=$PARTS | dataset=$DS"
          run_eval "$MODEL_ID_7B" "$BASE_7B" "$LORA_SPEC_R50" "$DS" "$LOG"; echo
        done
      else
        echo "SKIP 7B RAND50 split=${SPLIT}: no adapters."
      fi
    fi
  done
else
  echo "WARN: 7B base not found at $BASE_7B — skipping all 7B runs."
fi

echo "[DONE] Nemotron ensemble evaluations dispatched."
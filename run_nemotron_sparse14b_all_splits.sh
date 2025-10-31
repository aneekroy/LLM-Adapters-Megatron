#!/usr/bin/env bash
set -Eeuo pipefail

# ===== GPU selection =====
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1

# ===== WANDB (offline; sync later with `wandb sync`) =====
export WANDB_MODE=offline
export WANDB_PROJECT="huggingface"
# export WANDB_ENTITY="your-entity"

# ===== Paths =====
SP_ROOT="/home/models/nvidia-sparse"
TRAINED_ROOT="/home/aneek/LLM-Adapters/trained_models"
FT_ROOT="/home/aneek/LLM-Adapters/ft-training_set"
PY_FT="finetune.py"
PY_AL="active_learning_fast.py"

# ===== Which sparsity levels to run (skip 0.20: assumed done) =====
SP_LEVELS=("0.25" "0.33" "0.50")

# ===== Split mapping and part counts =====
declare -A SP2SPLIT=( ["0.25"]="split_25" ["0.33"]="split_33" ["0.50"]="split_50" )
declare -A SPLIT_PARTS=( ["split_25"]=4 ["split_33"]=3 ["split_50"]=2 )

# ===== Train hyperparams (FT = full finetune) =====
FT_BATCH_SIZE=4
FT_MICRO_BATCH_SIZE=1
FT_NUM_EPOCHS=3
FT_LR=3e-5
FT_CUTOFF_LEN=256
FT_VAL_SIZE=120

# ===== Active Learning configs =====
# rand50: label ~50% in one shot
AL_RAND_ROUNDS=1
AL_RAND_INIT=0.5
AL_RAND_ACQ=0.1

# al50: 10% seed + 2x20% acquisitions
AL_AL50_ROUNDS=3
AL_AL50_INIT=0.1
AL_AL50_ACQ=0.2

AL_UNCERT="logppl"
AL_SCORE_BS=8
AL_NUM_EPOCHS="${FT_NUM_EPOCHS}"
AL_LR="${FT_LR}"
AL_BATCH_SIZE="${FT_BATCH_SIZE}"
AL_MICRO_BATCH_SIZE="${FT_MICRO_BATCH_SIZE}"
AL_VAL_SIZE="${FT_VAL_SIZE}"
AL_CUTOFF_LEN="${FT_CUTOFF_LEN}"

# ===== Stage selector (comma-separated: ft,rand50,al50). Default: all. =====
STAGES="${STAGES:-ft,rand50,al50}"

ts(){ date +"%Y-%m-%d_%H-%M-%S"; }
log(){ echo "[$(ts)] $*"; }

# DRY_RUN=1 to only print commands
DRY_RUN="${DRY_RUN:-0}"
run(){
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY-RUN] '; printf '%q ' "$@"; printf '\n'
  else
    "$@"
  fi
}

need_stage(){ [[ ",${STAGES}," == *",$1,"* ]]; }

# ===== Pre-flight checks =====
[[ -x "$(command -v python)" ]] || { echo "python not found"; exit 1; }
[[ -f "${PY_FT}" ]] || { echo "Missing ${PY_FT} in $(pwd)"; exit 1; }
[[ -f "${PY_AL}" ]] || { echo "Missing ${PY_AL} in $(pwd)"; exit 1; }

trap 'echo "[INFO] done. To upload W&B offline runs: wandb sync --sync-all"' EXIT

# ===== Main =====
for SP in "${SP_LEVELS[@]}"; do
  SPLIT="${SP2SPLIT[$SP]:-}"
  [[ -n "${SPLIT}" ]] || { echo "No split mapping for SP=${SP}"; exit 1; }
  PARTS="${SPLIT_PARTS[$SPLIT]:-}"
  [[ -n "${PARTS}" ]] || { echo "No PARTS count for ${SPLIT}"; exit 1; }

  BASE_MODEL="${SP_ROOT}/OpenReasoning-Nemotron-14B-Sparse-${SP}"
  [[ -d "${BASE_MODEL}" ]] || { echo "Base model dir not found: ${BASE_MODEL}"; exit 1; }
  MODEL_SLUG="$(basename "${BASE_MODEL}")"

  OUT_ROOT="${TRAINED_ROOT}/${MODEL_SLUG}-Ensemble_${SPLIT}"
  LOG_DIR="${OUT_ROOT}/logs"
  run mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

  log "====== MODEL=${MODEL_SLUG} | SPLIT=${SPLIT} (parts=${PARTS}) | STAGES=${STAGES} ======"

  for ((p=1; p<=PARTS; p++)); do
    PART_JSON="${FT_ROOT}/${SPLIT}/math_14k_part${p}_of_${PARTS}.json"
    [[ -f "${PART_JSON}" ]] || { echo "Missing split file: ${PART_JSON}"; exit 1; }

    PART_TAG="part${p}"
    log "----- ${MODEL_SLUG} :: ${SPLIT} :: ${PART_TAG} -----"

    ########################################
    # (1) FULL FINETUNE ON THE WHOLE PART #
    ########################################
    if need_stage ft; then
      FT_OUT="${OUT_ROOT}/${MODEL_SLUG}-Math14k-${PART_TAG}"
      FT_DONE="${FT_OUT}/.done"
      FT_LOG="${LOG_DIR}/$(ts)_finetune_${MODEL_SLUG}_${SPLIT}_${PART_TAG}.log"
      FT_RUN="FT-${MODEL_SLUG}-${SPLIT}-${PART_TAG}"

      if [[ -f "${FT_DONE}" ]]; then
        log "SKIP (done): FT :: ${MODEL_SLUG} :: ${SPLIT} :: ${PART_TAG}"
      else
        log "RUN  : FT  :: ${MODEL_SLUG} :: ${SPLIT} :: ${PART_TAG}"
        run mkdir -p "${FT_OUT}"
        if [[ "${DRY_RUN}" == "1" ]]; then
          printf '[DRY-RUN] python %q \\\n' "${PY_FT}"
          cat <<EOF
  --base_model "${BASE_MODEL}" \
  --data_path "${PART_JSON}" \
  --output_dir "${FT_OUT}" \
  --batch_size "${FT_BATCH_SIZE}" \
  --micro_batch_size "${FT_MICRO_BATCH_SIZE}" \
  --num_epochs "${FT_NUM_EPOCHS}" \
  --learning_rate "${FT_LR}" \
  --cutoff_len "${FT_CUTOFF_LEN}" \
  --val_set_size "${FT_VAL_SIZE}" \
  --wandb_run_name "${FT_RUN}"
EOF
        else
          set +e
          python "${PY_FT}" \
            --base_model "${BASE_MODEL}" \
            --data_path "${PART_JSON}" \
            --output_dir "${FT_OUT}" \
            --batch_size "${FT_BATCH_SIZE}" \
            --micro_batch_size "${FT_MICRO_BATCH_SIZE}" \
            --num_epochs "${FT_NUM_EPOCHS}" \
            --learning_rate "${FT_LR}" \
            --cutoff_len "${FT_CUTOFF_LEN}" \
            --val_set_size "${FT_VAL_SIZE}" \
            --wandb_run_name "${FT_RUN}" \
            2>&1 | tee "${FT_LOG}"
          rc=$?; set -e
          if [[ $rc -eq 0 ]]; then
            run touch "${FT_DONE}"
            log "SUCCESS: FT :: ${MODEL_SLUG} :: ${SPLIT} :: ${PART_TAG}"
          else
            echo "[ERROR] FT failed (rc=$rc). See: ${FT_LOG}" >&2
            exit $rc
          fi
        fi
      fi
    fi

    ###########################
    # (2) ACTIVE: rand50     #
    ###########################
    if need_stage rand50; then
      AL_RAND_OUT="${OUT_ROOT}/${MODEL_SLUG}-Math14k-${PART_TAG}-rand50"
      AL_RAND_DONE="${AL_RAND_OUT}/.done"
      AL_RAND_LOG="${LOG_DIR}/$(ts)_al_rand50_${MODEL_SLUG}_${SPLIT}_${PART_TAG}.log"
      AL_RAND_RUN="AL-rand50-${MODEL_SLUG}-${SPLIT}-${PART_TAG}"

      if [[ -f "${AL_RAND_DONE}" ]]; then
        log "SKIP (done): rand50 :: ${MODEL_SLUG} :: ${SPLIT} :: ${PART_TAG}"
      else
        log "RUN  : rand50 :: ${MODEL_SLUG} :: ${SPLIT} :: ${PART_TAG}"
        run mkdir -p "${AL_RAND_OUT}"
        if [[ "${DRY_RUN}" == "1" ]]; then
          printf '[DRY-RUN] python %q \\\n' "${PY_AL}"
          cat <<EOF
  --base_model "${BASE_MODEL}" \
  --data_path "${PART_JSON}" \
  --output_dir "${AL_RAND_OUT}" \
  --rounds "${AL_RAND_ROUNDS}" \
  --init_frac "${AL_RAND_INIT}" \
  --acq_frac "${AL_RAND_ACQ}" \
  --uncertainty ${AL_UNCERT} \
  --cutoff_len "${AL_CUTOFF_LEN}" \
  --scoring_batch_size ${AL_SCORE_BS} \
  --num_epochs "${AL_NUM_EPOCHS}" \
  --learning_rate "${AL_LR}" \
  --per_device_train_batch_size "${AL_BATCH_SIZE}" \
  --micro_batch_size "${AL_MICRO_BATCH_SIZE}" \
  --val_set_size "${AL_VAL_SIZE}" \
  --wandb_run_name "${AL_RAND_RUN}"
EOF
        else
          set +e
          python "${PY_AL}" \
            --base_model "${BASE_MODEL}" \
            --data_path "${PART_JSON}" \
            --output_dir "${AL_RAND_OUT}" \
            --rounds "${AL_RAND_ROUNDS}" \
            --init_frac "${AL_RAND_INIT}" \
            --acq_frac "${AL_RAND_ACQ}" \
            --uncertainty "${AL_UNCERT}" \
            --cutoff_len "${AL_CUTOFF_LEN}" \
            --scoring_batch_size "${AL_SCORE_BS}" \
            --num_epochs "${AL_NUM_EPOCHS}" \
            --learning_rate "${AL_LR}" \
            --per_device_train_batch_size "${AL_BATCH_SIZE}" \
            --micro_batch_size "${AL_MICRO_BATCH_SIZE}" \
            --val_set_size "${AL_VAL_SIZE}" \
            --wandb_run_name "${AL_RAND_RUN}" \
            2>&1 | tee "${AL_RAND_LOG}"
          rc=$?; set -e
          if [[ $rc -eq 0 ]]; then
            run touch "${AL_RAND_DONE}"
            log "SUCCESS: rand50 :: ${MODEL_SLUG} :: ${SPLIT} :: ${PART_TAG}"
          else
            echo "[ERROR] rand50 failed (rc=$rc). See: ${AL_RAND_LOG}" >&2
            exit $rc
          fi
        fi
      fi
    fi

#     ###########################
#     # (3) ACTIVE: al50       #
#     ###########################
#     if need_stage al50; then
#       AL_AL50_OUT="${OUT_ROOT}/${MODEL_SLUG}-Math14k-${PART_TAG}-al50"
#       AL_AL50_DONE="${AL_AL50_OUT}/.done"
#       AL_AL50_LOG="${LOG_DIR}/$(ts)_al_al50_${MODEL_SLUG}_${SPLIT}_${PART_TAG}.log"
#       AL_AL50_RUN="AL-al50-${MODEL_SLUG}-${SPLIT}-${PART_TAG}"

#       if [[ -f "${AL_AL50_DONE}" ]]; then
#         log "SKIP (done): al50 :: ${MODEL_SLUG} :: ${SPLIT} :: ${PART_TAG}"
#       else
#         log "RUN  : al50 :: ${MODEL_SLUG} :: ${SPLIT} :: ${PART_TAG}"
#         run mkdir -p "${AL_AL50_OUT}"
#         if [[ "${DRY_RUN}" == "1" ]]; then
#           printf '[DRY-RUN] python %q \\\n' "${PY_AL}"
#           cat <<EOF
#   --base_model "${BASE_MODEL}" \
#   --data_path "${PART_JSON}" \
#   --output_dir "${AL_AL50_OUT}" \
#   --rounds "${AL_AL50_ROUNDS}" \
#   --init_frac "${AL_AL50_INIT}" \
#   --acq_frac "${AL_AL50_ACQ}" \
#   --uncertainty ${AL_UNCERT} \
#   --cutoff_len "${AL_CUTOFF_LEN}" \
#   --scoring_batch_size ${AL_SCORE_BS} \
#   --num_epochs "${AL_NUM_EPOCHS}" \
#   --learning_rate "${AL_LR}" \
#   --per_device_train_batch_size "${AL_BATCH_SIZE}" \
#   --micro_batch_size "${AL_MICRO_BATCH_SIZE}" \
#   --val_set_size "${AL_VAL_SIZE}" \
#   --wandb_run_name "${AL_AL50_RUN}"
# EOF
#         else
#           set +e
#           python "${PY_AL}" \
#             --base_model "${BASE_MODEL}" \
#             --data_path "${PART_JSON}" \
#             --output_dir "${AL_AL50_OUT}" \
#             --rounds "${AL_AL50_ROUNDS}" \
#             --init_frac "${AL_AL50_INIT}" \
#             --acq_frac "${AL_AL50_ACQ}" \
#             --uncertainty "${AL_UNCERT}" \
#             --cutoff_len "${AL_CUTOFF_LEN}" \
#             --scoring_batch_size "${AL_SCORE_BS}" \
#             --num_epochs "${AL_NUM_EPOCHS}" \
#             --learning_rate "${AL_LR}" \
#             --per_device_train_batch_size "${AL_BATCH_SIZE}" \
#             --micro_batch_size "${AL_MICRO_BATCH_SIZE}" \
#             --val_set_size "${AL_VAL_SIZE}" \
#             --wandb_run_name "${AL_AL50_RUN}" \
#             2>&1 | tee "${AL_AL50_LOG}"
#           rc=$?; set -e
#           if [[ $rc -eq 0 ]]; then
#             run touch "${AL_AL50_DONE}"
#             log "SUCCESS: al50 :: ${MODEL_SLUG} :: ${SPLIT} :: ${PART_TAG}"
#           else
#             echo "[ERROR] al50 failed (rc=$rc). See: ${AL_AL50_LOG}" >&2
#             exit $rc
#           fi
#         fi
#       fi
#     fi

  done
done

echo "[DONE] FT+rand50+al50 complete for SP in {0.25,0.33,0.50}. To upload W&B: wandb sync --sync-all"

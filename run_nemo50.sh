#!/usr/bin/env bash
set -Eeuo pipefail

export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
export WANDB_MODE=offline
export WANDB_PROJECT="huggingface"
export VLLM_USE_FLASHINFER=0                                                                          
export VLLM_LOGGING_LEVEL=WARNING

SP_ROOT="/home/models/nvidia-sparse"
BASE_DENSE="/home/models/nvidia/OpenReasoning-Nemotron-14B"   # tokenizer source
TRAINED_ROOT="/home/aneek/LLM-Adapters/trained_models"
FT_ROOT="/home/aneek/LLM-Adapters/ft-training_set"
PY_FT="finetune.py"
PY_AL="active_learning_fast.py"

# Run only 0.50
SP_LEVELS=("0.50")
declare -A SP2SPLIT=( ["0.50"]="split_50" )
declare -A SPLIT_PARTS=( ["split_50"]=2 )

FT_BATCH_SIZE=4
FT_MICRO_BATCH_SIZE=1
FT_NUM_EPOCHS=3
FT_LR=3e-5
FT_CUTOFF_LEN=256
FT_VAL_SIZE=120

AL_RAND_ROUNDS=1
AL_RAND_INIT=0.5
AL_RAND_ACQ=0.1

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

STAGES="ft,rand50,al50"   # force all three
ts(){ date +"%Y-%m-%d_%H-%M-%S"; }
log(){ echo "[$(ts)] $*"; }
DRY_RUN="${DRY_RUN:-0}"
run(){ if [[ "$DRY_RUN" == 1 ]]; then printf '[DRY-RUN] '; printf '%q ' "$@"; printf '\n'; else "$@"; fi; }
need_stage(){ [[ ",${STAGES}," == *",$1,"* ]]; }

[[ -x "$(command -v python)" ]] || { echo "python not found"; exit 1; }
[[ -f "${PY_FT}" ]] || { echo "Missing ${PY_FT}"; exit 1; }
[[ -f "${PY_AL}" ]] || { echo "Missing ${PY_AL}"; exit 1; }

trap 'echo "[INFO] done. To upload W&B offline runs: wandb sync --sync-all"' EXIT

for SP in "${SP_LEVELS[@]}"; do
  SPLIT="${SP2SPLIT[$SP]}"
  PARTS="${SPLIT_PARTS[$SPLIT]}"
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

    # # (1) FT
    # if need_stage ft; then
    #   FT_OUT="${OUT_ROOT}/${MODEL_SLUG}-Math14k-${PART_TAG}"
    #   FT_DONE="${FT_OUT}/.done"
    #   FT_LOG="${LOG_DIR}/$(ts)_finetune_${MODEL_SLUG}_${SPLIT}_${PART_TAG}.log"
    #   FT_RUN="FT-${MODEL_SLUG}-${SPLIT}-${PART_TAG}"
    #   if [[ -f "${FT_DONE}" ]]; then
    #     log "SKIP (done): FT :: ${PART_TAG}"
    #   else
    #     log "RUN  : FT  :: ${PART_TAG}"
    #     run mkdir -p "${FT_OUT}"
    #     set +e
    #     python "${PY_FT}" \
    #       --base_model "${BASE_MODEL}" \
    #       --tokenizer_path "${BASE_DENSE}" \
    #       --data_path "${PART_JSON}" \
    #       --output_dir "${FT_OUT}" \
    #       --batch_size "${FT_BATCH_SIZE}" \
    #       --micro_batch_size "${FT_MICRO_BATCH_SIZE}" \
    #       --num_epochs "${FT_NUM_EPOCHS}" \
    #       --learning_rate "${FT_LR}" \
    #       --cutoff_len "${FT_CUTOFF_LEN}" \
    #       --val_set_size "${FT_VAL_SIZE}" \
    #       --wandb_run_name "${FT_RUN}" \
    #       2>&1 | tee "${FT_LOG}"; rc=$?; set -e
    #     if [[ $rc -eq 0 ]]; then run touch "${FT_DONE}"; log "SUCCESS: FT :: ${PART_TAG}"; else echo "[ERROR] FT failed rc=$rc"; exit $rc; fi
    #   fi
    # fi

    # (2) rand50
    if need_stage rand50; then
      AL_RAND_OUT="${OUT_ROOT}/${MODEL_SLUG}-Math14k-${PART_TAG}-rand50"
      AL_RAND_DONE="${AL_RAND_OUT}/.done"
      AL_RAND_LOG="${LOG_DIR}/$(ts)_al_rand50_${MODEL_SLUG}_${SPLIT}_${PART_TAG}.log"
      AL_RAND_RUN="AL-rand50-${MODEL_SLUG}-${SPLIT}-${PART_TAG}"
      if [[ -f "${AL_RAND_DONE}" ]]; then
        log "SKIP (done): rand50 :: ${PART_TAG}"
      else
        log "RUN  : rand50 :: ${PART_TAG}"
        run mkdir -p "${AL_RAND_OUT}"
        set +e
        python "${PY_AL}" \
          --base_model "${BASE_MODEL}" \
          --tokenizer_path "${BASE_DENSE}" \
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
          2>&1 | tee "${AL_RAND_LOG}"; rc=$?; set -e
        if [[ $rc -eq 0 ]]; then run touch "${AL_RAND_DONE}"; log "SUCCESS: rand50 :: ${PART_TAG}"; else echo "[ERROR] rand50 failed rc=$rc"; exit $rc; fi
      fi
    fi

    # (3) al50
    if need_stage al50; then
      AL_AL50_OUT="${OUT_ROOT}/${MODEL_SLUG}-Math14k-${PART_TAG}-al50"
      AL_AL50_DONE="${AL_AL50_OUT}/.done"
      AL_AL50_LOG="${LOG_DIR}/$(ts)_al_al50_${MODEL_SLUG}_${SPLIT}_${PART_TAG}.log"
      AL_AL50_RUN="AL-al50-${MODEL_SLUG}-${SPLIT}-${PART_TAG}"
      if [[ -f "${AL_AL50_DONE}" ]]; then
        log "SKIP (done): al50 :: ${PART_TAG}"
      else
        log "RUN  : al50 :: ${PART_TAG}"
        run mkdir -p "${AL_AL50_OUT}"
        set +e
        python "${PY_AL}" \
          --base_model "${BASE_MODEL}" \
          --tokenizer_path "${BASE_DENSE}" \
          --data_path "${PART_JSON}" \
          --output_dir "${AL_AL50_OUT}" \
          --rounds "${AL_AL50_ROUNDS}" \
          --init_frac "${AL_AL50_INIT}" \
          --acq_frac "${AL_AL50_ACQ}" \
          --uncertainty "${AL_UNCERT}" \
          --cutoff_len "${AL_CUTOFF_LEN}" \
          --scoring_batch_size "${AL_SCORE_BS}" \
          --num_epochs "${AL_NUM_EPOCHS}" \
          --learning_rate "${AL_LR}" \
          --per_device_train_batch_size "${AL_BATCH_SIZE}" \
          --micro_batch_size "${AL_MICRO_BATCH_SIZE}" \
          --val_set_size "${AL_VAL_SIZE}" \
          --wandb_run_name "${AL_AL50_RUN}" \
          2>&1 | tee "${AL_AL50_LOG}"; rc=$?; set -e
        if [[ $rc -eq 0 ]]; then run touch "${AL_AL50_DONE}"; log "SUCCESS: al50 :: ${PART_TAG}"; else echo "[ERROR] al50 failed rc=$rc"; exit $rc; fi
      fi
    fi

  done
done

echo "[DONE] FT + rand50 + al50 for Nemotron-14B Sparse-0.50 (parts 1 & 2). To upload: wandb sync --sync-all"
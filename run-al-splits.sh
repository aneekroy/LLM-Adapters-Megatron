#!/usr/bin/env bash
# Sequential active-learning runs on GPU 0 with per-run timing + summary.
# Usage: bash run_al_splits.sh

set -euo pipefail
IFS=$'\n\t'

# ── Config you likely won't change often ─────────────────────────────────────
BASE_MODEL="/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.20"
DATA_DIR="/home/aneek/LLM-Adapters/ft-training_set/split_20"
OUT_ROOT="./trained_models/Qwen3_Sparse-Ensemble-Split-20"
SCRIPT="active_learning.py"
TORCHRUN_BIN="${CONDA_PREFIX:-}/bin/torchrun"

# Parts to run (1..4 as you specified). Ports must be unique per run.
PARTS=(1 2 3 4)
BASE_PORT=3200   # final port per run = BASE_PORT + part

# Static AL hyperparams
ROUNDS=3
INIT_FRAC=0.1
ACQ_FRAC=0.2

# Runtime env (offline-friendly + cleaner site-packages)
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1

# Optional: keep CPU thread usage sane
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

# Logs & summary
LOG_DIR="./logs/al_splits"
SUMMARY="${LOG_DIR}/al_runs_summary.tsv"
mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

# Check prerequisites
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "[ERR] CONDA_PREFIX is not set; activate your env first." >&2
  exit 1
fi
if [[ ! -x "${TORCHRUN_BIN}" ]]; then
  echo "[ERR] torchrun not found at ${TORCHRUN_BIN}" >&2
  exit 1
fi
command -v python >/dev/null || { echo "[ERR] python not found"; exit 1; }

# Helpers
hms() { # format seconds to H:MM:SS
  local T=$1 H=$((T/3600)) M=$(( (T%3600)/60 )) S=$((T%60))
  printf "%d:%02d:%02d" "$H" "$M" "$S"
}

# Initialize summary
echo -e "part\tstart_iso\tend_iso\tduration_sec\tduration_hms\texit_code\tlog_path\toutput_dir" > "${SUMMARY}"

run_one() {
  local part="$1"
  local port=$(( BASE_PORT + part ))
  local data="${DATA_DIR}/math_14k_part${part}_of_5.json"
  local outdir="${OUT_ROOT}/Qwen3-8B-Sparse-0.20-Math14k-part${part}-al50"
  local log="${LOG_DIR}/part${part}.log"
  local run_name="Qwen3-8B-Sparse-0.20-Math14k-part${part}-al50"

  if [[ ! -f "${data}" ]]; then
    echo "[ERR] missing data file: ${data}" | tee -a "${log}"
    return 2
  fi

  echo "────────────────────────────────────────────────────────"
  echo "[RUN] part ${part} | port ${port}"
  echo "      data: ${data}"
  echo "      out : ${outdir}"
  echo "      log : ${log}"
  echo "────────────────────────────────────────────────────────"

  mkdir -p "${outdir}"
  local start_epoch end_epoch dur code
  local start_iso end_iso
  start_epoch=$(date +%s)
  start_iso=$(date -Iseconds)

  # Execute
  set +e
  WORLD_SIZE=1 "${TORCHRUN_BIN}" --nproc_per_node=1 --master_port="${port}" "${SCRIPT}" \
    --base_model "${BASE_MODEL}" \
    --data_path "${data}" \
    --output_dir "${outdir}" \
    --rounds "${ROUNDS}" \
    --init_frac "${INIT_FRAC}" \
    --acq_frac "${ACQ_FRAC}" \
    --wandb_run_name "${run_name}" \
    2>&1 | tee "${log}"
  code=${PIPESTATUS[0]}
  set -e

  end_epoch=$(date +%s)
  end_iso=$(date -Iseconds)
  dur=$(( end_epoch - start_epoch ))
  printf "[DONE] part %d | exit=%d | %s\n" "${part}" "${code}" "$(hms ${dur})"

  # Append to summary
  echo -e "${part}\t${start_iso}\t${end_iso}\t${dur}\t$(hms ${dur})\t${code}\t${log}\t${outdir}" >> "${SUMMARY}"

  return "${code}"
}

# Run all parts sequentially
overall_rc=0
for p in "${PARTS[@]}"; do
  if ! run_one "${p}"; then
    rc=$?
    echo "[WARN] part ${p} exited with code ${rc}"
    overall_rc=$rc
  fi
done

echo "================================================================"
echo "Summary written to: ${SUMMARY}"
column -t -s $'\t' "${SUMMARY}" || cat "${SUMMARY}"
exit "${overall_rc}"
# #!/usr/bin/env bash
# set -euo pipefail

# REPO="/home/aneek/LLM-Adapters"
# PY="${CONDA_PREFIX:-}/bin/python"
# [[ -x "$PY" ]] || PY="$(command -v python3 || command -v python)"
# SCRIPT="$REPO/split_math14k.py"

# SRC="/home/aneek/LLM-Adapters/ft-training_set/commonsense_15k.json"
# BASE_OUT="/home/aneek/LLM-Adapters/ft-training_set"

# [[ -f "$SRC" ]] || { echo "Missing: $SRC"; exit 1; }
# [[ -f "$SCRIPT" ]] || { echo "Missing: $SCRIPT"; exit 1; }

# echo "[commonsense_15k] splitting…"
# mkdir -p "$BASE_OUT/split_50" "$BASE_OUT/split_33" "$BASE_OUT/split_25" "$BASE_OUT/split_20"

# # K=2 (50%)
# "$PY" "$SCRIPT" --src "$SRC" --outdir "$BASE_OUT/split_50" --parts 2 --seed 42
# # K=3 (33%)
# "$PY" "$SCRIPT" --src "$SRC" --outdir "$BASE_OUT/split_33" --parts 3 --seed 42
# # K=4 (25%)
# "$PY" "$SCRIPT" --src "$SRC" --outdir "$BASE_OUT/split_25" --parts 4 --seed 42
# # K=5 (20%)
# "$PY" "$SCRIPT" --src "$SRC" --outdir "$BASE_OUT/split_20" --parts 5 --seed 42

# echo "[commonsense_15k] done."

#!/usr/bin/env bash
set -euo pipefail

REPO="/home/aneek/LLM-Adapters"
PY="${CONDA_PREFIX:-}/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3 || command -v python)"
SCRIPT="$REPO/split_math14k.py"

SRC="/home/aneek/LLM-Adapters/ft-training_set/dolly-15k.json"
BASE_OUT="/home/aneek/LLM-Adapters/ft-training_set"

[[ -f "$SRC" ]] || { echo "Missing: $SRC"; exit 1; }
[[ -f "$SCRIPT" ]] || { echo "Missing: $SCRIPT"; exit 1; }

echo "[dolly-15k] splitting…"
mkdir -p "$BASE_OUT/split_50" "$BASE_OUT/split_33" "$BASE_OUT/split_25" "$BASE_OUT/split_20"

# K=2 (50%)
"$PY" "$SCRIPT" --src "$SRC" --outdir "$BASE_OUT/split_50" --parts 2 --seed 42
# K=3 (33%)
"$PY" "$SCRIPT" --src "$SRC" --outdir "$BASE_OUT/split_33" --parts 3 --seed 42
# K=4 (25%)
"$PY" "$SCRIPT" --src "$SRC" --outdir "$BASE_OUT/split_25" --parts 4 --seed 42
# K=5 (20%)
"$PY" "$SCRIPT" --src "$SRC" --outdir "$BASE_OUT/split_20" --parts 5 --seed 42

echo "[dolly-15k] done."
#!/usr/bin/env bash
set -euo pipefail

# CACHE_ROOT="/some/other/cache" DEST="$HOME/LLM-Adapters/ft-training_set/dataset/dolly15k"

# --- config (override via env if you want) ---
CACHE_ROOT="${CACHE_ROOT:-$HOME/.cache/huggingface/hub/datasets--databricks--databricks-dolly-15k/snapshots}"
DEST="${DEST:-$HOME/LLM-Adapters/ft-training_set/dataset/dolly15k}"

# --- locate latest snapshot in cache ---
if [[ ! -d "$CACHE_ROOT" ]]; then
  echo "ERROR: Cache root not found: $CACHE_ROOT"
  exit 1
fi
SNAPSHOT_DIR="$(ls -1dt "$CACHE_ROOT"/* 2>/dev/null | head -n1 || true)"
if [[ -z "${SNAPSHOT_DIR}" || ! -d "${SNAPSHOT_DIR}" ]]; then
  echo "ERROR: No snapshot dir found under $CACHE_ROOT"
  exit 1
fi

# --- copy files to project ---
mkdir -p "$DEST"
cp -v "$SNAPSHOT_DIR/databricks-dolly-15k.jsonl" "$DEST/"
cp -v "$SNAPSHOT_DIR/README.md"                  "$DEST/"

# --- convert JSONL -> JSON array ---
export DEST
python3 - <<'PY'
import os, json, pathlib, sys
dest = pathlib.Path(os.environ["DEST"])
src = dest / "databricks-dolly-15k.jsonl"
dst = dest / "databricks-dolly-15k.json"

if not src.exists():
    print(f"ERROR: Missing file {src}", file=sys.stderr); sys.exit(1)

arr = []
with open(src, 'r', encoding='utf-8') as f:
    for ln, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            arr.append(json.loads(line))
        except Exception as e:
            print(f"ERROR: JSON parse failed at line {ln}: {e}", file=sys.stderr)
            sys.exit(1)

with open(dst, 'w', encoding='utf-8') as g:
    json.dump(arr, g, ensure_ascii=False, indent=2)

print(f"Wrote {dst} ({len(arr)} records)")
PY

# --- show results ---
ls -lh "$DEST"
echo "Done."
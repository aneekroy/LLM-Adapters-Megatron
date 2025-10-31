#!/usr/bin/env python3
"""
Split a JSON list dataset into K roughly equal chunks (K in {2,3,4,5}).

Usage
-----
python split_math14k.py \
  --src /home/aneek/LLM-Adapters/ft-training_set/math_14k.json \
  --outdir /home/aneek/LLM-Adapters/ft-training_set/split_50/ \
  --parts 2 \
  --seed 42


python split_k.py --src /.../math_14k.json --outdir /.../split --parts 2 --seed 42
"""

import json, random, argparse, pathlib, os
from typing import List

def split_k(lst: List, k: int) -> List[List]:
    """Split lst into k parts with sizes as equal as possible."""
    n = len(lst)
    if k < 2 or k > 5:
        raise ValueError("k must be between 2 and 5")
    if n < k:
        raise ValueError(f"Dataset too small ({n}) to split into {k} parts.")

    base = n // k         # minimum size per part
    rem  = n %  k         # first `rem` parts get +1
    parts = []
    start = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        parts.append(lst[start:start+size])
        start += size
    assert sum(len(p) for p in parts) == n
    return parts

def main(src: str, outdir: str, parts: int, seed: int):
    # 1) Load JSON list
    src_path = pathlib.Path(src)
    with open(src_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError("Expected a JSON array/list at top level.")

    # 2) Shuffle once for reproducibility
    random.seed(seed)
    random.shuffle(data)

    # 3) Split
    chunks = split_k(data, parts)

    # 4) Write
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = src_path.stem

    for idx, chunk in enumerate(chunks, start=1):
        out_path = outdir / f"{stem}_part{idx}_of_{parts}.json"
        with open(out_path, "w") as f:
            json.dump(chunk, f, indent=2, ensure_ascii=False)
        print(f"[✓] wrote {len(chunk):>6} examples → {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Split a JSON list dataset into K parts.")
    ap.add_argument("--src",    required=True, help="Path to source JSON (list).")
    ap.add_argument("--outdir", required=True, help="Directory to save chunks.")
    ap.add_argument("--parts",  type=int, choices=[2,3,4,5], required=True, help="Number of splits.")
    ap.add_argument("--seed",   type=int, default=42, help="RNG seed for shuffling.")
    args = ap.parse_args()
    main(args.src, args.outdir, args.parts, args.seed)
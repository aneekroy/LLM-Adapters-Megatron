#!/usr/bin/env python3
import os, sys, json, glob, argparse
from pathlib import Path

TOK_RANDOM = ("random50","rand50","random_50","split_50","split50","random","split-random","split_random")
TOK_AL     = ("al50","al_50","active-learning","active_learning","active","heuristic","al")

def _iter_json_files(paths):
    if not paths:
        return
    for p in paths:
        matched = glob.glob(p, recursive=True)
        if not matched:
            if os.path.exists(p):
                matched = [p]
        for m in matched:
            if os.path.isdir(m):
                for root, _dirs, files in os.walk(m):
                    for fn in files:
                        if fn.lower().endswith(".json"):
                            yield os.path.join(root, fn)
            else:
                if m.lower().endswith(".json"):
                    yield m

def _count_json_list(path):
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return len(obj)
    if isinstance(obj, dict):
        for key in ("data", "train", "examples", "items"):
            if key in obj and isinstance(obj[key], list):
                return len(obj[key])
    raise TypeError(f"{path} is not a JSON list (or known container).")

def count_from_inputs(specs):
    total = 0
    files = list(_iter_json_files(specs))
    for fp in files:
        try:
            total += _count_json_list(fp)
        except Exception:
            pass
    return total, files

def looks_like_random50(path):
    p = path.lower()
    return p.endswith(".json") and any(tok in p for tok in TOK_RANDOM)

def looks_like_al50(path):
    p = path.lower()
    return p.endswith(".json") and any(tok in p for tok in TOK_AL)

def discover_nearby(full_path):
    roots = set()
    full_path = Path(full_path)
    if full_path.is_file():
        roots.add(full_path.parent)
    else:
        roots.add(full_path)
    parents = list(roots.copy())
    for r in list(roots):
        parents.extend([r.parent, r.parent.parent])
    for pr in parents:
        if pr and str(pr) != "/":
            roots.add(pr)
            for name in ("splits","split_50","random50","rand50","random","al50","al","active","heuristic","split"):
                roots.add(pr / name)

    rand_files, al_files = [], []
    for root in list(roots):
        if not Path(root).exists():
            continue
        for rp, _dirs, files in os.walk(root):
            for fn in files:
                fp = os.path.join(rp, fn)
                if looks_like_random50(fp):
                    rand_files.append(fp)
                if looks_like_al50(fp):
                    al_files.append(fp)
    rand_files = sorted(set(rand_files))
    al_files   = sorted(set(al_files))
    return rand_files, al_files

def one_part_sizes(total, K):
    base = total // K
    rem  = total %  K
    return [base + (1 if i < rem else 0) for i in range(K)]

def main():
    ap = argparse.ArgumentParser(description="Report Math-14k counts given only --full; auto-discovers/estimates 50% splits.")
    ap.add_argument("--full",  nargs="+", required=True,  help="Path(s) for Full data (file/dir/glob).")
    ap.add_argument("--all_parts", action="store_true", help="Also print one-part sizes for K in {2,3,4,5}.")
    args = ap.parse_args()

    full_n, full_files = count_from_inputs(args.full)
    probe_root = args.full[0]
    rand_files, al_files = discover_nearby(probe_root)
    rand_n, _ = count_from_inputs(rand_files) if rand_files else (0, [])
    al_n, _   = count_from_inputs(al_files)   if al_files   else (0, [])

    rand_estimated = False
    al_estimated = False
    if rand_n == 0 and full_n:
        rand_n = full_n // 2
        rand_estimated = True
    if al_n == 0 and full_n:
        al_n = full_n // 2
        al_estimated = True

    print("I. Math 14k — Data Amounts\n")
    print(f"Full Data : {full_n}")
    print(f"50% of Data for Random Split : {rand_n}" + ("  [estimated]" if rand_estimated else ""))
    print(f"50% of Data for Active Learning : {al_n}" + ("  [estimated]" if al_estimated else ""))
    print()

    headers = [
        "50% Sparsity for one-part out of the ensemble",
        "67% Sparsity for one-part out of the ensemble",
        "75% Sparsity for one-part out of the ensemble",
        "80% Sparsity for one-part out of the ensemble",
    ]

    if args.all_parts:
        for h in headers:
            print(h)
            for K in (2,3,4,5):
                print(f"K={K} (one-part sizes)")
                print(f"Full Data : {one_part_sizes(full_n, K)}")
                print(f"50% of Data for Random Split : {one_part_sizes(rand_n, K)}")
                print(f"50% of Data for Active Learning : {one_part_sizes(al_n, K)}")
                print()
    else:
        for h in headers:
            print(h)
            print(f"Full Data : {full_n}")
            print(f"50% of Data for Random Split : {rand_n}" + ("  [estimated]" if rand_estimated else ""))
            print(f"50% of Data for Active Learning : {al_n}" + ("  [estimated]" if al_estimated else ""))
            print()

    if full_n:
        r_pct = 100.0 * rand_n / full_n
        a_pct = 100.0 * al_n   / full_n
        if abs(r_pct - 50.0) > 2.0 and not rand_estimated:
            print(f"[!] Warning: Random50 is {r_pct:.2f}% of Full (expected ~50%).")
        if abs(a_pct - 50.0) > 2.0 and not al_estimated:
            print(f"[!] Warning: AL50 is {a_pct:.2f}% of Full (expected ~50%).")

if __name__ == "__main__":
    main()

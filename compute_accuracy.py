#!/usr/bin/env python3
import argparse, json, math, os, sys

def try_json_load_all(text):
    """Try to parse whole-file JSON (array or single object)."""
    try:
        data = json.loads(text)
        if isinstance(data, dict):  # single object → wrap
            return [data]
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return None

def try_json_lines(text):
    """Parse JSON Lines: one JSON object per line."""
    recs = []
    ok = False
    for ln in text.splitlines():
        s = ln.strip()
        if not s or not s.startswith("{"):
            continue
        try:
            recs.append(json.loads(s))
            ok = True
        except Exception:
            return None
    return recs if ok else None

def chunk_loose_objects(text):
    """Chunk concatenated `{...}` blocks by brace depth, ignoring commas/newlines."""
    recs, buf, depth, in_str, esc = [], [], 0, False, False
    for ch in text:
        buf.append(ch)
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                # end of one JSON object
                obj_txt = "".join(buf).strip()
                # strip trailing commas/newlines after object
                try:
                    recs.append(json.loads(obj_txt))
                except Exception:
                    # try to trim trailing commas
                    obj_txt2 = obj_txt.rstrip().rstrip(",")
                    recs.append(json.loads(obj_txt2))
                buf = []
    # If nothing parsed, return None
    return recs if recs else None

def as_number(x):
    try:
        return float(str(x).strip())
    except Exception:
        return None

def is_correct(rec):
    # Priority 1: explicit boolean/flag
    if 'correct' in rec:
        v = rec['correct']
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            return v.strip().lower() in {"true","1","yes","y"}
    # Priority 2: compare answer vs prediction
    a = rec.get('answer')
    p = rec.get('prediction')
    # Sometimes 'answer' is nested or as str; sometimes 'output' has the reasoning
    a_num, p_num = as_number(a), as_number(p)
    if a_num is not None and p_num is not None:
        # numeric compare with relative+absolute tolerance
        tol = max(1e-9, 1e-6 * max(1.0, abs(a_num), abs(p_num)))
        return abs(a_num - p_num) <= tol
    # fallback: normalized string compare
    if a is not None and p is not None:
        sa = str(a).strip().lower()
        sp = str(p).strip().lower()
        # common cleanup
        for bad in [".0", ",", " "]:
            if sa.endswith(bad): sa = sa[:-len(bad)]
            if sp.endswith(bad): sp = sp[:-len(bad)]
        return sa == sp
    return False

def main():
    ap = argparse.ArgumentParser(description="Compute accuracy from a results JSON.")
    ap.add_argument("path", help="Path to JSON results file")
    args = ap.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        text = f.read()

    recs = (try_json_load_all(text) or
            try_json_lines(text) or
            chunk_loose_objects(text))
    if recs is None:
        print("Could not parse JSON in any supported format.", file=sys.stderr)
        sys.exit(2)

    total = len(recs)
    correct = sum(1 for r in recs if is_correct(r))
    acc = (correct / total) * 100.0 if total else 0.0

    print(f"File: {args.path}")
    print(f"Total examples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()
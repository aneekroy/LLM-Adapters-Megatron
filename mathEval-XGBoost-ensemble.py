#!/usr/bin/env python
# Fast vLLM ensemble evaluator with multi-LoRA preloading and minimal overhead
# Adds XGBoost-based stacking ensemble (`--ensemble_rule xgb`)

import argparse, copy, datetime, glob, hashlib, json, os, re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import numpy as np  # NEW
import wandb
from tqdm.auto import tqdm

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# --------------------- small utils ---------------------
NUM_REGEX = re.compile(r"-?\d+\.?\d*")

def _slugify(s: str) -> str:
    s = re.sub(r'[^A-Za-z0-9._-]+', '-', s)
    return re.sub(r'-+', '-', s).strip('-')

def _safe_fname(stem: str, suffix: str = ".json", max_len: int = 200) -> str:
    stem = re.sub(r"[^A-Za-z0-9._+-]+", "-", stem).strip("-")
    max_stem = max_len - len(suffix)
    if len(stem) > max_stem:
        stem = stem[:max_stem]
    return f"{stem}{suffix}"

def _lora_tag(spec_csv: str) -> str:
    parts = []
    for seg in spec_csv.split(','):
        seg = seg.strip()
        if not seg:
            continue
        if '=' in seg:
            k, p = seg.split('=', 1)
            p = p.rstrip('/')
            leaf = os.path.basename(p)
            parent = os.path.basename(os.path.dirname(p))
            parts.append(f"{k}-{parent}-{leaf}")
        else:
            parts.append(os.path.basename(seg.rstrip('/')))
    return _slugify('_'.join(parts)) or "base"

def _expand_one_token(token: str):
    out = []
    if "=" in token:
        name, raw = token.split("=", 1)
        name = name.strip(); raw = raw.strip()
        paths = sorted(glob.glob(raw)) if any(ch in raw for ch in "*?[") else [raw.rstrip("/")]
        for p in paths:
            out.append((name, p))
    else:
        raw = token.strip()
        paths = sorted(glob.glob(raw)) if any(ch in raw for ch in "*?[") else [raw.rstrip("/")]
        for p in paths:
            inferred = os.path.basename(p.rstrip("/")) or f"adapter_{len(out)+1}"
            out.append((inferred, p))
    return out

def expand_adapters(spec: str):
    out, seen = [], set()
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        for name, path in _expand_one_token(tok):
            if not os.path.isdir(path):
                raise FileNotFoundError(f"LoRA path not found or not a dir: {path}")
            base, i, nm = name, 1, name
            while nm in seen:
                i += 1; nm = f"{base}_{i}"
            seen.add(nm)
            out.append((nm, path))
    if not out:
        raise ValueError(f"No adapter paths expand from spec: {spec}")
    return out  # list[(name, path)]

def generate_prompt(instr: str) -> str:
    return ("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instr}\n\n### Response:\n")

def load_data(name: str):
    path = f"dataset/{name}/test.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset file {path}")
    return json.load(open(path))

def extract_answer(ds_name: str, text: str):
    if ds_name.lower() == "aqua":
        m = re.search(r"[ABCDE]", text)
        return m.group(0) if m else ""
    text = text.replace(",", "")
    nums = NUM_REGEX.findall(text)
    if not nums:
        return float("nan")   # CHANGED: use NaN for missing instead of inf
    try:
        return float(nums[-1])
    except ValueError:
        return float("nan")

def numeric_equal(a: float, b: float, tol: float):
    if np.isnan(a) or np.isnan(b):
        return False
    return abs(a - b) <= tol


def build_llm(args, num_adapters: int):
    eff_max_loras = max(args.max_loras, num_adapters)
    eff_max_cpu   = args.max_cpu_loras if args.max_cpu_loras is not None else eff_max_loras
    llm = LLM(
        model=args.base_model,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        trust_remote_code=True,
        enable_lora=True,
        max_loras=eff_max_loras,
        max_cpu_loras=eff_max_cpu,
        max_lora_rank=args.max_lora_rank,
    )
    sp = SamplingParams(temperature=0.1, top_p=0.75, top_k=40, max_tokens=args.max_tokens)  # CHANGED: use CLI max_tokens
    return llm, sp, eff_max_loras, eff_max_cpu

def preload_adapters(llm, adapters):
    warm = SamplingParams(temperature=0.0, max_tokens=1)
    for idx, (name, path) in enumerate(adapters, start=1):
        llm.generate(["."], warm, lora_request=LoRARequest(name, idx, path))

# --------------------- NEW: meta-feature builders ---------------------
def _features_numeric(votes_matrix: np.ndarray):
    """
    votes_matrix: shape [N, M] with floats, possibly NaN
    returns X: [N, M + 6] with per-adapter preds + aggregates
    """
    N, M = votes_matrix.shape
    X = votes_matrix.copy()  # keep per-adapter cols (sparsity-aware NaN)
    # aggregates
    mean = np.nanmean(votes_matrix, axis=1)
    med  = np.nanmedian(votes_matrix, axis=1)
    std  = np.nanstd(votes_matrix, axis=1)
    minv = np.nanmin(votes_matrix, axis=1)
    maxv = np.nanmax(votes_matrix, axis=1)
    rng  = maxv - minv
    agg = np.stack([mean, med, std, minv, maxv, rng], axis=1)
    return np.concatenate([X, agg], axis=1)

def _entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) if p.size else 0.0

def _features_aqua(letter_votes: list, adapters_count: int):
    """
    letter_votes: list length N; each item is list/tuple length M with 'A'..'E' or '' for missing
    returns X: [N, M*5 + 5 + 2]
      - per-adapter one-hots (M*5)
      - global counts per option (5)
      - entropy + max_ratio (2)
    """
    N = len(letter_votes)
    M = adapters_count
    X = np.zeros((N, M * 5 + 5 + 2), dtype=np.float32)
    for i, row in enumerate(letter_votes):
        counts = np.zeros(5, dtype=np.float32)
        # per-adapter one-hots
        for j, c in enumerate(row):
            if not c:
                continue
            k = ord(c) - ord('A')
            if 0 <= k < 5:
                X[i, j*5 + k] = 1.0
                counts[k] += 1.0
        # global counts
        base = M * 5
        X[i, base:base+5] = counts
        total = counts.sum()
        if total > 0:
            probs = counts / total
            ent = _entropy(probs)
            max_ratio = probs.max()
        else:
            ent, max_ratio = 0.0, 0.0
        X[i, base+5]   = ent
        X[i, base+6]   = max_ratio
    return X

# --------------------- arg parsing ---------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["AddSub","MultiArith","SingleEq","gsm8k","AQuA","SVAMP"], required=True)
    p.add_argument("--model", choices=[
        "LLaMA-7B","BLOOM-7B","GPT-j-6B",
        "Llama-3.2-1B","Llama-3.2-3B",
        "Llama-3.2-1B-Instruct","Llama-3.2-3B-Instruct",
        "Llama-3.2-70B-Instruct","Llama-3.2-70B","Llama-3.2-3B-Instruct-Sparse",
        "Qwen3-4B-Instruct","Qwen3-8B-Sparse","Nemotron-14B-Sparse","Nemotron-7B","Nemotron-7B-Sparse","Qwen3-4B-Sparse"
    ], required=True)
    p.add_argument("--base_model", required=True)
    p.add_argument("--lora_weights", required=True,
                   help="Comma list or globs of LoRA dirs; optional name=path for each.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_tokens", type=int, default=256)  # CHANGED: respected upstream
    p.add_argument("--ensemble_rule", choices=["vote","median","xgb"], default="vote")  # NEW
    p.add_argument("--tolerance", type=float, default=1e-3)

    # vLLM & LoRA cache
    p.add_argument("--tp_size", type=int, default=1)
    p.add_argument("--max_model_len", type=int, default=None)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--max_loras", type=int, default=1)
    p.add_argument("--max_cpu_loras", type=int, default=None)
    p.add_argument("--max_lora_rank", type=int, default=32)

    # preloading
    p.add_argument("--preload", action="store_true")

    # logging
    p.add_argument("--log_every", type=int, default=5, help="W&B log freq in batches (per-adapter)")
    p.add_argument("--wandb_project", default="lora_math_ensemble_eval_al50")
    p.add_argument("--wandb_run_name", default=None)

    # NEW: XGBoost hyperparams
    p.add_argument("--xgb_tree_method", choices=["hist","approx","auto"], default="hist")
    p.add_argument("--xgb_eta", type=float, default=0.10)
    p.add_argument("--xgb_max_depth", type=int, default=4)
    p.add_argument("--xgb_subsample", type=float, default=0.80)
    p.add_argument("--xgb_colsample_bytree", type=float, default=0.70)
    p.add_argument("--xgb_lambda", type=float, default=1.0)
    p.add_argument("--xgb_alpha", type=float, default=0.0)
    p.add_argument("--xgb_rounds", type=int, default=200)
    p.add_argument("--xgb_cv_folds", type=int, default=5)  # set 0 to disable
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

# --------------------- main ---------------------
def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Expand adapters once
    adapters = expand_adapters(args.lora_weights)
    M = len(adapters)

    # Data once
    data = load_data(args.dataset)
    instructions = [d["instruction"] for d in data]

    # Prebuild prompts once
    prompts = [generate_prompt(instr) for instr in instructions]

    # Prebuild batches
    N = len(prompts)
    batch_starts = list(range(0, N, args.batch_size))

    # Labels once
    if args.dataset.lower() == "aqua":
        labels_letters = [d["answer"] for d in data]
    else:
        labels = []
        for d in data:
            v = d["answer"]
            labels.append(float(v) if not isinstance(v, (int, float)) else v)
        labels = np.array(labels, dtype=float)

    # Build LLM
    llm, sampling_params, eff_max_loras, eff_max_cpu = build_llm(args, num_adapters=M)

    # Preload caches if asked
    if args.preload:
        print(f"Preloading {M} LoRAs into cache …")
        preload_adapters(llm, adapters)

    # W&B
    lora_tag = _lora_tag(args.lora_weights)
    run_id = args.wandb_run_name or f"{args.model}-{args.dataset}-{os.path.basename(args.base_model).strip('/')}-{lora_tag}-{datetime.datetime.now():%Y%m%d_%H%M%S}"
    wandb.init(project=args.wandb_project, name=run_id, config={
        **vars(args),
        "num_adapters": M,
        "N_samples": N,
    }, reinit=True)

    # Votes: [M][N]
    votes = [[None] * N for _ in range(M)]

    # Generate per adapter
    for a_idx, (adapter_name, path) in enumerate(adapters, start=1):
        print(f">> [{a_idx}/{M}] {adapter_name} → {path}")
        lora_req = LoRARequest(adapter_name, a_idx, path)
        seen = 0
        for bi, start in enumerate(tqdm(batch_starts, leave=False)):
            end = min(start + args.batch_size, N)
            outs = llm.generate(prompts[start:end], sampling_params, lora_request=lora_req)
            # Extract predictions
            if args.dataset.lower() == "aqua":
                preds = []
                for o in outs:
                    txt = (o.outputs[0].text.strip() if o.outputs else "")
                    m = re.search(r"[ABCDE]", txt)
                    preds.append(m.group(0) if m else "")
            else:
                preds = []
                for o in outs:
                    txt = (o.outputs[0].text.strip() if o.outputs else "")
                    v = extract_answer(args.dataset, txt)
                    preds.append(v)
                preds = [float(x) if x != "" else float("nan") for x in preds]
            votes[a_idx - 1][start:end] = preds
            seen = end

            # Online per-adapter acc (rough)
            if (bi + 1) % args.log_every == 0 or end == N:
                if args.dataset.lower() == "aqua":
                    corr = sum(p == lbl for p, lbl in zip(votes[a_idx-1][:seen], labels_letters[:seen]))
                else:
                    corr = sum(numeric_equal(p, lbl, args.tolerance) for p, lbl in zip(votes[a_idx-1][:seen], labels[:seen]))
                wandb.log({f"{adapter_name}_acc": corr / max(1, seen), "samples_seen": seen})

    # --------------------- ENSEMBLE ---------------------
    # Prepare arrays
    if args.dataset.lower() == "aqua":
        # shape [N, M] of strings
        vote_letters = list(zip(*votes))  # list of length N, each entry tuple len M
        vote_letters = [list(t) for t in vote_letters]
    else:
        vote_nums = np.array(votes, dtype=float).T  # [N, M]

    # XGBoost path
    if args.ensemble_rule == "xgb":
        try:
            import xgboost as xgb
        except Exception as e:
            raise RuntimeError(
                "xgboost is required for --ensemble_rule xgb. Install via `pip install xgboost`."
            ) from e

        if args.dataset.lower() == "aqua":
            X = _features_aqua(vote_letters, M)
            y = np.array([ord(c) - ord('A') for c in labels_letters], dtype=int)
            param = {
                "objective": "multi:softprob",
                "num_class": 5,
                "eval_metric": "mlogloss",
                "eta": args.xgb_eta,
                "max_depth": args.xgb_max_depth,
                "subsample": args.xgb_subsample,
                "colsample_bytree": args.xgb_colsample_bytree,
                "lambda": args.xgb_lambda,
                "alpha": args.xgb_alpha,
                "tree_method": args.xgb_tree_method,
                "seed": args.seed,
            }
            dmat = xgb.DMatrix(X, label=y, missing=np.nan)
        else:
            X = _features_numeric(vote_nums)  # NaNs preserved
            y = labels
            param = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "eta": args.xgb_eta,
                "max_depth": args.xgb_max_depth,
                "subsample": args.xgb_subsample,
                "colsample_bytree": args.xgb_colsample_bytree,
                "lambda": args.xgb_lambda,
                "alpha": args.xgb_alpha,
                "tree_method": args.xgb_tree_method,
                "seed": args.seed,
            }
            dmat = xgb.DMatrix(X, label=y, missing=np.nan)

        # Optional CV (honest OOF estimate)
        cv_acc = None
        if args.xgb_cv_folds and args.xgb_cv_folds > 1:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=args.xgb_cv_folds, shuffle=True, random_state=args.seed)
            oof = np.zeros((N, 5), dtype=float) if args.dataset.lower()=="aqua" else np.zeros(N, dtype=float)
            for tr, va in kf.split(X):
                dtr = xgb.DMatrix(X[tr], label=y[tr], missing=np.nan)
                dva = xgb.DMatrix(X[va], label=y[va], missing=np.nan)
                bst = xgb.train(param, dtr, num_boost_round=args.xgb_rounds)
                if args.dataset.lower()=="aqua":
                    pred = bst.predict(dva)  # [len(va), 5]
                    oof[va] = pred
                else:
                    pred = bst.predict(dva)
                    oof[va] = pred
            if args.dataset.lower()=="aqua":
                pred_cls = oof.argmax(axis=1)
                cv_acc = float((pred_cls == y).mean())
            else:
                pred = oof
                cv_acc = float(np.mean([numeric_equal(p, t, args.tolerance) for p, t in zip(pred, y)]))
            wandb.log({"xgb_cv_acc": cv_acc})

        # Fit full model, predict
        bst = xgb.train(param, dmat, num_boost_round=args.xgb_rounds)

        if args.dataset.lower()=="aqua":
            probs = bst.predict(xgb.DMatrix(X, missing=np.nan))
            pred_cls = probs.argmax(axis=1)
            final_preds = [chr(int(k)+ord('A')) for k in pred_cls]
            acc = float(np.mean([p == l for p, l in zip(final_preds, labels_letters)]))
        else:
            pred = bst.predict(xgb.DMatrix(X, missing=np.nan))
            final_preds = pred.tolist()
            acc = float(np.mean([numeric_equal(p, t, args.tolerance) for p, t in zip(pred, y)]))

        wandb.log({"ensemble_accuracy": acc, "xgb_fullfit_acc": acc})
        if cv_acc is not None:
            print(f"\n==== XGB ENSEMBLE | OOF_ACC={cv_acc:.4f} | FULLFIT_ACC={acc:.4f} ====\n")
        else:
            print(f"\n==== XGB ENSEMBLE | FULLFIT_ACC={acc:.4f} ====\n")

    else:
        # Original rule-based paths preserved
        if args.dataset.lower() == "aqua":
            final_preds = []
            for i in range(N):
                pred_list = [votes[a][i] for a in range(M)]
                tally = Counter(pred_list)
                final_preds.append(tally.most_common(1)[0][0] if tally else "")
            acc = sum(p == l for p, l in zip(final_preds, labels_letters)) / N
        else:
            if args.ensemble_rule == "median":
                final_preds = [np.nanmedian(vote_nums[i]) if not np.isnan(vote_nums[i]).all() else np.nan
                               for i in range(N)]
            else:
                final_preds = []
                for i in range(N):
                    bins: defaultdict[float, int] = defaultdict(int)
                    row = vote_nums[i]
                    row = row[~np.isnan(row)]
                    for v in row:
                        placed = False
                        for key in list(bins.keys()):
                            if numeric_equal(v, key, args.tolerance):
                                bins[key] += 1; placed = True; break
                        if not placed:
                            bins[v] = 1
                    final_preds.append(max(bins.items(), key=lambda x: x[1])[0] if bins else float("nan"))
            acc = float(np.mean([numeric_equal(p, t, args.tolerance) for p, t in zip(final_preds, labels)]))
        print(f"\n==== ENSEMBLE (M={M}) | ACC={acc:.4f} ====\n")
        wandb.log({"ensemble_accuracy": acc})

    # Write compact output
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = _slugify(args.model)
    ds_tag = _slugify(args.dataset)
    short = hashlib.sha1(args.lora_weights.encode()).hexdigest()[:10]
    stem = f"{model_tag}-{ds_tag}-{ts}-{short}"
    out_dir = Path("experiment") / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _safe_fname(stem, suffix=".json", max_len=120)

    for rec, pred in zip(data, final_preds):
        rec["prediction"] = pred
        if args.dataset.lower() == "aqua":
            rec["correct"] = (pred == rec["answer"])
        else:
            try:
                rec["correct"] = numeric_equal(float(pred), float(rec["answer"]), args.tolerance)
            except Exception:
                rec["correct"] = False

    with open(out_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Saved →", str(out_path))
    wandb.finish()

if __name__ == "__main__":
    main()
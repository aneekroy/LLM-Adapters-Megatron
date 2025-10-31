#!/usr/bin/env python
# Fast vLLM ensemble evaluator with multi-LoRA preloading and minimal overhead

import argparse, copy, datetime, glob, hashlib, json, os, re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

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
        return float("inf")
    try:
        return float(nums[-1])
    except ValueError:
        return float("inf")

def numeric_equal(a: float, b: float, tol: float):
    return abs(a - b) <= tol



def build_llm(args, num_adapters: int):
    # plan effective cache sizes we intend to use
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
    sp = SamplingParams(temperature=0.1, top_p=0.75, top_k=40, max_tokens=256)
    return llm, sp, eff_max_loras, eff_max_cpu


def preload_adapters(llm, adapters):
    warm = SamplingParams(temperature=0.0, max_tokens=1)
    # One tiny request per adapter to populate caches
    for idx, (name, path) in enumerate(adapters, start=1):
        llm.generate(["."], warm, lora_request=LoRARequest(name, idx, path))

# --------------------- arg parsing ---------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["AddSub","MultiArith","SingleEq","gsm8k","AQuA","SVAMP"], required=True)
    p.add_argument("--model", choices=[
        "LLaMA-7B","BLOOM-7B","GPT-j-6B",
        "Llama-3.2-1B","Llama-3.2-3B",
        "Llama-3.2-1B-Instruct","Llama-3.2-3B-Instruct",
        "Llama-3.2-70B-Instruct","Llama-3.2-70B","Llama-3.2-3B-Instruct-Sparse",
        "Qwen3-4B-Instruct","Qwen3-8B-Sparse","Nemotron-14B-Sparse","Nemotron-7B","Nemotron-7B-Sparse",'Qwen3-4B-Sparse'
    ], required=True)
    p.add_argument("--base_model", required=True)
    p.add_argument("--lora_weights", required=True,
                   help="Comma list or globs of LoRA dirs; optional name=path for each.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--ensemble_rule", choices=["vote","median"], default="vote")
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
    return p.parse_args()

# --------------------- main ---------------------
def main():
    args = parse_args()

    # Expand adapters once
    adapters = expand_adapters(args.lora_weights)
    M = len(adapters)

    # Data once
    data = load_data(args.dataset)
    instructions = [d["instruction"] for d in data]

    # Prebuild prompts once (reused across adapters)
    prompts = [generate_prompt(instr) for instr in instructions]

    # Prebuild batch index windows once
    N = len(prompts)
    batch_starts = list(range(0, N, args.batch_size))

    # Labels once
    if args.dataset.lower() == "aqua":
        labels = [d["answer"] for d in data]
    else:
        labels = [float(d["answer"]) if not isinstance(d["answer"], (int, float)) else d["answer"] for d in data]

    # Build LLM (pin as many LoRAs as possible to avoid swapping)
    llm, sampling_params, eff_max_loras, eff_max_cpu = build_llm(args, num_adapters=M)

    # Preload (keeps adapters hot in cache)
    if args.preload:
        print(f"Preloading {M} LoRAs into cache …")
        preload_adapters(llm, adapters)

    # W&B lightweight init
    lora_tag = _lora_tag(args.lora_weights)
    run_id = args.wandb_run_name or f"{args.model}-{args.dataset}-{os.path.basename(args.base_model).strip('/')}-{lora_tag}-{datetime.datetime.now():%Y%m%d_%H%M%S}"
    wandb.init(project=args.wandb_project, name=run_id, config={
        "dataset": args.dataset,
        "model": args.model,
        "base_model": args.base_model,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "ensemble_rule": args.ensemble_rule,
        "tolerance": args.tolerance,
        "tp_size": args.tp_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "max_loras": eff_max_loras,   # actual
        "max_cpu_loras": eff_max_cpu,
        "max_lora_rank": args.max_lora_rank,
        "preload": args.preload,
        "num_adapters": M,
        "N_samples": N,
    }, reinit=True)

    # Votes: [M][N]
    votes = [[None] * N for _ in range(M)]

    # Generate per adapter with minimal per-batch overhead
    for a_idx, (adapter_name, path) in enumerate(adapters, start=1):
        print(f">> [{a_idx}/{M}] {adapter_name} → {path}")
        lora_req = LoRARequest(adapter_name, a_idx, path)

        seen = 0
        for bi, start in enumerate(tqdm(batch_starts, leave=False)):
            end = min(start + args.batch_size, N)
            outs = llm.generate(prompts[start:end], sampling_params, lora_request=lora_req)
            # Minimal postproc
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
                    txt = txt.replace(",", "")
                    nums = NUM_REGEX.findall(txt)
                    if not nums:
                        preds.append(float("inf"))
                    else:
                        try:
                            preds.append(float(nums[-1]))
                        except ValueError:
                            preds.append(float("inf"))

            votes[a_idx - 1][start:end] = preds
            seen = end

            # Throttle W&B logs
            if (bi + 1) % args.log_every == 0 or end == N:
                if args.dataset.lower() == "aqua":
                    corr = sum(p == lbl for p, lbl in zip(votes[a_idx-1][:seen], labels[:seen]))
                else:
                    corr = sum(numeric_equal(p, lbl, args.tolerance) for p, lbl in zip(votes[a_idx-1][:seen], labels[:seen]))
                wandb.log({f"{adapter_name}_acc": corr / max(1, seen), "samples_seen": seen})

    # Ensemble
    final_preds = []
    if args.dataset.lower() == "aqua":
        for i in range(N):
            pred_list = [votes[a][i] for a in range(M)]
            tally = Counter(pred_list)
            final_preds.append(tally.most_common(1)[0][0] if tally else "")
        acc = sum(p == l for p, l in zip(final_preds, labels)) / N
    else:
        if args.ensemble_rule == "median":
            for i in range(N):
                numerics = [v for v in (votes[a][i] for a in range(M)) if isinstance(v, (int, float)) and v != float("inf")]
                final_preds.append(median(numerics) if numerics else float("inf"))
        else:
            for i in range(N):
                bins: defaultdict[float, int] = defaultdict(int)
                for v in (votes[a][i] for a in range(M)):
                    if not isinstance(v, (int, float)) or v == float("inf"):
                        continue
                    placed = False
                    for key in list(bins.keys()):
                        if numeric_equal(v, key, args.tolerance):
                            bins[key] += 1; placed = True; break
                    if not placed:
                        bins[v] = 1
                final_preds.append(max(bins.items(), key=lambda x: x[1])[0] if bins else float("inf"))
        acc = sum(numeric_equal(p, l, args.tolerance) for p, l in zip(final_preds, labels)) / N

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
    out_path = out_dir / _safe_fname(stem, suffix=".json", max_len=120)  # well below 255

    # Avoid huge deepcopy; just attach predictions & correctness
    for rec, pred in zip(data, final_preds):
        rec["prediction"] = pred
        if args.dataset.lower() == "aqua":
            rec["correct"] = (pred == rec["answer"])
        else:
            rec["correct"] = numeric_equal(pred, float(rec["answer"]), args.tolerance)

    with open(out_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Saved →", str(out_path))
    wandb.finish()

if __name__ == "__main__":
    main()
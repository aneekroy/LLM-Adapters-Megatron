#!/usr/bin/env python
# vLLM ensemble evaluator with multi-LoRA preloading and original path semantics

import hashlib
import argparse, copy, datetime, glob, json, os, re
from collections import Counter, defaultdict
from statistics import median

import wandb
from tqdm.auto import tqdm

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# ───────────────────────── args ─────────────────────────
import os, re, hashlib, time
from pathlib import Path

def _short_lora_tag(lora_weights_csv: str) -> str:
    """
    Produce a short, stable tag for a set of LoRA adapters.
    Example: 'base-p5-a1' or 'al50-p3-hf3a21e7' (variant, part count, 8-hex hash).
    """
    if not lora_weights_csv:
        return "base-p0"

    # split "part1=/abs/dir1,part2=/abs/dir2,..."
    parts = [seg.split("=", 1)[1] if "=" in seg else seg
             for seg in lora_weights_csv.split(",") if seg.strip()]
    n = len(parts)

    # detect variant from any path
    s = ",".join(parts)
    variant = "al50" if "-al50" in s else ("rand50" if "-rand50" in s else "base")

    # stable short hash of the full CSV
    h = hashlib.sha1(lora_weights_csv.encode("utf-8")).hexdigest()[:8]
    return f"{variant}-p{n}-{h}"

def _safe_fname(stem: str, suffix: str = ".json", max_len: int = 200) -> str:
    """Sanitize and clamp filename length (excluding directory)."""
    # simple slug
    stem = re.sub(r"[^A-Za-z0-9._+-]+", "-", stem).strip("-")
    # clamp length accounting for suffix
    max_stem = max_len - len(suffix)
    if len(stem) > max_stem:
        stem = stem[:max_stem]
    return f"{stem}{suffix}"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["AddSub","MultiArith","SingleEq","gsm8k","AQuA","SVAMP"], required=True)
    p.add_argument("--model", choices=[
        "LLaMA-7B","BLOOM-7B","GPT-j-6B",
        "Llama-3.2-1B","Llama-3.2-3B",
        "Llama-3.2-1B-Instruct","Llama-3.2-3B-Instruct",
        "Llama-3.2-70B-Instruct","Llama-3.2-70B","Llama-3.2-3B-Instruct-Sparse",
        "Qwen3-4B-Instruct","Qwen3-8B-Sparse"
    ], required=True)
    p.add_argument("--base_model", required=True)
    p.add_argument(
        "--lora_weights",
        required=True,
        help=("Comma-separated list or glob of LoRA adapter directories, "
              "e.g. '/path/a1,/path/a2,/exp/*/adapter'. "
              "Optionally use name=path, e.g. 'math=/p/a1,reason=/p/a2'")
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--ensemble_rule", choices=["vote","median"], default="vote")
    p.add_argument("--tolerance", type=float, default=1e-3)

    # vLLM & LoRA cache
    p.add_argument("--tp_size", type=int, default=1)
    p.add_argument("--max_model_len", type=int, default=None)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--max_loras", type=int, default=1, help="Max LoRAs in GPU cache")
    p.add_argument("--max_cpu_loras", type=int, default=None, help="Max LoRAs in CPU cache (default=max_loras)")
    p.add_argument("--max_lora_rank", type=int, default=32)

    # preloading
    p.add_argument("--preload", action="store_true", help="Warm all adapters with a 1-token generate")
    p.add_argument("--preload_gpu_keep", type=int, default=None,
                   help="Override --max_loras during preloading to keep N adapters hot on GPU")
    return p.parse_args()

# ───────────────────────── helpers ─────────────────────────

def _expand_one_token(token: str):
    """Return list of (name, path) pairs from one comma token.
       Supports 'name=path' or plain path/glob (name inferred from basename)."""
    pairs = []
    if "=" in token:
        name, raw = token.split("=", 1)
        name = name.strip()
        raw = raw.strip()
        paths = sorted(glob.glob(raw)) if any(ch in raw for ch in "*?[") else [raw.rstrip("/")]
        for p in paths:
            pairs.append((name, p))
    else:
        raw = token.strip()
        paths = sorted(glob.glob(raw)) if any(ch in raw for ch in "*?[") else [raw.rstrip("/")]
        for p in paths:
            inferred = os.path.basename(p.rstrip("/")) or f"adapter_{len(pairs)+1}"
            pairs.append((inferred, p))
    return pairs

def expand_adapters(spec: str):
    """Original semantics: comma-separated entries; each can be a path or a glob.
       NEW: optional name=path. Ensures unique names and existing dirs."""
    out = []
    seen_names = set()
    for part in spec.split(","):
        if not part.strip():
            continue
        for name, path in _expand_one_token(part):
            if not os.path.isdir(path):
                raise FileNotFoundError(f"LoRA path not found or not a dir: {path}")
            base = name
            i = 1
            while name in seen_names:
                i += 1
                name = f"{base}_{i}"
            seen_names.add(name)
            out.append((name, path))
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

def batches(lst, bs):
    for i in range(0, len(lst), bs):
        yield lst[i : i + bs]

def build_llm(args):
    max_loras = args.preload_gpu_keep if args.preload_gpu_keep is not None else args.max_loras
    llm = LLM(
        model=args.base_model,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        trust_remote_code=True,
        enable_lora=True,
        max_loras=max_loras,
        max_cpu_loras=(args.max_cpu_loras if args.max_cpu_loras is not None else max_loras),
        max_lora_rank=args.max_lora_rank,
    )
    sp = SamplingParams(temperature=0.1, top_p=0.75, top_k=40, max_tokens=256)
    return llm, sp

def vllm_generate(llm, sampling_params, instructions, lora_req=None):
    prompts = [generate_prompt(i) for i in instructions]
    outs = llm.generate(prompts, sampling_params, lora_request=lora_req)
    return [o.outputs[0].text.strip() if o.outputs else "" for o in outs]

NUM_REGEX = re.compile(r"-?\d+\.?\d*")

def extract_answer(args, text: str):
    text = text.replace(",", "")
    if args.dataset.lower() == "aqua":
        m = re.search(r"[ABCDE]", text)
        return m.group(0) if m else ""
    nums = NUM_REGEX.findall(text)
    if not nums:
        return float("inf")
    try:
        return float(nums[-1])
    except ValueError:
        return float("inf")

def numeric_equal(a: float, b: float, tol: float):
    return abs(a - b) <= tol

def _slugify(s: str) -> str:
    # keep letters/digits/._-; replace everything else with '-'
    s = re.sub(r'[^A-Za-z0-9._-]+', '-', s)
    return re.sub(r'-+', '-', s).strip('-')


def preload_adapters(llm, adapters):
    """Warm each adapter once to populate the caches."""
    warm = SamplingParams(temperature=0.0, max_tokens=1)
    for idx, (name, path) in enumerate(adapters, start=1):
        llm.generate(["[warmup]"], warm, lora_request=LoRARequest(name, idx, path))

args = parse_args()

def _lora_tag(s: str) -> str:
    """
    Turn 'part1=/a/b/ensA_part1/round_0,part2=/a/b/ensA_part2/round_0,...'
    into a short, filesystem-safe tag like:
    'part1-ensA_part1-round_0_part2-ensA_part2-round_0_part3-ensA_part3-round_0'
    """
    parts = []
    for seg in s.split(','):
        if '=' in seg:
            k, p = seg.split('=', 1)
            p = p.rstrip('/')
            leaf = os.path.basename(p)
            parent = os.path.basename(os.path.dirname(p))
            parts.append(f"{k}-{parent}-{leaf}")
        else:
            parts.append(os.path.basename(seg.rstrip('/')))
    return _slugify('_'.join(parts))

# Build safe output path
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # if you already have a ts, reuse it
model_tag = _slugify(args.model)
lora_tag  = _lora_tag(args.lora_weights)
ds_tag    = _slugify(args.dataset)

out_dir = os.path.join("experiment", model_tag)
os.makedirs(out_dir, exist_ok=True)

# If you worry about filename length, append a short hash:
short = hashlib.sha1(args.lora_weights.encode()).hexdigest()[:10]

out_path = os.path.join(
    out_dir, f"{model_tag}-{lora_tag}-{ds_tag}-{ts}-{short}.json"
)
# ───────────────────────── main ─────────────────────────

def main():
    args = parse_args()
    run_id = f"{args.model}-{args.dataset}-{os.path.basename(args.base_model).strip('/')}-{lora_tag}-{datetime.datetime.now():%Y%m%d_%H%M%S}"
    wandb.init(project="lora_math_ensemble_eval_al50", name=run_id, config=vars(args), reinit=True)

    data = load_data(args.dataset)
    labels = [d["answer"] for d in data]
    if args.dataset.lower() != "aqua":
        labels = [float(x) if not isinstance(x, (int, float)) else x for x in labels]

    llm, sampling_params = build_llm(args)

    # << HERE: original semantics preserved (+ optional naming)
    adapters = expand_adapters(args.lora_weights)   # list[(name, path)]
    M = len(adapters)

    if args.preload:
        # print(f"Preloading {M} LoRA adapters (GPU slots={llm.engine_args.max_loras}, CPU cache={llm.engine_args.max_cpu_loras}) …")
        gpu_slots = args.preload_gpu_keep if args.preload_gpu_keep is not None else args.max_loras
        cpu_slots = args.max_cpu_loras if args.max_cpu_loras is not None else gpu_slots
        print(f"Preloading {M} LoRA adapters (GPU slots={gpu_slots}, CPU cache={cpu_slots}) …")
        preload_adapters(llm, adapters)

    votes = [[None] * len(data) for _ in range(M)]

    for a_idx, (adapter_name, path) in enumerate(adapters, start=1):
        print(f"\n>> [{a_idx}/{M}] {adapter_name} → {path}")
        lora_req = LoRARequest(adapter_name, a_idx, path)

        seen = 0
        for batch in tqdm(list(batches(data, args.batch_size)), leave=False):
            instr = [d["instruction"] for d in batch]
            outs = vllm_generate(llm, sampling_params, instr, lora_req)
            preds = [extract_answer(args, o) for o in outs]
            for j, p in enumerate(preds):
                votes[a_idx-1][seen + j] = p
            seen += len(batch)

            if args.dataset.lower() == "aqua":
                corr = sum(p == lbl for p, lbl in zip(votes[a_idx-1][:seen], labels[:seen]))
            else:
                corr = sum(numeric_equal(p, lbl, args.tolerance) for p, lbl in zip(votes[a_idx-1][:seen], labels[:seen]))
            wandb.log({f"{adapter_name}_acc": corr / max(1, seen), "samples_seen": seen})

    # ensemble
    final_preds = []
    for i in range(len(data)):
        pred_list = [votes[a][i] for a in range(M)]
        if args.dataset.lower() == "aqua":
            tally = Counter(pred_list)
            final_preds.append(tally.most_common(1)[0][0] if tally else "")
        else:
            if args.ensemble_rule == "median":
                numerics = [p for p in pred_list if isinstance(p, (int, float)) and p != float("inf")]
                final_preds.append(median(numerics) if numerics else float("inf"))
            else:
                bins: defaultdict[float, int] = defaultdict(int)
                for p in pred_list:
                    if not isinstance(p, (int, float)) or p == float("inf"):
                        continue
                    placed = False
                    for key in list(bins.keys()):
                        if numeric_equal(p, key, args.tolerance):
                            bins[key] += 1; placed = True; break
                    if not placed: bins[p] = 1
                final_preds.append(max(bins.items(), key=lambda x: x[1])[0] if bins else float("inf"))

    if args.dataset.lower() == "aqua":
        acc = sum(p == l for p, l in zip(final_preds, labels)) / len(labels)
    else:
        acc = sum(numeric_equal(p, l, args.tolerance) for p, l in zip(final_preds, labels)) / len(labels)

    print(f"\n==== ENSEMBLE (M={M}) | ACC={acc:.4f} ====\n")
    wandb.log({"ensemble_accuracy": acc})

    out = copy.deepcopy(data)
    for rec, pred in zip(out, final_preds):
        rec["prediction"] = pred
        if args.dataset.lower() == "aqua":
            rec["correct"] = pred == rec["answer"]
        else:
            rec["correct"] = numeric_equal(pred, float(rec["answer"]), args.tolerance)


    fname = _safe_fname(run_id, suffix=".json", max_len=200)  # keep plenty below 255
    out_dir = Path("experiment")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / fname)

    json.dump(out, open(out_path, "w"), indent=2)
    print("Saved →", out_path)
    wandb.finish()

if __name__ == "__main__":
    main()
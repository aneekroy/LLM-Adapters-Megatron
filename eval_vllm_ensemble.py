#!/usr/bin/env python
# vLLM 3-model ensemble evaluator (no LoRA). Loads models sequentially.

import argparse, copy, datetime, glob, json, os, re, gc, hashlib
from collections import Counter, defaultdict
from statistics import median

import wandb
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

# ───────────────────────── args ─────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["AddSub","MultiArith","SingleEq","gsm8k","AQuA","SVAMP"], required=True)

    # NEW: exactly 3 base models, comma-separated. Can be HF IDs or local dirs.
    p.add_argument(
        "--models", required=True,
        help=("Comma-separated list of THREE base models (no LoRA), e.g.\n"
              "  meta-llama/Llama-3.2-1B-Instruct,meta-llama/Llama-3.2-3B-Instruct,/path/to/local-3B\n"
              "Glob patterns allowed for local paths. Exactly three must resolve.")
    )

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--ensemble_rule", choices=["vote","median"], default="vote")
    p.add_argument("--tolerance", type=float, default=1e-3)

    # vLLM runtime knobs
    p.add_argument("--tp_size", type=int, default=1)
    p.add_argument("--max_model_len", type=int, default=None)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--max_tokens", type=int, default=256)

    return p.parse_args()

# ───────────────────────── helpers ─────────────────────────

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

def build_llm(model_id_or_path: str, args):
    return LLM(
        model=model_id_or_path,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        trust_remote_code=True,
        # LoRA explicitly NOT enabled here
    ), SamplingParams(temperature=0.1, top_p=0.75, top_k=40, max_tokens=args.max_tokens)

def vllm_generate(llm, sp, instructions):
    prompts = [generate_prompt(i) for i in instructions]
    outs = llm.generate(prompts, sp)
    return [o.outputs[0].text.strip() if o.outputs else "" for o in outs]

NUM_REGEX = re.compile(r"-?\d+\.?\d*")

def extract_answer(dataset_name: str, text: str):
    text = text.replace(",", "")
    if dataset_name.lower() == "aqua":
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
    s = re.sub(r'[^A-Za-z0-9._/-]+', '-', s)
    s = re.sub(r'-+', '-', s).strip('-')
    return s.replace('/', '_')

def _expand_models(spec: str):
    out = []
    for token in spec.split(","):
        raw = token.strip()
        if not raw:
            continue
        if any(ch in raw for ch in "*?["):
            out.extend(sorted(glob.glob(raw)))
        else:
            out.append(raw.rstrip("/"))
    # Strict: exactly 3 as requested
    if len(out) != 3:
        raise ValueError(f"--models must resolve to exactly 3 entries, got {len(out)}: {out}")
    return out

# ───────────────────────── main ─────────────────────────

def main():
    args = parse_args()
    data = load_data(args.dataset)
    labels = [d["answer"] for d in data]
    if args.dataset.lower() != "aqua":
        labels = [float(x) if not isinstance(x, (int, float)) else x for x in labels]

    models = _expand_models(args.models)
    model_names = [os.path.basename(m).strip("/") or m for m in models]
    model_tag = "__".join(_slugify(m) for m in model_names)
    ds_tag = _slugify(args.dataset)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short = hashlib.sha1(args.models.encode()).hexdigest()[:10]

    run_id = f"{ds_tag}-{model_tag}-{ts}-{short}"
    wandb.init(project="vll_math_ensemble_eval_noLora", name=run_id, config=vars(args), reinit=True)

    M = len(models)
    votes = [[None] * len(data) for _ in range(M)]

    # Evaluate each model sequentially (keeps VRAM usage sane)
    for m_idx, (model_id, nice_name) in enumerate(zip(models, model_names), start=1):
        print(f"\n>> [{m_idx}/{M}] {nice_name} → {model_id}")
        llm, sp = build_llm(model_id, args)

        seen = 0
        for batch in tqdm(list(batches(data, args.batch_size)), leave=False):
            instr = [d["instruction"] for d in batch]
            outs = vllm_generate(llm, sp, instr)
            preds = [extract_answer(args.dataset, o) for o in outs]
            for j, p in enumerate(preds):
                votes[m_idx-1][seen + j] = p
            seen += len(batch)

            # running accuracy logging
            if args.dataset.lower() == "aqua":
                corr = sum(p == lbl for p, lbl in zip(votes[m_idx-1][:seen], labels[:seen]))
            else:
                corr = sum(numeric_equal(p, lbl, args.tolerance) for p, lbl in zip(votes[m_idx-1][:seen], labels[:seen]))
            wandb.log({f"{nice_name}_acc": corr / max(1, seen), "samples_seen": seen})

        # free memory before next model
        del llm
        gc.collect()

    # Ensemble across models
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
        rec["correct"] = (pred == rec["answer"]) if args.dataset.lower() == "aqua" \
            else numeric_equal(pred, float(rec["answer"]), args.tolerance)

    os.makedirs("experiment", exist_ok=True)
    out_path = f"experiment/{run_id}.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print("Saved →", out_path)
    wandb.finish()

if __name__ == "__main__":
    main()
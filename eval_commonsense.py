#!/usr/bin/env python
import os, sys, re, json, argparse, copy, datetime
from typing import List, Dict, Tuple, Optional

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer,
    GenerationConfig, set_seed
)

# Optional: PEFT/ID3 support (HF path)
try:
    from peft import PeftModel, PeftConfig
except Exception:
    PeftModel = None
    PeftConfig = None

try:
    from selective_optimizers.load_store import load_summary_from_disk, load_weights_from_summary
except Exception:
    load_summary_from_disk = None
    load_weights_from_summary = None

# Optional: vLLM
VLLM_AVAILABLE = False
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except Exception:
    pass

# Optional: W&B
WANDB_AVAILABLE = False
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', choices=['hf','vllm'], default='vllm')
    parser.add_argument('--dataset', choices=["boolq","piqa","social_i_qa","hellaswag","winogrande","ARC-Challenge","ARC-Easy","openbookqa"], required=True)

    parser.add_argument('--model', choices=['LLaMA-7B','LLaMA-13B','BLOOM-7B','GPT-j-6B','Other',
                                            'Llama-3.2-1B','Llama-3.2-3B','Llama-3.2-1B-Instruct','Llama-3.2-3B-Instruct','Qwen3-4B-Instruct','Qwen3-4B-Sparse','Qwen3-8B','Qwen3-8B-Sparse',
                                            'Llama-3.2-11B-Vision-Instruct','Qwen3-8B-Instruct','Qwen3-8B-Sparse','Qwen3-8B'],
                        required=True)

    parser.add_argument('--adapter', choices=['none','LoRA','ID3','AdapterP','AdapterH','Parallel'], default='none')
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--tokenizer_path', required=True)
    parser.add_argument('--adapter_weights', default="")
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--load_8bit', action='store_true', default=False)
    parser.add_argument('--cutoff_len', type=int, default=2048)

    # sampling/generation
    parser.add_argument('--do_sample', type=lambda x: str(x).lower()=='true', default=False)  # default deterministic
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=32)

    # vLLM extras
    parser.add_argument('--tp_size', type=int, default=1)
    parser.add_argument('--dtype', choices=['auto','float16','bfloat16'], default='auto')

    # logging/output
    parser.add_argument('--output_dir', default="")
    parser.add_argument('--report_to', default="")  # 'wandb' or ''
    parser.add_argument('--wandb_project', default="commonsense-eval")
    parser.add_argument('--wandb_run_name', default="")
    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():  # type: ignore
            return "mps"
    except Exception:
        pass
    return "cpu"


# -------------------- Dataset helpers -------------------- #

DATASET_CANON = {
    "boolq": ["true", "false"],
    "piqa": ["solution1","solution2"],
    "hellaswag": ["ending1","ending2","ending3","ending4"],
    "winogrande": ["option1","option2"],
    "ARC-Challenge": ["A","B","C","D"],
    "ARC-Easy": ["A","B","C","D"],
    "openbookqa": ["A","B","C","D"],
    # social_i_qa has 3 options traditionally; we allow up to 5 canonical tokens in case of variants
    "social_i_qa": ["answer1","answer2","answer3","answer4","answer5"],
}

# def dataset_instruction_suffix(dataset: str) -> str:
#     opts = DATASET_CANON[dataset]
#     if dataset in ("ARC-Challenge","ARC-Easy","openbookqa"):
#         return f"Answer with a single token from: {', '.join(opts)}."
#     elif dataset == "boolq":
#         return 'Answer with a single token: "true" or "false".'
#     else:
#         return f"Answer with exactly one of: {', '.join(opts)}."
    
def dataset_instruction_suffix(dataset: str) -> str:
    opts = DATASET_CANON[dataset]
    if dataset in ("ARC-Challenge","ARC-Easy","openbookqa"):
        return "Respond with ONLY the single letter A, B, C, or D on one line. No words or punctuation."
    elif dataset == "boolq":
        return 'Respond with ONLY one token: "true" or "false" (lowercase), on one line.'
    else:
        return f"Respond with EXACTLY one of: {', '.join(opts)}. Output only that token, on one line."
    
def format_prompt(dataset: str, instruction: str, input_text: Optional[str]=None) -> str:
    suffix = dataset_instruction_suffix(dataset)
    if input_text and input_text.strip():
        return f"""Below is an instruction that describes a task, paired with an input that provides context. Write a response that completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Constraint:
{suffix}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that completes the request.

### Instruction:
{instruction}

### Constraint:
{suffix}

### Response:
"""

# Canonicalize labels and predictions aggressively
def canon_label(dataset: str, raw) -> str:
    if raw is None:
        return ""
    s = str(raw).strip().lower()

    if dataset == "boolq":
        if s in ("true","t","yes","y","1"): return "true"
        if s in ("false","f","no","n","0"): return "false"
        return s

    if dataset == "piqa":
        if s in ("solution1","1","a"): return "solution1"
        if s in ("solution2","2","b"): return "solution2"
        return s

    if dataset == "winogrande":
        if s in ("option1","1","a"): return "option1"
        if s in ("option2","2","b"): return "option2"
        return s

    if dataset in ("ARC-Challenge","ARC-Easy","openbookqa"):
        # Accept A/B/C/D, 'answer1'..'answer4', and 1..4
        m = re.search(r'\banswer\s*([1-4])\b', s)
        if m:
            return "ABCD"[int(m.group(1)) - 1]
        m = re.search(r'\b([1-4])\b', s)
        if m:
            return "ABCD"[int(m.group(1)) - 1]
        m = re.search(r'\b([abcd])\b', s)
        if m:
            return m.group(1).upper()
        return s.upper() if s in ("a","b","c","d") else s

    if dataset == "hellaswag":
        # ending1..4; also accept 1..4 / a..d
        m = re.search(r'\bending([1-4])\b', s)
        if m: return f"ending{m.group(1)}"
        m = re.search(r'\b([1-4])\b', s)
        if m: return f"ending{m.group(1)}"
        m = re.search(r'\b([abcd])\b', s)
        if m:
            idx = "abcd".index(m.group(1))
            return f"ending{idx+1}"
        return s

    if dataset == "social_i_qa":
        # usually 1..3; allow up to 5
        m = re.search(r'\banswer([1-5])\b', s)
        if m: return f"answer{m.group(1)}"
        m = re.search(r'\b([1-5])\b', s)
        if m: return f"answer{m.group(1)}"
        m = re.search(r'\b([abcde])\b', s)
        if m:
            idx = "abcde".index(m.group(1))
            return f"answer{idx+1}"
        return s

    return s


def extract_prediction(dataset: str, text: str) -> str:
    """
    Extract the model's choice robustly.
    """
    if text is None:
        return ""
    s = str(text).strip().lower()

    # First try canonical keys directly
    opts = DATASET_CANON[dataset]
    pat = r'\b(' + "|".join([re.escape(o) for o in opts]) + r')\b'
    m = re.search(pat, s)
    if m:
        return m.group(1)

    # Fall back to numeric/letter aliases
    return canon_label(dataset, s)


# -------------------- Data / batching -------------------- #

def load_data(dataset: str) -> List[Dict]:
    path = f"dataset/{dataset}/test.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find dataset file: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a list of examples")
    # Expect keys: instruction, answer (optionally input)
    return data

def make_batches(items: List[Dict], batch_size: int) -> List[List[Dict]]:
    out = []
    for i in range(0, len(items), batch_size):
        out.append(items[i:i+batch_size])
    return out


# -------------------- HF path -------------------- #

def hf_load_tokenizer(tokenizer_path: str, base_model: str, is_llama: bool):
    if is_llama:
        tok = LlamaTokenizer.from_pretrained(base_model, trust_remote_code=True)
    else:
        tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = 0
    return tok

def hf_load_model(args, device: str):
    is_llama = "LLaMA" in args.model or "Llama" in args.model
    tokenizer = hf_load_tokenizer(args.tokenizer_path, args.base_model, is_llama)

    torch_dtype = torch.float16
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    if args.dtype in dtype_map:
        torch_dtype = dtype_map[args.dtype]

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=args.load_8bit,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else {"": device},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    # Optional: attach & merge adapter
    if args.adapter.lower() != "none" and args.adapter_weights:
        if PeftModel is None:
            raise RuntimeError("peft not available but adapter requested.")
        if args.adapter.lower() == "id3":
            if load_summary_from_disk is None or load_weights_from_summary is None:
                raise RuntimeError("ID3 support requested but selective_optimizers not available.")
        # Load PEFT config
        peft_cfg = PeftConfig.from_pretrained(args.adapter_weights)
        model = PeftModel.from_pretrained(
            model, args.adapter_weights,
            torch_dtype=torch_dtype,
            device_map=None,
            trust_remote_code=True,
        )
        # If ID3, materialize weights into the backbone before merge
        if args.adapter.lower() == "id3":
            summary_path = os.path.join(args.adapter_weights, "summary.pt")
            summary = load_summary_from_disk(summary_path)
            load_weights_from_summary(model, summary)

        # Merge and unload to vanilla model (faster inference)
        model = model.merge_and_unload()
        model = model.to(device)
    else:
        model = model.to(device)

    model.eval()
    return tokenizer, model

def hf_generate(model, tokenizer, prompts: List[str], gen_cfg: GenerationConfig, max_new_tokens: int, device: str) -> List[str]:
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=getattr(gen_cfg, "max_length", 2048),
    )
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            generation_config=gen_cfg,
            return_dict_in_generate=True,
            output_scores=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
    decoded = tokenizer.batch_decode(out.sequences, skip_special_tokens=True)
    # Split at the last "### Response:" to grab only the answer segment
    answers = []
    for d in decoded:
        parts = d.split("### Response:")
        ans = parts[-1].strip() if parts else d.strip()
        answers.append(ans)
    return answers


# -------------------- vLLM path -------------------- #

def to_vllm_dtype(s: str):
    if s == "bfloat16": return "bfloat16"
    if s == "float16":  return "float16"
    return "auto"

def vllm_load(args):
    if not VLLM_AVAILABLE:
        raise RuntimeError("vLLM is not installed but engine=vllm was requested.")

    llm = LLM(
        model=args.base_model,
        trust_remote_code=True,
        tokenizer=args.tokenizer_path,   # add this
        tensor_parallel_size=max(1, int(args.tp_size)),
        dtype=to_vllm_dtype(args.dtype),
    )
    adapter_name = None
    if args.adapter.lower() == "lora" and args.adapter_weights:
        adapter_name = "eval_adapter"
        # vLLM runtime LoRA
        llm.load_adapter(adapter_name, args.adapter_weights)

    # vLLM will use the tokenizer from the model; no need for separate tokenizer here.
    return llm, adapter_name

def vllm_generate(llm, prompts: List[str], do_sample: bool, temperature: float,
                  top_p: float, top_k: int, num_beams: int, max_new_tokens: int,
                  adapter_name: Optional[str]) -> List[str]:
    # vLLM does not support beam search with SamplingParams; stick to greedy or sampling.
    from vllm import SamplingParams
    import inspect

    params = SamplingParams(
        temperature=temperature if do_sample else 0.0,
        top_p=top_p if do_sample else 1.0,
        top_k=top_k if do_sample else 0,
        max_tokens=max_new_tokens,
    )

    sig = inspect.signature(llm.generate)
    kwargs = dict(use_tqdm=False)

    # Only add adapter argument if the installed vLLM supports it
    if adapter_name:
        if 'adapter_name' in sig.parameters:
            kwargs['adapter_name'] = adapter_name
        elif 'lora_request' in sig.parameters:
            try:
                from vllm.lora.request import LoRARequest
                # weight=1.0; adapter must be pre-loaded via llm.load_adapter(...)
                kwargs['lora_request'] = LoRARequest(adapter_name, adapter_weight=1.0)
            except Exception:
                pass  # fall back to no adapter
                print("No Adapter Found !!")

    outputs = llm.generate(prompts, params, **kwargs)
    answers = []
    for o in outputs:
        text = o.outputs[0].text if o.outputs else ""
        parts = text.split("### Response:")
        answers.append((parts[-1] if parts else text).strip())
    return answers


# -------------------- Evaluation loop -------------------- #

def main():
    args = parse_args()
    device = get_device()
    if args.seed is not None:
        print("=" * 72)
        print(f"Setting seed to {args.seed}")
        set_seed(int(args.seed))

    # W&B
    use_wandb = (args.report_to.lower() == "wandb") and WANDB_AVAILABLE
    if args.report_to.lower() == "wandb" and not WANDB_AVAILABLE:
        print("WARNING: report_to=wandb but wandb not installed; continuing without W&B.")

    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"{args.dataset}-{args.engine}-{datetime.datetime.now().isoformat(timespec='seconds')}",
            config={
                "engine": args.engine,
                "dataset": args.dataset,
                "model": args.base_model,
                "adapter": args.adapter,
                "adapter_weights": args.adapter_weights,
                "do_sample": args.do_sample,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "num_beams": args.num_beams,
                "max_new_tokens": args.max_new_tokens,
                "batch_size": args.batch_size,
                "seed": args.seed,
            }
        )

    data = load_data(args.dataset)
    n_examples = len(data)
    batches = make_batches(data, args.batch_size)

    # Output dirs/files
    if args.output_dir:
        results_dir = os.path.join(args.output_dir, args.dataset)
    elif args.adapter_weights:
        results_dir = os.path.join(args.adapter_weights, args.dataset)
    else:
        raise ValueError("Provide --output_dir or --adapter_weights to place results.")
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, f"seed_{args.seed}.json")
    log_path  = os.path.join(results_dir, f"seed_{args.seed}.txt")
    lf = open(log_path, "w")

    # Build prompt fn and generation config
    gen_cfg = GenerationConfig(
        do_sample=bool(args.do_sample),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        num_beams=int(args.num_beams),
    )

    # Load engine
    tokenizer = None
    model = None
    llm = None
    adapter_name = None

    if args.engine == "hf":
        tokenizer, model = hf_load_model(args, device)
    else:
        llm, adapter_name = vllm_load(args)

    correct = 0
    seen = 0
    outputs_all = []
    pbar = tqdm(total=len(batches), desc="Evaluating")

    for bidx, batch in enumerate(batches):
        instructions = [ex.get("instruction","") for ex in batch]
        inputs = [ex.get("input","") for ex in batch]
        prompts = [format_prompt(args.dataset, ins, inp) for ins, inp in zip(instructions, inputs)]

        if args.engine == "hf":
            outs = hf_generate(model, tokenizer, prompts, gen_cfg, args.max_new_tokens, device)
        else:
            outs = vllm_generate(llm, prompts, args.do_sample, args.temperature,
                                 args.top_p, args.top_k, args.num_beams,
                                 args.max_new_tokens, adapter_name)

        for ex, raw in zip(batch, outs):
            gold_raw = ex.get("answer", None)
            gold = canon_label(args.dataset, gold_raw)
            pred = canon_label(args.dataset, extract_prediction(args.dataset, raw))

            is_correct = (gold == pred) and (gold in DATASET_CANON[args.dataset])
            correct += 1 if is_correct else 0
            seen += 1

            rec = copy.deepcopy(ex)
            rec["output_pred_text"] = raw
            rec["pred"] = pred
            rec["gold"] = gold
            rec["correct"] = bool(is_correct)
            outputs_all.append(rec)

        # Running logs (*** fix: divide by examples, not batches ***)
        running_acc = correct / max(1, seen)
        lf.write(f"batch {bidx+1}/{len(batches)} | seen {seen}/{n_examples} | correct {correct} | acc {running_acc:.4f}\n")
        lf.flush()

        if use_wandb:
            wandb.log({
                "bidx": bidx,
                "seen_examples": seen,
                "correct_count": correct,
                "running_accuracy": running_acc
            })

        # persist JSON incrementally
        with open(json_path, "w") as jf:
            json.dump(outputs_all, jf, indent=2)

        pbar.update(1)

    pbar.close()
    final_acc = correct / max(1, n_examples)
    print(f"\nFINAL: {correct}/{n_examples} = {final_acc:.8f} accuracy")
    lf.write(f"\nFINAL: {correct}/{n_examples} = {final_acc:.8f} accuracy\n")
    lf.close()

    if use_wandb:
        wandb.log({"final_accuracy": final_acc})
        wandb.finish()


if __name__ == "__main__":
    main()
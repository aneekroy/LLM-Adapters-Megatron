#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vLLM evaluator with LoRA auto-merge, version-safe decoding (beams if supported,
otherwise deterministic greedy), engine-level seeding, batched generation, and GSM8K parsing.
"""

import argparse
import copy
import datetime
import json
import math
import os
import re
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

# Optional W&B
try:
    import wandb
    WANDB_OK = True
except Exception:
    WANDB_OK = False

# vLLM
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
import inspect


# ---------------------------- CLI ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("vLLM + LoRA evaluator")

    p.add_argument('--dataset', choices=['AddSub','MultiArith','SingleEq','gsm8k','AQuA','SVAMP'], required=True)
    p.add_argument('--model', type=str, required=True, help="Label used in filenames/logs")

    # LoRA + base model
    p.add_argument('--adapter', choices=['None','LoRA','AdapterP','AdapterH','Parallel','Prefix'], default='None')
    p.add_argument('--base_model', required=True, help="Path or HF id to base/merged model")
    p.add_argument('--lora_weights', default='', help="Path/HF id to LoRA weights (if adapter=LoRA)")
    p.add_argument('--merged_out', default='', help="Dir to save merged weights (auto if empty)")

    # vLLM engine / resources
    p.add_argument('--dtype', choices=['float16','bfloat16','auto'], default='float16')
    p.add_argument('--tp_size', type=int, default=1)
    p.add_argument('--gpu_mem_util', type=float, default=0.9)
    p.add_argument('--max_model_len', type=int, default=None)
    p.add_argument('--enforce_eager', action='store_true')
    p.add_argument('--seed', type=int, default=1234, help='Engine seed (reproducibility)')

    # Decoding (HF-like defaults)
    p.add_argument('--num_beams', type=int, default=4, help='If unsupported, falls back to greedy')
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--top_p', type=float, default=1.0)
    p.add_argument('--top_k', type=int, default=0)
    p.add_argument('--max_new_tokens', type=int, default=256)
    p.add_argument('--stop', type=str, nargs='*', default=[])

    # Eval + logging
    p.add_argument('--batch_size_gen', type=int, default=1)
    p.add_argument('--miss', type=float, default=1e-3)  # numeric tolerance
    p.add_argument('--save_every', type=int, default=25)
    p.add_argument('--output_dir', type=str, default='experiment')
    p.add_argument('--wandb_project', type=str, default='')
    p.add_argument('--wandb_entity', type=str, default='')
    p.add_argument('--wandb_mode', choices=['online','offline','disabled'], default='online')

    return p.parse_args()


# ---------------------------- Main ----------------------------
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("vLLM requires CUDA GPUs.")

    # TP sanity
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    n_vis = len(vis.split(",")) if vis else torch.cuda.device_count()
    if args.tp_size > max(1, n_vis):
        raise ValueError(f"--tp_size={args.tp_size} exceeds available GPUs ({n_vis})")

    # W&B (optional)
    run = None
    if WANDB_OK and args.wandb_project and args.wandb_mode != 'disabled':
        os.environ["WANDB_MODE"] = args.wandb_mode
        run_name = f"{args.model}-{args.adapter}-{args.dataset}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        kw: Dict[str, Any] = dict(project=args.wandb_project, config=vars(args))
        if args.wandb_entity:
            kw["entity"] = args.wandb_entity
        run = wandb.init(name=run_name, **kw)

    # Data
    data_path = os.path.join("dataset", args.dataset, "test.json")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"cannot find dataset file: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        raise ValueError("dataset JSON must be a list of examples")

    # LoRA: auto-merge if provided
    model_path = args.base_model
    if args.adapter.lower() == "lora" and args.lora_weights:
        merged_dir = args.merged_out or os.path.abspath(
            f".merged_{os.path.basename(args.base_model).replace('/', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        model_path = merge_lora_auto(args.base_model, args.lora_weights, merged_dir)

    # Engine (seed here; NOT in SamplingParams)
    engine_args = EngineArgs(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=max(1, int(args.tp_size)),
        gpu_memory_utilization=float(args.gpu_mem_util),
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        enforce_eager=bool(args.enforce_eager),
        seed=int(args.seed),
    )
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=max(1, int(args.tp_size)),
        gpu_memory_utilization=float(args.gpu_mem_util),
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        enforce_eager=bool(args.enforce_eager),
        seed=int(args.seed),
    )

    # Build version-safe SamplingParams
    sp = build_sampling_params_safe(args)

    # Output file
    os.makedirs(args.output_dir, exist_ok=True)
    save_file = os.path.join(args.output_dir, f'{args.model}-{args.adapter}-{args.dataset}.json')

    # Eval
    total = len(dataset)
    correct = 0
    out_buf: List[Dict[str, Any]] = []
    pbar = tqdm(total=total, ncols=100, desc="Evaluating")

    bsz = max(1, int(args.batch_size_gen))
    idx = 0
    while idx < total:
        chunk = dataset[idx: idx + bsz]
        prompts = [generate_prompt(ex.get("instruction", ""), None) for ex in chunk]

        try:
            outs = llm.generate(prompts, sp)
        except Exception as e:
            print(f"[error] vLLM generate failed at idx={idx}: {e}")
            outs = []

        # Extract text after "### Response:" if present
        texts: List[str] = []
        for o in outs:
            txt = o.outputs[0].text if (o and getattr(o, "outputs", None)) else ""
            parts = txt.split("### Response:")
            texts.append(parts[1].strip() if len(parts) > 1 else txt.strip())
        while len(texts) < len(chunk):
            texts.append("")

        # Score
        for j, ex in enumerate(chunk):
            pred_text = texts[j]
            label_raw = ex.get("answer")
            ds = args.dataset.lower()

            flag = False
            record: Dict[str, Any]

            if ds == 'aqua':
                pred_letter = extract_answer_letter(pred_text)
                label_letter = (str(label_raw).strip().upper() if label_raw is not None else '')
                if pred_letter and label_letter and pred_letter == label_letter:
                    correct += 1
                    flag = True
                record = {**copy.deepcopy(ex), "output_pred": pred_text,
                          "pred_letter": pred_letter, "label_letter": label_letter, "flag": flag}
            else:
                label_val = coerce_float(label_raw)
                pred_val = extract_answer_number(pred_text)
                if all(v is not None and math.isfinite(v) for v in (label_val, pred_val)):
                    if abs(label_val - pred_val) <= float(args.miss):
                        correct += 1
                        flag = True
                record = {**copy.deepcopy(ex), "output_pred": pred_text,
                          "pred_value": pred_val, "label_value": label_val, "flag": flag}

            out_buf.append(record)

            cur = idx + j + 1
            if run:
                if ds == 'aqua':
                    wandb.log({"step": cur, "running_accuracy": correct / cur,
                               "flag_correct": flag,
                               "prediction_text": record.get("pred_letter", ""),
                               "label_text": record.get("label_letter", "")}, step=cur)
                else:
                    wandb.log({"step": cur, "running_accuracy": correct / cur,
                               "flag_correct": flag,
                               "prediction_value": record.get("pred_value", float("nan")),
                               "label_value": record.get("label_value", float("nan"))}, step=cur)

            if (cur % max(1, int(args.save_every))) == 0:
                safe_dump_json(out_buf, save_file)

            pbar.update(1)

        idx += bsz

    pbar.close()
    acc = correct / total if total else 0.0
    safe_dump_json(out_buf, save_file)
    print(f"\nTest finished. Accuracy: {acc:.6f} ({correct}/{total})")

    if run:
        wandb.log({"final_accuracy": acc, "correct": correct, "total": total})
        wandb.finish()


# ---------------------------- Helpers ----------------------------
def build_sampling_params_safe(args) -> SamplingParams:
    """
    Build SamplingParams that works across vLLM versions:
    - If this vLLM supports beam search (has 'use_beam_search' & 'beam_width'),
      use it when num_beams > 1.
    - Else, fall back to deterministic greedy (temperature=0, top_k=0, top_p=1).
    """
    sig = inspect.signature(SamplingParams.__init__)
    supports_beam = ("use_beam_search" in sig.parameters) and ("beam_width" in sig.parameters)

    kwargs: Dict[str, Any] = {
        "n": 1,
        "best_of": None,
        "max_tokens": int(args.max_new_tokens),
        "stop": (args.stop or None),
    }

    if supports_beam and args.num_beams and args.num_beams > 1:
        kwargs.update({
            "use_beam_search": True,
            "beam_width": int(args.num_beams),
        })
    else:
        # Deterministic greedy/sampling
        kwargs.update({
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(args.top_k),
        })

    return SamplingParams(**kwargs)


def safe_dump_json(obj: Any, path: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def generate_prompt(instruction: str, input_text: Optional[str] = None) -> str:
    instruction = instruction or ""
    if input_text:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
            "### Response:\n"
        )
    else:
        return (
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Response:\n"
        )


def coerce_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return None


def extract_answer_number(text: str) -> Optional[float]:
    if not text:
        return None
    s = text.replace(",", "")
    m = re.findall(r'(?i)(?:answer|ans)\s*[:=]\s*(-?\d+(?:\.\d+)?)', s)
    if m:
        try:
            return float(m[-1])
        except Exception:
            pass
    nums = re.findall(r'-?\d+(?:\.\d+)?', s)
    if nums:
        try:
            return float(nums[-1])
        except Exception:
            return None
    return None


def extract_answer_letter(text: str) -> str:
    if not text:
        return ''
    matches = re.findall(r'\b([A-E])\b', text.upper())
    return matches[-1] if matches else ''


def merge_lora_auto(base_model: str, lora_weights: str, out_dir: str) -> str:
    """Merge LoRA into base model and return merged dir path.

    If `out_dir` already exists and is non-empty, assume it's a previously
    merged model and just return the path without doing any work.
    """
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Fast path: reuse existing merge
    if os.path.isdir(out_dir):
        files = set(os.listdir(out_dir))
        has_cfg = "config.json" in files
        has_weights = any(
            name.startswith(("model", "pytorch_model")) and name.endswith((".safetensors", ".bin"))
            for name in files
        )
        if has_cfg and has_weights:
            print(f"[merge] Reusing existing merged directory: {out_dir}")
            return out_dir

    print(f"[merge] Loading base model: {base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"[merge] Loading LoRA weights: {lora_weights}")
    peft_model = PeftModel.from_pretrained(
        base,
        lora_weights,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    print("[merge] Merging and unloading...")
    merged = peft_model.merge_and_unload()

    print(f"[merge] Saving merged model to: {out_dir}")
    merged.save_pretrained(out_dir, safe_serialization=True)
    try:
        tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
        tok.save_pretrained(out_dir)
    except Exception:
        pass
    print("[merge] Done.")
    return out_dir


# ---------------------------- Entrypoint ----------------------------
if __name__ == "__main__":
    main()
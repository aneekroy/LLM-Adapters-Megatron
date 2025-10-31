#!/usr/bin/env python
# active_learning.py

import os, math, json, random, inspect, torch, transformers, fire
from typing import Callable, Dict
from datasets import load_dataset, concatenate_datasets

# Try to import the prompt fn from your finetuner, allow override via kwargs
try:
    from finetune import train as finetune_train
    try:
        from finetune import generate_and_tokenize_prompt as DEFAULT_PROMPT_FN  # optional
    except Exception:
        DEFAULT_PROMPT_FN = None
except Exception as e:
    raise RuntimeError(f"Cannot import finetune.train: {e}")

# ───────────── helpers ─────────────

def make_al_prompt_fn(tokenizer, cutoff_len):
    """Fallback prompt→tokens for common JSON schemas (instruction/input/output, question/answer, prompt/response)."""
    def _fn(x):
        # Build a plain text prompt
        if "instruction" in x:
            prompt = x["instruction"] + (("\n" + x.get("input","")) if x.get("input") else "")
        else:
            prompt = x.get("prompt") or x.get("question") or x.get("input") or x.get("text") or ""

        target = x.get("output") or x.get("answer") or x.get("response") or ""

        text = prompt + (("\n\n" + target) if target else "")
        toks = tokenizer(text, truncation=True, max_length=cutoff_len, padding=False)
        ids = toks["input_ids"]; att = toks["attention_mask"]
        return {"input_ids": ids, "attention_mask": att, "labels": ids.copy()}
    return _fn

def _ensure_pad(tokenizer: transformers.PreTrainedTokenizer):
    if tokenizer.pad_token_id is None:
        # fall back to eos
        if tokenizer.eos_token_id is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        else:
            tokenizer.pad_token = tokenizer.eos_token

def _filter_and_map_finetune_kwargs(fn: Callable, cutoff_len: int, kwargs: Dict):
    """
    Map common aliases to whatever 'finetune_train' exposes and drop unknowns.
    """
    sig = inspect.signature(fn)
    params = set(sig.parameters)

    # alias candidates per incoming key (ordered by preference)
    alias_map = {
        "num_train_epochs": ["num_train_epochs", "epochs", "num_epochs"],
        "epochs": ["epochs", "num_epochs", "num_train_epochs"],
        "learning_rate": ["learning_rate", "lr"],
        "lr": ["lr", "learning_rate"],
        "micro_batch_size": ["micro_batch_size", "per_device_train_batch_size"],
        "per_device_train_batch_size": ["per_device_train_batch_size", "micro_batch_size"],
        "gradient_accumulation_steps": ["gradient_accumulation_steps"],
        "seed": ["seed"],
        "lora_r": ["lora_r", "r"],
        "lora_alpha": ["lora_alpha", "alpha"],
        "lora_dropout": ["lora_dropout", "dropout"],
        "cutoff_len": ["cutoff_len", "max_seq_len"],
        "val_set_size": ["val_set_size", "validation_size", "eval_holdout_size"],
    }

    # Always try to pass cutoff_len, but only if accepted
    kwargs = dict(kwargs)  # shallow copy
    kwargs.setdefault("cutoff_len", cutoff_len)

    mapped = {}
    for k, v in kwargs.items():
        candidates = [k] + alias_map.get(k, [])
        target = next((c for c in candidates if c in params), None)
        if target is not None:
            mapped[target] = v  # last one wins if duplicated
    return mapped

def _batched_loss(model, tokenizer, ds, batch_size=8):
    """
    Return per-example negative-log-likelihood proxy: loss * (seq_len!=pad).
    Assumes ds items already have input_ids/attention_mask/labels.
    """
    _ensure_pad(tokenizer)
    collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, collate_fn=collator, shuffle=False, pin_memory=True
    )
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(model.device, non_blocking=True) for k, v in batch.items()}
            out = model(**batch)  # out.loss: mean over tokens/batch
            # approximate per-seq NLL by scaling mean loss with token counts per seq
            tok_counts = batch["input_ids"].ne(tokenizer.pad_token_id).sum(dim=1)  # [B]
            seq_loss = out.loss.detach() * tok_counts  # broadcast scalar → [B]
            losses.extend(seq_loss.float().cpu().tolist())
    return losses

def _select_top_k(pool_ds, scores, k):
    idx = torch.as_tensor(scores).topk(k).indices.tolist()
    sel = pool_ds.select(idx)
    rem = pool_ds.select(sorted(set(range(len(pool_ds))) - set(idx)))
    return sel, rem

def _tokenize_pool(pool, map_fn: Callable, keep_cols=("input_ids","attention_mask","labels")):
    tok = pool.map(map_fn, remove_columns=[c for c in pool.column_names if c not in keep_cols])
    return tok

def _load_peft_for_scoring(base_model: str, ckpt_dir: str):
    """
    Load base model + LoRA adapters from ckpt_dir if available.
    Fallback to base model if adapters missing.
    """
    AutoModel = transformers.AutoModelForCausalLM
    base = AutoModel.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    # Prefer PEFT directory layout (adapter_config.json present)
    try:
        from peft import PeftModel
        if os.path.exists(os.path.join(ckpt_dir, "adapter_config.json")):
            return PeftModel.from_pretrained(base, ckpt_dir)
        # Legacy: explicit state dict (adapter_model.bin)
        adapter_bin = os.path.join(ckpt_dir, "adapter_model.bin")
        if os.path.exists(adapter_bin):
            from peft import set_peft_model_state_dict, get_peft_model, LoraConfig
            # Try reading config alongside; if missing, still try to load state dict
            state = torch.load(adapter_bin, map_location="cpu")
            # We rely on PEFT internals to attach modules; if config missing, this may fail.
            set_peft_model_state_dict(base, state)
            return base
    except Exception as e:
        print(f"[WARN] Could not load LoRA adapters from {ckpt_dir}: {e}")
    print("[WARN] Using base model for scoring (no adapters found).")
    return base

# ────────────────────────────────────────────────

def active_learning(base_model:str,
                    data_path:str,
                    output_dir:str="./al-run",
                    rounds:int=5,
                    init_frac:float=0.1,
                    acq_frac:float=0.1,
                    cutoff_len:int=256,
                    seed:int=42,
                    **finetune_kwargs):
    """
    Pool-based AL with uncertainty sampling (loss).
    - data_path: JSON file or glob that datasets.load_dataset("json", ...) accepts.
    - Pass any extra training kwargs; unsupported ones are ignored safely.
    """
    transformers.set_seed(seed)
    random.seed(seed)

    full = load_dataset("json", data_files=data_path)["train"]
    assert 0 < init_frac < 1 and 0 < acq_frac < 1

    # Step 0 – seed labelled set
    init_n = math.ceil(len(full) * init_frac)
    split = full.train_test_split(test_size=len(full) - init_n, shuffle=True, seed=seed)
    labelled, pool = split["train"], split["test"]

    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.padding_side = "right"
    _ensure_pad(tokenizer)

    # Choose prompt/tokenize fn
    safe_val = max(1, int(0.1 * len(labelled)))
    safe_val = min(safe_val, max(1, len(labelled) - 1))
    finetune_kwargs.setdefault("val_set_size", safe_val)

    map_fn = finetune_kwargs.get("generate_and_tokenize_prompt", DEFAULT_PROMPT_FN)
    if map_fn is None:
        try:
            map_fn = make_al_prompt_fn(tokenizer, cutoff_len)
        except Exception as e:
            raise RuntimeError(
                "No 'generate_and_tokenize_prompt' provided and finetune.generate_and_tokenize_prompt not found.\n"
                "Pass one via CLI: --generate_and_tokenize_prompt <callable-in-scope> or implement import in this file."
            ) from e

    for r in range(rounds):
        print(f"\n=== Active-learning round {r} | labelled={len(labelled)} ===")
        round_out = os.path.join(output_dir, f"round_{r}")
        os.makedirs(round_out, exist_ok=True)

        # ─── train on current labelled set ───
        tmp_json = os.path.join(round_out, "labelled.json")
        with open(tmp_json, "w") as f:
            json.dump(labelled.to_list(), f, ensure_ascii=False, indent=2)

        mapped_kwargs = _filter_and_map_finetune_kwargs(
            finetune_train, cutoff_len, finetune_kwargs
        )

        # Call your trainer with only supported args
        finetune_train(
            base_model=base_model,
            data_path=tmp_json,
            output_dir=round_out,
            **mapped_kwargs
        )

        # ─── stop if pool is empty or last round ───
        if len(pool) == 0 or r == rounds - 1:
            print("AL loop finished.")
            break

        # ─── compute uncertainty on pool ───
        model = _load_peft_for_scoring(base_model, round_out)
        # Prepare tokenized pool (expects map_fn to build input_ids/attention_mask/labels)
        pool_tok = _tokenize_pool(pool, map_fn)
        scores = _batched_loss(model, tokenizer, pool_tok, batch_size=8)

        # ─── acquire top-k by uncertainty ───
        k = min(math.ceil(len(full) * acq_frac), len(pool))
        acquired, pool = _select_top_k(pool, scores, k)
        labelled = concatenate_datasets([labelled, acquired])

if __name__ == "__main__":
    fire.Fire(active_learning)
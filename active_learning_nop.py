#!/usr/bin/env python
# active_learning.py
# Pool-based Active Learning with non-overlapping selections,
# per-sample uncertainty (default = log-perplexity), W&B logging, and roundwise artifacts.

import os, math, json, random, inspect
from typing import Callable, Dict, List, Optional

import fire
import torch
import torch.nn.functional as F
import transformers
import numpy as np
from datasets import load_dataset, concatenate_datasets

# ──────────────────────────────────────────────────────────────────────────────
# W&B (optional; keep training runnable if wandb is missing)
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False

def _wandb_init(
    enable: bool,
    project: Optional[str],
    entity: Optional[str],
    name: Optional[str],
    group: Optional[str],
    mode: Optional[str],
    tags: Optional[List[str]],
    config: Dict,
):
    if not enable or not _WANDB_AVAILABLE:
        return None
    kwargs = dict(project=project, entity=entity, name=name, group=group, config=config)
    if mode is not None:
        kwargs["mode"] = mode
    if tags:
        kwargs["tags"] = tags
    try:
        return wandb.init(**{k: v for k, v in kwargs.items() if v is not None})
    except Exception as e:
        print(f"[WARN] wandb.init failed: {e}. Continuing without W&B.")
        return None

def _wandb_log(run, payload: Dict):
    if run is None:
        return
    try:
        wandb.log(payload)
    except Exception as e:
        print(f"[WARN] wandb.log failed: {e}")

def _wandb_artifact_add(run, name: str, type_: str, files: List[str]):
    if run is None:
        return
    try:
        art = wandb.Artifact(name=name, type=type_)
        for f in files:
            if f and os.path.exists(f):
                art.add_file(f, name=os.path.basename(f))
        run.log_artifact(art)
    except Exception as e:
        print(f"[WARN] wandb artifact failed: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Try to import your finetuner entrypoints
try:
    from finetune import train as finetune_train
    try:
        from finetune import generate_and_tokenize_prompt as DEFAULT_PROMPT_FN  # optional
    except Exception:
        DEFAULT_PROMPT_FN = None
except Exception as e:
    raise RuntimeError(f"Cannot import finetune.train: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Tokenization & prompt helpers

def make_al_prompt_fn(tokenizer, cutoff_len: int):
    """Fallback prompt→tokens for common JSON schemas (instruction/input/output, Q/A, prompt/response)."""
    def _fn(x):
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
        if tokenizer.eos_token_id is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        else:
            tokenizer.pad_token = tokenizer.eos_token

def _tokenize_pool(pool, map_fn: Callable,
                   keep_cols=("input_ids","attention_mask","labels")):
    drop_cols = [c for c in pool.column_names if c not in keep_cols]
    return pool.map(map_fn, remove_columns=drop_cols)

# ──────────────────────────────────────────────────────────────────────────────
# Finetune-args mapping

def _filter_and_map_finetune_kwargs(fn: Callable, cutoff_len: int, kwargs: Dict) -> Dict:
    """Map common aliases to the signature of finetune_train; drop unknowns safely."""
    sig = inspect.signature(fn)
    params = set(sig.parameters)
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
    kws = dict(kwargs)
    kws.setdefault("cutoff_len", cutoff_len)

    mapped = {}
    for k, v in kws.items():
        candidates = [k] + alias_map.get(k, [])
        target = next((c for c in candidates if c in params), None)
        if target is not None:
            mapped[target] = v
    return mapped

# ──────────────────────────────────────────────────────────────────────────────
# Device helpers (fix CPU↔CUDA mismatches)

def _pick_device_from_env() -> torch.device:
    if torch.cuda.is_available():
        try:
            lr = int(os.environ.get("LOCAL_RANK", "0"))
        except ValueError:
            lr = 0
        return torch.device(f"cuda:{lr}")
    return torch.device("cpu")

def _pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        try:
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except Exception:
            return torch.float16
    return torch.float32

def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")

# ──────────────────────────────────────────────────────────────────────────────
# Uncertainty scoring

def _per_sample_logppl(model, tokenizer, ds, batch_size=8) -> List[float]:
    """Per-sequence log perplexity = mean NLL/token. Higher ⇒ more uncertain."""
    _ensure_pad(tokenizer)
    collator = transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, collate_fn=collator, shuffle=False, pin_memory=True)
    logppls: List[float] = []
    model.eval()
    dev = _model_device(model)
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            # causal shift
            labels = batch["labels"][:, 1:].contiguous()
            input_ids = batch["input_ids"][:, :-1].contiguous()
            attn = batch["attention_mask"][:, :-1].contiguous()
            out = model(input_ids=input_ids, attention_mask=attn)
            logits = out.logits
            loss_tok = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).view(labels.size())
            # use shifted attention to mask pads
            mask = attn.ne(0)
            tok_sums = (loss_tok * mask).sum(dim=1)
            tok_counts = mask.sum(dim=1).clamp_min(1)
            logppl = (tok_sums / tok_counts).float()
            logppls.extend(logppl.cpu().tolist())
    return logppls

def _batched_loss_proxy(model, tokenizer, ds, batch_size=8) -> List[float]:
    """Approx per-example NLL proxy via model.loss * non-pad token count."""
    _ensure_pad(tokenizer)
    collator = transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, collate_fn=collator, shuffle=False, pin_memory=True)
    scores: List[float] = []
    model.eval()
    dev = _model_device(model)
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            out = model(**batch)
            tok_counts = batch["input_ids"].ne(tokenizer.pad_token_id).sum(dim=1)
            seq_score = out.loss.detach() * tok_counts
            scores.extend(seq_score.float().cpu().tolist())
    return scores

# ──────────────────────────────────────────────────────────────────────────────
# PEFT/adapter for scoring

def _load_peft_for_scoring(base_model: str, ckpt_dir: str):
    """
    Load base model + LoRA adapters from ckpt_dir if available; otherwise fallback to base.
    IMPORTANT: keep the entire model on ONE device to avoid CPU↔CUDA mismatches.
    """
    dev = _pick_device_from_env()
    dtype = _pick_dtype()

    AutoModel = transformers.AutoModelForCausalLM
    model = AutoModel.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=None,           # <<< single device, not "auto"
        trust_remote_code=True,
    ).to(dev)

    try:
        from peft import PeftModel, set_peft_model_state_dict
        if os.path.exists(os.path.join(ckpt_dir, "adapter_config.json")):
            model = PeftModel.from_pretrained(model, ckpt_dir)
            model.to(dev)
            return model
        adapter_bin = os.path.join(ckpt_dir, "adapter_model.bin")
        if os.path.exists(adapter_bin):
            state = torch.load(adapter_bin, map_location="cpu")
            set_peft_model_state_dict(model, state)
            model.to(dev)
            return model
    except Exception as e:
        print(f"[WARN] Could not load LoRA adapters from {ckpt_dir}: {e}")
    print("[WARN] Using base model for scoring (no adapters found).")
    return model

# ──────────────────────────────────────────────────────────────────────────────
# Utility: selection, I/O, dedup

def _select_top_k(pool_ds, scores: List[float], k: int):
    idx = torch.as_tensor(scores).topk(k).indices.tolist()
    sel = pool_ds.select(idx)
    rem = pool_ds.select(sorted(set(range(len(pool_ds))) - set(idx)))
    return sel, rem

def _save_json(ds, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(ds.to_list(), f, ensure_ascii=False, indent=2)

def _dedup_by_idx(ds, idx_col="_al_idx"):
    seen, keep = set(), []
    for i, v in enumerate(ds[idx_col]):
        if v in seen:
            continue
        seen.add(v); keep.append(i)
    return ds.select(keep)

# ──────────────────────────────────────────────────────────────────────────────
# Main AL

def active_learning(
    base_model: str,
    data_path: str,
    output_dir: str = "./al-run",
    rounds: int = 3,                # al50 default: 0.1 + 0.2 + 0.2
    init_frac: float = 0.1,
    acq_frac: float = 0.2,
    cutoff_len: int = 256,
    seed: int = 42,
    uncertainty: str = "logppl",    # {"logppl", "loss"}
    scoring_batch_size: int = 8,

    # W&B controls
    wandb_enable: bool = True,                      # set False to disable explicitly
    wandb_project: Optional[str] = None,            # or via env WANDB_PROJECT
    wandb_entity: Optional[str] = None,             # or env WANDB_ENTITY
    wandb_run_name: Optional[str] = None,           # or env WANDB_NAME
    wandb_group: Optional[str] = None,
    wandb_mode: Optional[str] = None,               # "online" | "offline" | "disabled"
    wandb_tags: Optional[str] = None,               # comma-separated
    wandb_log_artifacts: bool = True,

    **finetune_kwargs,
):
    """
    Active Learning with non-overlapping selections and W&B logging.

    Artifacts per round r:
      round_r/labelled.json
      round_0/selection_seed_0pXX.json
      round_*/selection_acquire_0pYY.json
    """
    transformers.set_seed(seed)
    random.seed(seed)

    # Dataset + stable indices
    full = load_dataset("json", data_files=data_path)["train"]
    total_n = len(full)
    assert total_n > 0, "Empty dataset."
    assert 0 < init_frac < 1 and 0 < acq_frac < 1, "Fractions must be in (0,1)."
    full = full.add_column("_al_idx", list(range(total_n)))

    # Seed split
    init_n = math.ceil(total_n * init_frac)
    split = full.train_test_split(test_size=total_n - init_n, shuffle=True, seed=seed)
    labelled, pool = split["train"], split["test"]
    chosen_ids = set(labelled["_al_idx"])

    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.padding_side = "right"
    _ensure_pad(tokenizer)

    # Default val size for small seeds
    safe_val = max(1, int(0.1 * len(labelled)))
    safe_val = min(safe_val, max(1, len(labelled) - 1))
    finetune_kwargs.setdefault("val_set_size", safe_val)

    # Prompt fn
    map_fn = finetune_kwargs.get("generate_and_tokenize_prompt", DEFAULT_PROMPT_FN)
    if map_fn is None:
        map_fn = make_al_prompt_fn(tokenizer, cutoff_len)

    # W&B setup
    tags = [t.strip() for t in wandb_tags.split(",")] if wandb_tags else None
    wb_config = dict(
        base_model=base_model, data_path=data_path, output_dir=output_dir,
        rounds=rounds, init_frac=init_frac, acq_frac=acq_frac, cutoff_len=cutoff_len,
        seed=seed, uncertainty=uncertainty, scoring_batch_size=scoring_batch_size,
        **{f"ft_{k}": v for k, v in finetune_kwargs.items()}
    )
    wb = _wandb_init(
        enable=wandb_enable, project=wandb_project, entity=wandb_entity,
        name=wandb_run_name, group=wandb_group, mode=wandb_mode, tags=tags, config=wb_config
    )

    # Persist seed selection
    os.makedirs(output_dir, exist_ok=True)
    r0_dir = os.path.join(output_dir, "round_0")
    os.makedirs(r0_dir, exist_ok=True)
    seed_sel_path = os.path.join(r0_dir, f"selection_seed_0p{int(init_frac*100):02d}.json")
    _save_json(labelled, seed_sel_path)

    _wandb_log(wb, {
        "round": 0,
        "labelled_size": int(len(labelled)),
        "pool_size": int(len(pool)),
        "val_set_size": int(finetune_kwargs["val_set_size"]),
        "event": "seed_selection_saved",
    })
    if wandb_log_artifacts:
        _wandb_artifact_add(wb, name="al_seed_selection", type_="al-round", files=[seed_sel_path])

    # Round loop
    for r in range(rounds):
        print(f"\n=== Active-learning round {r} | labelled={len(labelled)} | pool={len(pool)} ===")
        round_out = os.path.join(output_dir, f"round_{r}")
        os.makedirs(round_out, exist_ok=True)

        # Cumulative labelled set for this round
        labelled = _dedup_by_idx(labelled)
        labelled_path = os.path.join(round_out, "labelled.json")
        _save_json(labelled, labelled_path)

        # Train
        tmp_json = os.path.join(round_out, "labelled_tmp.json")
        _save_json(labelled, tmp_json)
        mapped_kwargs = _filter_and_map_finetune_kwargs(finetune_train, cutoff_len, finetune_kwargs)

        _wandb_log(wb, {
            "round": r,
            "labelled_size": int(len(labelled)),
            "pool_size": int(len(pool)),
            "event": "train_start",
        })

        finetune_train(
            base_model=base_model,
            data_path=tmp_json,
            output_dir=round_out,
            **mapped_kwargs
        )

        _wandb_log(wb, {"round": r, "event": "train_done", "round_output_dir": round_out})

        # Stop if last round or pool empty
        if r == rounds - 1 or len(pool) == 0:
            _wandb_log(wb, {"round": r, "event": "al_finished"})
            break

        # Score pool
        model = _load_peft_for_scoring(base_model, round_out)
        pool_tok = _tokenize_pool(pool, map_fn)

        if uncertainty.lower() == "logppl":
            scores = _per_sample_logppl(model, tokenizer, pool_tok, batch_size=scoring_batch_size)
        elif uncertainty.lower() == "loss":
            scores = _batched_loss_proxy(model, tokenizer, pool_tok, batch_size=scoring_batch_size)
        else:
            raise ValueError("uncertainty must be one of {'logppl','loss'}")

        # Acquire next non-overlapping acq_frac of FULL
        k = min(math.ceil(total_n * acq_frac), len(pool))
        acquired, pool = _select_top_k(pool, scores, k)

        # Non-overlap guard
        acquired_ids = set(acquired["_al_idx"])
        inter = acquired_ids & chosen_ids
        if inter:
            print(f"[WARN] Overlap detected ({len(inter)}); filtering and topping up.")
            keep_mask = [(_id not in chosen_ids) for _id in acquired["_al_idx"]]
            keep_idx = [i for i, ok in enumerate(keep_mask) if ok]
            acquired = acquired.select(keep_idx)
            acquired_ids = set(acquired["_al_idx"])
            need = k - len(acquired)
            if need > 0 and len(pool) > 0:
                pool_tok2 = _tokenize_pool(pool, map_fn)
                if uncertainty.lower() == "logppl":
                    scores2 = _per_sample_logppl(model, tokenizer, pool_tok2, batch_size=scoring_batch_size)
                else:
                    scores2 = _batched_loss_proxy(model, tokenizer, pool_tok2, batch_size=scoring_batch_size)
                topup, pool = _select_top_k(pool, scores2, min(need, len(pool)))
                acquired = concatenate_datasets([acquired, topup])
                acquired_ids |= set(topup["_al_idx"])

        # Persist newly acquired set
        sel_name = f"selection_acquire_0p{int(acq_frac*100):02d}.json"
        sel_path = os.path.join(round_out, sel_name)
        _save_json(acquired, sel_path)

        # Round metrics (aggregate stats on uncertainty scores)
        scores_np = np.asarray(scores, dtype=np.float32)
        _wandb_log(wb, {
            "round": r,
            "acquired_k": int(len(acquired)),
            "pool_after": int(len(pool)),
            "uncertainty_metric": uncertainty,
            "uncertainty_mean": float(scores_np.mean()) if scores_np.size else float("nan"),
            "uncertainty_std": float(scores_np.std()) if scores_np.size else float("nan"),
            "uncertainty_p50": float(np.percentile(scores_np, 50)) if scores_np.size else float("nan"),
            "uncertainty_p90": float(np.percentile(scores_np, 90)) if scores_np.size else float("nan"),
            "selection_path": sel_path,
            "labelled_path": labelled_path,
            "event": "acquire_done",
        })
        if wandb_log_artifacts:
            _wandb_artifact_add(wb, name=f"al_round_{r}", type_="al-round", files=[labelled_path, sel_path])

        # Update trackers
        chosen_ids |= acquired_ids
        labelled = concatenate_datasets([labelled, acquired])

    print(f"[OK] Active learning completed. Outputs in: {output_dir}")
    if wb is not None:
        try:
            wb.finish()
        except Exception:
            pass

# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fire.Fire(active_learning)
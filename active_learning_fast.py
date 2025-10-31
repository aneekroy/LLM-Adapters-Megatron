#!/usr/bin/env python
# active_learning_fast.py (fixed tokenizer routing & robust finetune subprocess + stable scoring)

import os, re, math, json, random, inspect, hashlib, time, sys, subprocess
from contextlib import nullcontext
from typing import Callable, Dict, List, Optional, Iterable, Tuple

import fire
import torch
import torch.nn.functional as F
import transformers
import numpy as np
from datasets import load_dataset, concatenate_datasets, Dataset

# ===================== Global perf switches =====================
# Disable Inductor CUDA Graphs (they’re brittle with PEFT + dynamic shapes)
os.environ.setdefault("TORCHINDUCTOR_DISABLE_CUDAGRAPHS", "1")
try:
    import torch._inductor.config as _ind_cfg
    _ind_cfg.cudagraphs = False
    _ind_cfg.triton.cudagraphs = False
except Exception:
    pass

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# IMPORTANT: disable autotuned algo changes across batches for stable replays
torch.backends.cudnn.benchmark = False

# Sensible default to limit fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ===================== W&B (optional) =====================
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False

def _wandb_init(enable, project, entity, name, group, mode, tags, config):
    if (not enable) or (not _WANDB_AVAILABLE):
        return None
    try:
        kw = dict(project=project, entity=entity, name=name, group=group, config=config)
        if mode is not None: kw["mode"] = mode
        if tags: kw["tags"] = tags
        return wandb.init(**{k: v for k, v in kw.items() if v is not None})
    except Exception as e:
        print(f"[WARN] wandb.init failed: {e}. Continuing without W&B.")
        return None

def _wandb_log(run, payload):
    if run is not None:
        try: wandb.log(payload)
        except Exception as e: print(f"[WARN] wandb.log failed: {e}")

def _wandb_artifact_add(run, name: str, type_: str, files: List[str]):
    if run is None: return
    try:
        art = wandb.Artifact(name=name, type=type_)
        for f in files:
            if f and os.path.exists(f): art.add_file(f, name=os.path.basename(f))
        run.log_artifact(art)
    except Exception as e:
        print(f"[WARN] wandb artifact failed: {e}")

# ===================== Finetune entrypoints =====================
try:
    from finetune import train as finetune_train  # for signature only
    try:
        from finetune import generate_and_tokenize_prompt as DEFAULT_PROMPT_FN
    except Exception:
        DEFAULT_PROMPT_FN = None
except Exception as e:
    raise RuntimeError(f"Cannot import finetune.train: {e}")

# ===================== Tokenization helpers (scoring uses fixed length) =====================
def make_al_prompt_fn(tokenizer, cutoff_len: int):
    def _fn(x):
        if "instruction" in x:
            prompt = x["instruction"] + (("\n" + x.get("input","")) if x.get("input") else "")
        else:
            prompt = x.get("prompt") or x.get("question") or x.get("input") or x.get("text") or ""
        target = x.get("output") or x.get("answer") or x.get("response") or ""
        text = prompt + (("\n\n" + target) if target else "")
        toks = tokenizer(
            text,
            truncation=True,
            max_length=cutoff_len,
            padding="max_length"   # fixed-length to stabilize shapes during scoring
        )
        ids = toks["input_ids"]; att = toks["attention_mask"]
        return {"input_ids": ids, "attention_mask": att, "labels": ids.copy()}
    return _fn

def _ensure_pad(tk: transformers.PreTrainedTokenizer):
    if tk.pad_token_id is None:
        if tk.eos_token_id is None:
            tk.add_special_tokens({"pad_token":"[PAD]"})
        else:
            tk.pad_token = tk.eos_token

def _tokenize_pool(ds, map_fn: Callable, keep=("input_ids","attention_mask","labels","_al_idx")) -> Dataset:
    drop = [c for c in ds.column_names if c not in keep]
    return ds.map(map_fn, remove_columns=drop)

def _resolve_tokenizer_src(base_model: str, tokenizer_path: Optional[str]) -> str:
    """
    1) If tokenizer_path provided, use it.
    2) If base_model under .../nvidia-sparse/...-Sparse-x, try sibling dense dir .../nvidia/... (strip -Sparse-...).
    3) Else, use base_model.
    """
    if tokenizer_path:
        return tokenizer_path
    base_dir = os.path.dirname(base_model)
    name = os.path.basename(base_model)
    dense_name = re.sub(r"-Sparse-\d+(?:\.\d+)?$", "", name)
    if "nvidia-sparse" in base_dir:
        dense_dir = base_dir.replace("nvidia-sparse", "nvidia")
        candidate = os.path.join(dense_dir, dense_name)
        if os.path.isdir(candidate):
            return candidate
    return base_model

# ===================== Finetune-arg mapping =====================
def _filter_and_map_finetune_kwargs(fn: Callable, cutoff_len: int, kwargs: Dict) -> Dict:
    sig = inspect.signature(fn); params = set(sig.parameters)
    alias = {
        "batch_size": ["batch_size"],  # ensure batch_size is forwarded
        "num_train_epochs": ["num_train_epochs", "epochs", "num_epochs"],
        "epochs": ["epochs", "num_train_epochs"],
        "learning_rate": ["learning_rate", "lr"],
        "lr": ["lr", "learning_rate"],
        "micro_batch_size": ["micro_batch_size"],
        "per_device_train_batch_size": ["per_device_train_batch_size"],
        "gradient_accumulation_steps": ["gradient_accumulation_steps"],
        "seed": ["seed"],
        "lora_r": ["lora_r", "r"],
        "lora_alpha": ["lora_alpha", "alpha"],
        "lora_dropout": ["lora_dropout", "dropout"],
        "cutoff_len": ["cutoff_len", "max_seq_len"],
        "val_set_size": ["val_set_size", "validation_size", "eval_holdout_size"],
        "wandb_run_name": ["wandb_run_name"],
    }
    kws = dict(kwargs)
    kws.setdefault("cutoff_len", cutoff_len)

    # If user didn’t give batch_size, derive from per_device_train_batch_size or fallback to 4.
    if "batch_size" not in kws:
        if "per_device_train_batch_size" in kws:
            kws["batch_size"] = kws["per_device_train_batch_size"]
        else:
            kws["batch_size"] = 4

    out = {}
    for k, v in kws.items():
        cand = [k] + alias.get(k, [])
        tgt = next((c for c in cand if c in params), None)
        if tgt is not None:
            out[tgt] = v
    return out

# ===================== Device helpers =====================
def _pick_device_from_env() -> torch.device:
    if torch.cuda.is_available():
        try: lr = int(os.environ.get("LOCAL_RANK","0"))
        except ValueError: lr = 0
        return torch.device(f"cuda:{lr}")
    return torch.device("cpu")

def _pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        try: return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        except Exception: return torch.float16
    return torch.float32

def _model_device(m: torch.nn.Module) -> torch.device:
    try: return next(m.parameters()).device
    except StopIteration: return torch.device("cpu")

# ===================== DataLoader (fixed-shape scoring) =====================
def _mk_loader(ds: Dataset, tokenizer, batch_size: int):
    # Already padded to max_length during tokenization → do NOT repad
    coll = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=None,
        return_tensors="pt",
        padding=False
    )
    nw = max(0, (os.cpu_count() or 2) // 2)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, collate_fn=coll, shuffle=False,
        pin_memory=True, num_workers=nw, persistent_workers=(nw > 0),
        prefetch_factor=(2 if nw > 0 else None)
    )

# ===================== Scoring =====================
def _per_sample_logppl(model, tokenizer, ds_tok: Dataset, batch_size=8) -> List[float]:
    _ensure_pad(tokenizer); loader = _mk_loader(ds_tok, tokenizer, batch_size)
    out: List[float] = []; model.eval(); dev = _model_device(model)
    amp = torch.cuda.amp.autocast if (torch.cuda.is_available() and dev.type == "cuda") else nullcontext
    with torch.inference_mode():
        for batch in loader:
            batch = {k:v.to(dev, non_blocking=True) for k,v in batch.items()}
            # shift for next-token prediction
            labels = batch["labels"][:,1:].contiguous()
            input_ids = batch["input_ids"][:,:-1].contiguous()
            attn = batch["attention_mask"][:,:-1].contiguous()
            with amp(dtype=(torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16)):
                logits = model(input_ids=input_ids, attention_mask=attn).logits
                loss_tok = F.cross_entropy(
                    logits.transpose(1,2), labels, ignore_index=-100, reduction="none"
                )
            mask = labels.ne(-100)
            tok_sums = (loss_tok * mask).sum(dim=1)
            tok_counts = mask.sum(dim=1).clamp_min(1)
            out.extend((tok_sums / tok_counts).float().cpu().tolist())
    return out

def _batched_loss_proxy(model, tokenizer, ds_tok: Dataset, batch_size=8) -> List[float]:
    _ensure_pad(tokenizer); loader = _mk_loader(ds_tok, tokenizer, batch_size)
    scores: List[float] = []; model.eval(); dev = _model_device(model)
    amp = torch.cuda.amp.autocast if (torch.cuda.is_available() and dev.type == "cuda") else nullcontext
    with torch.inference_mode():
        for batch in loader:
            batch = {k:v.to(dev, non_blocking=True) for k,v in batch.items()}
            with amp(dtype=(torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16)):
                out = model(**batch)
            tok_counts = batch["input_ids"].ne(tokenizer.pad_token_id).sum(dim=1)
            scores.extend((out.loss.detach() * tok_counts).float().cpu().tolist())
    return scores

# ===================== Base model load (for scoring) =====================
def _load_base_for_scoring(base_model: str, score_on_cpu: bool):
    # Eager attention avoids Flash-Attn/graph capture surprises; no compile here.
    common = dict(trust_remote_code=True, low_cpu_mem_usage=True, attn_implementation="eager")
    if score_on_cpu:
        dev = torch.device("cpu")
        m = transformers.AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.float32, device_map=None, **common
        ).to(dev)
        return m
    dev = _pick_device_from_env(); dtype = _pick_dtype()
    m = transformers.AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, device_map=None, **common
    ).to(dev)
    # DO NOT torch.compile() for scoring (PEFT + Inductor cudagraphs are fragile).
    return m

def _attach_adapter(base_model_obj, ckpt_dir: str):
    model = base_model_obj
    try:
        from peft import PeftModel, set_peft_model_state_dict
        cfg = os.path.join(ckpt_dir, "adapter_config.json")
        if os.path.exists(cfg):
            model = PeftModel.from_pretrained(base_model_obj, ckpt_dir)
            return model.to(_model_device(base_model_obj))
        abin = os.path.join(ckpt_dir, "adapter_model.bin")
        if os.path.exists(abin):
            state = torch.load(abin, map_location="cpu")
            set_peft_model_state_dict(model, state)
            return model.to(_model_device(base_model_obj))
    except Exception as e:
        print(f"[WARN] Could not load LoRA from {ckpt_dir}: {e}")
    print("[WARN] Scoring with base (no adapters found).")
    return model.to(_model_device(base_model_obj))

# ===================== Utils =====================
def _topk_ids(ids: List[int], scores: List[float], k: int) -> List[int]:
    top = torch.as_tensor(scores).topk(k).indices.tolist()
    return [ids[i] for i in top]

def _save_json(ds, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(ds.to_list(), f, ensure_ascii=False, indent=2)

def _dedup_by_idx(ds, idx_col="_al_idx"):
    seen, keep = set(), []
    for i, v in enumerate(ds[idx_col]):
        if v in seen: continue
        seen.add(v); keep.append(i)
    return ds.select(keep)

def _build_index_map(ds_tok: Dataset, idx_col="_al_idx") -> Dict[int,int]:
    return { ds_tok[idx_col][i]: i for i in range(len(ds_tok)) }

def _select_view_by_ids(ds_tok: Dataset, ids: Iterable[int], id_to_pos: Dict[int,int]) -> Dataset:
    pos = [id_to_pos[i] for i in ids]
    return ds_tok.select(pos)

# ===================== CLI helpers =====================
def _supports_flag(script_path: str, flag: str) -> bool:
    try:
        out = subprocess.run(
            [sys.executable, script_path, "--help"],
            capture_output=True, text=True, check=False
        )
        hay = (out.stdout or "") + (out.stderr or "")
        return flag in hay
    except Exception:
        return False

def _sanitize_cmd(cmd: List[str]) -> List[str]:
    bad = [t for t in cmd if str(t).strip() == '-' or str(t).startswith('- ')]
    if bad:
        raise RuntimeError(f"Illegal lone '-' or malformed flag(s): {bad}\nCMD: {' '.join(map(str,cmd))}")
    return [str(t) for t in cmd]

# ===================== Subprocess finetune =====================
def _run_finetune_subprocess(base_model, tmp_json, rd, mapped, cuda_visible="0", tokenizer_path: Optional[str]=None):
    finetune_py = os.path.join(os.path.dirname(__file__), "finetune.py")
    cmd = [
        sys.executable, finetune_py,
        "--base_model", base_model,
        "--data_path", tmp_json,
        "--output_dir", rd,
    ]

    # Conditionally append tokenizer_path ONLY if finetune.py exposes it
    if tokenizer_path and _supports_flag(finetune_py, "--tokenizer_path"):
        cmd += ["--tokenizer_path", tokenizer_path]

    # Ensure batch_size present
    if "batch_size" not in mapped:
        mapped["batch_size"] = mapped.get("per_device_train_batch_size", 4)

    for k, v in mapped.items():
        if isinstance(v, bool):
            cmd += [f"--{k}", str(v).lower()]
        else:
            cmd += [f"--{k}", str(v)]

    cmd = _sanitize_cmd(cmd)

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", cuda_visible)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    try:
        print(f"[INFO] Launching finetune subprocess: {' '.join(cmd)}")
        res = subprocess.run(cmd, env=env, check=True)
        return res.returncode
    except subprocess.CalledProcessError as e:
        print(f"[ERR] finetune subprocess failed with code {e.returncode}")
        raise

def _hard_cuda_cleanup():
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

# ===================== Main =====================
def active_learning(
    base_model: str,
    data_path: str,
    output_dir: str = "./al-run",
    rounds: int = 3,
    init_frac: float = 0.1,
    acq_frac: float = 0.2,
    cutoff_len: int = 256,
    seed: int = 42,
    uncertainty: str = "logppl",          # {"logppl", "loss"}
    scoring_batch_size: int = 8,

    # speed/robustness knobs
    two_stage: bool = False,              # shortlist -> exact scoring
    preselect_factor: float = 10.0,       # shortlist size ≈ k * factor
    preselect_cutoff_len: Optional[int] = None,  # if None, uses cutoff_len//2
    score_on_cpu: bool = False,           # fallback if GPU memory is ultra-tight

    # recovery
    resume_round: Optional[int] = None,   # jump to a particular round index

    # W&B controls
    wandb_enable: bool = True,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_group: Optional[str] = None,
    wandb_mode: Optional[str] = None,     # "online"|"offline"|"disabled"
    wandb_tags: Optional[str] = None,     # comma-separated
    wandb_log_artifacts: bool = True,

    # TOKENIZER override (optional)
    tokenizer_path: Optional[str] = None,

    **finetune_kwargs,
):
    transformers.set_seed(seed); random.seed(seed)

    # Resolve tokenizer source
    tok_src = _resolve_tokenizer_src(base_model, tokenizer_path)

    # Dataset + stable indices
    full = load_dataset("json", data_files=data_path)["train"]
    total_n = len(full)
    assert total_n > 0, "Empty dataset."
    assert 0 < init_frac < 1 and 0 < acq_frac < 1, "Fractions must be in (0,1)."
    if "_al_idx" not in full.column_names:
        full = full.add_column("_al_idx", list(range(total_n)))

    # Seed split
    init_n = math.ceil(total_n * init_frac)
    split = full.train_test_split(test_size=total_n - init_n, shuffle=True, seed=seed)
    labelled, pool = split["train"], split["test"]
    chosen_ids = set(labelled["_al_idx"]); pool_ids = set(pool["_al_idx"])

    # Tokenizer (from tok_src)
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True, use_fast=True)
    except Exception:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True, use_fast=False)
    tokenizer.padding_side = "right"; _ensure_pad(tokenizer)

    map_fn = finetune_kwargs.get("generate_and_tokenize_prompt", DEFAULT_PROMPT_FN) or make_al_prompt_fn(tokenizer, cutoff_len)

    # One-time tokenization of FULL (fixed length)
    tokenized_full = _tokenize_pool(full, map_fn)
    id_to_pos = _build_index_map(tokenized_full)

    # Optional two-stage tokenization
    if two_stage:
        st1_cut = preselect_cutoff_len or max(64, cutoff_len // 2)
        map_fn_st1 = make_al_prompt_fn(tokenizer, st1_cut)
        tokenized_full_st1 = _tokenize_pool(full, map_fn_st1)
        id_to_pos_st1 = _build_index_map(tokenized_full_st1)
    else:
        tokenized_full_st1 = None
        id_to_pos_st1 = {}

    # Val size heuristic (if not supplied)
    safe_val = max(1, int(0.1 * len(labelled)))
    safe_val = min(safe_val, max(1, len(labelled) - 1))
    finetune_kwargs.setdefault("val_set_size", safe_val)

    # W&B
    tags = [t.strip() for t in wandb_tags.split(",")] if wandb_tags else None
    wb = _wandb_init(
        enable=wandb_enable, project=wandb_project, entity=wandb_entity,
        name=wandb_run_name, group=wandb_group, mode=wandb_mode, tags=tags,
        config=dict(
            base_model=base_model, tokenizer_path=tok_src, data_path=data_path, output_dir=output_dir,
            rounds=rounds, init_frac=init_frac, acq_frac=acq_frac, cutoff_len=cutoff_len,
            seed=seed, uncertainty=uncertainty, scoring_batch_size=scoring_batch_size,
            two_stage=two_stage, preselect_factor=preselect_factor, preselect_cutoff_len=preselect_cutoff_len,
            resume_round=resume_round, score_on_cpu=score_on_cpu,
            **{f"ft_{k}": v for k,v in finetune_kwargs.items()}
        )
    )

    # Persist seed selection
    os.makedirs(output_dir, exist_ok=True)
    r0 = os.path.join(output_dir, "round_0"); os.makedirs(r0, exist_ok=True)
    _save_json(labelled, os.path.join(r0, f"selection_seed_0p{int(init_frac*100):02d}.json"))
    _wandb_log(wb, {"round":0,"labelled_size":int(len(labelled)),"pool_size":int(len(pool_ids)),"val_set_size":int(finetune_kwargs["val_set_size"]),"event":"seed_selection_saved"})
    if wandb_log_artifacts: _wandb_artifact_add(wb, name="al_seed_selection", type_="al-round", files=[os.path.join(r0, f"selection_seed_0p{int(init_frac*100):02d}.json")])

    # Finetune kwargs mapped to finetune.train signature (inject batch_size if missing)
    mapped = _filter_and_map_finetune_kwargs(finetune_train, cutoff_len, finetune_kwargs)

    # ===== Rounds =====
    for r in range(rounds):
        if (resume_round is not None) and (r != int(resume_round)):
            continue

        print(f"\n=== Active-learning round {r} | labelled={len(chosen_ids)} | pool={len(pool_ids)} ===")
        rd = os.path.join(output_dir, f"round_{r}"); os.makedirs(rd, exist_ok=True)

        # De-dup & persist current labelled
        labelled = _dedup_by_idx(labelled)
        labelled_path = os.path.join(rd, "labelled.json"); _save_json(labelled, labelled_path)

        # Trainer input file
        tmp_json = os.path.join(rd, "labelled_tmp.json"); _save_json(labelled, tmp_json)

        _wandb_log(wb, {"round":r, "labelled_size":int(len(labelled)), "pool_size":int(len(pool_ids)), "event":"train_start"})
        _run_finetune_subprocess(
            base_model=base_model,
            tmp_json=tmp_json,
            rd=rd,
            mapped=mapped,
            cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES","0"),
            tokenizer_path=_resolve_tokenizer_src(base_model, tokenizer_path),  # conditionally forwarded
        )
        _hard_cuda_cleanup()
        _wandb_log(wb, {"round":r, "event":"train_done", "round_output_dir":rd})

        # Stop if last or pool empty
        if r == rounds - 1 or len(pool_ids) == 0:
            _wandb_log(wb, {"round":r, "event":"al_finished"}); break

        # === Scoring on pool ===
        model = _load_base_for_scoring(base_model, score_on_cpu=score_on_cpu)
        try:
            model = _attach_adapter(model, rd)

            def _score_ids(ids: Iterable[int], stage1: bool=False) -> Tuple[List[int], List[float]]:
                ids = list(ids)
                if not ids: return [], []
                ds_tok = _select_view_by_ids(
                    tokenized_full_st1 if stage1 else tokenized_full,
                    ids,
                    id_to_pos_st1 if stage1 else id_to_pos
                )
                if uncertainty.lower() == "logppl":
                    scores = _per_sample_logppl(model, tokenizer, ds_tok, batch_size=scoring_batch_size)
                elif uncertainty.lower() == "loss":
                    scores = _batched_loss_proxy(model, tokenizer, ds_tok, batch_size=scoring_batch_size)
                else:
                    raise ValueError("uncertainty must be one of {'logppl','loss'}")
                return ids, scores

            k_target = min(math.ceil(total_n * acq_frac), len(pool_ids))

            if two_stage and len(pool_ids) > 0:
                shortlist_size = min(len(pool_ids), max(k_target, int(preselect_factor * k_target)))
                ids1, s1 = _score_ids(pool_ids, stage1=True)
                short_ids = _topk_ids(ids1, s1, shortlist_size)
                ids2, s2 = _score_ids(short_ids, stage1=False)
                acquired_ids = _topk_ids(ids2, s2, k_target)
                scores_for_round = s2
            else:
                ids2, s2 = _score_ids(pool_ids, stage1=False)
                acquired_ids = _topk_ids(ids2, s2, k_target)
                scores_for_round = s2
        finally:
            try: del model
            except Exception: pass
            _hard_cuda_cleanup()

        # Build acquired Dataset and update sets
        acq_set = set(acquired_ids)
        acq_idx = [i for i, _id in enumerate(full["_al_idx"]) if _id in acq_set]
        acquired = full.select(acq_idx)

        # Persist selection
        sel_path = os.path.join(rd, f"selection_acquire_0p{int(acq_frac*100):02d}.json")
        _save_json(acquired, sel_path)

        # Round metrics
        sp = np.asarray(scores_for_round, dtype=np.float32)
        _wandb_log(wb, {
            "round": r,
            "acquired_k": int(len(acquired_ids)),
            "pool_after": int(len(pool_ids) - len(acquired_ids)),
            "uncertainty_metric": uncertainty,
            "uncertainty_mean": float(sp.mean()) if sp.size else float("nan"),
            "uncertainty_std": float(sp.std()) if sp.size else float("nan"),
            "uncertainty_p50": float(np.percentile(sp, 50)) if sp.size else float("nan"),
            "uncertainty_p90": float(np.percentile(sp, 90)) if sp.size else float("nan"),
            "selection_path": sel_path,
            "labelled_path": labelled_path,
            "event": "acquire_done",
        })
        if wandb_log_artifacts:
            _wandb_artifact_add(wb, name=f"al_round_{r}", type_="al-round", files=[labelled_path, sel_path])

        # Update trackers/state
        chosen_ids.update(acquired_ids)
        pool_ids.difference_update(acquired_ids)
        labelled = concatenate_datasets([labelled, acquired])

    print(f"[OK] Active learning completed. Outputs in: {output_dir}")
    if _WANDB_AVAILABLE and wandb.run is not None:
        try: wandb.finish()
        except Exception: pass

if __name__ == "__main__":
    fire.Fire(active_learning)
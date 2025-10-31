# ========================= finetune.py =========================
# Fast LoRA finetune with SDPA/FA2, fused AdamW, TF32, presets, and flash_only.
# ===============================================================

# --- env must be set before importing torch ---
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # CHANGED: no max_split_size_mb
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from typing import List, Optional, Union
import fire
import torch
import transformers
from datasets import load_dataset

torch.cuda.empty_cache()

# Optional WeightWatcher (gated)
try:
    import weightwatcher as ww
except Exception:
    ww = None

# Local PEFT checkout support
sys.path.append(os.path.join(os.getcwd(), "peft/src/"))

from peft import (   # noqa: E402
    LoraConfig,
    PrefixTuningConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

try:
    from peft import BottleneckConfig
except ImportError:
    BottleneckConfig = None

# Helper for quantised LoRA prep – PEFT keeps renaming/moving it.
try:
    # PEFT ≤ 0.10
    from peft import prepare_model_for_int8_training as prepare_model_for_int8_training
except ImportError:
    try:
        # PEFT ≥ 0.11
        from peft.tuners.lora import (
            prepare_model_for_kbit_training as prepare_model_for_int8_training
        )
    except ImportError:
        try:
            # PEFT dev branch
            from peft.utils.other import (
                prepare_model_for_kbit_training as prepare_model_for_int8_training
            )
        except ImportError:
            def prepare_model_for_int8_training(model, *args, **kwargs):  # type: ignore
                return model

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer


# ---------- Attention backends & TF32 ----------
# NEW: enable TF32 fast path on A100/90 etc.
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

from contextlib import contextmanager

@contextmanager
def attention_ctx(flash_only: bool):
    """
    NEW: Context that forces PyTorch SDPA Flash if `flash_only=True`,
    otherwise prefers FA2 and falls back gracefully.
    """
    try:
        if flash_only:
            # Force SDPA flash kernel, disable math/mem_efficient
            torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
        else:
            # Prefer flash; allow mem_efficient; disable math
            torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
        yield
    finally:
        # keep same settings on exit to avoid surprises
        if flash_only:
            torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
        else:
            torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)


# ---------- Helpers ----------
def _present(names, model):
    mods = set(dict(model.named_modules()).keys())
    return [n for n in names if any(k.endswith(n) or (("." + n) in k) for k in mods)]


def default_lora_targets(model, explicit=None):
    if explicit:
        return _present(explicit, model)

    mt = (getattr(model.config, "model_type", "") or "").lower()
    families = {
        "llama":   ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        "mistral": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        "mixtral": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        "gemma":   ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        "qwen":    ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        "qwen3":   ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        "opt":     ["q_proj","k_proj","v_proj","out_proj","fc1","fc2"],
        "gpt_neox":["query_key_value","dense","dense_h_to_4h","dense_4h_to_h"],
        "falcon":  ["query_key_value","dense","dense_h_to_4h","dense_4h_to_h"],
    }
    guess = families.get(mt, families["llama"])
    chosen = _present(guess, model)
    if chosen:
        return chosen

    heuristic = [
        "q_proj","k_proj","v_proj","o_proj","out_proj",
        "up_proj","down_proj","gate_proj",
        "proj","fc1","fc2","Wqkv","query_key_value","dense","dense_4h_to_h","dense_h_to_4h"
    ]
    chosen = _present(heuristic, model)
    if chosen:
        return chosen

    target_names = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and (("attn" in name) or ("mlp" in name) or ("proj" in name)):
            leaf = name.split(".")[-1]
            target_names.append(leaf)
    return sorted(set(target_names)) or ["q_proj","v_proj","o_proj"]


# NEW: align LoRA dtype to base model to avoid hidden casts
def _align_lora_dtype(model: torch.nn.Module):
    model_dtype = next(p for p in model.parameters() if p.requires_grad).dtype
    for _, m in model.named_modules():
        if hasattr(m, "lora_A") and hasattr(m.lora_A, "default"):
            m.lora_A.default.weight.data = m.lora_A.default.weight.data.to(model_dtype)
        if hasattr(m, "lora_B") and hasattr(m.lora_B, "default"):
            m.lora_B.default.weight.data = m.lora_B.default.weight.data.to(model_dtype)


# ======================= TRAIN =======================
def train(
    # model/data params
    base_model: str = "",
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    adapter_name: str = "lora",
    load_8bit: bool = False,

    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    use_gradient_checkpointing: bool = False,
    eval_step: int = 200,
    save_step: int = 200,

    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = None,

    # bottleneck adapter hyperparams
    bottleneck_size: int = 256,
    non_linearity: str = "tanh",
    adapter_dropout: float = 0.0,
    use_parallel_adapter: bool = False,
    use_adapterp: bool = False,
    target_modules: List[str] = None,
    scaling: Union[float, str] = 1.0,

    # prefix tuning hyperparams
    num_virtual_tokens: int = 30,

    # llm hyperparams
    train_on_inputs: bool = True,
    group_by_length: bool = False,

    # wandb
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",
    wandb_log_model: str = "",
    resume_from_checkpoint: str = None,
    enable_weightwatcher: bool = False,

    # NEW: mem/speed knobs
    restrict_lora_to_attention: bool = True,
    speed_preset: str = "fast",          # NEW: 'safe' | 'fast' | 'max'
    log_every: int = 50,                  # NEW
    flash_only: bool = False,             # NEW: force SDPA-Flash even if FA2 is unavailable
):
    print(
        f"Finetuning model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"bottleneck_size: {bottleneck_size}\n"
        f"non_linearity: {non_linearity}\n"
        f"adapter_dropout: {adapter_dropout}\n"
        f"use_parallel_adapter: {use_parallel_adapter}\n"
        f"use_adapterp: {use_adapterp}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"scaling: {scaling}\n"
        f"adapter_name: {adapter_name}\n"
        f"target_modules: {target_modules}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
        f"restrict_lora_to_attention: {restrict_lora_to_attention}\n"
        f"speed_preset: {speed_preset}\n"
        f"log_every: {log_every}\n"
        f"flash_only: {flash_only}\n"
    )
    assert base_model, "Please specify --base_model"

    gradient_accumulation_steps = max(1, batch_size // micro_batch_size)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = max(1, gradient_accumulation_steps // world_size)

    # wandb env
    use_wandb = len(wandb_project) > 0 or ("WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0)
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    # Prefer BF16 on A100; safe init order
    use_bf16 = torch.cuda.is_available()
    try:
        use_bf16 = use_bf16 and torch.cuda.is_bf16_supported()
    except Exception:
        pass

    # ---- Speed preset knobs (NEW) ----
    preset = (speed_preset or "fast").lower()
    if preset not in {"safe", "fast", "max"}:
        preset = "fast"
    if preset == "safe":
        use_gc = True
        per_device_bs = max(4, micro_batch_size)
        group_by = False
        optim_name = "adamw_torch"
        compile_model = False
        eval_every = max(1000, eval_step)
        save_every = max(2000, save_step)
    elif preset == "fast":
        use_gc = False
        per_device_bs = max(2, micro_batch_size)
        group_by = True
        optim_name = "adamw_torch_fused"
        compile_model = False
        eval_every = max(1500, eval_step)
        save_every = max(3000, save_step)
    else:  # max
        use_gc = False
        per_device_bs = max(4, micro_batch_size)
        group_by = True
        optim_name = "adamw_torch_fused"
        compile_model = True
        eval_every = 999999
        save_every = 999999

    # ---- Load model; prefer FA2 unless flash_only forces SDPA ----
    attn_impl = "sdpa" if flash_only else "flash_attention_2"  # NEW
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
        trust_remote_code=True,
        attn_implementation=attn_impl,  # NEW
    )

    # Grad checkpointing policy (NEW)
    if use_gradient_checkpointing and not use_gc:
        print("[speed] Overriding: turning OFF gradient checkpointing for speed.")
    use_gradient_checkpointing = use_gc
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    else:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()

    # ---- Tokenizer ----
    if getattr(model.config, "model_type", "") == "llama":
        try:
            tokenizer = LlamaTokenizer.from_pretrained(base_model, legacy=True)
        except Exception as err:
            print(f"[warn] LlamaTokenizer failed: {err}\nFalling back to AutoTokenizer.")
            tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)

    # Optional WW
    if enable_weightwatcher and ww is not None:
        watcher = ww.WeightWatcher(model=model)
        details = watcher.analyze()
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model {base_model} has {total_params:,} params; trainable={trainable_params:,} ({trainable_params/total_params:.2%})")
        print(details)

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(prompt, truncation=True, max_length=cutoff_len, padding=False, return_tensors=None)
        if (result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < cutoff_len and add_eos_token):
            result["input_ids"].append(tokenizer.eos_token_id)
            if "chatglm" not in base_model:
                result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        if "chatglm" in base_model:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        return tokenized_full_prompt

    # Prepare int8/kbit path if requested
    model = prepare_model_for_int8_training(model, use_gradient_checkpointing=use_gradient_checkpointing)

    # LoRA targets
    chosen_targets = (target_modules or lora_target_modules)
    if adapter_name in ("lora", "bottleneck"):
        if restrict_lora_to_attention:
            chosen_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]  # CHANGED: smaller, faster
        else:
            chosen_targets = default_lora_targets(model, explicit=chosen_targets)
        if not chosen_targets:
            raise RuntimeError("Could not infer target_modules; please pass --target_modules")
        print(f"[peft] Using target_modules: {chosen_targets}")

    if adapter_name == "lora":
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=chosen_targets,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif adapter_name == "bottleneck":
        if BottleneckConfig is None:
            raise RuntimeError("BottleneckConfig not available in this PEFT version.")
        config = BottleneckConfig(
            bottleneck_size=bottleneck_size,
            non_linearity=non_linearity,
            adapter_dropout=adapter_dropout,
            use_parallel_adapter=use_parallel_adapter,
            use_adapterp=use_adapterp,
            target_modules=chosen_targets,
            scaling=scaling,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif adapter_name == "prefix-tuning":
        config = PrefixTuningConfig(num_virtual_tokens=num_virtual_tokens, task_type="CAUSAL_LM")
    else:
        raise ValueError(f"Unknown adapter_name: {adapter_name}")

    model = get_peft_model(model, config)
    _align_lora_dtype(model)  # CHANGED: avoid PEFT hidden casts

    if adapter_name == "prefix-tuning":
        model.to('cuda')

    # Data
    if data_path.endswith(".json"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if resume_from_checkpoint:
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(resume_from_checkpoint, "adapter_model.bin")
            resume_from_checkpoint = False
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name, map_location="cpu")
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()

    if val_set_size > 0:
        train_val = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    # ---- TrainingArguments (NEW fused/TF32/preset knobs) ----
    _args_dict = dict(
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=max(1, batch_size // per_device_bs // max(1, world_size)),
        warmup_steps=50,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=False if use_bf16 else True,
        bf16=True if use_bf16 else False,
        logging_steps=log_every,
        optim=optim_name,  # NEW: fused on A100 when available
        save_strategy="steps",
        save_steps=save_every,
        output_dir=output_dir,
        save_total_limit=2,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by,  # NEW
        report_to="wandb" if use_wandb else None,
        run_name=wandb_run_name if use_wandb else None,
        dataloader_num_workers=4,   # NEW
        dataloader_pin_memory=True, # NEW
    )

    _sig = __import__("inspect").signature(transformers.TrainingArguments.__init__)
    if "evaluation_strategy" in _sig.parameters and val_set_size > 0:
        _args_dict["evaluation_strategy"] = "steps"
        _args_dict["eval_steps"] = eval_every
        _args_dict["load_best_model_at_end"] = False  # faster

    if "ddp_find_unused_parameters" not in _sig.parameters:
        _args_dict.pop("ddp_find_unused_parameters", None)

    training_args = transformers.TrainingArguments(**_args_dict)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    # Optional compile in 'max' preset
    if compile_model and torch.__version__ >= "2" and sys.platform != "win32":
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
            print("[speed] torch.compile enabled (reduce-overhead).")
        except Exception as e:
            print(f"[speed] torch.compile disabled ({e}).")

    # Save only adapter weights
    old_state_dict = model.state_dict
    model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(model, type(model))

    # Train with attention context
    with attention_ctx(flash_only=flash_only):  # NEW
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
    print("\nDone. If there's a warning about missing keys above, please disregard :)")

# ======================= PROMPT BUILDER =======================
def generate_prompt(data_point):
    """
    Build a textual prompt from a single dataset record.
    Supports:
      1) Alpaca-style:  {"instruction": str, "input": str or "", "output": str}
      2) BoolQ-style:   {"question": str, "passage": str, "answer": bool}
    """
    instruction = data_point.get("instruction", "Answer the question based on the passage.")
    user_input = data_point.get("input")

    if user_input is None and "question" in data_point:
        user_input = data_point["question"]
        if "passage" in data_point:
            user_input = f"Passage:\n{data_point['passage']}\n\nQuestion:\n{user_input}"

    output = data_point.get("output")
    if output is None and "answer" in data_point:
        output = "yes" if data_point["answer"] else "no"

    if user_input:
        return (
            "Below is an instruction that describes a task, paired with an "
            "input that provides further context. Write a response that "
            "appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{user_input}\n\n"
            "### Response:\n"
            f"{output}"
        )
    else:
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Response:\n"
            f"{output}"
        )

# ======================= ENTRY =======================
if __name__ == "__main__":
    fire.Fire(train)

# export_llava_language_model_gpu_v2.py
import os, json, shutil, time, torch
from transformers import AutoModelForVision2Seq, AutoModelForCausalLM, AutoConfig, AutoTokenizer

SRC = "/home/models/Llama-3.2-11B-Vision-Instruct"
DST = "/home/models/Llama-3.2-11B-Text-from-Vision"

def log(msg):
    print(time.strftime("[%H:%M:%S]"), msg, flush=True)

def pick_dtype():
    if torch.cuda.is_available():
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32

def find_text_tower(mm):
    """
    Try common attributes. If those fail, search by presence of lm_head & embed_tokens.
    Returns the text module and the attribute path (string for logging).
    """
    # 1) common wrappers
    for name in ("language_model", "text_model"):
        if hasattr(mm, name):
            return getattr(mm, name), name

    # 2) Some wrappers hang the causal LM under .model
    #    But be careful: some mm.model is still a wrapper. Check for causal LM bits.
    if hasattr(mm, "model"):
        m = getattr(mm, "model")
        # quick heuristic: must have .lm_head and a child named ".model" with layers
        has_lm_head = hasattr(m, "lm_head")
        has_inner_model = hasattr(m, "model")
        if has_lm_head and has_inner_model:
            return m, "model"

    # 3) Fallback: scan child modules for something that *looks* like LlamaForCausalLM
    for n, mod in mm.named_modules():
        if hasattr(mod, "lm_head") and hasattr(mod, "model"):
            # looks like LlamaForCausalLM: has .model (LlamaModel) + .lm_head
            return mod, n or "<root>"

    raise RuntimeError("Could not find the text tower (LlamaForCausalLM-like) inside the composite model.")

def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.makedirs(DST, exist_ok=True)
    torch.set_grad_enabled(False)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = pick_dtype()
    log(f"Using device={device}, dtype={dtype}")

    # Tokenizer
    log("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)
    tok.save_pretrained(DST)

    # Read config to choose the right Auto class (some repos register as Vision2Seq)
    cfg = AutoConfig.from_pretrained(SRC, trust_remote_code=True)
    model_type = getattr(cfg, "model_type", "").lower()
    log(f"Detected model_type={model_type}")

    # Let HF/accelerate place things; don't invent device_map keys
    # Try Vision2Seq first, then fall back to CausalLM (some repos override that entry-point).
    log("Loading composite model with device_map='auto'...")
    mm = None
    tried = []
    try:
        mm = AutoModelForVision2Seq.from_pretrained(
            SRC, trust_remote_code=True, torch_dtype=dtype,
            device_map="auto", low_cpu_mem_usage=True, offload_folder=os.path.join(DST, "_offload")
        )
        tried.append("AutoModelForVision2Seq")
    except Exception as e1:
        log(f"AutoModelForVision2Seq failed: {repr(e1)}")
        try:
            mm = AutoModelForCausalLM.from_pretrained(
                SRC, trust_remote_code=True, torch_dtype=dtype,
                device_map="auto", low_cpu_mem_usage=True, offload_folder=os.path.join(DST, "_offload")
            )
            tried.append("AutoModelForCausalLM")
        except Exception as e2:
            raise RuntimeError(f"Failed to load with Vision2Seq and CausalLM. "
                               f"Errors: {repr(e1)} | {repr(e2)}")

    mm.eval()
    log(f"Loaded via: {tried[-1]}")

    # Locate the text tower
    text, where = find_text_tower(mm)
    log(f"Found text tower at: .{where}")

    # Put text on GPU for faster serialization
    try:
        text.to(device)
    except Exception as e:
        log(f"Warning: moving text tower to {device} failed ({e}); continuing on current device")

    # Save standalone Llama (writes its own correct config.json)
    log("Saving text tower as a plain Llama checkpoint...")
    text.save_pretrained(DST, safe_serialization=True)

    # Ensure model_type is 'llama' in the saved config (some classes already do this)
    cfg_path = os.path.join(DST, "config.json")
    try:
        with open(cfg_path, "r") as f:
            saved_cfg = json.load(f)
        if str(saved_cfg.get("model_type", "")).lower() != "llama":
            saved_cfg["model_type"] = "llama"
            with open(cfg_path, "w") as f:
                json.dump(saved_cfg, f, indent=2)
            log("Forced model_type='llama' in saved config.json")
    except Exception as e:
        log(f"Note: couldn't adjust saved config.json ({e}); likely fine.")

    # Copy common extras
    for fn in ("special_tokens_map.json", "tokenizer_config.json",
               "generation_config.json", "chat_template.json"):
        sp = os.path.join(SRC, fn)
        if os.path.exists(sp):
            shutil.copy2(sp, os.path.join(DST, fn))

    log(f"Done. Saved plain text model to: {DST}")
    log("Sanity tip: first shard should contain keys like 'model.embed_tokens.weight' (no 'language_model.' prefix).")

if __name__ == "__main__":
    main()
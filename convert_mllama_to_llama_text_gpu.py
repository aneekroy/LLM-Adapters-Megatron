#!/usr/bin/env python
import os, json, glob, argparse, torch
from safetensors.torch import load_file as st_load
from transformers import AutoConfig, AutoTokenizer, LlamaConfig, LlamaForCausalLM

ALLOWED_ROPE_TYPES = {None, "linear", "dynamic", "yarn"}

SKIP_SUBSTRINGS = (
    "vision_", "multi_modal_projector", "visual_", "image_",
    ".cross_attn", "_cross_attn", "cross_attn_",
)

PREFIXES = (
    "model.language_model.",  # common mllama layout
    "language_model.",        # some repos
)

def sanitize_rope(cfg: dict):
    rs = cfg.get("rope_scaling")
    if isinstance(rs, dict):
        if "type" not in rs and "rope_type" in rs:
            rs["type"] = rs.pop("rope_type")
        if rs.get("type") not in ALLOWED_ROPE_TYPES:
            cfg.pop("rope_scaling", None)

def list_shards(src: str):
    idx = os.path.join(src, "model.safetensors.index.json")
    if os.path.exists(idx):
        j = json.load(open(idx, "r"))
        files = sorted(set(j["weight_map"].values()))
        return [os.path.join(src, f) for f in files]
    cand = sorted(glob.glob(os.path.join(src, "model-*.safetensors")))
    if cand:
        return cand
    single = os.path.join(src, "model.safetensors")
    if os.path.exists(single):
        return [single]
    raise FileNotFoundError(f"No .safetensors shards found at {src}")

def wants_key(k: str) -> bool:
    for bad in SKIP_SUBSTRINGS:
        if bad in k:
            return False
    return k.startswith("model.") or k == "lm_head.weight"

def normalize_key(k: str) -> str:
    # Strip known prefixes if present; otherwise return original key.
    for p in PREFIXES:
        if k.startswith(p):
            return k[len(p):]
    return k

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--dtype", default="float16",
                    choices=["float16","bfloat16","float32"])
    ap.add_argument("--device", default="cuda:0", help="cuda:N (set visible devices externally)")
    args = ap.parse_args()

    # ---- GPU setup (GPU-only) ----
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; this script is GPU-only.")
    if not args.device.startswith("cuda:"):
        raise ValueError("Use --device cuda:N")
    idx = int(args.device.split(":")[1])
    if idx >= torch.cuda.device_count():
        raise RuntimeError(f"Requested {args.device}, but only {torch.cuda.device_count()} CUDA device(s) visible.")
    torch.cuda.set_device(idx)
    device = torch.device(args.device)
    ST_DEVICE = "cuda"  # safetensors loads to current CUDA

    os.makedirs(args.dst, exist_ok=True)

    # ---- Build a clean text-only config ----
    cfg = AutoConfig.from_pretrained(args.src, trust_remote_code=True)
    base = cfg.text_config.to_dict() if getattr(cfg, "text_config", None) else cfg.to_dict()

    # Drop multimodal fields
    for k in list(base.keys()):
        if k.startswith("vision_") or k.startswith("mm_") or k in {
            "vision_config","vision_tower","image_token_id","image_eos_token_id",
            "image_start_token_id","image_end_token_id"
        }:
            base.pop(k, None)

    sanitize_rope(base)
    base["architectures"] = ["LlamaForCausalLM"]
    base["model_type"] = "llama"

    # ---- PASS 0: scan shards to find vocab sizes for embed/head ----
    shards = list_shards(args.src)
    vocab_embed = None
    vocab_head = None
    for shard in shards:
        blob = st_load(shard, device=ST_DEVICE)
        for k, v in blob.items():
            nk = normalize_key(k)
            if nk.endswith("model.embed_tokens.weight"):
                vocab_embed = v.shape[0]
            elif nk == "lm_head.weight":
                vocab_head = v.shape[0]
        # free shard tensors promptly
        del blob
        torch.cuda.empty_cache()

    if vocab_embed is None and vocab_head is None:
        raise RuntimeError("Could not find embed or head tensors in shards. Are you sure this is an mllama-style ckpt?")

    target_vocab = max([x for x in [vocab_embed, vocab_head, base.get("vocab_size")] if x is not None])

    # Build model on GPU with target vocab so we never need to expand during load
    base["vocab_size"] = target_vocab
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    model = LlamaForCausalLM(LlamaConfig(**base)).to(device=device, dtype=torch_dtype)
    model.eval()

    # ---- PASS 1: load all NON-embed/head weights via load_state_dict ----
    with torch.inference_mode():
        for shard in shards:
            blob = st_load(shard, device=ST_DEVICE)
            partial = {}
            for k, v in blob.items():
                nk = normalize_key(k)
                if not wants_key(nk):
                    continue
                if nk in ("model.embed_tokens.weight", "lm_head.weight"):
                    continue  # handled manually later
                partial[nk] = v
            if partial:
                model.load_state_dict(partial, strict=False)
            del blob, partial
            torch.cuda.empty_cache()

    # ---- PASS 2: manually place embed/head with safe slicing on-GPU ----
    with torch.inference_mode():
        # embed_tokens
        placed_embed = False
        for shard in shards:
            blob = st_load(shard, device=ST_DEVICE)
            for k, v in blob.items():
                nk = normalize_key(k)
                if nk == "model.embed_tokens.weight":
                    dst = model.get_input_embeddings().weight  # [Vt, H]
                    vt_src, vt_dst = v.shape[0], dst.shape[0]
                    rows = min(vt_src, vt_dst)
                    dst[:rows].copy_(v[:rows])
                    placed_embed = True
                    break
            del blob
            torch.cuda.empty_cache()
            if placed_embed:
                break
        if not placed_embed and vocab_embed is not None:
            raise RuntimeError("Embed tokens weight not found across shards.")

        # lm_head
        placed_head = False
        for shard in shards:
            blob = st_load(shard, device=ST_DEVICE)
            for k, v in blob.items():
                nk = normalize_key(k)
                if nk == "lm_head.weight":
                    dst = model.lm_head.weight  # [Vt, H]
                    vt_src, vt_dst = v.shape[0], dst.shape[0]
                    rows = min(vt_src, vt_dst)
                    dst[:rows].copy_(v[:rows])
                    placed_head = True
                    break
            del blob
            torch.cuda.empty_cache()
            if placed_head:
                break
        if not placed_head and vocab_head is not None:
            raise RuntimeError("LM head weight not found across shards.")

    # ---- Save tokenizer & model ----
    tok = AutoTokenizer.from_pretrained(args.src, use_fast=True, trust_remote_code=True)
    tok.save_pretrained(args.dst)

    model.save_pretrained(args.dst)
    cleaned = model.config.to_dict()
    sanitize_rope(cleaned)
    with open(os.path.join(args.dst, "config.json"), "w") as f:
        json.dump(cleaned, f, indent=2, sort_keys=True)

    print(f"[convert][GPU] OK. Text-only model saved to: {args.dst}")
    print(f"[convert] vocab_embed={vocab_embed}, vocab_head={vocab_head}, target_vocab={target_vocab}")

if __name__ == "__main__":
    main()
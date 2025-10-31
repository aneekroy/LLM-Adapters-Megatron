

import copy, json, os, re, sys, argparse, datetime
import torch, wandb
from tqdm import tqdm

# vLLM imports
import vllm.utils
vllm.utils.MM_PLUGIN_ENABLED = False
from vllm import LLM, SamplingParams

DEVICE_MSG = "vLLM needs CUDA GPUs. MPS/CPU aren’t supported for this script."

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(
    load_8bit: bool = False,     # kept for CLI compatibility; unused with vLLM
    base_model: str = "",
    lora_weights: str = "",
    share_gradio: bool = False,  # kept for CLI compatibility; unused
):
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(DEVICE_MSG)

    # ── Weights & Biases ─────────────────────────────────────────────────────
    run_name = f"{args.model}-{args.adapter}-{args.dataset}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project="Llama-3.2-11B--Vision-Instruct_baseline",  # your original project name
        name=run_name,
        config=vars(args),
        reinit=True
    )

    save_file = f'experiment/{args.model}-{args.adapter}-{args.dataset}.json'
    create_dir('experiment/')

    dataset = load_data(args)
    llm, tokenizer = load_model_vllm(args)

    # ── Generation helpers ───────────────────────────────────────────────────
    def evaluate_one(instruction,
                     temperature=0.1, top_p=0.75, top_k=40,
                     num_beams=4, max_new_tokens=256, **kwargs):
        prompt = generate_prompt(instruction, None)
        sp = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
        )
        outs = llm.generate([prompt], sp)
        text = outs[0].outputs[0].text
        parts = text.split("### Response:")
        return parts[1].strip() if len(parts) > 1 else text.strip()

    def evaluate_batch(inst_list,
                       temperature=0.1, top_p=0.75, top_k=40,
                       num_beams=4, max_new_tokens=256):
        prompts = [generate_prompt(ins, None) for ins in inst_list]
        sp = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
        )
        outs = llm.generate(prompts, sp)
        texts = []
        for o in outs:
            t = o.outputs[0].text
            parts = t.split("### Response:")
            texts.append(parts[1].strip() if len(parts) > 1 else t.strip())
        return texts

    # ── Eval loop ────────────────────────────────────────────────────────────
    total = len(dataset)
    correct = 0
    miss = 0.001
    output_data = []
    pbar = tqdm(total=total)

    bsz = max(1, args.batch_size_gen)
    idx = 0
    while idx < total:
        chunk = dataset[idx: idx + bsz]
        instructions = [ex.get("instruction") for ex in chunk]
        outputs = [evaluate_one(instructions[0])] if bsz == 1 else evaluate_batch(instructions)

        for j, data in enumerate(chunk):
            outputs_j = outputs[j]
            label = data.get('answer')
            flag = False
            if args.dataset.lower() in ['aqua']:
                predict = extract_answer_letter(args, outputs_j)
                if label == predict:
                    correct += 1
                    flag = True
            else:
                if isinstance(label, str):
                    try:
                        label = float(label)
                    except Exception:
                        label = float('inf')
                predict = extract_answer_number(args, outputs_j)
                if abs(label - predict) <= miss:
                    correct += 1
                    flag = True

            new_data = copy.deepcopy(data)
            new_data['output_pred'] = outputs_j
            new_data['pred'] = predict
            new_data['flag'] = flag
            output_data.append(new_data)

            cur = idx + j + 1
            print('\n---------------')
            print(outputs_j)
            print('prediction:', predict)
            print('label:', label)
            print('---------------')
            print(f'\rtest:{cur}/{total} | accuracy {correct}  {correct / (cur)}')
            wandb.log({
                "step": cur,
                "running_accuracy": correct / cur,
                "prediction": predict,
                "label": label,
                "flag_correct": flag
            }, step=cur)
            with open(save_file, 'w+') as f:
                json.dump(output_data, f, indent=4)
            pbar.update(1)

        idx += bsz

    pbar.close()
    final_acc = correct / total
    wandb.log({"final_accuracy": final_acc})
    print('\nTest finished')
    wandb.finish()

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

### Instruction:
{instruction}

### Response:
"""

# ─────────────────────────────────────────────────────────────────────────────
# Patches for compatibility
# ─────────────────────────────────────────────────────────────────────────────
# ── drop-in replacement for your config patch & loader ───────────────────────

def _backup_once(path: str):
    bak = path + ".backup"
    if os.path.exists(path) and not os.path.exists(bak):
        try:
            with open(path, "rb") as fsrc, open(bak, "wb") as fdst:
                fdst.write(fsrc.read())
            print(f"[patch] Backed up {path} -> {bak}")
        except Exception as e:
            print(f"[warn] Backup failed for {path}: {e}")

def _is_vllm_rope_supported(rs):
    if not isinstance(rs, dict): return True
    t = rs.get("type", None)
    # Conservative allow-list for older vLLM builds:
    return t in (None, "linear", "dynamic", "yarn")

def force_pure_text_config(model_path: str):
    """
    Convert an Mllama (vision) config into a pure LLaMA text config for vLLM:
    - Flattens `text_config` to top-level
    - Sets architectures/model_type for text
    - Removes all vision/mm fields
    - Drops rope_scaling if type is not supported by vLLM (e.g., 'su')
    """
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.exists(cfg_path):
        print(f"[patch] No config.json at {cfg_path}")
        return

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    # If nested structure exists, take text_config as the base
    base = None
    if isinstance(cfg.get("text_config"), dict):
        base = cfg["text_config"].copy()
    else:
        # Some repos already expose a flat text config; use it as-is
        base = cfg.copy()

    # Remove/ignore any mm/vision bits (top-level and in base)
    def _purge_mm(d):
        ks = [k for k in list(d.keys())
              if k.startswith("vision_") or k.startswith("mm_") or
                 k in {"vision_config","vision_tower","image_token_id",
                       "image_eos_token_id","image_start_token_id",
                       "image_end_token_id"}]
        for k in ks:
            d.pop(k, None)

    _purge_mm(cfg)
    _purge_mm(base)

    # vLLM-friendly text identifiers
    base["architectures"] = ["LlamaForCausalLM"]
    base["model_type"] = "llama"

    # Bring over useful tokenizer/special token fields if they were only on the outer cfg
    for k in ["bos_token_id","eos_token_id","pad_token_id","tokenizer_class",
              "tie_word_embeddings","vocab_size"]:
        if k not in base and k in cfg:
            base[k] = cfg[k]

    # Handle rope_scaling compatibility
    rs = base.get("rope_scaling")
    if isinstance(rs, dict) and "type" not in rs and "rope_type" in rs:
        rs["type"] = rs.pop("rope_type")  # normalize

    if not _is_vllm_rope_supported(base.get("rope_scaling")):
        print(f"[patch] Removing unsupported rope_scaling: {base.get('rope_scaling')}")
        base.pop("rope_scaling", None)

    # Write back flattened, pure-text config
    _backup_once(cfg_path)
    with open(cfg_path, "w") as f:
        json.dump(base, f, indent=2, sort_keys=True)
    print("[patch] Wrote flattened pure-text LLaMA config.json")

    # Clean tokenizer_config.json of image hooks (harmless if absent)
    tcfg_path = os.path.join(model_path, "tokenizer_config.json")
    if os.path.exists(tcfg_path):
        try:
            with open(tcfg_path, "r") as f:
                tcfg = json.load(f)
            changed = False
            for k in ["image_token_id","image_eos_token_id","image_start_token_id","image_end_token_id"]:
                if k in tcfg:
                    tcfg.pop(k, None); changed = True
            if changed:
                _backup_once(tcfg_path)
                with open(tcfg_path, "w") as f:
                    json.dump(tcfg, f, indent=2, sort_keys=True)
                print("[patch] Cleaned tokenizer_config.json of image_* ids")
        except Exception as e:
            print(f"[warn] tokenizer_config patch skipped: {e}")





def _normalize_rope_scaling_inplace(model_path: str):
    """
    Convert {'rope_type': '...'} → {'type': '...'} in rope_scaling if needed.
    """
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.exists(cfg_path):
        return
    try:
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        rs = cfg.get("rope_scaling")
        if isinstance(rs, dict) and "type" not in rs and "rope_type" in rs:
            rs["type"] = rs.pop("rope_type")
            _backup_once(cfg_path)
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)
            print("[patch] Updated rope_scaling in config.json: added 'type' key")
    except Exception as e:
        print(f"[warn] rope_scaling patch skipped: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Data & Args
# ─────────────────────────────────────────────────────────────────────────────
def load_data(args) -> list:
    file_path = f'dataset/{args.dataset}/test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    return json.load(open(file_path, 'r'))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['AddSub','MultiArith','SingleEq','gsm8k','AQuA','SVAMP'], required=True)

    # Include your local models explicitly (plus others you had)
    parser.add_argument('--model', choices=[
        'Llama-3.2-11B-Vision-Instruct',
        'Llama-3.2-1B-Instruct',
        'Llama-3.2-3B','Llama-3.2-3B-Instruct',
        'Llama-3.2-3B-Instruct-0.33-Wanda',
        'Llama-3.2-3B-Instruct-0.5-Wanda-Structured_24',
        'Llama-3.2-3B-Instruct-0.5-Wanda-Structured_48',
        'Llama-3.2-3B-Instruct-Sparse-0.33',
        'LLaMA-7B','BLOOM-7B','GPT-j-6B',
        'Llama-3.1-8B','Llama-3.1-8B-Instruct',
        'Llama-3.3-70B-Instruct','Llama-3.3-70B-Instruct-Sparse',
        'Llama-4-Scout-17B-16E-Instruct','Llama-4-Scout-17B-16E-Instruct-Sparse',
        'Llama-3.1-8B-Instruct-0.33-Wanda','Llama-3.1-8B-Instruct-0.73-Wanda',
        'Llama-3.2-3B-Instruct-0.33-Wanda',  # (duplicate allowed; harmless)
    ], required=True)

    parser.add_argument('--adapter', choices=['None','LoRA','AdapterP','AdapterH','Parallel','Prefix'], default='None')
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--lora_weights', default='')  # if you’ll pre-merge, point to merged dir instead
    parser.add_argument('--load_8bit', action='store_true', default=False)  # unused with vLLM
    parser.add_argument('--batch_size_gen', type=int, default=1, help='vLLM batch size for generation')
    parser.add_argument('--tp_size', type=int, default=1, help='Tensor parallel size for vLLM (multi-GPU)')
    parser.add_argument('--gpu_mem_util', type=float, default=0.9, help='vLLM GPU memory utilization [0-1]')
    parser.add_argument('--max_model_len', type=int, default=None, help='Override max model len for vLLM')
    parser.add_argument('--dtype', choices=['auto','float16','bfloat16'], default='auto', help='KV/weights dtype for vLLM')
    return parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Model Loader (vLLM)
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_dtype(arg_dtype: str) -> str:
    if arg_dtype == 'auto':
        return 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
    return arg_dtype

def load_model_vllm(args):
    model_path = args.base_model
    force_pure_text_config(model_path)             # <-- NEW: robust flattener
    # keep your existing rope normalizer if you want, but not necessary now
    # _normalize_rope_scaling_inplace(model_path)

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=max(1, args.tp_size),
        gpu_memory_utilization=args.gpu_mem_util,
        dtype=_resolve_dtype(getattr(args, "dtype", "auto")),
        max_model_len=args.max_model_len,
        enforce_eager=True,
    )
    try:
        tokenizer = llm.get_tokenizer()
    except Exception:
        tokenizer = None
    return llm, tokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Answer extractors
# ─────────────────────────────────────────────────────────────────────────────
def extract_answer_number(args, sentence: str) -> float:
    dataset = args.dataset.lower()
    if dataset in ["multiarith", "addsub", "singleeq", "gsm8k", "svamp"]:
        sentence = sentence.replace(',', '')
        pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
        if not pred:
            return float('inf')
        pred_answer = float(pred[-1])
    else:
        raise NotImplementedError(f'not support dataset: {dataset}')
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except Exception:
            pred_answer = float('inf')
    return pred_answer

def extract_answer_letter(args, sentence: str) -> str:
    sentence_ = sentence.strip()
    pred_answers = re.findall(r'A|B|C|D|E', sentence_)
    if pred_answers:
        return pred_answers[0]
    return ''

# ─────────────────────────────────────────────────────────────────────────────
# Optional: LoRA merge (imports inside function to avoid hard deps)
# ─────────────────────────────────────────────────────────────────────────────
def merge_lora(base_model: str, lora_weights: str, out_dir: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    os.makedirs(out_dir, exist_ok=True)
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="cpu")
    tok = AutoTokenizer.from_pretrained(base_model)
    peft_model = PeftModel.from_pretrained(base, lora_weights, torch_dtype=torch.float16, device_map="cpu")
    peft_model = peft_model.merge_and_unload()
    peft_model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[merge] Saved merged model to: {out_dir}")

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
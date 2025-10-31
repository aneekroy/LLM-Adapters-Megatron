import copy, json, os, re, sys, argparse, datetime
import torch, wandb
from tqdm import tqdm

from vllm.engine.arg_utils import EngineArgs
import vllm.utils
vllm.utils.MM_PLUGIN_ENABLED = False

# vLLM imports
from vllm import LLM, SamplingParams

from vllm.engine.arg_utils import EngineArgs




DEVICE_MSG = "vLLM needs CUDA GPUs."

def main(
    load_8bit: bool = False,     # kept for CLI compatibility; unused with vLLM
    base_model: str = "",
    lora_weights: str = "",
    share_gradio: bool = False,  # kept for CLI compatibility; unused
):
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(DEVICE_MSG)

    # ── Weights & Biases setup ────────────────────────────────────────────────
    run_name = f"{args.model}-{args.adapter}-{args.dataset}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project=f"{args.model}-evaluation",
        name=run_name,
        config=vars(args),
        reinit=True
    )

    save_file = f'experiment/{args.model}-{args.adapter}-{args.dataset}.json'
    create_dir('experiment/')

    dataset = load_data(args)
    llm, tokenizer = load_model_vllm(args)

    # Single-call generation helper (keeps your original flow)
    def evaluate_one(instruction,
                     temperature=0.1, top_p=0.75, top_k=40,
                     num_beams=4, max_new_tokens=256, **kwargs):
        prompt = generate_prompt(instruction, None)
        sp = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
            # use_beam_search=(num_beams and num_beams > 1),
            # beam_width=(num_beams if num_beams and num_beams > 1 else None),
        )
        outs = llm.generate([prompt], sp)
        text = outs[0].outputs[0].text
        # Preserve your splitter
        parts = text.split("### Response:")
        return parts[1].strip() if len(parts) > 1 else text.strip()

    # Optional: batched generation for speed (set --batch_size_gen > 1)
    def evaluate_batch(inst_list,
                       temperature=0.1, top_p=0.75, top_k=40,
                       num_beams=4, max_new_tokens=256):
        prompts = [generate_prompt(ins, None) for ins in inst_list]
        sp = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
            # use_beam_search=(num_beams and num_beams > 1),
            # beam_width=(num_beams if num_beams and num_beams > 1 else None),
        )
        outs = llm.generate(prompts, sp)
        texts = []
        for o in outs:
            t = o.outputs[0].text
            parts = t.split("### Response:")
            texts.append(parts[1].strip() if len(parts) > 1 else t.strip())
        return texts

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
        # switch seamlessly between single/batch
        if bsz == 1:
            outputs = [evaluate_one(instructions[0])]
        else:
            outputs = evaluate_batch(instructions)

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

# Normalize rope_scaling in config.json so older vLLM versions don't need a runtime kwarg
def _normalize_rope_scaling_inplace(model_path: str):
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.exists(cfg_path):
        return
    try:
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        rs = cfg.get("rope_scaling")
        if isinstance(rs, dict) and "type" not in rs and "rope_type" in rs:
            rs["type"] = rs.pop("rope_type")
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)
            print("[patch] Updated rope_scaling in config.json: added 'type' key")
    except Exception as e:
        print(f"[warn] rope_scaling patch skipped: {e}")

def load_data(args) -> list:
    file_path = f'dataset/{args.dataset}/test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    return json.load(open(file_path, 'r'))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['AddSub','MultiArith','SingleEq','gsm8k','AQuA','SVAMP'], required=True)
    parser.add_argument('--model', choices=[
        'LLaMA-7B','BLOOM-7B','GPT-j-6B',
        'Llama-3.2-1B','Llama-3.2-3B',
        'Llama-3.2-1B-Instruct','Llama-3.2-3B-Instruct','Llama-3.2-3B-Instruct-Sparse','Llama-4-Scout-17B-16E-Instruct','Llama-3.1-8B','Llama-3.1-8B-Instruct',
        'Llama-3.3-70B-Instruct','Llama-3.3-70B-Instruct-Sparse','Llama-3.2-11B-Vision-Instruct','Llama-4-Scout-17B-16E-Instruct-Sparse',
        'Llama-3.1-8B-Instruct-0.33-Wanda','Llama-3.1-8B-Instruct-0.73-Wanda','Llama-3.2-3B-Instruct-0.5-Wanda-Structured_24','Llama-3.2-3B-Instruct-0.5-Wanda-Structured_48',
        'Llama-3.2-3B-Instruct-0.33-Wanda','Llama-3.2-3B-Instruct-0.5-Wanda-Structured_24','Llama-3.2-3B-Instruct-0.5-Wanda-Structured_48','Qwen3-8B','Qwen3-8B-Sparse','Qwen3-4B-Instruct','Fin-o1-8B','Qwen3-4B-Sparse',
    ], required=True)

    parser.add_argument('--adapter', choices=['None','LoRA','AdapterP','AdapterH','Parallel','Prefix'], default='None')
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--lora_weights', default='')  # if you’ll pre-merge, point to merged dir instead
    parser.add_argument('--batch_size_gen', type=int, default=1, help='vLLM batch size for generation')
    parser.add_argument('--tp_size', type=int, default=1, help='Tensor parallel size for vLLM (multi-GPU)')
    parser.add_argument('--gpu_mem_util', type=float, default=0.9, help='vLLM GPU memory utilization [0-1]')
    parser.add_argument('--max_model_len', type=int, default=None, help='Override max model len for vLLM')
    parser.add_argument('--seed', type=int, default=1234,
                    help='Global engine seed for reproducible decoding')
    return parser.parse_args()



def load_model_vllm(args):
    model_path = args.base_model

    # Normalize the on-disk config so vLLM doesn't choke on rope_scaling
    _normalize_rope_scaling_inplace(model_path)

    from vllm.engine.arg_utils import EngineArgs



    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=max(1, args.tp_size),
        gpu_memory_utilization=args.gpu_mem_util,
        dtype="float16",
        max_model_len=args.max_model_len,
        enforce_eager=True,
        seed=int(args.seed),   # engine-level seed
    )
    try:
        tokenizer = llm.get_tokenizer()
    except Exception:
        tokenizer = None
    return llm, tokenizer



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

# Optional utility if you want to pre-merge LoRA into base_model here
def merge_lora(base_model: str, lora_weights: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="cpu")
    tok = AutoTokenizer.from_pretrained(base_model)
    peft_model = PeftModel.from_pretrained(base, lora_weights, torch_dtype=torch.float16, device_map="cpu")
    peft_model = peft_model.merge_and_unload()
    peft_model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[merge] Saved merged model to: {out_dir}")

if __name__ == "__main__":
    # fire.Fire(main)   # Fire is overkill now; stick to argparse
    main()
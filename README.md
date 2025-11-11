<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<h1 align="center"> 
<img src="picture.jpg" width="73" height="114">
<p> LLM-Adapters</p>
</h1>

<h3 align="center">
    <p>LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models </p>
</h3>
LLM-Adapters is an easy-to-use framework that integrates various adapters into LLMs and can execute adapter-based PEFT methods of LLMs for different tasks. LLM-Adapter is an extension of HuggingFace's PEFT library, many thanks for their amazing work! Please find our paper at this link: https://arxiv.org/abs/2304.01933.

The framework includes state-of-the-art open-access LLMs: LLaMa, OPT, BLOOM, and GPT-J, as well as widely used adapters such as Bottleneck adapters, Parallel adapters, and LoRA.

Supported Adapters:

1. LoRA: [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685.pdf)
2. AdapterH: [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/pdf/1902.00751.pdf)
3. AdapterP: [GMAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer](https://arxiv.org/pdf/2005.00052.pdf)
4. Parallel: [TOWARDS A UNIFIED VIEW OF PARAMETER-EFFICIENT TRANSFER LEARNING](https://arxiv.org/pdf/2110.04366.pdf)
5. Prefix Tuning: [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://aclanthology.org/2021.acl-long.353/), [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/pdf/2110.07602.pdf)
6. P-Tuning: [GPT Understands, Too](https://arxiv.org/pdf/2103.10385.pdf)
7. Prompt Tuning: [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/pdf/2104.08691.pdf) 

## Latest News 🔥🔥

* [2023-08-10] LLM-Adapters has been accepted by EMNLP 2023.
* [2023-07-16] we released commonsense170k dataset and the  The LLaMA-13B-Parallel model outformances ChatGPT on 8 commonsense benchmarks.
* [2023-04-21] We released math10k dataset and the [LLaMA-13B adapter checkpoints](https://drive.google.com/file/d/1NqUv-Hn_mAkGXsUOqpJKmPKW5Gp8mRlO/view?usp=sharing). The LLaMA-13B-Parallel model achieves **91%** of GPT-3.5 performance!
* [2023-04-10] We can support GPT-Neo and ChatGLM now!
* [2023-04-04] [Release code and dataset](https://github.com/AGI-Edgerunners/LLM-Adapters)

## Special Announcement
The `math_10k.json` data is collected with the training sets of GSM8K, MAWPS, and AQuA(1000 examples). However, MAWPS consists of AddSub, MultiArith, SingleOp, SingleEq, SimulEq-S, SimulEq-L. Thus, we can't utilize MultiArith, AddSub, and SingleEq as evaluation benchmarks with models trained with `math_10k.json`. We evaluate the PEFT methods on the MAWPS test set instead, and the result table has been updated (The findings in the paper are consistent). Furthermore, two variations of `math_10k.json` have been uploaded, `math_7K.json` where the MAWPS samples have been deleted, and `math_14k.json` where the MAWPS samples have been deleted as well and we combine ChatGPT and GPT-4 rationales. Sincerely apologize for any inconvenience!

## Setup

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Set environment variables, or modify the files referencing `BASE_MODEL`:

```bash
# Files referencing `BASE_MODEL`
# export_hf_checkpoint.py
# export_state_dict_checkpoint.py

export BASE_MODEL=/home/aneek/models/Llama-3.2-1B
```

Both `finetune.py` and `generate.py` use `--base_model` flag as shown further below.

3. If bitsandbytes doesn't work, [install it from source.](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md) Windows users can follow [these instructions](https://github.com/tloen/alpaca-lora/issues/17).

## Training(finetune.py)

This file contains some code related to prompt construction and tokenization.In this file, specify different adapters and different sets of data, so that different models can be trained. 

Example usage for multiple GPUs:

```bash
WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=3192 finetune.py \
  --base_model '/home/aneek/models/Llama-3.2-1B' \
  --data_path '/home/aneek/ActiveLearning/data/raw/classification/boolq' \
  --output_dir './trained_models/llama-lora' \
  --batch_size 16 \
  --micro_batch_size 4 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name lora

  WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=2,0 torchrun --nproc_per_node=1 --master_port=3193 finetune.py   --base_model '/home/models/Llama-3.3-70B-Instruct'   --data_path '/home/aneek/LLM-Adapters/ft-training_set/dataset/combined/train.json'   --output_dir './trained_models/instruction/llama-3.3-70B-combined-lora'   --batch_size 16   --micro_batch_size 4   --num_epochs 3   --learning_rate 3e-4   --cutoff_len 256   --val_set_size 120   --adapter_name lora


  WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=3192 finetune.py   --base_model '/home/models/Llama-3.2-11B-Vision-Instruct'   --data_path '/home/aneek/LLM-Adapters/ft-training_set/dataset/combined/train_combined.json'   --output_dir './trained_models/vision_11b/llama-11B-combined-lora'   --batch_size 16   --micro_batch_size 4   --num_epochs 3   --learning_rate 3e-4   --cutoff_len 256   --val_set_size 120   --adapter_name lora

  WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2   finetune.py   --base_model /home/aneek/models/Llama-3.2-1B   --data_path /home/aneek/LLM-Adapters/dataset/AddSub/AddSub.json   --output_dir ./trained_models/llama-addsub-1B-overfit   --batch_size 4   --micro_batch_size 1   --num_epochs 15   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --adapter_name lora

  WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2   finetune.py   --base_model /home/aneek/models/Llama-3.2-1B   --data_path /home/aneek/LLM-Adapters/dataset/AQuA/AQuA.json   --output_dir ./trained_models/llama-AQuA-1B-lora   --batch_size 4   --micro_batch_size 1   --num_epochs 20   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --adapter_name lora

  WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2   finetune.py   --base_model /home/aneek/models/Llama-3.2-1B   --data_path /home/aneek/LLM-Adapters/dataset/ARC-Challenge/train.json   --output_dir ./trained_models/llama-ARC-Challenge-1B-lora   --batch_size 4   --micro_batch_size 1   --num_epochs 15   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --adapter_name lora

  WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2   finetune.py   --base_model /home/aneek/models/Llama-3.2-1B   --data_path /home/aneek/LLM-Adapters/dataset/MultiArith/MultiArith.json   --output_dir ./trained_models/llama-MultiArith-1B-lora  --batch_size 4   --micro_batch_size 1   --num_epochs 15   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --adapter_name lora

WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2   finetune.py   --base_model /home/aneek/models/Llama-3.2-1B   --data_path /home/aneek/LLM-Adapters/dataset/hellaswag/train.json   --output_dir ./trained_models/llama-hellaswag-1B   --batch_size 4   --micro_batch_size 1   --num_epochs 15   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 

1. --pending 

WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2  --master_port=3190 finetune.py   --base_model /home/aneek/models/Llama-3.2-1B   --data_path /home/aneek/LLM-Adapters/dataset/SingleEq/SingleEq.json   --output_dir ./trained_models/llama-SingleEq-1B-lora   --batch_size 4   --micro_batch_size 1   --num_epochs 15   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --adapter_name lora


WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2  --master_port=3189 finetune.py   --base_model /home/aneek/models/Llama-3.2-1B   --data_path /home/aneek/LLM-Adapters/dataset/SingleEq/SingleEq.json   --output_dir ./trained_models/llama-SingleEq-1B  --batch_size 4   --micro_batch_size 1   --num_epochs 15   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 

2. --pending 

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1  --master_port=3191 finetune.py   --base_model /home/aneek/models/Llama-3.2-1B   --data_path /home/aneek/LLM-Adapters/dataset/boolq/train.json   --output_dir ./trained_models/llama-boolq-1B-lora   --batch_size 4   --micro_batch_size 1   --num_epochs 5   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --adapter_name lora


WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2  --master_port=3192 finetune.py   --base_model /home/aneek/models/Llama-3.2-1B   --data_path /home/aneek/LLM-Adapters/dataset/boolq/train.json   --output_dir ./trained_models/llama-boolq-1B  --batch_size 4   --micro_batch_size 1   --num_epochs 15   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 



3. --pending 

WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2  --master_port=3193 finetune.py   --base_model /home/aneek/models/Llama-3.2-1B   --data_path /home/aneek/LLM-Adapters/dataset/AQuA/AQuA.json   --output_dir ./trained_models/llama-AQuA-1B-lora   --batch_size 4   --micro_batch_size 1   --num_epochs 15   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --adapter_name lora


WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2  --master_port=3194 finetune.py   --base_model /home/aneek/models/Llama-3.2-1B   --data_path /home/aneek/LLM-Adapters/dataset/SingleEq/SingleEq.json   --output_dir ./trained_models/llama-SingleEq-1B  --batch_size 4   --micro_batch_size 1   --num_epochs 15   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 


4. --pending 

WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2  --master_port=3195 finetune.py   --base_model /home/aneek/models/Llama-3.2-1B   --data_path /home/aneek/LLM-Adapters/dataset/mawps/trainset.json  --output_dir ./trained_models/llama-mawps-1B-lora   --batch_size 4   --micro_batch_size 1   --num_epochs 15   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --adapter_name lora

# ----- didnt work ------


WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2  --master_port=3196 finetune.py   --base_model /home/aneek/models/Llama-3.2-1B   --data_path /home/aneek/LLM-Adapters/dataset/SingleEq/SingleEq.json   --output_dir ./trained_models/llama-SingleEq-1B  --batch_size 4   --micro_batch_size 1   --num_epochs 15   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 

5. --done

WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2  --master_port=3197 finetune.py   --base_model /home/aneek/models/Llama-3.2-1B   --data_path /home/aneek/LLM-Adapters/dataset/social_i_qa/train.json   --output_dir ./trained_models/llama-social_i_qa-1B-lora   --batch_size 4   --micro_batch_size 1   --num_epochs 3   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --adapter_name lora


WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2  --master_port=3198 finetune.py   --base_model /home/aneek/models/Llama-3.2-1B   --data_path /home/aneek/LLM-Adapters/dataset/social_i_qa/train.json   --output_dir ./trained_models/llama-social_i_qa-1B  --batch_size 4   --micro_batch_size 1   --num_epochs 15   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 

6. --pending 

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1  --master_port=3202 finetune.py   --base_model /home/aneek/models/Llama-3.2-1B   --data_path /home/aneek/LLM-Adapters/dataset/ARC-Easy/train.json   --output_dir ./trained_models/llama-ARC-Easy-1B-lora   --batch_size 4   --micro_batch_size 1   --num_epochs 5   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --adapter_name lora


WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2  --master_port=3199 finetune.py   --base_model /home/aneek/models/Llama-3.2-1B   --data_path /home/aneek/LLM-Adapters/dataset/ARC-Easy/train.json   --output_dir ./trained_models/llama-ARC-Easy-1B  --batch_size 4   --micro_batch_size 1   --num_epochs 15   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 

# ----done------



7. --pending 

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=3201 finetune.py   --base_model /home/aneek/models/Llama-3.2-1B   --data_path /home/aneek/LLM-Adapters/dataset/openbookqa/train.json   --output_dir ./trained_models/llama-openbookqa-1B-lora   --batch_size 4   --micro_batch_size 1   --num_epochs 5   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --adapter_name lora


WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2  --master_port=3202 finetune.py   --base_model /home/aneek/models/Llama-3.2-1B   --data_path /home/aneek/LLM-Adapters/dataset/openbookqa/train.json   --output_dir ./trained_models/llama-openbookqa-1B  --batch_size 4   --micro_batch_size 1   --num_epochs 15   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 

8. --pending 

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1  --master_port=3203 finetune.py   --base_model /home/aneek/models/Llama-3.2-1B   --data_path /home/aneek/LLM-Adapters/dataset/piqa/train.json   --output_dir ./trained_models/llama-piqa-1B-lora   --batch_size 4   --micro_batch_size 1   --num_epochs 5   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --adapter_name lora


WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2  --master_port=3204 finetune.py   --base_model /home/aneek/models/Llama-3.2-1B   --data_path /home/aneek/LLM-Adapters/dataset/piqa/train.json  --output_dir ./trained_models/llama-piqa-1B  --batch_size 4   --micro_batch_size 1   --num_epochs 15   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 



9. --pending 

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1  --master_port=3205 finetune.py   --base_model /home/aneek/models/Llama-3.2-3B-Instruct   --data_path /home/aneek/LLM-Adapters/dataset/SVAMP/SVAMP.json  --output_dir ./trained_models/llama-SVAMP-1B-lora   --batch_size 4   --micro_batch_size 1   --num_epochs 15   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --adapter_name lora


WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2  --master_port=3206 finetune.py   --base_model /home/aneek/models/Llama-3.2-1B   --data_path /home/aneek/LLM-Adapters/dataset/SVAMP/SVAMP.json   --output_dir ./trained_models/llama-SVAMP-1B  --batch_size 4   --micro_batch_size 1   --num_epochs 15   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 

10. --pending 

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1  --master_port=3207 finetune.py   --base_model /home/aneek/models/Llama-3.2-1B   --data_path /home/aneek/LLM-Adapters/dataset/winogrande/train.json  --output_dir ./trained_models/llama-winogrande-1B-lora   --batch_size 4   --micro_batch_size 1   --num_epochs 3   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --adapter_name lora


WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2  --master_port=3208 finetune.py   --base_model /home/aneek/models/Llama-3.2-1B   --data_path /home/aneek/LLM-Adapters/dataset/winogrande/train.json  --output_dir ./trained_models/llama-winogrande-3B  --batch_size 4   --micro_batch_size 1   --num_epochs 15   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 

/home/aneek/LLM-Adapters/ft-training_set/alpaca_data_cleaned.json
/home/aneek/LLM-Adapters/ft-training_set/commonsense_15k.json
/home/aneek/LLM-Adapters/ft-training_set/math_14k.json

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=3208 finetune.py   --base_model /home/models/Llama-3.2-3B-Instruct-Sparse-0.33/   --data_path /home/aneek/LLM-Adapters/ft-training_set/alpaca_data_cleaned.json  --output_dir ./trained_models/instruct/llama-alpaca_data_cleaned-3B-Sparse0.33-lora  --batch_size 4   --micro_batch_size 1   --num_epochs 3   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 

echo 'WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=3210 finetune.py   --base_model /home/aneek/models/Llama-3.2-3B   --data_path /home/aneek/LLM-Adapters/ft-training_set/commonsense_15k.json --output_dir ./trained_models/llama-commonsense_15k-3B  --batch_size 4   --micro_batch_size 1   --num_epochs 3   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120' | at now + 1 hour

WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=3210 finetune.py   --base_model /home/aneek/models/Llama-3.2-3B   --data_path /home/aneek/LLM-Adapters/ft-training_set/dataset/combined/train.json --output_dir ./trained_models/llama-combined-3B-lora  --batch_size 4   --micro_batch_size 1   --num_epochs 3   --learning_rate 3e-5   --cutoff_len 256  --adapter_name lora --val_set_size 120

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=3090 finetune.py   --base_model /home2/models/Llama-3.2-3B-Instruct-Sparse-0.33/ --data_path /home/aneek/LLM-Adapters/ft-training_set/math_50k.json --output_dir ./trained_models/instruct/llama-math_50k-3B-Sparse0.33-lora  --batch_size 4   --micro_batch_size 1   --num_epochs 3   --learning_rate 3e-5   --cutoff_len 256  --adapter_name lora --val_set_size 120

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=3092 finetune.py   --base_model '/home2/models/Llama-3.2-3B-Instruct-Sparse-0.33/'   --data_path /home/aneek/LLM-Adapters/ft-training_set/commonsense_170k.json --output_dir ./trained_models/instruct_sparse/llama-commonsense_170k-3B-sparse-lora  --batch_size 4   --micro_batch_size 1   --num_epochs 3   --learning_rate 3e-5   --cutoff_len 256  --adapter_name lora --val_set_size 120

/home/aneek/LLM-Adapters/ft-training_set/commonsense_170k.json


/home/aneek/LLM-Adapters/dataset/SingleEq/SingleEq.json
/home/aneek/LLM-Adapters/dataset/boolq/train.json
/home/aneek/LLM-Adapters/dataset/social_i_qa/train.json
/home/aneek/LLM-Adapters/dataset/ARC-Easy/train.json
/home/aneek/LLM-Adapters/dataset/openbookqa/train.json
/home/aneek/LLM-Adapters/dataset/piqa/train.json
/home/aneek/LLM-Adapters/dataset/SVAMP/SVAMP.json
/home/aneek/LLM-Adapters/dataset/winogrande/train.json


```

The `math_10k.json` data is collected with the training sets of GSM8K, MAWPS, and AQuA(1000 examples). `yahma/llama-7b-hf` is a base model, LLaMa-7B. Add `lora` adapter to this model.

Example usage for Single GPUs:

```bash
CUDA_VISIBLE_DEVICES=2 python /home/aneek/LLM-Adapters/finetune.py \
  --base_model /home/models/nvidia-sparse/OpenReasoning-Nemotron-14B-Sparse-0.67 \
  --data_path '/home/aneek/LLM-Adapters/ft-training_set/math_14k.json' \
  --output_dir './trained_models/OpenReasoning-Nemotron-14B-Sparse-Ensemble_fullMath14k/OpenReasoning-Nemotron-14B-Sparse-0.67-Math14k-fullFT' \
  --batch_size 16 \
  --micro_batch_size 4 \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --cutoff_len 256 \
  --val_set_size 120 \
  --adapter_name lora
```


 python /home/aneek/LLM-Adapters/finetune.py   --base_model "/home/models/nvidia-sparse/OpenReasoning-Nemotron-14B-Sparse-0.67"   --data_path "/home/aneek/LLM-Adapters/ft-training_set/math_14k.json"   --output_dir "./trained_models/OpenReasoning-Nemotron-14B-Sparse-Ensemble_fullMath14k/OpenReasoning-Nemotron-14B-Sparse-0.67-Math14k-fullFT"   --batch_size 1   --micro_batch_size 1   --num_epochs 3   --learning_rate 3e-5   --cutoff_len 64   --val_set_size 120   --bf16   --gradient_checkpointing



/home/aneek/LLM-Adapters/finetune.py --base_model /home/models/nvidia-sparse/OpenReasoning-Nemotron-14B-Sparse-0.33 --data_path /home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part3_of_3.json --output_dir /home/aneek/LLM-Adapters/trained_models/OpenReasoning-Nemotron-14B-Sparse-0.33-Ensemble_split_33/OpenReasoning-Nemotron-14B-Sparse-0.33-Math14k-part3 --batch_size 4 --micro_batch_size 1 --num_epochs 3 --learning_rate 3e-5 --cutoff_len 256 --val_set_size 120 --wandb_run_name FT-OpenReasoning-Nemotron-14B-Sparse-0.33-split_33-part3

Moreover, you can use `--use_gradient_checkpointing` to save more GPU memory, but it will increase the training time.

To use the AdapterH, just add the following arguments:

```bash
--adapter_name bottleneck # use the bottleneck adapter, refers to AdapterH in the result table
```

To use the AdapterP, just add the following arguments:

```bash
--adapter_name bottleneck 
--use_adapterp  # use the AdapterP, refers to AdapterP in the result table
```

To use parallel adapter, just add the following arguments:

```bash
--adapter_name bottleneck
--use_parallel_adapter
```

Note that, In order to facilitate INT8 training of large models with parallel adapters, we have adopted a technique whereby the parallel adapter layers are incorporated into multi-head attention layers and MLP layers, in parallel with Linear layers. It is different from [Hu et al. (2021)](https://arxiv.org/pdf/2106.09685.pdf). 

## Inference (generate.py)

This file reads the foundation model from the Hugging Face model hub and the LoRA weights from `'./trained_models/llama-lora'` , and runs a Gradio interface for inference on a specified input. Users should treat this as example code for the use of the model, and modify it as needed.
Example usage:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun generate.py \
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights './trained_models/llama-lora'
```

## Evaluation (evaluate.py)

To evaluate the performance of the finetuned model on the Arithmetic Reasoning tasks, you can use the following command:

```bash
CUDA_VISIBLE_DEVICES=0,1 python evaluate.py  --model 'Llama-3.2-3B' --adapter LoRA --dataset gsm8k --base_model '/home/aneek/models/Llama-3.2-3B' --lora_weights '/home/aneek/LLM-Adapters/trained_models/llama-math_50k-3B-lora'


```

["multiarith", "addsub", "singleeq", "gsm8k", "svamp"]:


CUDA_VISIBLE_DEVICES=0 python evaluate.py 
    --model LLaMA-7B
    --adapter LoRA \   #specify the adapter name ["LoRA", "AdapterH", "AdapterP", "Parallel"， "Scaled_Parallel""]
    --dataset SVAMP \  #specify the test dataset
    --base_model 'yahma/llama-7b-hf' \
    --lora_weights '/home/aneek/LLM-Adapters/trained_models/llama-1B-gsm8k-lora'



#!/usr/bin/env python
"""
active_learning.py – adds an uncertainty-sampling loop around finetune.py
Usage example (2 AL rounds, 10 % → 30 % of data):
python active_learning.py \
    --base_model /path/llama \
    --data_path /path/data.json \
    --output_dir ./exp \
    --rounds 3 --init_frac 0.1 --acq_frac 0.2


    active learning fraction

<!-- ## Resource Consumption

There is a table of resouce needed for different adapters, which contains Trainable Parameters, GPU RAM Usage, and Fine-tuning Time on the Arithmetic Reasoning dataset `math_10k.json`

Hyper-parameter setting: num_epochs=3, lora_r=8, lora_alpha=16, bottleneck_size=256

Models: LLaMA-13B, LLaMA-7B, BLOOM-6.7B, GPT-j-6B
Dataset: 3.2K math word problems

Hardware: 2*3090 GPUs

| Model                 | Trainable Parameters | GPU RAM Usage | Fine-tuning Time |
|-----------------------|----------------------|---------------|------------------|
| LLaMA-7B-LoRA         | 4.2M                 | 18GB          |     4h           | 
| LLaMA-7B-AdapterH     | 200M                 | 22GB          |     4h           | 
| LLaMA-7B-AdapterP     | 200M                 | 22GB          |     4h           | 
| LLaMA-7B-Parallel     | 200M                 | 22GB          |     4h           |  -->


## Finetune Result
There are the finetune results in different models with 4 math reasoning datasets, which contains GSM8K, AQuA, SVAMP, and MAWPS. In this table, we use the optimal configuration and placement of Prefix-Tuning, Series Adapter, LoRA, and Parallel Adapter according to the empirical study in our [paper](https://aclanthology.org/2023.emnlp-main.319/).

| Model                 | GSM8K  | AQuA   |   MAWPS  |  SVAMP | Average |
|-----------------------|--------|--------|----------|--------|---------|
| GPT-3.5               |**56.4**|**38.9**| **87.4** |**69.9**|**63.2** |
| BLOOMz-7B-Prefix	    | 13.8   |  12.5  |   47.5   |  24.1  |  24.5   |
| BLOOMz-7B-Series	    | 14.3   |  20.5  |   62.2   |  38.1  |  33.8   |
| BLOOMz-7B-Parallel	| 18.5   |  18.9  |   70.6   |  36.4  |  36.1   |
| BLOOMz-7B-LoRA	    | 17.4	 |  21.3  |   70.2   |  41.0  |  37.5   |
| GPT-j-6B-Prefix	    | 16.0	 |  14.7  |   59.2   |  31.0  |  30.2   |
| GPT-j-6B-Series	    | 19.5	 |  15.0  |   80.3   |  43.6  |  39.6   |
| GPT-j-6B-Parallel	    | 18.9	 |  17.9  |   78.2   |  41.1  |  39.0   |
| GPT-j-6B-LoRA	        | 23.0	 |  16.1  |   79.4   |  46.0  |  41.1   |
| LLaMA-7B-Prefix	    | 24.4	 |  14.2  |   63.4   |  38.1  |  35.0   |
| LLaMA-7B-Series	    | 33.3	 |  15.0  |   77.7   |  52.3  |  44.6   |
| LLaMA-7B-Parallel	    | 35.3	 |  18.1  |   82.4   |  49.6  |  46.4   |
| LLaMA-7B-LoRA	        | 37.5	 |  18.9  |   79.0   |  52.1  |  46.9   |
| LLaMA-13B-Prefix	    | 31.1	 |  15.7  |   66.8   |  41.4  |  38.8   |
| LLaMA-13B-Series	    | 44.0	 |  22.0  |   78.6   |  50.8  |  48.9   |
| LLaMA-13B-Parallel	| 43.3	 |  20.5  |   81.1   |  55.7  |  50.2   |
| LLaMA-13B-LoRA	    | 47.5	 |  18.5  |   83.6   |  54.6  |  51.1   |


There are the finetune results in different models with eight commonsense reasoning datasets.

| Model                 |  BoolQ  |  PIQA  |  SIQA  |  HellaSwag  |  WinoGrande  |  ARC-e  |  ARC-c  |  OBQA  |  Average  |
|-----------------------|---------|--------|--------|-------------|--------------|---------|---------|--------|-----------|
| ChatGPT               | **73.1**|**85.4**|  68.5  |  78.5       |  66.1        |**89.8** |**79.9** |  74.8  |  77.0     |
| BLOOMz-7B-Prefix	    |   45.6  |  53.7  |  46.3  |  26.7       |  49.5        |  52.1   |  39.7   |  44.3  |  44.7     |
| BLOOMz-7B-Series	    |   65.4  |  70.4  |  73.6  |  53.4       |  69.3        |  72.3   |  55.9   |  68.0  |  66.0     |
| BLOOMz-7B-Parallel	  |   64.1  |  71.5  |  72.1  |  52.9       |  67.0        |  70.5   |  54.7   |  69.6  |  65.3     |
| BLOOMz-7B-LoRA	      |   65.9  |  75.3  |  74.5  |  57.3       |  72.5        |  74.6   |  57.8   |  73.4  |  68.9     |
| GPT-j-6B-Prefix	      |   63.1  |  66.9  |  68.7  |  34.4       |  64.5        |  64.4   |  46.8   |  59.0  |  58.5     |
| GPT-j-6B-Series	      |   62.1  |  63.5  |  72.3  |  30.6       |  68.0        |  63.9   |  48.1   |  63.8  |  59.0     |
| GPT-j-6B-Parallel	    |   62.2  |  69.7  |  70.0  |  41.7       |  65.0        |  60.2   |  44.6   |  58.2  |  59.0     |
| GPT-j-6B-LoRA	        |   62.4  |  68.6  |  49.5  |  43.1       |  57.3        |  43.4   |  31.0   |  46.6  |  50.2     |
| LLaMA-7B-Prefix	      |   64.3  |  76.8  |  73.9  |  42.1       |  72.1        |  72.9   |  54.0   |  60.6  |  64.6     |
| LLaMA-7B-Series	      |   63.0  |  79.2  |  76.3  |  67.9       |  75.7        |  74.5   |  57.1   |  72.4  |  70.8     |
| LLaMA-7B-Parallel	    |   67.9  |  76.4  |  78.8  |  69.8       |  78.9        |  73.7   |  57.3   |  75.2  |  72.3     |
| LLaMA-7B-LoRA	        |   68.9  |  80.7  |  77.4  |  78.1       |  78.8        |  77.8   |  61.3   |  74.8  |  74.7     |
| LLaMA-13B-Prefix	    |   65.3  |  75.4  |  72.1  |  55.2       |  68.6        |  79.5   |  62.9   |  68.0  |  68.4     |
| LLaMA-13B-Series	    |   71.8  |  83.0  |  79.2  |  88.1       |  82.4        |  82.5   |  67.3   |  81.8  |  79.5     |
| LLaMA-13B-Parallel	  |   72.5  |  84.8  |  79.8  |**92.1**     |**84.7**      |  84.2   |  71.2   |**82.4**|**81.5**   |
| LLaMA-13B-LoRA	      |   72.1  |  83.5  |**80.5**|  90.5       |  83.7        |  82.8   |  68.3   |**82.4**|  80.5     |


### Adapter support matrix
This metrix shows whether different models can use LoRA,AdapterH,AdapterP,Parallel and Scaled Parallel adapters.

| Adapter      | LoRA | AdapterH | AdapterP | Parallel| Prefix Tuning	|P-Tuning|Prompt Tuning|
|--------------|-------|-------|----------|-------|-------|-------|-------|
| LLaMA        | ✅     | ✅     | ✅        |✅     | ✅     | ✅     | ✅     |
| BLOOM        | ✅     | ✅     | ✅        |✅     | ✅     | ✅     | ✅     | 
| GPT-J        | ✅     | ✅     | ✅        |✅     | ✅     | ✅     | ✅     |
| OPT          | ✅     | ✅     | ✅        |✅     | ✅     | ✅     | ✅     |
| GPT-2        | ✅     | 🔧Developing | 🔧Developing|🔧Developing | ✅     | ✅     | ✅     | 
| GPT-Neo      | ✅     | ✅     | ✅        | ✅    | ✅     | ✅     | ✅     | 
| GPT-NeoX-20B | ✅     | 🔧Developing | 🔧Developing|🔧Developing | ✅     | ✅     | ✅     |
| ChatGLM      | ✅     | ✅     | ✅        |✅     | ✅     | ✅     | ✅     | 


### TODO List
- [x] Add AdapterH
- [x] Add AdapterP
- [x] Add Parallel Adapter
- [ ] Support More LLMs
- [ ] Support Multiple Adapter
- [ ] Support Adapter Composition
- [ ] Support Adapter Fusion


## :star: Star History

[![Star History Chart](https://api.star-history.com/svg?repos=AGI-Edgerunners/LLM-Adapters&type=Date)](https://star-history.com/#AGI-Edgerunners/LLM-Adapters&Date)

## Citing <img src="picture.jpg" width="14px" height="14px"> LLM-Adapter

If you use <img src="picture.jpg" width="14px" height="14px"> LLM-Adapters in your publication, please cite it by using the following BibTeX entry.

```bibtex
@article{hu2023llm,
  title={LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models},
  author={Hu, Zhiqiang and Lan, Yihuai and Wang, Lei and Xu, Wanyu and Lim, Ee-Peng and Lee, Roy Ka-Wei and Bing, Lidong and Poria, Soujanya},
  journal={arXiv preprint arXiv:2304.01933},
  year={2023}
}
```

## Acknowledgement

This repo benefits from [PEFT](https://github.com/huggingface/peft), [Adapter-Transformer](https://github.com/adapter-hub/adapter-transformers), [Alpaca-lora](https://github.com/tloen/alpaca-lora). Thanks for their wonderful works. Additionally, we thank DONG Shan and [dream.ai](https://dream.ai/create) for the exceptional logo design, which has added immense value to our project.



WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=3208 finetune.py   --base_model /home/models/Llama-3.2-1B-Instruct   --data_path /home/aneek/LLM-Adapters/ft-training_set/commonsense_15k.json --output_dir ./trained_models/instruct/llama-commonsense_15k-1B-lora  --batch_size 4   --micro_batch_size 1   --num_epochs 1   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 



WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=3211 finetune.py   --base_model /home/models/Llama-3.2-1B-Instruct   --data_path /home/aneek/LLM-Adapters/ft-training_set/math_14k_part2.json --output_dir ./trained_models/instruct/llama-math-14k-part2-1B-lora  --batch_size 4   --micro_batch_size 1   --num_epochs 1   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120


./trained_models/instruct/llama-math-7k-1B-lora

```bash

CUDA_VISIBLE_DEVICES=1 python evaluate.py  --model 'Llama-3.2-1B' --adapter LoRA --dataset gsm8k --base_model '/home/models/Llama-3.2-1b-Instruct' --lora_weights '/home/aneek/LLM-Adapters/trained_models/instruct/llama-math-7k-1B-lora'

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3213 finetune.py \
  --base_model /home/models/Llama-3.2-3B-Instruct-0.33-Wanda \
  --data_path /home/aneek/LLM-Adapters/ft-training_set/math_14k.json \
  --output_dir ./trained_models/instruct_3B/llama-math-14k-3B-Wanda-lora \
  --batch_size 4 --micro_batch_size 1 --num_epochs 3 \
  --learning_rate 3e-5 --cutoff_len 256 --val_set_size 120



WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3215 active_learning.py \
    --base_model /home/models/Llama-3.2-3B-Instruct-0.33-Wanda \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/math_14k.json \
    --output_dir ./trained_models/instruct_3B/llama-math-14k-al50-3B-Wanda-lora \
    --rounds 3 \
    --init_frac 0.1 \
    --acq_frac 0.2


WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=3210 finetune.py   --base_model /home/models/Llama-3.2-3B-Instruct   --data_path /home/aneek/LLM-Adapters/ft-training_set/math_50k.json --output_dir ./trained_models/instruct/llama-math_50-1B-lora  --batch_size 4   --micro_batch_size 1   --num_epochs 3   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 

/home/aneek/LLM-Adapters/ft-training_set/alpaca_data_cleaned.json
/home/aneek/LLM-Adapters/ft-training_set/commonsense_15k.json


WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=3211 finetune.py   --base_model /home/models/Llama-3.2-1B-Instruct   --data_path /home/aneek/LLM-Adapters/ft-training_set/alpaca_data_cleaned.json --output_dir ./trained_models/instruct/llama-alpaca_data_cleaned-1B-lora  --batch_size 4   --micro_batch_size 1   --num_epochs 3   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 


WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=3212 finetune.py   --base_model /home/models/Llama-3.2-3B-Instruct   --data_path /home/aneek/LLM-Adapters/ft-training_set/commonsense_170k.json --output_dir ./trained_models/instruct/llama-commonsense_170k-3B-lora  --batch_size 4   --micro_batch_size 1   --num_epochs 2   --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 

```
copying data from megatron to bumblebee:

rsync -avh --progress /home/aneek/models/  aneek@10.225.65.83:/home/aneek/LLM-Adapters/

Evaluation benchmarks on Math / Commonsense datasets for Llama-3.2-1B trained on math and commonsense datasets:

````bash

Commons_sense_eval : ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"] for commonsense_170k_1B model


CUDA_VISIBLE_DEVICES=0,1 python commonsense_evaluate.py  --model 'Llama-3.2-1B-Instruct' --adapter LoRA --dataset boolq --base_model '/home/models/Llama-3.2-1B-Instruct/' --lora_weights '/home/aneek/LLM-Adapters/trained_models/instruct/llama-commonsense_170k-1B-lora'  --batch_size 16

CUDA_VISIBLE_DEVICES=0,1 python commonsense_evaluate.py  --model 'Llama-3.2-1B-Instruct' --adapter LoRA --dataset piqa --base_model '/home/models/Llama-3.2-1B-Instruct/' --lora_weights '/home/aneek/LLM-Adapters/trained_models/instruct/llama-commonsense_170k-1B-lora'  --batch_size 16

CUDA_VISIBLE_DEVICES=0,1 python commonsense_evaluate.py  --model 'Llama-3.2-1B-Instruct' --adapter LoRA --dataset social_i_qa --base_model '/home/models/Llama-3.2-1B-Instruct/' --lora_weights '/home/aneek/LLM-Adapters/trained_models/instruct/llama-commonsense_170k-1B-lora'  --batch_size 16

CUDA_VISIBLE_DEVICES=0,1 python commonsense_evaluate.py  --model 'Llama-3.2-1B-Instruct' --adapter LoRA --dataset hellaswag --base_model '/home/models/Llama-3.2-1B-Instruct/' --lora_weights '/home/aneek/LLM-Adapters/trained_models/instruct/llama-commonsense_170k-1B-lora'  --batch_size 16

CUDA_VISIBLE_DEVICES=0,1 python commonsense_evaluate.py  --model 'Llama-3.2-1B-Instruct' --adapter LoRA --dataset winogrande --base_model '/home/models/Llama-3.2-1B-Instruct/' --lora_weights '/home/aneek/LLM-Adapters/trained_models/instruct/llama-commonsense_170k-1B-lora'  --batch_size 16
------------------------------------------------------------------------------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0,1 python commonsense_evaluate.py  --model 'Llama-3.2-1B-Instruct' --adapter LoRA --dataset ARC-Challenge --base_model '/home/models/Llama-3.2-1B-Instruct/' --lora_weights '/home/aneek/LLM-Adapters/trained_models/instruct/llama-commonsense_170k-1B-lora'  --batch_size 16

CUDA_VISIBLE_DEVICES=0,1 python commonsense_evaluate.py  --model 'Llama-3.2-1B-Instruct' --adapter LoRA --dataset openbookqa --base_model '/home/models/Llama-3.2-1B-Instruct/' --lora_weights '/home/aneek/LLM-Adapters/trained_models/instruct/llama-commonsense_170k-1B-lora'  --batch_size 16

CUDA_VISIBLE_DEVICES=0,1 python commonsense_evaluate.py  --model 'Llama-3.2-1B-Instruct' --adapter LoRA --dataset ARC-Easy --base_model '/home/models/Llama-3.2-1B-Instruct/' --lora_weights '/home/aneek/LLM-Adapters/trained_models/instruct/llama-commonsense_170k-1B-lora'  --batch_size 16




Commons_sense_eval : ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"] for commonsense_170k_1B model


CUDA_VISIBLE_DEVICES=0 python commonsense_evaluate.py  --model 'Llama-3.2-3B-Instruct' --adapter LoRA --dataset boolq --wandb_run_name boolq   --base_model '/home2/models/Llama-3.2-3B-Instruct-Sparse-0.33/' --lora_weights '/home2/palash/aneek/LLM-Adapters/trained_models/instruct_sparse/llama-commonsense_170k-3B-sparse-lora'  --batch_size 16

CUDA_VISIBLE_DEVICES=0 python commonsense_evaluate.py  --model 'Llama-3.2-3B-Instruct' --adapter LoRA --dataset piqa --base_model '/home2/models/Llama-3.2-3B-Instruct-Sparse-0.33/' --lora_weights '/home/aneek/LLM-Adapters/trained_models/instruct/llama-commonsense_170k-3B-lora'  --batch_size 16

CUDA_VISIBLE_DEVICES=0 python commonsense_evaluate.py  --model 'Llama-3.2-3B-Instruct' --adapter LoRA --dataset social_i_qa --base_model '/home/models/Llama-3.2-3B-Instruct-Sparse-0.33/' --lora_weights '/home/aneek/LLM-Adapters/trained_models/instruct/llama-commonsense_170k-3B-lora'  --batch_size 16

CUDA_VISIBLE_DEVICES=1 python commonsense_evaluate.py  --model 'Llama-3.2-3B-Instruct' --adapter LoRA --dataset hellaswag --base_model '/home/models/Llama-3.2-3B-Instruct-Sparse-0.33/' --lora_weights '/home/aneek/LLM-Adapters/trained_models/instruct/llama-commonsense_170k-3B-lora'  --batch_size 16

CUDA_VISIBLE_DEVICES=0,1 python commonsense_evaluate.py  --model 'Llama-3.2-3B-Instruct' --adapter LoRA --dataset winogrande --base_model '/home/models/Llama-3.2-3B-Instruct-Sparse-0.33/' --lora_weights '/home/aneek/LLM-Adapters/trained_models/instruct/llama-commonsense_170k-3B-lora'  --batch_size 16

CUDA_VISIBLE_DEVICES=0,1 python commonsense_evaluate.py  --model 'Llama-3.2-3B-Instruct' --adapter LoRA --dataset ARC-Challenge --base_model '/home/models/Llama-3.2-3B-Instruct-Sparse-0.33/' --lora_weights '/home/aneek/LLM-Adapters/trained_models/instruct/llama-commonsense_170k-3B-lora'  --batch_size 16

CUDA_VISIBLE_DEVICES=3 python commonsense_evaluate.py  --model 'Llama-3.2-3B-Instruct' --adapter LoRA --dataset openbookqa --base_model '/home/models/Llama-3.2-3B-Instruct-Sparse-0.33/' --lora_weights '/home/aneek/LLM-Adapters/trained_models/instruct/llama-commonsense_170k-3B-lora'  --batch_size 16

CUDA_VISIBLE_DEVICES=2 python commonsense_evaluate.py  --model 'Llama-3.2-3B-Instruct' --adapter LoRA --dataset ARC-Easy --base_model '/home/models/Llama-3.2-3B-Instruct-Sparse-0.33/' --lora_weights '/home/aneek/LLM-Adapters/trained_models/instruct/llama-commonsense_170k-3B-lora'  --batch_size 16


----------------------------------------------------------------------------------------------------------------------------------------


/home2/palash/aneek/LLM-Adapters/trained_models/instruct_sparse/llama-commonsense_170k-3B-sparse-lora


`````

Math: ['AddSub', 'MultiArith', 'SingleEq', 'gsm8k', 'AQuA', 'SVAMP'] -- Math_14k -- /home2/palash/aneek/LLM-Adapters/trained_models/al-math14k-3B-rand50/round_0

/home2/palash/aneek/LLM-Adapters/trained_models/al-math14k-3B-sparse-rand50/round_0

CUDA_VISIBLE_DEVICES=1 python evaluate.py  --model 'Llama-4-Scout-17B-16E-Instruct' --adapter LoRA --dataset gsm8k --base_model '/home/models/Llama-4-Scout-17B-16E-Instruct' --lora_weight

CUDA_VISIBLE_DEVICES=1 python evaluate.py  --model 'Llama-3.2-1B-Instruct' --adapter LoRA --dataset AddSub --base_model '/home2/models/Llama-3.2-1B-Instruct/' --lora_weights '/home2/palash/aneek/LLM-Adapters/trained_models/al-math14k-1B-heur50/round_2'

CUDA_VISIBLE_DEVICES=1 python evaluate.py  --model 'Llama-3.2-1B-Instruct' --adapter LoRA --dataset MultiArith --base_model '/home2/models/Llama-3.2-1B-Instruct/' --lora_weights '/home2/palash/aneek/LLM-Adapters/trained_models/al-math14k-1B-heur50/round_2'

CUDA_VISIBLE_DEVICES=1 python evaluate.py  --model 'Llama-3.2-1B-Instruct' --adapter LoRA --dataset SingleEq --base_model '/home2/models/Llama-3.2-1B-Instruct/' --lora_weights '/home2/palash/aneek/LLM-Adapters/trained_models/al-math14k-1B-heur50/round_2'

CUDA_VISIBLE_DEVICES=1 python evaluate.py  --model 'Llama-3.2-1B-Instruct' --adapter LoRA --dataset AQuA --base_model '/home2/models/Llama-3.2-1B-Instruct/' --lora_weights '/home2/palash/aneek/LLM-Adapters/trained_models/al-math14k-1B-heur50/round_2'


CUDA_VISIBLE_DEVICES=1 python evaluate.py  --model 'Llama-3.2-3B-Instruct-Sparse' --adapter LoRA --dataset SVAMP --base_model '/home2/models/Llama-3.2-3B-Instruct-Sparse-0.33/' --lora_weights '/home2/palash/aneek/LLM-Adapters/trained_models/al-math14k-3B-Sparse-heur50/round_2'


run for 10% random split: Llama-3.2-1B-Instruct 

CUDA_VISIBLE_DEVICES=1 python evaluate.py  --model 'Llama-3.2-1B-Instruct' --adapter LoRA --dataset gsm8k --base_model '/home2/models/Llama-3.2-1B-Instruct/' --lora_weights '/home2/palash/aneek/LLM-Adapters/trained_models/instruction_new_14k/al-math14k-1B-rand50_10/round_0'

CUDA_VISIBLE_DEVICES=1 python evaluate.py  --model 'Llama-3.2-1B-Instruct' --adapter LoRA --dataset AddSub --base_model '/home2/models/Llama-3.2-1B-Instruct/' --lora_weights '/home2/palash/aneek/LLM-Adapters/trained_models/instruction_new_14k/al-math14k-1B-rand50_10/round_0'

CUDA_VISIBLE_DEVICES=1 python evaluate.py  --model 'Llama-3.2-1B-Instruct' --adapter LoRA --dataset MultiArith --base_model '/home2/models/Llama-3.2-1B-Instruct/' --lora_weights '/home2/palash/aneek/LLM-Adapters/trained_models/instruction_new_14k/al-math14k-1B-rand50_10/round_0'

CUDA_VISIBLE_DEVICES=1 python evaluate.py  --model 'Llama-3.2-1B-Instruct' --adapter LoRA --dataset SingleEq --base_model '/home2/models/Llama-3.2-1B-Instruct/' --lora_weights '/home2/palash/aneek/LLM-Adapters/trained_models/instruction_new_14k/al-math14k-1B-rand50_10/round_0'

CUDA_VISIBLE_DEVICES=1 python evaluate.py  --model 'Llama-3.2-1B-Instruct' --adapter LoRA --dataset AQuA --base_model '/home2/models/Llama-3.2-1B-Instruct/' --lora_weights '/home2/palash/aneek/LLM-Adapters/trained_models/instruction_new_14k/al-math14k-1B-rand50_10/round_0'


CUDA_VISIBLE_DEVICES=1 python evaluate.py  --model 'Llama-3.2-1B-Instruct' --adapter LoRA --dataset SVAMP --base_model '/home2/models/Llama-3.2-1B-Instruct/' --lora_weights '/home2/palash/aneek/LLM-Adapters/trained_models/instruction_new_14k/al-math14k-1B-rand50_10/round_0'


['AddSub', 'MultiArith', 'SingleEq', 'gsm8k', 'AQuA', 'SVAMP']


CUDA_VISIBLE_DEVICES=1  python /home/aneek/LLM-Adapters/new_eval_llm.py \
  --dataset gsm8k \
  --model Qwen3-4B-Instruct \
  --base_model "/home/models/Qwen/Qwen3-4B-Instruct-2507" \
  --adapter LoRA \
  --lora_weights ./trained_models/Qwen3/Qwen3-4B-Math14k-rnd50 \
  --batch_size_gen 16 \
  --tp_size 1 \
  --max_model_len 8192 \
  --gpu_mem_util 0.95

CUDA_VISIBLE_DEVICES=0  python /home/aneek/LLM-Adapters/evaluate.py \
  --dataset gsm8k \
  --model Qwen3-4B-Instruct \
  --base_model "/home/models/Qwen/Qwen3-4B-Instruct-2507" \
  --adapter LoRA \
  --lora_weights /home/aneek/LLM-Adapters/trained_models/Qwen3/Qwen3-4B-Math-14k\
  --batch_size_gen 16 \
  --tp_size 1 \
  --max_model_len 8192 \
  --gpu_mem_util 0.45

python - <<'PY'
from eval_llm import merge_lora
merge_lora(
  "/home/models/Qwen/Qwen3-4B-Instruct-2507",
  "/home/aneek/LLM-Adapters/trained_models/Qwen3/Qwen3-4B-Math-14k",
  "/home/aneek/merged/Qwen3-4B-Math14k-merged"
)
PY
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named 'vllm_eval'

CUDA_VISIBLE_DEVICES=1 python evaluate.py  --model 'Llama-3.2-1B-Instruct' --adapter LoRA --dataset SVAMP --base_model '/home2/models/Llama-3.2-1B-Instruct/' --lora_weights '/home2/palash/aneek/LLM-Adapters/trained_models/instruction_new_14k/al-math14k-1B-rand50_10/round_0'


WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 python  finetune.py   --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.20" \
  --data_path /home/aneek/LLM-Adapters/ft-training_set/math_14k.json \
  --output_dir ./trained_models/Qwen3_Sparse/Qwen3-8B-Sparse-0.20-Math-14k \
  --batch_size 4   --micro_batch_size 1   --num_epochs 3 \
  --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --wandb_run_name Qwen3-8B-Sparse-0.20-Math-14k

  WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 python  finetune.py   --base_model  "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.25" \
  --data_path /home/aneek/LLM-Adapters/ft-training_set/math_14k.json \
  --output_dir ./trained_models/Qwen3_Sparse/Qwen3-8B-Sparse-0.25-Math-14k \
  --batch_size 4   --micro_batch_size 1   --num_epochs 3 \
  --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --wandb_run_name Qwen3-8B-Sparse-0.25-Math-14k

  WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 python  finetune.py   --base_model  "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.33" \
  --data_path /home/aneek/LLM-Adapters/ft-training_set/math_14k.json \
  --output_dir ./trained_models/Qwen3_Sparse/Qwen3-8B-Sparse-0.33-Math-14k \
  --batch_size 4   --micro_batch_size 1   --num_epochs 3 \
  --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --wandb_run_name Qwen3-8B-Sparse-0.33-Math-14k

  WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 python  finetune.py   --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.20" \
  --data_path /home/aneek/LLM-Adapters/ft-training_set/math_14k.json \
  --output_dir ./trained_models/Qwen3_Sparse/Qwen3-8B-Sparse-0.20-Math-14k \
  --batch_size 4   --micro_batch_size 1   --num_epochs 3 \
  --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --wandb_run_name Qwen3-8B-Sparse-0.20-Math-14k

====================================================================================================================================

For Ensemble Data split the data into 20% / 20% / 20% / 20% / 20%  -- run the 100% / 50% rand / 50% active learning

part1/part2/part3/part4/part5

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 python  finetune.py   --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.20" \
  --data_path /home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part1_of_5.json \
  --output_dir ./trained_models/Qwen3_Sparse_Ensemble_Split20/Qwen3-8B-Sparse-0.20-Math14k-part1 \
  --batch_size 4   --micro_batch_size 1   --num_epochs 3 \
  --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --wandb_run_name Qwen3-8B-Sparse-0.20-Math14k-part1

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 python  finetune.py   --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.20" \
  --data_path /home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part2_of_5.json \
  --output_dir ./trained_models/Qwen3_Sparse_Ensemble_Split20/Qwen3-8B-Sparse-0.20-Math14k-part2 \
  --batch_size 4   --micro_batch_size 1   --num_epochs 3 \
  --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --wandb_run_name Qwen3-8B-Sparse-0.20-Math14k-part2

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 python  finetune.py   --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.20" \
  --data_path /home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part3_of_5.json \
  --output_dir ./trained_models/Qwen3_Sparse_Ensemble_Split20/Qwen3-8B-Sparse-0.20-Math14k-part3 \
  --batch_size 4   --micro_batch_size 1   --num_epochs 3 \
  --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --wandb_run_name Qwen3-8B-Sparse-0.20-Math14k-part3

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 python  finetune.py   --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.20" \
  --data_path /home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part4_of_5.json \
  --output_dir ./trained_models/Qwen3_Sparse_Ensemble_Split20/Qwen3-8B-Sparse-0.20-Math14k-part4 \
  --batch_size 4   --micro_batch_size 1   --num_epochs 3 \
  --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --wandb_run_name Qwen3-8B-Sparse-0.20-Math14k-part4

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1 python  finetune.py   --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.20" \
  --data_path /home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part5_of_5.json \
  --output_dir ./trained_models/Qwen3_Sparse_Ensemble_Split20/Qwen3-8B-Sparse-0.20-Math14k-part5 \
  --batch_size 4   --micro_batch_size 1   --num_epochs 3 \
  --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --wandb_run_name Qwen3-8B-Sparse-0.20-Math14k-part5

==========================================================================================================================

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3201 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.20" \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part1_of_5.json   \
    --output_dir ./trained_models/Qwen3_Sparse-Ensemble-Split-20/Qwen3-8B-Sparse-0.20-Math14k-part1-rand50 \
    --rounds 1 \
    --init_frac 0.5 \
    --acq_frac 0.1




WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3202 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.20" \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part2_of_5.json   \
    --output_dir ./trained_models/Qwen3_Sparse-Ensemble-Split-20/Qwen3-8B-Sparse-0.20-Math14k-part2-rand50 \
    --rounds 1 \
    --init_frac 0.5 \
    --acq_frac 0.1


WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3203 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.20" \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part3_of_5.json   \
    --output_dir ./trained_models/Qwen3_Sparse-Ensemble-Split-20/Qwen3-8B-Sparse-0.20-Math14k-part3-rand50 \
    --rounds 1 \
    --init_frac 0.5 \
    --acq_frac 0.1


WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3204 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.20" \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part4_of_5.json   \
    --output_dir ./trained_models/Qwen3_Sparse-Ensemble-Split-20/Qwen3-8B-Sparse-0.20-Math14k-part4-rand50 \
    --rounds 1 \
    --init_frac 0.5 \
    --acq_frac 0.1


WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3205 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.20" \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part5_of_5.json   \
    --output_dir ./trained_models/Qwen3_Sparse-Ensemble-Split-20/Qwen3-8B-Sparse-0.20-Math14k-part5-rand50 \
    --rounds 1 \
    --init_frac 0.5 \
    --acq_frac 0.1

=========================================================================================================================

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3201 active_learning_fast.py \
    --base_model "/home/models/nvidia-sparse/OpenReasoning-Nemotron-7B-Sparse-0.75" \
    --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part1_of_4.json"   \
    --output_dir ./trained_models/Nemotron-7B-Sparse/OpenReasoning-Nemotron-7B-Sparse-0.75-Ensemble_split_25/OpenReasoning-Nemotron-7B-Sparse-0.80-Math14k-part1-al50 \
    --rounds 3 \
    --init_frac 0.1 \
    --acq_frac 0.2 

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3201 active_learning_fast.py \
    --base_model "/home/models/nvidia-sparse/OpenReasoning-Nemotron-7B-Sparse-0.75" \
    --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part2_of_4.json"   \
    --output_dir ./trained_models/Nemotron-7B-Sparse/OpenReasoning-Nemotron-7B-Sparse-0.75-Ensemble_split_25/OpenReasoning-Nemotron-7B-Sparse-0.80-Math14k-part2-al50 \
    --rounds 3 \
    --init_frac 0.1 \
    --acq_frac 0.2 

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3201 active_learning_fast.py \
    --base_model "/home/models/nvidia-sparse/OpenReasoning-Nemotron-7B-Sparse-0.75" \
    --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part3_of_4.json"   \
    --output_dir ./trained_models/Nemotron-7B-Sparse/OpenReasoning-Nemotron-7B-Sparse-0.75-Ensemble_split_25/OpenReasoning-Nemotron-7B-Sparse-0.80-Math14k-part3-rand50 \
    --rounds 3 \
    --init_frac 0.1 \
    --acq_frac 0.2  

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3201 active_learning_fast.py \
    --base_model "/home/models/nvidia-sparse/OpenReasoning-Nemotron-7B-Sparse-0.75" \
    --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part4_of_4.json"   \
    --output_dir ./trained_models/Nemotron-7B-Sparse/OpenReasoning-Nemotron-7B-Sparse-0.75-Ensemble_split_25/OpenReasoning-Nemotron-7B-Sparse-0.80-Math14k-part4-rand50 \
    --rounds 3 \
    --init_frac 0.1 \
    --acq_frac 0.2 


WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=2 python  finetune.py   --base_model "/home/models/nvidia-sparse/OpenReasoning-Nemotron-7B-Sparse-0.75" \
  --data_path /home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part1_of_4.json \
  --output_dir ./trained_models/Nemotron-7B-Sparse/OpenReasoning-Nemotron-7B-Sparse-0.75-Ensemble_split_20/OpenReasoning-Nemotron-7B-Sparse-0.75-Math14k-part1 \
  --batch_size 4   --micro_batch_size 1   --num_epochs 3 \
  --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --wandb_run_name OpenReasoning-Nemotron-7B-Sparse-0.75-Math14k-part1



WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3202 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.20" \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part2_of_5.json   \
    --output_dir ./trained_models/Qwen3_Sparse-Ensemble-Split-20/Qwen3-8B-Sparse-0.20-Math14k-part2-al50 \
    --rounds 3 \
    --init_frac 0.1 \
    --acq_frac 0.2 



WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3203 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.20" \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part3_of_5.json   \
    --output_dir ./trained_models/Qwen3_Sparse-Ensemble-Split-20/Qwen3-8B-Sparse-0.20-Math14k-part3-al50 \
    --rounds 3 \
    --init_frac 0.1 \
    --acq_frac 0.2 


WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3204 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.20" \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part4_of_5.json   \
    --output_dir ./trained_models/Qwen3_Sparse-Ensemble-Split-20/Qwen3-8B-Sparse-0.20-Math14k-part4-al50 \
    --rounds 3 \
    --init_frac 0.1 \
    --acq_frac 0.2 


WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3205 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.20" \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part5_of_5.json   \
    --output_dir ./trained_models/Qwen3_Sparse-Ensemble-Split-20/Qwen3-8B-Sparse-0.20-Math14k-part5-al50 \
    --rounds 3 \
    --init_frac 0.1 \
    --acq_frac 0.2 


=========================================================================================================================

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3210 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.25" \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part1_of_4.json  \
    --output_dir ./trained_models/Qwen3_Sparse-Ensemble-Split-25/Qwen3-8B-Sparse-0.25-Math14k-part1-al50 \
    --rounds 3 \
    --init_frac 0.1 \
    --acq_frac 0.2 



WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3202 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.25" \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part2_of_4.json  \
    --output_dir ./trained_models/Qwen3_Sparse-Ensemble-Split-25/Qwen3-8B-Sparse-0.25-Math14k-part2-al50 \
    --rounds 3 \
    --init_frac 0.1 \
    --acq_frac 0.2 


WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3203 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.25" \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part3_of_4.json  \
    --output_dir ./trained_models/Qwen3_Sparse-Ensemble-Split-25/Qwen3-8B-Sparse-0.25-Math14k-part3-al50 \
    --rounds 3 \
    --init_frac 0.1 \
    --acq_frac 0.2 


WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3204 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.25" \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part4_of_4.json  \
    --output_dir ./trained_models/Qwen3_Sparse-Ensemble-Split-25/Qwen3-8B-Sparse-0.25-Math14k-part4-al50 \
    --rounds 3 \
    --init_frac 0.1 \
    --acq_frac 0.2 




------------------------------------------------------------------------------------------------------------


WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3201 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.25"  \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/math_14k.json \
    --output_dir ./trained_models/Qwen3_Sparse/Qwen3-8B-Sparse-0.25-Math14k-rand50\
    --rounds 1 \
    --init_frac 0.5 \
    --acq_frac 0.1

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3202 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.33" \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/math_14k.json \
    --output_dir ./trained_models/Qwen3_Sparse/Qwen3-8B-Sparse-0.33-Math14k-rand50\
    --rounds 1 \
    --init_frac 0.5 \
    --acq_frac 0.1 

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3203 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.50" \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/math_14k.json \
    --output_dir ./trained_models/Qwen3_Sparse/Qwen3-8B-Sparse-0.50-Math14k-rand50\
    --rounds 1 \
    --init_frac 0.5 \
    --acq_frac 0.1 \
    --wandb_run_name Qwen3-8B-Sparse-0.50-Math14k-rand50

=============================================================================================================


WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3209 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.20" \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/math_14k.json \
    --output_dir ./trained_models/Qwen3_Sparse/Qwen3-8B-Sparse-0.20-Math14k-al50\
    --rounds 3 \
    --init_frac 0.1 \
    --acq_frac 0.2 \
    --wandb_run_name Qwen3-8B-Sparse-0.50-Math14k-al50

    WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3200 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.25" \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/math_14k.json \
    --output_dir ./trained_models/Qwen3_Sparse/Qwen3-8B-Sparse-0.25-Math14k-al50\
    --rounds 3 \
    --init_frac 0.1 \
    --acq_frac 0.2 \
    --wandb_run_name Qwen3-8B-Sparse-0.50-Math14k-al50

    WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3203 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.33" \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/math_14k.json \
    --output_dir ./trained_models/Qwen3_Sparse/Qwen3-8B-Sparse-0.33-Math14k-al50\
    --rounds 3 \
    --init_frac 0.1 \
    --acq_frac 0.2 \
    --wandb_run_name Qwen3-8B-Sparse-0.50-Math14k-al50

WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,2 python active_learning.py     --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.33"     --data_path /home/aneek/LLM-Adapters/ft-training_set/math_14k.json     --output_dir ./trained_models/Qwen3_Sparse/Qwen3-8B-Sparse-0.33-Math14k-al50    --rounds 3     --init_frac 0.1     --acq_frac 0.2     --wandb_run_name Qwen3-8B-Sparse-0.33-Math14k-al50

Qwen3-8B-Sparse-0.20-Math14k-part1-al50
 CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29515 active_learning.py   --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.33"   --data_path /home/aneek/LLM-Adapters/ft-training_set/math_14k.json   --output_dir ./trained_models/Qwen3_Sparse/Qwen3-8B-Sparse-0.33-Math14k-al50   --rounds 3   --init_frac 0.1   --acq_frac 0.2   --wandb_run_name Qwen3-8B-Sparse-0.33-Math14k-al50

    WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3203 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.20" \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/math_14k.json \
    --output_dir ./trained_models/Qwen3_Sparse/Qwen3-8B-Sparse-0.20-Math14k-al50\
    --rounds 3 \
    --init_frac 0.1 \
    --acq_frac 0.2 \
    --wandb_run_name Qwen3-8B-Sparse-0.50-Math14k-al50

=============================================================================================================

For Ensemble Data split the data into 20% / 20% / 20% / 20% / 20%  -- run the 100% / 50% rand / 50% active learning
part1/part2/part3/part4/part5


For Ensemble Data split the data into 25% / 25% / 25%/ 25% -- run the 100% / 50% rand / 50% active learning
part1/part2/part3/part4


WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3201 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.25"  \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part1_of_4.json \
    --output_dir ./trained_models/Qwen3_Sparse-Ensemble-Split-25/Qwen3-8B-Sparse-0.25-Math14k-part1-rand50\
    --rounds 1 \
    --init_frac 0.5 \
    --acq_frac 0.1


WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3202 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.25"  \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part2_of_4.json \
    --output_dir ./trained_models/Qwen3_Sparse-Ensemble-Split-25/Qwen3-8B-Sparse-0.25-Math14k-part2-rand50\
    --rounds 1 \
    --init_frac 0.5 \
    --acq_frac 0.1

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3203 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.25"  \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part3_of_4.json \
    --output_dir ./trained_models/Qwen3_Sparse-Ensemble-Split-25/Qwen3-8B-Sparse-0.25-Math14k-part3-rand50\
    --rounds 1 \
    --init_frac 0.5 \
    --acq_frac 0.1


    WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3204 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.25"  \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part4_of_4.json \
    --output_dir ./trained_models/Qwen3_Sparse-Ensemble-Split-25/Qwen3-8B-Sparse-0.25-Math14k-part4-rand50\
    --rounds 1 \
    --init_frac 0.5 \
    --acq_frac 0.1
/home/aneek/LLM-Adapters/trained_models/Qwen3_Sparse-Ensemble-Split-20

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 python  finetune.py   --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.25" \
  --data_path /home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part1_of_4.json \
  --output_dir ./trained_models/Qwen3_Sparse_Ensemble_Split25/Qwen3-8B-Sparse-0.25-Math14k-part1 \
  --batch_size 4   --micro_batch_size 1   --num_epochs 3 \
  --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --wandb_run_name Qwen3-8B-Sparse-0.25-Math14k-part1

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 python  finetune.py   --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.25" \
  --data_path /home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part2_of_4.json \
  --output_dir ./trained_models/Qwen3_Sparse_Ensemble_Split25/Qwen3-8B-Sparse-0.25-Math14k-part2 \
  --batch_size 4   --micro_batch_size 1   --num_epochs 3 \
  --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --wandb_run_name Qwen3-8B-Sparse-0.25-Math14k-part2

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 python  finetune.py   --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.25" \
  --data_path /home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part3_of_4.json \
  --output_dir ./trained_models/Qwen3_Sparse_Ensemble_Split25/Qwen3-8B-Sparse-0.25-Math14k-part3 \
  --batch_size 4   --micro_batch_size 1   --num_epochs 3 \
  --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --wandb_run_name Qwen3-8B-Sparse-0.25-Math14k-part3

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 python  finetune.py   --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.25" \
  --data_path /home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part4_of_4.json \
  --output_dir ./trained_models/Qwen3_Sparse_Ensemble_Split25/Qwen3-8B-Sparse-0.25-Math14k-part4 \
  --batch_size 4   --micro_batch_size 1   --num_epochs 3 \
  --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --wandb_run_name Qwen3-8B-Sparse-0.25-Math14k-part4



For Ensemble Data split the data into 33% / 33% / 33%  -- run the 100% / 50% rand / 50% active learning

part1/part2/part3

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3301 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.33"  \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part1_of_3.json \
    --output_dir ./trained_models/Qwen3_Sparse-Ensemble-Split-33/Qwen3-8B-Sparse-0.33-Math14k-part1-rand50 \
    --rounds 1 \
    --init_frac 0.5 \
    --acq_frac 0.1


WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3302 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.33"  \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part2_of_3.json \
    --output_dir ./trained_models/Qwen3_Sparse-Ensemble-Split-33/Qwen3-8B-Sparse-0.33-Math14k-part2-rand50 \
    --rounds 1 \
    --init_frac 0.5 \
    --acq_frac 0.1

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3303 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.33"  \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part3_of_3.json \
    --output_dir ./trained_models/Qwen3_Sparse-Ensemble-Split-33/Qwen3-8B-Sparse-0.33-Math14k-part3-rand50 \
    --rounds 1 \
    --init_frac 0.5 \
    --acq_frac 0.1


part1/part2/part3



WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3301 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.33"  \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part1_of_3.json \
    --output_dir ./trained_models/Qwen3_Sparse-Ensemble-Split-33/Qwen3-8B-Sparse-0.33-Math14k-part1-al50 \
    --rounds 3 \
    --init_frac 0.1 \
    --acq_frac 0.2 \
    --wandb_run_name Qwen3-8B-Sparse-0.33-Math14k-part1-al50


WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3302 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.33"  \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part2_of_3.json \
    --output_dir ./trained_models/Qwen3_Sparse-Ensemble-Split-33/Qwen3-8B-Sparse-0.33-Math14k-part2-al50 \
    --rounds 3 \
    --init_frac 0.1 \
    --acq_frac 0.2 \
    --wandb_run_name Qwen3-8B-Sparse-0.33-Math14k-part2-al50

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3303 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.33"  \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part3_of_3.json \
    --output_dir ./trained_models/Qwen3_Sparse-Ensemble-Split-33/Qwen3-8B-Sparse-0.33-Math14k-part3-al50 \
    --rounds 3 \
    --init_frac 0.1 \
    --acq_frac 0.2 \
    --wandb_run_name Qwen3-8B-Sparse-0.33-Math14k-part3-al50

=============================================================================================================


WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3401 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.50"  \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/split_50/math_14k_part1_of_2.json\
    --output_dir ./trained_models/Qwen3_Sparse-Ensemble-Split-50/Qwen3-8B-Sparse-0.50-Math14k-part1-rand50 \
    --rounds 3 \
    --init_frac 0.1 \
    --acq_frac 0.2 \
    --wandb_run_name Qwen3-8B-Sparse-0.50-Math14k-part1-al50


WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3402 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.50"  \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/split_50/math_14k_part2_of_2.json \
    --output_dir ./trained_models/Qwen3_Sparse-Ensemble-Split-50/Qwen3-8B-Sparse-0.50-Math14k-part2-rand50 \
    --rounds 3 \
    --init_frac 0.1 \
    --acq_frac 0.2 \
    --wandb_run_name Qwen3-8B-Sparse-0.50-Math14k-part2-al50






WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1 python  finetune.py   --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.33" \
  --data_path /home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part1_of_3.json \
  --output_dir ./trained_models/Qwen3_Sparse_Ensemble_Split33/Qwen3-8B-Sparse-0.33-Math14k-part1 \
  --batch_size 4   --micro_batch_size 1   --num_epochs 3 \
  --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --wandb_run_name Qwen3-8B-Sparse-0.33-Math14k-part1

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1 python  finetune.py   --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.33" \
  --data_path /home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part2_of_3.json \
  --output_dir ./trained_models/Qwen3_Sparse_Ensemble_Split33/Qwen3-8B-Sparse-0.33-Math14k-part2 \
  --batch_size 4   --micro_batch_size 1   --num_epochs 3 \
  --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --wandb_run_name Qwen3-8B-Sparse-0.33-Math14k-part2

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1 python  finetune.py   --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.33" \
  --data_path /home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part3_of_3.json \
  --output_dir ./trained_models/Qwen3_Sparse_Ensemble_Split33/Qwen3-8B-Sparse-0.33-Math14k-part3 \
  --batch_size 4   --micro_batch_size 1   --num_epochs 3 \
  --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --wandb_run_name Qwen3-8B-Sparse-0.33-Math14k-part3

For Ensemble Data split the data into 50% / 50%   -- run the 100% / 50% rand / 50% active learning

part1/part2


WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3401 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.50"  \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/split_50/math_14k_part1_of_2.json\
    --output_dir ./trained_models/Qwen3_Sparse-Ensemble-Split-50/Qwen3-8B-Sparse-0.50-Math14k-part1-rand50 \
    --rounds 1 \
    --init_frac 0.5 \
    --acq_frac 0.1


WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3402 active_learning.py \
    --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.50"  \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/split_50/math_14k_part2_of_2.json \
    --output_dir ./trained_models/Qwen3_Sparse-Ensemble-Split-50/Qwen3-8B-Sparse-0.50-Math14k-part2-rand50 \
    --rounds 1 \
    --init_frac 0.5 \
    --acq_frac 0.1



WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1 python  finetune.py   --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.50" \
  --data_path /home/aneek/LLM-Adapters/ft-training_set/split_50/math_14k_part1_of_2.json \
  --output_dir ./trained_models/Qwen3_Sparse_Ensemble_Split50/Qwen3-8B-Sparse-0.50-Math14k-part1 \
  --batch_size 4   --micro_batch_size 1   --num_epochs 3 \
  --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --wandb_run_name Qwen3-8B-Sparse-0.50-Math14k-part1

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=1 python  finetune.py   --base_model "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.50" \
  --data_path /home/aneek/LLM-Adapters/ft-training_set/split_50/math_14k_part2_of_2.json \
  --output_dir ./trained_models/Qwen3_Sparse_Ensemble_Split50/Qwen3-8B-Sparse-0.50-Math14k-part2 \
  --batch_size 4   --micro_batch_size 1   --num_epochs 3 \
  --learning_rate 3e-5   --cutoff_len 256   --val_set_size 120 --wandb_run_name Qwen3-8B-Sparse-0.50-Math14k-part2


=============================================================================================================

WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 \
"$CONDA_PREFIX/bin/torchrun" --nproc_per_node=1 --master_port=3200 active_learning.py \
    --base_model /home/models/Qwen/Qwen3-8B/ \
    --data_path /home/aneek/LLM-Adapters/ft-training_set/math_14k.json \
    --output_dir ./trained_models/Qwen3/Qwen3-8B-Math14k-rand50\
    --rounds 1 \
    --init_frac 0.5 \
    --acq_frac 0.1



  "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.20"
  "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.25"
  "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.33"
  "/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.50"

  
['AddSub', 'MultiArith', 'SingleEq', 'gsm8k', 'AQuA', 'SVAMP']
  CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python /home/aneek/LLM-Adapters/eval_llm.py \
  --dataset gsm8k \
  --model Llama-3.2-3B-Instruct-0.33-Wanda \
  --base_model /home/models/Llama-3.2-3B-Instruct-0.33-Wanda \
  --adapter LoRA \
  --lora_weights /home/aneek/LLM-Adapters/trained_models/instruct_3B/llama-math-14k-3B-Wanda-lora-ep3 \
  --batch_size_gen 16 \
  --tp_size 1 \
  --max_model_len 8192 \
  --gpu_mem_util 0.97

AddSub,MultiArith,SingleEq,gsm8k,AQuA,SVAMP

   CUDA_VISIBLE_DEVICES=1,2 PYTHONNOUSERSITE=1 "$CONDA_PREFIX/bin/python" /home/aneek/LLM-Adapters/eval_llm.py   --dataset AddSub   --model Llama-4-Scout-17B-16E-Instruct   --base_model /home/models/Llama-4-Scout-17B-16E-Instruct   --batch_size_gen 8   --tp_size 1   --max_model_len 4096  



  CUDA_VISIBLE_DEVICES=2,0 PYTHONNOUSERSITE=1 python /home/aneek/LLM-Adapters/eval_llm.py  \
  --dataset AddSub \
  --model Llama-3.2-11B-Vision-Instruct \
  --base_model /home/models/Llama-3.2-11B-Vision-Instruct \
  --batch_size_gen 4 \
  --tp_size 2 \
  --max_model_len 8192 \
  --gpu_mem_util 0.95 \ 

AddSub,MultiArith,SingleEq,gsm8k,AQuA,SVAMP

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=1 \
PYTHONNOUSERSITE=1 \
python /home/aneek/LLM-Adapters/eval_llm.py \
  --dataset SVAMP \
  --model Llama-3.2-3B-Instruct-0.33-Wanda \
  --base_model /home/models/Llama-3.2-3B-Instruct-0.33-Wanda/ \
  --batch_size_gen 16 \
  --tp_size 1 \
  --max_model_len 2048 \
  --gpu_mem_util 0.95

/home/models/Llama-3.1-8B-Instruct-0.33-Wanda

AddSub,MultiArith,SingleEq,gsm8k,AQuA,SVAMP

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=2 \
PYTHONNOUSERSITE=1 \
python /home/aneek/LLM-Adapters/eval_llm.py \
  --dataset SVAMP \
  --model Llama-3.1-8B-Instruct-0.33-Wanda \
  --base_model /home/models/Llama-3.1-8B-Instruct-0.33-Wanda \
  --batch_size_gen 16 \
  --tp_size 1 \
  --max_model_len 2048 \
  --gpu_mem_util 0.95


===========================================================================


/home/aneek/LLM-Adapters/trained_models/al-math14k-3B-heur50/round_2
CUDA_VISIBLE_DEVICES=2 \
PYTHONNOUSERSITE=1 \
python /home/aneek/LLM-Adapters/eval_llm.py \
  --dataset SVAMP \
  --model Llama-3.2-3B-Instruct \
  --base_model /home/models/Llama-3.2-3B-Instruct \
  --batch_size_gen 16 \
  --lora_weights '/home/aneek/LLM-Adapters/trained_models/al-math14k-3B-heur50/round_2'
  --tp_size 1 \
  --max_model_len 2048 \
  --gpu_mem_util 0.95



/home/aneek/LLM-Adapters/trained_models/al-math14k-3B-sparse-heur50/round_2

CUDA_VISIBLE_DEVICES=1 \
PYTHONNOUSERSITE=1 \
python /home/aneek/LLM-Adapters/eval_llm.py \
  --dataset SVAMP \
  --model Llama-3.2-3B-Instruct-Sparse \
  --base_model /home/models/Llama-3.2-3B-Instruct-Sparse-0.33 \
  --batch_size_gen 16 \
  --lora_weights '/home/aneek/LLM-Adapters/trained_models/al-math14k-3B-heur50/round_2'
  --tp_size 1 \
  --max_model_len 2048 \
  --gpu_mem_util 0.95




accelerate launch --config_file accelerate_gpus_0_2.yaml eval_llm.py \
    --dataset AddSub \
  --model Llama-3.1-8B-Instruct \
  --base_model /home/models/Llama-3.1-8B-Instruct \
  --batch_size_gen 16 \
  --tp_size 1 \
  --max_model_len 8192 \
  --gpu_mem_util 0.95 \
  --max_model_len 2048


AddSub,MultiArith,SingleEq,gsm8k,AQuA,SVAMP

CUDA_VISIBLE_DEVICES=1 python /home/aneek/LLM-Adapters/eval_vllm_ensemble_math.py\
  --dataset SVAMP \
  --model Llama-3.2-3B-Instruct-Sparse \
  --base_model /home/models/Llama-3.2-3B-Instruct-Sparse-0.33 \
  --lora_weights "part1=/home/aneek/LLM-Adapters/trained_models/instruct_al_3B_ens50/ensA_part1/round_2,part2=/home/aneek/LLM-Adapters/trained_models/instruct_al_3B_ens50/ensB_part2/round_2,part3=/home/aneek/LLM-Adapters/trained_models/instruct_al_3B_ens50/ensC_part3/round_2" \
  --batch_size 16 \
  --tp_size 1 \
  --gpu_memory_utilization 0.95 \
  --max_loras 3 --max_cpu_loras 3 --max_lora_rank 32 \
  --preload --preload_gpu_keep 2

['AddSub', 'MultiArith', 'SingleEq', 'gsm8k', 'AQuA', 'SVAMP']
  CUDA_VISIBLE_DEVICES=0 python /home/aneek/LLM-Adapters/eval_vllm_ensemble_math.py  --dataset SVAMP   --model Llama-3.2-1B-Instruct   --base_model /home/models/Llama-3.2-1B-Instruct/   --lora_weights "part1=/home/aneek/LLM-Adapters/trained_models/instruct_new/llama-math-14k-part1-1B-lora,part2=/home/aneek/LLM-Adapters/trained_models/instruct_new/llama-math-14k-part2-1B-lora,part3=/home/aneek/LLM-Adapters/trained_models/instruct_new/llama-math-14k-part3-1B-lora"   --batch_size 16   --tp_size 1   --gpu_memory_utilization 0.95   --max_loras 3 --max_cpu_loras 3 --max_lora_rank 32 


CUDA_VISIBLE_DEVICES=1 \
python /home/aneek/LLM-Adapters/eval_vllm_ensemble.py \
  --dataset gsm8k \
  --models /home/models/Llama-3.2-1B-Instruct,/home/models/Llama-3.2-3B-Instruct,/home/models/Llama-3.2-1B-Instruct \
  --batch_size 16 \
  --tp_size 1 \
  --gpu_memory_utilization 0.95 \
  --ensemble_rule vote

  CUDA_VISIBLE_DEVICES=1 python /home/aneek/LLM-Adapters/eval_vllm_ensemble.py   --dataset SVAMP   --models /home/models/Llama-3.2-3B-Instruct-Sparse-0.33,/home/models/Llama-3.2-3B-Instruct-Sparse-0.33,/home/models/Llama-3.2-3B-Instruct-Sparse-0.33   --batch_size 16   --tp_size 1   --gpu_memory_utilization 0.95   --ensemble_rule vote

['AddSub', 'MultiArith', 'SingleEq', 'gsm8k', 'AQuA', 'SVAMP']

CUDA_VISIBLE_DEVICES=2 \
python radial_router_3models.py \
  --dataset gsm8k \
  --models /home/models/Llama-3.2-1B-Instruct,/home/models/Llama-3.2-1B-Instruct,/home/models/Llama-3.2-1B-Instruct \
  --alpha 0.02 \
  --batch_size 16 --tp_size 1 --gpu_memory_utilization 0.95 \
  --router_epochs 30 --router_batch 64 --contrastive_lambda 0.5

CUDA_VISIBLE_DEVICES=0,2 python eval_text_baseline.py \
  --dataset AddSub \
  --model Llama-3.2-11B-Vision-Instruct \
  --base_model /home/models/Llama-3.2-11B-TextOnly \
  --tp_size 2 \
  --batch_size_gen 4 \
  --gpu_mem_util 0.9 \
  --max_model_len 8192 \
  --dtype auto

python -c 'from transformers import AutoTokenizer, AutoModelForCausalLM; m=AutoModelForCausalLM.from_pretrained("/home/models/Llama-3.2-1B-Instruct", torch_dtype="auto").cuda(); t=AutoTokenizer.from_pretrained("/home/models/Llama-3.2-1B-Instruct"); i=t("Q: What is 4 + 5?\nA:", return_tensors="pt").to("cuda"); print(t.decode(m.generate(**i, max_new_tokens=10, pad_token_id=t.eos_token_id)[0]))'


1. Run the ensemble code without any pre-training difference.
2. Check for different ensemble.
3. Run Wanda / PruneNet 


CUDA_VISIBLE_DEVICES=2 python convert_mllama_to_llama_text_gpu.py   --src /home/models/Llama-3.2-11B-Vision-Instruct   --dst /home/models/Llama-3.2-11B-TextOnly   --dtype float16   --device cuda:0


(sparsegpt) aneek@LCS2-IITD:~/LLM-Adapters$ python split_math14k.py \                                                                                   
  --src /home/aneek/LLM-Adapters/ft-training_set/math_14k.json \                                                                                        
  --outdir /home/aneek/LLM-Adapters/ft-training_set/split_20/ \                                                                                         
  --parts 5 \                                                                                                                                           
  --seed 42                                                                                                                                             
[✓] wrote   2785 examples → /home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part1_of_5.json                                                  
[✓] wrote   2784 examples → /home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part2_of_5.json                                                  
[✓] wrote   2784 examples → /home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part3_of_5.json                                                  
[✓] wrote   2784 examples → /home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part4_of_5.json                                                  
[✓] wrote   2784 examples → /home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part5_of_5.json                                                  
(sparsegpt) aneek@LCS2-IITD:~/LLM-Adapters$ python split_math14k.py \                                                                                   
  --src /home/aneek/LLM-Adapters/ft-training_set/math_14k.json \                                                                                        
  --outdir /home/aneek/LLM-Adapters/ft-training_set/split_25/ \                                                                                         
  --parts 4 \                                                                                                                                           
  --seed 42                                                                                                                                             
[✓] wrote   3481 examples → /home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part1_of_4.json                                                  
[✓] wrote   3480 examples → /home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part2_of_4.json                                                  
[✓] wrote   3480 examples → /home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part3_of_4.json                                                  
[✓] wrote   3480 examples → /home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part4_of_4.json                                                  
(sparsegpt) aneek@LCS2-IITD:~/LLM-Adapters$ python split_math14k.py \                                                                                   
  --src /home/aneek/LLM-Adapters/ft-training_set/math_14k.json \                
  --outdir /home/aneek/LLM-Adapters/ft-training_set/split_33/ \                                                                                         
  --parts 3 \
  --seed 42
[✓] wrote   4641 examples → /home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part1_of_3.json                                                  
[✓] wrote   4640 examples → /home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part2_of_3.json                                                  
[✓] wrote   4640 examples → /home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part3_of_3.json                                                  
(sparsegpt) aneek@LCS2-IITD:~/LLM-Adapters$ python split_math14k.py \                                                                                   
  --src /home/aneek/LLM-Adapters/ft-training_set/math_14k.json \                                                                                        
  --outdir /home/aneek/LLM-Adapters/ft-training_set/split_50/ \                                                                                         
  --parts 2 \
  --seed 42
[✓] wrote   6961 examples → /home/aneek/LLM-Adapters/ft-training_set/split_50/math_14k_part1_of_2.json                                                  
[✓] wrote   6960 examples → /home/aneek/LLM-Adapters/ft-training_set/split_50/math_14k_part2_of_2.json   
````



CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python active_learning.py \
  --base_model "/home/models/Qwen/Qwen3-4B-Instruct-2507" \
  --data_path /home/aneek/LLM-Adapters/ft-training_set/math_14k.json \
  --output_dir ./trained_models/Qwen3_Ensemble/Qwen3-4B-2507-Math14k-al50 \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2 \
  --uncertainty logppl --cutoff_len 256 --scoring_batch_size 8 \
  --num_epochs 3 --learning_rate 3e-5 \
  --per_device_train_batch_size 4 --micro_batch_size 1 --val_set_size 120


CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python active_learning.py \
  --base_model "/home/models/Qwen/Qwen3-4B-Instruct-2507" \
  --data_path /home/aneek/LLM-Adapters/ft-training_set/math_14k.json \
  --output_dir ./trained_models/Qwen3_Ensemble/Qwen3-4B-2507-Math14k-rand50 \
  --rounds 1 --init_frac 0.5 --acq_frac 0.1 \
  --uncertainty logppl --cutoff_len 256 --scoring_batch_size 8 \
  --num_epochs 3 --learning_rate 3e-5 \
  --per_device_train_batch_size 4 --micro_batch_size 1 --val_set_size 120


PART=N  # e.g., 3
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python active_learning.py \
  --base_model "/home/models/Qwen/Qwen3-4B-Instruct-2507" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part${PART}_of_4.json" \
  --output_dir "./trained_models/Qwen3_Ensemble_Split25/Qwen3-4B-2507-Math14k-part${PART}-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2 \
  --uncertainty logppl --cutoff_len 256 --scoring_batch_size 8 \
  --num_epochs 3 --learning_rate 3e-5 \
  --per_device_train_batch_size 4 --micro_batch_size 1 --val_set_size 120

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \ 
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python /home/aneek/LLM-Adapters/active_learning_nop_bbl.py --base_model /home/models/nvidia/OpenReasoning-Nemotron-14B --data_path /home/aneek/LLM-Adapters/ft-training_set/math_14k.json --output_dir /home/aneek/LLM-Adapters/trained_models/Nemotron//OpenReasoning-Nemotron-14B-Math14k/OpenReasoning-Nemotron-14B-al50 --rounds 3 --init_frac 0.1 --acq_frac 0.2 --uncertainty logppl --cutoff_len 256 --scoring_batch_size 8 --num_epochs 3 --learning_rate 3e-5 --per_device_train_batch_size 4 --micro_batch_size 1 --val_set_size 120 --wandb_run_name AL-al50-OpenReasoning-Nemotron-14B-Math14k



CUDA_VISIBLE_DEVICES=1 python evaluate.py  --model 'Llama-3.2-1B-Instruct' --adapter LoRA --dataset SVAMP --base_model '/home2/models/Llama-3.2-1B-Instruct/' --lora_weights '/home2/palash/aneek/LLM-Adapters/trained_models/instruction_new_14k/al-math14k-1B-rand50_10/round_0'



CUDA_VISIBLE_DEVICES=1 python /home/aneek/LLM-Adapters/new_eval_llm.py \
  --dataset gsm8k \
  --model Qwen3-4B-Instruct \
  --base_model "/home/models/Qwen/Qwen3-4B-Instruct-2507" \
  --adapter LoRA \
  --lora_weights "/home/aneek/LLM-Adapters/trained_models/Qwen3/Qwen3-4B-Math-14k" \
  --merged_out "/home/aneek/merged/Qwen3-4B-Math14k-merged" \
  --num_beams 4 --temperature 0.0 --top_p 1.0 --top_k 0 \
  --batch_size_gen 16 --tp_size 1 --max_model_len 8192 --gpu_mem_util 0.95 \
  --seed 42 \
  --wandb_project llm_adapter_eval_math_14k_baseline


  CUDA_VISIBLE_DEVICES=1 python /home/aneek/LLM-Adapters/eval_llm.py \
  --dataset gsm8k \
  --model Qwen3-4B-Instruct \
  --base_model "/home/models/Qwen/Qwen3-4B-Instruct-2507" \
  --adapter LoRA \
  --lora_weights "/home/aneek/LLM-Adapters/trained_models/Qwen3/Qwen3-4B-Math-14k" \
  --merged_out "/home/aneek/merged/Qwen3-4B-Math14k-merged" \
  --num_beams 4 --temperature 0.0 --top_p 1.0 --top_k 0 \
  --batch_size_gen 16 --tp_size 1 --max_model_len 8192 --gpu_mem_util 0.95 \
  --seed 42



  CUDA_VISIBLE_DEVICES=2 python /home/aneek/LLM-Adapters/eval_vllm_ensemble_math_b.py\
  --dataset gsm8k \
  --model Qwen3-4B-Instruct \
  --base_model "/home/models/Qwen/Qwen3-4B-Instruct-2507" \
  --lora_weights "part1=/home/aneek/LLM-Adapters/trained_models/Qwen3/Qwen3-4B-Math-14k" \
  --batch_size 16 \
  --tp_size 1 \
  --gpu_memory_utilization 0.95 \
  --max_loras 3 --max_cpu_loras 3 --max_lora_rank 32 



  1. Visualisation -- 
  Math (6 tasks) 
  subplot 6 columns * 3 rows plots for 

  Ensemble line plots -- straight line for baseline for the rest *** 

  Delta in line plots
  Bar-plots 
  XG-Boost implementation

  2. Pseudo Code

  3. github 

  4. effective data utilisation for each data training strategy

  
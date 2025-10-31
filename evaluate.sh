#!/bin/bash

BASE=/home/models/Llama-3.2-1B-Instruct
SCRIPT=/home/aneek/LLM-Adapters/ensemble-math-lora.py
LORAS="/home/aneek/LLM-Adapters/trained_models/instruct/llama-alpaca_data_cleaned-1B-lora,\
/home/aneek/LLM-Adapters/trained_models/instruct/llama-commonsense_15k-1B-lora,\
/home/aneek/LLM-Adapters/trained_models/instruct/llama-math_50-1B-lora"

# LORAS="/home/aneek/LLM-Adapters/trained_models/llama-commonsense_170k-1B-lora,\
# /home/aneek/LLM-Adapters/trained_models/llama-math_50k-1B-lora,\
# /home/aneek/LLM-Adapters/trained_models/llama-alpaca_data_cleaned-1B-lora"

# Map each dataset to a specific GPU
declare -A DS_GPU=(
  [gsm8k]=0
  [AddSub]=1
  [MultiArith]=0
  [SingleEq]=1
  [SVAMP]=0
  [AQuA]=1
)

for DS in "${!DS_GPU[@]}"; do
  GPU=${DS_GPU[$DS]}
  echo "▶ Launching $DS on GPU $GPU"
  CUDA_VISIBLE_DEVICES=$GPU python "$SCRIPT" \
      --dataset "$DS" \
      --model Llama-3.2-1B-Instruct \
      --base_model "$BASE" \
      --lora_weights "$LORAS" \
      --batch_size 16 \
      --ensemble_rule vote \
      --load_8bit &   # backgrounded
done

wait  # Block until all backgrounded evaluations complete
echo "✅ All evaluations finished."
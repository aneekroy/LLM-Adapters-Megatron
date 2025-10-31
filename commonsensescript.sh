datasets=(piqa social_i_qa hellaswag winogrande ARC-Challenge ARC-Easy openbookqa)

for ds in "${datasets[@]}"; do
  CUDA_VISIBLE_DEVICES=1 python commonsense_evaluate.py \
    --model 'Llama-3.2-3B-Instruct' \
    --adapter LoRA \
    --dataset openbookqa \
    --base_model '/home2/models/Llama-3.2-3B-Instruct-Sparse-0.33/' \
    --tokenizer_path '/home2/models/Llama-3.2-3B-Instruct-Sparse-0.33/' \
    --adapter_weights '/home2/palash/aneek/LLM-Adapters/trained_models/instruct_sparse/llama-commonsense_170k-3B-sparse-lora' \
    --batch_size 16 \
    --report_to wandb \
    --wandb_project 'llama3.2-3B-commonsense170k_eval-Sparse0.33' \
    --wandb_run_name openbookqa \
    # --cutoff_len 2048           # ← keep or tweak as needed
done


datasets=(winogrande ARC-Challenge)

for ds in "${datasets[@]}"; do
  CUDA_VISIBLE_DEVICES=2 python commonsense_evaluate.py \
    --model 'Llama-3.2-3B-Instruct' \
    --adapter LoRA \
    --dataset "$ds" \
    --base_model '/home2/models/Llama-3.2-3B-Instruct-Sparse-0.33/' \
    --tokenizer_path '/home2/models/Llama-3.2-3B-Instruct-Sparse-0.33/' \
    --adapter_weights '/home2/palash/aneek/LLM-Adapters/trained_models/instruct_sparse/llama-commonsense_170k-3B-sparse-lora' \
    --batch_size 16 \
    --report_to wandb \
    --wandb_project 'llama3.2-3B-commonsense170k_eval-Sparse0.33' \
    --wandb_run_name "$ds" \
    # --cutoff_len 2048           # ← keep or tweak as needed
done


datasets=(ARC-Easy openbookqa)

for ds in "${datasets[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python commonsense_evaluate.py \
    --model 'Llama-3.2-3B-Instruct' \
    --adapter LoRA \
    --dataset "$ds" \
    --base_model '/home2/models/Llama-3.2-3B-Instruct-Sparse-0.33/' \
    --tokenizer_path '/home2/models/Llama-3.2-3B-Instruct-Sparse-0.33/' \
    --adapter_weights '/home2/palash/aneek/LLM-Adapters/trained_models/instruct_sparse/llama-commonsense_170k-3B-sparse-lora' \
    --batch_size 16 \
    --report_to wandb \
    --wandb_project 'llama3.2-3B-commonsense170k_eval-Sparse0.33' \
    --wandb_run_name "$ds" \
    # --cutoff_len 2048           # ← keep or tweak as needed
done
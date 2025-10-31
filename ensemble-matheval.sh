# Common
BASE=/home/models/Llama-3.2-1B-Instruct
DATADIR=/home/aneek/LLM-Adapters/ft-training_set
OUT=/home/aneek/LLM-Adapters/trained_models/instruct_al_1B_ens50
ROUNDS=3
INIT=0.10
ACQ=0.03335

# Part 1  (≈ 10% seed + 3.335% + 3.335% ≈ 16.67%)
CUDA_VISIBLE_DEVICES=0 python active_learning.py \
  --base_model /home/models/Llama-3.2-1B-Instruct \
  --data_path /home/aneek/LLM-Adapters/ft-training_set/math_14k_part1.json \
  --output_dir /home/aneek/LLM-Adapters/trained_models/instruct_al_1B_ens50/ensA_part1 \
  --rounds 3 --init_frac 0.10 --acq_frac 0.03335 --cutoff_len 256 \
  --val_set_size 0.1 \
  --learning_rate 2e-4 --micro_batch_size 4 --gradient_accumulation_steps 8 

# Part 2
CUDA_VISIBLE_DEVICES=0 python active_learning.py \
  --base_model /home/models/Llama-3.2-1B-Instruct \
  --data_path /home/aneek/LLM-Adapters/ft-training_set/math_14k_part2.json \
  --output_dir /home/aneek/LLM-Adapters/trained_models/instruct_al_1B_ens50/ensB_part2 \
  --rounds 3 --init_frac 0.10 --acq_frac 0.03335 --cutoff_len 256 \
  --val_set_size 0.1 \
  --learning_rate 2e-4 --micro_batch_size 4 --gradient_accumulation_steps 8 \
  --seed 42

# Part 3
CUDA_VISIBLE_DEVICES=0 python active_learning.py \
  --base_model /home/models/Llama-3.2-1B-Instruct \
  --data_path /home/aneek/LLM-Adapters/ft-training_set/math_14k_part3.json \
  --output_dir /home/aneek/LLM-Adapters/trained_models/instruct_al_1B_ens50/ensC_part3 \
  --rounds 3 --init_frac 0.10 --acq_frac 0.03335 --cutoff_len 256 \
  --val_set_size 0.1 \
  --learning_rate 2e-4 --micro_batch_size 4 --gradient_accumulation_steps 8 \
  --seed 43


  1. compare vs SparseGPT/PruneNet/WANDA both without and with finetuning 
  2. 6 Datasets performance.
  3. Larger Llama-3.2-70B drop Llama - 4

  Wednesday 9pm 
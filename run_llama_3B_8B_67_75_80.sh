
################################################################################
# Llama-3.1-8B-Instruct-Sparse-0.67
################################################################################
export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=2
export PYTHONNOUSERSITE=1
TORCHRUN="$CONDA_PREFIX/bin/torchrun"
SCRIPT="active_learning_fast.py"

BASE="/home/models/llama-sparse/Llama-3.1-8B-Instruct-Sparse-0.67"
TOK="$BASE"

# ===== Active Learning =====
$TORCHRUN --nproc_per_node=1 --master_port=3601 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part1_of_3.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.67-Ensemble_split_33/Llama-3.1-8B-Instruct-Sparse-0.67-Math14k-part1-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3601 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part2_of_3.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.67-Ensemble_split_33/Llama-3.1-8B-Instruct-Sparse-0.67-Math14k-part2-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3601 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part3_of_3.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.67-Ensemble_split_33/Llama-3.1-8B-Instruct-Sparse-0.67-Math14k-part3-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

# ===== Random 50% baseline =====
$TORCHRUN --nproc_per_node=1 --master_port=3601 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part1_of_3.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.67-Ensemble_split_33/Llama-3.1-8B-Instruct-Sparse-0.67-Math14k-part1-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3601 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part2_of_3.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.67-Ensemble_split_33/Llama-3.1-8B-Instruct-Sparse-0.67-Math14k-part2-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3601 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part3_of_3.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.67-Ensemble_split_33/Llama-3.1-8B-Instruct-Sparse-0.67-Math14k-part3-rand50" \
  --rounds 1 --init_frac 0.5

# ===== ********************************************************** =====
export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=2
export PYTHONNOUSERSITE=1
TORCHRUN="$CONDA_PREFIX/bin/torchrun"
SCRIPT="finetune_fast.py"

BASE="/home/models/llama-sparse/Llama-3.1-8B-Instruct-Sparse-0.67"
TOK="$BASE"

$TORCHRUN --nproc_per_node=1 --master_port=3611 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part1_of_3.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.67-Ensemble_split_33/Llama-3.1-8B-Instruct-Sparse-0.67-Math14k-part1-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3611 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part2_of_3.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.67-Ensemble_split_33/Llama-3.1-8B-Instruct-Sparse-0.67-Math14k-part2-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3611 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part3_of_3.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.67-Ensemble_split_33/Llama-3.1-8B-Instruct-Sparse-0.67-Math14k-part3-FT"


################################################################################
# Llama-3.1-8B-Instruct-Sparse-0.75
################################################################################
export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=2
export PYTHONNOUSERSITE=1
TORCHRUN="$CONDA_PREFIX/bin/torchrun"
SCRIPT="active_learning_fast.py"

BASE="/home/models/llama-sparse/Llama-3.1-8B-Instruct-Sparse-0.75"
TOK="$BASE"

# ===== Active Learning =====
$TORCHRUN --nproc_per_node=1 --master_port=3602 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part1_of_4.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.75-Ensemble_split_25/Llama-3.1-8B-Instruct-Sparse-0.75-Math14k-part1-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3602 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part2_of_4.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.75-Ensemble_split_25/Llama-3.1-8B-Instruct-Sparse-0.75-Math14k-part2-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3602 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part3_of_4.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.75-Ensemble_split_25/Llama-3.1-8B-Instruct-Sparse-0.75-Math14k-part3-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3602 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part4_of_4.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.75-Ensemble_split_25/Llama-3.1-8B-Instruct-Sparse-0.75-Math14k-part4-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

# ===== Random 50% baseline =====
$TORCHRUN --nproc_per_node=1 --master_port=3602 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part1_of_4.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.75-Ensemble_split_25/Llama-3.1-8B-Instruct-Sparse-0.75-Math14k-part1-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3602 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part2_of_4.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.75-Ensemble_split_25/Llama-3.1-8B-Instruct-Sparse-0.75-Math14k-part2-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3602 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part3_of_4.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.75-Ensemble_split_25/Llama-3.1-8B-Instruct-Sparse-0.75-Math14k-part3-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3602 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part4_of_4.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.75-Ensemble_split_25/Llama-3.1-8B-Instruct-Sparse-0.75-Math14k-part4-rand50" \
  --rounds 1 --init_frac 0.5

# ===== ********************************************************** =====
export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=2
export PYTHONNOUSERSITE=1
TORCHRUN="$CONDA_PREFIX/bin/torchrun"
SCRIPT="finetune_fast.py"

BASE="/home/models/llama-sparse/Llama-3.1-8B-Instruct-Sparse-0.75"
TOK="$BASE"

$TORCHRUN --nproc_per_node=1 --master_port=3612 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part1_of_4.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.75-Ensemble_split_25/Llama-3.1-8B-Instruct-Sparse-0.75-Math14k-part1-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3612 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part2_of_4.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.75-Ensemble_split_25/Llama-3.1-8B-Instruct-Sparse-0.75-Math14k-part2-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3612 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part3_of_4.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.75-Ensemble_split_25/Llama-3.1-8B-Instruct-Sparse-0.75-Math14k-part3-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3612 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part4_of_4.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.75-Ensemble_split_25/Llama-3.1-8B-Instruct-Sparse-0.75-Math14k-part4-FT"


################################################################################
# Llama-3.1-8B-Instruct-Sparse-0.80
################################################################################
export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=2
export PYTHONNOUSERSITE=1
TORCHRUN="$CONDA_PREFIX/bin/torchrun"
SCRIPT="active_learning_fast.py"

BASE="/home/models/llama-sparse/Llama-3.1-8B-Instruct-Sparse-0.80"
TOK="$BASE"

# ===== Active Learning =====
$TORCHRUN --nproc_per_node=1 --master_port=3603 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part1_of_5.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.80-Ensemble_split_20/Llama-3.1-8B-Instruct-Sparse-0.80-Math14k-part1-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3603 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part2_of_5.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.80-Ensemble_split_20/Llama-3.1-8B-Instruct-Sparse-0.80-Math14k-part2-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3603 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part3_of_5.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.80-Ensemble_split_20/Llama-3.1-8B-Instruct-Sparse-0.80-Math14k-part3-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3603 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part4_of_5.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.80-Ensemble_split_20/Llama-3.1-8B-Instruct-Sparse-0.80-Math14k-part4-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3603 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part5_of_5.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.80-Ensemble_split_20/Llama-3.1-8B-Instruct-Sparse-0.80-Math14k-part5-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

# ===== Random 50% baseline =====
$TORCHRUN --nproc_per_node=1 --master_port=3603 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part1_of_5.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.80-Ensemble_split_20/Llama-3.1-8B-Instruct-Sparse-0.80-Math14k-part1-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3603 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part2_of_5.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.80-Ensemble_split_20/Llama-3.1-8B-Instruct-Sparse-0.80-Math14k-part2-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3603 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part3_of_5.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.80-Ensemble_split_20/Llama-3.1-8B-Instruct-Sparse-0.80-Math14k-part3-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3603 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part4_of_5.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.80-Ensemble_split_20/Llama-3.1-8B-Instruct-Sparse-0.80-Math14k-part4-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3603 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part5_of_5.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.80-Ensemble_split_20/Llama-3.1-8B-Instruct-Sparse-0.80-Math14k-part5-rand50" \
  --rounds 1 --init_frac 0.5

# ===== ********************************************************** =====
export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=2
export PYTHONNOUSERSITE=1
TORCHRUN="$CONDA_PREFIX/bin/torchrun"
SCRIPT="finetune_fast.py"

BASE="/home/models/llama-sparse/Llama-3.1-8B-Instruct-Sparse-0.80"
TOK="$BASE"

$TORCHRUN --nproc_per_node=1 --master_port=3613 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part1_of_5.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.80-Ensemble_split_20/Llama-3.1-8B-Instruct-Sparse-0.80-Math14k-part1-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3613 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part2_of_5.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.80-Ensemble_split_20/Llama-3.1-8B-Instruct-Sparse-0.80-Math14k-part2-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3613 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part3_of_5.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.80-Ensemble_split_20/Llama-3.1-8B-Instruct-Sparse-0.80-Math14k-part3-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3613 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part4_of_5.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.80-Ensemble_split_20/Llama-3.1-8B-Instruct-Sparse-0.80-Math14k-part4-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3613 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part5_of_5.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.1-8B-Instruct-Sparse/Llama-3.1-8B-Instruct-Sparse-0.80-Ensemble_split_20/Llama-3.1-8B-Instruct-Sparse-0.80-Math14k-part5-FT"



########################################
# Llama-3.2-3B-Instruct-Sparse-0.67
########################################
export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=2
export PYTHONNOUSERSITE=1
TORCHRUN="$CONDA_PREFIX/bin/torchrun"
SCRIPT="active_learning_fast.py"

BASE="/home/models/llama-sparse/Llama-3.2-3B-Instruct-Sparse-0.67"
TOK="$BASE"

# ===== Active Learning (AL 10% seed, acquire 20%, 3 rounds) =====
$TORCHRUN --nproc_per_node=1 --master_port=3501 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part1_of_3.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.67-Ensemble_split_33/Llama-3.2-3B-Instruct-Sparse-0.67-Math14k-part1-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3501 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part2_of_3.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.67-Ensemble_split_33/Llama-3.2-3B-Instruct-Sparse-0.67-Math14k-part2-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3501 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part3_of_3.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.67-Ensemble_split_33/Llama-3.2-3B-Instruct-Sparse-0.67-Math14k-part3-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

# ===== Random 50% baseline (single round) =====
$TORCHRUN --nproc_per_node=1 --master_port=3501 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part1_of_3.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.67-Ensemble_split_33/Llama-3.2-3B-Instruct-Sparse-0.67-Math14k-part1-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3501 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part2_of_3.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.67-Ensemble_split_33/Llama-3.2-3B-Instruct-Sparse-0.67-Math14k-part2-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3501 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part3_of_3.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.67-Ensemble_split_33/Llama-3.2-3B-Instruct-Sparse-0.67-Math14k-part3-rand50" \
  --rounds 1 --init_frac 0.5

# ===== ********************************************************** =====
export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=2
export PYTHONNOUSERSITE=1
TORCHRUN="$CONDA_PREFIX/bin/torchrun"
SCRIPT="finetune_fast.py"

BASE="/home/models/llama-sparse/Llama-3.2-3B-Instruct-Sparse-0.67"
TOK="$BASE"

$TORCHRUN --nproc_per_node=1 --master_port=3511 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part1_of_3.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.67-Ensemble_split_33/Llama-3.2-3B-Instruct-Sparse-0.67-Math14k-part1-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3511 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part2_of_3.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.67-Ensemble_split_33/Llama-3.2-3B-Instruct-Sparse-0.67-Math14k-part2-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3511 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part3_of_3.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.67-Ensemble_split_33/Llama-3.2-3B-Instruct-Sparse-0.67-Math14k-part3-FT"


########################################
# Llama-3.2-3B-Instruct-Sparse-0.75
########################################
export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=2
export PYTHONNOUSERSITE=1
TORCHRUN="$CONDA_PREFIX/bin/torchrun"
SCRIPT="active_learning_fast.py"

BASE="/home/models/llama-sparse/Llama-3.2-3B-Instruct-Sparse-0.75"
TOK="$BASE"

# ===== Active Learning =====
$TORCHRUN --nproc_per_node=1 --master_port=3502 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part1_of_4.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.75-Ensemble_split_25/Llama-3.2-3B-Instruct-Sparse-0.75-Math14k-part1-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3502 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part2_of_4.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.75-Ensemble_split_25/Llama-3.2-3B-Instruct-Sparse-0.75-Math14k-part2-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3502 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part3_of_4.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.75-Ensemble_split_25/Llama-3.2-3B-Instruct-Sparse-0.75-Math14k-part3-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3502 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part4_of_4.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.75-Ensemble_split_25/Llama-3.2-3B-Instruct-Sparse-0.75-Math14k-part4-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

# ===== Random 50% baseline =====
$TORCHRUN --nproc_per_node=1 --master_port=3502 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part1_of_4.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.75-Ensemble_split_25/Llama-3.2-3B-Instruct-Sparse-0.75-Math14k-part1-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3502 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part2_of_4.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.75-Ensemble_split_25/Llama-3.2-3B-Instruct-Sparse-0.75-Math14k-part2-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3502 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part3_of_4.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.75-Ensemble_split_25/Llama-3.2-3B-Instruct-Sparse-0.75-Math14k-part3-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3502 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part4_of_4.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.75-Ensemble_split_25/Llama-3.2-3B-Instruct-Sparse-0.75-Math14k-part4-rand50" \
  --rounds 1 --init_frac 0.5

# ===== ********************************************************** =====
export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=2
export PYTHONNOUSERSITE=1
TORCHRUN="$CONDA_PREFIX/bin/torchrun"
SCRIPT="finetune_fast.py"

BASE="/home/models/llama-sparse/Llama-3.2-3B-Instruct-Sparse-0.75"
TOK="$BASE"

$TORCHRUN --nproc_per_node=1 --master_port=3512 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part1_of_4.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.75-Ensemble_split_25/Llama-3.2-3B-Instruct-Sparse-0.75-Math14k-part1-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3512 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part2_of_4.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.75-Ensemble_split_25/Llama-3.2-3B-Instruct-Sparse-0.75-Math14k-part2-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3512 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part3_of_4.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.75-Ensemble_split_25/Llama-3.2-3B-Instruct-Sparse-0.75-Math14k-part3-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3512 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part4_of_4.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.75-Ensemble_split_25/Llama-3.2-3B-Instruct-Sparse-0.75-Math14k-part4-FT"


########################################
# Llama-3.2-3B-Instruct-Sparse-0.80
########################################
export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=2
export PYTHONNOUSERSITE=1
TORCHRUN="$CONDA_PREFIX/bin/torchrun"
SCRIPT="active_learning_fast.py"

BASE="/home/models/llama-sparse/Llama-3.2-3B-Instruct-Sparse-0.80"
TOK="$BASE"

# ===== Active Learning =====
$TORCHRUN --nproc_per_node=1 --master_port=3503 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part1_of_5.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.80-Ensemble_split_20/Llama-3.2-3B-Instruct-Sparse-0.80-Math14k-part1-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3503 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part2_of_5.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.80-Ensemble_split_20/Llama-3.2-3B-Instruct-Sparse-0.80-Math14k-part2-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3503 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part3_of_5.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.80-Ensemble_split_20/Llama-3.2-3B-Instruct-Sparse-0.80-Math14k-part3-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3503 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part4_of_5.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.80-Ensemble_split_20/Llama-3.2-3B-Instruct-Sparse-0.80-Math14k-part4-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3503 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part5_of_5.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.80-Ensemble_split_20/Llama-3.2-3B-Instruct-Sparse-0.80-Math14k-part5-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

# ===== Random 50% baseline =====
$TORCHRUN --nproc_per_node=1 --master_port=3503 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part1_of_5.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.80-Ensemble_split_20/Llama-3.2-3B-Instruct-Sparse-0.80-Math14k-part1-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3503 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part2_of_5.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.80-Ensemble_split_20/Llama-3.2-3B-Instruct-Sparse-0.80-Math14k-part2-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3503 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part3_of_5.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.80-Ensemble_split_20/Llama-3.2-3B-Instruct-Sparse-0.80-Math14k-part3-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3503 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part4_of_5.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.80-Ensemble_split_20/Llama-3.2-3B-Instruct-Sparse-0.80-Math14k-part4-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3503 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part5_of_5.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.80-Ensemble_split_20/Llama-3.2-3B-Instruct-Sparse-0.80-Math14k-part5-rand50" \
  --rounds 1 --init_frac 0.5

# ===== ********************************************************** =====
export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=2
export PYTHONNOUSERSITE=1
TORCHRUN="$CONDA_PREFIX/bin/torchrun"
SCRIPT="finetune_fast.py"

BASE="/home/models/llama-sparse/Llama-3.2-3B-Instruct-Sparse-0.80"
TOK="$BASE"

$TORCHRUN --nproc_per_node=1 --master_port=3513 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part1_of_5.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.80-Ensemble_split_20/Llama-3.2-3B-Instruct-Sparse-0.80-Math14k-part1-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3513 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part2_of_5.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.80-Ensemble_split_20/Llama-3.2-3B-Instruct-Sparse-0.80-Math14k-part2-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3513 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part3_of_5.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.80-Ensemble_split_20/Llama-3.2-3B-Instruct-Sparse-0.80-Math14k-part3-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3513 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part4_of_5.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.80-Ensemble_split_20/Llama-3.2-3B-Instruct-Sparse-0.80-Math14k-part4-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3513 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part5_of_5.json" \
  --output_dir "./trained_models/llama-sparse/Llama-3.2-3B-Instruct-Sparse/Llama-3.2-3B-Instruct-Sparse-0.80-Ensemble_split_20/Llama-3.2-3B-Instruct-Sparse-0.80-Math14k-part5-FT"


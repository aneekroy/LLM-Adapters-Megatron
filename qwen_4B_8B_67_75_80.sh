export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=2
export PYTHONNOUSERSITE=1
TORCHRUN="$CONDA_PREFIX/bin/torchrun"
SCRIPT="active_learning_fast.py"

BASE="/home/models/Qwen_Sparse/Qwen3-4B-Sparse-0.67"
TOK="$BASE"

# ===== Active Learning (AL 10% seed, acquire 20%, 3 rounds) =====
$TORCHRUN --nproc_per_node=1 --master_port=3301 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part1_of_3.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.67-Ensemble_split_33/Qwen3-4B-Sparse-0.67-Math14k-part1-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3301 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part2_of_3.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.67-Ensemble_split_33/Qwen3-4B-Sparse-0.67-Math14k-part2-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3301 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part3_of_3.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.67-Ensemble_split_33/Qwen3-4B-Sparse-0.67-Math14k-part3-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

# ===== Random 50% baseline (single round) =====
$TORCHRUN --nproc_per_node=1 --master_port=3301 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part1_of_3.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.67-Ensemble_split_33/Qwen3-4B-Sparse-0.67-Math14k-part1-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3301 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part2_of_3.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.67-Ensemble_split_33/Qwen3-4B-Sparse-0.67-Math14k-part2-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3301 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part3_of_3.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.67-Ensemble_split_33/Qwen3-4B-Sparse-0.67-Math14k-part3-rand50" \
  --rounds 1 --init_frac 0.5


# ===== ********************************************************** =====
export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=2
export PYTHONNOUSERSITE=1
TORCHRUN="$CONDA_PREFIX/bin/torchrun"
SCRIPT="finetune_fast.py"

BASE="/home/models/Qwen_Sparse/Qwen3-4B-Sparse-0.67"
TOK="$BASE"

$TORCHRUN --nproc_per_node=1 --master_port=3311 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part1_of_3.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.67-Ensemble_split_33/Qwen3-4B-Sparse-0.67-Math14k-part1-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3311 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part2_of_3.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.67-Ensemble_split_33/Qwen3-4B-Sparse-0.67-Math14k-part2-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3311 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part3_of_3.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.67-Ensemble_split_33/Qwen3-4B-Sparse-0.67-Math14k-part3-FT"

# ===== ********************************************************** =====

export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=2
export PYTHONNOUSERSITE=1
TORCHRUN="$CONDA_PREFIX/bin/torchrun"
SCRIPT="active_learning_fast.py"

BASE="/home/models/Qwen_Sparse/Qwen3-4B-Sparse-0.75"
TOK="$BASE"

# ===== Active Learning =====
$TORCHRUN --nproc_per_node=1 --master_port=3302 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part1_of_4.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.75-Ensemble_split_25/Qwen3-4B-Sparse-0.75-Math14k-part1-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3302 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part2_of_4.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.75-Ensemble_split_25/Qwen3-4B-Sparse-0.75-Math14k-part2-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3302 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part3_of_4.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.75-Ensemble_split_25/Qwen3-4B-Sparse-0.75-Math14k-part3-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3302 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part4_of_4.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.75-Ensemble_split_25/Qwen3-4B-Sparse-0.75-Math14k-part4-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

# ===== Random 50% baseline =====
$TORCHRUN --nproc_per_node=1 --master_port=3302 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part1_of_4.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.75-Ensemble_split_25/Qwen3-4B-Sparse-0.75-Math14k-part1-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3302 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part2_of_4.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.75-Ensemble_split_25/Qwen3-4B-Sparse-0.75-Math14k-part2-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3302 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part3_of_4.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.75-Ensemble_split_25/Qwen3-4B-Sparse-0.75-Math14k-part3-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3302 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part4_of_4.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.75-Ensemble_split_25/Qwen3-4B-Sparse-0.75-Math14k-part4-rand50" \
  --rounds 1 --init_frac 0.5

export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=2
export PYTHONNOUSERSITE=1
TORCHRUN="$CONDA_PREFIX/bin/torchrun"
SCRIPT="finetune_fast.py"

BASE="/home/models/Qwen_Sparse/Qwen3-4B-Sparse-0.75"
TOK="$BASE"

$TORCHRUN --nproc_per_node=1 --master_port=3312 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part1_of_4.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.75-Ensemble_split_25/Qwen3-4B-Sparse-0.75-Math14k-part1-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3312 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part2_of_4.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.75-Ensemble_split_25/Qwen3-4B-Sparse-0.75-Math14k-part2-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3312 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part3_of_4.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.75-Ensemble_split_25/Qwen3-4B-Sparse-0.75-Math14k-part3-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3312 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part4_of_4.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.75-Ensemble_split_25/Qwen3-4B-Sparse-0.75-Math14k-part4-FT"


export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=2
export PYTHONNOUSERSITE=1
TORCHRUN="$CONDA_PREFIX/bin/torchrun"
SCRIPT="active_learning_fast.py"

BASE="/home/models/Qwen_Sparse/Qwen3-4B-Sparse-0.80"
TOK="$BASE"

# ===== Active Learning =====
$TORCHRUN --nproc_per_node=1 --master_port=3303 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part1_of_5.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.80-Ensemble_split_20/Qwen3-4B-Sparse-0.80-Math14k-part1-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3303 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part2_of_5.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.80-Ensemble_split_20/Qwen3-4B-Sparse-0.80-Math14k-part2-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3303 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part3_of_5.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.80-Ensemble_split_20/Qwen3-4B-Sparse-0.80-Math14k-part3-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3303 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part4_of_5.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.80-Ensemble_split_20/Qwen3-4B-Sparse-0.80-Math14k-part4-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3303 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part5_of_5.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.80-Ensemble_split_20/Qwen3-4B-Sparse-0.80-Math14k-part5-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

# ===== Random 50% baseline =====
$TORCHRUN --nproc_per_node=1 --master_port=3303 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part1_of_5.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.80-Ensemble_split_20/Qwen3-4B-Sparse-0.80-Math14k-part1-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3303 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part2_of_5.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.80-Ensemble_split_20/Qwen3-4B-Sparse-0.80-Math14k-part2-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3303 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part3_of_5.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.80-Ensemble_split_20/Qwen3-4B-Sparse-0.80-Math14k-part3-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3303 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part4_of_5.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.80-Ensemble_split_20/Qwen3-4B-Sparse-0.80-Math14k-part4-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3303 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part5_of_5.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.80-Ensemble_split_20/Qwen3-4B-Sparse-0.80-Math14k-part5-rand50" \
  --rounds 1 --init_frac 0.5

# ===== Active Learning =====

export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=2
export PYTHONNOUSERSITE=1
TORCHRUN="$CONDA_PREFIX/bin/torchrun"
SCRIPT="finetune_fast.py"

BASE="/home/models/Qwen_Sparse/Qwen3-4B-Sparse-0.80"
TOK="$BASE"

$TORCHRUN --nproc_per_node=1 --master_port=3313 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part1_of_5.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.80-Ensemble_split_20/Qwen3-4B-Sparse-0.80-Math14k-part1-ft100"

$TORCHRUN --nproc_per_node=1 --master_port=3313 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part2_of_5.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.80-Ensemble_split_20/Qwen3-4B-Sparse-0.80-Math14k-part2-ft100"

$TORCHRUN --nproc_per_node=1 --master_port=3313 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part3_of_5.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.80-Ensemble_split_20/Qwen3-4B-Sparse-0.80-Math14k-part3-ft100"

$TORCHRUN --nproc_per_node=1 --master_port=3313 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part4_of_5.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.80-Ensemble_split_20/Qwen3-4B-Sparse-0.80-Math14k-part4-ft100"

$TORCHRUN --nproc_per_node=1 --master_port=3313 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part5_of_5.json" \
  --output_dir "./trained_models/Qwen3-4B-Sparse/Qwen3-4B-Sparse-0.80-Ensemble_split_20/Qwen3-4B-Sparse-0.80-Math14k-part5-ft100"



# ############################################################################################

export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=2
export PYTHONNOUSERSITE=1
TORCHRUN="$CONDA_PREFIX/bin/torchrun"
SCRIPT="active_learning_fast.py"

BASE="/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.67"
TOK="$BASE"

# ===== Active Learning =====
$TORCHRUN --nproc_per_node=1 --master_port=3401 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part1_of_3.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.67-Ensemble_split_33/Qwen3-8B-Sparse-0.67-Math14k-part1-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3401 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part2_of_3.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.67-Ensemble_split_33/Qwen3-8B-Sparse-0.67-Math14k-part2-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3401 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part3_of_3.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.67-Ensemble_split_33/Qwen3-8B-Sparse-0.67-Math14k-part3-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

# ===== Random 50% baseline =====
$TORCHRUN --nproc_per_node=1 --master_port=3401 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part1_of_3.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.67-Ensemble_split_33/Qwen3-8B-Sparse-0.67-Math14k-part1-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3401 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part2_of_3.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.67-Ensemble_split_33/Qwen3-8B-Sparse-0.67-Math14k-part2-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3401 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part3_of_3.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.67-Ensemble_split_33/Qwen3-8B-Sparse-0.67-Math14k-part3-rand50" \
  --rounds 1 --init_frac 0.5


export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=2
export PYTHONNOUSERSITE=1
TORCHRUN="$CONDA_PREFIX/bin/torchrun"
SCRIPT="finetune_fast.py"

BASE="/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.67"
TOK="$BASE"

$TORCHRUN --nproc_per_node=1 --master_port=3411 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part1_of_3.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.67-Ensemble_split_33/Qwen3-8B-Sparse-0.67-Math14k-part1-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3411 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part2_of_3.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.67-Ensemble_split_33/Qwen3-8B-Sparse-0.67-Math14k-part2-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3411 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_33/math_14k_part3_of_3.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.67-Ensemble_split_33/Qwen3-8B-Sparse-0.67-Math14k-part3-FT"


# ===== ********************************************************** =====

export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=2
export PYTHONNOUSERSITE=1
TORCHRUN="$CONDA_PREFIX/bin/torchrun"
SCRIPT="active_learning_fast.py"

BASE="/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.75"
TOK="$BASE"

# ===== Active Learning =====
$TORCHRUN --nproc_per_node=1 --master_port=3402 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part1_of_4.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.75-Ensemble_split_25/Qwen3-8B-Sparse-0.75-Math14k-part1-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3402 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part2_of_4.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.75-Ensemble_split_25/Qwen3-8B-Sparse-0.75-Math14k-part2-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3402 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part3_of_4.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.75-Ensemble_split_25/Qwen3-8B-Sparse-0.75-Math14k-part3-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3402 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part4_of_4.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.75-Ensemble_split_25/Qwen3-8B-Sparse-0.75-Math14k-part4-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

# ===== Random 50% baseline =====
$TORCHRUN --nproc_per_node=1 --master_port=3402 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part1_of_4.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.75-Ensemble_split_25/Qwen3-8B-Sparse-0.75-Math14k-part1-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3402 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part2_of_4.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.75-Ensemble_split_25/Qwen3-8B-Sparse-0.75-Math14k-part2-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3402 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part3_of_4.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.75-Ensemble_split_25/Qwen3-8B-Sparse-0.75-Math14k-part3-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3402 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part4_of_4.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.75-Ensemble_split_25/Qwen3-8B-Sparse-0.75-Math14k-part4-rand50" \
  --rounds 1 --init_frac 0.5

export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=2
export PYTHONNOUSERSITE=1
TORCHRUN="$CONDA_PREFIX/bin/torchrun"
SCRIPT="finetune_fast.py"

BASE="/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.75"
TOK="$BASE"

$TORCHRUN --nproc_per_node=1 --master_port=3412 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part1_of_4.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.75-Ensemble_split_25/Qwen3-8B-Sparse-0.75-Math14k-part1-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3412 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part2_of_4.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.75-Ensemble_split_25/Qwen3-8B-Sparse-0.75-Math14k-part2-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3412 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part3_of_4.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.75-Ensemble_split_25/Qwen3-8B-Sparse-0.75-Math14k-part3-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3412 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_25/math_14k_part4_of_4.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.75-Ensemble_split_25/Qwen3-8B-Sparse-0.75-Math14k-part4-FT"

# ===== ********************************************************** =====    


export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=2
export PYTHONNOUSERSITE=1
TORCHRUN="$CONDA_PREFIX/bin/torchrun"
SCRIPT="active_learning_fast.py"

BASE="/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.80"
TOK="$BASE"

# ===== Active Learning =====
$TORCHRUN --nproc_per_node=1 --master_port=3403 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part1_of_5.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.80-Ensemble_split_20/Qwen3-8B-Sparse-0.80-Math14k-part1-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3403 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part2_of_5.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.80-Ensemble_split_20/Qwen3-8B-Sparse-0.80-Math14k-part2-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3403 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part3_of_5.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.80-Ensemble_split_20/Qwen3-8B-Sparse-0.80-Math14k-part3-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3403 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part4_of_5.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.80-Ensemble_split_20/Qwen3-8B-Sparse-0.80-Math14k-part4-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3403 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part5_of_5.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.80-Ensemble_split_20/Qwen3-8B-Sparse-0.80-Math14k-part5-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

# ===== Random 50% baseline =====
$TORCHRUN --nproc_per_node=1 --master_port=3403 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part1_of_5.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.80-Ensemble_split_20/Qwen3-8B-Sparse-0.80-Math14k-part1-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3403 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part2_of_5.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.80-Ensemble_split_20/Qwen3-8B-Sparse-0.80-Math14k-part2-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3403 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part3_of_5.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.80-Ensemble_split_20/Qwen3-8B-Sparse-0.80-Math14k-part3-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3403 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part4_of_5.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.80-Ensemble_split_20/Qwen3-8B-Sparse-0.80-Math14k-part4-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3403 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part5_of_5.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.80-Ensemble_split_20/Qwen3-8B-Sparse-0.80-Math14k-part5-rand50" \
  --rounds 1 --init_frac 0.5


export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=2
export PYTHONNOUSERSITE=1
TORCHRUN="$CONDA_PREFIX/bin/torchrun"
SCRIPT="finetune_fast.py"

BASE="/home/models/Qwen_Sparse/Qwen3-8B-Sparse-0.80"
TOK="$BASE"

$TORCHRUN --nproc_per_node=1 --master_port=3413 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part1_of_5.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.80-Ensemble_split_20/Qwen3-8B-Sparse-0.80-Math14k-part1-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3413 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part2_of_5.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.80-Ensemble_split_20/Qwen3-8B-Sparse-0.80-Math14k-part2-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3413 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part3_of_5.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.80-Ensemble_split_20/Qwen3-8B-Sparse-0.80-Math14k-part3-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3413 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part4_of_5.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.80-Ensemble_split_20/Qwen3-8B-Sparse-0.80-Math14k-part4-FT"

$TORCHRUN --nproc_per_node=1 --master_port=3413 "$SCRIPT" \
  --base_model "$BASE" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/split_20/math_14k_part5_of_5.json" \
  --output_dir "./trained_models/Qwen3-8B-Sparse/Qwen3-8B-Sparse-0.80-Ensemble_split_20/Qwen3-8B-Sparse-0.80-Math14k-part5-FT"
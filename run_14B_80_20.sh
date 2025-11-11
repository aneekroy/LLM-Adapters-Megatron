export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=1
export PYTHONNOUSERSITE=1
TORCHRUN="$CONDA_PREFIX/bin/torchrun"
SCRIPT="active_learning_fast.py"   # use the updated file you gave me

TOK="/home/models/nvidia/OpenReasoning-Nemotron-14B"
BASE="/home/models/nvidia-sparse/OpenReasoning-Nemotron-14B-Sparse-0.80"

# ===== Active Learning (AL 10% seed, acquire 20%, 3 rounds) =====
$TORCHRUN --nproc_per_node=1 --master_port=3202 "$SCRIPT" \
  --base_model "/home/models/nvidia-sparse/OpenReasoning-Nemotron-14B-Sparse-0.80" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/math_14k.json" \
  --output_dir "./trained_models/OpenReasoning-Nemotron-14B-Sparse-Ensemble_fullMath14k/OpenReasoning-Nemotron-14B-Sparse-0.80-Math14k-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3202 "$SCRIPT" \
  --base_model "/home/models/nvidia-sparse/OpenReasoning-Nemotron-14B-Sparse-0.75" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/math_14k.json" \
  --output_dir "./trained_models/OpenReasoning-Nemotron-14B-Sparse-Ensemble_fullMath14k/OpenReasoning-Nemotron-14B-Sparse-0.75-Math14k-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

$TORCHRUN --nproc_per_node=1 --master_port=3202 "$SCRIPT" \
  --base_model "/home/models/nvidia-sparse/OpenReasoning-Nemotron-14B-Sparse-0.67" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/math_14k.json" \
  --output_dir "./trained_models/OpenReasoning-Nemotron-14B-Sparse-Ensemble_fullMath14k/OpenReasoning-Nemotron-14B-Sparse-0.67-Math14k-al50" \
  --rounds 3 --init_frac 0.1 --acq_frac 0.2

# ===== Random 50% baseline (single round) =====
$TORCHRUN --nproc_per_node=1 --master_port=3202 "$SCRIPT" \
  --base_model "/home/models/nvidia-sparse/OpenReasoning-Nemotron-14B-Sparse-0.80" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/math_14k.json" \
  --output_dir "./trained_models/OpenReasoning-Nemotron-14B-Sparse-Ensemble_fullMath14k/OpenReasoning-Nemotron-14B-Sparse-0.80-Math14k-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3202 "$SCRIPT" \
  --base_model "/home/models/nvidia-sparse/OpenReasoning-Nemotron-14B-Sparse-0.75" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/math_14k.json" \
  --output_dir "./trained_models/OpenReasoning-Nemotron-14B-Sparse-Ensemble_fullMath14k/OpenReasoning-Nemotron-14B-Sparse-0.75-Math14k-rand50" \
  --rounds 1 --init_frac 0.5

$TORCHRUN --nproc_per_node=1 --master_port=3202 "$SCRIPT" \
  --base_model "/home/models/nvidia-sparse/OpenReasoning-Nemotron-14B-Sparse-0.67" --tokenizer_path "$TOK" \
  --data_path "/home/aneek/LLM-Adapters/ft-training_set/math_14k.json" \
  --output_dir "./trained_models/OpenReasoning-Nemotron-14B-Sparse-Ensemble_fullMath14k/OpenReasoning-Nemotron-14B-Sparse-0.67-Math14k-rand50" \
  --rounds 1 --init_frac 0.5
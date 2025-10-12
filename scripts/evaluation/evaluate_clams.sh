#!/bin/bash
################################################################################
# evaluate_grammar.sh
# Script for evaluating grammar acceptability with SLURM
#
# Usage:
#   bash scripts/evaluation/evaluate_grammar.sh \
#       <base_model_path> <training_data> <grammar_tests_dir> [seeds]
#
# Example:
#   bash scripts/evaluation/evaluate_grammar.sh \
#       models/pretrained/babyberta-en-seed42 \
#       data/pre-training/CHILDES/2.5M/EN.txt \
#       data/evaluation/blimp \
#       "42 51 71"
################################################################################

# Check arguments
if [ $# -lt 3 ]; then
    echo "Usage: bash evaluate_grammar.sh <base_model_path> <training_data> <grammar_tests_dir> [seeds]"
    echo ""
    echo "Arguments:"
    echo "  base_model_path     Base model path with 'seed42' placeholder"
    echo "  training_data       Training data file (for tokenizer)"
    echo "  grammar_tests_dir   Directory containing grammar test files"
    echo "  seeds               Space-separated seeds (default: '42 51 71')"
    echo ""
    echo "Example:"
    echo "  bash evaluate_grammar.sh \\"
    echo "      models/pretrained/babyberta-en-seed42 \\"
    echo "      data/pre-training/CHILDES/2.5M/EN.txt \\"
    echo "      data/evaluation/blimp \\"
    echo "      '42 51 71'"
    exit 1
fi

BASE_MODEL_PATH=$1
TRAINING_DATA=$2
GRAMMAR_TESTS_DIR=$3
SEEDS=${4:-"42 51 71"}

echo "Submitting grammar evaluation job..."
echo "Base model: $BASE_MODEL_PATH"
echo "Training data: $TRAINING_DATA"
echo "Grammar tests: $GRAMMAR_TESTS_DIR"
echo "Seeds: $SEEDS"

# Submit SLURM job
sbatch --job-name="eval_grammar" <<EOT
#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=04:00:00
#SBATCH --job-name="eval_grammar"
#SBATCH --output=logs/eval-grammar-%J.out
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=32G

echo "========================================="
echo "Grammar Evaluation"
echo "========================================="
echo "Base model: $BASE_MODEL_PATH"
echo "Training data: $TRAINING_DATA"
echo "Grammar tests: $GRAMMAR_TESTS_DIR"
echo "Seeds: $SEEDS"
echo "Job ID: \$SLURM_JOBID"
echo "Node: \$SLURM_JOB_NODELIST"
echo "========================================="
date

# Disable wandb
export WANDB_DISABLED=true

# Load environment
module load anaconda
source activate myenv

# Run evaluation
python src/evaluation/evaluate_grammar.py \\
    --base_model_path "$BASE_MODEL_PATH" \\
    --training_data_path "$TRAINING_DATA" \\
    --data_dir "$GRAMMAR_TESTS_DIR" \\
    --seeds $SEEDS \\
    --output_dir results/grammar

echo "Evaluation completed!"
date
EOT

echo "Job submitted successfully!"
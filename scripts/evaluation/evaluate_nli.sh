#!/bin/bash
################################################################################
# evaluate_nli.sh
# Script for evaluating NLI models with SLURM
#
# Usage:
#   bash scripts/evaluation/evaluate_nli.sh <model_path> <validation_file> [dataset_name]
#
# Example:
#   bash scripts/evaluation/evaluate_nli.sh \
#       models/finetuned/babyberta-xnli \
#       data/finetune/XNLI/EN/xnli_dev.json \
#       xnli-en
################################################################################

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: bash evaluate_nli.sh <model_path> <validation_file> [dataset_name]"
    echo "Example: bash evaluate_nli.sh models/babyberta-xnli data/xnli_dev.json xnli-en"
    exit 1
fi

MODEL_PATH=$1
VALIDATION_FILE=$2
DATASET_NAME=${3:-"nli"}

echo "Submitting NLI evaluation job..."
echo "Model: $MODEL_PATH"
echo "Validation file: $VALIDATION_FILE"
echo "Dataset name: $DATASET_NAME"

# Submit SLURM job
sbatch --job-name="eval_nli_${DATASET_NAME}" <<EOT
#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --job-name="eval_nli_${DATASET_NAME}"
#SBATCH --output=logs/eval-nli-%J.out
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=32G

echo "========================================="
echo "NLI Evaluation: ${DATASET_NAME}"
echo "========================================="
echo "Model: $MODEL_PATH"
echo "Validation file: $VALIDATION_FILE"
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
python src/evaluation/evaluate_nli.py \\
    "$MODEL_PATH" \\
    "$VALIDATION_FILE" \\
    --dataset_name "$DATASET_NAME" \\
    --batch_size 32

echo "Evaluation completed!"
date
EOT

echo "Job submitted successfully!"
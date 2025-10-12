#!/bin/bash
################################################################################
# finetune_single_task.sh
# Script for fine-tuning pretrained models on downstream tasks
#
# Usage:
#   bash scripts/finetuning/finetune_single_task.sh \
#       <model_checkpoint> <task> <dataset_name> <output_dir>
#
# Tasks: qa, nli
# Model types: extractive (for BERT-like), generative (for T5)
#
# Example (QA):
#   bash scripts/finetuning/finetune_single_task.sh \
#       models/pretrained/babyberta/checkpoint-260000 \
#       qa \
#       squad \
#       models/finetuned/babyberta-squad
#
# Example (NLI):
#   bash scripts/finetuning/finetune_single_task.sh \
#       models/pretrained/t5/checkpoint-50000 \
#       nli \
#       xnli-en \
#       models/finetuned/t5-xnli
################################################################################

# Check arguments
if [ $# -lt 4 ]; then
    echo "Usage: bash finetune_single_task.sh <model_checkpoint> <task> <dataset_name> <output_dir> [model_type]"
    echo ""
    echo "Arguments:"
    echo "  model_checkpoint  Path to pretrained model"
    echo "  task              Task type: qa or nli"
    echo "  dataset_name      Dataset name (e.g., squad, xnli-en)"
    echo "  output_dir        Output directory for fine-tuned model"
    echo "  model_type        Model type: extractive or generative (optional, auto-detect from path)"
    echo ""
    echo "Examples:"
    echo "  # BabyBERTa on SQuAD"
    echo "  bash finetune_single_task.sh models/babyberta/checkpoint-260000 qa squad models/finetuned/babyberta-squad"
    echo ""
    echo "  # T5 on XNLI"
    echo "  bash finetune_single_task.sh models/t5/checkpoint-50000 nli xnli-en models/finetuned/t5-xnli generative"
    exit 1
fi

MODEL_CHECKPOINT=$1
TASK=$2
DATASET_NAME=$3
OUTPUT_DIR=$4
MODEL_TYPE=${5:-""}  # Optional: extractive or generative

# Auto-detect model type if not provided
if [ -z "$MODEL_TYPE" ]; then
    if [[ "$MODEL_CHECKPOINT" == *"t5"* ]] || [[ "$MODEL_CHECKPOINT" == *"T5"* ]]; then
        MODEL_TYPE="generative"
        echo "Auto-detected model type: generative (T5)"
    else
        MODEL_TYPE="extractive"
        echo "Auto-detected model type: extractive (BERT-like)"
    fi
fi

# Validate task
if [ "$TASK" != "qa" ] && [ "$TASK" != "nli" ]; then
    echo "Error: Task must be 'qa' or 'nli'"
    exit 1
fi

# Validate model type
if [ "$MODEL_TYPE" != "extractive" ] && [ "$MODEL_TYPE" != "generative" ]; then
    echo "Error: Model type must be 'extractive' or 'generative'"
    exit 1
fi

echo "Submitting fine-tuning job..."
echo "Model: $MODEL_CHECKPOINT"
echo "Task: $TASK"
echo "Dataset: $DATASET_NAME"
echo "Output: $OUTPUT_DIR"
echo "Model Type: $MODEL_TYPE"

# Create job name
JOB_NAME="finetune_${TASK}_${DATASET_NAME}"

# Submit SLURM job
sbatch --job-name="$JOB_NAME" <<EOT
#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=24:00:00
#SBATCH --job-name="$JOB_NAME"
#SBATCH --output=logs/finetune-%J.out
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=58G

echo "========================================="
echo "Fine-tuning: $TASK on $DATASET_NAME"
echo "========================================="
echo "Model checkpoint: $MODEL_CHECKPOINT"
echo "Dataset: $DATASET_NAME"
echo "Output dir: $OUTPUT_DIR"
echo "Model type: $MODEL_TYPE"
echo "Job ID: \$SLURM_JOBID"
echo "Node: \$SLURM_JOB_NODELIST"
echo "========================================="
date

# Disable wandb
export WANDB_DISABLED=true

# Load environment
module load anaconda
source activate myenv

# Run fine-tuning based on task
if [ "$TASK" == "qa" ]; then
    echo "Running QA fine-tuning..."
    python src/finetuning/finetune_qa.py \\
        --model_checkpoint "$MODEL_CHECKPOINT" \\
        --dataset_name "$DATASET_NAME" \\
        --output_dir "$OUTPUT_DIR" \\
        --model_type "$MODEL_TYPE" \\
        --batch_size 8 \\
        --num_epochs 3 \\
        --learning_rate 1e-4
elif [ "$TASK" == "nli" ]; then
    echo "Running NLI fine-tuning..."
    python src/finetuning/finetune_nli.py \\
        --model_checkpoint "$MODEL_CHECKPOINT" \\
        --dataset_name "$DATASET_NAME" \\
        --output_dir "$OUTPUT_DIR" \\
        --model_type "$MODEL_TYPE" \\
        --batch_size 16 \\
        --num_epochs 3 \\
        --learning_rate 2e-5
fi

echo "Fine-tuning completed!"
date
EOT

echo "Job submitted successfully!"
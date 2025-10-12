#!/bin/bash
################################################################################
# run_t5.sh
# Script for pretraining T5 models with SLURM
#
# Usage:
#   bash scripts/pretraining/run_t5.sh <data_path> <output_dir> <seeds>
#
# Example:
#   bash scripts/pretraining/run_t5.sh \
#       data/pre-training/CHILDES/2.5M/EN.txt \
#       models/pretrained/t5 \
#       "42 51 71"
################################################################################

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: bash run_t5.sh <data_path> <output_dir> [seeds]"
    echo "Example: bash run_t5.sh data/EN.txt models/t5 '42 51 71'"
    exit 1
fi

DATA_PATH=$1
OUTPUT_DIR=$2
SEEDS=${3:-"42"}  # Default seed is 42

echo "Submitting T5 pretraining job..."
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Seeds: $SEEDS"

# Submit SLURM job
sbatch --job-name="pretrain_t5" <<EOT
#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=48:00:00
#SBATCH --job-name="pretrain_t5"
#SBATCH --output=logs/t5-%J.out
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=64G

echo "========================================="
echo "T5 Pretraining"
echo "========================================="
echo "Data path: $DATA_PATH"
echo "Output dir: $OUTPUT_DIR"
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

# Run pretraining
python src/pretraining/pretrain_t5.py \\
    --data_path "$DATA_PATH" \\
    --output_dir "$OUTPUT_DIR" \\
    --seeds $SEEDS \\
    --model_name_or_path t5-small \\
    --batch_size 8 \\
    --learning_rate 5e-4 \\
    --num_epochs 3

echo "Training completed!"
date
EOT

echo "Job submitted successfully!"
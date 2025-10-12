#!/bin/bash
################################################################################
# run_ltgbert.sh
# Script for pretraining LTG-BERT models with SLURM
#
# Usage:
#   bash scripts/pretraining/run_ltgbert.sh <data_path> <output_dir> <seeds>
#
# Example:
#   bash scripts/pretraining/run_ltgbert.sh \
#       data/pre-training/CHILDES/2.5M/EN.txt \
#       models/pretrained/ltgbert \
#       "42 51 71"
################################################################################

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: bash run_ltgbert.sh <data_path> <output_dir> [seeds]"
    echo "Example: bash run_ltgbert.sh data/EN.txt models/ltgbert '42 51 71'"
    exit 1
fi

DATA_PATH=$1
OUTPUT_DIR=$2
SEEDS=${3:-"42"}  # Default seed is 42

echo "Submitting LTG-BERT pretraining job..."
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Seeds: $SEEDS"

# Submit SLURM job
sbatch --job-name="pretrain_ltgbert" <<EOT
#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=48:00:00
#SBATCH --job-name="pretrain_ltgbert"
#SBATCH --output=logs/ltgbert-%J.out
#SBATCH --gpus=rtx_4090:1
#SBATCH --mem=64G

echo "========================================="
echo "LTG-BERT Pretraining"
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
python src/pretraining/pretrain_ltgbert.py \\
    --data_path "$DATA_PATH" \\
    --output_dir "$OUTPUT_DIR" \\
    --seeds $SEEDS \\
    --vocab_size 30000 \\
    --hidden_size 768 \\
    --num_layers 12 \\
    --batch_size 8 \\
    --learning_rate 1e-4 \\
    --max_steps 100000

echo "Training completed!"
date
EOT

echo "Job submitted successfully!"
#!/bin/bash
# run_t5.sh - Simple T5 pretraining script

set -e

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <data_file> [seeds...]"
    echo "Example: $0 data/en_childes_2.5M.txt 1 2 3"
    exit 1
fi

DATA_PATH=$1
shift

# Default seeds if none provided
SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
    SEEDS=(1 2 3)
fi

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/src/pretraining"
OUTPUT_DIR="outputs/pretrained_models/t5"

# Training configuration
BATCH_SIZE=8
LEARNING_RATE=5e-4
MAX_STEPS=50000
WARMUP_STEPS=5000

echo "Training T5 on: $DATA_PATH"
echo "Seeds: ${SEEDS[@]}"
echo ""

python "${SCRIPT_DIR}/train_t5.py" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --seeds ${SEEDS[@]} \
    --model_name_or_path "t5-small" \
    --max_seq_length 128 \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --max_steps $MAX_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --logging_steps 500 \
    --save_steps 5000

echo "Training completed!"
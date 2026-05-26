#!/bin/bash
# Learning-rate sweep on the four-task GLUE FedLEASE baseline, single seed.
# Sweeps {1e-4, 3e-4, 1e-3, 3e-3, 5e-3}.
#
# Usage:
#   bash scripts/repro/sweep_lr.sh
#
# Activate your Python environment first, e.g.
#   uv sync && source .venv/bin/activate

set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LRS="1e-4 3e-4 1e-3 3e-3 5e-3"
SEED=42
BASE_ARGS="--model_name roberta-large \
    --tasks sst2 sst2 sst2 sst2 qnli qnli qnli qnli mrpc mrpc mrpc mrpc qqp qqp qqp qqp \
    --global_rounds 25 --warmup_rounds 5 --local_epochs 2 \
    --rank 4 --max_clusters 8 \
    --train_samples 1000 --test_samples 200 --batch_size 128"

for lr in $LRS; do
    echo "========================================="
    echo "Running lr=$lr seed=$SEED at $(date)"
    echo "========================================="
    python -u main.py $BASE_ARGS \
        --lr "$lr" --seed "$SEED" \
        --output_dir "./output/sweep_lr_${lr}_seed${SEED}"
    echo "lr=$lr completed at $(date)"
    echo ""
done

echo "All lr sweeps completed."

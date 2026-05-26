#!/bin/bash
# Reproduce the four-task GLUE FedLEASE baseline (RoBERTa-Large, lr=1e-3) over
# five seeds. Mean accuracy across seeds matches the upstream FedLEASE
# in-distribution number (~87.88).
#
# Usage:
#   bash scripts/repro/run_all_seeds.sh                 # seeds 42 43 44 45 46
#   bash scripts/repro/run_all_seeds.sh "42 43 44"      # custom seed list
#
# Activate your Python environment first, e.g.
#   uv sync && source .venv/bin/activate
# or
#   python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

set -e

SEEDS="${1:-42 43 44 45 46}"
BASE_ARGS="--model_name roberta-large \
    --tasks sst2 sst2 sst2 sst2 qnli qnli qnli qnli mrpc mrpc mrpc mrpc qqp qqp qqp qqp \
    --global_rounds 25 --warmup_rounds 5 --local_epochs 2 \
    --assignment_mode oracle --rank 4 --max_clusters 4 \
    --train_samples 1000 --test_samples 200 \
    --batch_size 128 --lr 1e-3 \
    --save_final_params --cross_eval"

for seed in $SEEDS; do
    echo "========================================="
    echo "Running seed $seed at $(date)"
    echo "========================================="
    python -u main.py $BASE_ARGS \
        --seed "$seed" \
        --output_dir "./output/fedlease_4task_seed${seed}"
    echo "Seed $seed completed at $(date)"
    echo ""
done

echo "All seeds completed. Aggregating results..."
python scripts/analysis/analyze_results.py --output_dir ./output --seeds $SEEDS
echo "Done."

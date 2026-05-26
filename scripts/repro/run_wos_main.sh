#!/bin/bash
# Six-task GLUE variant of the dual-forward objective with shuffled pooling on
# both the client and server sides (RoBERTa-Large, sst2/qnli/mrpc/qqp/mnli/rte,
# 18 clients, K=6, lambda=0.20).
#
# Usage:
#   bash scripts/repro/run_wos_main.sh [seed]            # default seed 42
#
# Activate your Python environment first, e.g.
#   uv sync && source .venv/bin/activate

set -e

SEED="${1:-42}"

python -u main.py \
  --model_name roberta-large \
  --tasks sst2 sst2 sst2 qnli qnli qnli mrpc mrpc mrpc \
          qqp qqp qqp mnli mnli mnli rte rte rte \
  --global_rounds 25 --warmup_rounds 5 --local_epochs 2 \
  --assignment_mode oracle --rank 4 --max_clusters 6 --lr 1e-3 \
  --batch_size 64 --train_samples 1000 --test_samples 200 \
  --universal_expert --additive_residual --universal_warmup_rounds 2 \
  --soft_membership task_family --visa_coeff 0.20 \
  --client_exposure_mode shuffled --server_exposure_mode shuffled \
  --save_final_params --cross_eval \
  --seed "$SEED" \
  --output_dir "./output/wos_shuffled_6task_seed${SEED}"

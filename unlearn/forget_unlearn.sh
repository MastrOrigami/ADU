#!/usr/bin/env bash


MODEL_DIR="/path/to/model/dir"
OUT_DIR="/path/to/output/dir"

python3 forget_unlearn.py \
  --model_name_or_path "${MODEL_DIR}" \
  --output_dir "${OUT_DIR}" \
  --max_num_batches 400 \
  --lr 5e-5 \
  --seed 42 \
  --layer_ids 12,13,14,15,16 \
  --param_ids 6 \
  --forget_subsets bio-forget-corpus,cyber-forget-corpus \
  --shuffle \
  --max_new_tokens 64 \
  --q_preplan 0.10 \
  --q_anchor 0.10 \
  --W 32 \
  --k 8 \
  --lambda_block 1.0 \
  --gamma_amp 1.5 \
  --log_every 10

#!/usr/bin/env bash


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FORGOT_MODEL_DIR="/path/to/model/dir"

# Reference/original model (theta_ref) for attention constraints
REF_MODEL_DIR="/path/to/output/dir"

# Output: forgotten+retain model
OUT_DIR="${SCRIPT_DIR}/models/adu_after_retain-2"
mkdir -p "${OUT_DIR}"

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"



python3 "${SCRIPT_DIR}/retain_learn.py" \
  --model_dir "${FORGOT_MODEL_DIR}" \
  --ref_model_dir "${REF_MODEL_DIR}" \
  --output_dir "${OUT_DIR}" \
  --mmlu_dataset "cais/mmlu" \
  --mmlu_config "all" \
  --mmlu_splits "auxiliary_train,dev,val" \
  --max_num_batches 400 \
  --lr 2e-6 \
  --seed 42 \
  --layer_ids "12,13,14,15,16" \
  --param_ids "6" \
  --frac_local 0.20 \
  --frac_global 0.20 \
  --W 32 \
  --q_anchor 0.20 \
  --gamma_amp 1.0 \
  --max_seq_len 512 \
  --log_every 10 \
  --w_syntax 1.0 \
  --w_reason 0.0 


#!/usr/bin/env bash

MODEL_DIR="/path/to/model/dir"
OUT_DIR="/path/to/output/dir"
OUTPUT_NAME="v0-origin-temp-shot"
mkdir -p "${OUT_DIR}"

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -m lm_eval \
  --model hf \
  --model_args "pretrained=${MODEL_DIR},dtype=float16" \
  --tasks "wmdp_bio,wmdp_cyber" \
  --gen_kwargs "do_sample=False,temperature=0.0,max_gen_toks=256" \
  --apply_chat_template \
  --batch_size 1 \
  --device cuda:0 \
  --output_path "${OUT_DIR}"

# retain/retain_learn.py
# -*- coding: utf-8 -*-



import argparse
import datetime
import inspect
import os
import random
from typing import Any, Optional

import numpy as np
import torch
from torch.optim import AdamW
import tqdm

from utils import load_model, get_params
from retain_utils import (
    RetainHyper,
    load_mmlu_retain,
    build_mmlu_prompt,
    build_model_input_ids,
    classify_heads_from_ref_attn,
    compute_T_glob_from_ref_attn,
    l_retain,
)


def parse_args():
    p = argparse.ArgumentParser("ADU retain module (L_retain)")

    # models
    p.add_argument("--model_dir", type=str, default=None, help="Forgotten model directory (input).")
    p.add_argument(
        "--forgot_model_dir",
        type=str,
        default=None,
        help="Alias for --model_dir (deprecated; keep for old .sh scripts).",
    )
    p.add_argument("--ref_model_dir", type=str, default=None, help="Reference/original model for constraints (theta_ref).")
    p.add_argument("--output_dir", type=str, default="retain/models/adu_retain_out")

    # update params (reuse your RMU-style selection)
    p.add_argument("--layer_ids", type=str, default="12,13,14,15,16")
    p.add_argument("--param_ids", type=str, default="6")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_num_batches", type=int, default=400)

    # Backward-compatible aliases (older scripts)
    p.add_argument("--max_steps", type=int, default=None, help="Alias for --max_num_batches.")
    p.add_argument("--batch_size", type=int, default=1, help="(Optional) micro-batch size (currently only 1 is supported).")

    # data
    p.add_argument("--mmlu_dataset", type=str, default="cais/mmlu")
    p.add_argument("--mmlu_config", type=str, default="all", help="For cais/mmlu, use config 'all' (recommended).")
    p.add_argument("--mmlu_splits", type=str, default="auxiliary_train,dev,val")
    p.add_argument("--shuffle", action="store_true")

    # retain hyper
    p.add_argument("--frac_local", type=float, default=0.30)
    p.add_argument("--frac_global", type=float, default=0.30)
    p.add_argument("--W", type=int, default=32)
    p.add_argument("--q_anchor", type=float, default=0.40)  # Method.txt uses 0.4
    p.add_argument("--gamma_amp", type=float, default=1.5)
    p.add_argument("--w_syntax", type=float, default=1.0)
    p.add_argument("--w_reason", type=float, default=1.0)

    # sequence control
    p.add_argument("--max_seq_len", type=int, default=512, help="Truncate to last max_seq_len tokens if longer.")
    p.add_argument("--anchor_on_prompt_only", action="store_true", help="(Debug) Restrict T_glob to prompt positions.")

    # logging / saving
    p.add_argument("--log_every", type=int, default=10)

    args = p.parse_args()
    # Map legacy flags to canonical ones
    if (getattr(args, "model_dir", None) is None or args.model_dir == "") and getattr(args, "forgot_model_dir", None):
        args.model_dir = args.forgot_model_dir
    if getattr(args, "max_steps", None) is not None:
        args.max_num_batches = int(args.max_steps)
    # batch_size is currently unused (retains for compatibility); keep as attribute.
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_attn_impl(m) -> Optional[str]:
    cfg = getattr(m, "config", None)
    if cfg is None:
        return None
    for attr in ("attn_implementation", "_attn_implementation"):
        if hasattr(cfg, attr):
            try:
                v = getattr(cfg, attr)
                if isinstance(v, str) and v:
                    return v
            except Exception:
                pass
    return None


def _force_eager_attn(m):
    """Force attention implementation to eager in the most compatible way."""
    # 1) Newer Transformers: model.set_attn_implementation("eager")
    if hasattr(m, "set_attn_implementation"):
        try:
            m.set_attn_implementation("eager")
        except Exception:
            pass

    # 2) Try config fields (Transformers checks these for sdpa gating)
    cfg = getattr(m, "config", None)
    if cfg is not None:
        for attr in ("attn_implementation", "_attn_implementation"):
            try:
                if hasattr(cfg, attr):
                    setattr(cfg, attr, "eager")
            except Exception:
                pass

    # 3) Some models mirror this in generation_config
    gen = getattr(m, "generation_config", None)
    if gen is not None:
        for attr in ("attn_implementation", "_attn_implementation"):
            try:
                if hasattr(gen, attr):
                    setattr(gen, attr, "eager")
            except Exception:
                pass


def _load_mmlu_examples(dataset_path: str, config: str, splits: str):
    """Call retain_utils.load_mmlu_retain with signature-compatibility across variants."""
    sig = inspect.signature(load_mmlu_retain)
    kwargs: dict[str, Any] = {}

    # dataset path
    if "dataset_path" in sig.parameters:
        kwargs["dataset_path"] = dataset_path
    else:
        kwargs["dataset_path"] = dataset_path  # most versions

    # splits
    if "splits" in sig.parameters:
        kwargs["splits"] = splits

    # config name parameter varies across earlier patches
    if "config_name" in sig.parameters:
        kwargs["config_name"] = config
    elif "config" in sig.parameters:
        kwargs["config"] = config

    return load_mmlu_retain(**kwargs)


def main():
    args = parse_args()
    set_seed(args.seed)

    if not args.model_dir:
        raise ValueError("--model_dir is required and cannot be empty.")
    if not args.ref_model_dir:
        raise ValueError("--ref_model_dir is required and cannot be empty.")

    # load models
    updated_model, _tok_u = load_model(args.model_dir)
    ref_model, tokenizer = load_model(args.ref_model_dir)

    device = next(updated_model.parameters()).device

    # IMPORTANT: ensure attentions can be returned
    _force_eager_attn(updated_model)
    _force_eager_attn(ref_model)

    # Disable checkpointing to avoid detached attentions in some model impls
    for m in (updated_model, ref_model):
        try:
            m.gradient_checkpointing_disable()
        except Exception:
            pass
        try:
            m.config.use_cache = False
        except Exception:
            pass

    # Sanity check: if still sdpa, fail early with actionable guidance
    impl_u = _get_attn_impl(updated_model)
    impl_r = _get_attn_impl(ref_model)
    if (impl_u == "sdpa") or (impl_r == "sdpa"):
        raise RuntimeError(
            "Your model is still using attn_implementation=sdpa, which cannot return attentions. "
            "Please modify utils.load_model() to pass attn_implementation='eager' into from_pretrained(), "
            "or explicitly reload the model with eager attention."
        )

    ref_model.eval()
    updated_model.train()

    # params to update
    layer_ids = [int(x) for x in args.layer_ids.split(",") if x.strip()]
    param_ids = [int(x) for x in args.param_ids.split(",") if x.strip()]
    params = get_params(updated_model, layer_ids, param_ids)
    opt = AdamW(params, lr=args.lr)

    # data
    examples = _load_mmlu_examples(args.mmlu_dataset, args.mmlu_config, args.mmlu_splits)
    if len(examples) == 0:
        raise RuntimeError("Loaded 0 MMLU examples. Check dataset/config/splits.")
    if args.shuffle:
        random.shuffle(examples)

    hp = RetainHyper(
        frac_local=args.frac_local,
        frac_global=args.frac_global,
        W=args.W,
        q_anchor=args.q_anchor,
        gamma_amp=args.gamma_amp,
    )

    H_loc = None
    H_glob = None

    # training loop
    pbar = tqdm.tqdm(range(args.max_num_batches))
    running = 0.0

    for step in pbar:
        ex = examples[step % len(examples)]
        prompt, target = build_mmlu_prompt(ex)

        # teacher-forced fixed sequence: prompt + target token(s)
        ids_prompt = build_model_input_ids(tokenizer, args.ref_model_dir, prompt)[0]
        ids_target = tokenizer(target, add_special_tokens=False, return_tensors="pt").input_ids[0]
        full_ids = torch.cat([ids_prompt, ids_target], dim=0)
        prompt_len = int(ids_prompt.shape[0])

        if full_ids.shape[0] > args.max_seq_len:
            full_ids = full_ids[-args.max_seq_len:]
            # adjust prompt_len consistently
            total_before = int(ids_prompt.shape[0] + ids_target.shape[0])
            cut = max(0, total_before - args.max_seq_len)
            prompt_len = max(0, prompt_len - cut)

        full_ids = full_ids.unsqueeze(0).to(device)
        attn_mask = torch.ones_like(full_ids, device=device)

        # reference attentions (no grad)
        with torch.no_grad():
            out_ref = ref_model(
                input_ids=full_ids,
                attention_mask=attn_mask,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
            ref_attn = list(out_ref.attentions)

        # classify heads once using ref attentions
        if H_loc is None or H_glob is None:
            H_loc, H_glob = classify_heads_from_ref_attn(ref_attn, prompt_len, hp)

        # updated attentions (grad)
        out_upd = updated_model(
            input_ids=full_ids,
            attention_mask=attn_mask,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
        upd_attn = list(out_upd.attentions)

        # anchors from ref attentions
        T_glob = compute_T_glob_from_ref_attn(
            ref_attn,
            H_glob=H_glob,
            q_anchor=hp.q_anchor,
            anchor_on_prompt_only=args.anchor_on_prompt_only,
            prompt_len=prompt_len,
        )

        loss = l_retain(
            updated_attentions=upd_attn,
            ref_attentions=ref_attn,
            H_loc=H_loc,
            H_glob=H_glob,
            T_glob=T_glob,
            gamma_amp=hp.gamma_amp,
            w_syntax=args.w_syntax,
            w_reason=args.w_reason,
        )

        if not loss.requires_grad:
            pbar.set_postfix({"loss": "0.0000", "note": "no_grad"})
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        running = 0.98 * running + 0.02 * float(loss.detach().item()) if step > 0 else float(loss.detach().item())

        if (step + 1) % max(1, args.log_every) == 0:
            pbar.set_postfix({"loss": f"{running:.4f}"})

    # save
    os.makedirs(args.output_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    meta_path = os.path.join(args.output_dir, f"retain_meta_{stamp}.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(str(vars(args)) + "\n")

    # save model + tokenizer
    updated_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"[RETAIN] saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

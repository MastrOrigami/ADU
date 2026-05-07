# unlearn/adu_unlearn.py
# -*- coding: utf-8 -*-

import os
import argparse
import datetime
import random
import numpy as np
import torch
from torch.optim import AdamW
import tqdm

from utils import load_model, get_params
from forget_utils import (
    ADUHyper,
    load_wmdp_train,
    build_prompt_wmdp_mcq,
    greedy_generate_full_ids,
    compute_sets_from_frozen,
    swld_forget_loss,
)

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="unlearn/models/adu_llama_wmdp")

    # update params (reuse RMU style)
    p.add_argument("--layer_ids", type=str, default="12,13,14,15,16")
    p.add_argument("--param_ids", type=str, default="6")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_num_batches", type=int, default=400)

    # generation
    p.add_argument("--max_new_tokens", type=int, default=256)

    # data
    p.add_argument("--forget_subsets", type=str, default="bio-forget-corpus,cyber-forget-corpus")
    p.add_argument("--shuffle", action="store_true")

    # ADU hyper
    p.add_argument("--frac_local", type=float, default=0.30)
    p.add_argument("--frac_global", type=float, default=0.30)
    p.add_argument("--W", type=int, default=32)
    p.add_argument("--q_preplan", type=float, default=0.20)
    p.add_argument("--q_anchor", type=float, default=0.20)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--tau_waad_quantile", type=float, default=0.30)
    p.add_argument("--tau_delta_quantile", type=float, default=0.70)

    # SWLD
    p.add_argument("--lambda_block", type=float, default=1.0)
    p.add_argument("--gamma_amp", type=float, default=1.5)

    p.add_argument("--log_every", type=int, default=10)
    return p.parse_args()


def main():
    args = get_args()

    # parse lists
    args.layer_ids = [int(x) for x in args.layer_ids.split(",") if x.strip()]
    args.param_ids = [int(x) for x in args.param_ids.split(",") if x.strip()]
    subsets = [s.strip() for s in args.forget_subsets.split(",") if s.strip()]

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    frozen_model, tokenizer = load_model(args.model_name_or_path)
    updated_model, _ = load_model(args.model_name_or_path)

    try:
        frozen_model.set_attn_implementation("eager")
        updated_model.set_attn_implementation("eager")
    except Exception:
        pass

    params = get_params(updated_model, args.layer_ids, args.param_ids)
    opt = AdamW(params, lr=args.lr)

    # load forget data: WMDP benchmark train
    examples = load_wmdp_train(subsets)
    if args.shuffle:
        random.shuffle(examples)

    hp = ADUHyper(
        frac_local=args.frac_local,
        frac_global=args.frac_global,
        W=args.W,
        q_preplan=args.q_preplan,
        q_anchor=args.q_anchor,
        k=args.k,
        tau_waad_quantile=args.tau_waad_quantile,
        tau_delta_quantile=args.tau_delta_quantile,
    )

    updated_model.train()
    frozen_model.eval()

    skip_cnt = 0
    pbar = tqdm.tqdm(range(args.max_num_batches))
    
    for step in pbar:
        subset_name, subset_idx, ex = examples[step % len(examples)]
        prompt = build_prompt_wmdp_mcq(ex, cot=True)

        # 1) frozen generate a fixed response (sequence)
        with torch.no_grad():
            full_ids, prompt_len, _out_text = greedy_generate_full_ids(
                frozen_model, tokenizer, args.model_name_or_path, prompt, max_new_tokens=args.max_new_tokens
            )

        # 2) compute structural sets from frozen attentions
        with torch.no_grad():
            sets = compute_sets_from_frozen(
                frozen_model, tokenizer, full_ids, prompt_len, hp, ex,
                use_correct_choice_as_sanc=True
            )

        # 3) SWLD forget loss on updated model (Eq.8/9/10)
        loss = swld_forget_loss(
            updated_model,
            full_ids,
            sets,
            lambda_block=args.lambda_block,
            gamma_amp=args.gamma_amp,
        )

        if (not loss.requires_grad):
            skip_cnt += 1
            tqdm.tqdm.write(f"[skip] Tloc={len(sets['T_loc'])} Sanc={len(sets['S_anc'])} Hglob={len(sets['H_glob'])}")
            continue

        opt.zero_grad(set_to_none=True)
        if (not loss.requires_grad) or (loss.detach().item() == 0.0):
            print(f"[skip] empty signal: Tloc={len(sets['T_loc'])} Sanc={len(sets['S_anc'])} Hglob={len(sets['H_glob'])}")
            continue
        loss.backward()
        opt.step()
        
        pbar.set_postfix({
        "loss": f"{loss.item():.4f}",
        "skip%": f"{100*skip_cnt/(step+1):.1f}",
        "Tloc": len(sets["T_loc"]),
        "Sanc": len(sets["S_anc"]),
    })

        if step % args.log_every == 0:
            print(
                f"[step {step}] subset={subset_name} idx={subset_idx} "
                f"Tloc={len(sets['T_loc'])} Sanc={len(sets['S_anc'])} Hglob={len(sets['H_glob'])} "
                f"loss={loss.item():.6f}"
            )

    # save
    out_dir = args.output_dir
    if out_dir.endswith("/"):
        out_dir = out_dir[:-1]
    if out_dir == "unlearn/models/adu_llama_wmdp":
        # default add timestamp to avoid overwriting
        out_dir = f"{out_dir}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    os.makedirs(out_dir, exist_ok=True)
    updated_model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()

# unlearn/adu_utils.py
# -*- coding: utf-8 -*-

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Set

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset


# -----------------------------
# Prompt & chat template utils
# -----------------------------

def is_instruct_model(model_path_or_name: str, tokenizer) -> bool:
    if "instruct" in (model_path_or_name or "").lower():
        return True
    if getattr(tokenizer, "chat_template", None):
        return True
    return False


def build_model_input_ids(tokenizer, model_path_or_name: str, raw_prompt_text: str) -> torch.Tensor:
    use_chat = is_instruct_model(model_path_or_name, tokenizer) and hasattr(tokenizer, "apply_chat_template")
    if use_chat:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Follow the user's instructions carefully."},
            {"role": "user", "content": raw_prompt_text},
        ]
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
    return tokenizer(raw_prompt_text, return_tensors="pt", add_special_tokens=True)["input_ids"]


def build_prompt_wmdp_mcq(ex: Dict[str, Any], cot: bool = True) -> str:
    """
    WMDP benchmark sample:
      - ex["question"], ex["choices"] (len=4), ex["answer"] (0..3)
    """
    q = ex["question"].strip()
    ch = ex["choices"]
    prompt = (
        f"Q: {q}\n"
        f"A. {ch[0]}\n"
        f"B. {ch[1]}\n"
        f"C. {ch[2]}\n"
        f"D. {ch[3]}\n"
    )
    if cot:
        prompt += (
            "A: Let's think step by step.\n"
            "First give a brief reasoning, then output the final answer as a single letter on the last line in the form:\n"
            "Final answer: <A/B/C/D>\n"
        )
    else:
        prompt += "A: Answer with a single letter.\n"
    return prompt


def load_wmdp_train(subsets: List[str]) -> List[Tuple[str, int, Dict[str, Any]]]:
    """
    Return list of (subset_name, index_in_subset, example_dict)
    """
    all_ex = []
    for subset in subsets:
        ds = load_dataset("cais/wmdp", subset, split="test")
        for i in range(len(ds)):
            all_ex.append((subset, i, ds[i]))
    return all_ex


@torch.no_grad()
def greedy_generate_full_ids(
    model, tokenizer, model_path_or_name: str, prompt_text: str, max_new_tokens: int
) -> Tuple[torch.Tensor, int, str]:
    input_ids = build_model_input_ids(tokenizer, model_path_or_name, prompt_text).to(model.device)
    prompt_len = int(input_ids.shape[-1])

    attn = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    full_ids = gen[0].to(model.device)  # [T_total]
    out_ids = full_ids[prompt_len:]
    out_text = tokenizer.decode(out_ids, skip_special_tokens=True).strip()
    return full_ids, prompt_len, out_text


# -----------------------------
# Special token extraction (Tloc/Tglob/D/I(D)) + head grouping
# -----------------------------

@dataclass
class ADUHyper:
    frac_local: float = 0.30
    frac_global: float = 0.30
    W: int = 32
    q_preplan: float = 0.10
    q_anchor: float = 0.10
    k: int = 8
    tau_waad_quantile: float = 0.30
    tau_delta_quantile: float = 0.70


def _topq_indices(values: np.ndarray, q: float) -> List[int]:
    n = len(values)
    if n == 0:
        return []
    k = max(1, int(np.ceil(n * q)))
    idx = np.argpartition(values, -k)[-k:]
    idx = idx[np.argsort(values[idx])[::-1]]
    return idx.tolist()


@torch.no_grad()
def compute_sets_from_frozen(
    frozen_model,
    tokenizer,
    full_ids_1d: torch.Tensor,
    prompt_len: int,
    hp: ADUHyper,
    ex: Dict[str, Any],
    use_correct_choice_as_sanc: bool = True,
) -> Dict[str, Any]:

    device = frozen_model.device
    full_ids = full_ids_1d.unsqueeze(0).to(device)
    T_total = int(full_ids.shape[-1])
    
    attn = torch.ones_like(full_ids, dtype=torch.long, device=full_ids.device)
    out = frozen_model(full_ids, attention_mask=attn, output_attentions=True, use_cache=False)

    attentions = out.attentions
    L = len(attentions)
    H = int(attentions[0].shape[1])

    # response positions
    R = np.arange(prompt_len, T_total, dtype=np.int64)
    if len(R) < 2:
        return {
            "H_glob": [],
            "T_loc": [],
            "T_glob": [],
            "D": [],
            "I_D": set(),
            "S_anc": set(),
            "prompt_len": prompt_len,
            "T_total": T_total,
        }

    # dist[t,s]=t-s for s<=t
    # compute d^(l,h) = mean_{t in R} sum_{s<=t} A[t,s]*(t-s)
    d_vals = []
    for l in range(L):
        A_l = attentions[l][0].float().cpu().numpy()  # [H,T,T]
        for h in range(H):
            acc = 0.0
            for t in R:
                s_idx = np.arange(0, t + 1, dtype=np.int64)
                dist = (t - s_idx).astype(np.float32)
                acc += float((A_l[h, t, s_idx] * dist).sum())
            d = acc / float(len(R))
            d_vals.append((d, l, h))
    d_vals.sort(key=lambda x: x[0])

    n_loc = max(1, int(np.ceil(hp.frac_local * L * H)))
    n_glob = max(1, int(np.ceil(hp.frac_global * L * H)))
    loc_set = set((l, h) for (_, l, h) in d_vals[:n_loc])
    glob_set = set((l, h) for (_, l, h) in d_vals[-n_glob:])
    H_glob = sorted(list(glob_set))


    A_loc_sum = np.zeros((T_total, T_total), dtype=np.float32)
    A_glob_sum = np.zeros((T_total, T_total), dtype=np.float32)
    cnt_loc, cnt_glob = 0, 0
    for l in range(L):
        A_l = attentions[l][0].float().cpu().numpy()  # [H,T,T]
        for h in range(H):
            if (l, h) in loc_set:
                A_loc_sum += A_l[h]
                cnt_loc += 1
            if (l, h) in glob_set:
                A_glob_sum += A_l[h]
                cnt_glob += 1
    Abar_loc = A_loc_sum / max(1, cnt_loc)
    Abar_glob = A_glob_sum / max(1, cnt_glob)

    # WAAD over response
    WAAD = np.zeros(T_total, dtype=np.float32)
    for t in R:
        s_idx = np.arange(0, t + 1, dtype=np.int64)
        dcap = np.minimum((t - s_idx).astype(np.int32), hp.W).astype(np.float32)
        WAAD[t] = float((Abar_loc[t, s_idx] * dcap).sum())

    Delta = np.zeros(T_total, dtype=np.float32)
    for i in range(len(R) - 1):
        t = int(R[i])
        tp1 = int(R[i + 1])
        Delta[t] = abs(WAAD[t] - WAAD[tp1])

    # T_loc (Preplan): TopQ(Delta on response)
    delta_positions = R[:-1]
    delta_vals = Delta[delta_positions]
    pre_idx = _topq_indices(delta_vals, hp.q_preplan)
    T_loc = [int(delta_positions[i]) for i in pre_idx]

    # T_glob (Anchor): TopQ(FAI over all s)
    FAI = np.zeros(T_total, dtype=np.float32)
    for s in range(T_total):
        future_ts = R[R > s]
        if len(future_ts) == 0:
            FAI[s] = 0.0
        else:
            FAI[s] = float(Abar_glob[future_ts, s].mean())
    # ---- Anchor candidates restricted to prompt positions ----
    prompt_positions = np.arange(0, prompt_len, dtype=np.int64)

    FAI_prompt = np.zeros(prompt_len, dtype=np.float32)
    for s in prompt_positions:
        future_ts = R[R > s]
        if len(future_ts) == 0:
            FAI_prompt[s] = 0.0
        else:
            FAI_prompt[s] = float(Abar_glob[future_ts, s].mean())

    anc_idx = _topq_indices(FAI_prompt, hp.q_anchor)  # indices are in [0, prompt_len)
    T_glob = [int(i) for i in anc_idx]


    # D (Locally dominant anchors)
    # tau_waad: low WAAD threshold on response; tau_delta: high Delta threshold on response
    tau_waad = float(np.quantile(WAAD[R], hp.tau_waad_quantile))
    tau_delta = float(np.quantile(Delta[delta_positions], hp.tau_delta_quantile)) if len(delta_positions) else 0.0

    D = []
    for t in T_glob:
        if WAAD[t] > tau_waad:
            continue
        left = max(0, t - hp.k)
        right = max(0, t)
        if right <= left:
            continue
        if float(Delta[left:right].max()) >= tau_delta:
            D.append(int(t))


    I_D = set()
    T_loc_set = set(T_loc)
    for d in D:
        for u in range(max(0, d - hp.k), d):
            if u in T_loc_set:
                I_D.add(u)


    S_anc = set()
    if use_correct_choice_as_sanc:
        ans = int(ex["answer"])
        correct_choice = ex["choices"][ans]
        choice_ids = tokenizer(correct_choice, add_special_tokens=False)["input_ids"]
        choice_id_set = set(choice_ids)

        full_ids_cpu = full_ids_1d.detach().cpu().tolist()
        for p in T_glob:
            if p < prompt_len and full_ids_cpu[p] in choice_id_set:
                S_anc.add(int(p))
    else:
        for p in T_glob:
            if p < prompt_len:
                S_anc.add(int(p))

    return {
        "H_glob": H_glob,
        "T_loc": T_loc,
        "T_glob": T_glob,
        "D": D,
        "I_D": I_D,
        "S_anc": S_anc,
        "prompt_len": prompt_len,
        "T_total": T_total,
    }


# -----------------------------
# SWLD loss (Eq.8/9/10)
# -----------------------------

def swld_forget_loss(
    updated_model,
    full_ids_1d: torch.Tensor,
    sets: Dict[str, Any],
    lambda_block: float = 1.0,
    gamma_amp: float = 1.5,
    eps: float = 1e-8,
) -> torch.Tensor:

    device = updated_model.device
    full_ids = full_ids_1d.unsqueeze(0).to(device)  # [1,T]
    
    attn = torch.ones_like(full_ids, dtype=torch.long, device=full_ids.device)
    out = updated_model(full_ids, attention_mask=attn, output_attentions=True, use_cache=False)

    attentions = out.attentions  # list[L] of [1,H,T,T]

    H_glob: List[Tuple[int, int]] = sets["H_glob"]
    T_loc: List[int] = sets["T_loc"]
    I_D: Set[int] = sets["I_D"]
    S_anc: Set[int] = sets["S_anc"]


    if len(H_glob) == 0 or len(T_loc) == 0 or len(S_anc) == 0:
        return torch.zeros((), device=device, dtype=torch.float32)

    loss = torch.zeros((), device=device, dtype=torch.float32)


    for t in T_loc:
        # omega_t: Eq.(9) with Sp approximation:
        # Sp_t := (sum over global heads attention mass to S_anc at time t) > eps
        sp_mass = 0.0
        for (l, h) in H_glob:
            A = attentions[l][0, h, t, : t + 1]  # [t+1]
            if A.dtype != torch.float32:
                A = A.float()
            # sum attention to sensitive anchors within prefix
            for s in S_anc:
                if s <= t:
                    sp_mass += float(A[s].detach().cpu().item())
        sp_flag = (sp_mass > 1e-4)

        omega_t = 1.0 + (gamma_amp - 1.0) * (1.0 if (t in I_D and sp_flag) else 0.0)

        for (l, h) in H_glob:
            A = attentions[l][0, h, t, : t + 1]  # [t+1]
            if A.dtype != torch.float32:
                A = A.float()

            # build scale vector (1 - lambda * M_{t,s}); only s in S_anc are suppressed
            scale = torch.ones_like(A)
            if lambda_block != 0.0:
                for s in S_anc:
                    if s <= t:
                        scale[s] = scale[s] * (1.0 - lambda_block)

            Ahat = A * scale
            denom = Ahat.sum().clamp_min(eps)
            Ahat = Ahat / denom

            # KL(A || Ahat) = sum A * (log A - log Ahat)
            A_cl = A.clamp_min(eps)
            Ahat_cl = Ahat.clamp_min(eps)
            kl = (A_cl * (A_cl.log() - Ahat_cl.log())).sum()

            loss = loss + float(omega_t) * kl

    # normalize a bit to keep scale stable
    loss = loss / max(1.0, float(len(T_loc) * len(H_glob)))
    return loss

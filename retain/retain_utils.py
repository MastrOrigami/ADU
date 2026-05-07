# retain/retain_utils.py
# -*- coding: utf-8 -*-
"""
ADU Retain module (L_retain) aligned to Method.txt:

- Identify head categories H_loc / H_glob from attention distance (same idea as forget stage).
- For each sample in general data D_r (here: MMLU full set), compute anchor tokens T_glob via FAI.
- Define omega_t^{retain} = 1 + (gamma_amp - 1) * I{t in T_glob}.
- L_retain = Local Syntax Constraint + Weighted Reasoning Preservation

Local Syntax Constraint:
  sum_{(l,h) in H_loc} KL(A_theta^{(l,h)} || A_ref^{(l,h)})

Weighted Reasoning Preservation:
  sum_{(l,h) in H_glob} sum_t omega_t^{retain} KL(A_theta^{(l,h)} || A_ref^{(l,h)})

Note: We compute KL on causal attention matrices (masked to tril).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, get_dataset_config_names


# -----------------------------
# Tokenization / prompt helpers
# -----------------------------
def is_instruct_model(model_path_or_name: str, tokenizer) -> bool:
    name = (model_path_or_name or "").lower()
    if "instruct" in name or "chat" in name:
        return True
    # some tokenizers expose chat_template
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        return True
    return False


def build_model_input_ids(tokenizer, model_path_or_name: str, raw_prompt_text: str) -> torch.Tensor:
    """
    Mirror forget_utils behavior: if it's an instruct/chat model and tokenizer supports apply_chat_template,
    wrap the prompt into a simple system+user conversation, and add generation prompt.
    """
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
    return tokenizer(raw_prompt_text, return_tensors="pt").input_ids


def _answer_to_letter(ans: Any) -> str:
    if isinstance(ans, int):
        return "ABCD"[int(ans)]
    if isinstance(ans, str):
        a = ans.strip()
        if a.isdigit():
            return "ABCD"[int(a)]
        a = a.upper()
        if a in ["A", "B", "C", "D"]:
            return a
        # sometimes stored as "(A)"
        if len(a) >= 3 and a[0] == "(" and a[2] == ")":
            if a[1] in "ABCD":
                return a[1]
    # fallback
    return "A"


def build_mmlu_prompt(ex: Dict[str, Any]) -> Tuple[str, str]:
    """
    Align to lm-eval config:
      doc_to_text:  "Q: {question}\n(A) {c0} (B) {c1} (C) {c2} (D) {c3}\nA:"
      doc_to_choice: ["(A)","(B)","(C)","(D)"]
      target: answer
    Returns:
      prompt_text, target_choice_text (e.g. "(C)")
    """
    q = (ex.get("question") or "").strip()
    choices = ex.get("choices") or ex.get("options") or ex.get("answers") or None
    if choices is None:
        # some variants store as A/B/C/D fields
        choices = [ex.get("A"), ex.get("B"), ex.get("C"), ex.get("D")]
    if not isinstance(choices, (list, tuple)) or len(choices) < 4:
        # last resort: make placeholders
        choices = list(choices) if choices else ["", "", "", ""]
        while len(choices) < 4:
            choices.append("")
    c0, c1, c2, c3 = [str(x) for x in choices[:4]]

    prompt = f"Q: {q}\n(A) {c0} (B) {c1} (C) {c2} (D) {c3}\nA:"
    letter = _answer_to_letter(ex.get("answer"))
    target = f"({letter})"
    return prompt, target


def load_mmlu_retain(
    dataset_path: str = "cais/mmlu",
    splits: str = "auxiliary_train,dev,val",
    config: str | None = None,
    revision: str | None = None,
) -> List[Dict[str, Any]]:


    requested = [s.strip() for s in splits.split(",") if s.strip()]
    # normalize aliases
    alias = {"val": "validation"}
    requested = [alias.get(s, s) for s in requested]

    # Auto-pick config when missing
    if not config:
        try:
            cfgs = get_dataset_config_names(dataset_path)
        except Exception:
            cfgs = []
        if "all" in cfgs:
            config = "all"
        elif len(cfgs) == 1:
            config = cfgs[0]

    if not config:
        raise ValueError(
            f"Config name is missing for dataset '{dataset_path}'. "
            "For cais/mmlu, pass config like 'all' or a subject, e.g. "
            "load_dataset('cais/mmlu','all'). Set --mmlu_config all."
        )

    ds_dict = load_dataset(dataset_path, config, revision=revision)
    available = set(ds_dict.keys())

    # resolve validation naming
    def resolve(sp: str) -> str | None:
        if sp in available:
            return sp
        if sp == "validation" and "val" in available:
            return "val"
        if sp == "val" and "validation" in available:
            return "validation"
        return None

    examples: List[Dict[str, Any]] = []
    for sp in requested:
        real = resolve(sp)
        if real is None:
            print(f"[WARN] split '{sp}' not found in {dataset_path}/{config}. Available={sorted(available)}")
            continue
        for ex in ds_dict[real]:
            ex = dict(ex)
            if "choices" not in ex:
                if "options" in ex:
                    ex["choices"] = ex["options"]
                elif "answer_choices" in ex:
                    ex["choices"] = ex["answer_choices"]
            examples.append(ex)

    return examples


# -----------------------------
# Head decomposition utilities
# -----------------------------
@dataclass
class RetainHyper:
    frac_local: float = 0.30
    frac_global: float = 0.30
    W: int = 32
    q_anchor: float = 0.40  # Method.txt uses q=0.4 for T_glob
    gamma_amp: float = 1.5  # omega amplification


def _topq_indices(values: np.ndarray, q: float) -> List[int]:
    n = int(values.shape[0])
    if n <= 0:
        return []
    k = max(1, int(math.ceil(n * float(q))))
    # argpartition for top-k
    idx = np.argpartition(values, -k)[-k:]
    # sort descending
    idx = idx[np.argsort(values[idx])[::-1]]
    return [int(i) for i in idx.tolist()]


@torch.no_grad()
def classify_heads_from_ref_attn(
    ref_attentions: List[torch.Tensor],
    prompt_len: int,
    hp: RetainHyper,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Compute head attention distance d^(l,h) on response range R=[prompt_len, T_total),
    then take bottom frac_local as H_loc and top frac_global as H_glob.
    This mirrors the idea used in forget_utils.
    """
    L = len(ref_attentions)
    H = int(ref_attentions[0].shape[1])
    T_total = int(ref_attentions[0].shape[-1])

    R = np.arange(prompt_len, T_total, dtype=np.int64)
    if len(R) == 0:
        R = np.arange(0, T_total, dtype=np.int64)

    d_vals: List[Tuple[float, int, int]] = []
    for l in range(L):
        A_l = ref_attentions[l][0].float().cpu().numpy()  # [H,T,T]
        for h in range(H):
            acc = 0.0
            for t in R:
                s_idx = np.arange(0, t + 1, dtype=np.int64)
                dist = (t - s_idx).astype(np.float32)
                acc += float((A_l[h, t, s_idx] * dist).sum())
            d = acc / float(len(R))
            d_vals.append((d, l, h))

    d_vals.sort(key=lambda x: x[0])
    n_loc = int(math.ceil(hp.frac_local * L * H)) if hp.frac_local and hp.frac_local > 0 else 0
    n_glob = int(math.ceil(hp.frac_global * L * H)) if hp.frac_global and hp.frac_global > 0 else 0

    H_loc = [(l, h) for (_, l, h) in d_vals[:n_loc]] if n_loc > 0 else []
    H_glob = [(l, h) for (_, l, h) in d_vals[-n_glob:]] if n_glob > 0 else []
    return H_loc, H_glob


@torch.no_grad()
def compute_T_glob_from_ref_attn(
    ref_attentions: List[torch.Tensor],
    H_glob: List[Tuple[int, int]],
    q_anchor: float,
    anchor_on_prompt_only: bool = False,
    prompt_len: int = 0,
) -> List[int]:
    """
    Compute anchor tokens T_glob = TopQ(FAI, q=q_anchor).
    FAI_s = mean_{t>s} mean_{(l,h) in H_glob} A^{(l,h)}[t,s]

    If anchor_on_prompt_only=True, restrict s to [0,prompt_len).
    """
    if len(H_glob) == 0:
        return []
    device = ref_attentions[0].device
    T = int(ref_attentions[0].shape[-1])

    # average global attention matrices
    Abar = None
    for (l, h) in H_glob:
        A = ref_attentions[l][0, h]  # [T,T]
        Abar = A if Abar is None else (Abar + A)
    Abar = Abar / float(len(H_glob))

    # compute FAI per token s
    if anchor_on_prompt_only:
        S = list(range(min(prompt_len, T)))
    else:
        S = list(range(T))

    fai = torch.zeros((T,), device=device, dtype=torch.float32)
    for s in S:
        if s + 1 >= T:
            fai[s] = 0.0
        else:
            fai[s] = Abar[(s + 1):, s].mean()

    fai_np = fai.detach().cpu().numpy()
    if anchor_on_prompt_only:
        fai_np = fai_np[: min(prompt_len, T)]
        idx = _topq_indices(fai_np, q_anchor)
        return [int(i) for i in idx]
    idx = _topq_indices(fai_np, q_anchor)
    return [int(i) for i in idx]


# -----------------------------
# L_retain computation
# -----------------------------
def _kl_attn(updated: torch.Tensor, ref: torch.Tensor, row_weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    KL(updated || ref) for a single head attention matrix, masked to causal tril.
    updated/ref: [T,T]
    row_weights: [T] (broadcast to [T,1])
    """
    T = updated.shape[0]
    # causal mask
    mask = torch.tril(torch.ones((T, T), device=updated.device, dtype=updated.dtype))
    Au = updated * mask
    Ar = ref * mask

    # KL with eps; masked positions in Au are 0 so contribute 0.
    logAu = torch.log(Au + eps)
    logAr = torch.log(Ar + eps)
    w = row_weights.view(T, 1).to(dtype=updated.dtype)
    return (w * Au * (logAu - logAr)).sum()


def l_retain(
    updated_attentions: List[torch.Tensor],
    ref_attentions: List[torch.Tensor],
    H_loc: List[Tuple[int, int]],
    H_glob: List[Tuple[int, int]],
    T_glob: List[int],
    gamma_amp: float = 1.5,
    w_syntax: float = 1.0,
    w_reason: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Returns scalar L_retain = L_syntax + L_reason.
    """
    device = updated_attentions[0].device
    T = int(updated_attentions[0].shape[-1])

    omega = torch.ones((T,), device=device, dtype=torch.float32)
    if gamma_amp is not None and gamma_amp > 1.0:
        for t in T_glob:
            if 0 <= int(t) < T:
                omega[int(t)] = float(gamma_amp)

    ones = torch.ones((T,), device=device, dtype=torch.float32)

    loss_syntax = torch.zeros((), device=device, dtype=torch.float32)
    for (l, h) in H_loc:
        Au = updated_attentions[l][0, h].float()
        Ar = ref_attentions[l][0, h].float()
        loss_syntax = loss_syntax + _kl_attn(Au, Ar, ones, eps=eps)

    loss_reason = torch.zeros((), device=device, dtype=torch.float32)
    for (l, h) in H_glob:
        Au = updated_attentions[l][0, h].float()
        Ar = ref_attentions[l][0, h].float()
        loss_reason = loss_reason + _kl_attn(Au, Ar, omega, eps=eps)

    # normalize to keep scale stable
    denom1 = max(1, len(H_loc) * T)
    denom2 = max(1, len(H_glob) * T)
    loss_syntax = loss_syntax / float(denom1)
    loss_reason = loss_reason / float(denom2)
    return float(w_syntax) * loss_syntax + float(w_reason) * loss_reason

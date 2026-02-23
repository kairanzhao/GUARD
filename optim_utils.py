import torch
import random
import re
import numpy as np
import glob
import json
from PIL import Image as PILImage
from io import BytesIO
import pandas as pd
import torchvision.transforms as T
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModelForMaskedLM

from PIL import Image as PILImage
import string
import os
from pathlib import Path
import math
import itertools # We need this for the combinations
import torch.nn.functional as F
from diffusers import DDPMScheduler
# from diffusers import DDIMScheduler
from diffusers.models.attention_processor import AttnProcessor
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple

try:
    from io_utils import read_jsonlines
except ModuleNotFoundError:
    import os; os.chdir("..")
    from io_utils import read_jsonlines


import matplotlib as m
m.use("Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import datasets
from datasets import load_dataset, Dataset, Image

from io_utils import *
# import nltk
# from nltk.corpus import wordnet
# nltk.download('wordnet')
from transformers import pipeline
# NEW: neighbor paraphrase generator
from transformers import PegasusForConditionalGeneration, PegasusTokenizer


# embed_mix_utils.py
import torch
from typing import List, Tuple, Optional

@torch.no_grad()
def _encode_prompt_only(
    pipe,
    prompt: str,
    device: torch.device,
    do_cfg: bool = False,
) -> torch.Tensor:
    """
    Returns ONLY the conditional [1, 77, 768] embeddings for a single prompt.
    Uses pipe.encode_prompt if available; falls back to _encode_prompt.
    """
    if hasattr(pipe, "encode_prompt"):
        pe, _ = pipe.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=None,
        )
    else:
        pe, _ = pipe._encode_prompt(
            prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=None,
        )
    if pe.dim() == 2:  
        pe = pe.unsqueeze(0)
    return pe

def _align_dtype_device(x: torch.Tensor, pipe) -> torch.Tensor:
    text_dtype = getattr(pipe, "text_encoder", None)
    text_dtype = text_dtype.dtype if text_dtype is not None else x.dtype
    return x.to(device=pipe.device, dtype=text_dtype)

def pool_eot(e: torch.Tensor) -> torch.Tensor:
    """Take EOT token as pooled rep. e: [1,77,768] -> [1,768]."""
    return e[:, -1, :]

def cosine_sim(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    a = torch.nn.functional.normalize(a, dim=dim)
    b = torch.nn.functional.normalize(b, dim=dim)
    return (a * b).sum(dim=dim, keepdim=True)


@torch.no_grad()
def safe_slerp_tokenwise(alpha: float, a: torch.Tensor, b: torch.Tensor,
                         min_theta: float = 1e-3, eps: float = 1e-7) -> torch.Tensor:
    """
    Token-wise SLERP over last dim with numeric guards.
    a, b: [B, 77, D] (bf16/fp16/fp32). Returns same dtype as a.
    Uses fp32 for the math; falls back to LERP when angles are tiny.
    """
    assert a.shape == b.shape and a.dim() == 3, f"expected [B,77,D], got {a.shape} vs {b.shape}"
    out_dtype, dev = a.dtype, a.device

    # do math in fp32 to avoid bf16/fp16 underflow
    a32, b32 = a.float(), b.float()

    # normalize per token to get directions
    a_n = a32 / (a32.norm(dim=-1, keepdim=True) + eps)
    b_n = b32 / (b32.norm(dim=-1, keepdim=True) + eps)

    # angle per token
    dot = (a_n * b_n).sum(dim=-1, keepdim=True).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(dot)                              # [B,77,1]
    sin_theta = torch.sin(theta)

    # slerp and lerp in fp32
    s1 = torch.sin((1 - alpha) * theta) / (sin_theta + eps)
    s2 = torch.sin(alpha * theta) / (sin_theta + eps)
    slerp_res = s1 * a32 + s2 * b32
    lerp_res  = (1 - alpha) * a32 + alpha * b32

    # small-angle fallback
    mask_small = (theta < min_theta)
    out = torch.where(mask_small, lerp_res, slerp_res).to(dtype=out_dtype, device=dev)

    # last-resort guard (should stay False if above is correct)
    if not torch.isfinite(out).all():
        out = torch.nan_to_num(out)  # avoid poisoning downstream; log it
        print("[safe_slerp_tokenwise] WARNING: non-finite values encountered; applied nan_to_num.")
    return out

@torch.no_grad()
def most_similar_neighbor_embed(
    pipe,
    gt_prompt: str,
    nb_set: List[str],
    device: torch.device,
) -> Tuple[str, torch.Tensor]:
    """
    Finds the neighbor with highest cosine similarity to GT (pooled EOT),
    returns (neighbor_text, neighbor_embed [1,77,768]).
    """
    gt_e = _encode_prompt_only(pipe, gt_prompt, device=device, do_cfg=False)
    gt_pooled = pool_eot(gt_e)  # [1,768]

    best_nb = None
    best_e = None
    best_sim = -1.0

    for nb in nb_set:
        nb_e = _encode_prompt_only(pipe, nb, device=device, do_cfg=False)
        sim = cosine_sim(pool_eot(nb_e), gt_pooled, dim=-1).item()
        if sim > best_sim:
            best_sim = sim
            best_nb = nb
            best_e = nb_e

    return best_nb, best_e

@torch.no_grad()
def build_negative_embeds(pipe, n_prompt, num_images_per_prompt: int, do_cfg: bool) -> Optional[torch.Tensor]:
    if not do_cfg:
        return None
    if hasattr(pipe, "encode_prompt"):
        _, neg = pipe.encode_prompt(
            prompt=n_prompt,
            device=pipe.device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=True,
            negative_prompt=None,
        )
    else:
        _, neg = pipe._encode_prompt(
            n_prompt,
            device=pipe.device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=True,
            negative_prompt=None,
        )
    return neg


# -------------------------
# Text embeddings utilities
# -------------------------
@torch.no_grad()
def text_embed_openclip(model, tokenizer, prompts: List[str], device):
    tok = tokenizer(prompts)
    if isinstance(tok, torch.Tensor):  # some open_clip tokenizers return Tensor directly
        text = tok.to(device)
    else:
        text = torch.tensor(tok).to(device)
    feats = model.encode_text(text)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats

@torch.no_grad()
def text_embed_sd(pipe, prompts: List[str], device):
    # Use SD text encoder as fallback; pool via mean over sequence
    tok = pipe.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    tok = {k: v.to(device) for k, v in tok.items()}
    outputs = pipe.text_encoder(**tok, output_hidden_states=False)
    last = outputs.last_hidden_state  # [B, seq, hidden]
    pooled = last.mean(dim=1)
    pooled = pooled / pooled.norm(dim=-1, keepdim=True)
    return pooled

@torch.no_grad()
def text_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    # a,b: [D] or [1,D]
    a = a.flatten(0, -1)
    b = b.flatten(0, -1)
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

# ------------------------------------------
# Cross-attention capture (unet attn processor)
# ------------------------------------------
class CrossAttnCapture(AttnProcessor):
    """
    Drop-in attention processor that stores cross-attention probabilities.
    Works with diffusers' UNet2DConditionModel. We only store when encoder_hidden_states is not None (cross-attn).
    """
    def __init__(self, store: Dict, name: str):
        super().__init__()
        self.store = store
        self.name = name

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        # Replicate vanilla AttnProcessor forward with explicit attn_probs capture.
        residual = hidden_states
        batch, sequence, _ = hidden_states.shape

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Head dims
        query = attn.head_to_batch_dim(query)      # [B*H, Q, Dh]
        key   = attn.head_to_batch_dim(key)        # [B*H, K, Dh]
        value = attn.head_to_batch_dim(value)      # [B*H, K, Dh]

        attn_scores = torch.bmm(
            query, key.transpose(-1, -2)
        ) * attn.scale  # [B*H, Q, K]

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = F.softmax(attn_scores.float(), dim=-1).to(value.dtype)  # [B*H, Q, K]

        if is_cross:
            self.store.setdefault(self.name, []).append(attn_probs.detach().cpu())

        hidden_states = torch.bmm(attn_probs, value)  # [B*H, Q, Dh]
        hidden_states = attn.batch_to_head_dim(hidden_states)  # [B, Q, H*Dh]

        # out proj + dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states + residual

def install_capture_processors(unet) -> Tuple[Dict[str, AttnProcessor], Dict[str, List[torch.Tensor]]]:
    prev = dict(unet.attn_processors)  # save original processors
    store: Dict[str, List[torch.Tensor]] = {}
    new = {name: CrossAttnCapture(store, name) for name in prev.keys()}
    unet.set_attn_processor(new)
    return prev, store

def uninstall_processors(unet, prev):
    unet.set_attn_processor(prev)

# -------------------------
# Attention map aggregation
# -------------------------
def aggregate_attn(store, target_side=32):
    """
    Aggregate cross-attention only from blocks whose spatial resolution is target_side x target_side.
    Each captured attn map has shape [B*H, Q, K]; Q should be a perfect square.
    
    Return:
      - spatial_vec: np.ndarray of shape [Q]   (sum over tokens K, mean over heads and layers)
      - token_mass:  np.ndarray of shape [K]   (sum over spatial Q, mean over heads and layers)
      - Q, K        : query and key lengths
    """
    spatial_list = []
    token_list   = []
    K_final = None
    target_Q = target_side * target_side

    # collect all attn2 cross-attn maps at the requested resolution
    for name, lst in store.items():
        if "attn2" not in name:
            continue
        for m in lst:
            # m: [B*H, Q, K] (CPU, detached)
            m = m.float()
            Q, K = m.shape[-2], m.shape[-1]

            # CHANGED: filter by target spatial size
            if Q != target_Q:
                continue

            if K_final is None:
                K_final = int(K)

            # mean over heads (B==1 so B*H==H)
            m = m.mean(dim=0)  # [Q, K]

            token_i   = m.sum(dim=0).cpu().numpy()  # [K]
            spatial_i = m.sum(dim=1).cpu().numpy()  # [Q]

            token_list.append(token_i)
            spatial_list.append(spatial_i)

    if len(spatial_list) == 0:
        # nothing matched; return None so caller can handle it
        return None

    # normalize each, then average (more robust to scale)
    spatial_mat = np.stack([v / (np.linalg.norm(v) + 1e-8) for v in spatial_list], axis=0)  # [L, Q]
    token_mat   = np.stack([v / (np.linalg.norm(v) + 1e-8) for v in token_list],   axis=0)  # [L, K]

    spatial_vec = spatial_mat.mean(axis=0)
    spatial_vec = spatial_vec / (np.linalg.norm(spatial_vec) + 1e-8)

    token_mass  = token_mat.mean(axis=0)
    token_mass  = token_mass / (np.linalg.norm(token_mass) + 1e-8)

    return spatial_vec, token_mass, int(target_Q), int(K_final)

def topk_iou(a: np.ndarray, b: np.ndarray, topk_percent: int = 10) -> float:
    k = max(1, int(math.ceil(len(a) * topk_percent / 100.0)))
    idx_a = np.argpartition(a, -k)[-k:]
    idx_b = np.argpartition(b, -k)[-k:]
    set_a, set_b = set(idx_a.tolist()), set(idx_b.tolist())
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / max(union, 1)

def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))

def save_heatmap(vec: np.ndarray, out_path: str):
    side = int(math.sqrt(len(vec)))
    if side * side != len(vec):
        side = int(np.sqrt(len(vec)))
        side = max(1, side)
        target = side * side
        vec = vec[:target]
    img = vec.reshape(side, side)
    plt.figure()
    plt.imshow(img) 
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.05)
    plt.close()

def save_multi_heatmap(spatial_gt, neigh_spatials, labels, out_path, side):
    """
    spatial_gt: np.array [Q]
    neigh_spatials: list of np.array [Q]
    labels: list[str], same length as neigh_spatials; each label may contain a newline,
            e.g. 'nb0 | cos=0.972\\nshort neighbor prompt'
    out_path: file to save
    side: int (e.g., 32)
    """
    import numpy as np
    import math
    import matplotlib.pyplot as plt

    panels = [spatial_gt] + list(neigh_spatials)
    titles = ["GT"] + list(labels)

    all_vals = np.concatenate([p for p in panels])
    vmin, vmax = np.percentile(all_vals, 5), np.percentile(all_vals, 95)

    n = len(panels)
    ncols = min(6, n)
    nrows = math.ceil(n / ncols)
    plt.figure(figsize=(2.2 * ncols, 2.4 * nrows))

    for idx, (vec, title) in enumerate(zip(panels, titles), start=1):
        img = vec.reshape(side, side)
        ax = plt.subplot(nrows, ncols, idx)
        ax.imshow(img, vmin=vmin, vmax=vmax)
        ax.axis("off")

        if "\n" in title:
            line1, line2 = title.split("\n", 1)
            if len(line2) > 50:
                line2 = line2[:47] + "…"
            ax.set_title(f"{line1}\n{line2}", fontsize=8)
        else:
            if len(title) > 60:
                title = title[:57] + "…"
            ax.set_title(title, fontsize=8)

    plt.tight_layout(pad=0.2)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close()

def capture_spatial_for_timestep(pipe, latents, prompt_embeds, t, target_side):
    prev_proc, store = install_capture_processors(pipe.unet)
    _ = pipe.unet(latents, t, encoder_hidden_states=prompt_embeds).sample
    uninstall_processors(pipe.unet, prev_proc)
    agg = aggregate_attn(store, target_side=target_side)
    return agg 
def save_mega_panel(gt_spatials_by_t, all_nb_spatials_by_t, col_labels, ts, out_path, side):
    """
    gt_spatials_by_t: list[np.array(Q)] length = len(ts)
    all_nb_spatials_by_t: list over neighbors; each is list[np.array(Q) or None], length = len(ts)
    col_labels: ["GT", "nb0 | cos=...", "nb1 | cos=...", ...]  (titles for each column)
    ts: list[int] timesteps
    side: int (e.g., 32)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    rows = len(ts)
    cols = 1 + len(all_nb_spatials_by_t)

    vals = []
    for vec in gt_spatials_by_t:
        if vec is not None: vals.append(vec)
    for nb_row in all_nb_spatials_by_t:
        for vec in nb_row:
            if vec is not None: vals.append(vec)
    if not vals:
        return
    all_vals = np.concatenate(vals)
    vmin, vmax = np.percentile(all_vals, 5), np.percentile(all_vals, 95)

    plt.figure(figsize=(2.2 * cols, 2.2 * rows), constrained_layout=False)
    plt.subplots_adjust(left=0.22, right=0.98, top=0.92, bottom=0.06, wspace=0.05, hspace=0.2)

    def _draw(ax, vec):
        if vec is None:
            ax.axis("off")
            ax.text(0.5, 0.5, "—", ha="center", va="center", fontsize=10)
            return
        img = vec.reshape(side, side)
        ax.imshow(img, vmin=vmin, vmax=vmax, cmap="gray", interpolation="nearest")
        ax.axis("off")

    for r, t in enumerate(ts):
        ax = plt.subplot(rows, cols, r * cols + 1)
        _draw(ax, gt_spatials_by_t[r])
        ax.set_title("GT", fontsize=9)
        ax.text(-0.12, 0.5, f"t={int(t)}", transform=ax.transAxes,
                fontsize=9, va="center", ha="right")

        for c in range(1, cols):
            ax = plt.subplot(rows, cols, r * cols + c + 1)
            vec = all_nb_spatials_by_t[c - 1][r]  # neighbor (c-1), timestep r
            _draw(ax, vec)
            ax.set_title(col_labels[c], fontsize=8)

    # optional caption
    plt.suptitle("Rows: timesteps  |  Cols: GT + neighbors", fontsize=10, y=0.995)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close()

def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


_cached_clean_vocab = None

def generate_random_word_prompt(tokenizer, num_words=8):
    """
    Generates a prompt consisting of a number of random words sampled from 
    the tokenizer's vocabulary.

    Args:
        tokenizer: The tokenizer from the diffusion model pipeline (e.g., pipe.tokenizer).
        num_words (int): The number of random words to include in the prompt.

    Returns:
        str: A space-separated string of random words.
    """
    global _cached_clean_vocab

    if _cached_clean_vocab is None:
        print("Building and cleaning vocabulary for prompt generation...")
        vocab = tokenizer.get_vocab()
        
        clean_vocab = []
        for token in vocab.keys():
            cleaned_token = token.replace("</w>", "")
            if re.match("^[a-zA-Z]{3,}$", cleaned_token):
                clean_vocab.append(cleaned_token)
        
        _cached_clean_vocab = clean_vocab
        print(f"Found {len(_cached_clean_vocab)} usable words in vocabulary for random sampling.")

    sampled_words = random.sample(_cached_clean_vocab, num_words)

    return " ".join(sampled_words)


def process_in_batches(process_fn, data, batch_size, desc=""):
    """Applies a processing function to data in batches."""
    results = []
    num_batches = math.ceil(len(data) / batch_size)
    
    for i in tqdm(range(num_batches), desc=desc):
        batch_data = data[i * batch_size : (i + 1) * batch_size]
        if not batch_data:
            continue
        batch_result = process_fn(batch_data)
        results.append(batch_result.cpu()) 
    
    return torch.cat(results, dim=0)

def _encode_latents(pipe, pil_batch):
    imgs = pipe.image_processor.preprocess(pil_batch).to(device=pipe.device, dtype=pipe.unet.dtype)
    with torch.no_grad():
        latents = pipe.vae.encode(imgs).latent_dist.mean * pipe.vae.config.scaling_factor
    return latents  # [B,4,H/8,W/8]

def _prompt_embeds(pipe, prompts):
    tok = pipe.tokenizer(
        prompts, padding="max_length", truncation=True,
        max_length=pipe.tokenizer.model_max_length, return_tensors="pt").to(pipe.device)
    with torch.no_grad():
        pe = pipe.text_encoder(**tok)[0]
    return pe.to(dtype=pipe.unet.dtype)

def diffusion_pair_loss(pipe, scheduler, latents, prompt_embeds, K=64, M=1,
                        weighting="none", common_ts=None, common_eps=None):
    B = latents.size(0)
    T = scheduler.num_train_timesteps
    device = latents.device
    pred_type = getattr(pipe.unet.config, "prediction_type", getattr(scheduler.config, "prediction_type", "epsilon"))

    if common_ts is None:
        ts = torch.randint(0, T, (K,), device=device)
    else:
        ts = common_ts
    eps_list = []
    for _ in range(M):
        if common_eps is None:
            eps_list.append(torch.randn((K,)+latents.shape[1:], device=device, dtype=latents.dtype))
        else:
            eps_list.append(common_eps)

    total = torch.zeros(B, device=device, dtype=latents.dtype)
    for m in range(M):
        eps = eps_list[m]
        for k in range(K):
            t = ts[k].repeat(B)
            # build z_t
            noise = eps[k].unsqueeze(0).expand_as(latents)
            z_t = scheduler.add_noise(latents, noise, t)

            model_in = scheduler.scale_model_input(z_t, t)
            with torch.no_grad():
                pred = pipe.unet(model_in, t, encoder_hidden_states=prompt_embeds).sample

            if pred_type == "epsilon":
                target = noise
            elif "v" in pred_type:  # v_prediction
                target = scheduler.get_velocity(latents, noise, t)
            else:  # x0
                target = latents

            mse = F.mse_loss(pred, target, reduction="none")
            mse = mse.view(B, -1).mean(dim=1)

            if weighting != "none":
                alphas_cumprod = scheduler.alphas_cumprod.to(device=device, dtype=latents.dtype)
                a = alphas_cumprod[t]
                snr = a / (1 - a)
                w = snr if weighting == "snr" else (snr.clamp(max=5.0) / (snr + 1.0))  # example gamma=5.0
                mse = mse * w

            total += mse
    return (total / (K * M)).detach().float()  # [B]

@torch.no_grad()
def generate_neighbors_from_algorithm(
    text: str,
    n: int, 
    p_replacements: float,  
    top_k_for_combinations: int = 3
):
    """
    Implements Algorithm 1.
    If n > 0, generates 'n' neighbors using the lock-step method.
    If n = -1, generates all possible combinations of the top 'k' candidates.
    """

    MODEL_NAME = "roberta-base"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    
    words = text.split()
    L = len(words)
    m = max(1, int(L * p_replacements))

    tokenized_prompt = tokenizer(text, return_tensors="pt").to(DEVICE)
    input_ids = tokenized_prompt["input_ids"]

    positional_swap_scores = []
    for i in range(L):
        word_ids = tokenized_prompt.word_ids(batch_index=0)
        if i not in word_ids: continue
        token_idx = word_ids.index(i)
        
        embeddings = model.roberta.embeddings.word_embeddings(input_ids)
        original_token_id = input_ids[0, token_idx].item()
        mask_token_id = tokenizer.mask_token_id
        mask_embedding = model.roberta.embeddings.word_embeddings(torch.tensor([[mask_token_id]], device=DEVICE))
        corrupted_embeddings = embeddings.clone()
        corrupted_embeddings[0, token_idx] = mask_embedding

        outputs = model(inputs_embeds=corrupted_embeddings)
        logits = outputs.logits[0, token_idx]
        probabilities = torch.softmax(logits, dim=-1)
        p_original = probabilities[original_token_id].item()
        if p_original >= 1.0: continue

        top_vals, top_indices = torch.topk(probabilities, k=2)
        best_candidate_id = top_indices[0].item()
        if best_candidate_id == original_token_id: best_candidate_id = top_indices[1].item()
        
        p_theta_tilde = probabilities[best_candidate_id].item()
        pswap = p_theta_tilde / (1 - p_original)
        positional_swap_scores.append({"word_idx": i, "score": pswap})

    positional_swap_scores.sort(key=lambda x: x["score"], reverse=True)
    indices_to_replace = [d["word_idx"] for d in positional_swap_scores[:m]]

    num_candidates_to_fetch = n if n > 0 else top_k_for_combinations
    
    replacements_for_top_m = {}
    for word_idx in indices_to_replace:
        token_idx = tokenized_prompt.word_ids(batch_index=0).index(word_idx)
        original_token_id = input_ids[0, token_idx].item()
        
        embeddings = model.roberta.embeddings.word_embeddings(input_ids)
        mask_embedding = model.roberta.embeddings.word_embeddings(torch.tensor([[tokenizer.mask_token_id]], device=DEVICE))
        corrupted_embeddings = embeddings.clone()
        corrupted_embeddings[0, token_idx] = mask_embedding
        outputs = model(inputs_embeds=corrupted_embeddings)
        logits = outputs.logits[0, token_idx]
        probabilities = torch.softmax(logits, dim=-1)

        top_k_vals, top_k_indices = torch.topk(probabilities, k=num_candidates_to_fetch + 1)
        candidates = []
        for k in range(num_candidates_to_fetch + 1):
            candidate_id = top_k_indices[k].item()
            if candidate_id == original_token_id: continue
            candidates.append(tokenizer.decode(candidate_id).strip())
            if len(candidates) == num_candidates_to_fetch: break
        replacements_for_top_m[word_idx] = candidates

    final_neighbors = []
    if n > 0:
        for i in range(n):
            new_words = list(words)
            for word_idx in indices_to_replace:
                if i < len(replacements_for_top_m.get(word_idx, [])):
                    replacement = replacements_for_top_m[word_idx][i]
                    new_words[word_idx] = replacement
            final_neighbors.append(" ".join(new_words))
    
    elif n == -1:
        candidate_lists = [replacements_for_top_m[idx] for idx in indices_to_replace]
        
        num_combinations = len(candidate_lists[0]) ** len(candidate_lists)
        if num_combinations > 1000:
             print(f"Warning: This will generate {num_combinations} neighbors, which might be very large. Proceeding...")

        for combination in itertools.product(*candidate_lists):
            new_words = list(words)
            for i, replacement in enumerate(combination):
                word_idx = indices_to_replace[i]
                new_words[word_idx] = replacement
            final_neighbors.append(" ".join(new_words))
            
    return final_neighbors, m

def get_single_synonym(word):
    """
    Returns a single synonym for 'word' excluding the exact same string,
    or None if no suitable synonym is found.
    """
    lookup_key = word

    for syn in wordnet.synsets(lookup_key):
        for lemma in syn.lemmas():
            lemma_name = lemma.name()
            if lemma_name.lower() != word.lower():
                print(f"Found synonym for '{word}': {lemma_name}")
                return lemma_name.replace("_", " ")
    print(f"No synonym found for '{word}'")
    return None

def fill_masked_token(fill_mask, text, token_idx):
    """
    Given a whitespace-split text, replaces token_idx with '[MASK]',
    then uses the fill_mask pipeline to fill it with the model's best guess.
    Returns the newly augmented sentence (string).
    
    Note: If the mask-filler returns multiple tokens, your final text might
    end up with extra tokens in that position. Also, if there's punctuation
    etc., splitting may not perfectly align with subword tokens. 
    """
    tokens = text.split()
    if token_idx < 0 or token_idx >= len(tokens):
        return text
    tokens = tokens[:token_idx] + ["[MASK]"] + tokens[token_idx:]
    masked_text = " ".join(tokens)
    print(f"Masked text: {masked_text}")
    
    candidates = fill_mask(masked_text, top_k=10)  # or 50, etc.
    if not candidates:
        return text

    best_sequence = candidates[0]["sequence"]
    print(f"Best sequence: {best_sequence}")
    new_prompt = " ".join(best_sequence.split())

    return new_prompt

def prompt_augmentation_synonym(prompt, pipe, significance_threshold=None, top_k=None):
    """
    1. Computes token significance scores for `prompt`.
    2. Identifies "important" tokens (above the threshold).
    3. Replaces those tokens with synonyms (or removes them if they contain '<').
    4. Returns the new, augmented prompt string.
    """

    device = pipe._execution_device  
    token_grads = pipe.get_text_cond_grad(
        prompt=prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        num_images_per_prompt=4,
        target_steps=list(range(10)) 
    )

    prompt_token_ids = pipe.tokenizer.encode(prompt)
    prompt_token_ids = prompt_token_ids[1:-1]
    prompt_token_ids = prompt_token_ids[:75]  
    token_grads = token_grads[1 : (1 + len(prompt_token_ids))]
    token_grads = token_grads.cpu().tolist()
    decoded_tokens = [pipe.tokenizer.decode(tid) for tid in prompt_token_ids]

    num_tokens = len(decoded_tokens)
    token_indices = list(range(num_tokens))

    if top_k is not None and top_k > 0:
        print(f'replace the top {top_k} tokens by gradient')
        sorted_by_grad = sorted(
            token_indices,
            key=lambda i: token_grads[i],
            reverse=True
        )
        important_indices = set(sorted_by_grad[:top_k])
    elif significance_threshold is not None:
        print(f'replace tokens with gradient > {significance_threshold}')
        important_indices = set(
            i for i, grad_val in enumerate(token_grads)
            if grad_val > significance_threshold
        )
    else:
        threshold = np.percentile(token_grads, 95) 
        important_indices = { i for i, g in enumerate(token_grads) if g >= threshold }
        print(f'replace top 5% pencentile tokens by gradient > {threshold}, {len(important_indices)} tokens in total')

    fill_mask = pipeline("fill-mask", model="bert-base-uncased")
    new_tokens = " ".join(decoded_tokens)
    for idx in sorted(important_indices, reverse=True):
        new_tokens = fill_masked_token(fill_mask, new_tokens, idx)


    new_prompt = new_tokens
    print(f"Original prompt: {prompt}")
    print(f"Augmented prompt: {new_prompt}")
    return new_prompt

### credit to https://github.com/somepago/DCR
def insert_rand_word(sentence, word):
    sent_list = sentence.split(" ")
    sent_list.insert(random.randint(0, len(sent_list)), word)
    new_sent = " ".join(sent_list)
    return new_sent

def insert_prefix(sentence, word):
    """
    Adds a word at the beginning of the sentence.
    """
    return word + " " + sentence

def insert_suffix(sentence, word):
    """
    Adds a word at the end of the sentence.
    """
    return sentence + " " + word


def prompt_augmentation(prompt, aug_style, tokenizer=None, repeat_num=4):
    if aug_style == "rand_numb_add":
        for i in range(repeat_num):
            randnum = np.random.choice(100000)
            prompt = insert_rand_word(prompt, str(randnum))
    elif aug_style == "rand_word_add":
        for i in range(repeat_num):
            randword = tokenizer.decode(list(np.random.randint(49400, size=1)))
            prompt = insert_rand_word(prompt, randword)
    elif aug_style == "word_add":
        for i in range(repeat_num):
            word = "DEMEMORIZE"
            prompt = insert_rand_word(prompt, word)
    elif aug_style == "rand_word_add_shuffle":
        for i in range(repeat_num):
            randword = tokenizer.decode(list(np.random.randint(49400, size=1)))
            #shuffle the characters in the word
            randword = ''.join(random.sample(randword, len(randword)))
            prompt = insert_rand_word(prompt, randword)
    elif aug_style == "rand_word_repeat":
        wordlist = prompt.split(" ")
        for i in range(repeat_num):
            randword = np.random.choice(wordlist)
            prompt = insert_rand_word(prompt, randword)
    elif aug_style == "rand_word_prefix":
        for i in range(repeat_num):
            randword = tokenizer.decode(list(np.random.randint(49400, size=1)))
            prompt = insert_prefix(prompt, randword)
    elif aug_style == "rand_word_suffix":
        for i in range(repeat_num):
            randword = tokenizer.decode(list(np.random.randint(49400, size=1)))
            prompt = insert_suffix(prompt, randword)
    elif aug_style == "rand_word_suffix_shuffle":
        for i in range(repeat_num):
            randword = tokenizer.decode(list(np.random.randint(49400, size=1)))
            randword = ''.join(random.sample(randword, len(randword)))
            prompt = insert_suffix(prompt, randword)
    elif aug_style == "rand_word_suffix_repeat_shuffle":
        randword = tokenizer.decode(list(np.random.randint(49400, size=1)))
        randword = ''.join(random.sample(randword, len(randword)))
        for i in range(repeat_num):
            prompt = insert_suffix(prompt, randword)

    elif aug_style == "add_mem_randloc":
        for i in range(repeat_num):
            prompt = insert_rand_word(prompt, "memorization")
    elif aug_style == "neighbor_replace":
        pass
    else:
        raise Exception("This style of prompt augmnentation is not written")
    return prompt


def get_dataset(dataset_name, pipe=None): 
    if "jsonl" in dataset_name:
        dataset = load_jsonlines(dataset_name)
        prompt_key = "caption"
    elif dataset_name == "random":
        dataset = []
        for _ in range(2000):
            k = random.randrange(pipe.tokenizer.model_max_length)
            rand_tokens = random.sample(range(pipe.tokenizer.vocab_size), k)
            dataset.append({"Prompt": pipe.tokenizer.decode(rand_tokens)})
        prompt_key = "Prompt"
    elif dataset_name == "ChristophSchuhmann/MS_COCO_2017_URL_TEXT":
        dataset = load_dataset(dataset_name)["train"]
        prompt_key = "TEXT"
    elif dataset_name == "Gustavosta/Stable-Diffusion-Prompts":
        dataset = load_dataset(dataset_name)["test"]
        prompt_key = "Prompt"
    
    elif dataset_name == "laion_10k":
        dataset, prompt_key = get_dataset_finetune_v2(dataset_name)
    elif dataset_name == "laion_aesthetics":
        dataset, prompt_key = get_dataset_finetune_v2(dataset_name)
    elif dataset_name == "memorized_images":
        dataset, prompt_key = get_dataset_finetune(dataset_name)
    else:
        raise NotImplementedError

    return dataset, prompt_key

def _is_main_process():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            return torch.distributed.get_rank() == 0
        except RuntimeError:
            pass
    try:
        from accelerate.state import AcceleratorState
        return AcceleratorState().is_main_process
    except Exception:
        return int(os.environ.get("RANK", "0")) == 0

def get_dataset_finetune(
    dataset_name, non_mem_dataset=None, end=None, repeats=1, non_mem_ratio=0, args=None,
    accelerator=None, 
    tokenizer=None
):

    if "groundtruth" in dataset_name:
        dataset = load_jsonlines(f"{args.data_path}/{dataset_name}/{dataset_name}.jsonl")
        prompt_key = "caption"

        return dataset, prompt_key
    
    elif dataset_name == "sdv1_500_mem" or dataset_name == "sdv1_500_mem_RL":
        images_dir = f"{args.data_path}/sdv1_500_mem_groundtruth/gt_images"
        captions_file = f"{args.data_path}/sdv1_500_mem_groundtruth/sdv1_500_mem_groundtruth.jsonl"
        captions_data = {}
        with open(captions_file, "r") as f:
            for line in f:
                entry = json.loads(line)
                captions_data[entry["index"]] = entry["caption"]  

        indices = list(captions_data.keys())
        
        shuffled_captions = list(captions_data.values())
        random.shuffle(shuffled_captions)
        shuffled_caption_data = dict(zip(indices, shuffled_captions)) 

        # -----------------------
        # PHASE 1: Deduplication
        # -----------------------
        deduped_data = []
        for curr_index in captions_data:
            image_files = glob.glob(f"{images_dir}/{curr_index}/*.png")
            image_files.sort()
            
            if not image_files:
                print(f"Warning: No image found for index {curr_index}.")
                continue
            elif len(image_files) > 1:
                continue

            # Assign captions based on dataset type
            if dataset_name == "sdv1_500_mem_RL":
                caption = shuffled_caption_data[curr_index]  
            elif dataset_name == "sdv1_500_mem": 
                caption = captions_data[curr_index]
            
            deduped_data.append({
                "index": curr_index,
                "image": image_files[0],
                "caption": caption,
                "mask": 1,
            })

        print(f"Deduplication complete: Collected {len(deduped_data)} examples from captions data.")

        # -----------------------
        # PHASE 2: SSC-D Selection
        # -----------------------
        if args.topk is not None:
            path = './results/inference_memNone_nonmem_None_gen_sdv1_500_mem_groundtruth_start0_end500_no_mitigation_s0'
            results = pd.read_pickle(f"{path}/results_with_scores.pkl")
            sscd_scores = np.array(results['sscd_scores'])
            
            if args.topk > 1.0:
                topk_indices = np.argsort(sscd_scores)[-args.topk:]
                print(f"Top-k SSCD scores: {sscd_scores[topk_indices]}")
                deduped_data = [deduped_data[i] for i in topk_indices]
            
            elif args.topk <= 1.0:
                print(f"Selecting examples with SSCD score > {args.topk}")
                deduped_data = [deduped_data[i] for i in range(len(deduped_data)) if sscd_scores[i] > args.topk]
        # -----------------------
        # PHASE 3: Build Training Data (+ neighbors)
        # -----------------------
        device = accelerator.device if accelerator is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        use_neighbors = getattr(args, "prompt_aug_style", None) == "neighbor_replace"
        num_neighbors = int(getattr(args, "num_neighbors", 1))
        neighbor_mask = 0 if args.mem_strategy == "NGplus" else 1
        paraphrase_model_name = "tuner007/pegasus_paraphrase"

        if use_neighbors and not hasattr(get_dataset_finetune, "_pegasus"):
            print(f"[neighbor_replace] Loading paraphrase model: {paraphrase_model_name} on {device}")
            _tok = PegasusTokenizer.from_pretrained(paraphrase_model_name)
            _mdl = PegasusForConditionalGeneration.from_pretrained(paraphrase_model_name).to(device)
            get_dataset_finetune._pegasus = (_tok, _mdl)

        def _paraphrase(sentence: str, k: int):
            if not use_neighbors or k <= 0:
                return []
            tok, mdl = get_dataset_finetune._pegasus
            batch = tok([sentence], truncation=True, padding="longest", return_tensors="pt").to(device)
            with torch.no_grad():
                out = mdl.generate(
                    **batch,
                    max_length=60,
                    do_sample=True,             
                    num_beams=min(10, max(1, k)),
                    num_return_sequences=k,
                    temperature=2.5
                )
            cands = tok.batch_decode(out, skip_special_tokens=True)
            seen, uniq = set(), []
            s_l = sentence.strip().lower()
            for t in cands:
                t = t.strip()
                if not t: 
                    continue
                if t.lower() == s_l:
                    continue
                if t not in seen:
                    uniq.append(t)
                    seen.add(t)
            return uniq[:k]

        # -----------------------
        # PHASE 4: Build Training Data
        # -----------------------
        all_data = {"image": [], "text": [], "mask": []}
        added_neighbors = 0
        for entry in deduped_data:
            all_data["image"].append(entry["image"])
            all_data["text"].append(entry["caption"])
            all_data["mask"].append(entry["mask"])

            if use_neighbors and num_neighbors > 0:
                try:
                    nb_texts = _paraphrase(entry["caption"], num_neighbors)
                except Exception as e:
                    print(f"[neighbor_replace] paraphrase failed for index {entry.get('index','?')}: {e}")
                    nb_texts = []

                for nb in nb_texts:
                    all_data["image"].append(entry["image"])
                    all_data["text"].append(nb)
                    all_data["mask"].append(neighbor_mask)
                    added_neighbors += 1

        print(
            f"Collected {len(all_data['image'])} total examples for Dataset {dataset_name}."
            + (f" Added {added_neighbors} neighbor examples (mask={neighbor_mask})." if use_neighbors else "")
        )

    else:  
        all_files = glob.glob(f"{dataset_name}/*.jpg")
        all_files.sort()
        print('check1: mem_images: len_all_files:', len(all_files))

        if end is not None:
            all_files = all_files[:end]

        all_data = {"image": [], "text": [], "mask": []}
        for file in all_files:
            f = open(file.replace("jpg", "txt"), "r")
            captions = f.read()

            all_data["image"].append(file)
            all_data["text"].append(captions)
            all_data["mask"].append(1)

        all_data["image"] = all_data["image"] * repeats
        all_data["text"] = all_data["text"] * repeats
        all_data["mask"] = all_data["mask"] * repeats
        mem_len = len(all_data["image"])
        print(f"check: total mem images (each repeats {repeats} times): {mem_len}")

    if non_mem_dataset is not None:
        if non_mem_dataset == 'laion_10k':
            with open(f"{args.data_path}/{non_mem_dataset}/laion_combined_captions.json", "r") as f:
                captions_data = json.load(f)

            non_mem_files = glob.glob(f"{args.data_path}/{non_mem_dataset}/images_large/*.png")
            non_mem_files.sort()

            if args.mem is not None:
                non_mem_files = remain_files
                
                remain_data = {"image": [], "text": []}
                for file in non_mem_files:
                    filename = file.split("/")[-1]
                    for key in captions_data:
                        if key.endswith(f"images_large/{filename}"):
                            caption = captions_data[key][0]
                            remain_data["image"].append(file)
                            remain_data["text"].append(caption)
                            break

            else:
                for file in non_mem_files:
                    if len(all_data["image"]) >= mem_len * (1 + non_mem_ratio):
                        break

                    filename = file.split("/")[-1]

                    for key in captions_data:
                        if key.endswith(f"images_large/{filename}"):
                            caption = captions_data[key][0]
                            all_data["image"].append(file)
                            all_data["text"].append(caption)
                            break
        
        elif non_mem_dataset == 'laion_aesthetics':
            print("Adding LAION aesthetics 6.5+ dataset...")
            images_dir = f"{args.data_path}/laion_aesthetics_6.5plus/preprocessed/images"
            captions_file = f"{args.data_path}/laion_aesthetics_6.5plus/preprocessed/captions.json"

            with open(captions_file, "r") as f:
                captions_data = json.load(f)

            non_mem_files = glob.glob(f"{images_dir}/*.png")
            non_mem_files.sort()

            if dataset_name == "sdv1_500_mem" or dataset_name == "sdv1_500_mem_RL":
                print(f"Number of available non-mem images: {len(non_mem_files)} checkkkk")
                
                remain_data = {"image": [], "text": [], "mask": []}
                for file in non_mem_files:
                    filename = os.path.basename(file)
                    if filename in captions_data:
                        caption = captions_data[filename]
                        remain_data["image"].append(file)
                        remain_data["text"].append(caption)
                        remain_data["mask"].append(0)
                    else:
                        print(f"Warning: Caption for {filename} not found.")
                        continue                

            else: 
                print(f"Number of available non-mem images: {len(non_mem_files)}")
                

                remain_data = {"image": [], "text": [], "mask": []}
                for file in non_mem_files:
                    if args.non_mem_ratio > 0:
                        if len(all_data["image"]) >= mem_len * (1 + non_mem_ratio): 
                            break

                    filename = os.path.basename(file)
                    if filename in captions_data:
                        caption = captions_data[filename]
                        remain_data["image"].append(file)
                        remain_data["text"].append(caption)
                        remain_data["mask"].append(0)
                    else:
                        print(f"Warning: Caption for {filename} not found.")
                        continue

        else:
            all_files = glob.glob(f"{non_mem_dataset}/*.jpg")
            all_files.sort()
            print('len_all_files:', len(all_files))

            for file in all_files:
                if len(all_data["image"]) >= mem_len * (1 + non_mem_ratio):
                    break

                f = open(file.replace("jpg", "txt"), "r")
                captions = f.read()

                all_data["image"].append(file)
                all_data["text"].append(captions)

            all_data["image"] = all_data["image"][: int(mem_len * (1 + non_mem_ratio))]
            all_data["text"] = all_data["text"][: int(mem_len * (1 + non_mem_ratio))]
            print(f"check: mem + non-mem : {len(all_data['image'])}")
        
        all_data["image"].extend(remain_data["image"])
        all_data["text"].extend(remain_data["text"])
        all_data["mask"].extend(remain_data["mask"])
        
    dataset = Dataset.from_dict(all_data).cast_column("image", datasets.Image())
    prompt_key = "text"
    
    return dataset, prompt_key

def get_dataset_finetune_v2(
    dataset_name, non_mem_dataset=None, end=None, repeats=1, non_mem_ratio=0, args=None
):
    if "groundtruth" in dataset_name:
        dataset = load_jsonlines(f"{dataset_name}/{dataset_name}.jsonl")
        prompt_key = "caption"
    else:
        if dataset_name == "laion_10k":
            with open(f"{args.data_path}/{dataset_name}/laion_combined_captions.json", "r") as f:
                captions_data = json.load(f)

            # Get image file paths
            all_files = glob.glob(f"{args.data_path}/{dataset_name}/images_large/*.png")
            all_files.sort()

            if end is not None:
                all_files = all_files[:end]

            # Prepare the dataset
            all_data = {"image": [], "text": []}
            for file in all_files:
                filename = file.split("/")[-1]

                for key in captions_data:
                    if key.endswith(f"images_large/{filename}"):
                        caption = captions_data[key][0]
                        all_data["image"].append(file)
                        all_data["text"].append(caption)
                        break
        
        elif dataset_name == "laion_aesthetics":
            images_dir = f"{args.data_path}/laion_aesthetics_6.5plus/preprocessed/images"
            captions_file = f"{args.data_path}/laion_aesthetics_6.5plus/preprocessed/captions.json"

            with open(captions_file, "r") as f:
                captions_data = json.load(f)

            # Get image file paths
            all_files = glob.glob(f"{images_dir}/*.png")
            all_files.sort()

            if end is not None:
                all_files = all_files[:end]

            # Prepare the dataset
            all_data = {"image": [], "text": []}
            for file in all_files:
                filename = os.path.basename(file)

                if filename in captions_data:
                    caption = captions_data[filename]
                    all_data["image"].append(file)
                    all_data["text"].append(caption)
                else:
                    print(f"Warning: Caption for {filename} not found.")
                    continue

        # Repeat the dataset
        all_data["image"] = all_data["image"] * repeats
        all_data["text"] = all_data["text"] * repeats
        mem_len = len(all_data["image"])

        # Create a HuggingFace Dataset
        dataset = Dataset.from_dict(all_data).cast_column("image", Image())
        prompt_key = "text"

    return dataset, prompt_key

def measure_CLIP_similarity(images, prompt, model, clip_preprocess, tokenizer, device):
    with torch.no_grad():
        img_batch = [clip_preprocess(i).unsqueeze(0) for i in images]
        img_batch = torch.cat(img_batch, dim=0).to(device) 
        image_features = model.encode_image(img_batch)

        text = tokenizer([prompt]).to(device)
        text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return (image_features @ text_features.T).mean(-1)
    
def measure_CLIP_similarity_batch(images, prompts, model, clip_preprocess, tokenizer, device):
    similarities = []
    for img, prompt in zip(images, prompts):
        sim = measure_CLIP_similarity([img], prompt, model, clip_preprocess, tokenizer, device)
        similarities.append(sim)
    return similarities



### credit: https://github.com/somepago/DCR
def measure_SSCD_similarity(gt_images, images, model, device):
    ret_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    gt_images = torch.stack([ret_transform(x.convert("RGB")) for x in gt_images]).to(
        device
    )
    images = torch.stack([ret_transform(x.convert("RGB")) for x in images]).to(device)

    with torch.no_grad():
        feat_1 = model(gt_images).clone()
        feat_1 = nn.functional.normalize(feat_1, dim=1, p=2)

        feat_2 = model(images).clone()
        feat_2 = nn.functional.normalize(feat_2, dim=1, p=2)

        return torch.mm(feat_1, feat_2.T)
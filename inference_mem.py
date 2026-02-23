import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean
import PIL
from PIL import Image 
import pickle
from rich import print as rich_print
import torch
import numpy as np
import time
from statistics import mean
import shutil
import os
import json
import sys
from io_utils import *
from local_sd_pipeline import LocalStableDiffusionPipeline
from local_sd_pipeline import find_pre_eot_and_eot

from diffusers import DDIMScheduler, UNet2DConditionModel
import lpips
import open_clip
from optim_utils import *
from optim_utils import _encode_latents, _prompt_embeds, diffusion_pair_loss, process_in_batches
# from optim_utils import (
#     _encode_prompt_only,
#     _align_dtype_device,
#     slerp_tokenwise,
#     most_similar_neighbor_embed,
#     build_negative_embeds,
# )
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration, PegasusForConditionalGeneration, PegasusTokenizer


print(Image)
print(PIL.Image)

import PIL.Image as PILImage
from PIL.Image import Image as PILImageType  # the actual PIL image class
import torch.nn.functional as F

def image_embed_openclip(model, preprocess, images, device):
    """
    images: PIL.Image or list[ PIL.Image ]
    returns: tensor [B, D] normalized to unit length
    """
    import PIL.Image as PILImage

    if isinstance(images, (PILImage.Image, torch.Tensor)):
        images = [images]

    # normalize inputs to PIL.Image.Image
    pil_images = []
    for im in images:
        if isinstance(im, PILImage.Image):
            pil = im
        elif isinstance(im, str):  # path
            pil = PILImage.open(im).convert("RGB")
        else:
            if hasattr(im, "convert"):
                pil = im.convert("RGB")
            elif hasattr(im, "to_pil"):
                pil = im.to_pil().convert("RGB")
            else:
                raise TypeError(f"Unsupported image type: {type(im)}. "
                                "Expected PIL.Image.Image, path, or object with .convert/.to_pil().")
        pil_images.append(pil)

    batch = torch.stack([preprocess(p).to(device) for p in pil_images], dim=0)  # [B,3,H,W]


    model.eval()
    with torch.no_grad():
        feats = model.encode_image(batch)     
        feats = F.normalize(feats.float(), dim=-1) # unit norm
    return feats


def main(args):
    if args.benchmark:
        args.with_tracking = False

    gen_start_time = time.time()
    table = None
    if args.with_tracking:
        wandb.init(
            project="mitigate_memorization", name=f'{args.unlearn_type}-{args.run_name}', tags=["run_mem"]
        )
        wandb.config.update(args)
        table = wandb.Table(
            columns=[
                "gt_prompt",
                "gen_prompt",
                "gt_clip_score",
                "gen_clip_score",
                "SSCD_sim",
                "SSCD_sim_max",
                "SSCD_sim_min",
            ]
        )

    # load diffusion model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    if args.unet_id is not None:
        unet = UNet2DConditionModel.from_pretrained(
            args.unet_id, torch_dtype=torch.bfloat16
        )
        pipe = LocalStableDiffusionPipeline.from_pretrained(
            args.model_id,
            unet=unet,
            torch_dtype=torch.bfloat16,
            safety_checker=None,
            requires_safety_checker=False,
        )
    else:
        pipe = LocalStableDiffusionPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            safety_checker=None,
            requires_safety_checker=False,
        )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    EOT_ID = 49407  # CLIP text EOT for SD-1.x

    def _first_eot_index(input_ids_1xT: torch.Tensor, eot_id: int = EOT_ID) -> int:
        ids = input_ids_1xT[0].tolist()
        return ids.index(eot_id) if eot_id in ids else len(ids) - 1

    @torch.no_grad()
    def representation_checks_A1_A2(pipe, core_prompt: str, suffix_prompt: str, device: str, save_path: str = None, tag: str = ""):
        # Tokenize
        tok_core = pipe.tokenizer([core_prompt], padding="max_length",
                                max_length=pipe.tokenizer.model_max_length,
                                truncation=True, return_tensors="pt").to(device)
        tok_suf = pipe.tokenizer([suffix_prompt], padding="max_length",
                                max_length=pipe.tokenizer.model_max_length,
                                truncation=True, return_tensors="pt").to(device)

        # Encode
        h_core = pipe.text_encoder(input_ids=tok_core["input_ids"]).last_hidden_state.float()  # [1, T, H]
        h_suf  = pipe.text_encoder(input_ids=tok_suf["input_ids"]).last_hidden_state.float()   # [1, T, H]

        # EOT indices
        eot_core = _first_eot_index(tok_core["input_ids"])
        eot_suf  = _first_eot_index(tok_suf["input_ids"])

        n_core_positions = int(eot_core)
        if n_core_positions > 0:
            cos_list = F.cosine_similarity(
                h_core[0, :n_core_positions, :],
                h_suf[0,  :n_core_positions, :],
                dim=-1,
            ).detach().cpu().numpy()
            core_mean_cos   = float(np.mean(cos_list))
            core_frac_0999  = float((cos_list >= 0.999).mean())
        else:
            cos_list = np.array([], dtype=np.float32)
            core_mean_cos = 1.0
            core_frac_0999 = 1.0

        eot_cos = float(F.cosine_similarity(h_core[0, eot_core, :], h_suf[0, eot_suf, :], dim=0).item())

        suffix_positions = list(range(int(eot_core), int(eot_suf))) if int(eot_suf) > int(eot_core) else []
        suffix_len = len(suffix_positions)

        suffix_cos_to_eot_suf_all = None
        suffix_cos_to_eot_suf_mean = None
        suffix_cos_to_eot_suf_min  = None
        suffix_cos_to_eot_suf_max  = None
        suffix_cos_to_eot_core_mean = None
        suffix_keynorm_mean = None

        if suffix_len > 0:
            eot_vec_suf = h_suf[0, eot_suf, :].unsqueeze(0).expand(suffix_len, -1)  # [L, H]
            cos_to_eot_suf = F.cosine_similarity(h_suf[0, suffix_positions, :], eot_vec_suf, dim=-1)
            cos_to_eot_suf_np = cos_to_eot_suf.detach().cpu().numpy()
            suffix_cos_to_eot_suf_all  = cos_to_eot_suf_np.tolist()
            suffix_cos_to_eot_suf_mean = float(np.mean(cos_to_eot_suf_np))
            suffix_cos_to_eot_suf_min  = float(np.min(cos_to_eot_suf_np))
            suffix_cos_to_eot_suf_max  = float(np.max(cos_to_eot_suf_np))

            eot_vec_core = h_core[0, eot_core, :].unsqueeze(0).expand(suffix_len, -1)
            cos_to_eot_core = F.cosine_similarity(h_suf[0, suffix_positions, :], eot_vec_core, dim=-1)
            suffix_cos_to_eot_core_mean = float(cos_to_eot_core.detach().cpu().numpy().mean())

            suffix_keynorm_mean = float(h_suf[0, suffix_positions, :].norm(dim=-1).mean().item())

        core_cos_to_eot_core_all = None
        core_cos_to_eot_core_mean = None
        if n_core_positions > 0:
            eot_vec_core_full = h_core[0, eot_core, :].unsqueeze(0).expand(n_core_positions, -1)
            cos_core_core = F.cosine_similarity(h_core[0, :n_core_positions, :], eot_vec_core_full, dim=-1)
            core_cos_to_eot_core_np = cos_core_core.detach().cpu().numpy()
            core_cos_to_eot_core_all  = core_cos_to_eot_core_np.tolist()
            core_cos_to_eot_core_mean = float(np.mean(core_cos_to_eot_core_np))

        core_cos_to_eot_suf_all = None
        core_cos_to_eot_suf_mean = None
        if n_core_positions > 0:
            eot_vec_suf_full = h_suf[0, eot_suf, :].unsqueeze(0).expand(n_core_positions, -1)
            cos_core_suf = F.cosine_similarity(h_suf[0, :n_core_positions, :], eot_vec_suf_full, dim=-1)
            core_cos_to_eot_suf_np = cos_core_suf.detach().cpu().numpy()
            core_cos_to_eot_suf_all  = core_cos_to_eot_suf_np.tolist()
            core_cos_to_eot_suf_mean = float(np.mean(core_cos_to_eot_suf_np))

        # ---------------- Summary dict ----------------
        summary = {
            "A1_core_mean_cos": core_mean_cos,
            "A1_core_frac_ge_0.999": core_frac_0999,
            "A1_n_positions": n_core_positions,
            "A2_eot_cos": eot_cos,
            "A2_eot_cos_before": None,  
            "A2_eot_cos_after": None,  
            "eot_mean_pool_used": bool(args.eot_mean_pool),
            "eot_shrink_alpha_used": float(args.eot_shrink_alpha if args.eot_shrink_alpha is not None else 0.0),
            "eot_noise_sigma_used": float(args.eot_noise_sigma if args.eot_noise_sigma is not None else 0.0),

            "A2_eot_idx_core": int(eot_core),
            "A2_eot_idx_suffix": int(eot_suf),
            "token_len_core": int(tok_core["input_ids"].shape[-1]),
            "token_len_suffix": int(tok_suf["input_ids"].shape[-1]),
            "suffix_len_tokens": suffix_len,

            "S_cos_suffix_to_EOT_suf_all":  suffix_cos_to_eot_suf_all,
            "S_cos_suffix_to_EOT_suf_mean": suffix_cos_to_eot_suf_mean,
            "S_cos_suffix_to_EOT_suf_min":  suffix_cos_to_eot_suf_min,
            "S_cos_suffix_to_EOT_suf_max":  suffix_cos_to_eot_suf_max,
            "S_cos_suffix_to_EOT_core_mean": suffix_cos_to_eot_core_mean,
            "S_suffix_hidden_norm_mean": suffix_keynorm_mean,

            "C_cos_core_to_EOT_core_all":  core_cos_to_eot_core_all,
            "C_cos_core_to_EOT_core_mean": core_cos_to_eot_core_mean,
            "C_cos_core_to_EOT_suf_all":   core_cos_to_eot_suf_all,
            "C_cos_core_to_EOT_suf_mean":  core_cos_to_eot_suf_mean,

            "tag": tag,
        }

        if save_path is not None:
            try:
                with open(save_path, "w") as f:
                    json.dump(summary, f, indent=2)
            except Exception as e:
                print(f"[A1/A2] Failed to save JSON to {save_path}: {e}")

        return summary

    # dataset
    set_random_seed(args.gen_seed)
    if args.dataset == 'memorized_images':
        dataset, prompt_key = get_dataset_finetune(args.dataset, args=args)
    elif args.dataset == 'laion_10k':
        if args.mem is not None:
            dataset, prompt_key = get_dataset_finetune(
                args.dataset,
                args.non_mem_dataset,
                end=args.end,
                # repeats=args.repeats,
                # non_mem_ratio=args.non_mem_ratio,
                args=args,
            )

            mask_filter = {'forget': 1, 'retain': 0}
            print(f'args.split: {args.split}, mask value: {mask_filter[args.split]}')

            if args.split in mask_filter:
                dataset = [d for d in dataset if d['mask'] == mask_filter[args.split]]
            else:
                raise ValueError(f"Invalid split: {args.split}")

            print(f"Loaded {len(dataset)} images from {args.dataset} -- split: {args.split}")
        else:
            dataset, prompt_key = get_dataset_finetune_v2(args.dataset)
            # dataset, prompt_key = get_dataset_finetune_v2(dataset_name=args.dataset, end=args.end)
            print(f"Loaded {len(dataset)} images from {args.dataset}")
    
    elif args.dataset == 'laion_aesthetics':
        if args.mem is not None:
            dataset, prompt_key = get_dataset_finetune(
                args.dataset,
                args.non_mem_dataset,
                end=args.end,
                # repeats=args.repeats,
                # non_mem_ratio=args.non_mem_ratio,
                args=args,
            )

            mask_filter = {'forget': 1, 'retain': 0}
            print(f'args.split: {args.split}, mask value: {mask_filter[args.split]}')
            if args.split in mask_filter:
                dataset = [d for d in dataset if d['mask'] == mask_filter[args.split]]
            else:
                raise ValueError(f"Invalid split: {args.split}")

            print(f"Loaded {len(dataset)} images from {args.dataset} -- split: {args.split}")
        else:
            dataset, prompt_key = get_dataset_finetune_v2(args.dataset, args=args)
        # dataset, prompt_key = get_dataset_finetune_v2(dataset_name=args.dataset, end=args.end)
        print(f"Loaded {len(dataset)} images from {args.dataset}")

    else:
        dataset, prompt_key = get_dataset_finetune(args.dataset, args=args)
        print(f"Loaded {len(dataset)} images from {args.dataset}")
        print('**********************************************')

    args.end = min(args.end, len(dataset))

    chosen_dataset_indices = None
    sampled_idx_set = None
    if args.sample_size is not None:
        total = len(dataset)

        if args.sample_indices_path and os.path.exists(args.sample_indices_path):
            with open(args.sample_indices_path, "r") as f:
                chosen_dataset_indices = json.load(f)
            # Keep only valid integers within bounds
            chosen_dataset_indices = [int(x) for x in chosen_dataset_indices if 0 <= int(x) < total]
            print(f"Loaded {len(chosen_dataset_indices)} sampled indices from {args.sample_indices_path}.")
        else:
            # 2) Draw a new subset deterministically
            rng = np.random.RandomState(args.gen_seed)
            k = min(int(args.sample_size), total)
            chosen_dataset_indices = rng.choice(np.arange(total), size=k, replace=False).tolist()
            if args.sample_indices_path:
                with open(args.sample_indices_path, "w") as f:
                    json.dump(sorted(chosen_dataset_indices), f, indent=2)
                print(f"Saved {len(chosen_dataset_indices)} sampled indices to {args.sample_indices_path}.")

        sampled_idx_set = set(chosen_dataset_indices)
        print(f"Sampling active -> restricting to {len(sampled_idx_set)} dataset indices (global sampling).")


    # -----------------------
    # PHASE 1: Deduplication
    # -----------------------
    if 'stable-diffusion-2-base' in args.model_id:
        dedup = False
    else:
        dedup = True  

    if args.template_mem:
        dedup = True

    print("Starting deduplication (filtering out examples with multiple GT images)...")
    deduped_examples = []
    iter_indices = range(args.start, args.end) if sampled_idx_set is None else sorted(sampled_idx_set)
    for i in tqdm(iter_indices):
        seed = i + args.gen_seed
        gt_prompt = dataset[i][prompt_key]
        
        if "groundtruth" in args.dataset:
            gt_images = []
            curr_index = dataset[i]["index"]
            for filename in glob.glob(f"{args.data_path}/{args.dataset}/gt_images/{curr_index}/*.png"):
                im = PIL.Image.open(filename)
                gt_images.append(im)

        else:
            gt_images = [dataset[i]["image"]]

        if args.template_mem:
            if len(gt_images) <= 1:
                continue
        elif dedup and len(gt_images) > 1:
            continue

        deduped_examples.append({
            "original_index": i,  
            "seed": seed,
            "gt_prompt": gt_prompt,
            "gt_images": gt_images,
        })
        
    print(f"Deduplication complete. {len(deduped_examples)} examples remain.")
    

    # -----------------------
    # PHASE 2: SSCD Selection
    # -----------------------
    print("Starting SSCD selection (if applicable)...")
    selected_examples = []
    if args.dataset == 'sdv1_500_mem_groundtruth' and args.topk is not None:
        if 'stable-diffusion-2-base' in args.model_id:
            if args.template_mem:
                path = './results/inference_no_mitigation_sdv2_tempMem_memNone_num258_nonmem_None_gen_sdv1_500_mem_groundtruth_topNone_start0_end500_s0'
            elif dedup:
                path = './results/inference_no_mitigation_sdv2_memNone_num242_nonmem_None_gen_sdv1_500_mem_groundtruth_topNone_start0_end500_s0'
            else:
                path = './results/inference_no_mitigation_sdv2_memNone_num500_nonmem_None_gen_sdv1_500_mem_groundtruth_topNone_start0_end500_s0'
        else:
            if args.template_mem:
                path = './results/inference_no_mitigation_tempMem_memNone_num258_nonmem_None_gen_sdv1_500_mem_groundtruth_topNone_start0_end500_s0'
            elif dedup:
                path = './results/inference_memNone_nonmem_None_gen_sdv1_500_mem_groundtruth_start0_end500_no_mitigation_s0'
            
        sscd_results = pd.read_pickle(f"{path}/results_with_scores.pkl")
        sscd_scores = np.array(sscd_results['sscd_scores'])
        # Note: sscd_scores should be aligned with the deduplicated examples.
        
        if args.topk > 1.0:
        # Option 1: Select top-k indices (the results file order is assumed to match deduped_data order)
            topk_indices = np.argsort(sscd_scores)[-args.topk:]
            print(f"Top-k SSCD scores: {sscd_scores[topk_indices]}")
            for idx in topk_indices:
                selected_examples.append(deduped_examples[idx])
        
        elif args.topk <= 1.0:
        # Option 2: Select examples with SSCD score > threshold
            print(f"Selecting examples with SSCD score > {args.topk}")
            for idx, score in enumerate(sscd_scores):
                if score > args.topk:
                    selected_examples.append(deduped_examples[idx])
            selected_scores = [sscd_scores[idx] for idx, score in enumerate(sscd_scores) if score > args.topk]
            print(f"Selected SSCD scores: {selected_scores}")

        # Option 3: Select examples with kth percentile sscd scores (e.g., 90 or 95)
        # threshold = np.percentile(sscd_scores, args.topk)
        # print(f"Selecting examples with SSCD score > {threshold}")
        # for idx, score in enumerate(sscd_scores):
        #     if score > threshold:
        #         selected_examples.append(deduped_examples[idx])

        print(f"Selecting examples with SSCD score > {args.topk}")

    else:
        selected_examples = deduped_examples

    print(f"****** Selected {len(selected_examples)} examples for generation ******")
    num_selected_examples = len(selected_examples)

    results = {
        "gen_images": [], "gt_images": [], "gen_prompts": [], "gt_prompts": [],
        'sscd_scores': [], 'clip_scores': [],
        "sscd_scores_per_image": [], "clip_scores_per_image": [],
        "loss_exposure": [], "loss_exposure_rank": [],
        "loss_exposure_L_canary": [], "loss_exposure_L_refs": [],
        "sscd_exposure": [], "sscd_exposure_rank": [],
        "basin_histories": [],
    }
    if sampled_idx_set is not None:
        results["sampled_dataset_indices"] = sorted(chosen_dataset_indices)

    examples_to_generate = selected_examples
    num_canaries_for_sscd = 0 # Will be used to split results later

    non_mem_ds = None
    if args.unet_id and 'laion_10k' in args.unet_id:
        non_mem_ds = 'laion_10k'
    elif args.unet_id and 'laion_aesthetic' in args.unet_id:
        non_mem_ds = 'laion_aesthetic'
    
    if args.dememorize_token is not None:
        args.run_name = f"{args.run_name}_genw{args.dememorize_token}"

    if args.model_id and 'stable-diffusion-2' in args.model_id:
        args.run_name = f"{args.run_name}_sdv2"
    
    if args.template_mem:
        args.run_name = f"{args.run_name}_tempMem"

    print(f"Output directory: {args.run_name}")
     
    if args.mem is not None:
        output_dir = f"./results/{args.unlearn_type}_{args.run_name}_mem{args.mem_dataset}_num{num_selected_examples}_nonmem_{non_mem_ds}_gen_{args.dataset}_mem{args.mem}_split{args.split}_start{args.start}_end{args.end}_s{args.gen_seed}"
    else:
        output_dir = f"./results/{args.unlearn_type}_{args.run_name}_mem{args.mem_dataset}_num{num_selected_examples}_nonmem_{non_mem_ds}_gen_{args.dataset}_top{args.topk}_start{args.start}_end{args.end}_s{args.gen_seed}"
    print('overall output_dir: ',output_dir)
    os.makedirs(output_dir, exist_ok=True)
    rep_dir = os.path.join(output_dir, "representative_images")
    os.makedirs(rep_dir, exist_ok=True)
    
    if sampled_idx_set is not None:
        with open(os.path.join(output_dir, "sampled_dataset_indices.json"), "w") as f:
            json.dump(sorted(chosen_dataset_indices), f, indent=2)


    # -----------------------
    # PHASE 3: Generation
    # -----------------------
    print("Starting generation...")
    all_gen_images = []
    all_gt_images = []
    all_gen_prompts = []
    all_gt_prompts = []
    # all_cos_sims = []
    # all_cond_norms = []

    examples_to_generate = selected_examples

    check_neighbors = False
    # only_check_neighbors = True

    if args.benchmark:
        # Force these off
        check_neighbors = False

    log_attention_viz = True
    attn_target_side = 32
    height = width = args.image_length
    if check_neighbors:
        print(">>> Running neighbor check (text-embedding proximity + cross-attention similarity).")

        # prepare timesteps
        ts = []
        for t in args.attn_timesteps.split(","):
            t = int(t.strip())
            t = max(0, min(pipe.scheduler.config.num_train_timesteps - 1, t))
            ts.append(t)

        out_dir = os.path.join("./assets/neighbors", f"neighbor_checks_{args.run_name}_gen_{args.dataset}_start{args.start}_end{args.end}_s{args.gen_seed}")
        os.makedirs(out_dir, exist_ok=True)
        if log_attention_viz:
            os.makedirs(os.path.join(out_dir, "attn_viz"), exist_ok=True)

        # Optional OpenCLIP
        ref_model = ref_tokenizer = None
        if args.reference_model is not None:
            ref_model, _, _ = open_clip.create_model_and_transforms(
                args.reference_model, pretrained=args.reference_model_pretrain, device=device
            )
            ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

        # height = width = args.image_length
        latent_h, latent_w = height // 8, width // 8

        neighbor_rows = []
        for i, example in enumerate(tqdm(examples_to_generate)):
            seed = example.get('seed', i + args.gen_seed)
            gt_prompt = example["gt_prompt"]

            if args.prompt_aug_style == 'neighbor_replace':
                print(f'gndtruth_prompt: {gt_prompt}')
                neighs, _ = generate_neighbors_from_algorithm(
                    gt_prompt,
                    n=args.num_neighbors,  # Number of neighbors to generate, -1 to return all neighbors
                    p_replacements=args.p_replacements,  # Proportion of words to replace
                    top_k_for_combinations=3  # New parameter for n=-1 feature
                )

                if not isinstance(neighs, (list, tuple)):
                    neighs = [neighs]

                def _norm_prompt(p):
                    # handle cases like ['some prompt'] or already a string
                    return p[0] if isinstance(p, list) and len(p) == 1 else p
                # neighs = [_norm_prompt(p) for p in neighs]
                neighs = [str(_norm_prompt(p)).strip() for p in neighs]

                seen = set()
                neighs_to_eval = []
                for p in neighs:
                    # p = str(p).strip()
                    if p == gt_prompt.strip():
                        continue
                    if p not in seen:
                        seen.add(p)
                        neighs_to_eval.append(p)

                if isinstance(args.num_neighbors, int) and args.num_neighbors > 0:
                    neighs_to_eval = neighs_to_eval[:args.num_neighbors]
                print(f"evaluating {len(neighs_to_eval)} neighbors")
            
            else:
                prompt_nb = gt_prompt

            # text cosine
            if args.reference_model is not None:
                e_gt = text_embed_openclip(ref_model, ref_tokenizer, [gt_prompt], device)[0]
            else:
                e_gt = text_embed_sd(pipe, [gt_prompt], device)[0]
            
            pe_gt = _prompt_embeds(pipe, [gt_prompt]).to(device)
            set_random_seed(seed)
            latents = torch.randn(1, pipe.unet.in_channels, latent_h, latent_w,
                                device=device, dtype=pipe.unet.dtype)
            
            gt_spatials_by_t = []        # list[np.array(Q)] aligned with ts
            gt_tokens_by_t   = []        # list[np.array(K)] (kept for metrics if needed)
            Q = K = None
            for t in ts:
                agg_gt = capture_spatial_for_timestep(pipe, latents, pe_gt, t, attn_target_side)
                if agg_gt is None:
                    print(f"[warn] No cross-attn at {attn_target_side}x{attn_target_side} for GT @ t={t}; skipping example {i}.")
                    gt_spatials_by_t = []
                    break
                spatial_gt, token_gt, Q, K = agg_gt
                gt_spatials_by_t.append(spatial_gt)
                gt_tokens_by_t.append(token_gt)

            if not gt_spatials_by_t:
                continue  

            all_nb_spatials_by_t = []   
            panel_labels_base    = []  

            for nb_idx, prompt_nb in enumerate(neighs_to_eval):
                if args.reference_model is not None:
                    e_nb = text_embed_openclip(ref_model, ref_tokenizer, [prompt_nb], device)[0]
                else:
                    e_nb = text_embed_sd(pipe, [prompt_nb], device)[0]
                text_cos = text_cosine(e_gt, e_nb)

                pe_nb = _prompt_embeds(pipe, [prompt_nb]).to(device)

                nb_spatials_by_t = []
                nb_tokens_by_t   = []

                for t in ts:
                    agg_nb = capture_spatial_for_timestep(pipe, latents, pe_nb, t, attn_target_side)
                    if agg_nb is None:
                        print(f"[warn] No cross-attn at {attn_target_side}x{attn_target_side} for neighbor {nb_idx} @ t={t}; skipping this t.")
                        nb_spatials_by_t.append(None)
                        nb_tokens_by_t.append(None)
                        continue
                    spatial_nb, token_nb, _, _ = agg_nb
                    nb_spatials_by_t.append(spatial_nb)
                    nb_tokens_by_t.append(token_nb)

                    spatial_gt_t = gt_spatials_by_t[ts.index(t)]
                    token_gt_t   = gt_tokens_by_t[ts.index(t)]
                    attn_spatial_cos = cosine_np(spatial_gt_t, spatial_nb)
                    attn_spatial_iou = topk_iou(spatial_gt_t, spatial_nb, topk_percent=args.attn_topk_percent)
                    attn_token_cos   = cosine_np(token_gt_t, token_nb)

                    row = {
                        "idx": i,
                        "nb_idx": nb_idx,   
                        "timestep": int(t),        
                        "seed": seed,
                        "gt_prompt": gt_prompt,
                        "neighbor_prompt": prompt_nb,
                        "text_cosine": text_cos,
                        "attn_spatial_cos": attn_spatial_cos,
                        "attn_spatial_iou_topk": attn_spatial_iou,
                        "attn_tokenmass_cos": attn_token_cos,
                        "Q": int(Q),
                        "K": int(K),
                    }
                    neighbor_rows.append(row)

                    if args.with_tracking:
                        wandb.log({k: v for k, v in row.items() if isinstance(v, (int, float))})

                all_nb_spatials_by_t.append(nb_spatials_by_t)
                short = prompt_nb if len(prompt_nb) <= 60 else (prompt_nb[:57] + "…")
                panel_labels_base.append(f"nb{nb_idx} | cos={text_cos:.3f}")
                
                
                if log_attention_viz:
                    col_labels = ["GT"] + panel_labels_base  
                    panel_path = os.path.join(
                        out_dir, "attn_viz",
                        f"{i:05d}_MEGA_{attn_target_side}px.png"
                    )
                    save_mega_panel(
                        gt_spatials_by_t=gt_spatials_by_t,
                        all_nb_spatials_by_t=all_nb_spatials_by_t,
                        col_labels=col_labels,
                        ts=ts,
                        out_path=panel_path,
                        side=attn_target_side
                    )
                    if args.with_tracking:
                        wandb.log({"attn_panel_mega": wandb.Image(panel_path)})

        with open(os.path.join(out_dir, "neighbor_checks.json"), "w") as f:
            json.dump(neighbor_rows, f, indent=2)
        print(f"Saved neighbor checks to {out_dir}")

    logs_dir = os.path.join(output_dir, "logs")
    current_log_stats = True
    current_trace = True
    current_save_json = f"{logs_dir}/{args.run_name}_xattn_stats.json"

    if args.benchmark:
        current_log_stats = False
        current_trace = False
        current_save_json = None
        print(">>> BENCHMARK: Stats logging and Tracing DISABLED for efficiency. <<<")
    
    for i, example in enumerate(tqdm(examples_to_generate)):
        seed = example.get('seed', i + args.gen_seed)
        gt_prompt = example["gt_prompt"]

        # Optional prompt augmentation
        if args.prompt_aug_style == 'synonym_replace':
            model_name = "tuner007/pegasus_paraphrase"
            tokenizer = PegasusTokenizer.from_pretrained(model_name)
            model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

            def get_paraphrases(sentence, num_return_sequences=5, num_beams=5):
                # Tokenize the input sentence
                batch = tokenizer([sentence], truncation=True, padding="longest", return_tensors="pt").to(device)
                # Generate paraphrases
                translated = model.generate(
                    **batch,
                    max_length=60,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    temperature=3.5 # Increase temperature for more diversity
                )
                # Decode the generated tokens
                paraphrases = tokenizer.batch_decode(translated, skip_special_tokens=True)
                return paraphrases
            paraphrased_neighbors = get_paraphrases(gt_prompt)
            prompt = paraphrased_neighbors[0] if isinstance(paraphrased_neighbors, list) else paraphrased_neighbors

            print(f"Original: {gt_prompt}")
            print(f"Paraphrased: {prompt}")

        elif args.prompt_aug_style == 'neighbor_replace':
            print(f'gndtruth_prompt: {gt_prompt}')

            model_name = "tuner007/pegasus_paraphrase"
            tokenizer = PegasusTokenizer.from_pretrained(model_name)
            model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

            def get_paraphrases(sentence, num_return_sequences=5, num_beams=5):
                # Tokenize the input sentence
                batch = tokenizer([sentence], truncation=True, padding="longest", return_tensors="pt").to(device)
                # Generate paraphrases
                translated = model.generate(
                    **batch,
                    max_length=60,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    temperature=3.5 # Increase temperature for more diversity
                )
                # Decode the generated tokens
                paraphrases = tokenizer.batch_decode(translated, skip_special_tokens=True)
                return paraphrases
            paraphrased_neighbors = get_paraphrases(gt_prompt)
            prompt = paraphrased_neighbors[0] if isinstance(paraphrased_neighbors, list) else paraphrased_neighbors
            print(f"neighbor_prompt: {prompt}")

        elif args.prompt_aug_style is not None:
            prompt = prompt_augmentation(
                gt_prompt,
                args.prompt_aug_style,
                tokenizer=pipe.tokenizer,
                repeat_num=args.repeat_num,
            )

        else:
            prompt = gt_prompt
        print(f"Prompt: {prompt}")


        if (args.repeat_token is not None or args.repeat_num is not None) and not args.benchmark:
            tag = f"sample_{i}_seed_{seed}"
            if not args.benchmark: 
                os.makedirs(logs_dir, exist_ok=True)
            a12_json = os.path.join(logs_dir, f"A12_{tag}.json") if 'output_dir' in locals() else None

            a12 = representation_checks_A1_A2(
                pipe=pipe,
                core_prompt=gt_prompt,
                suffix_prompt=prompt,
                device=device,
                save_path=a12_json,
                tag=tag,
            )

            if args.with_tracking:
                wandb.log({
                    "A1_core_mean_cos": a12["A1_core_mean_cos"],
                    "A1_core_frac_ge_0.999": a12["A1_core_frac_ge_0.999"],
                    "A1_n_positions": a12["A1_n_positions"],
                    "A2_eot_cos": a12["A2_eot_cos"],
                    "A2_eot_idx_core": a12["A2_eot_idx_core"],
                    "A2_eot_idx_suffix": a12["A2_eot_idx_suffix"],
                    "S_suffix_len_tokens": a12["suffix_len_tokens"],
                    "S_cos_suffix_to_EOT_suf_mean": a12["S_cos_suffix_to_EOT_suf_mean"],
                    "S_cos_suffix_to_EOT_suf_max": a12["S_cos_suffix_to_EOT_suf_max"],
                    "S_cos_suffix_to_EOT_suf_min": a12["S_cos_suffix_to_EOT_suf_min"],
                    "S_cos_suffix_to_EOT_core_mean": a12["S_cos_suffix_to_EOT_core_mean"],
                    "S_suffix_hidden_norm_mean": a12["S_suffix_hidden_norm_mean"],
                })
  

        elif args.num_neighbors is not None and args.prompt_aug_style == 'neighbor_replace':
            set_random_seed(seed)

            if args.xattn_rescale:
                print(f"[Combined Method] Using GT prompt as positive neighbor WITH CA attenuation.")
                BLOCKS_DOWN_EARLY = [
                    "down_blocks.0.attentions.0.transformer_blocks.0.attn2",
                    "down_blocks.0.attentions.1.transformer_blocks.0.attn2",
                    "down_blocks.1.attentions.0.transformer_blocks.0.attn2",
                    "down_blocks.1.attentions.1.transformer_blocks.0.attn2",
                ]

                BLOCKS_DOWN_LATE = [
                    "down_blocks.2.attentions.0.transformer_blocks.0.attn2",
                    "down_blocks.2.attentions.1.transformer_blocks.0.attn2",
                    "down_blocks.3.attentions.0.transformer_blocks.0.attn2",
                    "down_blocks.3.attentions.1.transformer_blocks.0.attn2",    
                    "down_blocks.3.attentions.2.transformer_blocks.0.attn2",    
                ]

                BLOCK_MID = [
                    "mid_block.attentions.0.transformer_blocks.0.attn2"
                ]

                BLOCKS_UP_EARLY = [
                    "up_blocks.0.attentions.0.transformer_blocks.0.attn2",    
                    "up_blocks.0.attentions.1.transformer_blocks.0.attn2",    
                    "up_blocks.0.attentions.2.transformer_blocks.0.attn2",    
                    "up_blocks.1.attentions.0.transformer_blocks.0.attn2",
                    "up_blocks.1.attentions.1.transformer_blocks.0.attn2",
                    "up_blocks.1.attentions.2.transformer_blocks.0.attn2",
                ]

                BLOCKS_UP_LATE = [
                    "up_blocks.2.attentions.0.transformer_blocks.0.attn2",
                    "up_blocks.2.attentions.1.transformer_blocks.0.attn2",
                    "up_blocks.2.attentions.2.transformer_blocks.0.attn2",
                    "up_blocks.3.attentions.0.transformer_blocks.0.attn2",
                    "up_blocks.3.attentions.1.transformer_blocks.0.attn2",
                    "up_blocks.3.attentions.2.transformer_blocks.0.attn2",
                ]

                # --- GROUP DEFINITIONS ---
                intervention_blocks = None
                if args.intervention_blocks == 'BLOCKS_ALL':
                    intervention_blocks = BLOCKS_DOWN_EARLY + BLOCKS_DOWN_LATE + BLOCK_MID + BLOCKS_UP_EARLY + BLOCKS_UP_LATE
                elif args.intervention_blocks == 'BLOCKS_SEMANTIC':
                    intervention_blocks = BLOCKS_DOWN_LATE + BLOCK_MID + BLOCKS_UP_EARLY
                elif args.intervention_blocks == 'BLOCKS_DETAIL':
                    intervention_blocks = BLOCKS_DOWN_EARLY + BLOCKS_UP_LATE
                elif args.intervention_blocks == 'BLOCK_MID':
                    intervention_blocks = BLOCK_MID
                elif args.intervention_blocks == 'BLOCKS_EARLY':
                    intervention_blocks = BLOCKS_DOWN_EARLY + BLOCKS_DOWN_LATE + BLOCK_MID
                elif args.intervention_blocks == 'BLOCKS_LATE':
                    intervention_blocks = BLOCK_MID + BLOCKS_UP_EARLY + BLOCKS_UP_LATE
                elif args.intervention_blocks == 'BLOCKS_MIDDETAIL':
                    intervention_blocks = BLOCK_MID + BLOCKS_DOWN_EARLY + BLOCKS_UP_LATE
                else:
                    intervention_blocks = None

                paraphrased_neighbors = [gt_prompt]
                
                current_neighbor_gating = "none" 
                current_ortho = False # Ortho not needed if text is identical (unless relying purely on attenuation delta)
                current_pst_bias = args.pst_bias
                current_eot_bias = args.eot_bias
                current_spike_threshold = args.spike_threshold
                current_spike_penalty = args.spike_penalty
                current_spike_scale = args.spike_scale
                current_intervention_blocks = intervention_blocks # Defined earlier in script for BLOCKS_EARLY etc
            else:
                current_neighbor_gating = args.neighbor_gating_mode
                current_ortho = args.orthogonalize_neighbors
                current_pst_bias = 0.0
                current_eot_bias = 0.0
                current_spike_threshold = 0.0
                current_spike_penalty = 0.0
                current_spike_scale = 1.0
                current_intervention_blocks = None
            # ------------------------------------------------
            outputs = pipe.dual_guidance_call(
                prompt_positive=paraphrased_neighbors,              # <= list[str]
                prompt_negative_repulsive=gt_prompt,
                guidance_scale_positive=args.guidance_scale,        
                guidance_scale_negative=args.repulsion_scale,     
                negative_guidance_decay_schedule="cosine",  
                positive_guidance_schedule="none",    
                neighbor_gating_mode=current_neighbor_gating,   
                # neighbor_gating_threshold=0.9,  
                # neighbor_gating_soft_sigma=0.5, 
                # use_only_nearest_for_positive=False,
                # soft_per_neighbor=True,   
                # orthogonalize_neighbors=current_ortho, 
                # ortho_cos_threshold=0.9,
                # ortho_coef_clip=2.0,
                ###
                # # -------------------- TD-REG (NEW) --------------------
                # td_reg= False,
                # td_beta= 0.88,      ###TODO:0-0.88, 2-0.93, 0.9
                # td_eps= 1e-8,       ## prevents divide-by-zero
                # td_tiny= 1e-6,      ## Prevents over-amplifying noise  
                # ------------------------------------------------------
                num_inference_steps=args.num_inference_steps,
                num_images_per_prompt=args.num_images_per_prompt,
                xattn_log=True, 
                trace_per_block=current_trace, 
                xattn_save_json=current_save_json,
                log_stats=current_log_stats,
                pst_bias=current_pst_bias,
                eot_bias=current_eot_bias,
                spike_threshold=current_spike_threshold,
                spike_penalty=current_spike_penalty,
                spike_scale=current_spike_scale,
                intervention_blocks=current_intervention_blocks,
            )

            if hasattr(pipe, "_xattn_logger") and pipe._xattn_logger is not None:
                basin_hist = getattr(pipe._xattn_logger, "basin_history", [])
                results["basin_histories"].append({
                    "prompt": prompt,
                    "type": "ar_guidance",
                    "history": copy.deepcopy(basin_hist)
                })
            else:
                results["basin_histories"].append(None)

        elif args.xattn_rescale:
            logs_dir = os.path.join(output_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            BLOCKS_DOWN_EARLY = [
                "down_blocks.0.attentions.0.transformer_blocks.0.attn2",
                "down_blocks.0.attentions.1.transformer_blocks.0.attn2",
                "down_blocks.1.attentions.0.transformer_blocks.0.attn2",
                "down_blocks.1.attentions.1.transformer_blocks.0.attn2",
            ]

            BLOCKS_DOWN_LATE = [
                "down_blocks.2.attentions.0.transformer_blocks.0.attn2",
                "down_blocks.2.attentions.1.transformer_blocks.0.attn2",
                "down_blocks.3.attentions.0.transformer_blocks.0.attn2",
                "down_blocks.3.attentions.1.transformer_blocks.0.attn2",  
                "down_blocks.3.attentions.2.transformer_blocks.0.attn2",  
            ]

            BLOCK_MID = [
                "mid_block.attentions.0.transformer_blocks.0.attn2"
            ]

            BLOCKS_UP_EARLY = [
                "up_blocks.0.attentions.0.transformer_blocks.0.attn2",   
                "up_blocks.0.attentions.1.transformer_blocks.0.attn2", 
                "up_blocks.0.attentions.2.transformer_blocks.0.attn2",  
                "up_blocks.1.attentions.0.transformer_blocks.0.attn2",
                "up_blocks.1.attentions.1.transformer_blocks.0.attn2",
                "up_blocks.1.attentions.2.transformer_blocks.0.attn2",
            ]

            BLOCKS_UP_LATE = [
                "up_blocks.2.attentions.0.transformer_blocks.0.attn2",
                "up_blocks.2.attentions.1.transformer_blocks.0.attn2",
                "up_blocks.2.attentions.2.transformer_blocks.0.attn2",
                "up_blocks.3.attentions.0.transformer_blocks.0.attn2",
                "up_blocks.3.attentions.1.transformer_blocks.0.attn2",
                "up_blocks.3.attentions.2.transformer_blocks.0.attn2",
            ]

            intervention_blocks = None
            if args.intervention_blocks == 'BLOCKS_ALL':
                intervention_blocks = BLOCKS_DOWN_EARLY + BLOCKS_DOWN_LATE + BLOCK_MID + BLOCKS_UP_EARLY + BLOCKS_UP_LATE
            elif args.intervention_blocks == 'BLOCKS_SEMANTIC':
                intervention_blocks = BLOCKS_DOWN_LATE + BLOCK_MID + BLOCKS_UP_EARLY
            elif args.intervention_blocks == 'BLOCKS_DETAIL':
                intervention_blocks = BLOCKS_DOWN_EARLY + BLOCKS_UP_LATE
            elif args.intervention_blocks == 'BLOCK_MID':
                intervention_blocks = BLOCK_MID
            elif args.intervention_blocks == 'BLOCKS_EARLY':
                intervention_blocks = BLOCKS_DOWN_EARLY + BLOCKS_DOWN_LATE + BLOCK_MID
            elif args.intervention_blocks == 'BLOCKS_LATE':
                intervention_blocks = BLOCK_MID + BLOCKS_UP_EARLY + BLOCKS_UP_LATE
            elif args.intervention_blocks == 'BLOCKS_MIDDETAIL':
                intervention_blocks = BLOCK_MID + BLOCKS_DOWN_EARLY + BLOCKS_UP_LATE
            else:
                intervention_blocks = None

            outputs = pipe(
                prompt=prompt,
                gt_prompt=gt_prompt, 
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=args.num_images_per_prompt,

                pst_bias=args.pst_bias,
                eot_bias=args.eot_bias,
                bot_bias=args.bot_bias,
                bias_schedule=args.bias_schedule, 
                bias_start_frac=args.bias_start_frac, 
                bias_end_frac=args.bias_end_frac,  
                bias_on='cond', 
                xattn_log=True, 
                xattn_save_json=current_save_json,
                trace_per_block=current_trace,
                log_stats=current_log_stats,

                dynamic_head_selection=args.dynamic_head_selection,   
                dynamic_scout_steps=5, 
                dynamic_top_k_per_block=args.dynamic_top_k_per_block,
                dynamic_top_k_global=args.dynamic_top_k_global, 
                dynamic_metric="eot_mass_peakiness",  
                dynamic_min_threshold=0.0,     
                intervention_blocks=intervention_blocks,  

                spike_penalty=args.spike_penalty,
                spike_threshold=args.spike_threshold,    
                spike_scale=args.spike_scale,
            )

            if hasattr(pipe, "_xattn_logger") and pipe._xattn_logger is not None:
                basin_hist = getattr(pipe._xattn_logger, "basin_history", [])
                results["basin_histories"].append({
                    "prompt": prompt,
                    "type": "attenuation",
                    "history": copy.deepcopy(basin_hist)
                })
            else:
                results["basin_histories"].append(None)

        else:

            outputs = pipe(
                prompt=prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=args.num_images_per_prompt,
                xattn_log=args.xattn_log if not args.benchmark else False,
                trace_per_block=current_trace,
                xattn_save_json=current_save_json,
                log_stats=current_log_stats,
            )

            if hasattr(pipe, "_xattn_logger") and pipe._xattn_logger is not None:
                basin_hist = getattr(pipe._xattn_logger, "basin_history", [])
                results["basin_histories"].append({
                    "prompt": prompt,
                    "type": "baseline",
                    "history": copy.deepcopy(basin_hist)
                })
            else:
                results["basin_histories"].append(None)

        gen_images = outputs.images

        all_gen_images.append(gen_images)
        all_gt_images.append(example["gt_images"])
        all_gen_prompts.append(prompt)
        all_gt_prompts.append(gt_prompt)

    print(f"Overall {len(all_gt_images)} ground truth examples processed.")

    gen_end_time = time.time()
    gen_duration = gen_end_time - gen_start_time
    print(f"-------- Generation runtime: {gen_duration:.3f} seconds ---------")
    if len(all_gt_images) > 0:
        print(f"-------- Average time per example: {gen_duration / len(all_gt_images):.3f} seconds ---------")

    if args.benchmark:
        print(">>> BENCHMARK MODE END: Exiting before evaluation metrics calculation. <<<")
        return 

    
    results["gen_images"] = all_gen_images
    results["gt_images"] = all_gt_images
    results["gen_prompts"] = all_gen_prompts
    results["gt_prompts"] = all_gt_prompts

    metadata = []
    for i in range(min(10, len(all_gen_images))):
        for j, gen_image in enumerate(all_gen_images[i][:1]):  
            gen_path = os.path.join(rep_dir, f"gen_{i}_{j}.png")
            gen_image.save(gen_path, format="PNG")
            metadata.append({
                "image_type": "generated",
                "prompt": all_gen_prompts[i],
                "image_path": gen_path
            })            

        for j, gt_image in enumerate(all_gt_images[i][:1]):  
            gt_path = os.path.join(rep_dir, f"gt_{i}_{j}.png")
            gt_image.save(gt_path, format="PNG")
            metadata.append({
                "image_type": "ground_truth",
                "prompt": all_gt_prompts[i],
                "image_path": gt_path
            })
    metadata_path = os.path.join(rep_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"10 representative images & metadata saved to {rep_dir}")
    
    pipe = pipe.to(torch.device("cpu"))
    del pipe
    if "pez_model" in args:
        pez_model = args.pez_model.to(torch.device("cpu"))
        del pez_model
        del args.pez_model
    torch.cuda.empty_cache()

    sim_model = torch.jit.load("sscd_disc_large.torchscript.pt").to(device)

    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
            args.reference_model,
            pretrained=args.reference_model_pretrain,
            device=device,
        )
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    # eval
    print("eval")
    gt_clip_scores = []
    gen_clip_scores = []
    SSCD_sims = []
    SSCD_sims_max = []
    SSCD_sims_min = []

    for i in tqdm(range(len(all_gen_images))):
        gen_images = all_gen_images[i]
        gt_images = all_gt_images[i]
        prompt = all_gen_prompts[i]
        gt_prompt = all_gt_prompts[i]

        ### SSCD sim
        SSCD_sim = measure_SSCD_similarity(gt_images, gen_images, sim_model, device)
        gt_image = gt_images[SSCD_sim.argmax(dim=0)[0].item()]
        SSCD_sim_per_image = SSCD_sim.tolist()  
        SSCD_sim = SSCD_sim.max(0).values
        SSCD_sim_max = SSCD_sim.max().item()
        SSCD_sim_min = SSCD_sim.min().item()
        SSCD_sim = SSCD_sim.mean().item()

        SSCD_sims.append(SSCD_sim)
        SSCD_sims_max.append(SSCD_sim_max)
        SSCD_sims_min.append(SSCD_sim_min)

        results["sscd_scores_per_image"].append({
            "gt": None, 
            "gen": SSCD_sim_per_image, 
        })

        ### clip score
        if args.reference_model is not None:
            sims = measure_CLIP_similarity(
                [gt_image] + gen_images,
                gt_prompt,
                ref_model,
                ref_clip_preprocess,
                ref_tokenizer,
                device,
            )
            gt_clip_score = sims[0:1].mean().item()
            gen_clip_scores_per_image = sims[1:].tolist()  #TODO:
            gen_clip_score = sims[1:].mean().item()
        else:
            gt_clip_score = 0
            gen_clip_scores_per_image = [0] * len(gen_images) #TODO:
            gen_clip_score = 0

        gt_clip_scores.append(gt_clip_score)
        gen_clip_scores.append(gen_clip_score)

        results["clip_scores_per_image"].append({
            "gt": gt_clip_score, 
            "gen": gen_clip_scores_per_image, 
        })

        if args.with_tracking:
            table.add_data(
                gt_prompt,
                prompt,
                gt_clip_score,
                gen_clip_score,
                SSCD_sim,
                SSCD_sim_max,
                SSCD_sim_min,
            )

    results["clip_scores"] = gen_clip_scores  
    results["sscd_scores"] = SSCD_sims  

    if args.num_images_to_save >= 0:
        num_to_save = min(args.num_images_to_save, len(results["gen_images"]))
        print(f"Truncating image lists: saving the first {num_to_save} image pairs to the results file.")
        results["gen_images"] = results["gen_images"][:num_to_save]
        results["gt_images"] = results["gt_images"][:num_to_save]
    else:
        print("Saving all generated and ground-truth images to the results file (num_images_to_save < 0).")

    if args.no_save_image:
        results["gen_images"] = []
        results["gt_images"] = []
        if hasattr(args, "num_images_to_save"):
            args.num_images_to_save = 0

    args.result_path = os.path.join(output_dir, "results_with_scores.pkl")
    with open(args.result_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Updated results with per-image evaluation scores saved to {args.result_path}")

    if args.with_tracking:
        wandb.log({"Table": table})
        wandb.log(
            {
                "gt_clip_score_mean": mean(gt_clip_scores),
                "gen_clip_score_mean": mean(gen_clip_scores),
                "SSCD_sim_mean": mean(SSCD_sims),
                "SSCD_sim_max_mean": mean(SSCD_sims_max),
                "SSCD_sim_min_mean": mean(SSCD_sims_min),
            }
        )

    print(f"gt_clip_score_mean: {mean(gt_clip_scores)}")
    print(f"gen_clip_score_mean: {mean(gen_clip_scores)}, 25th percentile: {np.percentile(gen_clip_scores, 25)}, 75th percentile: {np.percentile(gen_clip_scores, 75)}")
    print(f"SSCD_sim_mean: {mean(SSCD_sims)}, 25th percentile: {np.percentile(SSCD_sims, 25)}, 75th percentile: {np.percentile(SSCD_sims, 75)}")
    print(f"SSCD_sim_max_mean: {mean(SSCD_sims_max)}")
    print(f"SSCD_sim_min_mean: {mean(SSCD_sims_min)}")
    print(f"Loss-Rank Exposure: {mean(results['loss_exposure']) if 'loss_exposure' in results and results['loss_exposure'] else 'N/A'}")
    print(f"SSCD-Rank Exposure: {mean(results['sscd_exposure']) if 'sscd_exposure' in results and results['sscd_exposure'] else 'N/A'}")


    print(f'avg_sscd\t25th_sscd\t75th_sscd\tavg_clip\t25th_clip\t75th_clip')
    print(f'{mean(SSCD_sims):.3f}\t{np.percentile(SSCD_sims, 25):.3f}\t{np.percentile(SSCD_sims, 75):.3f}\t{mean(gen_clip_scores):.3f}\t{np.percentile(gen_clip_scores, 25):.3f}\t{np.percentile(gen_clip_scores, 75):.3f}')

    print('----------------- mean & CI -----------------')
    loaded_results = pd.read_pickle(f'{output_dir}/results_with_scores.pkl')
    sscd_scores_per_image = loaded_results['sscd_scores_per_image']
    clip_scores_per_image = loaded_results['clip_scores_per_image']

    def aggregate_scores_with_ci(scores_per_image, key='gen'):
        row_means = []
        for score in scores_per_image:
            data = score[key]
            
            if not data:
                continue

            if isinstance(data[0], list):
                matrix = np.array(data)
                best_match_scores = matrix.max(axis=0) 
                row_means.append(np.mean(best_match_scores))
                
            else:
                row_means.append(np.mean(data))
        
        mean, ci = compute_mean_ci(row_means)
        return mean, ci
    
    def compute_mean_ci(data):
        data = np.array(data)
        mean = np.mean(data)
        ci = 1.96 * np.std(data) / np.sqrt(len(data)) if len(data) > 1 else 0.0
        return mean, ci

    sscd_mean, sscd_ci = aggregate_scores_with_ci(sscd_scores_per_image, key='gen')
    clip_mean, clip_ci = aggregate_scores_with_ci(clip_scores_per_image, key='gen')

    print(f'Final avg_sscd: {sscd_mean:.4f} ± {sscd_ci:.4f}, Final avg_clip: {clip_mean:.4f} ± {clip_ci:.4f}')
    print(f'sscd_avg\tsscd_ci\t\tclip_avg\tclip_ci')
    print(f'{sscd_mean:.3f}\t{sscd_ci:.3f}\t\t{clip_mean:.3f}\t{clip_ci:.3f}')

    print(f"\n---------- Zipping the results folder: {output_dir} ----------")
    try:
        archive_path = shutil.make_archive(base_name=output_dir,
                                           format='zip',
                                           root_dir=os.path.dirname(output_dir),
                                           base_dir=os.path.basename(output_dir))
    except Exception as e:
        print(f"Unexpected error occurred during zipping: {e}")

if __name__ == "__main__":
    def parse_topk(x):
        try:
            if '.' in x:
                return float(x)
            else:
                return int(x)
        except ValueError:
            raise argparse.ArgumentTypeError("Topk must be either an integer or a float (e.g. 0.7).")

    
    parser = argparse.ArgumentParser(description="diffusion memorization")
    parser.add_argument("--unlearn_type", default=None)
    parser.add_argument("--benchmark", action="store_true", help="Run in benchmark mode: disable logging, wandb, extra checks, and evaluation to measure pure generation time.")

    parser.add_argument("--run_name", default="test")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=500, type=int)
    parser.add_argument("--image_length", default=512, type=int)
    parser.add_argument("--model_id", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--unet_id", default=None)
    parser.add_argument("--with_tracking", action="store_true")
    parser.add_argument("--num_images_per_prompt", default=4, type=int)
    parser.add_argument("--guidance_scale", default=7.5, type=float)
    ##### for dual-guidance:
    parser.add_argument("--repulsion_scale", default=3.0, type=float, help="Guidance scale for negative repulsive (memorized) prompt in dual-guidance.")
    parser.add_argument("--num_inference_steps", default=50, type=int)
    parser.add_argument("--reference_model", default=None)
    parser.add_argument("--reference_model_pretrain", default="laion2b_s12b_b42k")
    parser.add_argument("--gen_seed", default=0, type=int)
    # parser.add_argument("--result_path", default="results.pkl", type=str)
    parser.add_argument("--data_path", default=None, type=str)
    # mitigation strategy
    # baseline
    parser.add_argument("--prompt_aug_style", default=None)  # rand_numb_add, rand_word_add, rand_word_repeat
    parser.add_argument("--repeat_num", default=1, type=int)
    parser.add_argument("--template_mem", action="store_true") # one-to-many / many-to-many prompt-image pairs

    # ours
    parser.add_argument("--optim_target_steps", default=0, type=int)
    parser.add_argument("--optim_lr", default=0.05, type=float)
    parser.add_argument("--optim_iters", default=10, type=int)
    parser.add_argument("--optim_target_loss", default=None, type=float)
    # use paraphrased prompt to replace the loss function term: empty-prompt embedding
    parser.add_argument("--use_customised_prompt", action="store_true")
    import ast
    parser.add_argument('--sim', default=None, type=str) # ssim / lpips
    parser.add_argument('--repeat_token', default=None, type=int) 
    # neighbor replacement
    parser.add_argument('--p_replacements', default=0.3, type=float, help='Proportion of words to replace')
    parser.add_argument('--num_neighbors', default=-1, type=int, help='Number of neighbor prompts to generate')
    parser.add_argument('--neighbor_gating_mode', default='none', type=str, help='none / hard / soft')
    parser.add_argument('--orthogonalize_neighbors', action='store_true', help='Orthogonalize neighbor embeddings')
    parser.add_argument("--num_images_to_save", type=int, default=100, help="Number of GT/Gen image pairs to save in the final results file. Set to -1 to save all.")
    # visualization check for neighbors
    parser.add_argument("--attn_timesteps", type=str, default="999,750,500,250,0", help="Comma-separated denoise timesteps for attention capture.")
    parser.add_argument("--attn_topk_percent", type=int, default=10, help="Top-k%% for IoU over spatial attention.")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="If set, randomly sample this many examples (after dedup/selection).")
    parser.add_argument("--sample_indices_path", type=str, default=None, help="JSON path to save or load sampled ORIGINAL indices for reproducibility.")
    parser.add_argument("--no_save_image", action="store_true", help="If set, do NOT store any GT/gen image objects in the results pickle. ")

    parser.add_argument("--xattn_log", action="store_true")
    parser.add_argument("--eot_shrink_alpha", type=float, default=0.0, help="scale EOT down by (1-alpha).")  # 0.A (default 0.0)
    parser.add_argument("--eot_noise_sigma", type=float, default=0.0, help="add noise sigma to EOT cross-attention.")  # 0.A (default 0.0)
    parser.add_argument("--eot_mean_pool", action="store_true", help="If set, use EOT mean pooling.")  # 0.C (flag)
    parser.add_argument("--eot_logit_bias_beta", type=float, default=0.0, help="EOT logit bias beta.")  # 0.B (default 0.0)
    parser.add_argument("--ref_suffix", type=str, default=None, help="good suffix for EOT replacement")
    parser.add_argument("--eot_blend_alpha", type=float, default=0.0, help="EOT blend alpha for causal re-encode.")
    ### cross-attention rescale
    parser.add_argument("--xattn_rescale", action="store_true", help="If set, enable cross-attention rescaling.")
    parser.add_argument("--pst_bias", type=float, default=0.0, help="Pre-Suffix Token (old EOT) bias value.")
    parser.add_argument("--eot_bias", type=float, default=0.0, help="EOT bias value.")
    parser.add_argument("--bot_bias", type=float, default=0.0, help="Beginning of Text (BOT) bias value.")
    parser.add_argument("--bias_schedule", type=str, default="constant", choices=["constant", "linear_fadeout", "cosine_fadeout", "linear_fadein", "cosine_fadein"], help="Bias schedule type.")    
    parser.add_argument("--bias_start_frac", type=float, default=0.0, help="Start fraction for bias schedule.")
    parser.add_argument("--bias_end_frac", type=float, default=1.0, help="End fraction for bias schedule.")
    parser.add_argument("--bias_on", type=str, default="cond", choices=["both", "cond"], help="Which parts to apply bias to.")
    parser.add_argument("--dynamic_head_selection", action="store_true", help="If set, enable dynamic head selection based on scouting.")
    parser.add_argument("--dynamic_top_k_per_block", type=float, default=1.0, help="Proportion of top heads to select per U-Net block during dynamic head selection.")
    parser.add_argument("--dynamic_top_k_global", type=float, default=0.0, help="Proportion of top heads to select globally during dynamic head selection.")
    parser.add_argument("--intervention_blocks", type=str, default=None, help='BLOCKS_ALL / BLOCK_MID / BLOCKS_SEMANTIC / BLOCKS_DETAIL / custom list (e.g., ["mid_block", "up_blocks.1.attentions.1"])')
    parser.add_argument("--spike_threshold", type=float, default=0.1, help="Threshold for spike detection in cross-attention heads.") # default 3.0
    parser.add_argument("--spike_penalty", type=float, default=0.0, help="Penalty bias for detected spikes in cross-attention heads.")
    parser.add_argument("--spike_scale", type=float, default=1.0, help="Scaling factor for spike penalty.")
    # basin
    parser.add_argument("--basin_beta", type=float, default=0.0, help="Basin attenuation beta value.")
    # parser.add_argument("--mem", default=None)  # high / med / low
    parser.add_argument("--topk", type=parse_topk, default=None, help="For sdv1_500_mem_groundtruth: use a fixed topk (int) or a hard threshold (float, e.g. 0.7) for selection.")

    parser.add_argument("--mem_file", default=None)
    parser.add_argument("--collect_step", default=1, type=int)
    parser.add_argument("--split", default=None, type=str) # forget / retain
    parser.add_argument("--non_mem_dataset", type=str, default=None,)
    parser.add_argument("--mem_dataset", type=str, default=None,)
    args = parser.parse_args()
    rich_print(args)

    total_start_time = time.time()
    main(args)
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"-------- Total runtime: {total_duration:.3f} seconds ---------")

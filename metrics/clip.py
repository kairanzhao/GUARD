# clip_eval.py
import pickle
import torch
import wandb
import copy
from statistics import mean
import glob
import PIL
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import open_clip
from optim_utils import *
from io_utils import *

from local_sd_pipeline import LocalStableDiffusionPipeline
from diffusers import DDIMScheduler, UNet2DConditionModel


def main(args):
    if 'laion_10k' in args.unet_id:
        non_mem_ds = 'laion_10k'
    elif 'laion_aesthetic' in args.unet_id:
        non_mem_ds = 'laion_aesthetic'
    else:
        non_mem_ds = args.non_mem_dataset
    output_dir = f"./results/{args.unlearn_type}_nonmem_{non_mem_ds}_gen_{args.dataset}_start{args.start}_end{args.end}_{args.run_name}_s{args.gen_seed}"
    args.result_path = os.path.join(output_dir, "results.pkl")

    os.makedirs(output_dir, exist_ok=True) 
    image_output_dir = os.path.join(output_dir, "gt_images_diff")  
    os.makedirs(image_output_dir, exist_ok=True)  

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
                # "SSCD_sim",
                # "SSCD_sim_max",
                # "SSCD_sim_min",
            ]
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ######## start: calculate clip when results files are available ########
    # Load intermediate results
    # with open(args.result_path, "rb") as f:
    #     results = pickle.load(f)

    # all_gen_images = results["gen_images"]
    # all_gt_images = results["gt_images"]
    # all_gen_prompts = results["gen_prompts"]
    # all_gt_prompts = results["gt_prompts"]
    ######## end: calculate clip when results files are available ########

    ######## start: calculate clip directly from the dataset ########
    all_gen_images = []
    all_gt_images = []
    all_gen_prompts = []
    all_gt_prompts = []
    all_image_paths = []

    set_random_seed(args.gen_seed)
    if args.dataset == 'memorized_images':
        dataset, prompt_key = get_dataset_finetune(args.dataset)
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
        print(f"Loaded {len(dataset)} images from {args.dataset}")
    else:
        dataset, prompt_key = get_dataset_finetune(args.dataset, args=args)
        print(f"Loaded {len(dataset)} images from {args.dataset}")

    args.end = len(dataset)

    for i in tqdm(range(args.start, args.end)):
        seed = i + args.gen_seed

        gt_prompt = dataset[i][prompt_key]
        prompt = gt_prompt    

        if "groundtruth" in args.dataset:
            gt_images = []

            curr_index = dataset[i]["index"]
            for filename in glob.glob(f"{args.data_path}/{args.dataset}/gt_images/{curr_index}/*.png"):
                im = PIL.Image.open(filename)
                gt_images.append(im)
        else:
            gt_images = [dataset[i]["image"]]

        # if there are multiple images, skip this iteration
        if len(gt_images) > 1:
            print(f"multiple gt_images! {len(gt_images)} images in {curr_index}")
            continue

        
        all_gt_images.append(gt_images)
        all_gt_prompts.append(gt_prompt)

        # Save images to disk
        image_filename = f"gt_image_{i}.png"  
        image_path = os.path.join(image_output_dir, image_filename)  
        gt_images[0].save(image_path)  
        all_image_paths.append(image_path) 

    print(f"Loaded {len(all_gt_images)} images from {args.dataset}")
######## end: calculate clip directly from the dataset ########

    # # Load CLIP model
    # ref_model, ref_preprocess, ref_tokenizer = open_clip.create_model_and_transforms(
    #     args.reference_model,
    #     pretrained=args.reference_model_pretrain,
    #     device="cuda",
    # )
    
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
            args.reference_model,
            pretrained=args.reference_model_pretrain,
            device=device,
        )
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)
 

    # CLIP evaluation
    gt_clip_scores = []
    gen_clip_scores = []

    print("Starting CLIP evaluation...")
    # args.end = min(args.end, len(all_gen_images))
    args.end = min(args.end, len(all_gt_images))

    for i in tqdm(range(args.end)):
        # gen_images = all_gen_images[i]
        gt_image = all_gt_images[i][0]
        # prompt = all_gen_prompts[i]
        gt_prompt = all_gt_prompts[i]
        gen_images = []
        prompt = []

        if args.reference_model is not None:
            sims = measure_CLIP_similarity(
                [gt_image] + gen_images, 
                gt_prompt,
                ref_model,
                ref_clip_preprocess,
                ref_tokenizer,
                device,
            )
            print(f"gt_clip_score: {sims[0]}, gen_clip_score: {sims[1:]}")
            gt_clip_score = sims[0:1].mean().item()
            gen_clip_score = sims[1:].mean().item()
        else:
            gt_clip_score = 0
            gen_clip_score = 0            

        gt_clip_scores.append(gt_clip_score)
        gen_clip_scores.append(gen_clip_score)

        if args.with_tracking:
            table.add_data(
                gt_prompt,
                prompt,
                gt_clip_score,
                gen_clip_score,
                # SSCD_sim,
                # SSCD_sim_max,
                # SSCD_sim_min,
            )

        print(f'image {i}: gt_clip_score: {gt_clip_score}, gen_clip_score: {gen_clip_score}')

    if args.with_tracking:
        wandb.log({"Table": table})
        wandb.log(
            {
                "gt_clip_score_mean": mean(gt_clip_scores),
                "gen_clip_score_mean": mean(gen_clip_scores),
                # "SSCD_sim_mean": mean(SSCD_sims),
                # "SSCD_sim_max_mean": mean(SSCD_sims_max),
                # "SSCD_sim_min_mean": mean(SSCD_sims_min),
            }
        )

    print(f"gt_clip_score_mean: {mean(gt_clip_scores)}, gt_min_clip_score: {min(gt_clip_scores)}, gt_max_clip_score: {max(gt_clip_scores)}")
    print(f"gen_clip_score_mean: {mean(gen_clip_scores)}, gen_min_clip_score: {min(gen_clip_scores)}, gen_max_clip_score: {max(gen_clip_scores)}")


    with open(os.path.join(output_dir, "clip_scores.pkl"), "wb") as f:
        pickle.dump(
            {
                "gt_clip_scores": gt_clip_scores,
                # "gen_clip_scores": gen_clip_scores,
                "gt_prompts": all_gt_prompts,
                # "gen_prompts": all_gen_prompts,
                "gt_image_paths": all_image_paths,
            },
            f,
        )

    # plot the distribution of clip scores
    plt.hist(gt_clip_scores, bins=30, alpha=0.5, label="gt_clip_scores")
    plt.xlim(0, 1)
    plt.xlabel("CLIP score")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "clip_scores_diff.png"))
    plt.show()



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CLIP Evaluation")
    parser.add_argument("--result_path", default="results.pkl", type=str)
    parser.add_argument("--unlearn_type", default=None) # unlearning type: training / inference

    # parser.add_argument("--reference_model", default="ViT-B/32", type=str)
    parser.add_argument("--reference_model", default=None)
    parser.add_argument("--reference_model_pretrain", default="laion2b_s12b_b42k")
    
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
    parser.add_argument("--num_inference_steps", default=50, type=int)
    # parser.add_argument("--reference_model", default=None)
    # parser.add_argument("--reference_model_pretrain", default="laion2b_s12b_b42k")
    parser.add_argument("--gen_seed", default=0, type=int)

    parser.add_argument(
        "--prompt_aug_style", default=None
    )  
    parser.add_argument("--repeat_num", default=1, type=int)

    parser.add_argument("--optim_target_steps", default=0, type=int)
    parser.add_argument("--optim_lr", default=0.05, type=float)
    parser.add_argument("--optim_iters", default=10, type=int)
    parser.add_argument("--optim_target_loss", default=None, type=float)

    parser.add_argument("--non_mem_dataset", type=str, default=None,)
    parser.add_argument("--data_path", default="/home/k/xxxxxxxxxx/data/laion", type=str)
    parser.add_argument("--mem", default=None)


    args = parser.parse_args()
    main(args)

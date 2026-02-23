import pickle
import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean
from PIL import Image
import torch
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
    output_dir = f"./results/{args.unlearn_type}_nonmem_{non_mem_ds}_gen_{args.dataset}_start{args.start}_end{args.end}_{args.run_name}_s{args.gen_seed}"
    args.result_path = os.path.join(output_dir, "results.pkl")

    # Load intermediate results
    with open(args.result_path, "rb") as f:
        results = pickle.load(f)

    all_gen_images = results["gen_images"]
    all_gt_images = results["gt_images"]

    table = None
    if args.with_tracking:
        wandb.init(
            project="mitigate_memorization", name=f'{args.unlearn_type}-{args.run_name}', tags=["run_mem"]
        )
        wandb.config.update(args)
        table = wandb.Table(
            columns=[
                "SSCD_sim",
                "SSCD_sim_max",
                "SSCD_sim_min",
                "SSCD_sim_mean",
            ]
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load SSCD similarity model
    sim_model = torch.jit.load("sscd_disc_large.torchscript.pt").to(device)
    sim_model.eval()  
    # eval
    print("Starting SSCD evaluation...")
    SSCD_sims = []
    SSCD_sims_max = []
    SSCD_sims_min = []
    SSCD_sims_mean = []

    for i in tqdm(range(len(all_gen_images))):
        gen_images = all_gen_images[i]
        gt_images = all_gt_images[i]

        ### SSCD sim
        SSCD_sim = measure_SSCD_similarity(gt_images, gen_images, sim_model, device)
        gt_image = gt_images[SSCD_sim.argmax(dim=0)[0].item()]
        SSCD_sim = SSCD_sim.max(0).values

        SSCD_sim_max = SSCD_sim.max().item()
        SSCD_sim_min = SSCD_sim.min().item()
        SSCD_sim = SSCD_sim.mean().item()

        SSCD_sims.append(SSCD_sim)
        SSCD_sims_max.append(SSCD_sim_max)
        SSCD_sims_min.append(SSCD_sim_min)


        if args.with_tracking:
            table.add_data(
                SSCD_sim,
                SSCD_sim_max,
                SSCD_sim_min,
            )

    if args.with_tracking:
        wandb.log({"Table": table})
        wandb.log(
            {
                "SSCD_sim_mean": mean(SSCD_sims),
                "SSCD_sim_max_mean": mean(SSCD_sims_max),
                "SSCD_sim_min_mean": mean(SSCD_sims_min),
            }
        )

    print(f"SSCD_sim_mean: {mean(SSCD_sims)}")
    print(f"SSCD_sim_max_mean: {mean(SSCD_sims_max)}, len: {len(SSCD_sims_max)}, max: {max(SSCD_sims_max)}")
    print(f"SSCD_sim_min_mean: {mean(SSCD_sims_min)}, len: {len(SSCD_sims_min)}, min: {min(SSCD_sims_min)}")

    plt.hist(SSCD_sims, bins=30)
    plt.xlim(-0.1, 1)
    plt.xlabel("SSCD similarity score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "SSCD_sims.png"))
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="diffusion memorization")
    parser.add_argument("--result_path", default="results.pkl", type=str)
    parser.add_argument("--unlearn_type", default=None) # unlearning type: training / inference

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
    parser.add_argument("--reference_model", default=None)
    parser.add_argument("--reference_model_pretrain", default="laion2b_s12b_b42k")
    parser.add_argument("--gen_seed", default=0, type=int)

    parser.add_argument(
        "--prompt_aug_style", default=None
    )  
    parser.add_argument("--repeat_num", default=1, type=int)

    parser.add_argument("--optim_target_steps", default=0, type=int)
    parser.add_argument("--optim_lr", default=0.05, type=float)
    parser.add_argument("--optim_iters", default=10, type=int)
    parser.add_argument("--optim_target_loss", default=None, type=float)

    args = parser.parse_args()

    main(args)
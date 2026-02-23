import argparse
import os
import torch
from tqdm import tqdm
from accelerate.utils import ProjectConfiguration, set_seed
from local_sd_pipeline import LocalStableDiffusionPipeline
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer
from optim_utils import *

def parse_topk(x):
    try:
        # If the string contains a dot, treat it as a float.
        if '.' in x:
            return float(x)
        else:
            return int(x)
    except ValueError:
        raise argparse.ArgumentTypeError("Topk must be either an integer or a float (e.g. 0.7).")
    
def generate_mask(
    args=None,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.unet_id is not None:
        unet = UNet2DConditionModel.from_pretrained(
            args.unet_id,
        ).to(device)
    else:
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
        ).to(device)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    ).to(device)

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    ).to(device)

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Image transformations
    train_transforms = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    # Load dataset
    set_seed(args.seed)
    dataset, prompt_key = get_dataset_finetune(args.dataset, args=args)
    print(f"Dataset: {dataset}")

    def collate_fn(batch):
        images = torch.stack([train_transforms(ex["image"]) for ex in batch])
        prompts = [ex[prompt_key] for ex in batch]
        return images.to(device), prompts

    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=args.dataloader_num_workers, collate_fn=collate_fn
    )

    criteria = torch.nn.MSELoss()

    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Initialize gradient storage
    gradients = {name: torch.zeros_like(param, device=device) for name, param in unet.named_parameters()}
        
    def get_text_embeddings(prompts, tokenizer, text_encoder, device):
        # inputs = tokenizer(
        inputs = tokenizer.batch_encode_plus(
            prompts,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            return text_encoder(inputs.input_ids.to(device))[0]  # Returns (batch_size, seq_len, hidden_dim)


    with tqdm(total=len(dataloader)) as t:
        for images, prompts in tqdm(dataloader):
            optimizer.zero_grad()

            # Encode images into latents
            with torch.no_grad():
                latents = vae.encode(images.to(vae.dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            null_prompt = ""
            # conditioned_embeds = [get_text_embeddings(prompt, tokenizer, text_encoder, device) for prompt in prompts]
            # unconditioned_embeds = [get_text_embeddings(null_prompt, tokenizer, text_encoder, device) for _ in prompts]

            conditioned_embeds = get_text_embeddings(prompts, tokenizer, text_encoder, device)
            unconditioned_embeds = get_text_embeddings([null_prompt] * len(prompts), tokenizer, text_encoder, device)
            
            # Generate noise and perform diffusion step
            t_step = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device).long()

            noise = torch.randn_like(latents, device=device)
            # noisy_image = noise_scheduler.add_noise(image, noise, t_step)
            noisy_latents = noise_scheduler.add_noise(latents, noise, t_step)
            
            conditioned_out = unet(noisy_latents, t_step, conditioned_embeds).sample
            unconditioned_out = unet(noisy_latents, t_step, unconditioned_embeds).sample
            
            # Compute loss
            preds = (1 + args.c_guidance) * conditioned_out - args.c_guidance * unconditioned_out
            loss = -criteria(noise, preds)
            loss.backward()
            
            # Accumulate gradients
            with torch.no_grad():
                for name, param in unet.named_parameters():
                    if param.grad is not None:
                        gradients[name] += param.grad.detach()
            
            t.update(1)

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs(gradients[name])

        mask_path = os.path.join("mask", args.dataset)
        os.makedirs(mask_path, exist_ok=True)

        threshold = 0.5
        all_elements = - torch.cat([tensor.flatten().cpu() for tensor in gradients.values()])
        threshold_index = int(len(all_elements) * threshold)
        torch.cuda.empty_cache()
        
        positions = torch.argsort(all_elements)
        ranks = torch.argsort(positions)
        
        mask_dict = {}
        start_idx = 0
        for name, tensor in gradients.items():
            num_elements = tensor.numel()
            tensor_ranks = ranks[start_idx : start_idx + num_elements]
            threshold_tensor = (tensor_ranks < threshold_index).reshape(tensor.shape).to(torch.uint8)
            mask_dict[name] = threshold_tensor
            start_idx += num_elements
        
        # Save mask
        torch.save(mask_dict, os.path.join(mask_path, "mask.pt"))
        print(f"Gradient mask saved to {mask_path}/mask.pt")


def generate_nsfw_mask(
    c_guidance,
    batch_size,
    epochs,
    lr,
    config_path,
    ckpt_path,
    diffusers_config_path,
    device,
    image_size=512,
    num_timesteps=1000,
):
    # MODEL TRAINING SETUP
    model = setup_model(config_path, ckpt_path, device)
    train_dl, _ = setup_forget_nsfw_data(batch_size, image_size)

    model.eval()
    criteria = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.model.diffusion_model.parameters(), lr=lr)
    word_nude = "a photo of a nude person"

    gradients = {}
    for name, param in model.model.diffusion_model.named_parameters():
        gradients[name] = 0

    with tqdm(total=len(train_dl)) as t:
        for i, images in enumerate(train_dl):
            optimizer.zero_grad()

            images = images.to(device)
            t = torch.randint(
                0, num_timesteps, (images.shape[0],), device=device
            ).long()

            null_prompts = [""] * batch_size
            prompts = [word_nude] * batch_size

            forget_batch = {"jpg": images.permute(0, 2, 3, 1), "txt": prompts}
            null_batch = {"jpg": images.permute(0, 2, 3, 1), "txt": null_prompts}

            forget_input, forget_emb = model.get_input(
                forget_batch, model.first_stage_key
            )
            null_input, null_emb = model.get_input(null_batch, model.first_stage_key)

            t = torch.randint(
                0, model.num_timesteps, (forget_input.shape[0],), device=device
            ).long()
            noise = torch.randn_like(forget_input, device=device)

            forget_noisy = model.q_sample(x_start=forget_input, t=t, noise=noise)

            forget_out = model.apply_model(forget_noisy, t, forget_emb)
            null_out = model.apply_model(forget_noisy, t, null_emb)

            preds = (1 + c_guidance) * forget_out - c_guidance * null_out

            loss = - criteria(noise, preds)
            loss.backward()

            with torch.no_grad():
                for name, param in model.model.diffusion_model.named_parameters():
                    if param.grad is not None:
                        gradients[name] += param.grad.data.cpu()

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

        threshold_list = [0.5]
        for i in threshold_list:
            sorted_dict_positions = {}
            hard_dict = {}

            # Concatenate all tensors into a single tensor
            all_elements = - torch.cat([tensor.flatten() for tensor in gradients.values()])

            # Calculate the threshold index for the top 10% elements
            threshold_index = int(len(all_elements) * i)

            # Calculate positions of all elements
            positions = torch.argsort(all_elements)
            ranks = torch.argsort(positions)

            start_index = 0
            for key, tensor in gradients.items():
                num_elements = tensor.numel()
                # tensor_positions = positions[start_index: start_index + num_elements]
                tensor_ranks = ranks[start_index : start_index + num_elements]

                sorted_positions = tensor_ranks.reshape(tensor.shape)
                sorted_dict_positions[key] = sorted_positions

                # Set the corresponding elements to 1
                threshold_tensor = torch.zeros_like(tensor_ranks)
                threshold_tensor[tensor_ranks < threshold_index] = 1
                threshold_tensor = threshold_tensor.reshape(tensor.shape)
                hard_dict[key] = threshold_tensor
                start_index += num_elements

            torch.save(hard_dict, os.path.join("mask/nude_{}.pt".format(i)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Train", description="train a stable diffusion model from scratch"
    )

    parser.add_argument(
        "--classes",
        help="class corresponding to concept to erase",
        type=str,
        required=False,
        default="6",
    )
    parser.add_argument(
        "--c_guidance",
        help="guidance of start image used to train",
        type=float,
        required=False,
        default=7.5,
    )
    parser.add_argument(
        "--train_batch_size",
        help="batch_size used to train",
        type=int,
        required=False,
        default=16,
    )
    parser.add_argument(
        "--epochs", help="epochs used to train", type=int, required=False, default=1
    )
    parser.add_argument(
        "--learning_rate",
        help="learning rate used to train",
        type=float,
        required=False,
        default=1e-5,
    )
    parser.add_argument(
        "--ckpt_path",
        help="ckpt path for stable diffusion v1-4",
        type=str,
        required=False,
        default="models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt",
    )
    parser.add_argument(
        "--config_path",
        help="config path for stable diffusion v1-4 inference",
        type=str,
        required=False,
        default="configs/stable-diffusion/v1-inference.yaml",
    )
    parser.add_argument(
        "--diffusers_config_path",
        help="diffusers unet config json path",
        type=str,
        required=False,
        default="diffusers_unet_config.json",
    )
    parser.add_argument(
        "--device",
        help="cuda devices to train on",
        type=str,
        required=False,
        default="0",
    )
    parser.add_argument(
        "--image_size",
        help="image size used to train",
        type=int,
        required=False,
        default=512,
    )
    parser.add_argument(
        "--num_timesteps",
        help="ddim steps of inference used to train",
        type=int,
        required=False,
        default=1000,
    )
    parser.add_argument(
        "--nsfw", help="class or nsfw", type=bool, required=False, default=False
    )

    # parser.add_argument("--model_id", default="CompVis/stable-diffusion-v1-4")
    # parser.add_argument("--unet_id", default=None)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--unet_id", default=None)

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )

    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    
    parser.add_argument("--data_path", default=None, type=str)

    parser.add_argument("--topk", type=parse_topk, default=0.7, help="For sdv1_500_mem_groundtruth: use a fixed topk (int) or a hard threshold (float, e.g. 0.7) for selection.")
    parser.add_argument("--mem", default=None)


    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    c_guidance = args.c_guidance
    batch_size = args.train_batch_size
    epochs = args.epochs
    lr = args.learning_rate
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    diffusers_config_path = args.diffusers_config_path
    device = f"cuda:{int(args.device)}"
    image_size = args.image_size
    num_timesteps = args.num_timesteps

    if args.nsfw:
        generate_nsfw_mask(
            c_guidance,
            batch_size,
            epochs,
            lr,
            config_path,
            ckpt_path,
            diffusers_config_path,
            device,
            image_size,
            num_timesteps,
        )
    else:
        generate_mask(
            args,
        )
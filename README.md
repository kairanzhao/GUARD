# Surgical Memorization Mitigation in Text-to-Image Diffusion Models

## Dependencies
- PyTorch/2.0-Miniconda3-4.12.0
- transformers == 4.46.3
- diffusers == 0.31.0
- accelerate == 1.0.1
- datasets == 3.1.0
- lpips == 0.1.4

## Data Preparation
- sdv1_500_mem ("natural memorization") from [[Webster, 2023](https://arxiv.org/pdf/2305.08694)]

    - template memorization
    - verbatim memorization



## Inference-time memorization mitigation
<!-- ```bash
python inference_mem.py --data_path {DATA_PATH} --unlearn_type inference --run_name {CUSTOMISED_NAME} --repeat_token 10 --dataset sdv1_500_mem_groundtruth --topk 0.7 --end 500 --gen_seed {SEED} --reference_model ViT-g-14 
```

- `--repeat_token`: repeats cues to the suffix of the prompt -->

### CA attenuation
```bash
python inference_mem.py --data_path {DATA_PATH} --unlearn_type inference --run_name {CUSTOMISED_NAME} --xattn_rescale --intervention_blocks BLOCKS_EARLY --spike_scale 0.3 --spike_threshold 1.0 --topk 0.7 --dataset sdv1_500_mem_groundtruth --end 500 --gen_seed 0 --reference_model ViT-g-14 
```


<!-- - `--sim`: specifies the similarity metric for the quality-preserving regularizer. Options include clip, mse, l2, lpips, etc.

- `--sim_gamma`: controls the strength of the quality-preserving term in the loss. Defaults to None (i.e., no quality augmentation). -->

### CA-in-GUARD

```bash
python inference_mem.py --data_path {DATA_PATH} --unlearn_type inference --run_name {CUSTOMISED_NAME} --xattn_rescale --intervention_blocks BLOCKS_EARLY --repulsion_scale 0.1 --spike_scale 0.5 --spike_threshold 0.1 --guidance_scale 7.5 --dataset sdv1_500_mem_groundtruth --topk 0.7 --end 500 --gen_seed 0 --prompt_aug_style neighbor_replace --reference_model ViT-g-14
```

<!-- --template_mem  -->

| Method |  Optional Hyperparameters                  |
| --------: | :------------------------------------- |
| SD v1.4 verbatim memorization |  `--model_id {SDV1.4}`         |
| SD v1.4 template memorization |  `--model_id {SDV1.4}`, `--template_mem `          |
| SD v2.0 template memorization |  `--model_id {SDV2.0}`, `--template_mem`          |

<!-- ## Evaluation

For training-time mitigation & unlearning, the ``--dataset`` argument specifies which dataset to evaluate on. This can be either memorized / non-memorized dataset.

```bash
python inference_mem.py --unlearn_type training --run_name {RUN_NAME} --unet_id {OUTPUT_DIR}/checkpoint-20000/unet --dataset {EVAL_DATASET} --mem_dataset {MEMORIZED_DATASET} --non_mem_dataset laion_aesthetics --end 200 --gen_seed {SEED} --reference_model ViT-g-14 
```

| Eval_dataset |    Flag    | Optional Hyperparameters                                 |
| --------: | :--------: | :------------------------------ |
| sdv1_500_mem |  `sdv1_500_mem_groundtruth` | `--topk 0.7`                | -->



# References
- [Wen et al.](https://github.com/YuxinWenRick/diffusion_memorization)
- [RTA](https://github.com/somepago/DCR)
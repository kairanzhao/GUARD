# GUARD

This is the official code repository of the ICML 2026 paper:

[You Don’t Need All That Attention: Surgical Memorization Mitigation in Text-to-Image Diffusion Models](https://arxiv.org/abs/2603.00133)

## Dependencies
- PyTorch/2.0-Miniconda3-4.12.0
- transformers == 4.46.3
- diffusers == 0.31.0
- accelerate == 1.0.1
- datasets == 3.1.0
- lpips == 0.1.4

## Data Preparation
`sdv1_500_mem` is a 500-example Stable Diffusion memorization benchmark proposed by [Webster, 2023](https://arxiv.org/pdf/2305.08694). It contains prompts associated with memorized training images and covers both:

    - verbatim memorization
    - template memorization

  In this repository, the prompts and ground-truth images follow the release from [Wen et al.](https://github.com/YuxinWenRick/diffusion_memorization)'s codebase. Download them from [this link](https://drive.google.com/file/d/1mdhkyTlDBZIW6LaO_Q1J3roaU2Be0Nvo/view?usp=sharing).



## Inference-time memorization mitigation


### CA attenuation
```bash
python inference_mem.py --data_path {DATA_PATH} --unlearn_type inference --run_name {CUSTOMISED_NAME} --xattn_rescale --intervention_blocks BLOCKS_EARLY --spike_scale 0.3 --spike_threshold 0.01 --topk 0.7 --dataset sdv1_500_mem_groundtruth --end 500 --gen_seed 0 --reference_model ViT-g-14 
```


### CA-in-GUARD

```bash
python inference_mem.py --data_path {DATA_PATH} --unlearn_type inference --run_name {CUSTOMISED_NAME} --xattn_rescale --intervention_blocks BLOCKS_EARLY --repulsion_scale 2.0 --spike_scale 0.3 --spike_threshold 0.1 --guidance_scale 7.5 --dataset sdv1_500_mem_groundtruth --topk 0.7 --end 500 --gen_seed 0 --prompt_aug_style neighbor_replace --reference_model ViT-g-14
```

<!-- --template_mem  -->

| Method |  Optional Hyperparameters                  |
| --------: | :------------------------------------- |
| SD v1.4 verbatim memorization |  `--model_id {SDV1.4_ID}`         |
| SD v1.4 template memorization |  `--model_id {SDV1.4_ID}`, `--template_mem `          |
| SD v2.0 template memorization |  `--model_id {SDV2.0_ID}`, `--template_mem`          |


## Citation
If you find this work useful, please cite our paper:
```
@article{zhao2026you,
  title={You Don't Need All That Attention: Surgical Memorization Mitigation in Text-to-Image Diffusion Models},
  author={Zhao, Kairan and Triantafillou, Eleni and Triantafillou, Peter},
  journal={Forty-third International Conference on Machine Learning},
  year={2026}
}
```


## Acknowledgements
This codebase builds on components from the following open-source repositories:

- [Wen et al.](https://github.com/YuxinWenRick/diffusion_memorization)
- [RTA](https://github.com/somepago/DCR)

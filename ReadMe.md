Official implementation of the paper "Diverse Data Generation for Retinal Layer Segmentation with Potential Structure Modelling".

# Data

Public data 'OCTA-500' can be downloaded at: https://ieee-dataport.org/open-access/octa-500.

The pathes of data in `./configs` are required to completed after downloaded the data. 

The partitions of data of our experiments are provided at `./jsons`.

 The examples of prompts for generating large-scale datasets are given as `./jsons/generated_31500.json` and  `./jsons/generated_94500.json`.

# Label generation

Layer code can be pre-transformed from layer labels by `./data_process/gen_layer_data.py`.

1. Pretrain a TDPM:

```shell
# on the OCTA-6MM dataset
python pretrain_TDPM.py --exp_name TDPM_0_6mm --result_root path/to/results  --data_config configs/layergen/octa6mm.yaml
```

2. Finetune a TDPM:
```shell
# on the OCTA-6MM dataset
python finetune_TDPM.py --exp_name TDPM_1_6mm --result_root path/to/results  --data_config configs/layergen/octa6mm.yaml --first_stage_ckpt path/to/pretrained/checkpoint
```


3. Merge LoRA matrics for faster inference
```shell
python TDPM_lora_merge.py --ckpt_path path/to/finetuned/ckpt --save_path path/to/merged/ckpt
```
4. Generate labels by given prompts:
```shell
python test_TDPM.py --result_save_dir path/to/generated/label --ckpt_path path/to/MLDM/checkpoint --test_json_path jsons/generated_94500.json
```

# Label-to-img translation

1.  Train a VQVAE:

```shell
# on the OCTA-6MM dataset
python train_VQVAE.py --exp_name VQVAE_6mm --result_root path/to/results  --data_config configs/imggen/octa6mm.yaml
```

2. Train a MLDM with a trained VQVAE:

```shell
# on the OCTA-6MM dataset
python train_MLDM.py --exp_name MLDM_6mm --result_root path/to/results  --data_config configs/layer2img/octa6mm.yaml --first_stage_ckpt path/to/VQVAE/checkpoint
```

3.  Generate images by given label and prompts:

```shell
python test_MLDM.py --result_save_dir path/to/generated/image --ckpt_path path/to/MLDM/checkpoint  --test_img_root path/to/label --test_json_path jsons/generated_94500.json --first_stage_ckpt path/to/VQVAE/checkpoint
```
---
Our codebase builds on [LatentDiffusion](https://github.com/CompVis/latent-diffusion), [LoRA](https://github.com/microsoft/LoRA), and [MONAI](https://docs.monai.io/en/stable/networks.html#vitautoenc).



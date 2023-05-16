# diffusers

Install requirements

```
pip install -r requirements.txt
```

Set up Hugging Face account to get access token.

## Stage 1 VAE training 

VAE training is performed at the  [modified latent-diffusion repository](https://github.com/adammoss/latent-diffusion).

Once trained models can be uploaded to the HF hub by 

```
python convert_vae.py --config_path <config-path.yaml> --checkpoint_path <checkpoint-path.ckpt> --output_path <output-path> --hub_token <hub-token> --push_to_hub
```

## Stage 2 diffuser training

### Example runs in pixel space: 

SIMBA N-body 64x64 resolution 

```
accelerate launch train.py --dataset_name="Mtot_Nbody_SIMBA" --resolution=64 --data_size=13500 \
--train_batch_size=32 --cache_dir="data" --ddpm_num_steps=4000 --ddpm_num_inference_steps=4000 \
--checkpointing_steps=10000 --use_ema --prediction_type="sample" \
--logger=wandb --push_to_hub --hub_token=<hub-token> 
```

SIMBA N-body 64x64 resolution with conditional parameters

```
accelerate launch train.py --dataset_name="Mtot_Nbody_SIMBA" --resolution=64 --data_size=13500 \
--train_batch_size=32 --cache_dir="data" --ddpm_num_steps=4000 --ddpm_num_inference_steps=4000 \
--checkpointing_steps=10000 --use_ema --prediction_type="sample" --conditional --base_channels=64 \
--logger=wandb --push_to_hub --hub_token=<hub-token>  
```

### Example runs in latent-space

SIMBA N-body and gas mass 128x128 resolution with conditional parameters

```
accelerate launch train.py --dataset_name="Mtot_Nbody_SIMBA" --dataset_name="Mgas_SIMBA" --resolution=128 --data_size 13500 \
--train_batch_size=32 --num_epochs=100 --gradient_accumulation_steps=1 --prediction_type="sample" \
--learning_rate=1e-4 --lr_warmup_steps=500 --mixed_precision=no --cache_dir="data" \
--checkpointing_steps=10000 --use_ema --ddpm_num_steps=4000 --ddpm_num_inference_steps=4000 --conditional \
--vae_from_pretrained="adammoss/cmd_f2_d3_128" --vae="kl" --vae_scaling_factor=0.03 \
--logger=wandb --push_to_hub --hub_token=<hub-token>
```

SIMBA N-body 256x256 resolution with conditional parameters

```
accelerate launch train.py --dataset_name="Mtot_Nbody_SIMBA" --resolution=256 --data_size 13500 \
--train_batch_size=32 --num_epochs=100 --gradient_accumulation_steps=1 --prediction_type="sample" \
--learning_rate=1e-4 --lr_warmup_steps=500 --mixed_precision=no --cache_dir="data" \
--checkpointing_steps=10000 --use_ema --ddpm_num_steps=4000 --ddpm_num_inference_steps=4000 --conditional \
--vae_from_pretrained="adammoss/cmd_f2_d3_256" --vae="kl" --vae_scaling_factor=0.03 \
--logger=wandb --push_to_hub --hub_token=<hub-token>
```

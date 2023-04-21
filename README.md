# diffusers

Install requirements

```
pip install -r requirements.txt
```

Set up Hugging Face account and login to hub

```
huggingface-cli login
```

### Example runs: 

D-sprites 64x64 resolution

```
accelerate launch train.py --dataset_name="dsprites" --resolution=64 \
--train_batch_size=32 --cache_dir="data" --ddpm_num_steps 4000 --ddpm_num_inference_steps 4000 \
--checkpointing_steps=10000 --use_ema --data_size=15000 --push_to_hub
```

D-sprites 64x64 resolution with conditional parameters 

```
accelerate launch train.py --dataset_name="dsprites" --resolution=64 \
--train_batch_size=32 --cache_dir="data" --ddpm_num_steps 4000 --ddpm_num_inference_steps 4000 \
--checkpointing_steps=10000 --use_ema --data_size=15000 --conditional --push_to_hub
```

SIMBA N-body 64x64 resolution

```
accelerate launch train.py --dataset_name="Mtot_Nbody_SIMBA" --resolution=64 --data_size 13500 \
--train_batch_size=32 --cache_dir="data" --ddpm_num_steps 4000 --ddpm_num_inference_steps 4000 \
--checkpointing_steps=10000 --use_ema --prediction_type sample --push_to_hub
```

SIMBA N-body 64x64 resolution with conditional parameters

```
accelerate launch train.py --dataset_name="Mtot_Nbody_SIMBA" --resolution=64 --data_size 13500 \
--train_batch_size=32 --cache_dir="data" --ddpm_num_steps 4000 --ddpm_num_inference_steps 4000 \
--checkpointing_steps=10000 --use_ema --prediction_type sample --conditional --push_to_hub
```

SIMBA N-body 128x128 resolution

```
accelerate launch train.py --dataset_name="Mtot_Nbody_SIMBA" --resolution=128 --data_size 13500 \
--train_batch_size=32  --ddpm_num_steps 4000 --ddpm_num_inference_steps 4000 \
--cache_dir="data" --checkpointing_steps=10000 --use_ema --prediction_type sample --push_to_hub
```

SIMBA N-body 128x128 resolution with conditional parameters

```
accelerate launch train.py --dataset_name="Mtot_Nbody_SIMBA" --resolution=128 --data_size 13500 \
--train_batch_size=4 --gradient_accumulation_steps=1  --ddpm_num_steps 4000 --ddpm_num_inference_steps 400 \
--cache_dir="data" --checkpointing_steps=10000 --use_ema --prediction_type sample --conditional --push_to_hub
```



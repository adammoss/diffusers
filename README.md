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
accelerate launch train_cmd.py --dataset_name="dsprites" --resolution=64  \
--train_batch_size=32 --cache_dir="data" \
--checkpointing_steps=5000 --use_ema --data_size=15000 --push_to_hub
```

D-sprites 64x64 resolution with conditional parameters 

```
accelerate launch train_cmd.py --dataset_name="dsprites" --resolution=64  \
--train_batch_size=32 --cache_dir="data" \
--checkpointing_steps=5000 --use_ema --data_size=15000 --conditional --push_to_hub
```

SIMBA N-body 64x64 resolution

```
accelerate launch train_cmd.py --dataset_name="Mtot_Nbody_SIMBA" --resolution=64  \
--train_batch_size=32 --cache_dir="data" \
--checkpointing_steps=5000 --use_ema --push_to_hub
```

SIMBA N-body 64x64 resolution with conditional parameters

```
accelerate launch train_cmd.py --dataset_name="Mtot_Nbody_SIMBA" --resolution=64  \
--train_batch_size=32 --cache_dir="data" \
--checkpointing_steps=5000 --use_ema --conditional --push_to_hub
```

SIMBA N-body 128x128 resolution with conditional parameters

```
accelerate launch train_cmd.py --dataset_name="Mtot_Nbody_SIMBA" --resolution=128  \
--train_batch_size=8 --gradient_accumulation_steps=4  \
--cache_dir="data" --checkpointing_steps=5000 --use_ema --conditional --push_to_hub
```


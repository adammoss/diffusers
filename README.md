# diffusers

First login to hub

```
huggingface-cli login
```

### Example runs: 

SIMBA N-body 64x64 resolution

```
python train_cmd.py --dataset_field="Mtot_Nbody_SIMBA" --resolution=64  --output_dir="output/test" --train_batch_size=32 --num_epochs=100 --gradient_accumulation_steps=1  --learning_rate=1e-4 --lr_warmup_steps=500 --mixed_precision=no --cache_dir="data" --push_to_hub
```
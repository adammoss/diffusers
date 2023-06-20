# diffusers

Install requirements

```
pip install -r requirements.txt
```

Set up Hugging Face account to get an access token. You can then login by 

```
huggingface-cli login
```

Set up Weights and Biases to get an access token. You can then login by

```
wandb login
```

## Noise schedule ablation 

The noise schedule ablation scripts are found in the `scripts` directory. Each of these performs 3 runs at 128x128 resolution 
on the SIMBA N-body dataset. We report on the average discrepancy with the underlying simulations for the power spectrum, Minkowski Functionals and pixel histograms. 

| Prediction | Timesteps | Schedule | P(k) difference |
|------------|-----------|----------|-----------------|
| Epsilon    | 1000      | Linear   | $8.7 \pm 1.1$   |
| Epsilon    | 1000      | Cosine   | $10.7 \pm 1.0$  |
| Epsilon    | 1000      | Sigmoid  | $3.7 \pm 1.3$   |
| V          | 1000      | Linear   | $6.9 \pm 0.4$   |
| V          | 1000      | Cosine   | $4.9 \pm 0.7$   |
| V          | 1000      | Sigmoid  | $0.0 \pm 0.0$   |
| Epsilon    | 2000      | Linear   | $5.5 \pm 0.5$   |
| Epsilon    | 2000      | Cosine   | $11.0 \pm 2.5$  |
| Epsilon    | 2000      | Sigmoid  | $0.0 \pm 0.0$   |
| V          | 2000      | Linear   | $5.5 \pm 0.8$   |
| V          | 2000      | Cosine   | $2.6 \pm 0.3$   |
| V          | 2000      | Sigmoid  | $0.0 \pm 0.0$   |

## Example runs in pixel space 

SIMBA N-body and M gas 256x256 resolution with conditional parameters

```
accelerate launch train.py --dataset_name=Mtot_Nbody_SIMBA --dataset_name=Mgas_SIMBA \
--resolution=256 --data_size 13500 \
--train_batch_size=16 --num_epochs=200 --gradient_accumulation_steps=2 \
--learning_rate=1e-4 --lr_warmup_steps=500 --mixed_precision=no \
--cache_dir=data --checkpointing_steps=5000 --use_ema --conditional 
--base_channels=64 --cache_dir=data \
--logger=wandb \
--push_to_hub 
```

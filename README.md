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

| Prediction | Timesteps | Schedule | P(k) difference | Hist different  |
|------------|-----------|----------|-----------------|-----------------|
| Epsilon    | 1000      | Linear   | $9.2 \pm 1.1$   | $7.0 \pm 3.2$   |
| Epsilon    | 1000      | Cosine   | $10.5 \pm 0.9$  | $107.4 \pm 6.0$ |
| Epsilon    | 1000      | Sigmoid  | $4.1 \pm 1.4$   | $6.7 \pm 2.2$   |
| V          | 1000      | Linear   | $7.4 \pm 0.4$   | $6.5 \pm 1.0$   |
| V          | 1000      | Cosine   | $5.3 \pm 0.7$   | $7.9 \pm 0.9$   |
| V          | 1000      | Sigmoid  | $5.8 \pm 1.2$   | $8.0 \pm 2.8$   |
| Sample SN  | 1000      | Linear   | $5.1 \pm 0.2$   | $8.3 \pm 0.9$   |
| Sample SN  | 1000      | Cosine   | $2.9 \pm 0.4$   | $12.8 \pm 4.5$  |
| Sample SN  | 1000      | Sigmoid  | $4.7 \pm 1.0$   | $10.2 \pm 1.9$  |
| Epsilon    | 2000      | Linear   | $6.0 \pm 0.5$   | $5.9 \pm 0.8$   |
| Epsilon    | 2000      | Cosine   | $10.5 \pm 2.5$  | $89.9 \pm 20.9$ |
| Epsilon    | 2000      | Sigmoid  | $4.3 \pm 0.1$   | $6.5 \pm 1.8$   |
| V          | 2000      | Linear   | $6.0 \pm 0.8$   | $7.8 \pm 2.0$   |
| V          | 2000      | Cosine   | $3.0 \pm 0.3$   | $5.9 \pm 1.9$   |
| V          | 2000      | Sigmoid  | $6.2 \pm 0.6$   | $8.4 \pm 0.9$   |
| Sample SN  | 2000      | Linear   | $4.0 \pm 0.0$   | $7.8 \pm 2.1$   |
| Sample SN  | 2000      | Cosine   | $1.7 \pm 0.2$   | $11.8 \pm 2.9$  |
| Sample SN  | 2000      | Sigmoid  | $4.3 \pm 0.9$   | $10.5 \pm 1.0$  |

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

for RUN in 1 2 3
do
  python train.py --output_dir="output/ddpm-128-sample-2000-cos-run$RUN" --base_channels=64 \
  --resolution=128 --data_size 13500 --ddpm_beta_schedule="squaredcos_cap_v2"  \
  --dataset_name="Mtot_Nbody_SIMBA" --cache_dir="data" \
  --train_batch_size=64 --eval_batch_size=16 --num_epochs=200 --prediction_type="sample" \
  --learning_rate=1e-4 --lr_warmup_steps=500 --mixed_precision=no \
  --checkpointing_steps=10000 --use_ema \
  --ddpm_num_steps=2000 --base_channels=64 \
  --logger="wandb" \
  --push_to_hub \
  --tag="na1"
done
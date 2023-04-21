from fabric import task, Connection


@task
def install(c):
    if not c.run('test -f diffusers', warn=True).failed:
        c.run('git clone https://github.com/adammoss/diffusers')
    with c.cd('diffusers'):
        c.run('pip install -r requirements.txt')
        c.run('pip install protobuf==3.20.*')
        c.run('sudo apt-get install git-lfs')
        c.run('git lfs install')


@task
def run(c, dataset='', token='', resolution=128, timesteps=4000, conditional=False):
    with c.cd('diffusers'):
        if conditional:
            c.run('/home/ubuntu/.local/bin/accelerate launch train.py --dataset_name=%s --resolution=%s '
                  '--data_size=13500 --train_batch_size=32 --gradient_accumulation_steps=1 '
                  '--ddpm_num_steps=%s --ddpm_num_inference_steps=%s --conditional '
                  '--cache_dir="data" --checkpointing_steps=10000 --use_ema --prediction_type sample '
                  '--push_to_hub --hub_token=%s' % (dataset, resolution, timesteps, timesteps, token))
        else:
            c.run('/home/ubuntu/.local/bin/accelerate launch train.py --dataset_name=%s --resolution=%s '
                  '--data_size=13500 --train_batch_size=32 --gradient_accumulation_steps=1 '
                  '--ddpm_num_steps=%s --ddpm_num_inference_steps=%s '
                  '--cache_dir="data" --checkpointing_steps=10000 --use_ema --prediction_type sample '
                  '--push_to_hub --hub_token=%s' % (dataset, resolution, timesteps, timesteps, token))

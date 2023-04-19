import torch

from diffusers import RePaintPipeline

from pipeline import DDPMConditionPipeline
from repaint_scheduler import RePaintScheduler


def generate_samples(model, batch_size, device=None):
    pipeline = DDPMConditionPipeline.from_pretrained(model)
    if device is not None:
        pipeline = pipeline.to(device)
    images = pipeline(
        batch_size=batch_size,
        num_inference_steps=pipeline.scheduler.num_train_timesteps,
        output_type="numpy",
    ).images
    return images


def inpaint(model, image, mask, device=None):
    pipeline = DDPMConditionPipeline.from_pretrained(model)
    scheduler = RePaintScheduler.from_config(pipeline.scheduler.config)
    pipeline = RePaintPipeline.from_pretrained(model, scheduler=scheduler)
    if device is not None:
        pipeline = pipeline.to(device)
    images = pipeline(
        image=torch.from_numpy(image),
        mask_image=torch.from_numpy(mask),
        num_inference_steps=pipeline.scheduler.num_train_timesteps,
        eta=0.0,
        jump_length=10,
        jump_n_sample=10,
        output_type="numpy"
    ).images
    return images






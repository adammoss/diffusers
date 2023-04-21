import torch

from diffusers import RePaintPipeline

from pipelines import DDPMConditionPipeline
from schedulers import RePaintScheduler


def generate_samples(model, batch_size, device=None, num_inference_steps=None):
    pipeline = DDPMConditionPipeline.from_pretrained(model)
    if device is not None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pipeline = pipeline.to(device)
    if num_inference_steps is None:
        num_inference_steps = pipeline.scheduler.num_train_timesteps
    images = pipeline(
        batch_size=batch_size,
        num_inference_steps=num_inference_steps,
        output_type="numpy",
    ).images
    return images


def inpaint(model, images, mask, device=None, num_inference_steps=None):
    pipeline = DDPMConditionPipeline.from_pretrained(model)
    scheduler = RePaintScheduler.from_config(pipeline.scheduler.config)
    pipeline = RePaintPipeline.from_pretrained(model, scheduler=scheduler)
    if device is not None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pipeline = pipeline.to(device)
    if num_inference_steps is None:
        num_inference_steps = pipeline.scheduler.num_train_timesteps
    images = pipeline(
        image=torch.from_numpy(images),
        mask_image=torch.from_numpy(mask),
        num_inference_steps=num_inference_steps,
        eta=0.0,
        jump_length=10,
        jump_n_sample=10,
        output_type="numpy"
    ).images
    return images






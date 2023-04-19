from pipeline import DDPMConditionPipeline


def run_model(model, batch_size, device=None):
    pipeline = DDPMConditionPipeline.from_pretrained(model)
    if device is not None:
        pipeline = pipeline.to(device)
    images = pipeline(
        batch_size=batch_size,
        num_inference_steps=pipeline.scheduler.num_train_timesteps,
        output_type="numpy",
    ).images
    return images

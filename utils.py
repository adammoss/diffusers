import os
import argparse

import torch
import numpy as np

from diffusers import DiffusionPipeline, RePaintPipeline

from pipelines import DDPMConditionPipeline, LatentDDPMConditionPipeline
from schedulers import RePaintScheduler


def generate_samples(model, batch_size=1, device=None, num_inference_steps=None,
                     encoder_hidden_states=None, average_out_channels=False, generator=None):
    config = DiffusionPipeline.load_config(model)
    if ('vae' in config) or ('vqvae' in config):
        pipeline = LatentDDPMConditionPipeline.from_pretrained(model)
    else:
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
        encoder_hidden_states=encoder_hidden_states,
        average_out_channels=average_out_channels,
        generator=generator,
    ).images
    return images


def inpaint(model, images, mask, device=None, num_inference_steps=None, generator=None):
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
        output_type="numpy",
        generator=generator,
    ).images
    return images


def img2img(model, images, device=None, num_inference_steps=None, batch_size=1, generator=None):
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
        conditional_image=torch.from_numpy(images),
        generator=generator,
    ).images
    return images


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action",
        type=str,
        default="samples",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.action == "samples":
        images = generate_samples("adammoss/%s" % args.model, num_inference_steps=args.num_inference_steps,
                                  batch_size=args.num_samples, device=args.device)
        np.save(os.path.join(args.output_dir, args.model + "-samples.npy"), images)
    elif args.action == "conditional_samples":
        encoder_hidden_states = []
        for i in range(args.num_samples):
            encoder_hidden_states.append([
                0.1 * np.random.rand(),
                0.1 * np.random.rand(),
                np.random.rand(),
                np.random.rand(),
                np.random.rand(),
                np.random.rand()
            ])
        images = generate_samples("adammoss/%s" % args.model,
                                  batch_size=len(encoder_hidden_states),
                                  device=args.device,
                                  encoder_hidden_states=encoder_hidden_states,
                                  num_inference_steps=args.num_inference_steps)
        np.save(os.path.join(args.output_dir, args.model + "-slice1.npy"), images)
        encoder_hidden_states = []
        for i in range(args.num_samples):
            encoder_hidden_states.append([
                0.9 + 0.1 * np.random.rand(),
                0.1 * np.random.rand(),
                np.random.rand(),
                np.random.rand(),
                np.random.rand(),
                np.random.rand()
            ])
        images = generate_samples("adammoss/%s" % args.model,
                                  batch_size=len(encoder_hidden_states),
                                  device=args.device,
                                  encoder_hidden_states=encoder_hidden_states,
                                  num_inference_steps=args.num_inference_steps)
        np.save(os.path.join(args.output_dir, args.model + "-slice2.npy"), images)
        encoder_hidden_states = []
        for i in range(args.num_samples):
            encoder_hidden_states.append([
                0.1 * np.random.rand(),
                0.9 + 0.1 * np.random.rand(),
                np.random.rand(),
                np.random.rand(),
                np.random.rand(),
                np.random.rand()
            ])
        images = generate_samples("adammoss/%s" % args.model,
                                  batch_size=len(encoder_hidden_states),
                                  device=args.device,
                                  encoder_hidden_states=encoder_hidden_states,
                                  num_inference_steps=args.num_inference_steps)
        np.save(os.path.join(args.output_dir, args.model + "-slice3.npy"), images)
        encoder_hidden_states = []
        for i in range(args.num_samples):
            encoder_hidden_states.append([
                0.9 + 0.1 * np.random.rand(),
                0.9 + 0.1 * np.random.rand(),
                np.random.rand(),
                np.random.rand(),
                np.random.rand(),
                np.random.rand()
            ])
        images = generate_samples("adammoss/%s" % args.model,
                                  batch_size=len(encoder_hidden_states),
                                  device=args.device,
                                  encoder_hidden_states=encoder_hidden_states,
                                  num_inference_steps=args.num_inference_steps)
        np.save(os.path.join(args.output_dir, args.model + "-slice4.npy"), images)
        generator = []
        encoder_hidden_states = []
        for i in range(args.num_samples):
            generator += [
                torch.Generator(device='cuda').manual_seed(i),
                torch.Generator(device='cuda').manual_seed(i),
            ]
            encoder_hidden_states += [
                [0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1],
            ]
        images = generate_samples("adammoss/%s" % args.model,
                                  batch_size=len(encoder_hidden_states),
                                  device=args.device, encoder_hidden_states=encoder_hidden_states,
                                  generator=generator, average_out_channels=True)
        np.save(os.path.join(args.output_dir, args.model + "-fix-generator.npy"), images)
    elif args.action == "class_conditional_samples":
        for class_label in [0, 1]:
            encoder_hidden_states = []
            for i in range(args.num_samples):
                encoder_hidden_states.append([
                    0.1 * np.random.rand(),
                    0.1 * np.random.rand(),
                    np.random.rand(),
                    np.random.rand(),
                    np.random.rand(),
                    np.random.rand(),
                    class_label
                ])
            images = generate_samples("adammoss/%s" % args.model,
                                      batch_size=len(encoder_hidden_states),
                                      device=args.device,
                                      encoder_hidden_states=encoder_hidden_states,
                                      num_inference_steps=args.num_inference_steps)
            np.save(os.path.join(args.output_dir, args.model + "-class%s-slice1.npy" % class_label), images)
            encoder_hidden_states = []
            for i in range(args.num_samples):
                encoder_hidden_states.append([
                    0.9 + 0.1 * np.random.rand(),
                    0.1 * np.random.rand(),
                    np.random.rand(),
                    np.random.rand(),
                    np.random.rand(),
                    np.random.rand(),
                    class_label
                ])
            images = generate_samples("adammoss/%s" % args.model,
                                      batch_size=len(encoder_hidden_states),
                                      device=args.device,
                                      encoder_hidden_states=encoder_hidden_states,
                                      num_inference_steps=args.num_inference_steps)
            np.save(os.path.join(args.output_dir, args.model + "-class%s-slice2.npy" % class_label), images)
            encoder_hidden_states = []
            for i in range(args.num_samples):
                encoder_hidden_states.append([
                    0.1 * np.random.rand(),
                    0.9 + 0.1 * np.random.rand(),
                    np.random.rand(),
                    np.random.rand(),
                    np.random.rand(),
                    np.random.rand(),
                    class_label
                ])
            images = generate_samples("adammoss/%s" % args.model,
                                      batch_size=len(encoder_hidden_states),
                                      device=args.device,
                                      encoder_hidden_states=encoder_hidden_states,
                                      num_inference_steps=args.num_inference_steps)
            np.save(os.path.join(args.output_dir, args.model + "-class%s-slice3.npy" % class_label), images)
            encoder_hidden_states = []
            for i in range(args.num_samples):
                encoder_hidden_states.append([
                    0.9 + 0.1 * np.random.rand(),
                    0.9 + 0.1 * np.random.rand(),
                    np.random.rand(),
                    np.random.rand(),
                    np.random.rand(),
                    np.random.rand(),
                    class_label
                ])
            images = generate_samples("adammoss/%s" % args.model,
                                      batch_size=len(encoder_hidden_states),
                                      device=args.device,
                                      encoder_hidden_states=encoder_hidden_states,
                                      num_inference_steps=args.num_inference_steps)
            np.save(os.path.join(args.output_dir, args.model + "-class%s-slice4.npy" % class_label), images)
        generator = []
        encoder_hidden_states = []
        for i in range(args.num_samples):
            generator += [
                torch.Generator(device='cuda').manual_seed(i),
                torch.Generator(device='cuda').manual_seed(i),
                torch.Generator(device='cuda').manual_seed(i),
                torch.Generator(device='cuda').manual_seed(i),
            ]
            encoder_hidden_states += [
                [0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1],
            ]
        images = generate_samples("adammoss/%s" % args.model,
                                  batch_size=len(encoder_hidden_states),
                                  device=args.device, encoder_hidden_states=encoder_hidden_states,
                                  generator=generator, average_out_channels=True)
        np.save(os.path.join(args.output_dir, args.model + "-fix-generator.npy"), images)


import argparse
import inspect
import logging
import math
import time
import os
from pathlib import Path
from typing import Optional, Union
import json

import numpy as np
from sklearn.model_selection import train_test_split

import accelerate
import datasets
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available

from data import CustomDataset, get_cmd_dataset, get_dsprites_dataset
from losses import LPIPSWithDiscriminator


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.15.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        action='append',
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HF Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="The config of the VAE model to train, leave as None to use standard configuration.",
    )
    parser.add_argument(
        "--disc_start",
        type=int,
        default=50001,
    )
    parser.add_argument(
        "--kl_weight",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--disc_weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/vae",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default='data',
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--data_size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=16, help="Batch size (per device) for the test dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_images_epochs", type=int, default=10, help="How often to save images during training.")
    parser.add_argument(
        "--save_model_epochs", type=int, default=10, help="How often to save the model during training."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.5, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.9, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_private_repo", action="store_true", help="Whether or not to create a private repository."
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
        help=(
            "Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai)"
            " for experiment tracking and logging of model metrics and model checkpoints"
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="lpips",
        choices=["lpips"],
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("You must specify either a dataset name from the hub or a train data directory.")

    return args


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main(args):
    if args.output_dir == "output/vae":
        # Default output
        args.output_dir += '-%s' % args.resolution
        args.output_dir += '-' + '-'.join(args.dataset_name).replace("_", "-")
        args.output_dir += '-%s' % int(time.time())

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")

    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Which VAEmodel to use
    VAEModel = AutoencoderKL

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "vae"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = VAEModel.from_pretrained(input_dir, subfolder="vae")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")

        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        with open(os.path.join(args.output_dir, "params.json"), "w+") as file:
            json.dump(vars(args), file, indent=4)

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.

    if 'simba' in args.dataset_name[0].lower() or 'illustris' in args.dataset_name[0].lower():

        data = []
        for dataset_name in args.dataset_name:
            data.append(get_cmd_dataset(dataset_name, cache_dir=args.cache_dir, resolution=args.resolution,
                                        data_size=args.data_size, transform=np.log, accelerator=accelerator))
        X = np.concatenate([d[0] for d in data], axis=0)
        Y = np.concatenate([d[1] for d in data], axis=0)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=args.test_size, random_state=42)
        train_dataset = CustomDataset(X_train, Y_train, augment=True)
        test_dataset = CustomDataset(X_test, Y_test, augment=False)

    elif args.dataset_name[0] == 'dsprites':

        X, Y = get_dsprites_dataset(cache_dir=args.cache_dir, data_size=args.data_size, accelerator=accelerator)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=args.test_size, random_state=42)
        train_dataset = CustomDataset(X_train, Y_train, augment=False)
        test_dataset = CustomDataset(X_test, Y_test, augment=False)

    else:

        if args.dataset_name[0] is not None:
            train_dataset = load_dataset(
                args.dataset_name[0],
                args.dataset_config_name,
                cache_dir=args.cache_dir,
                split="train",
            )
            test_dataset = load_dataset(
                args.dataset_name[0],
                args.dataset_config_name,
                cache_dir=args.cache_dir,
                split="test",
            )

        else:
            train_dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir,
                                         split="train")
            test_dataset = load_dataset("imagefolder", data_dir=args.test_data_dir, cache_dir=args.cache_dir,
                                         split="test")
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

        # Preprocessing the datasets and DataLoaders creation.
        augmentations = transforms.Compose(
            [
                transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(
                    args.resolution),
                transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        def transform_images(examples):
            images = [augmentations(image.convert("RGB")) for image in examples["image"]]
            return {"input": images}

        train_dataset.set_transform(transform_images)

    d = train_dataset[0]

    in_channels = d["input"].size()[0]
    out_channels = d["input"].size()[0]

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )
    test_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.dataloader_num_workers
    )

    # Initialize the model
    if args.model_config_name_or_path is None:
        model = VAEModel(
            sample_size=args.resolution,
            in_channels=in_channels,
            out_channels=out_channels,
            latent_channels=4,
            scaling_factor=0.18215,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            down_block_types=(
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ),
            up_block_types=(
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
            ),
        )
    else:
        config = VAEModel.load_config(args.model_config_name_or_path)
        model = VAEModel.from_config(config)

    # Loss
    if args.loss == 'lpips':
        loss_fn = LPIPSWithDiscriminator(args.disc_start, kl_weight=args.kl_weight,
                                         disc_weight=args.disc_weight, disc_in_channels=in_channels)
    else:
        raise ValueError(f"Unsupported loss type: {args.loss}")

    accelerator.print('Number of parameters: %s' % sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Initialize the optimizers
    opt_ae = torch.optim.Adam(list(model.encoder.parameters()) +
                              list(model.decoder.parameters()) +
                              list(model.quant_conv.parameters()) +
                              list(model.post_quant_conv.parameters()),
                              lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2))
    opt_disc = torch.optim.Adam(loss_fn.discriminator.parameters(),
                                lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2))

    # Prepare everything with our `accelerator`.
    model, train_dataloader, test_dataloader, opt_ae, opt_disc, loss_fn = \
        accelerator.prepare(model, train_dataloader, test_dataloader, opt_ae, opt_disc, loss_fn)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num test examples = {len(test_dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Train!
    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            inputs = batch["input"]

            with accelerator.accumulate(model):

                posterior = model.encode(inputs).latent_dist
                z = posterior.sample()
                reconstructions = model.decode(z).sample

                last_layer = model.decoder.conv_out.weight

                aeloss, log_dict_ae = loss_fn(inputs, reconstructions, posterior, 0, global_step,
                                              last_layer=last_layer, split="train")

                discloss, log_dict_disc = loss_fn(inputs, reconstructions, posterior, 1, global_step,
                                                  last_layer=last_layer, split="train")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(aeloss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                avg_loss = accelerator.gather(discloss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(aeloss)
                accelerator.backward(discloss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                opt_ae.step()
                opt_disc.step()
                opt_ae.zero_grad()
                opt_disc.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"ae_loss": aeloss.detach().item(), "disc_loss": discloss.detach().item(), "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        progress_bar.close()

        accelerator.wait_for_everyone()

        model.eval()
        test_loss = 0
        for step, batch in enumerate(test_dataloader):
            inputs = batch["input"]

            posterior = model.encode(inputs).latent_dist
            z = posterior.sample()
            reconstructions = model.decode(z).sample

            aeloss, log_dict_ae = loss_fn(inputs, reconstructions, posterior, 0, global_step,
                                          last_layer=last_layer, split="test")

            discloss, log_dict_disc = loss_fn(inputs, reconstructions, posterior, 1, global_step,
                                              last_layer=last_layer, split="test")

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(aeloss.repeat(args.eval_batch_size)).mean()
            test_loss += avg_loss.item() / args.gradient_accumulation_steps
            avg_loss = accelerator.gather(discloss.repeat(args.eval_batch_size)).mean()
            test_loss += avg_loss.item() / args.gradient_accumulation_steps

            if accelerator.is_main_process and step == 0 and \
                    (epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1):

                images_processed = (reconstructions / 2 + 0.5).clamp(0, 1)
                images_processed = images_processed.detach().cpu().numpy()
                images_processed = (images_processed * 255).round().astype("uint8")

                if args.logger == "tensorboard":
                    if is_accelerate_version(">=", "0.17.0.dev0"):
                        tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                    else:
                        tracker = accelerator.get_tracker("tensorboard")
                    tracker.add_images("test_samples", images_processed, epoch)
                elif args.logger == "wandb":
                    # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
                    accelerator.get_tracker("wandb").log(
                        {"test_samples": [wandb.Image(img) for img in images_processed], "epoch": epoch},
                        step=global_step,
                    )

        accelerator.log({"test_loss": test_loss}, step=global_step)
        test_loss = 0.0

        # Generate sample images for visual inspection
        if accelerator.is_main_process and (epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1):
            # save the model
            vae = accelerator.unwrap_model(model)
            vae.save_pretrained(os.path.join(args.output_dir, 'vae'))

            if args.push_to_hub:
                repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=False)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)

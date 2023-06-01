import argparse
import inspect
import logging
import math
import time
import os
from pathlib import Path
from typing import Optional
import json

import numpy as np

import accelerate
import datasets
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import AutoencoderKL, VQModel
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from pipelines import DDPMConditionPipeline, LatentDDPMConditionPipeline
from data import CustomDataset, get_cmd_dataset, get_dsprites_dataset, get_low_resolution
from plotting import calc_1dps_img2d

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.15.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps.cpu()].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


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
        "--dataset_conditional_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model_config_name_or_path",
        type=str,
        default=None,
        help="The config of the UNet model to train, leave as None to use standard DDPM configuration.",
    )
    parser.add_argument(
        "--base_channels",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--cross_attention_dim",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--vae",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--vae_from_pretrained",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--vae_scaling_factor",
        type=float,
        default=None,
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
        "--output_dir",
        type=str,
        default="output/ddpm",
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
        "--super_resolution",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--conditional",
        default=False,
        action="store_true",
        help="whether to use conditional U-net",
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
        "--eval_batch_size", type=int, default=16, help="The number of images to generate for evaluation."
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
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")
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
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "sample", "v_prediction"],
        help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=["mse", "huber"],
    )
    parser.add_argument("--ddpm_num_steps", type=int, default=1000)
    parser.add_argument("--ddpm_num_inference_steps", type=int, default=1000)
    parser.add_argument("--ddpm_beta_schedule", type=str, default="squaredcos_cap_v2")
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
        default='latest',
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
    if args.output_dir == "output/ddpm":
        # Default output
        if args.use_ema:
            args.output_dir += "-ema"
        args.output_dir += '-%s' % args.resolution
        args.output_dir += '-' + '-'.join(args.dataset_name).replace("_", "-")
        if args.conditional:
            args.output_dir += "-cond"
        if args.super_resolution is not None:
            args.output_dir += '-SR%s' % args.super_resolution
        if args.dataset_conditional_name is not None:
            args.output_dir += '-' + args.dataset_conditional_name.replace("_", "-")
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

    # Which U-net model to use
    UNetModel = UNet2DConditionModel if args.conditional else UNet2DModel

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNetModel)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNetModel.from_pretrained(input_dir, subfolder="unet")
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

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.

    if 'simba' in args.dataset_name[0].lower() or 'illustris' in args.dataset_name[0].lower():

        if len(args.dataset_name) == 1:

            X, Y = get_cmd_dataset(args.dataset_name[0], cache_dir=args.cache_dir, resolution=args.resolution,
                                   data_size=args.data_size, transform=np.log, accelerator=accelerator)
            if args.dataset_conditional_name is not None:
                X_conditional, Y_conditional = get_cmd_dataset(args.dataset_conditional_name, cache_dir=args.cache_dir,
                                                               resolution=args.resolution, data_size=args.data_size,
                                                               transform=np.log, accelerator=accelerator)
            elif args.super_resolution is not None:
                X_conditional = get_low_resolution(X, args.super_resolution)
            else:
                X_conditional = None
            dataset = CustomDataset(X, Y, augment=True, data_conditional=X_conditional)

        else:

            if args.dataset_conditional_name is not None:
                raise ValueError("Num datasets > 1 not compatible with conditional data")

            data = []
            data_conditional = []
            for i, dataset_name in enumerate(args.dataset_name):
                X, Y = get_cmd_dataset(dataset_name, cache_dir=args.cache_dir, resolution=args.resolution,
                                       data_size=args.data_size, transform=np.log, accelerator=accelerator)
                if len(args.dataset_name) == 2:
                    Y_class = np.ones((Y.shape[0], 1, 1)) * i
                    Y = np.concatenate((Y, Y_class), axis=2).astype(np.float32)
                if args.super_resolution is not None:
                    data_conditional.append(get_low_resolution(X, args.super_resolution))
                data.append([X, Y])

            X = np.concatenate([d[0] for d in data], axis=0)
            Y = np.concatenate([d[1] for d in data], axis=0)

            if len(data_conditional) > 0:
                dataset = CustomDataset(X, Y, augment=True,
                                        data_conditional=np.concatenate(data_conditional, axis=0))
            else:
                dataset = CustomDataset(X, Y, augment=True)

    elif args.dataset_name == 'dsprites':

        X, Y = get_dsprites_dataset(cache_dir=args.cache_dir, data_size=args.data_size, accelerator=accelerator)
        if args.super_resolution is not None:
            X_conditional = get_low_resolution(X, args.super_resolution)
        else:
            X_conditional = None
        dataset = CustomDataset(X, Y, augment=False, data_conditional=X_conditional)

    else:

        if args.dataset_name is not None:
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
                split="train",
            )

        else:
            dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

        # Preprocessing the datasets and DataLoaders creation.
        augmentations = transforms.Compose(
            [
                transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
                transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        def transform_images(examples):
            images = [augmentations(image.convert("RGB")) for image in examples["image"]]
            return {"input": images}

        dataset.set_transform(transform_images)

    d = dataset[0]

    data_in_channels = d["input"].size()[0]
    data_out_channels = d["input"].size()[0]
    if args.conditional:
        encoder_hid_dim = d["parameters"].size()[1]
    else:
        encoder_hid_dim = None

    if "conditional_input" in d:
        conditional_channels = d["conditional_input"].size()[0]
        conditional_test = d["conditional_input"]
    else:
        conditional_channels = 0

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
    )

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

    # Initialize the VAE if given
    if args.vae_from_pretrained is not None:
        if args.vae == 'kl':
            if args.vae_scaling_factor is not None:
                try:
                    vae = AutoencoderKL.from_pretrained(args.vae_from_pretrained, subfolder="vae",
                                                        scaling_factor=args.vae_scaling_factor)
                except:
                    vae = AutoencoderKL.from_pretrained(args.vae_from_pretrained,
                                                        scaling_factor=args.vae_scaling_factor)
            else:
                try:
                    vae = AutoencoderKL.from_pretrained(args.vae_from_pretrained,
                                                        subfolder="vae")
                except:
                    vae = AutoencoderKL.from_pretrained(args.vae_from_pretrained)
        elif args.vae == 'vq':
            if args.vae_scaling_factor is not None:
                try:
                    vae = VQModel.from_pretrained(args.vae_from_pretrained, subfolder="vqvae",
                                                  scaling_factor=args.vae_scaling_factor)
                except:
                    vae = VQModel.from_pretrained(args.vae_from_pretrained,
                                                  scaling_factor=args.vae_scaling_factor)
            else:
                try:
                    vae = VQModel.from_pretrained(args.vae_from_pretrained, subfolder="vqvae")
                except:
                    vae = VQModel.from_pretrained(args.vae_from_pretrained)
        # Freeze the VAE
        vae.requires_grad_(False)
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        vae.to(accelerator.device, dtype=weight_dtype)
        if vae.__class__ == VQModel:
            latent_shape = vae.encode(
                torch.rand((1, vae.config.in_channels, args.resolution, args.resolution)).to(accelerator.device,
                                                                                             dtype=weight_dtype)).latents.size()
        else:
            latent_shape = vae.encode(
                torch.rand((1, vae.config.in_channels, args.resolution, args.resolution)).to(accelerator.device,
                                                                                             dtype=weight_dtype)).latent_dist.sample().size()
        sample_size = latent_shape[2]
        in_channels = latent_shape[1]
        out_channels = latent_shape[1]
        accelerator.print('VAE scaling factor: %s' % vae.config.scaling_factor)
    else:
        vae = None
        sample_size = args.resolution
        in_channels = data_in_channels
        out_channels = data_out_channels

    # Initialize the model
    if args.model_config_name_or_path is None:
        if args.conditional:
            if args.cross_attention_dim is not None:
                cross_attention_dim = args.cross_attention_dim
            else:
                cross_attention_dim = 4 * args.base_channels
            if sample_size <= 64:
                # LDM-8 like config from https://arxiv.org/pdf/2112.10752.pdf
                block_out_channels = (
                    args.base_channels,
                    2 * args.base_channels,
                    4 * args.base_channels,
                    4 * args.base_channels,
                )
                down_block_types = (
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                )
                up_block_types = (
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                )
            elif sample_size == 128:
                if args.super_resolution is not None:
                    block_out_channels = (
                        args.base_channels,
                        2 * args.base_channels,
                        4 * args.base_channels,
                        4 * args.base_channels,
                    )
                    down_block_types = (
                        "DownBlock2D",
                        "DownBlock2D",
                        "CrossAttnDownBlock2D",
                        "CrossAttnDownBlock2D",
                    )
                    up_block_types = (
                        "UpBlock2D",
                        "UpBlock2D",
                        "CrossAttnUpBlock2D",
                        "CrossAttnUpBlock2D",
                    )
                else:
                    # LDM-2 like config from https://arxiv.org/pdf/2112.10752.pdf
                    block_out_channels = (
                        args.base_channels,
                        2 * args.base_channels,
                        2 * args.base_channels,
                        4 * args.base_channels,
                        4 * args.base_channels,
                    )
                    down_block_types = (
                        "DownBlock2D",
                        "CrossAttnDownBlock2D",
                        "CrossAttnDownBlock2D",
                        "CrossAttnDownBlock2D",
                        "DownBlock2D",
                    )
                    up_block_types = (
                        "UpBlock2D",
                        "CrossAttnUpBlock2D",
                        "CrossAttnUpBlock2D",
                        "CrossAttnUpBlock2D",
                        "UpBlock2D",
                    )
            else:
                # LDM-1 like config from https://arxiv.org/pdf/2112.10752.pdf
                block_out_channels = (
                    args.base_channels,
                    args.base_channels,
                    2 * args.base_channels,
                    2 * args.base_channels,
                    4 * args.base_channels,
                    4 * args.base_channels,
                )
                down_block_types = (
                    "DownBlock2D",
                    "DownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "DownBlock2D",
                )
                up_block_types = (
                    "UpBlock2D",
                    "UpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                    "UpBlock2D",
                )
            model = UNetModel(
                sample_size=sample_size,
                in_channels=in_channels + conditional_channels,
                out_channels=out_channels,
                encoder_hid_dim=encoder_hid_dim,
                block_out_channels=block_out_channels,
                cross_attention_dim=cross_attention_dim,
                down_block_types=down_block_types,
                up_block_types=up_block_types,
            )
        else:
            # Base model from https://github.com/huggingface/diffusers/blob/main/examples/unconditional_image_generation/train_unconditional.py
            model = UNetModel(
                sample_size=sample_size,
                in_channels=in_channels + conditional_channels,
                out_channels=out_channels,
                layers_per_block=2,
                block_out_channels=(
                    args.base_channels,
                    args.base_channels,
                    2 * args.base_channels,
                    2 * args.base_channels,
                    4 * args.base_channels,
                    4 * args.base_channels
                ),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
            )
    else:
        config = UNetModel.load_config(args.model_config_name_or_path)
        model = UNetModel.from_config(config)

    accelerator.print('Number of parameters: %s' % sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Create EMA for the model.
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNetModel,
            model_config=model.config,
        )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Initialize the scheduler
    accepts_prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())
    if accepts_prediction_type:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.ddpm_num_steps,
            beta_schedule=args.ddpm_beta_schedule,
            prediction_type=args.prediction_type,
        )
    else:
        noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
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

            average_out_channels = False
            if vae is not None:
                clean_images = batch["input"]
                if vae.config.in_channels > 1 and data_in_channels == 1:
                    clean_images = torch.cat([clean_images] * vae.config.in_channels, dim=1)
                    average_out_channels = True
                if vae.__class__ == VQModel:
                    clean_images = vae.encode(clean_images.to(weight_dtype)).latents
                else:
                    clean_images = vae.encode(clean_images.to(weight_dtype)).latent_dist.sample()
                clean_images = clean_images * vae.config.scaling_factor
                if "conditional_input" in batch:
                    conditional_images = batch["conditional_input"]
                    if vae.config.in_channels > 1 and data_in_channels == 1:
                        conditional_images = torch.cat([conditional_images] * vae.config.in_channels, dim=1)
                    if vae.__class__ == VQModel:
                        conditional_images = vae.encode(conditional_images.to(weight_dtype)).latents
                    else:
                        conditional_images = vae.encode(conditional_images.to(weight_dtype)).latent_dist.sample()
                    conditional_images = conditional_images * vae.config.scaling_factor
            else:
                clean_images = batch["input"]
                if "conditional_input" in batch:
                    conditional_images = batch["conditional_input"]

            # Sample noise that we'll add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bsz = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            if "conditional_input" in batch:
                noisy_images = torch.cat((noisy_images, conditional_images), dim=1)

            with accelerator.accumulate(model):
                # Predict the noise residual
                if args.conditional:
                    model_output = model(noisy_images, timesteps, batch["parameters"]).sample
                else:
                    model_output = model(noisy_images, timesteps).sample

                if args.prediction_type == "epsilon":
                    if args.loss == 'mse':
                        loss = F.mse_loss(model_output, noise)  # this could have different weights!
                    elif args.loss == 'huber':
                        loss = F.huber_loss(model_output, noise)
                elif args.prediction_type == "sample":
                    alpha_t = _extract_into_tensor(
                        noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                    )
                    snr_weights = alpha_t / (1 - alpha_t)
                    loss = snr_weights * F.mse_loss(
                        model_output, clean_images, reduction="none"
                    )  # use SNR weighting from distillation paper
                    loss = loss.mean()
                elif args.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(clean_images, noise, timesteps)
                    loss = F.mse_loss(model_output, target)
                else:
                    raise ValueError(f"Unsupported prediction type: {args.prediction_type}")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1:
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                if vae is not None:
                    pipeline = LatentDDPMConditionPipeline(
                        unet=unet,
                        scheduler=noise_scheduler,
                        vae=vae,
                    )
                else:
                    pipeline = DDPMConditionPipeline(
                        unet=unet,
                        scheduler=noise_scheduler,
                    )

                # run pipeline in inference (sample random noise and denoise)
                if args.conditional:
                    generator = []
                    encoder_hidden_states = []
                    if len(args.dataset_name) == 1:
                        for i in range(args.eval_batch_size):
                            encoder_hidden_states.append(
                                [i / (args.eval_batch_size - 1)] * dataset[0]["parameters"].size()[1])
                            generator.append(torch.Generator(device=pipeline.device).manual_seed(i))
                    else:
                        for i in range(args.eval_batch_size)[:args.eval_batch_size//2]:
                            encoder_hidden_states.append(
                                [i / (args.eval_batch_size / 2 - 1)] * (dataset[0]["parameters"].size()[1] - 1) + [0])
                            generator.append(torch.Generator(device=pipeline.device).manual_seed(i))
                        for i in range(args.eval_batch_size)[args.eval_batch_size//2:]:
                            encoder_hidden_states.append(
                                [i / (args.eval_batch_size / 2 - 1)] * (dataset[0]["parameters"].size()[1] - 1) + [1])
                            generator.append(torch.Generator(device=pipeline.device).manual_seed(i))
                    if "conditional_input" in batch:
                        images = pipeline(
                            batch_size=len(encoder_hidden_states),
                            generator=generator,
                            num_inference_steps=args.ddpm_num_inference_steps,
                            output_type="numpy",
                            conditional_image=conditional_test,
                            encoder_hidden_states=encoder_hidden_states,
                            average_out_channels=average_out_channels,
                        ).images
                    else:
                        images = pipeline(
                            batch_size=len(encoder_hidden_states),
                            generator=generator,
                            num_inference_steps=args.ddpm_num_inference_steps,
                            output_type="numpy",
                            encoder_hidden_states=encoder_hidden_states,
                            average_out_channels=average_out_channels,
                        ).images
                else:
                    generator = torch.Generator(device=pipeline.device).manual_seed(0)
                    if "conditional_input" in batch:
                        images = pipeline(
                            generator=generator,
                            batch_size=args.eval_batch_size,
                            num_inference_steps=args.ddpm_num_inference_steps,
                            output_type="numpy",
                            conditional_image=conditional_test,
                            average_out_channels=average_out_channels,
                        ).images
                    else:
                        images = pipeline(
                            generator=generator,
                            batch_size=args.eval_batch_size,
                            num_inference_steps=args.ddpm_num_inference_steps,
                            output_type="numpy",
                            average_out_channels=average_out_channels,
                        ).images

                if args.use_ema:
                    ema_model.restore(unet.parameters())

                accelerator.log({'sample_mean': images.mean(), 'sample_sd': images.std()}, step=global_step)

                # denormalize the images and save to tensorboard
                images_processed = (images * 255).round().astype("uint8")

                if args.logger == "tensorboard":
                    if is_accelerate_version(">=", "0.17.0.dev0"):
                        tracker = accelerator.get_tracker("tensorboard", unwrap=True)
                    else:
                        tracker = accelerator.get_tracker("tensorboard")
                    tracker.add_images("test_samples", images_processed.transpose(0, 3, 1, 2), epoch)
                elif args.logger == "wandb":
                    # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
                    tracker = accelerator.get_tracker("wandb")
                    tracker.log(
                        {"test_samples": [wandb.Image(img) for img in images_processed], "epoch": epoch},
                        step=global_step,
                    )
                    if images.shape[-1] == 1:
                        # 1 channel, channel dim now last
                        kvals = np.arange(0, images.shape[1] / 2)
                        ps = []
                        for image in images:
                            ps.append(list(calc_1dps_img2d(image, smoothed=0.25)[1] * kvals**2))
                        tracker.log({"power_spectrum": wandb.plot.line_series(
                            xs=list(kvals),
                            ys=ps,
                            title="Power spectrum",
                            xname="k")}, step=global_step)

            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                # save the model
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                if vae is not None:
                    pipeline = LatentDDPMConditionPipeline(
                        unet=unet,
                        scheduler=noise_scheduler,
                        vae=vae,
                    )
                else:
                    pipeline = DDPMConditionPipeline(
                        unet=unet,
                        scheduler=noise_scheduler,
                    )

                pipeline.save_pretrained(args.output_dir)

                if args.use_ema:
                    ema_model.restore(unet.parameters())

                if args.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=False)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)

import argparse

import torch
from omegaconf import OmegaConf

from diffusers import AutoencoderKL
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    assign_to_checkpoint,
    conv_attn_to_linear,
    renew_vae_attention_paths,
    renew_vae_resnet_paths,
)

from huggingface_hub import HfFolder, Repository, create_repo, whoami


def custom_convert_ldm_vae_checkpoint(checkpoint, config):
    vae_state_dict = checkpoint

    new_checkpoint = {}

    new_checkpoint["encoder.conv_in.weight"] = vae_state_dict["encoder.conv_in.weight"]
    new_checkpoint["encoder.conv_in.bias"] = vae_state_dict["encoder.conv_in.bias"]
    new_checkpoint["encoder.conv_out.weight"] = vae_state_dict["encoder.conv_out.weight"]
    new_checkpoint["encoder.conv_out.bias"] = vae_state_dict["encoder.conv_out.bias"]
    new_checkpoint["encoder.conv_norm_out.weight"] = vae_state_dict["encoder.norm_out.weight"]
    new_checkpoint["encoder.conv_norm_out.bias"] = vae_state_dict["encoder.norm_out.bias"]

    new_checkpoint["decoder.conv_in.weight"] = vae_state_dict["decoder.conv_in.weight"]
    new_checkpoint["decoder.conv_in.bias"] = vae_state_dict["decoder.conv_in.bias"]
    new_checkpoint["decoder.conv_out.weight"] = vae_state_dict["decoder.conv_out.weight"]
    new_checkpoint["decoder.conv_out.bias"] = vae_state_dict["decoder.conv_out.bias"]
    new_checkpoint["decoder.conv_norm_out.weight"] = vae_state_dict["decoder.norm_out.weight"]
    new_checkpoint["decoder.conv_norm_out.bias"] = vae_state_dict["decoder.norm_out.bias"]

    new_checkpoint["quant_conv.weight"] = vae_state_dict["quant_conv.weight"]
    new_checkpoint["quant_conv.bias"] = vae_state_dict["quant_conv.bias"]
    new_checkpoint["post_quant_conv.weight"] = vae_state_dict["post_quant_conv.weight"]
    new_checkpoint["post_quant_conv.bias"] = vae_state_dict["post_quant_conv.bias"]

    # Retrieves the keys for the encoder down blocks only
    num_down_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "encoder.down" in layer})
    down_blocks = {
        layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key] for layer_id in range(num_down_blocks)
    }

    # Retrieves the keys for the decoder up blocks only
    num_up_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "decoder.up" in layer})
    up_blocks = {
        layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key] for layer_id in range(num_up_blocks)
    }

    for i in range(num_down_blocks):
        resnets = [key for key in down_blocks[i] if f"down.{i}" in key and f"down.{i}.downsample" not in key]

        if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.weight"
            )
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.bias"
            )

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    conv_attn_to_linear(new_checkpoint)

    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        resnets = [
            key for key in up_blocks[block_id] if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
        ]

        if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.weight"
            ]
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.bias"
            ]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    conv_attn_to_linear(new_checkpoint)
    return new_checkpoint


def create_vae_diffusers_config(original_config):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    vae_params = original_config.model.params.ddconfig

    block_out_channels = [vae_params.ch * mult for mult in vae_params.ch_mult]
    down_block_types = ["DownEncoderBlock2D"] * len(block_out_channels)
    up_block_types = ["UpDecoderBlock2D"] * len(block_out_channels)

    config = {
        "sample_size": vae_params.resolution,
        "in_channels": vae_params.in_channels,
        "out_channels": vae_params.out_ch,
        "down_block_types": tuple(down_block_types),
        "up_block_types": tuple(up_block_types),
        "block_out_channels": tuple(block_out_channels),
        "latent_channels": vae_params.z_channels,
        "layers_per_block": vae_params.num_res_blocks,
    }
    return config


def vae_to_vae_diffuser(
        config_path: str,
        checkpoint_path: str,
        output_path: str,
        hub_token: str,
        push_to_hub: bool,
):
    original_config = OmegaConf.load(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Loading checkpoint')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print('Done')

    # Convert the VAE model.
    vae_config = create_vae_diffusers_config(original_config)
    print(vae_config)
    print('Converting checkpoint')
    converted_vae_checkpoint = custom_convert_ldm_vae_checkpoint(checkpoint["state_dict"], vae_config)
    print('Done')

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(converted_vae_checkpoint)

    if push_to_hub:
        create_repo("adammoss/%s" % output_path, exist_ok=True, token=hub_token)
        repo = Repository(output_path, clone_from="adammoss/%s" % output_path, token=hub_token)

    vae.save_pretrained(output_path)

    if push_to_hub:
        repo.push_to_hub(commit_message="VAE conversion", blocking=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", default=None, type=str, required=True,
                        help="Path to the VAE config to convert.")
    parser.add_argument("--checkpoint_path", default=None, type=str, required=True,
                        help="Path to the VAE.pt to convert.")
    parser.add_argument("--output_path", default=None, type=str, required=True, help="Output path.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")

    args = parser.parse_args()

    vae_to_vae_diffuser(args.config_path,
                        args.checkpoint_path,
                        args.output_path,
                        args.hub_token,
                        args.push_to_hub)

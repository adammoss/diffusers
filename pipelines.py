from typing import List, Optional, Tuple, Union

import torch
import numpy as np

from diffusers.utils import randn_tensor
from diffusers.pipeline_utils import DiffusionPipeline, ImagePipelineOutput


class DDPMConditionPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        encoder_hidden_states: List[float] = None,
        image: Optional[torch.FloatTensor] = None,
        conditional_image: Optional[torch.FloatTensor] = None,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            encoder_hidden_states:
        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """

        if image is None:

            # Sample gaussian noise to begin loop
            if isinstance(self.unet.sample_size, int):
                image_shape = (batch_size, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size)
            else:
                image_shape = (batch_size, self.unet.in_channels, *self.unet.sample_size)

            if self.device.type == "mps":
                # randn does not work reproducibly on mps
                image = randn_tensor(image_shape, generator=generator)
                image = image.to(self.device)
            else:
                image = randn_tensor(image_shape, generator=generator, device=self.device)

        if conditional_image is not None:
            if len(conditional_image.size()) == 3:
                conditional_image = conditional_image.unsqueeze(0)
                conditional_image = conditional_image.repeat(batch_size, 1, 1, 1)
            print(conditional_image.size())
            print(image.size())
            conditional_image = conditional_image.to(self.device)
            image = torch.cat((image, conditional_image), axis=1)
            print(image.size())

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        # set encoder hidden states
        if encoder_hidden_states is not None:
            encoder_hidden_states = np.array(encoder_hidden_states, dtype=np.float32)
            if len(encoder_hidden_states.shape) == 1:
                encoder_hidden_states = np.expand_dims(encoder_hidden_states, 0)
                encoder_hidden_states = np.repeat(encoder_hidden_states, batch_size, axis=0)
                encoder_hidden_states = np.expand_dims(encoder_hidden_states, 1)
            elif len(encoder_hidden_states.shape) == 2 and encoder_hidden_states.shape[0] == batch_size:
                encoder_hidden_states = np.expand_dims(encoder_hidden_states, 1)
            else:
                raise ValueError('Incorrect shape for encoder hidden states')
            encoder_hidden_states = torch.from_numpy(encoder_hidden_states).to(self.device)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            if encoder_hidden_states is not None:
                model_output = self.unet(image, t, encoder_hidden_states).sample
            else:
                model_output = self.unet(image, t).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
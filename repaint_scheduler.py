from typing import Optional, Tuple, Union

import torch

from diffusers import RePaintScheduler as RePaintSchedulerBase
from diffusers.schedulers.scheduling_repaint import RePaintSchedulerOutput
from diffusers.utils import randn_tensor


class RepaintScheduler(RePaintSchedulerBase):

    def step(
            self,
            model_output: torch.FloatTensor,
            timestep: int,
            sample: torch.FloatTensor,
            original_image: torch.FloatTensor,
            mask: torch.FloatTensor,
            generator: Optional[torch.Generator] = None,
            return_dict: bool = True,
    ) -> Union[RePaintSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).
        Args:
            model_output (`torch.FloatTensor`): direct output from learned
                diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            original_image (`torch.FloatTensor`):
                the original image to inpaint on.
            mask (`torch.FloatTensor`):
                the mask where 0.0 values define which part of the original image to inpaint (change).
            generator (`torch.Generator`, *optional*): random number generator.
            return_dict (`bool`): option for returning tuple rather than
                DDPMSchedulerOutput class
        Returns:
            [`~schedulers.scheduling_utils.RePaintSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.RePaintSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        t = timestep
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
            pred_epsilon = (alpha_prod_t ** 0.5) * model_output + (beta_prod_t ** 0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 3. Clip "predicted x_0"
        if self.config.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # We choose to follow RePaint Algorithm 1 to get x_{t-1}, however we
        # substitute formula (7) in the algorithm coming from DDPM paper
        # (formula (4) Algorithm 2 - Sampling) with formula (12) from DDIM paper.
        # DDIM schedule gives the same results as DDPM with eta = 1.0
        # Noise is being reused in 7. and 8., but no impact on quality has
        # been observed.

        # 5. Add noise
        device = model_output.device
        noise = randn_tensor(model_output.shape, generator=generator, device=device, dtype=model_output.dtype)
        std_dev_t = self.eta * self._get_variance(timestep) ** 0.5

        variance = 0
        if t > 0 and self.eta > 0:
            variance = std_dev_t * noise

        # 6. compute "direction pointing to x_t" of formula (12)
        # from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** 0.5 * pred_epsilon

        # 7. compute x_{t-1} of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_unknown_part = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction + variance

        # 8. Algorithm 1 Line 5 https://arxiv.org/pdf/2201.09865.pdf
        prev_known_part = (alpha_prod_t_prev ** 0.5) * original_image + ((1 - alpha_prod_t_prev) ** 0.5) * noise

        # 9. Algorithm 1 Line 8 https://arxiv.org/pdf/2201.09865.pdf
        pred_prev_sample = mask * prev_known_part + (1.0 - mask) * prev_unknown_part

        if not return_dict:
            return (
                pred_prev_sample,
                pred_original_sample,
            )

        return RePaintSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)

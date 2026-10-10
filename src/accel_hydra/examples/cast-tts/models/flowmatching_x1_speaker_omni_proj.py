"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations

from random import random
from typing import Callable
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint
from diffusers.utils.torch_utils import randn_tensor
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling
from tqdm import tqdm
from accel_hydra.utils.torch import create_mask_from_length as lens_to_mask
from accel_hydra.models.common import CountParamsBase, LoadPretrainedBase

from utils.audio import MelSpec
from utils.general import (
    default,
    exists,
)
from utils.torch import list_str_to_idx, list_str_to_tensor, mask_from_frac_lengths
from utils.flow_matching import get_epss_timesteps
from utils.tokenize import get_tokenizer
import warnings
class FlowMatchingMixin:
    def __init__(
        self,
        cfg_drop_ratio: float = 0.2,
        sample_strategy: str = 'normal',
        num_train_steps: int = 1000
    ) -> None:
        r"""
        Args:
            cfg_drop_ratio (float): Dropout ratio for the autoencoder.
            sample_strategy (str): Sampling strategy for timesteps during training.
            num_train_steps (int): Number of training steps for the noise scheduler.
        """
        self.sample_strategy = sample_strategy
        # self.infer_noise_scheduler = FlowMatchEulerDiscreteScheduler(
        #     num_train_timesteps=num_train_steps
        # )
        # self.train_noise_scheduler = copy.deepcopy(self.infer_noise_scheduler)

        self.classifier_free_guidance = cfg_drop_ratio > 0.0
        self.cfg_drop_ratio = cfg_drop_ratio

    def get_input_target_and_timesteps(
        self,
        latent: torch.Tensor,
        training: bool,
    ):
        batch_size = latent.shape[0]
        noise = torch.randn_like(latent)

        if training:
            if self.sample_strategy == 'normal':
                t = compute_density_for_timestep_sampling(
                    weighting_scheme="logit_normal",
                    batch_size=batch_size,
                    logit_mean=0,
                    logit_std=1,
                    mode_scale=None,
                )
            elif self.sample_strategy == 'uniform':
                t = torch.rand(batch_size)
            else:
                raise NotImplementedError(
                    f"{self.sample_strategy} samlping for timesteps is not supported now"
                )
        else:
            t = torch.tensor([0.5] * batch_size)

        t = t.to(latent.device)
        t_expanded = t
        while t_expanded.ndim < latent.ndim:
            t_expanded = t_expanded.unsqueeze(-1)
        noisy_latent = (1.0 - t_expanded) * noise + t_expanded * latent

        target = latent - noise

        return noisy_latent, target, t

    # def retrieve_timesteps(
    #     self,
    #     num_inference_steps: int | None = None,
    #     device: str | torch.device | None = None,
    #     timesteps: list[int] | None = None,
    #     sigmas: list[float] | None = None,
    #     **kwargs,
    # ):
    #     # used in inference, retrieve new timesteps on given inference timesteps
    #     scheduler = self.infer_noise_scheduler

    #     if timesteps is not None and sigmas is not None:
    #         raise ValueError(
    #             "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
    #         )
    #     if timesteps is not None:
    #         accepts_timesteps = "timesteps" in set(
    #             inspect.signature(scheduler.set_timesteps).parameters.keys()
    #         )
    #         if not accepts_timesteps:
    #             raise ValueError(
    #                 f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
    #                 f" timestep schedules. Please check whether you are using the correct scheduler."
    #             )
    #         scheduler.set_timesteps(
    #             timesteps=timesteps, device=device, **kwargs
    #         )
    #         timesteps = scheduler.timesteps
    #         num_inference_steps = len(timesteps)
    #     elif sigmas is not None:
    #         accept_sigmas = "sigmas" in set(
    #             inspect.signature(scheduler.set_timesteps).parameters.keys()
    #         )
    #         if not accept_sigmas:
    #             raise ValueError(
    #                 f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
    #                 f" sigmas schedules. Please check whether you are using the correct scheduler."
    #             )
    #         scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
    #         timesteps = scheduler.timesteps
    #         num_inference_steps = len(timesteps)
    #     else:
    #         scheduler.set_timesteps(
    #             num_inference_steps, device=device, **kwargs
    #         )
    #         timesteps = scheduler.timesteps
    #     return timesteps, num_inference_steps


class MultiConditionFlowMatchingMixin(FlowMatchingMixin):
    def __init__(
        self,
        cfg_drop_config: dict | float,
        sample_strategy: str = "normal",
        num_train_steps: int = 1000
    ) -> None:
        super().__init__(
            cfg_drop_ratio=0.0,
            sample_strategy=sample_strategy,
            num_train_steps=num_train_steps
        )
        if isinstance(cfg_drop_config, float):
            self.cfg_drop_config = {"all": cfg_drop_config}
        elif isinstance(cfg_drop_config, dict):
            self.cfg_drop_config = cfg_drop_config
        else:
            raise ValueError(f"Invalid cfg_drop_config: {cfg_drop_config}")
        self.classifier_free_guidance = any(
            v > 0.0 for v in self.cfg_drop_config.values()
        )

    def drop_content(
        self, content_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # FIXME `content_dict` is modified in-place, should be a deep copy, but deep copy may increase memory usage
        content_dict_copy = content_dict.copy()
        for content_key, content in content_dict_copy.items():
            if content_key not in self.cfg_drop_config:
                continue
            if random() < self.cfg_drop_config[content_key]:
                # 赋值给拷贝后的字典
                content_dict_copy[content_key] = torch.zeros_like(content)
                # zeros = torch.zeros_like(content)
                # zeros.requires_grad_(content.requires_grad)
                # content_dict_copy[content_key] = zeros
        return content_dict_copy


class CASTFlowMatching(
    MultiConditionFlowMatchingMixin, LoadPretrainedBase, CountParamsBase
):
    def __init__(
        self,
        autoencoder: nn.Module,
        text_encoder: nn.Module,
        content_encoder: nn.Module,
        backbone: nn.Module,
        tokenizer_path: str,
        tokenizer: str = "char",
        cfg_drop_config: dict | float | None = None,
        sample_strategy: str = "normal",
        num_train_steps: int = 1000,
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
    ) -> None:
        nn.Module.__init__(self)
        if cfg_drop_config is None:
            cfg_drop_config = {"masked_mel": 0.3, "text_embed": 0.2}
        super().__init__(
            cfg_drop_config=cfg_drop_config,
            sample_strategy=sample_strategy,
            num_train_steps=num_train_steps
        )
        self.autoencoder = autoencoder
        self.text_encoder = text_encoder
        self.content_encoder = content_encoder
        self.backbone = backbone
        self.vocab_char_map = get_tokenizer(tokenizer_path, tokenizer)[0]
        self.frac_lengths_mask = frac_lengths_mask
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.backbone.parameters():
            param.requires_grad = False
        #self.text_encoder.eval()
        #self.backbone.eval()
        


    def encode_text(
        self, text: torch.Tensor, target_length: int
    ) -> torch.Tensor:
        text_embed = self.text_encoder(text, target_length)
        return text_embed

    def get_backbone_input(
        self, text_embed: float["b n d"], caption_embed: float["b n d"], 
    ):
        return  text_embed, caption_embed

    def forward(
        self,
        prompt_waveform: torch.Tensor,  # ground truth
        target_waveform: torch.Tensor,  # ground truth
        text: list[str],
        caption: list[str],
        prompt_waveform_lengths: torch.Tensor | None = None,
        target_waveform_lengths: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        with torch.no_grad():
            target_latent, target_latent_lengths, target_mask = self.autoencoder.encode(target_waveform, target_waveform_lengths)
        batch, seq_len = target_latent.shape[:2]
        dtype, device = target_latent.dtype, target_latent.device

        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # lens and mask
        if not exists(target_latent_lengths):
            # if lens not acquired by trainer from collate_fn
            target_latent_lengths = torch.full((batch, ), seq_len, device=device)
        mask = lens_to_mask(target_latent_lengths, max_length=seq_len)

        if not torch.equal(mask.bool(), target_mask.bool()):
            batch_lens = mask.bool().sum(dim=1)
            mel_batch_lens = target_mask.bool().sum(dim=1)
            warnings.warn(
                f"mel_mask != mask: lengths error\n"
                f"mask sum per sample: {batch_lens.tolist()}\n"
                f"mel_mask sum per sample: {mel_batch_lens.tolist()}\n"
                f"difference: {(batch_lens - mel_batch_lens).tolist()}"
            )



        noisy_latent, target, timesteps = self.get_input_target_and_timesteps(
            target_latent, training=self.training
        )


        text_embedding = self.encode_text(text, target_latent.shape[1])

        caption_embed, caption_mask = self.content_encoder.encode_content(prompt_waveform, prompt_waveform_lengths, caption, infer_type=-1, device=device)

        # for name, param in self.content_encoder.feat_proj.named_parameters():
        #     print(name, param.requires_grad, flush=True)
        # print(caption_embed.grad_fn, flush=True)
        # print(caption_embed.requires_grad, flush=True)

        content_dict = {"text_embed": text_embedding, "caption_embed": caption_embed}

        if self.training and self.classifier_free_guidance:
            content_dict = self.drop_content(content_dict)
        
        time_aligned_content, non_time_aligned_content = self.get_backbone_input(**content_dict)
        non_time_aligned_content_mask = caption_mask
        #with torch.enable_grad():
        pred: torch.Tensor = self.backbone(
            x=noisy_latent,
            time=timesteps,
            # context=content,
            x_mask=mask,
            time_aligned_content=time_aligned_content,
            non_time_aligned_content=non_time_aligned_content,
            # non_time_aligned_content=non_time_aligned_content.requires_grad_(True),
            non_time_aligned_content_mask=non_time_aligned_content_mask,
            # context_mask=content_mask
        )
        # import pdb
        # pdb.set_trace()
        # print("pred", pred.requires_grad, flush=True)
        # print("target", target.requires_grad, flush=True)

        # loss = F.mse_loss(pred, target, reduction="none")
        # print("loss", loss.requires_grad, flush=True)
        # print("mask", mask.requires_grad, flush=True)
        # loss = loss[mask]
        # #print(loss, loss.requires_grad, loss.grad_fn)
        # return loss.mean()
        # print(f"[DEBUG] pred: requires_grad={pred.requires_grad}, grad_fn={pred.grad_fn}", flush=True)
        # print(f"[DEBUG] target: requires_grad={target.requires_grad}, grad_fn={target.grad_fn}", flush=True)

        loss = F.mse_loss(pred, target, reduction="none")
        # print(f"[DEBUG] loss (unreduced): requires_grad={loss.requires_grad}, grad_fn={loss.grad_fn}", flush=True)

        loss = loss[mask]
        # print(f"[DEBUG] loss (masked): requires_grad={loss.requires_grad}, grad_fn={loss.grad_fn}", flush=True)

        final_loss = loss.mean()
        # print(f"[DEBUG] final_loss: requires_grad={final_loss.requires_grad}, grad_fn={final_loss.grad_fn}", flush=True)

        # assert final_loss.grad_fn is not None, "Loss grad_fn is None!"

        return final_loss

    def iterative_denoise(
        self,
        latent: torch.Tensor,
        timesteps: list[int],
        # noise_scheduler: FlowMatchEulerDiscreteScheduler,
        num_steps: int,
        verbose: bool,
        cfg: bool,
        cfg_scale: float,
        backbone_input: dict,
    ):

        def fn(t, x):

            if cfg:
                x = torch.cat([x, x])

            noise_pred = self.backbone(
                x=x,
                time=t,
                **backbone_input,
            )
            if cfg:
                pred_uncond, pred_cond = torch.chunk(noise_pred, 2, dim=0)
                noise_pred = pred_uncond + cfg_scale * (
                    pred_cond - pred_uncond
                )

            return noise_pred

        odeint_kwargs = {"method": "euler"}
        trajectory = odeint(fn, latent, timesteps, **odeint_kwargs)
        result = trajectory[-1]

        return result

    @torch.inference_mode()
    def inference(
        self,
        prompt_waveform: torch.Tensor,
        text: "int['b nt'] | list[str]",
        caption: "list[str]",
        duration: "int | int['b']",
        infer_type: "int | int['b']",
        latent_shape: Sequence[int],
        # noise_scheduler: FlowMatchEulerDiscreteScheduler,
        prompt_waveform_lengths: torch.Tensor = None,
        num_steps: int = 20,
        use_epss: bool = True,
        sway_sampling_coef: float | None = -1.0,
        guidance_scale: float = 3.0,
        disable_progress: bool = True,
        **kwargs
    ) -> torch.Tensor:
        with torch.no_grad():
            prompt_latent, _, _ = self.autoencoder.encode(prompt_waveform, prompt_waveform_lengths)
        device = prompt_latent.device
        batch, prompt_seq_len = prompt_latent.shape[0], prompt_latent.shape[1]
        classifier_free_guidance = guidance_scale > 1.0

        # text

        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # # duration
        # if not exists(prompt_latent_lengths):
        #     prompt_latent_lengths = torch.full((batch, ),
        #                                   prompt_seq_len,
        #                                   device=device)
        # prompt_mask = lens_to_mask(prompt_latent_lengths)
        # if edit_mask is not None:
        #     mel_spec_mask = mel_spec_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch, ),
                                  duration,
                                  device=device,
                                  dtype=torch.long)

        duration = torch.maximum(
            (text != -1).sum(dim=-1) + 1,
            duration
        )  # duration at least text/audio prompt length plus one token, so something is generated
        max_duration = duration.amax()


        latent_mask = lens_to_mask(duration)

        # prepare content input to the backbone
        text_embedding = self.encode_text(text, max_duration)
        
        caption_embed, caption_mask = self.content_encoder.encode_content(prompt_waveform, prompt_waveform_lengths, caption, infer_type=infer_type, device=device)

        content_dict = {"text_embed": text_embedding, "caption_embed": caption_embed}

        # --------------------------------------------------------------------
        # prepare unconditional input
        # --------------------------------------------------------------------
        if classifier_free_guidance:
            for key in self.cfg_drop_config:
                uncond_content = torch.zeros_like(content_dict[key])
                content_dict[key] = torch.cat([
                    uncond_content, content_dict[key]
                ])

            latent_mask = torch.cat([
                latent_mask, latent_mask.detach().clone()
            ])


        time_aligned_content, non_time_aligned_content = self.get_backbone_input(**content_dict)

        non_time_aligned_content_mask = caption_mask

        # prepare noise input to the backbone
        latent_shape = tuple(
            max_duration if dim is None else dim for dim in latent_shape
        )
        shape = (batch, *latent_shape)
        latent_generated = randn_tensor(
            shape, generator=None, device=device, dtype=prompt_latent.dtype
        )

        if use_epss:  # use Empirically Pruned Step Sampling for low NFE
            t = get_epss_timesteps(
                num_steps, device=device, dtype=prompt_latent.dtype
            )
        else:
            t = torch.linspace(
                0, 1, num_steps + 1, device=self.device, dtype=prompt_latent.dtype
            )
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        # noise_scheduler.set_timesteps(sigmas=t.cpu(), device=device)
        latent_generated = self.iterative_denoise(
            latent=latent_generated,
            timesteps=t,
            # noise_scheduler=noise_scheduler,
            num_steps=num_steps,
            verbose=not disable_progress,
            cfg=classifier_free_guidance,
            cfg_scale=guidance_scale,
            backbone_input={
                "x_mask": latent_mask,
                "time_aligned_content": time_aligned_content,
                "non_time_aligned_content": non_time_aligned_content,
                "non_time_aligned_content_mask": non_time_aligned_content_mask,
            }
        )

        #print(latent_generated.shape)
        #print(latent_mask.shape)
        target_latent = latent_generated[0][latent_mask[0]]  # shape: [T', 100]
        target_latent = target_latent.unsqueeze(0).permute(0, 2, 1)
        #print(target_latent.shape)
        waveform = self.autoencoder.decode(target_latent)
        return waveform

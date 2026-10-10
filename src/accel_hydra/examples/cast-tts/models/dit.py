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

import math
from typing import Optional

import torch
import torch.nn.functional as F
from accel_hydra.utils.general import is_package_available
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from x_transformers.x_transformers import RotaryEmbedding, apply_rotary_pos_emb


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0
):
    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    # https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
    theta *= theta_rescale_factor**(dim / (dim - 2))
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


# AdaLayerNorm
# return with modulated x for attn input, and params for later mlp modulation


class AdaLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(
            emb, 6, dim=1
        )

        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


# RMSNorm


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.native_rms_norm = float(torch.__version__[:3]) >= 2.4

    def forward(self, x):
        if self.native_rms_norm:
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                x = x.to(self.weight.dtype)
            x = F.rms_norm(
                x,
                normalized_shape=(x.shape[-1], ),
                weight=self.weight,
                eps=self.eps
            )
        else:
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                x = x.to(self.weight.dtype)
            x = x * self.weight

        return x


# Attention processor

if is_package_available("flash_attn"):
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input


class AttnProcessor:
    def __init__(
        self,
        pe_attn_head: int |
        None = None,  # number of attention head to apply rope, None for all
        attn_backend: str = "torch",  # "torch" or "flash_attn"
        attn_mask_enabled: bool = True,
    ):
        if attn_backend == "flash_attn":
            assert is_package_available(
                "flash_attn"
            ), "Please install flash-attn first."

        self.pe_attn_head = pe_attn_head
        self.attn_backend = attn_backend
        self.attn_mask_enabled = attn_mask_enabled

    def __call__(
        self,
        attn: Attention,
        x: float["b n d"],  # noised input x
        mask: bool["b n"] | None = None,
        rope=None,  # rotary position embedding
    ) -> torch.FloatTensor:
        batch_size = x.shape[0]

        # `sample` projections
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads,
                           head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads,
                           head_dim).transpose(1, 2)

        # qk norm
        if attn.q_norm is not None:
            query = attn.q_norm(query)
        if attn.k_norm is not None:
            key = attn.k_norm(key)

        # apply rotary position embedding
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (
                xpos_scale, xpos_scale**-1.0
            ) if xpos_scale is not None else (1.0, 1.0)

            if self.pe_attn_head is not None:
                pn = self.pe_attn_head
                query[:, :pn, :, :] = apply_rotary_pos_emb(
                    query[:, :pn, :, :], freqs, q_xpos_scale
                )
                key[:, :pn, :, :] = apply_rotary_pos_emb(
                    key[:, :pn, :, :], freqs, k_xpos_scale
                )
            else:
                query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
                key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        if self.attn_backend == "torch":
            # mask. e.g. inference got a batch with different target durations, mask out the padding
            if self.attn_mask_enabled and mask is not None:
                attn_mask = mask
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(
                    1
                )  # 'b n -> b 1 1 n'
                attn_mask = attn_mask.expand(
                    batch_size, attn.heads, query.shape[-2], key.shape[-2]
                )
            else:
                attn_mask = None
            x = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False
            )
            x = x.transpose(1,
                            2).reshape(batch_size, -1, attn.heads * head_dim)

        elif self.attn_backend == "flash_attn":
            query = query.transpose(1, 2)  # [b, h, n, d] -> [b, n, h, d]
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            if self.attn_mask_enabled and mask is not None:
                query, indices, q_cu_seqlens, q_max_seqlen_in_batch, _ = unpad_input(
                    query, mask
                )
                key, _, k_cu_seqlens, k_max_seqlen_in_batch, _ = unpad_input(
                    key, mask
                )
                value, _, _, _, _ = unpad_input(value, mask)
                x = flash_attn_varlen_func(
                    query,
                    key,
                    value,
                    q_cu_seqlens,
                    k_cu_seqlens,
                    q_max_seqlen_in_batch,
                    k_max_seqlen_in_batch,
                )
                x = pad_input(x, indices, batch_size, q_max_seqlen_in_batch)
                x = x.reshape(batch_size, -1, attn.heads * head_dim)
            else:
                x = flash_attn_func(
                    query, key, value, dropout_p=0.0, causal=False
                )
                x = x.reshape(batch_size, -1, attn.heads * head_dim)

        x = x.to(query.dtype)

        # linear proj
        x = attn.to_out[0](x)
        # dropout
        x = attn.to_out[1](x)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        return x


# Joint Attention processor for MM-DiT
# modified from diffusers/src/diffusers/models/attention_processor.py


class JointAttnProcessor:
    def __init__(self):
        pass

    def __call__(
        self,
        attn: Attention,
        x: float["b n d"],  # noised input x
        c: float["b nt d"] = None,  # context c, here text
        mask: bool["b n"] | None = None,
        rope=None,  # rotary position embedding for x
        c_rope=None,  # rotary position embedding for c
    ) -> torch.FloatTensor:
        residual = x

        batch_size = c.shape[0]

        # `sample` projections
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        # `context` projections
        c_query = attn.to_q_c(c)
        c_key = attn.to_k_c(c)
        c_value = attn.to_v_c(c)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads,
                           head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads,
                           head_dim).transpose(1, 2)
        c_query = c_query.view(batch_size, -1, attn.heads,
                               head_dim).transpose(1, 2)
        c_key = c_key.view(batch_size, -1, attn.heads,
                           head_dim).transpose(1, 2)
        c_value = c_value.view(batch_size, -1, attn.heads,
                               head_dim).transpose(1, 2)

        # qk norm
        if attn.q_norm is not None:
            query = attn.q_norm(query)
        if attn.k_norm is not None:
            key = attn.k_norm(key)
        if attn.c_q_norm is not None:
            c_query = attn.c_q_norm(c_query)
        if attn.c_k_norm is not None:
            c_key = attn.c_k_norm(c_key)

        # apply rope for context and noised input independently
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (
                xpos_scale, xpos_scale**-1.0
            ) if xpos_scale is not None else (1.0, 1.0)
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)
        if c_rope is not None:
            freqs, xpos_scale = c_rope
            q_xpos_scale, k_xpos_scale = (
                xpos_scale, xpos_scale**-1.0
            ) if xpos_scale is not None else (1.0, 1.0)
            c_query = apply_rotary_pos_emb(c_query, freqs, q_xpos_scale)
            c_key = apply_rotary_pos_emb(c_key, freqs, k_xpos_scale)

        # joint attention
        query = torch.cat([query, c_query], dim=2)
        key = torch.cat([key, c_key], dim=2)
        value = torch.cat([value, c_value], dim=2)

        # mask. e.g. inference got a batch with different target durations, mask out the padding
        if mask is not None:
            attn_mask = F.pad(
                mask, (0, c.shape[1]), value=True
            )  # no mask for c (text)
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # 'b n -> b 1 1 n'
            attn_mask = attn_mask.expand(
                batch_size, attn.heads, query.shape[-2], key.shape[-2]
            )
        else:
            attn_mask = None

        x = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False
        )
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)

        # Split the attention outputs.
        x, c = (
            x[:, :residual.shape[1]],
            x[:, residual.shape[1]:],
        )

        # linear proj
        x = attn.to_out[0](x)
        # dropout
        x = attn.to_out[1](x)
        if not attn.context_pre_only:
            c = attn.to_out_c(c)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)
            # c = c.masked_fill(~mask, 0.)  # no mask for c (text)

        return x, c


# Attention with possible joint part
# modified from diffusers/src/diffusers/models/attention_processor.py


class Attention(nn.Module):
    def __init__(
        self,
        processor: JointAttnProcessor | AttnProcessor,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,  # if not None -> joint attention
        context_pre_only: bool = False,
        qk_norm: Optional[str] = None,
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "Attention equires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.processor = processor

        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout

        self.context_dim = context_dim
        self.context_pre_only = context_pre_only

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        if qk_norm is None:
            self.q_norm = None
            self.k_norm = None
        elif qk_norm == "rms_norm":
            self.q_norm = RMSNorm(dim_head, eps=1e-6)
            self.k_norm = RMSNorm(dim_head, eps=1e-6)
        else:
            raise ValueError(f"Unimplemented qk_norm: {qk_norm}")

        if self.context_dim is not None:
            self.to_q_c = nn.Linear(context_dim, self.inner_dim)
            self.to_k_c = nn.Linear(context_dim, self.inner_dim)
            self.to_v_c = nn.Linear(context_dim, self.inner_dim)
            if qk_norm is None:
                self.c_q_norm = None
                self.c_k_norm = None
            elif qk_norm == "rms_norm":
                self.c_q_norm = RMSNorm(dim_head, eps=1e-6)
                self.c_k_norm = RMSNorm(dim_head, eps=1e-6)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, dim))
        self.to_out.append(nn.Dropout(dropout))

        if self.context_dim is not None and not self.context_pre_only:
            self.to_out_c = nn.Linear(self.inner_dim, context_dim)

    def forward(
        self,
        x: float["b n d"],  # noised input x
        c: float["b n d"] = None,  # context c
        mask: bool["b n"] | None = None,
        rope=None,  # rotary position embedding for x
        c_rope=None,  # rotary position embedding for c
    ) -> torch.Tensor:
        if c is not None:
            return self.processor(
                self, x, c=c, mask=mask, rope=rope, c_rope=c_rope
            )
        else:
            return self.processor(self, x, mask=mask, rope=rope)


# FeedForward


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        mult=4,
        dropout=0.0,
        approximate: str = "none"
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.ff(x)


# DiT Block


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head,
        ff_mult=4,
        dropout=0.1,
        qk_norm=None,
        pe_attn_head=None,
        attn_backend="torch",  # "torch" or "flash_attn"
        attn_mask_enabled=True,
    ):
        super().__init__()

        self.attn_norm = AdaLayerNorm(dim)
        self.attn = Attention(
            processor=AttnProcessor(
                pe_attn_head=pe_attn_head,
                attn_backend=attn_backend,
                attn_mask_enabled=attn_mask_enabled,
            ),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            qk_norm=qk_norm,
        )

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(
            dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh"
        )

    def forward(
        self,
        x,
        t,
        mask=None,
        rope=None
    ):  # x: noised input, t: time embedding
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(
            x, emb=t
        )

        # attention
        attn_output = self.attn(x=norm, mask=mask, rope=rope)

        # process attention output for input x
        x = x + gate_msa.unsqueeze(1) * attn_output

        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x


# sinusoidal position embedding


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        # a little different from UniFlow-Audio: `scale` here
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# time step conditioning embedding


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def forward(self, timestep: float["b"]):
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        time = self.time_mlp(time_hidden)  # b d
        return time


# convolutional position embedding


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.conv1d = nn.Sequential(
            nn.Conv1d(
                dim, dim, kernel_size, groups=groups, padding=kernel_size // 2
            ),
            nn.Mish(),
            nn.Conv1d(
                dim, dim, kernel_size, groups=groups, padding=kernel_size // 2
            ),
            nn.Mish(),
        )
        self.layer_need_mask_idx = [
            i for i, layer in enumerate(self.conv1d)
            if isinstance(layer, nn.Conv1d)
        ]

    def forward(self, x: float["b n d"], mask: bool["b n"] | None = None):
        if mask is not None:
            mask = mask.unsqueeze(1)  # [B 1 N]
        x = x.permute(0, 2, 1)  # [B D N]

        if mask is not None:
            x = x.masked_fill(~mask, 0.0)
        for i, block in enumerate(self.conv1d):
            x = block(x)
            if mask is not None and i in self.layer_need_mask_idx:
                x = x.masked_fill(~mask, 0.0)

        x = x.permute(0, 2, 1)  # [B N D]

        return x


# AdaLayerNorm for final layer
# return only with modulated x for attn input, cuz no more mlp modulation


class AdaLayerNorm_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))  # different from UniFlow-Audio
        scale, shift = torch.chunk(emb, 2, dim=1)

        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


# Global Response Normalization layer (Instance Normalization ?)


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


# ConvNeXt-V2 Block https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
# ref: https://github.com/bfs18/e2_tts/blob/main/rfwave/modules.py#L108


class ConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=7,
            padding=padding,
            groups=dim,
            dilation=dilation
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)  # b n d -> b d n
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


# Text embedding


class TextEmbedding(nn.Module):
    def __init__(
        self,
        text_num_embeds,
        text_dim,
        mask_padding=True,
        average_upsampling=False,
        conv_layers=0,
        conv_mult=2
    ):
        super().__init__()
        self.text_embed = nn.Embedding(
            text_num_embeds + 1, text_dim
        )  # use 0 as filler token

        self.mask_padding = mask_padding  # mask filler and batch padding tokens or not
        self.average_upsampling = average_upsampling  # zipvoice-style text late average upsampling (after text encoder)
        if average_upsampling:
            assert mask_padding, "text_embedding_average_upsampling requires text_mask_padding to be True"

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 8192  # 8192 is ~87.38s of 24khz audio; 4096 is ~43.69s of 24khz audio
            self.register_buffer(
                "freqs_cis",
                precompute_freqs_cis(text_dim, self.precompute_max_pos),
                persistent=False
            )
            self.text_blocks = nn.Sequential(
                *[
                    ConvNeXtV2Block(text_dim, text_dim * conv_mult)
                    for _ in range(conv_layers)
                ]
            )
        else:
            self.extra_modeling = False

    def average_upsample_text_by_mask(self, text, text_mask):
        batch, text_len, text_dim = text.shape

        audio_len = text_len  # cuz text already padded to same length as audio sequence
        text_lens = text_mask.sum(dim=1)  # [batch]

        upsampled_text = torch.zeros_like(text)

        for i in range(batch):
            text_len = text_lens[i].item()

            if text_len == 0:
                continue

            valid_ind = torch.where(text_mask[i])[0]
            valid_data = text[i, valid_ind, :]  # [text_len, text_dim]

            base_repeat = audio_len // text_len
            remainder = audio_len % text_len

            indices = []
            for j in range(text_len):
                repeat_count = base_repeat + (
                    1 if j >= text_len - remainder else 0
                )
                indices.extend([j] * repeat_count)

            indices = torch.tensor(
                indices[:audio_len], device=text.device, dtype=torch.long
            )
            upsampled = valid_data[indices]  # [audio_len, text_dim]

            upsampled_text[i, :audio_len, :] = upsampled

        return upsampled_text

    def forward(self, text: int["b nt"], seq_len, drop_text=False):
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len
                   ]  # curtail if character tokens are more than the mel spec tokens
        text = F.pad(
            text, (0, seq_len - text.shape[1]), value=0
        )  # (opt.) if not self.average_upsampling:
        if self.mask_padding:
            text_mask = text == 0

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            text = text + self.freqs_cis[:seq_len, :]

            # convnextv2 blocks
            if self.mask_padding:
                text = text.masked_fill(
                    text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0
                )
                for block in self.text_blocks:
                    text = block(text)
                    text = text.masked_fill(
                        text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)),
                        0.0
                    )
            else:
                text = self.text_blocks(text)

        if self.average_upsampling:
            text = self.average_upsample_text_by_mask(text, ~text_mask)

        return text


# noised input audio and context mixing embedding


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(
        self,
        x: float["b n d"],
        cond: float["b n d"],
        text_embed: float["b n d"],
        drop_audio_cond=False,
        audio_mask: bool["b n"] | None = None,
    ):
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)

        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed(x, mask=audio_mask) + x
        return x


# Transformer backbone using DiT blocks


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        text_mask_padding=True,
        text_embedding_average_upsampling=False,
        qk_norm=None,
        conv_layers=0,
        pe_attn_head=None,
        attn_backend="torch",  # "torch" | "flash_attn"
        attn_mask_enabled=False,
        long_skip_connection=False,
        checkpoint_activations=False,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(
            text_num_embeds,
            text_dim,
            mask_padding=text_mask_padding,
            average_upsampling=text_embedding_average_upsampling,
            conv_layers=conv_layers,
        )
        self.text_cond, self.text_uncond = None, None  # text cache
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList([
            DiTBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout,
                qk_norm=qk_norm,
                pe_attn_head=pe_attn_head,
                attn_backend=attn_backend,
                attn_mask_enabled=attn_mask_enabled,
            ) for _ in range(depth)
        ])
        self.long_skip_connection = nn.Linear(
            dim * 2, dim, bias=False
        ) if long_skip_connection else None

        self.norm_out = AdaLayerNorm_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations

        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out AdaLN layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def ckpt_wrapper(self, module):
        # https://github.com/chuanyangjin/fast-DiT/blob/main/models.py
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def get_input_embed(
        self,
        x,  # b n d
        cond,  # b n d
        text,  # b nt
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cache: bool = True,
        audio_mask: bool["b n"] | None = None,
    ):
        if self.text_uncond is None or self.text_cond is None or not cache:
            if audio_mask is None:
                text_embed = self.text_embed(
                    text, x.shape[1], drop_text=drop_text
                )
            else:
                batch = x.shape[0]
                seq_lens = audio_mask.sum(
                    dim=1
                )  # Calculate the actual sequence length for each sample
                text_embed_list = []
                for i in range(batch):
                    text_embed_i = self.text_embed(
                        text[i].unsqueeze(0),
                        seq_len=seq_lens[i].item(),
                        drop_text=drop_text,
                    )
                    text_embed_list.append(text_embed_i[0])
                text_embed = pad_sequence(
                    text_embed_list, batch_first=True, padding_value=0
                )
            if cache:
                if drop_text:
                    self.text_uncond = text_embed
                else:
                    self.text_cond = text_embed

        if cache:
            if drop_text:
                text_embed = self.text_uncond
            else:
                text_embed = self.text_cond

        x = self.input_embed(
            x,
            cond,
            text_embed,
            drop_audio_cond=drop_audio_cond,
            audio_mask=audio_mask
        )

        return x

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None

    def forward(
        self,
        x: float["b n d"],  # nosied input audio
        cond: float["b n d"],  # masked cond audio
        text: int["b nt"],  # text
        time: float["b"] | float[""],  # time step
        mask: bool["b n"] | None = None,
        drop_audio_cond: bool = False,  # cfg for cond audio
        drop_text: bool = False,  # cfg for text
        cfg_infer: bool = False,  # cfg inference, pack cond & uncond forward
        cache: bool = False,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, text: text, x: noised audio + cond audio + text
        t = self.time_embed(time)
        if cfg_infer:  # pack cond & uncond forward: b n d -> 2b n d
            x_cond = self.get_input_embed(
                x,
                cond,
                text,
                drop_audio_cond=False,
                drop_text=False,
                cache=cache,
                audio_mask=mask
            )
            x_uncond = self.get_input_embed(
                x,
                cond,
                text,
                drop_audio_cond=True,
                drop_text=True,
                cache=cache,
                audio_mask=mask
            )
            x = torch.cat((x_cond, x_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
        else:
            x = self.get_input_embed(
                x,
                cond,
                text,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
                cache=cache,
                audio_mask=mask
            )

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint
                x = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(block),
                    x,
                    t,
                    mask,
                    rope,
                    use_reentrant=False
                )
            else:
                x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output

class DiT_Test(DiT):
    def forward(
        self,
        x: float["b n d"],  # nosied input audio
        cond: float["b n d"],  # masked cond audio
        text: int["b nt"],  # text
        time: float["b"] | float[""],  # time step
        mask: bool["b n"] | None = None,
        drop_audio_cond: bool = False,  # cfg for cond audio
        drop_text: bool = False,  # cfg for text
        cfg_infer: bool = False,  # cfg inference, pack cond & uncond forward
        cache: bool = False,
        return_hidden_states=False,   # 新增
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, text: text, x: noised audio + cond audio + text
        t = self.time_embed(time)
        if cfg_infer:  # pack cond & uncond forward: b n d -> 2b n d
            x_cond = self.get_input_embed(
                x,
                cond,
                text,
                drop_audio_cond=False,
                drop_text=False,
                cache=cache,
                audio_mask=mask
            )
            x_uncond = self.get_input_embed(
                x,
                cond,
                text,
                drop_audio_cond=True,
                drop_text=True,
                cache=cache,
                audio_mask=mask
            )
            x = torch.cat((x_cond, x_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
        else:
            x = self.get_input_embed(
                x,
                cond,
                text,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
                cache=cache,
                audio_mask=mask
            )

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        hidden_states = []            # 用于收集每层输出
        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(block), x, t, mask, rope, use_reentrant=False
                )
            else:
                x = block(x, t, mask=mask, rope=rope)
            if return_hidden_states:
                hidden_states.append(x)   # 保存该层输出（未经过最终 norm/out）

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        if return_hidden_states:
            return output, hidden_states
        else:
            return output
        
class InputFusionDiT(nn.Module):
    def __init__(
        self,
        *,
        in_dim,
        out_dim,
        embed_dim,
        ta_content_dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        qk_norm=None,
        pe_attn_head=None,
        attn_backend="torch",  # "torch" | "flash_attn"
        attn_mask_enabled=False,
        long_skip_connection=False,
        checkpoint_activations=False,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(embed_dim)
        # self.text_cond, self.text_uncond = None, None  # text cache
        # self.input_embed = InputEmbedding(mel_dim, text_dim, dim)
        self.input_proj = nn.Linear(in_dim + ta_content_dim, embed_dim)
        self.input_pos_embed = ConvPositionEmbedding(dim=embed_dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = embed_dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList([
            DiTBlock(
                dim=embed_dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout,
                qk_norm=qk_norm,
                pe_attn_head=pe_attn_head,
                attn_backend=attn_backend,
                attn_mask_enabled=attn_mask_enabled,
            ) for _ in range(depth)
        ])
        self.long_skip_connection = nn.Linear(
            embed_dim * 2, embed_dim, bias=False
        ) if long_skip_connection else None

        self.norm_out = AdaLayerNorm_Final(embed_dim)  # final modulation
        self.proj_out = nn.Linear(embed_dim, out_dim)

        self.checkpoint_activations = checkpoint_activations

        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out AdaLN layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def ckpt_wrapper(self, module):
        # https://github.com/chuanyangjin/fast-DiT/blob/main/models.py
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    # def clear_cache(self):
    #     self.text_cond, self.text_uncond = None, None

    def input_fusion(
        self,
        x: float["b n d"],
        x_mask: bool["b n"] | None,
        time_aligned_content: float["b n d"],
    ):
        x = self.input_proj(torch.cat([x, time_aligned_content], dim=-1))
        x = self.input_pos_embed(x, mask=x_mask) + x
        return x

    def forward(
        self,
        x: float["b n d"],  # nosied input audio
        # masked_mel: float["b n d"],  # masked cond audio
        # text_embed: int["b nt"],  # text
        time: float["b"] | float[""],  # time step
        x_mask: bool["b n"] | None = None,
        time_aligned_content: float["b n d"] | None = None,
        # drop_audio_cond: bool = False,  # cfg for cond audio
        # drop_text: bool = False,  # cfg for text
        # cfg_infer: bool = False,  # cfg inference, pack cond & uncond forward
        # cache: bool = False,
    ):

        batch, seq_len = x.shape[0], x.shape[1]

        x = self.input_fusion(x, x_mask, time_aligned_content)

        if time.ndim == 0:
            time = time.repeat(batch).to(x.device)

        # t: conditioning time, text: text, x: noised audio + cond audio + text
        t = self.time_embed(time)
        # if cfg_infer:  # pack cond & uncond forward: b n d -> 2b n d
        #     x_cond = self.get_input_embed(
        #         x,
        #         cond,
        #         text,
        #         drop_audio_cond=False,
        #         drop_text=False,
        #         cache=cache,
        #         audio_mask=mask
        #     )
        #     x_uncond = self.get_input_embed(
        #         x,
        #         cond,
        #         text,
        #         drop_audio_cond=True,
        #         drop_text=True,
        #         cache=cache,
        #         audio_mask=mask
        #     )
        #     x = torch.cat((x_cond, x_uncond), dim=0)
        #     t = torch.cat((t, t), dim=0)
        #     mask = torch.cat((mask, mask), dim=0) if mask is not None else None
        # else:
        #     x = self.get_input_embed(
        #         x,
        #         cond,
        #         text,
        #         drop_audio_cond=drop_audio_cond,
        #         drop_text=drop_text,
        #         cache=cache,
        #         audio_mask=mask
        #     )

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint
                x = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(block),
                    x,
                    t,
                    x_mask,
                    rope,
                    use_reentrant=False
                )
            else:
                x = block(x, t, mask=x_mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output

class DiT_Controlnet(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        text_mask_padding=True,
        text_embedding_average_upsampling=False,
        qk_norm=None,
        conv_layers=0,
        pe_attn_head=None,
        attn_backend="torch",
        attn_mask_enabled=False,
        long_skip_connection=False,
        checkpoint_activations=False,
        # ---------- ControlNet 新增参数 ----------
        use_controlnet: bool = False,
        controlnet_input_dim: int = 7,
        controlnet_time_steps: int = 3,   # 固定时间步，用于上采样
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(
            text_num_embeds,
            text_dim,
            mask_padding=text_mask_padding,
            average_upsampling=text_embedding_average_upsampling,
            conv_layers=conv_layers,
        )
        self.text_cond, self.text_uncond = None, None  # text cache
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        # 原始 DiT 块
        self.transformer_blocks = nn.ModuleList([
            DiTBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout,
                qk_norm=qk_norm,
                pe_attn_head=pe_attn_head,
                attn_backend=attn_backend,
                attn_mask_enabled=attn_mask_enabled,
            ) for _ in range(depth)
        ])
        self.long_skip_connection = nn.Linear(
            dim * 2, dim, bias=False
        ) if long_skip_connection else None

        self.norm_out = AdaLayerNorm_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations

        # ---------- ControlNet 相关模块 ----------
        self.use_controlnet = use_controlnet
        if use_controlnet:
            # 1. 可训练的 ControlNet 块（结构与原始块相同）
            self.controlnet_blocks = nn.ModuleList([
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                ) for _ in range(depth)
            ])
            # 2. 零卷积层（线性层，初始化为0）
            self.zero_convs = nn.ModuleList([
                nn.Linear(dim, dim, bias=False) for _ in range(depth)
            ])
            for conv in self.zero_convs:
                nn.init.zeros_(conv.weight)
            # 3. 额外特征编码器 (7 -> dim)
            self.extra_feature_linear = nn.Linear(controlnet_input_dim, dim)

        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out AdaLN layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

        # ControlNet 的零卷积已经在 __init__ 中初始化为0，无需额外操作

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    def get_input_embed(
        self,
        x,
        cond,
        text,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cache: bool = True,
        audio_mask: Optional[torch.Tensor] = None,
    ):
        if self.text_uncond is None or self.text_cond is None or not cache:
            if audio_mask is None:
                text_embed = self.text_embed(
                    text, x.shape[1], drop_text=drop_text
                )
            else:
                batch = x.shape[0]
                seq_lens = audio_mask.sum(dim=1)
                text_embed_list = []
                for i in range(batch):
                    text_embed_i = self.text_embed(
                        text[i].unsqueeze(0),
                        seq_len=seq_lens[i].item(),
                        drop_text=drop_text,
                    )
                    text_embed_list.append(text_embed_i[0])
                text_embed = pad_sequence(
                    text_embed_list, batch_first=True, padding_value=0
                )
            if cache:
                if drop_text:
                    self.text_uncond = text_embed
                else:
                    self.text_cond = text_embed

        if cache:
            if drop_text:
                text_embed = self.text_uncond
            else:
                text_embed = self.text_cond

        x = self.input_embed(
            x,
            cond,
            text_embed,
            drop_audio_cond=drop_audio_cond,
            audio_mask=audio_mask
        )
        return x

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None

    # ---------- ControlNet 权重复制与冻结 ----------
    def copy_controlnet_weights(self):
        """将原始 transformer_blocks 的权重复制到 controlnet_blocks"""
        if self.use_controlnet:
            for orig, ctrl in zip(self.transformer_blocks, self.controlnet_blocks):
                ctrl.load_state_dict(orig.state_dict())

    def freeze_original(self):
        """冻结所有非 ControlNet 的参数（原始 DiT 全部参数）"""
        for name, param in self.named_parameters():
            if not (name.startswith('controlnet_') or
                    name.startswith('zero_convs') or
                    name.startswith('extra_feature_linear')):
                param.requires_grad = False

    # ---------- 主 forward ----------
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        text: torch.Tensor,
        time: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cfg_infer: bool = False,
        cache: bool = False,
        extra_features: Optional[torch.Tensor] = None,   # 新增：形状 [b, 3, 7]
        controlnet_scale: float = 1.0,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # 时间嵌入
        t = self.time_embed(time)

        # --- 原始路径的输入嵌入 ---
        if cfg_infer:
            x_cond = self.get_input_embed(
                x,
                cond,
                text,
                drop_audio_cond=False,
                drop_text=False,
                cache=cache,
                audio_mask=mask
            )
            x_uncond = self.get_input_embed(
                x,
                cond,
                text,
                drop_audio_cond=True,
                drop_text=True,
                cache=cache,
                audio_mask=mask
            )
            x = torch.cat((x_cond, x_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
            # extra_features 也需要重复
            if self.use_controlnet and extra_features is not None:
                extra_features = torch.cat([extra_features, extra_features], dim=0)
        else:
            x = self.get_input_embed(
                x,
                cond,
                text,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
                cache=cache,
                audio_mask=mask
            )

        # --- ControlNet 输入准备 ---
        if self.use_controlnet and extra_features is not None:
            # 编码 extra_features: [b, 3, 7] -> [b, 3, dim]
            extra_embed = self.extra_feature_linear(extra_features)   # [b, 3, dim]
            # 上采样到 seq_len（线性插值）
            extra_embed = extra_embed.permute(0, 2, 1)                # [b, dim, 3]
            extra_embed = F.interpolate(extra_embed, size=seq_len, mode='linear', align_corners=False)  # [b, dim, seq_len]
            extra_embed = extra_embed.permute(0, 2, 1)                # [b, seq_len, dim]
            c_input = x + extra_embed   # ControlNet 的初始输入（与原始输入相加）
        else:
            c_input = x if self.use_controlnet else None

        # --- 位置编码和长跳跃 ---
        rope = self.rotary_embed.forward_from_seq_len(seq_len)
        residual = x if self.long_skip_connection is not None else None

        # --- 逐层处理 ---
        for i, block in enumerate(self.transformer_blocks):
            # 原始 block
            y = block(x, t, mask=mask, rope=rope)

            if self.use_controlnet and extra_features is not None and controlnet_scale != 0.0:
                # ControlNet block
                c = self.controlnet_blocks[i](c_input, t, mask=mask, rope=rope)
                delta = self.zero_convs[i](c)   # 零卷积输出
                x = y + controlnet_scale * delta                   # 加到原始输出上
                c_input = c                     # 更新 ControlNet 的下一层输入
            else:
                x = y

        # 长跳跃连接
        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        # 最终输出
        x = self.norm_out(x, t)
        output = self.proj_out(x)
        return output

class DiT_Controlnet_Test(DiT_Controlnet):
    # ---------- 主 forward ----------
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        text: torch.Tensor,
        time: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cfg_infer: bool = False,
        cache: bool = False,
        extra_features: Optional[torch.Tensor] = None,   # 新增：形状 [b, 3, 7]
        controlnet_scale: float = 1.0,
        return_hidden_states=False,   # 新增
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # 时间嵌入
        t = self.time_embed(time)

        # --- 原始路径的输入嵌入 ---
        if cfg_infer:
            x_cond = self.get_input_embed(
                x,
                cond,
                text,
                drop_audio_cond=False,
                drop_text=False,
                cache=cache,
                audio_mask=mask
            )
            x_uncond = self.get_input_embed(
                x,
                cond,
                text,
                drop_audio_cond=True,
                drop_text=True,
                cache=cache,
                audio_mask=mask
            )
            x = torch.cat((x_cond, x_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
            # extra_features 也需要重复
            if self.use_controlnet and extra_features is not None:
                extra_features = torch.cat([extra_features, extra_features], dim=0)
        else:
            x = self.get_input_embed(
                x,
                cond,
                text,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
                cache=cache,
                audio_mask=mask
            )

        # --- ControlNet 输入准备 ---
        if self.use_controlnet and extra_features is not None:
            # 编码 extra_features: [b, 3, 7] -> [b, 3, dim]
            extra_embed = self.extra_feature_linear(extra_features)   # [b, 3, dim]
            # 上采样到 seq_len（线性插值）
            extra_embed = extra_embed.permute(0, 2, 1)                # [b, dim, 3]
            extra_embed = F.interpolate(extra_embed, size=seq_len, mode='linear', align_corners=False)  # [b, dim, seq_len]
            extra_embed = extra_embed.permute(0, 2, 1)                # [b, seq_len, dim]
            c_input = x + extra_embed   # ControlNet 的初始输入（与原始输入相加）
        else:
            c_input = x if self.use_controlnet else None

        # --- 位置编码和长跳跃 ---
        rope = self.rotary_embed.forward_from_seq_len(seq_len)
        residual = x if self.long_skip_connection is not None else None
        hidden_states = []            # 用于收集每层输出
        # --- 逐层处理 ---
        for i, block in enumerate(self.transformer_blocks):
            # 原始 block
            y = block(x, t, mask=mask, rope=rope)

            if self.use_controlnet and extra_features is not None and controlnet_scale != 0.0:
                # ControlNet block
                c = self.controlnet_blocks[i](c_input, t, mask=mask, rope=rope)
                delta = self.zero_convs[i](c)   # 零卷积输出
                x = y + controlnet_scale * delta                   # 加到原始输出上
                c_input = c                     # 更新 ControlNet 的下一层输入
            else:
                x = y
            if return_hidden_states:
                hidden_states.append(delta)   # 保存该层输出（未经过最终 norm/out）

        # 长跳跃连接
        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        # 最终输出
        x = self.norm_out(x, t)
        output = self.proj_out(x)
        if return_hidden_states:
            return output, hidden_states
        else:
            return output
    
class DiT_Controlnet_tune(nn.Module):
    """
    带 ControlNet 分支的 DiT，支持额外的 3×7 维特征输入（时间步3，特征维7）。
    推理时支持 control_scale (λ) 和 control_time_range (t_control)。
    """
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        text_mask_padding=True,
        text_embedding_average_upsampling=False,
        qk_norm=None,
        conv_layers=0,
        pe_attn_head=None,
        attn_backend="torch",
        attn_mask_enabled=False,
        long_skip_connection=False,
        checkpoint_activations=False,
        # ---------- ControlNet 新增参数 ----------
        use_controlnet: bool = False,
        controlnet_input_dim: int = 7,
        controlnet_time_steps: int = 3,   # 固定时间步，用于上采样
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(
            text_num_embeds,
            text_dim,
            mask_padding=text_mask_padding,
            average_upsampling=text_embedding_average_upsampling,
            conv_layers=conv_layers,
        )
        self.text_cond, self.text_uncond = None, None  # text cache
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        # 原始 DiT 块
        self.transformer_blocks = nn.ModuleList([
            DiTBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout,
                qk_norm=qk_norm,
                pe_attn_head=pe_attn_head,
                attn_backend=attn_backend,
                attn_mask_enabled=attn_mask_enabled,
            ) for _ in range(depth)
        ])
        self.long_skip_connection = nn.Linear(
            dim * 2, dim, bias=False
        ) if long_skip_connection else None

        self.norm_out = AdaLayerNorm_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations

        # ---------- ControlNet 相关模块 ----------
        self.use_controlnet = use_controlnet
        if use_controlnet:
            # 1. 可训练的 ControlNet 块（结构与原始块相同）
            self.controlnet_blocks = nn.ModuleList([
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                ) for _ in range(depth)
            ])
            # 2. 零卷积层（线性层，初始化为0）
            self.zero_convs = nn.ModuleList([
                nn.Linear(dim, dim, bias=False) for _ in range(depth)
            ])
            for conv in self.zero_convs:
                nn.init.zeros_(conv.weight)
            # 3. 额外特征编码器 (7 -> dim)
            self.extra_feature_linear = nn.Linear(controlnet_input_dim, dim)

        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out AdaLN layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

        # ControlNet 的零卷积已经在 __init__ 中初始化为0，无需额外操作

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    def get_input_embed(
        self,
        x,
        cond,
        text,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cache: bool = True,
        audio_mask: Optional[torch.Tensor] = None,
    ):
        if self.text_uncond is None or self.text_cond is None or not cache:
            if audio_mask is None:
                text_embed = self.text_embed(
                    text, x.shape[1], drop_text=drop_text
                )
            else:
                batch = x.shape[0]
                seq_lens = audio_mask.sum(dim=1)
                text_embed_list = []
                for i in range(batch):
                    text_embed_i = self.text_embed(
                        text[i].unsqueeze(0),
                        seq_len=seq_lens[i].item(),
                        drop_text=drop_text,
                    )
                    text_embed_list.append(text_embed_i[0])
                text_embed = pad_sequence(
                    text_embed_list, batch_first=True, padding_value=0
                )
            if cache:
                if drop_text:
                    self.text_uncond = text_embed
                else:
                    self.text_cond = text_embed

        if cache:
            if drop_text:
                text_embed = self.text_uncond
            else:
                text_embed = self.text_cond

        x = self.input_embed(
            x,
            cond,
            text_embed,
            drop_audio_cond=drop_audio_cond,
            audio_mask=audio_mask
        )
        return x

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None

    # ---------- ControlNet 权重复制与冻结 ----------
    def copy_controlnet_weights(self):
        """将原始 transformer_blocks 的权重复制到 controlnet_blocks"""
        if self.use_controlnet:
            for orig, ctrl in zip(self.transformer_blocks, self.controlnet_blocks):
                ctrl.load_state_dict(orig.state_dict())

    def freeze_original(self):
        """冻结所有非 ControlNet 的参数（原始 DiT 全部参数）"""
        for name, param in self.named_parameters():
            if not (name.startswith('controlnet_') or
                    name.startswith('zero_convs') or
                    name.startswith('extra_feature_linear')):
                param.requires_grad = False

    # ---------- 主 forward ----------
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        text: torch.Tensor,
        time: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cfg_infer: bool = False,
        cache: bool = False,
        extra_features: Optional[torch.Tensor] = None,   # 形状 [b, 3, 7]
        controlnet_scale: float = 1.0,                    # λ，推理时使用
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # 时间嵌入
        t = self.time_embed(time)

        # --- 原始路径的输入嵌入 ---
        if cfg_infer:
            x_cond = self.get_input_embed(
                x,
                cond,
                text,
                drop_audio_cond=False,
                drop_text=False,
                cache=cache,
                audio_mask=mask
            )
            x_uncond = self.get_input_embed(
                x,
                cond,
                text,
                drop_audio_cond=True,
                drop_text=True,
                cache=cache,
                audio_mask=mask
            )
            x = torch.cat((x_cond, x_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
            # extra_features 也需要重复
            if self.use_controlnet and extra_features is not None:
                extra_features = torch.cat([extra_features, extra_features], dim=0)
        else:
            x = self.get_input_embed(
                x,
                cond,
                text,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
                cache=cache,
                audio_mask=mask
            )

        # --- ControlNet 输入准备 ---
        if self.use_controlnet and extra_features is not None:
            # 编码 extra_features: [b, 3, 7] -> [b, 3, dim]
            extra_embed = self.extra_feature_linear(extra_features)   # [b, 3, dim]
            # 上采样到 seq_len（线性插值）
            extra_embed = extra_embed.permute(0, 2, 1)                # [b, dim, 3]
            extra_embed = F.interpolate(extra_embed, size=seq_len, mode='linear', align_corners=False)  # [b, dim, seq_len]
            extra_embed = extra_embed.permute(0, 2, 1)                # [b, seq_len, dim]
            c_input = x + extra_embed   # ControlNet 的初始输入（与原始输入相加）
        else:
            c_input = x if self.use_controlnet else None

        # --- 位置编码和长跳跃 ---
        rope = self.rotary_embed.forward_from_seq_len(seq_len)
        residual = x if self.long_skip_connection is not None else None

        # --- 逐层处理 ---
        for i, block in enumerate(self.transformer_blocks):
            # 原始 block
            y = block(x, t, mask=mask, rope=rope)

            if self.use_controlnet and extra_features is not None and controlnet_scale != 0.0:
                # ControlNet block
                c = self.controlnet_blocks[i](c_input, t, mask=mask, rope=rope)
                delta = self.zero_convs[i](c)
                x = y + controlnet_scale * delta      # 应用缩放
                c_input = c                           # 更新 ControlNet 的下一层输入
            else:
                x = y

        # 长跳跃连接
        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        # 最终输出
        x = self.norm_out(x, t)
        output = self.proj_out(x)
        return output


class DiT_Controlnet_tune_emo(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        text_mask_padding=True,
        text_embedding_average_upsampling=False,
        qk_norm=None,
        conv_layers=0,
        pe_attn_head=None,
        attn_backend="torch",
        attn_mask_enabled=False,
        long_skip_connection=False,
        checkpoint_activations=False,
        # ---------- ControlNet 参数 ----------
        use_controlnet: bool = False,
        controlnet_input_dim: int = 7,
        controlnet_time_steps: int = 3,
        num_emotions: Optional[int] = None,   # 情感类别数
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(
            text_num_embeds,
            text_dim,
            mask_padding=text_mask_padding,
            average_upsampling=text_embedding_average_upsampling,
            conv_layers=conv_layers,
        )
        self.text_cond, self.text_uncond = None, None
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        # 原始 DiT 块
        self.transformer_blocks = nn.ModuleList([
            DiTBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout,
                qk_norm=qk_norm,
                pe_attn_head=pe_attn_head,
                attn_backend=attn_backend,
                attn_mask_enabled=attn_mask_enabled,
            ) for _ in range(depth)
        ])
        self.long_skip_connection = nn.Linear(
            dim * 2, dim, bias=False
        ) if long_skip_connection else None

        self.norm_out = AdaLayerNorm_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations

        # ---------- ControlNet 模块 ----------
        self.use_controlnet = use_controlnet
        if use_controlnet:
            self.controlnet_blocks = nn.ModuleList([
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                ) for _ in range(depth)
            ])
            self.zero_convs = nn.ModuleList([
                nn.Linear(dim, dim, bias=False) for _ in range(depth)
            ])
            for conv in self.zero_convs:
                nn.init.zeros_(conv.weight)

            # 编码 extra_features (7 -> dim)
            self.extra_feature_linear = nn.Linear(controlnet_input_dim, dim)

            # 编码 emo_id (类别数 -> dim)
            if num_emotions is not None:
                self.emo_embed = nn.Embedding(num_emotions, dim)
            else:
                self.emo_embed = None

            # 分别归一化两个模态
            self.norm_extra = nn.LayerNorm(dim)
            self.norm_emo = nn.LayerNorm(dim)

        self.initialize_weights()
    
    def initialize_weights(self):
        # Zero-out AdaLN layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

        # ControlNet 的零卷积已经在 __init__ 中初始化为0，无需额外操作

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    def get_input_embed(
        self,
        x,
        cond,
        text,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cache: bool = True,
        audio_mask: Optional[torch.Tensor] = None,
    ):
        if self.text_uncond is None or self.text_cond is None or not cache:
            if audio_mask is None:
                text_embed = self.text_embed(
                    text, x.shape[1], drop_text=drop_text
                )
            else:
                batch = x.shape[0]
                seq_lens = audio_mask.sum(dim=1)
                text_embed_list = []
                for i in range(batch):
                    text_embed_i = self.text_embed(
                        text[i].unsqueeze(0),
                        seq_len=seq_lens[i].item(),
                        drop_text=drop_text,
                    )
                    text_embed_list.append(text_embed_i[0])
                text_embed = pad_sequence(
                    text_embed_list, batch_first=True, padding_value=0
                )
            if cache:
                if drop_text:
                    self.text_uncond = text_embed
                else:
                    self.text_cond = text_embed

        if cache:
            if drop_text:
                text_embed = self.text_uncond
            else:
                text_embed = self.text_cond

        x = self.input_embed(
            x,
            cond,
            text_embed,
            drop_audio_cond=drop_audio_cond,
            audio_mask=audio_mask
        )
        return x

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None

    # ---------- ControlNet 权重复制与冻结 ----------
    def copy_controlnet_weights(self):
        """将原始 transformer_blocks 的权重复制到 controlnet_blocks"""
        if self.use_controlnet:
            for orig, ctrl in zip(self.transformer_blocks, self.controlnet_blocks):
                ctrl.load_state_dict(orig.state_dict())

    def freeze_original(self):
        """冻结所有非 ControlNet 的参数（原始 DiT 全部参数）"""
        for name, param in self.named_parameters():
            if not (name.startswith('controlnet_') or
                    name.startswith('zero_convs') or
                    name.startswith('extra_feature_linear')):
                param.requires_grad = False
    # ------- 其余方法（initialize_weights, ckpt_wrapper, get_input_embed, clear_cache, copy_controlnet_weights, freeze_original）与之前完全相同 -------

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        text: torch.Tensor,
        time: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cfg_infer: bool = False,
        cache: bool = False,
        extra_features: Optional[torch.Tensor] = None,   # [B, 3, 7]
        emo_id: Optional[torch.Tensor] = None,           # [B] 或 [B,1]
        controlnet_scale: float = 1.0,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        t = self.time_embed(time)

        # --- 原始路径输入嵌入 ---
        if cfg_infer:
            x_cond = self.get_input_embed(
                x,
                cond,
                text,
                drop_audio_cond=False,
                drop_text=False,
                cache=cache,
                audio_mask=mask
            )
            x_uncond = self.get_input_embed(
                x,
                cond,
                text,
                drop_audio_cond=True,
                drop_text=True,
                cache=cache,
                audio_mask=mask
            )
            x = torch.cat((x_cond, x_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
            # 对 extra_features 和 emo_id 做同样的 batch 拼接
            if self.use_controlnet and extra_features is not None:
                extra_features = torch.cat([extra_features, extra_features], dim=0)
            if self.use_controlnet and emo_id is not None:
                emo_id = torch.cat([emo_id, emo_id], dim=0)
        else:
            x = self.get_input_embed(
                x,
                cond,
                text,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
                cache=cache,
                audio_mask=mask
            )

        # --- ControlNet 输入准备（分别编码、归一化、相加融合）---
        combined_embed = None
        if self.use_controlnet:
            # 处理 extra_features
            if extra_features is not None:
                extra_embed = self.extra_feature_linear(extra_features)  # [B, 3, dim]
                extra_embed = self.norm_extra(extra_embed)               # 归一化
                combined_embed = extra_embed

            # 处理 emo_id
            if emo_id is not None and self.emo_embed is not None:
                if emo_id.dim() == 2 and emo_id.size(1) == 1:
                    emo_id = emo_id.squeeze(1)   # [B]
                emo_emb = self.emo_embed(emo_id)                     # [B, dim]
                emo_emb = emo_emb.unsqueeze(1).expand(-1, 3, -1)    # [B, 3, dim]
                emo_emb = self.norm_emo(emo_emb)                     # 归一化
                if combined_embed is not None:
                    combined_embed = combined_embed + emo_emb        # 相加融合
                else:
                    combined_embed = emo_emb       
            # 若融合后非空，则上采样到 seq_len 并加到 x 上
            if combined_embed is not None:
                # combined_embed: [B, 3, dim]
                extra_embed = combined_embed.permute(0, 2, 1)       # [B, dim, 3]
                extra_embed = F.interpolate(
                    extra_embed, size=seq_len, mode='linear', align_corners=False
                )                                                  # [B, dim, seq_len]
                extra_embed = extra_embed.permute(0, 2, 1)         # [B, seq_len, dim]
                c_input = x + extra_embed
            else:
                c_input = x
        else:
            c_input = x if self.use_controlnet else None

        # --- 位置编码、逐层处理（同原来）---
        rope = self.rotary_embed.forward_from_seq_len(seq_len)
        residual = x if self.long_skip_connection is not None else None

        for i, block in enumerate(self.transformer_blocks):
            y = block(x, t, mask=mask, rope=rope)
            if self.use_controlnet and combined_embed is not None and controlnet_scale != 0.0:
                c = self.controlnet_blocks[i](c_input, t, mask=mask, rope=rope)
                delta = self.zero_convs[i](c)
                x = y + controlnet_scale * delta
                c_input = c
            else:
                x = y

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)
        return output
    
class DiT_Controlnet_tune_emo_only(DiT_Controlnet_tune_emo):
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        text: torch.Tensor,
        time: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cfg_infer: bool = False,
        cache: bool = False,
        extra_features: Optional[torch.Tensor] = None,   # [B, 3, 7]
        emo_id: Optional[torch.Tensor] = None,           # [B] 或 [B,1]
        controlnet_scale: float = 1.0,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        t = self.time_embed(time)

        # --- 原始路径输入嵌入 ---
        if cfg_infer:
            x_cond = self.get_input_embed(
                x,
                cond,
                text,
                drop_audio_cond=False,
                drop_text=False,
                cache=cache,
                audio_mask=mask
            )
            x_uncond = self.get_input_embed(
                x,
                cond,
                text,
                drop_audio_cond=True,
                drop_text=True,
                cache=cache,
                audio_mask=mask
            )
            x = torch.cat((x_cond, x_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
            # 对 extra_features 和 emo_id 做同样的 batch 拼接
            if self.use_controlnet and extra_features is not None:
                extra_features = torch.cat([extra_features, extra_features], dim=0)
            if self.use_controlnet and emo_id is not None:
                emo_id = torch.cat([emo_id, emo_id], dim=0)
        else:
            x = self.get_input_embed(
                x,
                cond,
                text,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
                cache=cache,
                audio_mask=mask
            )

        # --- ControlNet 输入准备（分别编码、归一化、相加融合）---
        combined_embed = None
        if self.use_controlnet:
            # 处理 extra_features
            if extra_features is not None:
                extra_embed = self.extra_feature_linear(extra_features)  # [B, 3, dim]
                extra_embed = self.norm_extra(extra_embed)               # 归一化
                combined_embed = extra_embed
            combined_embed = torch.zeros_like(combined_embed)
            # 处理 emo_id
            if emo_id is not None and self.emo_embed is not None:
                if emo_id.dim() == 2 and emo_id.size(1) == 1:
                    emo_id = emo_id.squeeze(1)   # [B]
                emo_emb = self.emo_embed(emo_id)                     # [B, dim]
                emo_emb = emo_emb.unsqueeze(1).expand(-1, 3, -1)    # [B, 3, dim]
                emo_emb = self.norm_emo(emo_emb)                     # 归一化
                if combined_embed is not None:
                    combined_embed = combined_embed + emo_emb        # 相加融合
                else:
                    combined_embed = emo_emb       
            # 若融合后非空，则上采样到 seq_len 并加到 x 上
            if combined_embed is not None:
                # combined_embed: [B, 3, dim]
                extra_embed = combined_embed.permute(0, 2, 1)       # [B, dim, 3]
                extra_embed = F.interpolate(
                    extra_embed, size=seq_len, mode='linear', align_corners=False
                )                                                  # [B, dim, seq_len]
                extra_embed = extra_embed.permute(0, 2, 1)         # [B, seq_len, dim]
                c_input = x + extra_embed
            else:
                c_input = x
        else:
            c_input = x if self.use_controlnet else None

        # --- 位置编码、逐层处理（同原来）---
        rope = self.rotary_embed.forward_from_seq_len(seq_len)
        residual = x if self.long_skip_connection is not None else None

        for i, block in enumerate(self.transformer_blocks):
            y = block(x, t, mask=mask, rope=rope)
            if self.use_controlnet and combined_embed is not None and controlnet_scale != 0.0:
                c = self.controlnet_blocks[i](c_input, t, mask=mask, rope=rope)
                delta = self.zero_convs[i](c)
                x = y + controlnet_scale * delta
                c_input = c
            else:
                x = y

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)
        return output

class DiT_Controlnet_tune_emo_select(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        text_mask_padding=True,
        text_embedding_average_upsampling=False,
        qk_norm=None,
        conv_layers=0,
        pe_attn_head=None,
        attn_backend="torch",
        attn_mask_enabled=False,
        long_skip_connection=False,
        checkpoint_activations=False,
        # ---------- ControlNet 参数 ----------
        use_controlnet: bool = False,
        controlnet_input_dim: int = 7,
        controlnet_time_steps: int = 3,
        num_emotions: Optional[int] = None,   # 情感类别数
        active_blocks: Optional[List[int]] = None,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(
            text_num_embeds,
            text_dim,
            mask_padding=text_mask_padding,
            average_upsampling=text_embedding_average_upsampling,
            conv_layers=conv_layers,
        )
        self.text_cond, self.text_uncond = None, None
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        # 原始 DiT 块
        self.transformer_blocks = nn.ModuleList([
            DiTBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=ff_mult,
                dropout=dropout,
                qk_norm=qk_norm,
                pe_attn_head=pe_attn_head,
                attn_backend=attn_backend,
                attn_mask_enabled=attn_mask_enabled,
            ) for _ in range(depth)
        ])
        self.long_skip_connection = nn.Linear(
            dim * 2, dim, bias=False
        ) if long_skip_connection else None

        self.norm_out = AdaLayerNorm_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations

        # ---------- ControlNet 模块 ----------
        self.use_controlnet = use_controlnet
        if use_controlnet:
            self.controlnet_blocks = nn.ModuleList([
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                ) for _ in range(depth)
            ])
            self.zero_convs = nn.ModuleList([
                nn.Linear(dim, dim, bias=False) for _ in range(depth)
            ])
            for conv in self.zero_convs:
                nn.init.zeros_(conv.weight)

            # 编码 extra_features (7 -> dim)
            self.extra_feature_linear = nn.Linear(controlnet_input_dim, dim)

            # 编码 emo_id (类别数 -> dim)
            if num_emotions is not None:
                self.emo_embed = nn.Embedding(num_emotions, dim)
            else:
                self.emo_embed = None

            # 分别归一化两个模态
            self.norm_extra = nn.LayerNorm(dim)
            self.norm_emo = nn.LayerNorm(dim)

            if active_blocks is None:
                self.active_blocks = list(range(depth))   # 默认全部激活
            else:
                # 验证所有索引在有效范围内
                for idx in active_blocks:
                    if idx < 0 or idx >= depth:
                        raise ValueError(f"Block index {idx} out of range (0-{depth-1})")
                self.active_blocks = sorted(set(active_blocks))   # 去重排序
            print(f"ControlNet active blocks: {self.active_blocks}")   # 打印激活块
        else:
            self.active_blocks = []
        self.initialize_weights()
    
    def initialize_weights(self):
        # Zero-out AdaLN layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

        # ControlNet 的零卷积已经在 __init__ 中初始化为0，无需额外操作

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    def get_input_embed(
        self,
        x,
        cond,
        text,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cache: bool = True,
        audio_mask: Optional[torch.Tensor] = None,
    ):
        if self.text_uncond is None or self.text_cond is None or not cache:
            if audio_mask is None:
                text_embed = self.text_embed(
                    text, x.shape[1], drop_text=drop_text
                )
            else:
                batch = x.shape[0]
                seq_lens = audio_mask.sum(dim=1)
                text_embed_list = []
                for i in range(batch):
                    text_embed_i = self.text_embed(
                        text[i].unsqueeze(0),
                        seq_len=seq_lens[i].item(),
                        drop_text=drop_text,
                    )
                    text_embed_list.append(text_embed_i[0])
                text_embed = pad_sequence(
                    text_embed_list, batch_first=True, padding_value=0
                )
            if cache:
                if drop_text:
                    self.text_uncond = text_embed
                else:
                    self.text_cond = text_embed

        if cache:
            if drop_text:
                text_embed = self.text_uncond
            else:
                text_embed = self.text_cond

        x = self.input_embed(
            x,
            cond,
            text_embed,
            drop_audio_cond=drop_audio_cond,
            audio_mask=audio_mask
        )
        return x

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None

    # ---------- ControlNet 权重复制与冻结 ----------
    def copy_controlnet_weights(self):
        """将原始 transformer_blocks 的权重复制到 controlnet_blocks"""
        if self.use_controlnet:
            for orig, ctrl in zip(self.transformer_blocks, self.controlnet_blocks):
                ctrl.load_state_dict(orig.state_dict())

    def freeze_original(self):
        """冻结所有非 ControlNet 的参数（原始 DiT 全部参数）"""
        for name, param in self.named_parameters():
            if not (name.startswith('controlnet_') or
                    name.startswith('zero_convs') or
                    name.startswith('extra_feature_linear')):
                param.requires_grad = False
    # ------- 其余方法（initialize_weights, ckpt_wrapper, get_input_embed, clear_cache, copy_controlnet_weights, freeze_original）与之前完全相同 -------

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        text: torch.Tensor,
        time: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cfg_infer: bool = False,
        cache: bool = False,
        extra_features: Optional[torch.Tensor] = None,   # [B, 3, 7]
        emo_id: Optional[torch.Tensor] = None,           # [B] 或 [B,1]
        controlnet_scale: float = 1.0,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        t = self.time_embed(time)

        # --- 原始路径输入嵌入 ---
        if cfg_infer:
            x_cond = self.get_input_embed(
                x,
                cond,
                text,
                drop_audio_cond=False,
                drop_text=False,
                cache=cache,
                audio_mask=mask
            )
            x_uncond = self.get_input_embed(
                x,
                cond,
                text,
                drop_audio_cond=True,
                drop_text=True,
                cache=cache,
                audio_mask=mask
            )
            x = torch.cat((x_cond, x_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
            # 对 extra_features 和 emo_id 做同样的 batch 拼接
            if self.use_controlnet and extra_features is not None:
                extra_features = torch.cat([extra_features, extra_features], dim=0)
            if self.use_controlnet and emo_id is not None:
                emo_id = torch.cat([emo_id, emo_id], dim=0)
        else:
            x = self.get_input_embed(
                x,
                cond,
                text,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
                cache=cache,
                audio_mask=mask
            )

        # --- ControlNet 输入准备（分别编码、归一化、相加融合）---
        combined_embed = None
        if self.use_controlnet:
            # 处理 extra_features
            if extra_features is not None:
                extra_embed = self.extra_feature_linear(extra_features)  # [B, 3, dim]
                extra_embed = self.norm_extra(extra_embed)               # 归一化
                combined_embed = extra_embed

            # 处理 emo_id
            if emo_id is not None and self.emo_embed is not None:
                if emo_id.dim() == 2 and emo_id.size(1) == 1:
                    emo_id = emo_id.squeeze(1)   # [B]
                emo_emb = self.emo_embed(emo_id)                     # [B, dim]
                emo_emb = emo_emb.unsqueeze(1).expand(-1, 3, -1)    # [B, 3, dim]
                emo_emb = self.norm_emo(emo_emb)                     # 归一化
                if combined_embed is not None:
                    combined_embed = combined_embed + emo_emb        # 相加融合
                else:
                    combined_embed = emo_emb

            # 若融合后非空，则上采样到 seq_len 并加到 x 上
            if combined_embed is not None:
                # combined_embed: [B, 3, dim]
                extra_embed = combined_embed.permute(0, 2, 1)       # [B, dim, 3]
                extra_embed = F.interpolate(
                    extra_embed, size=seq_len, mode='linear', align_corners=False
                )                                                  # [B, dim, seq_len]
                extra_embed = extra_embed.permute(0, 2, 1)         # [B, seq_len, dim]
                c_input = x + extra_embed
            else:
                c_input = x
        else:
            c_input = x if self.use_controlnet else None

        # --- 位置编码、逐层处理（同原来）---
        rope = self.rotary_embed.forward_from_seq_len(seq_len)
        residual = x if self.long_skip_connection is not None else None

        # for i, block in enumerate(self.transformer_blocks):
        #     y = block(x, t, mask=mask, rope=rope)
        #     if self.use_controlnet and combined_embed is not None and controlnet_scale != 0.0:
        #         c = self.controlnet_blocks[i](c_input, t, mask=mask, rope=rope)
        #         delta = self.zero_convs[i](c)
        #         x = y + controlnet_scale * delta
        #         c_input = c
        #     else:
        #         x = y

        for i, block in enumerate(self.transformer_blocks):
            y = block(x, t, mask=mask, rope=rope)   # 原始块输出

            if self.use_controlnet and combined_embed is not None and controlnet_scale != 0.0:
                if i in self.active_blocks:        # 当前块被激活
                    c = self.controlnet_blocks[i](c_input, t, mask=mask, rope=rope)
                    delta = self.zero_convs[i](c)
                    x = y + controlnet_scale * delta
                    c_input = c                    # 更新 c_input 以便下一层使用
                else:
                    x = y                          # 旁路：不添加 ControlNet 输出
                    # c_input 保持不变（继续传递之前的控制特征）
            else:
                x = y

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)
        return output
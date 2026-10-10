from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from huggingface_hub import hf_hub_download

from eval.ecapa_tdnn import ECAPA_TDNN_SMALL
from accel_hydra.utils.torch import create_mask_from_length as lens_to_mask


class ContentEncoder(nn.Module):
    def __init__(
        self,
        speaker_model: str,
        text_encoder: nn.Module = None,
        p: int=0.5,
        input_dim: int=1024,
        output_dim: int=1536,
    ):
        super().__init__()
        self.speaker_encoder = ECAPA_TDNN_SMALL(
                feat_dim=1024, feat_type="wavlm_large", config_path=None
            )
        speaker_checkpoint = speaker_model
        if "/" in speaker_model and not speaker_model.endswith((".pt", ".pth", ".bin")):
            speaker_checkpoint = hf_hub_download(
                repo_id=speaker_model,
                filename="wavlm_large_finetune.pth",
            )
        state_dict = torch.load(
            speaker_checkpoint,
            weights_only=True,
            map_location=lambda storage, loc: storage)
        self.speaker_encoder.load_state_dict(state_dict["model"], strict=False)
        self.speaker_encoder.eval()
        self.text_encoder = text_encoder
        self.p=p
        #self.task_embeddings = nn.Embedding(2, output_dim)
        self.feat_proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            #nn.GELU(),
            #nn.Linear(output_dim, output_dim),
            #nn.LayerNorm(output_dim)
            )
    
    def encode_content(
        self, prompt_waveform: torch.Tensor, prompt_waveform_lengths: torch.Tensor, caption: list, infer_type: int, 
        device: str | torch.device
    ):
        
        B = prompt_waveform.shape[0]
        batch_caption_embed = []
        batch_caption_mask = []
        for idx in range(B):
            caption_idx = caption[idx]
            if caption_idx=="":     # prompt  
                prompt_waveform_idx_len = prompt_waveform_lengths[idx]
                prompt_waveform_idx = prompt_waveform[idx][..., :prompt_waveform_idx_len]
                if prompt_waveform_idx.dim() == 1:  
                    prompt_waveform_idx = prompt_waveform_idx.unsqueeze(0)  # [1, T]
                caption_embed = self.speaker_encoder.encode(prompt_waveform_idx).squeeze(0)
            else:     #caption
                caption_embed = self.text_encoder(caption_idx)[0].squeeze(0)
                caption_embed = self.feat_proj(caption_embed)
            
            #task_id = torch.tensor(type, device=device)    
            #task_embed = self.task_embeddings(task_id).unsqueeze(0)
            #caption_embed = torch.cat([task_embed, caption_embed], dim=0)

            caption_mask = torch.ones(caption_embed.shape[0], dtype=torch.bool, device=device)  # mel部分有效

            batch_caption_embed.append(caption_embed)
            batch_caption_mask.append(caption_mask)

        batch_caption_embed = nn.utils.rnn.pad_sequence(batch_caption_embed, batch_first=True, padding_value=0).to(device)
        batch_caption_mask = nn.utils.rnn.pad_sequence(batch_caption_mask, batch_first=True, padding_value=False).to(device)

        return batch_caption_embed, batch_caption_mask

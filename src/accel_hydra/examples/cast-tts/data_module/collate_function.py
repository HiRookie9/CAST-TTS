from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass
class CASTMelCollate:
    def __call__(self, batch: dict) -> dict:
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        mel_specs = [item["mel_spec"].squeeze(0) for item in batch]
        mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
        max_mel_length = mel_lengths.amax()

        padded_mel_specs = []
        for spec in mel_specs:
            padding = (0, max_mel_length - spec.size(-1))
            padded_spec = F.pad(spec, padding, value=0)
            padded_mel_specs.append(padded_spec)

        mel_specs = torch.stack(padded_mel_specs)

        text = [item["text"] for item in batch]
        text_lengths = torch.LongTensor([len(item) for item in text])
        return dict(
            inp=mel_specs.permute(0, 2, 1),
            lens=mel_lengths,  # records for padding mask
            text=text,
            text_lengths=text_lengths,
        )


@dataclass
class CASTMelCollateWithoutRename(CASTMelCollate):
    def __call__(self, batch: dict) -> dict:
        output = super().__call__(batch)
        if output is None:
            return None
        return {
            "mel_spec": output["inp"],
            "mel_spec_lengths": output["lens"],
            "text": output["text"],
            "text_lengths": output["text_lengths"]
        }

@dataclass
class CASTCollate:
    def collate_wav(self, batch: list, key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        waves = []
        wave_lens = []
        for item in batch:
            wav = item[key]  
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)          
            elif wav.dim() == 2 and wav.shape[0] > wav.shape[1]:
                wav = wav.t()
            waves.append(wav)
            wave_lens.append(wav.shape[-1])

        wave_lens = torch.LongTensor(wave_lens)
        max_len = wave_lens.amax()

        # padding 
        padded_waves = []
        for wav in waves:
            pad = (0, max_len - wav.shape[-1])
            padded_wav = F.pad(wav, pad, value=0.0)  # [C, max_len]
            padded_waves.append(padded_wav)

        waves = torch.stack(padded_waves)  # [B, C, max_len]
        return waves, wave_lens

    def __call__(self, batch: dict) -> dict:
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        
        prompt_waveform, prompt_waveform_lengths = self.collate_wav(batch, "prompt_waveform")
        target_waveform, target_waveform_lengths = self.collate_wav(batch, "target_waveform")
        
        text = [item["text"] for item in batch]
        text_lengths = torch.LongTensor([len(item) for item in text])
        caption = [item["caption"] for item in batch]
        mask_len = torch.tensor([item["mask_len"] for item in batch], dtype=torch.long)

        return dict(
            prompt_waveform=prompt_waveform,
            prompt_waveform_lengths = prompt_waveform_lengths,
            target_waveform=target_waveform,  
            target_waveform_lengths = target_waveform_lengths,
            text=text,
            caption=caption,
            text_lengths=text_lengths,
            mask_len=mask_len,
        )

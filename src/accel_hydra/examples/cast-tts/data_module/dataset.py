import json
from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from pathlib import Path

def read_jsonl(jsonl_file: Union[str, Path]) -> List[Dict]:
    """
    Read single jsonl file, return list of dicts.
    """
    out = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line:  # 跳过空行
                data = json.loads(stripped_line)
                out.append(data)
    return out


def read_multiple_jsonl(jsonl_files: List[Union[str, Path]]) -> List[Dict]:
    """
    Read multiple jsonl files and merge data into one list.
    """
    all_data = []
    for file_path in jsonl_files:
        try:
            file_data = read_jsonl(file_path)
            all_data.extend(file_data)
        except Exception as e:
            print(f"Warning: Failed to load {file_path} - {e}")
    return all_data


@dataclass
class CustomDataset(torch.utils.data.Dataset):
    jsonl_files: Union[str, Path, List[Union[str, Path]]]
    min_duration: float = field(default=0.3)
    max_duration: float = field(default=30.0)
    target_sample_rate: int = field(default=24_000)
    prompt_sample_rate: int = field(default=16_000)
    
    hop_length: int = field(default=256)
    max_samples: Optional[int] = field(default=None)

    def __post_init__(self):
        if not isinstance(self.jsonl_files, list):
            self.jsonl_files = [self.jsonl_files]
        
        self.data = read_multiple_jsonl(self.jsonl_files)
        if not self.data:
            raise ValueError("No valid data loaded from the provided jsonl files!")
        
        self.durations = [item["duration"] for item in self.data]
        
        if self.max_samples is not None and self.max_samples < len(self.data):
            np.random.seed(42)  
            sample_idx = np.random.choice(len(self.data), self.max_samples, replace=False)
            self.data = [self.data[i] for i in sample_idx]
            self.durations = [self.durations[i] for i in sample_idx]
            print(f"Sampled {self.max_samples} samples from total {len(self.data)}")


    def get_length(self, index: int) -> float:
        """Calculate mel spec length (time steps)"""
        if self.durations is not None:
            return self.durations[index] * self.target_sample_rate / self.hop_length
        return self.data[index]["duration"] * self.target_sample_rate / self.hop_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str]]:
        while True:
            row = self.data[index]
            audio_path = row["location"]
            text = row.get("text", "")
            mask_duration = row.get("mask_len", 0)
            caption = row.get("caption", "")
            duration = row["duration"]

            if self.min_duration <= duration <= self.max_duration:
                break  
            
            index = (index + 1) % len(self.data)
       
        audio, source_sample_rate = torchaudio.load(audio_path)

        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Resample
        if source_sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                source_sample_rate, self.target_sample_rate
            )
            target_audio = resampler(audio)
        else:
            target_audio = audio

        if source_sample_rate != self.prompt_sample_rate:
            resampler = torchaudio.transforms.Resample(
                source_sample_rate, self.prompt_sample_rate
            )
            prompt_audio = resampler(audio)
        else:
            prompt_audio = audio
        
        if mask_duration==0:  #caption
            assert caption != "", f"[AssertError] mask_duration==0 but caption==''! \nRow: {row}"
            prompt_waveform = prompt_audio[..., :1]
            target_waveform = target_audio[..., 0:]
            target_mask_len=0
        else:
            assert caption == "", f"[AssertError] mask_duration==0 but caption==''! \nRow: {row}"
            target_mask_len = int(mask_duration * self.target_sample_rate)
            prompt_mask_len = int(mask_duration * self.prompt_sample_rate)
            
            prompt_waveform = prompt_audio[..., :prompt_mask_len]
            target_waveform = target_audio[..., target_mask_len:]
        return {
            "prompt_waveform": prompt_waveform,
            "target_waveform": target_waveform,
            "text": text,
            "caption": caption,
            "mask_len": target_mask_len,
        }

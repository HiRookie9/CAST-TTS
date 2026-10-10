import torch
from accel_hydra.models.common import LoadPretrainedBase
from utils.audio import MelSpec


class MEL(LoadPretrainedBase, torch.nn.Module):
    def __init__(self, encoder, decoder, latent_dim, downsampling_ratio,
                 sample_rate, **kwargs):
        LoadPretrainedBase.__init__(self)
        torch.nn.Module.__init__(self)
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.downsampling_ratio = downsampling_ratio
        self.sample_rate = sample_rate

    def encode(self, waveform, waveform_lengths):
        if waveform.ndim == 3 and waveform.size(1) == 1:
            waveform = waveform.squeeze(1)
        mel = self.encoder(waveform)
        mel_lengths = torch.div(
            waveform_lengths, self.downsampling_ratio, rounding_mode="floor"
        ).long().to(mel.device)
        mel_mask = torch.arange(mel.shape[2], device=mel.device).unsqueeze(0) < mel_lengths.unsqueeze(1)
        mel = (mel * mel_mask.unsqueeze(1).to(mel.dtype)).permute(0, 2, 1)
        return mel, mel_lengths, mel_mask

    def decode(self, mel, mel_mask=None):
        if mel_mask is not None:
            mel = mel.clone().masked_fill(
                ~mel_mask.unsqueeze(1),
                torch.log(torch.tensor(1e-5, device=mel.device, dtype=mel.dtype)),
            )
        return self.decoder(mel)

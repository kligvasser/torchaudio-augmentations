import os
import random
import numpy as np
import torch

import torchaudio.functional as F

from .misc import (
    get_files_by_extension,
    load_audio,
    downsample_audio,
    cut_random_segment_repeat,
)
from .noise import Noise


class RandomBackgroundNoise(torch.nn.Module):
    def __init__(
        self,
        noise_root,
        sample_rate=8000,
        segment_size=16000,
        bank_size=64,
        snr_dbs_range=(10, 30),
    ):
        super().__init__()
        self.snr_dbs_range = snr_dbs_range
        self.noiser = Noise()
        self.min_snr, self.max_snr = snr_dbs_range

        self._create_noise_bank(noise_root, bank_size, segment_size, sample_rate)
        self.keys = list(self.bank.keys())

    def _create_noise_bank(self, root, bank_size, segment_size, sample_rate):
        wav_list = get_files_by_extension(root)

        self.bank = dict()
        for i in range(bank_size):
            wav_path = random.choice(wav_list)
            audio, sr = load_audio(wav_path)
            if sr > sample_rate:
                audio = downsample_audio(audio, sr, sample_rate)
            self.bank[i] = cut_random_segment_repeat(audio, segment_size)

        return

    def forward(self, audio):
        key = random.choice(self.keys)
        noise = self.noiser(self.bank[key])
        snr_db = random.randint(self.min_snr, self.max_snr)

        noise_t = torch.from_numpy(noise).unsqueeze(0)
        audio_t = torch.from_numpy(audio).unsqueeze(0)
        snr_dbs = torch.tensor([snr_db])

        noisy_t = F.add_noise(audio_t, noise_t, snr_dbs)
        noisy = noisy_t.squeeze().numpy()

        return noisy

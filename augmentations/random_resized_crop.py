import numpy as np
import random
import torch

from .misc import cut_random_segment_zeros, cut_random_segment_repeat


class RandomResizedCrop(torch.nn.Module):
    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples

    def forward(self, audio):
        max_samples = audio.shape[-1]
        start_idx = random.randint(0, max_samples - self.n_samples)
        audio = audio[..., start_idx : start_idx + self.n_samples]
        return audio


class RandomCrop(torch.nn.Module):
    def __init__(self, segment_size, pad_type="zero"):
        super().__init__()
        self.segment_size = segment_size
        self.pad_type = pad_type

    def forward(self, audio):
        if self.pad_type == "zero":
            audio = cut_random_segment_zeros(audio, self.segment_size)
        else:
            audio = cut_random_segment_repeat(audio, self.segment_size)
        return audio


class RandomCropExpand(torch.nn.Module):
    def __init__(self, crop_size, expand_size, pad_type="zero"):
        super().__init__()
        self.crop_size = crop_size
        self.expand_size = expand_size
        self.pad_type = pad_type

    def forward(self, audio):
        if self.pad_type == "zero":
            audio = cut_random_segment_zeros(audio, self.crop_size)
            audio = cut_random_segment_zeros(audio, self.expand_size)
        else:
            audio = cut_random_segment_repeat(audio, self.crop_size)
            audio = cut_random_segment_repeat(audio, self.expand_size)
        return audio

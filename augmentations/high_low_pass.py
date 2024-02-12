import random
import torch
from . import HighPassFilter, LowPassFilter
from .misc import numpy_to_tensor, tensor_to_numpy


class HighLowPass(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int,
        lowpass_freq_low: float = 2200,
        lowpass_freq_high: float = 4000,
        highpass_freq_low: float = 200,
        highpass_freq_high: float = 1200,
    ):
        super().__init__()
        self.sample_rate = sample_rate

        self.high_pass_filter = HighPassFilter(
            sample_rate, highpass_freq_low, highpass_freq_high
        )
        self.low_pass_filter = LowPassFilter(
            sample_rate, lowpass_freq_low, lowpass_freq_high
        )

    def forward(self, audio):
        audio_t = numpy_to_tensor(audio)
        highlowband = random.randint(0, 1)
        if highlowband == 0:
            audio_t = self.high_pass_filter(audio_t)
        else:
            audio_t = self.low_pass_filter(audio_t)
        audio = tensor_to_numpy(audio_t)
        return audio

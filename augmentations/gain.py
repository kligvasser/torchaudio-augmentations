import torch
import random
from torchaudio.transforms import Vol
from .misc import numpy_to_tensor, tensor_to_numpy


class RandomGain(torch.nn.Module):
    def __init__(self, min_gain: float = -20.0, max_gain: float = -1):
        super().__init__()
        self.min_gain = min_gain
        self.max_gain = max_gain

    def forward(self, audio):
        gain = random.uniform(self.min_gain, self.max_gain)
        audio_t = numpy_to_tensor(audio)
        audio_t = Vol(gain, gain_type="db")(audio_t)
        audio = tensor_to_numpy(audio_t)
        return audio

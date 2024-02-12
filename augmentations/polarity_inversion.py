import torch

from .misc import numpy_to_tensor, tensor_to_numpy


class PolarityInversion(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, audio):
        audio_t = numpy_to_tensor(audio)
        audio_t = torch.neg(audio_t)
        audio = tensor_to_numpy(audio_t)
        return audio

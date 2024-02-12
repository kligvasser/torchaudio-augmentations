import random
import torch

from .effects import EffectChain
from .misc import numpy_to_tensor, tensor_to_numpy


class PitchShift:
    def __init__(
        self, n_samples, sample_rate, pitch_shift_min=-7.0, pitch_shift_max=7.0
    ):
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.pitch_shift_cents_min = int(pitch_shift_min * 100)
        self.pitch_shift_cents_max = int(pitch_shift_max * 100)
        self.src_info = {"rate": self.sample_rate}

    def process(self, x):
        n_steps = random.randint(self.pitch_shift_cents_min, self.pitch_shift_cents_max)
        effect_chain = EffectChain().pitch(n_steps).rate(self.sample_rate)
        num_channels = x.shape[0]
        target_info = {
            "channels": num_channels,
            "length": self.n_samples,
            "rate": self.sample_rate,
        }
        y = effect_chain.apply(x, src_info=self.src_info, target_info=target_info)

        # sox might misbehave sometimes by giving nan/inf if sequences are too short (or silent)
        # and the effect chain includes eg `pitch`
        if torch.isnan(y).any() or torch.isinf(y).any():
            return x.clone()

        if y.shape[1] != x.shape[1]:
            if y.shape[1] > x.shape[1]:
                y = y[:, : x.shape[1]]
            else:
                y0 = torch.zeros(num_channels, x.shape[1]).to(y.device)
                y0[:, : y.shape[1]] = y
                y = y0
        return y

    def __call__(self, audio):
        audio_t = numpy_to_tensor(audio)
        if audio_t.ndim == 3:
            for b in range(audio_t.shape[0]):
                audio_t[b] = self.process(audio_t[b])
        else:
            audio_t = self.process(audio_t)

        return tensor_to_numpy(audio_t)

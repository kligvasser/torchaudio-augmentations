import torch
import random
from torchaudio.transforms import Vol


class Gain(torch.nn.Module):
    def __init__(self, min_gain: float = -20.0, max_gain: float = -1):
        super().__init__()
        self.min_gain = min_gain
        self.max_gain = max_gain

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        gain = random.uniform(self.min_gain, self.max_gain)
        audio = Vol(gain, gain_type="db")(audio)
        return audio


class RandomGain(torch.nn.Module):
    def __init__(self, min_gain: float = -20.0, max_gain: float = -1):
        super().__init__()
        self.min_gain = min_gain
        self.max_gain = max_gain

    def forward(self, audio):
        gain = random.uniform(self.min_gain, self.max_gain)

        audio_t = torch.from_numpy(audio).unsqueeze(0)
        audio_t = Vol(gain, gain_type="db")(audio_t)
        audio = audio_t.squeeze().numpy()

        return audio

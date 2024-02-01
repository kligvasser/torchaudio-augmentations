import os
import random
import numpy as np
import torch
import torchaudio


class RandomEncoder(torch.nn.Module):
    def __init__(self, sample_rate, codecs=("pcm_mulaw", "g722", "vorbis")):
        super().__init__()
        self.sample_rate = sample_rate
        self.keys = list(codecs)
        self.encoders = {
            "pcm_mulaw": torchaudio.io.AudioEffector(format="wav", encoder="pcm_mulaw"),
            "g722": torchaudio.io.AudioEffector(format="g722", encoder=None),
            "vorbis": torchaudio.io.AudioEffector(format="ogg", encoder="vorbis"),
        }

    def _apply_codec(self, audio, sample_rate, format, encoder=None):
        encoder = torchaudio.io.AudioEffector(format=format, encoder=encoder)
        return encoder.apply(audio, sample_rate)

    def forward(self, audio):
        key = random.choice(self.keys)
        audio_t = torch.from_numpy(audio).unsqueeze(-1)
        audio_t = self.encoders[key].apply(audio_t, self.sample_rate)
        audio = audio_t.squeeze().numpy()

        return audio

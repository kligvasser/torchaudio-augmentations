import random
import numpy as np
import torch
import os

from .misc import load_dict_from_pickle

BANK_8K_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "bank", "rir_8k.pkl"
)


class RandomRIR(torch.nn.Module):
    def __init__(
        self,
        bank_path=BANK_8K_PATH,
    ):
        super().__init__()
        self.bank = load_dict_from_pickle(bank_path)
        self.keys = list(self.bank.keys())

    def forward(self, audio):
        key = random.choice(self.keys)
        rir = self.bank[key]
        audio = np.convolve(audio, rir, mode="same")
        return audio

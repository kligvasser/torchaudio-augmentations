import pickle
import random
import numpy as np
import torch


def load_dict_from_pickle(file_path):
    with open(file_path, "rb") as file:
        loaded_dict = pickle.load(file)
    return loaded_dict


class RandomRIR(torch.nn.Module):
    def __init__(
        self,
        bank_path="../../bank/rir_8k.pkl",
    ):
        super().__init__()
        self.bank = load_dict_from_pickle(bank_path)
        self.keys = list(self.bank.keys())

    def forward(self, audio):
        key = random.choice(self.keys)
        rir = self.bank[key]
        audio = np.convolve(audio, rir, mode="same")
        return audio

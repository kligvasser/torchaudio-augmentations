# PyTorch Audio Augmentations
Forked from [torchaudio-agmentations](https://github.com/Spijkervet/torchaudio-augmentations) project.

Audio data augmentations library for PyTorch for audio in the time-domain. The focus of this repository is to:
- Provide many audio transformations in an easy Python interface.
- Have a high test coverage.
- Easily control stochastic (sequential) audio transformations.
- Make every audio transformation differentiable with PyTorch's `nn.Module`.
- Optimise audio transformations for CPU and GPU.
- *Added RIR, codec and background noise augmentations.*

It supports stochastic transformations as used often in self-supervised, semi-supervised learning methods. One can apply a single stochastic augmentation or create as many stochastically transformed audio examples from a single interface.

This package follows the conventions set out by `torchvision` and `torchaudio`, with audio defined as a tensor of `[channel, time]`, or a batched representation `[batch, channel, time]`. Each individual augmentation can be initialized on its own, or be wrapped around a `RandomApply` interface which will apply the augmentation with probability `p`.

## Installation
```
sudo apt install libsox-dev
conda env create -f environment.yaml
conda activate audio
```

## Usage
We can define a single or several audio augmentations, which are applied sequentially to an audio waveform.
```python
from augmentations import *


audio, sample_rate = torchaudio.load("tests/classical.00002.wav")

segment_size = sample_rate * 10
transforms = Compose(
        [
            RandomCrop(segment_size=segment_size),
            RandomApply(
                RandomBackgroundNoise(
                    noise_root="/media/klig/disk/datasets/arabic-natural-audio",
                    sample_rate=sample_rate,
                    segment_size=segment_size,
                    bank_size=256,
                    snr_dbs_range=[20, 30],
                ),
                p=0.2,
            ),
            RandomApply(RandomRIR(), 0.3),
            RandomApply(Noise(min_snr=0.001, max_snr=0.005), p=0.3),
            RandomApply(RandomEncoder(sample_rate=sample_rate), p=0.5),
        ]
    )


audio_transformed = transforms(audio)

```

# Cite
You can cite this work with the following BibTeX:
```
@misc{spijkervet_torchaudio_augmentations,
  doi = {10.5281/ZENODO.4748582},
  url = {https://zenodo.org/record/4748582},
  author = {Spijkervet,  Janne},
  title = {Spijkervet/torchaudio-augmentations},
  publisher = {Zenodo},
  year = {2021},
  copyright = {MIT License}
}
```

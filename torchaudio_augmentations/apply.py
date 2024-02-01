import torch


class RandomApplys(torch.nn.Module):
    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, audio):
        if self.p < torch.rand(1):
            return audio
        for t in self.transforms:
            audio = t(audio)
        return audio

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "\n    p={}".format(self.p)
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomApply(torch.nn.Module):
    def __init__(self, transform, p=0.5):
        super().__init__()
        self.transform = transform
        self.p = p

    def forward(self, audio):
        if self.p < torch.rand(1):
            return audio
        else:
            return self.transform(audio)

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += "\n    p={}".format(self.p)
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

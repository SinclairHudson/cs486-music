import torch
from torchaudio.transforms import InverseSpectrogram, Spectrogram
from colorsys import hsv_to_rgb
import numpy as np

def indices_to_rgb_image(indices : torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    takes a batch of vqvae indices, maps each index to a unique colour, and then
    forms an rgb image of the indices.
    """
    bsz, x, y = indices.shape
    # switch to numpy since it allows assignment
    rgb_tensor = np.zeros((bsz, x, y, 3))
    for img in range(bsz):
        for i in range(x):
            for j in range(y):
                index = indices[img, i, j]
                hue = float(index) / vocab_size
                res = hsv_to_rgb(hue, 1, 1)
                rgb_tensor[img, i, j] = list(res)

    rgb_tensor = np.moveaxis(rgb_tensor, -1, 1)
    return torch.Tensor(rgb_tensor)

s = Spectrogram(n_fft=800, return_complex=True, power=None)
inv = InverseSpectrogram(n_fft=800)

def ml_representation_to_audio(x):
    assert len(x.shape) == 4
    x = torch.moveaxis(x, 1, -1).contiguous()  # move complex dimension to the end
    x = torch.view_as_complex(x)
    audio_one_channel = inv(x[0].cpu())
    audio = audio_one_channel.expand(2, -1)
    return audio

def spectrogram_to_ml_representation(x):
    """
    x is BxNxL of a complex spectrogram
    """
    x_ml = torch.view_as_real(x).contiguous()
    x_ml = torch.moveaxis(x_ml, -1, 1)  # move complex dimensions to channels
    return x_ml


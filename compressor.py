import torch.nn as nn
from quantize import VQVAEQuantize

class Encoder(nn.Module):
    def __init__(self, step_size=4):
        super(Encoder, self).__init__()
        self.enc = nn.ModuleList([
            nn.Conv2d(2, 32, kernel_size=5, stride=1, padding=4, dilation=2),
            nn.GroupNorm(1, 32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=6, dilation=2),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=7, stride=(4, 2), padding=6, dilation=2),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=7, stride=(4, 2), padding=6, dilation=2),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(26, 1), padding=0),
        ])

    def forward(self, x):
        for layer in self.enc:
            x = layer(x)
        return x



class Decoder(nn.Module):
    def __init__(self, embedding_dim=64, step_size=4):
        super(Decoder, self).__init__()
        self.dec = nn.ModuleList([
            nn.ConvTranspose2d(64, 64, kernel_size=(26, 1), padding=0),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=7, stride=(4, 2), padding=6, dilation=2),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=7, stride=(4, 2), padding=6, dilation=2),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=7, stride=1, padding=6, dilation=2),
            nn.GroupNorm(1, 32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 2, kernel_size=(5, 6), stride=1, padding=4, dilation=2),
        ])

    def forward(self, x):
        for layer in self.dec:
            x = layer(x)
        return x


class Compressor(nn.Module):
    def __init__(self, step_size=16, vocab_size=512):
        super(Compressor, self).__init__()
        self.encoder = Encoder(step_size=step_size)
        self.decoder = Decoder(step_size=step_size)
        self.quantizer = VQVAEQuantize(num_hiddens=64, n_embed=vocab_size, embedding_dim=64)

    def forward(self, x):
        """
        x and the return object, xhat, are assumed to be Bx2xNxL
        B is batch size, 2 for real and imaginary part, N for number of bins in the FFT,
        and L is length of the spectrogram.
        """
        encoded = self.encoder(x)
        z_q, diff, ind = self.quantizer(encoded)
        initial = self.decoder(z_q)
        return initial[:,:,:,:-1], diff  # cut the last

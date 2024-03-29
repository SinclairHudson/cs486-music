import torch.nn as nn
import torch
from quantize import VQVAEQuantize, VectorQuantizer2d

class Encoder(nn.Module):
    def __init__(self, step_size=4):
        super(Encoder, self).__init__()
        self.enc1 = nn.ModuleList([
            nn.Conv2d(2, 32, kernel_size=5, stride=1, padding=4, dilation=2),
            nn.GroupNorm(1, 32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=6, dilation=2),
            nn.GroupNorm(1, 64),
            nn.ReLU()
        ])
        self.enc2 = nn.ModuleList([
            nn.Conv2d(2 + 64, 64, kernel_size=7, stride=(4, 2), padding=6, dilation=2),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=7, stride=(4, 2), padding=6, dilation=2),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(13, 1), padding=0),
        ])

    def forward(self, x):
        y = x
        for layer in self.enc1:
            y = layer(y)
        z = torch.cat((y, x), dim=1)

        for layer in self.enc2:
            z = layer(z)
        return z


class Decoder(nn.Module):
    def __init__(self, embedding_dim=64, step_size=4):
        super(Decoder, self).__init__()
        self.dec1 = nn.ModuleList([

            nn.ConvTranspose2d(64, 92, kernel_size=(1, 1), padding=0),
            nn.GroupNorm(1, 92),
            nn.ReLU(),

            nn.ConvTranspose2d(92, 64, kernel_size=(13, 1), padding=0),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=7, stride=(4, 2), padding=6, dilation=2),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=7, stride=(4, 2), padding=7, dilation=2),
            nn.GroupNorm(1, 64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=7, stride=1, padding=(2, 1), dilation=1),
            nn.GroupNorm(1, 64),
            nn.ReLU(),
        ])

        # ensure that dec2 doesn't change size
        self.dec2 = nn.ModuleList([
            nn.ConvTranspose2d(64, 32, kernel_size=7, stride=1, padding=3),
            nn.GroupNorm(1, 32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=7, stride=1, padding=3),
            nn.GroupNorm(1, 32),
            nn.ReLU(),
        ])

        self.head = nn.Conv2d(32+64, 2, kernel_size=1)

    def forward(self, x):
        for layer in self.dec1:
            x = layer(x)
        y = x
        for layer in self.dec2:
            y = layer(y)

        return self.head(torch.cat((x, y), dim=1))


class Compressor(nn.Module):
    def __init__(self, step_size=16, vocab_size=16, beta=1):
        super(Compressor, self).__init__()
        self.encoder = Encoder(step_size=step_size)
        self.decoder = Decoder(step_size=step_size)
        self.quantizer = VectorQuantizer2d(n_e=vocab_size, e_dim=64, beta=beta)

    def forward(self, x):
        """
        x and the return object, xhat, are assumed to be Bx2xNxL
        B is batch size, 2 for real and imaginary part, N for number of bins in the FFT,
        and L is length of the spectrogram.
        """
        encoded = self.encoder(x)
        z_q, codebook_loss, ind = self.quantizer(encoded, return_indices=True)
        initial = self.decoder(z_q)
        return initial[:,:,:,:], codebook_loss, ind  # cut the last

    def random_restart(self):
        proportion = self.quantizer.random_restart()
        self.quantizer.reset_usage()
        return proportion

    def get_perplexity(self):
        return self.quantizer.perplexity

    def get_latent_space_size(self, x):
        with torch.no_grad():
            latent = self.encoder(x)
            z_q = self.quantizer(encoded, return_indices=True)
            return z_q.size()



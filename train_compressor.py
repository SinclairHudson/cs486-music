import torch
import torch.nn as nn
import torchaudio
from torch.optim import Adam
from dataset import LofiDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchaudio.transforms import InverseSpectrogram, Spectrogram
from compressor import Compressor


N_EPOCHS = 100
BATCH_SIZE = 4
LR = 0.001

s = Spectrogram(n_fft=800, return_complex=True, power=None)
inv = InverseSpectrogram(n_fft=800)

device = torch.device("cuda:0")
train_dataset = LofiDataset("/media/sinclair/datasets4/lofi/good_splits",
                            spectrogram=s,
                            length=45)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

compressor = Compressor().to(device)
optimizer = Adam(compressor.parameters(), lr=LR)

inv = InverseSpectrogram(n_fft=800)

loss_fn = nn.MSELoss()

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


for epoch in range(N_EPOCHS):
    epoch_loss = []
    sample = None
    inds= None
    for x_imaginary in tqdm(train_loader):
        optimizer.zero_grad()
        x = spectrogram_to_ml_representation(x_imaginary.to(device))
        xhat, diff, ind = compressor(x)
        recon_loss = loss_fn(x, xhat)
        loss = diff * 500 + recon_loss
        epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        torchaudio.save("x.wav", ml_representation_to_audio(x), sample_rate=44100)
        sample = ml_representation_to_audio(xhat)
        inds = ind

    print(inds)
    print(f"avg epoch loss: {sum(epoch_loss)/len(epoch_loss)}")
    torchaudio.save(f"sample_epoch_{epoch}.wav", sample, sample_rate=44100)


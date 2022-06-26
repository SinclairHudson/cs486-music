import torch
import torch.nn as nn
import torchaudio
from torch.optim import Adam
from dataset import LofiDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchaudio.transforms import InverseSpectrogram, Spectrogram
from compressor import Compressor


N_EPOCHS = 500
BATCH_SIZE = 4
LR = 0.01
VOCAB_SIZE = 64
BETA = 1
SONG_LENGTH = 5

# n_fft controls the height of the spectrogram, how many frequency bins there are
s = Spectrogram(n_fft=800, return_complex=True, power=None)
inv = InverseSpectrogram(n_fft=800)

device = torch.device("cuda:0")
train_dataset = LofiDataset("/media/sinclair/datasets4/lofi/good_splits",
                            spectrogram=s,
                            length=SONG_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

compressor = Compressor(step_size=16, vocab_size=VOCAB_SIZE, beta=BETA).to(device)

optimizer = Adam(compressor.parameters(), lr=LR)


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
    print("=" * 10 + f"starting epoch {epoch}." + "=" * 10)
    epoch_loss = []
    epoch_codebook_loss = []
    epoch_reconstruction_loss = []
    sample = None

    for x_imaginary in tqdm(train_loader):
        optimizer.zero_grad()
        x = spectrogram_to_ml_representation(x_imaginary.to(device))
        xhat, codebook_loss, ind = compressor(x)
        recon_loss = loss_fn(x, xhat)
        loss = 100 * recon_loss + codebook_loss

        epoch_codebook_loss.append(codebook_loss.item())
        epoch_reconstruction_loss.append(recon_loss.item())
        epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        sample = ml_representation_to_audio(xhat)

    compressor.random_restart()
    embedding = compressor.quantizer.embedding.weight
    print(embedding)
    print(f"latent space perplexity: {compressor.quantizer.perplexity}")
    print(f"avg epoch loss: {sum(epoch_loss)/len(epoch_loss)}")
    print(f"avg recon loss loss: {sum(epoch_reconstruction_loss)/len(epoch_reconstruction_loss)}")
    print(f"avg codebook loss: {sum(epoch_codebook_loss)/len(epoch_codebook_loss)}")
    if epoch % 5 == 0 or epoch == N_EPOCHS-1:
        torchaudio.save(f"io/sample_epoch_{epoch}.wav", sample, sample_rate=44100)


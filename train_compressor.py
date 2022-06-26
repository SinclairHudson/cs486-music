import wandb
import torch
import torch.nn as nn
import torchaudio
from torch.optim import Adam
from dataset import LofiDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchaudio.transforms import InverseSpectrogram, Spectrogram
from compressor import Compressor
import pprint

pp = pprint.PrettyPrinter()


c = {
    "N_EPOCHS": 500,
    "BATCH_SIZE": 4,
    "LR": 0.005,
    "VOCAB_SIZE": 64,
    "BETA": 1,
    "SONG_LENGTH": 5,
    "CODEBOOK_LOSS_W": 1,
    "RECON_LOSS_W": 100,
}

wandb.init(project="lofi-compressor", config=c)

c = wandb.config

## normalize the loss weights to sum to 1, so that it's not implicitly learning rate
loss_w = c.CODEBOOK_LOSS_W + c.RECON_LOSS_W
codebook_loss_w = c.CODEBOOK_LOSS_W / loss_w
recon_loss_w = c.RECON_LOSS_W / loss_w

# n_fft controls the height of the spectrogram, how many frequency bins there are
s = Spectrogram(n_fft=800, return_complex=True, power=None)
inv = InverseSpectrogram(n_fft=800)

device = torch.device("cuda:0")
train_dataset = LofiDataset("/media/sinclair/datasets4/lofi/good_splits",
                            spectrogram=s,
                            length=c.SONG_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=c.BATCH_SIZE, shuffle=True)

compressor = Compressor(step_size=16, vocab_size=c.VOCAB_SIZE, beta=c.BETA).to(device)

optimizer = Adam(compressor.parameters(), lr=c.LR)


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


best_avg_recon_loss = 0.25

for epoch in range(c.N_EPOCHS):
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
        loss = recon_loss_w * recon_loss + codebook_loss_w *codebook_loss

        epoch_codebook_loss.append(codebook_loss.item())
        epoch_reconstruction_loss.append(recon_loss.item())
        epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        sample = ml_representation_to_audio(xhat)

    compressor.random_restart()
    embedding = compressor.quantizer.embedding.weight
    avg_recon_loss = sum(epoch_reconstruction_loss)/len(epoch_reconstruction_loss)

    print(embedding)
    summary = {
        "latent_space_perplexity": compressor.quantizer.perplexity,
        "avg_epoch_loss": sum(epoch_loss)/len(epoch_loss),
        "avg_codebook_loss": sum(epoch_codebook_loss)/len(epoch_codebook_loss),
        "avg_recon_loss": avg_recon_loss
        }
    pp.pprint(summary)
    wandb.log(summary)
    if avg_recon_loss < best_avg_recon_loss:
        # save the model
        best_avg_recon_loss = avg_recon_loss

    if epoch % 5 == 0 or epoch == c.N_EPOCHS-1:
        torchaudio.save(f"io/sample_epoch_{epoch}.wav", sample, sample_rate=44100)


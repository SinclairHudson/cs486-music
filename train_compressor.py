import wandb
import torch
import torchaudio
from torch.optim import Adam
from dataset import LofiDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchaudio.transforms import InverseSpectrogram, Spectrogram
from compressor import Compressor
import pprint
from utils import spectrogram_to_ml_representation, ml_representation_to_audio
from loss import SpectrogramLoss

pp = pprint.PrettyPrinter()


c = {
    "N_EPOCHS_0": 50,
    "N_EPOCHS_1": 200,
    "BATCH_SIZE": 8,
    "LR_0": 0.01,
    "LR_1": 0.005,
    "VOCAB_SIZE": 256,
    "BETA": 1,
    "SONG_LENGTH": 10,
    "CODEBOOK_LOSS_W": 1,
    "RECON_LOSS_W": 50,
    "UNDERLYING_0": "L2",
    "UNDERLYING_1": "L1",
    "N_FFT": 800,
}

wandb.init(project="lofi-compressor", config=c)

c = wandb.config

## normalize the loss weights to sum to 1, so that it's not implicitly learning rate
loss_w = c.CODEBOOK_LOSS_W + c.RECON_LOSS_W
codebook_loss_w = c.CODEBOOK_LOSS_W / loss_w
recon_loss_w = c.RECON_LOSS_W / loss_w

# n_fft controls the height of the spectrogram, how many frequency bins there are
s = Spectrogram(n_fft=c.N_FFT, return_complex=True, power=None)
inv = InverseSpectrogram(n_fft=c.N_FFT)

device = torch.device("cuda:0")
train_dataset = LofiDataset("/media/sinclair/datasets4/lofi/good_splits",
                            spectrogram=s,
                            length=c.SONG_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=c.BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, prefetch_factor=2)

compressor = Compressor(step_size=16, vocab_size=c.VOCAB_SIZE, beta=c.BETA).to(device)

start_point = None
if not start_point is None:
    compressor.load_state_dict(torch.load(start_point))

with torch.no_grad():
    x = spectrogram_to_ml_representation(next(iter(train_loader)).to(device))
    latent = compressor.encoder(x)
    density = latent.shape[-1] * latent.shape[-2] / c.SONG_LENGTH
    print(f"num latent codes per second: {density}")
    print(f"size of the latent space for a single clip: {latent.shape[-2:]}")

pp.pprint(c)


best_l2_recon_loss = 0.45

loss_fn = SpectrogramLoss(underlying=c.UNDERLYING_0)
optimizer = Adam(compressor.parameters(), lr=c.LR_0)

for epoch in range(c.N_EPOCHS_0):
    print("=" * 10 + f"starting epoch {epoch}." + "=" * 10)
    epoch_loss = []
    epoch_l2 = []
    epoch_codebook_loss = []
    epoch_reconstruction_loss = []
    sample = None
    sample_inds = None

    for x_imaginary in tqdm(train_loader):
        optimizer.zero_grad()
        x = spectrogram_to_ml_representation(x_imaginary.to(device))
        xhat, codebook_loss, ind = compressor(x)
        recon_loss = loss_fn(x, xhat)
        loss = recon_loss_w * recon_loss + codebook_loss_w *codebook_loss

        epoch_codebook_loss.append(codebook_loss.item())
        epoch_reconstruction_loss.append(recon_loss.item())
        epoch_loss.append(loss.item())
        epoch_l2.append(torch.mean((x - xhat) ** 2).item())
        loss.backward()
        optimizer.step()
        sample = ml_representation_to_audio(xhat)
        sample_inds = ind

    prop_restarted = compressor.random_restart()
    embedding = compressor.quantizer.embedding.weight
    avg_recon_loss = sum(epoch_reconstruction_loss)/len(epoch_reconstruction_loss)

    summary = {
        "latent_space_perplexity": compressor.quantizer.perplexity,
        "avg_epoch_loss": sum(epoch_loss)/len(epoch_loss),
        "avg_codebook_loss": sum(epoch_codebook_loss)/len(epoch_codebook_loss),
        "avg_recon_loss": avg_recon_loss,
        "avg_l2_loss": sum(epoch_l2)/len(epoch_l2),
        "proportion_restarted": prop_restarted,
        "indices": sample_inds
        }
    pp.pprint(summary)
    wandb.log(summary)

print("starting stage 2")
loss_fn = SpectrogramLoss(underlying=c.UNDERLYING_1)
optimizer = Adam(compressor.parameters(), lr=c.LR_1)


for epoch in range(c.N_EPOCHS_1):
    print("=" * 10 + f"starting epoch {epoch}." + "=" * 10)
    epoch_loss = []
    epoch_l2 = []
    epoch_codebook_loss = []
    epoch_reconstruction_loss = []
    sample = None
    sample_inds = None

    for x_imaginary in tqdm(train_loader):
        optimizer.zero_grad()
        x = spectrogram_to_ml_representation(x_imaginary.to(device))
        xhat, codebook_loss, ind = compressor(x)
        recon_loss = loss_fn(x, xhat)
        loss = recon_loss_w * recon_loss + codebook_loss_w *codebook_loss

        epoch_codebook_loss.append(codebook_loss.item())
        epoch_reconstruction_loss.append(recon_loss.item())
        epoch_loss.append(loss.item())
        epoch_l2.append(torch.mean((x - xhat) ** 2).item())
        loss.backward()
        optimizer.step()
        sample = ml_representation_to_audio(xhat)
        sample_inds = ind

    prop_restarted = compressor.random_restart()
    embedding = compressor.quantizer.embedding.weight
    avg_recon_loss = sum(epoch_reconstruction_loss)/len(epoch_reconstruction_loss)

    summary = {
        "latent_space_perplexity": compressor.quantizer.perplexity,
        "avg_epoch_loss": sum(epoch_loss)/len(epoch_loss),
        "avg_codebook_loss": sum(epoch_codebook_loss)/len(epoch_codebook_loss),
        "avg_recon_loss": avg_recon_loss,
        "avg_l2_loss": sum(epoch_l2)/len(epoch_l2),
        "proportion_restarted": prop_restarted,
        "indices": sample_inds
        }
    pp.pprint(summary)
    wandb.log(summary)
    if sum(epoch_l2)/len(epoch_l2) < best_l2_recon_loss:
        # save the model
        best_avg_recon_loss = avg_recon_loss
        torch.save(compressor.state_dict(), f"io/best_epoch_run_{wandb.run.name}.pth")
        torchaudio.save(f"io/best_epoch_{wandb.run.name}_clip.wav", sample, sample_rate=44100)



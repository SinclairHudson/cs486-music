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
from loss import SpectrogramLoss

pp = pprint.PrettyPrinter()


c = {
    "N_EPOCHS": 750,
    "BATCH_SIZE": 8,
    "LR": 0.01,
    "VOCAB_SIZE": 128,
    "BETA": 1,
    "SONG_LENGTH": 10,
    "CODEBOOK_LOSS_W": 1,
    "RECON_LOSS_W": 50,
    "UNDERLYING": "L2",
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

train_loader = DataLoader(train_dataset, batch_size=c.BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, prefetch_factor=2)

compressor = Compressor(step_size=16, vocab_size=c.VOCAB_SIZE, beta=c.BETA).to(device)

start_point = None
if not start_point is None:
    compressor.load_state_dict(torch.load(start_point))


optimizer = Adam(compressor.parameters(), lr=c.LR)

pp.pprint(c)

loss_fn = SpectrogramLoss(underlying=c.UNDERLYING)

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


best_l2_recon_loss = 0.45

for epoch in range(c.N_EPOCHS):
    print("=" * 10 + f"starting epoch {epoch}." + "=" * 10)
    epoch_loss = []
    epoch_l2 = []
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
        epoch_l2.append(torch.mean((x - xhat) ** 2).item())
        loss.backward()
        optimizer.step()
        sample = ml_representation_to_audio(xhat)

    prop_restarted = compressor.random_restart()
    embedding = compressor.quantizer.embedding.weight
    avg_recon_loss = sum(epoch_reconstruction_loss)/len(epoch_reconstruction_loss)

    summary = {
        "latent_space_perplexity": compressor.quantizer.perplexity,
        "avg_epoch_loss": sum(epoch_loss)/len(epoch_loss),
        "avg_codebook_loss": sum(epoch_codebook_loss)/len(epoch_codebook_loss),
        "avg_recon_loss": avg_recon_loss,
        "avg_l2_loss": sum(epoch_l2)/len(epoch_l2),
        "proportion_restarted": prop_restarted
        }
    pp.pprint(summary)
    wandb.log(summary)
    if sum(epoch_l2)/len(epoch_l2) < best_l2_recon_loss:
        # save the model
        best_avg_recon_loss = avg_recon_loss
        torch.save(compressor.state_dict(), f"io/best_epoch_run_{wandb.run.name}.pth")
        torchaudio.save(f"io/best_epoch_{wandb.run.name}_clip.wav", sample, sample_rate=44100)



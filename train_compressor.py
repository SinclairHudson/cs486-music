import wandb
import torch
from dataclasses import dataclass
import time
import torchaudio
from torch.optim import Adam
from dataset import LofiDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchaudio.transforms import InverseSpectrogram, Spectrogram
import pprint
from utils import spectrogram_to_ml_representation, ml_representation_to_audio, indices_to_rgb_image
from modules.loss import SpectrogramLoss
from modules.resnet_compressor import ResCompressor
from test import test

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

@dataclass
class Config:
    N_EPOCHS = 600
    BATCH_SIZE = 4
    LR = 0.01
    VOCAB_SIZE = 256
    BETA = 1
    SONG_LENGTH = 10  # in seconds
    CODEBOOK_LOSS_W = 1
    RECON_LOSS_W = 50
    UNDERLYING = "L2" # underlying loss
    N_FFT = 800  # resolution of the spectrogram
    TEST_EVERY_N_EPOCHS = 10
    BASS_BIAS = 0.2
    NUM_WORKERS = 3

def train(debug_flag=False):
    pp = pprint.PrettyPrinter()

    c = Config()
    if not debug_flag:
        wandb.init(project="lofi-compressor", config=c)


## normalize the loss weights to sum to 1, so that it's not implicitly learning rate
    loss_w = c.CODEBOOK_LOSS_W + c.RECON_LOSS_W
    codebook_loss_w = c.CODEBOOK_LOSS_W / loss_w
    recon_loss_w = c.RECON_LOSS_W / loss_w

# n_fft controls the height of the spectrogram, how many frequency bins there are
    s = Spectrogram(n_fft=c.N_FFT, return_complex=True, power=None)

    device = torch.device("cuda:0")
    train_dataset = LofiDataset("/media/sinclair/datasets/lofi/train_splits",
                                spectrogram=s,
                                length=c.SONG_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=c.BATCH_SIZE, shuffle=True,
                              num_workers=c.NUM_WORKERS, pin_memory=False, prefetch_factor=2)

    test_dataset = LofiDataset("/media/sinclair/datasets/lofi/test_splits",
                                spectrogram=s,
                                length=c.SONG_LENGTH)

    test_loader = DataLoader(test_dataset, batch_size=c.BATCH_SIZE, shuffle=False,
                              num_workers=c.NUM_WORKERS, pin_memory=False, prefetch_factor=2)

    compressor = ResCompressor(step_size=16, vocab_size=c.VOCAB_SIZE, beta=c.BETA).to(device)

# start_point = "io/best_epoch_run_glad-microwave-77.pth"
    start_point=None
    if not start_point is None:
        compressor.load_state_dict(torch.load(start_point))
        print(f"starting from {start_point}.")

    with torch.no_grad():
        x = spectrogram_to_ml_representation(next(iter(train_loader)).to(device))
        latent = compressor.encoder(x)
        density = latent.shape[-1] * latent.shape[-2] / c.SONG_LENGTH
        print(f"num latent codes per second: {density}")
        print(f"size of the latent space for a single clip: {latent.shape[-2:]}")
        wandb.config.update({"density": density,
                             "latent_shape": latent.shape[-2:],
                             "start_point": start_point})

    pp.pprint(c.__dict__)

    best_l2_recon_loss = 0.45

    loss_fn = SpectrogramLoss(bass_bias=c.BASS_BIAS, underlying=c.UNDERLYING)
    optimizer = Adam(compressor.parameters(), lr=c.LR)

    for epoch in range(c.N_EPOCHS):
        print("=" * 10 + f"starting epoch {epoch}" + "=" * 10)
        epoch_loss = []
        epoch_l2 = []
        epoch_codebook_loss = []
        epoch_reconstruction_loss = []
        sample = None
        sample_inds = None
        true_sample = None

        epoch_start_time = time.time()
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
            true_sample = ml_representation_to_audio(x)
            sample_inds = ind

        prop_restarted = compressor.random_restart()
        embedding = compressor.quantizer.embedding.weight
        avg_recon_loss = sum(epoch_reconstruction_loss)/len(epoch_reconstruction_loss)

        summary = {
            "latent_space_perplexity": compressor.quantizer.perplexity,
            "epoch_time": time.time() - epoch_start_time,
            "avg_epoch_loss": sum(epoch_loss)/len(epoch_loss),
            "avg_codebook_loss": sum(epoch_codebook_loss)/len(epoch_codebook_loss),
            "avg_recon_loss": avg_recon_loss,
            "avg_l2_loss": sum(epoch_l2)/len(epoch_l2),
            "proportion_restarted": prop_restarted,
            "indices": sample_inds,
            "latent_code_viz": wandb.Image(indices_to_rgb_image(sample_inds, c.VOCAB_SIZE))
            }

        if epoch % c.TEST_EVERY_N_EPOCHS == 0:
            test_summary = test(compressor, test_loader, loss_fn, recon_loss_w, codebook_loss_w)
            summary.update(test_summary)

        if sum(epoch_l2)/len(epoch_l2) < best_l2_recon_loss:
            # save the model
            best_avg_recon_loss = avg_recon_loss
            torch.save(compressor.state_dict(), f"io/best_epoch_run_{wandb.run.name}.pth")
            torchaudio.save(f"io/best_epoch_{wandb.run.name}_clip.wav", torch.cat((true_sample, sample), dim=1), sample_rate=44100)

        wandb.log(summary)

if __name__ == "__main__":
    train()

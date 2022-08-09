import os
import torchaudio
import torch
import torch.nn as nn
from resnet_compressor import ResCompressor
from torch.utils.data import DataLoader
from utils import spectrogram_to_ml_representation, ml_representation_to_audio
from dataset import LofiDataset
from torchaudio.transforms import InverseSpectrogram, Spectrogram
from generator import TransformerModel
from tqdm import tqdm

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


timesteps = 4000

c = {
    "N_EPOCHS": 300,
    "BATCH_SIZE": 512,
    "LR": 0.01,
    "VOCAB_SIZE": 512,
    "N_FFT": 800,
    "N_HEAD": 8,
    "D_HID": 400,
    "EMBED_SIZE": 400,
    "N_LAYERS": 4,
    "DROPOUT": 0.2,
    "TEST_EVERY_N_EPOCHS": 10,
    "RANDOM_STATE": 10,
    "GAMMA": 0.95,
    "LR_STEP_SIZE": 5,
    "RECEPTIVE_FIELD": 2300,
}

HEIGHT = 23

assert RECEPTIVE_FIELD % HEIGHT == 0  # ensures that we never have partial time steps

stub = torch.load(f"best_epoch_run_azure-yogurt-87_sequences/song1_indices.pt")
stub = stub.reshape(-1)[-c.RECEPTIVE_FIELD:] # last 


src_mask = generate_square_subsequent_mask(c.RECEPTIVE_FIELD).to(device)

model = TransformerModel(c.VOCAB_SIZE, c.EMBED_SIZE, c.N_HEAD, c.D_HID, c.N_LAYERS, c.DROPOUT).to(device)

accumulated_music = []

for step in range(timesteps):
    for _ in range(HEIGHT):
        inp = stub.unsqueeze(1)  # add batch dimension
        inp = inp.to(device)
        output = model(inp, src_mask).squeeze(1)
        # sample from the distribution:
        next_token = torch.argmax(output)
        stub = torch.cat((stub, next_token), dim=0)
        accumulated_music.append(stub[1:])
        stub = stub[1:]  # kick the oldest element


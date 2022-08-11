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
import torch.nn.functional as F

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


timesteps = 1000

c = {
    "N_EPOCHS": 300,
    "BATCH_SIZE": 512,
    "LR": 0.01,
    "VOCAB_SIZE": 512,
    "N_FFT": 800,
    "N_HEAD": 4,
    "D_HID": 200,
    "EMBED_SIZE": 200,
    "N_LAYERS": 3,
    "DROPOUT": 0.2,
    "TEST_EVERY_N_EPOCHS": 10,
    "RANDOM_STATE": 10,
    "GAMMA": 0.95,
    "LR_STEP_SIZE": 100,
    "RECEPTIVE_FIELD": 2300,
}


HEIGHT = 23

assert c["RECEPTIVE_FIELD"]% HEIGHT == 0  # ensures that we never have partial time steps

stub = torch.load(f"best_epoch_run_azure-yogurt-87_sequences/song9_indices.pt")
stub = stub.reshape(-1)[-c["RECEPTIVE_FIELD"]:] # last 


src_mask = generate_square_subsequent_mask(c["RECEPTIVE_FIELD"]).to(device)

model = TransformerModel(c["VOCAB_SIZE"], c["EMBED_SIZE"], c["N_HEAD"], c["D_HID"], c["N_LAYERS"], c["DROPOUT"]).to(device)
model.load_state_dict(torch.load("io/rich-meadow-15_best_val.pth"))

accumulated_music = []

def max_sample(logits):
    return torch.argmax(logits)

def softmax_sample(logits):
    dist = torch.distributions.Categorical(F.softmax(logits, dim=0))
    return dist.sample()

def softmax_sample_with_temp(logits, temp=1):
    dist = torch.distributions.Categorical(F.softmax(logits/temp, dim=0))
    return dist.sample()

for step in tqdm(range(timesteps)):
    for _ in range(HEIGHT):
        inp = stub.unsqueeze(1)  # add batch dimension
        inp = inp.to(device)
        output = model(inp, src_mask).squeeze(1)
        # sample from the distribution:
        next_token = softmax_sample_with_temp(output[-1], temp=0.5)
        stub = torch.cat((stub, next_token.unsqueeze(0)), dim=0)
        accumulated_music.append(stub[:1])
        stub = stub[1:]  # kick the oldest element

generated_sequence = torch.cat(accumulated_music, dim=0)
latents = generated_sequence.reshape((HEIGHT, -1))
torch.save(latents.unsqueeze(0), "generated_music.pt")



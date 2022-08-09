import os
import torchaudio
import torch
import torch.nn as nn
from resnet_compressor import ResCompressor
from torch.utils.data import DataLoader
from utils import spectrogram_to_ml_representation, ml_representation_to_audio
from dataset import LofiDataset
from torchaudio.transforms import InverseSpectrogram, Spectrogram
from tqdm import tqdm

device = torch.device("cuda:0")
# load model

compressor = ResCompressor(step_size=16, vocab_size=512, beta=1).to(device)
start_point = "io/best_epoch_run_azure-yogurt-87.pth"
compressor.load_state_dict(torch.load(start_point))
compressor.eval()

def restore(filename: str):
        with torch.no_grad():
                latent = torch.load(f"best_epoch_run_azure-yogurt-87_sequences/{filename}")
                output_ml = compressor.decoder(compressor.quantizer.dequantize(latent))
                audio = ml_representation_to_audio(output_ml)
                torchaudio.save(f"io/{filename}.wav", audio, sample_rate=44100)

if __name__ == "__main__":
        restore("song1_indices.pt")




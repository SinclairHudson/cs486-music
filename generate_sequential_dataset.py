import os
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

compressor = ResCompressor(step_size=16, vocab_size=256, beta=1).to(device)
start_point = "io/best_epoch_run_spring-silence-58.pth"
compressor.load_state_dict(torch.load(start_point))
compressor.eval()

s = Spectrogram(n_fft=800, return_complex=True, power=None)

train_dataset = LofiDataset("/media/sinclair/datasets4/lofi/good_splits",
                            spectrogram=s,
                            length=45)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                          num_workers=4, pin_memory=True, prefetch_factor=2)

os.mkdir(f"{start_point[3:-4]}_sequences")

with torch.no_grad():
    for i, x_imaginary in tqdm(enumerate(train_loader)):
        x = spectrogram_to_ml_representation(x_imaginary).to(device)

        _, _, indices = compressor.quantizer(compressor.encoder(x), return_indices=True)
        torch.save(indices, f"{start_point[3:-4]}_sequences/song{i}_indices.pt")


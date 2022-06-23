import torch
from dataset import LofiDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchaudio.transforms import InverseSpectrogram, Spectrogram


N_EPOCHS = 100
BATCH_SIZE = 4

spectrogram_params = {
    "n_fft": 800,
    "normalized": True,
}

s = Spectrogram(**spectrogram_params)
inv = InverseSpectrogram(**spectrogram_params)

device = torch.device("cuda:0")
train_dataset = LofiDataset("/media/sinclair/datasets4/lofi/good_splits",
                            spectrogram=s,
                            length=15)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(N_EPOCHS):
    for x in tqdm(train_loader):
        x.to(device)
        breakpoint()

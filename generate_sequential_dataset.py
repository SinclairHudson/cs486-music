import torch
import torch.nn as nn
from resnet_compressor import ResCompressor

device = torch.device("cuda:0")
# load model

model = ResCompressor()

train_dataset = LofiDataset("/media/sinclair/datasets4/lofi/good_splits",
                            spectrogram=s,
                            length=c.SONG_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=c.BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, prefetch_factor=2)

with torch.no_grad():
    x = spectrogram_to_ml_representation(x_imaginary).to(device))
    latent = compressor.encoder(x)
    density = latent.shape[-1] * latent.shape[-2] / c.SONG_LENGTH
    print(f"num latent codes per second: {density}")
    print(f"size of the latent space for a single clip: {latent.shape[-2:]}")

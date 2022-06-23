import torch
from dataset import LofiDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


N_EPOCHS = 100

train_dataset = LofiDataset("/media/sinclair/datasets4/lofi/good_splits")

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
for epoch in range(N_EPOCHS):
    for x in tqdm(train_loader):
        print(wefe)

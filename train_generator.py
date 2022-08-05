import wandb
import torch.nn as nn
from generator import TransformerModel


c = {
    "N_EPOCHS": 200,
    "BATCH_SIZE": 16,
    "LR": 0.01,
    "VOCAB_SIZE": 256,
    "N_FFT": 800,
    "N_HEAD": 2,
    "D_HID": 200,
    "EMBED_SIZE": 200,
    "N_LAYERS": 3,
    "DROPOUT": 0.2,
    "TEST_EVERY_N_EPOCHS": 10,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.init(project="lofi-generator", config=c)

c = wandb.config

train_dataset = GeneratorDataset(encoder, clip_dataset)

model = TransformerModel(c.VOCAB_SIZE, c.EMBED_SIZE, c.N_HEAD, c.D_HID, c.N_LAYERS, c.DROPOUT).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# the transformer is going to take in latent_dim x N sequences, and predicts 
# the next along the vocabulary. That's going to be softmax'd values over the 
# vocabulary

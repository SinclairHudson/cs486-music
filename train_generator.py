
import torch.nn as nn

generator = nn.Transformer(batch_first=True)

# load the trained encoder

encoder = 

clip_dataset = 

train_dataset = GeneratorDataset(encoder, clip_dataset)


import torch.nn as nn

generator = nn.Transformer(batch_first=True)

# load the trained encoder

encoder = 

clip_dataset = 

train_dataset = GeneratorDataset(encoder, clip_dataset)

# the transformer is going to take in latent_dim x N sequences, and predicts 
# the next along the vocabulary. That's going to be softmax'd values over the 
# vocabulary

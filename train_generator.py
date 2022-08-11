import wandb
import math
import torch.nn as nn
import torch
from dataset import GeneratorDataset
from generator import TransformerModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm


c = {
    "N_EPOCHS": 500,
    "BATCH_SIZE": 512,
    "LR": 0.01,
    "VOCAB_SIZE": 512,
    "N_FFT": 800,
    "N_HEAD": 4,
    "D_HID": 200,
    "EMBED_SIZE": 200,
    "N_LAYERS": 4,
    "DROPOUT": 0.2,
    "TEST_EVERY_N_EPOCHS": 10,
    "RANDOM_STATE": 10,
    "GAMMA": 0.95,
    "LR_STEP_SIZE": 100,
    "RECEPTIVE_FIELD": 4600,
}

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.init(project="lofi-generator", config=c)

c = wandb.config

train_dataset = GeneratorDataset("best_epoch_run_azure-yogurt-87_sequences",
                                 batch_size=c.BATCH_SIZE, bptt=c.RECEPTIVE_FIELD)

train, test = train_test_split(train_dataset, test_size=0.3,
                               random_state=c.RANDOM_STATE,
                               shuffle=False)  # don't shuffle so that songs stay independent

criterion = nn.CrossEntropyLoss()
model = TransformerModel(c.VOCAB_SIZE, c.EMBED_SIZE, c.N_HEAD, c.D_HID, c.N_LAYERS, c.DROPOUT).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=c.LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, c.LR_STEP_SIZE, gamma=c.GAMMA)

# the transformer is going to take in latent_dim x N sequences, and predicts 
# the next along the vocabulary. That's going to be softmax'd values over the 
# vocabulary

best_test_loss = float('inf')

src_mask = generate_square_subsequent_mask(c.RECEPTIVE_FIELD).to(device)

def evaluate(model):
    print("evaluating")
    model.eval()
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(c.RECEPTIVE_FIELD).to(device)
    with torch.no_grad():
        for batch in tqdm(test):
            data, targets = batch
            batch_length = data.size(0)
            if batch_length != c.RECEPTIVE_FIELD:  # only on last batch
                mask = src_mask[:batch_length, :batch_length]
            else:
                mask = src_mask

            data = data.to(device)
            mask = mask.to(device)
            output = model(data, mask)
            output_flat = output.view(-1, c.VOCAB_SIZE)
            total_loss += criterion(output_flat, targets).item()

    val_loss = total_loss / len(test)
    val_summary = {"val_avg_CE_loss": val_loss,
                   "val_ppl": math.exp(val_loss)
    }
    return val_summary


print("beginning training")
best_val_loss = 3.5
for e in range(c.N_EPOCHS):
    model.train()
    cross_entropy_loss = []
    for batch in tqdm(train):
        data, targets = batch
        batch_length = data.size(0)
        if batch_length != c.RECEPTIVE_FIELD:  # only on last batch
            mask = src_mask[:batch_length, :batch_length]
        else:
            mask = src_mask

        data = data.to(device)
        mask = mask.to(device)
        output = model(data, mask)
        loss = criterion(output.view(-1, c.VOCAB_SIZE), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        cross_entropy_loss.append(loss.item())

    summary = {"learning_rate": scheduler.get_last_lr()[0],
               "mean_CE_loss": sum(cross_entropy_loss)/len(cross_entropy_loss),
               }

    if e % c.TEST_EVERY_N_EPOCHS == 0 and e != 0:
        # validation
        val_loss = evaluate(model)
        summary.update(val_loss)
        if val_loss["val_avg_CE_loss"] < best_val_loss:
            print("saving the model")
            torch.save(model.state_dict(), f"io/{wandb.run.name}_best_val.pth")
            best_val_loss = val_loss["val_avg_CE_loss"]

    scheduler.step()
    print(summary)
    wandb.log(summary)


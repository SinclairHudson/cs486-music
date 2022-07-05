import torch
from tqdm import tqdm
from utils import spectrogram_to_ml_representation, ml_representation_to_audio

device = torch.device("cuda:0")

def test(compressor, test_loader, loss_fn, recon_loss_w, codebook_loss_w):
    print("testing")
    epoch_loss = []
    epoch_l2 = []
    epoch_codebook_loss = []
    epoch_reconstruction_loss = []
    with torch.no_grad():
        for x_imaginary in tqdm(test_loader):
            x = spectrogram_to_ml_representation(x_imaginary.to(device))
            xhat, codebook_loss, ind = compressor(x)
            recon_loss = loss_fn(x, xhat)
            loss = recon_loss_w * recon_loss + codebook_loss_w * codebook_loss

            epoch_codebook_loss.append(codebook_loss.item())
            epoch_reconstruction_loss.append(recon_loss.item())
            epoch_loss.append(loss.item())
            epoch_l2.append(torch.mean((x - xhat) ** 2).item())

    avg_recon_loss = sum(epoch_reconstruction_loss)/len(epoch_reconstruction_loss)
    summary = {
        "test_avg_epoch_loss": sum(epoch_loss)/len(epoch_loss),
        "test_avg_codebook_loss": sum(epoch_codebook_loss)/len(epoch_codebook_loss),
        "test_avg_recon_loss": avg_recon_loss,
        "test_avg_l2_loss": sum(epoch_l2)/len(epoch_l2),
        }
    return summary

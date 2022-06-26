import torch
import torch.nn as nn

class SpectrogramLoss(nn.Module):
    def __init__(self, underlying="L2"):
        super(SpectrogramLoss, self).__init__()
        self.underlying = underlying

    def forward(self, yhat, y_true):
        B, C, H, W = yhat.size()
        aranged = 1 + (torch.arange(start=0, end=H) / H)
        aranged = aranged.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        aranged = aranged.to(yhat.device)
        weights = aranged.expand(B, C, H, W)

        if self.underlying == "L2":
            unweighted_loss = (yhat - y_true) ** 2

        elif self.underlying == "L1":
            unweighted_loss = torch.abs(yhat - y_true)
        else:
            raise ValueError("underlying loss must be L1 or L2.")

        return torch.mean(unweighted_loss * weights)  # elementwise


class SpectrogramMagnitudeWeightedLoss(nn.Module):
    def __init__(self, underlying="L2"):
        super(SpectrogramMagnitudeWeightedLoss, self).__init__()
        self.underlying = underlying

    def forward(self, yhat, y_true):
        if self.underlying == "L2":
            unweighted_loss = (yhat - y_true) ** 2

        elif self.underlying == "L1":
            unweighted_loss = torch.abs(yhat - y_true)
        else:
            raise ValueError("underlying loss must be L1 or L2.")

        return torch.mean(unweighted_loss * y_true)  # elementwise weight by magnitude


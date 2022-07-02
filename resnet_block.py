import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels)
        )

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.r1 = nn.ReLU()
        self.r2 = nn.ReLU()
        self.r3 = nn.ReLU()


    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.r1(self.bn1(self.conv1(x)))
        x = self.r2(self.bn2(self.conv2(x)))
        x = x + shortcut
        return self.r3(x)

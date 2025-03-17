import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels=3, num_filters=64):
        super(PatchDiscriminator,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_filters, num_filters*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_filters*2, num_filters*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_filters*4, num_filters*8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(num_filters*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(num_filters*8, 1, kernel_size=4, stride=1, padding=1)  # Patch size 4x4
        )

    def forward(self, x):
        return self.model(x)  # Output: [B, 1, H, W]

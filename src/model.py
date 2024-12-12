import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class SimpleMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Initial feature extraction - no padding
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=0, bias=False),  # 28->26, 3x3
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        # Deep feature extraction - no padding
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=3, padding=0, bias=False),  # 26->24, 5x5
            nn.BatchNorm2d(24),
            nn.ReLU(),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(24,24, kernel_size=3, padding=1, bias=False),  # 24->24, 7x7
            nn.BatchNorm2d(24),
            nn.ReLU(),
        )

        self.maxpool = nn.MaxPool2d(2, 2)  # 24->12 , 8x8
        self.block4 = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=3, padding=1, bias=False),  # 12 > 10, 12x12
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )



        # Feature refinement with 3x3 convs
        self.block5 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False), # 10->8 , 14x14
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.block6 = nn.Sequential(
            nn.Conv2d(16, 24, kernel_size=3, padding=1, bias=False),# 8->6 , 16x16
            nn.BatchNorm2d(24),
            nn.ReLU(),
        )

        self.block7 = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=3, padding=0, bias=False), # 6 > 4 , 18x18
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.block8 = nn.Sequential(
            nn.Conv2d(16, 10, kernel_size=3, padding=0, bias=False), # 6 > 4 , 20x20
            nn.BatchNorm2d(10),
            nn.ReLU(),
        )

        self.gap = nn.AvgPool2d(2)

        self.final_conv = nn.Conv2d(10, 10, kernel_size=4) # 4 > 1



    def forward(self, x):
        # Initial features without padding
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.maxpool(x)

        # Feature refinement
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)

        x = self.gap(x)
        x = self.final_conv(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=-1)

    def count_parameters(self) -> int:
        # Return the total number of parameters as an integer
        return sum(p.numel() for p in self.parameters())

    def print_model_summary(self):
        params = self.count_parameters()

        print("\nModel Parameter Summary:")
        print("-" * 40)
        print(f"Total Parameters: {params:,}")
        print("-" * 40)

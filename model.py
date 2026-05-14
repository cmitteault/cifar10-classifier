import torch
import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Bloc convolutif 1 : 3 → 32 canaux
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # (B, 32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)                              # (B, 32, 16, 16)
        )

        # Bloc convolutif 2 : 32 → 64 canaux
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (B, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)                              # (B, 64, 8, 8)
        )

        # Bloc convolutif 3 : 64 → 128 canaux
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # (B, 128, 8, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)                               # (B, 128, 4, 4)
        )

        # Tête de classification
        self.classifier = nn.Sequential(
            nn.Flatten(),           # (B, 128*4*4) = (B, 2048)
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)
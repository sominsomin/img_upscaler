import torch.nn as nn

class ImageUpscalerCNN(nn.Module):
    def __init__(self, SCALING_FACTOR=2):
        super(ImageUpscalerCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=SCALING_FACTOR, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.model(x)

import torch
import torch.nn as nn


class AutoencoderBaseline(nn.Module):
    def __init__(self):
        super(AutoencoderBaseline, self).__init__()        

        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 6, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 1, 1), 
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, img):
        encoded = self.encoder(img)
        decoded = self.decoder(encoded)
        return decoded
import torch, torchvision
import torch.nn as nn


class AutoencoderBackbone(nn.Module):
    def __init__(self):
        super().__init__()        
        # N(BATCHSIZE), 3, 256, 512
        #[(W-K+2P)/S]+1
        resnet_net = torchvision.models.resnet50(pretrained=True)
        resnet_net.conv1=nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        delattr(resnet_net, "maxpool")
        modules = list(resnet_net.children())[:-2]
        backbone = nn.Sequential(*modules)

        self.encoder = backbone
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 512, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, output_padding=1), 
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
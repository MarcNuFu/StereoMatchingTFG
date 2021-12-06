import torch, torchvision
import torch.nn as nn


def get_conv2d(in_channels, out_channels, kernel_param, stride_param=1, padding_param=0, output_padding_param=0):
    return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_param, stride=stride_param, padding=padding_param, output_padding=output_padding_param), 
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True))
                        
                        
class AutoencoderBackbone(nn.Module):
    def __init__(self):
        super(AutoencoderBackbone, self).__init__()        
        
        #Get modified resnet as backbone encoder
        resnet_net = torchvision.models.resnet50(pretrained=True)
        resnet_net.conv1=nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        delattr(resnet_net, "maxpool")
        modules = list(resnet_net.children())[:-2]
        backbone = nn.Sequential(*modules)
        self.encoder = backbone
        
        #Decoder
        self.decoder = nn.Sequential(
            get_conv2d(in_channels=2048, out_channels=1024, kernel_param=3, stride_param=2, padding_param=1, output_padding_param=1),
           
            get_conv2d(in_channels=1024, out_channels=512, kernel_param=3, stride_param=2, padding_param=1, output_padding_param=1),

            get_conv2d(in_channels=512, out_channels=256, kernel_param=3, stride_param=2, padding_param=1, output_padding_param=1),

            get_conv2d(in_channels=256, out_channels=128, kernel_param=3, stride_param=2, padding_param=1, output_padding_param=1),
  
            get_conv2d(in_channels=128, out_channels=64, kernel_param=1),

            get_conv2d(in_channels=64, out_channels=32, kernel_param=1),

            get_conv2d(in_channels=32, out_channels=16, kernel_param=1),

            get_conv2d(in_channels=16, out_channels=8, kernel_param=1),

            get_conv2d(in_channels=8, out_channels=4, kernel_param=1),

            get_conv2d(in_channels=4, out_channels=2, kernel_param=1),

            nn.ConvTranspose2d(2, 1, 1), 
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, imgL, imgR):
        img = torch.cat([imgL, imgR], dim=1)
        encoded = self.encoder(img)
        decoded = self.decoder(encoded)
        pred = torch.squeeze(decoded,1)
        return pred
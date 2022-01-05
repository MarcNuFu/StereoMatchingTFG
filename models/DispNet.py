import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
import torch.nn.functional as F

def conv(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(
                          nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
                          nn.BatchNorm2d(num_features=out_channels),
                          nn.LeakyReLU(negative_slope=0.1, inplace=True)
                         )


def predict_flow(in_channels):
    return nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)


def upconv(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(
                          nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
                          nn.BatchNorm2d(num_features=out_channels),
                          nn.LeakyReLU(negative_slope=0.1, inplace=True)
                         )
    
    
class DispNet(nn.Module):
    def __init__(self):
        super(DispNet,self).__init__()

        self.conv1  = conv(in_channels=6,    out_channels=64,   kernel_size=7, stride=2)
        self.conv2  = conv(in_channels=64,   out_channels=128,  kernel_size=5, stride=2)
        self.conv3a = conv(in_channels=128,  out_channels=256,  kernel_size=5, stride=2)
        self.conv3b = conv(in_channels=256,  out_channels=256,  kernel_size=3, stride=1)
        self.conv4a = conv(in_channels=256,  out_channels=512,  kernel_size=3, stride=2)
        self.conv4b = conv(in_channels=512,  out_channels=512,  kernel_size=3, stride=1)
        self.conv5a = conv(in_channels=512,  out_channels=512,  kernel_size=3, stride=2)
        self.conv5b = conv(in_channels=512,  out_channels=512,  kernel_size=3, stride=1)
        self.conv6a = conv(in_channels=512,  out_channels=1024, kernel_size=3, stride=2)
        self.conv6b = conv(in_channels=1024, out_channels=1024, kernel_size=3, stride=1)

        self.upconv5 = upconv(in_channels=1024, out_channels=512,  kernel_size=4, stride=2)
        self.upconv4 = upconv(in_channels=512,  out_channels=256,  kernel_size=4, stride=2)
        self.upconv3 = upconv(in_channels=256,  out_channels=128,  kernel_size=4, stride=2)
        self.upconv2 = upconv(in_channels=128,  out_channels=64,   kernel_size=4, stride=2)
        self.upconv1 = upconv(in_channels=64,   out_channels=32,   kernel_size=4, stride=2)

        self.pr6 = predict_flow(in_channels=1024)
        self.pr5 = predict_flow(in_channels=512)
        self.pr4 = predict_flow(in_channels=256)
        self.pr3 = predict_flow(in_channels=128)
        self.pr2 = predict_flow(in_channels=64)
        self.pr1 = predict_flow(in_channels=32)
        
        self.iconv5 = upconv(in_channels=1025, out_channels=512, kernel_size=3, stride=1) 
        self.iconv4 = upconv(in_channels=769,  out_channels=256, kernel_size=3, stride=1) 
        self.iconv3 = upconv(in_channels=385,  out_channels=128, kernel_size=3, stride=1)  
        self.iconv2 = upconv(in_channels=193,  out_channels=64,  kernel_size=3, stride=1)  
        self.iconv1 = upconv(in_channels=97,   out_channels=32,  kernel_size=3, stride=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')


    def forward(self, imgL, imgR):
        images = torch.cat([imgL, imgR], dim=1)

        conv1   = self.conv1(images)
        conv2   = self.conv2(conv1)
        
        conv3a  = self.conv3a(conv2)
        conv3b  = self.conv3b(conv3a)
        conv4a  = self.conv4a(conv3b)
        conv4b  = self.conv4b(conv4a)
        conv5a  = self.conv5a(conv4b)
        conv5b  = self.conv5b(conv5a)
        conv6a  = self.conv6a(conv5b)
        conv6b  = self.conv6b(conv6a)

        pr6     = self.pr6(conv6b)
        pr6_up  = self.upsample(pr6)
                
        upconv5 = self.upconv5(conv6b)
        iconv5  = self.iconv5(torch.cat([upconv5, pr6_up, conv5b], dim=1))
        pr5     = self.pr5(iconv5)
        pr5_up  = self.upsample(pr5)
        
        upconv4 = self.upconv4(iconv5)
        iconv4  = self.iconv4(torch.cat([upconv4, pr5_up, conv4b], dim=1))
        pr4     = self.pr4(iconv4)
        pr4_up  = self.upsample(pr4)

        upconv3 = self.upconv3(iconv4)
        iconv3  = self.iconv3(torch.cat([upconv3, pr4_up, conv3b], dim=1))
        pr3     = self.pr3(iconv3)
        pr3_up  = self.upsample(pr3)

        upconv2 = self.upconv2(iconv3)
        iconv2  = self.iconv2(torch.cat([upconv2, pr3_up, conv2], dim=1))
        pr2     = self.pr2(iconv2)
        pr2_up  = self.upsample(pr2)

        upconv1 = self.upconv1(iconv2)
        iconv1  = self.iconv1(torch.cat([upconv1, pr2_up, conv1], dim=1))
        pr1     = self.pr1(iconv1)
        pr1_up  = self.upsample(pr1)


        if self.training:
            return pr1_up, (pr1, pr2, pr3, pr4, pr5, pr6)
        else:
            return pr1_up, pr1
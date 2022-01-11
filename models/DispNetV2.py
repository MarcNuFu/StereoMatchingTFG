import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
import torch.nn.functional as F
from models.ResBlock import ResBlock


def conv(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(
                          nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
                          nn.LeakyReLU(negative_slope=0.1, inplace=True)
                         )


def predict_flow(in_channels):
    return nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)


def iconv(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True)


def upconv(in_channels, out_channels):
    return nn.Sequential(
                          nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                          nn.LeakyReLU(negative_slope=0.1, inplace=True)
                         )
    
    
class DispNetV2(nn.Module):
    def __init__(self):
        super(DispNetV2,self).__init__()

        self.conv1  = conv(in_channels=6, out_channels=64, kernel_size=7, stride=2)
        
        self.conv2  = ResBlock(in_channels=64,   out_channels=128,  stride=2)
        self.conv3a = ResBlock(in_channels=128,  out_channels=256,  stride=2)
        self.conv3b = ResBlock(in_channels=256,  out_channels=256,  stride=1)
        self.conv4a = ResBlock(in_channels=256,  out_channels=512,  stride=2)
        self.conv4b = ResBlock(in_channels=512,  out_channels=512,  stride=1)
        self.conv5a = ResBlock(in_channels=512,  out_channels=512,  stride=2)
        self.conv5b = ResBlock(in_channels=512,  out_channels=512,  stride=1)
        self.conv6a = ResBlock(in_channels=512,  out_channels=1024, stride=2)
        self.conv6b = ResBlock(in_channels=1024, out_channels=1024, stride=1)
        

        self.upconv5 = upconv(in_channels=1024, out_channels=512)
        self.upconv4 = upconv(in_channels=512,  out_channels=256)
        self.upconv3 = upconv(in_channels=256,  out_channels=128)
        self.upconv2 = upconv(in_channels=128,  out_channels=64)
        self.upconv1 = upconv(in_channels=64,   out_channels=32)
        self.upconv0 = upconv(in_channels=32,   out_channels=16)
        

        self.pr6 = predict_flow(in_channels=1024)
        self.pr5 = predict_flow(in_channels=512)
        self.pr4 = predict_flow(in_channels=256)
        self.pr3 = predict_flow(in_channels=128)
        self.pr2 = predict_flow(in_channels=64)
        self.pr1 = predict_flow(in_channels=32)
        self.pr0 = predict_flow(in_channels=16)
        
        
        self.iconv5 = iconv(in_channels=1025, out_channels=512) 
        self.iconv4 = iconv(in_channels=769,  out_channels=256) 
        self.iconv3 = iconv(in_channels=385,  out_channels=128)  
        self.iconv2 = iconv(in_channels=193,  out_channels=64)  
        self.iconv1 = iconv(in_channels=97,   out_channels=32)
        self.iconv0 = iconv(in_channels=20,   out_channels=16)
        

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
      
      
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
                
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
        
        upconv0 = self.upconv0(iconv1)
        iconv0  = self.iconv0(torch.cat([upconv0, pr1_up, imgL], dim=1))
        pr0     = self.pr0(iconv0)


        if self.training:
            return pr0, (pr0, pr1, pr2, pr3, pr4, pr5, pr6)
        else:
            return pr0
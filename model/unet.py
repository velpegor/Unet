import torch
import torch.nn as nn
from torchsummary import summary
from .ops import *

class UNet(nn.Module):
    def __init__(self, num_of_cls=81, use_upconv=True, tau=-1.0):
        super(UNet, self).__init__()
        
        self.tau=nn.Parameter(torch.tensor([tau]))
        self.input_channel = 3 
        self.channel = 64
        self.num_of_cls = num_of_cls

        self.inconv = RepeatedConv2d(self.input_channel, self.channel, tau=self.tau)
        
        self.down1 = Down(self.channel, self.channel*2, tau=self.tau)
        self.down2 = Down(self.channel*2, self.channel*4, tau=self.tau)
        self.down3 = Down(self.channel*4, self.channel*8, tau=self.tau)
        self.down4 = Down(self.channel*8, self.channel*16, tau=self.tau)

        self.up1 = Up(self.channel*16, self.channel*8, use_upconv=use_upconv, tau=self.tau)
        self.up2 = Up(self.channel*8, self.channel*4, use_upconv=use_upconv, tau=self.tau)
        self.up3 = Up(self.channel*4, self.channel*2, use_upconv=use_upconv, tau=self.tau)
        self.up4 = Up(self.channel*2, self.channel, use_upconv=use_upconv, tau=self.tau)

        self.outconv = nn.Conv2d(self.channel, self.num_of_cls, kernel_size=(1,1))

    def forward(self, x):
        x1 = self.inconv(x)
        
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)

        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.outconv(x)
        return x


if __name__ == '__main__':
    model = UNet(num_of_cls=81, use_upconv=False).cuda()
    summary(model, (3, 512, 512))
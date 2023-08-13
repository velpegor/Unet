import torch
import torch.nn as nn

class BiasedSoftSparseTanh(nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau

    def forward(self, input):
        return nn.functional.tanh(input+self.tau) - nn.functional.tanh(self.tau)
    
    
class RepeatedConv2d(nn.Module):
    def __init__(self, in_chn, out_chn, mid_chn=None, ksize=3, stride=1, padding=1, bias=True, tau=None):
        super(RepeatedConv2d, self).__init__()
        if mid_chn is None:
            mid_chn = out_chn
        
        self.tau = tau

        self.conv1 = nn.Conv2d(in_channels=in_chn, out_channels=mid_chn, kernel_size=ksize, stride=stride, padding=padding, bias=bias)
        self.conv2 = nn.Conv2d(in_channels=mid_chn, out_channels=out_chn, kernel_size=ksize, stride=stride, padding=padding, bias=bias)

        self.bn1 = nn.BatchNorm2d(num_features=mid_chn)
        self.bn2 = nn.BatchNorm2d(num_features=out_chn)

        self.trainable_tau = BiasedSoftSparseTanh(tau=self.tau)

    def forward(self, x):
        x = self.trainable_tau(self.bn1(self.conv1(x)))
        x = self.trainable_tau(self.bn2(self.conv2(x)))
        return x


class Down(nn.Module):
    def __init__(self, in_chn, out_chn, tau=None):
        super(Down, self).__init__()
        self.tau = tau
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.conv = RepeatedConv2d(in_chn, out_chn, tau=self.tau)
    
    def forward(self, x):
        return self.conv(self.maxpool(x))


class Up(nn.Module):
    def __init__(self, in_chn, out_chn, use_upconv, tau=None):
        super(Up, self).__init__()
        self.use_upconv = use_upconv
        self.tau = tau

        if self.use_upconv:
            self.upsample = nn.ConvTranspose2d(in_chn, out_chn, kernel_size=(2,2), stride=2)
            self.conv = RepeatedConv2d(in_chn, out_chn, tau=self.tau)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            self.conv = RepeatedConv2d(in_chn + out_chn, out_chn, in_chn, tau=self.tau)
            
        
        # self.conv = RepeatedConv2d(in_chn, out_chn)
    
    def forward(self, x, hres):
        x = self.upsample(x)
        x = self._concat(hres, x)
        x = self.conv(x)
        return x
    
    def _concat(self, hres, x):
        _, _, w, h = x.size()
        hres_crop = self._crop(hres, w, h)
        return torch.cat((hres_crop, x), 1)

    def _crop(self, x, tw, th):
        w, h = x.size()[-2:]
        dw = (w-tw) // 2
        dh = (h-th) // 2
        return x[:,:,dw:dw+tw,dh:dh+th]
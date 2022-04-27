import numpy as np
import torch
import torch.nn as nn

class mainNet(nn.Module):
    def __init__(self,in_size, out_size):
        super(mainNet,self).__init__()
        self.Conv_right = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU(),
            nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU()
        )
        self.Conv_down = nn.Sequential(
            nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU()
        )

    def forward(self,x):
        two_conv = self.Conv_right(x)
        down2r = self.Conv_down(x)
        return two_conv,down2r

class strengthNet(nn.Module):
    def __init__(self,in_size, out_size):
        super(strengthNet, self).__init__()
        self.Conv_right2 = nn.Sequential(
            nn.Conv2d(in_size, out_size*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_size*2),
            nn.ReLU(),
            nn.Conv2d(out_size*2, out_size*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_size*2),
            nn.ReLU()
        )
        self.Conv_up = nn.Sequential( # 使用反卷积替换插值上采样+卷积
            nn.ConvTranspose2d(in_channels=out_size*2, out_channels=out_size,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(out_size),
            nn.ReLU()
        )

    def forward(self,x,out):
        two_conv2 = self.Conv_right2(x)
        up2r = self.Conv_up(two_conv2)
        cat_out = torch.cat((up2r,out),dim=1)
        return cat_out

class Unet(nn.Module):
    def __init__(self,in_size=3,num_classes=21):
        super(Unet, self).__init__()
        # 下采样
        self.main1 = mainNet(in_size,64) # 3-64 channels
        self.main2 = mainNet(64,128) #64-128
        self.main3 = mainNet(128,256) # 128-256
        self.main4 = mainNet(256, 512) # 256-512
        # 上采样
        self.strength1 = strengthNet(512,512) # 512-1024-512
        self.strength2 = strengthNet(1024, 256) # 1024-512-256
        self.strength3 = strengthNet(512, 128) # 512-256-128
        self.strength4 = strengthNet(256, 64) # 256-128-64
        # 输出conv
        self.out = nn.Sequential(
            nn.Conv2d(128,64,3,padding=1),
            nn.Conv2d(64,num_classes,1)
        )

    def forward(self,x):
        out_1, out1 = self.main1(x)
        out_2, out2 = self.main2(out1)
        out_3, out3 = self.main3(out2)
        out_4, out4 = self.main4(out3)
        out5 = self.strength1(out4, out_4)
        out6 = self.strength2(out5, out_3)
        out7 = self.strength3(out6, out_2)
        out8 = self.strength4(out7, out_1)
        out = self.out(out8)
        return out

if __name__ == '__main__':
    lr = 0.00001
    net = Unet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net.to(device)
    print(model)

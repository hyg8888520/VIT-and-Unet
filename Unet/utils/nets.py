from turtle import forward
import torch.nn as nn
import torch

# 在下采样过程中，特征图缩小的尺度是上一层的一半；而在上采样过程中特征图变为上一层的一倍。通道数量变化相反。 因此能看到out*2
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()
        # Encoder
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        # Decoder
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.output = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        conv5 = self.conv5(pool4)

        up6 = self.up6(conv5)
        meger6 = torch.cat([up6, conv4], dim=1)
        conv6 = self.conv6(meger6)
        up7 = self.up7(conv6)
        meger7 = torch.cat([up7, conv3], dim=1)
        conv7 = self.conv7(meger7)
        up8 = self.up8(conv7)
        meger8 = torch.cat([up8, conv2], dim=1)
        conv8 = self.conv8(meger8)
        up9 = self.up9(conv8)
        meger9 = torch.cat([up9, conv1], dim=1)
        conv9 = self.conv9(meger9)
        out = self.output(conv9)

        return out

if __name__ == '__main__':
    lr = 0.00001
    net = Unet(3,21)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net.to(device)
    print(model)
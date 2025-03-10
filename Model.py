import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dropout=False, dropout_p=0.5):
        super().__init__()
        if dropout:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Dropout(p=dropout_p),
                DoubleConv(in_channels, out_channels)
            )
        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels)
            )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, dropout_flag=False, dropout_p=0.5):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
        self.dropout_flag = dropout_flag
        if dropout_flag:
            self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x1, x2):
        # x1 --> C, H, W
        # x2 --> C/2, H*2, W*2

        x1 = self.up(x1)  # C, H, W --> C/2, H*2, W*2
        # Pad x1 before concatination
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        # C/2, H*2, W*2 and C/2, H*2, W*2 --> C,  H*2, W*2
        x = torch.cat([x2, x1], dim=1)
        if self.dropout_flag:
            x = self.dropout(x)
        # DoubleConv C, H*2, W*2 --> C/2, H*2, W*2
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, initial_feature_map=64, usa_cuda=True, dropout=False, dropout_p=0.5):
        super(UNet, self).__init__()
        self.usa_cuda = usa_cuda
        if n_channels==-2:
            self.n_channels = 3
        elif n_channels==-1:
            self.n_channels = 1
        else:
            self.n_channels = n_channels
        self.n_classes = n_classes
        self.initial_feature_map = initial_feature_map
        self.dropout = dropout
        self.dropout_p = dropout_p

        # Encoder
        self.inc = DoubleConv(self.n_channels, self.initial_feature_map)
        self.inc.apply(self.weights_init)
        self.down1 = Down(self.initial_feature_map,
                          self.initial_feature_map*2, self.dropout, self.dropout_p)
        self.down1.apply(self.weights_init)
        self.down2 = Down(self.initial_feature_map*2,
                          self.initial_feature_map*4, self.dropout, self.dropout_p)
        self.down2.apply(self.weights_init)
        self.down3 = Down(self.initial_feature_map*4,
                          self.initial_feature_map*8, self.dropout, self.dropout_p)
        self.down3.apply(self.weights_init)
        self.down4 = Down(self.initial_feature_map*8,
                          self.initial_feature_map*16, self.dropout, self.dropout_p)
        self.down4.apply(self.weights_init)

        # decoder
        self.up1 = Up(self.initial_feature_map*16,
                      self.initial_feature_map*8, self.dropout, self.dropout_p)
        self.up1.apply(self.weights_init)
        self.up2 = Up(self.initial_feature_map*8,
                      self.initial_feature_map*4, self.dropout, self.dropout_p)
        self.up2.apply(self.weights_init)
        self.up3 = Up(self.initial_feature_map*4,
                      self.initial_feature_map*2, self.dropout, self.dropout_p)
        self.up3.apply(self.weights_init)
        self.up4 = Up(self.initial_feature_map*2,
                      self.initial_feature_map, self.dropout, self.dropout_p)
        self.up4.apply(self.weights_init)
        self.outc = OutConv(self.initial_feature_map, n_classes)
        self.outc.apply(self.weights_init)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)

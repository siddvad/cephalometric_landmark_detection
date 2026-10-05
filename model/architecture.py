import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, X):
        return self.block(X)

class UNetLandmark(nn.Module):
    def __init__(self, n_landmarks=29):
        super().__init__()

        self.enc1 = ConvBlock(1,16)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(16,32)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(32,64)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(64,128)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(128,128)

        self.up4 = nn.ConvTranspose2d(128,128,2,stride=2)
        self.dec4 = ConvBlock(256,128)

        self.up3 = nn.ConvTranspose2d(128,64,2,stride=2)
        self.dec3 = ConvBlock(128,64)

        self.up2 = nn.ConvTranspose2d(64,32,2,stride=2)
        self.dec2 = ConvBlock(64,32)

        self.up1 = nn.ConvTranspose2d(32,16,2,stride=2)
        self.dec1 = ConvBlock(32,16)

        self.head = nn.Conv2d(16,n_landmarks,1)

    def forward(self,x):

        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        u4 = self.up4(b)
        u4 = torch.cat([u4,e4],dim=1)
        d4 = self.dec4(u4)

        u3 = self.up3(d4)
        u3 = torch.cat([u3,e3],dim=1)
        d3 = self.dec3(u3)

        u2 = self.up2(d3)
        u2 = torch.cat([u2,e2],dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1,e1],dim=1)
        d1 = self.dec1(u1)

        heatmaps = self.head(d1)

        heatmaps = F.interpolate(heatmaps,size=(128,128),mode="bilinear",align_corners=False)

        return heatmaps
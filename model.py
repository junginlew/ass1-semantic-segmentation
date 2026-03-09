import torch
import torch.nn as nn

class DoubleConv(nn.Module):  #(Conv2d -> BatchNorm2d -> ReLU) * 2
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1): # RGB에서 흑백 마스크로
        super().__init__()
        
        #Encoder(downsampling)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2) #해상도 절반으로 줄임
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        
        #Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        #Decoder(upsampling)
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) #해상도 2배, 채널 절반
        self.up1 = DoubleConv(1024, 512) # Skip connection: bottleneck에서 온 512 채널+인코더(down4)에서 온 512 = 1024개 채널 -> 512개 채널로 줄임
        
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = DoubleConv(512, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3 = DoubleConv(256, 128)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up4 = DoubleConv(128, 64)
        
        #Final output
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1) #최종 마스크 (이진 분류), 채널 1개로 줄임
   
    def forward(self, x):
        #Encoder
        skip1 = self.down1(x)
        x = self.pool(skip1)
        
        skip2 = self.down2(x)
        x = self.pool(skip2)
        
        skip3 = self.down3(x)
        x = self.pool(skip3)

        skip4 = self.down4(x)
        x = self.pool(skip4)
        
        # 2. 병목 층 통과
        x = self.bottleneck(x)
        
        #Decoder, skip connection 결합
        x = self.upconv1(x)
        x = torch.cat([skip4, x], dim=1) #dim=1로 Channel 축을 따라 합침
        x = self.up1(x) # 1024채널 -> 512채널로 줄임
        
        x = self.upconv2(x)
        x = torch.cat([skip3, x], dim=1)
        x = self.up2(x) #256채널로 줄임

        x = self.upconv3(x)
        x = torch.cat([skip2, x], dim=1)
        x = self.up3(x) #128채널

        x = self.upconv4(x)
        x = torch.cat([skip1, x], dim=1)
        x = self.up4(x) #64채널
        
        x = self.final_conv(x) #채널 1개 , (-inf, +inf) 범위의 logit 출력 
        
        return x
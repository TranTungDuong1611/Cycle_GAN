import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary

from torchgan.models.conv_blocks import *

def init_generator(in_channels=3):
    """
        initialize generator for CycleGAN
    """
    return Generator()


class Generator(nn.Module):
    def __init__(self, in_channels=3):
        """ Generator using UNet architecture

        Args:
            in_channels (int, optional): input channels. Defaults to 3.
        """
        super(Generator, self).__init__()
        
        # Define conv down block
        self.conv_down1 = ConvDownBlock(in_channels, 64, kernel_size=7, stride=2, padding=3)   # (3, 256, 256) -> (64, 128, 128)
        self.conv_down2 = ConvDownBlock(64, 128)    # (64, 128, 128) -> (128, 64, 64)
        self.conv_down3 = ConvDownBlock(128, 256)   # (128, 64, 64) -> (256, 32, 32)
        self.conv_down4 = ConvDownBlock(256, 512)   # (256, 32, 32) -> (512, 16, 16)
        self.conv_down5 = ConvDownBlock(512, 512)   # (512, 16, 16) -> (512, 8, 8)
        self.conv_down6 = ConvDownBlock(512, 512)  # (512, 8, 8) -> (512, 4, 4)
        self.conv_down7 = ConvDownBlock(512, 512)  # (512, 4, 4) -> (512, 2, 2)
        
        # bottle neck
        self.bottle_neck = ConvDownBlock(512, 512, norm='none')  # (512, 2, 2) -> (512, 1, 1)
        
        # Define conv up block
        self.conv_up7 = ConvUpBlock(512, 512)   # (512, 1, 1) -> (512, 2, 2)
        self.conv_up6 = ConvUpBlock(1024, 512)    # (512, 2, 2) -> (512, 4, 4)
        self.conv_up5 = ConvUpBlock(1024, 512)   # (512, 4, 4) -> (512, 8, 8)
        self.conv_up4 = ConvUpBlock(1024, 512)   # (512, 8, 8) -> (512, 16, 16)
        self.conv_up3 = ConvUpBlock(1024, 256)   # (512, 16, 16) -> (256, 32, 32)
        self.conv_up2 = ConvUpBlock(512, 128)  # (256, 32, 32) -> (128, 64, 64)
        self.conv_up1 = ConvUpBlock(256, 64)  # (128, 64, 64) -> (64, 128, 128)
        self.output = ConvUpBlock(64, 3, act='tanh')  # (64, 128, 128) -> (3, 256, 256)
        
    def forward(self, x):
        # Conv down
        xd1 = self.conv_down1(x)
        xd2 = self.conv_down2(xd1)
        xd3 = self.conv_down3(xd2)
        xd4 = self.conv_down4(xd3)
        xd5 = self.conv_down5(xd4)
        xd6 = self.conv_down6(xd5)
        xd7 = self.conv_down7(xd6)
        
        # Bottle neck
        bottle_neck = self.bottle_neck(xd7)
        
        # Conv up
        xu7 = self.conv_up7(bottle_neck)
        xu6 = self.conv_up6(torch.cat((xu7, xd7), 1))
        xu5 = self.conv_up5(torch.cat((xu6, xd6), 1))
        xu4 = self.conv_up4(torch.cat((xu5, xd5), 1))
        xu3 = self.conv_up3(torch.cat((xu4, xd4), 1))
        xu2 = self.conv_up2(torch.cat((xu3, xd3), 1))
        xu1 = self.conv_up1(torch.cat((xu2, xd2), 1))
        
        # Output
        output = self.output(xu1)
        
        return output
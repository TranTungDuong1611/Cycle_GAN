import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from torchsummary import summary
from models.generator import ConvDownBlock

def init_discriminator(in_channels=3):
    """
        initialize discriminator for CycleGAN
    """
    return Discriminator(ConvDownBlock, in_channels)

class Discriminator(nn.Module):
    def __init__(self, ConvDownBlock, in_channels=3):
        """ discriminator class for CycleGAN using PatchGAN

        Args:
            ConvDownBlock: Convolution downsampling block
            in_channels (int, optional): Defaults to 3.
            num_conv_blocks (int, optional): Number of conv down blocks. Defaults to 8.
        """
        
        super(Discriminator, self).__init__()
        
        conv_downs = []
        self.middle_channels = 64
        for i in range(5):
            conv_downs.append(ConvDownBlock(
                in_channels = in_channels,
                out_channels = self.middle_channels,
                kernel_size = 4,
                stride = 2 if i < 3 else 1,
                padding = 0,
                act='leaky_relu',
                norm='instance' if i != 4 else 'none'
            ))
            in_channels = self.middle_channels
            self.middle_channels = self.middle_channels * 2
            
        # Final layer is 1 channel
        conv_downs.append(ConvDownBlock(in_channels, 1, kernel_size=4, stride=1, padding=1, act='sigmoid', norm='none'))
        self.model = nn.Sequential(*conv_downs)
        
    def forward(self, x):
        x = self.model(x)
        return x
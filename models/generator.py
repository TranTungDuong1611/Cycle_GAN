import torch
import torch.nn as nn
from torchsummary import summary

class Generator(nn.Module):
    def __init__(self, ConvDownBlock, ConvUpBlock, in_channels=3, num_conv_blocks=8, down_factor=3):
        """ generator class for CycleGAN

        Args:
            ConvDownBlock: Convolution downsampling block
            ConvUpBlock: Convolution upsampling block
            in_channels (int, optional): Defaults to 3.
            num_conv_blocks (int, optional): Number of conv down blocks. Defaults to 8.
            down_factor (int, optional): consider number of channel in the deepest block. Defaults to 3.
                                        if down_factor = 3 means the number of channels in the deepest block is 64 * 2^3 = 512
        """
        
        super(Generator, self).__init__()
        
        self.middle_channels = 64
                
        # Define the generator architecture
        # Conv downsampling blocks
        conv_downs = []
        for i in range(num_conv_blocks):
            conv_downs.append(ConvDownBlock(
                in_channels = in_channels,
                out_channels = self.middle_channels,
                kernel_size = 7 if i == 0 else 3,
                stride = 2,
                padding = 3 if i == 0 else 1,
                act='leaky_relu',
                norm='instance' if i != num_conv_blocks - 1 else 'none'
            ))
            
            in_channels = self.middle_channels
            # Double the number of channels maximum (middle_channels * 2^4) (64 -> 1024)
            if i < down_factor:
                self.middle_channels = self.middle_channels * 2
            
        self.conv_downs = nn.Sequential(*conv_downs)
        
        conv_ups = []
        for i in range(num_conv_blocks):
            in_channels = self.middle_channels
            # Halve the number of channels maximum (middle_channels / 2^4) (1024 -> 64)
            if i > (num_conv_blocks - down_factor) and i < num_conv_blocks - 1:
                self.middle_channels = self.middle_channels // 2
            elif i == num_conv_blocks - 1:          # final layer is 3 channels
                self.middle_channels = 3
                
            conv_ups.append(ConvUpBlock(
                in_channels = in_channels,
                out_channels = self.middle_channels,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                act='leaky_relu' if i < num_conv_blocks - 1 else 'tanh',
                norm='instance'
            ))
        
        self.conv_ups = nn.Sequential(*conv_ups)
        
    def forward(self, x):
        # Pass through the downsampling blocks
        x = self.conv_downs(x)
        
        # Pass through the upsampling blocks
        x = self.conv_ups(x)
        
        return x
        
        
class ConvDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, act='leaky_relu', norm='instance'):
        super(ConvDownBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        
        # Normalization layer
        if norm == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=False)
        elif norm == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'none':
            self.norm = nn.Identity()
        else:
            raise ValueError("Unsupported normalization type: {}".format(norm))

        # Activation function
        if act == 'leaky_relu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError("Unsupported activation function: {}".format(act))
    
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    
class ConvUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, act='leaky_relu', norm='instance'):
        super(ConvUpBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        
        # Normalization layer
        if norm == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=False)
        elif norm == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'none':
            self.norm = nn.Identity()
        else:
            raise ValueError("Unsupported normalization type: {}".format(norm))

        # Activation function
        if act == 'leaky_relu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError("Unsupported activation function: {}".format(act))
        
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    
    
generator = Generator(ConvDownBlock, ConvUpBlock)
# Print the model summary
summary(generator, (3, 256, 256), device='cpu')
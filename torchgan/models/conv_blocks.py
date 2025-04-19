import torch
import torch.nn as nn

class ConvDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, act='leaky_relu', norm='instance'):
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
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
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
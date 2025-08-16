import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        # in_channels: number of input layers in the block
        # out_channels: number of output layers in the block
        # stride: stride of the convolution i.e. how many pixels to skip
        super().__init__() # initialize the parent class nn.Module constructor
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               3, stride, padding=1, bias=False) # first convolution layer of residual block is a 3x3 convolution (conv2d)
        self.bn1 = nn.BatchNorm2d(out_channels) # batch normalization layer after the first convolution, i.e. normalizes the output of the convolution layer
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               3, padding=1, bias=False) # second convolution layer of residual block is also a 3x3 convolution
        self.bn2 = nn.BatchNorm2d(out_channels) # batch normalization layer after the second convolution, i.e. normalizes the output of the second convolution layer

        self.shortcut = nn.Sequential() # shortcut connection, i.e. the input to the block is added to the output of next layer
        self.use_shortcut = stride != 1 or in_channels != out_channels # if the stride is not 1 or the number of input channels is not equal to the 
        # number of output channels, we need a shortcut connection with transformation, as stride > 1 will reduce the spatial dimensions which will not match 
        # the output dimensions of the block and if the number of input channels is not equal to the number of output channels, it can cause a mismatch in the 
        # number of channels , this mostly occurs when we pass input from one residual block to another residual block with different number of channels
        if self.use_shortcut:
            self.shortcut = nn.Sequential( # making a sequential layer for the shortcut connection as it will contain a convolution layer and a batch normalization layer
                # which will transform the input to match the output dimensions of the block,here we use a 1x1 convolution to change the number of channels and a batch normalization layer
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))

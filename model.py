import torch.nn as nn
import torch


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
          # init - This is where you define the layers once (Conv2d, BatchNorm, Linear, etc.). Think of it as declaring the tools your block will use.  
            
    def forward(self, x, fmap_dict=None, prefix=""):  # This is where you define how the input flows through those layers. # You can use the same layers 
        # multiple times, combine outputs, add shortcuts, etc. Every time you call the block, forward() runs.
        # the arguments are: X: input tensor, fmap_dict: dictionary to store feature maps, prefix: string to prefix the keys in fmap_dict
            out = self.conv1(x) # first convolution layer 
            out = self.bn1(out) # batch normalization after the first convolution
            out = torch.relu(out) # activation function after the first convolution
            out = self.conv2(out) # second convolution layer
            out = self.bn2(out) # batch normalization after the second convolution
            shortcut = self.shortcut(x) if self.use_shortcut else x # if we are using a shortcut connection, we check if we need to transform the input using the use_shortcut flag
            # else if not needed, we just pass the input as it is 
            out_add = out + shortcut # adding the output of the block to the input (shortcut connection), this main idea of residual blocks 

            if fmap_dict is not None: # store the feature map after the addition in fmap_dict
                fmap_dict[f"{prefix}.conv"] = out_add

            out = torch.relu(out_add) # applying ReLU activation after adding the shortcut connection for non-linearity
            if fmap_dict is not None: # store the feature map after the ReLU activation in fmap_dict
                fmap_dict[f"{prefix}.relu"] = out

            return out

class AudioCNN(nn.Module): # This is the main class for the audio CNN model
    def __init__(self, num_classes=50):
        # num_classes: number of output classes for classification according to the dataset
        # super().__init__() initializes the parent class nn.Module constructor
        super().__init__()
        self.conv1 = nn.Sequential( # defining the first convolutional layer with a 7x7 kernel, stride of 2, padding of 3, and batch normalization and 
                                   #ReLU activation and max pooling layer, this will be used later in the forward pass for exact use
            nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(3, stride=2, padding=1))
        self.layer1 = nn.ModuleList([ResidualBlock(64, 64) for i in range(3)]) # defining 3 residual blocks each with 64 input and output channels in the first layer
        self.layer2 = nn.ModuleList( # defining 4 residual blocks in the second layer, first block has 64 input channels and 128 output channels, 
                                    #rest have 128 input and output channels, this is done to match output of previous layer with input of next layer
            [ResidualBlock(64 if i == 0 else 128, 128, stride=2 if i == 0 else 1) for i in range(4)])
        self.layer3 = nn.ModuleList( # defining 6 residual blocks in the third layer, first block has 128 input channels and 256 output channels, 
                                    # rest have 256 input and output channels
            [ResidualBlock(128 if i == 0 else 256, 256, stride=2 if i == 0 else 1) for i in range(6)])
        self.layer4 = nn.ModuleList(
            [ResidualBlock(256 if i == 0 else 512, 512, stride=2 if i == 0 else 1) for i in range(3)])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # adaptive average pooling layer to reduce the spatial dimensions to 1x1
        self.dropout = nn.Dropout(0.5) # dropout layer with a probability of 0.5 to prevent overfitting , this will randomly set 50% of the input to zero during training
        # this is done to prevent overfitting by randomly dropping out some neurons during training
        self.fc = nn.Linear(512, num_classes)  # fully connected layer with 512 input features and num_classes output features for classification

    def forward(self, x, return_feature_maps=False): # This is the main function where the input tensor x flows through the model
        # return_feature_maps: boolean flag to return feature maps for visualization or debugging
        # if return_feature_maps is False, we just pass the input through the model and return the output, input x comes from the dataset in the form of a tensor
        if not return_feature_maps: # if we do not want to return feature maps
            x = self.conv1(x) # first convolutional layer
            for block in self.layer1: # passing the input through the first layer of residual blocks and iterating through each block as mentioned in the init function
                x = block(x) # passing the input through each residual block and updating the input for the next block
            for block in self.layer2: # passing the input through the second layer of residual blocks
                x = block(x)
            for block in self.layer3:
                x = block(x)
            for block in self.layer4:
                x = block(x)
            x = self.avgpool(x) # adaptive average pooling to reduce the spatial dimensions to 1x1
            x = x.view(x.size(0), -1) # flattening the output to a 1D tensor
            x = self.dropout(x) # applying dropout to the flattened output
            x = self.fc(x) # passing the output through the fully connected layer to get the final output
            # this will return the output tensor which is the final output of the model
            return x
        else: # if we want to return feature maps, we will store the feature maps in a dictionary and return it along with the output
            # this is useful for visualization or debugging purposes to see the intermediate outputs of the model
            feature_maps = {}
            x = self.conv1(x) # first convolutional layer
            feature_maps["conv1"] = x # storing the feature map after the first convolutional layer in the dictionary

            for i, block in enumerate(self.layer1): # passing the input through the first layer of residual blocks and iterating through each block
                x = block(x, feature_maps, prefix=f"layer1.block{i}") # X is feature map returned after passing through each residual block, this intermediate 
                # feature map is stored in the dictionary with the key as "layer1.block{i}" where i is the index of the block in the layer
                # like keys as "layer1.block0", "layer1.block1", etc. enumerate is used to get the index of the block in the layer
            feature_maps["layer1"] = x # the combined feature map after completing all blocks in layer1 is also stored in the dictionary with the key as "layer1"

            for i, block in enumerate(self.layer2): # passing the input through the second layer of residual blocks
                x = block(x, feature_maps, prefix=f"layer2.block{i}") # passing the feature_maps dictionary and prefix to store the feature maps with appropriate keys
            feature_maps["layer2"] = x # storing the feature map after the second layer of residual blocks in the dictionary

            for i, block in enumerate(self.layer3):
                x = block(x, feature_maps, prefix=f"layer3.block{i}")
            feature_maps["layer3"] = x

            for i, block in enumerate(self.layer4):
                x = block(x, feature_maps, prefix=f"layer4.block{i}")
            feature_maps["layer4"] = x

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = self.fc(x)
            return x, feature_maps

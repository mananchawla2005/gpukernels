import torch
import torch.nn as nn
from resnet_kernels import fused_add_relu, fused_conv_batch_relu

class OptimisedBasicBlock(nn.Module):
    """
    Optimised Resnet BasicBlock using fused add and relu
    """
    def __init__(self, original_block):
        super().__init__()
        self.conv1_weight = original_block.conv1.weight.data
        self.conv2_weight = original_block.conv2.weight.data
        
        # Store batch norm parameters
        self.bn1_weight = original_block.bn1.weight.data
        self.bn1_bias = original_block.bn1.bias.data
        self.bn1_mean = original_block.bn1.running_mean
        self.bn1_var = original_block.bn1.running_var
        self.conv2 = original_block.conv2
        self.bn2 = original_block.bn2
        self.downsample = original_block.downsample
        self.stride = original_block.stride
        
        # Not needed
        self.relu1 = original_block.relu
    
    def forward(self, x):
        identity = x

        out = fused_conv_batch_relu(
            x,
            self.conv1_weight,
            self.bn1_weight,
            self.bn1_bias,
            self.bn1_mean,
            self.bn1_var
        )


        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # using fused kernel
        out = fused_add_relu(out, identity)

        return out

def optimize_resnet18(model):
    """
    Replacing standard ResNet18 blocks with our version
    """
    # Access the layer groups in ResNet
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(model, layer_name)
        # Each layer is a Sequential container of BasicBlocks
        for i in range(len(layer)):
            # Replace each BasicBlock with our optimized version
            original_block = layer[i]
            layer[i] = OptimisedBasicBlock(original_block)
    
    return model
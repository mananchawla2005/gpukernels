import torch
import torch.nn as nn
from resnet_kernels import fused_add_relu

class OptimisedBasicBlock(nn.Module):
    """
    Optimised Resnet BasicBlock using fused add and relu
    """
    def __init__(self, original_block):
        super().__init__()
        self.conv1 = original_block.conv1
        self.bn1 = original_block.bn1
        self.conv2 = original_block.conv2
        self.bn2 = original_block.bn2
        self.downsample = original_block.downsample
        self.stride = original_block.stride
        
        # Not needed
        self.relu1 = original_block.relu
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

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
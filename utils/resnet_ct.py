import torch.nn as nn
from torchvision.models.resnet import Bottleneck, _resnet, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights


class CTBottleneck(Bottleneck):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Use ReLU during init so weights match
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)  # after residual

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out

def resnet50_ct(*, weights = None, progress: bool = True, **kwargs):
    weights = ResNet50_Weights.verify(weights)
    return _resnet(CTBottleneck, [3, 4, 6, 3], weights, progress, **kwargs)

def resnet101_ct(*, weights = None, progress: bool = True, **kwargs):
    weights = ResNet101_Weights.verify(weights)
    return _resnet(CTBottleneck, [3, 4, 23, 3], weights, progress, **kwargs)

def resnet152_ct(*, weights = None, progress: bool = True, **kwargs):
    weights = ResNet152_Weights.verify(weights)
    return _resnet(CTBottleneck, [3, 8, 36, 3], weights, progress, **kwargs)

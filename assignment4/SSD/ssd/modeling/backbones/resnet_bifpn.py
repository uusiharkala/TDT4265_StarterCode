import torch
from typing import Tuple, List
from torch import nn
import torchvision
from collections import OrderedDict

from .bifpn import BiFPN

class ResNet_BiFPN(torch.nn.Module):

    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.resnet = nn.ModuleList(list(torchvision.models.resnet18(pretrained=True).children())[:-1])
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes
        self.bifpn = BiFPN(output_channels, feature_size=64, num_layers=1, epsilon=0.0001)

        self.extras = nn.ModuleList([
            nn.Sequential(
                #nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1),
                nn.Conv2d(output_channels[3], output_channels[4], kernel_size=2, stride=2),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv2d(output_channels[4], output_channels[5], kernel_size=2, stride=2),
                nn.ReLU(),
            ),
        ])
        self.init_parameters()
        self.out_channels = [64, 64, 64, 64, 64, 64]

    def init_parameters(self):
        for m in self.extras.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        # First 5 children including conv 1 and Res0
        for i in range(5):
            x = self.resnet[i](x)
        out_features.append(x)
        # Child number 5 containing Res1
        x = self.resnet[5](x)
        out_features.append(x)
        # Child number 6 containing Res2
        x = self.resnet[6](x)
        out_features.append(x)
        # Child number 7 and 8 containing Res3
        for i in range(7, len(self.resnet) - 1):
            x = self.resnet[i](x)
        out_features.append(x)

        # out_features_dict = OrderedDict()
        # for i, feature in enumerate(out_features):
        #   out_features_dict[str(i)] = feature
        #
        # out_features = self.bifpn(out_features_dict)
        # out_features = list(out_features.values())

        out_features = self.bifpn(out_features)

        # Dimension Check
        for idx, feature in enumerate(out_features):
           out_channel = self.out_channels[idx]
           h, w = self.output_feature_shape[idx]
           expected_shape = (out_channel, h, w)
           #print("Expected shape: "+ str(out_channel) + ", " + str(h) + ", " + str(w) + ", got: " + str(feature.shape[1:]) + " at output IDX: {idx}")
           assert feature.shape[1:] == expected_shape, \
               f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
           assert len(out_features) == len(self.output_feature_shape),\
           f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, " \
           f"but it was: {len(out_features)}"
        # End Dimension Check



        return tuple(out_features)

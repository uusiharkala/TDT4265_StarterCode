from ssd.modeling import SSD300_ext_heads, AnchorBoxes
from .task_2_2 import (
    train,
    optimizer,
    schedulers,
    #loss_objective,
    #model,
    #backbone,
    data_train,
    data_val,
    val_cpu_transform,
    train_cpu_transform,
    gpu_transform_train,
    label_map,
    #anchors
    )

from .task_2_3_2 import loss_objective
from ssd.modeling.backbones import ResNet_BiFPN
from .task_2_3_3 import model

from tops.config import LazyCall as L

train.epochs = 50

anchors = L(AnchorBoxes)(
    feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
    strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
    min_sizes=[[16, 16], [32, 32], [48, 48], [64, 64], [86, 86], [128, 128], [128, 400]],
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
    # aspect ratio is defined per feature map (first index is largest feature map (38x38))
    # aspect ratio is used to define two boxes per element in the list.
    # if ratio=[2], boxes will be created with ratio 1:2 and 2:1
    # Number of boxes per location is in total 2 + 2 per aspect ratio
    aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2
)

backbone = L(ResNet_BiFPN)(
    #output_channels=[128, 256, 128, 128, 64, 64],
    #output_channels=[256, 512, 1024, 2048, 4096, 8192], #ResNet50
    output_channels=[64, 128, 256, 512, 1024, 2048], #ResNet18
    image_channels="${train.image_channels}",
    output_feature_sizes="${anchors.feature_sizes}"
)

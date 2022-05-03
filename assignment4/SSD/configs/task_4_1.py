from ssd.modeling import SSD300_ext_heads, AnchorBoxes
from .task_2_3_3 import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    model,
    #backbone,
    data_train,
    data_val,
    val_cpu_transform,
    train_cpu_transform,
    gpu_transform_train,
    label_map,
    anchors
    )


from ssd.modeling.backbones import ResNet_BiFPN
from tops.config import LazyCall as L

train.epochs = 50

backbone = L(ResNet_BiFPN)(
    #output_channels=[256, 512, 1024, 2048, 4096, 8192], #ResNet50
    output_channels=[64, 128, 256, 512, 1024, 2048], #ResNet18
    image_channels="${train.image_channels}",
    output_feature_sizes="${anchors.feature_sizes}"
)

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

from .task_2_3_2 import loss_objective  # Keep Focal Loss
from ssd.modeling.backbones import FPN50    # Get FPN with Retina50
from tops.config import LazyCall as L

# Goal: Retrain the model from Task 2.3.3 with ResNet50 for the comparison
# with the RetinaNet50 from the detectron2 repository


train.epochs = 50

anchors = L(AnchorBoxes)(
    feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
    strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
    min_sizes=[[16, 16], [32, 32], [48, 48], [64, 64], [86, 86], [128, 128], [128, 400]],
    # Added aspect ratios such that all classes have 6 boxes per anchor
    # to enable the heads which share parameters over the feature maps
    aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2
)

backbone = L(FPN50)(
    output_channels=[256, 512, 1024, 2048, 4096, 8192], #ResNet50
    image_channels="${train.image_channels}",
    output_feature_sizes="${anchors.feature_sizes}"
)

model = L(SSD300_ext_heads)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=8 + 1  # Add 1 for background
)

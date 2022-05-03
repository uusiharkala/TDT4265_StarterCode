from .task_2_2 import ( # Import config using augmentations
    train,
    optimizer,
    schedulers,
    loss_objective,
    model,
    backbone,
    data_train,
    data_val,
    val_cpu_transform,
    train_cpu_transform,
    gpu_transform_train,
    label_map,
    anchors
    )

from tops.config import LazyCall as L
from ssd.modeling.backbones import ResNet
train.epochs = 50

# Goal of task_2_3_0: Replace basic backbone by pretrained ResNet
# Attention: check in ssd.py if initialization from task 2.3 is deactivated

backbone = L(ResNet)(
    #output_channels=[256, 512, 1024, 2048, 4096, 8192], #ResNet50
    output_channels=[64, 128, 256, 512, 1024, 2048], #ResNet18
    image_channels="${train.image_channels}",
    output_feature_sizes="${anchors.feature_sizes}"
)

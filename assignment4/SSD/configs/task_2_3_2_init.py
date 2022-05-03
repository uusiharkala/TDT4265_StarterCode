from .task_2_3_2 import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    #model,
    backbone,
    data_train,
    data_val,
    val_cpu_transform,
    train_cpu_transform,
    gpu_transform_train,
    label_map,
    anchors
    )

from .task_2_3_1 import backbone

from tops.config import LazyCall as L

train.epochs = 50

# Goal: Improve the initialisation from task_2_3_2

model = L(SSD300_init)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=8 + 1  # Add 1 for background
)

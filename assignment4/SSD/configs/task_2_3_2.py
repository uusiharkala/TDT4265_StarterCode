from .task_2_2 import (
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

from .task_2_3_1 import backbone

from tops.config import LazyCall as L
from ssd.modeling.ssd_focal_loss import SSDFocalLoss
train.epochs = 50

loss_objective = L(SSDFocalLoss)(anchors="${anchors}")

from .test_anchors.base import anchors
from .tdt4265 import (
    train,
    optimizer,
    schedulers,
    backbone,
    model,
    data_train,
    data_val,
    loss_objective,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map)

# Goal: Get baseline model
# Attention: check in ssd.py if initialization from task 2.3 is deactivated

model.anchors = anchors
train.epochs = 50
# 50 iterations per epoch
# batch size == 32, dataset size == 1604
# --> 1604 / 32 ~= 50 iterations
# 1 batch == 1 image

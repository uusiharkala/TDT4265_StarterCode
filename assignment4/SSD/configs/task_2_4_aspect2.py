from ssd.modeling import AnchorBoxes
from tops.config import LazyCall as L
from .task_2_1 import train, optimizer, schedulers, backbone, model, data_train, data_val, loss_objective, train_cpu_transform, val_cpu_transform, gpu_transform, label_map

train.epochs = 50

anchors = L(AnchorBoxes)(
    feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
    min_sizes=[[8, 8], [16, 16], [32, 32], [48, 48], [86, 86], [128, 128], [128, 400]],
    aspect_ratios=[[3,5], [3,5], [2,4], [2,4], [2,3], [2,3]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2
)

model.anchors = anchors

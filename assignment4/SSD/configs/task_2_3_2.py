import torch
import torchvision
from torch.optim.lr_scheduler import MultiStepLR, LinearLR
from ssd.modeling import SSD300, SSDMultiboxLoss, SSDFocalLoss, backbones, AnchorBoxes
from tops.config import LazyCall as L
from ssd.data.mnist import MNISTDetectionDataset
from ssd import utils
from ssd.data.transforms import Normalize, ToTensor, GroundTruthBoxesToAnchors
from .utils import get_dataset_dir, get_output_dir

## This is the config file for the second iteration to develop retina net.
## The backbone and the loss are changed

# Import everything from the baseline of Task 2.1
from .task_2_1 import train, optimizer, schedulers, backbone, model, data_train, data_val, loss_objective, val_cpu_transform, label_map, anchors
from .task_2_2 import train_cpu_transform, gpu_transform_train

train.epochs = 50

backbone = L(backbones.ResNet)(
    #output_channels=[128, 256, 128, 128, 64, 64], #BasicModel
    #output_channels=[256, 512, 1024, 2048, 4096, 8192], #ResNet50
    output_channels=[64, 128, 256, 512, 1024, 2048], #ResNet18
    image_channels="${train.image_channels}",
    output_feature_sizes="${anchors.feature_sizes}"
)

loss_objective = L(SSDFocalLoss)(anchors="${anchors}")

model = L(SSD300)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=8 + 1  # Add 1 for background
)

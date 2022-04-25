import torch
import torchvision
from torch.optim.lr_scheduler import MultiStepLR, LinearLR
from ssd.modeling import SSD300, SSDMultiboxLoss, backbones, AnchorBoxes
from tops.config import LazyCall as L
from ssd.data.mnist import MNISTDetectionDataset
from ssd import utils
from ssd.data.transforms import Normalize, ToTensor, GroundTruthBoxesToAnchors
from .utils import get_dataset_dir, get_output_dir

## This is the config file for the first iteration of implementing RetinaNet. Only the backbone is changed relative
## to Task 2.2

# Import everything from the baseline of Task 2.1
from .task_2_3_0 import train, optimizer, schedulers, backbone, model, data_train, data_val, loss_objective, val_cpu_transform, label_map, anchors, train_cpu_transform, gpu_transform_train


train.epochs = 50

backbone = L(backbones.ResNet)(
    #output_channels=[128, 256, 128, 128, 64, 64],
    #output_channels=[256, 512, 1024, 2048, 4096, 8192], #ResNet50
    output_channels=[64, 128, 256, 512, 1024, 2048], #ResNet18
    image_channels="${train.image_channels}",
    output_feature_sizes="${anchors.feature_sizes}"
)

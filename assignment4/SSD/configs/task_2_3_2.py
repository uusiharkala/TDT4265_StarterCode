import torch
import torchvision
from torch.optim.lr_scheduler import MultiStepLR, LinearLR
from ssd.modeling import SSD300, SSDMultiboxLoss, backbones, AnchorBoxes
from tops.config import LazyCall as L
from ssd.data.mnist import MNISTDetectionDataset
from ssd import utils
from ssd.data.transforms import Normalize, ToTensor, GroundTruthBoxesToAnchors
from .utils import get_dataset_dir, get_output_dir

## This is the config file for the second iteration to develop retina net.
## The backbone and the loss are changed

# Import everything from the baseline of Task 2.1
from .task_2_3_1 import train, optimizer, schedulers, backbone, model, data_train, data_val, loss_objective, val_cpu_transform, label_map, anchors, train_cpu_transform, gpu_transform_train

from ssd.modeling.ssd_focal_loss import SSDFocalLoss
train.epochs = 50

loss_objective = L(SSDFocalLoss)(anchors="${anchors}")

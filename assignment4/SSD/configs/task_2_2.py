import torchvision
from ssd.data import TDT4265Dataset
from tops.config import LazyCall as L
from ssd.data.transforms import (
    ToTensor, Normalize, Resize,
    GroundTruthBoxesToAnchors, RandomSampleCrop, RandomHorizontalFlip,
    ColorJitter, GaussianBlur)
from .utils import get_dataset_dir

# Import everything from the baseline of Task 2.1
from .task_2_1 import (
    train,
    optimizer,
    schedulers,
    backbone,
    model,
    data_train,
    data_val,
    loss_objective,
    val_cpu_transform,
    label_map,
    anchors
    )

train.epochs = 50

train_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(RandomSampleCrop)(),
    L(ToTensor)(),
    L(RandomHorizontalFlip)(p=0.5),
    L(Resize)(imshape="${train.imshape}"),
    L(GroundTruthBoxesToAnchors)(anchors="${anchors}", iou_threshold=0.5)
])

gpu_transform_train = L(torchvision.transforms.Compose)(transforms=[
    L(Normalize)(mean=[0.4765, 0.4774, 0.2259], std=[0.2951, 0.2864, 0.2878]),
#    L(ColorJitter)()
    L(GaussianBlur)(kernel_size=5)
])

data_train.dataset = L(TDT4265Dataset)(
    img_folder=get_dataset_dir("tdt4265_2022_updated"),
    transform="${train_cpu_transform}",
    annotation_file=get_dataset_dir("tdt4265_2022_updated/train_annotations.json"))

data_train.gpu_transform = gpu_transform_train

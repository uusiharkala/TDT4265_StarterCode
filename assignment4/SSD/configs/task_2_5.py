from .task_2_3_3 import (
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

train.epochs = 50

# Goal: Change datasets to extended datasets and train the model from
# task_2_3_3 on it
#

from .utils import get_dataset_dir
data_train.dataset.img_folder = get_dataset_dir("tdt4265_2022_updated")
data_train.dataset.annotation_file = get_dataset_dir("tdt4265_2022_updated/train_annotations.json")
data_val.dataset.img_folder = get_dataset_dir("tdt4265_2022_updated")
data_val.dataset.annotation_file = get_dataset_dir("tdt4265_2022_updated/val_annotations.json")

data_train.dataset = L(TDT4265Dataset)(
    img_folder=get_dataset_dir("tdt4265_2022_updated"),
    transform="${train_cpu_transform}",
    annotation_file=get_dataset_dir("tdt4265_2022_updated/train_annotations.json"))

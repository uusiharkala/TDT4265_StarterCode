import time
import click
import torch
import tops
from ssd import utils
from pathlib import Path
from tops.config import instantiate
from tops.checkpointer import load_checkpoint

#Additional imports
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import cv2
import numpy as np
import torchvision
from pytorch_grad_cam import AblationCAM, EigenCAM
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels, scale_cam_image
import requests
from PIL import Image
from vizer.draw import draw_boxes
from ssd import utils
from tqdm import tqdm
from ssd.data.transforms import ToTensor


coco_names = ['__background__', 'car', 'truck', 'bus', 'motorcycle', 'bicycle',
                'scooter', 'person', 'rider']
# coco_names = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', \
#               'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
#               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
#               'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
#               'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
#               'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
#               'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
#               'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
#               'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#               'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
#               'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
#               'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
#               'scissors', 'teddy bear', 'hair drier', 'toothbrush']


# This will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

class RetinaNetModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        # Forward pass
        output = self.model(x)
        # convert the original model output (from init) into a dict
        return_dict = {}
        return_dict["boxes"] = output[0][0]
        return_dict["labels"] = output[0][1]
        return_dict["scores"] = output[0][2]
        return return_dict

def get_config(config_path):
    cfg = utils.load_config(config_path)
    cfg.train.batch_size = 1
    cfg.data_train.dataloader.shuffle = False
    cfg.data_val.dataloader.shuffle = False
    return cfg


def get_trained_model(cfg):
    model = tops.to_cuda(instantiate(cfg.model))
    model.eval()
    ckpt = load_checkpoint(cfg.output_dir.joinpath("checkpoints"), map_location=tops.get_device())
    model.load_state_dict(ckpt["model"])
    return model


def get_dataloader(cfg, dataset_to_visualize):
    # We use just to_tensor to get rid of all data augmentation, etc...
    to_tensor_transform = [
        L(ToTensor)()
    ]
    if dataset_to_visualize == "train":
        cfg.data_train.dataset.transform.transforms = to_tensor_transform
        data_loader = instantiate(cfg.data_train.dataloader)
    else:
        cfg.data_val.dataset.transform.transforms = to_tensor_transform
        cfg.data_val.dataloader.collate_fn = utils.batch_collate
        data_loader = instantiate(cfg.data_val.dataloader)

    return data_loader


def convert_boxes_coords_to_pixel_coords(boxes, width, height):
    boxes[:, [0, 2]] *= width
    boxes[:, [1, 3]] *= height
    return boxes.cpu().numpy()


def convert_image_to_hwc_byte(image):
    first_image_in_batch = image[0]  # This is the only image in batch
    image_pixel_values = (first_image_in_batch * 255).byte()
    image_h_w_c_format = image_pixel_values.permute(1, 2, 0)
    return image_h_w_c_format.cpu().numpy()


def visualize_annotations_on_image(image, batch, label_map):
    boxes = convert_boxes_coords_to_pixel_coords(batch["boxes"][0], batch["width"], batch["height"])
    labels = batch["labels"][0].cpu().numpy().tolist()

    image_with_boxes = draw_boxes(image, boxes, labels, class_name_map=label_map)
    return image_with_boxes


def visualize_model_predictions_on_image(image, img_transform, batch, model, label_map, score_threshold):
    pred_image = tops.to_cuda(batch["image"])
    transformed_image = img_transform({"image": pred_image})["image"]

    boxes, categories, scores = model(transformed_image, score_threshold=score_threshold)[0]
    boxes = convert_boxes_coords_to_pixel_coords(boxes.detach().cpu(), batch["width"], batch["height"])
    categories = categories.cpu().numpy().tolist()

    image_with_predicted_boxes = draw_boxes(image, boxes, categories, scores, class_name_map=label_map)
    return image_with_predicted_boxes


def create_filepath(save_folder, image_id):
    filename = "image_" + str(image_id) + ".png"
    return os.path.join(save_folder, filename)


def create_comparison_image(batch, model, img_transform, label_map, score_threshold):
    image = convert_image_to_hwc_byte(batch["image"])
    image_with_annotations = visualize_annotations_on_image(image, batch, label_map)
    image_with_model_predictions = visualize_model_predictions_on_image(
        image, img_transform, batch, model, label_map, score_threshold)

    concatinated_image = np.concatenate([
        image,
        image_with_annotations,
        image_with_model_predictions
    ], axis=0)
    return concatinated_image


def create_and_save_comparison_images(dataloader, model, cfg, save_folder, score_threshold, num_images):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    print("Saving images to", save_folder)

    num_images_to_save = min(len(dataloader), num_images)
    dataloader = iter(dataloader)

    img_transform = instantiate(cfg.data_val.gpu_transform)
    for i in tqdm(range(num_images_to_save)):
        batch = next(dataloader)
        comparison_image = create_comparison_image(batch, model, img_transform, cfg.label_map, score_threshold)
        filepath = create_filepath(save_folder, i)
        cv2.imwrite(filepath, comparison_image[:, :, ::-1])


def get_save_folder_name(cfg, dataset_to_visualize):
    return os.path.join(
        "performance_assessment",
        cfg.run_name,
        dataset_to_visualize
    )


@click.command()
@click.argument("config_path")
@click.option("--train", default=False, is_flag=True, help="Use the train dataset instead of val")
@click.option("-n", "--num_images", default=500, type=int, help="The max number of images to save")
@click.option("-c", "--conf_threshold", default=0.3, type=float, help="The confidence threshold for predictions")
def main(config_path, train, num_images, conf_threshold):
    cfg = get_config(config_path)
    model = get_trained_model(cfg)

    if train:
        dataset_to_visualize = "train"
    else:
        dataset_to_visualize = "val"

    dataloader = get_dataloader(cfg, dataset_to_visualize)
    save_folder = get_save_folder_name(cfg, dataset_to_visualize)

    create_and_save_comparison_images(dataloader, model, cfg, save_folder, conf_threshold, num_images)


if __name__ == '__main__':
    main()

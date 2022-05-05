# Framework for implementation was taken from
# https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/Class%20Activation%20Maps%20for%20Object%20Detection%20With%20Faster%20RCNN.ipynb
# and adjusted for our needs

import cv2
import os
import tops
import click
import numpy as np
from tops.config import instantiate
from tops.config import LazyCall as L
from tops.checkpointer import load_checkpoint
from vizer.draw import draw_boxes
from ssd import utils
from tqdm import tqdm
from ssd.data.transforms import ToTensor

import torch
import torch.nn as nn

#Additional imports
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torchvision
from pytorch_grad_cam import AblationCAM, EigenCAM
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels, scale_cam_image
import requests
from PIL import Image



coco_names = ['__background__', 'car', 'truck', 'bus', 'motorcycle', 'bicycle',
                'scooter', 'person', 'rider']

# This will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

# Wrapper class to get a dictionary as output of our model
class RetinaNetModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.feature_extractor = model.feature_extractor
    def forward(self, x):
        # Forward pass
        output = self.model(x)

        # convert the original model output (from init) into a dict
        return_dict = {}
        return_dict["boxes"] = output[0][0]
        return_dict["labels"] = output[0][1]
        return_dict["scores"] = output[0][2]
        return [return_dict]

# Renormailzes the CAM into the bounding boxes
def renormalize_cam_in_bounding_boxes(boxes, image_float, grayscale_cam, labels, label_map, scores):
    """Normalize the CAM to be in the range [0, 1]
    inside every bounding boxes, and zero outside of the bounding boxes. """
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    images = []
    for x1, y1, x2, y2 in boxes:
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        #print("x1: " + str(x1) + ", x2: " + str(x2) + ", y1: " + str(y1) + ", y2: " + str(y2))
        img = renormalized_cam * 0
        img[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        images.append(img)

    renormalized_cam = np.max(np.float32(images), axis = 0)
    renormalized_cam = scale_cam_image(renormalized_cam)
    eigencam_image_renormalized = show_cam_on_image(image_float, renormalized_cam, use_rgb=True)
    image_with_bounding_boxes = draw_boxes(eigencam_image_renormalized, boxes, labels, scores, class_name_map=label_map)
    #image_with_bounding_boxes = draw_boxes(boxes, labels, label_map, eigencam_image_renormalized)
    return image_with_bounding_boxes


def get_config(config_path):
    cfg = utils.load_config(config_path)
    cfg.train.batch_size = 1
    cfg.data_train.dataloader.shuffle = False
    cfg.data_val.dataloader.shuffle = False
    return cfg


def get_trained_model(cfg):
    model = tops.to_cuda(instantiate(cfg.model))
    ckpt = load_checkpoint(cfg.output_dir.joinpath("checkpoints"), map_location=tops.get_device())
    model.load_state_dict(ckpt["model"])
    model.eval()
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

def cam_reshape_transform(x):
    target_size = torch.Size([4, 32])
    activations = []
    for key, value in x.items():
        activations.append(torch.nn.functional.interpolate(torch.abs(value), target_size, mode='bilinear'))
    activations = torch.cat(activations, axis=1)
    return activations

def visualize_model_predictions_on_image(image, img_transform, batch, model, label_map, score_threshold, renormalized):
    pred_image = tops.to_cuda(batch["image"])
    transformed_image = img_transform({"image": pred_image})["image"]
    boxes, categories, scores = model(transformed_image, score_threshold=score_threshold)[0]
    boxes = convert_boxes_coords_to_pixel_coords(boxes.detach().cpu(), batch["width"], batch["height"])
    categories = categories.cpu().numpy().tolist()
    # CAM stuff
    wrapped_model = RetinaNetModelOutputWrapper(model)
    target_layers = [wrapped_model.feature_extractor.fpn]
    wrapped_model.model.eval()
    targets = [FasterRCNNBoxScoreTarget(labels=categories, bounding_boxes=boxes)]
    cam = EigenCAM(wrapped_model,
               target_layers,
               use_cuda=torch.cuda.is_available(),
               reshape_transform=cam_reshape_transform)

    grayscale_cam = cam(transformed_image, targets=targets)
    # Take the first image in the batch:
    grayscale_cam = grayscale_cam[0, :]

    if renormalized:
        image_with_predicted_boxes = renormalize_cam_in_bounding_boxes(boxes, image/255, grayscale_cam, categories, label_map, scores)
    else:
        cam_image = show_cam_on_image(image/255, grayscale_cam, use_rgb=True)
        image_with_predicted_boxes = draw_boxes(cam_image, boxes, categories, scores, class_name_map=label_map)


    return image_with_predicted_boxes


def create_filepath(save_folder, image_id):
    filename = "image_" + (image_id<100)*"0" + (image_id<10)*"0" +str(image_id) + ".png"
    return os.path.join(save_folder, filename)


def create_cam_image(batch, model, img_transform, label_map, score_threshold, renormalized):
    image = convert_image_to_hwc_byte(batch["image"])
    image_with_model_predictions = visualize_model_predictions_on_image(
        image, img_transform, batch, model, label_map, score_threshold, renormalized)
    return image_with_model_predictions


def create_and_save_cam_images(dataloader, model, cfg, save_folder, score_threshold, num_images, renormalized):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    print("Saving images to", save_folder)

    num_images_to_save = min(len(dataloader), num_images)
    dataloader = iter(dataloader)

    img_transform = instantiate(cfg.data_val.gpu_transform)
    for i in tqdm(range(num_images_to_save)):
        batch = next(dataloader)
        comparison_image = create_cam_image(batch, model, img_transform, cfg.label_map, score_threshold, renormalized)
        filepath = create_filepath(save_folder, i)
        cv2.imwrite(filepath, comparison_image[:, :, ::-1])


def get_save_folder_name(cfg, dataset_to_visualize):
    return os.path.join(
        "cam_representation",
        cfg.run_name,
        dataset_to_visualize
    )


@click.command()
@click.argument("config_path")
@click.option("--train", default=False, is_flag=True, help="Use the train dataset instead of val")
@click.option("-n", "--num_images", default=20, type=int, help="The max number of images to save")
@click.option("-c", "--conf_threshold", default=0.5, type=float, help="The confidence threshold for predictions")
@click.option("--renormalized", default=False, is_flag=True, help="If the CAM should be renormalized")
def main(config_path, train, num_images, conf_threshold, renormalized):
    cfg = get_config(config_path)
    model = get_trained_model(cfg)

    if train:
        dataset_to_visualize = "train"
    else:
        dataset_to_visualize = "val"

    dataloader = get_dataloader(cfg, dataset_to_visualize)
    save_folder = get_save_folder_name(cfg, dataset_to_visualize)

    create_and_save_cam_images(dataloader, model, cfg, save_folder, conf_threshold, num_images, renormalized)


if __name__ == '__main__':
    main()

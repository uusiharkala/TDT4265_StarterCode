import detectron2

# import some common libraries
import numpy as np
import cv2
import random
import os
# #from google.colab.patches import cv2_imshow
#
# # import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

from detectron2.evaluation import COCOEvaluator

from detectron2.utils.visualizer import Visualizer
from detectron2.utils.logger import setup_logger
#
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances

class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)


def set_up_training():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("tdt4265_train",)
    cfg.DATASETS.TEST = ("tdt4265_val",)

    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001

    cfg.OUTPUT_DIR = "outputs/task_4_3_retina50"

    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.MAX_ITER = 5000 #adjust up if val mAP is still rising, adjust down if overfit
    cfg.SOLVER.STEPS = (1000, 1500)
    cfg.SOLVER.GAMMA = 0.05

    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
    cfg.MODEL.RETINANET.NUM_CLASSES = 8
    cfg.MODEL.RETINANET.BATCH_SIZE_PER_IMAGE = 64
    cfg.TEST.EVAL_PERIOD = 500

    return cfg



if __name__ == '__main__':
    print("Start")
    setup_logger()
    # Register Datasets
    register_coco_instances("tdt4265_train", {}, "data/tdt4265_2022/train_annotations.json", "data/tdt4265_2022")
    register_coco_instances("tdt4265_val", {}, "data/tdt4265_2022/val_annotations.json", "data/tdt4265_2022")
    # Get Metadata from Datasets
    my_dataset_train_metadata = MetadataCatalog.get("tdt4265_train")
    dataset_dicts = DatasetCatalog.get("tdt4265_val")
    # Display some data
    # for d in random.sample(dataset_dicts, 3):
    #     img = cv2.imread(d["file_name"])
    #     visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=1)
    #     vis = visualizer.draw_dataset_dict(d)
    #     cv2.imshow('jadajada',vis.get_image()[:, :, ::-1])
    #     cv2.waitKey(0)
    cfg = set_up_training()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

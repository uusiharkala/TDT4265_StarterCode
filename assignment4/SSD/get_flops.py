import time
import click
import torch
import tops
from ssd import utils
from pathlib import Path
from tops.config import instantiate
from tops.checkpointer import load_checkpoint

from tqdm import tqdm

from ptflops import get_model_complexity_info


def get_macs(model):
    with torch.cuda.device(0):
      macs, params = get_model_complexity_info(model, (3, 128, 1024), as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
      print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
      print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    return macs


@torch.no_grad()
def evaluation(cfg):
    model =instantiate(cfg.model)
    model.eval()
    model = tops.to_cuda(model)
    ckpt = load_checkpoint(cfg.output_dir.joinpath("checkpoints"), map_location=tops.get_device())
    model.load_state_dict(ckpt["model"])
    dataloader_val = instantiate(cfg.data_val.dataloader)
    batch = next(iter(dataloader_val))
    gpu_transform = instantiate(cfg.data_val.gpu_transform)
    batch = tops.to_cuda(batch)
    batch = gpu_transform(batch)
    get_macs(model)

@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def main(config_path: Path):
    cfg = utils.load_config(config_path)
    evaluation(cfg)


if __name__ == '__main__':
    main()

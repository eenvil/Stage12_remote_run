import argparse
from omegaconf import OmegaConf
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="configs/train_finetune.yaml", type=str)
parser.add_argument("--entry", type=str, default="train_net")

args, unknown = parser.parse_known_args()
cfg = OmegaConf.load(args.cfg_file)
cli_cfg = OmegaConf.from_cli(unknown)
cfg = OmegaConf.merge(cfg, cli_cfg)
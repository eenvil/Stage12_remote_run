import hydra
import pytorch_lightning as pl
import rich
import rich.syntax
import rich.tree
import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from ppd.utils.logger import Log, monitor_process_wrapper


@monitor_process_wrapper
def get_data(cfg: DictConfig, wo_train: bool = False) -> pl.LightningDataModule:
    datamodule = hydra.utils.instantiate(cfg.data, wo_train=wo_train, _recursive_=False)
    return datamodule


@monitor_process_wrapper
def get_model(cfg: DictConfig) -> pl.LightningModule:
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    return model


@monitor_process_wrapper
def get_callbacks(cfg: DictConfig) -> list:
    if not hasattr(cfg, "callbacks"):
        return None
    callbacks = []
    for callback in cfg.callbacks.values():
        if callback is not None:
            callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))
    return callbacks


@rank_zero_only
def print_cfg(cfg: DictConfig, use_rich: bool = False):
    if use_rich:
        print_order = ("data", "model", "callbacks", "logger", "pl_trainer", "exp")
        style = "dim"
        tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

        # add fields from `print_order` to queue
        # add all the other fields to queue (not specified in `print_order`)
        queue = []
        for field in print_order:
            queue.append(field) if field in cfg else Log.warn(f"Field '{field}' not found in config. Skipping.")
        for field in cfg:
            if field not in queue:
                queue.append(field)

        # generate config tree from queue
        for field in queue:
            branch = tree.add(field, style=style, guide_style=style)
            config_group = cfg[field]
            if isinstance(config_group, DictConfig):
                branch_content = OmegaConf.to_yaml(config_group, resolve=False)
            else:
                branch_content = str(config_group)
            branch.add(rich.syntax.Syntax(branch_content, "yaml"))
        rich.print(tree)
    else:
        Log.info(OmegaConf.to_yaml(cfg, resolve=False))

def find_last_ckpt_path(dirpath):
    """
    Assume ckpt is named as e{}* or last*, following the convention of pytorch-lightning.
    """
    dirpath = Path(dirpath)
    model_paths = []
    for p in sorted(list(dirpath.glob("*.ckpt"))):
        if "last" in p.name:
            continue
        model_paths.append(p)
    if len(model_paths) > 0:
        return model_paths[-1]
    else:
        Log.info("No checkpoint found, set model_path to None")
        return None
import hydra
import pytorch_lightning as pl
import os

from omegaconf import DictConfig
from ppd.entrys.utils import get_data, get_model, get_callbacks, print_cfg, find_last_ckpt_path
from typing import Tuple
from tqdm.auto import tqdm
from ppd.utils.logger import Log

def setup_trainer(cfg: DictConfig) -> Tuple[pl.Trainer, pl.LightningModule, pl.LightningDataModule]:
    """
    Set up the PyTorch Lightning trainer, model, and data module.
    """
    if cfg.print_cfg: print_cfg(cfg, use_rich=True)
    pl.seed_everything(cfg.seed)
    # preparation
    datamodule = get_data(cfg, wo_train=True)
    model = get_model(cfg)
    if cfg.pretrained_model:
        model.load_pretrained_model_eval(cfg.pretrained_model)
    else:
        raise FileNotFoundError("Pretrained model not found. Please specify 'pretrained_model' path in config.")

    # PL callbacks and logger
    callbacks = get_callbacks(cfg)
    cfg_logger = DictConfig.copy(cfg.logger)
    cfg_logger.update({'version': 'val_metrics'})
    logger = hydra.utils.instantiate(cfg_logger, _recursive_=False)

    # PL-Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        logger=logger if logger is not None else False,
        callbacks=callbacks,
        **cfg.pl_trainer,
    )

    return trainer, model, datamodule

def val(cfg: DictConfig) -> None:
    """
    Validate the model.
    """
    trainer, model, datamodule = setup_trainer(cfg)
    trainer.validate(model, datamodule.val_dataloader())

def predict(cfg: DictConfig) -> None:
    """
    Predict using the model.
    """
    trainer, model, datamodule = setup_trainer(cfg)
    trainer.predict(model, datamodule.val_dataloader())

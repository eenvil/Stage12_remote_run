import os
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from ppd.utils.logger import Log
from ppd.entrys.utils import get_data, get_model, get_callbacks, print_cfg, find_last_ckpt_path


def train_net(cfg: DictConfig) -> None:
    """
    Instantiate the trainer, and then train the model.
    """
    if cfg.print_cfg: print_cfg(cfg, use_rich=True)
    callbacks = get_callbacks(cfg)
    logger = hydra.utils.instantiate(cfg.logger, _recursive_=False)
    trainer = pl.Trainer(
        accelerator="gpu",
        logger=logger if logger is not None else False,
        callbacks=callbacks,
        **cfg.pl_trainer,
    )
    # seed everything before loading data
    pl.seed_everything(cfg.seed)
    datamodule: pl.LightningDataModule = get_data(cfg)
    model: pl.LightningModule = get_model(cfg)
    
    # load pretrained model
    ckpt_path = find_last_ckpt_path(cfg.callbacks.model_checkpoint.dirpath)
    if ckpt_path:
        model.load_pretrained_model(ckpt_path)
    
    # training loop
    trainer.fit(model, datamodule, ckpt_path=None)
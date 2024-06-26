import logging
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from ml4gw.transforms import ChannelWiseScaler

Tensor = torch.Tensor
Distribution = torch.distributions.Distribution


class AmplfiModel(pl.LightningModule):
    """
    Amplfi model base class

    Encodes common functionality for all models,
    such as on-device augmentation and preprocessing,
    """

    def __init__(
        self,
        inference_params: list[str],
        outdir: Path,
        learning_rate: float,
        weight_decay: float = 0.0,
        save_top_k_models: int = 10,
        patience: Optional[int] = None,
    ):
        super().__init__()
        self.outdir = outdir
        outdir.mkdir(exist_ok=True, parents=True)
        self.inference_params = inference_params
        self.save_hyperparameters()

        # initialize an unfit scaler here so that it is available
        # for the LightningModule to save and load from checkpoints
        self.scaler = ChannelWiseScaler(len(inference_params))

    def get_logger(self):
        logger_name = self.__class__.__name__
        return logging.getLogger(logger_name)

    def setup(self, stage):
        if stage == "fit":
            # if we're fitting, store an instance of the
            # fit scaler in the model
            # so that its weights can be checkpointed
            self.scaler = self.trainer.datamodule.scaler

    def configure_optimizers(self):
        if not torch.distributed.is_initialized():
            world_size = 1
        else:
            world_size = torch.distributed.get_world_size()

        lr = self.hparams.learning_rate * world_size
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "valid_loss"},
        }

    def configure_callbacks(self):
        checkpoint = ModelCheckpoint(
            monitor="valid_loss",
            save_top_k=self.hparams.save_top_k_models,
            save_last=True,
            auto_insert_metric_name=False,
            mode="min",
        )
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        return [checkpoint, lr_monitor]

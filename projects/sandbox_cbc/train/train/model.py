import logging
from typing import Optional

import lightning.pytorch as pl
import torch
import train.testing as test_utils
from lightning.pytorch.callbacks import ModelCheckpoint
from train.architectures.flows import FlowArchitecture
from train.callbacks import SaveAugmentedBatch

Tensor = torch.Tensor


class PEModel(pl.LightningModule):
    """
    Args:
        arch:
            Neural network architecture to train.
            This should be a subclass of `FlowArchitecture`.
        patience:
            Number of epochs to wait for validation loss to improve
        save_top_k_models:
            Maximum number of best-performing model checkpoints
            to keep during training
    """

    def __init__(
        self,
        arch: FlowArchitecture,
        learning_rate: float,
        weight_decay: float = 0.0,
        patience: Optional[int] = None,
        save_top_k_models: int = 10,
    ) -> None:
        super().__init__()
        # construct our model up front and record all
        # our hyperparameters to our logdir
        self.model = arch
        self.save_hyperparameters(ignore=["arch"])
        self._logger = self.get_logger()

    def get_logger(self):
        logger_name = "PEModel"
        return logging.getLogger(logger_name)

    def forward(self, parameters, strain) -> Tensor:
        return -self.model.log_prob(parameters, context=strain)

    def training_step(self, batch, _):
        strain, parameters = batch
        loss = self(parameters, strain).mean()
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
            logger=True,
        )
        return loss

    def validation_step(self, batch, _):
        strain, parameters = batch
        loss = self(parameters, strain).mean()
        self.log(
            "valid_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )
        return loss

    def on_test_epoch_start(self):
        self.test_results = []
        self.num_plotted = 0

    def test_step(self, batch, batch_idx):
        strain, parameters = batch
        res = test_utils.draw_samples_from_model(
            strain,
            parameters,
            self,
            self.preprocessor,
            self.inference_params,
            self.num_samples_draw,
            self.priors,
        )
        self.test_results.append(res)
        if batch_idx % 100 == 0 and self.num_plotted < self.num_plot_corner:
            skymap_filename = f"{self.num_plotted}_mollview.png"
            res.plot_corner(
                save=True,
                filename=f"{self.num_plotted}_corner.png",
                levels=(0.5, 0.9),
            )
            test_utils.plot_mollview(
                res.posterior["phi"],
                res.posterior["dec"],
                truth=(
                    res.injection_parameters["phi"],
                    res.injection_parameters["dec"],
                ),
                outpath=skymap_filename,
            )
            self.num_plotted += 1
            self.print("Made corner plots and skymap for ", batch_idx)

    def on_test_epoch_end(self):
        import bilby

        bilby.result.make_pp_plot(
            self.test_results,
            save=True,
            filename="pp-plot.png",
            keys=self.inference_params,
        )
        del self.test_results, self.num_plotted

    def configure_callbacks(self):
        checkpoint = ModelCheckpoint(
            monitor="valid_loss",
            save_top_k=self.hparams.save_top_k_models,
            save_last=True,
            auto_insert_metric_name=False,
            mode="min",
        )
        return [SaveAugmentedBatch(), checkpoint]

    def configure_optimizers(self):
        if not torch.distributed.is_initialized():
            world_size = 1
        else:
            world_size = torch.distributed.get_world_size()

        lr = self.hparams.learning_rate * world_size
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.9, patience=30, cooldown=10, min_lr=1e-8
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "valid_loss"},
        }

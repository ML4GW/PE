import torch
from train.callbacks import SaveAugmentedSimilarityBatch
from train.losses import VICRegLoss
from train.models.base import AmplfiModel


class SimilarityModel(AmplfiModel):
    """
    A LightningModule for training similarity embeddings

    Args:
        arch:
            A neural network architecture that maps waveforms
            to lower dimensional embedded space
    """

    def __init__(
        self,
        *args,
        arch: torch.nn.Module,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # TODO: parmeterize cov, std, repr weights
        self.model = arch
        self.loss = VICRegLoss()

    def forward(self, ref, aug):
        ref = self.model(ref)
        aug = self.model(aug)
        loss, *_ = self.loss(ref, aug)
        return loss

    def validation_step(self, batch, _):
        [ref, aug], _ = batch
        loss = self(ref, aug)
        self.log(
            "valid_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True
        )

    def training_step(self, batch, _):
        [ref, aug], _ = batch
        loss = self(ref, aug)
        self.log(
            "train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=False
        )
        return loss

    def configure_callbacks(self):

        callbacks = super().configure_callbacks()
        callbacks.append(SaveAugmentedSimilarityBatch())
        return callbacks

import os

import h5py
from lightning.pytorch import Callback


class SaveAugmentedBatch(Callback):
    def on_train_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # save an example training batch
            # and parameters to disk
            X = next(iter(trainer.train_dataloader))
            X, parameters = trainer.datamodule.inject(X[0])
            save_dir = trainer.logger.log_dir or trainer.logger.save_dir

            with h5py.File(os.path.join(save_dir, "train-batch.h5"), "w") as f:
                f["X"] = X.cpu().numpy()
                f["parameters"] = parameters.cpu().numpy()

            # save an example validation batch
            # and parameters to disk
            X, parameters = next(iter(trainer.datamodule.val_dataloader()))
            with h5py.File(os.path.join(save_dir, "val-batch.h5"), "w") as f:
                f["X"] = X.cpu().numpy()
                f["parameters"] = parameters.cpu().numpy()

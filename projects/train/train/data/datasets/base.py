import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import h5py
import lightning.pytorch as pl
import torch
from ml4gw.dataloading import Hdf5TimeSeriesDataset, InMemoryDataset
from ml4gw.transforms import ChannelWiseScaler, Whiten

from train.augmentations import PsdEstimator, WaveformProjector
from train.data.utils import ZippedDataset
from train.data.waveforms.sampler import WaveformSampler

Tensor = torch.Tensor
Distribution = torch.distributions.Distribution


class AmplfiDataset(pl.LightningDataModule):
    """
    Base LightningDataModule for loading data to train AMPLFI models.

    Subclasses must define the `inject` method
    which encodes how background strain,
    cross/plus polarizations and parameters
    are processed before being passed to a model

    Args:
        data_dir:
            Path to directory containing training and testing data
        inference_params:
            List of parameters to perform inference on. Can be a subset
            of the parameters that fully describes the waveforms
        highpass:
            Highpass frequency in Hz
        sample_rate:
            Rate data is sampled in Hz
        kernel_length:
            Length of the kernel seen by model in seconds
        fduration:
            The length of the whitening filter's impulse
            response, in seconds. `fduration / 2` seconds
            worth of data will be cropped from the edges
            of the whitened timeseries.
        psd_length:
            Length of data used to calculate psd in seconds
        batches_per_epoch:
            Number of batches for each training epoch.
        batch_size:
            Number of samples in each batch
        ifos:
            List of interferometers
        waveform_sampler:
            `WaveformSampler` object that produces waveforms and parameters
            for training, validation and testing.
            See `train.data.waveforms.sampler`
            for methods this object should define.
        fftlength:
            Length of the fft used to calculate the psd.
            Defaults to `kernel_length`
        min_valid_duration:
            Minimum number of seconds of validation background data

    """

    def __init__(
        self,
        data_dir: Path,
        inference_params: list[str],
        highpass: float,
        sample_rate: float,
        kernel_length: float,
        fduration: float,
        psd_length: float,
        batches_per_epoch: int,
        batch_size: int,
        ifos: List[str],
        waveform_sampler: WaveformSampler,
        fftlength: Optional[int] = None,
        min_valid_duration: float = 10000,
        verbose: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["waveform_sampler"])
        self.init_logging(verbose)
        self.data_dir = data_dir
        self.waveform_sampler = waveform_sampler
        self.train_fnames, self.val_fnames = self.train_val_split()

    def init_logging(self, verbose: bool):
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            format=log_format,
            level=logging.DEBUG if verbose else logging.INFO,
            stream=sys.stdout,
        )

    # ================================================ #
    # Distribution utilities
    # ================================================ #
    def get_world_size_and_rank(self) -> tuple[int, int]:
        """
        Name says it all, but generalizes to the case
        where we aren't running distributed training.
        """
        if not torch.distributed.is_initialized():
            return 1, 0
        else:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            return world_size, rank

    def get_logger(self, world_size, rank):
        logger_name = "AmplfiDataset"
        if world_size > 1:
            logger_name += f":{rank}"
        return logging.getLogger(logger_name)

    @property
    def device(self):
        """Return the device of the associated lightning module"""
        return self.trainer.lightning_module.device

    # ================================================ #
    # Helper utilities for preprocessing
    # ================================================ #

    def transform(self, parameters: Dict[str, Tensor]):
        """
        Make transforms to parameters before scaling
        and performing training/inference.
        For example, taking logarithm of hrss
        """
        return self.waveform_sampler.parameter_transformer(parameters)

    def scale(self, parameters, reverse: bool = False):
        """
        Apply standard scaling to transformed parameters
        """
        parameters = parameters.transpose(1, 0)
        scaled = self.scaler(parameters, reverse=reverse)
        scaled = scaled.transpose(1, 0)
        return scaled

    # ================================================ #
    # Re-parameterizing some attributes
    # ================================================ #
    @property
    def sample_length(self):
        return (
            self.hparams.kernel_length
            + self.hparams.fduration
            + self.hparams.psd_length
        )

    @property
    def num_ifos(self):
        return len(self.hparams.ifos)

    @property
    def num_params(self):
        return len(self.hparams.inference_params)

    @property
    def num_workers(self):
        local_world_size = len(self.trainer.device_ids)
        return min(6, int(os.cpu_count() / local_world_size))

    @property
    def val_batch_size(self):
        """Use larger batch sizes when we don't need gradients."""
        return int(2 * self.hparams.batch_size)

    @property
    def train_val_fnames(self):
        """List of background files used for both training and validation"""
        background_dir = self.data_dir / "train" / "background"
        fnames = list(background_dir.glob("*.hdf5"))
        return fnames

    @property
    def test_fnames(self):
        """List of background files used for testing a trained model"""
        test_dir = self.data_dir / "test" / "background"
        fnames = list(test_dir.glob("*.hdf5"))
        return fnames

    def train_val_split(self) -> Sequence[str]:
        """
        Split background files into training and validation sets
        based on the requested duration of the validation set
        """
        fnames = sorted(self.train_val_fnames)
        durations = [int(fname.stem.split("-")[-1]) for fname in fnames]
        valid_fnames = []
        valid_duration = 0
        while valid_duration < self.hparams.min_valid_duration:
            fname, duration = fnames.pop(-1), durations.pop(-1)
            valid_duration += duration
            valid_fnames.append(str(fname))

        train_fnames = fnames
        return train_fnames, valid_fnames

    # ================================================ #
    # Utilities for initial data loading and preparation
    # ================================================ #
    def build_modules(self, stage):
        """
        Build torch.nn.Modules that will be used for on-device
        augmentation and preprocessing. Transfer these modules
        to the appropiate device
        """
        self._logger.info("Building torch Modules and transferring to device")
        window_length = self.hparams.kernel_length + self.hparams.fduration
        fftlength = self.hparams.fftlength or window_length
        self.psd_estimator = PsdEstimator(
            window_length,
            self.hparams.sample_rate,
            fftlength,
            fast=self.hparams.highpass is not None,
            average="median",
        ).to(self.device)

        self.whitener = Whiten(
            self.hparams.fduration,
            self.hparams.sample_rate,
            self.hparams.highpass,
        ).to(self.device)

        # build standard scaler object and fit to parameters;
        # waveform_sampler subclasses will decide how to generate
        # parameters to fit the scaler
        self._logger.info("Fitting standard scaler to parameters")
        scaler = ChannelWiseScaler(self.num_params).to(self.device)
        self.scaler = self.waveform_sampler.fit_scaler(scaler)

        self.projector = WaveformProjector(
            self.hparams.ifos, self.hparams.sample_rate
        ).to(self.device)

        self.waveform_sampler.to(self.device)

    def setup(self, stage: str) -> None:
        world_size, rank = self.get_world_size_and_rank()
        self._logger = self.get_logger(world_size, rank)

        self._logger.info(f"Setting up data for stage {stage}")

        # infer sample rate directly from background data
        with h5py.File(self.train_fnames[0], "r") as f:
            sample_rate = 1 / f[self.hparams.ifos[0]].attrs["dx"]
            assert sample_rate == self.hparams.sample_rate

        self._logger.info(f"Inferred sample rate of {sample_rate} Hz")

        # load validation/testing waveforms, parameters, and background
        # and build the fixed background dataset while data and augmentations
        # modules are all still on CPU.
        # get_val_waveforms should be implemented by waveform_sampler object
        if stage in ["fit", "validate"]:
            self.val_background = self.load_background(self.val_fnames)
            self._logger.info(
                f"Loaded background files {self.val_fnames} for validation"
            )

            cross, plus, parameters = self.waveform_sampler.get_val_waveforms(
                rank, world_size
            )
            self._logger.info(f"Loaded {len(cross)} waveforms for validation")
            params = []
            for k in self.hparams.inference_params:
                if k in parameters.keys():
                    params.append(torch.Tensor(parameters[k]))

            self.val_waveforms = torch.stack([cross, plus], dim=0)
            self.val_parameters = torch.column_stack(params)

        elif stage == "test":
            self.test_background = self.load_background(self.test_fnames)
            self._logger.info(
                f"Loaded background files {self.test_fnames} for testing"
            )
            (
                cross,
                plus,
                parameters,
            ) = self.waveform_sampler.get_test_waveforms()

            self._logger.info(f"Loaded {len(cross)} waveforms for testing")

            params = []
            for k in self.hparams.inference_params:
                if k in parameters.keys():
                    params.append(torch.Tensor(parameters[k]))

            self.test_waveforms = torch.stack([cross, plus], dim=0)
            self.test_parameters = torch.column_stack(params)

        # once we've generated validation/testing waveforms on cpu,
        # build data augmentation modules
        # and transfer them to appropiate device
        self.build_modules(stage)

    def load_background(self, fnames: Sequence[str]):
        background = []
        for fname in fnames:
            data = []
            with h5py.File(fname, "r") as f:
                for ifo in self.hparams.ifos:
                    back = f[ifo][:]
                    data.append(torch.tensor(back, dtype=torch.float32))
            data = torch.stack(data)
            background.append(data)
        return background

    def on_after_batch_transfer(self, batch, _):
        """
        This is a Lightning `hook` that gets called after
        data returned by a dataloader gets put on the local device,
        but before it gets passed to model for inference.

        Use this to do on-device augmentation/preprocessing
        """
        if self.trainer.training:
            [batch] = batch
            cross, plus, parameters = self.waveform_sampler.sample(batch)
            strain, parameters = self.inject(batch, cross, plus, parameters)

        elif self.trainer.validating or self.trainer.sanity_checking:
            [cross, plus, parameters], [background] = batch

            background = background[: len(cross)]
            cross = cross.to(self.device)
            plus = plus.to(self.device)
            parameters = parameters.to(self.device)
            keys = [
                k
                for k in self.hparams.inference_params
                if k not in ["dec", "psi", "phi"]
            ]
            parameters = {k: parameters[:, i] for i, k in enumerate(keys)}
            strain, parameters = self.inject(
                background, cross, plus, parameters
            )

        elif self.trainer.testing:
            [cross, plus, parameters], [background] = batch
            cross = cross.to(self.device)
            plus = plus.to(self.device)
            parameters = parameters.to(self.device)
            keys = [
                k
                for k in self.hparams.inference_params
                if k not in ["dec", "psi", "phi"]
            ]
            parameters = {k: parameters[:, i] for i, k in enumerate(keys)}
            strain, parameters = self.inject(
                background, cross, plus, parameters
            )

        return strain, parameters

    # ================================================ #
    # Dataloaders used by lightning
    # ================================================ #

    def train_dataloader(self):
        # if we only have one training file
        # load it into memory and use InMemoryDataset
        if len(self.train_fnames) == 1:
            train_background = self.load_background(self.train_fnames)[0]
            dataset = InMemoryDataset(
                train_background,
                kernel_size=int(self.hparams.sample_rate * self.sample_length),
                batch_size=self.hparams.batch_size,
                coincident=False,
                batches_per_epoch=self.hparams.batches_per_epoch,
                shuffle=True,
            )
        else:
            dataset = Hdf5TimeSeriesDataset(
                self.train_fnames,
                channels=self.hparams.ifos,
                kernel_size=int(self.hparams.sample_rate * self.sample_length),
                batch_size=self.hparams.batch_size,
                batches_per_epoch=self.hparams.batches_per_epoch,
                coincident=False,
            )

        self._logger.info(
            f"Using a {dataset.__class__.__name__} class for training"
        )
        pin_memory = isinstance(
            self.trainer.accelerator, pl.accelerators.CUDAAccelerator
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=self.num_workers, pin_memory=pin_memory
        )
        return dataloader

    def val_dataloader(self):
        # TODO: allow for multiple validation segment files

        # offset the start of the validation background data
        # by the device id to add more diversity in the validation set
        _, rank = self.get_world_size_and_rank()

        # build waveform dataloader
        cross, plus = self.val_waveforms
        waveform_dataset = torch.utils.data.TensorDataset(
            cross, plus, self.val_parameters
        )
        waveform_dataloader = torch.utils.data.DataLoader(
            waveform_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            pin_memory=False,
        )

        # build background dataloader
        val_background = self.val_background[0][:, rank:]

        background_dataset = InMemoryDataset(
            val_background,
            kernel_size=int(self.hparams.sample_rate * self.sample_length),
            batch_size=self.val_batch_size,
            batches_per_epoch=len(waveform_dataloader),
            coincident=False,
            shuffle=False,
        )

        background_dataloader = torch.utils.data.DataLoader(
            background_dataset, pin_memory=False
        )
        return ZippedDataset(
            waveform_dataloader,
            background_dataloader,
        )

    def test_dataloader(self):
        # TODO: allow for multiple test segment files

        # build waveform dataloader
        cross, plus = self.test_waveforms
        waveform_dataset = torch.utils.data.TensorDataset(
            cross, plus, self.test_parameters
        )
        waveform_dataloader = torch.utils.data.DataLoader(
            waveform_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
        )

        background_dataset = InMemoryDataset(
            self.test_background[0],
            kernel_size=int(self.hparams.sample_rate * self.sample_length),
            batch_size=1,
            batches_per_epoch=len(waveform_dataloader),
            coincident=False,
            shuffle=False,
        )

        background_dataloader = torch.utils.data.DataLoader(
            background_dataset, pin_memory=False
        )
        return ZippedDataset(
            waveform_dataloader,
            background_dataloader,
        )

    def inject(self, *args, **kwargs):
        """
        Subclasses should implement this method
        for different training use cases,
        like training a similarity embedding
        or training a normalizing flow. This is called
        after the data is transferred to the local device
        """
        raise NotImplementedError

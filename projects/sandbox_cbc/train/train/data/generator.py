from typing import TYPE_CHECKING, Callable

import torch
from train.data.base import BaseDataset
from train.data.utils import ParameterSampler

from ml4gw.waveforms import SineGaussian

if TYPE_CHECKING:
    from ml4gw.transforms import ChannelWiseScaler


# type-alias for callable that takes dictionary of parameters
# and returns a tensor timeseries of waveforms
WaveformGenerator = Callable


class WaveformGeneratorDataset(BaseDataset):
    """
    Dataset that generates waveforms on the fly with torch
    and injects into background noise

    Args:
        waveform_generator:
            Callable that takes a dictionary of Tensors, each of length `N`,
            as input and returns a Tensor containing `N`
            waveforms of shape `(N, num_ifos, strain_dim)`
        parameter_sampler:
            Import path to a Callable that takes an integer `N` as input and
            returnes a dictionary of Tensors, each of length `N`
        num_val_waveforms:
            Total number of validaton waveforms to use.
            This total will be split up among all devices
        num_fit_params: N
            Number of parameters to use for fitting the standard scaler
    """

    def __init__(
        self,
        *args,
        # waveform_generator: ml4gw.waveforms.SineGaussian,
        parameter_sampler: ParameterSampler,
        num_val_waveforms: int = 10000,
        num_fit_params: int = 100000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_val_waveforms = num_val_waveforms
        self.num_fit_params = num_fit_params

        # TODO: generalize this
        self.waveform_generator = SineGaussian(
            self.sample_rate, self.sample_length
        )
        self.parameter_sampler = parameter_sampler

    @property
    def val_waveforms_per_device(self):
        world_size, _ = self.get_world_size_and_rank()
        return self.num_val_waveforms // world_size

    def fit_scaler(self, scaler: "ChannelWiseScaler") -> "ChannelWiseScaler":
        # sample parameters from parameter sampler
        # so we can fit the standard scaler
        parameters = self.parameter_sampler(self.num_fit_params)
        dec, phi, psi = self.sample_extrinsic(
            self.num_fit_params, device="cpu"
        )
        parameters.update({"dec": dec, "phi": phi, "psi": psi})

        # downselect only to those requested to do inference on
        fit_params = {
            k: v for k, v in parameters.items() if k in self.inference_params
        }

        # transform any relevant parameters and fit the scaler
        transformed = self.transform(fit_params)
        transformed = [v for k, v in transformed.items()]
        parameters = torch.vstack(transformed)
        scaler.fit(parameters)
        return scaler

    def get_val_waveforms(self):
        cross, plus, parameters = self.sample_waveforms(
            self.val_waveforms_per_device, device="cpu"
        )
        dec, phi, psi = self.sample_extrinsic(
            self.val_waveforms_per_device, device="cpu"
        )
        waveforms = self.projector(dec, phi, psi, cross=cross, plus=plus)
        parameters.update({"dec": dec, "phi": phi, "psi": psi})
        parameters = self.transform(parameters)
        return waveforms, parameters

    def sample_waveforms(self, N: int, device: torch.device):
        # sample intrinsic parameters and generate
        # intrinsic h+ and hx polarizations
        parameters = self.parameter_sampler(N, device=device)
        cross, plus = self.waveform_generator(**parameters)
        cross, plus = cross.float(), plus.float()
        return cross, plus, parameters

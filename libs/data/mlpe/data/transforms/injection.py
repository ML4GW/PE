from typing import Callable, List

import torch

from ml4gw import gw


class PEInjector(torch.nn.Module):
    """
    A torch module that generates waveforms and injects them into
    background kernels
    """

    def __init__(
        self,
        sample_rate: float,
        ifos: List[str],
        intrinsic_parameter_sampler: Callable,
        dec: Callable,
        psi: Callable,
        phi: Callable,
        waveform: Callable,
    ):
        super().__init__()

        self.waveform = waveform
        self.parameter_sampler = intrinsic_parameter_sampler
        self.sample_rate = sample_rate

        self.dec = dec
        self.psi = psi
        self.phi = phi

        tensors, vertices = gw.get_ifo_geometry(*ifos)
        self.register_buffer("tensors", tensors)
        self.register_buffer("vertices", vertices)

    def transform(self, parameters: dict):
        # take logarithm of hrss since that is what we train with;
        # remove eccentricity from parameters since we don't train with it
        parameters.pop("eccentricity")
        parameters["hrss"] = torch.log10(parameters["hrss"])
        return parameters

    def sample_waveforms(self, N: int):
        # randomly sample intrinsic parameters and generate raw polarizations
        parameters = self.parameter_sampler(N, device=self.tensors.device)
        cross, plus = self.waveform(**parameters)
        cross, plus = cross.float(), plus.float()
        dec, psi, phi = (
            self.dec(N),
            self.psi(N),
            self.phi(N),
        )

        dec = dec.to(self.tensors.device)
        psi = psi.to(self.tensors.device)
        phi = phi.to(self.tensors.device)
        waveforms = gw.compute_observed_strain(
            dec,
            psi,
            phi,
            detector_tensors=self.tensors,
            detector_vertices=self.vertices,
            sample_rate=self.sample_rate,
            plus=plus,
            cross=cross,
        )

        transformed = self.transform(parameters)
        # concatenate intrinsic parameters with sampled extrinsic parameters
        transformed = torch.column_stack(list(transformed.values()))
        transformed = torch.column_stack((transformed, dec, psi, phi))

        return waveforms, transformed

    def forward(self, X: "gw.WaveformTensor"):
        if self.training:
            # inject waveforms into every kernel
            N = len(X)

            # infer kernel size from background
            kernel_size = X.shape[-1]
            waveforms, parameters = self.sample_waveforms(N)

            # calculate the fixed location
            # where waveform T_c will placed
            center = waveforms.shape[-1] // 2
            start = center - (kernel_size // 2)
            stop = center + (kernel_size // 2)

            waveforms = waveforms[:, :, start:stop]
            X += waveforms

            return X, parameters
        return X

import torch

from ml4gw.waveforms import SineGaussian


class SGGenerator(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def init(self, sample_rate, duration):
        self.sine_gaussian = SineGaussian(sample_rate, duration)

    def slice_waveforms(self, waveforms: torch.Tensor, waveform_size: int):
        # for sine gaussians, place waveform in center of kernel
        center = waveforms.shape[-1] // 2
        half = waveform_size // 2
        start = center - half
        stop = center + half
        return waveforms[..., start:stop]

    def forward(self, parameters):
        return self.sine_gaussian(**parameters)

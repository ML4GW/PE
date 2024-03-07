from math import ceil

import torch

from ml4gw import waveforms


class FrequencyDomainWaveformGenerator:
    """Loose implementation of bilby's WaveformGenerator using torch"""

    def __init__(
        self,
        duration: float,
        sample_rate: float,
        f_min: float,
        f_max: float,
        f_ref: float,
        approximant: waveforms.IMRPhenomD,
        start_time: float = 0.0,
        post_padding: float = 0.0,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.sampling_frequency = sampling_frequency
        self.time_duration = time_duration
        self.number_of_samples = ceil(time_duration * sampling_frequency)
        self.number_of_frequencies = self.number_of_samples // 2 + 1
        self.start_time = start_time
        self.post_padding = post_padding

        self.frequency_array = torch.linspace(
            0,
            self.sampling_frequency / 2,
            self.number_of_frequencies,
            device=self.device,
            dtype=torch.float32,
        )
        self.time_array = torch.linspace(
            self.start_time,
            self.time_duration + self.start_time - 1 / self.sampling_frequency,
            self.number_of_samples,
            device=self.device,
            dtype=torch.float32,
        )
        # append the times corresponding to padding
        self.number_of_post_padding = ceil(post_padding * sampling_frequency)
        post_time_array = torch.linspace(
            self.time_duration + self.start_time,
            self.time_duration
            + self.start_time
            + self.post_padding
            - 1 / self.sampling_frequency,
            self.number_of_post_padding,
            device=self.device,
            dtype=torch.float32,
        )
        self.time_array = torch.cat((self.time_array, post_time_array))
        self.f_min = f_min
        self.f_max = f_max
        self.f_ref = f_ref
        if isinstance(approximant, str):
            try:
                self.approximant = getattr(waveforms, approximant)
            except AttributeError:
                # FIXME: provide pathway into lalsimulation/bilby API
                raise RuntimeError(
                    "Waveform not implemented in ml4gw.waveforms"
                )
        else:
            self.approximant = approximant

    def frequency_domain_strain(self, *args, **kwargs):
        kwargs["f_ref"] = self.f_ref  # FIXME: handle this better

        freq_mask = self.frequency_array >= self.f_min
        freq_mask *= self.frequency_array < self.f_max

        _hp, _hc = self.approximant(
            self.frequency_array[freq_mask], *args, **kwargs
        )

        _frequency_domain_h_plus = torch.zeros(
            _hp.shape[0],
            self.frequency_array.shape[-1],
            device=self.device,
            dtype=_hp.dtype,
        )
        _frequency_domain_h_cross = torch.zeros(
            _hc.shape[0],
            self.frequency_array.shape[-1],
            device=self.device,
            dtype=_hc.dtype,
        )
        _frequency_domain_h_plus[..., freq_mask] = _hp
        _frequency_domain_h_cross[..., freq_mask] = _hc
        return _frequency_domain_h_plus, _frequency_domain_h_cross

    def time_domain_strain(self, *args, **kwargs):
        (
            _frequency_domain_h_plus,
            _frequency_domain_h_cross,
        ) = self.frequency_domain_strain(*args, **kwargs)
        # implemented based on https://git.ligo.org/lscsoft/bilby/-/blob/master/bilby/core/utils/series.py  # noqa
        _hp = torch.fft.irfft(_frequency_domain_h_plus)
        _hc = torch.fft.irfft(_frequency_domain_h_cross)
        _time_domain_h_plus = _hp * self.sampling_frequency
        _time_domain_h_cross = _hc * self.sampling_frequency

        post_padding_samples = int(self.post_padding * self.sampling_frequency)
        _time_domain_h_plus = torch.nn.functional.pad(
            _time_domain_h_plus, (0, post_padding_samples, 0, 0)
        )
        _time_domain_h_cross = torch.nn.functional.pad(
            _time_domain_h_cross, (0, post_padding_samples, 0, 0)
        )

        return _time_domain_h_plus, _time_domain_h_cross

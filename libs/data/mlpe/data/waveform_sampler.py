from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np
from bilby.core.prior import Cosine, PriorDict, Uniform
from gwpy.frequencyseries import FrequencySeries
from mlpe.data.utils import sample_kernels
from mlpe.injection import project_raw_gw

PRIORS = PriorDict(
    {
        "ra": Uniform(minimum=0, maximum=2 * np.pi),
        "dec": Cosine(),
        "psi": Uniform(minimum=0, maximum=np.pi),
    }
)


@dataclass
class _DummyWaveformGenerator:
    sampling_frequency: float
    duration: float


class WaveformSampler:
    def __init__(
        self,
        dataset: str,
        domain: str,
        duration: float,
        min_snr: float,
        max_snr: float,
        highpass: Optional[float] = 20,
    ):
        if max_snr <= min_snr:
            raise ValueError(
                f"max_snr {max_snr} must be greater than min_snr {min_snr}"
            )
        self.min_snr = min_snr
        self.max_snr = max_snr

        self.domain = domain
        self.duration = duration

        with h5py.File(dataset, "r") as f:
            # extract waveforms, parameters
            self.waveforms = f["signals"][:]
            inference_parameter_group = f["inference_parameters"]
            inference_parameter_names = inference_parameter_group.keys()
            self.inference_parameters = np.column_stack(
                [
                    inference_parameter_group[key][:]
                    for key in inference_parameter_names
                ]
            )

        self.priors = PRIORS.copy()
        self.df = 1 / duration

        if domain == "time":
            self.sample_rate = self.waveforms.shape[-1] / duration
        elif domain == "frequency":
            self.sample_rate = self.df * self.waveforms.shape[-1] / 2

        freqs = np.arange(0, self.sample_rate // 2 + self.df, self.df)
        highpass = highpass or 0
        self.mask = freqs >= highpass

        # initialize some attributes that need
        #  to be fit to a particular background
        self.background_asd = self.ifos = None

    def fit(
        self, t0: float, tf: float, *background_asds: FrequencySeries
    ) -> None:
        ifos = [asd.channel.name.split(":")[0] for asd in background_asds]
        background_asds = [asd.interpolate(self.df) for asd in background_asds]
        background_asds = np.stack([asd.value for asd in background_asds])

        # for now, make sure our background asds
        # don't have 0 values
        if (background_asds == 0).any():
            raise ValueError("The background asds contain 0 values")

        self.priors["geocent_time"] = Uniform(minimum=t0, maximum=tf)
        self.background_asd = background_asds
        self.ifos = ifos

    def compute_snrs(self, signals: np.ndarray, domain: str) -> np.ndarray:
        if domain == "time":
            ffts = np.fft.rfft(signals, axis=-1) / self.sample_rate
        elif domain == "freq":
            ffts = signals
        snrs = 2 * np.abs(ffts) / self.background_asd
        snrs = self.df * self.mask * snrs**2
        return snrs.sum(axis=-1) ** 0.5

    def reweight_snrs(self, signals: np.ndarray) -> np.ndarray:
        snrs = self.compute_snrs(signals)
        snrs = (snrs**2).sum(axis=1) ** 0.5

        target_snrs = np.random.uniform(
            self.min_snr, self.max_snr, size=len(snrs)
        )
        weights = target_snrs / snrs
        signals = signals.transpose(1, 2, 0) * weights
        return signals.transpose(2, 0, 1)

    def sample(
        self, N: int, size: int, trigger_distance_size: int = 0
    ) -> np.ndarray:
        if self.background_asd is None:
            raise RuntimeError(
                "Must fit WaveformGenerator to background asd before sampling"
            )

        # sample some waveform indices to inject as well
        # as sky localization parameters for computing
        # the antenna response in real-time
        idx = np.random.choice(len(self.waveforms), size=N, replace=False)
        extrinsic_parameters = self.priors.sample(N)

        # get the parameters corresponding to these waveform samples
        # and append the random sky localization parameters
        inference_parameters = self.inference_parameters[idx]
        inference_parameters = np.hstack(
            [inference_parameters, extrinsic_parameters]
        )

        # initialize the output array and a dummy object
        # which has a couple attributes expected by the
        # argument passed to project_raw_gw
        # TODO: project_raw_gw should accept these arguments on their own
        signals = np.zeros((N, len(self.ifos), self.waveforms.shape[-1]))
        waveform_generator = _DummyWaveformGenerator(
            self.sample_rate, self.waveforms.shape[-1] // self.sample_rate
        )

        # for each one of the interferometers used in
        # the background asds passed to `.fit`, compute
        # its response to the waveform given the sky
        # localization parameters
        for i, ifo in enumerate(self.ifos):
            signal = project_raw_gw(
                self.waveforms[idx],
                inference_parameters,
                waveform_generator,
                ifo,
                get_snr=False,
            )
            signals[:, i] = signal

        # randomly sample kernels from these signals
        return sample_kernels(
            signals,
            size,
            trigger_distance_size,
            parameters=inference_parameters,
        )

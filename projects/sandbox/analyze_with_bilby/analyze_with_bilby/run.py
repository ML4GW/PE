import logging
import signal
import sys
from pathlib import Path
from typing import Callable, List, Optional

import bilby
import h5py
import matplotlib.pyplot as plt
import numpy as np
from gwpy.timeseries import TimeSeries

from typeo import scriptify


@scriptify
def main(
    trigger_time: float,
    ifos: List[str],
    channel: str,
    duration: float,
    sample_rate: float,
    fmin: float,
    waveform: Callable,
    prior: Callable,
    outdir: Path,
    label: str,
    post_trigger_duration: float = 2,
    psd_buffer: float = 2,
    psd_duration: Optional[float] = None,
    request_cpus: int = 1,
    nlive: int = 1024,
    nact: int = 5,
    seed: int = 112296,
):

    outdir = Path(outdir) / label
    np.random.seed(seed)
    prior = prior()
    bilby.core.utils.setup_logger(outdir=outdir, label=label)

    injection_parameters = prior.sample()
    injection_parameters["geocent_time"] = trigger_time
    psd_duration = psd_duration or 32 * duration

    # Create the waveform_generator using a sine Gaussian source function
    waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
        duration=duration,
        sampling_frequency=sample_rate,
        frequency_domain_source_model=waveform,
    )

    roll_off = 0.4
    stop = trigger_time + post_trigger_duration
    start = stop - duration

    psd_start = start - psd_duration - psd_buffer
    psd_stop = start - psd_buffer

    logging.info("Querying Data and PSDs")
    detectors = bilby.gw.detector.InterferometerList([])
    for ifo in ifos:
        channel_name = f"{ifo}:{channel}"
        interferometer = bilby.gw.detector.get_empty_interferometer(ifo)

        # reset the mininimum and maximum frequency to full range
        # so that we can save the raw time domain data for use with
        # our network
        interferometer.minimum_frequency = 0
        interferometer.maximum_frequency = sample_rate / 2

        data = TimeSeries.get(channel_name, start, stop)
        data = data.resample(sample_rate)
        interferometer.strain_data.set_from_gwpy_timeseries(data)

        psd_data = TimeSeries.get(channel_name, psd_start, psd_stop)
        psd_data = psd_data.resample(sample_rate)
        psd_alpha = 2 * roll_off / duration
        psd = psd_data.psd(
            fftlength=duration,
            overlap=0,
            window=("tukey", psd_alpha),
            method="median",
        )

        interferometer.power_spectral_density = (
            bilby.gw.detector.PowerSpectralDensity(
                frequency_array=psd.frequencies.value, psd_array=psd.value
            )
        )
        detectors.append(interferometer)

    bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)
    detectors.plot_data(outdir=outdir, label="raw")

    prior["geocent_time"] = bilby.core.prior.Uniform(
        trigger_time - 0.1, trigger_time + 0.1, name="geocent_time"
    )

    logging.info("Injecting signal into data")
    detectors.inject_signal(
        waveform_generator=waveform_generator,
        parameters=injection_parameters,
        raise_error=False,
    )
    detectors.plot_data(outdir=outdir, label="injected")

    logging.info("Plotting data")
    data = []
    for detector in detectors:
        timeseries = TimeSeries(
            data=detector.strain_data.time_domain_strain,
            times=detector.strain_data.time_array,
        )
        timeseries = timeseries.highpass(fmin)
        fig, ax = plt.subplots()

        x = detector.strain_data.time_array
        xlabel = "GPS time [s]"

        ax.plot(x, timeseries)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Strain")
        fig.tight_layout()
        fig.savefig("{}/{}_time_domain_data.png".format(outdir, detector.name))
        plt.close(fig)

        data.append(detector.strain_data.time_domain_strain)

        # set minimum and maximum frequency used during nested sampling here
        detector.minimum_frequency = fmin
        detector.maximum_frequency = sample_rate * 0.5

    with h5py.File(outdir / "data.hdf5", "w") as f:
        f.create_dataset("strain", data=np.stack(data))

    # Initialise the likelihood by passing
    # in the interferometer data (IFOs) and the waveform generator
    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=detectors, waveform_generator=waveform_generator
    )

    # copied from bilby_pipe
    def sighandler(signum, frame):
        logging.info("Performing periodic eviction")
        sys.exit(77)

    signal.signal(signal.SIGALRM, handler=sighandler)
    signal.alarm(28800)

    logging.info("Launching sampler")
    result = bilby.core.sampler.run_sampler(
        likelihood=likelihood,
        priors=prior,
        sampler="dynesty",
        nlive=nlive,
        walks=10,
        nact=nact,
        injection_parameters=injection_parameters,
        outdir=outdir,
        label=label,
        npool=request_cpus,
        exit_code=77,
        sample="rwalk",
    )

    # make some plots of the outputs
    result.plot_corner()

    # save
    result.save_to_file("result.hdf5", outdir=outdir, extension="hdf5")


if __name__ == "__main__":
    main()

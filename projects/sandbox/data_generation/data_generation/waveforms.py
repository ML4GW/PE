import logging
from pathlib import Path
from typing import Callable, Dict, Optional

import h5py

import mlpe.injection as injection
from mlpe.logging import configure_logging
from typeo import scriptify


@scriptify
def main(
    prior: Callable,
    waveform: Callable,
    sample_rate: float,
    n_samples: int,
    waveform_duration: float,
    datadir: Path,
    logdir: Path,
    waveform_arguments: Optional[Dict] = None,
    parameter_conversion: Optional[Callable] = None,
    force_generation: bool = False,
    verbose: bool = False,
):
    """Generates a dataset of raw waveforms. The goal was to make this
    project waveform agnositic

    Args:

        prior: Callable that instantiates a bilby prior
        waveform: A callable compatible with bilby waveform generator
        sample_rate: sample rate for generating waveform
        n_samples: number of signal to inject
        waveform_duration: length of injected waveforms
        datadir: Path to store data
        logdir: Path to store logs
        waveform_arguments:
            Additional arguments to pass to waveform generator,
            that will ultimately get passed
            to the waveform callable specified. For example,
            generating BBH waveforms requires the specification of a
            waveform_approximant
    """

    configure_logging(logdir / "generate_waveforms.log", verbose)
    # make data dir
    datadir.mkdir(exist_ok=True, parents=True)
    signal_file = datadir / "signals.h5"

    if signal_file.exists() and not force_generation:
        logging.info(
            "Signal data already exists and forced generation is off. "
            "Not generating signals."
        )
        return signal_file

    priors = prior()

    sample_params = priors.sample(n_samples)

    signals = injection.generate_gw(
        sample_params,
        sample_rate,
        waveform_duration,
        waveform,
        waveform_arguments=waveform_arguments,
        parameter_conversion=parameter_conversion,
    )

    # write signals and parameters used to generate them
    with h5py.File(signal_file, "w") as f:

        f.create_dataset("signals", data=signals)

        for k, v in sample_params.items():
            f.create_dataset(k, data=v)

        # write attributes
        f.attrs.update(
            {
                "size": n_samples,
                "sample_rate": sample_rate,
                "waveform_duration": waveform_duration,
                "waveform": waveform.__name__,
            }
        )
        if waveform_arguments is not None:
            f.attrs.update(waveform_arguments)

    return signal_file


if __name__ == "__main__":
    main()

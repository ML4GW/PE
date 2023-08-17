import logging
from pathlib import Path
from typing import List, Optional

import bilby
import h5py
import numpy as np
from mldatafind.segments import query_segments

from mlpe.deploy import condor
from typeo import scriptify


def post_process(outdir: Path):
    # read in all the bilby results into a list
    # next, read in all the data for the flow
    results, data = [], []
    for run in outdir.iterdir():
        result = bilby.core.result.read_in_result(run / "result.hdf5")
        results.append(result)
        with h5py.File(run / "data.hdf5") as f:
            data.append(f["strain"][:])

    data = np.stack(data)

    results.make_pp_plot()

    with h5py.File(outdir / "bilby_injections.h5", "w") as f:
        f.create_dataset("injections", data=data)


@scriptify
def main(
    start: float,
    stop: float,
    n_samples: int,
    duration: float,
    sample_rate: float,
    fmin: float,
    prior: str,
    waveform: str,
    ifos: List[str],
    channel: str,
    state_flag: str,
    outdir: Path,
    request_cpus: int,
    accounting_group: str,
    accounting_group_user: str,
    min_duration: float = 2048,
    psd_buffer: float = 2,
    post_trigger_duration: float = 1,
    psd_duration: Optional[float] = None,
    nlive: int = 1024,
    nact: int = 5,
    seed: int = 112296,
    request_memory: float = 16384,
    request_disk: float = 8194,
    verbose: bool = False,
):

    segment_names = [f"{ifo}:{state_flag}" for ifo in ifos]
    start, stop = query_segments(
        segment_names,
        start,
        stop,
        min_duration,
    )[0]

    condordir = Path(outdir) / "condor"
    outdir = Path(outdir) / "runs"
    condordir.mkdir(exist_ok=True, parents=True)
    outdir.mkdir(exist_ok=True, parents=True)
    psd_duration = psd_duration or 32 * duration
    np.random.seed(seed)
    # randomly sample times across the start and stop
    times = np.random.uniform(
        start + duration / 2, stop - duration / 2, n_samples
    )

    parameters = "time,label,seed\n"
    for i, time in enumerate(times):
        parameters += f"{time},{i},{seed + i}\n"

    arguments = "--trigger-time $(time) --label $(label) --seed $(seed) "
    arguments += f"--ifos {' '.join(ifos)} --channel {channel} "
    arguments += f"--outdir {outdir} --request-cpus {request_cpus} "
    arguments += f"--nlive {nlive} --nact {nact} "
    arguments += f"--prior {prior} --waveform {waveform} "
    arguments += f"--duration {duration} --sample-rate {sample_rate} "
    arguments += f"--fmin {fmin} "
    arguments += f"--post-trigger-duration {post_trigger_duration} "
    arguments += f"--psd-buffer {psd_buffer} "
    arguments += f"--psd-duration {psd_duration} "

    if verbose:
        arguments += "--verbose "

    subfile = condor.make_submit_file(
        executable="run-bilby",
        name="run-bilby",
        parameters=parameters,
        arguments=arguments,
        submit_dir=condordir,
        accounting_group=accounting_group,
        accounting_group_user=accounting_group_user,
        clear=True,
        requirements="",
        request_disk="5.0GB",
        request_memory="8.0GB",
        request_cpus=request_cpus,
        use_x509userproxy=True,
        environment="OMP_NUM_THREADS=1 USE_HDF5_FILE_LOCKING=FALSE",
        should_transfer_files="YES",
        when_to_transfer_output="ON_EXIT_OR_EVICT",
        stream_error=True,
        stream_output=True,
        transfer_output_files=outdir,
        checkpoint_exit_code=77,
    )
    dag_id = condor.submit(subfile)
    logging.info(f"Launching bilby analyses with dag id {dag_id}")
    condor.watch(dag_id, condordir, held=True)
    logging.info("Completed bilby analysis jobs")

    post_process(outdir)

import os
import re
import shutil
import subprocess
from pathlib import Path
from textwrap import dedent
from typing import Union


def get_executable(name: str) -> str:
    """Get the path to an executable based on its name"""
    ex = shutil.which(name)
    if ex is None:
        raise ValueError(f"No executable {name}")
    return str(ex)


def make_submit_file(
    executable: str,
    name: str,
    parameters: str,
    arguments: str,
    submit_dir: Path,
    accounting_group: str,
    accounting_group_user: str,
    clear: bool = True,
    **kwargs,
):
    """
    Construct a condor `.sub` file

    Args:
        executable:
            The executable to be run with condor. Can be either the
            absolute path or the name of the executable
        name:
            Prefix used for naming the log, output, and error condor files
        parameters:
            Values that condor will use to fill in arguments for each job
        arguments:
            Arguments for the executable
        submit_dir:
            Directory to which condor-related files will be written
        accounting_group:
            Accounting group for the condor jobs
        accounting_group_user:
            Username of the person running the condor jobs
        clear:
            If true, remove any log files from the submit directory
        **kwargs:
            Additional arguments for the submit file
    """
    if not os.path.isabs(executable):
        executable = get_executable(executable)

    param_names, parameters = parameters.split("\n", maxsplit=1)
    with open(submit_dir / "parameters.txt", "w") as f:
        f.write(parameters)

    log_dir = submit_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{name}-$(ProcId)"
    if clear:
        for fname in log_dir.glob("*.log"):
            fname.unlink()

    default_kwargs = {"request_memory": "1024", "request_disk": "1024"}
    default_kwargs.update(kwargs)

    subfile = f"""
        universe = vanilla
        executable = {executable}
        arguments =  {arguments}
        log = {log_dir}/{stem}.log
        output = {log_dir}/{stem}.out
        error = {log_dir}/{stem}.err
        getenv = True
        accounting_group = {accounting_group}
        accounting_group_user = {accounting_group_user}
    """
    subfile = dedent(subfile)
    for key, value in default_kwargs.items():
        subfile += f"{key} = {value}\n"
    subfile += f"queue {param_names} from {submit_dir}/parameters.txt\n"

    fname = submit_dir / f"{name}.sub"
    with open(fname, "w") as f:
        f.write(subfile)
    return fname


def submit(sub_file: Union[str, Path]) -> str:
    """Submit a condor `.sub` file. Return the id associated with the jobs."""
    condor_submit = get_executable("condor_submit")
    cmd = [condor_submit, str(sub_file)]
    out = subprocess.check_output(cmd, text=True)

    # re for extracting cluster id from condor_submit output
    # stolen from pyomicron:
    # https://github.com/ML4GW/pyomicron/blob/master/omicron/condor.py
    dag_id = re.search(r"(?<=submitted\sto\scluster )[0-9]+", out).group(0)

    return dag_id


def check_failed(submit_dir: Path):
    """Crawl through condor logs and count failed jobs"""
    log_dir = submit_dir / "logs"
    failed_jobs, total_jobs = [], 0
    for f in log_dir.glob("*.log"):
        log = f.read_text()
        for line in log.splitlines()[-5:]:
            line = line.strip()
            if line.startswith("Job terminated") and not line.endswith(
                "exit-code 0."
            ):
                err_file = log_dir / (f.stem + ".err")
                failed_jobs.append(str(err_file))
                break
        total_jobs += 1

    if failed_jobs:
        err_files = "\n".join(failed_jobs[:10])
        if len(failed_jobs) > 10:
            err_files += "\n..."
        raise RuntimeError(
            "{}/{} jobs failed, consult error files:\n{}".format(
                len(failed_jobs), total_jobs, err_files
            )
        )


def watch(dag_id: str, submit_dir: Path, held: bool = True):
    """
    Display the status of the condor jobs specified by `dag_id`.
    Once the jobs are complete, check whether any failed.
    """
    cwq = get_executable("condor_watch_q")
    cmd = [
        cwq,
        "-exit",
        "all,done,0",
        "-clusters",
        dag_id,
    ]
    if held:
        cmd.extend(["-exit", "any,held,1"])

    subprocess.check_call(cmd)
    check_failed(submit_dir)

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import bilby
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def cast_samples_as_bilby_result(
    samples: np.ndarray,
    truth: np.ndarray,
    inference_params: List[str],
    priors: "bilby.core.prior.Priordict",
    label: str,
):
    """Cast posterior samples as bilby Result object"""
    # samples shape (1, num_samples, num_params)
    # inference_params shape (1, num_params)

    injections = {k: float(v) for k, v in zip(inference_params, truth)}

    posterior = dict()
    for idx, k in enumerate(inference_params):
        posterior[k] = samples.T[idx].flatten()
    posterior = pd.DataFrame(posterior)

    return bilby.result.Result(
        label=label,
        injection_parameters=injections,
        posterior=posterior,
        search_parameter_keys=inference_params,
        priors=priors,
    )


def plot_mollview(
    ra_samples: np.ndarray,
    dec_samples: np.ndarray,
    nside: int = 32,
    truth: Optional[Tuple[float, float]] = None,
    outpath: Path = None,
):
    """Plot mollview of posterior samples

    Args:
        ra_samples: array of right ascension samples in radians (-pi, pi)
        dec_samples: array of declination samples in radians (-pi/2, pi/2)
        nside: nside parameter for healpy
        truth: tuple of true ra and dec
    """

    # mask out non physical samples;
    ra_samples_mask = (ra_samples > -np.pi) * (ra_samples < np.pi)
    dec_samples += np.pi / 2
    dec_samples_mask = (dec_samples > 0) * (dec_samples < np.pi)

    net_mask = ra_samples_mask * dec_samples_mask
    ra_samples = ra_samples[net_mask]
    dec_samples = dec_samples[net_mask]

    # calculate number of samples in each pixel
    NPIX = hp.nside2npix(nside)
    ipix = hp.ang2pix(nside, dec_samples, ra_samples)
    ipix = np.sort(ipix)
    uniq, counts = np.unique(ipix, return_counts=True)

    # create empty map and then fill in non-zero pix with counts
    m = np.zeros(NPIX)
    m[np.in1d(range(NPIX), uniq)] = counts

    plt.close()
    # plot molleweide
    fig = hp.mollview(m)
    if truth is not None:
        ra_inj, dec_inj = truth
        dec_inj += np.pi / 2
        hp.visufunc.projscatter(
            dec_inj, ra_inj, marker="x", color="red", s=150
        )

    plt.savefig(outpath)

    return fig


def load_and_initialize_flow(
    flow: Callable,
    embedding: Callable,
    model_state_path: Path,
    n_ifos: int,
    strain_dim: int,
    param_dim: int,
    device: str,
):
    model_state = torch.load(model_state_path, map_location=device)
    embedding = embedding((n_ifos, strain_dim))
    flow_obj = flow((param_dim, n_ifos, strain_dim), embedding)
    flow_obj.build_flow(device=device, model_state=model_state)

    return flow_obj


def draw_samples_from_model(
    signal,
    param,
    flow: torch.nn.Module,
    preprocessor: torch.nn.Module,
    inference_params: List[str],
    num_samples_draw: int,
    priors: dict,
    label: str = "testing_samples",
):
    strain, scaled_param = preprocessor(signal, param)
    with torch.no_grad():
        samples = flow.sample([1, num_samples_draw], context=strain)
        descaled_samples = preprocessor.scaler(
            samples[0].transpose(1, 0), reverse=True
        )
    descaled_samples = descaled_samples.unsqueeze(0).transpose(2, 1)
    descaled_res = cast_samples_as_bilby_result(
        descaled_samples.cpu().numpy()[0],
        param.cpu().numpy()[0],
        inference_params,
        priors,
        label=label,
    )
    return descaled_res

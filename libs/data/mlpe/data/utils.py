from typing import Optional

import numpy as np


def sample_kernels(
    x: np.ndarray,
    size: int,
    trigger_distance_size: int = 0,
    N: Optional[int] = None,
    parameters: Optional[np.ndarray] = None,
    center_trigger: bool = True,
) -> np.ndarray:
    """Sample fixed-size kernels from a timeseries array
    Given an array of timeseries `x`, whose 0th dimension
    indexes independent samples and whos last dimension
    indexes time, sample `N` kernels from among the samples
    in `x` with length `size`. If `x` has only a single dimension,
    `N` must be provided and `N` kernels will be sampled from
    that single timeseries
    Args:
        x: array of timeseries to sample from
        size: the size of kernels to sample from `x` (in samples)
        N: The number of samples to generate. If left as `None`,
            a kernel will be generated from each row in `x` in order
            if `x.ndim > 1`, otherwise a `ValueError` will be
            raised.
        trigger_distance_size:
            Maximum amount of samples away from the closest edge
            of the kernel the trigger time (t0) is allowed. A positive
            value allows kernels where the trigger is not in the kernel.
            A value of 0 allows the trigger to lie anywhere in the kernel.
            A negative value sets a 'padding' from either kernel edge where
            trigger cannot lie.
        parameters:
            For signal arrays, the parameters corresponding to the injections
            in x.
        center_trigger:
            If true, the trigger t0 will always lie in the center
            of the kernel
        Returns:
            Array of sampled kernels of size `(n, ..., size)`, where
            `n = len(x) if N is None else N` and `...` represents
            all of `x`'s intermediate dimensions between its first
            and last.
    """

    # if parameters is passed do some checks
    if parameters is not None:
        if len(x) != len(parameters):
            raise ValueError(
                "The number of waveforms and parameters passed is not the same"
            )

    if trigger_distance_size < 0 and abs(trigger_distance_size) >= size / 2:
        raise ValueError(
            "If trigger_distance_size is negative, "
            "its magnitude cannot be more than half the kernel size"
        )

    if len(x.shape) == 1 and N is None:
        raise ValueError(
            "Must specify number of samples N when x is 1 dimensional"
        )
    elif len(x.shape) == 1:
        # this is a 1D array, so we'll just be sampling
        # the N kernels from the existing timeseries
        if N > (len(x) - size):
            raise ValueError(
                "Array of length {} too short to sample {} "
                "kernels of length {}".format(len(x), N, size)
            )

        idx = np.random.choice(len(x) - size, size=N, replace=False)
    elif N is None:
        # x is >1 dimensional, indicating a sample dimension, and
        # we didn't specify N, so sample kernels from each sample
        idx = np.arange(len(x))

    elif N > len(x):
        # x has samples but we asked for more samples than
        # x has, so we'll need to sample with replacement
        idx = np.random.choice(len(x), size=N, replace=True)
    else:
        # x has samples and we asked for a subset of them,
        # so sample without replacmenet
        idx = np.random.choice(len(x), size=N, replace=False)

    # always make sure that the center of x's
    # 1st axis is in the kernel that we sample
    # if we're doing >1D sampling

    # t0 is the center sample

    dim = x.shape[-1]
    min_sample_start = max(dim // 2 + 1 - trigger_distance_size - size, 0)
    right_pad = min(trigger_distance_size, 0)
    max_sample_start = min(dim // 2 - 1 + right_pad, x.shape[-1] - size)

    # now iterate through and grab all the kernels
    # TODO: is there a more array-friendly way of doing this?
    samples = []
    for i in idx:
        # in the 1D case, the idx represent start samples
        # from the timeseries
        if len(x.shape) == 1:
            samples.append(x[i : i + size])
            continue

        # otherwise, i represents an index along x's 0th dimension,
        # and we sample from within a kernel's length of
        # the center of each sampled index, where it is assumed
        # that the "trigger" of the relevant event will live
        if max_sample_start <= 0:
            start = 0
        else:
            start = np.random.randint(min_sample_start, max_sample_start)

        # if center trigger is True
        # set start to half the size
        # before the center sample
        if center_trigger:
            start = (dim // 2) - size // 2
        slc = slice(start, start + size)

        # unfortunately can't think of a cleaner
        # way to make sure we're slicing from the
        # last dimension
        # TODO: won't generalize to >3 dim
        if len(x.shape) == 2:
            samples.append(x[i, slc])
        else:
            samples.append(x[i, :, slc])

    # if we passed parameters
    # get the parameters associated
    # with the sampled waveforms
    if parameters is not None:
        parameters = parameters[idx]
    return samples, parameters

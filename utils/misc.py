# © 2025 NVIDIA CORPORATION & AFFILIATES

"""
Miscellaneous utility functions
"""

import numpy as np
from utils import Sigmoid


def generate_is_nack(sinr,
                     mcs,
                     bler_sigmoid_params,
                     return_bler=False):
    """
    Generate ACK/NACK from given SINR and MCS

    Input:
    ------
        sinr: `float` or `list` of `float`
            SINR value(s)

        mcs: `int` or `list` of `int`
            MCS value(s)

        bler_sigmoid_params: `dict`
            Sigmoid parameters approximating the BLER function

        return_pr_ack: `bool` (default: `False`)
            If `True`, return the probability of an ACK

    Output:
    ------
        is_nack: `int` or `list` of `int`
            Whether HARQ feedback is NACK

        bler: `float` or `list` of `float`
            BLER. Only returned if `return_bler` is `True`
    """
    if not hasattr(sinr, '__len__'):
        sinr = np.array([sinr])
        return_scalar = True
    else:
        sinr = np.array(sinr)
        return_scalar = False
    mcs = np.array([mcs]) if not hasattr(mcs, '__len__') else np.array(mcs)
    assert len(sinr) == len(mcs)
    
    # Compute BLER
    pr_ack_fun = Sigmoid(center=bler_sigmoid_params['center'][mcs],
                         scale=bler_sigmoid_params['scale'][mcs])
    bler = 1 - pr_ack_fun(sinr)

    # Generate ACK/NACK
    is_nack = (np.random.rand(len(sinr)) < bler).astype(int)
    is_nack = is_nack[0] if return_scalar else is_nack
    if return_bler:
        return is_nack, bler
    else:
        return is_nack


def get_bler_sigmoid_params(bler_table_df,
                            cbs,
                            table_index,
                            return_mcs_min_available=False):
    """
    Get BLER sigmoid parameters from a BLER table stored in `bler_table_df`

    Input:
    ------
        bler_table_df: `pandas.DataFrame`
            BLER table

        cbs: `int`
            Code block size

        table_index: `int`
            Table index

    Output:
    ------
        bler_sigmoid_params: `dict`
            Sigmoid parameters approximating the BLER function

        mcs_min_available: `int`
            Minimum MCS index available
    """

    # Select PDSCH and appropriate table index
    bler_table_df = bler_table_df[(bler_table_df['category'] == 'PDSCH') &
                                  (bler_table_df['table_index'] == table_index)]

    if cbs not in bler_table_df['CBS_num_info_bits'].unique():
        raise ValueError(f'Code block size {cbs} not found in the data. ' +
                         f'Available CBS: {bler_table_df["CBS_num_info_bits"].unique()}')

    # Select appropriate code block size
    df1 = bler_table_df[bler_table_df['CBS_num_info_bits'] == cbs]
    df1 = df1.sort_values(by='MCS')

    # Extract sigmoid parameters to approximate BLER tables
    bler_sigmoid_params = {
        'center': df1['sigmoid_center_db'].values,
        'scale': df1['sigmoid_scale_db'].values
    }

    if return_mcs_min_available:
        mcs_min_available = df1['MCS'].min()
        return bler_sigmoid_params, mcs_min_available
    else:
        return bler_sigmoid_params


def run_la(la_algo,
           sinr_hist,
           likelihood_params,
           nack_report_batch_size=1,
           mcs_to_se=None):
    """
    Run the link adaptation algorithm

    Input:
    ------

        la_algo: `object`
            Link adaptation algorithm

        sinr_hist: `list` of `float`
            History of SINR values

        likelihood_params: `dict`
            Sigmoid parameters approximating the BLER function

        nack_report_batch_size: `int`
            ACK/NACK are received in batch every `nack_report_batch_size` slots

        mcs_to_se: `list` of `float` (default: `None`)
            Mapping from MCS to SE

    Output:
    ------

        nack_hist: `list` of `int`
            History of ACK/NACK values

        rate_hist: `list` of `float`
            History of achieved rates. Only returned if `mcs_to_se` is not `None`

        la_algo: `object`
            Link adaptation algorithm after running the simulation

        mcs_hist: `list` of `int`
            History of MCS values

    """

    n_obs = len(sinr_hist)
    # History of control variables and observations
    is_nack_hist = np.zeros(n_obs)
    mcs_hist = np.zeros(n_obs)
    if mcs_to_se is not None:
        rate_hist = np.zeros(n_obs)
    else:
        rate_hist = None

    mcs_curr = la_algo()
    mcs_hist[0] = mcs_curr

    # Initialize counter and wait time
    n_iter_since_last_feedback = 0

    for ii in range(n_obs):
        n_iter_since_last_feedback += 1

        # ACK/NACK observation
        is_nack = generate_is_nack(sinr_hist[ii],
                                   mcs_curr,
                                   likelihood_params)
        is_nack_hist[ii] = is_nack

        # Achieved rate
        if mcs_to_se is not None:
            rate_hist[ii] = (is_nack == 0) * mcs_to_se[mcs_curr]

        # Feedback
        if n_iter_since_last_feedback == nack_report_batch_size:
            is_nack_reported = is_nack_hist[ii -
                                            nack_report_batch_size + 1: ii+1]
            # Reinitialize counter and wait time
            n_iter_since_last_feedback = 0

            # Select new MCS
            mcs_curr = la_algo(is_nack=is_nack_reported)
        else:
            mcs_curr = la_algo()
        if ii < n_obs - 1:
            mcs_hist[ii+1] = mcs_curr

    if mcs_to_se is not None:
        return is_nack_hist, rate_hist, la_algo, mcs_hist
    else:
        return is_nack_hist, la_algo, mcs_hist


def discounted_average(x, discount_factor=.95):
    """
    Compute the `discount_factor`-discounted average of sequence `x`

    Input:
    ------
        x: `list` of `float`
            Sequence

        discount_factor: `float` (default: 0.95)
            Discount factor

    Output:
    ------
        avg: `list` of `float`
            Discounted average of `x`
    """
    x = np.array(x)
    assert (discount_factor > 0) and (discount_factor < 1)
    assert len(x) > 0
    assert len(x.shape) == 1
    avg = np.zeros(len(x))
    avg[0] = x[0]
    for i in range(1, len(x)):
        avg[i] = discount_factor * avg[i-1] + (1 - discount_factor) * x[i]
    return avg


def sliding_window_average(x, window_size):
    """
    Compute the sliding window average of sequence `x` with window size `window_size`

    Input:
    ------
        x: `list` of `float`
            Sequence

        window_size: `int`
            Sliding window size 

    Output:
    ------
        avg: `list` of `float`
            Sliding window average of `x`
    """
    assert len(x) > 0
    assert len(x.shape) == 1

    return np.array([np.mean(x[i-window_size:i]) for i in range(window_size, len(x))])


def rescale(y, bounds, return_coeffs=False):
    """
    Rescale `y` to the range `bounds`

    Input:
    ------
        y: `list` of `float`
            Sequence

        bounds: `tuple` of `float`
            Lower/upper bounds to which `y` is rescaled

    Output:
    ------
        y: `list` of `float`
            Rescaled sequence
    """
    M, m = max(y), min(y)
    if M != m:
        a = (bounds[1] - bounds[0]) / (M - m)
        b = bounds[0] - m * (bounds[1] - bounds[0]) / (M - m)
    else:
        a, b = 1, 0
    y = a * y + b
    if return_coeffs:
        return y, a, b
    else:
        return y


def generate_rect(n_samples, n_jumps, bounds):
    """
    Generate a rectangular function with `n_jumps` jumps between `bounds`

    Input:
    ------
        n_samples: `int`
            Number of samples

        n_jumps: `int`
            Number of jumps

        bounds: `tuple` of `float`
            Lower/upper bounds

    Output:
    ------
        f: `list` of `float`
            Rectangular function
    """
    f = np.ones(n_samples)
    for t in range(n_jumps+1):
        f[t*2*n_samples//(n_jumps+1):(t*2+1)*n_samples//(n_jumps+1)] = 0
    f = rescale(f, bounds)
    return f


def generate_ar(n_samples, coef, std_noise, bounds):
    """
    Generate an AR(1) process
    """
    x = np.zeros(n_samples)
    for t in range(1, n_samples):
        x[t] = coef * x[t-1] + np.random.randn() * std_noise
    x = rescale(x, bounds)
    return x

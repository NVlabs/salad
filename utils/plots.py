# © 2025 NVIDIA CORPORATION & AFFILIATES

"""
Plotting utilities
"""

import matplotlib.pyplot as plt
import numpy as np
from utils import sliding_window_average


def plot_results(sinr_true : list[float]    ,
                 la_algos : dict[str, object],
                 mcs_hist : dict[str, list[int]],
                 is_nack_hist : dict[str, list[int]],
                 rate_hist : dict[str, list[float]],
                 bler_target : float,
                 mcs_oracle : list[int] | None = None,
                 min_available_mcs : int = 0,
                 linewidths : dict[str, float] | None = None,
                 legends : dict[str, str] | None = None,
                 markers : dict[str, str] | None = None,
                 colors : dict[str, str] | None = None,
                 window_size : int =50,
                 markevery : int =30,
                 markersize : int =7,
                 fs : dict[str, float] | None = None,
                 plot_only : list[str] | None = None):
    """
    Plot the results of the link adaptation algorithms

    Input
    -----
        sinr_true: `list` of `float`
            History of true SINR values

        la_algos: `dict` of `object`
            Link adaptation algorithms

        mcs_hist: `dict` of `list` of `int`
            History of MCS values

        is_nack_hist: `dict` of `list` of `int`
            History of ACK/NACK values

        rate_hist: `dict` of `list` of `float`
            History of achieved rates

        bler_target: `float`
            Target BLER

        mcs_oracle: `list` of `int` (default: `None`)
            History of MCS values from an oracle algorithm that knows the true SINR

        mcs_min_available: `int` (default: `0`)
            Minimum MCS index available

        linewidths: `dict` of `float` (default: `None`)
            Linewidths for the plots

        legends: `dict` of `str` (default: `None`)
            Legends for the plots

        legends: `dict` of `str` (default: `None`)
            Legends for the plots

        markers: `dict` of `str` (default: `None`)
            Markers for the plots

        colors: `dict` of `str` (default: `None`)
            Colors for the plots

        window_size: `int` (default: `50`)
            Window size for the sliding average

        markevery: `int` (default: `30`)
            Mark every n-th point for the plots

        markersize: `int` (default: `7`)
            Marker size for the plots

        fs: `dict` of `float` (default: `None`)
            Font size for the plots

        plot_only: `list` of `str` (default: `None`)
            List of labels to plot. If `None`, all labels are plotted

    Output
    ------
        fig: `matplotlib.figure.Figure`
            Figure object
    """
    n_slots = len(sinr_true)
    if plot_only is not None:
        labels = plot_only
    else:
        labels = list(la_algos.keys())
    if fs is None:
        # Font size
        fs = {}
        fs['title'] = 20
        fs['ylabel'] = 17
        fs['xlabel'] = 17
        fs['tick'] = 15
        fs['legend'] = 15
    if linewidths is None:
        linewidths = {}
        for lab in labels:
            linewidths[lab] = 1
    if legends is None:
        legends = {}
        for lab in labels:
            legends[lab] = lab
    if colors is None:
        colors = {}
        for ii, lab in enumerate(labels):
            colors[lab] = f'C{ii}'
    if markers is None:
        markers = {}
        for lab in labels:
            markers[lab] = ''

    fig, axs = plt.subplots(5, 1, figsize=(
        8, 12), height_ratios=[1, 1, 1, .7, .7])

    axs[0].plot(sinr_true, '--r', label='Ground truth')

    for ii, lab in enumerate(labels):

        # SINR (true and estimates)
        axs[0].plot(la_algos[lab].sinr_estimator.sinr_hist,
                    marker=markers[lab], markevery=markevery,
                    markersize=markersize,
                    label=legends[lab], color=colors[lab],
                    linewidth=linewidths[lab])

        # MCS index
        axs[1].plot(mcs_hist[lab] + min_available_mcs,
                    marker=markers[lab], markevery=markevery,
                    markersize=markersize, color=colors[lab], linewidth=linewidths[lab])

        # Spectral efficiency
        rate_avg_win = sliding_window_average(
            rate_hist[lab], window_size=window_size)
        axs[2].plot(rate_avg_win,
                    marker=markers[lab], markevery=markevery,
                    markersize=markersize, color=colors[lab], linewidth=linewidths[lab])

        # BLER (sliding average)
        axs[3].plot(np.arange(window_size, n_slots),
                    sliding_window_average(is_nack_hist[lab],
                                           window_size=window_size),
                    marker=markers[lab], markevery=markevery,
                    markersize=markersize, color=colors[lab], linewidth=linewidths[lab])

        # BLER (long-term average)
        axs[4].plot(np.cumsum(is_nack_hist[lab]) / np.arange(1, n_slots+1),
                    marker=markers[lab], markevery=markevery,
                    markersize=markersize, color=colors[lab], linewidth=linewidths[lab])

    for ax in axs[3:]:
        ax.plot(bler_target * np.ones(n_slots), '--k',
                label=r'Long-term target $\tau$')

    axs[0].set_ylim(min(sinr_true)-5, max(sinr_true)+5)
    axs[0].legend(labelspacing=0.1, fontsize=fs['legend'], loc='upper right',
                  bbox_to_anchor=(1.02, 1.05), handletextpad=.5, handlelength=1.2)
    axs[0].set_ylabel('SINR [dB]', fontsize=fs['ylabel'])
    if mcs_oracle is not None:
        axs[1].plot(mcs_oracle + min_available_mcs, 'r', label='Oracle')
    axs[1].legend(fontsize=fs['legend'])
    axs[1].set_ylabel('MCS index', fontsize=fs['ylabel'])

    axs[2].set_ylabel('SE [bit/s/Hz]', fontsize=fs['ylabel'])

    axs[3].set_ylabel('Short-term \n BLER', fontsize=fs['ylabel'])
    axs[3].legend(labelspacing=0.2, fontsize=fs['legend'])

    axs[4].set_ylabel('Long-term \n BLER', fontsize=fs['ylabel'])
    axs[4].set_ylim(axs[3].get_ylim())

    for ax in axs:
        ax.grid()
        ax.tick_params(axis='both', labelsize=fs['tick'])

    axs[4].set_xlabel('Slot', fontsize=fs['xlabel'])

    fig.tight_layout()
    return fig

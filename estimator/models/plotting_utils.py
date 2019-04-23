#!/usr/bin/env python3
"""
Plotting functions
"""
from typing import List, Optional

import numpy as np
import plotly.graph_objs as go
from estimator.models.toy_dataset import RecordedData
from plotly.offline import plot

__author__ = "Avi"
__version__ = "0.1.0"


# visualising the data
def plot_toy_dataset(
    data_list: List[RecordedData], fname: Optional[str] = "signal_and_raw_data.html"
):

    all_x = np.concatenate([d.x for d in data_list])
    all_y = np.concatenate([d.y for d in data_list])
    all_sig = np.concatenate([d.signal.y for d in data_list])

    traces = []

    true_data = go.Scatter(
        x=all_x, y=all_y, line=dict(dash="dash"), name="data = signal+noise"
    )
    traces.append(true_data)

    for idx, d in enumerate(data_list):
        trace = go.Scatter(x=d.x, y=d.y, mode="markers", name=r"data_{}".format(idx))
        traces.append(trace)

    true_signal = go.Scatter(x=all_x, y=all_sig, mode="lines+markers", name="signal")
    traces.append(true_signal)

    layout = dict(
        title="Visualising the true signal and data",
        xaxis=dict(title="Time (s)"),
        yaxis=dict(title="Amplitude"),
    )
    fig = dict(data=traces, layout=layout)

    plot(fig, filename=fname)

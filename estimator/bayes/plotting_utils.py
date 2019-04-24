#!/usr/bin/env python3
import estimator.bayes.priors as priors
import plotly.graph_objs as go
from plotly.offline import plot


def plot_contour(twod_z_values, title="Post Prob"):
    # Contour data
    contour_data = [
        {
            "z": twod_z_values,
            "colorscale": "Jet",
            "type": u"contour",
            "dx": priors.DM,
            "x0": priors.M_MIN,
            "dy": priors.DC,
            "y0": priors.C_MIN,
        }
    ]
    # True data
    true_lines = [
        {
            "type": "line",
            "x0": priors.M_MIN,
            "y0": priors.C_TRUE,
            "x1": priors.M_MAX,
            "y1": priors.C_TRUE,
        },
        {
            "type": "line",
            "x0": priors.M_TRUE,
            "y0": priors.C_MIN,
            "x1": priors.M_TRUE,
            "y1": priors.C_MAX,
        },
    ]

    true_annotation = dict(
        x=priors.M_TRUE,
        y=priors.C_TRUE,
        xref="x",
        yref="y",
        text="True Vals (m={}, c={})".format(priors.M_TRUE, priors.C_TRUE),
        showarrow=True,
        arrowhead=7,
        ax=0,
        ay=-40,
        font=dict(color="black", size=12),
        bgcolor="#c7c7c7",
        opacity=0.8,
    )

    # Setting preferences
    layout = go.Layout(
        title=title,
        xaxis=dict(title="m", range=[-1.1, -0.8]),
        yaxis=dict(title="c", range=[3, 5.2]),
        showlegend=False,
        shapes=true_lines,
        annotations=[true_annotation],
    )

    # ploting
    fig = go.Figure(data=contour_data, layout=layout)
    plot(fig, filename=title)

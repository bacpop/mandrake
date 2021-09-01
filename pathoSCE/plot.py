# vim: set fileencoding=<utf-8> :
# Copyright 2020 John Lees

'''Methods for making plots of embeddings'''

import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def plotSCE(embedding, names, labels, output_prefix):
    not_noise = labels != -1
    not_noise_list = list(np.where(not_noise)[0])
    plot_df = pd.DataFrame({'x': embedding[not_noise, 0],
                            'y': embedding[not_noise, 1],
                            'names': [names[i] for i in not_noise_list],
                            'cluster': [str(labels[x]) for x in not_noise_list]})

    # Plot clustered points
    fig = px.scatter(plot_df, x="x", y="y",
                     hover_name='names',
                     color='cluster',
                     render_mode='webgl')
    fig.layout.update(showlegend=False)
    fig.update_traces(marker=dict(size=10,
                             line=dict(width=2,
                                       color='DarkSlateGrey')),
                      text=plot_df['names'],
                      hoverinfo="text",
                      opacity=0.8,
                      selector=dict(mode='markers'))
    """ fig = go.Figure(data=go.Scattergl(
            mode='markers',
            x=plot_df['x'],
            y=plot_df['y'],
            marker_color=plot_df['cluster'],
            text=plot_df['names'],
            hoverinfo="text",
            opacity=0.5,
            marker=dict(
                size=15
            ),
            showlegend=False
        )
    ) """

    # Plot noise points
    fig.add_trace(
        go.Scattergl(
            mode='markers',
            x=embedding[labels == -1, 0],
            y=embedding[labels == -1, 1],
            text=[names[i] for i in list(np.where(labels == -1)[0])],
            hoverinfo="text",
            opacity=0.5,
            marker=dict(
                color='black',
                size=8
            ),
            showlegend=False
        )
    )

    fig.write_html(output_prefix + '_SCE_result.html')
    # needs separate library for static image
    try:
        fig.write_image(output_prefix + ".embedding.png", engine="auto")
    except ValueError as e:
        sys.stderr.write("Need to install orca ('plotly-orca') or kaleido "
        "('python-kaleido') to draw png image output")


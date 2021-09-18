# vim: set fileencoding=<utf-8> :
# Copyright 2020 John Lees

'''Methods for making plots of embeddings'''

import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams.update({'font.size': 18})
import matplotlib.pyplot as plt

def plotSCE(embedding, names, labels, output_prefix, dbscan=True):
    if dbscan:
        not_noise = labels != -1
        not_noise_list = list(np.where(not_noise)[0])
        plot_df = pd.DataFrame({'SCE dimension 1': embedding[not_noise, 0],
                                'SCE dimension 2': embedding[not_noise, 1],
                                'names': [names[i] for i in not_noise_list],
                                'Label': [str(labels[x]) for x in not_noise_list]})
    else:
        plot_df = pd.DataFrame({'SCE dimension 1': embedding[:, 0],
                                'SCE dimension 2': embedding[:, 1],
                                'names': names,
                                'Label': [str(x) for x in labels]})

    random_colour_map = {}
    rng = np.random.default_rng(1)
    for label in pd.unique(plot_df['Label']):
      # Alternative approach with hsl representation
      # from hsluv import hsluv_to_hex ## outside of loop
      # hue = rng.uniform(0, 360)
      # saturation = rng.uniform(60, 100)
      # luminosity = rng.uniform(50, 90)
      # random_colour_map[label] = hsluv_to_hex([hue, saturation, luminosity])

      # Random in rbg seems to give better contrast
      rgb = rng.integers(low=0, high=255, size=3)
      random_colour_map[label] = ",".join(["rgb(" + str(rgb[0]),
                                             str(rgb[1]),
                                             str(rgb[2]) + ")"])

    # Plot clustered points
    fig = px.scatter(plot_df, x="SCE dimension 1", y="SCE dimension 2",
                     hover_name='names',
                     color='Label',
                     color_discrete_map=random_colour_map,
                     render_mode='webgl')
    fig.layout.update(showlegend=False)
    fig.update_traces(marker=dict(size=10,
                             line=dict(width=2,
                                       color='DarkSlateGrey')),
                      text=plot_df['names'],
                      hoverinfo="text",
                      opacity=1.0,
                      selector=dict(mode='markers'))
    if dbscan:
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

    fig.write_html(output_prefix + '.embedding.html')
    # needs separate library for static image
    try:
        fig.write_image(output_prefix + ".embedding.png", engine="auto")
    except ValueError as e:
        sys.stderr.write("Need to install orca ('plotly-orca') or kaleido "
        "('python-kaleido') to draw png image output\n")
        sys.stderr.write("Falling back to matplotlib\n")
        plotSCE_static(embedding, labels, output_prefix, dbscan=dbscan)

# Fallback function if kaledio or orca are missing
def plotSCE_static(embedding, labels, output_prefix, dbscan=True):
    # Black removed and is used for noise instead.
    unique_labels = set(labels)

    if embedding.shape[0] > 10000:
        pt_scale = 1
    else:
        pt_scale = 7

    plt.figure(figsize=(11, 8), dpi= 160, facecolor='w', edgecolor='k')
    rng = np.random.default_rng(1)
    for k in unique_labels:
        if k == -1 and dbscan:
            ptsize = 1 * pt_scale
            col = 'k'
            mec = None
            mew = 0
        else:
            ptsize = 2 * pt_scale
            col = tuple(rng.uniform(size=3))
            mec = 'k'
            mew = 0.2 * pt_scale
        class_member_mask = (labels == k)
        xy = embedding[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], '.', color=col, markersize=ptsize, mec=mec, mew=mew)

    # plot output
    if dbscan:
        plt.title('HDBSCAN – estimated number of spatial clusters: %d' % (len(unique_labels) - 1))
    plt.xlabel('SCE dimension 1')
    plt.ylabel('SCE dimension 2')
    plt.savefig(output_prefix + ".embedding.png")
    plt.close()

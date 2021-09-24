# vim: set fileencoding=<utf-8> :
# Copyright 2020 John Lees

'''Methods for making plots of embeddings'''

import sys
import collections
import pandas as pd
import numpy as np
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go

import matplotlib as mpl
mpl.use('Agg')
#mpl.rcParams.update({'font.size': 8})
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation

# Interactive HTML plot using plotly
def plotSCE_html(embedding, names, labels, output_prefix, dbscan=True):
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

# Hexagon density plot to see overplotting
def plotSCE_hex(embedding, output_prefix):
    # Set up figure with scale bar
    plt.figure(figsize=(8, 8), dpi=320, facecolor='w', edgecolor='k')
    ax = plt.subplot()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Hex plot
    hb = ax.hexbin(embedding[:, 0], embedding[:, 1],
                   mincnt=1, gridsize=50, cmap='inferno')

    # Colour bar
    cbar = plt.colorbar(hb, cax=cax)
    cbar.set_label('Samples')

    # Draw the plot
    ax.set_title('Embedding density')
    ax.set_xlabel('SCE dimension 1')
    ax.set_ylabel('SCE dimension 2')
    plt.savefig(output_prefix + ".embedding_density.pdf")

# Matplotlib static plot, and animation if available
def plotSCE_mpl(embedding, results, labels, output_prefix, dbscan=True):
    # Set the style by group
    if embedding.shape[0] > 10000:
        pt_scale = 1
    else:
        pt_scale = 7
    unique_labels = set(labels)
    rng = np.random.default_rng(1)
    style_dict = collections.defaultdict(dict)
    for k in unique_labels:
        if k == -1 and dbscan:
            style_dict['ptsize'][k] = 1 * pt_scale
            style_dict['col'][k] = 'k'
            style_dict['mec'][k] = None
            style_dict['mew'][k] = 0
        else:
            style_dict['ptsize'][k] = 2 * pt_scale
            style_dict['col'][k] = tuple(rng.uniform(size=3))
            style_dict['mec'][k] = 'k'
            style_dict['mew'][k] = 0.2 * pt_scale

    # Static figure is a scatter plot, drawn by class
    plt.figure(figsize=(8, 8), dpi=320, facecolor='w', edgecolor='k')
    for k in unique_labels:
        class_member_mask = (labels == k)
        xy = embedding[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], '.',
                 color=style_dict['col'][k],
                 markersize=style_dict['ptsize'][k],
                 mec=style_dict['mec'][k],
                 mew=style_dict['mew'][k])

    # plot output
    if dbscan:
        plt.title('HDBSCAN – estimated number of spatial clusters: %d' % (len(unique_labels) - 1))
    plt.xlabel('SCE dimension 1')
    plt.ylabel('SCE dimension 2')
    plt.savefig(output_prefix + ".embedding_static.png")
    plt.close()

    # Make animation
    if results.animated():
        sys.stderr.write("Creating animation\n")
        plt.figure(figsize=(13, 11), dpi=160, facecolor='w', edgecolor='k')
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
        ax1.set_xlabel('SCE dimension 1')
        ax1.set_ylabel('SCE dimension 2')
        if dbscan:
            ax1.set_title('HDBSCAN – estimated number of spatial clusters: %d' % (len(unique_labels) - 1))
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Eq')
        ax2.set_ylim(bottom=0)

        ims = []
        iter_series, eq_series = results.get_eq()
        plt.tight_layout()
        for frame in tqdm(range(results.n_frames()), unit="frames"):
            animated = True if frame > 0 else False

            # Eq plot at bottom, for current frame
            eq_im, = ax2.plot(iter_series[0:(frame+1)], eq_series[0:(frame+1)],
                              color='cornflowerblue', lw=1, animated=animated)
            frame_ims = [eq_im]

            # Scatter plot at top, for current frame
            embedding = np.array(results.get_embedding_frame(frame)).reshape(-1, 2)
            _norm_and_centre(embedding)
            for k in unique_labels:
                class_member_mask = (labels == k)
                xy = embedding[class_member_mask]
                im, = ax1.plot(xy[:, 0], xy[:, 1], '.',
                          color=style_dict['col'][k],
                          markersize=style_dict['ptsize'][k],
                          mec=style_dict['mec'][k],
                          mew=style_dict['mew'][k],
                          animated=animated)
                frame_ims.append(im)
            ims.append(frame_ims)

        # Write the animation (list of lists) to an mp4
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                        repeat=False)
        writer = animation.FFMpegWriter(
            fps=20, metadata=dict(title='mandrake animation'), bitrate=-1)
        progress_callback = \
          lambda i, n: sys.stderr.write('Saving frame ' + str(i) + ' of ' + str(len(ims)) + '\r')
        ani.save(output_prefix + ".embedding_animation.mp4", writer=writer,
                dpi=320, progress_callback=progress_callback)
        progress_callback(len(ims), len(ims))
        sys.stderr.write("\n")

# Internal functions

# Transforms the provided array to normalise and centre it
def _norm_and_centre(array):
    # scale 1
    scales = 0.5*(np.amax(array, axis=0) - np.amin(array, axis=0))
    array /= scales
    means = np.mean(array, axis=0)
    array -= means
    # scale 2
    #means = np.mean(array, axis=0)
    #array -= means
    #scales = np.std(array, axis=0) - np.amin(array, axis=0)
    #array /= scales
    # scale 3
    #means = np.mean(array, axis=0)
    #array -= means
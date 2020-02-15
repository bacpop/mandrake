# vim: set fileencoding=<utf-8> :
# Copyright 2020 John Lees

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
                            'cluster': [str(labels[x]) for x in not_noise_list],
                            'size': [0.5 for x in not_noise_list]})
    
    # Plot clustered points
    fig = px.scatter(plot_df, x="x", y="y", 
                     hover_name='names',
                     color='cluster',
                     size='size')
    fig.layout.update(showlegend=False)

    # Plot noise points
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=embedding[labels == -1, 0],
            y=embedding[labels == -1, 1],
            hovertext=[names[i] for i in list(np.where(labels == -1)[0])],
            hoverinfo="text",
            opacity=0.5,
            marker=dict(
                color='black',
                size=15
            ),
            showlegend=False
        )
    )

    # fig.write_image(output_prefix + ".embedding.png")
    fig.show()
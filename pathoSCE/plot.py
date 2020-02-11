# vim: set fileencoding=<utf-8> :
# Copyright 2020 John Lee

import pandas as pd
import plotly.express as px

def plotSCE(embedding, names, output_prefix):
    plot_df = pd.DataFrame({'x': embedding[:, 0],
                            'y': embedding[:, 1],
                            'names': names})
    fig = px.scatter(plot_df, x="x", y="y", hover_name='names')
    fig.write_image(output_prefix + ".embedding.png")
    fig.show()
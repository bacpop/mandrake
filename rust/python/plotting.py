"""Plotting adapter embedded by Mandrake's Rust CLI.

The plotting behavior follows the original mandrake/plot.py and
mandrake/clustering.py implementation, but accepts plain Python arrays and a
portable NPZ frame archive instead of the legacy C++ result object.
"""

import operator
from collections import defaultdict

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm


ARCHIVE_VERSION = 1


def save_frame_archive(embeddings, iterations, worker_updates, eq, path):
    frames = np.asarray(embeddings, dtype=np.float64)
    if frames.ndim != 3 or frames.shape[2] != 2:
        raise ValueError("embeddings must have shape (frames, samples, 2)")
    if frames.shape[0] < 2:
        raise ValueError("animation archive requires at least two frames")
    iterations = np.asarray(iterations, dtype=np.uint64)
    worker_updates = np.asarray(worker_updates, dtype=np.uint64)
    eq = np.asarray(eq, dtype=np.float64)
    if not (len(iterations) == len(worker_updates) == len(eq) == frames.shape[0]):
        raise ValueError("frame metadata lengths do not match embeddings")
    np.savez_compressed(
        path,
        format_version=np.asarray(ARCHIVE_VERSION, dtype=np.uint64),
        embeddings=frames,
        iterations=iterations,
        worker_updates=worker_updates,
        eq=eq,
    )


def _load_frame_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        version = int(np.asarray(archive["format_version"]))
        if version != ARCHIVE_VERSION:
            raise ValueError("unsupported animation archive version %d" % version)
        frames = np.asarray(archive["embeddings"], dtype=np.float64)
        iterations = np.asarray(archive["iterations"], dtype=np.uint64)
        worker_updates = np.asarray(archive["worker_updates"], dtype=np.uint64)
        eq = np.asarray(archive["eq"], dtype=np.float64)
    if frames.ndim != 3 or frames.shape[2] != 2 or frames.shape[0] < 2:
        raise ValueError("animation archive embeddings have an invalid shape")
    if not (len(iterations) == len(worker_updates) == len(eq) == frames.shape[0]):
        raise ValueError("animation archive metadata lengths do not match embeddings")
    if not np.isfinite(frames).all() or not np.isfinite(eq).all():
        raise ValueError("animation archive contains non-finite values")
    return frames, worker_updates, eq


def _normalise_and_centre(array):
    array = np.asarray(array, dtype=np.float64).copy()
    array -= np.mean(array, axis=0)
    scales = np.std(array, axis=0)
    scales[scales == 0] = 1.0
    array /= scales
    return array


def _cluster_labels(embedding):
    import hdbscan

    scaled = _normalise_and_centre(embedding)
    hdb = hdbscan.HDBSCAN(
        algorithm="boruvka_balltree",
        min_cluster_size=2,
        min_samples=2,
        cluster_selection_epsilon=0.02,
        allow_single_cluster=True,
    ).fit(scaled)
    return hdb.labels_


def _write_hdbscan_clusters(labels, names, output_prefix):
    pd.DataFrame(
        {
            "id": list(names),
            "hdbscan_cluster__autocolour": list(labels),
        }
    ).to_csv(output_prefix + ".embedding_hdbscan_clusters.csv", index=False)


def _colour_map(labels, embedding_size, dbscan, seed):
    unique_labels = list(set(labels.tolist()))
    rng = np.random.default_rng(seed=seed)
    style = defaultdict(dict)
    for label in sorted(unique_labels, key=str):
        if label == -1 and dbscan:
            style["ptsize"][label] = 1 if embedding_size > 10000 else (1.5 if embedding_size > 1000 else 7)
            style["col"][label] = "k"
            style["mec"][label] = None
            style["mew"][label] = 0
        else:
            pt_scale = 1.5 if embedding_size > 10000 else (3 if embedding_size > 1000 else 7)
            style["ptsize"][label] = 2 * pt_scale
            style["col"][label] = tuple(rng.uniform(size=3))
            style["mec"][label] = "k" if embedding_size <= 10000 else None
            style["mew"][label] = 0.2 * pt_scale if embedding_size <= 10000 else 0
    return style, unique_labels


def _plot_html(embedding, names, labels, output_prefix, hover_labels, dbscan, seed):
    labels = np.asarray(labels)
    names = list(names)
    if dbscan:
        not_noise = labels != -1
        indices = list(np.where(not_noise)[0])
        plot_df = pd.DataFrame(
            {
                "SCE dimension 1": embedding[not_noise, 0],
                "SCE dimension 2": embedding[not_noise, 1],
                "names": [names[i] for i in indices],
                "Label": [str(labels[i]) for i in indices],
            }
        )
    else:
        plot_df = pd.DataFrame(
            {
                "SCE dimension 1": embedding[:, 0],
                "SCE dimension 2": embedding[:, 1],
                "names": names,
                "Label": [str(label) for label in labels],
            }
        )

    rng = np.random.default_rng(seed=seed)
    colour_map = {}
    for label in sorted(pd.unique(plot_df["Label"]), key=str):
        rgb = rng.integers(low=0, high=255, size=3)
        colour_map[label] = "rgb(" + str(rgb[0]) + "," + str(rgb[1]) + "," + str(rgb[2]) + ")"

    fig = px.scatter(
        plot_df,
        x="SCE dimension 1",
        y="SCE dimension 2",
        hover_name="names" if hover_labels else None,
        color="Label",
        color_discrete_map=colour_map,
        render_mode="webgl",
    )
    fig.layout.update(showlegend=False)
    fig.update_traces(
        marker=dict(size=10, line=dict(width=2, color="DarkSlateGrey")),
        text=plot_df["names"] if hover_labels else None,
        hoverinfo="text" if hover_labels else None,
        opacity=1.0,
        selector=dict(mode="markers"),
    )
    if dbscan:
        noise = labels == -1
        fig.add_trace(
            go.Scattergl(
                mode="markers",
                x=embedding[noise, 0],
                y=embedding[noise, 1],
                text=[names[i] for i in list(np.where(noise)[0])] if hover_labels else None,
                hoverinfo="text" if hover_labels else None,
                opacity=0.5,
                marker=dict(color="black", size=8),
                showlegend=False,
            )
        )
    fig.write_html(output_prefix + ".embedding.html")


def _plot_density(embedding, output_prefix):
    plt.figure(figsize=(8, 8), dpi=320, facecolor="w", edgecolor="k")
    ax = plt.subplot()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    hb = ax.hexbin(embedding[:, 0], embedding[:, 1], mincnt=1, gridsize=50, cmap="inferno")
    cbar = plt.colorbar(hb, cax=cax)
    cbar.set_label("Samples")
    ax.set_title("Embedding density")
    ax.set_xlabel("SCE dimension 1")
    ax.set_ylabel("SCE dimension 2")
    plt.savefig(output_prefix + ".embedding_density.pdf")
    plt.close()


def _plot_static(embedding, labels, output_prefix, dbscan, seed):
    style, unique_labels = _colour_map(labels, embedding.shape[0], dbscan, seed)
    plt.figure(figsize=(8, 8), dpi=320, facecolor="w", edgecolor="k")
    for label in sorted(unique_labels, key=str):
        xy = embedding[labels == label]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            ".",
            color=style["col"][label],
            markersize=style["ptsize"][label],
            mec=style["mec"][label],
            mew=style["mew"][label],
        )
    if dbscan:
        plt.title("HDBSCAN – estimated number of spatial clusters: %d" % (len(unique_labels) - 1))
    plt.xlabel("SCE dimension 1")
    plt.ylabel("SCE dimension 2")
    plt.savefig(output_prefix + ".embedding_static.png")
    plt.close()


def _plot_animation(frames, worker_updates, eq, labels, output_prefix, dbscan, seed):
    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError("FFmpeg is required for embedding animation output")
    style, unique_labels = _colour_map(labels, frames.shape[1], dbscan, seed)
    fig = plt.figure(facecolor="k", edgecolor="w", constrained_layout=True)
    fig.set_size_inches(16, 8, True)
    grid = fig.add_gridspec(2, 2)
    ax_em = fig.add_subplot(grid[:, 0])
    ax_em.set_xlabel("SCE dimension 1")
    ax_em.set_ylabel("SCE dimension 2")
    ax_eq = fig.add_subplot(grid[1, 1])
    ax_eq.set_xlabel("Worker updates")
    ax_eq.set_ylabel("Eq")
    ax_eq.set_ylim(bottom=0)
    ax_leg = fig.add_subplot(grid[0, 1])
    ax_leg.axis("off")

    cluster_sizes = sorted(
        ((label, int(np.sum(labels == label))) for label in unique_labels),
        key=operator.itemgetter(1),
        reverse=True,
    )
    for index, (label, size) in enumerate(cluster_sizes):
        style["label"][label] = str(label) + " (" + str(size) + ")" if index < 30 else None

    images = []
    for frame_index in tqdm(range(frames.shape[0]), unit="frames"):
        animated = frame_index > 0
        eq_image, = ax_eq.plot(
            worker_updates[: frame_index + 1],
            eq[: frame_index + 1],
            color="cornflowerblue",
            lw=2,
            animated=animated,
        )
        frame_images = [eq_image]
        current = _normalise_and_centre(frames[frame_index])
        for label in unique_labels:
            xy = current[labels == label]
            image, = ax_em.plot(
                xy[:, 0],
                xy[:, 1],
                ".",
                color=style["col"][label],
                markersize=style["ptsize"][label],
                mec=style["mec"][label],
                mew=style["mew"][label],
                label=style["label"][label],
                animated=animated,
            )
            frame_images.append(image)
        if frame_index == 0:
            handles, labels_for_legend = ax_em.get_legend_handles_labels()
            legend = ax_leg.legend(
                handles,
                labels_for_legend,
                borderaxespad=0,
                loc="center",
                ncol=4,
                markerscale=7,
                mode="expand",
                title="30 largest classes (size)",
            )
        frame_images.append(legend)
        images.append(frame_images)

    ani = animation.ArtistAnimation(fig, images, interval=50, blit=True, repeat=False)
    writer = animation.FFMpegWriter(
        fps=20,
        metadata=dict(title="Mandrake animation"),
        bitrate=-1,
    )
    ani.save(output_prefix + ".embedding_animation.mp4", writer=writer, dpi=320)
    plt.close(fig)


def render_all(
    embedding,
    names,
    labels,
    output_prefix,
    no_clustering,
    hover_labels,
    seed,
    archive_path,
):
    embedding = np.asarray(embedding, dtype=np.float64)
    names = list(names)
    if labels is None:
        if no_clustering:
            labels = np.full((embedding.shape[0],), -1, dtype=np.int64)
            dbscan = False
        else:
            labels = _cluster_labels(embedding)
            dbscan = True
            _write_hdbscan_clusters(labels, names, output_prefix)
    else:
        labels = np.asarray(labels, dtype=object)
        dbscan = False
    if len(labels) != embedding.shape[0]:
        raise ValueError("label count does not match embedding rows")

    _plot_html(embedding, names, labels, output_prefix, hover_labels, dbscan, seed)
    _plot_density(embedding, output_prefix)
    _plot_static(embedding, labels, output_prefix, dbscan, seed)
    if archive_path is not None:
        frames, worker_updates, eq = _load_frame_archive(archive_path)
        if frames.shape[1] != embedding.shape[0]:
            raise ValueError("animation frame sample count does not match embedding")
        _plot_animation(frames, worker_updates, eq, labels, output_prefix, dbscan, seed)

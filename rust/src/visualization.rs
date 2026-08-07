use anyhow::{Context, Result, anyhow, bail};
use mandrake::SceResults;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

const PLOT_CODE: &str = include_str!("../python/plotting.py");

pub fn save_frame_archive(output: &Path, results: &SceResults) -> Result<()> {
    let mut embeddings = Vec::with_capacity(results.frames().len());
    let mut iterations = Vec::with_capacity(results.frames().len());
    let mut worker_updates = Vec::with_capacity(results.frames().len());
    let mut eq = Vec::with_capacity(results.frames().len());

    for frame in results.frames() {
        let shape = frame.embedding().shape();
        if shape.len() != 2 || shape[1] != 2 {
            bail!("animation frame has unexpected shape {:?}", shape);
        }
        embeddings.push(
            frame
                .embedding()
                .rows()
                .into_iter()
                .map(|row| vec![row[0], row[1]])
                .collect::<Vec<_>>(),
        );
        iterations.push(u64::try_from(frame.iteration()).context("frame iteration exceeds u64")?);
        worker_updates.push(
            u64::try_from(frame.worker_updates()).context("worker update count exceeds u64")?,
        );
        eq.push(frame.eq());
    }

    let archive = frame_archive_path(output);
    let archive_string = archive.to_string_lossy().to_string();
    with_python(|py| {
        let module = plotting_module(py)?;
        module.getattr("save_frame_archive")?.call1((
            embeddings,
            iterations,
            worker_updates,
            eq,
            archive_string,
        ))?;
        Ok(())
    })
    .with_context(|| format!("writing frame archive {}", archive.display()))
}

pub fn run_plot(
    input_prefix: &Path,
    output_prefix: Option<&Path>,
    labels_path: Option<&Path>,
    no_clustering: bool,
    no_html_labels: bool,
    animate: bool,
    seed: u64,
) -> Result<()> {
    let output_prefix = output_prefix.unwrap_or(input_prefix);
    let (embedding, names) = read_embedding(input_prefix)?;
    let labels = labels_path
        .map(|path| read_labels(path, &names))
        .transpose()?;
    let archive = if animate {
        let path = frame_archive_path(input_prefix);
        if !path.is_file() {
            bail!(
                "animation archive {} is missing; rerun embedding with --save-animation",
                path.display()
            );
        }
        Some(path)
    } else {
        None
    };
    let output_string = output_prefix.to_string_lossy().to_string();
    let archive_string = archive
        .as_ref()
        .map(|path| path.to_string_lossy().to_string());

    with_python(|py| {
        let module = plotting_module(py)?;
        let labels = labels.unwrap_or_default();
        let labels_object = if labels.is_empty() {
            py.None()
        } else {
            labels.into_py(py)
        };
        let archive_object = archive_string
            .map(|path| path.into_py(py))
            .unwrap_or_else(|| py.None());
        module.getattr("render_all")?.call1((
            embedding,
            names,
            labels_object,
            output_string,
            no_clustering,
            !no_html_labels,
            seed,
            archive_object,
        ))?;
        Ok(())
    })
    .context("running Python visualization")
}

fn with_python<T>(call: impl for<'py> FnOnce(Python<'py>) -> PyResult<T>) -> Result<T> {
    Python::with_gil(call).map_err(|error| anyhow!("{error}"))
}

fn plotting_module<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
    PyModule::from_code_bound(py, PLOT_CODE, "mandrake_rust_plot.py", "mandrake_rust_plot")
}

fn frame_archive_path(prefix: &Path) -> PathBuf {
    PathBuf::from(format!("{}.embedding_frames.npz", prefix.display()))
}

fn read_embedding(prefix: &Path) -> Result<(Vec<Vec<f64>>, Vec<String>)> {
    let embedding_path = PathBuf::from(format!("{}.embedding.txt", prefix.display()));
    let names_path = PathBuf::from(format!("{}.names.txt", prefix.display()));
    let embedding_file = File::open(&embedding_path)
        .with_context(|| format!("opening embedding file {}", embedding_path.display()))?;
    let mut embedding = Vec::new();
    for (line_number, line) in BufReader::new(embedding_file).lines().enumerate() {
        let line = line.with_context(|| format!("reading {}", embedding_path.display()))?;
        let values = line.split_whitespace().collect::<Vec<_>>();
        if values.len() != 2 {
            bail!(
                "{} line {} must contain exactly two coordinates",
                embedding_path.display(),
                line_number + 1
            );
        }
        let x = values[0].parse::<f64>().with_context(|| {
            format!(
                "parsing x coordinate on {} line {}",
                embedding_path.display(),
                line_number + 1
            )
        })?;
        let y = values[1].parse::<f64>().with_context(|| {
            format!(
                "parsing y coordinate on {} line {}",
                embedding_path.display(),
                line_number + 1
            )
        })?;
        if !x.is_finite() || !y.is_finite() {
            bail!(
                "{} line {} contains a non-finite coordinate",
                embedding_path.display(),
                line_number + 1
            );
        }
        embedding.push(vec![x, y]);
    }

    let names_file = File::open(&names_path)
        .with_context(|| format!("opening names file {}", names_path.display()))?;
    let names = BufReader::new(names_file)
        .lines()
        .map(|line| line.with_context(|| format!("reading {}", names_path.display())))
        .collect::<Result<Vec<_>>>()?;
    if embedding.is_empty() || embedding.len() != names.len() {
        bail!(
            "embedding/name count mismatch for {} ({} coordinates, {} names)",
            prefix.display(),
            embedding.len(),
            names.len()
        );
    }
    let mut seen = HashSet::with_capacity(names.len());
    for name in &names {
        if name.is_empty() || !seen.insert(name) {
            bail!("names file contains an empty or duplicate sample name: {name:?}");
        }
    }
    Ok((embedding, names))
}

fn read_labels(path: &Path, names: &[String]) -> Result<Vec<String>> {
    let file =
        File::open(path).with_context(|| format!("opening labels file {}", path.display()))?;
    let mut values = HashMap::new();
    for (line_number, line) in BufReader::new(file).lines().enumerate() {
        let line = line.with_context(|| format!("reading labels file {}", path.display()))?;
        let mut fields = line.splitn(2, '\t');
        let name = fields.next().unwrap_or_default();
        let label = fields.next().ok_or_else(|| {
            anyhow!(
                "{} line {} must contain sample name and label separated by a tab",
                path.display(),
                line_number + 1
            )
        })?;
        if name.is_empty() || label.is_empty() {
            bail!(
                "{} line {} contains an empty sample name or label",
                path.display(),
                line_number + 1
            );
        }
        if values.insert(name.to_owned(), label.to_owned()).is_some() {
            bail!("{} contains duplicate sample name {name:?}", path.display());
        }
    }
    if values.len() != names.len() || names.iter().any(|name| !values.contains_key(name)) {
        bail!(
            "labels file {} does not cover the embedding names exactly",
            path.display()
        );
    }
    Ok(names.iter().map(|name| values[name].clone()).collect())
}

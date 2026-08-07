use anyhow::{Context, Result, bail};
use clap::{ArgGroup, Parser};
use mandrake::{
    SketchOptions, SparseDistances, Sparsification, WtsneOptions, accessory_distances,
    pair_snp_distances, sketch_distances, sketch_distances_from_fasta_list, wtsne,
};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Parser)]
#[command(
    name = "mandrake",
    about = "Embed sparse genomic distances with Mandrake"
)]
#[command(group(
    ArgGroup::new("input")
        .required(true)
        .args(["alignment", "accessory", "sketches"])
))]
#[command(group(
    ArgGroup::new("sparsification")
        .required(true)
        .args(["knn", "threshold"])
))]
struct Args {
    /// Multiple-sequence FASTA alignment (plain or compressed).
    #[arg(long, value_name = "FILE")]
    alignment: Option<PathBuf>,
    /// Binary Roary-style accessory table (plain or .bz2).
    #[arg(long, value_name = "FILE")]
    accessory: Option<PathBuf>,
    /// Sketch prefix (.skm/.skd) or a text file containing one FASTA path per line.
    #[arg(long, value_name = "PREFIX_OR_LIST")]
    sketches: Option<PathBuf>,
    /// Number of neighbours to retain per sample; accepts the legacy --kNN spelling.
    #[arg(long, visible_alias = "kNN", value_name = "N")]
    knn: Option<usize>,
    /// Strict normalized distance threshold (not supported for sketches).
    #[arg(long, value_name = "DISTANCE")]
    threshold: Option<f64>,
    /// Use accessory rather than core sketch distances.
    #[arg(long)]
    use_accessory: bool,
    /// Output prefix for embedding and names files.
    #[arg(long, default_value = "mandrake", value_name = "PREFIX")]
    output: PathBuf,
    /// Conditional-probability perplexity; non-positive values use raw similarities.
    #[arg(long, default_value_t = 15.0)]
    perplexity: f64,
    /// Maximum optimisation iterations.
    #[arg(long, default_value_t = 100_000)]
    max_iterations: usize,
    /// Repulsion samples per worker and iteration.
    #[arg(long, default_value_t = 5)]
    repulsion_samples: usize,
    /// Initial learning rate.
    #[arg(long, default_value_t = 1.0)]
    learning_rate: f64,
    /// Apply initial attraction exaggeration.
    #[arg(long)]
    initial_exaggeration: bool,
    /// Number of optimisation workers.
    #[arg(long, default_value_t = 1)]
    workers: usize,
    /// Random seed.
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Disable the progress bar.
    #[arg(long)]
    no_progress: bool,
    /// K-mer size used when --sketches points to a FASTA list.
    #[arg(long, default_value_t = 21)]
    sketch_kmer: usize,
    /// Sketch bins used when --sketches points to a FASTA list.
    #[arg(long, default_value_t = 1_000)]
    sketch_size: u64,
}

pub fn run() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    let sparsification = parse_sparsification(&args)?;
    let distances = build_distances(&args, sparsification)?;

    let weights = vec![1.0; distances.n_samples()];
    let options = WtsneOptions {
        perplexity: args.perplexity,
        max_iterations: args.max_iterations,
        repulsion_samples: args.repulsion_samples,
        learning_rate: args.learning_rate,
        initial_exaggeration: args.initial_exaggeration,
        workers: args.workers,
        progress: !args.no_progress,
        seed: args.seed,
        ..WtsneOptions::default()
    };
    let results = wtsne(
        distances.rows(),
        distances.columns(),
        distances.distances(),
        &weights,
        &options,
    )
    .context("running wtsne")?;
    write_outputs(&args.output, &distances, results.embedding())?;
    Ok(())
}

fn parse_sparsification(args: &Args) -> Result<Sparsification> {
    match (args.knn, args.threshold) {
        (Some(k), None) if k > 0 => Ok(Sparsification::Knn(k)),
        (None, Some(threshold)) if threshold.is_finite() && threshold > 0.0 && threshold <= 1.0 => {
            Ok(Sparsification::Threshold(threshold))
        }
        (Some(_), None) => bail!("--knn must be greater than zero"),
        (None, Some(_)) => bail!("--threshold must be finite and in (0, 1]"),
        _ => bail!("specify exactly one of --knn or --threshold"),
    }
}

fn build_distances(args: &Args, sparsification: Sparsification) -> Result<SparseDistances> {
    match (&args.alignment, &args.accessory, &args.sketches) {
        (Some(path), None, None) => pair_snp_distances(path, sparsification),
        (None, Some(path), None) => {
            if args.use_accessory {
                bail!("--use-accessory is only valid with --sketches")
            }
            accessory_distances(path, sparsification)
        }
        (None, None, Some(path)) => {
            if matches!(sparsification, Sparsification::Threshold(_)) {
                bail!("threshold sparsification is not supported for sketch inputs; use --knn")
            }
            let options = SketchOptions {
                kmer_sizes: vec![args.sketch_kmer],
                sketch_size: args.sketch_size,
            };
            if sketch_prefix_exists(path) {
                sketch_distances(path, sparsification, args.use_accessory)
            } else {
                let files = read_fasta_list(path)?;
                sketch_distances_from_fasta_list(
                    &files,
                    sparsification,
                    args.use_accessory,
                    &options,
                )
            }
        }
        _ => bail!("specify exactly one input source"),
    }
}

fn sketch_prefix_exists(path: &Path) -> bool {
    let mut prefix = path.to_path_buf();
    if prefix
        .extension()
        .is_some_and(|extension| extension == "skm" || extension == "skd")
    {
        prefix.set_extension("");
        let mut value = prefix.to_string_lossy().to_string();
        while value.ends_with('.') {
            value.pop();
        }
        prefix = PathBuf::from(value);
    }
    let mut metadata = prefix.clone();
    metadata.set_extension("skm");
    let mut data = prefix;
    data.set_extension("skd");
    metadata.is_file() && data.is_file()
}

fn read_fasta_list(path: &Path) -> Result<Vec<PathBuf>> {
    let file =
        File::open(path).with_context(|| format!("opening FASTA list {}", path.display()))?;
    let mut files = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line.context("reading FASTA list")?;
        let line = line.trim();
        if !line.is_empty() && !line.starts_with('#') {
            files.push(PathBuf::from(line));
        }
    }
    if files.len() < 2 {
        bail!("FASTA list must contain at least two paths");
    }
    Ok(files)
}

fn write_outputs(
    output: &Path,
    distances: &SparseDistances,
    embedding: &ndarray::Array2<f64>,
) -> Result<()> {
    let embedding_path = PathBuf::from(format!("{}.embedding.txt", output.display()));
    let names_path = PathBuf::from(format!("{}.names.txt", output.display()));
    let mut embedding_file = File::create(&embedding_path)
        .with_context(|| format!("creating {}", embedding_path.display()))?;
    for row in embedding.rows() {
        writeln!(embedding_file, "{:.17e}\t{:.17e}", row[0], row[1])?;
    }
    let mut names_file =
        File::create(&names_path).with_context(|| format!("creating {}", names_path.display()))?;
    for name in distances.names() {
        writeln!(names_file, "{name}")?;
    }
    Ok(())
}

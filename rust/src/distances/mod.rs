//! Distance construction for the supported Mandrake input formats.

use anyhow::{Context, Result, anyhow, bail};
use needletail::parse_fastx_file;
use rayon::prelude::*;
use sketchlib::distances;
use sketchlib::distances::distance_matrix::DistVec;
use sketchlib::hashing::HashType;
use sketchlib::io::NeedletailIterator;
use sketchlib::sketch::multisketch::MultiSketch;
use sketchlib::sketch::{SketchingOpts, sketch_data};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

#[allow(dead_code)]
mod gene;

use self::gene::SampleBases;

/// A sparse distance matrix in coordinate (COO) form.
///
/// `rows`, `columns`, and `distances` have equal lengths. Indices are
/// zero-based and correspond to entries in `names`; distances are finite and
/// normalized to the inclusive range `[0, 1]`.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseDistances {
    /// Sample names in row/column order.
    pub names: Vec<String>,
    /// Zero-based COO row indices.
    pub rows: Vec<u64>,
    /// Zero-based COO column indices.
    pub columns: Vec<u64>,
    /// Normalized distances corresponding to `rows` and `columns`.
    pub distances: Vec<f64>,
}

impl SparseDistances {
    /// Construct a sparse matrix after validating its COO vectors.
    pub fn new(
        names: Vec<String>,
        rows: Vec<u64>,
        columns: Vec<u64>,
        distances: Vec<f64>,
    ) -> Result<Self> {
        if rows.len() != columns.len() || rows.len() != distances.len() {
            bail!("COO rows, columns, and distances must have the same length");
        }
        if names.len() < 2 {
            bail!("at least two sample names are required");
        }
        if rows
            .iter()
            .chain(&columns)
            .any(|&index| index >= names.len() as u64)
        {
            bail!("COO index is outside the sample-name vector");
        }
        if distances
            .iter()
            .any(|&distance| !distance.is_finite() || !(0.0..=1.0).contains(&distance))
        {
            bail!("distances must be finite and normalized to [0, 1]");
        }
        Ok(Self {
            names,
            rows,
            columns,
            distances,
        })
    }

    /// Sample names in the row/column order of this matrix.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Zero-based COO row indices.
    pub fn rows(&self) -> &[u64] {
        &self.rows
    }

    /// Zero-based COO column indices.
    pub fn columns(&self) -> &[u64] {
        &self.columns
    }

    /// Normalized distances corresponding to [`Self::rows`] and columns.
    pub fn distances(&self) -> &[f64] {
        &self.distances
    }

    /// Number of samples represented by this matrix.
    pub fn n_samples(&self) -> usize {
        self.names.len()
    }

    /// Number of retained COO edges.
    pub fn len(&self) -> usize {
        self.distances.len()
    }

    /// Whether the matrix contains no retained edges.
    pub fn is_empty(&self) -> bool {
        self.distances.is_empty()
    }

    /// Consume the matrix into names and COO vectors.
    pub fn into_parts(self) -> (Vec<String>, Vec<u64>, Vec<u64>, Vec<f64>) {
        (self.names, self.rows, self.columns, self.distances)
    }
}

/// Selects which edges are retained by a distance constructor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sparsification {
    /// Keep the `k` nearest neighbours per sample. A value of zero means all
    /// other samples, matching the legacy command-line convention.
    Knn(usize),
    /// Keep edges whose source-specific distance is below the threshold.
    Threshold(f64),
}

/// Options used when sketching a FASTA list in memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SketchOptions {
    /// K-mer sizes used for the generated sketch.
    pub kmer_sizes: Vec<usize>,
    /// Number of sketch bins.
    pub sketch_size: u64,
}

impl Default for SketchOptions {
    fn default() -> Self {
        Self {
            kmer_sizes: vec![21],
            sketch_size: 1_000,
        }
    }
}

/// Calculate normalized pair-SNP distances from a multiple-sequence FASTA.
pub fn pair_snp_distances<P: AsRef<Path>>(
    path: P,
    sparsification: Sparsification,
) -> Result<SparseDistances> {
    validate_sparsification(sparsification)?;
    let (names, sequences, alignment_len) = read_alignment(path)?;
    let n = names.len();
    let rows = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut comparisons = Vec::with_capacity(n);
            for j in 0..n {
                let matches = sequences[i].matching_sites(&sequences[j]).len() as usize;
                let gaps = sequences[i].either_gap_sites(&sequences[j]).len() as usize;
                let comparable = alignment_len.saturating_sub(gaps);
                let mismatches = comparable.saturating_sub(matches);
                comparisons.push((j, mismatches));
            }
            select_alignment_row(i, comparisons, alignment_len, sparsification)
        })
        .collect::<Vec<_>>();

    let (rows, columns, distances) = flatten_rows(rows);
    SparseDistances::new(names, rows, columns, distances)
}

/// Calculate Jaccard distances from a binary Roary-style `.Rtab` table.
pub fn accessory_distances<P: AsRef<Path>>(
    path: P,
    sparsification: Sparsification,
) -> Result<SparseDistances> {
    validate_sparsification(sparsification)?;
    let (names, columns) = read_accessory_table(path)?;
    let n = names.len();
    let rows = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut candidates = Vec::with_capacity(n.saturating_sub(1));
            for j in 0..n {
                if i != j {
                    candidates.push((j, jaccard_distance(&columns[i], &columns[j])));
                }
            }
            select_distance_row(i, candidates, sparsification)
        })
        .collect::<Vec<_>>();

    let (rows, columns, distances) = flatten_rows(rows);
    SparseDistances::new(names, rows, columns, distances)
}

/// Calculate core or accessory distances from an existing sketchlib database.
///
/// Sketchlib 0.4.1 exposes a sparse kNN implementation but no streaming
/// threshold operation, so threshold sparsification is rejected here.
pub fn sketch_distances<P: AsRef<Path>>(
    prefix: P,
    sparsification: Sparsification,
    use_accessory: bool,
) -> Result<SparseDistances> {
    if matches!(sparsification, Sparsification::Threshold(_)) {
        bail!("threshold sparsification is not supported for sketch inputs; use --knn");
    }
    let prefix = normalize_sketch_prefix(prefix.as_ref());
    let sketches = MultiSketch::load(
        prefix
            .to_str()
            .ok_or_else(|| anyhow!("sketch prefix is not valid UTF-8"))?,
    )
    .with_context(|| format!("loading sketch database {}", prefix.display()))?;
    sparse_sketch_distances(&sketches, sparsification, use_accessory)
}

/// Calculate sketch distances from one FASTA/FASTQ file per sample.
pub fn sketch_distances_from_fasta_list<P: AsRef<Path>>(
    files: &[P],
    sparsification: Sparsification,
    use_accessory: bool,
    options: &SketchOptions,
) -> Result<SparseDistances> {
    if files.is_empty() {
        bail!("at least two FASTA files are required");
    }
    if options.kmer_sizes.is_empty() || options.kmer_sizes.contains(&0) {
        bail!("sketch k-mer sizes must be positive");
    }
    if options.sketch_size == 0 {
        bail!("sketch size must be positive");
    }
    validate_sparsification(sparsification)?;
    let mut all_sketches = Vec::with_capacity(files.len());
    for path in files {
        let path = path.as_ref();
        let reader = parse_fastx_file(path)
            .map_err(|error| anyhow!(error))
            .with_context(|| format!("opening FASTA input {}", path.display()))?;
        let mut records = vec![NeedletailIterator::new(reader)];
        let mut opts = SketchingOpts::default();
        opts.name = sample_name(path);
        opts.k_vals = options.kmer_sizes.clone();
        opts.sketch_size = options.sketch_size;
        let mut sketches = sketch_data(&mut records, opts);
        all_sketches.append(&mut sketches);
    }
    if all_sketches.len() < 2 {
        bail!("at least two FASTA samples are required");
    }
    let sketches = MultiSketch::from_sketches(
        &mut all_sketches,
        options.sketch_size,
        &options.kmer_sizes,
        HashType::DNA,
    );
    sparse_sketch_distances(&sketches, sparsification, use_accessory)
}

fn sparse_sketch_distances(
    sketches: &MultiSketch,
    sparsification: Sparsification,
    use_accessory: bool,
) -> Result<SparseDistances> {
    let n = sketches.number_samples_loaded();
    if n < 2 {
        bail!("at least two sketch samples are required");
    }
    let requested_k = match sparsification {
        Sparsification::Knn(k) => k,
        Sparsification::Threshold(_) => unreachable!(),
    };
    let k = if requested_k == 0 {
        n - 1
    } else {
        requested_k.min(n - 1)
    };
    let names = (0..n)
        .map(|index| sketches.sketch_name(index).to_string())
        .collect::<Vec<_>>();

    if sketches.kmer_lengths().len() == 1 {
        if use_accessory {
            bail!("accessory sketch distances require at least two k-mer lengths");
        }
        let dist_type = distances::set_k(sketches, Some(sketches.kmer_lengths()[0]), false)
            .context("selecting sketch distance")?;
        let sparse = distances::self_dists_knn(sketches, n, k, dist_type, true, None, 0.0);
        let DistVec::Jaccard(values) = sparse.dists_as_ref() else {
            bail!("sketchlib returned an unexpected distance type");
        };
        let mut rows = Vec::with_capacity(values.len());
        let mut columns = Vec::with_capacity(values.len());
        let mut distances = Vec::with_capacity(values.len());
        for (edge, value) in values.iter().enumerate() {
            rows.push((edge / k) as u64);
            columns.push(value.0 as u64);
            distances.push(value.1 as f64);
        }
        return SparseDistances::new(names, rows, columns, distances);
    }

    if use_accessory {
        return sparse_sketch_accessory(sketches, names, k);
    }

    let dist_type = distances::set_k(sketches, None, false).context("selecting sketch distance")?;
    let sparse = distances::self_dists_knn(sketches, n, k, dist_type, true, None, 0.0);
    let DistVec::CoreAcc(values) = sparse.dists_as_ref() else {
        bail!("sketchlib returned an unexpected distance type");
    };
    let mut rows = Vec::with_capacity(values.len());
    let mut columns = Vec::with_capacity(values.len());
    let mut distances = Vec::with_capacity(values.len());
    for (edge, value) in values.iter().enumerate() {
        rows.push((edge / k) as u64);
        columns.push(value.0 as u64);
        distances.push(value.1 as f64);
    }
    SparseDistances::new(names, rows, columns, distances)
}

fn sparse_sketch_accessory(
    sketches: &MultiSketch,
    names: Vec<String>,
    k: usize,
) -> Result<SparseDistances> {
    let n = names.len();
    let dist_type = distances::set_k(sketches, None, false).context("selecting sketch distance")?;
    let mut writer = SketchAccessoryWriter::new(names.clone(), k);
    distances::self_dists_all_stream(&mut writer, sketches, n, dist_type, true, None, 0.0, 1)
        .map_err(|error| anyhow!(error))
        .context("streaming sketch accessory distances")?;
    let candidates = writer.finish();
    let rows = candidates
        .into_iter()
        .enumerate()
        .map(|(row, candidate)| select_distance_row(row, candidate, Sparsification::Knn(k)))
        .collect::<Vec<_>>();
    let (rows, columns, distances) = flatten_rows(rows);
    SparseDistances::new(names, rows, columns, distances)
}

struct SketchAccessoryWriter {
    names: HashMap<String, usize>,
    candidates: Vec<Vec<(usize, f64)>>,
    k: usize,
    partial: String,
}

impl SketchAccessoryWriter {
    fn new(names: Vec<String>, k: usize) -> Self {
        let count = names.len();
        Self {
            names: names
                .into_iter()
                .enumerate()
                .map(|(index, name)| (name, index))
                .collect(),
            candidates: vec![Vec::new(); count],
            k,
            partial: String::new(),
        }
    }

    fn add(&mut self, row: usize, column: usize, distance: f64) {
        let candidates = &mut self.candidates[row];
        candidates.push((column, distance));
        if candidates.len() > self.k {
            let (max_index, _) = candidates
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| {
                    left.1
                        .total_cmp(&right.1)
                        .then_with(|| left.0.cmp(&right.0))
                })
                .expect("candidate list is non-empty");
            candidates.swap_remove(max_index);
        }
    }

    fn parse_line(&mut self, line: &str) -> io::Result<()> {
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sketchlib accessory distance line has the wrong number of fields",
            ));
        }
        let row = *self.names.get(fields[0]).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "unknown sketch sample name")
        })?;
        let column = *self.names.get(fields[1]).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "unknown sketch sample name")
        })?;
        let distance = fields[3].parse::<f64>().map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid sketch distance: {error}"),
            )
        })?;
        self.add(row, column, distance);
        self.add(column, row, distance);
        Ok(())
    }

    fn finish(mut self) -> Vec<Vec<(usize, f64)>> {
        if !self.partial.is_empty() {
            let line = std::mem::take(&mut self.partial);
            self.parse_line(&line)
                .expect("sketchlib stream ended with an invalid distance line");
        }
        self.candidates
    }
}

impl Write for SketchAccessoryWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.partial.push_str(&String::from_utf8_lossy(buffer));
        while let Some(end) = self.partial.find('\n') {
            let line = self.partial[..end].to_string();
            self.partial.drain(..=end);
            self.parse_line(line.trim_end_matches('\r'))?;
        }
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn read_alignment<P: AsRef<Path>>(path: P) -> Result<(Vec<String>, Vec<SampleBases>, usize)> {
    let path = path.as_ref();
    let mut reader = parse_fastx_file(path)
        .map_err(|error| anyhow!(error))
        .with_context(|| format!("opening alignment {}", path.display()))?;
    let mut names = Vec::new();
    let mut sequences = Vec::new();
    let mut alignment_len = None;
    while let Some(record) = reader.next() {
        let record = record
            .map_err(|error| anyhow!(error))
            .context("reading alignment record")?;
        let sequence = record.seq().to_vec();
        if sequence.is_empty() {
            bail!("alignment contains an empty sequence");
        }
        if let Some(expected) = alignment_len {
            if sequence.len() != expected {
                bail!("alignment sequences have variable lengths");
            }
        } else {
            alignment_len = Some(sequence.len());
        }
        names.push(String::from_utf8_lossy(record.id()).into_owned());
        sequences.push(SampleBases::from_sequence_at(&sequence, 0));
    }
    let alignment_len = alignment_len.ok_or_else(|| anyhow!("alignment contains no sequences"))?;
    if names.len() < 2 {
        bail!("alignment must contain at least two sequences");
    }
    Ok((names, sequences, alignment_len))
}

fn read_accessory_table<P: AsRef<Path>>(path: P) -> Result<(Vec<String>, Vec<Vec<u8>>)> {
    let path = path.as_ref();
    let reader = open_table(path)?;
    let mut csv = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(false)
        .from_reader(reader);
    let header = csv
        .records()
        .next()
        .ok_or_else(|| anyhow!("accessory table has no header"))?
        .context("reading accessory table header")?;
    if header.len() < 3 || header.get(0) != Some("Gene") {
        bail!("accessory table must start with Gene and at least two samples");
    }
    let names = header
        .iter()
        .skip(1)
        .map(str::to_string)
        .collect::<Vec<_>>();
    let mut columns = vec![Vec::new(); names.len()];
    for record in csv.records() {
        let record = record.context("reading accessory table row")?;
        if record.len() != names.len() + 1 {
            bail!("accessory table row has the wrong number of columns");
        }
        for (column, value) in record.iter().skip(1).enumerate() {
            let value = match value {
                "0" => 0,
                "1" => 1,
                _ => bail!("accessory values must be binary 0/1"),
            };
            columns[column].push(value);
        }
    }
    Ok((names, columns))
}

fn open_table(path: &Path) -> Result<Box<dyn Read>> {
    let file =
        File::open(path).with_context(|| format!("opening accessory table {}", path.display()))?;
    if path.extension().is_some_and(|extension| extension == "bz2") {
        Ok(Box::new(bzip2::read::BzDecoder::new(file)))
    } else {
        Ok(Box::new(file))
    }
}

fn select_alignment_row(
    row: usize,
    mut comparisons: Vec<(usize, usize)>,
    alignment_len: usize,
    sparsification: Sparsification,
) -> Vec<(u64, u64, f64)> {
    let cutoff = match sparsification {
        Sparsification::Knn(k) => {
            comparisons.sort_unstable_by_key(|&(column, distance)| (distance, column));
            let neighbour_count = if k == 0 {
                comparisons.len() - 1
            } else {
                k.min(comparisons.len() - 1)
            };
            comparisons[neighbour_count].1
        }
        Sparsification::Threshold(threshold) => {
            let raw = (threshold * alignment_len as f64).floor() as usize;
            raw.saturating_add(1)
        }
    };
    comparisons
        .into_iter()
        .filter(|&(_, distance)| distance <= cutoff)
        .map(|(column, distance)| {
            (
                row as u64,
                column as u64,
                distance as f64 / alignment_len as f64,
            )
        })
        .collect()
}

fn validate_sparsification(sparsification: Sparsification) -> Result<()> {
    if let Sparsification::Threshold(threshold) = sparsification
        && (!threshold.is_finite() || !(0.0..=1.0).contains(&threshold))
    {
        bail!("distance threshold must be finite and in [0, 1]");
    }
    Ok(())
}

fn select_distance_row(
    row: usize,
    mut candidates: Vec<(usize, f64)>,
    sparsification: Sparsification,
) -> Vec<(u64, u64, f64)> {
    match sparsification {
        Sparsification::Knn(k) => {
            candidates.sort_unstable_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            let count = if k == 0 {
                candidates.len()
            } else {
                k.min(candidates.len())
            };
            candidates.truncate(count);
        }
        Sparsification::Threshold(threshold) => {
            candidates.retain(|&(_, distance)| distance < threshold);
        }
    }
    candidates
        .into_iter()
        .map(|(column, distance)| (row as u64, column as u64, distance))
        .collect()
}

fn flatten_rows(rows: Vec<Vec<(u64, u64, f64)>>) -> (Vec<u64>, Vec<u64>, Vec<f64>) {
    let total = rows.iter().map(Vec::len).sum();
    let mut row_indices = Vec::with_capacity(total);
    let mut column_indices = Vec::with_capacity(total);
    let mut distances = Vec::with_capacity(total);
    for row in rows {
        for (source, column, distance) in row {
            row_indices.push(source);
            column_indices.push(column);
            distances.push(distance);
        }
    }
    (row_indices, column_indices, distances)
}

fn jaccard_distance(left: &[u8], right: &[u8]) -> f64 {
    let mut intersection = 0usize;
    let mut union = 0usize;
    for (&left, &right) in left.iter().zip(right) {
        if left == 1 && right == 1 {
            intersection += 1;
        }
        if left == 1 || right == 1 {
            union += 1;
        }
    }
    if union == 0 {
        0.0
    } else {
        1.0 - intersection as f64 / union as f64
    }
}

fn normalize_sketch_prefix(path: &Path) -> PathBuf {
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
        PathBuf::from(value)
    } else {
        prefix
    }
}

fn sample_name(path: &Path) -> String {
    let value = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("sample")
        .to_string();
    [".bz2", ".gz", ".xz", ".fasta", ".fastq", ".fa", ".fq"]
        .iter()
        .find_map(|suffix| value.strip_suffix(suffix))
        .unwrap_or(&value)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn sparse_distances_validates_and_exposes_parts() {
        let distances =
            SparseDistances::new(vec!["a".into(), "b".into()], vec![0], vec![1], vec![0.5])
                .unwrap();
        assert_eq!(distances.n_samples(), 2);
        assert_eq!(distances.len(), 1);
        assert_eq!(distances.into_parts().3, vec![0.5]);
    }

    #[test]
    fn accessory_table_reads_bzip2_and_binary_values() {
        let path =
            std::env::temp_dir().join(format!("mandrake-table-{}.tsv.bz2", std::process::id()));
        let file = File::create(&path).unwrap();
        let mut encoder = bzip2::write::BzEncoder::new(file, bzip2::Compression::fast());
        writeln!(encoder, "Gene\ta\tb").unwrap();
        writeln!(encoder, "g1\t1\t0").unwrap();
        writeln!(encoder, "g2\t1\t1").unwrap();
        encoder.finish().unwrap();
        let (names, columns) = read_accessory_table(&path).unwrap();
        assert_eq!(names, vec!["a", "b"]);
        assert_eq!(columns, vec![vec![1, 1], vec![0, 1]]);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn threshold_rows_are_strict_for_binary_distances() {
        let row = select_distance_row(0, vec![(1, 0.5), (2, 0.49)], Sparsification::Threshold(0.5));
        assert_eq!(row.len(), 1);
        assert_eq!(row[0].1, 2);
    }
}

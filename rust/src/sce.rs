//! CPU implementation of Mandrake's stochastic cluster embedding.

use anyhow::{Context, Result, bail};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::Array2;
use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::{RngCore, SeedableRng};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::api::{FrameSchedule, SceFrame, SceResults, WtsneOptions};

type Xoshiro = Xoshiro256PlusPlus;

const DIMENSIONS: usize = 2;
const PROBABILITY_TOLERANCE: f64 = 1e-5;
const PERPLEXITY_STEPS: usize = 100;
const WORKER_STREAM_DOMAIN: u64 = 0x4d41_4e44_5241_4b45;
const INITIAL_STREAM_DOMAIN: u64 = 0x5343_455f_494e_4954;
const STREAM_MULTIPLIER: u64 = 0x9e37_79b9_7f4a_7c15;

fn seeded_rng(seed: u64, domain: u64, index: u64) -> Xoshiro {
    let stream_seed = seed
        .wrapping_add(domain)
        .wrapping_add(index.wrapping_mul(STREAM_MULTIPLIER));
    Xoshiro::seed_from_u64(stream_seed)
}

fn next_unit(rng: &mut Xoshiro) -> f64 {
    rng.next_u64() as f64 / 18_446_744_073_709_551_616.0
}

/// Run the CPU stochastic cluster embedding algorithm.
///
/// `i` and `j` are zero-based COO source and destination node indices;
/// `distances` contains one distance per COO edge and `weights` contains one
/// sampling weight per node. The final embedding has shape `(weights.len(), 2)`.
///
/// The COO source rows may be in any order, but every node must have at least
/// one outgoing edge. Inputs and optimiser settings are validated before any
/// optimisation starts. When a frame schedule is enabled, its frame count
/// includes the initial and final states.
pub fn wtsne(
    i: &[u64],
    j: &[u64],
    distances: &[f64],
    weights: &[f64],
    options: &WtsneOptions,
) -> Result<SceResults> {
    validate_inputs(i, j, distances, weights, options)?;

    let n_nodes = weights.len();
    let row_edges = make_row_edges(i, n_nodes)?;
    let probabilities = conditional_probabilities(&row_edges, distances, options.perplexity)?;
    let edge_table = AliasTable::new(&probabilities).context("building edge sampler")?;
    let node_table = AliasTable::new(weights).context("building node sampler")?;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.workers)
        .build()
        .context("building Rayon thread pool")?;
    let embedding = AtomicEmbedding::new(initial_embedding(n_nodes, options.seed, &pool));
    let mut rng_states: Vec<Xoshiro> = (0..options.workers)
        .map(|worker| seeded_rng(options.seed, WORKER_STREAM_DOMAIN, worker as u64))
        .collect();

    let progress = if options.progress {
        let bar = ProgressBar::new(options.max_iterations as u64);
        let style =
            ProgressStyle::with_template("Optimizing [{bar:40.cyan/blue}] {pos}/{len} eta={msg}")
                .unwrap_or_else(|_| ProgressStyle::default_bar());
        bar.set_style(style);
        Some(bar)
    } else {
        None
    };

    let n_squared = n_nodes as f64 * (n_nodes - 1) as f64;
    let mut eq = 1.0;
    let frame_iterations = frame_iterations(&options.frame_schedule, options.max_iterations)?;
    let mut next_frame = 0;
    let mut frames = Vec::with_capacity(frame_iterations.len());

    if frame_iterations[0] == 0 {
        frames.push(SceFrame {
            iteration: 0,
            worker_updates: 0,
            eq,
            embedding: embedding.to_array(n_nodes)?,
        });
        next_frame += 1;
    }

    for iteration in 0..options.max_iterations {
        let eta = (options.learning_rate
            * (1.0 - iteration as f64 / options.max_iterations as f64))
            .max(options.learning_rate * 1e-4);
        let attr_coef = if options.initial_exaggeration && iteration < options.max_iterations / 10 {
            8.0
        } else {
            2.0
        };
        let repulsion_coef = 2.0 / (eq * options.repulsion_samples as f64);
        let worker_context = WorkerContext {
            embedding: &embedding,
            edge_table: &edge_table,
            node_table: &node_table,
            i,
            j,
            eta,
            attraction_coefficient: attr_coef,
            repulsion_coefficient: repulsion_coef,
            repulsion_samples: options.repulsion_samples,
        };

        let (qsum, qcount, clashes) = pool.install(|| {
            rng_states
                .par_iter_mut()
                .map(|rng| run_worker(rng, &worker_context))
                .reduce(
                    || (0.0, 0_u64, 0_u64),
                    |left, right| (left.0 + right.0, left.1 + right.1, left.2 + right.2),
                )
        });
        eq = (eq * n_squared + qsum) / (n_squared + qcount as f64);

        let completed_iteration = iteration + 1;
        if next_frame < frame_iterations.len()
            && frame_iterations[next_frame] == completed_iteration
        {
            frames.push(SceFrame {
                iteration: completed_iteration,
                worker_updates: completed_iteration as u128 * options.workers as u128,
                eq,
                embedding: embedding.to_array(n_nodes)?,
            });
            next_frame += 1;
        }

        if let Some(bar) = &progress {
            bar.set_position((iteration + 1) as u64);
            bar.set_message(format!("{eta:.4} Eq={eq:.6} clashes={clashes}"));
        }
    }
    if let Some(bar) = progress {
        bar.finish_and_clear();
    }

    debug_assert_eq!(next_frame, frame_iterations.len());
    Ok(SceResults { frames })
}

fn validate_inputs(
    i: &[u64],
    j: &[u64],
    distances: &[f64],
    weights: &[f64],
    options: &WtsneOptions,
) -> Result<()> {
    if i.len() != j.len() || i.len() != distances.len() {
        bail!("COO index and distance vectors must have the same length");
    }
    if weights.len() < 2 {
        bail!("at least two node weights are required");
    }
    if i.is_empty() {
        bail!("at least one COO edge is required");
    }
    if i.iter()
        .chain(j)
        .any(|&index| index >= weights.len() as u64)
    {
        bail!("COO node index is outside the node-weight vector");
    }
    if distances
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        bail!("distances must be finite and non-negative");
    }
    if weights
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        || weights.iter().all(|value| *value == 0.0)
    {
        bail!("weights must be finite, non-negative, and have positive total");
    }
    if !options.perplexity.is_finite() {
        bail!("perplexity must be finite");
    }
    if options.max_iterations == 0 {
        bail!("max_iterations must be greater than zero");
    }
    if options.repulsion_samples == 0 {
        bail!("repulsion_samples must be greater than zero");
    }
    if options.workers == 0 {
        bail!("workers must be greater than zero");
    }
    if !options.learning_rate.is_finite() || options.learning_rate <= 0.0 {
        bail!("learning_rate must be finite and positive");
    }
    validate_frame_schedule(&options.frame_schedule, options.max_iterations)?;
    Ok(())
}

fn validate_frame_schedule(schedule: &FrameSchedule, max_iterations: usize) -> Result<()> {
    match schedule {
        FrameSchedule::FinalOnly => Ok(()),
        FrameSchedule::Linear { frame_count } | FrameSchedule::Exponential { frame_count } => {
            if *frame_count < 2 || *frame_count > max_iterations.saturating_add(1) {
                bail!(
                    "frame count must be between 2 and max_iterations + 1 ({})",
                    max_iterations.saturating_add(1)
                );
            }
            Ok(())
        }
    }
}

fn frame_iterations(schedule: &FrameSchedule, max_iterations: usize) -> Result<Vec<usize>> {
    validate_frame_schedule(schedule, max_iterations)?;
    match schedule {
        FrameSchedule::FinalOnly => Ok(vec![max_iterations]),
        FrameSchedule::Linear { frame_count } => {
            let denominator = (*frame_count - 1) as u128;
            let maximum = max_iterations as u128;
            Ok((0..*frame_count)
                .map(|index| ((index as u128 * maximum) + denominator / 2) / denominator)
                .map(|position| position as usize)
                .collect())
        }
        FrameSchedule::Exponential { frame_count } => {
            let denominator = (*frame_count - 1) as f64;
            let base = max_iterations as f64 + 1.0;
            let mut positions = Vec::with_capacity(*frame_count);
            positions.push(0);
            for index in 1..(*frame_count - 1) {
                let fraction = index as f64 / denominator;
                let raw = (base.powf(fraction) - 1.0).round();
                let candidate = raw.clamp(0.0, max_iterations as f64) as usize;
                let lower_bound = positions[index - 1] + 1;
                let upper_bound = max_iterations - (*frame_count - 1 - index);
                positions.push(candidate.clamp(lower_bound, upper_bound));
            }
            positions.push(max_iterations);
            Ok(positions)
        }
    }
}

fn make_row_edges(i: &[u64], n_nodes: usize) -> Result<Vec<Vec<usize>>> {
    let mut rows = vec![Vec::new(); n_nodes];
    for (edge, &source) in i.iter().enumerate() {
        rows[source as usize].push(edge);
    }
    if rows.iter().any(Vec::is_empty) {
        bail!("each node must have at least one outgoing COO edge");
    }
    Ok(rows)
}

fn conditional_probabilities(
    rows: &[Vec<usize>],
    distances: &[f64],
    perplexity: f64,
) -> Result<Vec<f64>> {
    let mut probabilities = vec![0.0; distances.len()];
    if perplexity <= 0.0 {
        for (index, &distance) in distances.iter().enumerate() {
            let probability = 1.0 - distance;
            if probability < 0.0 || !probability.is_finite() {
                bail!("raw similarities must be finite and non-negative");
            }
            probabilities[index] = probability;
        }
    } else {
        let desired_entropy = perplexity.ln();
        let row_probabilities: Vec<Vec<f64>> = rows
            .par_iter()
            .map(|row| {
                let mut beta = 1.0;
                let mut beta_min = f64::NEG_INFINITY;
                let mut beta_max = f64::INFINITY;
                let mut row_probabilities = vec![0.0; row.len()];

                for _ in 0..PERPLEXITY_STEPS {
                    let mut sum = 0.0;
                    for (slot, &edge) in row.iter().enumerate() {
                        let probability = (-distances[edge] * beta).exp();
                        row_probabilities[slot] = probability;
                        sum += probability;
                    }
                    if sum == 0.0 || !sum.is_finite() {
                        let uniform = 1.0 / row.len() as f64;
                        row_probabilities.fill(uniform);
                        break;
                    }
                    let mut weighted_distance = 0.0;
                    for (slot, &edge) in row.iter().enumerate() {
                        row_probabilities[slot] /= sum;
                        weighted_distance += distances[edge] * row_probabilities[slot];
                    }
                    let entropy = sum.ln() + beta * weighted_distance;
                    let difference = entropy - desired_entropy;
                    if difference.abs() <= PROBABILITY_TOLERANCE {
                        break;
                    }
                    if difference > 0.0 {
                        beta_min = beta;
                        beta = if beta_max.is_infinite() {
                            beta * 2.0
                        } else {
                            (beta + beta_max) * 0.5
                        };
                    } else {
                        beta_max = beta;
                        beta = if beta_min.is_infinite() {
                            beta * 0.5
                        } else {
                            (beta + beta_min) * 0.5
                        };
                    }
                }

                row_probabilities
            })
            .collect();
        for (row, row_values) in rows.iter().zip(row_probabilities) {
            for (slot, &edge) in row.iter().enumerate() {
                probabilities[edge] = row_values[slot];
            }
        }
    }

    normalize(&mut probabilities, "edge probabilities")?;
    Ok(probabilities)
}

fn normalize(values: &mut [f64], name: &str) -> Result<()> {
    let sum: f64 = values.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        bail!("{name} must have a positive finite total");
    }
    for value in values {
        *value /= sum;
    }
    Ok(())
}

struct AliasTable {
    probabilities: Vec<f64>,
    aliases: Vec<usize>,
}

impl AliasTable {
    fn new(values: &[f64]) -> Result<Self> {
        if values.is_empty() {
            bail!("probability table cannot be empty");
        }
        let total: f64 = values.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            bail!("probability table must have a positive finite total");
        }

        let n = values.len();
        let mut probabilities = vec![0.0; n];
        let mut aliases: Vec<usize> = (0..n).collect();
        let mut small = Vec::new();
        let mut large = Vec::new();
        for (index, value) in values.iter().enumerate() {
            probabilities[index] = value * n as f64 / total;
            if probabilities[index] < 1.0 {
                small.push(index);
            } else {
                large.push(index);
            }
        }
        while let (Some(s), Some(l)) = (small.pop(), large.pop()) {
            aliases[s] = l;
            probabilities[l] = probabilities[l] + probabilities[s] - 1.0;
            if probabilities[l] < 1.0 {
                small.push(l);
            } else {
                large.push(l);
            }
        }
        for index in small.into_iter().chain(large) {
            probabilities[index] = 1.0;
            aliases[index] = index;
        }
        Ok(Self {
            probabilities,
            aliases,
        })
    }

    fn draw(&self, rng: &mut Xoshiro) -> usize {
        let column = (next_unit(rng) * self.probabilities.len() as f64) as usize;
        if next_unit(rng) < self.probabilities[column] {
            column
        } else {
            self.aliases[column]
        }
    }
}

fn initial_embedding(n_nodes: usize, seed: u64, pool: &rayon::ThreadPool) -> Vec<f64> {
    let mut values = vec![0.0; n_nodes * DIMENSIONS];
    pool.install(|| {
        values
            .par_chunks_exact_mut(DIMENSIONS)
            .enumerate()
            .for_each(|(index, point)| {
                let mut rng = seeded_rng(seed, INITIAL_STREAM_DOMAIN, index as u64);
                loop {
                    let x = 2.0 * next_unit(&mut rng) - 1.0;
                    let y = 2.0 * next_unit(&mut rng) - 1.0;
                    let norm = x * x + y * y;
                    if norm > 0.0 && norm <= 1.0 {
                        point[0] = x * 1e-4;
                        point[1] = y * 1e-4;
                        break;
                    }
                }
            });
    });
    values
}

struct WorkerContext<'a> {
    embedding: &'a AtomicEmbedding,
    edge_table: &'a AliasTable,
    node_table: &'a AliasTable,
    i: &'a [u64],
    j: &'a [u64],
    eta: f64,
    attraction_coefficient: f64,
    repulsion_coefficient: f64,
    repulsion_samples: usize,
}

fn run_worker(rng: &mut Xoshiro, context: &WorkerContext<'_>) -> (f64, u64, u64) {
    let edge = context.edge_table.draw(rng);
    let attraction_i = context.i[edge] as usize;
    let attraction_j = context.j[edge] as usize;
    let mut qsum = 0.0;
    let mut qcount = 0_u64;
    let mut clashes = 0_u64;

    for sample in 0..=context.repulsion_samples {
        let (k, l) = if sample == 0 {
            (attraction_i, attraction_j)
        } else {
            (context.node_table.draw(rng), context.node_table.draw(rng))
        };
        if k == l {
            continue;
        }

        let k_offset = k * DIMENSIONS;
        let l_offset = l * DIMENSIONS;
        let k_read = [
            context.embedding.load(k_offset),
            context.embedding.load(k_offset + 1),
        ];
        let l_read = [
            context.embedding.load(l_offset),
            context.embedding.load(l_offset + 1),
        ];
        let delta = [k_read[0] - l_read[0], k_read[1] - l_read[1]];
        let dist_squared = delta[0] * delta[0] + delta[1] * delta[1];
        let q = 1.0 / (1.0 + dist_squared);
        let coefficient = if sample == 0 {
            -context.attraction_coefficient * q
        } else {
            context.repulsion_coefficient * q * q
        };
        let gain = [
            context.eta * coefficient * delta[0],
            context.eta * coefficient * delta[1],
        ];
        let changed_k0 = context.embedding.add(k_offset, gain[0], k_read[0]);
        let changed_l0 = context.embedding.add(l_offset, -gain[0], l_read[0]);
        let changed_k1 = context.embedding.add(k_offset + 1, gain[1], k_read[1]);
        let changed_l1 = context.embedding.add(l_offset + 1, -gain[1], l_read[1]);

        if changed_k0 && changed_l0 && changed_k1 && changed_l1 {
            qsum += q;
            qcount += 1;
        } else {
            context.embedding.add_unchecked(k_offset, -gain[0]);
            context.embedding.add_unchecked(l_offset, gain[0]);
            context.embedding.add_unchecked(k_offset + 1, -gain[1]);
            context.embedding.add_unchecked(l_offset + 1, gain[1]);
            clashes += 1;
        }
    }
    (qsum, qcount, clashes)
}

struct AtomicEmbedding {
    values: Vec<AtomicU64>,
}

impl AtomicEmbedding {
    fn new(values: Vec<f64>) -> Self {
        Self {
            values: values
                .into_iter()
                .map(|value| AtomicU64::new(value.to_bits()))
                .collect(),
        }
    }

    fn load(&self, index: usize) -> f64 {
        f64::from_bits(self.values[index].load(Ordering::SeqCst))
    }

    fn add(&self, index: usize, delta: f64, expected: f64) -> bool {
        let atomic = &self.values[index];
        let expected_result = (expected + delta).to_bits();
        let mut observed = atomic.load(Ordering::SeqCst);
        loop {
            let current = f64::from_bits(observed);
            let next = (current + delta).to_bits();
            match atomic.compare_exchange_weak(observed, next, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => return next == expected_result,
                Err(actual) => observed = actual,
            }
        }
    }

    fn add_unchecked(&self, index: usize, delta: f64) {
        let atomic = &self.values[index];
        let mut observed = atomic.load(Ordering::SeqCst);
        loop {
            let current = f64::from_bits(observed);
            let next = (current + delta).to_bits();
            match atomic.compare_exchange_weak(observed, next, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => return,
                Err(actual) => observed = actual,
            }
        }
    }

    fn to_array(&self, n_nodes: usize) -> Result<Array2<f64>> {
        let values: Vec<f64> = self
            .values
            .iter()
            .map(|value| f64::from_bits(value.load(Ordering::SeqCst)))
            .collect();
        Array2::from_shape_vec((n_nodes, DIMENSIONS), values).map_err(anyhow::Error::msg)
    }
}

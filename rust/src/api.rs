//! Public stochastic cluster embedding configuration and result types.

use ndarray::Array2;

/// Controls which optimisation states are retained by [`crate::wtsne`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum FrameSchedule {
    /// Retain only the final optimisation state.
    #[default]
    FinalOnly,
    /// Retain `frame_count` evenly spaced states, including initial and final.
    Linear { frame_count: usize },
    /// Retain `frame_count` geometrically spaced states, including initial and
    /// final, with more frames near the start of optimisation.
    Exponential { frame_count: usize },
}

/// One sampled state of a stochastic cluster embedding run.
#[derive(Clone, Debug)]
pub struct SceFrame {
    pub(crate) iteration: usize,
    pub(crate) worker_updates: u128,
    pub(crate) eq: f64,
    pub(crate) embedding: Array2<f64>,
}

impl SceFrame {
    /// Number of completed outer optimisation iterations at this frame.
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Number of worker updates completed at this frame.
    pub fn worker_updates(&self) -> u128 {
        self.worker_updates
    }

    /// Current SCE convergence statistic.
    pub fn eq(&self) -> f64 {
        self.eq
    }

    /// Borrow the 2D embedding stored at this frame.
    pub fn embedding(&self) -> &Array2<f64> {
        &self.embedding
    }
}

/// Final embedding and optional intermediate optimisation states.
///
/// Frames are chronological, non-empty, and the final frame is always the
/// authoritative final embedding. Configured schedules retain all snapshots
/// in memory, so storage scales with the number of frames and nodes.
#[derive(Clone, Debug)]
pub struct SceResults {
    pub(crate) frames: Vec<SceFrame>,
}

impl SceResults {
    /// Borrow the final 2D embedding.
    pub fn embedding(&self) -> &Array2<f64> {
        &self
            .frames
            .last()
            .expect("SceResults always contains a final frame")
            .embedding
    }

    /// Consume the result and return its final 2D embedding.
    pub fn into_embedding(self) -> Array2<f64> {
        self.frames
            .into_iter()
            .last()
            .expect("SceResults always contains a final frame")
            .embedding
    }

    /// Borrow all retained frames in chronological order.
    pub fn frames(&self) -> &[SceFrame] {
        &self.frames
    }

    /// Whether initial and intermediate states were retained.
    pub fn is_animated(&self) -> bool {
        self.frames.len() > 1
    }

    /// Return the final convergence statistic.
    pub fn final_eq(&self) -> f64 {
        self.frames
            .last()
            .expect("SceResults always contains a final frame")
            .eq
    }
}

/// Configuration for [`crate::wtsne`].
#[derive(Clone, Debug)]
pub struct WtsneOptions {
    /// Target entropy for conditional probability preprocessing. Values at or
    /// below zero use the input values as raw similarities (`1 - distance`).
    pub perplexity: f64,
    /// Number of optimisation iterations.
    pub max_iterations: usize,
    /// Number of randomly sampled repulsion pairs per worker and iteration.
    pub repulsion_samples: usize,
    /// Initial learning rate.
    pub learning_rate: f64,
    /// Use four-times stronger attraction during the first tenth of the run.
    pub initial_exaggeration: bool,
    /// Number of Rayon workers used for preprocessing, initialisation, and
    /// optimisation.
    pub workers: usize,
    /// Show an `indicatif` progress bar while optimising.
    pub progress: bool,
    /// Seed for all random-number streams.
    pub seed: u64,
    /// Which optimisation states to retain in the returned result.
    pub frame_schedule: FrameSchedule,
}

impl Default for WtsneOptions {
    fn default() -> Self {
        Self {
            perplexity: 15.0,
            max_iterations: 100_000,
            repulsion_samples: 5,
            learning_rate: 1.0,
            initial_exaggeration: false,
            workers: 1,
            progress: true,
            seed: 1,
            frame_schedule: FrameSchedule::FinalOnly,
        }
    }
}

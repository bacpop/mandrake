//! CPU implementation of Mandrake's stochastic cluster embedding.

pub mod api;
pub mod distances;
mod sce;

pub use api::{FrameSchedule, SceFrame, SceResults, WtsneOptions};
pub use distances::{
    SketchOptions, SparseDistances, Sparsification, accessory_distances, pair_snp_distances,
    sketch_distances, sketch_distances_from_fasta_list,
};
pub use sce::wtsne;

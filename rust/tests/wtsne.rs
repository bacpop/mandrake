use mandrake::{FrameSchedule, WtsneOptions, wtsne};

fn graph() -> (Vec<u64>, Vec<u64>, Vec<f64>, Vec<f64>) {
    (
        vec![0, 0, 1, 1, 2, 2, 3, 3],
        vec![1, 2, 0, 2, 0, 3, 0, 2],
        vec![0.1, 0.4, 0.1, 0.2, 0.4, 0.3, 0.3, 0.2],
        vec![1.0, 2.0, 1.0, 1.0],
    )
}

fn options() -> WtsneOptions {
    WtsneOptions {
        perplexity: 2.0,
        max_iterations: 20,
        repulsion_samples: 1,
        learning_rate: 1.0,
        initial_exaggeration: false,
        workers: 1,
        progress: false,
        seed: 42,
        frame_schedule: FrameSchedule::FinalOnly,
    }
}

#[test]
fn valid_graph_returns_finite_two_dimensional_embedding() {
    let (i, j, distances, weights) = graph();
    let result = wtsne(&i, &j, &distances, &weights, &options()).unwrap();
    let embedding = result.embedding();

    assert_eq!(embedding.shape(), &[4, 2]);
    assert!(embedding.iter().all(|value| value.is_finite()));
}

#[test]
fn fixed_seed_is_reproducible_with_one_worker() {
    let (i, j, distances, weights) = graph();
    let first = wtsne(&i, &j, &distances, &weights, &options())
        .unwrap()
        .into_embedding();
    let second = wtsne(&i, &j, &distances, &weights, &options())
        .unwrap()
        .into_embedding();

    assert_eq!(first, second);
}

#[test]
fn changing_seed_changes_reproducible_embedding() {
    let (i, j, distances, weights) = graph();
    let mut baseline = options();
    baseline.max_iterations = 1;
    let mut changed = options();
    changed.max_iterations = 1;
    changed.seed += 1;
    let first = wtsne(&i, &j, &distances, &weights, &baseline)
        .unwrap()
        .into_embedding();
    let second = wtsne(&i, &j, &distances, &weights, &changed)
        .unwrap()
        .into_embedding();

    assert_ne!(first, second);
}

#[test]
fn public_api_module_exposes_configuration_types() {
    let options = mandrake::api::WtsneOptions::default();
    assert_eq!(options.seed, 1);
}

#[test]
fn parallel_execution_completes_with_finite_output() {
    let (i, j, distances, weights) = graph();
    let mut parallel_options = options();
    parallel_options.workers = 2;
    let result = wtsne(&i, &j, &distances, &weights, &parallel_options).unwrap();
    let embedding = result.embedding();

    assert_eq!(embedding.shape(), &[4, 2]);
    assert!(embedding.iter().all(|value| value.is_finite()));
}

#[test]
fn raw_similarity_mode_is_supported() {
    let (i, j, mut distances, weights) = graph();
    distances
        .iter_mut()
        .for_each(|distance| *distance = 1.0 - *distance);
    let mut raw_options = options();
    raw_options.perplexity = 0.0;

    let result = wtsne(&i, &j, &distances, &weights, &raw_options).unwrap();
    let embedding = result.embedding();
    assert!(embedding.iter().all(|value| value.is_finite()));
}

#[test]
fn unordered_coo_rows_are_supported() {
    let (i, j, distances, weights) = graph();
    let order = [4, 0, 6, 2, 7, 1, 5, 3];
    let shuffled_i: Vec<u64> = order.iter().map(|&index| i[index]).collect();
    let shuffled_j: Vec<u64> = order.iter().map(|&index| j[index]).collect();
    let shuffled_distances: Vec<f64> = order.iter().map(|&index| distances[index]).collect();

    let result = wtsne(
        &shuffled_i,
        &shuffled_j,
        &shuffled_distances,
        &weights,
        &options(),
    )
    .unwrap();
    let embedding = result.embedding();
    assert_eq!(embedding.shape(), &[4, 2]);
    assert!(embedding.iter().all(|value| value.is_finite()));
}

#[test]
fn invalid_inputs_are_rejected() {
    let (i, j, distances, weights) = graph();
    let error = wtsne(&i[..7], &j, &distances, &weights, &options()).unwrap_err();
    assert!(error.to_string().contains("same length"));

    let mut bad_i = i;
    bad_i[0] = 4;
    let error = wtsne(&bad_i, &j, &distances, &weights, &options()).unwrap_err();
    assert!(error.to_string().contains("node index"));

    bad_i[0] = u64::MAX;
    let error = wtsne(&bad_i, &j, &distances, &weights, &options()).unwrap_err();
    assert!(error.to_string().contains("node index"));
}

#[test]
fn final_only_result_contains_final_metadata() {
    let (i, j, distances, weights) = graph();
    let result = wtsne(&i, &j, &distances, &weights, &options()).unwrap();

    assert!(!result.is_animated());
    assert_eq!(result.frames().len(), 1);
    let frame = &result.frames()[0];
    assert_eq!(frame.iteration(), 20);
    assert_eq!(frame.worker_updates(), 20);
    assert!(frame.eq().is_finite());
    assert_eq!(frame.embedding(), result.embedding());
}

#[test]
fn linear_schedule_includes_initial_and_final_states() {
    let (i, j, distances, weights) = graph();
    let mut scheduled = options();
    scheduled.frame_schedule = FrameSchedule::Linear { frame_count: 4 };
    let result = wtsne(&i, &j, &distances, &weights, &scheduled).unwrap();

    assert!(result.is_animated());
    assert_eq!(result.frames().len(), 4);
    assert_eq!(
        result
            .frames()
            .iter()
            .map(|frame| frame.iteration())
            .collect::<Vec<_>>(),
        vec![0, 7, 13, 20]
    );
    assert_eq!(result.frames().first().unwrap().eq(), 1.0);
    assert_eq!(
        result.frames().last().unwrap().embedding(),
        result.embedding()
    );
}

#[test]
fn exponential_schedule_uses_geometric_positions() {
    let (i, j, distances, weights) = graph();
    let mut scheduled = options();
    scheduled.max_iterations = 15;
    scheduled.frame_schedule = FrameSchedule::Exponential { frame_count: 5 };
    let result = wtsne(&i, &j, &distances, &weights, &scheduled).unwrap();

    assert_eq!(
        result
            .frames()
            .iter()
            .map(|frame| frame.iteration())
            .collect::<Vec<_>>(),
        vec![0, 1, 3, 7, 15]
    );
}

#[test]
fn recording_frames_does_not_change_final_embedding() {
    let (i, j, distances, weights) = graph();
    let final_only = wtsne(&i, &j, &distances, &weights, &options())
        .unwrap()
        .into_embedding();
    let mut scheduled = options();
    scheduled.frame_schedule = FrameSchedule::Linear { frame_count: 4 };
    let with_frames = wtsne(&i, &j, &distances, &weights, &scheduled)
        .unwrap()
        .into_embedding();

    assert_eq!(final_only, with_frames);
}

#[test]
fn invalid_frame_counts_are_rejected() {
    let (i, j, distances, weights) = graph();
    let mut scheduled = options();
    scheduled.frame_schedule = FrameSchedule::Linear { frame_count: 1 };
    let error = wtsne(&i, &j, &distances, &weights, &scheduled).unwrap_err();
    assert!(error.to_string().contains("frame count"));

    scheduled.frame_schedule = FrameSchedule::Exponential { frame_count: 22 };
    let error = wtsne(&i, &j, &distances, &weights, &scheduled).unwrap_err();
    assert!(error.to_string().contains("frame count"));
}

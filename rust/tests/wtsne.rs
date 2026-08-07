use mandrake::{WtsneOptions, wtsne};

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
    }
}

#[test]
fn valid_graph_returns_finite_two_dimensional_embedding() {
    let (i, j, distances, weights) = graph();
    let embedding = wtsne(&i, &j, &distances, &weights, &options()).unwrap();

    assert_eq!(embedding.shape(), &[4, 2]);
    assert!(embedding.iter().all(|value| value.is_finite()));
}

#[test]
fn fixed_seed_is_reproducible_with_one_worker() {
    let (i, j, distances, weights) = graph();
    let first = wtsne(&i, &j, &distances, &weights, &options()).unwrap();
    let second = wtsne(&i, &j, &distances, &weights, &options()).unwrap();

    assert_eq!(first, second);
}

#[test]
fn parallel_execution_completes_with_finite_output() {
    let (i, j, distances, weights) = graph();
    let mut parallel_options = options();
    parallel_options.workers = 2;
    let embedding = wtsne(&i, &j, &distances, &weights, &parallel_options).unwrap();

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

    let embedding = wtsne(&i, &j, &distances, &weights, &raw_options).unwrap();
    assert!(embedding.iter().all(|value| value.is_finite()));
}

#[test]
fn unordered_coo_rows_are_supported() {
    let (i, j, distances, weights) = graph();
    let order = [4, 0, 6, 2, 7, 1, 5, 3];
    let shuffled_i: Vec<u64> = order.iter().map(|&index| i[index]).collect();
    let shuffled_j: Vec<u64> = order.iter().map(|&index| j[index]).collect();
    let shuffled_distances: Vec<f64> = order.iter().map(|&index| distances[index]).collect();

    let embedding = wtsne(
        &shuffled_i,
        &shuffled_j,
        &shuffled_distances,
        &weights,
        &options(),
    )
    .unwrap();
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

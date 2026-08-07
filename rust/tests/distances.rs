use mandrake::{
    SketchOptions, Sparsification, accessory_distances, pair_snp_distances, sketch_distances,
    sketch_distances_from_fasta_list,
};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

fn temp_path(suffix: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "mandrake-distance-test-{}-{suffix}",
        std::process::id()
    ))
}

#[test]
fn pair_snp_distances_are_normalized_and_keep_legacy_self_edges() {
    let path = temp_path("alignment.fasta.bz2");
    let file = File::create(&path).unwrap();
    let mut encoder = bzip2::write::BzEncoder::new(file, bzip2::Compression::fast());
    write!(encoder, ">a\nACGT\n>b\nACGA\n>c\nACGN\n").unwrap();
    encoder.finish().unwrap();

    let distances = pair_snp_distances(&path, Sparsification::Knn(1)).unwrap();
    assert_eq!(distances.n_samples(), 3);
    assert!(
        distances
            .rows()
            .iter()
            .zip(distances.columns())
            .any(|(row, column)| row == column)
    );
    assert!(
        distances
            .distances()
            .iter()
            .all(|distance| (0.0..=1.0).contains(distance))
    );
    std::fs::remove_file(path).unwrap();
}

#[test]
fn accessory_distances_parse_binary_table_and_use_strict_thresholds() {
    let path = temp_path("accessory.tsv");
    let mut file = File::create(&path).unwrap();
    writeln!(file, "Gene\ta\tb\tc").unwrap();
    writeln!(file, "g1\t1\t1\t0").unwrap();
    writeln!(file, "g2\t1\t0\t0").unwrap();
    drop(file);

    let distances = accessory_distances(&path, Sparsification::Threshold(0.5)).unwrap();
    assert_eq!(distances.names(), ["a", "b", "c"]);
    assert!(distances.distances().iter().all(|&distance| distance < 0.5));
    assert!(
        distances
            .rows()
            .iter()
            .zip(distances.columns())
            .all(|(row, column)| row != column)
    );
    std::fs::remove_file(path).unwrap();
}

#[test]
fn sketch_fixture_loads_as_sparse_distances() {
    let prefix = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sketches.skm");
    let distances = sketch_distances(&prefix, Sparsification::Knn(2), false).unwrap();
    assert!(distances.n_samples() > 2);
    assert_eq!(distances.len(), distances.n_samples() * 2);
    assert!(
        distances
            .distances()
            .iter()
            .all(|distance| (0.0..=1.0).contains(distance))
    );
}

#[test]
fn sketch_fasta_list_supports_core_and_accessory_distances() {
    let first = temp_path("first.fasta");
    let second = temp_path("second.fasta");
    std::fs::write(&first, b">first\nACGTACGTACGTACGTACGTACGT\n").unwrap();
    std::fs::write(&second, b">second\nACGTACGTACGTACGTACGTTCGT\n").unwrap();
    let options = SketchOptions {
        kmer_sizes: vec![3, 5],
        sketch_size: 64,
    };
    let core = sketch_distances_from_fasta_list(
        &[first.clone(), second.clone()],
        Sparsification::Knn(1),
        false,
        &options,
    )
    .unwrap();
    let accessory = sketch_distances_from_fasta_list(
        &[first.clone(), second.clone()],
        Sparsification::Knn(1),
        true,
        &options,
    )
    .unwrap();
    assert_eq!(core.n_samples(), 2);
    assert_eq!(accessory.n_samples(), 2);
    assert_eq!(core.len(), 2);
    assert_eq!(accessory.len(), 2);
    std::fs::remove_file(first).unwrap();
    std::fs::remove_file(second).unwrap();
}

#[test]
fn cli_writes_embedding_and_names_for_accessory_input() {
    let input = temp_path("cli-accessory.tsv");
    let output = temp_path("cli-output");
    let mut file = File::create(&input).unwrap();
    writeln!(file, "Gene\ta\tb\tc").unwrap();
    writeln!(file, "g1\t1\t1\t0").unwrap();
    writeln!(file, "g2\t1\t0\t0").unwrap();
    drop(file);

    let status = Command::new(env!("CARGO_BIN_EXE_mandrake"))
        .args([
            "--accessory",
            input.to_str().unwrap(),
            "--knn",
            "1",
            "--max-iterations",
            "1",
            "--no-progress",
            "--output",
            output.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success());
    let embedding = PathBuf::from(format!("{}.embedding.txt", output.display()));
    let names = PathBuf::from(format!("{}.names.txt", output.display()));
    assert_eq!(
        std::fs::read_to_string(&embedding).unwrap().lines().count(),
        3
    );
    assert_eq!(std::fs::read_to_string(&names).unwrap().lines().count(), 3);
    std::fs::remove_file(input).unwrap();
    std::fs::remove_file(embedding).unwrap();
    std::fs::remove_file(names).unwrap();
}

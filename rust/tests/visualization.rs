#[cfg(feature = "PyO3")]
use std::path::PathBuf;
#[cfg(feature = "PyO3")]
use std::process::Command;

#[cfg(feature = "PyO3")]
fn missing_prefix() -> PathBuf {
    std::env::temp_dir().join(format!(
        "mandrake-visualization-missing-{}",
        std::process::id()
    ))
}

#[cfg(feature = "PyO3")]
#[test]
fn plot_subcommand_is_exposed_by_default_feature() {
    let output = Command::new(env!("CARGO_BIN_EXE_mandrake"))
        .args(["--help"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let help = String::from_utf8(output.stdout).unwrap();
    assert!(help.contains("plot"));
    assert!(help.contains("--save-animation"));
}

#[cfg(feature = "PyO3")]
#[test]
fn plot_reports_missing_embedding_prefix() {
    let output = Command::new(env!("CARGO_BIN_EXE_mandrake"))
        .args([
            "plot",
            "--input-prefix",
            missing_prefix().to_str().unwrap(),
            "--no-clustering",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("opening embedding file"));
}

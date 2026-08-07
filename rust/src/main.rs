mod cli;
#[cfg(feature = "PyO3")]
mod visualization;

fn main() -> anyhow::Result<()> {
    cli::run()
}

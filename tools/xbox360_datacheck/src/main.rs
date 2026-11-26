//! The PC side of the data check: the same fingerprints the console title
//! writes to its log, printed, for comparing the two.
//!
//!     cargo run -p xbox360_datacheck --release -- <data root>

fn main() {
    let root = std::env::args().nth(1).unwrap_or_else(|| "Data".into());
    let mut out = |line: &str| println!("{line}");
    xbox360_datacheck::run(std::path::Path::new(&root), &mut out);
}

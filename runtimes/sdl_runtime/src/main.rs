//! The SDL runtime as a desktop binary: the same code alpha3 runs on Android.
fn main() {
    std::process::exit(shock2::run(std::env::args().collect()));
}

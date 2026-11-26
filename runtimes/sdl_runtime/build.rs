#[path = "../../build-support/version.rs"]
mod version;

fn main() {
    version::emit_build_label();
    // sdl2-sys links `-lSDL2` without saying where it is; for Android that is
    // the arm64 libSDL2.so the host app (alpha3) ships, e.g.
    // alpha3/androidApp/src/main/jniLibs/arm64-v8a.
    println!("cargo:rerun-if-env-changed=SHOCK2_SDL2_LIB_DIR");
    if let Ok(dir) = std::env::var("SHOCK2_SDL2_LIB_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
    }
}

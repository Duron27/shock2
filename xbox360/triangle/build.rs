//! Turns `build/shaders.h`, which vulkan4xenon's `tools/build_shaders.py`
//! writes (xbox360/build.sh runs it first), into the SPIR-V sizes Rust needs:
//! the arrays themselves are C, in `build/shaders.c`, beside the bundle
//! `vkCreateShaderModule` looks shaders up in.
use std::fmt::Write;

fn main() {
    let dir = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let header = dir.join("build/shaders.h");
    println!("cargo:rerun-if-changed={}", header.display());
    let text = std::fs::read_to_string(&header)
        .unwrap_or_else(|e| panic!("{}: {e}; run xbox360/build.sh, which writes it", header.display()));
    let mut out = String::new();
    // enum { TRI_TRIANGLE_FRAG_SPIRV_BYTES = 500 };
    for line in text.lines() {
        let Some(body) = line.strip_prefix("enum { ").and_then(|l| l.strip_suffix(" };")) else {
            continue;
        };
        let (name, bytes) = body.split_once(" = ").expect("NAME = bytes");
        writeln!(out, "pub const {name}: usize = {bytes};").unwrap();
    }
    let path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("shader_sizes.rs");
    std::fs::write(path, out).unwrap();
}

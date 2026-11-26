//! Computes the self-test's expected value on the PC, so the title can check
//! that LLVM's 64-bit arithmetic (compiler-builtins on 32-bit PowerPC) gives
//! the console the same answer.
fn main() {
    let mut acc: u64 = 0x0123_4567_89AB_CDEF;
    for i in 0..1000u64 {
        acc = acc.wrapping_mul(6364136223846793005).wrapping_add(i);
    }
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("self_test.rs");
    std::fs::write(out, format!("const SELF_TEST_U64: u64 = {acc:#x};\n")).unwrap();
    println!("cargo:rerun-if-changed=build.rs");
}

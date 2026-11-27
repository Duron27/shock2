//! GLSL to SPIR-V for the Vulkan backend.
//!
//! The engine's shaders are GL-style GLSL: loose `uniform`s, unlocated
//! varyings. `libshaderc` (loaded at runtime) compiles them under glslang's
//! relaxed Vulkan rules, which gather the loose uniforms of each stage into
//! one uniform block. The SPIR-V is then read back ([`reflect`]) to learn
//! each uniform's byte offset, and rewritten ([`link`]) so both stages agree
//! on descriptor bindings and varying locations:
//!
//! * binding 0: the vertex stage's uniform block
//! * binding 1: the fragment stage's uniform block
//! * binding 2..: samplers, by name (one binding per name across stages)
//!
//! and each fragment input takes the location of the vertex output of the
//! same name.

use std::collections::BTreeMap;
use std::ffi::{CStr, CString, c_char, c_void};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stage {
    Vertex,
    Fragment,
}

// ---------------------------------------------------------------------------
// libshaderc, loaded at runtime (no headers or link-time dependency)
// ---------------------------------------------------------------------------

type Compiler = *mut c_void;
type Options = *mut c_void;
type CompileResult = *mut c_void;

#[cfg(not(target_os = "xbox360"))]
type ShadercLibrary = libloading::Library;
/// The Xbox 360 has no libshaderc: its shaders are compiled ahead of time.
#[cfg(target_os = "xbox360")]
type ShadercLibrary = ();

struct Shaderc {
    _lib: ShadercLibrary,
    compiler: Compiler,
    options: Options,
    compile: unsafe extern "C" fn(
        Compiler,
        *const c_char,
        usize,
        i32,
        *const c_char,
        *const c_char,
        Options,
    ) -> CompileResult,
    status: unsafe extern "C" fn(CompileResult) -> i32,
    length: unsafe extern "C" fn(CompileResult) -> usize,
    bytes: unsafe extern "C" fn(CompileResult) -> *const c_char,
    error: unsafe extern "C" fn(CompileResult) -> *const c_char,
    release: unsafe extern "C" fn(CompileResult),
}

#[cfg_attr(target_os = "xbox360", allow(dead_code))]
const SHADERC_TARGET_ENV_VULKAN: i32 = 0;
#[cfg_attr(target_os = "xbox360", allow(dead_code))]
const SHADERC_ENV_VERSION_VULKAN_1_0: u32 = 1 << 22;
#[cfg_attr(target_os = "xbox360", allow(dead_code))]
const SHADERC_VERTEX: i32 = 0;
#[cfg_attr(target_os = "xbox360", allow(dead_code))]
const SHADERC_FRAGMENT: i32 = 1;

impl Shaderc {
    #[cfg(target_os = "xbox360")]
    fn load() -> Result<Shaderc, String> {
        Err("no libshaderc on the Xbox 360: its shaders are compiled ahead of time".into())
    }

    #[cfg(not(target_os = "xbox360"))]
    fn load() -> Result<Shaderc, String> {
        let lib = [
            "libshaderc_shared.so.1",
            "libshaderc_shared.so",
            "shaderc_shared.dll",
        ]
        .iter()
        .find_map(|name| unsafe { libloading::Library::new(name) }.ok())
        .ok_or("libshaderc not found (Fedora: dnf install libshaderc)")?;
        unsafe {
            macro_rules! sym {
                ($name:literal) => {
                    *lib.get($name).map_err(|e| format!("libshaderc: {}", e))?
                };
            }
            let init: unsafe extern "C" fn() -> Compiler = sym!(b"shaderc_compiler_initialize\0");
            let options_init: unsafe extern "C" fn() -> Options =
                sym!(b"shaderc_compile_options_initialize\0");
            let set_env: unsafe extern "C" fn(Options, i32, u32) =
                sym!(b"shaderc_compile_options_set_target_env\0");
            let set_flag = |name: &[u8]| -> Result<unsafe extern "C" fn(Options, bool), String> {
                Ok(*lib.get(name).map_err(|e| format!("libshaderc: {e}"))?)
            };
            let relaxed = set_flag(b"shaderc_compile_options_set_vulkan_rules_relaxed\0")?;
            let auto_bind = set_flag(b"shaderc_compile_options_set_auto_bind_uniforms\0")?;
            let auto_map = set_flag(b"shaderc_compile_options_set_auto_map_locations\0")?;
            let compiler = init();
            let options = options_init();
            set_env(
                options,
                SHADERC_TARGET_ENV_VULKAN,
                SHADERC_ENV_VERSION_VULKAN_1_0,
            );
            relaxed(options, true);
            auto_bind(options, true);
            auto_map(options, true);
            Ok(Shaderc {
                compiler,
                options,
                compile: sym!(b"shaderc_compile_into_spv\0"),
                status: sym!(b"shaderc_result_get_compilation_status\0"),
                length: sym!(b"shaderc_result_get_length\0"),
                bytes: sym!(b"shaderc_result_get_bytes\0"),
                error: sym!(b"shaderc_result_get_error_message\0"),
                release: sym!(b"shaderc_result_release\0"),
                _lib: lib,
            })
        }
    }

    fn compile(&self, source: &str, stage: Stage) -> Result<Vec<u32>, String> {
        let text = CString::new(source).map_err(|e| e.to_string())?;
        let kind = match stage {
            Stage::Vertex => SHADERC_VERTEX,
            Stage::Fragment => SHADERC_FRAGMENT,
        };
        unsafe {
            let result = (self.compile)(
                self.compiler,
                text.as_ptr(),
                source.len(),
                kind,
                c"shader.glsl".as_ptr(),
                c"main".as_ptr(),
                self.options,
            );
            let out = if (self.status)(result) == 0 {
                let bytes = std::slice::from_raw_parts(
                    (self.bytes)(result) as *const u8,
                    (self.length)(result),
                );
                Ok(bytes
                    .chunks_exact(4)
                    .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                    .collect())
            } else {
                Err(CStr::from_ptr((self.error)(result))
                    .to_string_lossy()
                    .into_owned())
            };
            (self.release)(result);
            out
        }
    }
}

thread_local! {
    static SHADERC: Result<Shaderc, String> = Shaderc::load();
}

/// The engine's version-less GLSL as Vulkan GLSL 450. The vertex stage also
/// maps GL's -1..1 clip depth onto Vulkan's 0..1, so projection matrices
/// stay GL's, and multiplies y by the `flip_y` push constant: -1 on the
/// swapchain (Vulkan's framebuffer y points down), +1 into render targets so
/// their rows sit where GL put them.
///
/// The shader's own `main` becomes `shock_main`, writing the position into
/// an `out` parameter that stands in for `gl_Position`, and the real `main`
/// writes `gl_Position` once from it: never read back, since the Xbox 360's
/// shader compiler (xost) can't read an output (xbox360/README.md, "Shaders").
pub fn to_vulkan_glsl(source: &str, stage: Stage) -> String {
    to_vulkan_glsl_for(source, stage, Target::HOST)
}

/// The float uniform the console's `SHOCK_COUNT(n)` loop bounds scale by,
/// always 1.
pub const LOOP_SCALE: &str = "shockLoopScale";

/// Which GPU a shader is compiled for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Desktop,
    /// The Xbox 360, through vulkan4xenon and xost: `SHOCK_XBOX360` is
    /// defined, and bool and int uniforms are floats ([`console_uniforms`]).
    Xbox360,
}

impl Target {
    pub const HOST: Target = if cfg!(target_os = "xbox360") {
        Target::Xbox360
    } else {
        Target::Desktop
    };
}

/// [`to_vulkan_glsl`] for `target`.
pub fn to_vulkan_glsl_for(source: &str, stage: Stage, target: Target) -> String {
    let mut out = String::from(
        "#version 450\n#define highp\n#define mediump\n#define lowp\n#define SHOCK_VULKAN 1\n",
    );
    let console;
    let source = if target == Target::Xbox360 {
        // xost unrolls a loop of constant count, which the lit shaders' light
        // loops overflow 64 registers with, and keeps one bounded by int() of
        // a float uniform as a hardware loop: SHOCK_COUNT(n) is such a bound,
        // its scale 1 (vk_backend::create_program).
        out.push_str("#define SHOCK_XBOX360 1\n");
        // Declared only where used: a program without a uniform block (the
        // colour grade) has nowhere to put it.
        if source.contains("SHOCK_COUNT(") {
            out.push_str(&format!(
                "uniform float {LOOP_SCALE};\n#define SHOCK_COUNT(n) int({LOOP_SCALE} * float(n))\n"
            ));
        }
        console = console_uniforms(source);
        console.as_str()
    } else {
        out.push_str("#define SHOCK_COUNT(n) (n)\n");
        source
    };
    if stage == Stage::Vertex {
        // `invariant gl_Position;` keeps naming the real output, which main still writes.
        let replaced = replace_identifier(source, "gl_Position", "shock_position")
            .replace("invariant shock_position", "invariant gl_Position");
        let renamed = rename_main(&replaced);
        // `void shock_main(` -> `void shock_main(out vec4 shock_position`: main has no parameters.
        let open = renamed
            .find("void shock_main")
            .and_then(|at| renamed[at..].find('(').map(|p| at + p + 1))
            .expect("a vertex shader has a main");
        out.push_str(&renamed[..open]);
        out.push_str("out vec4 shock_position");
        out.push_str(&renamed[open..]);
        out.push_str(
            "\nlayout(push_constant) uniform ShockPush { float flip_y; } shock_push;\n\
             void main() {\n    vec4 shock_position;\n    shock_main(shock_position);\n\
             gl_Position = vec4(shock_position.x, shock_position.y * shock_push.flip_y,\n\
             (shock_position.z + shock_position.w) * 0.5, shock_position.w);\n}\n",
        );
    } else {
        out.push_str(source);
    }
    out
}

/// The console's uniforms: xost reads a block's int or bool only as D3D9 set
/// HLSL's, a float in its constant register, so each `uniform bool x;`
/// becomes `uniform float x;` read through `#define x (x != 0.0)`, and each
/// `uniform int`/`ivecN` a `float`/`vecN`; sources convert an int uniform
/// with `int()` where they need one. The console backend writes these
/// uniforms as floats (`vk_backend::set_ints`). One declaration per line.
fn console_uniforms(source: &str) -> String {
    let mut out = String::with_capacity(source.len() + 256);
    for line in source.split_inclusive('\n') {
        let decl = line.trim();
        let words: Vec<&str> = decl.trim_end_matches(';').split_whitespace().collect();
        let (ty, name) = match words.as_slice() {
            ["uniform", ty, name] if decl.ends_with(';') => (*ty, *name),
            ["uniform", "highp" | "mediump" | "lowp", ty, name] if decl.ends_with(';') => {
                (*ty, *name)
            }
            _ => {
                out.push_str(line);
                continue;
            }
        };
        let float = match ty {
            "bool" => "float",
            "int" => "float",
            "ivec2" => "vec2",
            "ivec3" => "vec3",
            "ivec4" => "vec4",
            _ => {
                out.push_str(line);
                continue;
            }
        };
        out.push_str(&format!("uniform {float} {name};\n"));
        if ty == "bool" && !name.contains('[') {
            out.push_str(&format!("#define {name} ({name} != 0.0)\n"));
        }
    }
    out
}

/// `source` with each whole-word `from` replaced by `to`.
fn replace_identifier(source: &str, from: &str, to: &str) -> String {
    let word = |c: char| c.is_alphanumeric() || c == '_';
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(i) = rest.find(from) {
        out.push_str(&rest[..i]);
        let whole = !out.ends_with(word) && !rest[i + from.len()..].starts_with(word);
        out.push_str(if whole { to } else { from });
        rest = &rest[i + from.len()..];
    }
    out.push_str(rest);
    out
}

fn rename_main(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(i) = rest.find("void") {
        out.push_str(&rest[..i]);
        let after = &rest[i + 4..];
        let trimmed = after.trim_start();
        let is_main = trimmed.starts_with("main")
            && trimmed[4..].trim_start().starts_with('(')
            && !after.starts_with(|c: char| c.is_alphanumeric() || c == '_')
            && !out.ends_with(|c: char| c.is_alphanumeric() || c == '_');
        if is_main {
            out.push_str("void shock_main");
            rest = &trimmed[4..];
        } else {
            out.push_str("void");
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

/// `SHOCK2_SHADER_DUMP=<dir>`: every shader the engine compiles and every
/// module it creates, written there for building them ahead of time (the
/// Xbox 360 has no runtime compiler; xbox360/README.md, "Shaders").
/// `compile()` writes `<hash>.<stage>.glsl` with the GLSL it was given and
/// `<hash>.<stage>.compiled.spv` with its SPIR-V; [`record_module`] writes
/// `<hash>.<stage>.spv`, the linked SPIR-V a module is made from. Each
/// `<hash>` is FNV-1a over the file's own content, as vulkan4xenon keys its
/// shader bundle by the SPIR-V's.
fn shader_dump_dir() -> Option<&'static std::path::Path> {
    static DIR: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::path::PathBuf::from(std::env::var_os("SHOCK2_SHADER_DUMP")?);
        std::fs::create_dir_all(&dir).ok()?;
        Some(dir)
    })
    .as_deref()
}

/// FNV-1a, 64-bit, over each word's bytes low byte first: vulkan4xenon's
/// hash of SPIR-V (its `src/shader.c`), used here for file names.
pub fn spirv_hash(words: &[u32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in words.iter().flat_map(|w| w.to_le_bytes()) {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn stage_suffix(stage: Stage) -> &'static str {
    match stage {
        Stage::Vertex => "vert",
        Stage::Fragment => "frag",
    }
}

fn spirv_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Records the SPIR-V a shader module is created from, when
/// `SHOCK2_SHADER_DUMP` is set ([`shader_dump_dir`]).
pub fn record_module(words: &[u32], stage: Stage) {
    if let Some(dir) = shader_dump_dir() {
        let name = format!("{:016x}.{}.spv", spirv_hash(words), stage_suffix(stage));
        let _ = std::fs::write(dir.join(name), spirv_bytes(words));
    }
}

pub(crate) fn source_hash(source: &str) -> u64 {
    spirv_hash(&source.bytes().map(u32::from).collect::<Vec<_>>())
}

/// Appends `<vertex module> <fragment module> <vertex GLSL> <fragment GLSL>`
/// hashes to `programs.txt` in the dump, tying each linked program's modules
/// to the sources they came from.
pub fn record_program(vertex_source: &str, fragment_source: &str, linked: &Linked) {
    if let Some(dir) = shader_dump_dir() {
        use std::io::Write;
        let line = format!(
            "{:016x} {:016x} {:016x} {:016x}\n",
            spirv_hash(&linked.vertex),
            spirv_hash(&linked.fragment),
            source_hash(vertex_source),
            source_hash(fragment_source)
        );
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("programs.txt"))
        {
            let _ = f.write_all(line.as_bytes());
        }
    }
}

/// With a dump, also compiles the program as the Xbox 360 will and writes
/// what the console needs into `<dump>/xbox360/`: each source's SPIR-V
/// (`<source hash>.<stage>.compiled.spv`, the table the console looks its
/// GLSL up in, having no shaderc) and each module it will hand
/// vkCreateShaderModule (`<module hash>.<stage>.spv`, which vulkan4xenon
/// looks its microcode up by), linked unless the program builds its own
/// layout (`linked` false, the colour grade).
pub fn record_console_program(vertex_source: &str, fragment_source: &str, linked: bool) {
    let Some(dir) = shader_dump_dir() else {
        return;
    };
    let dir = dir.join("xbox360");
    let _ = std::fs::create_dir_all(&dir);
    let mut modules = Vec::new();
    for (source, stage) in [
        (vertex_source, Stage::Vertex),
        (fragment_source, Stage::Fragment),
    ] {
        let base = format!("{:016x}.{}", source_hash(source), stage_suffix(stage));
        match compile_uncached(source, stage, Target::Xbox360) {
            Ok(words) => {
                let _ = std::fs::write(
                    dir.join(format!("{base}.compiled.spv")),
                    spirv_bytes(&words),
                );
                modules.push(words);
            }
            Err(e) => {
                let _ = std::fs::write(dir.join(format!("{base}.error.txt")), e);
                return;
            }
        }
    }
    let fragment = modules.pop().unwrap();
    let vertex = modules.pop().unwrap();
    let (vertex, fragment) = if linked {
        let l = link(vertex, fragment);
        (l.vertex, l.fragment)
    } else {
        (vertex, fragment)
    };
    for (words, stage) in [(vertex, Stage::Vertex), (fragment, Stage::Fragment)] {
        let name = format!("{:016x}.{}.spv", spirv_hash(&words), stage_suffix(stage));
        let _ = std::fs::write(dir.join(name), spirv_bytes(&words));
    }
}

pub fn compile(source: &str, stage: Stage) -> Result<Vec<u32>, String> {
    let result = compile_uncached(source, stage, Target::HOST);
    if let (Some(dir), Ok(words)) = (shader_dump_dir(), &result) {
        let base = format!("{:016x}.{}", source_hash(source), stage_suffix(stage));
        let _ = std::fs::write(dir.join(format!("{base}.glsl")), source);
        let _ = std::fs::write(dir.join(format!("{base}.compiled.spv")), spirv_bytes(words));
    }
    result
}

/// The console's GLSL -> SPIR-V table (xbox360/tools/shader_table.py),
/// which the title hands over before it makes a program: the console has no
/// shaderc.
static CONSOLE_TABLE: std::sync::OnceLock<&'static [u8]> = std::sync::OnceLock::new();

/// Give the console's [`compile`] its table. Only the first call counts.
pub fn set_console_table(table: &'static [u8]) {
    let _ = CONSOLE_TABLE.set(table);
}

/// `source`'s SPIR-V for `stage` in a table shader_table.py wrote.
fn table_lookup(table: &[u8], source: &str, stage: Stage) -> Result<Vec<u32>, String> {
    let word = |at: usize| -> Option<u32> {
        Some(u32::from_le_bytes(table.get(at..at + 4)?.try_into().ok()?))
    };
    if table.get(..4) != Some(b"S2ST".as_slice()) {
        return Err("not a shader table (shader_table.py)".into());
    }
    let (want, want_stage) = (source_hash(source), stage as u32);
    let mut at = 8;
    for _ in 0..word(4).ok_or("a truncated shader table")? {
        let hash = u64::from(word(at).ok_or("a truncated shader table")?)
            | u64::from(word(at + 4).ok_or("a truncated shader table")?) << 32;
        let entry_stage = word(at + 8).ok_or("a truncated shader table")?;
        let count = word(at + 12).ok_or("a truncated shader table")? as usize;
        let start = at + 16;
        at = start + count * 4;
        if hash == want && entry_stage == want_stage {
            return (0..count)
                .map(|i| word(start + i * 4).ok_or_else(|| "a truncated shader table".to_string()))
                .collect();
        }
    }
    Err(format!(
        "GLSL {want:016x}.{} isn't in the console's shader table: sweep the shaders again (xbox360/README.md, \"Shaders\")",
        stage_suffix(stage)
    ))
}

fn compile_uncached(source: &str, stage: Stage, target: Target) -> Result<Vec<u32>, String> {
    if cfg!(target_os = "xbox360") {
        let table = CONSOLE_TABLE
            .get()
            .ok_or("the console's shader table isn't set (set_console_table)")?;
        return table_lookup(table, source, stage);
    }
    let glsl = to_vulkan_glsl_for(source, stage, target);
    SHADERC.with(|compiler| match compiler {
        Ok(c) => c.compile(&glsl, stage).map_err(|e| {
            let numbered: String = glsl
                .lines()
                .enumerate()
                .map(|(i, l)| format!("{:4} {l}\n", i + 1))
                .collect();
            format!("{e}\n{numbered}")
        }),
        Err(e) => Err(e.clone()),
    })
}

// ---------------------------------------------------------------------------
// SPIR-V reflection and patching
// ---------------------------------------------------------------------------

const OP_NAME: u32 = 5;
const OP_MEMBER_NAME: u32 = 6;
const OP_TYPE_INT: u32 = 21;
const OP_TYPE_FLOAT: u32 = 22;
const OP_TYPE_VECTOR: u32 = 23;
const OP_TYPE_MATRIX: u32 = 24;
const OP_TYPE_IMAGE: u32 = 25;
const OP_TYPE_SAMPLED_IMAGE: u32 = 27;
const OP_TYPE_ARRAY: u32 = 28;
const OP_TYPE_STRUCT: u32 = 30;
const OP_TYPE_POINTER: u32 = 32;
const OP_CONSTANT: u32 = 43;
const OP_VARIABLE: u32 = 59;
const OP_DECORATE: u32 = 71;
const OP_MEMBER_DECORATE: u32 = 72;

const DEC_ARRAY_STRIDE: u32 = 6;
const DEC_MATRIX_STRIDE: u32 = 7;
const DEC_BUILTIN: u32 = 11;
const DEC_LOCATION: u32 = 30;
const DEC_BINDING: u32 = 33;
const DEC_DESCRIPTOR_SET: u32 = 34;
const DEC_OFFSET: u32 = 35;

const SC_UNIFORM_CONSTANT: u32 = 0;
const SC_INPUT: u32 = 1;
const SC_UNIFORM: u32 = 2;
const SC_OUTPUT: u32 = 3;

/// Where one uniform lives in its stage's block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Member {
    pub offset: u32,
    /// Bytes between array elements (0 when not an array).
    pub array_stride: u32,
    /// Bytes between matrix columns (0 when not a matrix).
    pub matrix_stride: u32,
    /// Array length (1 when not an array).
    pub count: u32,
}

/// What a stage's SPIR-V declares, plus where to patch it.
#[derive(Debug, Default)]
pub struct Reflection {
    /// The default uniform block: its size and members by name.
    pub block: Option<(u32, BTreeMap<String, Member>)>,
    block_binding_word: Option<usize>,
    block_set_word: Option<usize>,
    /// Samplers by name: the words holding each one's binding and set, and
    /// whether it samples a cube.
    samplers: BTreeMap<String, (usize, Option<usize>, bool)>,
    /// Non-builtin inputs: name -> (location word, location).
    pub inputs: BTreeMap<String, (usize, u32)>,
    /// Non-builtin outputs: name -> location.
    pub outputs: BTreeMap<String, u32>,
    /// Vertex inputs by location: (is integer, components).
    pub vertex_inputs: BTreeMap<u32, (bool, u32)>,
}

fn string_at(words: &[u32]) -> String {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

#[derive(Clone, Debug)]
enum Ty {
    Scalar {
        int: bool,
    },
    Vector {
        int: bool,
        n: u32,
    },
    Matrix {
        columns: u32,
        column: u32,
    },
    Array {
        length: u32,
    },
    Struct(Vec<u32>),
    /// An image type (`cube` when Dim is Cube) or a sampled image of one.
    Image {
        cube: bool,
    },
    SampledImage {
        image: u32,
    },
    Pointer {
        class: u32,
        pointee: u32,
    },
    Other,
}

pub fn reflect(spirv: &[u32]) -> Reflection {
    let mut names: BTreeMap<u32, String> = BTreeMap::new();
    let mut member_names: BTreeMap<(u32, u32), String> = BTreeMap::new();
    let mut decorations: BTreeMap<(u32, u32), (usize, u32)> = BTreeMap::new();
    let mut member_decorations: BTreeMap<(u32, u32, u32), u32> = BTreeMap::new();
    let mut types: BTreeMap<u32, Ty> = BTreeMap::new();
    let mut constants: BTreeMap<u32, u32> = BTreeMap::new();
    let mut variables: Vec<(u32, u32)> = Vec::new(); // (id, pointer type)

    let mut i = 5;
    while i < spirv.len() {
        let count = (spirv[i] >> 16) as usize;
        let op = spirv[i] & 0xffff;
        if count == 0 {
            break;
        }
        let w = &spirv[i..i + count];
        match op {
            OP_NAME => {
                names.insert(w[1], string_at(&w[2..]));
            }
            OP_MEMBER_NAME => {
                member_names.insert((w[1], w[2]), string_at(&w[3..]));
            }
            OP_DECORATE if count >= 3 => {
                let value = if count >= 4 { w[3] } else { 0 };
                decorations.insert((w[1], w[2]), (i + 3, value));
            }
            OP_MEMBER_DECORATE if count >= 5 => {
                member_decorations.insert((w[1], w[2], w[3]), w[4]);
            }
            OP_TYPE_INT => {
                types.insert(w[1], Ty::Scalar { int: true });
            }
            OP_TYPE_FLOAT => {
                types.insert(w[1], Ty::Scalar { int: false });
            }
            OP_TYPE_VECTOR => {
                let int = matches!(types.get(&w[2]), Some(Ty::Scalar { int: true }));
                types.insert(w[1], Ty::Vector { int, n: w[3] });
            }
            OP_TYPE_MATRIX => {
                let column = match types.get(&w[2]) {
                    Some(Ty::Vector { n, .. }) => *n,
                    _ => 4,
                };
                types.insert(
                    w[1],
                    Ty::Matrix {
                        columns: w[3],
                        column,
                    },
                );
            }
            OP_TYPE_ARRAY => {
                types.insert(w[1], Ty::Array { length: w[3] });
            }
            OP_TYPE_IMAGE if count >= 4 => {
                types.insert(w[1], Ty::Image { cube: w[3] == 3 });
            }
            OP_TYPE_SAMPLED_IMAGE => {
                types.insert(w[1], Ty::SampledImage { image: w[2] });
            }
            OP_TYPE_STRUCT => {
                types.insert(w[1], Ty::Struct(w[2..].to_vec()));
            }
            OP_TYPE_POINTER => {
                types.insert(
                    w[1],
                    Ty::Pointer {
                        class: w[2],
                        pointee: w[3],
                    },
                );
            }
            OP_CONSTANT if count >= 4 => {
                constants.insert(w[2], w[3]);
            }
            OP_VARIABLE => {
                variables.push((w[2], w[1]));
            }
            _ => {
                types.entry(0).or_insert(Ty::Other);
            }
        }
        i += count;
    }

    // Size in bytes of a member of type `ty` with the given strides.
    fn size_of(
        types: &BTreeMap<u32, Ty>,
        constants: &BTreeMap<u32, u32>,
        ty: u32,
        array_stride: u32,
        matrix_stride: u32,
    ) -> u32 {
        match types.get(&ty) {
            Some(Ty::Scalar { .. }) => 4,
            Some(Ty::Vector { n, .. }) => 4 * n,
            Some(Ty::Matrix { columns, column }) => {
                if matrix_stride > 0 {
                    matrix_stride * columns
                } else {
                    4 * columns * column
                }
            }
            Some(Ty::Array { length, .. }) => {
                array_stride * constants.get(length).copied().unwrap_or(1)
            }
            _ => 0,
        }
    }

    let mut reflection = Reflection::default();
    for (id, pointer) in variables {
        let Some(Ty::Pointer { class, pointee }) = types.get(&pointer).cloned() else {
            continue;
        };
        let builtin = decorations.contains_key(&(id, DEC_BUILTIN));
        match class {
            SC_UNIFORM => {
                let Some(Ty::Struct(members)) = types.get(&pointee).cloned() else {
                    continue;
                };
                let mut map = BTreeMap::new();
                let mut size = 0;
                for (m, &member_ty) in members.iter().enumerate() {
                    let m = m as u32;
                    let offset = member_decorations
                        .get(&(pointee, m, DEC_OFFSET))
                        .copied()
                        .unwrap_or(0);
                    let matrix_stride = member_decorations
                        .get(&(pointee, m, DEC_MATRIX_STRIDE))
                        .copied()
                        .unwrap_or(0);
                    let (array_stride, count) = match types.get(&member_ty) {
                        Some(Ty::Array { length, .. }) => (
                            decorations
                                .get(&(member_ty, DEC_ARRAY_STRIDE))
                                .map(|d| d.1)
                                .unwrap_or(16),
                            constants.get(length).copied().unwrap_or(1),
                        ),
                        _ => (0, 1),
                    };
                    // An array of matrices carries its matrix stride on the
                    // member too; size the element from it.
                    let bytes = size_of(&types, &constants, member_ty, array_stride, matrix_stride);
                    size = size.max(offset + bytes);
                    if let Some(name) = member_names.get(&(pointee, m)) {
                        map.insert(
                            name.clone(),
                            Member {
                                offset,
                                array_stride,
                                matrix_stride,
                                count,
                            },
                        );
                    }
                }
                reflection.block = Some((size.next_multiple_of(16), map));
                reflection.block_binding_word = decorations.get(&(id, DEC_BINDING)).map(|d| d.0);
                reflection.block_set_word = decorations.get(&(id, DEC_DESCRIPTOR_SET)).map(|d| d.0);
            }
            SC_UNIFORM_CONSTANT => {
                if let (Some(name), Some(binding)) =
                    (names.get(&id), decorations.get(&(id, DEC_BINDING)))
                {
                    let set = decorations.get(&(id, DEC_DESCRIPTOR_SET)).map(|d| d.0);
                    let cube = match types.get(&pointee) {
                        Some(Ty::SampledImage { image }) => {
                            matches!(types.get(image), Some(Ty::Image { cube: true }))
                        }
                        _ => false,
                    };
                    reflection
                        .samplers
                        .insert(name.clone(), (binding.0, set, cube));
                }
            }
            SC_INPUT if !builtin => {
                if let (Some(name), Some(location)) =
                    (names.get(&id), decorations.get(&(id, DEC_LOCATION)))
                {
                    reflection
                        .inputs
                        .insert(name.clone(), (location.0, location.1));
                    let (int, n) = match types.get(&pointee) {
                        Some(Ty::Scalar { int }) => (*int, 1),
                        Some(Ty::Vector { int, n }) => (*int, *n),
                        _ => (false, 4),
                    };
                    reflection.vertex_inputs.insert(location.1, (int, n));
                }
            }
            SC_OUTPUT if !builtin => {
                if let (Some(name), Some(location)) =
                    (names.get(&id), decorations.get(&(id, DEC_LOCATION)))
                {
                    reflection.outputs.insert(name.clone(), location.1);
                }
            }
            _ => {}
        }
    }
    reflection
}

/// A linked program's layout: what each stage's block holds and which
/// binding each sampler name uses.
#[derive(Debug)]
pub struct Linked {
    pub vertex: Vec<u32>,
    pub fragment: Vec<u32>,
    /// (size, members) of the vertex block (binding 0) and fragment block (binding 1).
    pub vertex_block: Option<(u32, BTreeMap<String, Member>)>,
    pub fragment_block: Option<(u32, BTreeMap<String, Member>)>,
    /// Sampler name -> binding and use.
    pub samplers: BTreeMap<String, SamplerUse>,
    /// Vertex inputs the program reads, by location.
    pub vertex_inputs: BTreeMap<u32, (bool, u32)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SamplerUse {
    pub binding: u32,
    pub vertex: bool,
    pub fragment: bool,
    pub cube: bool,
}

pub const VERTEX_BLOCK_BINDING: u32 = 0;
pub const FRAGMENT_BLOCK_BINDING: u32 = 1;
const FIRST_SAMPLER_BINDING: u32 = 2;

/// Rewrite both stages' bindings and the fragment inputs' locations (see the
/// module docs) and return the program's layout.
pub fn link(mut vertex: Vec<u32>, mut fragment: Vec<u32>) -> Linked {
    let vr = reflect(&vertex);
    let fr = reflect(&fragment);

    let mut samplers: BTreeMap<String, SamplerUse> = BTreeMap::new();
    for (name, &(_, _, cube)) in vr.samplers.iter().chain(fr.samplers.iter()) {
        let next = FIRST_SAMPLER_BINDING + samplers.len() as u32;
        samplers.entry(name.clone()).or_insert(SamplerUse {
            binding: next,
            vertex: false,
            fragment: false,
            cube,
        });
    }
    for (words, reflection, block_binding, is_vertex) in [
        (&mut vertex, &vr, VERTEX_BLOCK_BINDING, true),
        (&mut fragment, &fr, FRAGMENT_BLOCK_BINDING, false),
    ] {
        if let Some(word) = reflection.block_binding_word {
            words[word] = block_binding;
        }
        if let Some(word) = reflection.block_set_word {
            words[word] = 0;
        }
        for (name, (binding_word, set_word, _)) in &reflection.samplers {
            let entry = samplers.get_mut(name).unwrap();
            words[*binding_word] = entry.binding;
            if let Some(set_word) = set_word {
                words[*set_word] = 0;
            }
            if is_vertex {
                entry.vertex = true;
            } else {
                entry.fragment = true;
            }
        }
    }
    for (name, (word, _)) in &fr.inputs {
        if let Some(&location) = vr.outputs.get(name) {
            fragment[*word] = location;
        }
    }
    Linked {
        vertex,
        fragment,
        vertex_block: vr.block,
        fragment_block: fr.block,
        samplers,
        vertex_inputs: vr.vertex_inputs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_position_is_written_once_and_never_read() {
        let src = "invariant gl_Position;\nin vec3 p;\nvoid main() { gl_Position = vec4(p, 1.0); gl_Position.x += 1.0; }";
        let out = to_vulkan_glsl(src, Stage::Vertex);
        assert!(out.contains("invariant gl_Position;"), "{out}");
        assert!(out.contains("void shock_main(out vec4 shock_position)"));
        assert!(out.contains("shock_position = vec4(p, 1.0); shock_position.x += 1.0;"));
        assert_eq!(out.matches("gl_Position").count(), 2, "{out}");
        assert!(!out.contains("gl_Position."), "{out}");
    }

    /// `SHOCK2_CONSOLE_VS=<a.glsl> SHOCK2_CONSOLE_FS=<b.glsl> SHOCK2_CONSOLE_OUT=<dir>
    /// cargo test -p engine --lib console_link_files -- --ignored`: links a
    /// dumped program as the Xbox 360 will, into `<dir>/{vert,frag}.spv`, to
    /// try a source change through xost without running the game.
    #[test]
    #[ignore]
    fn console_link_files() {
        let read = |k: &str| std::fs::read_to_string(std::env::var(k).expect(k)).expect(k);
        let out = std::path::PathBuf::from(
            std::env::var("SHOCK2_CONSOLE_OUT").expect("SHOCK2_CONSOLE_OUT"),
        );
        let vs =
            compile_uncached(&read("SHOCK2_CONSOLE_VS"), Stage::Vertex, Target::Xbox360).unwrap();
        let fs =
            compile_uncached(&read("SHOCK2_CONSOLE_FS"), Stage::Fragment, Target::Xbox360).unwrap();
        let linked = link(vs, fs);
        std::fs::write(out.join("vert.spv"), spirv_bytes(&linked.vertex)).unwrap();
        std::fs::write(out.join("frag.spv"), spirv_bytes(&linked.fragment)).unwrap();
    }

    #[test]
    fn table_lookup_finds_a_source_by_hash_and_stage() {
        let entry = |source: &str, stage: u32, words: &[u32]| {
            let mut e = Vec::new();
            e.extend_from_slice(&source_hash(source).to_le_bytes());
            e.extend_from_slice(&stage.to_le_bytes());
            e.extend_from_slice(&(words.len() as u32).to_le_bytes());
            words
                .iter()
                .for_each(|w| e.extend_from_slice(&w.to_le_bytes()));
            e
        };
        let mut table = b"S2ST".to_vec();
        table.extend_from_slice(&3u32.to_le_bytes());
        table.extend(entry("a", 0, &[1, 2]));
        table.extend(entry("a", 1, &[3]));
        table.extend(entry("b", 0, &[0x0723_0203, 5, 6]));
        assert_eq!(table_lookup(&table, "a", Stage::Fragment), Ok(vec![3]));
        assert_eq!(
            table_lookup(&table, "b", Stage::Vertex),
            Ok(vec![0x0723_0203, 5, 6])
        );
        assert!(
            table_lookup(&table, "b", Stage::Fragment)
                .unwrap_err()
                .contains("isn't in")
        );
        assert!(table_lookup(&table[..20], "b", Stage::Vertex).is_err());
    }

    #[test]
    fn console_bool_and_int_uniforms_are_floats() {
        let src = "uniform bool lit;\n  uniform highp int modes[6];\nuniform ivec2 origin;\nuniform float k;\nvoid main() { if (lit) {} }\n";
        let out = console_uniforms(src);
        assert!(
            out.contains("uniform float lit;\n#define lit (lit != 0.0)\n"),
            "{out}"
        );
        assert!(out.contains("uniform float modes[6];\n"), "{out}");
        assert!(out.contains("uniform vec2 origin;\n"), "{out}");
        assert!(out.contains("uniform float k;\n"), "{out}");
        assert!(out.contains("if (lit)"), "{out}");
        let desktop = to_vulkan_glsl_for(src, Stage::Fragment, Target::Desktop);
        assert!(desktop.contains("uniform bool lit;") && !desktop.contains("SHOCK_XBOX360"));
    }

    #[test]
    fn identifier_replacement_is_whole_word() {
        assert_eq!(
            replace_identifier("gl_Position gl_PositionX xgl_Position", "gl_Position", "p"),
            "p gl_PositionX xgl_Position"
        );
    }

    #[test]
    fn only_the_real_main_is_renamed() {
        let src = "void mainish() {}\nvoid helper(){}\nvoid  main ( ) { mainish(); }";
        let out = rename_main(src);
        assert!(out.contains("void mainish()"));
        assert!(out.contains("void shock_main ( )") || out.contains("void shock_main( )"));
        assert_eq!(out.matches("shock_main").count(), 1);
    }

    /// Needs libshaderc; skipped where it is not installed.
    #[test]
    fn loose_uniforms_link_into_two_blocks_and_shared_samplers() {
        let vs = "layout (location = 0) in vec3 pos;\nlayout (location = 1) in vec2 uv;\n\
                  uniform mat4 world;\nuniform mat4 projection;\nout vec2 texCoord;\nout float fade;\n\
                  void main() { texCoord = uv; fade = 1.0; gl_Position = projection * world * vec4(pos, 1.0); }";
        let fs = "in float fade;\nin vec2 texCoord;\nout vec4 color;\nuniform sampler2D tex;\n\
                  uniform vec4 tint;\nuniform float lights[4];\nuniform mat3 rot;\nuniform samplerCube sky;\n\
                  void main() { color = texture(tex, texCoord) * tint * fade * lights[2] * rot[1][1]\n\
                  + texture(sky, vec3(texCoord, 1.0)); }";
        let (Ok(v), Ok(f)) = (compile(vs, Stage::Vertex), compile(fs, Stage::Fragment)) else {
            eprintln!("libshaderc unavailable; skipping");
            return;
        };
        let linked = link(v, f);
        let (_, vb) = linked.vertex_block.as_ref().unwrap();
        assert_eq!(vb["world"].matrix_stride, 16);
        assert!(vb.contains_key("projection"));
        let (size, fb) = linked.fragment_block.as_ref().unwrap();
        assert_eq!(fb["lights"].count, 4);
        assert_eq!(fb["lights"].array_stride, 16, "std140 pads float arrays");
        assert_eq!(fb["rot"].matrix_stride, 16, "std140 pads mat3 columns");
        assert!(*size >= fb["rot"].offset + 48);
        let tex = linked.samplers["tex"];
        assert!(tex.fragment && !tex.vertex && !tex.cube);
        assert!(linked.samplers["sky"].cube);
        assert_ne!(linked.samplers["sky"].binding, tex.binding);
        assert_eq!(linked.vertex_inputs[&0], (false, 3));
        // Fragment inputs now sit at the vertex outputs' locations.
        let vr = reflect(&linked.vertex);
        let fr = reflect(&linked.fragment);
        for (name, (_, location)) in &fr.inputs {
            assert_eq!(vr.outputs[name], *location, "{name}");
        }
    }
}

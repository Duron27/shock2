//! Parses the game's data the way the game does and fingerprints what comes
//! out, so the Xbox 360 (big-endian) and the PC (little-endian) can be
//! compared line by line: a reader that assumes the host's byte order gives
//! different counts or hashes on the console.
//!
//! Per file it reports counts and FNV-1a hashes: of every parsed property
//! and link through their `Debug` forms (Rust formats floats the same on
//! every target), and of the bit patterns of every world vertex, normal,
//! cell centre and AI path vertex. Lines that start with `time:` and `mem:`
//! are timings and sizes, which differ between the two by design.

use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

use dark::properties::get;
use dark::ss2_entity_info::SystemShock2EntityInfo;
use dark::{gamesys, mission, motion::MotionDB};
use engine::assets::asset_paths::{AbstractAssetPath, ReadableAndSeekable};

/// FNV-1a, 64-bit: small, and the same on every target.
struct Fnv(u64);

impl Fnv {
    fn new() -> Fnv {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn f32(&mut self, v: f32) {
        // The value's bits, in one byte order on every target.
        self.bytes(&v.to_bits().to_le_bytes());
    }
    fn floats(&mut self, values: &[f32]) {
        for &v in values {
            self.f32(v);
        }
    }
}

impl std::fmt::Write for Fnv {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.bytes(s.as_bytes());
        Ok(())
    }
}

/// `text` with every `0x<hex>` dropped: `WrappedProperty`'s `Debug` prints
/// its accumulator, a function pointer, as an address that changes per run.
fn without_addresses(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find("0x") {
        out.push_str(&rest[..at]);
        let hex = rest[at + 2..]
            .bytes()
            .take_while(u8::is_ascii_hexdigit)
            .count();
        out.push_str("0x_");
        rest = &rest[at + 2 + hex..];
    }
    out.push_str(rest);
    out
}

fn entity_lines(name: &str, info: &SystemShock2EntityInfo, out: &mut dyn FnMut(&str)) {
    let mut ids: Vec<_> = info.entity_to_properties.keys().copied().collect();
    ids.sort_unstable();
    let (mut props, mut hash) = (0usize, Fnv::new());
    // Each property's text, one file per data file, to diff the two machines' with: DATACHECK_DUMP=<dir> on the PC,
    // always on the console.
    #[cfg(target_os = "xbox360")]
    let dump = Some(r"game:\datacheck_props".to_owned());
    #[cfg(not(target_os = "xbox360"))]
    let dump = std::env::var("DATACHECK_DUMP").ok();
    if let Some(dir) = &dump {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut dumped = String::new();
    for id in &ids {
        let _ = write!(hash, "{id}:");
        // In one order: an entity's properties come out of a HashMap.
        let mut texts: Vec<String> = info.entity_to_properties[id]
            .iter()
            .map(|p| without_addresses(&format!("{p:?}")))
            .collect();
        texts.sort_unstable();
        for text in &texts {
            props += 1;
            let _ = write!(hash, "{text};");
            if dump.is_some() {
                let _ = writeln!(dumped, "{name} {id} {text}");
            }
        }
    }
    if let Some(dir) = dump {
        let _ = std::fs::write(Path::new(&dir).join(format!("{name}.props")), dumped);
    }
    let mut templates: Vec<_> = info.template_to_links.keys().copied().collect();
    templates.sort_unstable();
    let (mut links, mut link_hash) = (0usize, Fnv::new());
    for id in &templates {
        let to = &info.template_to_links[id].to_links;
        links += to.len();
        let _ = write!(link_hash, "{id}:{:?};", to);
    }
    let unparsed: usize = info.unparsed_properties.values().map(Vec::len).sum();
    out(&format!(
        "{name}: entities {}, properties {props} {:016x}, link sources {}, links {links} {:016x}, unparsed properties {unparsed}",
        ids.len(),
        hash.0,
        templates.len(),
        link_hash.0
    ));
}

fn level_lines(name: &str, level: &mission::SystemShock2Level, out: &mut dyn FnMut(&str)) {
    let (mut verts, mut geometry) = (0usize, Fnv::new());
    for g in &level.all_geometry {
        verts += g.verts.len();
        geometry.bytes(&g.texture_idx.to_le_bytes());
        for v in &g.verts {
            geometry.floats(&[v.position.x, v.position.y, v.position.z, v.uv.x, v.uv.y]);
            geometry.floats(&[
                v.lightmap_uv.x,
                v.lightmap_uv.y,
                v.normal.x,
                v.normal.y,
                v.normal.z,
            ]);
        }
    }
    let mut cells = Fnv::new();
    let mut polygons = 0usize;
    for c in &level.cells {
        cells.floats(&[c.center.x, c.center.y, c.center.z, c.radius]);
        cells.bytes(&[c.flags, c.portal_count]);
        polygons += c.polygons.len();
    }
    out(&format!(
        "{name}: geometry {} pieces, {verts} vertices {:016x}; cells {}, polygons {polygons} {:016x}; textures {}, lights {}",
        level.all_geometry.len(),
        geometry.0,
        level.cells.len(),
        cells.0,
        level.textures.0.len(),
        level.light_table.static_lights().len()
    ));
    match &level.path_database {
        Some(paths) => {
            let mut hash = Fnv::new();
            for v in &paths.vertices {
                hash.floats(&[v.x, v.y, v.z]);
            }
            out(&format!(
                "{name}: AI paths: cells {}, vertices {} {:016x}, links {}",
                paths.cells.len(),
                paths.vertices.len(),
                hash.0,
                paths.links.len()
            ));
        }
        None => out(&format!("{name}: AI paths: none")),
    }
    entity_lines(name, &level.entity_info, out);
}

fn reader(
    paths: &dyn AbstractAssetPath,
    base: &str,
    name: &str,
) -> Option<Box<dyn ReadableAndSeekable>> {
    paths
        .get_reader(base.to_owned(), name.to_ascii_lowercase())
        .or_else(|| paths.get_reader(base.to_owned(), name.to_owned()))
        .map(std::cell::RefCell::into_inner)
}

/// Every fingerprint line for the data under `root`, through `out`.
pub fn run(root: &Path, out: &mut dyn FnMut(&str)) {
    let start = Instant::now();
    let base = root.to_string_lossy().into_owned();
    let paths = shock2vr::data_files::asset_paths(root);
    let (properties, links, links_with_data) = get::<Box<dyn ReadableAndSeekable>>();
    out(&format!(
        "definitions: {} properties, {} links, {} links with data",
        properties.len(),
        links.len(),
        links_with_data.len()
    ));

    let t = Instant::now();
    let Some(mut game) = reader(&*paths, &base, "shock2.gam") else {
        out(&format!("FAIL: shock2.gam not found under {base}"));
        return;
    };
    let gamesys = gamesys::read(&mut game, &links, &links_with_data, &properties);
    drop(game);
    out(&format!("time: shock2.gam {:?}", t.elapsed()));
    entity_lines("shock2.gam", &gamesys.entity_info, out);

    let t = Instant::now();
    match reader(&*paths, &base, "motiondb.bin") {
        Some(mut r) => {
            let db = MotionDB::read(&mut r);
            out(&format!("time: motiondb.bin {:?}", t.elapsed()));
            let mut tags = db.get_all_tag_names();
            tags.sort();
            let mut hash = Fnv::new();
            let _ = write!(hash, "{:?}", tags);
            out(&format!(
                "motiondb.bin: {} creature types, {} tags {:016x}",
                db.get_creature_type_count(),
                tags.len(),
                hash.0
            ));
        }
        None => out("FAIL: motiondb.bin not found"),
    }

    let missions = shock2vr::data_files::mission_names(root);
    out(&format!("missions: {}", missions.len()));
    for name in &missions {
        let t = Instant::now();
        let Some(mut r) = reader(&*paths, &base, name) else {
            out(&format!("FAIL: {name} not found"));
            continue;
        };
        let level = mission::read(
            &*paths,
            &base,
            &mut r,
            &gamesys,
            &links,
            &links_with_data,
            &properties,
        );
        out(&format!("time: {name} {:?}", t.elapsed()));
        level_lines(name, &level, out);
    }
    out(&format!("time: all {:?}", start.elapsed()));
}

/// The console title: the same lines in `xbox360_datacheck.log` beside the
/// `.xex`, for the data copied to `Data` beside it.
#[cfg(target_os = "xbox360")]
#[unsafe(no_mangle)]
pub extern "C" fn main(_argc: i32, _argv: *const *const u8) -> i32 {
    unsafe extern "C" {
        fn xsdk_log_write(format: *const std::ffi::c_char, ...);
        fn exit(status: i32) -> !;
    }
    let mut out = |line: &str| unsafe {
        xsdk_log_write(
            c"%.*s".as_ptr(),
            line.len() as i32,
            line.as_ptr() as *const std::ffi::c_char,
        )
    };
    std::panic::set_hook(Box::new(|info| {
        let line = format!("panic: {info}");
        unsafe {
            xsdk_log_write(
                c"%.*s".as_ptr(),
                line.len() as i32,
                line.as_ptr() as *const std::ffi::c_char,
            );
            exit(3)
        }
    }));
    out(
        "xbox360_datacheck: the game's data parsed on the console; compare with the PC's lines. It ends by itself",
    );
    run(Path::new(r"game:\Data"), &mut out);
    out("xbox360_datacheck: done");
    0
}

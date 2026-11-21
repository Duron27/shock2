//! Inspect System Shock (1994) data without launching anything.
//!
//! ```text
//! cargo s1q res gamepal.res            # list a resource file's directory
//! cargo s1q res archive.dat --id 4005  # one resource: items and a hex peek
//! cargo s1q check                      # load every resource in every file
//! ```

use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use ss1::ResFile;

#[derive(Parser)]
#[command(about = "Inspect System Shock (1994) data")]
struct Cli {
    /// The game's data folder (default: $SS1_DATA_PATH, or an install beside the repo).
    #[arg(long, global = true)]
    data: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// List a resource file, or show one resource with --id.
    Res {
        file: String,
        #[arg(long)]
        id: Option<String>,
        /// Bytes of each item to hex-dump.
        #[arg(long, default_value_t = 32)]
        peek: usize,
    },
    /// Load and decompress every resource in every resource file.
    Check,
    /// Decode every 3D model in obj3d.res and summarize it.
    Models,
    /// Summarize a level from archive.dat (or list levels) and write a
    /// top-down PNG: floor height as brightness, solid tiles black.
    Level {
        number: Option<u16>,
        #[arg(long, default_value = "ss1_level.png")]
        out: PathBuf,
    },
    /// List a level's objects by class and render type; with --art, export
    /// each object type's first world frame as a PNG.
    Objects {
        level: u16,
        #[arg(long)]
        art: Option<PathBuf>,
        /// Print each object of this class (e.g. door) with its position.
        #[arg(long)]
        class: Option<String>,
    },
    /// Export image resources as PNGs (all of a file's images, or one --id).
    Image {
        file: String,
        #[arg(long)]
        id: Option<String>,
        /// Output folder.
        #[arg(long, default_value = "ss1_images")]
        out: PathBuf,
        /// Also write one contact sheet of every exported image.
        #[arg(long)]
        sheet: bool,
    },
}

fn parse_id(text: &str) -> Result<u16> {
    let parsed = match text.strip_prefix("0x") {
        Some(hex) => u16::from_str_radix(hex, 16),
        None => text.parse(),
    };
    parsed.with_context(|| format!("bad resource id {text:?}"))
}

fn hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let data = cli
        .data
        .or_else(ss1::paths::data_root)
        .ok_or_else(|| anyhow!("no System Shock data found; pass --data or set SS1_DATA_PATH"))?;

    match cli.command {
        Command::Res { file, id, peek } => {
            let path = data.join(&file);
            let res = ResFile::open(&path).with_context(|| path.display().to_string())?;
            match id {
                None => {
                    println!(
                        "{} - {:?}, {} resources",
                        file,
                        res.comment,
                        res.entries().len()
                    );
                    println!(
                        "{:>6} {:>6} {:>9} {:>9} {:<5} {}",
                        "id", "hex", "size", "csize", "flags", "type"
                    );
                    for e in res.entries() {
                        let flags = format!(
                            "{}{}",
                            if e.is_compressed() { "L" } else { "-" },
                            if e.is_compound() { "C" } else { "-" }
                        );
                        println!(
                            "{:>6} {:>#6x} {:>9} {:>9} {:<5} {}",
                            e.id,
                            e.id,
                            e.size,
                            e.csize,
                            flags,
                            e.res_type.name()
                        );
                    }
                }
                Some(id) => {
                    let resource = res.load(parse_id(&id)?)?;
                    let e = resource.entry;
                    println!(
                        "id {} ({:#x}) type {} size {} csize {} compressed {} compound {} items {}",
                        e.id,
                        e.id,
                        e.res_type.name(),
                        e.size,
                        e.csize,
                        e.is_compressed(),
                        e.is_compound(),
                        resource.len()
                    );
                    for i in 0..resource.len() {
                        let item = resource.item(i).unwrap();
                        println!(
                            "  [{i}] {} bytes: {}",
                            item.len(),
                            hex(&item[..item.len().min(peek)])
                        );
                    }
                }
            }
        }
        Command::Image {
            file,
            id,
            out,
            sheet,
        } => {
            let palette = ss1::Palette::load_game(&data)?;
            let res = ResFile::open(data.join(&file))?;
            let ids: Vec<u16> = match id {
                Some(id) => vec![parse_id(&id)?],
                None => res
                    .entries()
                    .iter()
                    .filter(|e| e.res_type == ss1::ResType::Image)
                    .map(|e| e.id)
                    .collect(),
            };
            std::fs::create_dir_all(&out)?;
            let (mut written, mut failed) = (0, 0);
            let mut decoded = Vec::new();
            for id in ids {
                let resource = res.load(id)?;
                for index in 0..resource.len() {
                    let item = resource.item(index).unwrap();
                    if item.is_empty() {
                        continue;
                    }
                    match ss1::Image::parse(item) {
                        Ok(image) if image.width > 0 && image.height > 0 => {
                            let path = out.join(format!("{id}_{index}.png"));
                            image::save_buffer(
                                &path,
                                &image.to_rgba(&palette),
                                image.width as u32,
                                image.height as u32,
                                image::ColorType::Rgba8,
                            )?;
                            written += 1;
                            if sheet {
                                decoded.push(image);
                            }
                        }
                        Ok(_) => {}
                        Err(err) => {
                            failed += 1;
                            if failed <= 5 {
                                println!("{id}[{index}]: {err}");
                            }
                        }
                    }
                }
            }
            if sheet && !decoded.is_empty() {
                write_sheet(&decoded, &palette, &out.join("sheet.png"))?;
            }
            println!(
                "wrote {written} images to {} ({failed} failed)",
                out.display()
            );
        }
        Command::Level { number, out } => {
            let archive = ResFile::open(data.join("archive.dat"))?;
            let Some(number) = number else {
                for n in ss1::Level::numbers(&archive) {
                    let level = ss1::Level::load(&archive, n)?;
                    let open = level.tiles.iter().filter(|t| !t.kind.is_solid()).count();
                    println!(
                        "level {n}: {}x{} z_shift {} open tiles {open} textures {}",
                        level.width,
                        level.height,
                        level.z_shift,
                        level.textures.len()
                    );
                }
                return Ok(());
            };
            let level = ss1::Level::load(&archive, number)?;
            let mut types = std::collections::BTreeMap::new();
            let mut modes = std::collections::BTreeMap::new();
            for t in &level.tiles {
                *types.entry(t.raw_type).or_insert(0) += 1;
                if t.slope > 0 {
                    *modes.entry(format!("{:?}", t.slope_mode())).or_insert(0) += 1;
                }
            }
            println!(
                "level {number}: {}x{} z_shift {}",
                level.width, level.height, level.z_shift
            );
            println!("tile types: {types:?}");
            println!("sloped tiles by mode: {modes:?}");
            println!("textures: {:?}", level.textures);
            const SCALE: u32 = 8;
            let mut img =
                image::RgbImage::new(level.width as u32 * SCALE, level.height as u32 * SCALE);
            for (px, py, pixel) in img.enumerate_pixels_mut() {
                let (x, y) = ((px / SCALE) as usize, (py / SCALE) as usize);
                let tile = level.tile(x, level.height - 1 - y).unwrap();
                *pixel = match tile.kind {
                    ss1::TileType::Solid => image::Rgb([0, 0, 0]),
                    kind => {
                        let v = 60 + tile.floor as u32 * 6;
                        let v = v.min(255) as u8;
                        match kind {
                            ss1::TileType::Slope { .. } => image::Rgb([v, v, 40]),
                            ss1::TileType::SolidDiagonal(_) => image::Rgb([40, v, v]),
                            ss1::TileType::Other(_) => image::Rgb([v, 40, v]),
                            _ => image::Rgb([v, v, v]),
                        }
                    }
                };
            }
            img.save(&out)?;
            println!("wrote {}", out.display());
        }
        Command::Objects { level, art, class } => {
            let archive = ResFile::open(data.join("archive.dat"))?;
            let props = ss1::objects::ObjProps::load(&data)?;
            let objs = ss1::objects::load_level_objects(&archive, level)?;
            let mut by_class = std::collections::BTreeMap::new();
            for o in &objs {
                let render = o
                    .triple()
                    .map(|t| format!("{:?}", props.props[t].render_type))
                    .unwrap_or_else(|| "bad-triple".into());
                let class = ss1::objects::CLASS_NAMES
                    .get(o.class as usize)
                    .copied()
                    .unwrap_or("?");
                *by_class.entry((class, render)).or_insert(0) += 1;
            }
            println!("level {level}: {} objects", objs.len());
            if let Some(name) = &class {
                for o in objs.iter().filter(|o| {
                    ss1::objects::CLASS_NAMES.get(o.class as usize) == Some(&name.as_str())
                }) {
                    println!(
                        "  #{:<4} {}/{}/{} {:?} model {} at ({:.2}, {:.2}) z {} heading {} pitch {} bank {} frame {}",
                        o.id,
                        o.class,
                        o.subclass,
                        o.kind,
                        o.triple().map(|t| props.props[t].render_type),
                        o.triple().map(|t| props.props[t].mfd_id).unwrap_or(-1),
                        o.x,
                        o.y,
                        o.z,
                        o.heading,
                        o.pitch,
                        o.bank,
                        o.frame
                    );
                }
            }
            for ((class, render), n) in by_class {
                println!("  {class:<11} {render:<10} {n}");
            }
            if let Some(out) = art {
                std::fs::create_dir_all(&out)?;
                let palette = ss1::Palette::load_game(&data)?;
                let objart = ResFile::open(data.join("objart.res"))?.load(1350)?;
                let mut done = std::collections::BTreeSet::new();
                for o in &objs {
                    let Some(t) = o.triple() else { continue };
                    if !done.insert(t) {
                        continue;
                    }
                    let Some(item) = props.world_art(t, 0).and_then(|i| objart.item(i)) else {
                        continue;
                    };
                    if let Ok(image) = ss1::Image::parse(item)
                        && image.width > 0
                    {
                        image::save_buffer(
                            out.join(format!("{t:03}_{}_{}_{}.png", o.class, o.subclass, o.kind)),
                            &image.to_rgba(&palette),
                            image.width as u32,
                            image.height as u32,
                            image::ColorType::Rgba8,
                        )?;
                    }
                }
                println!("wrote {} types to {}", done.len(), out.display());
            }
        }
        Command::Models => {
            let res = ss1::ResFile::open(data.join("obj3d.res"))?;
            for entry in res.entries() {
                let r = res.load(entry.id)?;
                let code = r.item(0).unwrap_or_default();
                let model = ss1::model::parse(code);
                let fills: std::collections::BTreeSet<String> = model
                    .polys
                    .iter()
                    .map(|p| format!("{:?}", p.fill))
                    .collect();
                let bounds = model.bounds().map(|(lo, hi)| {
                    format!(
                        "[{:.2} {:.2} {:.2}]..[{:.2} {:.2} {:.2}]",
                        lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]
                    )
                });
                println!(
                    "model {:3}: {:3} polys {} {}",
                    entry.id - ss1::model::MODEL_BASE,
                    model.polys.len(),
                    bounds.unwrap_or_default(),
                    fills.into_iter().collect::<Vec<_>>().join(",")
                );
            }
        }
        Command::Check => {
            let mut files: Vec<_> = std::fs::read_dir(&data)?
                .flatten()
                .map(|e| e.path())
                .filter(|p| {
                    p.extension().is_some_and(|x| x.eq_ignore_ascii_case("res"))
                        || p.file_name()
                            .is_some_and(|n| n.eq_ignore_ascii_case("archive.dat"))
                })
                .collect();
            files.sort();
            let mut failures = 0;
            for path in files {
                let name = path.file_name().unwrap().to_string_lossy().into_owned();
                let res = match ResFile::open(&path) {
                    Ok(res) => res,
                    Err(err) => {
                        println!("{name}: {err}");
                        failures += 1;
                        continue;
                    }
                };
                let bad: Vec<_> = res
                    .entries()
                    .iter()
                    .filter_map(|e| res.load(e.id).err().map(|err| format!("{}: {err}", e.id)))
                    .collect();
                println!(
                    "{name}: {} resources, {} failed",
                    res.entries().len(),
                    bad.len()
                );
                for line in bad.iter().take(5) {
                    println!("    {line}");
                }
                failures += bad.len();
            }
            if failures > 0 {
                return Err(anyhow!("{failures} failures"));
            }
        }
    }
    Ok(())
}

/// Lay images out in rows, up to 1024 pixels wide, on a dark backdrop.
fn write_sheet(
    images: &[ss1::Image],
    palette: &ss1::Palette,
    path: &std::path::Path,
) -> Result<()> {
    const WIDTH: usize = 1024;
    let (mut x, mut y, mut row_h) = (0usize, 0usize, 0usize);
    let mut spots = Vec::new();
    for image in images {
        let w = image.width.min(WIDTH);
        if x + w > WIDTH {
            x = 0;
            y += row_h + 2;
            row_h = 0;
        }
        spots.push((x, y));
        x += w + 2;
        row_h = row_h.max(image.height);
    }
    let height = y + row_h;
    let mut sheet =
        image::RgbaImage::from_pixel(WIDTH as u32, height as u32, image::Rgba([24, 24, 32, 255]));
    for (image, (x, y)) in images.iter().zip(spots) {
        let rgba = image.to_rgba(palette);
        for py in 0..image.height {
            for px in 0..image.width.min(WIDTH) {
                let i = (py * image.width + px) * 4;
                if rgba[i + 3] > 0 {
                    sheet.put_pixel(
                        (x + px) as u32,
                        (y + py) as u32,
                        image::Rgba([rgba[i], rgba[i + 1], rgba[i + 2], 255]),
                    );
                }
            }
        }
    }
    sheet.save(path)?;
    Ok(())
}

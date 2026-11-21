//! 8-bit images (`RTYPE_IMAGE` items) and palettes.
//!
//! An image item is a `FrameDesc` (shockmac `Libraries/H/2DRES.H`) followed
//! by its pixels:
//!
//! - `grs_bitmap` (16 bytes): bits pointer (unused on disk), type, align,
//!   flags, width, height, row bytes, log2 width, log2 height;
//! - an anchor/update rectangle (4 x `i16`);
//! - `pallOff`, an offset to a private palette (0 when none).
//!
//! Types used by the data are `BMT_FLAT8` (raw rows) and `BMT_RSD8` (the
//! run/skip/dump encoding of `RSD/RSDUnpack.c`). Colour 0 is transparent in
//! images flagged `BMF_TRANS`.

use crate::res::ResError;

pub const BMT_FLAT8: u8 = 2;
pub const BMT_RSD8: u8 = 4;
pub const BMF_TRANS: u16 = 1;
const FRAME_HEADER_LEN: usize = 28;

/// 256 colours, 8 bits per channel.
#[derive(Clone)]
pub struct Palette(pub [[u8; 3]; 256]);

impl Palette {
    pub fn from_bytes(bytes: &[u8]) -> Option<Palette> {
        let bytes = bytes.get(..768)?;
        let mut colours = [[0u8; 3]; 256];
        for (i, colour) in colours.iter_mut().enumerate() {
            colour.copy_from_slice(&bytes[i * 3..i * 3 + 3]);
        }
        Some(Palette(colours))
    }

    /// The main game palette (`gamepal.res` resource 700).
    pub fn load_game(data: &std::path::Path) -> Result<Palette, ResError> {
        let res = crate::ResFile::open(data.join("gamepal.res"))?;
        let item = res.load(700)?;
        Palette::from_bytes(item.item(0).unwrap_or(&[])).ok_or(ResError::Corrupt {
            id: 700,
            reason: "palette shorter than 768 bytes".into(),
        })
    }
}

/// A decoded 8-bit image.
#[derive(Debug, Clone)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub flags: u16,
    /// Anchor/update rectangle (left, top, right, bottom).
    pub anchor: [i16; 4],
    /// Palette indices, row-major, `width * height` of them.
    pub pixels: Vec<u8>,
}

impl Image {
    pub fn is_transparent(&self) -> bool {
        self.flags & BMF_TRANS != 0
    }

    /// Decode one image item.
    pub fn parse(item: &[u8]) -> Result<Image, String> {
        if item.len() < FRAME_HEADER_LEN {
            return Err(format!(
                "{} bytes is too short for a frame header",
                item.len()
            ));
        }
        let u16_at = |at: usize| u16::from_le_bytes([item[at], item[at + 1]]);
        let i16_at = |at: usize| i16::from_le_bytes([item[at], item[at + 1]]);
        let kind = item[4];
        let flags = u16_at(6);
        let width = i16_at(8).max(0) as usize;
        let height = i16_at(10).max(0) as usize;
        let row = (u16_at(12) as usize).max(width);
        let anchor = [i16_at(16), i16_at(18), i16_at(20), i16_at(22)];
        let bits = &item[FRAME_HEADER_LEN..];

        let pixels = match kind {
            BMT_FLAT8 => {
                let mut pixels = Vec::with_capacity(width * height);
                for y in 0..height {
                    let line = bits
                        .get(y * row..y * row + width)
                        .ok_or_else(|| format!("flat image row {y} runs past the data"))?;
                    pixels.extend_from_slice(line);
                }
                pixels
            }
            BMT_RSD8 => {
                let mut pixels = rsd8_unpack(bits)?;
                // The stream covers `row`-wide lines; keep `width` of each.
                pixels.resize(row * height, 0);
                if row != width {
                    pixels = (0..height)
                        .flat_map(|y| pixels[y * row..y * row + width].to_vec())
                        .collect();
                }
                pixels
            }
            other => return Err(format!("unsupported bitmap type {other}")),
        };

        Ok(Image {
            width,
            height,
            flags,
            anchor,
            pixels,
        })
    }

    /// RGBA pixels through `palette`; index 0 is clear in transparent images.
    pub fn to_rgba(&self, palette: &Palette) -> Vec<u8> {
        let transparent = self.is_transparent();
        let mut out = Vec::with_capacity(self.pixels.len() * 4);
        for &index in &self.pixels {
            let [r, g, b] = palette.0[index as usize];
            let alpha = if transparent && index == 0 { 0 } else { 255 };
            out.extend([r, g, b, alpha]);
        }
        out
    }
}

/// `gr_rsd8_unpack`: expand an RSD8 stream until its end code.
///
/// - `00 n v`: run of `n` copies of `v`
/// - `01..=7f`: dump that many literal bytes
/// - `81..=ff`: skip (`& 0x7f`) transparent pixels
/// - `80` then a little-endian `u16` code: `0` ends the stream, below
///   `0x8000` is a long skip, below `0xc000` a long dump of `code & 0x3fff`,
///   otherwise a long run of `code & 0x3fff` copies of the next byte.
pub fn rsd8_unpack(src: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let mut at = 0usize;
    let byte = |at: usize| {
        src.get(at)
            .copied()
            .ok_or("RSD stream ends early".to_string())
    };
    loop {
        let code = byte(at)?;
        at += 1;
        match code {
            0 => {
                let (count, value) = (byte(at)? as usize, byte(at + 1)?);
                at += 2;
                out.resize(out.len() + count, value);
            }
            1..=0x7f => {
                let count = code as usize;
                let run = src
                    .get(at..at + count)
                    .ok_or("RSD dump runs past the data")?;
                out.extend_from_slice(run);
                at += count;
            }
            0x81..=0xff => {
                out.resize(out.len() + (code & 0x7f) as usize, 0);
            }
            0x80 => {
                let long = u16::from_le_bytes([byte(at)?, byte(at + 1)?]);
                at += 2;
                let count = (long & 0x3fff) as usize;
                match long {
                    0 => break,
                    1..=0x7fff => out.resize(out.len() + long as usize, 0),
                    0x8000..=0xbfff => {
                        let run = src
                            .get(at..at + count)
                            .ok_or("RSD long dump runs past the data")?;
                        out.extend_from_slice(run);
                        at += count;
                    }
                    _ => {
                        let value = byte(at)?;
                        at += 1;
                        out.resize(out.len() + count, value);
                    }
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rsd_decodes_every_opcode() {
        let stream = [
            0x00, 3, 9, // run 3 x 9
            0x02, 5, 6,    // dump 5 6
            0x82, // skip 2
            0x80, 0x03, 0x00, // long skip 3
            0x80, 0x02, 0x80, 7, 8, // long dump 2
            0x80, 0x04, 0xc0, 4, // long run 4 x 4
            0x80, 0x00, 0x00, // end
        ];
        assert_eq!(
            rsd8_unpack(&stream).unwrap(),
            vec![9, 9, 9, 5, 6, 0, 0, 0, 0, 0, 7, 8, 4, 4, 4, 4]
        );
    }

    fn frame(kind: u8, flags: u16, w: i16, h: i16, row: u16, bits: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; 4];
        out.push(kind);
        out.push(0);
        out.extend(flags.to_le_bytes());
        out.extend(w.to_le_bytes());
        out.extend(h.to_le_bytes());
        out.extend(row.to_le_bytes());
        out.extend([0, 0]);
        out.extend([0u8; 8]);
        out.extend(0i32.to_le_bytes());
        out.extend(bits);
        out
    }

    #[test]
    fn flat_images_drop_row_padding() {
        let item = frame(BMT_FLAT8, 0, 2, 2, 3, &[1, 2, 99, 3, 4, 99]);
        let image = Image::parse(&item).unwrap();
        assert_eq!((image.width, image.height), (2, 2));
        assert_eq!(image.pixels, vec![1, 2, 3, 4]);
    }

    #[test]
    fn transparent_index_zero_is_clear() {
        let item = frame(BMT_RSD8, BMF_TRANS, 2, 1, 2, &[0x01, 5, 0x81, 0x80, 0, 0]);
        let image = Image::parse(&item).unwrap();
        let mut palette = Palette([[0; 3]; 256]);
        palette.0[5] = [10, 20, 30];
        assert_eq!(image.to_rgba(&palette), vec![10, 20, 30, 255, 0, 0, 0, 0]);
    }
}

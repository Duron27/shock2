//! LookingGlass resource files (`*.res`, and `archive.dat`), format
//! `LG ResFile v2`.
//!
//! Layout (shockmac `Libraries/RES/Source/RES.H`, PC byte order):
//!
//! - 128-byte header: 16-byte signature, 96-byte comment, 12 reserved
//!   bytes, then the `i32` offset of the directory.
//! - Directory: `u16` entry count, `i32` offset where resource data begins,
//!   then 10-byte entries: `u16` id, `u32` (size:24 | flags:8),
//!   `u32` (compressed size:24 | type:8).
//! - Resource data follows in directory order from the data offset, each
//!   resource padded to a 4-byte boundary.
//!
//! A *compound* resource starts with a ref table - `u16` count, then
//! `count + 1` `i32` offsets from the resource start - and holds several
//! sub-items. When a compound is also LZW-compressed, only the data after its
//! ref table is compressed.

use std::path::Path;

use crate::lzw;

pub const SIGNATURE: &[u8] = b"LG Res File v2\r\n";
const HEADER_LEN: usize = 128;
const DIR_ENTRY_LEN: usize = 10;

/// `RDF_LZW`: the resource is LZW-compressed on disk.
pub const RDF_LZW: u8 = 0x01;
/// `RDF_COMPOUND`: the resource is a table of sub-items.
pub const RDF_COMPOUND: u8 = 0x02;

#[derive(Debug, thiserror::Error)]
pub enum ResError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("not an LG resource file (signature {0:?})")]
    BadSignature(String),
    #[error("truncated resource file: {0}")]
    Truncated(&'static str),
    #[error("no resource with id {0} ({0:#x})")]
    Missing(u16),
    #[error("resource {id}: {reason}")]
    Corrupt { id: u16, reason: String },
}

/// `RTYPE_*` from `RESTYPES.H`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResType {
    Unknown,
    String,
    Image,
    Font,
    Anim,
    Palette,
    ShadeTable,
    Voc,
    Shape,
    Pict,
    B2Extern,
    B2Reloc,
    B2Code,
    B2Header,
    B2Reserved,
    Obj3d,
    Stencil,
    Movie,
    Rect,
    /// `RTYPE_APP + n`: a game-defined type (map data, object tables, ...).
    App(u8),
    Other(u8),
}

impl ResType {
    pub fn from_u8(value: u8) -> ResType {
        match value {
            0 => ResType::Unknown,
            1 => ResType::String,
            2 => ResType::Image,
            3 => ResType::Font,
            4 => ResType::Anim,
            5 => ResType::Palette,
            6 => ResType::ShadeTable,
            7 => ResType::Voc,
            8 => ResType::Shape,
            9 => ResType::Pict,
            10 => ResType::B2Extern,
            11 => ResType::B2Reloc,
            12 => ResType::B2Code,
            13 => ResType::B2Header,
            14 => ResType::B2Reserved,
            15 => ResType::Obj3d,
            16 => ResType::Stencil,
            17 => ResType::Movie,
            18 => ResType::Rect,
            48..=63 => ResType::App(value - 48),
            other => ResType::Other(other),
        }
    }

    pub fn name(self) -> String {
        match self {
            ResType::App(n) => format!("app{n}"),
            ResType::Other(n) => format!("type{n}"),
            other => format!("{other:?}").to_lowercase(),
        }
    }
}

/// One directory entry.
#[derive(Debug, Clone, Copy)]
pub struct ResEntry {
    pub id: u16,
    /// Uncompressed size, including a compound's ref table.
    pub size: u32,
    pub flags: u8,
    /// Size on disk.
    pub csize: u32,
    pub res_type: ResType,
    /// File offset of the data.
    pub offset: u32,
}

impl ResEntry {
    pub fn is_compressed(&self) -> bool {
        self.flags & RDF_LZW != 0
    }

    pub fn is_compound(&self) -> bool {
        self.flags & RDF_COMPOUND != 0
    }
}

/// A loaded, decompressed resource.
#[derive(Debug, Clone)]
pub struct Resource {
    pub entry: ResEntry,
    /// The whole resource, ref table included for a compound.
    pub data: Vec<u8>,
    /// For a compound: the byte range of each sub-item within `data`.
    pub refs: Option<Vec<std::ops::Range<usize>>>,
}

impl Resource {
    /// Number of items: a compound's sub-item count, or 1.
    pub fn len(&self) -> usize {
        self.refs.as_ref().map_or(1, Vec::len)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A compound's sub-item, or (index 0) a plain resource's whole data.
    pub fn item(&self, index: usize) -> Option<&[u8]> {
        match &self.refs {
            Some(refs) => refs.get(index).map(|range| &self.data[range.clone()]),
            None => (index == 0).then_some(self.data.as_slice()),
        }
    }
}

/// An open resource file: its directory plus the raw bytes.
pub struct ResFile {
    pub comment: String,
    entries: Vec<ResEntry>,
    bytes: Vec<u8>,
}

fn u16_at(bytes: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(bytes.get(at..at + 2)?.try_into().ok()?))
}

fn u32_at(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?))
}

impl ResFile {
    pub fn open(path: impl AsRef<Path>) -> Result<ResFile, ResError> {
        ResFile::parse(std::fs::read(path)?)
    }

    pub fn parse(bytes: Vec<u8>) -> Result<ResFile, ResError> {
        if bytes.len() < HEADER_LEN {
            return Err(ResError::Truncated("header"));
        }
        if &bytes[..SIGNATURE.len()] != SIGNATURE {
            return Err(ResError::BadSignature(
                String::from_utf8_lossy(&bytes[..16]).into_owned(),
            ));
        }
        let comment_bytes = &bytes[16..112];
        let comment_end = comment_bytes
            .iter()
            .position(|&b| b == 0 || b == 0x1a)
            .unwrap_or(comment_bytes.len());
        let comment = String::from_utf8_lossy(&comment_bytes[..comment_end])
            .trim()
            .to_owned();

        let dir = u32_at(&bytes, 124).ok_or(ResError::Truncated("header"))? as usize;
        let count = u16_at(&bytes, dir).ok_or(ResError::Truncated("directory"))? as usize;
        let mut offset = u32_at(&bytes, dir + 2).ok_or(ResError::Truncated("directory"))?;
        let mut entries = Vec::with_capacity(count);
        for i in 0..count {
            let at = dir + 6 + i * DIR_ENTRY_LEN;
            let (Some(id), Some(size_flags), Some(csize_type)) = (
                u16_at(&bytes, at),
                u32_at(&bytes, at + 2),
                u32_at(&bytes, at + 6),
            ) else {
                return Err(ResError::Truncated("directory entry"));
            };
            let entry = ResEntry {
                id,
                size: size_flags & 0x00FF_FFFF,
                flags: (size_flags >> 24) as u8,
                csize: csize_type & 0x00FF_FFFF,
                res_type: ResType::from_u8((csize_type >> 24) as u8),
                offset,
            };
            offset = (offset + entry.csize + 3) & !3;
            entries.push(entry);
        }
        Ok(ResFile {
            comment,
            entries,
            bytes,
        })
    }

    pub fn entries(&self) -> &[ResEntry] {
        &self.entries
    }

    pub fn entry(&self, id: u16) -> Option<&ResEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// Load and decompress resource `id`.
    pub fn load(&self, id: u16) -> Result<Resource, ResError> {
        let entry = *self.entry(id).ok_or(ResError::Missing(id))?;
        let corrupt = |reason: &str| ResError::Corrupt {
            id,
            reason: reason.to_owned(),
        };
        let start = entry.offset as usize;
        let raw = self
            .bytes
            .get(start..start + entry.csize as usize)
            .ok_or_else(|| corrupt("data runs past the end of the file"))?;

        // A compound's ref table is stored uncompressed even when the rest
        // of the resource is LZW.
        let table_len = if entry.is_compound() {
            let count = u16_at(raw, 0).ok_or_else(|| corrupt("missing ref table"))? as usize;
            2 + 4 * (count + 1)
        } else {
            0
        };
        if raw.len() < table_len {
            return Err(corrupt("ref table longer than the resource"));
        }

        let data = if entry.is_compressed() {
            let mut data = raw[..table_len].to_vec();
            let body_len = (entry.size as usize).saturating_sub(table_len);
            data.extend(lzw::expand(&raw[table_len..], Some(body_len)));
            if data.len() != entry.size as usize {
                return Err(corrupt(&format!(
                    "expanded to {} bytes, directory says {}",
                    data.len(),
                    entry.size
                )));
            }
            data
        } else {
            raw.to_vec()
        };

        let refs = if entry.is_compound() {
            let count = u16_at(&data, 0).unwrap() as usize;
            let mut refs = Vec::with_capacity(count);
            for i in 0..count {
                let from = u32_at(&data, 2 + 4 * i).unwrap() as usize;
                let to = u32_at(&data, 2 + 4 * (i + 1)).unwrap() as usize;
                if from > to || to > data.len() {
                    return Err(corrupt(&format!("bad ref {i}: {from}..{to}")));
                }
                refs.push(from..to);
            }
            Some(refs)
        } else {
            None
        };

        Ok(Resource { entry, data, refs })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a resource file from (id, type, flags, uncompressed data) items.
    fn build(items: &[(u16, u8, u8, Vec<u8>)]) -> Vec<u8> {
        let mut file = vec![0u8; HEADER_LEN];
        file[..SIGNATURE.len()].copy_from_slice(SIGNATURE);
        file[16..20].copy_from_slice(b"test");
        let data_offset = file.len() as u32;
        let mut dir = Vec::new();
        for (id, kind, flags, data) in items {
            let stored = if flags & RDF_LZW != 0 {
                let table = if flags & RDF_COMPOUND != 0 {
                    2 + 4 * (u16::from_le_bytes([data[0], data[1]]) as usize + 1)
                } else {
                    0
                };
                let mut stored = data[..table].to_vec();
                stored.extend(crate::lzw::tests::compress(&data[table..]));
                stored
            } else {
                data.clone()
            };
            dir.extend(id.to_le_bytes());
            dir.extend((data.len() as u32 | (u32::from(*flags) << 24)).to_le_bytes());
            dir.extend((stored.len() as u32 | (u32::from(*kind) << 24)).to_le_bytes());
            file.extend(&stored);
            while file.len() % 4 != 0 {
                file.push(0);
            }
        }
        let dir_offset = file.len() as u32;
        file.extend((items.len() as u16).to_le_bytes());
        file.extend(data_offset.to_le_bytes());
        file.extend(dir);
        file[124..128].copy_from_slice(&dir_offset.to_le_bytes());
        file
    }

    fn compound(items: &[&[u8]]) -> Vec<u8> {
        let mut out = (items.len() as u16).to_le_bytes().to_vec();
        let mut at = 2 + 4 * (items.len() as u32 + 1);
        for item in items {
            out.extend(at.to_le_bytes());
            at += item.len() as u32;
        }
        out.extend(at.to_le_bytes());
        for item in items {
            out.extend(*item);
        }
        out
    }

    #[test]
    fn reads_plain_compressed_and_compound_resources() {
        let packed = b"hello hello hello hello".repeat(8);
        let file = build(&[
            (7, 5, 0, b"abc".to_vec()),
            (8, 1, RDF_LZW, packed.clone()),
            (9, 1, RDF_COMPOUND, compound(&[b"one", b"", b"three"])),
            (10, 2, RDF_COMPOUND | RDF_LZW, compound(&[&packed, b"tail"])),
        ]);
        let res = ResFile::parse(file).unwrap();
        assert_eq!(res.comment, "test");
        assert_eq!(res.entries().len(), 4);

        let plain = res.load(7).unwrap();
        assert_eq!(plain.entry.res_type, ResType::Palette);
        assert_eq!(plain.item(0), Some(&b"abc"[..]));

        assert_eq!(res.load(8).unwrap().data, packed);

        let table = res.load(9).unwrap();
        assert_eq!(table.len(), 3);
        assert_eq!(table.item(1), Some(&b""[..]));
        assert_eq!(table.item(2), Some(&b"three"[..]));

        let both = res.load(10).unwrap();
        assert_eq!(both.item(0), Some(packed.as_slice()));
        assert_eq!(both.item(1), Some(&b"tail"[..]));

        assert!(matches!(res.load(99), Err(ResError::Missing(99))));
    }

    #[test]
    fn rejects_other_files() {
        assert!(matches!(
            ResFile::parse(vec![b'x'; 200]),
            Err(ResError::BadSignature(_))
        ));
    }
}

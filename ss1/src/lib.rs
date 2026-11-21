//! System Shock (1994) data readers.
//!
//! A from-scratch Rust reading of the original game's data, built on the
//! formats documented by Night Dive's GPL source release (`shockmac`). Nothing
//! here depends on the System Shock 2 crates.

pub mod image;
pub mod level;
pub mod lzw;
pub mod mesh;
pub mod model;
pub mod objects;
pub mod paths;
pub mod res;
pub mod scale;

pub use image::{Image, Palette};
pub use level::{Level, Tile, TileType};
pub use res::{ResEntry, ResError, ResFile, ResType, Resource};

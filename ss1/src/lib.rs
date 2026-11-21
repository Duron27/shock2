//! System Shock (1994) data readers.
//!
//! A from-scratch Rust reading of the original game's data, built on the
//! formats documented by Night Dive's GPL source release (`shockmac`). Nothing
//! here depends on the System Shock 2 crates.

pub mod lzw;
pub mod paths;
pub mod res;

pub use res::{ResEntry, ResError, ResFile, ResType, Resource};

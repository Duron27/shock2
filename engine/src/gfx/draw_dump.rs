//! One frame's draws as the Vulkan backend issues them, written to a file so
//! vulkan4xenon's `vkreplay` can issue them again on the console without the
//! game (to find what hangs the Xbox 360's GPU).
//!
//! `dump_draws.txt` beside the console's `.xex` names the frame (1 is the
//! first); its draws go to `draws.bin` beside it at that frame's end. The file
//! is a list of records, each a little-endian `u32` tag then its fields, all
//! `u32`/`f32` (byte blobs are a length then the bytes padded to 4):
//!
//! - `FRAM` width, height, colour format, depth format, samples: the frame's
//!   pass begins, cleared to black and depth 1.
//! - `CAPB` width, height, kept: the frame's pass pauses for a depth-only
//!   pass (format `D24_UNORM_S8_UINT` on the console), cleared unless kept.
//! - `CAPE`: the frame's pass resumes, loading what it held.
//! - `CLER` colour, depth, rgba, x, y, width, height: `vkCmdClearAttachments`.
//! - `PROG` id, vertex SPIR-V, fragment SPIR-V (each a word count then the
//!   words), the set layout's bindings (a count then binding, descriptor type,
//!   stage flags each): before a program's first draw.
//! - `DRAW` see [`Draw`].

use std::collections::HashSet;

pub const FRAME: u32 = u32::from_le_bytes(*b"FRAM");
pub const CAPTURE_BEGIN: u32 = u32::from_le_bytes(*b"CAPB");
pub const CAPTURE_END: u32 = u32::from_le_bytes(*b"CAPE");
pub const CLEAR: u32 = u32::from_le_bytes(*b"CLER");
pub const PROGRAM: u32 = u32::from_le_bytes(*b"PROG");
pub const DRAW: u32 = u32::from_le_bytes(*b"DRAW");

/// A draw as recorded, in the order [`DrawDump::draw`] writes it.
pub struct Draw<'a> {
    pub program: u32,
    /// 0 triangle list, 1 line list.
    pub topology: u32,
    /// Vertices (or indices) drawn; the vertex bytes start at the first.
    pub count: u32,
    pub stride: u32,
    /// Location, binding, `VkFormat`, offset; binding 1 is a zero-stride
    /// buffer of 16 zero bytes.
    pub attributes: &'a [[u32; 4]],
    /// `VkCullModeFlags`, `VkFrontFace`.
    pub cull: [u32; 2],
    pub depth_test: bool,
    pub depth_write: bool,
    pub depth_compare: u32,
    pub blend: bool,
    /// Source colour, destination colour, source alpha, destination alpha.
    pub blend_factors: [u32; 4],
    pub color_mask: u32,
    pub depth_bias: bool,
    /// Constant, slope.
    pub bias: [f32; 2],
    /// The pass's colour and depth `VkFormat` (colour 0 for none) and samples.
    pub formats: [u32; 3],
    /// x, y, width, height, min depth, max depth.
    pub viewport: [f32; 6],
    /// x, y, width, height.
    pub scissor: [i32; 4],
    /// The vertices from the draw's first (or the whole buffer for an
    /// indexed draw).
    pub vertices: &'a [u8],
    /// `u32` indices from the draw's first; empty for a plain draw.
    pub indices: &'a [u8],
    pub blocks: [&'a [u8]; 2],
    /// Binding, width, height, mips, layers, filter (0 nearest, 1 linear,
    /// 2 linear mipmapped), wrap.
    pub samplers: &'a [[u32; 7]],
}

#[derive(Default)]
pub struct DrawDump {
    target: Option<u32>,
    frame: u32,
    out: Option<Vec<u8>>,
    programs: HashSet<u32>,
}

impl DrawDump {
    /// The frame `dump_draws.txt` names, on the console.
    pub fn from_file() -> DrawDump {
        let target = if cfg!(target_os = "xbox360") {
            std::fs::read_to_string(r"game:\dump_draws.txt")
                .ok()
                .and_then(|t| t.trim().parse().ok())
        } else {
            None
        };
        if let Some(frame) = target {
            tracing::info!("draw dump: frame {frame} to draws.bin");
        }
        DrawDump {
            target,
            ..DrawDump::default()
        }
    }

    /// Whether the frame is still to come (or being recorded): buffers'
    /// bytes are kept until then.
    pub fn pending(&self) -> bool {
        self.target.is_some_and(|t| self.frame < t) || self.out.is_some()
    }

    pub fn active(&self) -> bool {
        self.out.is_some()
    }

    pub fn begin_frame(&mut self, width: u32, height: u32, formats: [u32; 3]) {
        self.frame += 1;
        if self.target == Some(self.frame) {
            self.out = Some(Vec::new());
            self.words(&[FRAME, width, height, formats[0], formats[1], formats[2]]);
        }
    }

    pub fn end_frame(&mut self) {
        let Some(out) = self.out.take() else { return };
        match std::fs::write(r"game:\draws.bin", &out) {
            Ok(()) => tracing::info!("draw dump: {} bytes to draws.bin", out.len()),
            Err(e) => tracing::warn!("draw dump: draws.bin: {e}"),
        }
    }

    pub fn capture_begin(&mut self, width: u32, height: u32, kept: bool) {
        self.words(&[CAPTURE_BEGIN, width, height, kept as u32]);
    }

    pub fn capture_end(&mut self) {
        self.words(&[CAPTURE_END]);
    }

    pub fn clear(&mut self, color: bool, depth: bool, rgba: [f32; 4], rect: [i32; 4]) {
        self.words(&[CLEAR, color as u32, depth as u32]);
        self.floats(&rgba);
        self.words(&rect.map(|v| v as u32));
    }

    /// Writes the program first if this frame hasn't yet.
    pub fn program(&mut self, id: u32, spirv: &[Vec<u32>; 2], bindings: &[[u32; 3]]) {
        if !self.active() || !self.programs.insert(id) {
            return;
        }
        self.words(&[PROGRAM, id]);
        for words in spirv {
            self.words(&[words.len() as u32]);
            self.words(words);
        }
        self.words(&[bindings.len() as u32]);
        for b in bindings {
            self.words(b);
        }
    }

    pub fn draw(&mut self, d: &Draw) {
        if !self.active() {
            return;
        }
        self.words(&[DRAW, d.program, d.topology, d.count, d.stride]);
        self.words(&[d.attributes.len() as u32]);
        for a in d.attributes {
            self.words(a);
        }
        self.words(&[
            d.cull[0],
            d.cull[1],
            d.depth_test as u32,
            d.depth_write as u32,
            d.depth_compare,
            d.blend as u32,
        ]);
        self.words(&d.blend_factors);
        self.words(&[d.color_mask, d.depth_bias as u32]);
        self.floats(&d.bias);
        self.words(&d.formats);
        self.floats(&d.viewport);
        self.words(&d.scissor.map(|v| v as u32));
        self.blob(d.vertices);
        self.blob(d.indices);
        self.blob(d.blocks[0]);
        self.blob(d.blocks[1]);
        self.words(&[d.samplers.len() as u32]);
        for s in d.samplers {
            self.words(s);
        }
    }

    fn words(&mut self, words: &[u32]) {
        if let Some(out) = self.out.as_mut() {
            for w in words {
                out.extend_from_slice(&w.to_le_bytes());
            }
        }
    }

    fn floats(&mut self, floats: &[f32]) {
        let words: Vec<u32> = floats.iter().map(|f| f.to_bits()).collect();
        self.words(&words);
    }

    fn blob(&mut self, bytes: &[u8]) {
        self.words(&[bytes.len() as u32]);
        if let Some(out) = self.out.as_mut() {
            out.extend_from_slice(bytes);
            out.resize(out.len().next_multiple_of(4), 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_target_frame_is_written() {
        let mut dump = DrawDump {
            target: Some(2),
            ..DrawDump::default()
        };
        dump.begin_frame(1280, 720, [44, 129, 1]);
        dump.capture_end();
        assert!(!dump.active());
        dump.begin_frame(1280, 720, [44, 129, 1]);
        assert!(dump.active());
        dump.capture_end();
        dump.clear(true, false, [0.0; 4], [0, 0, 8, 8]);
        let out = dump.out.as_ref().unwrap();
        assert_eq!(&out[..4], b"FRAM");
        assert_eq!(&out[24..28], b"CAPE");
        assert_eq!(&out[28..32], b"CLER");
        assert_eq!(out.len(), 24 + 4 + 4 + 8 + 16 + 16);
    }

    #[test]
    fn blobs_pad_to_words() {
        let mut dump = DrawDump {
            out: Some(Vec::new()),
            ..DrawDump::default()
        };
        dump.blob(&[1, 2, 3]);
        assert_eq!(dump.out.unwrap(), [3, 0, 0, 0, 1, 2, 3, 0]);
    }
}

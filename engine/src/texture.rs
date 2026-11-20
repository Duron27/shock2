use crate::EngineRenderContext;
use crate::texture_format;
use crate::texture_format::RawTextureData;
use crate::texture_format::TextureFormat;

use std::rc::Rc;
use std::time::Duration;

pub struct Texture {
    gpu: crate::gfx::Texture2D,
}

// Will this cause problems for multi-threading??
unsafe impl Send for Texture {}
unsafe impl Sync for Texture {}

pub trait TextureTrait {
    fn bind0(&self, render_context: &EngineRenderContext);
    fn bind1(&self, render_context: &EngineRenderContext);
    fn bind_to(&self, render_context: &EngineRenderContext, unit: u32);
}

impl TextureTrait for Texture {
    fn bind0(&self, _render_context: &EngineRenderContext) {
        bind0(self);
    }
    fn bind1(&self, _render_context: &EngineRenderContext) {
        bind1(self);
    }
    fn bind_to(&self, _render_context: &EngineRenderContext, unit: u32) {
        bind_to(self, unit);
    }
}

/// Texture sequences use the simulation render clock, including deterministic
/// debug stepping and pauses. Ping-pong turns around without holding endpoints.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlaybackMode {
    #[default]
    Loop,
    PingPong,
}

impl PlaybackMode {
    pub fn frame_index(self, time: f32, frame_time: f32, count: usize) -> usize {
        if count <= 1 || !time.is_finite() || !frame_time.is_finite() || frame_time <= 0.0 {
            return 0;
        }
        let period = match self {
            Self::Loop => count,
            Self::PingPong => 2 * (count - 1),
        };
        let phase = ((time.max(0.0) as f64 / frame_time as f64).floor() % period as f64) as usize;
        if phase < count { phase } else { period - phase }
    }
}

pub struct AnimatedTexture {
    textures: Vec<Rc<Texture>>,
    time_per_frame: f32,
    playback: PlaybackMode,
}

impl AnimatedTexture {
    pub fn new(textures: Vec<Rc<Texture>>, duration_per_frame: Duration) -> AnimatedTexture {
        Self::with_playback(textures, duration_per_frame, PlaybackMode::Loop)
    }

    pub fn with_playback(
        textures: Vec<Rc<Texture>>,
        duration_per_frame: Duration,
        playback: PlaybackMode,
    ) -> Self {
        assert!(
            !textures.is_empty(),
            "animated textures require at least one frame"
        );
        Self {
            textures,
            time_per_frame: duration_per_frame.as_secs_f32(),
            playback,
        }
    }

    fn current_frame(&self, context: &EngineRenderContext) -> &Texture {
        &self.textures[self.playback.frame_index(
            context.time,
            self.time_per_frame,
            self.textures.len(),
        )]
    }
}

impl TextureTrait for AnimatedTexture {
    fn bind0(&self, render_context: &EngineRenderContext) {
        bind0(self.current_frame(render_context));
    }
    fn bind1(&self, render_context: &EngineRenderContext) {
        bind1(self.current_frame(render_context));
    }
    fn bind_to(&self, render_context: &EngineRenderContext, unit: u32) {
        bind_to(self.current_frame(render_context), unit);
    }
}

#[cfg(test)]
mod animation_tests {
    use super::PlaybackMode;
    #[test]
    fn ping_pong_visits_every_frame_without_duplicate_endpoints() {
        let frames: Vec<_> = (0..15)
            .map(|tick| PlaybackMode::PingPong.frame_index(tick as f32 * 0.25, 0.25, 8))
            .collect();
        assert_eq!(frames, [0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0]);
        assert_eq!(PlaybackMode::PingPong.frame_index(0.249, 0.25, 8), 0);
        assert_eq!(PlaybackMode::Loop.frame_index(2.0, 0.25, 8), 0);
    }
    #[test]
    fn degenerate_and_large_clocks_are_defined() {
        for count in [0, 1] {
            assert_eq!(PlaybackMode::PingPong.frame_index(7.0, 0.25, count), 0);
        }
        assert_eq!(PlaybackMode::PingPong.frame_index(-1.0, 0.25, 8), 0);
        assert_eq!(PlaybackMode::PingPong.frame_index(f32::NAN, 0.25, 8), 0);
        assert_eq!(PlaybackMode::PingPong.frame_index(1.0, 0.0, 8), 0);
        assert!(PlaybackMode::PingPong.frame_index(f32::MAX, 0.25, 8) < 8);
    }
    #[test]
    fn half_speed_clock_shows_half_the_frames() {
        // 120 ticks of 1/60 s; the slowed clock is world time at scale 0.5.
        let frames = |scale: f32| -> Vec<usize> {
            (0..120)
                .map(|tick| PlaybackMode::Loop.frame_index(tick as f32 / 60.0 * scale, 0.25, 8))
                .collect()
        };
        let (full, half) = (frames(1.0), frames(0.5));
        assert_eq!(full[119], 7);
        assert_eq!(half[119], 3);
        // Each frame holds twice as many ticks.
        assert_eq!(full.iter().filter(|&&f| f == 1).count(), 15);
        assert_eq!(half.iter().filter(|&&f| f == 1).count(), 30);
    }
}

impl Texture {
    pub fn width(&self) -> u32 {
        self.gpu.width()
    }
    pub fn height(&self) -> u32 {
        self.gpu.height()
    }

    /// Overwrite a region with tightly packed RGB8 pixels.
    pub fn update_rgb_region(&self, x: u32, y: u32, width: u32, height: u32, pixels: &[u8]) {
        self.gpu.update_rgb_region(x, y, width, height, pixels);
    }

    pub fn gpu(&self) -> &crate::gfx::Texture2D {
        &self.gpu
    }
}

/// A six-face cube map, sampled by direction.
pub struct CubeTexture {
    gpu: crate::gfx::TextureCube,
}

impl std::fmt::Debug for CubeTexture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CubeTexture")
    }
}

impl CubeTexture {
    /// Upload RGBA8 faces with a full mip chain: reflections sample blurred
    /// mips, not just the sharp top level.
    pub fn new(cube: &crate::dds::CubeFaces) -> CubeTexture {
        let faces: Vec<u8> = cube.faces.iter().flat_map(|f| f.iter().copied()).collect();
        CubeTexture {
            gpu: crate::gfx::TextureCube::new(cube.size, &faces),
        }
    }

    pub fn bind_to(&self, unit: u32) {
        self.gpu.bind(unit);
    }
}

pub fn bind0(texture: &Texture) {
    texture.gpu.bind(0);
}
pub fn bind1(texture: &Texture) {
    texture.gpu.bind(1);
}
pub fn bind_to(texture: &Texture, unit: u32) {
    texture.gpu.bind(unit);
}
pub fn bind(texture: &Texture) {
    bind0(texture);
}

/// How a texture is sampled when it does not map 1:1 to pixels.
#[derive(Hash, Clone, Copy, PartialEq, Eq, Debug)]
pub enum TextureFilter {
    /// Blend neighbouring texels. Right for world and scaled-up art.
    Linear,
    /// Snap to the nearest texel. Right for colour-keyed HUD bitmaps drawn
    /// minified: blending across the key boundary turns it into a visible
    /// fringe that no colour-key test can then recognize.
    Nearest,
    /// Linear, with mipmaps for minification. Right for fine line art drawn
    /// smaller than its texels and at a viewpoint-dependent scale (a hologram
    /// grid on a VR panel): plain `Linear` point-samples the lines, so their
    /// brightness swims with sub-pixel phase as the panel moves.
    LinearMipmap,
}

#[derive(Hash)]
pub struct TextureOptions {
    pub wrap: bool,
    /// Treat palette index 0 as transparent when decoding paletted formats
    /// (PCX). Dark bitmap sprites (particles) are keyed this way; wall/UI
    /// textures are not, so this is opt-in.
    pub transparent_index_0: bool,
    /// Turn opaque line art into a translucent tinted overlay: each texel's
    /// alpha becomes its luminance and its colour becomes this tint. Black
    /// therefore drops out entirely, which is what makes a grid bitmap read as
    /// a hologram over the world instead of a black panel.
    pub luminance_alpha_tint: Option<[u8; 3]>,
    /// Luminance mapped to transparent/opaque when tinting. Defaults preserve
    /// the source contrast; a narrower range removes a dim icon background.
    pub luminance_alpha_range: [u8; 2],
    pub filter: TextureFilter,
}

impl Default for TextureOptions {
    fn default() -> TextureOptions {
        TextureOptions {
            wrap: true,
            transparent_index_0: false,
            luminance_alpha_tint: None,
            luminance_alpha_range: [0, 255],
            // World and model textures are mostly seen smaller than their
            // texels; without mips they shimmer and crawl as the view moves.
            filter: TextureFilter::LinearMipmap,
        }
    }
}

pub fn init_from_memory(raw_texture_data: RawTextureData) -> Texture {
    init_from_memory2(raw_texture_data, &TextureOptions::default())
}

pub fn init_from_memory2(raw_texture_data: RawTextureData, options: &TextureOptions) -> Texture {
    let (format, pixel_size_in_bytes) = match raw_texture_data.format {
        texture_format::PixelFormat::RGB => (crate::gfx::PixelFormat::Rgb8, 3),
        texture_format::PixelFormat::RGBA => (crate::gfx::PixelFormat::Rgba8, 4),
    };
    assert!(
        raw_texture_data.bytes.len()
            == (raw_texture_data.width * raw_texture_data.height * pixel_size_in_bytes) as usize,
        "Texture data size does not match width and height - width: {} height: {} pixel_size_in_bytes: {} actual_bytes: {}",
        raw_texture_data.width,
        raw_texture_data.height,
        pixel_size_in_bytes,
        raw_texture_data.bytes.len()
    );
    let filter = match options.filter {
        TextureFilter::Linear => crate::gfx::Filter::Linear,
        TextureFilter::Nearest => crate::gfx::Filter::Nearest,
        TextureFilter::LinearMipmap => crate::gfx::Filter::LinearMipmap,
    };
    Texture {
        gpu: crate::gfx::Texture2D::new(
            raw_texture_data.width,
            raw_texture_data.height,
            format,
            &raw_texture_data.bytes,
            crate::gfx::Sampling {
                filter,
                wrap: options.wrap,
            },
        ),
    }
}
pub fn init<T: crate::texture_format::TextureFormat>(
    buffer: &std::vec::Vec<u8>,
    format: T,
) -> Texture {
    // TODO:
    //let img = image::load_from_memory_with_format(&buffer, format).expect("Failed to load texture");

    let raw_texture_data = TextureFormat::load(&format, buffer);
    init_from_memory(raw_texture_data)
}

pub fn init2(buffer: &[u8], format: &dyn TextureFormat) -> Texture {
    // TODO:
    //let img = image::load_from_memory_with_format(&buffer, format).expect("Failed to load texture");

    let raw_texture_data = format.load(buffer);
    init_from_memory(raw_texture_data)
}

thread_local! {
    /// Uploaded once per thread, on first use. Needs a live GL context, which
    /// rules out a `const`/`static` (the `shared_builtin_font` pattern).
    /// `ManuallyDrop`: never deleted at exit, after the GL context is gone.
    static WHITE_PIXEL: std::mem::ManuallyDrop<std::cell::RefCell<Option<Rc<Texture>>>> =
        const { std::mem::ManuallyDrop::new(std::cell::RefCell::new(None)) };
}

/// A 1x1 opaque white texture, uploaded on first use.
///
/// The screen-space and panel materials all multiply a tint by a sampled
/// texel, so a solid-colour quad is just that tint over this texture - no
/// second shader, and one shared upload rather than a 1x1 bitmap per caller.
///
/// Requires a current GL context.
pub fn shared_white_pixel() -> Rc<Texture> {
    WHITE_PIXEL.with(|cell| {
        let mut cell = cell.borrow_mut();
        cell.get_or_insert_with(|| {
            Rc::new(init_from_memory(RawTextureData {
                bytes: vec![255, 255, 255, 255],
                width: 1,
                height: 1,
                format: texture_format::PixelFormat::RGBA,
            }))
        })
        .clone()
    })
}

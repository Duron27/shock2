//! Skipping a cutscene ([`SkipPress`]), and the radial hold-progress art
//! ([`RingArt`]) the Menu button's long press draws. Both are pure, so they are
//! testable without a running game.

use std::rc::Rc;

use engine::{
    scene::{SceneObject, basic_material},
    texture::{TextureOptions, TextureTrait, init_from_memory2},
    texture_format::{PixelFormat, RawTextureData},
};

/// Instant skip for the cutscene player: Space (the jump key) or Escape ends
/// the video at once, with nothing drawn on screen.
#[derive(Default)]
pub struct SkipPress {
    /// Space has been seen released since the cutscene began. A cutscene can
    /// start while Space is still down (a jump into a trigger), and that held
    /// key must not dismiss the movie it just started.
    armed: bool,
}

impl SkipPress {
    /// Whether this frame skips: a fresh Space press, or an Escape request.
    pub fn update(&mut self, space_down: bool, escape_pressed: bool) -> bool {
        if !space_down {
            self.armed = true;
            return escape_pressed;
        }
        let pressed = self.armed;
        self.armed = false;
        pressed || escape_pressed
    }
}

/// The ring's art, rasterized on demand and cached by fill step, plus the quad
/// that carries it. Shared by every hold in the game, so the readout cannot
/// look like two different things - the caller supplies only the transform.
#[derive(Default)]
pub struct RingArt {
    cached: Option<(u32, Rc<dyn TextureTrait>)>,
}

impl RingArt {
    /// A UNIT quad (1x1, centered, facing +Z) showing the ring filled to
    /// `progress` at `alpha` opacity. The caller places and scales it.
    pub fn quad(&mut self, progress: f32, alpha: f32) -> SceneObject {
        // The art changes only when the fill crosses a step, so a held button
        // rasterizes and uploads a few dozen textures rather than one a frame.
        let step = ring_step(progress);
        let texture = match &self.cached {
            Some((cached_step, texture)) if *cached_step == step => texture.clone(),
            _ => {
                let texture: Rc<dyn TextureTrait> = Rc::new(init_from_memory2(
                    ring_texture_for_step(step),
                    &TextureOptions {
                        wrap: false,
                        ..Default::default()
                    },
                ));
                self.cached = Some((step, texture.clone()));
                texture
            }
        };

        // Held just under fully opaque: the material only joins the blended
        // pass when it is transparent at all, and the ring is nothing but
        // per-pixel alpha.
        let transparency = (1.0 - alpha).max(0.02);
        SceneObject::new(
            basic_material::create_with_fixed_ambient(texture, 1.0, transparency),
            Box::new(engine::scene::quad::create()),
        )
    }
}

/// Edge length of the generated ring texture. Small: the ring draws a few dozen
/// pixels across, so anything larger is only oversampling.
const RING_TEXTURE_SIZE: usize = 64;

/// Steps the fill is drawn in. The art changes only when the arc crosses one,
/// so a hold rebuilds and re-uploads the texture this many times rather than
/// once per frame.
const RING_STEPS: u32 = 48;

/// Which step `progress` falls in - the cache key for [`ring_texture`].
fn ring_step(progress: f32) -> u32 {
    (progress.clamp(0.0, 1.0) * RING_STEPS as f32).round() as u32
}

/// The ring art for a step from [`ring_step`].
fn ring_texture_for_step(step: u32) -> RawTextureData {
    ring_texture(step as f32 / RING_STEPS as f32)
}

/// Inner and outer radius of the ring, in units of half the texture edge.
const RING_INNER: f32 = 0.66;
const RING_OUTER: f32 = 0.86;
/// Alpha of the part of the ring the hold has not reached yet - present enough
/// to read as a track for the fill to run around.
const TRACK_ALPHA: u8 = 90;

/// The radial progress ring for a hold `progress` in 0.0..=1.0: a white arc
/// filling clockwise from twelve o'clock over a faint full-circle track,
/// transparent everywhere else.
fn ring_texture(progress: f32) -> RawTextureData {
    let progress = progress.clamp(0.0, 1.0);
    let size = RING_TEXTURE_SIZE;
    let half = size as f32 / 2.0;
    let mut bytes = Vec::with_capacity(size * size * 4);

    for y in 0..size {
        for x in 0..size {
            // Pixel centers, in -1..1 with +y up (the quad's v axis).
            let dx = (x as f32 + 0.5 - half) / half;
            let dy = -(y as f32 + 0.5 - half) / half;
            let radius = (dx * dx + dy * dy).sqrt();

            let alpha = if !(RING_INNER..=RING_OUTER).contains(&radius) {
                0
            } else if angle_fraction(dx, dy) <= progress {
                255
            } else {
                TRACK_ALPHA
            };

            bytes.extend_from_slice(&[255, 255, 255, alpha]);
        }
    }

    RawTextureData {
        width: size as u32,
        height: size as u32,
        bytes,
        format: PixelFormat::RGBA,
    }
}

/// Where a point sits around the ring: 0.0 at twelve o'clock, rising clockwise
/// to 1.0 back at the top.
fn angle_fraction(dx: f32, dy: f32) -> f32 {
    let angle = dx.atan2(dy); // -PI..PI, zero pointing up
    let turns = angle / std::f32::consts::TAU;
    if turns < 0.0 { turns + 1.0 } else { turns }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_space_press_skips_at_once() {
        let mut skip = SkipPress::default();
        assert!(!skip.update(false, false));
        assert!(skip.update(true, false));
    }

    #[test]
    fn space_held_from_before_the_video_does_not_skip() {
        let mut skip = SkipPress::default();
        assert!(!skip.update(true, false));
        assert!(!skip.update(true, false));
        assert!(!skip.update(false, false));
        assert!(skip.update(true, false), "a new press after release skips");
    }

    #[test]
    fn escape_skips_whatever_space_is_doing() {
        let mut held = SkipPress::default();
        assert!(held.update(true, true));
        let mut idle = SkipPress::default();
        assert!(idle.update(false, true));
    }

    fn sample(texture: &RawTextureData, turns: f32) -> u8 {
        let size = texture.width as f32;
        let radius = (RING_INNER + RING_OUTER) / 2.0;
        let angle = turns * std::f32::consts::TAU;
        let x = (size / 2.0 + angle.sin() * radius * size / 2.0) as u32;
        let y = (size / 2.0 - angle.cos() * radius * size / 2.0) as u32;
        texture.bytes[((y * texture.width + x) * 4 + 3) as usize]
    }

    #[test]
    fn the_ring_fills_clockwise_from_the_top() {
        let quarter = ring_texture(0.25);

        assert_eq!(sample(&quarter, 0.1), 255, "just past the top is filled");
        assert_eq!(sample(&quarter, 0.2), 255, "the first quarter is filled");
        assert_eq!(
            sample(&quarter, 0.4),
            TRACK_ALPHA,
            "past the fill only the track shows"
        );
        assert_eq!(sample(&quarter, 0.9), TRACK_ALPHA);

        let full = ring_texture(1.0);
        for turns in [0.1, 0.35, 0.6, 0.85] {
            assert_eq!(sample(&full, turns), 255, "a complete hold fills the ring");
        }
    }

    #[test]
    fn the_ring_is_transparent_off_the_band() {
        let texture = ring_texture(1.0);
        let center = ((texture.height / 2) * texture.width + texture.width / 2) * 4 + 3;
        assert_eq!(texture.bytes[center as usize], 0, "hollow center");
        assert_eq!(texture.bytes[3], 0, "transparent corner");
    }
}

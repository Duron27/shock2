//! The original interface frame, sharpened, over the 3D view.
//!
//! SS1 draws its 3D view inside a 320x200 frame (`gamescr.res` 617): status
//! bars along the top, MFDs and the inventory along the bottom. The frame is
//! upscaled 4x with Scale2x (smooth diagonals, crisp outlines) and drawn as
//! one screen-space quad; the 3D projection is shifted and squeezed so its
//! centre and field of view sit in the frame's window.

use std::path::Path;
use std::rc::Rc;

use anyhow::{Context, Result, anyhow};
use cgmath::{Matrix4, vec2};
use engine::scene::{RenderLayer, SceneObject};
use engine::texture::{TextureFilter, TextureOptions, TextureTrait};
use engine::texture_format::{PixelFormat, RawTextureData};
use ss1::{Image, Palette, ResFile};

const FRAME_ID: u16 = 617;
const CANVAS: (f32, f32) = (320.0, 200.0);

pub struct Hud {
    frame: Rc<dyn TextureTrait>,
    /// The frame's transparent view window, in canvas pixels (x, y, w, h).
    window: (f32, f32, f32, f32),
}

/// Placement of the 320x200 canvas on a screen: letterboxed to keep its
/// aspect, returned as (x, y, scale).
fn canvas_on(screen: (f32, f32)) -> (f32, f32, f32) {
    let scale = (screen.0 / CANVAS.0).min(screen.1 / CANVAS.1);
    let x = (screen.0 - CANVAS.0 * scale) / 2.0;
    let y = (screen.1 - CANVAS.1 * scale) / 2.0;
    (x, y, scale)
}

/// The transparent region connected to the middle of the view, as a box,
/// plus which pixels belong to it.
fn view_window(image: &Image) -> Option<((f32, f32, f32, f32), Vec<bool>)> {
    let (w, h) = (image.width, image.height);
    let start = (w / 2, h / 3);
    if image.pixels[start.1 * w + start.0] != 0 {
        return None;
    }
    let mut seen = vec![false; w * h];
    let mut stack = vec![start];
    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0, 0);
    while let Some((x, y)) = stack.pop() {
        let i = y * w + x;
        if seen[i] || image.pixels[i] != 0 {
            continue;
        }
        seen[i] = true;
        (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
        if x > 0 {
            stack.push((x - 1, y));
        }
        if x + 1 < w {
            stack.push((x + 1, y));
        }
        if y > 0 {
            stack.push((x, y - 1));
        }
        if y + 1 < h {
            stack.push((x, y + 1));
        }
    }
    Some((
        (
            x0 as f32,
            y0 as f32,
            (x1 - x0 + 1) as f32,
            (y1 - y0 + 1) as f32,
        ),
        seen,
    ))
}

impl Hud {
    pub fn load(data: &Path) -> Result<Hud> {
        let palette = Palette::load_game(data)?;
        let res = ResFile::open(data.join("gamescr.res")).context("gamescr.res")?;
        let item = res.load(FRAME_ID)?;
        let mut frame = Image::parse(item.item(0).unwrap_or_default())
            .map_err(|e| anyhow!("frame image: {e}"))?;
        let (window, in_window) =
            view_window(&frame).ok_or_else(|| anyhow!("frame has no view window"))?;
        // Only the view window is see-through: the panels' own colour-0
        // holes (MFD dither, gaps) are black, as the game draws them.
        let black = (1..=255u8)
            .min_by_key(|&i| palette.0[i as usize].iter().map(|&c| c as u32).sum::<u32>())
            .unwrap();
        for (pixel, inside) in frame.pixels.iter_mut().zip(&in_window) {
            if *pixel == 0 && !inside {
                *pixel = black;
            }
        }
        let big = ss1::scale::scale2x(&ss1::scale::scale2x(&frame));
        let texture = engine::texture::init_from_memory2(
            RawTextureData {
                bytes: big.to_rgba(&palette),
                width: big.width as u32,
                height: big.height as u32,
                format: PixelFormat::RGBA,
            },
            &TextureOptions {
                wrap: false,
                filter: TextureFilter::Linear,
                ..TextureOptions::default()
            },
        );
        Ok(Hud {
            frame: Rc::new(texture),
            window,
        })
    }

    /// The view window on a screen, in pixels (x, y, w, h).
    pub fn window_on(&self, screen: (f32, f32)) -> (f32, f32, f32, f32) {
        let (ox, oy, s) = canvas_on(screen);
        let (x, y, w, h) = self.window;
        (ox + x * s, oy + y * s, w * s, h * s)
    }

    /// Remap a projection built for the window's aspect so it fills the
    /// window rather than the whole screen.
    pub fn window_projection(&self, screen: (f32, f32), projection: Matrix4<f32>) -> Matrix4<f32> {
        let (x, y, w, h) = self.window_on(screen);
        let sx = w / screen.0;
        let sy = h / screen.1;
        let cx = (x + w / 2.0) / screen.0 * 2.0 - 1.0;
        let cy = 1.0 - (y + h / 2.0) / screen.1 * 2.0;
        #[rustfmt::skip]
        let remap = Matrix4::new(
            sx, 0.0, 0.0, 0.0,
            0.0, sy, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            cx, cy, 0.0, 1.0,
        );
        remap * projection
    }

    pub fn window_aspect(&self) -> f32 {
        self.window.2 / self.window.3
    }

    pub fn objects(&self, screen: (f32, f32)) -> Vec<SceneObject> {
        let (ox, oy, s) = canvas_on(screen);
        let mut frame = SceneObject::screen_space_quad(
            self.frame.clone(),
            vec2(ox, oy),
            vec2(CANVAS.0 * s, CANVAS.1 * s),
        );
        frame.set_render_layer(RenderLayer::SceneUi);
        vec![frame]
    }
}

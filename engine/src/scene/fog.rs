//! Distance darkness: lit surfaces fade toward black with distance from the
//! eye, so a long corridor ends in shadow instead of a flat, evenly lit far
//! wall. Not in the original (SS2 authored no fog); part of this port's
//! horror look and off at density 0.
//!
//! Exponential: a surface `d` units away keeps `exp(-density * d)` of its
//! light, floored so nothing within a room ever goes fully black.
use std::cell::Cell;

use cgmath::{Matrix4, SquareMatrix};

use crate::gfx;

/// Fragment-side helper; materials multiply their lit colour by it.
pub const GLSL: &str = r#"
uniform highp vec4 fogEyeDensity;
// One return: the Xbox 360's shader compiler inlines only single-exit functions.
highp float fogFactor(highp vec3 worldPos) {
    highp float fog = 1.0;
    if (fogEyeDensity.w > 0.0) {
        highp float d = distance(worldPos, fogEyeDensity.xyz);
        fog = max(exp(-fogEyeDensity.w * d), 0.12);
    }
    return fog;
}
"#;

thread_local! {
    static FRAME: Cell<[f32; 4]> = const { Cell::new([0.0; 4]) };
    /// World gloss strength (see lightmap_material); rides along with the
    /// fog because both are per-frame view settings. 0 is off.
    static GLOSS: Cell<f32> = const { Cell::new(0.0) };
}

pub fn set_world_gloss(strength: f32) {
    GLOSS.with(|g| {
        g.set(if strength.is_finite() {
            strength.max(0.0)
        } else {
            0.0
        })
    });
}

/// Set this frame's eye (from the view matrix) and density.
pub fn set_frame(view: &Matrix4<f32>, density: f32) {
    let eye = view
        .invert()
        .map(|m| m.w.truncate())
        .unwrap_or_else(|| cgmath::vec3(0.0, 0.0, 0.0));
    let density = if density.is_finite() {
        density.max(0.0)
    } else {
        0.0
    };
    FRAME.with(|f| f.set([eye.x, eye.y, eye.z, density]));
}

pub struct Uniforms {
    eye_density: gfx::Uniform,
    /// Only the world shader declares it; elsewhere setting it is a no-op.
    world_gloss: gfx::Uniform,
}

impl Uniforms {
    pub fn new(program: &gfx::Program) -> Self {
        Self {
            eye_density: program.uniform("fogEyeDensity"),
            world_gloss: program.uniform("worldGloss"),
        }
    }

    pub fn bind(&self) {
        gfx::set_vec4(self.eye_density, FRAME.with(Cell::get));
        gfx::set_f32(self.world_gloss, GLOSS.with(Cell::get));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::vec3;

    #[test]
    fn eye_is_recovered_from_the_view_matrix() {
        let view = Matrix4::from_translation(vec3(-1.0, -2.0, -3.0));
        set_frame(&view, 0.02);
        let [x, y, z, d] = FRAME.with(Cell::get);
        assert!((x - 1.0).abs() < 1e-5 && (y - 2.0).abs() < 1e-5 && (z - 3.0).abs() < 1e-5);
        assert_eq!(d, 0.02);
        set_frame(&view, f32::NAN);
        assert_eq!(FRAME.with(Cell::get)[3], 0.0);
    }
}

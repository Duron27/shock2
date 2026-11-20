//! Bump detail derived from each texture's own brightness. Not in the
//! original (Dark had no bump maps, and neither 25AE nor the classic data
//! ship any); added so grout, seams and rivets catch the light instead of
//! reading as paint on a flat card.
//!
//! The height is the texel's luminance - dark texels sit lower, which is
//! how this era's art paints recesses - and its screen-space gradient tilts
//! the normal through a tangent frame built from derivatives (Mikkelsen's
//! surface-gradient bump), so meshes need no tangents. Distant surfaces
//! sample blurred mips, so the effect fades out instead of sparkling.
//! Strength 0 is off and costs nothing past one uniform branch.
use std::cell::Cell;

use crate::gfx;

pub const GLSL: &str = r#"
uniform float bumpStrength;
float bumpHeight(sampler2D tex, vec2 uv) {
    return dot(texture(tex, uv).rgb, vec3(0.299, 0.587, 0.114));
}
// `depth` is how deep, in world units, full black sits below full white.
vec3 bumpNormal(sampler2D tex, vec2 uv, highp vec3 pos, vec3 n, float depth) {
    if (bumpStrength <= 0.0) return n;
    vec2 duvdx = dFdx(uv);
    vec2 duvdy = dFdy(uv);
    highp vec3 dpdx = dFdx(pos);
    highp vec3 dpdy = dFdy(pos);
    // Height change across one pixel, sampled per pixel (not per 2x2 quad,
    // which would show as blocks).
    float h = bumpHeight(tex, uv);
    float dhdx = (bumpHeight(tex, uv + duvdx) - h) * depth * bumpStrength;
    float dhdy = (bumpHeight(tex, uv + duvdy) - h) * depth * bumpStrength;
    highp vec3 r1 = cross(dpdy, n);
    highp vec3 r2 = cross(n, dpdx);
    highp float det = dot(dpdx, r1);
    if (abs(det) < 1e-12) return n;
    highp vec3 grad = sign(det) * (dhdx * r1 + dhdy * r2);
    return normalize(abs(det) * n - grad);
}
"#;

thread_local! {
    static STRENGTH: Cell<f32> = const { Cell::new(0.0) };
}

pub fn set_strength(strength: f32) {
    STRENGTH.with(|s| {
        s.set(if strength.is_finite() {
            strength.max(0.0)
        } else {
            0.0
        })
    });
}

pub struct Uniforms {
    strength: gfx::Uniform,
}

impl Uniforms {
    pub fn new(program: &gfx::Program) -> Self {
        Self {
            strength: program.uniform("bumpStrength"),
        }
    }

    pub fn bind(&self) {
        gfx::set_f32(self.strength, STRENGTH.with(Cell::get));
    }
}

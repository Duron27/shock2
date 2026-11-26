//! A visual-only displacement shared by world meshes and their depth passes.
use crate::gfx;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldWave {
    /// Affine displacement per unit of height above the shared player origin.
    /// Zero disables the wave. Phase coefficients are evaluated once per frame.
    pub offset_per_height: cgmath::Vector3<f32>,
    pub origin_y: f32,
}

impl WorldWave {
    /// The wave as a matrix, for a position with w = 1: what
    /// `waveWorldPosition` does, for a shader that takes it in its view
    /// matrix instead (`SHOCK_WAVE_IN_VIEW`).
    pub fn matrix(&self) -> cgmath::Matrix4<f32> {
        let d = self.offset_per_height;
        let o = self.origin_y;
        cgmath::Matrix4::new(
            1.0,
            0.0,
            0.0,
            0.0, // column 0
            d.x,
            1.0 + d.y,
            d.z,
            0.0, // column 1: y moves along d
            0.0,
            0.0,
            1.0,
            0.0, // column 2
            -o * d.x,
            -o * d.y,
            -o * d.z,
            1.0, // column 3
        )
    }
}

impl Default for WorldWave {
    fn default() -> Self {
        Self {
            offset_per_height: cgmath::vec3(0.0, 0.0, 0.0),
            origin_y: 0.0,
        }
    }
}

// World-space, affine displacement preserves adjoining surfaces even when
// their tessellation differs (a nonlinear field tears coplanar wall/sign faces
// apart). Both eyes and all materials use the same field. Lighting continues
// to use the authored positions/normals; this is a visual intoxication effect.
// One vec4 keeps the extra vertex-uniform footprint small for skinned meshes.
// CPU-evaluated phase coefficients leave only multiply/add work per vertex.
pub(crate) const GLSL: &str = r#"
#ifdef SHOCK_WAVE_IN_VIEW
// The wave is in the view matrix (WorldWave::matrix).
#define waveWorldPosition(position) (position)
#else
uniform vec4 worldWave;
vec4 waveWorldPosition(vec4 position) {
    position.xyz += (position.y - worldWave.w) * worldWave.xyz;
    return position;
}
#endif
"#;

pub(crate) struct Uniforms {
    wave: gfx::Uniform,
}

impl Uniforms {
    pub fn new(program: &gfx::Program) -> Self {
        Self {
            wave: program.uniform("worldWave"),
        }
    }

    pub fn bind(&self, wave: WorldWave) {
        gfx::set_vec4(
            self.wave,
            [
                wave.offset_per_height.x,
                wave.offset_per_height.y,
                wave.offset_per_height.z,
                wave.origin_y,
            ],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{vec3, vec4};

    #[test]
    fn matrix_is_the_shader_wave() {
        let wave = WorldWave {
            offset_per_height: vec3(0.1, -0.05, 0.2),
            origin_y: 1.5,
        };
        let p = vec4(3.0f32, 4.0, -2.0, 1.0);
        let moved = wave.matrix() * p;
        let d = wave.offset_per_height * (p.y - wave.origin_y);
        for (got, want) in [
            (moved.x, p.x + d.x),
            (moved.y, p.y + d.y),
            (moved.z, p.z + d.z),
            (moved.w, 1.0),
        ] {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
    }
}

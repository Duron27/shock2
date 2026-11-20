//! A visual-only displacement shared by world meshes and their depth passes.
use crate::gfx;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldWave {
    /// Affine displacement per unit of height above the shared player origin.
    /// Zero disables the wave. Phase coefficients are evaluated once per frame.
    pub offset_per_height: cgmath::Vector3<f32>,
    pub origin_y: f32,
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
uniform vec4 worldWave;
vec4 waveWorldPosition(vec4 position) {
    position.xyz += (position.y - worldWave.w) * worldWave.xyz;
    return position;
}
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

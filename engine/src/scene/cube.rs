use once_cell::sync::OnceCell;

pub use crate::scene::Geometry;
use crate::scene::VertexPositionTexture;
use crate::scene::mesh::{self, Mesh};

/// A unit cube (position + uv), shared by every user.
pub struct Cube {}

static CUBE_GEOMETRY: OnceCell<Mesh> = OnceCell::new();

pub fn create() -> Cube {
    Cube {}
}

impl Geometry for Cube {
    fn draw(&self) {
        CUBE_GEOMETRY
            .get_or_init(|| {
                #[rustfmt::skip]
                let vertices: [f32; 180] = [
                -0.5, -0.5, -0.5, 0.0, 0.0, 0.5, -0.5, -0.5, 1.0, 0.0, 0.5, 0.5, -0.5, 1.0, 1.0,
                0.5, 0.5, -0.5, 1.0, 1.0, -0.5, 0.5, -0.5, 0.0, 1.0, -0.5, -0.5, -0.5, 0.0, 0.0,
                -0.5, -0.5, 0.5, 0.0, 0.0, 0.5, -0.5, 0.5, 1.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 0.5,
                0.5, 0.5, 1.0, 1.0, -0.5, 0.5, 0.5, 0.0, 1.0, -0.5, -0.5, 0.5, 0.0, 0.0, -0.5, 0.5,
                0.5, 1.0, 0.0, -0.5, 0.5, -0.5, 1.0, 1.0, -0.5, -0.5, -0.5, 0.0, 1.0, -0.5, -0.5,
                -0.5, 0.0, 1.0, -0.5, -0.5, 0.5, 0.0, 0.0, -0.5, 0.5, 0.5, 1.0, 0.0, 0.5, 0.5, 0.5,
                1.0, 0.0, 0.5, 0.5, -0.5, 1.0, 1.0, 0.5, -0.5, -0.5, 0.0, 1.0, 0.5, -0.5, -0.5,
                0.0, 1.0, 0.5, -0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 0.0, -0.5, -0.5, -0.5, 0.0,
                1.0, 0.5, -0.5, -0.5, 1.0, 1.0, 0.5, -0.5, 0.5, 1.0, 0.0, 0.5, -0.5, 0.5, 1.0, 0.0,
                -0.5, -0.5, 0.5, 0.0, 0.0, -0.5, -0.5, -0.5, 0.0, 1.0, -0.5, 0.5, -0.5, 0.0, 1.0,
                0.5, 0.5, -0.5, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 0.0, 0.5, 0.5, 0.5, 1.0, 0.0, -0.5,
                0.5, 0.5, 0.0, 0.0, -0.5, 0.5, -0.5, 0.0, 1.0,
            ];
                mesh::create(position_uv(&vertices))
            })
            .draw();
    }
}

/// Interleaved `x y z u v` floats as vertices.
pub(crate) fn position_uv(floats: &[f32]) -> Vec<VertexPositionTexture> {
    floats
        .chunks_exact(5)
        .map(|v| VertexPositionTexture {
            position: cgmath::vec3(v[0], v[1], v[2]),
            uv: cgmath::vec2(v[3], v[4]),
        })
        .collect()
}

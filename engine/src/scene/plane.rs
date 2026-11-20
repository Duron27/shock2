pub use crate::scene::Geometry;
use crate::scene::mesh::{self, Mesh};

/// A unit plane in XZ (position + uv).
pub struct Plane {
    mesh: Mesh,
}

pub fn create() -> Plane {
    create_with_uv_scale(100.0)
}

pub fn create_with_uv_scale(uv_scale: f32) -> Plane {
    #[rustfmt::skip]
    let vertices: [f32; 30] = [
        -0.5, 0.0, -0.5, 0.0, 0.0, 0.5, 0.0, -0.5, uv_scale, 0.0, 0.5, 0.0, 0.5, uv_scale,
        uv_scale, -0.5, 0.0, -0.5, 0.0, 0.0, -0.5, 0.0, 0.5, 0.0, uv_scale, 0.5, 0.0, 0.5,
        uv_scale, uv_scale,
    ];
    Plane {
        mesh: mesh::create(super::cube::position_uv(&vertices)),
    }
}

impl Geometry for Plane {
    fn draw(&self) {
        self.mesh.draw();
    }
}

pub use crate::scene::Geometry;
pub use crate::scene::VertexPosition;

/// Line segments: each pair of vertices is one line.
pub struct LinesMesh {
    mesh: crate::gfx::Mesh,
}

pub fn create(raw_vertices: Vec<VertexPosition>) -> LinesMesh {
    LinesMesh {
        mesh: crate::gfx::Mesh::new(&raw_vertices, None),
    }
}

impl Geometry for LinesMesh {
    fn draw(&self) {
        self.mesh.draw(crate::gfx::Topology::Lines);
    }
}

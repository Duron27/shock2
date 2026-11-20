use std::sync::Arc;

pub use crate::scene::Geometry;
pub use crate::scene::VertexPositionTexture;

use super::Vertex;

/// Non-indexed triangles. Cloning shares the GPU buffers.
#[derive(Clone)]
pub struct Mesh {
    mesh: Arc<crate::gfx::Mesh>,
}

pub fn create<T: Vertex>(raw_vertices: Vec<T>) -> Mesh {
    Mesh {
        mesh: Arc::new(crate::gfx::Mesh::new(&raw_vertices, None)),
    }
}

impl Geometry for Mesh {
    fn draw(&self) {
        self.mesh.draw(crate::gfx::Topology::Triangles);
    }
}

impl Mesh {
    /// Replace vertex data without allocating a new mesh per frame.
    pub fn update<T: Vertex>(&self, vertices: &[T]) {
        assert_eq!(vertices.len(), self.mesh.vertex_count() as usize);
        self.mesh.update_vertices(vertices);
    }
}

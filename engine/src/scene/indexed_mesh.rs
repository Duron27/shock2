pub use crate::scene::Geometry;

use super::Vertex;

/// Indexed triangles. Cloning shares the GPU buffers.
#[derive(Clone)]
pub struct IndexedMesh {
    mesh: std::sync::Arc<crate::gfx::Mesh>,
}

pub fn create<T: Vertex>(raw_vertices: Vec<T>, indices: Vec<u32>) -> IndexedMesh {
    IndexedMesh {
        mesh: std::sync::Arc::new(crate::gfx::Mesh::new(&raw_vertices, Some(&indices))),
    }
}

impl Geometry for IndexedMesh {
    fn draw(&self) {
        self.mesh.draw(crate::gfx::Topology::Triangles);
    }
}

use cgmath::{Matrix4, Vector2};

#[derive(Clone, Copy)]
pub struct CullingInfo {
    pub view: Matrix4<f32>,
    pub projection: Matrix4<f32>,
    pub screen_size: Vector2<f32>,
}

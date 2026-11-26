//! Rigid two-anchor posing. Geometry, scale and ownership stay with the primary hand.
use cgmath::{Quaternion, Rotation, Vector3};
use serde::Serialize;

/// Authored support sockets and regions currently enabled in gameplay.
/// Keep the Explorer eligibility notice and runtime policy together.
pub fn supports_model(model: &str) -> bool {
    matches!(
        model,
        "wrench_h" | "rapier_h" | "shard_h" | "atek_h" | "ar15_h" | "sg_h" | "fsn_h" | "al_h"
    )
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct GripPose {
    pub position: Vector3<f32>,
    pub rotation: Quaternion<f32>,
}

impl GripPose {
    pub fn point(self, local: Vector3<f32>) -> Vector3<f32> {
        self.position + self.rotation.rotate_vector(local)
    }

    pub fn is_tracked(self) -> bool {
        [
            self.position.x,
            self.position.y,
            self.position.z,
            self.rotation.s,
            self.rotation.v.x,
            self.rotation.v.y,
            self.rotation.v.z,
        ]
        .into_iter()
        .all(f32::is_finite)
            && crate::util::tracked_rotation(self.rotation).is_some()
    }
}

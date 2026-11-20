//! VR readout carried by a wielded psi amp: the selected power's hologram
//! floats over the sphere, and the overload meter wraps around it. Both follow
//! the amp's rendered transform, so a rescaled amp keeps them on its surface.
use cgmath::{EuclideanSpace, InnerSpace, Matrix3, Matrix4, Point3, Transform, Vector3, vec3};
use shipyard::{EntityId, Get, View, World};

use crate::Handedness;

/// The ball of `amp_h.bin` in model space: a least-squares sphere fit to its
/// surface vertices near the ball. The model also carries a cable, a baked
/// hand and a forearm, so its bounds do not locate the ball. The 25AE and
/// classic meshes fit within 0.02 of these.
pub(crate) const AMP_SPHERE_CENTER: Point3<f32> = Point3::new(-0.42, 0.094, -0.04);
const AMP_SPHERE_RADIUS: f32 = 0.155;
/// Reference anchor on the baked forearm, in model space (+x toward the elbow).
/// `wrist` tunes along this axis to seat the bracelet beside the cable inlet;
/// both bio bands and the implant socket consume that final frame.
const AMP_WRIST_CENTER: Point3<f32> = Point3::new(0.08, -0.12, -0.10);
/// Bands at this radius sit just off the forearm's skin, palm and back.
const AMP_FOREARM_RADIUS: f32 = 0.15;

/// World centre, radius and unit model axes of the amp's sphere, composed
/// exactly as the held amp mesh is drawn (charge offset, pose, fit scale,
/// left-hand mirror). Model +y runs palm to ball; +x points at the wrist.
pub(crate) fn sphere(
    world: &World,
    amp: EntityId,
    hand: Handedness,
) -> Option<(Vector3<f32>, f32, Matrix3<f32>)> {
    let transform = drawn_transform(world, amp, hand)?;
    Some((
        transform.transform_point(AMP_SPHERE_CENTER).to_vec(),
        AMP_SPHERE_RADIUS * transform.x.truncate().magnitude(),
        Matrix3::from_cols(
            transform.x.truncate().normalize(),
            transform.y.truncate().normalize(),
            transform.z.truncate().normalize(),
        ),
    ))
}

/// The amp's baked wrist as a glove-style wrist frame (+Y toward the fingers,
/// +Z out of the back of the hand, unit and right-handed in either hand), with
/// the forearm's world radius.
pub(crate) fn wrist(world: &World, amp: EntityId, hand: Handedness) -> Option<(Matrix4<f32>, f32)> {
    let transform = drawn_transform(world, amp, hand)?;
    // Apply physical centimetres after resolving the mirrored/scaled model,
    // along its wrist axis. Both bio bands and the implant use this result.
    let forward = crate::dev_params::get(crate::dev_params::PSI_BAND_FORWARD_CM)
        * (0.01 / crate::world_scale::meters_per_world_unit());
    let back = crate::dev_params::get(crate::dev_params::PSI_BAND_BACK_CM)
        * (0.01 / crate::world_scale::meters_per_world_unit());
    Some((
        wrist_frame(transform) * Matrix4::from_translation(vec3(0.0, forward, back)),
        AMP_FOREARM_RADIUS
            * transform.x.truncate().magnitude()
            * crate::dev_params::get(crate::dev_params::PSI_BAND_SCALE),
    ))
}

fn wrist_frame(transform: Matrix4<f32>) -> Matrix4<f32> {
    let fingers = -transform.x.truncate().normalize();
    let back = -transform.y.truncate().normalize();
    Matrix4::from_cols(
        fingers.cross(back).extend(0.0),
        fingers.extend(0.0),
        back.extend(0.0),
        transform.transform_point(AMP_WRIST_CENTER).to_homogeneous(),
    )
}

/// The amp's model-to-world transform, as its mesh is drawn (charge offset,
/// pose, fit scale, left-hand mirror).
fn drawn_transform(world: &World, amp: EntityId, hand: Handedness) -> Option<Matrix4<f32>> {
    model_transform(world, amp, Some(hand))
}

/// Physical amp model frame shared by attachments. Flat uses the authored
/// right-hand mesh; VR adds its live fit and the holding hand's reflection.
pub(crate) fn model_transform(
    world: &World,
    amp: EntityId,
    hand: Option<Handedness>,
) -> Option<Matrix4<f32>> {
    let transforms = world
        .borrow::<View<crate::runtime_props::RuntimePropTransform>>()
        .ok()?;
    let transform = transforms.get(amp).ok()?.0;
    Some(match hand {
        Some(hand) => transform * crate::vr_config::psi_amp_fit_scale() * hand.gun_mirror(),
        None => transform,
    })
}

#[cfg(test)]
mod tests {}

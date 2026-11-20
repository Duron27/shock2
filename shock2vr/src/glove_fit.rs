//! Global hand-frame calibration, plus fit-scene-only visual previews.
use cgmath::{Matrix4, Quaternion, Rotation, Vector3, vec3};

use crate::{dev_params, vr_config::Handedness, vr_support::GripPose};

/// Controller-to-hand translation, in the rotation's space and world units.
/// Subtract it when publishing a raw controller target for a calibrated grip.
pub(crate) fn forward_translation(rotation: Quaternion<f32>, forward_cm: f32) -> Vector3<f32> {
    crate::util::tracked_rotation(rotation).map_or(vec3(0.0, 0.0, 0.0), |rotation| {
        rotation.rotate_vector(vec3(
            0.0,
            0.0,
            -forward_cm * 0.01 / crate::world_scale::meters_per_world_unit(),
        ))
    })
}

/// Translate the consumed hand frame once, before menus and gameplay diverge.
/// Cached grip samples and wrist mounts remain relative to that frame, so live
/// tuning moves glove, held item and interactions together without rebaking.
/// Work on a copy: callers retain raw tracking and repeated updates never drift.
pub(crate) fn calibrated_input(
    input: &crate::input_context::InputContext,
    fit_scene: bool,
    forward_cm: f32,
) -> crate::input_context::InputContext {
    let mut calibrated = input.clone();
    if fit_scene {
        for (index, hand) in [&mut calibrated.left_hand, &mut calibrated.right_hand]
            .into_iter()
            .enumerate()
        {
            if input
                .pose_tracking
                .is_some_and(|tracking| !tracking.hands[index])
            {
                continue;
            }
            hand.position += forward_translation(hand.rotation, forward_cm);
        }
    }
    calibrated
}

#[derive(Clone, Copy)]
pub(crate) struct GloveFit {
    pub side_cm: f32,
    pub up_cm: f32,
    pub size: f32,
    pub visible: bool,
}

impl GloveFit {
    pub fn current() -> Self {
        Self {
            side_cm: dev_params::get(dev_params::GLOVE_SIDE_CM),
            up_cm: dev_params::get(dev_params::GLOVE_UP_CM),
            size: dev_params::get(dev_params::GLOVE_FIT_SIZE),
            visible: dev_params::get_bool(dev_params::GLOVE_FIT_VISIBLE),
        }
    }

    /// Physical centimeters in controller space; mirror lateral movement so
    /// both hands can be fitted symmetrically. Scale about the shifted origin.
    pub fn transform(self, pose: GripPose, side: Handedness) -> Matrix4<f32> {
        let offset = side.mirror_point(vec3(self.side_cm, self.up_cm, 0.0))
            * (0.01 / crate::world_scale::meters_per_world_unit());
        Matrix4::from_translation(pose.position)
            * Matrix4::from(pose.rotation)
            * Matrix4::from_translation(offset)
            * Matrix4::from_scale(self.size)
    }
}

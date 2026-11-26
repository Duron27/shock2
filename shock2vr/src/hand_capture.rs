//! Tracked-hand capture: OpenXR `XR_EXT_hand_tracking` joint positions to the
//! glove's finger bones, for authoring grips with real hands.
//!
//! The 26 OpenXR joints (palm, wrist, thumb metacarpal..tip, index..little
//! metacarpal..tip) share indices 1..=25 with the SteamVR skeleton the glove
//! poses use (index 0 is the palm there, the root here). Only joint positions
//! are used. The glove keeps its own size: its palm is fitted rigidly over the
//! tracked palm, then each finger is solved, with the glove's bone lengths and
//! within simple joint limits, to put its tip on the tracked tip while bending
//! like the tracked finger. Bone roll is not tracked, so the thumb's
//! opposition twist is the glove's own.
use cgmath::Vector3;

pub const HAND_JOINT_COUNT: usize = 26;

/// OpenXR joint positions, indexed like `XrHandJointEXT`.
pub type HandJoints = [Vector3<f32>; HAND_JOINT_COUNT];

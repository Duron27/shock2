//! One translation for the entire tracked rig: lowering the eye for crouch
//! must lower the hands by the same amount, preserving physical reach and IPD.
use cgmath::{InnerSpace, Vector3, Zero, vec3};
use shipyard::{Unique, UniqueView, World};

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct TrackingTransform {
    translation: Vector3<f32>,
    head_stage_y: f32,
    stage_offset_meters: f32,
    #[serde(default = "default_meters_per_world_unit")]
    meters_per_world_unit: f32,
}

fn default_meters_per_world_unit() -> f32 {
    crate::METERS_PER_WORLD_UNIT
}

impl TrackingTransform {
    /// All heights except `head_stage_y` and `stage_offset_meters` are world units.
    /// Resolve once from the head centre, then apply identically to both eyes,
    /// both hands, and gameplay's head pose. Never clamp individual tracked parts.
    pub fn new(
        center_above_floor: f32,
        eye_cap: f32,
        head_stage_y: f32,
        stage_offset_meters: f32,
    ) -> Self {
        Self::new_with_scale(
            center_above_floor,
            eye_cap,
            head_stage_y,
            stage_offset_meters,
            crate::world_scale::meters_per_world_unit(),
        )
    }

    fn new_with_scale(
        center_above_floor: f32,
        eye_cap: f32,
        head_stage_y: f32,
        stage_offset_meters: f32,
        scale: f32,
    ) -> Self {
        let offset = stage_offset_meters / scale;
        let head_y = head_stage_y / scale - center_above_floor;
        let lowering = (head_y - eye_cap).max(0.0);
        Self {
            translation: vec3(0.0, offset - center_above_floor - lowering, 0.0),
            head_stage_y,
            stage_offset_meters,
            meters_per_world_unit: scale,
        }
    }

    pub fn with_stance(self, center_above_floor: f32, eye_cap: f32) -> Self {
        Self::new(
            center_above_floor,
            eye_cap,
            self.head_stage_y,
            self.stage_offset_meters,
        )
    }

    /// Apply a stance transition to an already-converted input, including any
    /// pawn-local debug overrides, without changing relative tracked geometry.
    pub fn rebase_input(input: &mut crate::input_context::InputContext, center: f32, cap: f32) {
        if let Some(old) = input.tracking {
            let new = old.with_stance(center, cap);
            let ratio = old.meters_per_world_unit / new.meters_per_world_unit;
            let joints = [&mut input.left_hand.joints, &mut input.right_hand.joints]
                .into_iter()
                .flat_map(|joints| joints.iter_mut().flatten());
            for position in [
                &mut input.head.position,
                &mut input.left_hand.position,
                &mut input.right_hand.position,
            ]
            .into_iter()
            .chain(joints)
            {
                *position = (*position - old.translation) * ratio + new.translation;
            }
            input.tracking = Some(new);
        }
    }

    pub fn stage_to_pawn(self, stage: Vector3<f32>) -> Vector3<f32> {
        stage / self.meters_per_world_unit + self.translation
    }

    /// Raw eye height above the runtime's floor, before stance/eye-offset correction.
    pub fn head_height_meters(self) -> f32 {
        self.head_stage_y
    }
}

/// Horizontal roomscale walking (`user_settings::RoomscaleMode`). Tracking history is transient and belongs
/// to the mission, so a new mission/save establishes its own physical origin.
#[derive(Unique)]
pub struct RoomscaleState {
    previous_head: Option<Vector3<f32>>,
    offset: Vector3<f32>,
}

impl Default for RoomscaleState {
    fn default() -> Self {
        Self {
            previous_head: None,
            offset: Vector3::zero(),
        }
    }
}

impl RoomscaleState {
    /// Return a pawn-local horizontal movement request. Subtract the entire
    /// tracked delta from the rig, then let the ordinary character controller
    /// apply whatever travel is legal. The rejected part is never accumulated.
    pub fn sample(
        &mut self,
        input: &crate::input_context::InputContext,
        enabled: bool,
        advancing: bool,
        locomotion_allowed: bool,
    ) -> Vector3<f32> {
        if !enabled {
            *self = Self::default();
            return Vector3::zero();
        }
        // Debug idle updates are not simulation frames: do not consume a pose
        // before /v1/step has had a chance to move the character.
        if !advancing {
            return Vector3::zero();
        }
        let head = vec3(input.head.position.x, 0.0, input.head.position.z);
        if input.pose_tracking.is_some_and(|tracking| !tracking.head)
            || !head.x.is_finite()
            || !head.z.is_finite()
        {
            self.previous_head = None;
            return Vector3::zero();
        }
        if input.tracking_origin_changed {
            self.previous_head = None;
        }
        let previous = self.previous_head.replace(head);
        let Some(previous) = previous else {
            // Spawn/recovery uses the wearer's current position as the origin.
            self.offset = -head;
            return Vector3::zero();
        };
        let delta = head - previous;
        // Reject discontinuities, not ordinary walking speed. This is a coarse
        // fallback; OpenXR reference-space changes also explicitly reset above.
        if delta.magnitude() > 0.5 / crate::world_scale::meters_per_world_unit() {
            self.offset = -head;
            return Vector3::zero();
        }
        self.offset -= delta;
        if !locomotion_allowed {
            // Another system owns the body (grip, mantle, death, control
            // lock): treat the whole step as blocked, like a wall, so the rig
            // stays on the capsule instead of drifting and snapping later.
            return Vector3::zero();
        }
        delta
    }

    pub fn offset(world: &World) -> Vector3<f32> {
        world
            .borrow::<UniqueView<Self>>()
            .map_or(Vector3::zero(), |state| state.offset)
    }

    /// Pause, focus loss and save-load skip mission updates. Reset only the
    /// delta history so movement during that interval cannot become a resumed step.
    pub fn invalidate(world: &World) {
        if let Ok(mut state) = world.borrow::<shipyard::UniqueViewMut<Self>>() {
            state.previous_head = None;
        }
    }
}

//! Player interaction abstraction.
//!
//! `mission_core` drives the player's interaction through a
//! `Box<dyn PlayerInteraction>`; the flatscreen `FlatPlayerController` is
//! its implementation. It speaks the `VirtualHandEffect` language, which
//! `mission_core` processes in one place.

use cgmath::{Point3, Quaternion, Vector3};
use rapier3d::prelude::RigidBodyHandle;
use shipyard::{EntityId, World};

use crate::{
    flat_player_controller::FlatPlayerController, input_context::InputContext,
    physics::PhysicsWorld, virtual_hand::VirtualHandEffect, vr_config::Handedness,
};

/// Read-only per-frame inputs an interaction controller needs to update.
pub struct InteractionContext<'a> {
    pub physics: &'a PhysicsWorld,
    pub world: &'a World,
    pub input: &'a InputContext,
    pub player_pos: Vector3<f32>,
    pub player_rotation: Quaternion<f32>,
    /// Shared pawn-space eye for flat rendering, aiming and weapon placement.
    pub flat_eye: crate::death_camera::EyePose,
    pub step_dt: f32,
}

/// How the player interacts with the world. The effects returned by `update`
/// (and `grab`/`wield`) are applied by `mission_core::process_virtual_hand_effects`.
pub trait PlayerInteraction {
    /// Per-frame update; returns effects to apply.
    fn update(&mut self, ctx: &InteractionContext) -> Vec<VirtualHandEffect>;

    /// The wielded entity (as the "left") - for `PlayerInfo`.
    fn held_entities(&self) -> (Option<EntityId>, Option<EntityId>);

    /// Recoil's deflection of the fire ray off the camera axis, in radians
    /// (+x right, +y up) - what the reticle shifts by.
    fn flat_aim_bias(&self) -> cgmath::Vector2<f32>;

    /// Kick a firing weapon's viewmodel recoil spring.
    fn kick_viewmodel(&mut self, entity: EntityId, impulse: crate::weapon_recoil::RecoilImpulse);

    /// Entities under the reticle, for the hover-highlight overlay.
    fn highlighted_entities(&self) -> Vec<EntityId>;

    /// The reach-bounded crosshair target, where a cursor tool is applied.
    fn frob_target(&self) -> Option<EntityId>;

    /// What the reticle is aiming at, for readouts that name one object.
    fn highlighted_entity(&self, hand: Handedness) -> Option<EntityId>;

    /// The first-person viewmodel entity (drawn on top).
    fn viewmodel_entity(&self) -> Option<EntityId>;

    /// Wield `entity_id`. Returns any effects.
    fn grab(&mut self, world: &World, entity_id: EntityId, hand: Handedness)
    -> Vec<VirtualHandEffect>;

    /// A held entity was recreated as `new`; track the new id.
    fn replace_entity(&mut self, old: EntityId, new: EntityId, rigid_body: RigidBodyHandle);

    /// A held entity was destroyed; release it.
    fn on_entity_destroyed(&mut self, entity_id: EntityId);

    /// Which hand is wielding `entity_id`: the right, the hand that fires.
    fn holding_hand(&self, entity_id: EntityId) -> Option<Handedness>;

    /// Whether `entity_id` is currently held. Answered from
    /// [`Self::holding_hand`] so the two cannot disagree.
    fn is_holding(&self, entity_id: EntityId) -> bool {
        self.holding_hand(entity_id).is_some()
    }

    /// Wield `entity_id` as the first-person weapon.
    fn wield(&mut self, entity_id: EntityId) -> Vec<VirtualHandEffect>;

    fn is_wielding(&self) -> bool;

    /// The camera/crosshair fire ray (origin, forward).
    fn flat_aim_ray(&self) -> Option<(Point3<f32>, Vector3<f32>)>;
}

pub struct FlatInteraction {
    controller: FlatPlayerController,
    highlighted: Option<EntityId>,
}

impl FlatInteraction {
    pub fn new() -> Self {
        Self {
            controller: FlatPlayerController::new(),
            highlighted: None,
        }
    }
}

impl Default for FlatInteraction {
    fn default() -> Self {
        Self::new()
    }
}

impl PlayerInteraction for FlatInteraction {
    fn update(&mut self, ctx: &InteractionContext) -> Vec<VirtualHandEffect> {
        let (msgs, highlighted) = self.controller.update(
            &ctx.input.right_hand,
            ctx.player_pos,
            ctx.player_rotation,
            ctx.flat_eye,
            ctx.step_dt,
            ctx.world,
            ctx.physics,
        );
        self.highlighted = highlighted;
        msgs
    }

    fn held_entities(&self) -> (Option<EntityId>, Option<EntityId>) {
        (self.controller.wielded_entity(), None)
    }

    fn highlighted_entities(&self) -> Vec<EntityId> {
        self.highlighted.into_iter().collect()
    }

    fn frob_target(&self) -> Option<EntityId> {
        self.controller.frob_target()
    }

    fn highlighted_entity(&self, _hand: Handedness) -> Option<EntityId> {
        self.highlighted
    }

    fn viewmodel_entity(&self) -> Option<EntityId> {
        self.controller.wielded_entity()
    }

    fn grab(
        &mut self,
        _world: &World,
        entity_id: EntityId,
        _hand: Handedness,
    ) -> Vec<VirtualHandEffect> {
        self.controller.wield(entity_id)
    }

    fn replace_entity(&mut self, old: EntityId, new: EntityId, _rigid_body: RigidBodyHandle) {
        self.controller.replace_wielded(old, new);
    }

    fn on_entity_destroyed(&mut self, entity_id: EntityId) {
        self.controller.on_entity_destroyed(entity_id);
    }

    fn holding_hand(&self, entity_id: EntityId) -> Option<Handedness> {
        (self.controller.wielded_entity() == Some(entity_id)).then_some(Handedness::Right)
    }

    fn kick_viewmodel(&mut self, entity: EntityId, impulse: crate::weapon_recoil::RecoilImpulse) {
        self.controller.kick(entity, impulse);
    }

    fn wield(&mut self, entity_id: EntityId) -> Vec<VirtualHandEffect> {
        self.controller.wield(entity_id)
    }

    fn is_wielding(&self) -> bool {
        self.controller.is_wielding()
    }

    fn flat_aim_ray(&self) -> Option<(Point3<f32>, Vector3<f32>)> {
        self.controller.aim_ray()
    }

    fn flat_aim_bias(&self) -> cgmath::Vector2<f32> {
        self.controller.aim_bias()
    }
}

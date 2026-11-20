// Helper to convert the input context to a form more useful for gameplay / interacting with the world

use cgmath::{Quaternion, Rotation, Vector3, vec3};
use dark::{
    SCALE_FACTOR,
    properties::{
        FrobFlag, PropFrobInfo,
    },
};

use shipyard::{EntityId, Get, UniqueView, View, World};

use crate::scripts::Message;

const HAND_OFFSET: Vector3<f32> = vec3(0.0, 0.0, 0.0);

/// Where a tracked controller is in the world. Inputs are in pawn space, so
/// `vr_climb` resolves grips against exactly the pose the hand is drawn and
/// raycast from.
pub fn hand_world_position(
    pawn_pos: Vector3<f32>,
    pawn_rotation: Quaternion<f32>,
    hand_local: Vector3<f32>,
) -> Vector3<f32> {
    pawn_pos + HAND_OFFSET + pawn_rotation.rotate_vector(hand_local)
}

/// Maximum world-space distance from the hand/eye to a frob target's visible
/// surface. Retail `shock2.gam` authors `GAMEPARAM.Frob Dist = 50`; the
/// original `PickSetFocus` treats that as squared SS2 units, while this engine
/// divides authored world geometry by [`SCALE_FACTOR`].
pub(crate) const FROB_REACH: f32 = 7.071_068 / SCALE_FACTOR;


#[derive(Debug)]
pub enum VirtualHandEffect {
    OutMessage {
        message: Message,
    },
    // Keeping to allow for debugging
    #[allow(dead_code)]
    ApplyForce {
        entity_id: EntityId,
        force: Vector3<f32>,
        torque: Vector3<f32>,
    },
    /// Refit contact geometry once when a prepared melee grip is acquired.
    SetPositionRotation {
        entity_id: EntityId,
        position: Vector3<f32>,
        rotation: Quaternion<f32>,
        scale: Vector3<f32>,
    },
    // Keeping to allow for debugging
    #[allow(dead_code)]
    SpawnEntity {
        template_id: i32,
        position: Vector3<f32>,
        rotation: Quaternion<f32>,
    },
    HoldItem {
        entity_id: EntityId,
    },
    /// Store a world pickup and report only a successful transfer. Unlike a
    /// held pickup, storage can merge and destroy the incoming entity.
    StoreWorldPickup {
        entity_id: EntityId,
    },
    /// Move an item into the player's backpack - a flat world pickup of
    /// ordinary loot, the weapon a wield swap displaced, or a VR hand opened
    /// over the cyber interface's inventory strip. VR's ordinary release is
    /// still the physical `DropItem`; only the strip claims one.
    ///
    /// Storing an item that was HELD must be paired with a preceding
    /// `MessagePayload::Drop` to it (see `FlatPlayerController::wield` and
    /// `mission_core`'s `rewrite_strip_release`): unlike `DropItem`, this
    /// variant does not dispatch one, so the world-model restore and
    /// psi-charge cancel would otherwise be skipped.
    StoreItem {
        entity_id: EntityId,
    },
}




pub(crate) fn can_grab_item(world: &World, entity_id: EntityId) -> bool {
    let v_prop_frobinfo = world.borrow::<View<PropFrobInfo>>().unwrap();

    if let Ok(frob_info) = v_prop_frobinfo.get(entity_id) {
        if frob_info.world_action.contains(FrobFlag::MOVE)
            || frob_info.world_action.contains(FrobFlag::USE_AMMO)
        {
            return true;
        }
    }

    false
}

/// Whether this object is a key source. Every PropKeySrc receives the derived
/// `internal_keycard` script, whose Frob records the credential and consumes
/// the pickup; bypassing it creates a physical card that cannot unlock doors.
pub(crate) fn is_key_source(world: &World, entity_id: EntityId) -> bool {
    world
        .borrow::<View<dark::properties::PropKeySrc>>()
        .map(|keycards| keycards.get(entity_id).is_ok())
        .unwrap_or(false)
}

/// Whether taking this world item must go through a script before the physical
/// transfer. An authored `SCRIPT` world action owns its side effects and item
/// fate (for example `FrobQB` awards a quest bit and moves the item exactly
/// once). The always-collected categories - keycards, nanite piles, cyber
/// modules and logs - must route through Frob rather than a physical grab in
/// every presentation and on every path, even where their authored world action
/// is only `MOVE` (see `scripts::script_util::is_always_collected`, the shared
/// predicate every acquisition site consults).
pub(crate) fn uses_scripted_world_frob(world: &World, entity_id: EntityId) -> bool {
    let has_authored_world_script = world
        .borrow::<View<PropFrobInfo>>()
        .map(|frob_info| {
            frob_info
                .get(entity_id)
                .is_ok_and(|frob_info| frob_info.world_action.contains(FrobFlag::SCRIPT))
        })
        .unwrap_or(false);
    has_authored_world_script || crate::scripts::script_util::is_always_collected(world, entity_id)
}

/// Whether an inventory item is a wieldable weapon - a gun (`PropPlayerGun`) or
/// a melee arm (`PropLimbModel`). Clicking one in the backpack/strip wields it
/// (`Effect::GrabEntity`) instead of using it; shared by `ContainerGui` and the
/// flat cursor-drag wield so the two can't drift.
pub(crate) fn is_wieldable_weapon(world: &World, entity_id: EntityId) -> bool {
    world
        .borrow::<View<dark::properties::PropPlayerGun>>()
        .map(|v| v.get(entity_id).is_ok())
        .unwrap_or(false)
        || world
            .borrow::<View<dark::properties::PropLimbModel>>()
            .map(|v| v.get(entity_id).is_ok())
            .unwrap_or(false)
}

/// Whether the calibrated glove is drawn for a hand holding `held_entity`.
///
/// Successfully stripped weapon models keep the glove. Legacy weapon models
/// and the psi amp replace it with their authored hand; drawing both would
/// overlap. Empty hands and ordinary held objects also keep the glove.
pub(crate) fn shows_hand_visual(world: &World, held_entity: Option<EntityId>) -> bool {
    // Calibration override: draw the glove *as well as* the weapon model, so
    // the `_h` rig's baked fist can be compared against where the controller
    // actually is. See `dev_params::MELEE_GLOVE_OVERLAY`.
    if crate::dev_params::get(crate::dev_params::MELEE_GLOVE_OVERLAY) > 0.5 {
        return true;
    }
    match held_entity {
        None => true,
        Some(entity_id) => {
            !is_wieldable_weapon(world, entity_id)
                || world
                    .borrow::<View<crate::runtime_props::RuntimePropGloveWeapon>>()
                    .map(|v| v.get(entity_id).is_ok())
                    .unwrap_or(false)
        }
    }
}

/// Find the player's carried weapon matching an original gamesys weapon
/// archetype. Mission objects and carried items can have more-specific or
/// mission-local template ids, so selection follows the preserved canonical
/// class identity through the global inheritance hierarchy.
pub(crate) fn carried_weapon_by_class(world: &World, class_template_id: i32) -> Option<EntityId> {
    let hierarchy = world
        .borrow::<UniqueView<crate::mission::mission_core::GlobalTemplateHierarchy>>()
        .ok()?;

    crate::scripts::script_util::player_carried_items(world)
        .into_iter()
        .find(|entity| {
            is_wieldable_weapon(world, *entity)
                && crate::scripts::script_util::entity_class_template_id(world, *entity)
                    .is_some_and(|template_id| {
                        hierarchy.is_or_descends_from(template_id, class_template_id)
                    })
        })
}

#[cfg(test)]
mod tests {
    
    
    use super::*;
    
    use dark::properties::{PropFrobInfo, PropPlayerGun};

    fn scripted_inventory_use() -> PropFrobInfo {
        PropFrobInfo {
            world_action: FrobFlag::MOVE,
            inventory_action: FrobFlag::SCRIPT,
            tool_action: FrobFlag::empty(),
        }
    }

    fn player_gun() -> PropPlayerGun {
        PropPlayerGun {
            flags: 0,
            hand_model: "held_h".to_owned(),
            icon_file: String::new(),
            model_offset: vec3(0.0, 0.0, 0.0),
            fire_offset: vec3(0.0, 0.0, 0.0),
            heading: 0,
            reload_pitch: 0,
            reload_rate: 0,
            gun_type: 0,
        }
    }

    /// A wielded weapon draws its own first-person model at the hand, so the
    /// hand visual has to step aside - otherwise there is a hand inside the gun.
    #[test]
    fn wielded_weapon_replaces_the_hand_visual() {
        let mut world = World::new();
        let weapon = world.add_entity(player_gun());

        assert!(!shows_hand_visual(&world, Some(weapon)));
    }

    #[test]
    fn only_successfully_stripped_weapons_keep_the_glove() {
        let mut world = World::new();
        let gun = world.add_entity((player_gun(),));
        assert!(!shows_hand_visual(&world, Some(gun)));
        world.add_component(
            gun,
            crate::runtime_props::RuntimePropGloveWeapon {
                item_scale: 0.7,
                magazine_anchor: None,
            },
        );
        assert!(shows_hand_visual(&world, Some(gun)));
        world.remove::<crate::runtime_props::RuntimePropGloveWeapon>(gun);
        assert!(!shows_hand_visual(&world, Some(gun)));
    }

    /// An empty hand, or one holding something with no first-person weapon
    /// model (a crate, a consumable), still shows the hand.
    #[test]
    fn empty_and_plain_held_hands_keep_the_hand_visual() {
        let mut world = World::new();
        let consumable = world.add_entity(scripted_inventory_use());

        assert!(shows_hand_visual(&world, None));
        assert!(shows_hand_visual(&world, Some(consumable)));
    }

}

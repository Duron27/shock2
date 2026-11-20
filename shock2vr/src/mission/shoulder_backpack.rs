//! Release-to-stow behind either shoulder. Coordinates are in tracked pawn space.
//! Uses the head's slowly followed yaw, never its pitch/roll, for body direction.
use shipyard::{EntityId, Get, View, World};

/// Only live inventory membership can supply a shoulder recall. A marker never
/// creates ownership or allows a dropped/destroyed/transferred weapon to return.
pub(super) fn weapons(world: &World, inventory: EntityId) -> [Option<EntityId>; 2] {
    let mut slots = [None; 2];
    let contents = crate::inventory::Inventory::from_container(
        world,
        inventory,
        crate::inventory::grid_for(world, inventory),
    );
    let marked = world
        .borrow::<View<crate::runtime_props::RuntimePropShoulderWeapon>>()
        .unwrap();
    for item in contents.all_items() {
        if let Ok(slot) = marked.get(item.entity) {
            if let Some(out) = slots.get_mut(slot.0 as usize) {
                *out = Some(item.entity);
            }
        }
    }
    slots
}

//! Shared body-slot gestures for thigh weapons and backpack-backed chest items.

use crate::runtime_props::RuntimePropHolstered;
use shipyard::{EntityId, IntoIter, IntoWithId, View, World};

/// Slots are right/left; hand arrays are left/right.
pub(super) fn occupants(world: &World) -> [Option<EntityId>; 2] {
    let mut slots = [None; 2];
    let holstered = world.borrow::<View<RuntimePropHolstered>>().unwrap();
    for (id, slot) in holstered.iter().with_id() {
        if let Some(cell) = slots.get_mut(slot.slot as usize) {
            *cell = Some(id);
        }
    }
    slots
}

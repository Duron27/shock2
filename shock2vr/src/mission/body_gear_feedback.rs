//! Read-only body gear feedback. Inventory ownership stays in the pouch/holsters.
use cgmath::vec3;
use engine::scene::{self};
use shipyard::{EntityId, World};

#[derive(Clone, Debug, serde::Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(super) enum PouchState {
    Inactive,
    Ready,
    Empty,
    Refused,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(super) struct PouchReadout {
    pub weapon: Option<i32>,
    pub icon: Option<String>,
    pub count: Option<u32>,
    pub state: PouchState,
    pub near: bool,
}

impl PouchReadout {
    pub fn resolve(
        world: &World,
        held: [Option<EntityId>; 2],
        available: [bool; 2],
        near: [bool; 2],
        refused: bool,
        enabled: bool,
    ) -> Self {
        let guns = held.map(|item| {
            item.filter(|gun| {
                crate::scripts::script_util::active_gun_setting(world, *gun).is_some()
            })
        });
        // One shared pouch has no unambiguous draw while both hands hold guns.
        let (weapon, taking) = match guns {
            [Some(gun), None] => (Some(gun), 1),
            [None, Some(gun)] => (Some(gun), 0),
            _ => (None, 0),
        };
        let offer = weapon.and_then(|gun| super::reload::reserve_clip_for_pouch(world, gun));
        let state = if !enabled {
            PouchState::Inactive
        } else if refused {
            PouchState::Refused
        } else if weapon.is_none() || !available[taking] {
            PouchState::Inactive
        } else if offer.is_some() {
            PouchState::Ready
        } else {
            PouchState::Empty
        };
        Self {
            weapon: weapon.map(|id| id.inner() as i32),
            icon: offer.map_or_else(
                || {
                    if weapon.is_some_and(|gun| crate::wielded_weapon::is_energy_weapon(world, gun))
                    {
                        Some("i_batt".to_owned())
                    } else {
                        crate::hud::get_weapon_ammo_icon(world, weapon)
                    }
                },
                |offer| match offer.projectile {
                    Some(projectile) => crate::hud::get_projectile_ammo_icon(world, projectile),
                    None => crate::scripts::gui::inventory_icon(world, offer.reserve),
                },
            ),
            count: weapon.map(|_| {
                offer.map_or(0, |offer| {
                    super::reload::reserve_count(world, offer.projectile)
                })
            }),
            state,
            near: near[taking],
        }
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub(super) struct HolsterReadout {
    pub weapon: Option<i32>,
    pub ammo: Option<i32>,
    pub capacity: Option<i32>,
    pub segments: usize,
}

impl HolsterReadout {
    pub fn resolve(world: &World, item: Option<EntityId>) -> Self {
        let ammo = crate::hud::get_weapon_ammo(world, item);
        let capacity = item
            .and_then(|id| crate::scripts::script_util::active_gun_setting(world, id))
            .map(|setting| setting.clip)
            .filter(|capacity| *capacity > 0);
        Self {
            weapon: item.map(|id| id.inner() as i32),
            ammo,
            capacity,
            segments: match (ammo, capacity) {
                (Some(ammo), Some(capacity)) => lit_segments(ammo as f32 / capacity as f32),
                _ => 0,
            },
        }
    }
}

fn lit_segments(fraction: f32) -> usize {
    if fraction.is_finite() {
        (fraction.clamp(0.0, 1.0) * 4.0).ceil() as usize
    } else {
        0
    }
}

thread_local! {
    // Reuse one small annulus geometry; changing state changes only its color.
    // `ManuallyDrop`: never deleted at exit, after the GL context is gone.
    static RING: std::mem::ManuallyDrop<std::rc::Rc<Box<dyn scene::Geometry>>> = std::mem::ManuallyDrop::new({
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for i in 0..=48 {
            let angle = i as f32 * std::f32::consts::TAU / 48.0;
            for radius in [0.43, 0.5] {
                vertices.push(scene::VertexPosition { position: vec3(angle.cos()*radius, angle.sin()*radius, 0.0) });
            }
            if i < 48 { let a = i*2; indices.extend([a, a+2, a+1, a+1, a+2, a+3]); }
        }
        std::rc::Rc::new(Box::new(scene::indexed_mesh::create(vertices, indices)))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_holster_reads_its_own_gun_and_dual_wield_has_no_pouch_selection() {
        use dark::properties::{PropBaseGunDesc, PropGunState};
        let mut world = World::new();
        let guns = [(12, 12), (3, 6)].map(|(ammo, clip)| {
            let mut description = PropBaseGunDesc {
                settings: Default::default(),
            };
            for setting in &mut description.settings {
                setting.clip = clip;
            }
            world.add_entity((
                description,
                PropGunState {
                    ammo,
                    condition: 100.0,
                    setting: 0,
                    modification: 0,
                    silence_value: 0.0,
                },
            ))
        });
        let pistol = HolsterReadout::resolve(&world, Some(guns[0]));
        let shotgun = HolsterReadout::resolve(&world, Some(guns[1]));
        assert_eq!(
            (pistol.ammo, pistol.capacity, pistol.segments),
            (Some(12), Some(12), 4)
        );
        assert_eq!(
            (shotgun.ammo, shotgun.capacity, shotgun.segments),
            (Some(3), Some(6), 2)
        );
        let pouch =
            PouchReadout::resolve(&world, guns.map(Some), [false; 2], [true; 2], false, true);
        assert_eq!(pouch.weapon, None);
        assert_eq!(pouch.icon, None);
        assert_eq!(pouch.state, PouchState::Inactive);
        let disabled = PouchReadout::resolve(&world, [None; 2], [true; 2], [true; 2], true, false);
        assert_eq!(disabled.state, PouchState::Inactive);
    }

    #[test]
    fn segments_distinguish_empty_partial_and_full_and_reject_bad_values() {
        assert_eq!(lit_segments(0.0), 0);
        assert_eq!(lit_segments(0.01), 1);
        assert_eq!(lit_segments(0.5), 2);
        assert_eq!(lit_segments(1.0), 4);
        assert_eq!(lit_segments(2.0), 4);
        assert_eq!(lit_segments(f32::NAN), 0);
    }
}

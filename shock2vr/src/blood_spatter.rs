//! Wounds leave blood behind. When a bleeding creature is hit - the game
//! already marks that with a blood spang at the wound - some of it lands on
//! the wall behind and the floor below, and a creature that bled and then
//! died leaves a pool under it.
//!
//! Not in the original (Dark only drew the transient spang); added for the
//! horror look. The marks are the game's own blood decal archetypes, created
//! through `Effect::CreateEntity`, so they save with the level, take the
//! usual decal lift, and feed `blood_tracks` (walk through a fresh pool and
//! you track it onward).
use std::collections::{HashMap, HashSet};

use cgmath::{
    Deg, InnerSpace, Matrix4, Point3, Quaternion, Rotation3, SquareMatrix, Vector3, point3, vec3,
};
use dark::properties::{PropCreature, PropHitPoints, PropTemplateId};
use rand::Rng;
use shipyard::{EntityId, Get, IntoIter, IntoWithId, UniqueView, View, World};

use crate::{
    mission::PlayerInfo,
    mission::entity_creator::CreateEntityOptions,
    physics::{InternalCollisionGroups, PhysicsWorld},
    runtime_props::RuntimePropTransform,
    scripts::Effect,
    util::get_rotation_from_forward_vector,
};

/// "Standard Blood Spang" and "New Blood Spang": the wound effects.
const BLOOD_SPANGS: [i32; 2] = [-366, -1971];
const SPLATTER_SMALL: i32 = -864;
const SPLATTER_MEDIUM: i32 = -857;
const SPLATTER_FLOOR: i32 = -858;

/// How far behind a wound blood can reach a wall, and below it the floor.
const WALL_REACH: f32 = 2.8;
const FLOOR_REACH: f32 = 3.0;
/// The creature a wound belongs to is within this of the spang.
const WOUND_RADIUS: f32 = 1.6;
/// Marks per level, so a long firefight cannot grow the save without bound.
const MAX_MARKS: usize = 160;

#[derive(Default)]
pub(crate) struct BloodSpatter {
    seen: HashSet<EntityId>,
    is_spang: HashMap<i32, bool>,
    /// Creatures that have bled, and whether they have left their pool.
    bled: HashMap<EntityId, bool>,
    marks: usize,
}

impl BloodSpatter {
    pub(crate) fn update(
        &mut self,
        world: &World,
        physics: &PhysicsWorld,
        hierarchy: &HashMap<i32, Vec<i32>>,
    ) -> Vec<Effect> {
        let mut effects = Vec::new();
        let (Ok(templates), Ok(transforms), Ok(creatures), Ok(hit_points)) = (
            world.borrow::<View<PropTemplateId>>(),
            world.borrow::<View<RuntimePropTransform>>(),
            world.borrow::<View<PropCreature>>(),
            world.borrow::<View<PropHitPoints>>(),
        ) else {
            return effects;
        };
        let eye = world
            .borrow::<UniqueView<PlayerInfo>>()
            .map(|player| player.pos)
            .unwrap_or(vec3(0.0, 0.0, 0.0));

        let is_spang = &mut self.is_spang;
        let wounds: Vec<(EntityId, Vector3<f32>)> = (&templates, &transforms)
            .iter()
            .with_id()
            .filter(|(_, (template, _))| {
                let id = template.template_id;
                *is_spang.entry(id).or_insert_with(|| {
                    BLOOD_SPANGS.contains(&id)
                        || dark::ss2_entity_info::get_ancestors(hierarchy, &id)
                            .iter()
                            .any(|a| BLOOD_SPANGS.contains(a))
                })
            })
            .map(|(id, (_, transform))| (id, transform.0.w.truncate()))
            .collect();
        self.seen.retain(|id| wounds.iter().any(|(w, _)| w == id));

        let mut rng = rand::thread_rng();
        for (id, wound) in wounds {
            if !self.seen.insert(id) {
                continue;
            }
            // The creature it came from.
            if let Some(victim) = (&creatures, &transforms)
                .iter()
                .with_id()
                .map(|(c, (_, t))| (c, (t.0.w.truncate() - wound).magnitude()))
                .filter(|(_, d)| *d < WOUND_RADIUS)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(c, _)| c)
            {
                self.bled.entry(victim).or_insert(false);
            }
            if self.marks >= MAX_MARKS {
                continue;
            }
            // Spray carries on along the shot: away from the shooter's side.
            let mut away = wound - eye;
            away.y = 0.0;
            if away.magnitude2() > 1e-6 {
                let jitter = vec3(
                    rng.gen_range(-0.25..0.25),
                    rng.gen_range(-0.35..0.1),
                    rng.gen_range(-0.25..0.25),
                );
                let dir = (away.normalize() + jitter).normalize();
                if let Some((point, normal)) = surface(physics, wound, dir, WALL_REACH) {
                    let template = if rng.gen_bool(0.3) {
                        SPLATTER_MEDIUM
                    } else {
                        SPLATTER_SMALL
                    };
                    effects.push(decal(
                        template,
                        point,
                        normal,
                        rng.gen_range(0.28..0.55),
                        &mut rng,
                    ));
                    self.marks += 1;
                }
            }
            if rng.gen_bool(0.6) {
                let spread = vec3(rng.gen_range(-0.3..0.3), 0.0, rng.gen_range(-0.3..0.3));
                if let Some((point, normal)) =
                    surface(physics, wound + spread, vec3(0.0, -1.0, 0.0), FLOOR_REACH)
                {
                    effects.push(decal(
                        SPLATTER_SMALL,
                        point,
                        normal,
                        rng.gen_range(0.2..0.4),
                        &mut rng,
                    ));
                    self.marks += 1;
                }
            }
        }

        // A creature that bled and has now died leaves a pool under it.
        self.bled.retain(|id, _| creatures.contains(*id));
        for (id, pooled) in self.bled.iter_mut() {
            if *pooled || hit_points.get(*id).map_or(true, |hp| hp.hit_points > 0) {
                continue;
            }
            *pooled = true;
            if self.marks >= MAX_MARKS {
                continue;
            }
            let Ok(transform) = transforms.get(*id) else {
                continue;
            };
            if let Some((point, normal)) = surface(
                physics,
                transform.0.w.truncate(),
                vec3(0.0, -1.0, 0.0),
                FLOOR_REACH,
            ) {
                effects.push(decal(
                    SPLATTER_FLOOR,
                    point,
                    normal,
                    rng.gen_range(0.6..0.95),
                    &mut rng,
                ));
                self.marks += 1;
            }
        }
        effects
    }
}

/// The level surface `reach` along `dir` from `from`.
fn surface(
    physics: &PhysicsWorld,
    from: Vector3<f32>,
    dir: Vector3<f32>,
    reach: f32,
) -> Option<(Point3<f32>, Vector3<f32>)> {
    let hit = physics.ray_cast2(
        point3(from.x, from.y, from.z),
        dir,
        reach,
        InternalCollisionGroups::WORLD,
        None,
        true,
    )?;
    Some((hit.hit_point, hit.hit_normal.normalize()))
}

/// A blood decal on the surface at `point`: decals are flat along local X,
/// which points into the surface (the same frame a bullet hole rides).
fn decal(
    template_id: i32,
    point: Point3<f32>,
    normal: Vector3<f32>,
    scale: f32,
    rng: &mut impl Rng,
) -> Effect {
    Effect::CreateEntity {
        template_id,
        position: point,
        orientation: decal_orientation(normal, Deg(rng.gen_range(0.0..360.0))),
        root_transform: Matrix4::identity(),
        options: CreateEntityOptions {
            scale_override: Some(vec3(1.0, scale, scale)),
            ..CreateEntityOptions::default()
        },
    }
}

fn decal_orientation(normal: Vector3<f32>, roll: Deg<f32>) -> Quaternion<f32> {
    get_rotation_from_forward_vector(normal)
        * Quaternion::from_axis_angle(vec3(0.0, 1.0, 0.0), Deg(90.0))
        * Quaternion::from_axis_angle(vec3(1.0, 0.0, 0.0), roll)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::Rotation;

    #[test]
    fn decal_lies_flat_on_its_surface() {
        for normal in [
            vec3(0.0, 1.0, 0.0),
            vec3(1.0, 0.0, 0.0),
            vec3(0.0, 0.0, -1.0),
        ] {
            let q = decal_orientation(normal, Deg(37.0));
            let x = q.rotate_vector(vec3(1.0, 0.0, 0.0));
            assert!(
                (x + normal).magnitude() < 1e-4,
                "local X points into {normal:?}: {x:?}"
            );
        }
    }
}

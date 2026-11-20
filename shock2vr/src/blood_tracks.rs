//! Bloody footprints. Anything that walks through a pool of blood - the
//! player or any creature - tracks it onward: a trail of prints that fades
//! over the next several steps.
//!
//! Not in the original (Dark had no footprints); added for the horror look.
//! A pool is any floor-facing blood decal (the `Blood Splat*` family under
//! "Battle damage"). Prints are cosmetic: drawn as lit floor quads, capped,
//! and not saved.
use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    rc::Rc,
};

use cgmath::{InnerSpace, Matrix4, Point3, Transform, Vector3, point3, vec3};
use dark::{SCALE_FACTOR, properties::PropTemplateId};
use engine::scene::{Material, SceneObject, light::LightArray};
use shipyard::{EntityId, IntoIter, View, World};

use crate::{
    object_lighting::ObjectLighting,
    physics::{InternalCollisionGroups, PhysicsWorld},
    runtime_props::RuntimePropTransform,
};

/// Blood decal templates: Blood Splatter Small/Medium/on Floor, Blood Splat.
const BLOOD_DECALS: [i32; 4] = [-864, -857, -858, -601];

/// World units walked between prints (one foot to the other).
const STRIDE: f32 = 0.9;
/// Each print carries this share of the blood the one before it did.
const FADE_PER_STEP: f32 = 0.8;
/// Below this a foot has nothing left to leave.
const DRY: f32 = 0.08;
const PRINT_LENGTH: f32 = 0.42;
const PRINT_WIDTH: f32 = 0.17;
/// Each foot lands this far off the walker's line.
const FOOT_SPACING: f32 = 0.09;
/// Lift off the floor; the depth bias does the rest.
const LIFT: f32 = 0.006;
const MAX_PRINTS: usize = 400;
/// Rescan for pools this often (frames): decals are spawned during combat.
const RESCAN_FRAMES: u32 = 30;
/// Faster than this per frame is a teleport or level transition, not a step.
const MAX_STEP_PER_FRAME: f32 = 1.5;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Pool {
    center: Vector3<f32>,
    radius: f32,
}

#[derive(Clone, Copy, Debug)]
struct Walker {
    last: Vector3<f32>,
    travelled: f32,
    blood: f32,
    left: bool,
    heading: Vector3<f32>,
}

struct Print {
    transform: Matrix4<f32>,
    alpha: f32,
    left: bool,
    lights: Option<Rc<LightArray>>,
}

#[derive(Default)]
pub(crate) struct BloodTracks {
    pools: Vec<Pool>,
    rescan_in: u32,
    is_blood: HashMap<i32, bool>,
    walkers: HashMap<EntityId, Walker>,
    prints: VecDeque<Print>,
    /// Right and left foot materials, built on first draw.
    materials: Option<[Rc<RefCell<Box<dyn Material>>>; 2]>,
}

impl BloodTracks {
    /// `movers` are everything that can walk: (entity, position near its
    /// feet or body centre). The floor under each is found by ray.
    pub(crate) fn update(
        &mut self,
        world: &World,
        physics: &PhysicsWorld,
        hierarchy: &HashMap<i32, Vec<i32>>,
        movers: &[(EntityId, Vector3<f32>)],
    ) {
        if self.rescan_in == 0 {
            self.pools = self.scan_pools(world, hierarchy);
            self.rescan_in = RESCAN_FRAMES;
        }
        self.rescan_in -= 1;

        self.walkers
            .retain(|id, _| movers.iter().any(|(mover, _)| mover == id));
        for &(id, position) in movers {
            let walker = self.walkers.entry(id).or_insert(Walker {
                last: position,
                travelled: 0.0,
                blood: 0.0,
                left: false,
                heading: vec3(0.0, 0.0, -1.0),
            });
            let mut moved = position - walker.last;
            moved.y = 0.0;
            let distance = moved.magnitude();
            walker.last = position;
            if distance > MAX_STEP_PER_FRAME {
                walker.travelled = 0.0;
                continue;
            }
            if distance > 1e-4 {
                walker.heading = (walker.heading * 0.6 + moved / distance * 0.4).normalize();
            }
            if touching_pool(&self.pools, physics, position) {
                walker.blood = 1.0;
            }
            walker.travelled += distance;
            if walker.travelled < STRIDE {
                continue;
            }
            walker.travelled -= STRIDE;
            walker.left = !walker.left;
            if walker.blood < DRY {
                walker.blood = 0.0;
                continue;
            }
            let side = walker.heading.cross(vec3(0.0, 1.0, 0.0)).normalize();
            let foot = position
                + side
                    * if walker.left {
                        -FOOT_SPACING
                    } else {
                        FOOT_SPACING
                    };
            if let Some((point, normal)) = floor_under(physics, foot) {
                self.prints.push_back(Print {
                    transform: print_transform(point, normal, walker.heading),
                    alpha: walker.blood,
                    left: walker.left,
                    lights: None,
                });
                if self.prints.len() > MAX_PRINTS {
                    self.prints.pop_front();
                }
            }
            walker.blood *= FADE_PER_STEP;
        }
    }

    fn scan_pools(&mut self, world: &World, hierarchy: &HashMap<i32, Vec<i32>>) -> Vec<Pool> {
        let (Ok(templates), Ok(transforms)) = (
            world.borrow::<View<PropTemplateId>>(),
            world.borrow::<View<RuntimePropTransform>>(),
        ) else {
            return Vec::new();
        };
        let is_blood = &mut self.is_blood;
        (&templates, &transforms)
            .iter()
            .filter(|(template, _)| {
                let id = template.template_id;
                *is_blood.entry(id).or_insert_with(|| {
                    BLOOD_DECALS.contains(&id)
                        || dark::ss2_entity_info::get_ancestors(hierarchy, &id)
                            .iter()
                            .any(|a| BLOOD_DECALS.contains(a))
                })
            })
            .filter_map(|(_, transform)| floor_pool(&transform.0))
            .collect()
    }

    pub(crate) fn render(&mut self, lighting: Option<&ObjectLighting>) -> Vec<SceneObject> {
        if self.prints.is_empty() {
            return Vec::new();
        }
        let materials = self.materials.get_or_insert_with(|| {
            [false, true].map(|left| {
                let texture: Rc<dyn engine::texture::TextureTrait> =
                    Rc::new(engine::texture::init_from_memory(print_texture(left)));
                Rc::new(RefCell::new(engine::scene::basic_material::create(
                    texture, 0.0, 0.05,
                )))
            })
        });
        let geometry: Rc<Box<dyn engine::scene::Geometry>> =
            Rc::new(Box::new(engine::scene::quad::create()));
        self.prints
            .iter_mut()
            .map(|print| {
                if print.lights.is_none() {
                    // Lamps don't move: look the print's light up once.
                    print.lights = lighting.map(|l| l.at_position(print.transform.w.truncate()));
                }
                let mut object = SceneObject::create_with_shared_material(
                    materials[usize::from(print.left)].clone(),
                    geometry.clone(),
                );
                object.set_transform(print.transform);
                object.set_lights(print.lights.clone());
                object.set_depth_bias(true);
                object.set_transparency(Some((1.0 - print.alpha * 0.92).max(0.05)));
                object
            })
            .collect()
    }
}

/// A blood decal lying on a floor, as a pool: decals are flat along their
/// local X, 6x6 Dark units across in Y and Z before scaling.
fn floor_pool(transform: &Matrix4<f32>) -> Option<Pool> {
    let normal = transform.transform_vector(vec3(1.0, 0.0, 0.0));
    if normal.magnitude2() < 1e-12 || normal.normalize().y.abs() < 0.7 {
        return None;
    }
    let half = 3.0 / SCALE_FACTOR;
    let a = transform.transform_vector(vec3(0.0, half, 0.0)).magnitude();
    let b = transform.transform_vector(vec3(0.0, 0.0, half)).magnitude();
    Some(Pool {
        center: transform.w.truncate(),
        // A splatter does not fill its square.
        radius: (a + b) * 0.5 * 0.7,
    })
}

/// Whether `position` stands in a pool (on its floor, not above or below).
fn touching_pool(pools: &[Pool], physics: &PhysicsWorld, position: Vector3<f32>) -> bool {
    let pool = pools.iter().find(|pool| {
        let d = vec3(position.x - pool.center.x, 0.0, position.z - pool.center.z);
        d.magnitude2() < pool.radius * pool.radius && (position.y - pool.center.y).abs() < 2.5
    });
    pool.and_then(|pool| {
        floor_under(physics, position).map(|(floor, _)| (floor.y - pool.center.y).abs() < 0.35)
    })
    .unwrap_or(false)
}

fn floor_under(
    physics: &PhysicsWorld,
    position: Vector3<f32>,
) -> Option<(Point3<f32>, Vector3<f32>)> {
    const LIFT_PROBE: f32 = 0.3;
    let hit = physics.ray_cast2(
        point3(position.x, position.y + LIFT_PROBE, position.z),
        vec3(0.0, -1.0, 0.0),
        LIFT_PROBE + 3.0,
        InternalCollisionGroups::WORLD,
        None,
        true,
    )?;
    (hit.hit_normal.y > 0.6).then_some((hit.hit_point, hit.hit_normal.normalize()))
}

/// The unit quad (XY plane, facing +Z) laid on the floor: its Y along the
/// walking direction, its face up the floor's normal.
fn print_transform(point: Point3<f32>, up: Vector3<f32>, heading: Vector3<f32>) -> Matrix4<f32> {
    let forward = (heading - up * heading.dot(up)).normalize();
    let right = forward.cross(up).normalize();
    let origin = vec3(point.x, point.y, point.z) + up * LIFT;
    Matrix4::from_cols(
        (right * PRINT_WIDTH).extend(0.0),
        (forward * PRINT_LENGTH).extend(0.0),
        up.extend(0.0),
        origin.extend(1.0),
    )
}

/// A shoe print: a broad forefoot and a heel, mottled so it reads as smeared
/// blood rather than a stamp. `left` mirrors it.
const PRINT_TEXTURE_SIZE: (usize, usize) = (48, 96);

fn print_texture(left: bool) -> engine::texture_format::RawTextureData {
    const W: usize = PRINT_TEXTURE_SIZE.0;
    const H: usize = PRINT_TEXTURE_SIZE.1;
    let mut bytes = vec![0u8; W * H * 4];
    for y in 0..H {
        for x in 0..W {
            let u = (x as f32 + 0.5) / W as f32;
            let u = if left { 1.0 - u } else { u };
            let v = (y as f32 + 0.5) / H as f32;
            // A boot sole: broad forefoot (widest toward the big toe, at
            // low u), a heel, and the outer edge joining them - the arch
            // on the inner side leaves no mark.
            let fore = ellipse(u, v, 0.5, 0.68, 0.44, 0.3);
            let heel = ellipse(u, v, 0.53, 0.2, 0.36, 0.18);
            let outer = ellipse(u, v, 0.66, 0.44, 0.24, 0.3);
            let shape = fore.max(heel).max(outer);
            // Tread: bars across the sole, broken up by smearing.
            let tread = if (v * 16.0).fract() < 0.55 { 1.0 } else { 0.45 };
            let mottle = 0.6 + 0.4 * noise(x as u32 / 3, y as u32 / 3);
            let a = (shape * tread * mottle * 255.0) as u8;
            let i = (x + y * W) * 4;
            bytes[i..i + 4].copy_from_slice(&[120, 8, 6, a]);
        }
    }
    engine::texture_format::RawTextureData {
        bytes,
        width: W as u32,
        height: H as u32,
        format: engine::texture_format::PixelFormat::RGBA,
    }
}

/// 1 inside, fading to 0 across a soft rim.
fn ellipse(u: f32, v: f32, cu: f32, cv: f32, ru: f32, rv: f32) -> f32 {
    let d = ((u - cu) / ru).powi(2) + ((v - cv) / rv).powi(2);
    (1.0 - (d - 0.7) / 0.3).clamp(0.0, 1.0)
}

fn noise(x: u32, y: u32) -> f32 {
    let mut h = x.wrapping_mul(374_761_393) ^ y.wrapping_mul(668_265_263);
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    (h >> 8) as f32 / (1u32 << 24) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::SquareMatrix;

    #[test]
    fn only_floor_decals_are_pools() {
        // Decals are flat along local X: rotated so X points up, it lies flat.
        let floor = Matrix4::from_angle_z(cgmath::Deg(90.0));
        let pool = floor_pool(&floor).expect("a flat decal is a pool");
        assert!(pool.radius > 0.5 && pool.radius < 1.5, "{pool:?}");
        // Unrotated, X is horizontal: a decal on a wall.
        assert!(floor_pool(&Matrix4::from_scale(1.0)).is_none());
        // A mirrored scale (medsci1 authors these) still counts.
        let mirrored = floor * Matrix4::from_nonuniform_scale(-1.0, 0.7, 1.0);
        assert!(floor_pool(&mirrored).is_some());
    }

    #[test]
    fn prints_lie_flat_along_the_walk() {
        let m = print_transform(
            point3(1.0, 2.0, 3.0),
            vec3(0.0, 1.0, 0.0),
            vec3(0.0, 0.0, -1.0),
        );
        let face = m.transform_vector(vec3(0.0, 0.0, 1.0)).normalize();
        let along = m.transform_vector(vec3(0.0, 1.0, 0.0)).normalize();
        assert!((face.y - 1.0).abs() < 1e-5, "faces up");
        assert!((along.z + 1.0).abs() < 1e-5, "toe points where it walked");
        assert!(m.determinant() > 0.0, "keeps the quad's winding");
    }

    #[test]
    fn the_print_texture_has_a_shoe_shape() {
        let print = print_texture(false);
        let (w, h) = PRINT_TEXTURE_SIZE;
        // Alpha at a (u, v) in the sole, v = 0 heel to 1 toe.
        let alpha = |u: f32, v: f32| {
            let (x, y) = ((u * w as f32) as usize, (v * h as f32) as usize);
            print.bytes[(x + y * w) * 4 + 3]
        };
        assert!(alpha(0.5, 0.68) > 60, "forefoot");
        assert!(alpha(0.53, 0.2) > 60, "heel");
        assert!(alpha(0.7, 0.44) > 30, "outer edge joins them");
        assert_eq!(alpha(0.15, 0.42), 0, "inner arch is clear");
        assert_eq!(alpha(0.5, 0.995), 0, "nothing past the toe");
    }
}

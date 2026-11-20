//! A ladder's hand holds, read off its render mesh: the rungs (horizontal
//! bars, or a pole ladder's side pegs) and the rails (vertical members, or the
//! pole). Its physics body is a single box, so the mesh is the only place the
//! rungs exist. A VR hand holds a ladder only on these (see
//! [`crate::physics::PhysicsWorld::set_ladder_holds`]); debug tooling reads
//! them too, to put a test hand on a real rung.

use cgmath::{InnerSpace, Matrix4, Transform, Vector3, point3, vec3};
use dark::properties::PropModelName;
use engine::assets::{
    asset_cache::AssetCache, asset_importer::AssetImporter, asset_paths::ReadableAndSeekable,
};
use once_cell::sync::Lazy;
use shipyard::{EntityId, Get, View, World};

use crate::runtime_props::RuntimePropTransform;

/// A straight member of a ladder, as its two end points.
pub type Segment = [Vector3<f32>; 2];

#[derive(Debug, Clone, PartialEq, Default)]
pub struct LadderHolds {
    /// Rungs bottom to top, each from its low-x end to its high-x end.
    pub rungs: Vec<Segment>,
    /// Rails by x, each from bottom to top.
    pub rails: Vec<Segment>,
}

/// The model member selected by a snapped grip, kept distinct from the collider face.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemberKind {
    Rung,
    Rail,
}

#[derive(Debug, Clone, Copy)]
pub struct LadderMember {
    pub kind: MemberKind,
    pub axis: Vector3<f32>,
    pub face_normal: Vector3<f32>,
}

/// Faces whose extents along a member's key axis (y for rungs, x for rails)
/// overlap or come this close fold into one member: a rung's cylinder faces,
/// a rail's segments.
const MERGE_DISTANCE: f32 = 0.1;

/// A member thicker than this across (about 19 cm) is no bar a hand can
/// close around - an organic strand, a panel - so its mesh is not a bar
/// ladder and offers no holds.
const MAX_BAR_THICKNESS: f32 = 0.25;

type Bounds = (Vector3<f32>, Vector3<f32>);

fn grow((lo, hi): Bounds, v: Vector3<f32>) -> Bounds {
    (
        vec3(lo.x.min(v.x), lo.y.min(v.y), lo.z.min(v.z)),
        vec3(hi.x.max(v.x), hi.y.max(v.y), hi.z.max(v.z)),
    )
}

const EMPTY: Bounds = (
    vec3(f32::MAX, f32::MAX, f32::MAX),
    vec3(f32::MIN, f32::MIN, f32::MIN),
);

/// Sweep faces in key order into members, so the result does not depend on
/// the mesh's polygon order.
fn merge(mut faces: Vec<Bounds>, key: fn(Vector3<f32>) -> f32) -> Vec<Bounds> {
    faces.sort_by(|a, b| key(a.0).total_cmp(&key(b.0)));
    let mut members: Vec<Bounds> = vec![];
    for (lo, hi) in faces {
        match members.last_mut() {
            Some(member) if key(lo) <= key(member.1) + MERGE_DISTANCE => {
                *member = grow(grow(*member, lo), hi);
            }
            _ => members.push((lo, hi)),
        }
    }
    members
}

/// Classify each polygon by its bounds: a rung face is long in x (at least a
/// quarter of the ladder's width) and flat in y; a rail face is long in y and
/// narrow in x (a slab's side faces count, so a slab offers its edges). Faces
/// that are neither (caps, feet) are ignored, and so is a "rung" with no
/// thickness (a slab's top face). A mesh with a member too thick to grip
/// offers none.
pub fn ladder_holds(vertices: &[Vector3<f32>], polygons: &[Vec<u16>]) -> LadderHolds {
    let bounds = |indices: &[u16]| {
        indices
            .iter()
            .filter_map(|&i| vertices.get(i as usize))
            .fold(EMPTY, |b, &v| grow(b, v))
    };
    let (lo, hi) = vertices.iter().fold(EMPTY, |b, &v| grow(b, v));
    let width = hi.x - lo.x;

    let (mut rung_faces, mut rail_faces) = (vec![], vec![]);
    for polygon in polygons {
        let (lo, hi) = bounds(polygon);
        let (dx, dy) = (hi.x - lo.x, hi.y - lo.y);
        if dx >= 0.25 * width && dy < dx / 4.0 {
            rung_faces.push((lo, hi));
        } else if dy > 2.0 * dx && dy >= 0.2 && dx <= 0.25 * width {
            rail_faces.push((lo, hi));
        }
    }
    let rungs = merge(rung_faces, |v| v.y);
    let rails = merge(rail_faces, |v| v.x);

    let rungs: Vec<Bounds> = rungs
        .into_iter()
        .filter(|(a, b)| b.y - a.y > 0.01)
        .collect();
    let too_thick = |(a, b): &Bounds, across: fn(Vector3<f32>) -> f32| {
        across(b - a).max(b.z - a.z) > MAX_BAR_THICKNESS
    };
    if rungs.iter().any(|r| too_thick(r, |v| v.y)) || rails.iter().any(|r| too_thick(r, |v| v.x)) {
        return LadderHolds::default();
    }

    // A rung is its bar's centre line; a rail its member's centre line.
    let mut rungs: Vec<Segment> = rungs
        .into_iter()
        .map(|(a, b)| {
            let (y, z) = ((a.y + b.y) / 2.0, (a.z + b.z) / 2.0);
            [vec3(a.x, y, z), vec3(b.x, y, z)]
        })
        .collect();
    let mut rails: Vec<Segment> = rails
        .into_iter()
        .map(|(a, b)| {
            let (x, z) = ((a.x + b.x) / 2.0, (a.z + b.z) / 2.0);
            [vec3(x, a.y, z), vec3(x, b.y, z)]
        })
        .collect();
    rungs.sort_by(|a, b| a[0].y.total_cmp(&b[0].y));
    rails.sort_by(|a, b| a[0].x.total_cmp(&b[0].x));
    LadderHolds { rungs, rails }
}

impl LadderHolds {
    pub fn is_empty(&self) -> bool {
        self.rungs.is_empty() && self.rails.is_empty()
    }

    /// The point on the nearest rung or rail to `hand`, if one is within
    /// `reach` of it.
    pub fn nearest_hold(&self, hand: Vector3<f32>, reach: f32) -> Option<Vector3<f32>> {
        self.rungs
            .iter()
            .chain(&self.rails)
            .map(|&[a, b]| crate::pathfinding::closest_point_on_segment(a, b, hand))
            .map(|p| (p, (p - hand).magnitude()))
            .filter(|&(_, distance)| distance <= reach)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(p, _)| p)
    }

    /// Recover the held member from its snapped point, using the same ordering
    /// and distance policy as acquisition. Rungs win an exact rung/rail tie.
    pub fn member_at(&self, point: Vector3<f32>) -> Option<LadderMember> {
        let normal = self
            .rungs
            .first()
            .zip(self.rails.first())
            .and_then(|(rung, rail)| {
                let normal = (rung[1] - rung[0]).cross(rail[1] - rail[0]);
                (normal.magnitude2() > 1e-8).then(|| normal.normalize())
            })?;
        self.rungs
            .iter()
            .map(|s| (MemberKind::Rung, s))
            .chain(self.rails.iter().map(|s| (MemberKind::Rail, s)))
            .map(|(kind, &[a, b])| {
                (
                    kind,
                    b - a,
                    (crate::pathfinding::closest_point_on_segment(a, b, point) - point)
                        .magnitude2(),
                )
            })
            .filter(|(_, axis, distance)| axis.magnitude2() > 1e-8 && *distance < 1e-6)
            .min_by(|a, b| a.2.total_cmp(&b.2))
            .map(|(kind, axis, _)| LadderMember {
                kind,
                axis: axis.normalize(),
                face_normal: normal,
            })
    }

    /// These holds placed in the world by the ladder entity's transform.
    pub fn transformed(&self, transform: &Matrix4<f32>) -> LadderHolds {
        let place = |segments: &[Segment]| {
            segments
                .iter()
                .map(|s| {
                    s.map(|v| {
                        let p = transform.transform_point(point3(v.x, v.y, v.z));
                        vec3(p.x, p.y, p.z)
                    })
                })
                .collect()
        };
        LadderHolds {
            rungs: place(&self.rungs),
            rails: place(&self.rails),
        }
    }
}

/// The ladder face's normal in the world: the model's +z, which is its thin
/// axis. Either side may be the climbing side.
pub fn face_normal(transform: &Matrix4<f32>) -> Vector3<f32> {
    transform.transform_vector(Vector3::unit_z()).normalize()
}

/// A `.bin` model's holds in model space, cached per model by the asset
/// cache. Empty for a model that is not an object model.
static LADDER_HOLDS_IMPORTER: Lazy<AssetImporter<LadderHolds, LadderHolds, ()>> = Lazy::new(|| {
    AssetImporter::define(
        |_name, reader: &mut Box<dyn ReadableAndSeekable>, _assets, _config| {
            let header = dark::ss2_bin_header::read(reader);
            if !matches!(header.bin_type, dark::ss2_bin_header::BinFileType::Obj) {
                return LadderHolds::default();
            }
            // Raw vertices, no sub-object palette: every ladder model is one part.
            let mesh = dark::ss2_bin_obj_loader::read(reader, &header);
            let polygons: Vec<Vec<u16>> = mesh
                .polygons
                .iter()
                .map(|p| p.vertex_indices.clone())
                .collect();
            ladder_holds(&mesh.vertices, &polygons)
        },
        |holds, _assets, _config| holds,
    )
});

/// `entity`'s model name, its holds placed in the world, and its face normal;
/// `None` when it has no model, no transform, or its `.bin` cannot be opened.
pub fn entity_holds(
    asset_cache: &mut AssetCache,
    world: &World,
    entity: EntityId,
) -> Option<(String, LadderHolds, Vector3<f32>)> {
    let (model, transform) = world.run(
        |names: View<PropModelName>, transforms: View<RuntimePropTransform>| {
            Some((
                names.get(entity).ok()?.0.clone(),
                transforms.get(entity).ok()?.0,
            ))
        },
    )?;
    let holds = asset_cache.get_opt(&LADDER_HOLDS_IMPORTER, &format!("{model}.bin"))?;
    Some((
        model,
        holds.transformed(&transform),
        face_normal(&transform),
    ))
}

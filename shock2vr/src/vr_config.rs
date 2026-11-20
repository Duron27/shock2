use std::collections::HashMap;

use cgmath::{Quaternion, Rotation3, Vector3, vec3};
use dark::properties::PropModelName;

use once_cell::sync::Lazy;
use shipyard::{EntityId, Get, View, World};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Handedness {
    Left,
    Right,
}

impl Handedness {
    pub fn other(self) -> Handedness {
        match self {
            Handedness::Left => Handedness::Right,
            Handedness::Right => Handedness::Left,
        }
    }

    /// The reflection that turns right-handed geometry into this hand's.
    ///
    /// Every first-person asset the player wears or wields is authored for the
    /// **right** hand - the glove model, the melee `_h` arm rigs - so the left
    /// hand draws the mirror image, reflected across the hand frame's own X.
    /// This is the single definition of what "the other hand" means, shared by
    /// [`crate::hand_glove`] and the melee wield, so the arm and the glove
    /// cannot disagree about which way round the left hand is.
    ///
    /// Held guns reflect across a different plane of their own frame - see
    /// [`Self::gun_mirror`] - because their authored axes differ from the arm
    /// rigs'.
    ///
    /// A reflection has a negative determinant, which reverses the triangle
    /// winding the GPU sees. Geometry that culls backfaces must flip its
    /// front-face winding to match - [`dark::model::Model::apply_local_transform`]
    /// does that for meshes seated by a local transform. The glove needs
    /// nothing because GLB meshes are drawn double-sided.
    pub fn mirror(self) -> cgmath::Matrix4<f32> {
        use cgmath::{Matrix4, SquareMatrix};

        match self {
            Handedness::Right => Matrix4::identity(),
            Handedness::Left => Matrix4::from_nonuniform_scale(-1.0, 1.0, 1.0),
        }
    }

    /// [`Self::mirror`] applied to a hand-local point. Derived from the matrix
    /// rather than spelled out again, so the two cannot drift apart.
    pub fn mirror_point(self, point: Vector3<f32>) -> Vector3<f32> {
        use cgmath::Transform;

        self.mirror().transform_vector(point)
    }

    /// The reflection that turns a right-handed 25AE gun view model into this
    /// hand's, in the model's own frame.
    ///
    /// The `_h` guns bake a right hand onto the grip, so a left-hand wield
    /// draws the mirror image. Their authored frame runs the barrel along -X
    /// with +Y up, so the mirror plane is the one holding the barrel and the
    /// sights - it negates Z, "across the gun" - and the barrel keeps pointing
    /// the same way. The grip's thumb-side offset
    /// ([`VRHandModelPerHandAdjustments::flip_x`]) and the muzzle vhots
    /// reflect with it; the negative determinant is what
    /// [`dark::model::Model::apply_local_transform`] flips the winding for.
    pub fn gun_mirror(self) -> cgmath::Matrix4<f32> {
        use cgmath::{Matrix4, SquareMatrix};

        match self {
            Handedness::Right => Matrix4::identity(),
            Handedness::Left => Matrix4::from_nonuniform_scale(1.0, 1.0, -1.0),
        }
    }
}

/// Model-axis scale for held VR guns. The AR's local Z runs across its
/// receiver: reflecting only that axis exposes its bolt to the holding eye.
/// Uniform grip scale still controls size; flat and world models do not use this.
pub fn gun_model_scale(model: &str) -> Vector3<f32> {
    if model
        .trim_end_matches(".bin")
        .eq_ignore_ascii_case("ar15_h")
    {
        vec3(1.0, 1.0, -1.0)
    } else {
        vec3(1.0, 1.0, 1.0)
    }
}

/// Shared VR mesh/attachment frame, including the other hand's reflection.
pub fn gun_model_frame(model: &str, hand: Handedness) -> cgmath::Matrix4<f32> {
    let scale = gun_model_scale(model);
    hand.gun_mirror() * cgmath::Matrix4::from_nonuniform_scale(scale.x, scale.y, scale.z)
}

/// The index a hand occupies in a `[_; 2]` of per-hand state.
///
/// One conversion, crate-wide, so per-hand arrays built in one module can be
/// read in another. Anything that pairs the hands in an array (latches,
/// per-hand gestures) uses THIS ordering; a local array in some other order
/// must not be indexed with it.
pub fn hand_slot(hand: Handedness) -> usize {
    match hand {
        Handedness::Left => 0,
        Handedness::Right => 1,
    }
}

/// The 25th Anniversary Edition's remastered first-person gun models
/// (`obj/*_h.bin`, LGMD), shipped in `mods/sshock2ee.kpf` which outranks every
/// classic archive - so on a 25AE install these names always resolve to the
/// remastered meshes. mesh_audit measures them 0-10.5% open edges versus
/// 19-38% for the classic `_h` set, and they are authored barrel-along -X
/// with real muzzle vhots, so VR can wield them from any viewpoint.
///
/// Melee `_h` models (wrench/rapier/shard/psword) are LGMM skinned meshes
/// (classic geometry + remastered `PMNM` chunk); VR wields them posed to the
/// player-melee idle clip (see `Effect::ChangeModel` in mission_core). The
/// 25AE-only `pipewrench_h` has no gamesys template (it ships as a KEX squirrel
/// script we don't run), so it is not listed.
const VR_25AE_VIEW_MODELS: &[&str] = &[
    "atek_h", "ar15_h", "sg_h", "lasehand", "empgun_h", "gren_h", "sfg_h", "fsn_h", "al_h",
    "viro_h", "amp_h", "wrench_h", "rapier_h", "shard_h", "psword_h",
];

// --- Magazine anchors ---------------------------------------------------------
//
// Where a held clip has to reach to load a wielded gun, in the *weapon's own
// authored model frame* - NOT the hand-local space the grip offsets above use.
// The 25AE guns are authored barrel along -X, so +X is back toward the
// shooter, +Y up out of the gun body, +Z across it. World units (1 unit =
// 0.762 m), the space `RuntimePropTransform` lives in.

/// Per-model magazine anchors, keyed like [`HAND_MODEL_POSITIONING`]
/// (lowercased `PropModelName`). Where the art has a clip part (`ar15_h`'s
/// `@s02_cli`, `fsn_h`'s `@01_core`, found with `cargo dv <model>.bin
/// --debug-subobjects`) the anchor is its centre; the rest sit on the grip or
/// loading port. Every value is placed against the rendered wield with the
/// `clip_zone` dev param (which also marks the model origin), not from the
/// dump alone - the pistol's origin, for one, renders at the wrist, well
/// behind where its sub-object bounds suggest.
///
/// A model with no entry keeps the zone on its own origin: a classic install's
/// world models; the energy weapons (`lasehand`, `empgun_h`), which recharge
/// rather than take a clip; `sfg_h`, whose dump shows a stray part 5 m off the
/// model and is not trusted yet; and `al_h` / `viro_h`, not measured yet.
static MAGAZINE_ANCHORS: Lazy<HashMap<&str, Vector3<f32>>> = Lazy::new(|| {
    HashMap::from([
        ("atek_h", vec3(-0.20, -0.15, 0.0)),
        ("ar15_h", vec3(-0.04, -0.20, 0.0)),
        ("sg_h", vec3(0.08, -0.10, 0.0)),
        ("gren_h", vec3(0.10, -0.05, 0.0)),
        ("fsn_h", vec3(-0.65, 0.0, 0.0)),
    ])
});

/// Where `entity_id`'s magazine is, in its model's local frame - the origin
/// for a model with no entry.
pub fn magazine_anchor_from_entity(world: &World, entity_id: EntityId) -> Vector3<f32> {
    model_name_lower(world, entity_id)
        .map(|name| magazine_anchor_from_model(&name))
        .unwrap_or(vec3(0.0, 0.0, 0.0))
}

/// [`magazine_anchor_from_entity`] by model name (any case).
fn magazine_anchor_from_model(model_name: &str) -> Vector3<f32> {
    MAGAZINE_ANCHORS
        .get(model_name.to_ascii_lowercase().as_str())
        .copied()
        .unwrap_or(vec3(0.0, 0.0, 0.0))
}

/// `entity_id`'s `PropModelName`, lowercased - the key every per-model table
/// in this file is looked up by.
fn model_name_lower(world: &World, entity_id: EntityId) -> Option<String> {
    world
        .borrow::<View<PropModelName>>()
        .ok()
        .and_then(|names| {
            names
                .get(entity_id)
                .ok()
                .map(|name| name.0.to_ascii_lowercase())
        })
}

/// Whether `model_name` is a first-person view model VR should wield in place
/// of the world model (only meaningful on a 25AE install, where the remastered
/// copy is what resolves).
pub fn is_vr_view_model(model_name: &str) -> bool {
    let name = model_name.to_ascii_lowercase();
    VR_25AE_VIEW_MODELS.contains(&name.as_str())
}

/// Skeleton joints of the melee `_h` rigs: root, shoulder, **elbow**, **fist**,
/// **weapon** (see `cargo run -p shock2vr --example melee_grip`).
const MELEE_ARM_JOINT: usize = 2;
const MELEE_GRIP_JOINT: usize = 3;
const MELEE_WEAPON_JOINT: usize = 4;

/// The posed arm's joints, in model space, that the wield is built from:
/// elbow, fist and the weapon joint at the business end of the blade/head.
#[derive(Clone, Copy, Debug)]
pub struct MeleePosedArm {
    pub elbow: Vector3<f32>,
    pub fist: Vector3<f32>,
    pub weapon: Vector3<f32>,
}

impl MeleePosedArm {
    /// Read the three joints out of a posed melee `_h` skeleton
    /// (`AnimationPlayer::get_transforms`). `None` for a rig that does not
    /// have them - a patched or modded mesh must leave the wield unseated,
    /// not panic the frame.
    pub fn from_joints(joints: &[cgmath::Matrix4<f32>]) -> Option<MeleePosedArm> {
        Some(MeleePosedArm {
            elbow: joints.get(MELEE_ARM_JOINT)?.w.truncate(),
            fist: joints.get(MELEE_GRIP_JOINT)?.w.truncate(),
            weapon: joints.get(MELEE_WEAPON_JOINT)?.w.truncate(),
        })
    }
}

/// Rotation that takes the posed arm out of model space and into the hand's
/// own frame: the forearm runs back along the hand frame's **+Z** - the axis
/// opposite of the -Z the fingers/aim point down - and the weapon is rolled
/// into the hand's vertical plane so it stands up out of the fist.
///
/// Derived from the rig rather than hand-tuned: a single authored yaw can only
/// satisfy one of the two, and the shipped 120 degrees satisfied the weapon,
/// leaving the forearm 71-80 degrees off the hand's own arm axis (it stuck out
/// sideways past the wrist instead of running back toward the elbow).
fn melee_wield_alignment(arm: MeleePosedArm) -> Quaternion<f32> {
    use cgmath::{Rad, Rotation};

    let forearm = arm.elbow - arm.fist;
    let weapon = arm.weapon - arm.fist;
    let align = Quaternion::from_arc(forearm, Vector3::unit_z(), None);
    let aligned_weapon = align.rotate_vector(weapon);
    // Roll about the forearm until the weapon lies in the hand's YZ plane,
    // pointing along +Y (up out of the fist) rather than across the palm.
    let roll = Quaternion::from_angle_z(Rad(aligned_weapon.x.atan2(aligned_weapon.y)));
    roll * align
}

/// Where the weapon joint ends up, in hand-local space, once the arm is seated
/// by [`melee_wield_alignment`] - i.e. where the *rendered* head of the weapon
/// is, and therefore where the held entity's body (the melee contact collider)
/// has to sit for the drawn weapon and the damage volume to be the same place.
/// The wield stores it as `RuntimePropVrGripOffset`;
/// `cargo run -p shock2vr --example melee_grip` prints it per model.
///
/// The rigid-body origin remains on this weapon joint. `Effect::ChangeModel`
/// separately fits the body's collider around the rendered weapon vertices,
/// including the handle/blade extending back from the joint.
/// Hand-agnostic: [`melee_wield_alignment`] rolls the weapon into the hand's
/// YZ plane, so this lands on the hand's own centreline (x = 0) and the
/// left-hand mirror - a reflection across exactly that axis - leaves it where
/// it is. The damage volume is in the same place for either hand; only the
/// geometry around it swaps sides. Pinned by
/// `the_contact_point_is_on_the_hands_centreline`, because the left-hand
/// wield's "baked fist lands on the tracked hand" depends on it.
pub fn melee_contact_offset(arm: MeleePosedArm) -> Vector3<f32> {
    melee_contact_offset_scaled(arm, melee_wield_scale())
}

pub(crate) fn melee_contact_offset_scaled(arm: MeleePosedArm, scale: f32) -> Vector3<f32> {
    use cgmath::Rotation;

    scale * melee_wield_alignment(arm).rotate_vector(arm.weapon - arm.fist)
}

/// Live tuning for how large a wielded melee view model is drawn - see
/// [`crate::dev_params::MELEE_WIELD_SCALE`]. Both halves of the placement read
/// it, so the grip and the contact collider scale together and the
/// "rendered weapon head lands on the collider" invariant is unaffected.
fn melee_wield_scale() -> f32 {
    crate::dev_params::get(crate::dev_params::MELEE_WIELD_SCALE)
}

/// The held amp mesh's live fit scale, composed after its rendered transform.
pub(crate) fn psi_amp_fit_scale() -> cgmath::Matrix4<f32> {
    cgmath::Matrix4::from_scale(crate::dev_params::get(crate::dev_params::PSI_AMP_SCALE))
}

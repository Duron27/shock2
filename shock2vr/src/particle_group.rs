//! How an authored Dark particle group (`P$ParticleGroup`) maps onto the
//! engine particle system, where the mapping is more than a unit conversion.

use cgmath::{InnerSpace, Matrix4, Vector3};
use dark::properties::{PropParticleGroup, PropParticleLaunchInfo};

use crate::palette;

const PRT_SINGLE_COLOR_DISK: u32 = 4;
const PGMT_OBJECT_TRAILING: u32 = 3;
const PAT_LAUNCH_CONTINUOUS: u32 = 1;

/// Trailing groups launch into world space, so particles stay where they
/// were emitted and streak behind a moving host. Every other motion type
/// rides its host. Dark derives this from the motion type and ignores the
/// authored world-space flag; a trailing group with a fixed (non-launch)
/// animation falls back to riding its host.
pub fn is_world_space(pg: &PropParticleGroup) -> bool {
    pg.motion_type == PGMT_OBJECT_TRAILING && pg.animation_type <= PAT_LAUNCH_CONTINUOUS
}

/// The group's spin about its vertical axis, in radians per second. Dark
/// authors it in degrees per second (the training booths' teleport swirl
/// spins at 250).
pub fn swirl_rate(pg: &PropParticleGroup) -> f32 {
    pg.spin.y.to_radians()
}

/// The emitter frame a group launches in: its host's position and
/// orientation, without scale. Particle groups size themselves (`size`,
/// launch boxes and radii); the host's `P$Scale` never applies, and Earth's
/// teleport swirls carry a garbage one (~1e-36) that would otherwise
/// collapse every particle onto one point.
pub fn emitter_transform(host: Matrix4<f32>) -> Matrix4<f32> {
    // Rescale before normalising so a ~1e-36 column survives the division.
    let unit = |v: Vector3<f32>| {
        let big = v.x.abs().max(v.y.abs()).max(v.z.abs());
        if !(big.is_finite() && big > 0.0) {
            return None;
        }
        let v = v / big;
        let length = v.magnitude();
        (length > 1.0e-6).then(|| v / length)
    };
    let cols = [host.x.truncate(), host.y.truncate(), host.z.truncate()];
    let mut order = [0, 1, 2];
    order.sort_by(|&i, &j| {
        let size = |k: usize| cols[k].x.abs().max(cols[k].y.abs()).max(cols[k].z.abs());
        size(j).total_cmp(&size(i))
    });
    let [i, j, k] = order;
    // Rebuild the rotation from the two best-conditioned axes; the third comes
    // from their cross product with the frame's handedness.
    let rebuilt = unit(cols[i]).and_then(|a| {
        let b =
            unit(cols[j] / cols[j].magnitude().max(f32::MIN_POSITIVE)).or_else(|| unit(cols[j]))?;
        let b = unit(b - a * a.dot(b))?;
        let cyclic = (j + 3 - i) % 3 == 1;
        let c = if cyclic { a.cross(b) } else { b.cross(a) };
        let mut axes = [Vector3::unit_x(); 3];
        axes[i] = a;
        axes[j] = b;
        axes[k] = c;
        Some(axes)
    });
    let [x, y, z] = rebuilt.unwrap_or([Vector3::unit_x(), Vector3::unit_y(), Vector3::unit_z()]);
    Matrix4::from_cols(x.extend(0.0), y.extend(0.0), z.extend(0.0), host.w)
}

/// Seconds between launches. A continuous group spreads its particles evenly
/// over an average lifetime, overriding the authored period (often 0, which
/// would launch every frame until full and then pause - a dashed trail);
/// pre-launched and one-shot groups keep the authored period.
pub fn launch_period(pg: &PropParticleGroup, launch: &PropParticleLaunchInfo) -> f32 {
    if pg.animation_type == PAT_LAUNCH_CONTINUOUS && !pg.pre_launch && pg.num > 0 {
        (launch.min_time + launch.max_time) / 2.0 / pg.num as f32
    } else {
        pg.launch_time
    }
}

/// Colour over a disk particle's remaining life, as engine color keys. The
/// palette indices `g` / `b` are the fade targets: `r` until the shortest
/// lifetime remains, then to `g` at death, or to `g` at half of it and `b`
/// at death when both are set. Other render types and fixed (non-launch)
/// animations draw a single colour.
pub fn color_fade_keys(pg: &PropParticleGroup, min_lifetime: f32) -> Vec<(f32, Vector3<f32>)> {
    if pg.render_type != PRT_SINGLE_COLOR_DISK
        || pg.animation_type > PAT_LAUNCH_CONTINUOUS
        || pg.fade_time <= 0.0
        || (pg.g == 0 && pg.b == 0)
    {
        return Vec::new();
    }
    let rgb = palette::index_to_rgb;
    if pg.g != 0 && pg.b != 0 {
        vec![
            (min_lifetime, rgb(pg.r)),
            (min_lifetime / 2.0, rgb(pg.g)),
            (0.0, rgb(pg.b)),
        ]
    } else {
        vec![(min_lifetime, rgb(pg.r)), (0.0, rgb(pg.g))]
    }
}

#[cfg(test)]
mod tests {
    use cgmath::Vector3;
    use dark::properties::PropParticleGroup;

    use dark::properties::PropParticleLaunchInfo;

    use super::{color_fade_keys, emitter_transform, is_world_space, launch_period, swirl_rate};

    #[test]
    fn the_emitter_frame_drops_host_scale_even_when_degenerate() {
        use cgmath::{Deg, InnerSpace, Matrix4, Quaternion, Rotation3, vec3};
        let rotation = Matrix4::from(Quaternion::from_angle_y(Deg(90.0)));
        let at = vec3(1.0, 2.0, 3.0);
        let expected = Matrix4::from_translation(at) * rotation;
        for scale in [vec3(2.0, 3.0, 4.0), vec3(1.46e-36, 1.46e-36, 5.6e-42)] {
            let host = Matrix4::from_translation(at)
                * rotation
                * Matrix4::from_nonuniform_scale(scale.x, scale.y, scale.z);
            let frame = emitter_transform(host);
            for (a, b) in [(frame.x, expected.x), (frame.w, expected.w)] {
                assert!((a - b).magnitude() < 1.0e-5, "{scale:?}: {a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn spin_is_authored_in_degrees_per_second() {
        let mut pg = group(0, 1, 0, 0);
        pg.spin = Vector3::new(0.0, 250.0, 0.0);
        assert!((swirl_rate(&pg) - 250f32.to_radians()).abs() < 1.0e-6);
    }
    use crate::palette::index_to_rgb;

    fn group(motion_type: u32, animation_type: u32, g: u8, b: u8) -> PropParticleGroup {
        let zero = Vector3::new(0.0, 0.0, 0.0);
        PropParticleGroup {
            render_type: 4,
            motion_type,
            animation_type,
            num: 10,
            velocity: zero,
            gravity: zero,
            r: 10,
            g,
            b,
            a: 255,
            spin: zero,
            is_active: true,
            is_worldspace: false,
            pre_launch: false,
            size: 1.0,
            scale_vel: 0.0,
            prev_loc: zero,
            bbox_min: zero,
            bbox_max: zero,
            radius: 0.0,
            launch_time: 0.1,
            fade_time: 0.5,
            model_name: String::new(),
        }
    }

    #[test]
    fn only_launching_trailing_groups_are_world_space() {
        assert!(is_world_space(&group(3, 1, 0, 0)));
        assert!(is_world_space(&group(3, 0, 0, 0)));
        assert!(
            !is_world_space(&group(3, 2, 0, 0)),
            "fixed shape rides its host"
        );
        let mut attached = group(4, 1, 0, 0);
        attached.is_worldspace = true;
        assert!(!is_world_space(&attached), "authored flag is ignored");
    }

    #[test]
    fn color_fade_keys_follow_the_set_targets() {
        assert_eq!(color_fade_keys(&group(0, 1, 0, 0), 2.0), vec![]);
        assert_eq!(
            color_fade_keys(&group(0, 1, 20, 0), 2.0),
            vec![(2.0, index_to_rgb(10)), (0.0, index_to_rgb(20))]
        );
        assert_eq!(
            color_fade_keys(&group(0, 1, 20, 30), 2.0),
            vec![
                (2.0, index_to_rgb(10)),
                (1.0, index_to_rgb(20)),
                (0.0, index_to_rgb(30)),
            ]
        );
        let mut bitmap = group(0, 1, 20, 30);
        bitmap.render_type = 5;
        assert_eq!(color_fade_keys(&bitmap, 2.0), vec![], "disks only");
        assert_eq!(
            color_fade_keys(&group(0, 2, 20, 30), 2.0),
            vec![],
            "launch only"
        );
        let mut no_fade = group(0, 1, 20, 30);
        no_fade.fade_time = 0.0;
        assert_eq!(color_fade_keys(&no_fade, 2.0), vec![]);
    }

    #[test]
    fn continuous_groups_spread_launches_over_an_average_lifetime() {
        let zero = Vector3::new(0.0, 0.0, 0.0);
        let launch = PropParticleLaunchInfo {
            launch_type: 0,
            loc_min: zero,
            loc_max: zero,
            vel_min: zero,
            vel_max: zero,
            min_radius: 0.0,
            max_radius: 0.0,
            min_time: 0.4,
            max_time: 0.6,
        };
        // 10 particles over a 0.5 s average lifetime.
        assert!((launch_period(&group(3, 1, 0, 0), &launch) - 0.05).abs() < 1e-6);
        let mut pre = group(3, 1, 0, 0);
        pre.pre_launch = true;
        assert_eq!(launch_period(&pre, &launch), 0.1, "authored period");
        assert_eq!(
            launch_period(&group(3, 0, 0, 0), &launch),
            0.1,
            "one-shot keeps it"
        );
    }
}

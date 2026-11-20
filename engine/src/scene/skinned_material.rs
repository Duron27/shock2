extern crate gl;
use std::rc::Rc;

use super::incidence::{IncidencePass, IncidenceUniforms};
use super::shine::{Shine, ShineUniforms};
use crate::engine::EngineRenderContext;
use crate::gfx;
use crate::gfx::Program as ShaderProgram;
use crate::scene::Material;

use super::material::Glow;
use crate::texture::TextureTrait;
use cgmath::Matrix4;
use cgmath::Vector3;

use once_cell::sync::OnceCell;
use std::any::Any;

// Unified shader for single-pass lighting with up to 6 spotlights (skinned version)
const UNIFIED_VERTEX_SHADER_SOURCE: &str = r#"
        // Depth and color link this vertex shader into separate programs.
        invariant gl_Position;
        layout (location = 0) in highp vec3 inPos;
        layout (location = 1) in vec2 inTex;
        layout (location = 2) in ivec4 bone_ids;
        layout (location = 3) in vec4 bone_weights;
        layout (location = 4) in vec3 inNormal;

        uniform highp mat4 world;
        uniform highp mat4 view;
        uniform highp mat4 projection;
        uniform mat3 normalMatrix;
        // Affine bones packed as 3 vec4 rows each (the mat3x4's columns hold
        // the matrix rows), so the 80-slot palette costs 240 uniform vectors
        // and fits the GLES 3.0 guaranteed vertex budget (256); a mat4 array
        // would need 320. Apply with `vec4(v) * bone_matrices[i]`.
        uniform highp mat3x4 bone_matrices[80];

        out vec2 texCoord;
        out highp vec3 worldPos;
        out vec3 worldNormal;

        void main() {
            texCoord = inTex;

            // Apply weighted bone transformations to position and normal
            highp vec3 skinnedPos = vec3(0.0);
            vec3 skinnedNormal = vec3(0.0);
            float totalWeight = 0.0;

            // Blend up to 4 bones based on weights
            if (bone_weights.x > 0.0) {
                skinnedPos += bone_weights.x * (vec4(inPos, 1.0) * bone_matrices[bone_ids.x]);
                skinnedNormal += bone_weights.x * (vec4(inNormal, 0.0) * bone_matrices[bone_ids.x]);
                totalWeight += bone_weights.x;
            }
            if (bone_weights.y > 0.0) {
                skinnedPos += bone_weights.y * (vec4(inPos, 1.0) * bone_matrices[bone_ids.y]);
                skinnedNormal += bone_weights.y * (vec4(inNormal, 0.0) * bone_matrices[bone_ids.y]);
                totalWeight += bone_weights.y;
            }
            if (bone_weights.z > 0.0) {
                skinnedPos += bone_weights.z * (vec4(inPos, 1.0) * bone_matrices[bone_ids.z]);
                skinnedNormal += bone_weights.z * (vec4(inNormal, 0.0) * bone_matrices[bone_ids.z]);
                totalWeight += bone_weights.z;
            }
            if (bone_weights.w > 0.0) {
                skinnedPos += bone_weights.w * (vec4(inPos, 1.0) * bone_matrices[bone_ids.w]);
                skinnedNormal += bone_weights.w * (vec4(inNormal, 0.0) * bone_matrices[bone_ids.w]);
                totalWeight += bone_weights.w;
            }

            // Fallback to original position if no valid bones
            highp vec4 mod_position = (totalWeight > 0.0) ? vec4(skinnedPos, 1.0) : vec4(inPos, 1.0);
            vec3 mod_normal = (length(skinnedNormal) > 0.0) ? normalize(skinnedNormal) : inNormal;

            // Transform to world space
            highp vec4 worldPosition = world * mod_position;
            worldPos = worldPosition.xyz;
            worldNormal = normalize(normalMatrix * mod_normal);

            gl_Position = projection * view * waveWorldPosition(worldPosition);
        }
"#;

pub(crate) const UNIFIED_FRAGMENT_SHADER_SOURCE: &str = r#"
        out vec4 fragColor;

        in vec2 texCoord;
        in highp vec3 worldPos;
        in vec3 worldNormal;

        // Material properties
        uniform sampler2D texture1;
        uniform float emissivity;
        // Per-draw glow, added regardless of authored emission, inside a UV rect.
        uniform vec3 glow;
        uniform vec4 glowRect;
        // Darkens a glow part's lit colour along with its glow (see `lit_scale`).
        uniform float litScale;
        uniform float ambientIntensity;
        uniform float transparency;
        uniform vec3 silhouetteColor;
        uniform vec3 eyePosition;

        // A light painted onto part of the mesh: the mask's red channel says
        // where, `emissiveTint` says what colour and how bright. A zero tint
        // (the default) leaves the mesh exactly as it was, so meshes without a
        // light bind their own diffuse map here and pay only the sample.
        uniform sampler2D emissiveMask;
        uniform vec3 emissiveTint;

        // Spotlight array uniforms (up to 6 spotlights)
        uniform highp vec3 spotlightPos[6];
        uniform vec4 spotlightColorIntensity[6];  // RGB + intensity
        uniform highp vec3 spotlightDirection[6];
        uniform highp float spotlightInnerAngle[6];
        uniform highp float spotlightOuterAngle[6];
        uniform highp float spotlightRange[6];

        // Light every surface gets before any lamp reaches it.
        uniform vec3 ambientLight;
        // 0 = the renderer's smooth curve, 1 = inverse distance (how the
        // original lit objects - it falls off far more slowly).
        uniform int lightFalloffMode[6];
        // 0 = plain lambert, 1 = half-lambert (light wraps past the terminator).
        uniform float lambertWrap;

        // Lights are sources of this radius, not points: without a floor the
        // inverse-distance falloff runs away inside the model a lamp belongs
        // to. Must equal engine::scene::light::LIGHT_SOURCE_RADIUS.
        const float SOURCE_RADIUS = 0.8;

        // Calculate spotlight contribution
        vec3 calculateSpotlight(int i, highp vec3 worldPos, vec3 normal, vec3 texColor, inout vec3 specular, float power) {
            // Skip if light has zero intensity
            if (spotlightColorIntensity[i].w <= 0.0) {
                return vec3(0.0);
            }

            highp vec3 lightVec = spotlightPos[i] - worldPos;
            highp float distance = length(lightVec);

            // Range check
            if (distance > spotlightRange[i]) {
                return vec3(0.0);
            }

            highp vec3 lightDir = lightVec / max(distance, 0.00001);

            // Cone attenuation. A negative inner angle marks a point light -
            // it has no cone, so it lights every direction equally.
            highp float coneAttenuation = 1.0;
            if (spotlightInnerAngle[i] >= 0.0) {
                highp float cosOuterCone = cos(spotlightOuterAngle[i]);
                highp float cosInnerCone = cos(spotlightInnerAngle[i]);
                highp float spotFactor = dot(-lightDir, normalize(spotlightDirection[i]));

                if (spotFactor < cosOuterCone) {
                    return vec3(0.0);
                }

                if (lightFalloffMode[i] == 0) {
                    coneAttenuation = flashlightCone(spotFactor, cosInnerCone, cosOuterCone);
                } else if (spotFactor < cosInnerCone) {
                    // Preserve the mission's authored spotlight profile.
                    coneAttenuation = (spotFactor - cosOuterCone) / (cosInnerCone - cosOuterCone);
                }
            }

            // Distance attenuation
            highp float distanceAttenuation;
            if (lightFalloffMode[i] == 1) {
                distanceAttenuation = 1.0 / max(distance, SOURCE_RADIUS);
            } else {
                distanceAttenuation = flashlightDistance(distance, spotlightRange[i]);
            }

            // Diffuse lighting, optionally wrapped past the terminator so a
            // surface facing away is lifted rather than black.
            float ndl = dot(normal, lightDir);
            float wrap = lightFalloffMode[i] == 1 ? lambertWrap : 0.0;
            float lambertian = max((ndl + wrap) / (1.0 + wrap), 0.0);

            vec3 radiance = spotlightColorIntensity[i].rgb * spotlightColorIntensity[i].w
                            * coneAttenuation * distanceAttenuation;
            if (shineHighlights()) {
                specular += shineHighlight(radiance, lightDir, normal, worldPos, power);
            }
            return texColor * radiance * lambertian;
        }

        void main() {
            if (hiddenBySelfDepth()) discard;
            vec4 texColor = sampleMaterialPass(texture1, texCoord, worldPos, worldNormal);

            if (materialAlphaRejected(texColor.a)) discard;


            if (silhouetteColor.x >= 0.0) {
                float rim = pow(1.0 - abs(dot(normalize(worldNormal), normalize(eyePosition - worldPos))), 2.0);
                fragColor = vec4(silhouetteColor, texColor.a * (1.0 - transparency) * (0.18 + 0.82 * rim));
                return;
            }

            vec3 finalColor = texColor.rgb * emissivity;
            // Mostly flat, so dark metal heats up too; a little detail keeps its form.
            vec2 inRect = step(glowRect.xy, texCoord) * step(texCoord, glowRect.zw);
            finalColor += glow * inRect.x * inRect.y
                * (0.6 + 0.4 * dot(texColor.rgb, vec3(0.299, 0.587, 0.114)));

            // Add the painted-on light. The branch is uniform across the draw,
            // so a mesh without a light never pays the second fetch.
            if (any(greaterThan(emissiveTint, vec3(0.0)))) {
                finalColor += texture(emissiveMask, texCoord).r * emissiveTint;
            }

            // Light reaching the surface before albedo, shared with the shine.
            vec3 normal = normalize(worldNormal);
            vec3 light = ambientLight * ambientIntensity;
            vec3 specular = vec3(0.0);
            float power = shineHighlights() ? shinePower(length(worldNormal)) : SHINE_POWER;
            for (int i = 0; i < 6; i++) {
                light += calculateSpotlight(i, worldPos, normal, vec3(1.0), specular, power);
            }
            finalColor += texColor.rgb * light * litScale;
            vec4 shaded = applyShine(vec4(finalColor, texColor.a * (1.0 - transparency)), light, specular, 1.0 - transparency, texCoord, worldPos, normal);

            fragColor = applyMaterialPass(applyIncidence(shaded, texColor, worldPos, worldNormal), texColor, worldPos, worldNormal);
        }
"#;

struct UnifiedUniforms {
    world_wave: crate::scene::world_wave::Uniforms,
    self_depth: super::self_depth::Uniforms,
    incidence: IncidenceUniforms,
    render_pass: super::render_pass::Uniforms,
    shine: ShineUniforms,
    // Basic transformation matrices
    world_loc: gfx::Uniform,
    view_loc: gfx::Uniform,
    projection_loc: gfx::Uniform,
    normal_matrix_loc: gfx::Uniform,

    // Material properties
    emissivity_loc: gfx::Uniform,
    glow_loc: gfx::Uniform,
    glow_rect_loc: gfx::Uniform,
    lit_scale_loc: gfx::Uniform,
    ambient_intensity_loc: gfx::Uniform,
    transparency_loc: gfx::Uniform,
    emissive_tint_loc: gfx::Uniform,
    silhouette_color_loc: gfx::Uniform,
    eye_position_loc: gfx::Uniform,

    // Bone matrices for skeletal animation
    bone_matrices_loc: gfx::Uniform,

    // Spotlight array uniforms (6 spotlights)
    spotlight_pos_loc: [gfx::Uniform; 6],
    spotlight_color_intensity_loc: [gfx::Uniform; 6],
    spotlight_direction_loc: [gfx::Uniform; 6],
    spotlight_inner_angle_loc: [gfx::Uniform; 6],
    spotlight_outer_angle_loc: [gfx::Uniform; 6],
    spotlight_range_loc: [gfx::Uniform; 6],
    ambient_light_loc: gfx::Uniform,
    light_falloff_mode_loc: gfx::Uniform,
    lambert_wrap_loc: gfx::Uniform,
}

static UNIFIED_SHADER_PROGRAM: OnceCell<(ShaderProgram, UnifiedUniforms)> = OnceCell::new();
static SELF_DEPTH_SHADER_PROGRAM: OnceCell<(ShaderProgram, UnifiedUniforms)> = OnceCell::new();

#[derive(Clone)]
pub struct SkinnedMaterial {
    incidence: Option<IncidencePass>,
    render_pass: Option<super::render_pass::RenderPass>,
    shine: Option<Shine>,
    silhouette_color: Option<Vector3<f32>>,
    has_initialized: bool,
    diffuse_texture: Rc<dyn TextureTrait>,
    emissivity: f32,
    /// Per-draw multiplier from `Material::set_emissivity_scale`.
    emissivity_scale: f32,
    /// Per-draw flag from `Material::set_full_bright`.
    full_bright: bool,
    /// Per-draw glow from `Material::set_glow`.
    glow: Glow,
    transparency: f32,
    base_transparency: f32,
    /// Where the painted-on light sits, in the diffuse map's own UVs (red
    /// channel). `None` binds the diffuse map in its place, so the sampler
    /// always has a real texture; `emissive_tint` is then zero and the sample
    /// contributes nothing.
    emissive_mask: Option<Rc<dyn TextureTrait>>,
    emissive_tint: Vector3<f32>,
}

impl SkinnedMaterial {
    /// Reuse the animated mesh and skinning shader for an unlit rim silhouette.
    /// The source material and other instances of the model stay unchanged.
    pub fn silhouette(&self, color: Vector3<f32>) -> Box<dyn Material> {
        let mut material = self.clone();
        material.silhouette_color = Some(color);
        material.incidence = None;
        material.render_pass = None;
        material.shine = None;
        Box::new(material)
    }

    pub fn is_transparent(&self) -> bool {
        self.incidence.is_some() || self.transparency > 0.01
    }

    pub fn set_transparency_override(&mut self, transparency: f32) {
        self.transparency = transparency.clamp(0.0, 1.0);
    }

    pub fn reset_transparency(&mut self) {
        self.transparency = self.base_transparency;
    }

    pub fn draw_unified(
        &self,
        render_context: &EngineRenderContext,
        view_matrix: &Matrix4<f32>,
        world_matrix: &Matrix4<f32>,
        skinning_data: &[Matrix4<f32>],
        lights: &crate::scene::light::LightArray,
    ) {
        let program = if matches!(render_context.self_depth, super::self_depth::Phase::Capture) {
            &SELF_DEPTH_SHADER_PROGRAM
        } else {
            &UNIFIED_SHADER_PROGRAM
        };
        let (shader_program, uniforms) = program.get().expect("unified shader not compiled");
        self.diffuse_texture.bind0(render_context);
        self.emissive_mask
            .as_ref()
            .unwrap_or(&self.diffuse_texture)
            .bind1(render_context);
        shader_program.bind();
        uniforms.world_wave.bind(render_context.world_wave);
        uniforms.self_depth.bind(render_context.self_depth);
        uniforms
            .incidence
            .bind(self.incidence.as_ref(), render_context, view_matrix);
        uniforms
            .shine
            .bind(self.shine.as_ref(), lights, render_context, view_matrix);

        uniforms.render_pass.bind(
            self.render_pass.as_ref(),
            render_context,
            view_matrix,
            1.0 - self.transparency,
        );
        let projection = render_context.projection_matrix;

        // Set basic transformation matrices
        gfx::set_mat4(uniforms.world_loc, &world_matrix);
        gfx::set_mat4(uniforms.view_loc, &view_matrix);
        gfx::set_mat4(uniforms.projection_loc, &projection);
        let normal_matrix = crate::scene::material::normal_matrix(world_matrix);
        gfx::set_mat3(uniforms.normal_matrix_loc, &normal_matrix);

        let color = self
            .silhouette_color
            .unwrap_or(Vector3::new(-1.0, -1.0, -1.0));
        gfx::set_vec3(
            uniforms.silhouette_color_loc,
            *AsRef::<[f32; 3]>::as_ref(&color),
        );
        if self.silhouette_color.is_some() {
            gfx::set_vec3(
                uniforms.eye_position_loc,
                crate::scene::material::eye_position(view_matrix).into(),
            );
        }

        // Set material properties
        gfx::set_f32(uniforms.transparency_loc, self.transparency);
        let (emission, lit_scale) = super::material::emission_and_lit_scale(
            self.emissivity,
            self.emissivity_scale,
            self.full_bright,
        );
        gfx::set_f32(uniforms.emissivity_loc, emission);
        let Glow { color, uv_rect } = self.glow;
        gfx::set_vec3(uniforms.glow_loc, [color.x, color.y, color.z]);
        gfx::set_vec4(
            uniforms.glow_rect_loc,
            [uv_rect.x, uv_rect.y, uv_rect.z, uv_rect.w],
        );
        gfx::set_f32(uniforms.lit_scale_loc, lit_scale);
        gfx::set_f32(
            uniforms.ambient_intensity_loc,
            lights.ambient_intensity(render_context.ambient_light_intensity),
        );
        gfx::set_vec3(
            uniforms.emissive_tint_loc,
            [
                self.emissive_tint.x,
                self.emissive_tint.y,
                self.emissive_tint.z,
            ],
        );

        // Pack each affine bone as its 3 matrix rows (the shader-side
        // mat3x4's columns) and upload the whole palette in one call.
        let mut packed = [0f32; crate::scene::SKINNING_PALETTE_SIZE * 12];
        for (bone, mat) in skinning_data
            .iter()
            .enumerate()
            .take(crate::scene::SKINNING_PALETTE_SIZE)
        {
            let base = bone * 12;
            for row in 0..3 {
                packed[base + row * 4] = mat.x[row];
                packed[base + row * 4 + 1] = mat.y[row];
                packed[base + row * 4 + 2] = mat.z[row];
                packed[base + row * 4 + 3] = mat.w[row];
            }
        }
        gfx::set_mat3x4_array(uniforms.bone_matrices_loc, &packed);

        // How this array's lights behave: what an unlit surface shows, and
        // how the lights fall off.
        gfx::set_vec3(
            uniforms.ambient_light_loc,
            [lights.ambient.x, lights.ambient.y, lights.ambient.z],
        );
        gfx::set_f32(uniforms.lambert_wrap_loc, lights.lambert_wrap);
        let falloff_modes = lights.falloff.map(|mode| match mode {
            crate::scene::light::LightFalloff::Smooth => 0,
            crate::scene::light::LightFalloff::InverseDistance => 1,
        });
        gfx::set_i32_array(uniforms.light_falloff_mode_loc, &falloff_modes);

        // Set spotlight array uniforms
        for i in 0..6 {
            if let Some(light) = lights.get_light(i) {
                let pos = light.position();
                let color_intensity = light.color_intensity();
                let direction = light.direction();

                gfx::set_vec3(uniforms.spotlight_pos_loc[i], [pos.x, pos.y, pos.z]);
                gfx::set_vec4(
                    uniforms.spotlight_color_intensity_loc[i],
                    [
                        color_intensity.x,
                        color_intensity.y,
                        color_intensity.z,
                        color_intensity.w,
                    ],
                );
                gfx::set_vec3(
                    uniforms.spotlight_direction_loc[i],
                    [direction.x, direction.y, direction.z],
                );
                gfx::set_f32(
                    uniforms.spotlight_inner_angle_loc[i],
                    light.inner_cone_angle(),
                );
                gfx::set_f32(
                    uniforms.spotlight_outer_angle_loc[i],
                    light.outer_cone_angle(),
                );
                gfx::set_f32(uniforms.spotlight_range_loc[i], light.range());
            } else {
                // Disable this light slot by setting intensity to 0
                gfx::set_vec4(
                    uniforms.spotlight_color_intensity_loc[i],
                    [0.0, 0.0, 0.0, 0.0],
                );
            }
        }
    }
}
impl Material for SkinnedMaterial {
    fn emissivity(&self) -> f32 {
        self.emissivity
    }

    fn set_emissivity_scale(&mut self, scale: f32) {
        self.emissivity_scale = scale;
    }

    fn set_full_bright(&mut self, full_bright: bool) {
        self.full_bright = full_bright;
    }

    fn set_glow(&mut self, glow: Glow) {
        self.glow = glow;
    }

    fn set_diffuse_texture(&mut self, texture: Rc<dyn TextureTrait>) {
        self.diffuse_texture = texture;
    }

    fn set_render_pass(&mut self, pass: super::render_pass::RenderPass) {
        self.render_pass = Some(pass);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn set_shine(&mut self, shine: Shine) {
        self.shine = Some(shine);
    }

    fn set_transparency_override(&mut self, transparency: Option<f32>) {
        self.transparency = match transparency {
            Some(value) => value.clamp(0.0, 1.0),
            None => self.base_transparency,
        };
    }

    fn transparency(&self) -> Option<f32> {
        Some(self.transparency)
    }

    fn has_initialized(&self) -> bool {
        self.has_initialized
    }

    fn initialize(&mut self, _is_opengl_es: bool) {
        let build = |capture| {
            // Build and compile unified shader program with 6-spotlight support for skinned meshes
            let vertex_shader =
                &[crate::scene::world_wave::GLSL, UNIFIED_VERTEX_SHADER_SOURCE].concat();

            let fragment_shader = &format!(
                "{}\n{}\n{}\n{}\n{}\n{}\n{}",
                include_str!("flashlight.glsl"),
                super::self_depth::GLSL,
                super::incidence::GLSL,
                super::environment::GLSL,
                super::shine::GLSL,
                super::render_pass::GLSL,
                if capture {
                    super::self_depth::FRAGMENT
                } else {
                    UNIFIED_FRAGMENT_SHADER_SOURCE
                }
            );

            let shader = gfx::Program::new(&vertex_shader, &fragment_shader);

            // Which texture unit each sampler reads is program state, not
            // per-material state, so it is set once here rather than on
            // every draw. (`texture1` would default to unit 0 anyway; the
            // mask has to be told about unit 1.)
            shader.bind();
            gfx::set_i32(shader.uniform("texture1"), 0);
            gfx::set_i32(shader.uniform("emissiveMask"), 1);

            // Get uniform locations for all shader variables
            let bone_matrices_loc = shader.uniform("bone_matrices[0]");

            let uniforms = UnifiedUniforms {
                world_wave: crate::scene::world_wave::Uniforms::new(&shader),
                self_depth: super::self_depth::Uniforms::new(&shader),
                incidence: IncidenceUniforms::new(&shader),
                render_pass: super::render_pass::Uniforms::new(&shader),
                shine: ShineUniforms::new(&shader),
                // Basic transformation matrices
                world_loc: shader.uniform("world"),
                view_loc: shader.uniform("view"),
                projection_loc: shader.uniform("projection"),
                normal_matrix_loc: shader.uniform("normalMatrix"),

                // Material properties
                ambient_intensity_loc: shader.uniform("ambientIntensity"),
                emissivity_loc: shader.uniform("emissivity"),
                glow_loc: shader.uniform("glow"),
                glow_rect_loc: shader.uniform("glowRect"),
                lit_scale_loc: shader.uniform("litScale"),
                transparency_loc: shader.uniform("transparency"),
                emissive_tint_loc: shader.uniform("emissiveTint"),

                silhouette_color_loc: shader.uniform("silhouetteColor"),
                eye_position_loc: shader.uniform("eyePosition"),

                // Bone matrices
                bone_matrices_loc,

                // Spotlight array uniforms (6 spotlights)
                spotlight_pos_loc: [
                    shader.uniform("spotlightPos[0]"),
                    shader.uniform("spotlightPos[1]"),
                    shader.uniform("spotlightPos[2]"),
                    shader.uniform("spotlightPos[3]"),
                    shader.uniform("spotlightPos[4]"),
                    shader.uniform("spotlightPos[5]"),
                ],
                spotlight_color_intensity_loc: [
                    shader.uniform("spotlightColorIntensity[0]"),
                    shader.uniform("spotlightColorIntensity[1]"),
                    shader.uniform("spotlightColorIntensity[2]"),
                    shader.uniform("spotlightColorIntensity[3]"),
                    shader.uniform("spotlightColorIntensity[4]"),
                    shader.uniform("spotlightColorIntensity[5]"),
                ],
                spotlight_direction_loc: [
                    shader.uniform("spotlightDirection[0]"),
                    shader.uniform("spotlightDirection[1]"),
                    shader.uniform("spotlightDirection[2]"),
                    shader.uniform("spotlightDirection[3]"),
                    shader.uniform("spotlightDirection[4]"),
                    shader.uniform("spotlightDirection[5]"),
                ],
                spotlight_inner_angle_loc: [
                    shader.uniform("spotlightInnerAngle[0]"),
                    shader.uniform("spotlightInnerAngle[1]"),
                    shader.uniform("spotlightInnerAngle[2]"),
                    shader.uniform("spotlightInnerAngle[3]"),
                    shader.uniform("spotlightInnerAngle[4]"),
                    shader.uniform("spotlightInnerAngle[5]"),
                ],
                spotlight_outer_angle_loc: [
                    shader.uniform("spotlightOuterAngle[0]"),
                    shader.uniform("spotlightOuterAngle[1]"),
                    shader.uniform("spotlightOuterAngle[2]"),
                    shader.uniform("spotlightOuterAngle[3]"),
                    shader.uniform("spotlightOuterAngle[4]"),
                    shader.uniform("spotlightOuterAngle[5]"),
                ],
                spotlight_range_loc: [
                    shader.uniform("spotlightRange[0]"),
                    shader.uniform("spotlightRange[1]"),
                    shader.uniform("spotlightRange[2]"),
                    shader.uniform("spotlightRange[3]"),
                    shader.uniform("spotlightRange[4]"),
                    shader.uniform("spotlightRange[5]"),
                ],
                ambient_light_loc: shader.uniform("ambientLight"),
                light_falloff_mode_loc: shader.uniform("lightFalloffMode[0]"),
                lambert_wrap_loc: shader.uniform("lambertWrap"),
            };
            (shader, uniforms)
        };
        UNIFIED_SHADER_PROGRAM.get_or_init(|| build(false));
        SELF_DEPTH_SHADER_PROGRAM.get_or_init(|| build(true));
        self.has_initialized = true;
    }

    fn draw_opaque(
        &self,
        render_context: &EngineRenderContext,
        view_matrix: &Matrix4<f32>,
        world_matrix: &Matrix4<f32>,
        _skinning_data: &[Matrix4<f32>],
        _lights: &crate::scene::light::LightArray,
    ) -> bool {
        if !self.is_transparent() {
            self.draw_unified(
                render_context,
                view_matrix,
                world_matrix,
                _skinning_data,
                _lights,
            );
            true
        } else {
            false
        }
    }

    fn draw_transparent(
        &self,
        render_context: &EngineRenderContext,
        view_matrix: &Matrix4<f32>,
        world_matrix: &Matrix4<f32>,
        _skinning_data: &[Matrix4<f32>],
        _lights: &crate::scene::light::LightArray,
    ) -> bool {
        if self.is_transparent() {
            self.draw_unified(
                render_context,
                view_matrix,
                world_matrix,
                _skinning_data,
                _lights,
            );
            true
        } else {
            false
        }
    }
}

impl SkinnedMaterial {
    pub fn create(
        diffuse_texture: Rc<dyn TextureTrait>,
        emissivity: f32,
        transparency: f32,
    ) -> Box<dyn Material> {
        Self::build(
            diffuse_texture,
            emissivity,
            transparency,
            None,
            Vector3::new(0.0, 0.0, 0.0),
        )
    }

    /// As [`SkinnedMaterial::create`], plus a light painted onto the parts of
    /// the mesh `emissive_mask` marks (its red channel), in `emissive_tint`'s
    /// colour. A zero tint renders identically to `create`.
    pub fn create_with_light(
        diffuse_texture: Rc<dyn TextureTrait>,
        emissivity: f32,
        transparency: f32,
        emissive_mask: Rc<dyn TextureTrait>,
        emissive_tint: Vector3<f32>,
    ) -> Box<dyn Material> {
        Self::build(
            diffuse_texture,
            emissivity,
            transparency,
            Some(emissive_mask),
            emissive_tint,
        )
    }

    /// Use the normal skinning palette and lighting for a specular overlay.
    pub fn create_incidence(
        texture: Rc<dyn TextureTrait>,
        pass: IncidencePass,
    ) -> Box<dyn Material> {
        Box::new(Self {
            incidence: Some(pass),
            render_pass: None,
            shine: None,
            silhouette_color: None,
            diffuse_texture: texture,
            has_initialized: false,
            emissivity: 0.0,
            emissivity_scale: 1.0,
            full_bright: false,
            glow: Glow::NONE,
            transparency: 0.0,
            base_transparency: 0.0,
            emissive_mask: None,
            emissive_tint: Vector3::new(0.0, 0.0, 0.0),
        })
    }

    fn build(
        diffuse_texture: Rc<dyn TextureTrait>,
        emissivity: f32,
        transparency: f32,
        emissive_mask: Option<Rc<dyn TextureTrait>>,
        emissive_tint: Vector3<f32>,
    ) -> Box<dyn Material> {
        Box::new(SkinnedMaterial {
            incidence: None,
            render_pass: None,
            shine: None,
            silhouette_color: None,
            diffuse_texture,
            has_initialized: false,
            emissivity,
            emissivity_scale: 1.0,
            full_bright: false,
            glow: Glow::NONE,
            transparency,
            base_transparency: transparency,
            emissive_mask,
            emissive_tint,
        })
    }
}

/// The fragment shader source, for the test that pins its SOURCE_RADIUS to the
/// engine constant. Both lit materials declare it; both must agree.
#[cfg(test)]
pub(crate) const FRAGMENT_SHADER_SOURCE_FOR_TEST: &str = UNIFIED_FRAGMENT_SHADER_SOURCE;

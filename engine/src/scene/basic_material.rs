use std::any::Any;
use std::ops::Deref;

use super::incidence::{IncidencePass, IncidenceUniforms};
use super::shine::{Shine, ShineUniforms};
use crate::engine::EngineRenderContext;
use crate::gfx;
use crate::gfx::Program as ShaderProgram;
use crate::scene::Material;

use super::material::Glow;
use crate::texture::TextureTrait;
use cgmath::Matrix4;

use once_cell::sync::OnceCell;

// Unified shader for single-pass lighting with up to 6 spotlights
const UNIFIED_VERTEX_SHADER_SOURCE: &str = r#"
        // Depth and color link this vertex shader into separate programs.
        invariant gl_Position;
        layout (location = 0) in highp vec3 inPos;
        layout (location = 1) in vec2 inTex;
        layout (location = 2) in vec3 inNormal;

        uniform highp mat4 world;
        uniform highp mat4 view;
        uniform highp mat4 projection;
        uniform mat3 normalMatrix;
        uniform bool unlit;

        out vec2 texCoord;
        out highp vec3 worldPos;
        out vec3 worldNormal;

        void main() {
            texCoord = inTex;
            highp vec4 worldPosition = world * vec4(inPos, 1.0);
            worldPos = worldPosition.xyz;

            worldNormal = unlit ? vec3(0.0) : normalize(normalMatrix * inNormal);

            gl_Position = projection * view * waveWorldPosition(worldPosition);
        }
"#;

const UNIFIED_FRAGMENT_SHADER_SOURCE: &str = r#"
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
        uniform bool additiveUnlit;
        uniform bool unlit;

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
                            * coneAttenuation * distanceAttenuation
                            * pointShadowVisibility(lightShadowSlot[i], 0, worldPos, normal);
            if (shineHighlights()) {
                specular += shineHighlight(radiance, lightDir, normal, worldPos, power);
            }
            return texColor * radiance * lambertian;
        }

        uniform float heldReflection;
        uniform bool heldEnvironment;

        void main() {
            if (hiddenBySelfDepth()) discard;
            vec4 texColor = sampleMaterialPass(texture1, texCoord, worldPos, worldNormal);

            if (additiveUnlit) {
                // SRC_COLOR/ONE squares RGB. Scale by sqrt(opacity) so the
                // accumulated light fades linearly with authored alpha.
                fragColor = vec4(texColor.rgb * sqrt(texColor.a * (1.0 - transparency)), 1.0);
                return;
            }
            if (materialAlphaRejected(texColor.a)) discard;
            if (unlit) {
                // UI texels are already display colours. Skip every lighting
                // contribution, including lamps merged in by the renderer.
                fragColor = vec4(texColor.rgb, texColor.a * (1.0 - transparency));
                return;
            }


            vec3 finalColor = texColor.rgb * emissivity;
            // Mostly flat, so dark metal heats up too; a little detail keeps its form.
            vec2 inRect = step(glowRect.xy, texCoord) * step(texCoord, glowRect.zw);
            finalColor += glow * inRect.x * inRect.y
                * (0.6 + 0.4 * dot(texColor.rgb, vec3(0.299, 0.587, 0.114)));

            // Light reaching the surface before albedo, shared with the shine.
            // Models are small: shallow relief.
            vec3 normal = bumpNormal(texture1, texCoord, worldPos, normalize(worldNormal), 0.012);
            vec3 light = ambientLight * ambientIntensity;
            vec3 specular = vec3(0.0);
            float power = shineHighlights() ? shinePower(length(worldNormal)) : SHINE_POWER;
            for (int i = 0; i < 6; i++) {
                light += calculateSpotlight(i, worldPos, normal, vec3(1.0), specular, power);
            }
            finalColor += texColor.rgb * light * litScale;
            vec4 shaded = applyShine(vec4(finalColor, texColor.a * (1.0 - transparency)), light, specular, 1.0 - transparency, texCoord, worldPos, normal);

            // Held weapons are machined metal: they mirror the room, most
            // where the finish is silver (bright, unsaturated texels) and a
            // little where it is dark or painted. Scaled by the light that
            // reaches the gun, so it doesn't glow in a dark corridor.
            if (heldReflection > 0.0 && heldEnvironment) {
                highp vec3 toEye = normalize(fogEyeDensity.xyz - worldPos);
                float lum = dot(texColor.rgb, vec3(0.299, 0.587, 0.114));
                float top = max(texColor.r, max(texColor.g, texColor.b));
                float sat = (top - min(texColor.r, min(texColor.g, texColor.b))) / max(top, 1e-3);
                float metal = mix(0.12, 1.0, smoothstep(0.18, 0.55, lum) * (1.0 - smoothstep(0.12, 0.45, sat)));
                float facing = clamp(abs(dot(normal, toEye)), 0.0, 1.0);
                float fresnel = mix(0.55, 1.0, pow(1.0 - facing, 5.0));
                vec3 mirrored = reflect(-toEye, normal);
                vec3 seen = textureLod(shineEnvironment, environmentDirection(mirrored), 1.0).rgb;
                float lit = clamp(dot(light, vec3(0.299, 0.587, 0.114)) * 1.5, 0.1, 1.0);
                // Metal tints what it reflects with its own colour, and its
                // colour *is* the reflection: blend toward it rather than
                // adding, so a silver slide shows the room instead of
                // clipping to white.
                vec3 tint = mix(vec3(1.0), texColor.rgb / max(top, 1e-3), 0.4);
                vec3 mirror = seen * tint * lit * (0.5 + top);
                float amount = clamp(metal * fresnel * heldReflection * 0.75, 0.0, 0.85);
                shaded.rgb = mix(shaded.rgb, mirror, amount);
            }

            fragColor = applyMaterialPass(applyIncidence(shaded, texColor, worldPos, worldNormal), texColor, worldPos, worldNormal);
            fragColor.rgb *= fogFactor(worldPos);
        }
"#;

struct UnifiedUniforms {
    world_wave: crate::scene::world_wave::Uniforms,
    self_depth: super::self_depth::Uniforms,
    point_shadow: super::point_shadow::Uniforms,
    fog: super::fog::Uniforms,
    bump: super::bump::Uniforms,
    held_reflection: gfx::Uniform,
    held_environment: gfx::Uniform,
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
    additive_unlit_loc: gfx::Uniform,
    unlit_loc: gfx::Uniform,

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

pub struct BasicMaterial<T>
where
    T: Deref<Target = dyn TextureTrait + 'static>,
{
    has_initialized: bool,
    diffuse_texture: T,
    diffuse_override: Option<std::rc::Rc<dyn TextureTrait>>,
    emissivity: f32,
    /// Per-draw multiplier from `Material::set_emissivity_scale`.
    emissivity_scale: f32,
    /// Per-draw flag from `Material::set_full_bright`.
    full_bright: bool,
    /// Per-draw glow from `Material::set_glow`.
    glow: Glow,
    transparency: f32,
    base_transparency: f32,
    additive_unlit: bool,
    fixed_ambient: bool,
    unlit: bool,
    incidence: Option<IncidencePass>,
    render_pass: Option<super::render_pass::RenderPass>,
    shine: Option<Shine>,
}

impl<T> BasicMaterial<T>
where
    T: Deref<Target = dyn TextureTrait>,
{
    pub fn is_transparent(&self) -> bool {
        self.incidence.is_some() || self.additive_unlit || self.transparency > 0.01
    }

    pub fn draw_unified(
        &self,
        render_context: &EngineRenderContext,
        view_matrix: &Matrix4<f32>,
        world_matrix: &Matrix4<f32>,
        lights: &crate::scene::light::LightArray,
    ) {
        let program = if matches!(render_context.self_depth, super::self_depth::Phase::Capture) {
            &SELF_DEPTH_SHADER_PROGRAM
        } else {
            &UNIFIED_SHADER_PROGRAM
        };
        let (shader_program, uniforms) = program.get().expect("unified shader not compiled");
        if let Some(texture) = &self.diffuse_override {
            texture.bind0(render_context);
        } else {
            self.diffuse_texture.bind0(render_context);
        }
        shader_program.bind();
        uniforms.world_wave.bind(render_context.world_wave);
        uniforms.self_depth.bind(render_context.self_depth);
        uniforms.point_shadow.bind(lights, render_context);
        uniforms.fog.bind();
        uniforms.bump.bind();
        let held_environment = lights
            .environment
            .as_ref()
            .filter(|_| lights.held_reflection > 0.0);
        if let Some(environment) = held_environment {
            environment.bind_to(super::shine::ENVIRONMENT_UNIT);
        }
        gfx::set_i32(
            uniforms.held_environment,
            i32::from(held_environment.is_some()),
        );
        gfx::set_f32(uniforms.held_reflection, lights.held_reflection);
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

        // Set material properties
        gfx::set_i32(uniforms.additive_unlit_loc, i32::from(self.additive_unlit));
        gfx::set_i32(uniforms.unlit_loc, i32::from(self.unlit));
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
            if self.fixed_ambient {
                1.0
            } else {
                lights.ambient_intensity(render_context.ambient_light_intensity)
            },
        );

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
impl<T> Material for BasicMaterial<T>
where
    T: Deref<Target = dyn TextureTrait> + 'static,
{
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

    fn set_diffuse_texture(&mut self, texture: std::rc::Rc<dyn TextureTrait>) {
        self.diffuse_override = Some(texture);
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
            // Build and compile unified shader program with 6-spotlight support
            let vertex_shader =
                &[crate::scene::world_wave::GLSL, UNIFIED_VERTEX_SHADER_SOURCE].concat();

            let fragment_shader = &format!(
                "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
                include_str!("flashlight.glsl"),
                super::point_shadow::GLSL,
                super::fog::GLSL,
                super::bump::GLSL,
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

            // Get uniform locations for all shader variables
            let uniforms = UnifiedUniforms {
                world_wave: crate::scene::world_wave::Uniforms::new(&shader),
                self_depth: super::self_depth::Uniforms::new(&shader),
                point_shadow: super::point_shadow::Uniforms::new(&shader),
                fog: super::fog::Uniforms::new(&shader),
                bump: super::bump::Uniforms::new(&shader),
                held_reflection: shader.uniform("heldReflection"),
                held_environment: shader.uniform("heldEnvironment"),
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
                unlit_loc: shader.uniform("unlit"),
                additive_unlit_loc: shader.uniform("additiveUnlit"),
                transparency_loc: shader.uniform("transparency"),

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
        lights: &crate::scene::light::LightArray,
    ) -> bool {
        if !self.is_transparent() {
            self.draw_unified(render_context, view_matrix, world_matrix, lights);
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
        lights: &crate::scene::light::LightArray,
    ) -> bool {
        if self.is_transparent() {
            self.draw_unified(render_context, view_matrix, world_matrix, lights);
            true
        } else {
            false
        }
    }
}

pub fn create<T>(diffuse_texture: T, emissivity: f32, transparency: f32) -> Box<dyn Material>
where
    T: Deref<Target = dyn TextureTrait> + 'static,
{
    create_with_additive(diffuse_texture, emissivity, transparency, false)
}

pub fn create_with_additive<T>(
    diffuse_texture: T,
    emissivity: f32,
    transparency: f32,
    additive_unlit: bool,
) -> Box<dyn Material>
where
    T: Deref<Target = dyn TextureTrait> + 'static,
{
    Box::new(BasicMaterial {
        diffuse_texture,
        has_initialized: false,
        diffuse_override: None,
        emissivity,
        emissivity_scale: 1.0,
        full_bright: false,
        glow: Glow::NONE,
        transparency,
        base_transparency: transparency,
        additive_unlit,
        fixed_ambient: false,
        unlit: false,
        incidence: None,
        render_pass: None,
        shine: None,
    })
}

/// Preserve the ambient baseline for UI, video, and explicitly fullbright surfaces.
/// These materials must stay readable independently of world lighting tuning.
pub fn create_with_fixed_ambient<T>(
    diffuse_texture: T,
    emissivity: f32,
    transparency: f32,
) -> Box<dyn Material>
where
    T: Deref<Target = dyn TextureTrait> + 'static,
{
    Box::new(BasicMaterial {
        diffuse_texture,
        has_initialized: false,
        diffuse_override: None,
        emissivity,
        emissivity_scale: 1.0,
        full_bright: false,
        glow: Glow::NONE,
        transparency,
        base_transparency: transparency,
        additive_unlit: false,
        fixed_ambient: true,
        unlit: false,
        incidence: None,
        render_pass: None,
        shine: None,
    })
}

/// World-space UI, with the same texture colour and alpha as screen-space UI.
/// Emission alone is not unlit: the basic shader also adds scene lighting.
pub fn create_unlit<T>(diffuse_texture: T, transparency: f32) -> Box<dyn Material>
where
    T: Deref<Target = dyn TextureTrait> + 'static,
{
    Box::new(BasicMaterial {
        diffuse_texture,
        has_initialized: false,
        diffuse_override: None,
        emissivity: 1.0,
        emissivity_scale: 1.0,
        full_bright: false,
        glow: Glow::NONE,
        transparency,
        base_transparency: transparency,
        additive_unlit: false,
        fixed_ambient: false,
        unlit: true,
        incidence: None,
        render_pass: None,
        shine: None,
    })
}

/// A depth-tested additive overlay using the same geometry and lighting as
/// the base material. SceneObject supplies SRC_ALPHA/ONE composition.
pub fn create_incidence(
    texture: std::rc::Rc<dyn TextureTrait>,
    pass: IncidencePass,
) -> Box<dyn Material> {
    Box::new(BasicMaterial {
        diffuse_texture: texture,
        has_initialized: false,
        diffuse_override: None,
        emissivity: 0.0,
        emissivity_scale: 1.0,
        full_bright: false,
        glow: Glow::NONE,
        transparency: 0.0,
        base_transparency: 0.0,
        additive_unlit: false,
        fixed_ambient: false,
        unlit: false,
        incidence: Some(pass),
        render_pass: None,
        shine: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shader floors its inverse-distance falloff at a source radius, and
    /// the light selection ranks with the same floor. They are declared in two
    /// languages, so pin them to each other: if they drift, the lights chosen
    /// are not the lights drawn.
    #[test]
    fn shader_source_radius_matches_the_engine_constant() {
        let declared = format!(
            "const float SOURCE_RADIUS = {};",
            crate::scene::light::LIGHT_SOURCE_RADIUS
        );
        for (name, source) in [
            ("basic", UNIFIED_FRAGMENT_SHADER_SOURCE),
            (
                "skinned",
                crate::scene::skinned_material::FRAGMENT_SHADER_SOURCE_FOR_TEST,
            ),
        ] {
            assert!(
                source.contains(&declared),
                "{name} shader must declare `{declared}`"
            );
        }
    }
}

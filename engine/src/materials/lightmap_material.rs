use crate::engine::EngineRenderContext;
use crate::gfx;
use crate::gfx::Program as ShaderProgram;
use crate::scene::Material;
use crate::scene::render_pass::RenderPass;
use crate::scene::shine::{Shine, ShineUniforms};
use crate::scene::uv_motion::UvMotion;
use crate::texture::Texture;
use crate::texture::TextureTrait;

use cgmath::Matrix4;
use cgmath::Vector3;
use once_cell::sync::OnceCell;
use std::any::Any;
use std::rc::Rc;

// Unified shader for single-pass lighting with lightmaps + 6 dynamic spotlights
const UNIFIED_VERTEX_SHADER_SOURCE: &str = r#"
        layout (location = 0) in highp vec3 inPos;
        layout (location = 1) in vec2 inTex;
        layout (location = 2) in vec2 inLightMapTex;
        layout (location = 3) in vec4 inAtlas;
        layout (location = 4) in vec3 inNormal;

        uniform highp mat4 world;
        uniform highp mat4 view;
        uniform highp mat4 projection;

        out vec2 texCoord;
        out highp vec2 lightMapTexCoord;
        out highp vec4 atlasCoord;
        // Preserve sub-pixel light motion on GLES, including far from origin.
        out highp vec3 worldPos;
        out vec3 worldNormal;

        void main() {
            texCoord = inTex;
            lightMapTexCoord = inLightMapTex;
            atlasCoord = inAtlas;

            highp vec4 worldPosition = world * vec4(inPos, 1.0);
            worldPos = worldPosition.xyz;
            worldNormal = normalize(mat3(world) * inNormal);

            gl_Position = projection * view * waveWorldPosition(worldPosition);
        }
"#;

const UNIFIED_FRAGMENT_SHADER_SOURCE: &str = r#"
        out vec4 fragColor;

        in vec2 texCoord;
        in highp vec2 lightMapTexCoord;
        in highp vec4 atlasCoord;
        in highp vec3 worldPos;
        in vec3 worldNormal;

        // Material properties
        uniform sampler2D texture1; // lightmap
        uniform sampler2D texture2; // diffuse texture
        uniform vec3 ambientColor;  // authored mission-wide minimum lighting
        uniform float lightmapIntensity;
        uniform highp vec2 diffuseUvOffset;
        uniform bool unlit;
        uniform vec4 layerColorAlpha;
        uniform bool layerEnabled;
        uniform bool layerAlphaTest;
        // Polished-deck reflections; 0 is off. worldEnvironment says whether
        // the mission's capture is bound to shineEnvironment.
        uniform float worldGloss;
        uniform bool worldEnvironment;

        // Spotlight array uniforms (up to 6 spotlights)
        uniform highp vec3 spotlightPos[6];
        uniform vec4 spotlightColorIntensity[6];  // RGB + intensity
        uniform highp vec3 spotlightDirection[6];
        uniform highp float spotlightInnerAngle[6];
        uniform highp float spotlightOuterAngle[6];
        uniform highp float spotlightRange[6];
        // Slots 0..spotlightCount may hold a light; the rest are empty.
        uniform float spotlightCount;

        // Calculate spotlight contribution
        // One return: the Xbox 360's shader compiler inlines only single-exit
        // functions. Each early-out is a branch around the rest instead.
        vec3 calculateSpotlight(int i, highp vec3 worldPos, vec3 normal, vec3 texColor, inout vec3 specular, float power) {
            vec3 lit = vec3(0.0);
            highp vec3 lightVec = spotlightPos[i] - worldPos;
            highp float distance = length(lightVec);
            // Skip a light with zero intensity, or out of range.
            if (spotlightColorIntensity[i].w > 0.0 && !(distance > spotlightRange[i])) {
                highp vec3 lightDir = lightVec / max(distance, 0.00001);

                // Cone attenuation. A negative inner angle marks a point light -
                // it has no cone, so it lights every direction equally.
                highp float coneAttenuation = 1.0;
                if (spotlightInnerAngle[i] >= 0.0) {
                    highp float spotFactor = dot(-lightDir, normalize(spotlightDirection[i]));
                    coneAttenuation = flashlightCone(spotFactor, cos(spotlightInnerAngle[i]), cos(spotlightOuterAngle[i]));
                }

                // Distance attenuation
                highp float distanceAttenuation = flashlightDistance(distance, spotlightRange[i]);

                // Diffuse lighting
                float lambertian = max(dot(normal, lightDir), 0.0);

                if (shineHighlights()) {
                    vec3 radiance = spotlightColorIntensity[i].rgb * spotlightColorIntensity[i].w
                        * coneAttenuation * distanceAttenuation;
                    specular += shineHighlight(radiance, lightDir, normal, worldPos, power);
                }
                // Combine all factors
                lit = texColor * spotlightColorIntensity[i].rgb * spotlightColorIntensity[i].w
                       * lambertian * coneAttenuation * distanceAttenuation;
            }
            return lit;
        }

        void main() {
            // Sample lightmap and diffuse texture with proper UV wrapping
            float half_pixel = 0.5 / 4096.0;
            float full_pixel = half_pixel * 2.0;
            vec2 wrappedTexCoord = vec2(0.0, 0.0);
            float width = atlasCoord.z - full_pixel;
            float height = atlasCoord.w - full_pixel;

            wrappedTexCoord.x = mod(lightMapTexCoord.x * width, width) + atlasCoord.x + half_pixel;
            wrappedTexCoord.y = mod(lightMapTexCoord.y * height, height) + atlasCoord.y + half_pixel;

            vec4 lightmapColor = texture(texture1, wrappedTexCoord) * lightmapIntensity;
            vec4 diffuseColor = texture(texture2, texCoord + diffuseUvOffset);
            // Before any discard: the bump takes screen-space derivatives.
            vec3 geometric = normalize(worldNormal);
#ifdef SHOCK_XBOX360
            // No relief on the Xbox 360: its bump costs about 12 ms a frame.
            vec3 bumped = geometric;
#else
            vec3 bumped = unlit ? geometric
                : bumpNormal(texture2, texCoord + diffuseUvOffset, worldPos, geometric, 0.04);
#endif
            if (layerAlphaTest && diffuseColor.a * layerColorAlpha.a < 0.1) discard;

            // Dark's mission ambient is a minimum final intensity: preserve
            // brighter authored pixels while keeping unlit surfaces legible.
            vec3 finalColor = max(diffuseColor.rgb * lightmapColor.rgb, ambientColor);

            vec3 normal = geometric;
            // The Xbox 360's GPU takes about 10 ms a draw over these lamp
            // loops, even with each one's uniform off: it has no lamp
            // shadows, bump or gloss.
#ifndef SHOCK_XBOX360
            // An object between a shadow lamp and this surface takes back the
            // share of the baked light that lamp would have put here. World
            // casters are not in this set: the lightmap holds their shadows.
            if (!unlit) {
                highp float baked = max(dot(finalColor, vec3(0.299, 0.587, 0.114)), 0.001);
                // Each loop here sums into a local of its own: xost can't carry
                // into a hardware loop a value an if on a uniform changed.
                highp float kept = 1.0;
                for (int k = 0; k < SHOCK_COUNT(POINT_SHADOW_MAX); k++) {
                    highp vec4 lamp = pointShadowLights[k];
                    if (lamp.w <= 0.0) continue;
                    highp vec3 toLamp = lamp.xyz - worldPos;
                    highp float distance = length(toLamp);
                    if (distance >= lamp.w) continue;
                    float ndl = max(dot(normal, toLamp / max(distance, 0.00001)), 0.0);
                    if (ndl <= 0.0) continue;
                    float shadow = 1.0 - pointShadowVisibility(k, 1, worldPos, normal);
                    if (shadow <= 0.0) continue;
                    vec3 lampLight = diffuseColor.rgb * pointShadowColor[k].rgb * pointShadowColor[k].w
                        * ndl / max(distance, 0.8);
                    // The objects' lamp scale runs dimmer than the baked
                    // light it stands for; doubling the ratio lets a shadow
                    // take back what a player expects (a dark, readable shape).
                    highp float share = clamp(2.0 * dot(lampLight, vec3(0.299, 0.587, 0.114)) / baked, 0.0, 1.0);
                    kept *= 1.0 - pointShadowStrength * shadow * share;
                }
                finalColor = max(finalColor * kept, ambientColor);
            }

            // The lightmap was baked for the flat face; relight the relief
            // against the nearby lamps (and faintly from the eye, so an unlit
            // corridor still shows its texture) by how much the bumped normal
            // turns toward or away from them.
            if (!unlit && bumpStrength > 0.0) {
                highp vec3 toEyeDir = normalize(fogEyeDensity.xyz - worldPos);
                vec3 lightDir = toEyeDir * 0.35;
                for (int k = 0; k < SHOCK_COUNT(POINT_SHADOW_MAX); k++) {
                    highp vec4 lamp = pointShadowLights[k];
                    if (lamp.w <= 0.0) continue;
                    highp vec3 toLamp = lamp.xyz - worldPos;
                    highp float distance = length(toLamp);
                    if (distance >= lamp.w) continue;
                    lightDir += toLamp / max(distance, 0.00001) * (1.0 - distance / lamp.w);
                }
                float weight = length(lightDir);
                if (weight > 1e-3) {
                    vec3 l = lightDir / weight;
                    float relief = 1.0 + 1.6 * (dot(bumped, l) - dot(normal, l));
                    finalColor *= clamp(relief, 0.5, 1.5);
                }
            }

            // Reflections. Decks are polished metal and tile: floors most of
            // all, walls a little. The capture is a lit photograph of the
            // deck, so it is scaled by the light this spot actually gets -
            // a dark corner reflects nothing. Nearby lamps add sharp
            // highlights, which is what reads as a reflective floor.
            if (!unlit && worldGloss > 0.0) {
                highp vec3 toEye = normalize(fogEyeDensity.xyz - worldPos);
                float up = smoothstep(0.5, 0.9, geometric.y);
                float gloss = worldGloss * mix(0.08, 1.0, up);
                float facing = clamp(dot(bumped, toEye), 0.0, 1.0);
                float fresnel = mix(0.12, 1.0, pow(1.0 - facing, 5.0));
                float lit = clamp(dot(lightmapColor.rgb, vec3(0.299, 0.587, 0.114)) * 3.0, 0.0, 1.0);
                vec3 reflected = vec3(0.0);
                if (worldEnvironment) {
                    vec3 mirrored = reflect(-toEye, bumped);
                    reflected += textureLod(shineEnvironment, environmentDirection(mirrored), 1.2).rgb
                        * fresnel * lit;
                }
                vec3 lampGloss = vec3(0.0);
                for (int k = 0; k < SHOCK_COUNT(POINT_SHADOW_MAX); k++) {
                    highp vec4 lamp = pointShadowLights[k];
                    if (lamp.w <= 0.0) continue;
                    highp vec3 toLamp = lamp.xyz - worldPos;
                    highp float distance = length(toLamp);
                    if (distance >= lamp.w) continue;
                    vec3 l = toLamp / max(distance, 0.00001);
                    float ndl = max(dot(bumped, l), 0.0);
                    float spec = pow(max(dot(bumped, normalize(l + toEye)), 0.0), 72.0) * ndl;
                    float falloff = 1.0 - distance / lamp.w;
                    lampGloss += pointShadowColor[k].rgb * pointShadowColor[k].w
                        * spec * falloff * (0.6 + 2.0 * fresnel);
                }
                reflected += lampGloss;
                finalColor += reflected * gloss;
            }

#endif

            // Add dynamic spotlight contributions on top of baked lighting
            vec3 specular = vec3(0.0);
            float power = shineHighlights() ? shinePower(length(worldNormal)) : SHINE_POWER;
            vec3 spotlights = vec3(0.0);
            // Only the slots in use: the Xbox 360's shader compiler runs every
            // branch of a light, so each pass costs a full-screen layer about
            // 10 ms there, lit or not (vulkan4xenon's vkhang, 2026-11-26).
#ifdef SHOCK_XBOX360
            // Only slot 0 (the flashlight, which the mission adds first): a
            // loop that indexes the light arrays by its counter hangs the
            // Xbox 360's GPU, at any count (vulkan4xenon's vkreplay,
            // 2026-11-26), and each full light costs about 16 ms a frame
            // there even when off.
            if (spotlightCount > 0.5)
                spotlights += CONSOLE_FLASHLIGHT(worldPos, bumped, diffuseColor.rgb);
#else
            for (int i = 0; i < int(spotlightCount); i++) {
                spotlights += calculateSpotlight(i, worldPos, bumped, diffuseColor.rgb, specular, power);
            }
#endif
            finalColor += spotlights;

            vec4 base = vec4((unlit ? diffuseColor.rgb : finalColor) * layerColorAlpha.rgb,
                layerEnabled ? diffuseColor.a * layerColorAlpha.a : 1.0);
            // Terrain enables only the light-driven highlight (zero authored
            // sheen passes and no environment reflection). Reuse the object's
            // mask/coverage composition and move that mask with the diffuse.
#ifdef SHOCK_XBOX360
            // No highlight on the Xbox 360: the shine costs about 20 ms a frame there.
            fragColor = base;
#else
            fragColor = applyShine(base, vec3(0.0), specular, 1.0,
                texCoord + diffuseUvOffset, worldPos, normal);
#endif
            if (!unlit) fragColor.rgb *= fogFactor(worldPos);
        }
"#;

struct UnifiedUniforms {
    shine: ShineUniforms,
    point_shadow: crate::scene::point_shadow::Uniforms,
    fog: crate::scene::fog::Uniforms,
    bump: crate::scene::bump::Uniforms,
    world_wave: crate::scene::world_wave::Uniforms,
    uv_offset_loc: gfx::Uniform,
    unlit_loc: gfx::Uniform,
    layer_color_alpha_loc: gfx::Uniform,
    layer_enabled_loc: gfx::Uniform,
    layer_alpha_test_loc: gfx::Uniform,
    world_environment_loc: gfx::Uniform,
    // Basic transformation matrices
    world_loc: gfx::Uniform,
    view_loc: gfx::Uniform,
    projection_loc: gfx::Uniform,

    // Texture samplers
    texture1_loc: gfx::Uniform, // lightmap
    texture2_loc: gfx::Uniform, // diffuse
    ambient_color_loc: gfx::Uniform,
    lightmap_intensity_loc: gfx::Uniform,

    // Spotlight array uniforms (6 spotlights)
    spotlight_pos_loc: [gfx::Uniform; 6],
    spotlight_color_intensity_loc: [gfx::Uniform; 6],
    spotlight_direction_loc: [gfx::Uniform; 6],
    spotlight_inner_angle_loc: [gfx::Uniform; 6],
    spotlight_outer_angle_loc: [gfx::Uniform; 6],
    spotlight_range_loc: [gfx::Uniform; 6],
    spotlight_count_loc: gfx::Uniform,
}

static UNIFIED_SHADER_PROGRAM: OnceCell<(ShaderProgram, UnifiedUniforms)> = OnceCell::new();
/// The Xbox 360's lamp-shadow pass: how much of each world pixel a moving
/// caster's shadow keeps, multiplied into the world drawn already, inside
/// the shadows' screen rectangle alone (`begin_shadow_pass`). The world
/// shader can't take the shadow itself there: as both arms of its branches
/// run, it cost every world pixel about 40 ms a frame. Built everywhere so
/// the shader sweep (xbox360/tools/shader_sweep.sh) records it.
static SHADOW_PASS_PROGRAM: OnceCell<(ShaderProgram, ShadowPassUniforms)> = OnceCell::new();

struct ShadowPassUniforms {
    point_shadow: crate::scene::point_shadow::Uniforms,
    world_wave: crate::scene::world_wave::Uniforms,
    world_loc: gfx::Uniform,
    view_loc: gfx::Uniform,
    projection_loc: gfx::Uniform,
}

/// [`SHADOW_PASS_PROGRAM`]'s: the lamp's one shadow (point_shadow's
/// Xbox 360 receiver), as a factor. It shades by the lamp's angle and
/// range alone, not the bake under it as the desktop does, which would
/// take the lightmap and diffuse reads this pass is spared.
const SHADOW_PASS_FRAGMENT_SHADER_SOURCE: &str = r#"
        out vec4 fragColor;
        in highp vec3 worldPos;
        in vec3 worldNormal;
        // No discard (nor the texture read an alpha test takes): it would
        // turn off early depth rejection, and this pass redraws every world
        // draw over pixels the base pass already decided. Layered draws,
        // the only alpha-tested ones, take no shadow pass.
        void main() {
            vec3 normal = worldNormal * inversesqrt(max(dot(worldNormal, worldNormal), 1e-10));
            highp vec4 lamp = pointShadowLights[0];
            highp vec3 toLamp = lamp.xyz - worldPos;
            highp float distance = length(toLamp);
            float ndl = max(dot(normal, toLamp / max(distance, 0.00001)), 0.0);
            // In an if: straight-line, with no discard, this shader sends
            // xost into unbounded recursion ("maximum recursion depth").
            float shadow = 0.0;
            if (distance < lamp.w) shadow = 1.0 - pointShadowVisibility(0, 1, worldPos, normal);
            // Strongest facing the lamp near it, fading out toward its range.
            float facing = smoothstep(0.0, 0.3, ndl);
            float fade = 1.0 - smoothstep(0.5 * lamp.w, lamp.w, distance);
            float reaches = step(0.0001, lamp.w) * step(distance, lamp.w);
            float kept = 1.0 - pointShadowStrength * 0.75 * shadow * facing * fade * reaches;
            fragColor = vec4(kept, kept, kept, 1.0);
        }
"#;

/// Depth only, for shadow maps: no lighting, and above all no sampling of
/// the shadow atlas being drawn into. Alpha-tested layers (grates) keep
/// their holes.
const CAPTURE_FRAGMENT_SHADER_SOURCE: &str = r#"
        out vec4 fragColor;
        in vec2 texCoord;
        uniform sampler2D texture2;
        uniform highp vec2 diffuseUvOffset;
        uniform vec4 layerColorAlpha;
        uniform bool layerAlphaTest;
        void main() {
            vec4 diffuseColor = texture(texture2, texCoord + diffuseUvOffset);
            if (layerAlphaTest && diffuseColor.a * layerColorAlpha.a < 0.1) discard;
            fragColor = vec4(0.0);
        }
"#;

struct CaptureUniforms {
    world_wave: crate::scene::world_wave::Uniforms,
    world_loc: gfx::Uniform,
    view_loc: gfx::Uniform,
    projection_loc: gfx::Uniform,
    texture2_loc: gfx::Uniform,
    uv_offset_loc: gfx::Uniform,
    layer_color_alpha_loc: gfx::Uniform,
    layer_alpha_test_loc: gfx::Uniform,
}

static CAPTURE_SHADER_PROGRAM: OnceCell<(ShaderProgram, CaptureUniforms)> = OnceCell::new();

pub struct LightmapMaterial {
    shine: Option<Shine>,
    layer: Option<RenderPass>,
    uv_motion: UvMotion,
    unlit: bool,
    has_initialized: bool,
    lightmap_texture: Rc<Texture>,
    diffuse_texture: Rc<dyn TextureTrait>,
    ambient_color: Vector3<f32>,
}

impl LightmapMaterial {
    pub fn create(
        lightmap_texture: Rc<Texture>,
        diffuse_texture: Rc<dyn TextureTrait>,
        ambient_color: Vector3<f32>,
    ) -> Box<dyn Material> {
        Self::create_with_uv(
            lightmap_texture,
            diffuse_texture,
            ambient_color,
            UvMotion::None,
            false,
        )
    }

    pub fn create_with_uv(
        lightmap_texture: Rc<Texture>,
        diffuse_texture: Rc<dyn TextureTrait>,
        ambient_color: Vector3<f32>,
        uv_motion: UvMotion,
        unlit: bool,
    ) -> Box<dyn Material> {
        Box::new(LightmapMaterial {
            shine: None,
            layer: None,
            uv_motion,
            unlit,
            diffuse_texture,
            lightmap_texture,
            ambient_color,
            has_initialized: false,
        })
    }

    fn draw_capture(
        &self,
        render_context: &EngineRenderContext,
        view_matrix: &Matrix4<f32>,
        world_matrix: &Matrix4<f32>,
    ) {
        let (shader_program, uniforms) = CAPTURE_SHADER_PROGRAM
            .get()
            .expect("capture shader not compiled");
        self.diffuse_texture.bind1(render_context);
        shader_program.bind();
        uniforms.world_wave.bind(render_context.world_wave);
        gfx::set_mat4(uniforms.world_loc, world_matrix);
        gfx::set_mat4(uniforms.view_loc, view_matrix);
        gfx::set_mat4(uniforms.projection_loc, &render_context.projection_matrix);
        gfx::set_i32(uniforms.texture2_loc, 1);
        let uv = self.uv_motion.offset(render_context.time);
        gfx::set_vec2(uniforms.uv_offset_loc, [uv[0], uv[1]]);
        let alpha = self.layer.as_ref().map_or(1.0, |p| p.alpha);
        gfx::set_vec4(uniforms.layer_color_alpha_loc, [1.0, 1.0, 1.0, alpha]);
        gfx::set_i32(
            uniforms.layer_alpha_test_loc,
            i32::from(self.layer.as_ref().is_some_and(|p| p.alpha_test)),
        );
    }

    pub fn draw_unified(
        &self,
        render_context: &EngineRenderContext,
        view_matrix: &Matrix4<f32>,
        world_matrix: &Matrix4<f32>,
        lights: &crate::scene::light::LightArray,
    ) {
        if matches!(
            render_context.self_depth,
            crate::scene::self_depth::Phase::Capture
        ) {
            self.draw_capture(render_context, view_matrix, world_matrix);
            return;
        }
        let (shader_program, uniforms) = UNIFIED_SHADER_PROGRAM
            .get()
            .expect("unified shader not compiled");

        // Bind textures
        crate::texture::bind0(&self.lightmap_texture);
        self.diffuse_texture.bind1(render_context);

        shader_program.bind();
        uniforms.world_wave.bind(render_context.world_wave);
        uniforms.point_shadow.bind(lights, render_context);
        uniforms.fog.bind();
        uniforms.bump.bind();
        let environment = lights.environment.as_ref();
        if let Some(environment) = environment {
            environment.bind_to(crate::scene::shine::ENVIRONMENT_UNIT);
        }
        gfx::set_i32(
            uniforms.world_environment_loc,
            i32::from(environment.is_some()),
        );
        let strength = render_context.terrain_wetness;
        let shine = self
            .shine
            .as_ref()
            .filter(|_| strength > 0.0 && !self.unlit);
        uniforms.shine.bind_with_strengths(
            shine,
            lights,
            render_context,
            view_matrix,
            (strength, 0.0),
        );
        let uv = self.uv_motion.offset(render_context.time);
        gfx::set_vec2(uniforms.uv_offset_loc, [uv[0], uv[1]]);
        gfx::set_i32(uniforms.unlit_loc, i32::from(self.unlit));
        let (color, alpha) = self
            .layer
            .as_ref()
            .map_or((Vector3::new(1.0, 1.0, 1.0), 1.0), |p| (p.color, p.alpha));
        gfx::set_vec4(
            uniforms.layer_color_alpha_loc,
            [color.x, color.y, color.z, alpha],
        );
        gfx::set_i32(uniforms.layer_enabled_loc, i32::from(self.layer.is_some()));
        gfx::set_i32(
            uniforms.layer_alpha_test_loc,
            i32::from(self.layer.as_ref().is_some_and(|p| p.alpha_test)),
        );

        let projection = render_context.projection_matrix;

        // Set basic transformation matrices
        gfx::set_mat4(uniforms.world_loc, &world_matrix);
        gfx::set_mat4(uniforms.view_loc, &view_matrix);
        gfx::set_mat4(uniforms.projection_loc, &projection);

        // Set texture samplers
        gfx::set_i32(uniforms.texture1_loc, 0); // lightmap
        gfx::set_i32(uniforms.texture2_loc, 1); // diffuse
        gfx::set_f32(
            uniforms.lightmap_intensity_loc,
            render_context.level_light_intensity,
        );
        gfx::set_vec3(
            uniforms.ambient_color_loc,
            [
                self.ambient_color.x * render_context.ambient_light_intensity,
                self.ambient_color.y * render_context.ambient_light_intensity,
                self.ambient_color.z * render_context.ambient_light_intensity,
            ],
        );

        // Set spotlight array uniforms
        let used = (0..6)
            .rposition(|i| lights.get_light(i).is_some())
            .map_or(0, |last| last + 1);
        gfx::set_f32(uniforms.spotlight_count_loc, used as f32);
        // The Xbox 360's shaders read slot 0 alone (CONSOLE_FLASHLIGHT): the
        // rest would be thousands of wasted uniform writes a frame there.
        for i in 0..crate::scene::light::SHADER_SPOTLIGHT_SLOTS {
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

impl Material for LightmapMaterial {
    fn set_shine(&mut self, shine: Shine) {
        self.shine = Some(shine);
    }
    fn set_render_pass(&mut self, pass: RenderPass) {
        // Terrain's resolver admits only this shader's color/alpha subset.
        debug_assert!(
            pass.incidence.is_none()
                && pass.cube.is_none()
                && !pass.force_opaque
                && pass.mipmap_bias == 0.0
        );
        self.layer = Some(pass);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn has_initialized(&self) -> bool {
        self.has_initialized
    }

    fn initialize(&mut self, _is_opengl_es: bool) {
        let _ = CAPTURE_SHADER_PROGRAM.get_or_init(|| {
            let vertex_shader =
                &[crate::scene::world_wave::GLSL, UNIFIED_VERTEX_SHADER_SOURCE].concat();
            let shader = gfx::Program::new(vertex_shader, CAPTURE_FRAGMENT_SHADER_SOURCE);
            let uniforms = CaptureUniforms {
                world_wave: crate::scene::world_wave::Uniforms::new(&shader),
                world_loc: shader.uniform("world"),
                view_loc: shader.uniform("view"),
                projection_loc: shader.uniform("projection"),
                texture2_loc: shader.uniform("texture2"),
                uv_offset_loc: shader.uniform("diffuseUvOffset"),
                layer_color_alpha_loc: shader.uniform("layerColorAlpha"),
                layer_alpha_test_loc: shader.uniform("layerAlphaTest"),
            };
            (shader, uniforms)
        });
        let _ = UNIFIED_SHADER_PROGRAM.get_or_init(|| {
            // Build and compile unified shader program with lightmaps + 6-spotlight support
            let vertex_shader =
                &[crate::scene::world_wave::GLSL, UNIFIED_VERTEX_SHADER_SOURCE].concat();

            let fragment_shader = &format!(
                "{}\n{}\n{}\n{}\n{}\n{}\n{}",
                crate::scene::point_shadow::GLSL,
                crate::scene::fog::GLSL,
                crate::scene::bump::GLSL,
                crate::scene::environment::GLSL,
                crate::scene::shine::GLSL,
                include_str!("../scene/flashlight.glsl"),
                UNIFIED_FRAGMENT_SHADER_SOURCE
            );

            let shader = gfx::Program::new(&vertex_shader, &fragment_shader);

            // Get uniform locations for all shader variables
            let uniforms = UnifiedUniforms {
                shine: ShineUniforms::new(&shader),
                point_shadow: crate::scene::point_shadow::Uniforms::new(&shader),
                fog: crate::scene::fog::Uniforms::new(&shader),
                bump: crate::scene::bump::Uniforms::new(&shader),
                layer_color_alpha_loc: shader.uniform("layerColorAlpha"),
                layer_enabled_loc: shader.uniform("layerEnabled"),
                layer_alpha_test_loc: shader.uniform("layerAlphaTest"),
                world_environment_loc: shader.uniform("worldEnvironment"),
                uv_offset_loc: shader.uniform("diffuseUvOffset"),
                unlit_loc: shader.uniform("unlit"),
                world_wave: crate::scene::world_wave::Uniforms::new(&shader),
                // Basic transformation matrices
                world_loc: shader.uniform("world"),
                view_loc: shader.uniform("view"),
                projection_loc: shader.uniform("projection"),

                // Texture samplers
                texture1_loc: shader.uniform("texture1"),
                texture2_loc: shader.uniform("texture2"),
                ambient_color_loc: shader.uniform("ambientColor"),
                lightmap_intensity_loc: shader.uniform("lightmapIntensity"),

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
                spotlight_count_loc: shader.uniform("spotlightCount"),
            };
            (shader, uniforms)
        });
        let _ = SHADOW_PASS_PROGRAM.get_or_init(|| {
            let vertex_shader =
                &[crate::scene::world_wave::GLSL, UNIFIED_VERTEX_SHADER_SOURCE].concat();
            let fragment_shader = &[
                crate::scene::point_shadow::GLSL,
                SHADOW_PASS_FRAGMENT_SHADER_SOURCE,
            ]
            .concat();
            let shader = gfx::Program::new(vertex_shader, fragment_shader);
            let uniforms = ShadowPassUniforms {
                point_shadow: crate::scene::point_shadow::Uniforms::new(&shader),
                world_wave: crate::scene::world_wave::Uniforms::new(&shader),
                world_loc: shader.uniform("world"),
                view_loc: shader.uniform("view"),
                projection_loc: shader.uniform("projection"),
            };
            (shader, uniforms)
        });

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
        self.draw_unified(render_context, view_matrix, world_matrix, lights);
        true
    }

    fn begin_shadow_pass(
        &self,
        render_context: &EngineRenderContext,
        view_matrix: &Matrix4<f32>,
        world_matrix: &Matrix4<f32>,
        lights: &crate::scene::light::LightArray,
    ) -> bool {
        // A layer over the base would take the shadow twice.
        if self.layer.is_some() {
            return false;
        }
        let Some(viewport) = gfx::vk_backend::current_viewport() else {
            return false;
        };
        let view_projection = render_context.projection_matrix * view_matrix;
        let Some([x, y, w, h]) =
            crate::scene::point_shadow::receiver_scissor(&view_projection, viewport)
        else {
            return false;
        };
        let (shader_program, uniforms) = SHADOW_PASS_PROGRAM
            .get()
            .expect("shadow pass shader not compiled");
        shader_program.bind();
        uniforms.world_wave.bind(render_context.world_wave);
        uniforms.point_shadow.bind(lights, render_context);
        gfx::set_mat4(uniforms.world_loc, world_matrix);
        gfx::set_mat4(uniforms.view_loc, view_matrix);
        gfx::set_mat4(uniforms.projection_loc, &render_context.projection_matrix);
        gfx::enable(gfx::Cap::ScissorTest);
        gfx::scissor(x, y, w, h);
        // The pixel times what the shadow keeps of it; alpha left alone.
        gfx::enable(gfx::Cap::Blend);
        gfx::blend_func_separate(
            gfx::BlendFactor::DstColor,
            gfx::BlendFactor::Zero,
            gfx::BlendFactor::Zero,
            gfx::BlendFactor::One,
        );
        // The base pass wrote these pixels' depth: shade only those.
        gfx::depth_mask(false);
        gfx::depth_func(gfx::DepthFunc::LessEqual);
        true
    }

    fn draw_transparent(
        &self,
        _render_context: &EngineRenderContext,
        _view_matrix: &Matrix4<f32>,
        _world_matrix: &Matrix4<f32>,
        _skinning_data: &[Matrix4<f32>],
        _lights: &crate::scene::light::LightArray,
    ) -> bool {
        // Lightmap materials are typically opaque
        false
    }
}

#[cfg(test)]
mod tests {
    use super::UNIFIED_FRAGMENT_SHADER_SOURCE;

    #[test]
    fn world_shader_applies_authored_ambient_as_a_floor() {
        assert!(
            UNIFIED_FRAGMENT_SHADER_SOURCE
                .contains("max(diffuseColor.rgb * lightmapColor.rgb, ambientColor)"),
            "world pixels must not render below the mission ambient"
        );
        assert!(
            !UNIFIED_FRAGMENT_SHADER_SOURCE.contains("+ ambientColor"),
            "ambient is a floor, not an additive wash over authored lighting"
        );
    }
}

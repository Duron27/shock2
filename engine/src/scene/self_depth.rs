//! Private self-occlusion for translucent actors. Nothing here writes scene depth.
use super::{RenderLayer, scene::Scene};
use crate::{engine::EngineRenderContext, gfx, renderer::Renderer};
use cgmath::Matrix4;

/// Explicit phases prevent the scratch pass from sampling its own attachment.
#[derive(Clone, Copy, Default)]
pub enum Phase {
    #[default]
    None,
    Capture,
    Test(Mask),
}

/// Bound only for the actor whose nearest surface is in the scratch texture.
#[derive(Clone, Copy)]
pub struct Mask {
    /// The depth image's backend texture id.
    texture: u32,
    origin: [i32; 2],
}

// Units 0–1 hold diffuse/pass textures, 2–5 shine maps, and 7 the environment cube.
const TEXTURE_UNIT: u32 = 6;

pub(crate) const GLSL: &str = r#"
uniform bool selfDepthEnabled;
#ifdef MULTIVIEW
uniform highp sampler2DArray selfDepthTexture;
#else
uniform highp sampler2D selfDepthTexture;
#endif
uniform ivec2 selfDepthOrigin;
// One return: the Xbox 360's shader compiler inlines only single-exit functions.
bool hiddenBySelfDepth() {
    bool hidden = false;
    if (selfDepthEnabled) {
        ivec2 texel = ivec2(gl_FragCoord.xy) - selfDepthOrigin;
#ifdef MULTIVIEW
        highp float nearest = texelFetch(selfDepthTexture, ivec3(texel, int(gl_ViewID_OVR)), 0).r;
#else
        highp float nearest = texelFetch(selfDepthTexture, texel, 0).r;
#endif
        // One depth24 quantization step, plus rounding between the two draws.
        hidden = gl_FragCoord.z > nearest + 0.00000012;
    }
    return hidden;
}
"#;

/// Shares the material's vertex shader and texture sampling, but never executes
/// lighting or the color shader's control flow while capturing depth.
pub(crate) const FRAGMENT: &str = r#"
out vec4 fragColor;
in vec2 texCoord;
in highp vec3 worldPos;
in vec3 worldNormal;
uniform sampler2D texture1;
void main() {
    vec4 texel = sampleMaterialPass(texture1, texCoord, worldPos, worldNormal);
    if (materialAlphaRejected(texel.a) || texel.a <= 0.0
        || (materialPassEnabled && materialPassAlpha <= 0.0)) discard;
    fragColor = vec4(0.0);
}
"#;

pub(crate) struct Uniforms {
    enabled: gfx::Uniform,
    texture: gfx::Uniform,
    origin: gfx::Uniform,
}

impl Uniforms {
    pub fn new(program: &gfx::Program) -> Self {
        Self {
            enabled: program.uniform("selfDepthEnabled"),
            texture: program.uniform("selfDepthTexture"),
            origin: program.uniform("selfDepthOrigin"),
        }
    }

    pub fn bind(&self, phase: Phase) {
        let mask = match phase {
            Phase::Test(mask) => Some(mask),
            _ => None,
        };
        gfx::set_i32(self.enabled, i32::from(mask.is_some()));
        // With no mask, point at diffuse unit 0, never at the image being
        // rendered into. Even an untaken shader branch must avoid feedback.
        gfx::set_i32(
            self.texture,
            if mask.is_some() {
                TEXTURE_UNIT as i32
            } else {
                0
            },
        );
        if let Some(mask) = mask {
            gfx::vk_backend::bind_texture_id(TEXTURE_UNIT, Some(mask.texture));
            gfx::set_ivec2(self.origin, mask.origin);
        }
    }
}

/// The scratch capture: the backend pauses the frame's pass, captures depth
/// only (no colour, depth write, LESS) at the viewport's size into a reused
/// image, then resumes the frame.
#[derive(Default)]
pub(crate) struct Target;

impl Target {
    pub fn render(
        &mut self,
        group: u64,
        layer: RenderLayer,
        engine: &Renderer,
        context: &EngineRenderContext,
        view: &Matrix4<f32>,
        scene: &Scene,
    ) -> Mask {
        use gfx::vk_backend;
        let origin = vk_backend::begin_depth_capture().unwrap_or_default();
        let context = EngineRenderContext {
            self_depth: Phase::Capture,
            ..*context
        };
        for object in scene
            .objects_in_layer(layer)
            .filter(|object| object.self_depth_group == Some(group))
        {
            if let Some(depth) = object.self_depth_object() {
                depth.draw_opaque(engine, &context, view, scene.lights());
            }
        }
        Mask {
            texture: vk_backend::end_depth_capture().unwrap_or(0),
            origin,
        }
    }
}

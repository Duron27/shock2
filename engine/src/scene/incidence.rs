//! An optional view-angle lookup for an additive material pass. Shared by
//! static and skinned shaders so the ramp has one interpretation.
use crate::gfx;
use crate::{engine::EngineRenderContext, texture::TextureTrait};
use cgmath::{Matrix4, Vector3};
use std::rc::Rc;

#[derive(Clone)]
pub struct IncidencePass {
    pub ramp: Rc<dyn TextureTrait>,
    pub tint: Vector3<f32>,
    pub unlit: bool,
    /// Flat decals can carry vertex normals unrelated to their authored face.
    /// Derivatives recover that face without changing the base material.
    pub geometric_normal: bool,
}

pub(crate) const GLSL: &str = r#"
uniform bool incidenceEnabled;
uniform sampler2D incidenceRamp;
uniform vec3 incidenceEye;
uniform vec3 incidenceTint;
uniform bool incidenceUnlit;
uniform bool incidenceGeometricNormal;
// One return: the Xbox 360's shader compiler inlines only single-exit functions.
vec4 applyIncidence(vec4 lit, vec4 texel, vec3 position, vec3 normal) {
    vec4 result = lit;
    if (incidenceEnabled) {
        vec3 surfaceNormal = incidenceGeometricNormal ? cross(dFdx(position), dFdy(position)) : normal;
        float facing = clamp(abs(dot(normalize(surfaceNormal), normalize(incidenceEye - position))), 0.0, 1.0);
        float alpha = texture(incidenceRamp, vec2(facing, 0.5)).r;
        result = vec4((incidenceUnlit ? texel.rgb : lit.rgb) * incidenceTint, lit.a * alpha);
    }
    return result;
}
"#;

pub(crate) struct IncidenceUniforms {
    enabled: gfx::Uniform,
    ramp: gfx::Uniform,
    eye: gfx::Uniform,
    tint: gfx::Uniform,
    unlit: gfx::Uniform,
    geometric_normal: gfx::Uniform,
}
impl IncidenceUniforms {
    pub fn new(program: &gfx::Program) -> Self {
        let loc = |name: &str| program.uniform(name);
        Self {
            enabled: loc("incidenceEnabled"),
            ramp: loc("incidenceRamp"),
            eye: loc("incidenceEye"),
            tint: loc("incidenceTint"),
            unlit: loc("incidenceUnlit"),
            geometric_normal: loc("incidenceGeometricNormal"),
        }
    }
    pub fn bind(
        &self,
        pass: Option<&IncidencePass>,
        context: &EngineRenderContext,
        view: &Matrix4<f32>,
    ) {
        gfx::set_i32(self.enabled, i32::from(pass.is_some()));
        gfx::set_i32(self.ramp, 1);
        if let Some(pass) = pass {
            pass.ramp.bind1(context);
            gfx::set_vec3(self.eye, crate::scene::material::eye_position(view).into());
            gfx::set_vec3(self.tint, *AsRef::<[f32; 3]>::as_ref(&pass.tint));
            gfx::set_i32(self.unlit, i32::from(pass.unlit));
            gfx::set_i32(self.geometric_normal, i32::from(pass.geometric_normal));
        }
    }
}

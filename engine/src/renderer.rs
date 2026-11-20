//! Draws a [`Scene`] through `gfx`: opaque then transparent per
//! composition layer, with self-depth masks for translucent actors.

use crate::util;
use std::sync::Arc;

pub struct Renderer {
    self_depth: std::cell::RefCell<crate::scene::self_depth::Target>,
    pub is_opengl_es: bool,
    pub storage: Arc<dyn crate::file_system::Storage>,
}

impl Renderer {
    /// Draws `objects` in explicit composition-layer order: opaque then
    /// transparent per layer. `self_depth_scene` enables self-occluding
    /// translucent groups;
    unsafe fn draw_layers(
        &self,
        render_context: &EngineRenderContext,
        view: &cgmath::Matrix4<f32>,
        objects: &[crate::scene::SceneObject],
        lights: &crate::scene::light::LightArray,
        self_depth_scene: Option<&Scene>,
    ) {
        let in_layer = |layer: RenderLayer| {
            objects
                .iter()
                .filter(move |object| object.render_layer() == layer)
        };
        // Composition is keyed explicitly: debug/desktop append per-eye
        // objects while Oculus prepends them, but both must render world,
        // scene effects, scene UI, then system UI. Each non-world layer
        // starts with one depth clear at its group boundary, never from an
        // individual object mid-pass.
        for layer in RenderLayer::ORDERED {
            if in_layer(layer).next().is_none() {
                continue;
            }
            if layer.clears_depth() {
                crate::gfx::clear(false, true);
            }

            // SINGLE-PASS LIGHTING: Opaque pass with all lighting calculated
            // in shaders.
            in_layer(layer).for_each(|s| {
                // An object's own lights don't replace the scene's - the
                // player's hand lights live there, and a torch pointed at a
                // prop has to light it.
                let merged = s.lights().map(|o| o.merged_with(lights));
                s.draw_opaque(
                    self,
                    render_context,
                    view,
                    merged.as_ref().unwrap_or(lights),
                )
            });

            // Transparent pass with all lighting calculated in shaders
            crate::gfx::depth_mask(false);
            // Preserve authored composition order. A mask belongs to one
            // actor only; rebuilding for another actor never touches scene depth.
            let mut cached_mask = None;
            for s in in_layer(layer) {
                let mask = s
                    .self_depth_group
                    .zip(self_depth_scene)
                    .map(|(group, scene)| {
                        if cached_mask.is_none_or(|(id, _)| id != group) {
                            cached_mask = Some((
                                group,
                                self.self_depth.borrow_mut().render(
                                    group,
                                    layer,
                                    self,
                                    render_context,
                                    view,
                                    scene,
                                ),
                            ));
                        }
                        cached_mask.unwrap().1
                    });
                let context = EngineRenderContext {
                    self_depth: mask.map_or(
                        crate::scene::self_depth::Phase::None,
                        crate::scene::self_depth::Phase::Test,
                    ),
                    ..*render_context
                };
                let merged = s.lights().map(|o| o.merged_with(lights));
                s.draw_transparent(self, &context, view, merged.as_ref().unwrap_or(lights));
            }
            crate::gfx::depth_mask(true);
        }
    }
}

fn init_with(is_opengl_es: bool, storage: Arc<dyn crate::file_system::Storage>) -> Renderer {
    Renderer {
        self_depth: Default::default(),
        is_opengl_es,
        storage,
    }
}

use crate::engine::Engine;
use crate::engine::EngineRenderContext;
use crate::scene::{RenderLayer, scene::Scene};

/// Depth and blend state shared by every pass.
unsafe fn set_render_state() {
    use crate::gfx::{self, BlendFactor as F, Cap};
    gfx::enable(Cap::DepthTest);
    gfx::enable(Cap::Blend);
    // Shaders emit straight RGB; the framebuffer accumulates premultiplied
    // RGB, keeping coverage alpha rather than multiplying alpha by itself.
    gfx::blend_func_separate(
        F::SrcAlpha,
        F::OneMinusSrcAlpha,
        F::One,
        F::OneMinusSrcAlpha,
    );
}

impl Engine for Renderer {
    fn get_storage(&self) -> Arc<dyn crate::file_system::Storage> {
        Arc::clone(&self.storage)
    }

    fn render(&self, render_context: &EngineRenderContext, scene: &Scene) {
        unsafe {
            set_render_state();
            crate::gfx::clear_color([0.0, 0.0, 0.0, 0.0]);
            crate::gfx::clear(true, true);
            // Shadow depth first: it pauses the frame's pass, and the world
            // layer's receivers sample it. Bound until this frame is drawn.
            let _shadows =
                crate::scene::point_shadow::render_atlas(self, render_context, scene);
            let view = util::compute_view_matrix_from_render_context(render_context);
            crate::scene::fog::set_frame(&view, scene.fog_density);
            self.draw_layers(
                render_context,
                &view,
                &scene.objects,
                scene.lights(),
                Some(scene),
            );
        }
    }
}

pub fn init() -> Renderer {
    let bundle_file_system = Box::new(crate::file_system::DefaultFileSystem {
        root_path: Box::new(std::path::Path::new("./assets/")),
    });
    init_with(false, crate::file_system::storage::init(bundle_file_system))
}

use engine::{assets::asset_cache::AssetCache, scene::SceneObject};
use shipyard::{EntityId, World};

use std::sync::Arc;

use super::CullingInfo;
use crate::mission::SpatialQueryEngine;

pub trait VisibilityEngine {
    /// Start working out what the next render sees from this camera. The
    /// work may finish on another thread; [`resolve`](Self::resolve) collects it.
    fn prepare(
        &mut self,
        _spatial_data: Option<&Arc<dyn SpatialQueryEngine>>,
        _culling_info: &CullingInfo,
    ) {
    }

    /// Before the render's first [`is_visible`](Self::is_visible): finish what
    /// [`prepare`](Self::prepare) started, placing the entities where they
    /// are now.
    fn resolve(&mut self, _spatial_data: Option<&dyn SpatialQueryEngine>, _world: &World) {}

    fn is_visible(&mut self, entity_id: EntityId) -> bool;

    fn debug_render(&self, _asset_cache: &mut AssetCache) -> Vec<SceneObject> {
        Vec::new()
    }
}

pub struct AlwaysVisible;

impl VisibilityEngine for AlwaysVisible {
    fn is_visible(&mut self, _entity_id: EntityId) -> bool {
        true
    }
}

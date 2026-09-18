//! A moving tool gesture uses one stable part's authored grid throughout its life.

use bevy::prelude::*;
use mechanic_core::{ConstructionFrame, ConstructionFrameId, ConstructionGraph, PartId};

use crate::{AppSimulation, EditorGraph, EditorState, PlacedBearing};

#[derive(Clone, Copy, Debug)]
pub(crate) struct EditContext {
    pub(crate) anchor: PartId,
    pub(crate) frame: ConstructionFrameId,
    pub(crate) frame_to_world: ConstructionFrame,
}

impl EditContext {
    pub(crate) fn resolve(
        graph: &ConstructionGraph,
        simulation: &AppSimulation,
        anchor: PartId,
    ) -> Option<Self> {
        let frame = graph.part_frame_id(anchor)?;
        let authored = graph.part_frame(anchor)?;
        let frame_to_world = if let Some(creation) = simulation.creation.as_ref() {
            let body = creation
                .part_to_compound
                .iter()
                .find(|(part, _)| *part == anchor)?
                .1 as usize;
            let pose = simulation.transforms.get(body)?;
            let initial = &creation.compounds[body];
            let world_rotation =
                Quat::from_array(pose.rotation) * initial.root_rotation.conjugate();
            let world_translation =
                Vec3::from_slice(&pose.position[..3]) - world_rotation * initial.root_translation;
            ConstructionFrame::new(world_translation, world_rotation)
                .ok()?
                .compose(authored)
        } else {
            authored
        };
        Some(Self {
            anchor,
            frame,
            frame_to_world,
        })
    }

    pub(crate) fn accepts_part(
        &self,
        graph: &ConstructionGraph,
        simulation: &AppSimulation,
        part: PartId,
    ) -> bool {
        let Some(frame) = graph.part_frame_id(part) else {
            return false;
        };
        let Some(creation) = simulation.creation.as_ref() else {
            return frame == self.frame;
        };
        let body_for = |part| {
            creation
                .part_to_compound
                .iter()
                .find_map(|&(id, body)| (id == part).then_some(body))
        };
        let Some(body) = body_for(part) else {
            return frame == self.frame;
        };
        body_for(self.anchor) == Some(body)
    }

    pub(crate) fn ray(self, ray: Ray3d) -> Ray3d {
        let inverse = self.frame_to_world.inverse();
        Ray3d::new(
            inverse.point(ray.origin),
            Dir3::new(inverse.vector(ray.direction.as_vec3()))
                .expect("rigid frame preserves a ray direction"),
        )
    }
}

pub(crate) fn transform_bearing(
    mut bearing: PlacedBearing,
    frame: ConstructionFrame,
) -> PlacedBearing {
    bearing.anchor = frame.point(bearing.anchor);
    bearing.axis = frame.vector(bearing.axis);
    if let mechanic_core::BearingKind::Linear(rail) = &mut bearing.kind {
        rail.mount_normal = frame.vector(rail.mount_normal);
    }
    bearing
}

/// Access a canonical graph without marking an ECS resource changed until publication.
pub(crate) trait EditorViewTarget {
    fn graph(&self) -> &ConstructionGraph;
    fn publish(&mut self, graph: ConstructionGraph);
}

impl EditorViewTarget for EditorGraph {
    fn graph(&self) -> &ConstructionGraph {
        &self.0
    }

    fn publish(&mut self, graph: ConstructionGraph) {
        self.0 = graph;
    }
}

impl EditorViewTarget for ResMut<'_, EditorGraph> {
    fn graph(&self) -> &ConstructionGraph {
        &self.0
    }

    fn publish(&mut self, graph: ConstructionGraph) {
        self.0 = graph;
    }
}

/// A short-lived local view; only actual mutations are returned to the canonical graph.
pub(crate) struct EditorView<'a, T: EditorViewTarget> {
    target: &'a mut T,
    state: &'a mut EditorState,
    local: EditorGraph,
    baseline: ConstructionGraph,
    original_bearings: Vec<PlacedBearing>,
    local_bearings: Vec<PlacedBearing>,
    to_build: ConstructionFrame,
}

impl<'a, T: EditorViewTarget> EditorView<'a, T> {
    pub(crate) fn new(target: &'a mut T, state: &'a mut EditorState) -> Self {
        let local = state
            .edit_context
            .and_then(|context| target.graph().in_edit_frame(context.frame).ok())
            .unwrap_or_else(|| target.graph().clone());
        let to_build = local.view_to_build();
        let original_bearings = state.placed_bearings.clone();
        let local_bearings = original_bearings
            .iter()
            .map(|&bearing| transform_bearing(bearing, to_build.inverse()))
            .collect::<Vec<_>>();
        state.placed_bearings.clone_from(&local_bearings);
        Self {
            target,
            state,
            baseline: local.clone(),
            local: EditorGraph(local),
            original_bearings,
            local_bearings,
            to_build,
        }
    }

    pub(crate) fn parts(&mut self) -> (&mut EditorGraph, &mut EditorState) {
        (&mut self.local, self.state)
    }
}

impl<T: EditorViewTarget> Drop for EditorView<'_, T> {
    fn drop(&mut self) {
        if !self.local.0.shares_revision(&self.baseline) {
            let mut graph = self.local.0.canonicalized();
            if let Some(context) = self.state.edit_context {
                let added = graph
                    .parts()
                    .filter_map(|(part, _)| {
                        self.target.graph().part(part).is_none().then_some(part)
                    })
                    .collect::<Vec<_>>();
                for part in added {
                    if graph.edit_source(part).is_none() {
                        let _ = graph.set_edit_source(part, context.anchor);
                    }
                }
            }
            self.target.publish(graph);
        }
        for (index, bearing) in self.state.placed_bearings.iter_mut().enumerate() {
            *bearing = if self.local_bearings.get(index) == Some(bearing) {
                self.original_bearings[index]
            } else {
                transform_bearing(*bearing, self.to_build)
            };
        }
    }
}

/// Keep a gesture attached even while UI input temporarily prevents picking.
pub(crate) fn refresh_context(
    graph: Res<EditorGraph>,
    simulation: Res<AppSimulation>,
    mut state: ResMut<EditorState>,
) {
    if let Some(context) = state.edit_context
        && let Some(refreshed) = EditContext::resolve(&graph.0, &simulation, context.anchor)
    {
        state.edit_context = Some(refreshed);
    }
    // Keep a failed anchor until picking can cancel its attached gesture before
    // selecting another frame. Clearing it here loses that identity.
}

/// Preview meshes are authored in the active tool grid and follow its current pose.
#[expect(clippy::type_complexity)]
pub(crate) fn place_previews(
    state: Res<EditorState>,
    mut previews: Query<
        (&mut Transform, &Visibility),
        Or<(
            With<crate::ActionPreview>,
            With<crate::SelectionPreview>,
            With<crate::DeletePreview>,
        )>,
    >,
) {
    let Some(context) = state.edit_context else {
        return;
    };
    for (mut transform, visibility) in &mut previews {
        if *visibility == Visibility::Visible {
            transform.translation = context.frame_to_world.point(transform.translation);
            transform.rotation = context.frame_to_world.rotation() * transform.rotation;
        }
    }
}

#[cfg(test)]
mod tests;

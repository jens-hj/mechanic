//! Explicit weld-placement transactions. Ordinary edits retain pose-preserving transfer.

use crate::editor::history::{EditorHistory, EditorSnapshot};
use crate::{
    AppSimulation, EditorGraph, EditorState, PreparedWorldPhysics, WorldPhysicsPublication,
    builder::PlacementBounds,
    freeze::DimensionFreeze,
    weld_tool::{Pick, motion},
    world::WorldRuntime,
};
use bevy::{
    prelude::*,
    render::renderer::{RenderDevice, RenderQueue},
    tasks::{AsyncComputeTaskPool, Task},
};
use mechanic_core::{
    CompiledCreation, ConstructionFrame, ConstructionGraph, FaceOwner, PartId, WeldPlacement,
};
use mechanic_gpu::{GpuMechanismCoordinate, GpuTransform, GpuVelocity};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub(crate) struct Intent {
    pub(crate) baseline: ConstructionGraph,
    source: Pick,
    destination: Pick,
    transform: ConstructionFrame,
    pub(crate) parts: Vec<PartId>,
}

impl Intent {
    pub(crate) const fn destination(&self) -> &Pick {
        &self.destination
    }
    pub(crate) fn relocation(
        graph: &ConstructionGraph,
        source: &Pick,
        destination: &Pick,
        transform: ConstructionFrame,
    ) -> Self {
        Self {
            baseline: graph.clone(),
            source: source.clone(),
            destination: destination.clone(),
            transform,
            parts: graph
                .structural_component(source.part, [])
                .map(|c| c.parts().collect())
                .unwrap_or_default(),
        }
    }
    pub(crate) fn preview(
        &self,
        simulation: &AppSimulation,
    ) -> Result<(ConstructionGraph, Vec<PartId>, ConstructionFrame), String> {
        let frame = motion(simulation, self.destination.part, false)?.compose(self.transform);
        Ok((self.baseline.clone(), self.parts.clone(), frame))
    }
    pub(crate) fn place_sockets(&self, state: &mut EditorState) {
        for socket in &mut state.placed_bearings {
            if matches!(socket.source.owner, FaceOwner::Part(part) if self.parts.contains(&part)) {
                *socket = crate::live_edit::transform_bearing(*socket, self.transform);
            }
        }
    }

    pub(crate) fn stage(
        &self,
        graph: &ConstructionGraph,
        anchored: impl IntoIterator<Item = PartId>,
    ) -> Result<ConstructionGraph, String> {
        if !graph.shares_revision(&self.baseline) {
            return Err("The construction changed during weld placement".to_owned());
        }
        if let Some(socket) = self.destination.socket {
            return crate::weld_tool::socket::stage(
                graph,
                &self.source,
                socket,
                self.transform,
                anchored,
            );
        }
        WeldPlacement::stage(
            graph,
            self.source.face,
            self.destination.face,
            self.transform,
            anchored,
        )
        .map(|placement| placement.graph)
    }
    pub(crate) fn validate(
        &self,
        graph: &ConstructionGraph,
        simulation: &AppSimulation,
        world: &WorldRuntime,
        bounds: PlacementBounds,
    ) -> Result<(), String> {
        let anchored = simulation
            .world_revision
            .and(simulation.creation.as_ref())
            .into_iter()
            .flat_map(|c| c.compounds.iter())
            .filter(|b| b.is_static)
            .flat_map(|b| b.source_parts.iter().copied())
            .collect::<Vec<_>>();
        self.stage(graph, anchored)?;
        let creation = graph.compile().map_err(|e| e.to_string())?;
        let source_motion =
            motion(simulation, self.destination.part, true)?.compose(self.transform);
        let destination_motion = motion(simulation, self.destination.part, true)?;
        if self.destination.socket.is_none() {
            let source_faces = graph
                .weld_mating_faces(self.source.face)
                .map_err(|e| e.to_string())?;
            let destination_faces = graph
                .weld_mating_faces(self.destination.face)
                .map_err(|e| e.to_string())?;
            if matches!(
                self.destination.selection.feature,
                mechanic_core::WeldFeature::Face
            ) && !graph.weld_feature_on_faces(
                &destination_faces,
                self.source.selection.feature,
                self.transform,
            ) {
                return Err("Selected source feature leaves the destination material".to_owned());
            }
            if matches!(
                self.source.selection.feature,
                mechanic_core::WeldFeature::Face
            ) && !graph.weld_feature_on_faces(
                &source_faces,
                self.destination.selection.feature,
                self.transform.inverse(),
            ) {
                return Err("Selected destination feature leaves the source material".to_owned());
            }
            graph
                .weld_contact_square_transformed(
                    &source_faces,
                    &destination_faces,
                    source_motion,
                    destination_motion,
                )
                .map_err(|e| e.to_string())?;
        }
        for collider in creation
            .colliders
            .iter()
            .filter(|c| self.parts.contains(&c.source_part))
        {
            let body = &creation.compounds[collider.compound_index as usize];
            let frame = source_motion.compose(
                ConstructionFrame::new(body.root_translation, body.root_rotation)
                    .map_err(|e| e.to_string())?,
            );
            let geometry = mechanic_core::WeldCollider::new(collider, frame);
            if bounds.is_world() {
                if !crate::freeze::weld_terrain_clear(world, collider, pose(frame)) {
                    return Err(
                        "The default-pose source intersects terrain or unavailable terrain"
                            .to_owned(),
                    );
                }
            } else {
                let (low, high) = geometry.vertices.iter().fold(
                    (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
                    |(low, high), &point| (low.min(point), high.max(point)),
                );
                crate::builder::validate_world_bounds(low, high, bounds)
                    .map_err(|e| e.to_string())?;
            }
            for other in creation
                .colliders
                .iter()
                .filter(|c| !self.parts.contains(&c.source_part))
            {
                let body = &creation.compounds[other.compound_index as usize];
                let frame = motion(simulation, other.source_part, true)?.compose(
                    ConstructionFrame::new(body.root_translation, body.root_rotation)
                        .map_err(|e| e.to_string())?,
                );
                if geometry.penetration(&mechanic_core::WeldCollider::new(other, frame)) > 0.001 {
                    return Err("The default-pose source intersects another body".to_owned());
                }
            }
        }
        Ok(())
    }
    pub(crate) fn states(
        &self,
        creation: &CompiledCreation,
        graph: &ConstructionGraph,
        previous: &AppSimulation,
    ) -> Result<BodyStates, String> {
        let (mut transforms, mut velocities) =
            crate::rebuilt_body_states(creation, graph, previous);
        let old = previous
            .creation
            .as_ref()
            .ok_or("The source publication is unavailable")?;
        let state = previous
            .live_state
            .as_ref()
            .ok_or("Wait for an authoritative snapshot")?;
        let destination_body = old
            .part_to_compound
            .iter()
            .find_map(|&(part, body)| (part == self.destination.part).then_some(body as usize))
            .ok_or("Destination was removed")?;
        let destination_pose = state
            .transforms
            .get(destination_body)
            .ok_or("Destination snapshot is incomplete")?;
        let velocity = state
            .velocities
            .get(destination_body)
            .ok_or("Destination velocity is unavailable")?;
        let angular = Vec3::from_slice(&velocity.angular[..3]);
        let center = Vec3::from_slice(&destination_pose.position[..3]);
        let frame = motion(previous, self.destination.part, true)?;
        for (index, body) in creation.compounds.iter().enumerate().filter(|(_, body)| {
            body.source_parts
                .iter()
                .any(|part| self.parts.contains(part))
        }) {
            transforms[index] = pose(
                frame.compose(
                    ConstructionFrame::new(body.root_translation, body.root_rotation)
                        .map_err(|e| e.to_string())?,
                ),
            );
            let position = Vec3::from_slice(&transforms[index].position[..3]);
            velocities[index] = GpuVelocity {
                linear: (Vec3::from_slice(&velocity.linear[..3])
                    + angular.cross(position - center))
                .extend(0.0)
                .to_array(),
                angular: velocity.angular,
            };
        }
        let mut coordinates =
            crate::rebuilt_mechanism_coordinates(creation, previous, &transforms, &velocities);
        for (index, bearing) in creation.loop_topology.tree_bearings.iter().enumerate() {
            if self.baseline.bearing(*bearing).is_some_and(|joint| matches!(joint.source.owner, FaceOwner::Part(part) if self.parts.contains(&part))) {
                coordinates[index] = GpuMechanismCoordinate { position: 0.0, velocity: 0.0 };
            }
        }
        Ok((transforms, velocities, coordinates))
    }
}

pub(crate) type BodyStates = (
    Vec<GpuTransform>,
    Vec<GpuVelocity>,
    Vec<GpuMechanismCoordinate>,
);
fn pose(frame: ConstructionFrame) -> GpuTransform {
    GpuTransform {
        position: frame.translation().extend(0.0).to_array(),
        rotation: frame.rotation().to_array(),
    }
}

pub(crate) struct Publication {
    intent: Intent,
    task: Option<Task<Result<PreparedWorldPhysics, String>>>,
    ready: Option<PreparedWorldPhysics>,
    foundation: u64,
}
impl Publication {
    pub(crate) fn ready(&self) -> bool {
        self.ready.is_some()
    }
}

/// Returns true while placement owns the publication boundary.
#[expect(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn maintain(
    graph: &mut EditorGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    simulation: &mut AppSimulation,
    frozen: &mut DimensionFreeze,
    world: &mut WorldRuntime,
    publication: &mut WorldPhysicsPublication,
    device: &RenderDevice,
    queue: &RenderQueue,
) -> bool {
    if simulation.world_revision == Some((history.current_revision, world.foundation_revision()))
        && publication.placement.is_none()
        && publication.pending.is_none()
        && publication.ready.is_none()
        && let Some(intent) = state.weld.request.take()
    {
        let anchored = world.static_parts_for_physics(history.current_revision);
        let prepared_graph = anchored
            .as_ref()
            .ok_or("Wait for terrain publication".to_owned())
            .and_then(|parts| intent.stage(&graph.0, parts.iter().copied()));
        match prepared_graph {
            Err(error) => {
                state.weld.finish_publication();
                state.feedback = Some(error);
                return true;
            }
            Ok(staged) => {
                let suspension_sockets = crate::suspension_editor::sockets(&state.placed_bearings);
                let config = crate::GpuPhysicsConfig {
                    ground_plane_enabled: false,
                    mechanism_self_collisions: crate::world_mechanism_self_collisions(&staged),
                    ..default()
                };
                let device = device.clone();
                let queue = queue.clone();
                let pipelines = Arc::clone(&publication.pipelines);
                let generation = history.next_revision.saturating_add(1);
                publication.placement = Some(Publication {
                    intent,
                    foundation: world.foundation_revision(),
                    ready: None,
                    task: Some(AsyncComputeTaskPool::get().spawn(async move {
                        crate::prepare_world_physics(
                            staged,
                            generation,
                            suspension_sockets,
                            anchored.unwrap_or_default(),
                            config,
                            device,
                            queue,
                            pipelines,
                        )
                    })),
                });
            }
        }
    }
    let Some(mut transaction) = publication.placement.take() else {
        return false;
    };
    if !transaction.intent.baseline.shares_revision(&graph.0)
        || transaction.foundation != world.foundation_revision()
        || !state.weld.busy()
    {
        state.weld.finish_publication();
        state.feedback = Some("Weld placement cancelled because the scene changed".to_owned());
        return true;
    }
    if let Some(result) = transaction.task.as_mut().and_then(crate::check_ready) {
        transaction.task = None;
        match result {
            Ok(prepared) => transaction.ready = Some(prepared),
            Err(error) => {
                state.weld.finish_publication();
                state.feedback = Some(format!("Weld preparation failed: {error}"));
                return true;
            }
        }
    }
    if transaction.ready.is_none()
        || (simulation.is_running() && simulation.completed_tick + 1 < simulation.next_tick)
    {
        publication.placement = Some(transaction);
        return true;
    }
    let result = transaction
        .intent
        .validate(&graph.0, simulation, world, state.placement_bounds)
        .and_then(|()| {
            let revision = (
                history.next_revision.saturating_add(1),
                world.foundation_revision(),
            );
            let mut replacement = crate::replacement_simulation_for_weld(
                transaction
                    .ready
                    .take()
                    .expect("prepared transaction exists"),
                simulation,
                revision,
                queue,
                Some(&transaction.intent),
            )?;
            let hold = frozen.for_weld(
                simulation,
                &replacement,
                transaction.intent.destination.part,
                &transaction.intent.parts,
            );
            hold.install_publication(&mut replacement, queue)?;
            Ok((replacement, hold))
        });
    match result {
        Err(error) => {
            state.weld.finish_publication();
            state.feedback = Some(format!("Weld rejected: {error}"));
        }
        Ok((mut replacement, hold)) => {
            crate::inherit_terrain_residency(simulation, &mut replacement, device);
            let mut previous = EditorSnapshot::capture(&graph.0, state);
            let mut affected = transaction.intent.parts.clone();
            if let Ok(component) = graph
                .0
                .structural_component(transaction.intent.destination.part, [])
            {
                affected.extend(component.parts());
            }
            previous.weld_restore = Restore::capture(affected, &graph.0, simulation, frozen);
            let warning = crate::weld_lockup_warning(&graph.0, &replacement.published_graph);
            graph.0 = replacement.published_graph.clone();
            transaction.intent.place_sockets(state);
            history.commit(previous);
            *simulation = replacement;
            *frozen = hold;
            world.accept_weld_freeze(frozen.saved_record(), &graph.0, state);
            state.weld.cancel();
            state.construction_mesh_dirty = true;
            state.feedback = Some(warning.map_or_else(
                || "Welded in the source default pose".to_owned(),
                |warning| format!("Welded — {warning}"),
            ));
            publication.accepted_editor =
                Some((EditorSnapshot::capture(&graph.0, state), history.clone()));
        }
    }
    true
}

/// A history entry owns only the two affected assemblies, never the whole live world.
#[derive(Clone)]
pub(crate) struct Restore {
    parts: Vec<PartId>,
    frames: std::collections::BTreeMap<PartId, (ConstructionFrame, Vec3, GpuVelocity)>,
    coordinates: std::collections::BTreeMap<mechanic_core::BearingId, GpuMechanismCoordinate>,
    hold: DimensionFreeze,
}
impl std::fmt::Debug for Restore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeldRestore")
            .field("parts", &self.parts)
            .finish_non_exhaustive()
    }
}
impl Restore {
    pub(crate) fn recapture(
        &self,
        graph: &ConstructionGraph,
        simulation: &AppSimulation,
        frozen: &DimensionFreeze,
    ) -> Option<Self> {
        Self::capture(self.parts.clone(), graph, simulation, frozen)
    }
    fn capture(
        parts: Vec<PartId>,
        graph: &ConstructionGraph,
        simulation: &AppSimulation,
        frozen: &DimensionFreeze,
    ) -> Option<Self> {
        let creation = simulation.creation.as_ref()?;
        let live = simulation.live_state.as_ref()?;
        let mut frames = std::collections::BTreeMap::new();
        for &(part, body) in &creation.part_to_compound {
            if !parts.contains(&part) {
                continue;
            }
            let frame = motion(simulation, part, true)
                .ok()?
                .compose(graph.part_frame(part)?);
            let pose = live.transforms.get(body as usize)?;
            frames.insert(
                part,
                (
                    frame,
                    Vec3::from_slice(&pose.position[..3]),
                    *live.velocities.get(body as usize)?,
                ),
            );
        }
        let coordinates = creation
            .loop_topology
            .bearing_coordinates
            .iter()
            .filter_map(|(bearing, index)| {
                let joint = graph.bearing(*bearing)?;
                matches!(joint.source.owner, FaceOwner::Part(part) if parts.contains(&part))
                    .then(|| {
                        live.coordinates
                            .get(*index as usize)
                            .copied()
                            .map(|coordinate| (*bearing, coordinate))
                    })
                    .flatten()
            })
            .collect();
        Some(Self {
            parts,
            frames,
            coordinates,
            hold: frozen.clone(),
        })
    }
    pub(crate) fn states(
        &self,
        creation: &CompiledCreation,
        graph: &ConstructionGraph,
        previous: &AppSimulation,
    ) -> Result<BodyStates, String> {
        let (mut transforms, mut velocities) =
            crate::rebuilt_body_states(creation, graph, previous);
        for (index, body) in creation.compounds.iter().enumerate() {
            let Some((&part, &(world_frame, center, velocity))) = body
                .source_parts
                .iter()
                .find_map(|part| self.frames.get_key_value(part))
            else {
                continue;
            };
            let authored = ConstructionFrame::new(body.root_translation, body.root_rotation)
                .map_err(|e| e.to_string())?;
            let frame = world_frame
                .compose(
                    graph
                        .part_frame(part)
                        .ok_or("History part was removed")?
                        .inverse(),
                )
                .compose(authored);
            transforms[index] = pose(frame);
            let angular = Vec3::from_slice(&velocity.angular[..3]);
            velocities[index] = GpuVelocity {
                linear: (Vec3::from_slice(&velocity.linear[..3])
                    + angular.cross(frame.translation() - center))
                .extend(0.0)
                .to_array(),
                angular: velocity.angular,
            };
        }
        let mut coordinates =
            crate::rebuilt_mechanism_coordinates(creation, previous, &transforms, &velocities);
        for (index, bearing) in creation.loop_topology.tree_bearings.iter().enumerate() {
            if let Some(saved) = self.coordinates.get(bearing) {
                coordinates[index] = *saved;
            }
        }
        Ok((transforms, velocities, coordinates))
    }
    pub(crate) fn hold(
        &self,
        replacement: &AppSimulation,
        current: &DimensionFreeze,
    ) -> DimensionFreeze {
        if current
            .saved_record()
            .and_then(|record| replacement.published_graph.dimension_link(record.link))
            .is_some_and(|part| !self.parts.contains(&part))
        {
            return current.restored_for_weld(replacement);
        }
        if self
            .hold
            .saved_record()
            .and_then(|record| replacement.published_graph.dimension_link(record.link))
            .is_some_and(|part| self.parts.contains(&part))
        {
            self.hold.restored_for_weld(replacement)
        } else {
            DimensionFreeze::default()
        }
    }
}

#[cfg(test)]
mod tests;

//! Feature picking and a destination-relative weld gesture. Ghosts use authored geometry.

use crate::{
    AppSimulation, EditorState, builder, controls::GameAction, hotbar::WeldMode, weld_publication,
};
use bevy::prelude::*;
use mechanic_core::{
    ConstructionFrame, ConstructionGraph, FaceRef, PartId, SolidOwner, WeldAlignment,
    WeldConstraint, WeldFeature, WeldPick, WeldSelection, WeldSnap,
};

pub(crate) mod socket;

#[derive(Clone, Debug)]
pub(crate) struct Pick {
    pub(crate) part: PartId,
    pub(crate) face: FaceRef,
    pub(crate) socket: Option<crate::PlacedBearing>,
    pub(crate) selection: WeldSelection,
}

#[derive(Clone, Debug)]
struct Source {
    graph: ConstructionGraph,
    pick: Pick,
}

#[derive(Clone, Debug)]
struct Drag {
    destination: Pick,
    alignment: WeldAlignment,
    last_ray: Ray3d,
    raw: Vec3,
    initial_displacement: Vec3,
    snap: WeldSnap,
    displacement: Vec3,
    steps: u8,
}

#[derive(Default)]
pub(crate) struct WeldTool {
    mode: WeldMode,
    join_first: Option<(ConstructionGraph, PartId)>,
    pub(crate) join_hovered: Option<PartId>,
    join_candidate: Option<(PartId, Option<u64>, Result<ConstructionGraph, String>)>,
    source: Option<Source>,
    pub(crate) hovered: Option<Pick>,
    drag: Option<Drag>,
    pub(crate) preview: Option<(ConstructionGraph, Vec<PartId>, ConstructionFrame)>,
    pub(crate) error: Option<String>,
    pub(crate) request: Option<weld_publication::Intent>,
    publishing: Option<weld_publication::Intent>,
    candidate: Option<weld_publication::Intent>,
}

impl WeldTool {
    pub(crate) fn effects_target(&self, simulation: &AppSimulation) -> Option<(Vec3, Vec3)> {
        if self.candidate.is_none() || self.request.is_some() || self.publishing.is_some() {
            return None;
        }
        let pick = &self.drag.as_ref()?.destination;
        let frame = motion(simulation, pick.part, false).ok()?;
        Some((
            frame.point(pick.selection.point),
            frame.vector(pick.selection.normal),
        ))
    }

    pub(crate) fn busy(&self) -> bool {
        self.source.is_some() || self.request.is_some() || self.join_first.is_some()
    }
    pub(crate) fn cancel(&mut self) {
        *self = Self {
            mode: self.mode,
            ..Self::default()
        };
    }
    pub(crate) fn join_first(&self) -> Option<PartId> {
        self.join_first.as_ref().map(|(_, part)| *part)
    }
    /// Whether the hovered Join target would weld, once a first body is chosen.
    pub(crate) fn join_valid(&self) -> Option<bool> {
        self.join_candidate
            .as_ref()
            .map(|(_, _, staged)| staged.is_ok())
    }
    pub(crate) fn finish_publication(&mut self) {
        self.publishing = None;
        self.request = None;
    }
    pub(crate) fn cancel_drag(&mut self) {
        self.drag = None;
        self.candidate = None;
        self.preview = None;
    }
}

pub(crate) fn motion(
    simulation: &AppSimulation,
    part: PartId,
    authoritative: bool,
) -> Result<ConstructionFrame, String> {
    let Some(creation) = simulation.creation.as_ref() else {
        return Ok(ConstructionFrame::IDENTITY);
    };
    if simulation.world_revision.is_none() {
        return Ok(ConstructionFrame::IDENTITY);
    }
    let Some(body) = creation
        .part_to_compound
        .iter()
        .find_map(|&(id, body)| (id == part).then_some(body as usize))
    else {
        return Err("Wait for this creation to publish".to_owned());
    };
    let poses = if authoritative {
        &simulation
            .live_state
            .as_ref()
            .ok_or("Wait for an authoritative snapshot")?
            .transforms
    } else {
        &simulation.transforms
    };
    crate::live_weld::world_from_build(creation, poses, body)
}

fn pick(graph: &ConstructionGraph, simulation: &AppSimulation, ray: Ray3d) -> Option<Pick> {
    let mut nearest = None;
    for (part, spec) in graph.parts() {
        let Ok(frame) = motion(simulation, part, false) else {
            continue;
        };
        let inverse = frame.inverse();
        let owner = graph
            .region_of(part)
            .map_or(SolidOwner::Part(part), SolidOwner::Region);
        let solid = graph.evaluated_solid(owner).ok();
        let origin = inverse.point(ray.origin);
        let direction = inverse.vector(ray.direction.as_vec3());
        let hit = if let Some(solid) = &solid {
            solid
                .surfaces
                .iter()
                .filter_map(|surface| {
                    let (distance, point) =
                        builder::raycast_evaluated_surface(origin, direction, solid, surface)?;
                    Some(builder::SurfaceHit {
                        distance,
                        point,
                        face: FaceRef::patch(part, mechanic_core::FaceKind::PositiveY, surface.key),
                    })
                })
                .min_by(|a, b| a.distance.total_cmp(&b.distance))
        } else {
            builder::raycast_part_in_construction(graph, part, origin, direction)
        };
        let Some(hit) = hit else {
            continue;
        };
        if nearest
            .as_ref()
            .is_some_and(|(distance, _)| hit.distance >= *distance)
        {
            continue;
        }
        let Some(face) = builder::try_face_geometry_from_ref(hit.face, Some(graph)) else {
            continue;
        };
        let owner = graph
            .region_of(part)
            .map_or(SolidOwner::Part(part), SolidOwner::Region);
        let patch = hit
            .face
            .patch
            .unwrap_or_else(|| builder::primitive_surface_patch(*spec, hit.face.face));
        let selected = if solid.is_some() {
            WeldPick::nearest(graph, owner, patch, hit.point, 0.02)
                .and_then(|pick| pick.resolve(graph))
                .ok()
        } else {
            Some(WeldSelection {
                feature: WeldFeature::Face,
                point: hit.point,
                normal: face.normal,
                tangent: face.tangent_u,
            })
        };
        let selected = selected.map(|mut selection| {
            selection.tangent = grid_tangent(graph, part, selection.normal);
            selection
        });
        // A curved surface occludes a selectable surface behind it.
        nearest = Some((
            hit.distance,
            selected.map(|selection| Pick {
                part,
                face: hit.face,
                socket: None,
                selection,
            }),
        ));
    }
    nearest.and_then(|(_, pick)| pick)
}

fn plane_point(ray: Ray3d, destination: &Pick, simulation: &AppSimulation) -> Option<Vec3> {
    let inverse = motion(simulation, destination.part, false).ok()?.inverse();
    let origin = inverse.point(ray.origin);
    let direction = inverse.vector(ray.direction.as_vec3());
    let denominator = direction.dot(destination.selection.normal);
    if denominator.abs() < 1.0e-5 {
        return None;
    }
    let distance =
        (destination.selection.point - origin).dot(destination.selection.normal) / denominator;
    (distance >= 0.0).then_some(origin + direction * distance)
}

pub(crate) fn hover(
    graph: &ConstructionGraph,
    simulation: &AppSimulation,
    state: &mut EditorState,
    ray: Ray3d,
    actions: &ButtonInput<GameAction>,
    world: &crate::world::WorldRuntime,
) {
    state.edit_context = None;
    state.pointer_ray = Some((ray.origin, ray.direction.as_vec3()));
    if state
        .weld
        .source
        .as_ref()
        .is_some_and(|source| !source.graph.shares_revision(graph))
    {
        state.weld.cancel();
        state.feedback = Some("Weld selection changed; select the source again".to_owned());
    }
    let mut weld = std::mem::take(&mut state.weld);
    weld.hovered = if weld.source.is_some() {
        socket::pick(graph, simulation, &state.placed_bearings, ray)
    } else {
        pick(graph, simulation, ray)
    };
    state.hovered = weld.hovered.as_ref().map(|pick| builder::SurfaceHit {
        face: pick.face,
        point: pick.selection.point,
        distance: 0.0,
    });
    if let Some(intent) = &weld.publishing {
        weld.preview = intent.preview(simulation).ok();
        weld.error = intent
            .validate(graph, simulation, world, state.placement_bounds)
            .err();
        state.feedback = Some(
            weld.error
                .clone()
                .unwrap_or_else(|| "Preparing weld placement".to_owned()),
        );
        state.weld = weld;
        return;
    }
    weld.preview = None;
    weld.candidate = None;
    weld.error = None;
    if let Some(source) = &weld.source {
        let destination = weld
            .drag
            .as_ref()
            .map(|drag| &drag.destination)
            .or(weld.hovered.as_ref());
        if let Some(destination) = destination.cloned() {
            let result = build_candidate(
                graph,
                simulation,
                source,
                &destination,
                weld.drag.as_mut(),
                ray,
                actions,
                world,
                state.placement_bounds,
            );
            match result {
                Ok((intent, preview)) => {
                    weld.preview = Some(preview);
                    weld.candidate = Some(intent);
                }
                Err((error, preview)) => {
                    weld.error = Some(error);
                    weld.preview = preview;
                }
            }
        }
        if let Some(drag) = &weld.drag {
            let increment = if actions.pressed(GameAction::FinePlacement) {
                5
            } else {
                25
            };
            state.feedback = Some(format!(
                "Weld: {}° · {increment} cm snap{}",
                u16::from(drag.steps) * 15,
                weld.error
                    .as_ref()
                    .map_or(String::new(), |error| format!(" · {error}"))
            ));
        } else if let Some(error) = &weld.error {
            state.feedback = Some(error.clone());
        } else if weld.preview.is_some() {
            let increment = if actions.pressed(GameAction::FinePlacement) {
                5
            } else {
                25
            };
            state.feedback = Some(format!("Weld: 0° · {increment} cm snap · press to drag"));
        }
    }
    state.weld = weld;
}

type Preview = (ConstructionGraph, Vec<PartId>, ConstructionFrame);
#[expect(clippy::too_many_arguments)]
fn build_candidate(
    graph: &ConstructionGraph,
    simulation: &AppSimulation,
    source: &Source,
    destination: &Pick,
    drag: Option<&mut Drag>,
    ray: Ray3d,
    actions: &ButtonInput<GameAction>,
    world: &crate::world::WorldRuntime,
    bounds: builder::PlacementBounds,
) -> Result<(weld_publication::Intent, Preview), (String, Option<Preview>)> {
    let attempt = || -> Result<_, String> {
        let component = graph
            .structural_component(source.pick.part, [])
            .map_err(|e| e.to_string())?;
        if component.contains(destination.part) {
            if destination.socket.is_some() {
                return Err("Select a separate assembly to attach to this bearing".to_owned());
            }
            return Err("Parts of one creation weld where they are; use Join mode".to_owned());
        }
        let alignment = WeldAlignment::new(source.pick.selection, destination.selection)
            .and_then(WeldAlignment::align_tangent_grids)
            .map_err(|e| e.to_string())?;
        let frame = if let Some(drag) = drag {
            let point = plane_point(ray, destination, simulation)
                .ok_or("The drag ray no longer reaches the mating plane")?;
            let prior = plane_point(drag.last_ray, destination, simulation)
                .ok_or("The drag ray no longer reaches the mating plane")?;
            drag.raw += drag.alignment.displacement(point - prior);
            drag.last_ray = ray;
            let (u, v) = axes(drag.alignment, destination.selection);
            let snapped = drag.snap.update(
                Vec2::new(drag.raw.dot(u), drag.raw.dot(v)),
                actions.pressed(GameAction::FinePlacement),
            );
            drag.displacement = drag.initial_displacement + u * snapped.x + v * snapped.y;
            if actions.just_pressed(GameAction::Rotate) {
                drag.steps = drag
                    .alignment
                    .next_rotation(drag.displacement, drag.steps)
                    .unwrap_or(drag.steps);
            }
            drag.alignment.place(drag.displacement, drag.steps)
        } else {
            alignment.place(
                initial_displacement(
                    graph,
                    &source.pick,
                    destination,
                    alignment,
                    actions.pressed(GameAction::FinePlacement),
                ),
                0,
            )
        }
        .map_err(|e| e.to_string())?;
        Ok(weld_publication::Intent::relocation(
            graph,
            &source.pick,
            destination,
            frame,
        ))
    };
    let intent = attempt().map_err(|e| (e, None))?;
    let preview = intent.preview(simulation).map_err(|e| (e, None))?;
    intent
        .validate(graph, simulation, world, bounds)
        .map_err(|e| (e, Some(preview.clone())))?;
    Ok((intent, preview))
}

// Authored part axes keep the mating grid stable when the part is relocated or
// tilted. Projecting the best local axis also supports edited, sloping faces.
fn grid_tangent(graph: &ConstructionGraph, part: PartId, normal: Vec3) -> Vec3 {
    let frame = graph.part_frame(part).expect("picked part exists");
    let rotation = graph
        .part(part)
        .expect("picked part exists")
        .pose()
        .rotation
        .quaternion();
    let candidates = [Vec3::X, Vec3::Y, Vec3::Z].map(|axis| frame.vector(rotation * axis));
    let axis = candidates
        .into_iter()
        .min_by(|a, b| a.dot(normal).abs().total_cmp(&b.dot(normal).abs()))
        .unwrap();
    (axis - normal * axis.dot(normal)).normalize()
}

fn grid_origin(graph: &ConstructionGraph, pick: &Pick) -> Vec3 {
    let owner = graph
        .region_of(pick.part)
        .map_or(SolidOwner::Part(pick.part), SolidOwner::Region);
    let u = pick.selection.tangent;
    let v = pick.selection.normal.cross(u);
    let mut minimum = Vec2::splat(f32::INFINITY);
    if let Ok(solid) = graph.evaluated_solid(owner) {
        let patch = pick.face.patch.unwrap_or_else(|| {
            builder::primitive_surface_patch(*graph.part(pick.part).unwrap(), pick.face.face)
        });
        for surface in solid.surfaces.iter().filter(|surface| surface.key == patch) {
            let mut edge = surface.half_edge;
            loop {
                let half = solid.half_edges[edge as usize];
                let point = solid.vertices[half.origin as usize].position;
                minimum = minimum.min(Vec2::new(point.dot(u), point.dot(v)));
                edge = half.next;
                if edge == surface.half_edge {
                    break;
                }
            }
        }
    }
    if minimum.is_finite() {
        u * minimum.x
            + v * minimum.y
            + pick.selection.normal * pick.selection.point.dot(pick.selection.normal)
    } else {
        builder::try_face_geometry_from_ref(pick.face, Some(graph))
            .expect("picked face exists")
            .center
    }
}

// Align the face-local lattice origins, preserving the exact feature constraints.
fn initial_displacement(
    graph: &ConstructionGraph,
    source: &Pick,
    destination: &Pick,
    alignment: WeldAlignment,
    fine: bool,
) -> Vec3 {
    if destination.socket.is_some() {
        return Vec3::ZERO;
    }
    let (u, v) = axes(alignment, destination.selection);
    let offset =
        alignment.initial().point(grid_origin(graph, source)) - grid_origin(graph, destination);
    let coordinates = Vec2::new(offset.dot(u), offset.dot(v));
    let step = if fine { 0.05 } else { 0.25 };
    let correction = (coordinates / step).round() * step - coordinates;
    alignment.displacement(u * correction.x + v * correction.y)
}

fn axes(alignment: WeldAlignment, destination: WeldSelection) -> (Vec3, Vec3) {
    match alignment.constraint {
        WeldConstraint::Line { direction } => (direction, Vec3::ZERO),
        _ => (
            destination.tangent,
            destination.normal.cross(destination.tangent),
        ),
    }
}

pub(crate) fn actions(
    graph: &mut ConstructionGraph,
    simulation: &AppSimulation,
    state: &mut EditorState,
    history: &mut crate::editor::history::EditorHistory,
    actions: &ButtonInput<GameAction>,
    blocked: bool,
) {
    if actions.just_pressed(GameAction::Secondary) {
        state.weld.cancel();
        state.feedback = Some("Weld cancelled".to_owned());
        return;
    }
    if blocked {
        if actions.just_released(GameAction::Primary) {
            state.weld.cancel_drag();
        }
        return;
    }
    if state.weld.publishing.is_some() || state.weld.request.is_some() {
        return;
    }
    if actions.just_pressed(GameAction::Primary) {
        if let Some(source) = &state.weld.source {
            if let (Some(destination), Some((origin, direction))) =
                (state.weld.hovered.clone(), state.pointer_ray)
                && let Ok(alignment) =
                    WeldAlignment::new(source.pick.selection, destination.selection)
                        .and_then(WeldAlignment::align_tangent_grids)
            {
                let ray = Ray3d::new(
                    origin,
                    Dir3::new(direction).expect("pointer direction is normalized"),
                );
                if plane_point(ray, &destination, simulation).is_some() {
                    let initial_displacement = initial_displacement(
                        graph,
                        &source.pick,
                        &destination,
                        alignment,
                        actions.pressed(GameAction::FinePlacement),
                    );
                    let mut snap = WeldSnap::default();
                    snap.update(Vec2::ZERO, actions.pressed(GameAction::FinePlacement));
                    state.weld.drag = Some(Drag {
                        destination,
                        alignment,
                        last_ray: ray,
                        raw: Vec3::ZERO,
                        initial_displacement,
                        snap,
                        displacement: initial_displacement,
                        steps: 0,
                    });
                }
            }
        } else if let Some(pick) = state.weld.hovered.clone() {
            state.weld.source = Some(Source {
                graph: graph.clone(),
                pick,
            });
            state.feedback =
                Some("Source feature selected; press and drag a destination feature".to_owned());
        }
    }
    if actions.just_released(GameAction::Primary) && state.weld.drag.take().is_some() {
        if let Some(intent) = state.weld.candidate.take() {
            if simulation.world_revision.is_some() {
                state.weld.publishing = Some(intent.clone());
                state.weld.request = Some(intent);
                state.feedback = Some("Preparing weld placement".to_owned());
            } else {
                match intent.stage(graph, []) {
                    Ok(staged) => {
                        let previous =
                            crate::editor::history::EditorSnapshot::capture(graph, state);
                        *graph = staged;
                        intent.place_sockets(state);
                        history.commit(previous);
                        state.weld.cancel();
                        state.construction_mesh_dirty = true;
                        state.feedback = Some("Welded in the source default pose".to_owned());
                    }
                    Err(error) => state.feedback = Some(error),
                }
            }
        } else {
            state.feedback = Some(
                state
                    .weld
                    .error
                    .clone()
                    .unwrap_or_else(|| "Select a valid destination feature".to_owned()),
            );
        }
    }
}

/// Cancels a gesture begun in another mode, so switching never carries a
/// half-made weld across.
pub(crate) fn sync_mode(state: &mut EditorState, mode: WeldMode) {
    if state.weld.mode == mode {
        return;
    }
    state.weld.cancel();
    state.weld.mode = mode;
    state.hovered = None;
    state.feedback = Some(format!("Welder · {}", mode.label()));
}

/// The nearest part under the ray, whole bodies rather than features.
fn body_pick(
    graph: &ConstructionGraph,
    simulation: &AppSimulation,
    ray: Ray3d,
) -> Option<(PartId, builder::SurfaceHit)> {
    graph
        .parts()
        .filter_map(|(part, _)| {
            let inverse = motion(simulation, part, false).ok()?.inverse();
            builder::raycast_part_in_construction(
                graph,
                part,
                inverse.point(ray.origin),
                inverse.vector(ray.direction.as_vec3()),
            )
            .map(|hit| (part, hit))
        })
        .min_by(|a, b| a.1.distance.total_cmp(&b.1.distance))
}

/// Welds two touching bodies where they are: authored contact in the Garage,
/// current contact in the live world.
fn join_stage(
    graph: &ConstructionGraph,
    simulation: &AppSimulation,
    first: PartId,
    second: PartId,
) -> Result<ConstructionGraph, String> {
    if simulation.world_revision.is_some() && simulation.creation.is_some() {
        crate::live_weld::stage(graph, simulation, first, second)
    } else {
        builder::stage_weld_objects(
            graph,
            mechanic_core::FaceOwner::Part(first),
            mechanic_core::FaceOwner::Part(second),
        )
        .map_err(|error| error.to_string())
    }
}

pub(crate) fn join_hover(
    graph: &ConstructionGraph,
    simulation: &AppSimulation,
    state: &mut EditorState,
    ray: Ray3d,
) {
    state.edit_context = None;
    state.pointer_ray = Some((ray.origin, ray.direction.as_vec3()));
    if state
        .weld
        .join_first
        .as_ref()
        .is_some_and(|(baseline, _)| !baseline.shares_revision(graph))
    {
        state.weld.cancel();
        state.feedback = Some("Weld selection changed; select the first body again".to_owned());
    }
    let hit = body_pick(graph, simulation, ray);
    state.hovered = hit.map(|(_, hit)| hit);
    state.weld.join_hovered = hit.map(|(part, _)| part);
    let Some(first) = state.weld.join_first() else {
        state.weld.join_candidate = None;
        return;
    };
    let Some(second) = state.weld.join_hovered else {
        state.weld.join_candidate = None;
        state.feedback = Some("Select a touching body to weld to".to_owned());
        return;
    };
    let tick = simulation.live_state.as_ref().map(|live| live.tick);
    if !state
        .weld
        .join_candidate
        .as_ref()
        .is_some_and(|(part, at, _)| *part == second && *at == tick)
    {
        state.weld.join_candidate =
            Some((second, tick, join_stage(graph, simulation, first, second)));
    }
    if let Some((_, _, staged)) = &state.weld.join_candidate {
        state.feedback = Some(match staged {
            Ok(staged) => crate::weld_lockup_warning(graph, staged).map_or_else(
                || "Click to weld these bodies where they are".to_owned(),
                |warning| format!("Click to weld — {warning}"),
            ),
            Err(error) => error.clone(),
        });
    }
}

pub(crate) fn join_actions(
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut crate::editor::history::EditorHistory,
    actions: &ButtonInput<GameAction>,
    blocked: bool,
) {
    if actions.just_pressed(GameAction::Secondary) {
        state.weld.cancel();
        state.feedback = Some("Weld cancelled".to_owned());
        return;
    }
    if blocked || !actions.just_pressed(GameAction::Primary) {
        return;
    }
    if state.weld.join_first.is_none() {
        if let Some(part) = state.weld.join_hovered {
            state.weld.join_first = Some((graph.clone(), part));
            state.weld.join_candidate = None;
            state.feedback = Some("First body selected; click a touching body".to_owned());
        } else {
            state.feedback = Some("Select a body".to_owned());
        }
        return;
    }
    match state.weld.join_candidate.take() {
        Some((_, _, Ok(staged))) => {
            let lockup = crate::weld_lockup_warning(graph, &staged);
            let previous = crate::editor::history::EditorSnapshot::capture(graph, state);
            *graph = staged;
            history.commit(previous);
            state.weld.cancel();
            state.construction_mesh_dirty = true;
            state.feedback = Some(lockup.map_or_else(
                || "Welded the two bodies".to_owned(),
                |warning| format!("Welded the two bodies — {warning}"),
            ));
        }
        Some((_, _, Err(error))) => state.feedback = Some(error),
        None => state.feedback = Some("Select a touching body to weld to".to_owned()),
    }
}

/// Uses the same part and joint geometry as the scene, in the authored pose.
pub(crate) fn preview_mesh(
    graph: &ConstructionGraph,
    parts: &[PartId],
    sockets: &[crate::PlacedBearing],
) -> Mesh {
    use bevy::{
        asset::RenderAssetUsages, mesh::Indices, render::render_resource::PrimitiveTopology,
    };
    let mut mesh = crate::frame_visuals::parts_preview_mesh(
        graph,
        &AppSimulation::default(),
        parts,
        None,
        1.0,
    );
    let included =
        |owner| matches!(owner, mechanic_core::FaceOwner::Part(part) if parts.contains(&part));
    let sockets = sockets
        .iter()
        .copied()
        .filter(|socket| included(socket.source.owner))
        .filter(|socket| {
            !graph.bearings().any(|(_, joint)| {
                crate::bearing_uses_socket(joint, *socket) && !included(joint.target.owner)
            })
        })
        .collect::<Vec<_>>();
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut indices = Vec::new();
    for (_, joint) in graph.bearings().filter(|(_, joint)| {
        included(joint.source.owner)
            && included(joint.target.owner)
            && matches!(joint.kind, mechanic_core::BearingKind::Rotational)
            && !sockets
                .iter()
                .any(|&socket| crate::bearing_uses_socket(joint, socket))
    }) {
        crate::render::mesh::bearing::append_bearing_cylinder(
            joint.shared_anchor,
            joint.axis,
            joint.dimensions,
            &mut positions,
            &mut normals,
            &mut Vec::new(),
            &mut Vec::new(),
            &mut indices,
        );
    }
    for socket in &sockets {
        if matches!(socket.kind, mechanic_core::BearingKind::Rotational)
            && let Some(face) = builder::try_face_geometry_from_ref(socket.source, Some(graph))
        {
            crate::render::mesh::bearing::append_bearing_cylinder(
                socket.anchor,
                face.normal,
                socket.dimensions,
                &mut positions,
                &mut normals,
                &mut Vec::new(),
                &mut Vec::new(),
                &mut indices,
            );
        }
    }
    let joints = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(indices));
    mesh.merge(&joints)
        .expect("joint and part previews share triangle attributes");
    mesh.merge(&crate::linear_render::weld_preview_mesh(
        graph, parts, &sockets,
    ))
    .expect("rail and part previews share triangle attributes");
    mesh
}

#[derive(Component)]
pub(crate) struct WeldFeatureVisual;

#[expect(clippy::too_many_arguments)]
pub(crate) fn draw_features(
    graph: Res<crate::EditorGraph>,
    state: Res<EditorState>,
    simulation: Res<AppSimulation>,
    selected: Res<crate::hotbar::SelectedTool>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut overlay: Query<(&Mesh3d, &mut Visibility), With<WeldFeatureVisual>>,
) {
    let geometry = if selected.active_editor_tool() == Some(crate::Tool::Weld) {
        feature_geometry(&graph.0, &state, &simulation)
    } else {
        crate::editor::overlay::OverlayGeometry::default()
    };
    if let Ok((mesh, mut visibility)) = overlay.single_mut() {
        *visibility = crate::editor::overlay::write_overlay(&mut meshes, &mesh.0, geometry);
    } else {
        let mesh = meshes.add(crate::render::mesh::primitives::degenerate_overlay_mesh());
        let visibility = crate::editor::overlay::write_overlay(&mut meshes, &mesh, geometry);
        commands.spawn((
            Name::new("Weld feature highlights"),
            Mesh3d(mesh),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: crate::multitool::WELDER_COLOR,
                unlit: true,
                cull_mode: None,
                ..default()
            })),
            bevy::camera::visibility::RenderLayers::layer(1),
            bevy::camera::visibility::NoFrustumCulling,
            visibility,
            WeldFeatureVisual,
        ));
    }
}

fn feature_geometry(
    graph: &ConstructionGraph,
    state: &EditorState,
    simulation: &AppSimulation,
) -> crate::editor::overlay::OverlayGeometry {
    let mut geometry = crate::editor::overlay::OverlayGeometry::default();
    let target = state
        .weld
        .publishing
        .as_ref()
        .map(weld_publication::Intent::destination)
        .or_else(|| {
            state
                .weld
                .drag
                .as_ref()
                .map(|drag| &drag.destination)
                .or(state.weld.hovered.as_ref())
        });
    for pick in state
        .weld
        .source
        .as_ref()
        .map(|source| &source.pick)
        .into_iter()
        .chain(target)
    {
        let Ok(frame) = motion(simulation, pick.part, false) else {
            continue;
        };
        let selection = pick.selection;
        if let Some(socket) = pick.socket {
            socket::append_outline(socket, frame, &mut geometry);
            continue;
        }
        let owner = graph
            .region_of(pick.part)
            .map_or(SolidOwner::Part(pick.part), SolidOwner::Region);
        if let Ok(solid) = graph.evaluated_solid(owner) {
            let patch = pick.face.patch.or_else(|| {
                graph
                    .part(pick.part)
                    .map(|spec| builder::primitive_surface_patch(*spec, pick.face.face))
            });
            if let Some(patch) = patch {
                append_revealed_vertices(graph, state, owner, patch, frame, &solid, &mut geometry);
            }
            for surface in solid.surfaces.iter().filter(|s| Some(s.key) == patch) {
                let mut edge = surface.half_edge;
                loop {
                    let half = solid.half_edges[edge as usize];
                    let next = solid.half_edges[half.next as usize];
                    if half.logical_edge.is_some() {
                        crate::editor::overlay::append_overlay_bar(
                            frame.point(solid.vertices[half.origin as usize].position),
                            frame.point(solid.vertices[next.origin as usize].position),
                            0.010,
                            &mut geometry,
                        );
                    }
                    edge = half.next;
                    if edge == surface.half_edge {
                        break;
                    }
                }
            }
        }
        match selection.feature {
            WeldFeature::Edge([a, b]) => {
                crate::editor::overlay::append_overlay_bar(
                    frame.point(a),
                    frame.point(b),
                    0.022,
                    &mut geometry,
                );
            }
            WeldFeature::Vertex(point) => {
                append_vertex_marker(frame.point(point), frame.rotation(), 0.054, &mut geometry);
            }
            WeldFeature::Face => {}
        }
    }
    geometry
}

fn append_revealed_vertices(
    graph: &ConstructionGraph,
    state: &EditorState,
    owner: SolidOwner,
    patch: mechanic_core::SurfacePatchKey,
    frame: ConstructionFrame,
    solid: &mechanic_core::EvaluatedSolid,
    geometry: &mut crate::editor::overlay::OverlayGeometry,
) {
    let Some((origin, direction)) = state.pointer_ray else {
        return;
    };
    for (index, vertex) in solid.vertices.iter().enumerate() {
        let point = frame.point(vertex.position);
        let along = (point - origin).dot(direction);
        let distance = (point - origin - direction * along).length();
        if along < 0.0 || distance > crate::shape_tool::VERTEX_REVEAL_RADIUS {
            continue;
        }
        let Ok(index) = u32::try_from(index) else {
            continue;
        };
        if WeldPick::new(
            graph,
            owner,
            mechanic_core::WeldFeatureRef::Vertex(index),
            patch,
            vertex.position,
        )
        .is_ok()
        {
            append_vertex_marker(
                point,
                frame.rotation(),
                crate::shape_tool::vertex_marker_size(distance),
                geometry,
            );
        }
    }
}

fn append_vertex_marker(
    point: Vec3,
    rotation: Quat,
    size: f32,
    geometry: &mut crate::editor::overlay::OverlayGeometry,
) {
    crate::render::mesh::construction::append_transformed_cuboid(
        point,
        rotation,
        Vec3::splat(size),
        &mut geometry.positions,
        &mut geometry.normals,
        &mut geometry.indices,
    );
}

#[cfg(test)]
mod tests;

//! Meshes and transforms that follow published simulation state.

use crate::editor::build_actions::visible_bearing_count;
use crate::editor::preview::{
    BearingVisual, ConstructionVisual, EditorVisuals, FeaturePreviewKey, control_link_count,
    drive_xray_is_visible, feature_drag_preview_graph, feature_preview_key, joint_xray_is_visible,
};
use crate::editor::state::EditorState;
use crate::hotbar::SelectedTool;
use crate::pose::transform_from_gpu;
use crate::render::authored::{AuthoredPart, AuthoredPartVisual};
use crate::render::materials::material_index;
use crate::render::mesh::drive::combined_simulation_drive_xray_mesh;
use crate::render::mesh::primitives::renderable_mesh;
use crate::render::mesh::simulation::{
    SimulationMeshKind, combined_simulation_bearing_mesh, combined_simulation_material_mesh,
    local_simulation_authored_mesh, local_simulation_bearing_mesh, local_simulation_material_mesh,
    simulation_body_has_bearing, simulation_material_is_present,
    simulation_material_is_present_for_compound,
};
use crate::sequencer::DriveSequencer;
use crate::simulation::publication::WorldPhysicsRevision;
use crate::simulation::state::AppSimulation;
use crate::{performance_capture, world};
use bevy::prelude::{
    Assets, Bundle, Commands, Component, Entity, Handle, Material, Mesh, Mesh3d, MeshMaterial3d,
    Name, Or, Query, Res, ResMut, Resource, Transform, Visibility, With, Without, format, vec,
};
use mechanic_core::{ConstructionGraph, ConstructionMaterial, DimensionLinkId};
use mechanic_gpu::GpuTransform;

pub(crate) const SIMULATION_VISUAL_TICK_INTERVAL: u64 = 2;

#[derive(Resource, Default)]
pub(crate) struct SimulationVisualCache {
    pub(crate) revision: Option<WorldPhysicsRevision>,
    pub(crate) active_dimension_link: Option<DimensionLinkId>,
    /// Feature drag drawn into the body meshes, if any.
    pub(crate) feature_preview: Option<FeaturePreviewKey>,
    pub(crate) roots: Vec<Entity>,
}

impl SimulationVisualCache {
    pub(crate) fn needs_rebuild(
        &self,
        revision: WorldPhysicsRevision,
        active_dimension_link: Option<DimensionLinkId>,
    ) -> bool {
        self.revision != Some(revision) || self.active_dimension_link != active_dimension_link
    }
}

#[derive(Component)]
pub(crate) struct SimulationBodyVisualRoot(pub(crate) u32);

#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::type_complexity
)]
pub(crate) fn sync_simulation_visual_cache(
    mut commands: Commands,
    mut simulation: ResMut<AppSimulation>,
    mut cache: ResMut<SimulationVisualCache>,
    visuals: Res<EditorVisuals>,
    state: Res<EditorState>,
    world_runtime: Res<world::WorldRuntime>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut roots: Query<(&SimulationBodyVisualRoot, &mut Transform)>,
    mut legacy_visuals: Query<&mut Visibility, Or<(With<AuthoredPartVisual>, With<BearingVisual>)>>,
) {
    // The published scene owns its visuals even after a solver failure pauses it.
    let Some(revision) = simulation
        .world_revision
        .filter(|_| simulation.gpu.is_some())
    else {
        for entity in cache.roots.drain(..) {
            commands.entity(entity).despawn();
        }
        cache.revision = None;
        cache.active_dimension_link = None;
        cache.feature_preview = None;
        return;
    };
    let Some(creation) = simulation.creation.as_ref() else {
        return;
    };

    let started = std::time::Instant::now();
    let active_dimension_link = world_runtime.active_dimension_link();
    let feature_preview = feature_preview_key(state.feature_drag.as_ref());
    let rebuild = cache.needs_rebuild(revision, active_dimension_link)
        || cache.feature_preview != feature_preview;
    if rebuild {
        for entity in cache.roots.drain(..) {
            commands.entity(entity).despawn();
        }
        let preview_graph = state
            .feature_drag
            .as_ref()
            .and_then(|drag| feature_drag_preview_graph(&simulation.published_graph, drag));
        let graph = preview_graph
            .as_ref()
            .unwrap_or(&simulation.published_graph);
        let local_transforms = vec![
            GpuTransform {
                position: [0.0, 0.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            };
            creation.compounds.len()
        ];
        for (body_index, compound) in creation.compounds.iter().enumerate() {
            let body = u32::try_from(body_index).unwrap_or(u32::MAX);
            // Static construction materials use the shared world mesh, but
            // authored blocks and bearings need body visuals even when grounded.
            let ordinary = ConstructionMaterial::ALL
                .into_iter()
                .filter(|&material| {
                    !compound.is_static
                        && simulation_material_is_present_for_compound(
                            graph, creation, body, material,
                        )
                })
                .map(|material| {
                    let mesh = local_simulation_material_mesh(
                        graph,
                        creation,
                        &local_transforms,
                        body,
                        material,
                    );
                    (
                        meshes.add(mesh),
                        visuals.construction_materials[material_index(material)].clone(),
                        material,
                    )
                })
                .collect::<Vec<_>>();
            let authored = AuthoredPart::ALL
                .into_iter()
                .filter(|&appearance| {
                    creation.part_to_compound.iter().any(|&(part, compound)| {
                        compound == body
                            && graph.part(part).is_some_and(|spec| {
                                appearance.matches(graph, part, *spec, active_dimension_link)
                            })
                    })
                })
                .map(|appearance| {
                    (
                        meshes.add(local_simulation_authored_mesh(
                            graph,
                            creation,
                            &local_transforms,
                            body,
                            appearance,
                            active_dimension_link,
                        )),
                        visuals.authored_materials[appearance.index()].clone(),
                        appearance,
                    )
                })
                .collect::<Vec<_>>();
            let bearing =
                simulation_body_has_bearing(graph, creation, body, &state.placed_bearings).then(
                    || {
                        meshes.add(local_simulation_bearing_mesh(
                            graph,
                            creation,
                            &local_transforms,
                            body,
                            &state.placed_bearings,
                        ))
                    },
                );
            if ordinary.is_empty() && authored.is_empty() && bearing.is_none() {
                continue;
            }
            let transform = simulation
                .transforms
                .get(body_index)
                .copied()
                .map_or(Transform::IDENTITY, transform_from_gpu);
            let root = commands
                .spawn((
                    Name::new(format!("Simulation body {body}")),
                    transform,
                    Visibility::Visible,
                    SimulationBodyVisualRoot(body),
                ))
                .with_children(|root| {
                    for (mesh, material, kind) in ordinary {
                        root.spawn((
                            Name::new(format!("{} local simulation mesh", kind.label())),
                            simulation_body_mesh(mesh, material),
                        ));
                    }
                    for (mesh, material, appearance) in authored {
                        root.spawn((
                            Name::new(format!("{appearance:?} local simulation mesh")),
                            simulation_body_mesh(mesh, material),
                        ));
                    }
                    if let Some(mesh) = bearing {
                        root.spawn((
                            Name::new("Local bearing mesh"),
                            simulation_body_mesh(mesh, visuals.bearing_material.clone()),
                        ));
                    }
                })
                .id();
            cache.roots.push(root);
        }
        cache.revision = Some(revision);
        cache.active_dimension_link = active_dimension_link;
        cache.feature_preview = feature_preview;
        performance_capture::record("body_mesh_rebuild", || {
            serde_json::json!({
                "rebuild_ms": started.elapsed().as_secs_f64() * 1000.0,
                "bodies": creation.compounds.len(),
                "roots": cache.roots.len(),
            })
        });
    } else if simulation.render_dirty {
        for (root, mut transform) in &mut roots {
            if let Some(snapshot) = simulation.transforms.get(root.0 as usize).copied() {
                *transform = transform_from_gpu(snapshot);
            }
        }
    } else {
        return;
    }
    for mut visibility in &mut legacy_visuals {
        *visibility = Visibility::Hidden;
    }
    simulation.record_visual_update(started.elapsed());
}

// Body-local geometry is immutable between scene revisions. Bevy can calculate
// its AABB once and cull using the current propagated compound transform.
pub(crate) fn simulation_body_mesh<M: Material>(
    mesh: Handle<Mesh>,
    material: Handle<M>,
) -> impl Bundle {
    (Mesh3d(mesh), MeshMaterial3d(material))
}

/// Rebuilds the shared meshes that draw a published scene's static construction
/// and its x-ray overlays.
///
/// This owns no terrain state on purpose: static blocks are drawn only from
/// here, so a pending terrain collision publication must never delay them.
#[expect(clippy::too_many_arguments)]
pub(crate) fn refresh_published_construction_visuals(
    simulation: &mut AppSimulation,
    published_graph: &ConstructionGraph,
    state: &EditorState,
    selection: SelectedTool,
    sequencer: &DriveSequencer,
    visuals: &EditorVisuals,
    meshes: &mut Assets<Mesh>,
    construction_visuals: &mut Query<
        (&ConstructionVisual, &mut Visibility),
        Without<BearingVisual>,
    >,
) {
    let visual_started = std::time::Instant::now();

    // A live world draws the published graph, so an uncommitted chamfer or
    // fillet drag must be applied here as well as to the editor meshes.
    let feature_preview = feature_preview_key(state.feature_drag.as_ref());
    if simulation.rendered_feature_preview != feature_preview {
        simulation.static_mesh_dirty = true;
    }
    if simulation.static_mesh_dirty {
        let preview_graph = state
            .feature_drag
            .as_ref()
            .and_then(|drag| feature_drag_preview_graph(published_graph, drag));
        let published_graph = preview_graph.as_ref().unwrap_or(published_graph);
        let creation = simulation
            .creation
            .as_ref()
            .expect("running simulation has compiled creation");
        for material in ConstructionMaterial::ALL {
            let visible = simulation_material_is_present(
                published_graph,
                creation,
                SimulationMeshKind::Static,
                material,
            );
            if visible
                && let Some(mut asset) =
                    meshes.get_mut(&visuals.construction_meshes[material_index(material)])
            {
                *asset = renderable_mesh(combined_simulation_material_mesh(
                    published_graph,
                    creation,
                    &simulation.transforms,
                    SimulationMeshKind::Static,
                    material,
                ));
            }
            for (visual, mut visibility) in construction_visuals.iter_mut() {
                if visual.0 == material {
                    *visibility = if visible {
                        Visibility::Visible
                    } else {
                        Visibility::Hidden
                    };
                }
            }
        }
        simulation.static_mesh_dirty = false;
        simulation.rendered_feature_preview = feature_preview;
        performance_capture::record("static_mesh_rebuild", || {
            serde_json::json!({
                "rebuild_ms": visual_started.elapsed().as_secs_f64() * 1000.0,
                "static_parts": simulation.creation.as_ref().map(|creation| creation
                    .part_to_compound
                    .iter()
                    .filter(|(_, body)| creation.compounds[*body as usize].is_static)
                    .count()),
            })
        });
    }

    if !simulation.render_dirty {
        return;
    }
    let creation = simulation
        .creation
        .as_ref()
        .expect("running simulation has compiled creation");
    let bearings_visible = published_graph.bearing_count() > 0 || !state.placed_bearings.is_empty();
    if bearings_visible
        && joint_xray_is_visible(
            selection.active_editor_tool(),
            visible_bearing_count(published_graph, &state.placed_bearings),
        )
    {
        let rings = combined_simulation_bearing_mesh(
            published_graph,
            creation,
            &simulation.transforms,
            &state.placed_bearings,
        );
        if let Some(mut mesh) = meshes.get_mut(&visuals.joint_xray_mesh) {
            *mesh = renderable_mesh(rings);
        }
    }
    // The drive overlay follows the bodies while they move, so it is rebuilt
    // from the same published snapshot -- but only while it is on screen. A
    // hidden mesh has no slab allocation, so writing to it every frame both
    // wastes the rebuild and makes the renderer log a use-after-free.
    if drive_xray_is_visible(
        selection.active_editor_tool(),
        control_link_count(published_graph),
    ) && let Some(mut mesh) = meshes.get_mut(&visuals.drive_xray_mesh)
    {
        *mesh = combined_simulation_drive_xray_mesh(
            published_graph,
            creation,
            &simulation.transforms,
            &state.placed_bearings,
            sequencer,
        );
    }
    simulation.render_dirty = false;
    simulation.record_visual_update(visual_started.elapsed());
}

pub(crate) const fn visual_snapshot_is_due(snapshot_tick: u64, completed_tick: u64) -> bool {
    completed_tick.saturating_sub(snapshot_tick) >= SIMULATION_VISUAL_TICK_INTERVAL
}

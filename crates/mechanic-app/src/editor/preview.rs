//! Editor visuals: construction meshes, previews, x-rays, and the tool status line.

use crate::builder::{
    BLOCK_SIZE_METERS, PlacementError, bearing_anchor_from_hit_with_grid, face_geometry_from_ref,
    try_face_geometry_from_ref,
};
use crate::chroma::{ChromaBrush, ConstructionRenderMaterial};
use crate::editor::build_actions::visible_bearing_count;
use crate::editor::dimensions::BearingToolSettings;
use crate::editor::hover::DeleteTarget;
use crate::editor::raycast::hovered_part;
use crate::editor::state::{EditorGraph, EditorState};
use crate::hotbar::{SelectedMaterial, SelectedTool, Tool};
use crate::render::authored::{AuthoredPart, AuthoredPartVisual};
use crate::render::materials::material_index;
use crate::render::mesh::bearing::{combined_bearing_mesh, single_bearing_mesh};
use crate::render::mesh::construction::{
    combined_authored_construction_mesh, combined_material_construction_mesh,
    combined_parts_mesh_scaled, ordinary_materials, preview_region, single_cylinder_mesh,
};
use crate::render::mesh::drive::combined_drive_xray_mesh;
use crate::render::mesh::preview::{block_volume_preview_mesh, layer_preview_mesh};
use crate::render::mesh::primitives::renderable_mesh;
use crate::sequencer::DriveSequencer;
use crate::simulation::state::AppSimulation;
use crate::{
    chroma, control_panel, frame_visuals, hotbar, live_edit, performance_capture, shape_tool,
    weld_tool, world,
};
use bevy::prelude::{
    Alpha, Assets, Color, Component, Handle, Local, Mesh, Mesh3d, MeshMaterial3d, Mut, Quat, Query,
    Res, ResMut, Resource, Single, StandardMaterial, Transform, Vec3, Visibility, With, Without,
    format,
};
use mechanic_core::{
    BearingDimensions, BuildCommand, ConstructionEditDelta, ConstructionGraph,
    ConstructionMaterial, CuboidSpec, CylinderDimensions, EngineKind, FaceOwner,
    MaterialAppearance, PartId, PartSpec, PendingOperation, ShapeRegion,
};
use std::collections::HashSet;

#[derive(Resource)]
#[cfg_attr(test, derive(Default))]
pub(crate) struct EditorVisuals {
    pub(crate) construction_meshes: [Handle<Mesh>; ConstructionMaterial::ALL.len()],
    pub(crate) construction_materials:
        [Handle<ConstructionRenderMaterial>; ConstructionMaterial::ALL.len()],
    pub(crate) ghost_materials:
        [Handle<ConstructionRenderMaterial>; ConstructionMaterial::ALL.len()],
    pub(crate) authored_materials: [Handle<StandardMaterial>; AuthoredPart::ALL.len()],
    pub(crate) bearing_material: Handle<StandardMaterial>,
    pub(crate) bearing_mesh: Handle<Mesh>,
    pub(crate) joint_xray_mesh: Handle<Mesh>,
    pub(crate) shape_node_mesh: Handle<Mesh>,
    pub(crate) shape_selected_mesh: Handle<Mesh>,
    pub(crate) shape_plane_mesh: Handle<Mesh>,
    pub(crate) shape_arrow_mesh: Handle<Mesh>,
    pub(crate) controller_mesh: Handle<Mesh>,
    pub(crate) gas_engine_mesh: Handle<Mesh>,
    pub(crate) electric_engine_mesh: Handle<Mesh>,
    pub(crate) gas_transmission_mesh: Handle<Mesh>,
    pub(crate) electric_transmission_mesh: Handle<Mesh>,
    pub(crate) servo_mesh: Handle<Mesh>,
    pub(crate) seat_mesh: Handle<Mesh>,
    pub(crate) input_mesh: Handle<Mesh>,
    pub(crate) dimension_link_disabled_mesh: Handle<Mesh>,
    pub(crate) dimension_link_enabled_mesh: Handle<Mesh>,
    pub(crate) authored_preview_meshes: [Handle<Mesh>; AuthoredPart::ALL.len()],
    pub(crate) authored_preview_materials: [Handle<StandardMaterial>; AuthoredPart::ALL.len()],
    pub(crate) invalid_authored_preview_materials:
        [Handle<StandardMaterial>; AuthoredPart::ALL.len()],
    pub(crate) drive_xray_mesh: Handle<Mesh>,
    pub(crate) wire_drag_mesh: Handle<Mesh>,
    pub(crate) wire_hover_mesh: Handle<Mesh>,
    pub(crate) cube_preview_mesh: Handle<Mesh>,
    pub(crate) cylinder_preview_mesh: Handle<Mesh>,
    pub(crate) bearing_preview_mesh: Handle<Mesh>,
    pub(crate) white_preview_material: Handle<StandardMaterial>,
    pub(crate) chroma_preview_material: Handle<StandardMaterial>,
    pub(crate) green_preview_material: Handle<StandardMaterial>,
    pub(crate) red_preview_material: Handle<StandardMaterial>,
    /// Allowed, but with a consequence worth seeing first.
    pub(crate) amber_preview_material: Handle<StandardMaterial>,
    pub(crate) block_drag_preview_mesh: Handle<Mesh>,
    pub(crate) delete_drag_preview_mesh: Handle<Mesh>,
    pub(crate) weld_hover_preview_mesh: Handle<Mesh>,
    pub(crate) weld_selection_preview_mesh: Handle<Mesh>,
}

#[derive(Default)]
pub(crate) struct PreviewMeshRevisions {
    pub(crate) construction: Option<ConstructionPreviewMeshKey>,
    pub(crate) cylinder: Option<CylinderDimensions>,
    pub(crate) delete: u64,
}

#[derive(Clone, PartialEq)]
pub(crate) enum ConstructionPreviewMeshKey {
    Block(u64),
    Pipe(Vec<PartSpec>),
    Branch(mechanic_core::PipeJunctionSpec, mechanic_core::CylinderSpec),
    Layer(Vec<PartSpec>),
}

pub(crate) fn sync_preview_mesh<K: PartialEq>(
    meshes: &mut Assets<Mesh>,
    handle: &Handle<Mesh>,
    rendered: &mut Option<K>,
    key: K,
    build: impl FnOnce() -> Mesh,
) {
    if rendered.as_ref() != Some(&key)
        && let Some(mut mesh) = meshes.get_mut(handle)
    {
        *mesh = build();
        *rendered = Some(key);
    }
}

impl EditorVisuals {
    pub(crate) fn authored_mesh(&self, appearance: AuthoredPart) -> &Handle<Mesh> {
        match appearance {
            AuthoredPart::Controller => &self.controller_mesh,
            AuthoredPart::GasEngine => &self.gas_engine_mesh,
            AuthoredPart::ElectricEngine => &self.electric_engine_mesh,
            AuthoredPart::GasTransmission => &self.gas_transmission_mesh,
            AuthoredPart::ElectricTransmission => &self.electric_transmission_mesh,
            AuthoredPart::Servo => &self.servo_mesh,
            AuthoredPart::Seat => &self.seat_mesh,
            AuthoredPart::Input => &self.input_mesh,
            AuthoredPart::DimensionLinkDisabled => &self.dimension_link_disabled_mesh,
            AuthoredPart::DimensionLinkEnabled => &self.dimension_link_enabled_mesh,
        }
    }

    pub(crate) fn authored_preview_mesh(&self, appearance: AuthoredPart) -> &Handle<Mesh> {
        &self.authored_preview_meshes[appearance.index()]
    }

    pub(crate) fn authored_preview_material(
        &self,
        appearance: AuthoredPart,
        invalid: bool,
    ) -> &Handle<StandardMaterial> {
        let materials = if invalid {
            &self.invalid_authored_preview_materials
        } else {
            &self.authored_preview_materials
        };
        &materials[appearance.index()]
    }
}

#[derive(Component)]
pub(crate) struct ActionPreview;

#[derive(Component)]
pub(crate) struct SelectionPreview;

#[derive(Component)]
pub(crate) struct DeletePreview;

#[derive(Component)]
pub(crate) struct ConstructionVisual(pub(crate) ConstructionMaterial);

#[derive(Component)]
pub(crate) struct BearingVisual;

#[derive(Component)]
pub(crate) struct JointXrayVisual;

#[derive(Component)]
pub(crate) struct DriveXrayVisual;

pub(crate) const DELETE_PREVIEW_SCALE: f32 = 1.015;

/// Matches the 0.992 scale of a single 0.25 m block preview without making
/// large sheet previews shrink in proportion to their full width.
pub(crate) const BLOCK_SHEET_PREVIEW_INSET_METERS: f32 = 0.001;

#[expect(
    clippy::type_complexity,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
pub(crate) fn sync_visual_meshes(
    graph: Res<EditorGraph>,
    world_runtime: Res<world::WorldRuntime>,
    mirror: Res<shape_tool::ShapeMirror>,
    sequencer: Res<DriveSequencer>,
    selection: Res<SelectedTool>,
    simulation: Res<AppSimulation>,
    mut state: ResMut<EditorState>,
    visuals: Res<EditorVisuals>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut construction_visuals: Query<(&ConstructionVisual, &mut Visibility), Without<BearingVisual>>,
    mut bearing_visibility: Single<
        &mut Visibility,
        (With<BearingVisual>, Without<ConstructionVisual>),
    >,
    mut authored_visuals: Query<
        (&AuthoredPartVisual, &mut Visibility),
        (Without<ConstructionVisual>, Without<BearingVisual>),
    >,
) {
    // A new static publication takes ownership back from the moving-body meshes.
    // Even unchanged materials need their meshes and visibility restored.
    // A failed live scene still owns its last poses; do not redraw authored poses.
    let publication_changed = state.rendered_world_revision != simulation.world_revision;
    if !should_sync_editor_visual_meshes(
        state.construction_mesh_dirty || publication_changed,
        simulation.gpu.is_some(),
    ) {
        return;
    }
    let sync_started = std::time::Instant::now();
    let edit_delta = ConstructionEditDelta::between(&state.rendered_graph, &graph.0);
    let rebuild_all = publication_changed || edit_delta.is_empty();
    let affected_parts = edit_delta.affected_parts();
    let mut dirty_materials = affected_parts
        .iter()
        .flat_map(|&part| {
            graph
                .0
                .part(part)
                .into_iter()
                .chain(state.rendered_graph.part(part))
                .copied()
                .flat_map(ordinary_materials)
        })
        .collect::<HashSet<_>>();
    for &region in &edit_delta.region_owned_geometry {
        if let Some(material) = graph
            .0
            .region(region)
            .or_else(|| state.rendered_graph.region(region))
            .map(ShapeRegion::material)
        {
            dirty_materials.insert(material);
        }
    }
    let feature_preview_graph = state
        .feature_drag
        .as_ref()
        .and_then(|drag| feature_drag_preview_graph(&graph.0, drag));
    let mesh_graph = feature_preview_graph.as_ref().unwrap_or(&graph.0);
    let preview = preview_region(&graph.0, &state, *mirror);
    let active_dimension_link = world_runtime.active_dimension_link();
    for material in ConstructionMaterial::ALL {
        if !rebuild_all && !dirty_materials.contains(&material) {
            continue;
        }
        let mesh = combined_material_construction_mesh(mesh_graph, preview.as_ref(), material);
        let visible = mesh.count_vertices() > 0;
        if let Some(mut asset) =
            meshes.get_mut(&visuals.construction_meshes[material_index(material)])
        {
            *asset = renderable_mesh(mesh);
        }
        for (visual, mut visibility) in &mut construction_visuals {
            if visual.0 == material {
                *visibility = if visible {
                    Visibility::Visible
                } else {
                    Visibility::Hidden
                };
            }
        }
    }
    for appearance in AuthoredPart::ALL {
        let appearance_changed = rebuild_all
            || affected_parts.iter().any(|&part| {
                graph.0.part(part).is_some_and(|spec| {
                    appearance.matches(&graph.0, part, *spec, active_dimension_link)
                }) || state.rendered_graph.part(part).is_some_and(|spec| {
                    appearance.matches(&state.rendered_graph, part, *spec, active_dimension_link)
                })
            });
        if !appearance_changed {
            continue;
        }
        let visible = graph
            .0
            .parts()
            .any(|(part, spec)| appearance.matches(&graph.0, part, *spec, active_dimension_link));
        if visible && let Some(mut mesh) = meshes.get_mut(visuals.authored_mesh(appearance)) {
            *mesh =
                combined_authored_construction_mesh(&graph.0, appearance, active_dimension_link);
        }
        for (visual, mut visibility) in &mut authored_visuals {
            if visual.0 == appearance {
                *visibility = if visible {
                    Visibility::Visible
                } else {
                    Visibility::Hidden
                };
            }
        }
    }
    if graph.0.bearing_count() == 0 && state.placed_bearings.is_empty() {
        **bearing_visibility = Visibility::Hidden;
    } else {
        let rings = combined_bearing_mesh(&graph.0, &state.placed_bearings);
        if joint_xray_is_visible(
            selection.active_editor_tool(),
            visible_bearing_count(&graph.0, &state.placed_bearings),
        ) && let Some(mut mesh) = meshes.get_mut(&visuals.joint_xray_mesh)
        {
            *mesh = renderable_mesh(rings.clone());
        }
        let has_rings = rings.count_vertices() > 0;
        if let Some(mut mesh) = meshes.get_mut(&visuals.bearing_mesh) {
            *mesh = renderable_mesh(rings);
        }
        **bearing_visibility = if has_rings {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
    if drive_xray_is_visible(selection.active_editor_tool(), control_link_count(&graph.0))
        && let Some(mut mesh) = meshes.get_mut(&visuals.drive_xray_mesh)
    {
        *mesh = combined_drive_xray_mesh(&graph.0, &state.placed_bearings, &sequencer);
    }
    state.rendered_graph = graph.0.clone();
    state.rendered_world_revision = simulation.world_revision;
    state.construction_mesh_dirty = false;
    performance_capture::record("visual_mesh_sync", || {
        serde_json::json!({
            "rebuild_all": rebuild_all,
            "dirty_materials": dirty_materials.len(),
            "sync_ms": sync_started.elapsed().as_secs_f64() * 1000.0,
        })
    });
}

pub(crate) const fn should_sync_editor_visual_meshes(
    dirty: bool,
    simulation_running: bool,
) -> bool {
    dirty && !simulation_running
}

/// Identity of the feature a drag previews, so meshes rebuild when the previewed
/// amount changes rather than on every pointer sample.
pub(crate) type FeaturePreviewKey = (
    Option<mechanic_core::ShapeFeatureId>,
    Vec<mechanic_core::EdgeChainRef>,
    mechanic_core::EdgeTreatment,
    u32,
);

pub(crate) fn feature_preview_key(
    drag: Option<&shape_tool::FeatureDrag>,
) -> Option<FeaturePreviewKey> {
    drag.filter(|drag| drag.amount_ticks > 0).map(|drag| {
        (
            drag.feature,
            drag.targets.clone(),
            drag.treatment,
            drag.amount_ticks,
        )
    })
}

/// Applies an in-progress chamfer or fillet drag to a copy of `graph`.
///
/// The editor meshes and a live world's published meshes both draw from this,
/// so the drag previews wherever the construction is rendered.
pub(crate) fn feature_drag_preview_graph(
    graph: &ConstructionGraph,
    drag: &shape_tool::FeatureDrag,
) -> Option<ConstructionGraph> {
    if drag.amount_ticks == 0 {
        return None;
    }
    if let Some(preview) = &drag.validated_preview
        && preview.source.shares_revision(graph)
        && Some(&preview.key) == feature_preview_key(Some(drag)).as_ref()
    {
        return Some(preview.graph.clone());
    }
    let mut preview = graph.clone();
    let command = if let Some(feature) = drag.feature {
        BuildCommand::SetShapeFeatureAmount {
            feature,
            amount_ticks: drag.amount_ticks,
        }
    } else {
        BuildCommand::AddShapeFeature(mechanic_core::ShapeFeature::new(
            drag.targets.clone(),
            drag.treatment,
            drag.amount_ticks,
        ))
    };
    preview.apply(command).ok().map(|_| preview)
}

/// The bearing x-ray is also shown while wiring, so a drive wire can be traced
/// back through the construction to the block that owns it.
pub(crate) fn joint_xray_is_visible(tool: impl Into<Option<Tool>>, bearing_count: usize) -> bool {
    matches!(tool.into(), Some(Tool::Controller | Tool::Connector)) && bearing_count > 0
}

/// The drive overlay additionally stays up while simulating, so the joint a key
/// is driving can be seen moving. Its meshes are rebuilt from each published
/// snapshot, so the arcs and wires track the running bodies.
pub(crate) fn drive_xray_is_visible(tool: impl Into<Option<Tool>>, driven_count: usize) -> bool {
    matches!(tool.into(), Some(Tool::Controller | Tool::Connector)) && driven_count > 0
}

/// Number and world position of every driven joint.
///
/// Numbering comes from [`control_panel::panel_rows`], the same grouping the
/// panel lists, so `Joint 3` in the table is the joint wearing a floating `3`.
/// Two wires on one physical joint share a row, and so share one label.
pub(crate) fn joint_number_labels(
    graph: &ConstructionGraph,
    anchor_of: impl Fn(&mechanic_core::BearingSpec) -> Option<Vec3>,
) -> Vec<(usize, Vec3)> {
    let mut labels = Vec::new();
    for (controller, _) in graph.parts().filter(|(id, _)| graph.is_controller(*id)) {
        for (index, row) in control_panel::panel_rows(graph, controller)
            .iter()
            .enumerate()
        {
            let Some(bearing) = graph
                .drive_link(row.primary)
                .and_then(|link| graph.bearing(link.bearing))
            else {
                continue;
            };
            if let Some(anchor) = anchor_of(bearing) {
                labels.push((index + 1, anchor));
            }
        }
    }
    labels
}

pub(crate) fn driven_bearing_count(graph: &ConstructionGraph) -> usize {
    graph
        .drive_links()
        .filter(|(_, link)| graph.is_controller(link.controller))
        .count()
}

pub(crate) fn control_link_count(graph: &ConstructionGraph) -> usize {
    driven_bearing_count(graph)
        + graph.input_seat_links().count()
        + graph.seat_controller_links().count()
}

pub(crate) fn update_joint_xray(
    graph: Res<EditorGraph>,
    mut state: ResMut<EditorState>,
    mut simulation: ResMut<AppSimulation>,
    selection: Res<SelectedTool>,
    mut drive_visibility: Single<
        &mut Visibility,
        (With<DriveXrayVisual>, Without<JointXrayVisual>),
    >,
    mut visibility: Single<&mut Visibility, (With<JointXrayVisual>, Without<DriveXrayVisual>)>,
) {
    let drive_visible =
        drive_xray_is_visible(selection.active_editor_tool(), control_link_count(&graph.0));
    let joint_visible = joint_xray_is_visible(
        selection.active_editor_tool(),
        visible_bearing_count(&graph.0, &state.placed_bearings),
    );
    // An overlay's mesh is left alone while it is hidden, so it has to be
    // rebuilt on the frame it comes back. This system runs ahead of both mesh
    // builders, so the request is served without a stale frame.
    if (drive_visible && **drive_visibility == Visibility::Hidden)
        || (joint_visible && **visibility == Visibility::Hidden)
    {
        state.construction_mesh_dirty = true;
        if simulation.is_running() {
            simulation.render_dirty = true;
        }
    }
    **drive_visibility = if drive_visible {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };
    **visibility = if joint_visible {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };
}

#[derive(bevy::ecs::system::SystemParam)]
pub(crate) struct ChromaPreviewParams<'w> {
    pub(crate) selected_material: Res<'w, SelectedMaterial>,
    pub(crate) brush: Res<'w, ChromaBrush>,
    pub(crate) materials: ResMut<'w, Assets<StandardMaterial>>,
}

#[expect(
    clippy::type_complexity,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]
pub(crate) fn update_previews(
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
    simulation: Res<AppSimulation>,
    selected_tool: Res<SelectedTool>,
    mut chroma: ChromaPreviewParams,
    bearing_settings: Res<BearingToolSettings>,
    visuals: Res<EditorVisuals>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut rendered_revisions: Local<PreviewMeshRevisions>,
    mut rendered_bearing_dimensions: Local<BearingDimensions>,
    mut rendered_weld_hover: Local<Option<PartId>>,
    mut rendered_weld_selection: Local<Option<PartId>>,
    mut action: Single<
        (
            &mut Mesh3d,
            &mut Transform,
            &mut Visibility,
            &mut MeshMaterial3d<StandardMaterial>,
        ),
        (
            With<ActionPreview>,
            Without<SelectionPreview>,
            Without<DeletePreview>,
        ),
    >,
    mut selection: Single<
        (
            &mut Mesh3d,
            &mut Transform,
            &mut Visibility,
            &mut MeshMaterial3d<StandardMaterial>,
        ),
        (
            With<SelectionPreview>,
            Without<ActionPreview>,
            Without<DeletePreview>,
        ),
    >,
    mut delete: Single<
        (
            &mut Mesh3d,
            &mut Transform,
            &mut Visibility,
            &mut MeshMaterial3d<StandardMaterial>,
        ),
        (
            With<DeletePreview>,
            Without<ActionPreview>,
            Without<SelectionPreview>,
        ),
    >,
) {
    if selected_tool.active_editor_tool() == Some(Tool::Weld) {
        hide_preview(&mut action.2);
        hide_preview(&mut selection.2);
        hide_preview(&mut delete.2);
        if selected_tool.weld_mode == hotbar::WeldMode::Join {
            if let Some(part) = state.weld.join_hovered {
                if let Some(mut mesh) = meshes.get_mut(&visuals.weld_hover_preview_mesh) {
                    *mesh = frame_visuals::weld_preview_mesh(&graph.0, &simulation, part, None);
                }
                action.0.0 = visuals.weld_hover_preview_mesh.clone();
                *action.1 = Transform::default();
                action.3.0 = if state.weld.join_valid() == Some(false) {
                    visuals.red_preview_material.clone()
                } else {
                    visuals.green_preview_material.clone()
                };
                *action.2 = Visibility::Visible;
            }
            if let Some(part) = state.weld.join_first() {
                if let Some(mut mesh) = meshes.get_mut(&visuals.weld_selection_preview_mesh) {
                    *mesh = frame_visuals::weld_preview_mesh(&graph.0, &simulation, part, None);
                }
                selection.0.0 = visuals.weld_selection_preview_mesh.clone();
                *selection.1 = Transform::default();
                selection.3.0 = visuals.white_preview_material.clone();
                *selection.2 = Visibility::Visible;
            }
            return;
        }
        if let Some((preview_graph, parts, frame)) = &state.weld.preview {
            if let Some(mut mesh) = meshes.get_mut(&visuals.weld_hover_preview_mesh) {
                *mesh = weld_tool::preview_mesh(preview_graph, parts, &state.placed_bearings);
            }
            action.0.0 = visuals.weld_hover_preview_mesh.clone();
            *action.1 =
                Transform::from_translation(frame.translation()).with_rotation(frame.rotation());
            action.3.0 = if state.weld.error.is_some() {
                visuals.red_preview_material.clone()
            } else {
                visuals.green_preview_material.clone()
            };
            *action.2 = Visibility::Visible;
        }
        return;
    }
    let mut view = live_edit::EditorView::new(&mut graph, &mut state);
    let (graph, state) = view.parts();
    hide_preview(&mut action.2);
    hide_preview(&mut selection.2);
    hide_preview(&mut delete.2);

    if selected_tool.active_editor_tool() == Some(Tool::Weld) {
        if hovered_part(state.hovered).is_none() {
            *rendered_weld_hover = None;
        }
        if !matches!(graph.0.pending(), Some(PendingOperation::Weld(_))) {
            *rendered_weld_selection = None;
        }
    } else {
        *rendered_weld_hover = None;
        *rendered_weld_selection = None;
    }

    if let Some(drag) = state.delete_drag.as_ref() {
        if let Some(mut mesh) = meshes.get_mut(&visuals.delete_drag_preview_mesh) {
            *mesh = frame_visuals::parts_preview_mesh(
                &graph.0,
                &simulation,
                &drag.parts,
                state.edit_context,
                DELETE_PREVIEW_SCALE,
            );
        }
        rendered_revisions.delete = state.delete_preview_revision;
        delete.0.0 = visuals.delete_drag_preview_mesh.clone();
        *delete.1 = Transform::default();
        delete.3.0 = visuals.red_preview_material.clone();
        *delete.2 = Visibility::Visible;
        return;
    }

    if let Some(target) = state.delete_target {
        match target {
            DeleteTarget::PlacedBearing(index) => {
                if let Some(bearing) = state.placed_bearings.get(index) {
                    let normal = face_geometry_from_ref(bearing.source, Some(&graph.0)).normal;
                    update_bearing_preview_mesh(
                        &mut meshes,
                        &visuals.bearing_preview_mesh,
                        &mut rendered_bearing_dimensions,
                        bearing.dimensions,
                    );
                    show_bearing_preview(
                        &mut delete,
                        &visuals.bearing_preview_mesh,
                        &visuals.red_preview_material,
                        bearing.anchor,
                        normal,
                    );
                }
            }
            DeleteTarget::Part(part) => {
                if graph.0.part(part).is_some() {
                    if let Some(mut mesh) = meshes.get_mut(&visuals.delete_drag_preview_mesh) {
                        *mesh = frame_visuals::parts_preview_mesh(
                            &graph.0,
                            &simulation,
                            &[part],
                            state.edit_context,
                            DELETE_PREVIEW_SCALE,
                        );
                    }
                    delete.0.0 = visuals.delete_drag_preview_mesh.clone();
                    *delete.1 = Transform::default();
                    delete.3.0 = visuals.red_preview_material.clone();
                    *delete.2 = Visibility::Visible;
                }
            }
        }
        return;
    }

    let bearing_attachment_highlighted = selected_tool.active_editor_tool().is_some_and(|tool| {
        bearing_attachment_is_highlighted(
            tool,
            state.attachment_bearing,
            state.preview_error.as_ref(),
        )
    });
    let chroma_preview = matches!(
        selected_tool.active_editor_tool(),
        Some(Tool::Block | Tool::Cylinder)
    ) && chroma.brush.appearance != MaterialAppearance::BAKED;
    if chroma_preview {
        let [r, g, b] =
            chroma::representative_srgb(chroma.selected_material.0, chroma.brush.appearance);
        if let Some(mut material) = chroma.materials.get_mut(&visuals.chroma_preview_material) {
            material.base_color = Color::srgb_u8(r, g, b).with_alpha(0.46);
        }
    }
    let action_material = if state.preview_error.is_some() {
        &visuals.red_preview_material
    } else if state.preview_warning.is_some() {
        &visuals.amber_preview_material
    } else if bearing_attachment_highlighted {
        &visuals.green_preview_material
    } else if chroma_preview {
        &visuals.chroma_preview_material
    } else {
        &visuals.white_preview_material
    };
    if let Some(bearing) = state
        .attachment_bearing
        .and_then(|index| state.placed_bearings.get(index))
    {
        let normal = face_geometry_from_ref(bearing.source, Some(&graph.0)).normal;
        update_bearing_preview_mesh(
            &mut meshes,
            &visuals.bearing_preview_mesh,
            &mut rendered_bearing_dimensions,
            bearing.dimensions,
        );
        show_bearing_preview(
            &mut selection,
            &visuals.bearing_preview_mesh,
            if bearing_attachment_highlighted {
                &visuals.green_preview_material
            } else {
                &visuals.red_preview_material
            },
            bearing.anchor,
            normal,
        );
        selection.1.scale = Vec3::splat(1.12);
    }
    match (selected_tool.active_editor_tool(), graph.0.pending()) {
        (None | Some(Tool::Shape | Tool::Chroma), _) => {
            *action.2 = Visibility::Hidden;
        }
        (Some(Tool::Block), _) => {
            if let Some(drag) = state.block_drag.as_ref() {
                sync_preview_mesh(
                    &mut meshes,
                    &visuals.block_drag_preview_mesh,
                    &mut rendered_revisions.construction,
                    ConstructionPreviewMeshKey::Block(state.block_preview_revision),
                    || block_volume_preview_mesh(drag.volume),
                );
                action.0.0 = visuals.block_drag_preview_mesh.clone();
                *action.1 = Transform::default();
                action.3.0 = action_material.clone();
                *action.2 = Visibility::Visible;
            } else if let Some(candidate) = state.preview {
                show_cuboid_preview(
                    &mut action,
                    &visuals.cube_preview_mesh,
                    action_material,
                    candidate.spec,
                    0.992,
                );
            }
        }
        (Some(Tool::Cylinder), _) => {
            if let Some(drag) = state.pipe_drag.as_ref() {
                let specs = drag
                    .pieces
                    .iter()
                    .map(|piece| piece.spec)
                    .chain(
                        drag.branch
                            .map(|branch| PartSpec::PipeJunction(branch.junction)),
                    )
                    .collect::<Vec<_>>();
                sync_preview_mesh(
                    &mut meshes,
                    &visuals.block_drag_preview_mesh,
                    &mut rendered_revisions.construction,
                    ConstructionPreviewMeshKey::Pipe(specs.clone()),
                    || combined_parts_mesh_scaled(&specs, 1.0),
                );
                action.0.0 = visuals.block_drag_preview_mesh.clone();
                *action.1 = Transform::default();
                action.3.0 = action_material.clone();
                *action.2 = Visibility::Visible;
            } else if let Some((candidate, branch)) =
                state.cylinder_preview.zip(state.pipe_branch_preview)
            {
                // A branch previews its junction and the pipe leaving it.
                sync_preview_mesh(
                    &mut meshes,
                    &visuals.block_drag_preview_mesh,
                    &mut rendered_revisions.construction,
                    ConstructionPreviewMeshKey::Branch(branch.junction, candidate.spec),
                    || {
                        combined_parts_mesh_scaled(
                            &[
                                PartSpec::PipeJunction(branch.junction),
                                PartSpec::Cylinder(candidate.spec),
                            ],
                            1.004,
                        )
                    },
                );
                action.0.0 = visuals.block_drag_preview_mesh.clone();
                *action.1 = Transform::default();
                action.3.0 = action_material.clone();
                *action.2 = Visibility::Visible;
            } else if let Some(candidate) = state.cylinder_preview {
                sync_preview_mesh(
                    &mut meshes,
                    &visuals.cylinder_preview_mesh,
                    &mut rendered_revisions.cylinder,
                    candidate.spec.dimensions,
                    || single_cylinder_mesh(candidate.spec.dimensions),
                );
                show_cylinder_preview(
                    &mut action,
                    &visuals.cylinder_preview_mesh,
                    action_material,
                    candidate.spec,
                );
            }
        }
        (Some(Tool::Layer), _) => {
            // A drag follows the pointer off the part, so it previews from
            // the drag itself rather than from the hovered surface.
            let layered = match &state.layer_drag {
                Some(drag) => crate::builder::layered_parts(
                    &drag.target,
                    drag.thickness,
                    drag.material,
                    drag.appearance,
                )
                .ok()
                .map(|layered| (drag.target.frame, layered)),
                None => state
                    .layer_preview
                    .as_ref()
                    .map(|preview| (preview.target.frame, preview.layered.clone())),
            };
            if let Some((frame, layered)) = layered {
                let specs = layered.iter().map(|&(_, spec)| spec).collect::<Vec<_>>();
                sync_preview_mesh(
                    &mut meshes,
                    &visuals.block_drag_preview_mesh,
                    &mut rendered_revisions.construction,
                    ConstructionPreviewMeshKey::Layer(specs.clone()),
                    || layer_preview_mesh(&specs),
                );
                action.0.0 = visuals.block_drag_preview_mesh.clone();
                *action.1 = Transform::from_translation(frame.translation())
                    .with_rotation(frame.rotation());
                action.3.0 = action_material.clone();
                *action.2 = Visibility::Visible;
            } else {
                *action.2 = Visibility::Hidden;
            }
        }
        (Some(Tool::Weld), pending) => {
            if let Some(part) = state
                .world_hovered_part
                .or_else(|| hovered_part(state.hovered))
            {
                if let Some(mut mesh) = meshes.get_mut(&visuals.weld_hover_preview_mesh) {
                    *mesh = frame_visuals::weld_preview_mesh(
                        &graph.0,
                        &simulation,
                        part,
                        state.edit_context,
                    );
                }
                *rendered_weld_hover = Some(part);
                action.0.0 = visuals.weld_hover_preview_mesh.clone();
                *action.1 = Transform::default();
                action.3.0 = action_material.clone();
                *action.2 = Visibility::Visible;
            }
            if let Some(PendingOperation::Weld(first)) = pending
                && let FaceOwner::Part(part) = first.owner
            {
                if let Some(mut mesh) = meshes.get_mut(&visuals.weld_selection_preview_mesh) {
                    *mesh = frame_visuals::weld_preview_mesh(
                        &graph.0,
                        &simulation,
                        part,
                        state.edit_context,
                    );
                }
                *rendered_weld_selection = Some(part);
                selection.0.0 = visuals.weld_selection_preview_mesh.clone();
                *selection.1 = Transform::default();
                selection.3.0 = visuals.white_preview_material.clone();
                *selection.2 = Visibility::Visible;
            }
        }
        (Some(Tool::Bearing), _) => {
            if let Some(hit) = state.hovered
                && let Some(face) = try_face_geometry_from_ref(hit.face, Some(&graph.0))
            {
                let anchor = state.bearing_preview_anchor.unwrap_or_else(|| {
                    bearing_anchor_from_hit_with_grid(
                        &graph.0,
                        hit,
                        state.placement_grid,
                        state.placement_bounds,
                    )
                    .unwrap_or(hit.point)
                });
                update_bearing_preview_mesh(
                    &mut meshes,
                    &visuals.bearing_preview_mesh,
                    &mut rendered_bearing_dimensions,
                    bearing_settings.dimensions,
                );
                show_bearing_preview(
                    &mut action,
                    &visuals.bearing_preview_mesh,
                    action_material,
                    anchor,
                    face.normal,
                );
            }
        }
        (
            Some(
                Tool::Controller
                | Tool::GasEngine
                | Tool::ElectricEngine
                | Tool::Servo
                | Tool::Seat
                | Tool::Input
                | Tool::DimensionLink,
            ),
            _,
        ) => {
            if let (Some(candidate), Some(appearance)) = (
                state.preview,
                selected_tool
                    .active_editor_tool()
                    .and_then(AuthoredPart::from_tool),
            ) {
                show_cuboid_preview(
                    &mut action,
                    visuals.authored_preview_mesh(appearance),
                    visuals.authored_preview_material(appearance, state.preview_error.is_some()),
                    candidate.spec,
                    0.992,
                );
            }
        }
        (Some(Tool::Transmission), _) => {
            if let Some(candidate) = state.preview {
                let kind = state.hovered.and_then(|hit| match hit.face.owner {
                    FaceOwner::Part(part) => match graph.0.part(part) {
                        Some(PartSpec::Engine(engine)) => Some(engine.kind),
                        Some(PartSpec::Transmission(_)) => graph.0.transmission_kind(part),
                        _ => None,
                    },
                    FaceOwner::Ground => None,
                });
                let appearance = match kind.unwrap_or(EngineKind::Electric) {
                    EngineKind::Gas => AuthoredPart::GasTransmission,
                    EngineKind::Electric => AuthoredPart::ElectricTransmission,
                };
                show_cuboid_preview(
                    &mut action,
                    visuals.authored_preview_mesh(appearance),
                    visuals.authored_preview_material(appearance, state.preview_error.is_some()),
                    candidate.spec,
                    0.992,
                );
            }
        }
        (
            Some(Tool::Hammer | Tool::Connector | Tool::LinearBearing | Tool::Spring | Tool::Shock),
            _,
        ) => {}
    }
}

pub(crate) fn update_bearing_preview_mesh(
    meshes: &mut Assets<Mesh>,
    mesh_handle: &Handle<Mesh>,
    rendered_dimensions: &mut BearingDimensions,
    dimensions: BearingDimensions,
) {
    if !bearing_preview_dimensions_changed(rendered_dimensions, dimensions) {
        return;
    }
    if let Some(mut mesh) = meshes.get_mut(mesh_handle) {
        *mesh = single_bearing_mesh(dimensions);
    }
}

pub(crate) fn bearing_preview_dimensions_changed(
    rendered_dimensions: &mut BearingDimensions,
    dimensions: BearingDimensions,
) -> bool {
    if *rendered_dimensions == dimensions {
        false
    } else {
        *rendered_dimensions = dimensions;
        true
    }
}

pub(crate) fn bearing_attachment_is_highlighted(
    tool: Tool,
    attachment_bearing: Option<usize>,
    preview_error: Option<&PlacementError>,
) -> bool {
    matches!(tool, Tool::Block | Tool::Cylinder)
        && attachment_bearing.is_some()
        && preview_error.is_none()
}

pub(crate) type PreviewItem<'a> = (
    Mut<'a, Mesh3d>,
    Mut<'a, Transform>,
    Mut<'a, Visibility>,
    Mut<'a, MeshMaterial3d<StandardMaterial>>,
);

pub(crate) fn hide_preview(visibility: &mut Visibility) {
    *visibility = Visibility::Hidden;
}

pub(crate) fn show_cuboid_preview(
    preview: &mut PreviewItem<'_>,
    mesh_handle: &Handle<Mesh>,
    material_handle: &Handle<StandardMaterial>,
    spec: CuboidSpec,
    scale_factor: f32,
) {
    preview.0.0 = mesh_handle.clone();
    *preview.1 = Transform::from_translation(spec.pose.translation())
        .with_rotation(spec.pose.rotation.quaternion())
        .with_scale(spec.size_meters() * scale_factor);
    preview.3.0 = material_handle.clone();
    *preview.2 = Visibility::Visible;
}

pub(crate) fn show_bearing_preview(
    preview: &mut PreviewItem<'_>,
    mesh_handle: &Handle<Mesh>,
    material_handle: &Handle<StandardMaterial>,
    anchor: Vec3,
    normal: Vec3,
) {
    preview.0.0 = mesh_handle.clone();
    *preview.1 =
        Transform::from_translation(anchor).with_rotation(Quat::from_rotation_arc(Vec3::Y, normal));
    preview.3.0 = material_handle.clone();
    *preview.2 = Visibility::Visible;
}

pub(crate) fn show_cylinder_preview(
    preview: &mut PreviewItem<'_>,
    mesh_handle: &Handle<Mesh>,
    material_handle: &Handle<StandardMaterial>,
    spec: mechanic_core::CylinderSpec,
) {
    preview.0.0 = mesh_handle.clone();
    *preview.1 = Transform::from_translation(spec.pose.translation())
        .with_rotation(spec.pose.rotation.quaternion())
        .with_scale(Vec3::splat(0.992));
    preview.3.0 = material_handle.clone();
    *preview.2 = Visibility::Visible;
}

// Tool-specific guidance is kept together with its HUD layout.
pub(crate) fn tool_status_line(
    tool: impl Into<Option<Tool>>,
    bearing_dimensions: BearingDimensions,
    cylinder_dimensions: CylinderDimensions,
    selected_wires: Option<usize>,
    material: ConstructionMaterial,
) -> String {
    let Some(tool) = tool.into() else {
        return "Hand: Empty    Clear / Pipette picks the object under the reticle".to_owned();
    };
    match tool {
        Tool::Connector => format!(
            "Tool: Connector    Aim at suspension to adjust; drag a block to a motor bearing    {}    Right click a wired bearing changes its default direction",
            selected_wires.map_or_else(
                || "No block selected".to_owned(),
                |wires| format!(
                    "Selected block: {wires} bearing{} wired",
                    if wires == 1 { "" } else { "s" }
                )
            ),
        ),
        Tool::Controller => format!(
            "Tool: {}    Rotate action: 90°    {}    Interact opens its program",
            tool.label(),
            selected_wires.map_or_else(
                || "No block selected — click one to select it".to_owned(),
                |wires| format!(
                    "Selected block: {wires} bearing{} wired",
                    if wires == 1 { "" } else { "s" }
                )
            ),
        ),
        Tool::Block => format!(
            "Tool: Blocker Placer    Material: {}    Block size: {BLOCK_SIZE_METERS:.2} m",
            material.label(),
        ),
        Tool::Bearing => format!(
            "Tool: Bearing    Outer: {:.2} m ←/→  Inner: {:.2} m Shift+←/→",
            bearing_dimensions.outer_diameter(),
            bearing_dimensions.inner_diameter(),
        ),
        Tool::Cylinder => format!(
            "Tool: Pipe / Cylinder    Material: {}    Outer: {:.2} m ←/→  Inner: {:.2} m Shift+←/→  Length: {:.2} m ↓/↑  Sweep: {}° Shift+↓/↑    Hold primary to drag; R mode; F bend; wheel radius",
            material.label(),
            cylinder_dimensions.outer_diameter(),
            cylinder_dimensions.inner_diameter(),
            cylinder_dimensions.axial_length(),
            cylinder_dimensions.sweep_angle_degrees(),
        ),
        Tool::GasEngine | Tool::ElectricEngine | Tool::Servo | Tool::Seat | Tool::Input => {
            format!("Tool: {}    Rotate action: 90°", tool.label())
        }
        _ => format!("Tool: {}", tool.label()),
    }
}

//! Construction prototype with a GPU-physics preview.

#![allow(clippy::needless_pass_by_value)] // Bevy system parameters are value-typed wrappers.

mod builder;
mod camera;

use bevy::{
    asset::RenderAssetUsages,
    camera::visibility::NoFrustumCulling,
    core_pipeline::tonemapping::Tonemapping,
    mesh::Indices,
    prelude::*,
    render::{
        render_resource::PrimitiveTopology,
        renderer::{RenderDevice, RenderQueue},
    },
};
use builder::{
    BEARING_DEPTH, BEARING_DIAMETER, BuildTool, GROUND_HALF_SIZE, PlacementCandidate,
    PlacementError, SizePreset, SurfaceHit, bearing_anchor_from_hit, bearing_attachment_candidate,
    begin_bearing, begin_weld, candidate_from_hit, face_geometry_from_ref, raycast_construction,
    stage_bearing_attachment, stage_cuboid, stage_weld_objects,
};
use camera::OrbitCamera;
use mechanic_core::{
    BuildCommand, CompiledCreation, ConstructionGraph, CuboidSpec, FaceOwner, PartId,
    PendingOperation,
};
use mechanic_gpu::{FixedStepScheduler, GpuPhysics, GpuTransform};

#[derive(Resource, Default)]
struct EditorGraph(ConstructionGraph);

#[derive(Resource, Default)]
struct AppSimulation {
    gpu: Option<GpuPhysics>,
    creation: Option<CompiledCreation>,
    scheduler: FixedStepScheduler,
    transforms: Vec<GpuTransform>,
    render_dirty: bool,
}

fn handle_simulation_shortcut(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
    mut simulation: ResMut<AppSimulation>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    if !keyboard.just_pressed(KeyCode::Space) {
        return;
    }
    if simulation.is_running() {
        *simulation = AppSimulation::default();
        state.construction_mesh_dirty = true;
        state.feedback = Some("Simulation stopped; returned to build mode".to_owned());
        return;
    }

    if graph.0.pending().is_some() {
        let _ = graph.0.apply(BuildCommand::CancelPending);
    }
    let creation = match graph.0.compile() {
        Ok(creation) => creation,
        Err(error) => {
            state.feedback = Some(format!("Cannot start simulation: {error}"));
            return;
        }
    };
    let gpu = match GpuPhysics::new(render_device.wgpu_device(), &render_queue, &creation) {
        Ok(gpu) => gpu,
        Err(error) => {
            state.feedback = Some(format!("Cannot start simulation: {error}"));
            return;
        }
    };
    let transforms = creation
        .compounds
        .iter()
        .map(|compound| {
            let position = compound.root_translation;
            let rotation = compound.root_rotation;
            GpuTransform {
                position: [position.x, position.y, position.z, 0.0],
                rotation: [rotation.x, rotation.y, rotation.z, rotation.w],
            }
        })
        .collect();
    *simulation = AppSimulation {
        gpu: Some(gpu),
        creation: Some(creation),
        scheduler: FixedStepScheduler::new(),
        transforms,
        render_dirty: true,
    };
    state.feedback = Some("Simulation running".to_owned());
}

#[allow(clippy::too_many_arguments)]
fn advance_simulation(
    time: Res<Time>,
    graph: Res<EditorGraph>,
    mut state: ResMut<EditorState>,
    mut simulation: ResMut<AppSimulation>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    visuals: Res<EditorVisuals>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut construction_visibility: Single<
        &mut Visibility,
        (With<ConstructionVisual>, Without<BearingVisual>),
    >,
    mut bearing_visibility: Single<
        &mut Visibility,
        (With<BearingVisual>, Without<ConstructionVisual>),
    >,
) {
    if !simulation.is_running() {
        return;
    }

    let ticks = simulation
        .scheduler
        .advance(time.delta())
        .into_iter()
        .collect::<Vec<_>>();
    if let Some(&last_tick) = ticks.last() {
        let readback = {
            let gpu = simulation
                .gpu
                .as_ref()
                .expect("running simulation has GPU state");
            for &tick in &ticks {
                gpu.dispatch_tick(render_device.wgpu_device(), &render_queue, tick);
            }
            gpu.read_last_tick(render_device.wgpu_device())
                .map_err(|error| error.to_string())
                .and_then(|diagnostics| {
                    if diagnostics.error_flags != 0 {
                        return Err(format!(
                            "physics kernel reported flags {}",
                            diagnostics.error_flags
                        ));
                    }
                    gpu.read_snapshot_transforms(
                        render_device.wgpu_device(),
                        &render_queue,
                        u8::try_from(last_tick % 3).unwrap_or(0),
                    )
                    .map_err(|error| error.to_string())
                })
        };
        match readback {
            Ok(transforms) => {
                simulation.transforms = transforms;
                simulation.render_dirty = true;
            }
            Err(error) => {
                *simulation = AppSimulation::default();
                state.construction_mesh_dirty = true;
                state.feedback = Some(format!("Simulation stopped: {error}"));
                return;
            }
        }
    }

    if !simulation.render_dirty {
        return;
    }
    let creation = simulation
        .creation
        .as_ref()
        .expect("running simulation has compiled creation");
    if let Some(mut mesh) = meshes.get_mut(&visuals.construction_mesh) {
        *mesh = combined_simulation_mesh(&graph.0, creation, &simulation.transforms);
    }
    **construction_visibility = Visibility::Visible;
    if creation.bearings.is_empty() {
        **bearing_visibility = Visibility::Hidden;
    } else {
        if let Some(mut mesh) = meshes.get_mut(&visuals.bearing_mesh) {
            *mesh = combined_simulation_bearing_mesh(creation, &simulation.transforms);
        }
        **bearing_visibility = Visibility::Visible;
    }
    simulation.render_dirty = false;
}

impl AppSimulation {
    const fn is_running(&self) -> bool {
        self.gpu.is_some()
    }
}

#[derive(Resource, Default)]
struct EditorState {
    tool: BuildTool,
    size: SizePreset,
    hovered: Option<SurfaceHit>,
    preview: Option<PlacementCandidate>,
    preview_error: Option<PlacementError>,
    feedback: Option<String>,
    construction_mesh_dirty: bool,
    delete_target: Option<PartId>,
}

#[derive(Resource)]
struct EditorVisuals {
    construction_mesh: Handle<Mesh>,
    bearing_mesh: Handle<Mesh>,
    cube_preview_mesh: Handle<Mesh>,
    bearing_preview_mesh: Handle<Mesh>,
    white_preview_material: Handle<StandardMaterial>,
    red_preview_material: Handle<StandardMaterial>,
}

#[derive(Component)]
struct ActionPreview;

#[derive(Component)]
struct SelectionPreview;

#[derive(Component)]
struct DeletePreview;

#[derive(Component)]
struct ConstructionVisual;

#[derive(Component)]
struct BearingVisual;

#[derive(Component)]
struct HelpText;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Mechanic — construction and simulation prototype".to_owned(),
                resolution: (1280, 720).into(),
                ..default()
            }),
            ..default()
        }))
        .init_resource::<EditorGraph>()
        .init_resource::<EditorState>()
        .init_resource::<AppSimulation>()
        .insert_resource(GlobalAmbientLight {
            color: Color::srgb(0.75, 0.80, 0.90),
            brightness: 350.0,
            ..default()
        })
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                camera::update_orbit_camera,
                handle_simulation_shortcut,
                handle_shortcuts,
                update_hover,
                handle_build_actions,
                sync_visual_meshes,
                advance_simulation,
                update_previews,
                update_help_text,
            )
                .chain(),
        )
        .run();
}

#[allow(clippy::too_many_lines)] // One-time Bevy scene composition is clearest in declaration order.
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let construction_mesh = meshes.add(Cuboid::default());
    let bearing_mesh = meshes.add(Cuboid::default());
    let cube_preview_mesh = meshes.add(Cuboid::default());
    let bearing_preview_mesh = meshes.add(Cylinder::new(BEARING_DIAMETER * 0.5, BEARING_DEPTH));
    let construction_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.18, 0.48, 0.78),
        perceptual_roughness: 0.8,
        ..default()
    });
    let bearing_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.58, 0.08),
        metallic: 0.35,
        perceptual_roughness: 0.55,
        ..default()
    });
    let white_preview_material = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 1.0, 1.0, 0.34),
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        unlit: true,
        ..default()
    });
    let red_preview_material = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 0.06, 0.04, 0.46),
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        unlit: true,
        ..default()
    });

    commands.insert_resource(EditorVisuals {
        construction_mesh: construction_mesh.clone(),
        bearing_mesh: bearing_mesh.clone(),
        cube_preview_mesh: cube_preview_mesh.clone(),
        bearing_preview_mesh,
        white_preview_material: white_preview_material.clone(),
        red_preview_material: red_preview_material.clone(),
    });

    commands.spawn((
        Name::new("Ground platform"),
        Mesh3d(
            meshes.add(
                Plane3d::default()
                    .mesh()
                    .size(GROUND_HALF_SIZE * 2.0, GROUND_HALF_SIZE * 2.0),
            ),
        ),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.16, 0.19, 0.22),
            perceptual_roughness: 1.0,
            ..default()
        })),
    ));
    commands.spawn((
        Name::new("Construction mesh"),
        Mesh3d(construction_mesh),
        MeshMaterial3d(construction_material),
        NoFrustumCulling,
        Visibility::Hidden,
        ConstructionVisual,
    ));
    commands.spawn((
        Name::new("Bearing mesh"),
        Mesh3d(bearing_mesh),
        MeshMaterial3d(bearing_material),
        NoFrustumCulling,
        Visibility::Hidden,
        BearingVisual,
    ));
    commands.spawn((
        Name::new("Action preview"),
        Mesh3d(cube_preview_mesh.clone()),
        MeshMaterial3d(white_preview_material.clone()),
        Transform::default(),
        Visibility::Hidden,
        ActionPreview,
    ));
    commands.spawn((
        Name::new("Selection preview"),
        Mesh3d(cube_preview_mesh.clone()),
        MeshMaterial3d(white_preview_material),
        Transform::default(),
        Visibility::Hidden,
        SelectionPreview,
    ));
    commands.spawn((
        Name::new("Delete preview"),
        Mesh3d(cube_preview_mesh),
        MeshMaterial3d(red_preview_material),
        Transform::default(),
        Visibility::Hidden,
        DeletePreview,
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 12_000.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::from_xyz(8.0, 14.0, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    let orbit = OrbitCamera::default();
    commands.spawn((
        Name::new("Orbital camera"),
        Camera3d::default(),
        Tonemapping::None,
        orbit.transform(),
        orbit,
    ));

    commands.spawn((
        HelpText,
        Text::new(""),
        TextFont {
            font_size: FontSize::Px(17.0),
            ..default()
        },
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            top: px(14),
            left: px(14),
            padding: UiRect::all(px(10)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.02, 0.025, 0.035, 0.82)),
    ));
}

fn handle_shortcuts(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
    simulation: Res<AppSimulation>,
) {
    if simulation.is_running() {
        return;
    }
    let selected_tool = if keyboard.just_pressed(KeyCode::Digit1) {
        Some(BuildTool::Cuboid)
    } else if keyboard.just_pressed(KeyCode::Digit2) {
        Some(BuildTool::Weld)
    } else if keyboard.just_pressed(KeyCode::Digit3) {
        Some(BuildTool::Bearing)
    } else {
        None
    };
    if let Some(tool) = selected_tool {
        if tool != state.tool || graph.0.pending().is_some() {
            let _ = graph.0.apply(BuildCommand::CancelPending);
        }
        state.tool = tool;
        state.feedback = None;
    }
    if keyboard.just_pressed(KeyCode::Escape) {
        if graph.0.pending().is_some() {
            let _ = graph.0.apply(BuildCommand::CancelPending);
            state.feedback = Some("Selection cancelled".to_owned());
        } else {
            state.tool = BuildTool::Cuboid;
            state.feedback = None;
        }
    }
    if keyboard.just_pressed(KeyCode::KeyQ) {
        state.size.smaller();
        state.feedback = None;
    }
    if keyboard.just_pressed(KeyCode::KeyE) {
        state.size.larger();
        state.feedback = None;
    }
}

fn update_hover(
    graph: Res<EditorGraph>,
    mut state: ResMut<EditorState>,
    window: Single<&Window>,
    camera: Single<(&Camera, &GlobalTransform), With<OrbitCamera>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    simulation: Res<AppSimulation>,
) {
    if simulation.is_running() {
        clear_hover(&mut state);
        return;
    }
    if camera::orbit_input_active(&mouse_buttons, &keyboard) {
        clear_hover(&mut state);
        refresh_tool_preview(&graph.0, &mut state);
        return;
    }
    let Some(cursor) = window.cursor_position() else {
        clear_hover(&mut state);
        refresh_tool_preview(&graph.0, &mut state);
        return;
    };
    let (camera, camera_transform) = *camera;
    let Ok(ray) = camera.viewport_to_world(camera_transform, cursor) else {
        clear_hover(&mut state);
        refresh_tool_preview(&graph.0, &mut state);
        return;
    };
    let Some(hit) = raycast_construction(&graph.0, ray.origin, ray.direction.as_vec3()) else {
        clear_hover(&mut state);
        refresh_tool_preview(&graph.0, &mut state);
        return;
    };
    state.hovered = Some(hit);
    refresh_tool_preview(&graph.0, &mut state);
}

fn clear_hover(state: &mut EditorState) {
    state.hovered = None;
    state.preview = None;
    state.preview_error = None;
}

fn refresh_tool_preview(graph: &ConstructionGraph, state: &mut EditorState) {
    state.preview = None;
    state.preview_error = match (state.tool, graph.pending()) {
        (BuildTool::Cuboid, _) => state.hovered.and_then(|hit| {
            let candidate = candidate_from_hit(graph, state.size, hit);
            let error = stage_cuboid(graph, candidate).err();
            state.preview = Some(candidate);
            error
        }),
        (BuildTool::Weld, Some(PendingOperation::Weld(first))) => state
            .hovered
            .and_then(|hit| stage_weld_objects(graph, first.owner, hit.face.owner).err()),
        (BuildTool::Weld, _) => None,
        (BuildTool::Bearing, Some(PendingOperation::Bearing { source, anchor })) => {
            let candidate = bearing_attachment_candidate(graph, state.size, source, anchor);
            let error = stage_bearing_attachment(graph, candidate, source, anchor).err();
            state.preview = Some(candidate);
            error
        }
        (BuildTool::Bearing, _) => state
            .hovered
            .and_then(|hit| bearing_anchor_from_hit(graph, hit).err()),
    };
}

fn handle_build_actions(
    mouse: Res<ButtonInput<MouseButton>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
    simulation: Res<AppSimulation>,
) {
    if simulation.is_running() {
        return;
    }
    if camera::orbit_input_active(&mouse, &keyboard) {
        return;
    }
    if mouse.just_pressed(MouseButton::Right) {
        state.delete_target = hovered_part(state.hovered);
        if state.delete_target.is_some() {
            state.feedback = Some("Release right mouse to delete".to_owned());
        }
    }
    if mouse.just_released(MouseButton::Right) {
        if let Some(part) = state.delete_target.take() {
            match graph.0.apply(BuildCommand::Remove(part)) {
                Ok(_) => {
                    state.feedback = Some("Deleted cuboid and incident connections".to_owned());
                    state.construction_mesh_dirty = true;
                    clear_hover(&mut state);
                }
                Err(error) => state.feedback = Some(error.to_string()),
            }
        }
        return;
    }
    if mouse.pressed(MouseButton::Right) || !mouse.just_pressed(MouseButton::Left) {
        return;
    }

    match state.tool {
        BuildTool::Cuboid => {
            let Some(candidate) = state.preview else {
                state.feedback = Some("Point at the platform or a cuboid face".to_owned());
                return;
            };
            match stage_cuboid(&graph.0, candidate) {
                Ok(staged) => {
                    graph.0 = staged;
                    state.feedback = Some("Placed cuboid".to_owned());
                    state.construction_mesh_dirty = true;
                }
                Err(error) => state.feedback = Some(error.to_string()),
            }
        }
        BuildTool::Weld => {
            let Some(hit) = state.hovered else {
                state.feedback = Some("Select an object".to_owned());
                return;
            };
            match graph.0.pending() {
                Some(PendingOperation::Weld(first)) => {
                    match stage_weld_objects(&graph.0, first.owner, hit.face.owner) {
                        Ok(staged) => {
                            graph.0 = staged;
                            state.feedback = Some("Welded the two objects".to_owned());
                        }
                        Err(error) => state.feedback = Some(error.to_string()),
                    }
                }
                _ => match begin_weld(&mut graph.0, hit.face) {
                    Ok(()) => {
                        state.feedback =
                            Some("First object selected; choose a touching object".to_owned());
                    }
                    Err(error) => state.feedback = Some(error.to_string()),
                },
            }
        }
        BuildTool::Bearing => {
            if let Some(PendingOperation::Bearing { source, anchor }) = graph.0.pending() {
                let Some(candidate) = state.preview else {
                    state.feedback = Some("The attached cuboid has no valid position".to_owned());
                    return;
                };
                match stage_bearing_attachment(&graph.0, candidate, source, anchor) {
                    Ok(staged) => {
                        graph.0 = staged;
                        state.feedback = Some("Attached cuboid through bearing".to_owned());
                        state.construction_mesh_dirty = true;
                    }
                    Err(error) => state.feedback = Some(error.to_string()),
                }
            } else {
                let Some(hit) = state.hovered else {
                    state.feedback = Some("Point at a cuboid face".to_owned());
                    return;
                };
                match bearing_anchor_from_hit(&graph.0, hit)
                    .and_then(|anchor| begin_bearing(&mut graph.0, hit.face, anchor))
                {
                    Ok(()) => {
                        state.feedback = Some(
                            "Bearing placed; click again to attach the selected cuboid".to_owned(),
                        );
                    }
                    Err(error) => state.feedback = Some(error.to_string()),
                }
            }
        }
    }
    refresh_tool_preview(&graph.0, &mut state);
}

fn hovered_part(hit: Option<SurfaceHit>) -> Option<PartId> {
    match hit?.face.owner {
        FaceOwner::Part(part) => Some(part),
        FaceOwner::Ground => None,
    }
}

fn sync_visual_meshes(
    graph: Res<EditorGraph>,
    mut state: ResMut<EditorState>,
    visuals: Res<EditorVisuals>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut construction_visibility: Single<
        &mut Visibility,
        (With<ConstructionVisual>, Without<BearingVisual>),
    >,
    mut bearing_visibility: Single<
        &mut Visibility,
        (With<BearingVisual>, Without<ConstructionVisual>),
    >,
) {
    if !state.construction_mesh_dirty {
        return;
    }
    if graph.0.part_count() == 0 {
        **construction_visibility = Visibility::Hidden;
    } else {
        if let Some(mut mesh) = meshes.get_mut(&visuals.construction_mesh) {
            *mesh = combined_construction_mesh(&graph.0);
        }
        **construction_visibility = Visibility::Visible;
    }
    if graph.0.bearing_count() == 0 {
        **bearing_visibility = Visibility::Hidden;
    } else {
        if let Some(mut mesh) = meshes.get_mut(&visuals.bearing_mesh) {
            *mesh = combined_bearing_mesh(&graph.0);
        }
        **bearing_visibility = Visibility::Visible;
    }
    state.construction_mesh_dirty = false;
}

#[allow(clippy::type_complexity)]
fn update_previews(
    graph: Res<EditorGraph>,
    state: Res<EditorState>,
    simulation: Res<AppSimulation>,
    visuals: Res<EditorVisuals>,
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
    hide_preview(&mut action.2);
    hide_preview(&mut selection.2);
    hide_preview(&mut delete.2);

    if simulation.is_running() {
        return;
    }

    if let Some(part) = state.delete_target
        && let Some(spec) = graph.0.part(part)
    {
        show_cuboid_preview(
            &mut delete,
            &visuals.cube_preview_mesh,
            &visuals.red_preview_material,
            *spec,
            1.025,
        );
        return;
    }

    let action_material = if state.preview_error.is_none() {
        &visuals.white_preview_material
    } else {
        &visuals.red_preview_material
    };
    match (state.tool, graph.0.pending()) {
        (BuildTool::Cuboid, _) => {
            if let Some(candidate) = state.preview {
                show_cuboid_preview(
                    &mut action,
                    &visuals.cube_preview_mesh,
                    action_material,
                    candidate.spec,
                    0.992,
                );
            }
        }
        (BuildTool::Weld, pending) => {
            if let Some(part) = hovered_part(state.hovered)
                && let Some(spec) = graph.0.part(part)
            {
                show_cuboid_preview(
                    &mut action,
                    &visuals.cube_preview_mesh,
                    action_material,
                    *spec,
                    1.018,
                );
            }
            if let Some(PendingOperation::Weld(first)) = pending
                && let FaceOwner::Part(part) = first.owner
                && let Some(spec) = graph.0.part(part)
            {
                show_cuboid_preview(
                    &mut selection,
                    &visuals.cube_preview_mesh,
                    &visuals.white_preview_material,
                    *spec,
                    1.028,
                );
            }
        }
        (BuildTool::Bearing, Some(PendingOperation::Bearing { source, anchor })) => {
            let normal = face_geometry_from_ref(source, Some(&graph.0)).normal;
            show_bearing_preview(
                &mut selection,
                &visuals.bearing_preview_mesh,
                &visuals.white_preview_material,
                anchor,
                normal,
            );
            if let Some(candidate) = state.preview {
                show_cuboid_preview(
                    &mut action,
                    &visuals.cube_preview_mesh,
                    action_material,
                    candidate.spec,
                    0.992,
                );
            }
        }
        (BuildTool::Bearing, _) => {
            if let Some(hit) = state.hovered {
                let face = face_geometry_from_ref(hit.face, Some(&graph.0));
                let anchor = bearing_anchor_from_hit(&graph.0, hit).unwrap_or(hit.point);
                show_bearing_preview(
                    &mut action,
                    &visuals.bearing_preview_mesh,
                    action_material,
                    anchor,
                    face.normal,
                );
            }
        }
    }
}

type PreviewItem<'a> = (
    Mut<'a, Mesh3d>,
    Mut<'a, Transform>,
    Mut<'a, Visibility>,
    Mut<'a, MeshMaterial3d<StandardMaterial>>,
);

fn hide_preview(visibility: &mut Visibility) {
    *visibility = Visibility::Hidden;
}

fn show_cuboid_preview(
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

fn show_bearing_preview(
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

fn update_help_text(
    graph: Res<EditorGraph>,
    state: Res<EditorState>,
    simulation: Res<AppSimulation>,
    mut text: Single<&mut Text, With<HelpText>>,
) {
    let status = state.preview_error.as_ref().map_or_else(
        || state.feedback.clone().unwrap_or_else(|| "Ready".to_owned()),
        ToString::to_string,
    );
    let tool_hint = if simulation.is_running() {
        "Space stops simulation and returns to build mode"
    } else {
        match (state.tool, graph.0.pending()) {
            (BuildTool::Cuboid, _) => "Left click places the white cuboid ghost",
            (BuildTool::Weld, None) => "Left click selects the first object",
            (BuildTool::Weld, Some(_)) => "Left click a touching second object",
            (BuildTool::Bearing, None) => "Left click places the 0.25 m bearing",
            (BuildTool::Bearing, Some(_)) => "Left click attaches the cuboid ghost",
        }
    };
    let mode = if simulation.is_running() {
        "SIMULATING"
    } else {
        "BUILDING"
    };
    text.0 = format!(
        "MECHANIC — {mode}\n\
         1 Cuboid   2 Weld   3 Bearing\n\
         Q/E Size   Space Start/Stop   Left click Action   Right click Delete\n\
         Option+Left drag Orbit   Wheel Zoom   Esc Cancel\n\n\
         Tool: {}    Size: {:.1} m\n\
         Parts: {}   Welds: {}   Bearings: {}\n\
         {tool_hint}\n\
         {status}",
        state.tool.label(),
        state.size.meters(),
        graph.0.part_count(),
        graph.0.weld_count(),
        graph.0.bearing_count(),
    );
}

const CUBE_POSITIONS: [[f32; 3]; 24] = [
    [-0.5, 0.5, -0.5],
    [0.5, 0.5, -0.5],
    [0.5, 0.5, 0.5],
    [-0.5, 0.5, 0.5],
    [-0.5, -0.5, -0.5],
    [0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5],
    [-0.5, -0.5, 0.5],
    [0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, 0.5],
    [0.5, 0.5, -0.5],
    [-0.5, -0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, 0.5, 0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [-0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5],
    [-0.5, -0.5, -0.5],
    [-0.5, 0.5, -0.5],
    [0.5, 0.5, -0.5],
    [0.5, -0.5, -0.5],
];
const CUBE_NORMALS: [[f32; 3]; 24] = [
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, -1.0],
    [0.0, 0.0, -1.0],
    [0.0, 0.0, -1.0],
    [0.0, 0.0, -1.0],
];
const CUBE_INDICES: [u32; 36] = [
    0, 3, 1, 1, 3, 2, 4, 5, 7, 5, 6, 7, 8, 11, 9, 9, 11, 10, 12, 13, 15, 13, 14, 15, 16, 19, 17,
    17, 19, 18, 20, 21, 23, 21, 22, 23,
];

fn combined_construction_mesh(graph: &ConstructionGraph) -> Mesh {
    let mut positions = Vec::with_capacity(graph.part_count() * CUBE_POSITIONS.len());
    let mut normals = Vec::with_capacity(graph.part_count() * CUBE_NORMALS.len());
    let mut indices = Vec::with_capacity(graph.part_count() * CUBE_INDICES.len());
    for (_, spec) in graph.parts() {
        append_cuboid(spec, &mut positions, &mut normals, &mut indices);
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(indices))
}

fn combined_simulation_mesh(
    graph: &ConstructionGraph,
    creation: &CompiledCreation,
    transforms: &[GpuTransform],
) -> Mesh {
    let mut positions = Vec::with_capacity(creation.colliders.len() * CUBE_POSITIONS.len());
    let mut normals = Vec::with_capacity(creation.colliders.len() * CUBE_NORMALS.len());
    let mut indices = Vec::with_capacity(creation.colliders.len() * CUBE_INDICES.len());
    for collider in &creation.colliders {
        let transform = transforms[collider.compound_index as usize];
        let root_translation = Vec3::from_array(transform.position[..3].try_into().unwrap());
        let root_rotation = Quat::from_array(transform.rotation);
        let translation = root_translation + root_rotation * collider.local_center;
        let rotation = root_rotation * collider.local_rotation;
        let size = graph
            .part(collider.source_part)
            .expect("compiled collider source remains in graph")
            .size_meters();
        append_transformed_cuboid(
            translation,
            rotation,
            size,
            &mut positions,
            &mut normals,
            &mut indices,
        );
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(indices))
}

fn combined_bearing_mesh(graph: &ConstructionGraph) -> Mesh {
    const SEGMENTS: usize = 24;
    let vertices_per_bearing = SEGMENTS * 4 + 2;
    let indices_per_bearing = SEGMENTS * 12;
    let mut positions = Vec::with_capacity(graph.bearing_count() * vertices_per_bearing);
    let mut normals = Vec::with_capacity(graph.bearing_count() * vertices_per_bearing);
    let mut indices = Vec::with_capacity(graph.bearing_count() * indices_per_bearing);
    for (_, bearing) in graph.bearings() {
        append_bearing_cylinder(
            bearing.shared_anchor,
            bearing.axis,
            &mut positions,
            &mut normals,
            &mut indices,
        );
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(indices))
}

fn combined_simulation_bearing_mesh(
    creation: &CompiledCreation,
    transforms: &[GpuTransform],
) -> Mesh {
    const SEGMENTS: usize = 24;
    let vertices_per_bearing = SEGMENTS * 4 + 2;
    let indices_per_bearing = SEGMENTS * 12;
    let mut positions = Vec::with_capacity(creation.bearings.len() * vertices_per_bearing);
    let mut normals = Vec::with_capacity(creation.bearings.len() * vertices_per_bearing);
    let mut indices = Vec::with_capacity(creation.bearings.len() * indices_per_bearing);
    for bearing in &creation.bearings {
        let transform = transforms[bearing.compound_a as usize];
        let root_translation = Vec3::from_array(transform.position[..3].try_into().unwrap());
        let root_rotation = Quat::from_array(transform.rotation);
        append_bearing_cylinder(
            root_translation + root_rotation * bearing.local_anchor_a,
            root_rotation * bearing.local_axis_a,
            &mut positions,
            &mut normals,
            &mut indices,
        );
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_indices(Indices::U32(indices))
}

fn append_bearing_cylinder(
    anchor: Vec3,
    axis: Vec3,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    const SEGMENTS: u16 = 24;
    let axis = axis.normalize();
    let tangent_u = if axis.y.abs() < 0.9 {
        axis.cross(Vec3::Y).normalize()
    } else {
        axis.cross(Vec3::X).normalize()
    };
    let tangent_v = axis.cross(tangent_u);
    let radius = BEARING_DIAMETER * 0.5;
    let half_depth = BEARING_DEPTH * 0.5;
    let lower = anchor - axis * half_depth;
    let upper = anchor + axis * half_depth;
    let base = u32::try_from(positions.len()).expect("prototype mesh fits 32-bit indices");

    for segment in 0..SEGMENTS {
        let angle = std::f32::consts::TAU * f32::from(segment) / f32::from(SEGMENTS);
        let radial = tangent_u * angle.cos() + tangent_v * angle.sin();
        positions.push((lower + radial * radius).to_array());
        positions.push((upper + radial * radius).to_array());
        normals.push(radial.to_array());
        normals.push(radial.to_array());
    }
    for segment in 0..SEGMENTS {
        let next = (segment + 1) % SEGMENTS;
        let lower_current = base + u32::from(segment) * 2;
        let upper_current = lower_current + 1;
        let lower_next = base + u32::from(next) * 2;
        let upper_next = lower_next + 1;
        indices.extend([
            lower_current,
            lower_next,
            upper_current,
            upper_current,
            lower_next,
            upper_next,
        ]);
    }

    let lower_center = u32::try_from(positions.len()).unwrap();
    positions.push(lower.to_array());
    normals.push((-axis).to_array());
    let upper_center = u32::try_from(positions.len()).unwrap();
    positions.push(upper.to_array());
    normals.push(axis.to_array());
    let lower_ring = u32::try_from(positions.len()).unwrap();
    for segment in 0..SEGMENTS {
        let angle = std::f32::consts::TAU * f32::from(segment) / f32::from(SEGMENTS);
        let radial = tangent_u * angle.cos() + tangent_v * angle.sin();
        positions.push((lower + radial * radius).to_array());
        normals.push((-axis).to_array());
    }
    let upper_ring = u32::try_from(positions.len()).unwrap();
    for segment in 0..SEGMENTS {
        let angle = std::f32::consts::TAU * f32::from(segment) / f32::from(SEGMENTS);
        let radial = tangent_u * angle.cos() + tangent_v * angle.sin();
        positions.push((upper + radial * radius).to_array());
        normals.push(axis.to_array());
    }
    for segment in 0..SEGMENTS {
        let next = (segment + 1) % SEGMENTS;
        let current = u32::from(segment);
        let next = u32::from(next);
        indices.extend([
            lower_center,
            lower_ring + next,
            lower_ring + current,
            upper_center,
            upper_ring + current,
            upper_ring + next,
        ]);
    }
}

fn append_cuboid(
    spec: &CuboidSpec,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let size = spec.size_meters();
    let rotation = spec.pose.rotation.quaternion();
    let translation = spec.pose.translation();
    append_transformed_cuboid(translation, rotation, size, positions, normals, indices);
}

fn append_transformed_cuboid(
    translation: Vec3,
    rotation: Quat,
    size: Vec3,
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
) {
    let base_index = u32::try_from(positions.len()).expect("prototype mesh fits 32-bit indices");
    positions.extend(
        CUBE_POSITIONS.map(|position| {
            (translation + rotation * (Vec3::from_array(position) * size)).to_array()
        }),
    );
    normals.extend(CUBE_NORMALS.map(|normal| (rotation * Vec3::from_array(normal)).to_array()));
    indices.extend(CUBE_INDICES.map(|index| base_index + index));
}

#[cfg(test)]
mod rendering_tests {
    use bevy::prelude::Vec3;

    use super::{BEARING_DEPTH, BEARING_DIAMETER, append_bearing_cylinder};

    #[test]
    fn bearing_visual_is_quarter_metre_wide_and_five_centimetres_each_side() {
        let anchor = Vec3::new(2.0, 3.0, 4.0);
        let axis = Vec3::X;
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut indices = Vec::new();
        append_bearing_cylinder(anchor, axis, &mut positions, &mut normals, &mut indices);

        let offsets = positions
            .iter()
            .map(|position| Vec3::from_array(*position) - anchor)
            .collect::<Vec<_>>();
        let minimum_depth = offsets
            .iter()
            .map(|offset| offset.dot(axis))
            .fold(f32::INFINITY, f32::min);
        let maximum_depth = offsets
            .iter()
            .map(|offset| offset.dot(axis))
            .fold(f32::NEG_INFINITY, f32::max);
        let maximum_radius = offsets
            .iter()
            .map(|offset| (*offset - axis * offset.dot(axis)).length())
            .fold(0.0, f32::max);

        assert!((minimum_depth + BEARING_DEPTH * 0.5).abs() < 1.0e-6);
        assert!((maximum_depth - BEARING_DEPTH * 0.5).abs() < 1.0e-6);
        assert!((maximum_radius - BEARING_DIAMETER * 0.5).abs() < 1.0e-6);
    }
}

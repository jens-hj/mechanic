//! Construction prototype with a GPU-physics preview.

#![expect(
    clippy::needless_pass_by_value,
    reason = "bevy system parameters are value-typed wrappers"
)]

use std::collections::{HashMap, HashSet};

mod automation;
mod avatar;
mod builder;
mod camera;
mod chroma;
mod control_panel;
mod controls;
mod cpu_physics;
mod creation_menu;
mod creation_store;
mod debug_freeze;
mod editor;
mod frame_visuals;
mod freeze;
mod freeze_motion;
mod garage;
mod hotbar;
mod linear_editor;
mod linear_render;
mod live_edit;
mod live_weld;
mod multitool;
mod pause_menu;
mod performance;
mod performance_capture;
mod pose;
mod render;
mod render_diagnostics;
mod render_experiments;
mod scheduler;
mod seat;
mod sequencer;
mod settings;
mod shape_tool;
mod showcase;
mod simulation;
mod suspension_capture;
mod suspension_controls;
mod suspension_editor;
mod suspension_render;
mod terrain_publication;
mod tool_fx;
mod ui;
mod weld_publication;
mod weld_tool;
mod world;

use avatar::{spawn_player_avatar, sync_player_avatar};
use bevy::{
    asset::RenderAssetUsages,
    camera::visibility::{NoFrustumCulling, RenderLayers},
    core_pipeline::tonemapping::Tonemapping,
    diagnostic::FrameTimeDiagnosticsPlugin,
    image::ImageLoaderSettings,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
    tasks::futures::check_ready,
    window::{CursorGrabMode, CursorOptions, PrimaryWindow},
};
#[cfg(test)]
use builder::candidate_from_hit;
use builder::{
    BlockVolume, PlacementBounds, PlacementCandidate, PlacementError, PlacementGrid,
    PlacementPlane, PlacementSupport, SmartGuide, SurfaceHit, bearing_anchor_from_hit_with_grid,
    bearing_attachment_candidate, bearing_overlaps_candidate, bearing_overlaps_cylinder_candidate,
    bearing_support_face, block_box_bounds, block_box_specs, block_span_from_rays,
    candidate_from_hit_with_grid_and_supports, center_cylinder_candidate_on_bearing,
    cylinder_candidate_from_hit_with_grid, face_geometry_from_ref, free_cuboid_candidate,
    free_cylinder_candidate, oriented_cuboid_candidate_from_hit_with_grid, raycast_construction,
    raycast_construction_for_annulus, raycast_construction_for_annulus_with_ground,
    raycast_placement_plane_point, smart_snap_anchor, smart_snap_block_span,
    smart_snap_cuboid_candidate, smart_snap_cuboid_candidate_with_supports,
    smart_snap_cylinder_candidate, smart_snap_free_cuboid_candidate,
    smart_snap_free_cylinder_candidate, stage_bearing_attachment_in_bounds, stage_weld_objects,
    transmission_candidate_from_hit_in_bounds, try_face_geometry_from_ref,
    validate_block_batch_in_bounds, validate_block_volume_in_bounds,
    validate_cylinder_candidate_in_bounds, validate_indexed_block_batch_in_bounds,
};
use camera::{FovCamera, apply_camera_fov};
use camera::{
    MainCamera, MaterialWheelState, PlayerCamera, PlayerState, SEATED_EYE_HEIGHT,
    seated_view_rotation,
};
use chroma::{ChromaBrush, ConstructionRenderMaterial};
use control_panel::ControlPanelState;
use controls::GameAction;
use creation_menu::CreationMenuState;
use creation_store::CreationStore;
use debug_freeze::{DebugFrameFreeze, debug_frame_updates_enabled, update_debug_frame_freeze};
use editor::{
    build_actions::{
        PlacedBearing, bearing_socket_targets, bearing_uses_socket,
        editor_part_is_static_or_pending, handle_build_actions,
    },
    creation::{handle_creation_menu_shortcut, handle_creation_request},
    dimensions::{
        BearingToolSettings, CylinderToolSettings, handle_bearing_dimension_shortcuts,
        handle_cylinder_dimension_shortcuts, handle_dimension_link_interaction,
    },
    hover::{DRAG_DEAD_ZONE_RADIANS, PointerSample, clear_hover, handle_tool_change, update_hover},
    pipe::{
        PipeEditMode, invalidate_pipe_drag, pipe_pointer_delta, rebuild_pipe_drag,
        refresh_pipe_drag,
    },
    placement::{
        PlacementLatticeVisual, SmartGuideVisual, SmartSnapRangeVisual, active_placement_grid,
        free_placement_point_on_miss, rebuild_placement_snap_index, update_free_placement_settings,
        update_smart_snap_settings,
    },
    preview::{
        ActionPreview, BearingVisual, ConstructionVisual, DeletePreview, DriveXrayVisual,
        EditorVisuals, FeaturePreviewKey, JointXrayVisual, SelectionPreview, sync_visual_meshes,
        update_joint_xray, update_previews,
    },
    shape_actions::{
        LayerPreview, SHAPE_SELECTION_COLOR, ShapeArrowVisual, ShapeNodeVisual, ShapePlaneVisual,
        ShapeSelectedVisual, handle_shape_actions, sync_drag_plane, sync_region_focus,
        sync_shape_nodes,
    },
    shortcuts::{handle_control_panel_shortcut, handle_shortcuts},
    state::{CurrentCreation, EditorGraph, EditorState},
};
use editor::{
    hammer::{HammerInteraction, handle_hammer_actions},
    history::{
        EditorHistory, EditorSnapshot, cancel_transient_editor_state, handle_history_shortcut,
    },
    overlay::{
        OverlayGeometry, append_axis_arrows, append_dashed_overlay_bar, append_drag_plane,
        append_overlay_bar, append_plane_arrows, append_region_outline, region_world_bounds,
        sync_edit_overlay_transforms, sync_placement_overlays, write_overlay,
    },
    raycast::{
        hovered_part, raycast_live_placed_bearing_discs, raycast_live_placed_bearings,
        raycast_placed_bearings_with_pose, raycast_simulation,
    },
    wiring::{
        WireDragVisual, WireHoverVisual, update_wire_drag_preview, update_wire_hover_preview,
    },
};
use hotbar::{SelectedMaterial, SelectedTerrainMaterial, SelectedTool, Tool};
use mechanic_core::{
    BearingDimensions, BearingSocket, BuildCommand, BuildOutcome, CageIndex, CompiledCreation,
    ConstructionGraph, ConstructionMaterial, ControllerSpec, CuboidSpec, CylinderDimensions,
    DimensionLinkSpec, EngineKind, FaceOwner, FaceRef, GridRotation, InputSpec, MaterialAppearance,
    POSITION_TICK_METERS, POSITION_TICKS_PER_GRID_UNIT, PartId, PartSpec, PendingOperation,
    RegionId, SeatSpec, ServoSpec, ShapeRegion, TransmissionSpec, part_cells,
};
use mechanic_gpu::{GpuPhysicsConfig, GpuTransform};
use pause_menu::PauseMenuState;
use pause_menu::{
    begin_pause_frame, capture_control_binding, handle_pause_escape, handle_pause_request,
};
use performance::PerformanceMetrics;
use render::authored::{
    AuthoredPart, AuthoredPartVisual, CONTROLLER_SURFACE_COLOR, authored_orientation,
};
use render::mesh::{
    bearing::single_bearing_mesh,
    construction::{
        append_transformed_cuboid, ordinary_material, preview_region, single_authored_part_mesh,
        single_cylinder_mesh,
    },
    drive::wire_drag_preview_mesh,
    primitives::{append_mesh_quad, append_mesh_triangle, degenerate_overlay_mesh},
};
use render::{
    environment::{
        OneShotEnvironmentMapPlugin, SKY_CUBEMAP_SIZE, SKY_ENVIRONMENT_INTENSITY,
        StreamingMeshAllocatorPlugin, sky_cubemap,
    },
    materials::{
        BearingTextureMipsPending, PREVIEW_RENDER_DEPTH_BIAS, authored_part_material,
        authored_preview_material, bearing_surface_material, configure_repeating_texture,
        construction_material, construction_tint_mask_path, material_index,
        prepare_bearing_texture_mips, preview_material,
    },
};
use seat::handle_seat_interaction;
use sequencer::run_drive_sequencer;
use sequencer::{DriveSequencer, GearboxRuntime};
use settings::AppSettings;
use simulation::{
    publication::{WorldPhysicsPublication, maintain_space_simulation},
    state::AppSimulation,
    tick::{advance_simulation, poll_simulation_readbacks},
    visuals::{SimulationVisualCache, sync_simulation_visual_cache},
};

#[expect(
    clippy::too_many_lines,
    reason = "the app schedule is kept in visible execution order"
)]
fn main() {
    // Validate before starting the renderer; diagnostic modes are never persisted.
    let render_experiment = render_experiments::current();
    if render_experiment != render_experiments::RenderExperiment::Baseline {
        eprintln!("Rendering diagnostic: {}", render_experiment.label());
    }
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(bevy::winit::WinitPlugin {
                    prevent_activation: automation::background() || tool_fx::capture_active(),
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Mechanic — construction and simulation prototype".to_owned(),
                        resolution: if automation::enabled() {
                            bevy::window::WindowResolution::new(4112, 2524)
                                .with_scale_factor_override(2.0)
                        } else {
                            (1280, 720).into()
                        },
                        focused: !automation::background() && !tool_fx::capture_active(),
                        ..default()
                    }),
                    ..Default::default()
                }),
        )
        .add_plugins((
            FrameTimeDiagnosticsPlugin::new(120),
            bevy::render::diagnostic::RenderDiagnosticsPlugin,
            render_diagnostics::RenderTimingsPlugin,
        ))
        .add_plugins(StreamingMeshAllocatorPlugin)
        .add_plugins(OneShotEnvironmentMapPlugin)
        .add_plugins(MaterialPlugin::<world::TerrainRenderMaterial>::default())
        .add_plugins(MaterialPlugin::<ConstructionRenderMaterial>::default())
        // After DefaultPlugins: the overlay's render pass installs into the
        // render sub-app, which does not exist until RenderPlugin has run.
        .add_plugins(bevy_mosaic::MosaicPlugin)
        .init_resource::<EditorGraph>()
        .init_resource::<EditorState>()
        .init_resource::<EditorHistory>()
        .init_resource::<CreationMenuState>()
        .init_resource::<CreationStore>()
        .init_resource::<CurrentCreation>()
        .init_resource::<DebugFrameFreeze>()
        .init_resource::<PauseMenuState>()
        .init_resource::<PerformanceMetrics>()
        .init_resource::<performance_capture::Recorder>()
        .add_plugins(automation::AutomationPlugin)
        .add_plugins(suspension_capture::SuspensionCapturePlugin)
        .init_resource::<AppSettings>()
        .init_resource::<AppSimulation>()
        .init_resource::<SimulationVisualCache>()
        .init_resource::<WorldPhysicsPublication>()
        .init_resource::<freeze::DimensionFreeze>()
        .init_resource::<HammerInteraction>()
        .init_resource::<BearingToolSettings>()
        .init_resource::<ControlPanelState>()
        .init_resource::<DriveSequencer>()
        .init_resource::<GearboxRuntime>()
        .init_resource::<CylinderToolSettings>()
        .init_resource::<shape_tool::ShapeMirror>()
        .init_resource::<shape_tool::ShapeSnap>()
        .init_resource::<shape_tool::ShapeEditMode>()
        .init_resource::<SelectedTool>()
        .init_resource::<SelectedMaterial>()
        .init_resource::<ChromaBrush>()
        .init_resource::<SelectedTerrainMaterial>()
        .init_resource::<PlayerState>()
        .init_resource::<MaterialWheelState>()
        .init_resource::<ButtonInput<GameAction>>()
        .add_plugins(world::WorldPrototypePlugin)
        .add_plugins(multitool::MultitoolPlugin)
        .add_plugins(tool_fx::ToolFxPlugin)
        // A dim base fill keeps occluded construction readable without
        // overpowering the garage's authored lighting.
        .insert_resource(GlobalAmbientLight {
            color: Color::srgb_u8(43, 60, 76),
            brightness: 20.0,
            ..Default::default()
        })
        .insert_resource(ClearColor(garage::VOID_COLOR))
        .add_systems(Startup, (setup, ui::mount).chain())
        .add_systems(
            Update,
            (
                update_debug_frame_freeze,
                prepare_bearing_texture_mips,
                performance::toggle,
                (
                    (
                        (
                            begin_pause_frame,
                            controls::update_action_state,
                            update_smart_snap_settings,
                            update_free_placement_settings,
                            capture_control_binding,
                            handle_creation_menu_shortcut,
                            handle_dimension_link_interaction,
                            handle_control_panel_shortcut,
                            ui::drain,
                            handle_pause_request,
                            handle_pause_escape,
                        )
                            .chain(),
                        (
                            ui::push,
                            ui::push_help,
                            ui::push_markers,
                            ui::push_player,
                            ui::sync_input,
                            handle_history_shortcut,
                            handle_creation_request,
                        )
                            .chain(),
                    )
                        .chain(),
                    (
                        apply_camera_fov,
                        camera::update_material_wheel,
                        camera::update_player_camera,
                        poll_simulation_readbacks.run_if(world::world_playing),
                        freeze::update.run_if(world::world_playing),
                        live_edit::refresh_context,
                        handle_seat_interaction,
                        handle_shortcuts,
                    )
                        .chain(),
                    (
                        (
                            handle_bearing_dimension_shortcuts,
                            linear_editor::controls,
                            handle_cylinder_dimension_shortcuts,
                        )
                            .chain(),
                        (
                            handle_tool_change,
                            rebuild_placement_snap_index,
                            update_hover,
                            tool_fx::capture_gesture,
                            ui::push_suspension,
                            handle_build_actions,
                            tool_fx::finish_gesture,
                            handle_shape_actions,
                            ui::push_dimensions,
                            handle_hammer_actions,
                        )
                            .chain(),
                        update_joint_xray,
                        sync_placement_overlays,
                        sync_visual_meshes,
                        (
                            sync_shape_nodes,
                            sync_region_focus,
                            sync_drag_plane,
                            sync_edit_overlay_transforms,
                        )
                            .chain(),
                        update_wire_drag_preview,
                        update_wire_hover_preview,
                        maintain_space_simulation.after(world::sync_world_foundations),
                        sync_simulation_visual_cache,
                        linear_render::sync_linear_bearing_visuals,
                        suspension_render::sync_suspension_visuals,
                        run_drive_sequencer,
                        advance_simulation.run_if(world::world_playing),
                        sync_player_avatar,
                        (update_previews, live_edit::place_previews).chain(),
                        (
                            performance::sample,
                            performance_capture::sample,
                            ui::push_performance,
                            ui::push_driving,
                        )
                            .chain(),
                    )
                        .chain(),
                )
                    .chain()
                    .run_if(debug_frame_updates_enabled),
            )
                .chain(),
        )
        .add_systems(Update, weld_tool::draw_features)
        .run();
}

#[expect(
    clippy::too_many_lines,
    reason = "one-time Bevy scene composition is clearest in declaration order"
)]
fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    settings: Res<AppSettings>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut construction_render_materials: ResMut<Assets<ConstructionRenderMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    let construction_meshes = ConstructionMaterial::ALL.map(|_| meshes.add(Cuboid::default()));
    let bearing_mesh = meshes.add(Cuboid::default());
    let joint_xray_mesh = meshes.add(Cuboid::default());
    let shape_node_mesh = meshes.add(degenerate_overlay_mesh());
    let shape_selected_mesh = meshes.add(degenerate_overlay_mesh());
    let shape_plane_mesh = meshes.add(degenerate_overlay_mesh());
    let shape_arrow_mesh = meshes.add(degenerate_overlay_mesh());
    let placement_lattice_mesh = meshes.add(degenerate_overlay_mesh());
    let smart_guide_mesh = meshes.add(degenerate_overlay_mesh());
    let smart_snap_range_mesh = meshes.add(degenerate_overlay_mesh());
    let controller_mesh = meshes.add(Cuboid::default());
    let gas_engine_mesh = meshes.add(Cuboid::default());
    let electric_engine_mesh = meshes.add(Cuboid::default());
    let gas_transmission_mesh = meshes.add(Cuboid::default());
    let electric_transmission_mesh = meshes.add(Cuboid::default());
    let servo_mesh = meshes.add(Cuboid::default());
    let seat_mesh = meshes.add(Cuboid::default());
    let input_mesh = meshes.add(Cuboid::default());
    let dimension_link_disabled_mesh = meshes.add(Cuboid::default());
    let dimension_link_enabled_mesh = meshes.add(Cuboid::default());
    let authored_preview_meshes =
        AuthoredPart::ALL.map(|appearance| meshes.add(single_authored_part_mesh(appearance)));
    let drive_xray_mesh = meshes.add(Cuboid::default());
    let wire_drag_mesh = meshes.add(wire_drag_preview_mesh(Vec3::ZERO, Vec3::ZERO));
    let wire_hover_mesh = meshes.add(degenerate_overlay_mesh());
    let cube_preview_mesh = meshes.add(Cuboid::default());
    let cylinder_preview_mesh = meshes.add(single_cylinder_mesh(CylinderDimensions::default()));
    let bearing_preview_mesh = meshes.add(single_bearing_mesh(BearingDimensions::default()));
    let block_drag_preview_mesh = meshes.add(Cuboid::default());
    let delete_drag_preview_mesh = meshes.add(Cuboid::default());
    let weld_hover_preview_mesh = meshes.add(Cuboid::default());
    let weld_selection_preview_mesh = meshes.add(Cuboid::default());
    let white_tint_mask = images.add(Image::new_fill(
        Extent3d::default(),
        TextureDimension::D2,
        &[255, 255, 255, 255],
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    ));
    let tint_mask = |material| match construction_tint_mask_path(material) {
        Some(path) => asset_server
            .load_builder()
            .with_settings(|settings: &mut ImageLoaderSettings| {
                configure_repeating_texture(settings, false);
            })
            .load(path),
        None => white_tint_mask.clone(),
    };
    let construction_materials = ConstructionMaterial::ALL.map(|material| {
        construction_render_materials.add(construction_material(
            &asset_server,
            material,
            tint_mask(material),
        ))
    });
    // Faded copies, swapped in while a region is being edited so the area under
    // the cursor is the only thing that reads as solid.
    let ghost_materials = ConstructionMaterial::ALL.map(|material| {
        let mut ghost = construction_material(&asset_server, material, tint_mask(material));
        ghost.base.base_color = ghost.base.base_color.with_alpha(0.16);
        ghost.base.alpha_mode = AlphaMode::Blend;
        construction_render_materials.add(ghost)
    });
    let bearing_material = bearing_surface_material(&asset_server);
    commands.insert_resource(BearingTextureMipsPending(vec![
        bearing_material
            .base_color_texture
            .clone()
            .expect("the bearing has a base-color map"),
        bearing_material
            .normal_map_texture
            .clone()
            .expect("the bearing has a normal map"),
        bearing_material
            .metallic_roughness_texture
            .clone()
            .expect("the bearing has an ORM map"),
    ]));
    let bearing_material = materials.add(bearing_material);
    let authored_materials = [
        authored_part_material(&asset_server, "machines/controller/controller"),
        authored_part_material(&asset_server, "machines/gas_engine/gas_engine"),
        authored_part_material(&asset_server, "machines/electric_engine/electric_engine"),
        authored_part_material(&asset_server, "machines/transmission_gas/transmission_gas"),
        authored_part_material(
            &asset_server,
            "machines/transmission_electric/transmission_electric",
        ),
        authored_part_material(&asset_server, "machines/servo/servo"),
        authored_part_material(&asset_server, "machines/seat/seat"),
        authored_part_material(&asset_server, "machines/input/input"),
        authored_part_material(
            &asset_server,
            "machines/dimension_link/disabled/dimension_link",
        ),
        authored_part_material(
            &asset_server,
            "machines/dimension_link/enabled/dimension_link",
        ),
    ];
    let authored_preview_materials = std::array::from_fn(|index| {
        materials.add(authored_preview_material(
            authored_materials[index].clone(),
            Color::srgba(1.0, 1.0, 1.0, 0.46),
        ))
    });
    let invalid_authored_preview_materials = std::array::from_fn(|index| {
        materials.add(authored_preview_material(
            authored_materials[index].clone(),
            Color::srgba(1.0, 0.18, 0.16, 0.52),
        ))
    });
    let authored_materials = authored_materials.map(|material| materials.add(material));
    let drive_xray_material = materials.add(StandardMaterial {
        base_color: CONTROLLER_SURFACE_COLOR,
        cull_mode: None,
        unlit: true,
        ..default()
    });
    let wire_drag_material = drive_xray_material.clone();
    let wire_hover_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.86, 0.99, 1.0),
        cull_mode: None,
        unlit: true,
        ..default()
    });
    let joint_xray_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.58, 0.08),
        cull_mode: None,
        unlit: true,
        ..default()
    });
    let white_preview_material = materials.add(preview_material(Color::srgba(1.0, 1.0, 1.0, 0.34)));
    let chroma_preview_material =
        materials.add(preview_material(Color::srgba(1.0, 1.0, 1.0, 0.46)));
    let red_preview_material = materials.add(preview_material(Color::srgba(1.0, 0.06, 0.04, 0.46)));
    let amber_preview_material =
        materials.add(preview_material(Color::srgba(1.0, 0.60, 0.06, 0.46)));
    let green_preview_material =
        materials.add(preview_material(Color::srgba(0.12, 1.0, 0.28, 0.52)));
    let placement_lattice_material = materials.add(StandardMaterial {
        base_color: SHAPE_SELECTION_COLOR.with_alpha(0.24),
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        unlit: true,
        depth_bias: PREVIEW_RENDER_DEPTH_BIAS,
        ..default()
    });
    let smart_guide_material = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.86, 0.18),
        cull_mode: None,
        unlit: true,
        depth_bias: PREVIEW_RENDER_DEPTH_BIAS,
        ..default()
    });
    let smart_snap_range_material = materials.add(StandardMaterial {
        base_color: Color::srgba(1.0, 0.86, 0.18, 0.34),
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        unlit: true,
        depth_bias: PREVIEW_RENDER_DEPTH_BIAS,
        ..default()
    });

    spawn_player_avatar(&mut commands, &mut meshes, &mut materials);

    commands.insert_resource(EditorVisuals {
        construction_meshes: construction_meshes.clone(),
        construction_materials: construction_materials.clone(),
        ghost_materials: ghost_materials.clone(),
        authored_materials: authored_materials.clone(),
        bearing_material: bearing_material.clone(),
        bearing_mesh: bearing_mesh.clone(),
        joint_xray_mesh: joint_xray_mesh.clone(),
        shape_node_mesh: shape_node_mesh.clone(),
        shape_selected_mesh: shape_selected_mesh.clone(),
        shape_plane_mesh: shape_plane_mesh.clone(),
        shape_arrow_mesh: shape_arrow_mesh.clone(),
        controller_mesh: controller_mesh.clone(),
        gas_engine_mesh: gas_engine_mesh.clone(),
        electric_engine_mesh: electric_engine_mesh.clone(),
        gas_transmission_mesh: gas_transmission_mesh.clone(),
        electric_transmission_mesh: electric_transmission_mesh.clone(),
        servo_mesh: servo_mesh.clone(),
        seat_mesh: seat_mesh.clone(),
        input_mesh: input_mesh.clone(),
        dimension_link_disabled_mesh: dimension_link_disabled_mesh.clone(),
        dimension_link_enabled_mesh: dimension_link_enabled_mesh.clone(),
        authored_preview_meshes,
        authored_preview_materials,
        invalid_authored_preview_materials,
        drive_xray_mesh: drive_xray_mesh.clone(),
        wire_drag_mesh: wire_drag_mesh.clone(),
        wire_hover_mesh: wire_hover_mesh.clone(),
        cube_preview_mesh: cube_preview_mesh.clone(),
        cylinder_preview_mesh,
        bearing_preview_mesh,
        white_preview_material: white_preview_material.clone(),
        chroma_preview_material,
        green_preview_material,
        red_preview_material: red_preview_material.clone(),
        amber_preview_material,
        block_drag_preview_mesh,
        delete_drag_preview_mesh,
        weld_hover_preview_mesh,
        weld_selection_preview_mesh,
    });

    garage::spawn(&mut commands, &asset_server, &mut meshes, &mut materials);
    for material in ConstructionMaterial::ALL {
        let index = material_index(material);
        commands.spawn((
            Name::new(format!("{} construction mesh", material.label())),
            Mesh3d(construction_meshes[index].clone()),
            MeshMaterial3d(construction_materials[index].clone()),
            NoFrustumCulling,
            Visibility::Hidden,
            ConstructionVisual(material),
        ));
    }
    commands.spawn((
        Name::new("Bearing mesh"),
        Mesh3d(bearing_mesh.clone()),
        MeshMaterial3d(bearing_material),
        NoFrustumCulling,
        Visibility::Hidden,
        BearingVisual,
    ));
    commands.spawn((
        Name::new("Control block mesh"),
        Mesh3d(controller_mesh),
        MeshMaterial3d(authored_materials[AuthoredPart::Controller.index()].clone()),
        NoFrustumCulling,
        Visibility::Hidden,
        AuthoredPartVisual(AuthoredPart::Controller),
    ));
    commands.spawn((
        Name::new("Gas engine mesh"),
        Mesh3d(gas_engine_mesh),
        MeshMaterial3d(authored_materials[AuthoredPart::GasEngine.index()].clone()),
        NoFrustumCulling,
        Visibility::Hidden,
        AuthoredPartVisual(AuthoredPart::GasEngine),
    ));
    commands.spawn((
        Name::new("Electric engine mesh"),
        Mesh3d(electric_engine_mesh),
        MeshMaterial3d(authored_materials[AuthoredPart::ElectricEngine.index()].clone()),
        NoFrustumCulling,
        Visibility::Hidden,
        AuthoredPartVisual(AuthoredPart::ElectricEngine),
    ));
    commands.spawn((
        Name::new("Gas transmission mesh"),
        Mesh3d(gas_transmission_mesh),
        MeshMaterial3d(authored_materials[AuthoredPart::GasTransmission.index()].clone()),
        NoFrustumCulling,
        Visibility::Hidden,
        AuthoredPartVisual(AuthoredPart::GasTransmission),
    ));
    commands.spawn((
        Name::new("Electric transmission mesh"),
        Mesh3d(electric_transmission_mesh),
        MeshMaterial3d(authored_materials[AuthoredPart::ElectricTransmission.index()].clone()),
        NoFrustumCulling,
        Visibility::Hidden,
        AuthoredPartVisual(AuthoredPart::ElectricTransmission),
    ));
    commands.spawn((
        Name::new("Servo mesh"),
        Mesh3d(servo_mesh),
        MeshMaterial3d(authored_materials[AuthoredPart::Servo.index()].clone()),
        NoFrustumCulling,
        Visibility::Hidden,
        AuthoredPartVisual(AuthoredPart::Servo),
    ));
    commands.spawn((
        Name::new("Seat mesh"),
        Mesh3d(seat_mesh),
        MeshMaterial3d(authored_materials[AuthoredPart::Seat.index()].clone()),
        NoFrustumCulling,
        Visibility::Hidden,
        AuthoredPartVisual(AuthoredPart::Seat),
    ));
    commands.spawn((
        Name::new("Input mesh"),
        Mesh3d(input_mesh),
        MeshMaterial3d(authored_materials[AuthoredPart::Input.index()].clone()),
        NoFrustumCulling,
        Visibility::Hidden,
        AuthoredPartVisual(AuthoredPart::Input),
    ));
    commands.spawn((
        Name::new("Disabled Dimension Link mesh"),
        Mesh3d(dimension_link_disabled_mesh),
        MeshMaterial3d(authored_materials[AuthoredPart::DimensionLinkDisabled.index()].clone()),
        NoFrustumCulling,
        Visibility::Hidden,
        AuthoredPartVisual(AuthoredPart::DimensionLinkDisabled),
    ));
    commands.spawn((
        Name::new("Enabled Dimension Link mesh"),
        Mesh3d(dimension_link_enabled_mesh),
        MeshMaterial3d(authored_materials[AuthoredPart::DimensionLinkEnabled.index()].clone()),
        NoFrustumCulling,
        Visibility::Hidden,
        AuthoredPartVisual(AuthoredPart::DimensionLinkEnabled),
    ));
    commands.spawn((
        Name::new("Joint x-ray mesh"),
        Mesh3d(joint_xray_mesh),
        MeshMaterial3d(joint_xray_material.clone()),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Hidden,
        JointXrayVisual,
    ));
    commands.spawn((
        Name::new("Placement lattice"),
        Mesh3d(placement_lattice_mesh),
        MeshMaterial3d(placement_lattice_material),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Hidden,
        PlacementLatticeVisual::default(),
    ));
    commands.spawn((
        Name::new("Smart placement guides"),
        Mesh3d(smart_guide_mesh),
        MeshMaterial3d(smart_guide_material),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Hidden,
        SmartGuideVisual::default(),
    ));
    commands.spawn((
        Name::new("Smart snap range"),
        Mesh3d(smart_snap_range_mesh),
        MeshMaterial3d(smart_snap_range_material),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Hidden,
        SmartSnapRangeVisual::default(),
    ));
    let shape_selected_material = materials.add(StandardMaterial {
        base_color: SHAPE_SELECTION_COLOR,
        cull_mode: None,
        unlit: true,
        ..default()
    });
    commands.spawn((
        Name::new("Selected shape node markers"),
        Mesh3d(shape_selected_mesh),
        MeshMaterial3d(shape_selected_material.clone()),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Hidden,
        ShapeSelectedVisual,
    ));
    let shape_plane_material = materials.add(StandardMaterial {
        base_color: SHAPE_SELECTION_COLOR.with_alpha(0.14),
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        unlit: true,
        ..default()
    });
    commands.spawn((
        Name::new("Drag plane"),
        Mesh3d(shape_plane_mesh),
        MeshMaterial3d(shape_plane_material),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Hidden,
        ShapePlaneVisual,
    ));
    commands.spawn((
        Name::new("Drag plane arrows"),
        Mesh3d(shape_arrow_mesh),
        MeshMaterial3d(shape_selected_material),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Hidden,
        ShapeArrowVisual,
    ));
    commands.spawn((
        Name::new("Shape node markers"),
        Mesh3d(shape_node_mesh),
        MeshMaterial3d(joint_xray_material.clone()),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Hidden,
        ShapeNodeVisual,
    ));
    commands.spawn((
        Name::new("Drive x-ray mesh"),
        Mesh3d(drive_xray_mesh),
        MeshMaterial3d(drive_xray_material),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Hidden,
        DriveXrayVisual,
    ));
    // Kept visible with a degenerate mesh while idle: a hidden mesh has no slab
    // allocation, so writing the first frame of a drag into it would log a
    // use-after-free.
    commands.spawn((
        Name::new("Drive wire drag"),
        Mesh3d(wire_drag_mesh),
        MeshMaterial3d(wire_drag_material),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Visible,
        WireDragVisual,
    ));
    commands.spawn((
        Name::new("Drive wire hover"),
        Mesh3d(wire_hover_mesh),
        MeshMaterial3d(wire_hover_material),
        Transform::default(),
        RenderLayers::layer(1),
        NoFrustumCulling,
        Visibility::Visible,
        WireHoverVisual,
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

    // Filter the authored sky once, then retain the resulting diffuse and
    // roughness-aware specular maps without regenerating them every frame.
    let environment_map = images.add(sky_cubemap(SKY_CUBEMAP_SIZE));

    let player_camera = PlayerCamera::default();
    let projection = Projection::Perspective(PerspectiveProjection {
        fov: settings.camera_fov_degrees().to_radians(),
        ..default()
    });
    let camera_transform = player_camera.apply_pullback(
        PlayerState::default().position + Vec3::Y * camera::EYE_HEIGHT,
        player_camera.look_rotation(),
    );
    commands
        .spawn((
            Name::new("Player camera"),
            tool_fx::bloom(),
            tool_fx::FxCamera,
            Camera3d::default(),
            render_experiments::current().msaa(),
            projection.clone(),
            garage::EXPOSURE,
            Tonemapping::SomewhatBoringDisplayTransform,
            garage::fog(),
            GeneratedEnvironmentMapLight {
                environment_map,
                intensity: SKY_ENVIRONMENT_INTENSITY,
                ..default()
            },
            camera_transform,
            player_camera,
            MainCamera,
            render_diagnostics::ProfiledCamera,
            FovCamera,
        ))
        .with_children(|camera| {
            camera.spawn((
                Name::new("Joint x-ray camera"),
                Camera3d::default(),
                // Both cameras share the world target and must use the same MSAA.
                render_experiments::current().msaa(),
                projection,
                // The overlay rides the camera that draws last. This pass loads
                // rather than clears, so an overlay painted before it is drawn
                // over by the joints showing through.
                bevy_mosaic::MosaicCamera,
                Camera {
                    order: 2,
                    clear_color: ClearColorConfig::None,
                    output_mode: bevy::camera::CameraOutputMode::Write {
                        blend_state: Some(
                            bevy::render::render_resource::BlendState::PREMULTIPLIED_ALPHA_BLENDING,
                        ),
                        clear_color: ClearColorConfig::None,
                    },
                    ..default()
                },
                Tonemapping::None,
                RenderLayers::layer(1),
                FovCamera,
                Transform::default(),
            ));
        });
}

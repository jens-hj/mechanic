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
mod scene;
mod schedule;
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

use avatar::spawn_player_avatar;
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
use camera::FovCamera;
use camera::{
    MainCamera, MaterialWheelState, PlayerCamera, PlayerState, SEATED_EYE_HEIGHT,
    seated_view_rotation,
};
use chroma::{ChromaBrush, ConstructionRenderMaterial};
use control_panel::ControlPanelState;
use controls::GameAction;
use creation_menu::CreationMenuState;
use creation_store::CreationStore;
use debug_freeze::DebugFrameFreeze;
use editor::{
    build_actions::{
        PlacedBearing, bearing_socket_targets, bearing_uses_socket,
        editor_part_is_static_or_pending,
    },
    dimensions::{BearingToolSettings, CylinderToolSettings},
    hover::{DRAG_DEAD_ZONE_RADIANS, PointerSample, clear_hover},
    pipe::{
        PipeEditMode, invalidate_pipe_drag, pipe_pointer_delta, rebuild_pipe_drag,
        refresh_pipe_drag,
    },
    placement::{
        PlacementLatticeVisual, SmartGuideVisual, SmartSnapRangeVisual, active_placement_grid,
        free_placement_point_on_miss,
    },
    preview::{
        ActionPreview, BearingVisual, ConstructionVisual, DeletePreview, DriveXrayVisual,
        EditorVisuals, FeaturePreviewKey, JointXrayVisual, SelectionPreview,
    },
    shape_actions::{
        LayerPreview, SHAPE_SELECTION_COLOR, ShapeArrowVisual, ShapeNodeVisual, ShapePlaneVisual,
        ShapeSelectedVisual,
    },
    state::{CurrentCreation, EditorGraph, EditorState},
};
use editor::{
    hammer::HammerInteraction,
    history::{EditorHistory, EditorSnapshot, cancel_transient_editor_state},
    overlay::{
        OverlayGeometry, append_axis_arrows, append_dashed_overlay_bar, append_drag_plane,
        append_overlay_bar, append_plane_arrows, append_region_outline, region_world_bounds,
        write_overlay,
    },
    raycast::{
        hovered_part, raycast_live_placed_bearing_discs, raycast_live_placed_bearings,
        raycast_placed_bearings_with_pose, raycast_simulation,
    },
    wiring::{WireDragVisual, WireHoverVisual},
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
        construction_material, construction_tint_mask_path, material_index, preview_material,
    },
};
use sequencer::{DriveSequencer, GearboxRuntime};
use settings::AppSettings;
use simulation::{
    publication::WorldPhysicsPublication, state::AppSimulation, visuals::SimulationVisualCache,
};

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
        .add_plugins(schedule::FramePlugin)
        .run();
}

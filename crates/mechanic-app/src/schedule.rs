//! The frame, in order.
//!
//! Every system that takes part in the editor and simulation frame is listed
//! here, grouped into [`FrameSet`]s that run strictly one after another. A
//! module that needs to run relative to the frame orders itself against a set,
//! never against another module's system function, so moving or splitting a
//! system cannot silently break someone else's ordering.

use bevy::prelude::{App, IntoScheduleConfigs, Plugin, Startup, SystemSet, Update};

use crate::camera::{self, apply_camera_fov};
use crate::debug_freeze::{debug_frame_updates_enabled, update_debug_frame_freeze};
use crate::editor::build_actions::handle_build_actions;
use crate::editor::creation::{handle_creation_menu_shortcut, handle_creation_request};
use crate::editor::dimensions::{
    handle_bearing_dimension_shortcuts, handle_cylinder_dimension_shortcuts,
    handle_dimension_link_interaction,
};
use crate::editor::hammer::handle_hammer_actions;
use crate::editor::history::handle_history_shortcut;
use crate::editor::hover::{handle_tool_change, update_hover};
use crate::editor::overlay::{sync_edit_overlay_transforms, sync_placement_overlays};
use crate::editor::placement::{
    rebuild_placement_snap_index, update_free_placement_settings, update_smart_snap_settings,
};
use crate::editor::preview::{sync_visual_meshes, update_joint_xray, update_previews};
use crate::editor::shape_actions::{
    handle_shape_actions, sync_drag_plane, sync_region_focus, sync_shape_nodes,
};
use crate::editor::shortcuts::{handle_control_panel_shortcut, handle_shortcuts};
use crate::editor::wiring::{update_wire_drag_preview, update_wire_hover_preview};
use crate::pause_menu::{
    begin_pause_frame, capture_control_binding, handle_pause_escape, handle_pause_request,
};
use crate::render::materials::prepare_bearing_texture_mips;
use crate::seat::handle_seat_interaction;
use crate::sequencer::run_drive_sequencer;
use crate::simulation::publication::maintain_space_simulation;
use crate::simulation::tick::{advance_simulation, poll_simulation_readbacks};
use crate::simulation::visuals::sync_simulation_visual_cache;
use crate::world::world_playing;
use crate::{
    avatar, controls, freeze, linear_editor, linear_render, live_edit, performance,
    performance_capture, piston_editor, piston_render, scene, suspension_render, tool_fx, ui,
    weld_tool,
};

/// Startup phases other plugins order themselves against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum StartupSet {
    /// The garage scene, cameras, shared meshes and materials exist afterwards.
    Scene,
}

/// The phases of one `Update` frame, in the order they run.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum FrameSet {
    /// The debug frame-freeze key. Runs even while the frame is frozen.
    DebugFreeze,
    /// Asset post-processing that must not stall behind a frozen frame.
    Assets,
    /// The performance overlay toggle. Runs even while the frame is frozen.
    PerformanceToggle,
    /// Raw input becomes game actions.
    Input,
    /// Actions that change modes, settings, menus, and pause state.
    Commands,
    /// Panels receive this frame's snapshot; history and creation requests apply.
    Interface,
    /// Field of view, the material wheel, and the player camera.
    Camera,
    /// Completed physics ticks and the dimension freeze are read back.
    Readback,
    /// Live-edit context, seats, and tool shortcuts.
    Interaction,
    /// Bearing, linear-bearing, and cylinder dimension adjustments.
    Dimensions,
    /// Tool selection, the placement snap index, and what the cursor is over.
    Hover,
    /// The build gesture: effects capture, suspension panel, graph commits.
    Build,
    /// Shape, layer, and hammer actions that follow the build gesture.
    Shape,
    /// Editor meshes, overlays, and wire previews follow the edited graph.
    Visuals,
    /// World publication, simulation visuals, drive programs, and physics ticks.
    Simulation,
    /// Tool previews, placed after the simulation has moved its bodies.
    Previews,
    /// Frame metrics, captures, and the panels that show them.
    Metrics,
}

/// Everything in [`FrameSet`] from `Input` on: skipped while a debug freeze holds the frame.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Freezable;

/// Registers the frame's sets and the systems that run in them.
pub(crate) struct FramePlugin;

impl Plugin for FramePlugin {
    #[expect(
        clippy::too_many_lines,
        reason = "the frame is kept in one visible execution order"
    )]
    fn build(&self, app: &mut App) {
        app.init_resource::<crate::dial_assignment::DialAssignments>();
        app.init_resource::<crate::button_config::ButtonConfiguration>();
        app.init_resource::<crate::input_parts::InputParts>();
        app.init_resource::<crate::physical_controls::PhysicalControls>();
        app.configure_sets(
            Update,
            (
                FrameSet::DebugFreeze,
                FrameSet::Assets,
                FrameSet::PerformanceToggle,
                FrameSet::Input,
                FrameSet::Commands,
                FrameSet::Interface,
                FrameSet::Camera,
                FrameSet::Readback,
                FrameSet::Interaction,
                FrameSet::Dimensions,
                FrameSet::Hover,
                FrameSet::Build,
                FrameSet::Shape,
                FrameSet::Visuals,
                FrameSet::Simulation,
                FrameSet::Previews,
                FrameSet::Metrics,
            )
                .chain(),
        )
        .configure_sets(
            Update,
            (
                FrameSet::Input,
                FrameSet::Commands,
                FrameSet::Interface,
                FrameSet::Camera,
                FrameSet::Readback,
                FrameSet::Interaction,
                FrameSet::Dimensions,
                FrameSet::Hover,
                FrameSet::Build,
                FrameSet::Shape,
                FrameSet::Visuals,
                FrameSet::Simulation,
                FrameSet::Previews,
                FrameSet::Metrics,
            )
                .in_set(Freezable),
        )
        .configure_sets(Update, Freezable.run_if(debug_frame_updates_enabled))
        .add_systems(
            Startup,
            (scene::setup.in_set(StartupSet::Scene), ui::mount).chain(),
        )
        .add_systems(
            Update,
            (
                update_debug_frame_freeze.in_set(FrameSet::DebugFreeze),
                prepare_bearing_texture_mips.in_set(FrameSet::Assets),
                (
                    crate::button_config::capture,
                    crate::physical_controls::capture,
                    performance::toggle,
                )
                    .chain()
                    .in_set(FrameSet::PerformanceToggle),
                (begin_pause_frame, controls::update_action_state)
                    .chain()
                    .in_set(FrameSet::Input),
                (
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
                    .chain()
                    .in_set(FrameSet::Commands),
                (
                    ui::dials::push,
                    ui::push,
                    ui::push_help,
                    ui::push_markers,
                    ui::push_player,
                    ui::sync_input,
                    handle_history_shortcut,
                    handle_creation_request,
                )
                    .chain()
                    .in_set(FrameSet::Interface),
                (
                    apply_camera_fov,
                    camera::update_material_wheel,
                    camera::update_player_camera,
                )
                    .chain()
                    .in_set(FrameSet::Camera),
                (
                    poll_simulation_readbacks.run_if(world_playing),
                    freeze::update.run_if(world_playing),
                )
                    .chain()
                    .in_set(FrameSet::Readback),
                (
                    live_edit::refresh_context,
                    crate::physical_controls::interact,
                    crate::dial_assignment::highlight,
                    handle_seat_interaction,
                    handle_shortcuts,
                )
                    .chain()
                    .in_set(FrameSet::Interaction),
                (
                    handle_bearing_dimension_shortcuts,
                    linear_editor::controls,
                    piston_editor::controls,
                    handle_cylinder_dimension_shortcuts,
                )
                    .chain()
                    .in_set(FrameSet::Dimensions),
                (
                    handle_tool_change,
                    rebuild_placement_snap_index,
                    update_hover,
                )
                    .chain()
                    .in_set(FrameSet::Hover),
                (
                    tool_fx::capture_gesture,
                    ui::push_suspension,
                    ui::button_config::push,
                    handle_build_actions,
                    crate::input_parts::update_input_placement,
                )
                    .chain()
                    .in_set(FrameSet::Build),
                (
                    tool_fx::finish_gesture,
                    handle_shape_actions,
                    ui::push_dimensions,
                    handle_hammer_actions,
                )
                    .chain()
                    .in_set(FrameSet::Shape),
            ),
        )
        .add_systems(
            Update,
            (
                (
                    update_joint_xray,
                    sync_placement_overlays,
                    sync_visual_meshes,
                    sync_shape_nodes,
                    sync_region_focus,
                    sync_drag_plane,
                    sync_edit_overlay_transforms,
                    update_wire_drag_preview,
                    update_wire_hover_preview,
                )
                    .chain()
                    .in_set(FrameSet::Visuals),
                (
                    maintain_space_simulation,
                    sync_simulation_visual_cache,
                    linear_render::sync_linear_bearing_visuals,
                    piston_render::sync_piston_visuals,
                    suspension_render::sync_suspension_visuals,
                    run_drive_sequencer,
                    advance_simulation.run_if(world_playing),
                    avatar::sync_player_avatar,
                    crate::input_render::sync_input_visuals,
                )
                    .chain()
                    .in_set(FrameSet::Simulation),
                (update_previews, live_edit::place_previews)
                    .chain()
                    .in_set(FrameSet::Previews),
                (
                    performance::sample,
                    performance_capture::sample,
                    ui::push_performance,
                    ui::push_driving,
                )
                    .chain()
                    .in_set(FrameSet::Metrics),
            ),
        )
        // Ordered against nothing, as it always has been.
        .add_systems(Update, weld_tool::draw_features);
    }
}

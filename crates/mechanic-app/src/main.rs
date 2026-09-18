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
#[cfg(test)]
mod moving_tool_tests;
mod multitool;
mod pause_menu;
mod performance;
mod performance_capture;
mod pose;
#[cfg(test)]
mod publication_frame_tests;
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
#[cfg(test)]
use builder::{candidate_from_hit, stage_bearing_attachment};
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

#[cfg(test)]
mod debug_frame_freeze_tests {
    use crate::debug_freeze::{DebugFrameFreeze, DebugFrameFreezeEffect};

    #[test]
    fn freeze_stays_active_until_a_click_is_consumed() {
        let mut freeze = DebugFrameFreeze::default();
        assert_eq!(
            freeze.advance(true, false),
            DebugFrameFreezeEffect::PauseTime
        );
        assert!(freeze.blocks_updates());

        assert_eq!(freeze.advance(false, true), DebugFrameFreezeEffect::None);
        assert!(freeze.blocks_updates());

        assert_eq!(
            freeze.advance(false, false),
            DebugFrameFreezeEffect::ResumeTime
        );
        assert!(!freeze.blocks_updates());
    }
}

#[cfg(test)]
mod world_physics_publication_tests {
    use bevy::prelude::{IVec3, Quat, Vec3};
    use mechanic_core::{
        BuildCommand, BuildOutcome, BuildPose, CuboidSpec, DimensionLinkId, DimensionLinkSpec,
        FaceKind, FaceRef, GridRotation, WeldSpec,
    };
    use mechanic_gpu::GpuTransform;

    use super::ConstructionGraph;
    use crate::editor::build_actions::editor_part_is_static_or_pending;
    use crate::simulation::publication::{
        rebuilt_body_states, world_mechanism_self_collisions, world_physics_result_is_current,
    };
    use crate::simulation::state::AppSimulation;

    #[test]
    fn only_the_latest_graph_and_foundation_revision_can_publish_physics() {
        let desired = (42, 7);

        assert!(world_physics_result_is_current(desired, desired));
        assert!(!world_physics_result_is_current((41, 7), desired));
        assert!(!world_physics_result_is_current((42, 6), desired));
    }

    #[test]
    fn linked_vehicle_preserves_internal_mechanism_contacts() {
        let mut graph = ConstructionGraph::new();

        assert!(world_mechanism_self_collisions(&graph));
        graph
            .apply(BuildCommand::SpawnDimensionLink(DimensionLinkSpec::new(
                DimensionLinkId(1),
                BuildPose::default(),
            )))
            .unwrap();
        assert!(world_mechanism_self_collisions(&graph));
    }

    #[test]
    fn pending_editor_parts_remain_buildable_before_physics_publication() {
        let mut published = ConstructionGraph::new();
        let BuildOutcome::Spawned(published_part) = published
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([1; 3], BuildPose::default()).unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let mut graph = published.clone();
        let BuildOutcome::Spawned(pending_part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([1; 3], BuildPose::new(IVec3::X, GridRotation::default())).unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let simulation = AppSimulation {
            creation: Some(published.compile().unwrap()),
            published_graph: published,
            ..Default::default()
        };

        assert!(!editor_part_is_static_or_pending(
            &graph,
            &simulation,
            published_part
        ));
        assert!(editor_part_is_static_or_pending(
            &graph,
            &simulation,
            pending_part
        ));
    }

    #[test]
    fn published_static_parts_remain_buildable_and_moving_parts_do_not() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(static_part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([1; 3], BuildPose::default()).unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let BuildOutcome::Spawned(moving_part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([1; 3], BuildPose::new(IVec3::X, GridRotation::default())).unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let simulation = AppSimulation {
            creation: Some(graph.compile_with_static_parts([static_part]).unwrap()),
            published_graph: graph.clone(),
            ..Default::default()
        };

        assert!(editor_part_is_static_or_pending(
            &graph,
            &simulation,
            static_part
        ));
        assert!(!editor_part_is_static_or_pending(
            &graph,
            &simulation,
            moving_part
        ));
    }

    #[test]
    fn deleting_dimension_link_preserves_the_live_body_pose() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(block) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [2, 1, 1],
                    BuildPose::new(IVec3::new(0, 1, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let BuildOutcome::Spawned(link) = graph
            .apply(BuildCommand::SpawnDimensionLink(DimensionLinkSpec::new(
                DimensionLinkId(7),
                BuildPose::new(IVec3::new(2, 1, 0), GridRotation::default()),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(block, FaceKind::PositiveX),
                second: FaceRef::part(link, FaceKind::NegativeX),
            }))
            .unwrap();
        let previous_creation = graph.compile().unwrap();
        let live_rotation = Quat::from_rotation_y(0.7);
        let live_position = Vec3::new(4.0, 5.0, 6.0);
        let live_transform = GpuTransform {
            position: live_position.extend(0.0).to_array(),
            rotation: live_rotation.to_array(),
        };
        let previous = AppSimulation {
            creation: Some(previous_creation.clone()),
            published_graph: graph.clone(),
            previous_transforms: vec![live_transform],
            transforms: vec![live_transform],
            previous_snapshot_tick: 1,
            snapshot_tick: 2,
            live_state: Some(crate::simulation::state::LivePhysicsState {
                tick: 3,
                transforms: vec![GpuTransform {
                    position: (live_position + Vec3::Y).extend(0.0).to_array(),
                    ..live_transform
                }],
                velocities: vec![mechanic_gpu::GpuVelocity {
                    linear: [2.0, 3.0, 4.0, 0.0],
                    angular: [0.0, 0.0, 2.0, 0.0],
                }],
                coordinates: Vec::new(),
            }),
            ..Default::default()
        };

        graph.apply(BuildCommand::Remove(link)).unwrap();
        let creation = graph.compile().unwrap();
        let root_delta = creation.compounds[0].root_translation
            - previous_creation.compounds[0].root_translation;
        let expected_position = live_position + Vec3::Y + live_rotation * root_delta;
        let (transforms, velocities) = rebuilt_body_states(&creation, &graph, &previous);
        let rebuilt = transforms[0];

        assert!(Vec3::from_slice(&rebuilt.position[..3]).abs_diff_eq(expected_position, 1.0e-5));
        assert!(Quat::from_array(rebuilt.rotation).abs_diff_eq(live_rotation, 1.0e-5));
        let expected_velocity =
            Vec3::new(2.0, 3.0, 4.0) + Vec3::new(0.0, 0.0, 2.0).cross(live_rotation * root_delta);
        assert!(
            Vec3::from_slice(&velocities[0].linear[..3]).abs_diff_eq(expected_velocity, 1.0e-5)
        );
        assert_eq!(
            velocities[0].angular.map(f32::to_bits),
            [0.0_f32, 0.0, 2.0, 0.0].map(f32::to_bits)
        );
    }
    #[test]
    fn surviving_joint_state_follows_bearing_identity_after_removal() {
        use mechanic_core::BearingSpec;
        use mechanic_gpu::GpuMechanismCoordinate;
        let mut graph = ConstructionGraph::new();
        let mut joints = Vec::new();
        for z in [0_i16, 8] {
            let mut parts = Vec::new();
            for x in [0, 4] {
                let BuildOutcome::Spawned(part) = graph
                    .apply(BuildCommand::Spawn(
                        CuboidSpec::new(
                            [4; 3],
                            BuildPose::new(IVec3::new(x, 2, i32::from(z)), GridRotation::default()),
                        )
                        .unwrap(),
                    ))
                    .unwrap()
                else {
                    unreachable!()
                };
                parts.push(part);
            }
            let BuildOutcome::BearingAdded(bearing) = graph
                .apply(BuildCommand::AddBearing(BearingSpec::new(
                    FaceRef::part(parts[0], FaceKind::PositiveX),
                    FaceRef::part(parts[1], FaceKind::NegativeX),
                    Vec3::new(0.5, 0.5, f32::from(z) * 0.25),
                    Vec3::X,
                )))
                .unwrap()
            else {
                unreachable!()
            };
            joints.push(bearing);
        }
        let compiled = graph.compile().unwrap();
        let coordinates = vec![
            GpuMechanismCoordinate {
                position: 7.0,
                velocity: 2.0,
            },
            GpuMechanismCoordinate {
                position: -9.0,
                velocity: -3.0,
            },
        ];
        let surviving =
            coordinates[compiled.loop_topology.bearing_coordinates[&joints[1]] as usize];
        let previous = AppSimulation {
            creation: Some(compiled),
            live_state: Some(crate::simulation::state::LivePhysicsState {
                tick: 10,
                transforms: Vec::new(),
                velocities: Vec::new(),
                coordinates,
            }),
            ..Default::default()
        };
        graph.apply(BuildCommand::RemoveBearing(joints[0])).unwrap();
        let rebuilt = graph.compile().unwrap();
        assert_eq!(
            crate::simulation::publication::rebuilt_mechanism_coordinates(
                &rebuilt,
                &previous,
                &[],
                &[]
            ),
            vec![surviving]
        );
    }
    #[test]
    fn additions_removals_and_splits_inherit_the_live_rigid_velocity_field() {
        use mechanic_gpu::GpuVelocity;
        let mut original = ConstructionGraph::new();
        let mut parts = Vec::new();
        for x in 0..3 {
            let BuildOutcome::Spawned(part) = original
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(
                        [1; 3],
                        BuildPose::new(IVec3::new(x, 8, 0), GridRotation::default()),
                    )
                    .unwrap(),
                ))
                .unwrap()
            else {
                unreachable!()
            };
            if let Some(&last) = parts.last() {
                original
                    .apply(BuildCommand::Weld(WeldSpec {
                        first: FaceRef::part(last, FaceKind::PositiveX),
                        second: FaceRef::part(part, FaceKind::NegativeX),
                    }))
                    .unwrap();
            }
            parts.push(part);
        }
        let compiled = original.compile().unwrap();
        let old_root = compiled.compounds[0].root_translation;
        let live_position = Vec3::new(2.0, 4.0, 6.0);
        let live_rotation = Quat::from_rotation_y(0.8);
        let linear = Vec3::new(1.0, 2.0, 3.0);
        let angular = Vec3::new(0.0, 0.0, 2.0);
        let previous = AppSimulation {
            creation: Some(compiled),
            published_graph: original.clone(),
            live_state: Some(crate::simulation::state::LivePhysicsState {
                tick: 20,
                transforms: vec![GpuTransform {
                    position: live_position.extend(0.0).to_array(),
                    rotation: live_rotation.to_array(),
                }],
                velocities: vec![GpuVelocity {
                    linear: linear.extend(0.0).to_array(),
                    angular: angular.extend(0.0).to_array(),
                }],
                coordinates: Vec::new(),
            }),
            ..Default::default()
        };
        for edit in 0..3 {
            let mut graph = original.clone();
            match edit {
                0 => {
                    graph.apply(BuildCommand::Remove(parts[1])).unwrap();
                }
                1 => {
                    graph.apply(BuildCommand::Remove(parts[2])).unwrap();
                }
                _ => {
                    let BuildOutcome::Spawned(part) = graph
                        .apply(BuildCommand::Spawn(
                            CuboidSpec::new(
                                [1; 3],
                                BuildPose::new(IVec3::new(3, 8, 0), GridRotation::default()),
                            )
                            .unwrap(),
                        ))
                        .unwrap()
                    else {
                        unreachable!()
                    };
                    graph
                        .apply(BuildCommand::Weld(WeldSpec {
                            first: FaceRef::part(parts[2], FaceKind::PositiveX),
                            second: FaceRef::part(part, FaceKind::NegativeX),
                        }))
                        .unwrap();
                }
            }
            let rebuilt = graph.compile().unwrap();
            assert_eq!(rebuilt.compounds.len(), if edit == 0 { 2 } else { 1 });
            let (transforms, velocities) = rebuilt_body_states(&rebuilt, &graph, &previous);
            for (body, compound) in rebuilt.compounds.iter().enumerate() {
                let displacement = live_rotation * (compound.root_translation - old_root);
                assert!(
                    Vec3::from_slice(&transforms[body].position[..3])
                        .abs_diff_eq(live_position + displacement, 1.0e-5)
                );
                assert!(
                    Vec3::from_slice(&velocities[body].linear[..3])
                        .abs_diff_eq(linear + angular.cross(displacement), 1.0e-5)
                );
                assert_eq!(
                    velocities[body].angular.map(f32::to_bits),
                    angular.extend(0.0).to_array().map(f32::to_bits)
                );
            }
        }
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "share the moving-frame fixture across both joint kinds"
    )]
    fn promoted_closure_recovers_motion_in_a_rotated_world_frame() {
        use mechanic_core::BearingSpec;
        use mechanic_gpu::GpuVelocity;
        let mut graph = ConstructionGraph::new();
        let mut parts = Vec::new();
        for x in [0, 4] {
            let BuildOutcome::Spawned(part) = graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(
                        [4; 3],
                        BuildPose::new(IVec3::new(x, 2, 0), GridRotation::default()),
                    )
                    .unwrap(),
                ))
                .unwrap()
            else {
                unreachable!()
            };
            parts.push(part);
        }
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(parts[0], FaceKind::PositiveX),
                FaceRef::part(parts[1], FaceKind::NegativeX),
                Vec3::new(0.5, 0.5, 0.0),
                Vec3::X,
            )))
            .unwrap();
        let mut compiled = graph.compile().unwrap();
        let mut old = compiled.clone();
        old.loop_topology.bearing_coordinates.clear();
        old.loop_topology.tree_bearings.clear();
        old.bearings[0].coordinate_index = None;
        let previous = AppSimulation {
            creation: Some(old),
            live_state: Some(crate::simulation::state::LivePhysicsState {
                tick: 9,
                transforms: Vec::new(),
                velocities: Vec::new(),
                coordinates: Vec::new(),
            }),
            ..Default::default()
        };
        let rotation_a = Quat::from_rotation_y(0.9);
        let origin = Vec3::new(4.0, 8.0, 3.0);
        let linear_a = Vec3::new(2.0, 3.0, 4.0);
        let angular_a = Vec3::new(0.3, 0.2, 0.1);
        for linear in [false, true] {
            if linear {
                compiled.bearings[0].kind =
                    mechanic_core::BearingKind::Linear(mechanic_core::LinearBearing {
                        dimensions: mechanic_core::LinearBearingDimensions::default(),
                        mount_normal: Vec3::Y,
                        face: mechanic_core::CarriageFace::Top,
                    });
            }
            let bearing = compiled.bearings[0];
            let rotation_b = if linear {
                rotation_a
            } else {
                rotation_a * Quat::from_rotation_x(0.7)
            };
            let axis = rotation_a * Vec3::X;
            let arm_a = rotation_a * bearing.local_anchor_a;
            let arm_b = rotation_b * bearing.local_anchor_b;
            let separation = if linear { axis * 0.2 } else { Vec3::ZERO };
            let position_b = origin + arm_a - arm_b + separation;
            let angular_b = angular_a + if linear { Vec3::ZERO } else { axis * 1.2 };
            let linear_b = linear_a + angular_a.cross(arm_a + separation) - angular_b.cross(arm_b)
                + if linear { axis * 0.3 } else { Vec3::ZERO };
            let mut transforms = vec![
                GpuTransform {
                    position: [0.0; 4],
                    rotation: Quat::IDENTITY.to_array()
                };
                2
            ];
            let mut velocities = vec![
                GpuVelocity {
                    linear: [0.0; 4],
                    angular: [0.0; 4]
                };
                2
            ];
            for (body, position, rotation, velocity, angular) in [
                (bearing.compound_a, origin, rotation_a, linear_a, angular_a),
                (
                    bearing.compound_b,
                    position_b,
                    rotation_b,
                    linear_b,
                    angular_b,
                ),
            ] {
                transforms[body as usize] = GpuTransform {
                    position: position.extend(0.0).to_array(),
                    rotation: rotation.to_array(),
                };
                velocities[body as usize] = GpuVelocity {
                    linear: velocity.extend(0.0).to_array(),
                    angular: angular.extend(0.0).to_array(),
                };
            }
            let coordinates = crate::simulation::publication::rebuilt_mechanism_coordinates(
                &compiled,
                &previous,
                &transforms,
                &velocities,
            );
            assert!((coordinates[0].position - if linear { 0.2 } else { 0.7 }).abs() < 1.0e-5);
            assert!((coordinates[0].velocity - if linear { 0.3 } else { 1.2 }).abs() < 1.0e-5);
        }
    }
}

#[cfg(test)]
mod transmission_speed_tests {
    use bevy::prelude::{Quat, Vec3};

    use crate::sequencer::signed_joint_speed;

    #[test]
    fn snapshot_pairs_measure_signed_relative_joint_speed() {
        let delta_seconds = 0.1;
        let positive = signed_joint_speed(
            Quat::IDENTITY,
            Quat::IDENTITY,
            Quat::IDENTITY,
            Quat::from_axis_angle(Vec3::Z, 0.2),
            Vec3::Z,
            delta_seconds,
        );
        let negative = signed_joint_speed(
            Quat::IDENTITY,
            Quat::IDENTITY,
            Quat::IDENTITY,
            Quat::from_axis_angle(Vec3::Z, -0.2),
            Vec3::Z,
            delta_seconds,
        );
        assert!((positive - 2.0).abs() < 1.0e-5);
        assert!((negative + 2.0).abs() < 1.0e-5);
    }
}

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

#[cfg(test)]
mod pause_feature_tests {
    use super::*;
    use crate::camera::set_projection_fov;
    use crate::pause_menu::EscapeTarget;
    use crate::pause_menu::ExitDisposition;
    use crate::pause_menu::escape_target;
    use crate::pause_menu::exit_disposition;

    #[test]
    fn existing_escape_owners_take_priority_before_pause() {
        assert_eq!(
            escape_target(false, false, true, true, true),
            EscapeTarget::ExistingUi
        );
        assert_eq!(
            escape_target(false, false, false, true, true),
            EscapeTarget::ControlPanel
        );
        assert_eq!(
            escape_target(false, false, false, false, true),
            EscapeTarget::WorldState
        );
        assert_eq!(
            escape_target(false, false, false, false, false),
            EscapeTarget::OpenPause
        );
    }

    #[test]
    fn pause_confirmation_cancels_before_pause_continues() {
        assert_eq!(
            escape_target(true, true, true, true, true),
            EscapeTarget::PauseSubmenu
        );
        assert_eq!(
            escape_target(true, false, true, true, true),
            EscapeTarget::PauseMenu
        );
    }

    #[test]
    fn revisions_restore_clean_identity_through_undo_and_redo() {
        let graph = ConstructionGraph::default();
        let state = EditorState::default();
        let mut history = EditorHistory::default();
        assert!(
            !history.is_dirty(),
            "the initial blank construction is clean"
        );

        history.commit(EditorSnapshot::capture(&graph, &state));
        assert!(history.is_dirty(), "a successful edit creates a revision");
        history.mark_clean();
        assert!(
            !history.is_dirty(),
            "a successful save marks that revision clean"
        );

        history.commit(EditorSnapshot::capture(&graph, &state));
        assert!(history.is_dirty(), "a later edit is dirty");
        history
            .undo(EditorSnapshot::capture(&graph, &state))
            .expect("undo reaches the saved revision");
        assert!(!history.is_dirty());
        history
            .redo(EditorSnapshot::capture(&graph, &state))
            .expect("redo reaches the edited revision");
        assert!(history.is_dirty());
    }

    #[test]
    fn clean_exit_is_immediate_and_dirty_exit_requires_confirmation() {
        assert_eq!(exit_disposition(false), ExitDisposition::Exit);
        assert_eq!(exit_disposition(true), ExitDisposition::ConfirmUnsaved);
    }

    #[test]
    fn both_camera_projections_receive_the_same_fov() {
        let mut projections = [
            Projection::Perspective(PerspectiveProjection::default()),
            Projection::Perspective(PerspectiveProjection::default()),
        ];
        let wanted = 80.0_f32.to_radians();
        for projection in &mut projections {
            set_projection_fov(projection, wanted);
        }
        for projection in projections {
            let Projection::Perspective(perspective) = projection else {
                panic!("test projection remains perspective");
            };
            assert!((perspective.fov - wanted).abs() < f32::EPSILON);
        }
    }
}

#[cfg(test)]
mod rendering_tests {
    use std::time::Instant;

    #[test]
    #[ignore = "CPU-only saved-world visual rebuild measurement"]
    #[expect(
        clippy::too_many_lines,
        reason = "keep the saved-world stage measurements together"
    )]
    fn measure_builder_body_mesh_rebuild() {
        let source = std::env::var("MECHANIC_EDIT_FIXTURE").ok().map_or_else(
            || {
                include_str!(
                    "../../mechanic-bench/tests/fixtures/builder-world/generations/20/world.ron"
                )
                .to_owned()
            },
            |path| std::fs::read_to_string(path).unwrap(),
        );
        let instance: mechanic_world::WorldCreationInstanceDoc = ron::from_str(&source).unwrap();
        let graph = instance.creation.into_graph().unwrap().graph;
        let creation = graph.compile().unwrap();
        let collision_started = Instant::now();
        let _ = mechanic_physics::MachineCollisionGeometry::new(&creation, 1).unwrap();
        eprintln!(
            "CPU collision preparation {:?}",
            collision_started.elapsed()
        );
        let machine_started = Instant::now();
        let _ = mechanic_physics::CpuMachine::new(
            creation.clone(),
            1,
            mechanic_physics::MachineState::at_rest(&creation),
        )
        .unwrap();
        eprintln!("CPU machine initialization {:?}", machine_started.elapsed());
        for treatment in [
            mechanic_core::EdgeTreatment::Chamfer,
            mechanic_core::EdgeTreatment::Fillet,
        ] {
            let started = Instant::now();
            let mut worst = std::time::Duration::ZERO;
            let mut tries = 0;
            for (part, spec) in graph.parts() {
                if !matches!(spec, super::PartSpec::Cuboid(_)) {
                    continue;
                }
                let owner = mechanic_core::SolidOwner::Part(part);
                let solid = graph.evaluated_solid_shared(owner).unwrap();
                for edge in solid.logical_edges.iter().take(1) {
                    let mut preview = graph.clone();
                    let step = Instant::now();
                    let _ = preview.apply(super::BuildCommand::AddShapeFeature(
                        mechanic_core::ShapeFeature::new(
                            [mechanic_core::EdgeChainRef {
                                owner,
                                edge: edge.key,
                            }],
                            treatment,
                            1,
                        ),
                    ));
                    worst = worst.max(step.elapsed());
                    tries += 1;
                }
            }
            eprintln!(
                "{treatment:?}: tries={tries} total={:?} worst={worst:?}",
                started.elapsed()
            );
        }
        let transforms = vec![
            mechanic_gpu::GpuTransform {
                position: [0.0; 4],
                rotation: [0.0, 0.0, 0.0, 1.0],
            };
            creation.compounds.len()
        ];
        for sample in 0..3 {
            let started = Instant::now();
            let texture_started = Instant::now();
            let _ = crate::render::mesh::pipe::pipe_texture_offsets(&graph);
            eprintln!("texture offsets {:?}", texture_started.elapsed());
            let ends_started = Instant::now();
            let _ = crate::render::mesh::pipe::welded_pipe_ends(&graph);
            eprintln!("welded ends {:?}", ends_started.elapsed());
            let mut vertices = 0;
            for body in 0..creation.compounds.len() {
                let body = u32::try_from(body).unwrap();
                for material in super::ConstructionMaterial::ALL {
                    if crate::render::mesh::simulation::simulation_material_is_present_for_compound(
                        &graph, &creation, body, material,
                    ) {
                        let mesh_started = Instant::now();
                        vertices +=
                            crate::render::mesh::simulation::local_simulation_material_mesh(
                                &graph,
                                &creation,
                                &transforms,
                                body,
                                material,
                            )
                            .count_vertices();
                        if sample == 0 {
                            eprintln!(
                                "body={body} material={material:?} mesh={:?}",
                                mesh_started.elapsed()
                            );
                        }
                    }
                }
            }
            eprintln!(
                "builder bodies={} sample={sample} vertices={vertices} meshes_ms={:.3}",
                creation.compounds.len(),
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
    }

    use bevy::{
        image::{ImageAddressMode, ImageFilterMode, ImageLoaderSettings},
        mesh::VertexAttributeValues,
        prelude::{
            AlphaMode, App, Color, EnvironmentMapLight, GeneratedEnvironmentMapLight, Handle,
            IVec3, Image, Mesh, Quat, StandardMaterial, Update, Vec2, Vec3,
        },
    };
    use mechanic_core::{
        BearingDimensions, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph,
        ConstructionMaterial, ControllerSpec, CuboidSpec, CylinderDimensions, CylinderSpec,
        DimensionLinkId, DimensionLinkSpec, DriveLimits, DriveLinkSpec, DriveProgram, DriveState,
        DriveTarget, EdgeChainRef, EdgeTreatment, EngineKind, EngineSpec, FaceKind, FaceOwner,
        FaceRef, GridRotation, InputSpec, MaterialAppearance, MaterialColor, MaterialDye,
        MaterialFinish, MaterialShift, PartSpec, PipeBendDimensions, SeatSpec, ServoSpec,
        ShapeFeature, SolidOwner,
    };
    use mechanic_gpu::GpuTransform;

    use super::AuthoredPart;
    use crate::builder::BEARING_DEPTH;
    use crate::editor::build_actions::PlacedBearing;
    use crate::editor::preview::{
        BLOCK_SHEET_PREVIEW_INSET_METERS, bearing_preview_dimensions_changed,
        drive_xray_is_visible, joint_xray_is_visible, should_sync_editor_visual_meshes,
    };
    use crate::pose::transform_from_gpu;
    use crate::render::authored::authored_uvs;
    use crate::render::materials::{
        BEARING_RENDER_RADIAL_SKIN, authored_preview_material, bearing_pbr_material,
        configure_authored_texture, configure_bearing_texture, configure_repeating_texture,
        construction_tint_mask_path, preview_material,
    };
    use crate::render::mesh::bearing::{
        append_bearing_cylinder, bearing_profile_plan, bearing_u_repeat, combined_bearing_mesh,
        single_bearing_mesh,
    };
    use crate::render::mesh::construction::{
        MATERIAL_TEXTURE_METERS_PER_REPEAT, MATERIAL_TEXTURE_PIXELS_PER_BLOCK,
        MATERIAL_TEXTURE_PIXELS_PER_SIDE, append_merged_block_cuboids,
        append_pipe_bend_texture_coordinates, combined_authored_construction_mesh,
        combined_controller_mesh, combined_material_construction_mesh, single_authored_part_mesh,
        single_cylinder_mesh,
    };
    use crate::render::mesh::drive::combined_drive_xray_mesh;
    use crate::render::mesh::pipe::{append_cylinder_shape, append_pipe_bend_shape};
    use crate::render::mesh::preview::{
        block_sheet_bounds, block_sheet_preview_mesh, delete_preview_mesh,
    };
    use crate::render::mesh::primitives::renderable_mesh;
    use crate::render::mesh::simulation::{
        SimulationMeshKind, combined_simulation_bearing_mesh, combined_simulation_material_mesh,
        combined_simulation_mesh, local_simulation_material_mesh, simulation_material_is_present,
    };

    fn avatar_material_fixture() -> (App, crate::avatar::AvatarMaterials) {
        use bevy::{asset::AssetApp, prelude::*};

        let mut app = App::new();
        app.add_plugins((MinimalPlugins, AssetPlugin::default()))
            .init_asset::<StandardMaterial>();
        let mut materials = app.world_mut().resource_mut::<Assets<StandardMaterial>>();
        let handles = crate::avatar::AvatarMaterials {
            clothing: materials.add(crate::avatar::avatar_material(Color::srgb(
                0.08, 0.48, 0.46,
            ))),
            head: materials.add(crate::avatar::avatar_material(Color::srgb(
                0.72, 0.58, 0.46,
            ))),
            boots: materials.add(crate::avatar::avatar_material(Color::srgb(
                0.055, 0.065, 0.075,
            ))),
        };
        (app, handles)
    }

    #[test]
    fn avatar_is_opaque_only_at_full_visibility_and_preserves_its_fade() {
        use bevy::prelude::*;

        let (mut app, handles) = avatar_material_fixture();
        let mut materials = app.world_mut().resource_mut::<Assets<StandardMaterial>>();
        let original_colors = [&handles.clothing, &handles.head, &handles.boots]
            .map(|handle| materials.get(handle).unwrap().base_color);
        for (alpha, mode) in [
            (0.0, AlphaMode::Blend),
            (0.5, AlphaMode::Blend),
            (0.999, AlphaMode::Blend),
            (1.0, AlphaMode::Opaque),
            (0.5, AlphaMode::Blend),
            (0.0, AlphaMode::Blend),
        ] {
            crate::avatar::sync_avatar_materials(&mut materials, &handles, alpha);
            for (handle, color) in [&handles.clothing, &handles.head, &handles.boots]
                .into_iter()
                .zip(original_colors)
            {
                let material = materials.get(handle).unwrap();
                assert_eq!(material.alpha_mode, mode);
                assert_eq!(material.base_color, color.with_alpha(alpha));
                assert_eq!(material.perceptual_roughness.to_bits(), 0.92_f32.to_bits());
            }
        }
    }

    #[test]
    fn avatar_unchanged_opacity_does_not_emit_material_modifications() {
        use bevy::prelude::*;

        let (mut app, handles) = avatar_material_fixture();
        app.update();
        app.world_mut()
            .resource_mut::<Messages<AssetEvent<StandardMaterial>>>()
            .clear();
        for alpha in [0.5, 1.0, 0.5, 0.0] {
            for expected_modifications in [3, 0, 0] {
                crate::avatar::sync_avatar_materials(
                    &mut app.world_mut().resource_mut::<Assets<StandardMaterial>>(),
                    &handles,
                    alpha,
                );
                app.update();
                let modified = app
                    .world_mut()
                    .resource_mut::<Messages<AssetEvent<StandardMaterial>>>()
                    .drain()
                    .filter(|event| matches!(event, AssetEvent::Modified { .. }))
                    .count();
                assert_eq!(modified, expected_modifications);
            }
        }
    }

    #[test]
    fn tool_changes_do_not_rebuild_editor_meshes_over_a_running_simulation() {
        assert!(!should_sync_editor_visual_meshes(true, true));
        assert!(should_sync_editor_visual_meshes(true, false));
        assert!(!should_sync_editor_visual_meshes(false, false));
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "keep the rendering transition and its ECS fixture together"
    )]
    fn deleting_last_dimension_link_restores_static_creation_visuals() {
        use crate::editor::preview::{BearingVisual, ConstructionVisual, EditorVisuals};
        use crate::editor::state::{EditorGraph, EditorState};
        use crate::render::authored::AuthoredPartVisual;
        use crate::simulation::visuals::{SimulationBodyVisualRoot, SimulationVisualCache};
        use bevy::prelude::*;

        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([2, 1, 1], BuildPose::default()).unwrap(),
            ))
            .unwrap();
        graph
            .apply(BuildCommand::SpawnController(
                mechanic_core::ControllerSpec::new(BuildPose::new(
                    IVec3::new(4, 0, 0),
                    GridRotation::default(),
                )),
            ))
            .unwrap();
        let BuildOutcome::Spawned(link) = graph
            .apply(BuildCommand::SpawnDimensionLink(DimensionLinkSpec::new(
                DimensionLinkId(7),
                BuildPose::default(),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        let rendered_graph = graph.clone();
        graph.apply(BuildCommand::Remove(link)).unwrap();
        let creation = graph
            .compile_with_static_parts(graph.parts().map(|(part, _)| part))
            .unwrap();
        assert!(!crate::simulation::publication::creation_requires_live_physics(&creation));
        let mut meshes = Assets::<Mesh>::default();
        let visuals = EditorVisuals {
            construction_meshes: std::array::from_fn(|_| meshes.add(Cuboid::default())),
            controller_mesh: meshes.add(Cuboid::default()),
            ..Default::default()
        };
        let material = graph
            .parts()
            .find_map(|(_, spec)| crate::render::mesh::construction::ordinary_material(*spec))
            .unwrap();
        let block_mesh =
            visuals.construction_meshes[crate::render::materials::material_index(material)].clone();
        let mut app = App::new();
        let old_root = app
            .world_mut()
            .spawn((SimulationBodyVisualRoot(0), Transform::IDENTITY))
            .id();
        let block_visual = app
            .world_mut()
            .spawn((ConstructionVisual(material), Visibility::Hidden))
            .id();
        let controller_visual = app
            .world_mut()
            .spawn((
                AuthoredPartVisual(AuthoredPart::Controller),
                Visibility::Hidden,
            ))
            .id();
        app.world_mut().spawn((BearingVisual, Visibility::Hidden));
        app.insert_resource(EditorGraph(graph.clone()))
            .insert_resource(EditorState {
                rendered_graph,
                rendered_world_revision: Some((1, 1)),
                construction_mesh_dirty: true,
                ..Default::default()
            })
            .insert_resource(crate::simulation::state::AppSimulation {
                creation: Some(creation),
                published_graph: graph,
                world_revision: Some((2, 2)),
                ..Default::default()
            })
            .insert_resource(SimulationVisualCache {
                revision: Some((1, 1)),
                roots: vec![old_root],
                ..Default::default()
            })
            .insert_resource(visuals)
            .insert_resource(meshes)
            .init_resource::<super::world::WorldRuntime>()
            .init_resource::<super::shape_tool::ShapeMirror>()
            .init_resource::<super::DriveSequencer>()
            .init_resource::<super::SelectedTool>()
            .add_systems(
                Update,
                (
                    crate::editor::preview::sync_visual_meshes,
                    crate::simulation::visuals::sync_simulation_visual_cache,
                )
                    .chain(),
            );

        for _ in 0..2 {
            app.update();
            assert_eq!(
                *app.world().get::<Visibility>(block_visual).unwrap(),
                Visibility::Visible
            );
            assert_eq!(
                *app.world().get::<Visibility>(controller_visual).unwrap(),
                Visibility::Visible
            );
            assert!(
                app.world()
                    .resource::<Assets<Mesh>>()
                    .get(&block_mesh)
                    .unwrap()
                    .count_vertices()
                    > 0
            );
            assert!(app.world().get_entity(old_root).is_err());
            assert!(
                app.world()
                    .resource::<SimulationVisualCache>()
                    .roots
                    .is_empty()
            );
        }
    }

    /// A ground-anchored block is drawn only by the shared static mesh, and that
    /// mesh is rebuilt from the published scene alone. Terrain collision
    /// readiness must never be part of that path: while terrain streams, a
    /// pending cut once left new static blocks solid but invisible.
    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "keep the publication regression and its ECS fixture together"
    )]
    fn newly_published_static_blocks_reach_the_shared_construction_mesh() {
        use super::SelectedTool;
        use crate::editor::preview::{BearingVisual, ConstructionVisual, EditorVisuals};
        use crate::editor::state::EditorState;
        use crate::simulation::state::AppSimulation;
        use bevy::prelude::*;

        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([2, 2, 2], BuildPose::default()).unwrap(),
            ))
            .unwrap();
        let material = graph
            .parts()
            .find_map(|(_, spec)| crate::render::mesh::construction::ordinary_material(*spec))
            .unwrap();
        let publish = |graph: &ConstructionGraph| {
            let creation = graph
                .compile_with_static_parts(graph.parts().map(|(part, _)| part))
                .unwrap();
            assert!(
                creation.compounds.iter().all(|compound| compound.is_static),
                "an anchored construction publishes only static bodies"
            );
            let transforms = creation
                .compounds
                .iter()
                .map(|compound| GpuTransform {
                    position: compound.root_translation.extend(0.0).to_array(),
                    rotation: compound.root_rotation.to_array(),
                })
                .collect::<Vec<_>>();
            (creation, transforms)
        };
        let (creation, transforms) = publish(&graph);

        let mut meshes = Assets::<Mesh>::default();
        let visuals = EditorVisuals {
            construction_meshes: std::array::from_fn(|_| meshes.add(Cuboid::default())),
            ..Default::default()
        };
        let block_mesh =
            visuals.construction_meshes[crate::render::materials::material_index(material)].clone();
        let mut app = App::new();
        let block_visual = app
            .world_mut()
            .spawn((ConstructionVisual(material), Visibility::Hidden))
            .id();
        app.insert_resource(AppSimulation {
            creation: Some(creation),
            published_graph: graph.clone(),
            transforms,
            static_mesh_dirty: true,
            ..Default::default()
        })
        .insert_resource(visuals)
        .insert_resource(meshes)
        .init_resource::<EditorState>()
        .init_resource::<SelectedTool>()
        .init_resource::<DriveSequencer>()
        .add_systems(
            Update,
            |mut simulation: ResMut<AppSimulation>,
             state: Res<EditorState>,
             selection: Res<SelectedTool>,
             sequencer: Res<DriveSequencer>,
             visuals: Res<EditorVisuals>,
             mut meshes: ResMut<Assets<Mesh>>,
             mut construction_visuals: Query<
                (&ConstructionVisual, &mut Visibility),
                Without<BearingVisual>,
            >| {
                let published = simulation.published_graph.clone();
                crate::simulation::visuals::refresh_published_construction_visuals(
                    &mut simulation,
                    &published,
                    &state,
                    *selection,
                    &sequencer,
                    &visuals,
                    &mut meshes,
                    &mut construction_visuals,
                );
            },
        );

        let vertices = |app: &App| {
            app.world()
                .resource::<Assets<Mesh>>()
                .get(&block_mesh)
                .unwrap()
                .count_vertices()
        };
        app.update();
        let first = vertices(&app);
        assert!(first > 0, "the published block is drawn");
        assert_eq!(
            *app.world().get::<Visibility>(block_visual).unwrap(),
            Visibility::Visible
        );
        assert!(!app.world().resource::<AppSimulation>().static_mesh_dirty);

        // A second anchored block, published exactly as a ground placement is.
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [2, 2, 2],
                    BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap();
        let (creation, transforms) = publish(&graph);
        {
            let mut simulation = app.world_mut().resource_mut::<AppSimulation>();
            simulation.creation = Some(creation);
            simulation.published_graph = graph.clone();
            simulation.transforms = transforms;
            simulation.static_mesh_dirty = true;
        }
        app.update();
        assert!(
            vertices(&app) > first,
            "the newly published static block joins the shared mesh"
        );
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "keep the preview regression and its ECS fixture together"
    )]
    fn live_static_meshes_preview_an_uncommitted_feature_drag() {
        use super::SelectedTool;
        use crate::editor::preview::{BearingVisual, ConstructionVisual, EditorVisuals};
        use crate::editor::state::EditorState;
        use crate::simulation::state::AppSimulation;
        use bevy::prelude::*;

        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([4, 4, 4], BuildPose::default()).unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let material =
            crate::render::mesh::construction::ordinary_material(*graph.part(part).unwrap())
                .unwrap();
        let creation = graph
            .compile_with_static_parts(graph.parts().map(|(part, _)| part))
            .unwrap();
        let transforms = creation
            .compounds
            .iter()
            .map(|compound| GpuTransform {
                position: compound.root_translation.extend(0.0).to_array(),
                rotation: compound.root_rotation.to_array(),
            })
            .collect::<Vec<_>>();
        let owner = SolidOwner::Part(part);
        let target = EdgeChainRef {
            owner,
            edge: graph.evaluated_solid(owner).unwrap().logical_edges[0].key,
        };

        let mut meshes = Assets::<Mesh>::default();
        let visuals = EditorVisuals {
            construction_meshes: std::array::from_fn(|_| meshes.add(Cuboid::default())),
            ..Default::default()
        };
        let block_mesh =
            visuals.construction_meshes[crate::render::materials::material_index(material)].clone();
        let mut app = App::new();
        app.world_mut()
            .spawn((ConstructionVisual(material), Visibility::Hidden));
        app.insert_resource(AppSimulation {
            creation: Some(creation),
            published_graph: graph,
            transforms,
            static_mesh_dirty: true,
            ..Default::default()
        })
        .insert_resource(visuals)
        .insert_resource(meshes)
        .init_resource::<EditorState>()
        .init_resource::<SelectedTool>()
        .init_resource::<DriveSequencer>()
        .add_systems(
            Update,
            |mut simulation: ResMut<AppSimulation>,
             state: Res<EditorState>,
             selection: Res<SelectedTool>,
             sequencer: Res<DriveSequencer>,
             visuals: Res<EditorVisuals>,
             mut meshes: ResMut<Assets<Mesh>>,
             mut construction_visuals: Query<
                (&ConstructionVisual, &mut Visibility),
                Without<BearingVisual>,
            >| {
                let published = simulation.published_graph.clone();
                crate::simulation::visuals::refresh_published_construction_visuals(
                    &mut simulation,
                    &published,
                    &state,
                    *selection,
                    &sequencer,
                    &visuals,
                    &mut meshes,
                    &mut construction_visuals,
                );
            },
        );
        let vertices = |app: &App| {
            app.world()
                .resource::<Assets<Mesh>>()
                .get(&block_mesh)
                .unwrap()
                .count_vertices()
        };

        app.update();
        let plain = vertices(&app);
        let hit = crate::shape_tool::FeatureEdgeHit {
            target,
            point: Vec3::ZERO,
            tangent: Vec3::Z,
            bisector: Vec3::X,
            distance: 0.0,
        };
        app.world_mut().resource_mut::<EditorState>().feature_drag =
            Some(crate::shape_tool::FeatureDrag::begin(
                hit,
                vec![target],
                EdgeTreatment::Fillet,
                None,
                20,
                Vec3::Y,
                Vec3::NEG_Y,
            ));
        app.update();
        assert!(
            vertices(&app) > plain,
            "a live world draws the fillet while it is still being dragged"
        );

        app.world_mut().resource_mut::<EditorState>().feature_drag = None;
        app.update();
        assert_eq!(
            vertices(&app),
            plain,
            "a cancelled drag restores the published geometry"
        );
    }

    #[test]
    #[ignore = "requires a real GPU adapter"]
    #[expect(
        clippy::too_many_lines,
        reason = "keep the publication regression and its ECS fixture together"
    )]
    fn grounded_functional_blocks_keep_visuals_during_live_publication() {
        use bevy::prelude::*;

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("real GPU adapter required");
        eprintln!("Visual publication adapter: {:?}", adapter.get_info());
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(controller) = graph
            .apply(BuildCommand::SpawnController(ControllerSpec::new(
                BuildPose::default(),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1; 3],
                    BuildPose::new(IVec3::new(8, 8, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap();
        let creation = graph.compile_with_static_parts([controller]).unwrap();
        let controller_body = creation
            .part_to_compound
            .iter()
            .find(|(part, _)| *part == controller)
            .unwrap()
            .1;
        assert!(creation.compounds[controller_body as usize].is_static);
        assert!(crate::simulation::publication::creation_requires_live_physics(&creation));
        let gpu = mechanic_gpu::GpuPhysics::new_with_config(
            &device,
            &queue,
            &creation,
            super::GpuPhysicsConfig::default(),
        )
        .unwrap();
        let transforms = creation
            .compounds
            .iter()
            .map(|body| GpuTransform {
                position: body.root_translation.extend(0.0).to_array(),
                rotation: body.root_rotation.to_array(),
            })
            .collect();
        let mut app = App::new();
        app.insert_resource(crate::simulation::state::AppSimulation {
            gpu: Some(gpu),
            creation: Some(creation),
            published_graph: graph,
            transforms,
            world_revision: Some((1, 1)),
            ..Default::default()
        })
        .init_resource::<crate::simulation::visuals::SimulationVisualCache>()
        .init_resource::<crate::editor::preview::EditorVisuals>()
        .init_resource::<crate::editor::state::EditorState>()
        .init_resource::<super::world::WorldRuntime>()
        .init_resource::<Assets<Mesh>>()
        .add_systems(
            Update,
            crate::simulation::visuals::sync_simulation_visual_cache,
        );
        let legacy = app
            .world_mut()
            .spawn((
                crate::render::authored::AuthoredPartVisual(AuthoredPart::Controller),
                Visibility::Visible,
            ))
            .id();
        for (revision, failure) in [
            ((1, 1), None),
            ((2, 1), None),
            ((2, 1), Some("CPU tick failed".to_owned())),
        ] {
            {
                let mut simulation = app
                    .world_mut()
                    .resource_mut::<crate::simulation::state::AppSimulation>();
                simulation.world_revision = Some(revision);
                simulation.failure = failure;
                assert!(!should_sync_editor_visual_meshes(
                    true,
                    simulation.gpu.is_some()
                ));
            }
            app.update();
            assert_eq!(
                app.world().get::<Visibility>(legacy),
                Some(&Visibility::Hidden)
            );
            let mut roots = app.world_mut().query::<(
                &crate::simulation::visuals::SimulationBodyVisualRoot,
                &Children,
            )>();
            let children = roots
                .iter(app.world())
                .find(|(root, _)| root.0 == controller_body)
                .expect("grounded controller must have a replacement visual")
                .1;
            assert_eq!(children.len(), 1);
            let mesh = app.world().get::<Mesh3d>(children[0]).unwrap();
            assert!(
                app.world()
                    .resource::<Assets<Mesh>>()
                    .get(&mesh.0)
                    .unwrap()
                    .count_vertices()
                    > 0
            );
        }
    }

    #[test]
    fn dimension_link_toggles_refresh_world_visuals_without_a_physics_edit() {
        let revision = (3, 5);
        let first = Some(DimensionLinkId(7));
        let second = Some(DimensionLinkId(8));
        let mut cache = crate::simulation::visuals::SimulationVisualCache {
            revision: Some(revision),
            active_dimension_link: first,
            ..Default::default()
        };
        assert!(!cache.needs_rebuild(revision, first));
        // Turning off, turning on, and switching links all change the texture
        // without changing the construction or its physics revision.
        for active in [None, first, second, None] {
            assert!(cache.needs_rebuild(revision, active));
            cache.active_dimension_link = active;
            assert!(!cache.needs_rebuild(revision, active));
        }
        assert!(cache.needs_rebuild((4, 5), None));
    }

    #[test]
    fn cached_local_mesh_follows_one_compound_transform() {
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [3, 2, 1],
                    BuildPose::new(IVec3::new(2, 3, 4), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap();
        let creation = graph.compile().unwrap();
        let rotation = Quat::from_rotation_y(0.7);
        let snapshot = GpuTransform {
            position: [10.0, 2.0, -3.0, 0.0],
            rotation: rotation.to_array(),
        };
        let world = combined_simulation_material_mesh(
            &graph,
            &creation,
            &[snapshot],
            SimulationMeshKind::Dynamic,
            ConstructionMaterial::Steel,
        );
        let local = local_simulation_material_mesh(
            &graph,
            &creation,
            &[GpuTransform {
                position: [0.0; 4],
                rotation: Quat::IDENTITY.to_array(),
            }],
            0,
            ConstructionMaterial::Steel,
        );
        let transform = transform_from_gpu(snapshot);
        let transformed = positions(&local)
            .into_iter()
            .map(|position| transform.transform_point(position))
            .collect::<Vec<_>>();
        for (actual, expected) in transformed.into_iter().zip(positions(&world)) {
            assert!(actual.abs_diff_eq(expected, 1.0e-5));
        }
    }

    #[test]
    fn immutable_body_meshes_cull_offscreen_and_follow_motion_and_origin_rebases() {
        use bevy::camera::{
            CameraProjection,
            primitives::Aabb,
            visibility::{VisibilityPlugin, VisibleEntities},
        };
        use bevy::prelude::*;
        use std::any::TypeId;

        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            bevy::asset::AssetPlugin::default(),
            bevy::mesh::MeshPlugin,
            TransformPlugin,
            VisibilityPlugin,
        ));
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([3, 2, 1], BuildPose::default()).unwrap(),
            ))
            .unwrap();
        let creation = graph.compile().unwrap();
        let mesh = local_simulation_material_mesh(
            &graph,
            &creation,
            &[GpuTransform {
                position: [0.0; 4],
                rotation: Quat::IDENTITY.to_array(),
            }],
            0,
            ConstructionMaterial::Steel,
        );
        let original_positions = positions(&mesh);
        let handle = app.world_mut().resource_mut::<Assets<Mesh>>().add(mesh);
        let root = app
            .world_mut()
            .spawn((Transform::IDENTITY, Visibility::Inherited))
            .id();
        let body = app
            .world_mut()
            .spawn((
                crate::simulation::visuals::simulation_body_mesh(
                    handle.clone(),
                    Handle::<StandardMaterial>::default(),
                ),
                ChildOf(root),
            ))
            .id();
        let projection = PerspectiveProjection {
            fov: std::f32::consts::FRAC_PI_2,
            aspect_ratio: 1.0,
            ..default()
        };
        let camera = app
            .world_mut()
            .spawn((
                Camera::default(),
                VisibleEntities::default(),
                projection.compute_frustum(&GlobalTransform::IDENTITY),
            ))
            .id();
        for rebase in [Vec3::ZERO, Vec3::new(1024.0, -512.0, 2048.0)] {
            let camera_pose = GlobalTransform::from_translation(-rebase);
            app.world_mut()
                .entity_mut(camera)
                .insert(projection.compute_frustum(&camera_pose));
            for (position, visible) in [
                (Vec3::new(0.0, 0.0, -5.0), true),
                (Vec3::new(100.0, 0.0, -5.0), false),
                (Vec3::new(0.0, 0.0, 5.0), false),
                (Vec3::new(4.9, 0.0, -5.0), true),
                (Vec3::new(0.0, 0.0, -5.0), true),
            ] {
                app.world_mut().entity_mut(root).insert(Transform {
                    translation: position - rebase,
                    rotation: Quat::from_rotation_y(0.7),
                    ..default()
                });
                app.update();
                assert_eq!(
                    app.world()
                        .get::<VisibleEntities>(camera)
                        .unwrap()
                        .get(TypeId::of::<Mesh3d>())
                        .contains(&body),
                    visible,
                    "position={position} rebase={rebase}"
                );
                assert!(app.world().get::<Aabb>(body).is_some());
                assert_eq!(
                    positions(app.world().resource::<Assets<Mesh>>().get(&handle).unwrap()),
                    original_positions
                );
            }
        }
    }

    #[test]
    fn a_4096_block_sheet_renders_as_one_globally_mapped_cuboid() {
        let blocks = (0..64)
            .flat_map(|x| {
                (0..64).map(move |z| {
                    (
                        0,
                        CuboidSpec::new(
                            [1; 3],
                            BuildPose::from_position_ticks(
                                IVec3::new(x * 100, 50, z * 100),
                                GridRotation::default(),
                            ),
                        )
                        .unwrap(),
                    )
                })
            })
            .collect::<Vec<_>>();
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut tangents = Vec::new();
        let mut colors = Vec::new();
        let mut indices = Vec::new();

        append_merged_block_cuboids(
            &blocks,
            &mut positions,
            &mut normals,
            &mut uvs,
            &mut tangents,
            &mut colors,
            &mut indices,
        );

        assert_eq!(positions.len(), 24);
        assert_eq!(indices.len(), 36);
        assert_eq!(uvs.len(), 24);
        assert_eq!(tangents.len(), 24);
        assert_eq!(colors.len(), 24);
        let (minimum_u, maximum_u) = uvs.iter().flat_map(|uv| uv.iter()).copied().fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(minimum, maximum), value| (minimum.min(value), maximum.max(value)),
        );
        assert!((maximum_u - minimum_u - 16.0 / MATERIAL_TEXTURE_METERS_PER_REPEAT).abs() < 1.0e-4);
    }

    #[test]
    fn a_4096_block_sheet_mesh_publication_stays_within_one_frame() {
        let previous = ConstructionGraph::new();
        let start = PlacementCandidate {
            spec: CuboidSpec::new(
                [1; 3],
                BuildPose::from_position_ticks(
                    IVec3::new(-3_150, 50, -3_150),
                    GridRotation::default(),
                ),
            )
            .unwrap(),
            attached_face: FaceKind::NegativeY,
            anchor: Some(Vec3::ZERO),
            support: PlacementSupport::Surface(FaceOwner::Ground),
        };
        let placed = stage_block_volume_in_bounds(
            &previous,
            &PlacementSnapIndex::default(),
            start,
            BlockVolume::new(start.spec, IVec3::new(63, 0, 63)).unwrap(),
            None,
            Some(FaceOwner::Ground),
            PlacementBounds::Garage,
            1,
        )
        .unwrap();

        let started = Instant::now();
        let delta = mechanic_core::ConstructionEditDelta::between(&previous, &placed.graph);
        let mesh =
            combined_material_construction_mesh(&placed.graph, None, ConstructionMaterial::Steel);
        let elapsed = started.elapsed();

        assert_eq!(delta.added.len(), 4_096);
        assert_eq!(mesh.count_vertices(), 24);
        assert!(
            elapsed.as_secs_f64() <= 0.005,
            "mesh publication took {elapsed:?}"
        );
    }

    use crate::PlacementPlane;
    use crate::builder::{
        BlockVolume, PlacementBounds, PlacementCandidate, PlacementSnapIndex, PlacementSupport,
        block_sheet_specs, stage_block_volume_in_bounds,
    };
    use crate::editor::overlay::{
        OverlayGeometry, append_axis_arrows, append_drag_plane, append_plane_arrows,
        region_world_bounds,
    };
    use crate::editor::shape_actions::{append_feature_pull_arrow, region_focus_is_active};
    use crate::hotbar::Tool;
    use crate::render::environment::{
        EnvironmentMapGenerationReady, retain_generated_environment_map, sky_cubemap,
    };
    use crate::sequencer::DriveSequencer;

    fn positions(mesh: &Mesh) -> Vec<Vec3> {
        let Some(VertexAttributeValues::Float32x3(values)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("mesh must have float3 positions")
        };
        values.iter().copied().map(Vec3::from_array).collect()
    }

    #[test]
    fn only_vertex_mode_uses_region_focus_ghosting() {
        assert!(region_focus_is_active(
            Some(Tool::Shape),
            crate::shape_tool::ShapeEditMode::Vertex,
            true,
        ));
        assert!(!region_focus_is_active(
            Some(Tool::Shape),
            crate::shape_tool::ShapeEditMode::Chamfer,
            true,
        ));
        assert!(!region_focus_is_active(
            Some(Tool::Shape),
            crate::shape_tool::ShapeEditMode::Fillet,
            true,
        ));
        assert!(!region_focus_is_active(
            Some(Tool::Chroma),
            crate::shape_tool::ShapeEditMode::Vertex,
            true,
        ));
        assert!(!region_focus_is_active(
            Some(Tool::Shape),
            crate::shape_tool::ShapeEditMode::Vertex,
            false,
        ));
    }

    #[test]
    fn only_the_active_dimension_link_uses_the_enabled_batch() {
        let mut graph = ConstructionGraph::new();
        let spec = DimensionLinkSpec::new(DimensionLinkId(7), BuildPose::default());
        let BuildOutcome::Spawned(part) =
            graph.apply(BuildCommand::SpawnDimensionLink(spec)).unwrap()
        else {
            unreachable!()
        };
        let part_spec = *graph.part(part).unwrap();
        assert!(AuthoredPart::DimensionLinkDisabled.matches(&graph, part, part_spec, None));
        assert!(!AuthoredPart::DimensionLinkEnabled.matches(&graph, part, part_spec, None));
        assert!(AuthoredPart::DimensionLinkEnabled.matches(
            &graph,
            part,
            part_spec,
            Some(DimensionLinkId(7))
        ));
        assert!(!AuthoredPart::DimensionLinkDisabled.matches(
            &graph,
            part,
            part_spec,
            Some(DimensionLinkId(7))
        ));
    }

    /// A region offset from the origin, so a centred overlay cannot pass by
    /// sitting at zero.
    fn offset_region() -> mechanic_core::ShapeRegion {
        mechanic_core::ShapeRegion::new(
            IVec3::new(2, 4, 0),
            IVec3::new(3, 2, 1),
            ConstructionMaterial::Steel,
        )
        .unwrap()
    }

    fn overlay_bounds(geometry: &OverlayGeometry) -> (Vec3, Vec3) {
        geometry.positions.iter().fold(
            (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
            |(low, high), position| {
                let at = Vec3::from_array(*position);
                (low.min(at), high.max(at))
            },
        )
    }

    #[test]
    fn the_drag_plane_sits_in_the_middle_of_the_area() {
        let region = offset_region();
        // The area runs y = 0.50 m to 1.00 m, so its middle is 0.75 m.
        for (plane, normal_axis, middle) in [
            (PlacementPlane::Xz, 1, 0.75),
            (PlacementPlane::Xy, 2, 0.125),
            (PlacementPlane::Yz, 0, 0.625),
        ] {
            let (low, high) = region_world_bounds(&region);
            let mut geometry = OverlayGeometry::default();
            append_drag_plane(low, high, plane, &mut geometry);
            let (low, high) = overlay_bounds(&geometry);
            assert!(
                (f32::midpoint(low[normal_axis], high[normal_axis]) - middle).abs() < 1.0e-5,
                "{plane:?} sheet is centred on the area"
            );
            assert!(
                high[normal_axis] - low[normal_axis] < 0.01,
                "{plane:?} sheet is a sheet, not a slab"
            );
        }
    }

    #[test]
    fn the_drag_plane_points_along_both_of_its_axes() {
        let region = offset_region();
        for plane in [PlacementPlane::Xy, PlacementPlane::Xz, PlacementPlane::Yz] {
            let (low, high) = region_world_bounds(&region);
            let mut geometry = OverlayGeometry::default();
            append_plane_arrows(low, high, plane, &mut geometry);
            let (low, high) = overlay_bounds(&geometry);
            let centre = (
                f32::midpoint(0.25, 1.0),
                f32::midpoint(0.5, 1.0),
                f32::midpoint(0.0, 0.25),
            );
            let centre = Vec3::new(centre.0, centre.1, centre.2);
            for axis in plane.tangent_axes() {
                assert!(
                    low[axis] < centre[axis] - 0.1 && high[axis] > centre[axis] + 0.1,
                    "{plane:?} reaches out along axis {axis} in both directions"
                );
            }
            let normal_axis = plane.normal_axis();
            assert!(
                high[normal_axis] - low[normal_axis] < 0.02,
                "{plane:?} arrows lie flat on the plane"
            );
            assert!(
                (f32::midpoint(low[normal_axis], high[normal_axis]) - centre[normal_axis]).abs()
                    < 1.0e-5,
                "{plane:?} arrows are centred on the area with the sheet"
            );
        }
    }

    #[test]
    fn the_vertex_axis_guide_points_only_along_its_active_axis() {
        let at = Vec3::new(0.25, 0.5, 0.75);
        for axis in 0..3 {
            let mut geometry = OverlayGeometry::default();
            append_axis_arrows(at, axis, &mut geometry);
            let (low, high) = overlay_bounds(&geometry);
            assert!(low[axis] < at[axis] - 0.17);
            assert!(high[axis] > at[axis] + 0.17);
            for other in [0, 1, 2].into_iter().filter(|&other| other != axis) {
                assert!(
                    low[other] > at[other] - 0.03 && high[other] < at[other] + 0.03,
                    "axis {axis} guide must stay narrow on axis {other}"
                );
            }
        }
    }

    #[test]
    fn the_feature_guide_points_inward_along_the_cross_section_bisector() {
        let at = Vec3::new(0.25, 0.5, 0.75);
        let direction = Vec3::new(-1.0, -1.0, 0.0).normalize();
        let mut geometry = OverlayGeometry::default();
        append_feature_pull_arrow(at, direction, &mut geometry);
        let (minimum, maximum) = geometry.positions.iter().fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(minimum, maximum), position| {
                let along = (Vec3::from_array(*position) - at).dot(direction);
                (minimum.min(along), maximum.max(along))
            },
        );

        assert!(minimum < -0.15, "the shaft stays visible outside the edge");
        assert!(maximum > 0.014, "the head points inward through the edge");
    }

    #[test]
    fn the_source_sky_map_is_a_cube_of_the_requested_size() {
        let map = sky_cubemap(8);
        assert_eq!(map.texture_descriptor.size.depth_or_array_layers, 6);
        assert_eq!(map.texture_descriptor.size.width, 8);
        assert_eq!(map.texture_descriptor.mip_level_count, 1);
    }

    #[test]
    fn completed_generation_keeps_the_filtered_map_and_removes_its_generator() {
        let ready = EnvironmentMapGenerationReady::default();
        ready.0.store(true, std::sync::atomic::Ordering::Release);
        let mut app = App::new();
        app.insert_resource(ready)
            .add_systems(Update, retain_generated_environment_map);
        let entity = app
            .world_mut()
            .spawn((
                GeneratedEnvironmentMapLight::default(),
                EnvironmentMapLight::default(),
            ))
            .id();

        app.update();

        let entity = app.world().entity(entity);
        assert!(!entity.contains::<GeneratedEnvironmentMapLight>());
        assert!(entity.contains::<EnvironmentMapLight>());
    }

    #[test]
    fn ordinary_construction_uses_one_textured_batch_per_material() {
        let mut graph = ConstructionGraph::new();
        for (index, material) in ConstructionMaterial::ALL.into_iter().enumerate() {
            let x = i32::try_from(index).unwrap() * 12;
            graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(
                        [4; 3],
                        BuildPose::new(IVec3::new(x, 2, 0), GridRotation::default()),
                    )
                    .unwrap()
                    .with_material(material)
                    .with_appearance(MaterialAppearance::new(
                        MaterialColor::Dye(MaterialDye::new([42, 76, 199], 1.2).unwrap()),
                        MaterialFinish::Painted,
                    )),
                ))
                .unwrap();
            graph
                .apply(BuildCommand::SpawnCylinder(
                    CylinderSpec::new(
                        CylinderDimensions::new(1.0, 0.0, 1.0).unwrap(),
                        BuildPose::new(IVec3::new(x, 2, 8), GridRotation::default()),
                    )
                    .with_material(material)
                    .with_appearance(MaterialAppearance::new(
                        MaterialColor::Shift(MaterialShift::new(45.0, 1.3, 0.9).unwrap()),
                        MaterialFinish::Anodised,
                    )),
                ))
                .unwrap();
        }

        let creation = graph.compile().unwrap();
        let transforms = creation
            .compounds
            .iter()
            .map(|compound| GpuTransform {
                position: [
                    compound.root_translation.x,
                    compound.root_translation.y,
                    compound.root_translation.z,
                    0.0,
                ],
                rotation: compound.root_rotation.to_array(),
            })
            .collect::<Vec<_>>();
        for material in ConstructionMaterial::ALL {
            let build = combined_material_construction_mesh(&graph, None, material);
            let simulated = combined_simulation_material_mesh(
                &graph,
                &creation,
                &transforms,
                SimulationMeshKind::Dynamic,
                material,
            );
            assert!(build.count_vertices() > 24);
            assert_eq!(simulated.count_vertices(), build.count_vertices());
            assert_eq!(
                build.attribute(Mesh::ATTRIBUTE_COLOR).unwrap().len(),
                build.count_vertices(),
            );
            assert_eq!(
                simulated.attribute(Mesh::ATTRIBUTE_COLOR),
                build.attribute(Mesh::ATTRIBUTE_COLOR),
                "editor and simulation encode identical appearance payloads",
            );
            assert_eq!(
                build.attribute(Mesh::ATTRIBUTE_UV_0).unwrap().len(),
                build.count_vertices(),
            );
            assert_eq!(
                build.attribute(Mesh::ATTRIBUTE_TANGENT).unwrap().len(),
                build.count_vertices(),
            );
        }
    }

    #[test]
    fn cuboid_uvs_give_each_quarter_metre_block_512_texture_pixels() {
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [4; 3],
                    BuildPose::new(IVec3::new(40, 2, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap();
        let mesh = combined_material_construction_mesh(&graph, None, ConstructionMaterial::Steel);
        let positions = positions(&mesh);
        let Some(VertexAttributeValues::Float32x2(uvs)) = mesh.attribute(Mesh::ATTRIBUTE_UV_0)
        else {
            panic!("material UVs use Float32x2")
        };
        assert!((uvs[1][0] - positions[1].x / MATERIAL_TEXTURE_METERS_PER_REPEAT).abs() < 1.0e-6);
        assert!((uvs[0][0] - positions[0].x / MATERIAL_TEXTURE_METERS_PER_REPEAT).abs() < 1.0e-6);
        let expected_span =
            4.0 * MATERIAL_TEXTURE_PIXELS_PER_BLOCK / MATERIAL_TEXTURE_PIXELS_PER_SIDE;
        assert!(((uvs[1][0] - uvs[0][0]).abs() - expected_span).abs() < 1.0e-6);
    }

    #[test]
    fn material_maps_use_repeat_sampling_and_explicit_color_spaces() {
        let mut base_color = ImageLoaderSettings::default();
        configure_repeating_texture(&mut base_color, true);
        assert!(base_color.is_srgb);
        let base_sampler = base_color.sampler.get_or_init_descriptor();
        assert_eq!(base_sampler.address_mode_u, ImageAddressMode::Repeat);
        assert_eq!(base_sampler.address_mode_v, ImageAddressMode::Repeat);

        let mut data_map = ImageLoaderSettings::default();
        configure_repeating_texture(&mut data_map, false);
        assert!(!data_map.is_srgb);
        let data_sampler = data_map.sampler.get_or_init_descriptor();
        assert_eq!(data_sampler.address_mode_u, ImageAddressMode::Repeat);
        assert_eq!(data_sampler.address_mode_v, ImageAddressMode::Repeat);
    }

    #[test]
    fn only_copper_and_dirt_route_material_tint_masks() {
        for material in ConstructionMaterial::ALL {
            let path = construction_tint_mask_path(material);
            match material {
                ConstructionMaterial::Copper => {
                    assert_eq!(path, Some("materials/copper/copper_tint.png"));
                }
                ConstructionMaterial::Dirt => {
                    assert_eq!(path, Some("materials/dirt/dirt_tint.png"));
                }
                _ => assert_eq!(path, None),
            }
        }
    }

    #[test]
    fn authored_maps_clamp_and_use_their_declared_color_spaces() {
        let mut color_map = ImageLoaderSettings::default();
        configure_authored_texture(&mut color_map, true);
        assert!(color_map.is_srgb);
        let color_sampler = color_map.sampler.get_or_init_descriptor();
        assert_eq!(color_sampler.address_mode_u, ImageAddressMode::ClampToEdge);
        assert_eq!(color_sampler.address_mode_v, ImageAddressMode::ClampToEdge);
        assert_eq!(color_sampler.mag_filter, ImageFilterMode::Linear);
        assert_eq!(color_sampler.min_filter, ImageFilterMode::Linear);
        assert_eq!(color_sampler.mipmap_filter, ImageFilterMode::Linear);

        let mut data_map = ImageLoaderSettings::default();
        configure_authored_texture(&mut data_map, false);
        assert!(!data_map.is_srgb);
        let data_sampler = data_map.sampler.get_or_init_descriptor();
        assert_eq!(data_sampler.address_mode_u, ImageAddressMode::ClampToEdge);
        assert_eq!(data_sampler.address_mode_v, ImageAddressMode::ClampToEdge);
    }

    #[test]
    fn bearing_mesh_insets_radial_surfaces_without_changing_depth() {
        let anchor = Vec3::new(2.0, 3.0, 4.0);
        let axis = Vec3::X;
        let dimensions = BearingDimensions::new(0.80, 0.30).unwrap();
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut tangents = Vec::new();
        let mut indices = Vec::new();
        append_bearing_cylinder(
            anchor,
            axis,
            dimensions,
            &mut positions,
            &mut normals,
            &mut uvs,
            &mut tangents,
            &mut indices,
        );

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
        let minimum_radius = offsets
            .iter()
            .map(|offset| (*offset - axis * offset.dot(axis)).length())
            .fold(f32::INFINITY, f32::min);

        assert!((minimum_depth + BEARING_DEPTH * 0.5).abs() < 1.0e-6);
        assert!((maximum_depth - BEARING_DEPTH * 0.5).abs() < 1.0e-6);
        assert!(
            (maximum_radius - (dimensions.outer_diameter() * 0.5 - BEARING_RENDER_RADIAL_SKIN))
                .abs()
                < 1.0e-6
        );
        assert!(
            (minimum_radius - (dimensions.inner_diameter() * 0.5 + BEARING_RENDER_RADIAL_SKIN))
                .abs()
                < 1.0e-6
        );
    }

    #[test]
    fn bearing_material_biases_coplanar_surfaces_toward_the_camera() {
        let material = bearing_pbr_material(
            Handle::<Image>::default(),
            Handle::<Image>::default(),
            Handle::<Image>::default(),
        );
        assert!(material.depth_bias > 0.0);
    }

    #[test]
    fn bearing_maps_repeat_around_the_ring_and_clamp_across_the_profile() {
        let mut settings = ImageLoaderSettings::default();
        configure_bearing_texture(&mut settings, false);
        assert!(!settings.is_srgb);
        let sampler = settings.sampler.get_or_init_descriptor();
        assert_eq!(sampler.address_mode_u, ImageAddressMode::Repeat);
        assert_eq!(sampler.address_mode_v, ImageAddressMode::ClampToEdge);
        assert_eq!(sampler.mag_filter, ImageFilterMode::Linear);
        assert_eq!(sampler.min_filter, ImageFilterMode::Linear);
        assert_eq!(sampler.mipmap_filter, ImageFilterMode::Linear);
        assert_eq!(sampler.anisotropy_clamp, 8);
    }

    #[test]
    fn bearing_profile_fit_rule_matches_the_authored_sanity_sizes() {
        let minimum_wall = bearing_profile_plan(0.050, 0.025);
        assert_eq!(minimum_wall.steps, 1);
        assert!((minimum_wall.terrace_meters - 0.006).abs() < 1.0e-6);
        assert!((minimum_wall.relief_meters - 0.005).abs() < 1.0e-6);
        assert_eq!(minimum_wall.turns, 1);

        let common_ring = bearing_profile_plan(0.120, 0.050);
        assert_eq!(common_ring.steps, 4);
        assert!((common_ring.terrace_meters - 0.007).abs() < 1.0e-6);
        assert!((common_ring.relief_meters - 0.007).abs() < 1.0e-6);
        assert_eq!(common_ring.turns, 1);

        let wide_solid = bearing_profile_plan(0.200, 0.0);
        assert_eq!(wide_solid.steps, 4);
        assert!((wide_solid.terrace_meters - 0.0265).abs() < 1.0e-6);
        assert!((wide_solid.relief_meters - 0.020).abs() < 1.0e-6);
        assert_eq!(wide_solid.turns, 3);
    }

    #[test]
    fn bearing_mesh_carries_profile_uvs_and_normal_map_tangents() {
        let dimensions = BearingDimensions::default();
        let mesh = single_bearing_mesh(dimensions);
        let Some(VertexAttributeValues::Float32x2(uvs)) = mesh.attribute(Mesh::ATTRIBUTE_UV_0)
        else {
            panic!("bearing UVs use Float32x2")
        };
        let Some(VertexAttributeValues::Float32x4(tangents)) =
            mesh.attribute(Mesh::ATTRIBUTE_TANGENT)
        else {
            panic!("bearing tangents use Float32x4")
        };
        assert_eq!(uvs.len(), mesh.count_vertices());
        assert_eq!(tangents.len(), mesh.count_vertices());

        let outer_radius = dimensions.outer_diameter() * 0.5 - BEARING_RENDER_RADIAL_SKIN;
        let repeat = bearing_u_repeat(outer_radius);
        let seam = usize::from(crate::render::mesh::bearing::BEARING_SEGMENTS);
        assert!(uvs[0][0].abs() < f32::EPSILON);
        assert!((uvs[0][1] - 0.8125).abs() < f32::EPSILON);
        assert!((uvs[seam][0] - repeat).abs() < f32::EPSILON);
        assert!((uvs[0][1] - uvs[seam][1]).abs() < f32::EPSILON);
        assert!(uvs.iter().any(|uv| uv[1].abs() < f32::EPSILON));
        assert!(uvs.iter().any(|uv| (uv[1] - 1.0).abs() < f32::EPSILON));
    }

    #[test]
    fn every_preview_material_biases_coplanar_surfaces_toward_the_camera() {
        for color in [
            Color::srgba(1.0, 1.0, 1.0, 0.34),
            Color::srgba(1.0, 0.06, 0.04, 0.46),
            Color::srgba(0.12, 1.0, 0.28, 0.52),
        ] {
            assert!(preview_material(color).depth_bias > 0.0);
        }
    }

    #[test]
    fn delete_preview_surrounds_every_selected_block_face() {
        let spec = CuboidSpec::new([2, 4, 6], BuildPose::default()).unwrap();
        let half_extents = spec.size_meters() * 0.5;
        let vertices = positions(&delete_preview_mesh(&[PartSpec::Cuboid(spec)]));
        let minimum = vertices
            .iter()
            .copied()
            .fold(Vec3::splat(f32::INFINITY), Vec3::min);
        let maximum = vertices
            .iter()
            .copied()
            .fold(Vec3::splat(f32::NEG_INFINITY), Vec3::max);

        assert!(minimum.cmplt(-half_extents).all());
        assert!(maximum.cmpgt(half_extents).all());
    }

    #[test]
    fn block_sheet_preview_is_one_cuboid_inset_from_every_logical_contact_plane() {
        let start = CuboidSpec::new(
            [1; 3],
            BuildPose::from_half_grid(IVec3::ONE, GridRotation::default()),
        )
        .unwrap();
        let endpoint = start.pose.translation_half_units() + IVec3::new(10, 0, -6);
        let specs = block_sheet_specs(start, endpoint, PlacementPlane::Xz).unwrap();
        let expected = block_sheet_bounds(&specs).unwrap();
        let mesh = block_sheet_preview_mesh(&specs);
        let vertices = positions(&mesh);
        let actual_minimum = vertices
            .iter()
            .copied()
            .fold(Vec3::splat(f32::INFINITY), Vec3::min);
        let actual_maximum = vertices
            .iter()
            .copied()
            .fold(Vec3::splat(f32::NEG_INFINITY), Vec3::max);

        assert_eq!(vertices.len(), 24);
        assert_eq!(mesh.indices().unwrap().len(), 36);
        let inset = Vec3::splat(BLOCK_SHEET_PREVIEW_INSET_METERS);
        assert!(actual_minimum.abs_diff_eq(expected.0 + inset, 1.0e-6));
        assert!(actual_maximum.abs_diff_eq(expected.1 - inset, 1.0e-6));
    }

    #[test]
    fn solid_bearing_remains_closed_when_its_outer_radius_is_inset() {
        let mesh = single_bearing_mesh(BearingDimensions::new(0.4, 0.0).unwrap());
        assert!(positions(&mesh).iter().any(|position| {
            position.x.hypot(position.z) < f32::EPSILON && position.y.abs() <= BEARING_DEPTH * 0.5
        }));
    }

    #[test]
    fn cylinder_mesh_uses_exact_radii_and_variable_axial_length() {
        let mesh = single_cylinder_mesh(CylinderDimensions::new(1.2, 0.4, 2.0).unwrap());
        let positions = positions(&mesh);
        let maximum_radius = positions
            .iter()
            .map(|position| position.x.hypot(position.z))
            .fold(0.0_f32, f32::max);
        let maximum_y = positions
            .iter()
            .map(|position| position.y.abs())
            .fold(0.0_f32, f32::max);
        assert!((maximum_radius - 0.6).abs() < 1.0e-5);
        assert!((maximum_y - 1.0).abs() < 1.0e-5);
        assert!(
            positions
                .iter()
                .any(|position| { (position.x.hypot(position.z) - 0.2).abs() < 1.0e-5 })
        );
    }

    #[test]
    fn cylinder_sector_mesh_has_cut_walls_and_outward_winding() {
        let dimensions = CylinderDimensions::new(1.0, 0.5, 1.0)
            .unwrap()
            .with_sweep_angle_degrees(90)
            .unwrap();
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut indices = Vec::new();
        append_cylinder_shape(
            Vec3::ZERO,
            Quat::IDENTITY,
            dimensions,
            1.0,
            &mut positions,
            &mut normals,
            &mut indices,
        );

        assert!(positions.iter().all(|position| position[0] >= -1.0e-6));
        assert!(normals.iter().any(|normal| normal[1].abs() < 1.0e-6
            && normal[0].abs() > 0.5
            && normal[2].abs() > 0.5));
        for triangle in indices.chunks_exact(3) {
            let a = Vec3::from_array(positions[triangle[0] as usize]);
            let b = Vec3::from_array(positions[triangle[1] as usize]);
            let c = Vec3::from_array(positions[triangle[2] as usize]);
            let geometric_normal = (b - a).cross(c - a);
            let expected_normal = triangle
                .iter()
                .map(|&index| Vec3::from_array(normals[index as usize]))
                .sum::<Vec3>();
            assert!(geometric_normal.dot(expected_normal) > 0.0);
        }
    }

    #[test]
    fn pipe_bend_mesh_has_exact_bounds_bore_and_outward_winding() {
        let dimensions = PipeBendDimensions::new(0.25, 0.10, 2).unwrap();
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut indices = Vec::new();
        append_pipe_bend_shape(
            Vec3::ZERO,
            Quat::IDENTITY,
            dimensions,
            1.0,
            &mut positions,
            &mut normals,
            &mut indices,
        );
        let minimum = positions
            .iter()
            .map(|position| Vec3::from_array(*position))
            .fold(Vec3::splat(f32::INFINITY), Vec3::min);
        let maximum = positions
            .iter()
            .map(|position| Vec3::from_array(*position))
            .fold(Vec3::splat(f32::NEG_INFINITY), Vec3::max);
        assert!(minimum.abs_diff_eq(Vec3::new(-0.375, -0.125, -0.125), 1.0e-5));
        assert!(maximum.abs_diff_eq(Vec3::new(0.125, 0.375, 0.125), 1.0e-5));
        assert!(positions.iter().any(|position| {
            let point = Vec3::from_array(*position);
            let from_curve_center = Vec2::new(point.x + 0.375, point.y - 0.375).length();
            ((from_curve_center - 0.375).hypot(point.z) - 0.05).abs() < 1.0e-5
        }));
        for (triangle_index, triangle) in indices.chunks_exact(3).enumerate() {
            let a = Vec3::from_array(positions[triangle[0] as usize]);
            let b = Vec3::from_array(positions[triangle[1] as usize]);
            let c = Vec3::from_array(positions[triangle[2] as usize]);
            let geometric = (b - a).cross(c - a);
            let expected = triangle
                .iter()
                .map(|&index| Vec3::from_array(normals[index as usize]))
                .sum::<Vec3>();
            assert!(
                geometric.dot(expected) > 0.0,
                "triangle {triangle_index} has reversed winding"
            );
        }
    }

    #[test]
    fn pipe_bend_curved_vertices_share_analytic_smooth_normals() {
        let dimensions = PipeBendDimensions::new(0.25, 0.10, 2).unwrap();
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut indices = Vec::new();
        append_pipe_bend_shape(
            Vec3::ZERO,
            Quat::IDENTITY,
            dimensions,
            1.0,
            &mut positions,
            &mut normals,
            &mut indices,
        );

        let theta = -std::f32::consts::FRAC_PI_2 + std::f32::consts::FRAC_PI_2 / 12.0;
        let expected_normal = Vec3::new(theta.cos(), theta.sin(), 0.0);
        let expected_position = Vec3::new(-0.375, 0.375, 0.0) + expected_normal * (0.375 + 0.125);
        let seam_normals = positions
            .iter()
            .zip(&normals)
            .filter_map(|(&position, &normal)| {
                Vec3::from_array(position)
                    .abs_diff_eq(expected_position, 1.0e-5)
                    .then_some(Vec3::from_array(normal))
            })
            .collect::<Vec<_>>();

        assert!(seam_normals.len() >= 4);
        assert!(
            seam_normals
                .iter()
                .all(|normal| normal.abs_diff_eq(expected_normal, 1.0e-5))
        );
    }

    #[test]
    fn cuboid_texture_stays_attached_to_its_authored_grid_during_motion() {
        let spec = PartSpec::Cuboid(CuboidSpec::new([2, 3, 1], BuildPose::default()).unwrap());
        let make = |translation, rotation| {
            let mut positions = Vec::new();
            let mut normals = Vec::new();
            let mut uvs = Vec::new();
            let mut tangents = Vec::new();
            let mut indices = Vec::new();
            crate::render::mesh::construction::append_textured_part(
                spec,
                translation,
                rotation,
                crate::render::mesh::construction::BuildTransform::IDENTITY,
                crate::render::mesh::pipe::PipeTextureOffset::default(),
                crate::render::mesh::pipe::PipeEndFaces::ALL,
                &mut positions,
                &mut normals,
                &mut uvs,
                &mut tangents,
                &mut indices,
            );
            (uvs, tangents)
        };
        let rotation = Quat::from_euler(bevy::math::EulerRot::YXZ, 0.73, -0.4, 0.2);
        let (original_uvs, original_tangents) = make(Vec3::ZERO, Quat::IDENTITY);
        let (moved_uvs, moved_tangents) = make(Vec3::new(8.0, 2.0, -3.0), rotation);
        for (original, moved) in original_uvs.iter().zip(moved_uvs) {
            assert!((original[0] - moved[0]).abs() < 1.0e-5);
            assert!((original[1] - moved[1]).abs() < 1.0e-5);
        }
        for (original, moved) in original_tangents.iter().zip(moved_tangents) {
            let expected = rotation * Vec3::new(original[0], original[1], original[2]);
            assert!(expected.abs_diff_eq(Vec3::new(moved[0], moved[1], moved[2]), 1.0e-5));
        }
    }

    #[test]
    fn pipe_uvs_keep_texture_u_lengthwise_through_straights_and_bends() {
        let cylinder_dimensions = CylinderDimensions::new(0.25, 0.10, 1.0).unwrap();
        let mut straight_positions = Vec::new();
        let mut straight_normals = Vec::new();
        let mut straight_uvs = Vec::new();
        let mut straight_tangents = Vec::new();
        let mut straight_indices = Vec::new();
        crate::render::mesh::construction::append_textured_part(
            PartSpec::Cylinder(CylinderSpec::new(cylinder_dimensions, BuildPose::default())),
            Vec3::ZERO,
            Quat::IDENTITY,
            crate::render::mesh::construction::BuildTransform::IDENTITY,
            crate::render::mesh::pipe::PipeTextureOffset::default(),
            crate::render::mesh::pipe::PipeEndFaces::ALL,
            &mut straight_positions,
            &mut straight_normals,
            &mut straight_uvs,
            &mut straight_tangents,
            &mut straight_indices,
        );
        let straight_length =
            cylinder_dimensions.axial_length() / MATERIAL_TEXTURE_METERS_PER_REPEAT;
        assert!((straight_uvs[1][0] - straight_uvs[0][0] - straight_length).abs() < 1.0e-6);
        assert!((straight_uvs[1][1] - straight_uvs[0][1]).abs() < 1.0e-6);
        assert!(
            Vec3::from_array(straight_tangents[0][..3].try_into().unwrap())
                .abs_diff_eq(Vec3::Y, 1.0e-6)
        );
        let straight_circumference_step =
            std::f32::consts::TAU / 24.0 * cylinder_dimensions.outer_diameter() * 0.5
                / MATERIAL_TEXTURE_METERS_PER_REPEAT;
        for segment in 0..24 {
            let current = segment * 2;
            let next = (segment + 1) * 2;
            assert!(
                ((straight_uvs[next][1] - straight_uvs[current][1]).abs()
                    - straight_circumference_step)
                    .abs()
                    < 1.0e-6
            );
        }

        let dimensions = PipeBendDimensions::new(0.25, 0.10, 2).unwrap();
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut indices = Vec::new();
        append_pipe_bend_shape(
            Vec3::ZERO,
            Quat::IDENTITY,
            dimensions,
            1.0,
            &mut positions,
            &mut normals,
            &mut indices,
        );
        let mut uvs = Vec::new();
        let mut tangents = Vec::new();
        append_pipe_bend_texture_coordinates(
            dimensions,
            Vec3::ZERO,
            Quat::IDENTITY,
            0,
            &positions,
            &normals,
            0.0,
            &mut uvs,
            &mut tangents,
        );

        let arc_step = std::f32::consts::FRAC_PI_2 / 12.0 * dimensions.radius()
            / MATERIAL_TEXTURE_METERS_PER_REPEAT;
        let circumference_step = std::f32::consts::TAU / 24.0 * dimensions.outer_diameter() * 0.5
            / MATERIAL_TEXTURE_METERS_PER_REPEAT;
        assert!((uvs[1][0] - uvs[0][0] - arc_step).abs() < 1.0e-6);
        assert!((uvs[1][1] - uvs[0][1]).abs() < 1.0e-6);
        assert!((uvs[2][0] - uvs[1][0]).abs() < 1.0e-6);
        assert!((uvs[2][1] - uvs[1][1] - circumference_step).abs() < 1.0e-6);

        let tangent = Vec3::from_array(tangents[0][..3].try_into().unwrap());
        assert!(tangent.abs_diff_eq(Vec3::X, 1.0e-6));

        let curved_vertex_count = 12 * 24 * 8;
        let inner_circumference_step =
            std::f32::consts::TAU / 24.0 * dimensions.inner_diameter() * 0.5
                / MATERIAL_TEXTURE_METERS_PER_REPEAT;
        for (quad_uvs, quad_tangents) in uvs[..curved_vertex_count]
            .chunks_exact(8)
            .zip(tangents[..curved_vertex_count].chunks_exact(8))
        {
            assert!((quad_uvs[2][1] - quad_uvs[0][1] - circumference_step).abs() < 1.0e-6);
            assert!((quad_uvs[4][1] - quad_uvs[6][1] - inner_circumference_step).abs() < 1.0e-6);
            assert!(
                quad_tangents[..4]
                    .iter()
                    .all(|tangent| (tangent[3] + 1.0).abs() < f32::EPSILON)
            );
            assert!(
                quad_tangents[4..]
                    .iter()
                    .all(|tangent| (tangent[3] - 1.0).abs() < f32::EPSILON)
            );
        }
    }

    fn assert_pipe_circumference_phase(
        graph: &ConstructionGraph,
        part: mechanic_core::PartId,
        spec: PartSpec,
        offset: crate::render::mesh::pipe::PipeTextureOffset,
    ) {
        let face = FaceKind::PositiveY;
        let frame = crate::render::mesh::pipe::pipe_endpoint_texture_frame(spec, face).unwrap();
        let local_angle = std::f32::consts::FRAC_PI_2 - offset.v_angle;
        let radial = frame.radial_zero * local_angle.cos()
            - frame.direction.cross(frame.radial_zero) * local_angle.sin();
        let endpoint =
            crate::builder::face_geometry_from_ref(FaceRef::part(part, face), Some(graph));
        let target = endpoint.center + radial * 0.125;
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut tangents = Vec::new();
        let mut indices = Vec::new();
        crate::render::mesh::construction::append_textured_part(
            spec,
            spec.pose().translation(),
            spec.pose().rotation.quaternion(),
            crate::render::mesh::construction::BuildTransform::IDENTITY,
            offset,
            crate::render::mesh::pipe::PipeEndFaces {
                inlet: false,
                outlet: false,
            },
            &mut positions,
            &mut normals,
            &mut uvs,
            &mut tangents,
            &mut indices,
        );
        let matching_v = positions
            .iter()
            .zip(&uvs)
            .filter_map(|(&position, uv)| {
                Vec3::from_array(position)
                    .abs_diff_eq(target, 1.0e-5)
                    .then_some(uv[1])
            })
            .collect::<Vec<_>>();
        let expected_v = std::f32::consts::FRAC_PI_2 * 0.125
            / crate::render::mesh::construction::MATERIAL_TEXTURE_METERS_PER_REPEAT;
        assert!(!matching_v.is_empty());
        assert!(
            matching_v.iter().all(|&v| (v - expected_v).abs() < 1.0e-5),
            "part {part:?} circumference phase {matching_v:?} did not match {expected_v}"
        );
    }

    #[test]
    fn welded_pipe_pieces_share_texture_phase_and_hide_internal_caps() {
        let pieces = crate::builder::pipe_run_pieces(
            &[
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.875, 1.0, 0.0),
                Vec3::new(0.875, 1.0, 0.875),
            ],
            // A two-block bend keeps a full inner wall; a creased one-block
            // bend has a pinch vertex lying on its own cap planes.
            &[crate::builder::PipeNode::Bend { span: 2 }],
            CylinderDimensions::new(0.25, 0.10, 1.0).unwrap(),
            ConstructionMaterial::Wood,
        )
        .unwrap();
        let graph = crate::builder::stage_pipe_run(
            &ConstructionGraph::new(),
            &pieces,
            crate::builder::PipeRunAttachment::AutoWeld {
                source: FaceOwner::Ground,
            },
        )
        .unwrap();
        let offsets = crate::render::mesh::pipe::pipe_texture_offsets(&graph);
        let welded_ends = crate::render::mesh::pipe::welded_pipe_ends(&graph);
        let mut raw_length_mismatch_seen = false;
        let mut raw_circumference_mismatch_seen = false;
        let mut checked = 0;

        for (_, weld) in graph.welds() {
            let (FaceOwner::Part(first), FaceOwner::Part(second)) =
                (weld.first.owner, weld.second.owner)
            else {
                continue;
            };
            let Some(first_u) = graph.part(first).copied().and_then(|spec| {
                crate::render::mesh::pipe::pipe_endpoint_texture_u(spec, weld.first.face)
            }) else {
                continue;
            };
            let Some(second_u) = graph.part(second).copied().and_then(|spec| {
                crate::render::mesh::pipe::pipe_endpoint_texture_u(spec, weld.second.face)
            }) else {
                continue;
            };
            let first_frame = crate::render::mesh::pipe::pipe_endpoint_texture_frame(
                *graph.part(first).unwrap(),
                weld.first.face,
            )
            .unwrap();
            let second_frame = crate::render::mesh::pipe::pipe_endpoint_texture_frame(
                *graph.part(second).unwrap(),
                weld.second.face,
            )
            .unwrap();
            let angular = -second_frame.direction.cross(second_frame.radial_zero);
            let second_angle = first_frame
                .radial_zero
                .dot(angular)
                .atan2(first_frame.radial_zero.dot(second_frame.radial_zero));

            raw_length_mismatch_seen |= (first_u - second_u).abs() > 1.0e-6;
            raw_circumference_mismatch_seen |= second_angle.abs() > 1.0e-6;
            assert!((first_u + offsets[&first].u - second_u - offsets[&second].u).abs() < 1.0e-6);
            assert!(
                (offsets[&first].v_angle - second_angle - offsets[&second].v_angle).abs() < 1.0e-6
            );
            assert!(welded_ends.contains(&weld.first));
            assert!(welded_ends.contains(&weld.second));
            checked += 1;
        }

        assert!(raw_length_mismatch_seen);
        assert!(raw_circumference_mismatch_seen);
        assert_eq!(checked, 2);
        assert_eq!(welded_ends.len(), 4);

        for (part, spec) in graph.parts() {
            match spec {
                PartSpec::Cylinder(_) | PartSpec::PipeBend(_) => {
                    assert_pipe_circumference_phase(&graph, part, *spec, offsets[&part]);
                }
                _ => {}
            }
        }

        let mesh = crate::render::mesh::construction::combined_construction_mesh(&graph);
        let positions = positions(&mesh);
        let Some(VertexAttributeValues::Float32x3(normals)) =
            mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
        else {
            panic!("mesh must have float3 normals")
        };
        for (_, weld) in graph.welds() {
            let FaceOwner::Part(_) = weld.first.owner else {
                continue;
            };
            if !welded_ends.contains(&weld.first) {
                continue;
            }
            let face = crate::builder::face_geometry_from_ref(weld.first, Some(&graph));
            assert!(!positions.iter().zip(normals).any(|(position, normal)| {
                (position - face.center).dot(face.normal).abs() < 1.0e-5
                    && Vec3::from_array(*normal).dot(face.normal).abs() > 0.9
            }));
        }
    }

    #[test]
    fn welded_pipe_of_smaller_diameter_keeps_the_wider_cap_visible() {
        let mut pieces = crate::builder::pipe_run_pieces(
            &[
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.875, 1.0, 0.0),
                Vec3::new(0.875, 1.0, 0.875),
            ],
            &[crate::builder::PipeNode::Bend { span: 1 }],
            CylinderDimensions::new(0.25, 0.10, 1.0).unwrap(),
            ConstructionMaterial::Wood,
        )
        .unwrap();
        let last = pieces.last_mut().unwrap();
        let PartSpec::Cylinder(mut cylinder) = last.spec else {
            panic!("pipe run ends in a straight cylinder");
        };
        cylinder.dimensions =
            CylinderDimensions::new(0.15, 0.10, cylinder.dimensions.axial_length()).unwrap();
        last.spec = PartSpec::Cylinder(cylinder);
        let graph = crate::builder::stage_pipe_run(
            &ConstructionGraph::new(),
            &pieces,
            crate::builder::PipeRunAttachment::AutoWeld {
                source: FaceOwner::Ground,
            },
        )
        .unwrap();
        let hidden = crate::render::mesh::pipe::welded_pipe_ends(&graph);

        let mut narrowed = 0;
        for (_, weld) in graph.welds() {
            let (FaceOwner::Part(first), FaceOwner::Part(second)) =
                (weld.first.owner, weld.second.owner)
            else {
                continue;
            };
            let outer = |part| match graph.part(part) {
                Some(PartSpec::Cylinder(cylinder)) => cylinder.dimensions.outer_diameter(),
                Some(PartSpec::PipeBend(bend)) => bend.dimensions.outer_diameter(),
                _ => 0.0,
            };
            let (narrow, wide) = match outer(first).total_cmp(&outer(second)) {
                std::cmp::Ordering::Equal => {
                    assert!(hidden.contains(&weld.first) && hidden.contains(&weld.second));
                    continue;
                }
                std::cmp::Ordering::Less => (weld.first, weld.second),
                std::cmp::Ordering::Greater => (weld.second, weld.first),
            };
            assert!(
                hidden.contains(&narrow),
                "narrow cap sits inside the wide one"
            );
            assert!(
                !hidden.contains(&wide),
                "wide cap shows around the narrow pipe"
            );
            narrowed += 1;
        }
        assert_eq!(narrowed, 1);
        assert_eq!(hidden.len(), 3);
    }

    #[test]
    fn simulation_renders_one_cylinder_despite_sixteen_physical_colliders() {
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
                CylinderDimensions::new(1.0, 0.5, 1.0).unwrap(),
                BuildPose::default(),
            )))
            .unwrap();
        let creation = graph.compile().unwrap();
        assert_eq!(
            creation.colliders.len(),
            mechanic_core::CYLINDER_COLLIDER_COUNT
        );
        let transforms = [GpuTransform {
            position: [0.0, 0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
        }];
        let mesh =
            combined_simulation_mesh(&graph, &creation, &transforms, SimulationMeshKind::Dynamic);
        assert_eq!(
            mesh.count_vertices(),
            single_cylinder_mesh(CylinderDimensions::new(1.0, 0.5, 1.0).unwrap()).count_vertices()
        );
    }

    #[test]
    fn simulation_renders_the_same_feature_boundary_as_construction() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([4; 3], BuildPose::default()).unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let owner = SolidOwner::Part(part);
        let edge = graph.evaluated_solid(owner).unwrap().logical_edges[0].key;
        graph
            .apply(BuildCommand::AddShapeFeature(ShapeFeature::new(
                [EdgeChainRef { owner, edge }],
                EdgeTreatment::Fillet,
                20,
            )))
            .unwrap();
        let creation = graph.compile().unwrap();
        let transforms = [GpuTransform {
            position: [2.0, 0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
        }];

        let construction = crate::render::mesh::construction::combined_construction_mesh(&graph);
        let simulation =
            combined_simulation_mesh(&graph, &creation, &transforms, SimulationMeshKind::Dynamic);
        assert_eq!(simulation.count_vertices(), construction.count_vertices());
        let construction_min_x = positions(&construction)
            .into_iter()
            .map(|position| position.x)
            .fold(f32::INFINITY, f32::min);
        let simulation_min_x = positions(&simulation)
            .into_iter()
            .map(|position| position.x)
            .fold(f32::INFINITY, f32::min);
        assert!(
            (simulation_min_x - construction_min_x - 2.0).abs() < 1.0e-3,
            "construction {construction_min_x}, simulation {simulation_min_x}"
        );
    }

    #[test]
    fn rounded_mesh_normals_preserve_incident_faces_and_hard_seams() {
        for command in [
            BuildCommand::Spawn(CuboidSpec::new([4; 3], BuildPose::default()).unwrap()),
            BuildCommand::SpawnCylinder(CylinderSpec::new(
                CylinderDimensions::new(1.2, 0.0, 0.25).unwrap(),
                BuildPose::default(),
            )),
        ] {
            let mut graph = ConstructionGraph::new();
            let BuildOutcome::Spawned(part) = graph.apply(command).unwrap() else {
                unreachable!()
            };
            let owner = SolidOwner::Part(part);
            let edge = graph.evaluated_solid_shared(owner).unwrap().logical_edges[0].key;
            graph
                .apply(BuildCommand::AddShapeFeature(ShapeFeature::new(
                    [EdgeChainRef { owner, edge }],
                    EdgeTreatment::Fillet,
                    5,
                )))
                .unwrap();
            let solid = graph.evaluated_solid_shared(owner).unwrap();
            let mesh = crate::render::mesh::construction::combined_construction_mesh(&graph);
            let Some(VertexAttributeValues::Float32x3(actual)) =
                mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
            else {
                panic!("mesh has normals")
            };
            let mut expected = Vec::new();
            let mut hard = 0;
            let mut smooth = 0;
            for surface in &solid.surfaces {
                let mut edge = surface.half_edge;
                loop {
                    let vertex = solid.half_edges[edge as usize].origin;
                    let normal = if surface.smoothing_group == 0 {
                        hard += 1;
                        surface.normal
                    } else {
                        smooth += 1;
                        // Deliberately independent reference: inspect every incident face.
                        solid
                            .surfaces
                            .iter()
                            .enumerate()
                            .filter(|(face, candidate)| {
                                candidate.smoothing_group == surface.smoothing_group
                                    && solid.half_edges.iter().any(|edge| {
                                        edge.face as usize == *face && edge.origin == vertex
                                    })
                            })
                            .map(|(_, candidate)| candidate.normal)
                            .sum::<Vec3>()
                            .normalize_or_zero()
                    };
                    expected.push(normal.normalize_or_zero().to_array());
                    edge = solid.half_edges[edge as usize].next;
                    if edge == surface.half_edge {
                        break;
                    }
                }
            }
            assert!(hard > 0 && smooth > 0);
            assert_eq!(actual, &expected);
        }
    }

    #[test]
    fn fillet_keeps_one_texture_projection_through_its_profile() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([4; 3], BuildPose::default()).unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let owner = SolidOwner::Part(part);
        let solid = graph.evaluated_solid(owner).unwrap();
        let edge = solid
            .logical_edges
            .iter()
            .find(|logical| {
                let half_edge = solid.half_edges[logical.half_edges[0] as usize];
                let twin = solid.half_edges[half_edge.twin as usize];
                let patches = [
                    solid.surfaces[half_edge.face as usize].key.local,
                    solid.surfaces[twin.face as usize].key.local,
                ];
                patches.contains(&1) && patches.contains(&3)
            })
            .expect("the positive-X/positive-Y edge exists")
            .key;
        graph
            .apply(BuildCommand::AddShapeFeature(ShapeFeature::new(
                [EdgeChainRef { owner, edge }],
                EdgeTreatment::Fillet,
                20,
            )))
            .unwrap();

        let mesh = crate::render::mesh::construction::combined_construction_mesh(&graph);
        let Some(VertexAttributeValues::Float32x3(normals)) =
            mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
        else {
            panic!("mesh must have float3 normals")
        };
        let Some(VertexAttributeValues::Float32x4(tangents)) =
            mesh.attribute(Mesh::ATTRIBUTE_TANGENT)
        else {
            panic!("mesh must have float4 tangents")
        };
        let fillet_tangents = normals
            .iter()
            .zip(tangents)
            .filter_map(|(normal, tangent)| {
                let normal = Vec3::from_array(*normal).abs();
                (normal.x > 0.01 && normal.y > 0.01)
                    .then_some(Vec3::from_array(tangent[..3].try_into().unwrap()).abs())
            })
            .collect::<Vec<_>>();

        assert!(!fillet_tangents.is_empty());
        assert!(
            fillet_tangents
                .iter()
                .all(|tangent| tangent.abs_diff_eq(fillet_tangents[0], 1.0e-6)),
            "fillet changed texture projection through its profile: {fillet_tangents:?}"
        );
    }

    #[test]
    fn simulation_publication_only_touches_materials_in_each_motion_family() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(anchored) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([1; 3], BuildPose::default())
                    .unwrap()
                    .with_material(ConstructionMaterial::Steel),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1; 3],
                    BuildPose::new(IVec3::new(4, 0, 0), GridRotation::default()),
                )
                .unwrap()
                .with_material(ConstructionMaterial::Wood),
            ))
            .unwrap();
        let creation = graph.compile_with_static_parts([anchored]).unwrap();

        assert!(simulation_material_is_present(
            &graph,
            &creation,
            SimulationMeshKind::Static,
            ConstructionMaterial::Steel,
        ));
        assert!(!simulation_material_is_present(
            &graph,
            &creation,
            SimulationMeshKind::Dynamic,
            ConstructionMaterial::Steel,
        ));
        assert!(simulation_material_is_present(
            &graph,
            &creation,
            SimulationMeshKind::Dynamic,
            ConstructionMaterial::Wood,
        ));
        assert!(!simulation_material_is_present(
            &graph,
            &creation,
            SimulationMeshKind::Static,
            ConstructionMaterial::Wood,
        ));
    }

    #[test]
    fn zero_inner_diameter_generates_a_solid_disc_with_outward_winding() {
        let dimensions = BearingDimensions::new(0.50, 0.0).unwrap();
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut tangents = Vec::new();
        let mut indices = Vec::new();
        append_bearing_cylinder(
            Vec3::ZERO,
            Vec3::Y,
            dimensions,
            &mut positions,
            &mut normals,
            &mut uvs,
            &mut tangents,
            &mut indices,
        );

        assert!(positions.iter().any(|position| {
            let position = Vec3::from_array(*position);
            position.x.abs() < 1.0e-6 && position.z.abs() < 1.0e-6
        }));
        for triangle in indices.chunks_exact(3) {
            let a = Vec3::from_array(positions[triangle[0] as usize]);
            let b = Vec3::from_array(positions[triangle[1] as usize]);
            let c = Vec3::from_array(positions[triangle[2] as usize]);
            let geometric_normal = (b - a).cross(c - a);
            let expected_normal = triangle
                .iter()
                .map(|&index| Vec3::from_array(normals[index as usize]))
                .sum::<Vec3>();
            assert!(geometric_normal.dot(expected_normal) > 0.0);
        }
    }

    #[test]
    fn annular_mesh_inner_wall_and_faces_have_outward_winding() {
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut tangents = Vec::new();
        let mut indices = Vec::new();
        append_bearing_cylinder(
            Vec3::ZERO,
            Vec3::Y,
            BearingDimensions::default(),
            &mut positions,
            &mut normals,
            &mut uvs,
            &mut tangents,
            &mut indices,
        );

        for triangle in indices.chunks_exact(3) {
            let a = Vec3::from_array(positions[triangle[0] as usize]);
            let b = Vec3::from_array(positions[triangle[1] as usize]);
            let c = Vec3::from_array(positions[triangle[2] as usize]);
            let geometric_normal = (b - a).cross(c - a);
            let expected_normal = triangle
                .iter()
                .map(|&index| Vec3::from_array(normals[index as usize]))
                .sum::<Vec3>();
            assert!(geometric_normal.dot(expected_normal) > 0.0);
        }
    }

    #[test]
    fn unattached_bearing_is_included_in_the_visible_bearing_mesh() {
        let mut graph = ConstructionGraph::new();
        let support = CuboidSpec::new(
            [4, 4, 4],
            BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(support)).unwrap() else {
            unreachable!()
        };
        let bearing = PlacedBearing {
            kind: mechanic_core::BearingKind::Rotational,
            axis: Vec3::ZERO,
            source: FaceRef::part(part, FaceKind::PositiveY),
            anchor: Vec3::Y,
            dimensions: BearingDimensions::default(),
        };

        let mesh = combined_bearing_mesh(&graph, &[bearing]);

        assert!(mesh.count_vertices() > 0);
        assert_eq!(graph.bearing_count(), 0);
    }

    #[test]
    fn combined_bearing_mesh_preserves_each_bearings_dimensions() {
        let mut graph = ConstructionGraph::new();
        let specs = [IVec3::new(0, 2, 0), IVec3::new(4, 2, 0)].map(|center| {
            CuboidSpec::new([4, 4, 4], BuildPose::new(center, GridRotation::default())).unwrap()
        });
        let parts = specs.map(|spec| {
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                unreachable!()
            };
            part
        });
        let attached_dimensions = BearingDimensions::new(0.80, 0.30).unwrap();
        graph
            .apply(BuildCommand::AddBearing(
                mechanic_core::BearingSpec::new(
                    FaceRef::part(parts[0], FaceKind::PositiveX),
                    FaceRef::part(parts[1], FaceKind::NegativeX),
                    Vec3::new(0.5, 0.5, 0.0),
                    Vec3::X,
                )
                .with_dimensions(attached_dimensions),
            ))
            .unwrap();
        let placed_dimensions = BearingDimensions::new(0.40, 0.0).unwrap();
        let placed = PlacedBearing {
            kind: mechanic_core::BearingKind::Rotational,
            axis: Vec3::ZERO,
            source: FaceRef::part(parts[1], FaceKind::PositiveY),
            anchor: Vec3::new(1.0, 1.0, 0.0),
            dimensions: placed_dimensions,
        };

        let mesh = combined_bearing_mesh(&graph, &[placed]);
        let Some(VertexAttributeValues::Float32x3(positions)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("bearing mesh positions use Float32x3")
        };
        let attached_vertices = single_bearing_mesh(attached_dimensions).count_vertices();
        let attached_radius = positions[..attached_vertices]
            .iter()
            .map(|position| {
                let offset = Vec3::from_array(*position) - Vec3::new(0.5, 0.5, 0.0);
                (offset - Vec3::X * offset.x).length()
            })
            .fold(0.0, f32::max);
        let placed_radius = positions[attached_vertices..]
            .iter()
            .map(|position| {
                let offset = Vec3::from_array(*position) - placed.anchor;
                (offset - Vec3::Y * offset.y).length()
            })
            .fold(0.0, f32::max);
        assert!(
            (attached_radius
                - (attached_dimensions.outer_diameter() * 0.5 - BEARING_RENDER_RADIAL_SKIN))
                .abs()
                < 1.0e-6
        );
        assert!(
            (placed_radius
                - (placed_dimensions.outer_diameter() * 0.5 - BEARING_RENDER_RADIAL_SKIN))
                .abs()
                < 1.0e-6
        );
    }

    #[test]
    fn reusable_socket_with_multiple_attachments_renders_as_one_ring() {
        let mut graph = ConstructionGraph::new();
        let support = CuboidSpec::new(
            [4, 4, 4],
            BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(support) = graph.apply(BuildCommand::Spawn(support)).unwrap()
        else {
            unreachable!()
        };
        let targets = [IVec3::new(0, 9, 0), IVec3::new(2, 9, 0)].map(|center| {
            let spec = CuboidSpec::new(
                [1, 1, 1],
                BuildPose::from_half_grid(center, GridRotation::default()),
            )
            .unwrap();
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                unreachable!()
            };
            part
        });
        let socket = PlacedBearing {
            kind: mechanic_core::BearingKind::Rotational,
            axis: Vec3::ZERO,
            source: FaceRef::part(support, FaceKind::PositiveY),
            anchor: Vec3::Y,
            dimensions: BearingDimensions::new(0.80, 0.10).unwrap(),
        };
        for target in targets {
            graph
                .apply(BuildCommand::AddBearing(
                    mechanic_core::BearingSpec::new(
                        socket.source,
                        FaceRef::part(target, FaceKind::NegativeY),
                        socket.anchor,
                        Vec3::Y,
                    )
                    .with_dimensions(socket.dimensions),
                ))
                .unwrap();
        }
        graph
            .apply(BuildCommand::RigidLink(mechanic_core::RigidLinkSpec {
                first: targets[0],
                second: targets[1],
            }))
            .unwrap();

        let build_mesh = combined_bearing_mesh(&graph, &[socket]);
        let expected_vertices = single_bearing_mesh(socket.dimensions).count_vertices();
        assert_eq!(build_mesh.count_vertices(), expected_vertices);

        let creation = graph.compile().unwrap();
        assert_eq!(creation.bearings.len(), 1);
        let transforms = creation
            .compounds
            .iter()
            .map(|compound| GpuTransform {
                position: [
                    compound.root_translation.x,
                    compound.root_translation.y,
                    compound.root_translation.z,
                    0.0,
                ],
                rotation: compound.root_rotation.to_array(),
            })
            .collect::<Vec<_>>();
        let simulation_mesh =
            combined_simulation_bearing_mesh(&graph, &creation, &transforms, &[socket]);
        assert_eq!(simulation_mesh.count_vertices(), expected_vertices);
    }

    #[test]
    fn simulation_bearing_mesh_follows_attached_and_unattached_source_bodies() {
        let mut graph = ConstructionGraph::new();
        let specs = [IVec3::new(0, 2, 0), IVec3::new(4, 2, 0)].map(|center| {
            CuboidSpec::new([4, 4, 4], BuildPose::new(center, GridRotation::default())).unwrap()
        });
        let parts = specs.map(|spec| {
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                unreachable!()
            };
            part
        });
        let attached_dimensions = BearingDimensions::new(0.80, 0.30).unwrap();
        graph
            .apply(BuildCommand::AddBearing(
                mechanic_core::BearingSpec::new(
                    FaceRef::part(parts[0], FaceKind::PositiveX),
                    FaceRef::part(parts[1], FaceKind::NegativeX),
                    Vec3::new(0.5, 0.5, 0.0),
                    Vec3::X,
                )
                .with_dimensions(attached_dimensions),
            ))
            .unwrap();
        let placed = PlacedBearing {
            kind: mechanic_core::BearingKind::Rotational,
            axis: Vec3::ZERO,
            source: FaceRef::part(parts[0], FaceKind::PositiveY),
            anchor: Vec3::new(0.0, 1.0, 0.0),
            dimensions: BearingDimensions::new(0.40, 0.10).unwrap(),
        };
        let creation = graph.compile().unwrap();
        let source_compound = creation
            .part_to_compound
            .iter()
            .find_map(|&(part, compound)| (part == parts[0]).then_some(compound))
            .unwrap();
        let rotation = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let mut transforms = creation
            .compounds
            .iter()
            .map(|compound| GpuTransform {
                position: [
                    compound.root_translation.x,
                    compound.root_translation.y,
                    compound.root_translation.z,
                    0.0,
                ],
                rotation: compound.root_rotation.to_array(),
            })
            .collect::<Vec<_>>();
        transforms[source_compound as usize] = GpuTransform {
            position: [3.0, 4.0, 5.0, 0.0],
            rotation: rotation.to_array(),
        };

        let mesh = combined_simulation_bearing_mesh(&graph, &creation, &transforms, &[placed]);
        let Some(VertexAttributeValues::Float32x3(positions)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!("bearing mesh positions use Float32x3")
        };
        let attached_anchor = Vec3::new(3.0, 4.5, 5.0);
        let placed_anchor = Vec3::new(2.5, 4.0, 5.0);
        let attached_vertices = single_bearing_mesh(attached_dimensions).count_vertices();
        for (vertices, expected_anchor) in [
            (&positions[..attached_vertices], attached_anchor),
            (&positions[attached_vertices..], placed_anchor),
        ] {
            let minimum = vertices
                .iter()
                .map(|position| Vec3::from_array(*position))
                .fold(Vec3::splat(f32::INFINITY), Vec3::min);
            let maximum = vertices
                .iter()
                .map(|position| Vec3::from_array(*position))
                .fold(Vec3::splat(f32::NEG_INFINITY), Vec3::max);
            assert!(((minimum + maximum) * 0.5).abs_diff_eq(expected_anchor, 1.0e-5));
        }
    }

    #[test]
    fn joint_xray_requires_a_control_tool_and_bearing() {
        assert!(joint_xray_is_visible(Tool::Controller, 1));
        assert!(joint_xray_is_visible(Tool::Connector, 1));
        assert!(!joint_xray_is_visible(Tool::Controller, 0));
        assert!(!joint_xray_is_visible(Tool::Block, 1));
    }

    fn hinged_pair_with_control_block(reversed: bool) -> ConstructionGraph {
        let mut graph = ConstructionGraph::new();
        let spawn = |graph: &mut ConstructionGraph, x: i32| {
            let BuildOutcome::Spawned(id) = graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(
                        [4, 4, 4],
                        BuildPose::new(bevy::prelude::IVec3::new(x, 2, 0), GridRotation::default()),
                    )
                    .unwrap(),
                ))
                .unwrap()
            else {
                unreachable!()
            };
            id
        };
        let left = spawn(&mut graph, 0);
        let right = spawn(&mut graph, 4);
        let BuildOutcome::BearingAdded(bearing) = graph
            .apply(BuildCommand::AddBearing(mechanic_core::BearingSpec::new(
                FaceRef::part(left, FaceKind::PositiveX),
                FaceRef::part(right, FaceKind::NegativeX),
                Vec3::new(0.5, 0.5, 0.0),
                Vec3::X,
            )))
            .unwrap()
        else {
            unreachable!()
        };
        let BuildOutcome::Spawned(controller) = graph
            .apply(BuildCommand::SpawnController(ControllerSpec::new(
                BuildPose::new(bevy::prelude::IVec3::new(0, 12, 0), GridRotation::default()),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        let mut link = DriveLinkSpec::new(controller, bearing);
        link.reversed = reversed;
        link.limits = DriveLimits::new(2.0, 10.0, Some((-0.5, 0.5))).unwrap();
        link.program =
            DriveProgram::new(&[DriveState::new(DriveTarget::Speed(2.0)).unwrap()], false).unwrap();
        graph.apply(BuildCommand::AddDriveLink(link)).unwrap();
        graph
    }

    #[test]
    fn authored_parts_render_in_textured_meshes_not_the_construction_mesh() {
        let mut graph = hinged_pair_with_control_block(false);
        let mut engines = Vec::new();
        for (kind, x) in [(EngineKind::Gas, 20), (EngineKind::Electric, 24)] {
            let BuildOutcome::Spawned(engine) = graph
                .apply(BuildCommand::SpawnEngine(EngineSpec::new(
                    kind,
                    BuildPose::new(IVec3::new(x, 12, 0), GridRotation::default()),
                )))
                .unwrap()
            else {
                unreachable!()
            };
            engines.push(engine);
        }
        for engine in engines {
            let spec = graph.next_transmission_spec(engine).unwrap();
            graph
                .apply(BuildCommand::AttachTransmission {
                    parent: engine,
                    spec,
                })
                .unwrap();
        }
        graph
            .apply(BuildCommand::SpawnServo(ServoSpec::new(BuildPose::new(
                IVec3::new(28, 12, 0),
                GridRotation::default(),
            ))))
            .unwrap();
        graph
            .apply(BuildCommand::SpawnSeat(SeatSpec::new(BuildPose::new(
                IVec3::new(32, 12, 0),
                GridRotation::default(),
            ))))
            .unwrap();
        graph
            .apply(BuildCommand::SpawnInput(InputSpec::new(BuildPose::new(
                IVec3::new(36, 12, 0),
                GridRotation::default(),
            ))))
            .unwrap();
        let construction = crate::render::mesh::construction::combined_construction_mesh(&graph);
        let controllers = combined_controller_mesh(&graph);
        let gas = combined_authored_construction_mesh(&graph, AuthoredPart::GasEngine, None);
        let electric =
            combined_authored_construction_mesh(&graph, AuthoredPart::ElectricEngine, None);
        let gas_transmission =
            combined_authored_construction_mesh(&graph, AuthoredPart::GasTransmission, None);
        let electric_transmission =
            combined_authored_construction_mesh(&graph, AuthoredPart::ElectricTransmission, None);
        let servo = combined_authored_construction_mesh(&graph, AuthoredPart::Servo, None);
        let seat = combined_authored_construction_mesh(&graph, AuthoredPart::Seat, None);
        let input = combined_authored_construction_mesh(&graph, AuthoredPart::Input, None);

        // Two hinged blocks remain in the construction mesh; every authored part
        // has an independent batch so its material can use its own texture set.
        assert_eq!(positions(&construction).len(), 24 * 2);
        assert_eq!(positions(&controllers).len(), 24);
        assert_eq!(positions(&gas).len(), 24);
        assert_eq!(positions(&electric).len(), 24);
        assert_eq!(positions(&gas_transmission).len(), 24);
        assert_eq!(positions(&electric_transmission).len(), 24);
        assert_eq!(positions(&servo).len(), 24);
        assert_eq!(positions(&seat).len(), 24);
        assert_eq!(positions(&input).len(), 24);
        for mesh in [
            &controllers,
            &gas,
            &electric,
            &gas_transmission,
            &electric_transmission,
            &servo,
            &seat,
            &input,
        ] {
            assert_eq!(
                mesh.attribute(Mesh::ATTRIBUTE_UV_0).unwrap().len(),
                positions(mesh).len()
            );
            assert_eq!(
                mesh.attribute(Mesh::ATTRIBUTE_TANGENT).unwrap().len(),
                positions(mesh).len()
            );
        }
    }

    #[test]
    fn authored_preview_keeps_the_machine_uvs_and_texture_maps() {
        let mesh = single_authored_part_mesh(AuthoredPart::GasEngine);
        let Some(VertexAttributeValues::Float32x2(uvs)) = mesh.attribute(Mesh::ATTRIBUTE_UV_0)
        else {
            panic!("authored preview mesh must have float2 UVs")
        };
        assert_eq!(uvs, &authored_uvs(AuthoredPart::GasEngine));

        let texture = Handle::<Image>::default();
        let material = authored_preview_material(
            StandardMaterial {
                base_color_texture: Some(texture.clone()),
                normal_map_texture: Some(texture.clone()),
                metallic_roughness_texture: Some(texture.clone()),
                emissive_texture: Some(texture.clone()),
                ..Default::default()
            },
            Color::srgba(1.0, 1.0, 1.0, 0.46),
        );
        assert_eq!(material.alpha_mode, AlphaMode::Blend);
        assert_eq!(material.base_color_texture, Some(texture.clone()));
        assert_eq!(material.normal_map_texture, Some(texture.clone()));
        assert_eq!(material.metallic_roughness_texture, Some(texture.clone()));
        assert_eq!(material.emissive_texture, Some(texture));
    }

    #[test]
    fn authored_uvs_assign_each_controller_atlas_tile_to_its_named_face() {
        let uvs = authored_uvs(AuthoredPart::Controller);
        let bounds = |face: usize| {
            uvs[face * 4..face * 4 + 4].iter().fold(
                ([f32::INFINITY; 2], [f32::NEG_INFINITY; 2]),
                |(minimum, maximum), uv| {
                    (
                        [minimum[0].min(uv[0]), minimum[1].min(uv[1])],
                        [maximum[0].max(uv[0]), maximum[1].max(uv[1])],
                    )
                },
            )
        };

        // Vertex groups are +X, -X, +Y, -Y, +Z, -Z. These rectangles are the
        // labelled tiles in controller_reference.png.
        assert_eq!(bounds(0), ([0.0, 0.5], [0.25, 1.0]));
        assert_eq!(bounds(1), ([0.25, 0.5], [0.5, 1.0]));
        assert_eq!(bounds(2), ([0.5, 0.5], [1.0, 0.75]));
        assert_eq!(bounds(3), ([0.5, 0.75], [1.0, 1.0]));
        assert_eq!(bounds(4), ([0.0, 0.0], [0.5, 0.5]));
        assert_eq!(bounds(5), ([0.5, 0.0], [1.0, 0.5]));

        let approximately = |actual: [f32; 2], expected: [f32; 2]| {
            actual
                .iter()
                .zip(expected)
                .all(|(actual, expected)| (*actual - expected).abs() < 1.0e-6)
        };
        assert!(approximately(
            authored_uvs(AuthoredPart::GasEngine)[0],
            [0.0, 0.0]
        ));
        assert_eq!(
            authored_uvs(AuthoredPart::GasTransmission),
            authored_uvs(AuthoredPart::ElectricTransmission),
            "both imported transmission GLBs carry the same atlas UV layout",
        );
        assert_eq!(
            authored_uvs(AuthoredPart::GasTransmission),
            authored_uvs(AuthoredPart::Controller),
            "the extracted transmission atlas maps onto the shared authored cuboid ordering",
        );
        assert!(approximately(
            authored_uvs(AuthoredPart::ElectricEngine)[0],
            [0.666_667, 0.0]
        ));
        assert!(approximately(
            authored_uvs(AuthoredPart::Servo)[0],
            [0.666_667, 0.5]
        ));
        assert!(approximately(
            authored_uvs(AuthoredPart::Seat)[0],
            [0.5, 0.75]
        ));
        assert!(approximately(
            authored_uvs(AuthoredPart::Input)[0],
            [0.0, 0.75]
        ));
    }

    #[test]
    fn dimension_link_states_share_the_archive_atlas_layout() {
        let uvs = authored_uvs(AuthoredPart::DimensionLinkDisabled);
        assert_eq!(
            uvs,
            authored_uvs(AuthoredPart::DimensionLinkEnabled),
            "state changes swap maps without changing the mesh atlas",
        );
        let bounds = |face: usize| {
            uvs[face * 4..face * 4 + 4].iter().fold(
                ([f32::INFINITY; 2], [f32::NEG_INFINITY; 2]),
                |(minimum, maximum), uv| {
                    (
                        [minimum[0].min(uv[0]), minimum[1].min(uv[1])],
                        [maximum[0].max(uv[0]), maximum[1].max(uv[1])],
                    )
                },
            )
        };

        // +X, -X, +Y, -Y, +Z, -Z in the archive's 4x4 atlas.
        assert_eq!(bounds(0), ([0.0, 0.5], [0.25, 0.75]));
        assert_eq!(bounds(1), ([0.25, 0.5], [0.5, 0.75]));
        assert_eq!(bounds(2), ([0.0, 0.25], [0.5, 0.5]));
        assert_eq!(bounds(3), ([0.5, 0.25], [1.0, 0.5]));
        assert_eq!(bounds(4), ([0.0, 0.0], [0.5, 0.25]));
        assert_eq!(bounds(5), ([0.5, 0.0], [1.0, 0.25]));
    }

    #[test]
    fn input_uvs_do_not_fold_either_triangle_of_a_face() {
        let uvs = authored_uvs(AuthoredPart::Input);
        let signed_area = |a: [f32; 2], b: [f32; 2], c: [f32; 2]| {
            (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        };

        for face in uvs.chunks_exact(4) {
            let first = signed_area(face[0], face[1], face[2]);
            let second = signed_area(face[0], face[2], face[3]);
            assert!(
                first * second > 0.0,
                "both triangles must map the same way around the atlas tile"
            );
        }
    }

    #[test]
    fn empty_logical_batches_keep_an_invisible_gpu_allocation() {
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::SpawnController(ControllerSpec::new(
                BuildPose::new(IVec3::ZERO, GridRotation::default()),
            )))
            .unwrap();

        let logical = crate::render::mesh::construction::combined_construction_mesh(&graph);
        assert_eq!(logical.count_vertices(), 0);

        let allocated = renderable_mesh(logical);
        assert_eq!(allocated.count_vertices(), 3);
        assert!(
            positions(&allocated)
                .iter()
                .all(|position| position.length_squared() <= f32::EPSILON)
        );
        assert_eq!(allocated.indices().unwrap().len(), 3);
    }

    #[test]
    fn drive_overlay_is_empty_without_a_wire_and_mirrors_the_spin_direction() {
        let mut graph = ConstructionGraph::new();
        assert!(
            positions(&combined_drive_xray_mesh(
                &graph,
                &[],
                &DriveSequencer::default()
            ))
            .is_empty()
        );

        graph = hinged_pair_with_control_block(false);
        let forward = positions(&combined_drive_xray_mesh(
            &graph,
            &[],
            &DriveSequencer::default(),
        ));
        assert!(!forward.is_empty());

        let reversed = positions(&combined_drive_xray_mesh(
            &hinged_pair_with_control_block(true),
            &[],
            &DriveSequencer::default(),
        ));
        assert_eq!(forward.len(), reversed.len());
        // The arc sweeps the other way, so the two overlays are not identical.
        assert!(
            forward
                .iter()
                .zip(&reversed)
                .any(|(left, right)| !left.abs_diff_eq(*right, 1.0e-4))
        );
    }

    #[test]
    fn default_zero_speed_overlay_mirrors_the_wires_direction() {
        let with_default_program = |reversed| {
            let mut graph = hinged_pair_with_control_block(reversed);
            let (link_id, link) = graph
                .drive_links()
                .next()
                .map(|(id, link)| (id, *link))
                .unwrap();
            graph.apply(BuildCommand::RemoveDriveLink(link_id)).unwrap();
            graph
                .apply(BuildCommand::AddDriveLink(DriveLinkSpec {
                    program: DriveProgram::default(),
                    ..link
                }))
                .unwrap();
            graph
        };
        let forward = positions(&combined_drive_xray_mesh(
            &with_default_program(false),
            &[],
            &DriveSequencer::default(),
        ));
        let reversed = positions(&combined_drive_xray_mesh(
            &with_default_program(true),
            &[],
            &DriveSequencer::default(),
        ));

        assert_eq!(forward.len(), reversed.len());
        assert!(
            forward
                .iter()
                .zip(&reversed)
                .any(|(left, right)| !left.abs_diff_eq(*right, 1.0e-4))
        );
    }

    #[test]
    fn drive_overlay_shows_for_the_control_block_tools() {
        for tool in [Tool::Controller, Tool::Connector] {
            assert!(joint_xray_is_visible(tool, 1));
            assert!(drive_xray_is_visible(tool, 1));
            assert!(!drive_xray_is_visible(tool, 0));
        }
        assert!(!drive_xray_is_visible(Tool::Block, 1));
    }

    #[test]
    fn pipe_preview_keeps_its_mesh_until_geometry_or_shared_mesh_user_changes() {
        use crate::editor::preview::{ConstructionPreviewMeshKey, sync_preview_mesh};
        use crate::render::mesh::construction::combined_parts_mesh_scaled;
        let mut meshes = bevy::asset::Assets::<Mesh>::default();
        let pipe =
            mechanic_core::CylinderSpec::new(CylinderDimensions::default(), BuildPose::default());
        let specs = vec![PartSpec::Cylinder(pipe)];
        let key = ConstructionPreviewMeshKey::Pipe(specs.clone());
        let handle = meshes.add(combined_parts_mesh_scaled(&[], 1.0));
        let mut rendered = None;
        sync_preview_mesh(&mut meshes, &handle, &mut rendered, key.clone(), || {
            combined_parts_mesh_scaled(&specs, 1.0)
        });
        let pipe_positions = positions(meshes.get(&handle).unwrap());
        for _ in 0..20 {
            sync_preview_mesh(&mut meshes, &handle, &mut rendered, key.clone(), || {
                panic!("idle pipe preview rebuilt its mesh")
            });
        }
        let mut moved = pipe;
        moved.pose = BuildPose::new(IVec3::X * 4, GridRotation::default());
        sync_preview_mesh(
            &mut meshes,
            &handle,
            &mut rendered,
            ConstructionPreviewMeshKey::Layer(vec![PartSpec::Cylinder(moved)]),
            || combined_parts_mesh_scaled(&[PartSpec::Cylinder(moved)], 1.004),
        );
        assert_ne!(positions(meshes.get(&handle).unwrap()), pipe_positions);
        sync_preview_mesh(&mut meshes, &handle, &mut rendered, key, || {
            combined_parts_mesh_scaled(&specs, 1.0)
        });
        assert_eq!(positions(meshes.get(&handle).unwrap()), pipe_positions);
    }

    #[test]
    fn unchanged_bearing_preview_dimensions_do_not_rebuild_the_mesh() {
        let mut rendered = BearingDimensions::default();
        assert!(!bearing_preview_dimensions_changed(
            &mut rendered,
            BearingDimensions::default(),
        ));
        let custom = BearingDimensions::new(0.80, 0.20).unwrap();
        assert!(bearing_preview_dimensions_changed(&mut rendered, custom));
        assert!(!bearing_preview_dimensions_changed(&mut rendered, custom));
    }
}

#[cfg(test)]
mod interaction_tests {
    use bevy::prelude::{App, ButtonInput, IVec3, KeyCode, Quat, State, Update, Vec2, Vec3};
    use mechanic_core::{
        BearingDimensions, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph,
        ConstructionMaterial, ControllerSpec, CuboidSpec, CylinderDimensions, CylinderSpec,
        DimensionLinkId, DimensionLinkSpec, DriveLinkSpec, EdgeChainRef, EdgeTreatment, FaceKind,
        FaceOwner, FaceRef, GridRotation, MaterialAppearance, MaterialColor, MaterialDye,
        MaterialFinish, POSITION_TICK_METERS, PartId, PartSpec, PendingOperation, RigidLinkSpec,
        ShapeFeature, ShapeRegion, SolidOwner, WeldSpec,
    };
    use mechanic_gpu::GpuTransform;

    use super::{
        BlockVolume, MaterialWheelState, PlacementGrid, PlacementPlane, PlayerState, SelectedTool,
        SurfaceHit, Tool, bearing_attachment_candidate, candidate_from_hit, raycast_construction,
    };
    use crate::builder::{SmartGuide, block_sheet_specs};
    use crate::controls::GameAction;
    use crate::editor::build_actions::{
        PlacedBearing, handle_block_actions, handle_build_actions, handle_chroma_actions,
        stage_part_deletion_preserving_bearings,
    };
    use crate::editor::dimensions::{
        BearingDimensionTarget, BearingToolSettings, CylinderDimensionTarget, CylinderToolSettings,
        adjusted_bearing_dimensions, adjusted_cylinder_dimensions,
        requested_bearing_dimension_adjustment, requested_cylinder_dimension_adjustment,
    };
    use crate::editor::hammer::{
        HAMMER_CHARGE_SECONDS, HAMMER_MAX_IMPULSE, HAMMER_MIN_IMPULSE, hammer_delivery,
        hammer_impulse_magnitude, hammer_point_travel,
    };
    use crate::editor::history::{EditorHistory, HistoryAction, apply_history_action};
    use crate::editor::hover::{
        BearingOffsetDrag, BlockAttachment, BlockDrag, PointerSample, bearing_offset_from_rays,
        clear_editor_hover, delete_box_parts, handle_tool_change, refresh_bearing_offset_drag,
        refresh_block_drag, refresh_tool_preview,
    };
    use crate::editor::pipe::{
        PipeDrag, PipeEditMode, closest_axis_parameter, constrained_pipe_bend_span,
        pipe_corner_inset, pipe_pointer_delta, pipe_turn_direction, rebase_pipe_path,
    };
    use crate::editor::preview::{bearing_attachment_is_highlighted, tool_status_line};
    use crate::editor::raycast::{
        SimulationHit, raycast_placed_bearing_discs, raycast_placed_bearing_discs_with_pose,
        raycast_placed_bearings, raycast_simulation,
    };
    use crate::editor::shape_actions::{RegionDrag, commit_region_drag, region_area};
    use crate::editor::shape_actions::{
        active_drag_plane, choose_region, closer_feature_hit, handle_feature_shape_actions,
        refresh_region_drag, tangent_feature_chain, weld_connected_shape_owners,
    };
    use crate::editor::shortcuts::cycle_orientation;
    use crate::editor::state::{EditorGraph, EditorState};
    use crate::editor::wiring::{WireConnection, WireDrag, WireDragStep, WireEnd};
    use crate::editor::wiring::{
        connect_control_link, connect_drive_wire, wire_drag_step, wire_end_under_cursor,
    };
    use crate::pose::simulation_placed_bearing_pose;
    use crate::render::authored::{AUTHORED_ORIENTATION_COUNT, AUTHORED_ORIENTATIONS};
    use crate::render::mesh::preview::block_sheet_bounds;
    use crate::simulation::state::AppSimulation;

    fn pointer_sample(cursor: Vec2, ray_origin: Vec3, ray_direction: Vec3) -> PointerSample {
        PointerSample {
            cursor,
            ray_origin,
            ray_direction,
        }
    }

    #[test]
    fn blocked_delete_release_clears_the_target_that_would_block_tab() {
        let mut state = EditorState {
            delete_target: Some(crate::editor::hover::DeleteTarget::PlacedBearing(0)),
            ..Default::default()
        };

        assert!(state.world_drag_active());
        assert!(state.cancel_delete_gesture());
        assert!(!state.world_drag_active());
    }

    #[test]
    fn right_click_deletes_a_dimension_link_from_a_moving_construction() {
        assert_right_click_deletes_dimension_link(SelectedTool::from_editor_tool(
            Tool::DimensionLink,
        ));
    }

    #[test]
    fn empty_hands_can_delete_a_dimension_link_from_a_moving_construction() {
        assert_right_click_deletes_dimension_link(SelectedTool {
            tool: None,
            ..Default::default()
        });
    }

    fn assert_right_click_deletes_dimension_link(selection: SelectedTool) {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(link) = graph
            .apply(BuildCommand::SpawnDimensionLink(DimensionLinkSpec::new(
                DimensionLinkId(7),
                BuildPose::default(),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        let state = EditorState {
            hovered_simulation: Some(SimulationHit {
                part: link,
                body_index: 0,
                distance: 1.0,
                point: Vec3::ZERO,
                normal: Vec3::Y,
            }),
            ..Default::default()
        };
        let mut actions = ButtonInput::default();
        actions.press(GameAction::Secondary);
        let mut app = App::new();
        app.insert_resource(actions)
            .insert_resource(ButtonInput::<KeyCode>::default())
            .insert_resource(EditorGraph(graph))
            .insert_resource(state)
            .insert_resource(EditorHistory::default())
            .insert_resource(crate::chroma::ChromaBrush::default())
            .insert_resource(AppSimulation::default())
            .insert_resource(selection)
            .insert_resource(BearingToolSettings::default())
            .insert_resource(CylinderToolSettings::default())
            .insert_resource(crate::ui::UiInput::default())
            .insert_resource(MaterialWheelState::default())
            .insert_resource(PlayerState {
                input_captured: true,
                ..Default::default()
            })
            .init_resource::<bevy::input::mouse::AccumulatedMouseMotion>()
            .add_systems(Update, handle_build_actions);

        app.update();
        assert!(
            app.world()
                .resource::<EditorState>()
                .delete_target
                .is_some()
        );
        {
            let mut actions = app.world_mut().resource_mut::<ButtonInput<GameAction>>();
            actions.clear();
            actions.release(GameAction::Secondary);
        }
        app.update();

        assert!(app.world().resource::<EditorGraph>().0.part(link).is_none());
        assert_eq!(
            app.world().resource::<EditorState>().feedback.as_deref(),
            Some("Deleted Dimension Link and incident connections")
        );
    }

    #[test]
    fn connector_can_begin_wiring_a_moving_bearing() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [4, 4, 4],
                    BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let bearing = PlacedBearing {
            kind: mechanic_core::BearingKind::Rotational,
            axis: Vec3::ZERO,
            source: FaceRef::part(part, FaceKind::PositiveY),
            anchor: Vec3::Y,
            dimensions: BearingDimensions::default(),
        };
        let state = EditorState {
            hovered_bearing: Some(0),
            placed_bearings: vec![bearing],
            ..Default::default()
        };
        let mut actions = ButtonInput::default();
        actions.press(GameAction::Primary);
        let mut app = App::new();
        app.insert_resource(actions)
            .insert_resource(EditorGraph(graph))
            .insert_resource(state)
            .insert_resource(EditorHistory::default())
            .insert_resource(crate::chroma::ChromaBrush::default())
            .insert_resource(AppSimulation::default())
            .insert_resource(SelectedTool::from_editor_tool(Tool::Connector))
            .insert_resource(BearingToolSettings::default())
            .insert_resource(CylinderToolSettings::default())
            .insert_resource(crate::ui::UiInput::default())
            .insert_resource(MaterialWheelState::default())
            .insert_resource(PlayerState {
                input_captured: true,
                ..Default::default()
            })
            .init_resource::<bevy::input::mouse::AccumulatedMouseMotion>()
            .add_systems(Update, handle_build_actions);

        app.update();

        assert_eq!(
            app.world().resource::<EditorState>().wire_drag,
            Some(WireDrag {
                from: WireEnd::Bearing(0),
                armed: false,
            })
        );
    }

    #[test]
    fn connector_recognizes_a_moving_control_block() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(controller) = graph
            .apply(BuildCommand::SpawnController(ControllerSpec::new(
                BuildPose::default(),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        let state = EditorState {
            hovered_simulation: Some(SimulationHit {
                part: controller,
                body_index: 0,
                distance: 1.0,
                point: Vec3::ZERO,
                normal: Vec3::Y,
            }),
            ..Default::default()
        };

        assert_eq!(
            wire_end_under_cursor(&graph, &state),
            Some(WireEnd::Controller(controller))
        );
    }

    #[test]
    fn feature_drag_reuses_validated_geometry_only_for_matching_inputs() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([2; 3], BuildPose::default()).unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let owner = SolidOwner::Part(part);
        let target = EdgeChainRef {
            owner,
            edge: graph.evaluated_solid(owner).unwrap().logical_edges[0].key,
        };
        let mut drag = crate::shape_tool::FeatureDrag::begin(
            crate::shape_tool::FeatureEdgeHit {
                target,
                point: Vec3::ZERO,
                tangent: Vec3::Z,
                bisector: Vec3::X,
                distance: 0.0,
            },
            vec![target],
            EdgeTreatment::Fillet,
            None,
            0,
            Vec3::Y,
            Vec3::NEG_Y,
        );
        drag.amount_ticks =
            crate::editor::shape_actions::clamp_feature_amount(&graph, &mut drag, 10, 1);
        assert_eq!(drag.amount_ticks, 10);
        let preview = crate::editor::preview::feature_drag_preview_graph(&graph, &drag).unwrap();
        assert!(preview.shares_revision(&drag.validated_preview.as_ref().unwrap().graph));
        assert_eq!(
            graph.shape_features().count(),
            0,
            "a preview never commits the feature"
        );
        drag.amount_ticks = 20;
        let adjusted = crate::editor::preview::feature_drag_preview_graph(&graph, &drag).unwrap();
        assert!(
            adjusted.evaluated_solid(owner).unwrap().volume()
                < preview.evaluated_solid(owner).unwrap().volume()
        );
        drag.amount_ticks = 10;
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([1; 3], BuildPose::default()).unwrap(),
            ))
            .unwrap();
        let revised = crate::editor::preview::feature_drag_preview_graph(&graph, &drag).unwrap();
        assert_eq!(
            revised.part_count(),
            2,
            "a cached preview cannot hide a later placement"
        );
        assert_eq!(
            revised.evaluated_solid(owner).unwrap(),
            preview.evaluated_solid(owner).unwrap()
        );
    }

    #[test]
    fn committing_a_feature_clears_its_edge_selection() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([1; 3], BuildPose::default()).unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let owner = SolidOwner::Part(part);
        let target = EdgeChainRef {
            owner,
            edge: graph.evaluated_solid(owner).unwrap().logical_edges[0].key,
        };
        let hit = crate::shape_tool::FeatureEdgeHit {
            target,
            point: Vec3::ZERO,
            tangent: Vec3::Z,
            bisector: Vec3::X,
            distance: 0.0,
        };
        let ray_origin = Vec3::Y;
        let ray_direction = Vec3::NEG_Y;
        let mut state = EditorState {
            pointer_ray: Some((ray_origin, ray_direction)),
            feature_focus: Some(owner),
            selected_feature_edges: vec![target],
            feature_drag: Some(crate::shape_tool::FeatureDrag::begin(
                hit,
                vec![target],
                EdgeTreatment::Fillet,
                None,
                20,
                ray_origin,
                ray_direction,
            )),
            ..Default::default()
        };
        let mut actions = ButtonInput::default();
        actions.press(GameAction::Primary);
        actions.clear();
        actions.release(GameAction::Primary);
        let keys = ButtonInput::<KeyCode>::default();
        let mut history = EditorHistory::default();
        let player = PlayerState {
            input_captured: true,
            ..Default::default()
        };

        handle_feature_shape_actions(
            &actions,
            &keys,
            &mut graph,
            &mut state,
            &mut history,
            crate::shape_tool::ShapeSnap::feature_default(),
            crate::shape_tool::ShapeEditMode::Fillet,
            crate::ui::UiInput::default(),
            &player,
            &MaterialWheelState::default(),
        );

        assert_eq!(graph.shape_features().count(), 1);
        assert!(state.selected_feature_edges.is_empty());
        assert_eq!(state.selected_shape_feature, None);
        assert_eq!(history.undo.len(), 1);
    }

    #[test]
    fn every_pipe_run_cylinder_rim_accepts_a_fillet() {
        let pieces = crate::builder::pipe_run_pieces(
            &[
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.875, 1.0, 0.0),
                Vec3::new(0.875, 1.0, 0.875),
            ],
            &[crate::builder::PipeNode::Bend { span: 1 }],
            CylinderDimensions::new(0.25, 0.0, 1.0).unwrap(),
            ConstructionMaterial::Wood,
        )
        .unwrap();
        let graph = crate::builder::stage_pipe_run(
            &ConstructionGraph::new(),
            &pieces,
            crate::builder::PipeRunAttachment::AutoWeld {
                source: FaceOwner::Ground,
            },
        )
        .unwrap();

        let mut rims = 0;
        for (part, spec) in graph.parts() {
            let PartSpec::Cylinder(_) = spec else {
                continue;
            };
            let owner = SolidOwner::Part(part);
            let solid = graph.evaluated_solid(owner).unwrap();
            for logical in solid
                .logical_edges
                .iter()
                .filter(|edge| edge.closed && edge.convex)
            {
                rims += 1;
                graph
                    .clone()
                    .apply(BuildCommand::AddShapeFeature(ShapeFeature::new(
                        [EdgeChainRef {
                            owner,
                            edge: logical.key,
                        }],
                        EdgeTreatment::Fillet,
                        20,
                    )))
                    .unwrap_or_else(|error| panic!("fillet rejected on {part:?}: {error}"));
            }
        }
        assert_eq!(rims, 4, "both pipe cylinders offer two rims");
    }

    #[test]
    fn closer_generated_edge_wins_over_a_dashed_feature_source() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([2; 3], BuildPose::default()).unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let owner = SolidOwner::Part(part);
        let target = EdgeChainRef {
            owner,
            edge: graph.evaluated_solid(owner).unwrap().logical_edges[0].key,
        };
        let BuildOutcome::ShapeFeatureAdded(feature) = graph
            .apply(BuildCommand::AddShapeFeature(ShapeFeature::new(
                [target],
                EdgeTreatment::Chamfer,
                10,
            )))
            .unwrap()
        else {
            unreachable!()
        };
        let hit = |distance| crate::shape_tool::FeatureEdgeHit {
            target,
            point: Vec3::ZERO,
            tangent: Vec3::X,
            bisector: Vec3::Y,
            distance,
        };

        assert_eq!(
            closer_feature_hit(Some(hit(0.01)), Some((feature, hit(0.03)))),
            (None, Some(hit(0.01)))
        );
        assert_eq!(
            closer_feature_hit(Some(hit(0.04)), Some((feature, hit(0.02)))),
            (Some(feature), Some(hit(0.02)))
        );
        assert_eq!(
            closer_feature_hit(Some(hit(0.02)), Some((feature, hit(0.02)))),
            (None, Some(hit(0.02))),
            "an equal-distance real edge must not enter source adjustment mode"
        );
    }

    #[test]
    fn tangent_edge_selection_traverses_one_weld_component() {
        let mut graph = ConstructionGraph::new();
        let mut parts = Vec::new();
        for x in 0..4 {
            let BuildOutcome::Spawned(part) = graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(
                        [1; 3],
                        BuildPose::new(IVec3::new(x, 0, 0), GridRotation::default()),
                    )
                    .unwrap(),
                ))
                .unwrap()
            else {
                unreachable!()
            };
            parts.push(part);
        }
        for pair in parts[..3].windows(2) {
            graph
                .apply(BuildCommand::Weld(WeldSpec {
                    first: FaceRef::part(pair[0], FaceKind::PositiveX),
                    second: FaceRef::part(pair[1], FaceKind::NegativeX),
                }))
                .unwrap();
        }

        let target_for = |part| {
            let owner = SolidOwner::Part(part);
            let solid = graph.evaluated_solid(owner).unwrap();
            let edge = solid
                .logical_edges
                .iter()
                .find(|logical| {
                    let half_edge = solid.half_edges[logical.half_edges[0] as usize];
                    let twin = solid.half_edges[half_edge.twin as usize];
                    let patches = [
                        solid.surfaces[half_edge.face as usize].key.local,
                        solid.surfaces[twin.face as usize].key.local,
                    ];
                    patches.contains(&3) && patches.contains(&4)
                })
                .expect("the positive-Y/negative-Z edge exists")
                .key;
            EdgeChainRef { owner, edge }
        };
        let initial = target_for(parts[0]);

        let connected = weld_connected_shape_owners(&graph, initial.owner);
        assert_eq!(connected.len(), 3);
        assert!(!connected.contains(&SolidOwner::Part(parts[3])));
        assert_eq!(
            tangent_feature_chain(&graph, initial),
            parts[..3]
                .iter()
                .copied()
                .map(target_for)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn chroma_drag_paints_each_crossed_part_as_one_undoable_stroke() {
        let mut graph = ConstructionGraph::new();
        let mut ids = Vec::new();
        for x in [0, 4] {
            let BuildOutcome::Spawned(id) = graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(
                        [1, 1, 1],
                        BuildPose::new(IVec3::new(x, 0, 0), GridRotation::default()),
                    )
                    .unwrap(),
                ))
                .unwrap()
            else {
                unreachable!()
            };
            ids.push(id);
        }
        let paint = MaterialAppearance::new(
            MaterialColor::Dye(MaterialDye::new([42, 76, 199], 1.0).unwrap()),
            MaterialFinish::Painted,
        );
        let hit = |part| SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: FaceRef::part(part, FaceKind::PositiveY),
        };
        let mut state = EditorState {
            hovered: Some(hit(ids[0])),
            ..Default::default()
        };
        let mut history = EditorHistory::default();
        let mut mouse = ButtonInput::default();

        mouse.press(GameAction::Primary);
        handle_chroma_actions(&mouse, &mut graph, &mut state, &mut history, paint);
        mouse.clear();
        state.hovered = Some(hit(ids[1]));
        handle_chroma_actions(&mouse, &mut graph, &mut state, &mut history, paint);
        mouse.clear();
        mouse.release(GameAction::Primary);
        handle_chroma_actions(&mouse, &mut graph, &mut state, &mut history, paint);

        assert_eq!(history.undo.len(), 1);
        assert!(
            ids.into_iter()
                .all(|id| graph.part(id).unwrap().appearance() == Some(paint))
        );
        assert!(apply_history_action(
            HistoryAction::Undo,
            &mut graph,
            &mut state,
            &mut history,
        ));
        assert!(
            graph
                .parts()
                .all(|(_, part)| part.appearance() == Some(MaterialAppearance::BAKED))
        );
    }

    #[test]
    fn chroma_remove_restores_baked_once_and_a_noop_stroke_adds_no_history() {
        let paint = MaterialAppearance::new(
            MaterialColor::Dye(MaterialDye::new([224, 86, 31], 1.0).unwrap()),
            MaterialFinish::Anodised,
        );
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([1; 3], BuildPose::default())
                    .unwrap()
                    .with_appearance(paint),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let mut state = EditorState {
            hovered: Some(SurfaceHit {
                distance: 1.0,
                point: Vec3::ZERO,
                face: FaceRef::part(part, FaceKind::PositiveY),
            }),
            ..Default::default()
        };
        let mut history = EditorHistory::default();
        let mut mouse = ButtonInput::default();

        for expected_history in [1, 1] {
            mouse.press(GameAction::Secondary);
            handle_chroma_actions(&mouse, &mut graph, &mut state, &mut history, paint);
            mouse.clear();
            mouse.release(GameAction::Secondary);
            handle_chroma_actions(&mouse, &mut graph, &mut state, &mut history, paint);
            mouse.clear();
            assert_eq!(history.undo.len(), expected_history);
            assert_eq!(
                graph.part(part).unwrap().appearance(),
                Some(MaterialAppearance::BAKED)
            );
        }
    }

    #[test]
    fn rotate_cycles_every_authored_tool_through_all_grid_orientations() {
        for tool in [Tool::Controller, Tool::GasEngine, Tool::ElectricEngine] {
            let mut state = EditorState::default();
            for expected in (1..AUTHORED_ORIENTATION_COUNT).chain(std::iter::once(0)) {
                cycle_orientation(&mut state, tool);
                assert_eq!(
                    state.authored_orientation,
                    expected,
                    "{} should rotate",
                    tool.label()
                );
            }
        }
    }

    #[test]
    fn rotate_cycles_the_axis_of_an_active_vertex_drag() {
        let region =
            ShapeRegion::new(IVec3::ZERO, IVec3::ONE, ConstructionMaterial::Steel).unwrap();
        let start = crate::shape_tool::vertex_position(&region, [0, 0, 0]).unwrap();
        let ray_origin = start + Vec3::Z * 2.0;
        let ray_direction = Vec3::NEG_Z;
        let drag =
            crate::shape_tool::begin_group_drag(&region, [0, 0, 0], &[], ray_origin, ray_direction);
        let mut state = EditorState {
            pointer_position: Some(Vec2::ZERO),
            pointer_ray: Some((ray_origin, ray_direction)),
            vertex_drag: Some(drag),
            ..Default::default()
        };

        assert_eq!(cycle_orientation(&mut state, Tool::Shape), "Shape axis: Y");
        assert_eq!(state.vertex_drag.as_ref().unwrap().axis, 1);
    }

    #[test]
    fn authored_orientation_cycle_contains_all_24_cube_orientations_once() {
        let mut signatures = Vec::new();
        for rotation in AUTHORED_ORIENTATIONS {
            let quaternion = rotation.quaternion();
            let signature = [Vec3::X, Vec3::Y, Vec3::Z]
                .map(|axis| (quaternion * axis).round().as_ivec3().to_array());
            assert!(!signatures.contains(&signature));
            signatures.push(signature);
        }
        assert_eq!(signatures.len(), 24);
    }

    #[test]
    fn authored_preview_uses_a_tipped_rotation_selected_with_q() {
        let graph = ConstructionGraph::new();
        let mut state = EditorState {
            hovered: Some(SurfaceHit {
                distance: 1.0,
                point: Vec3::ZERO,
                face: FaceRef::ground(),
            }),
            authored_orientation: 16,
            ..Default::default()
        };

        refresh_tool_preview(&graph, &mut state, Tool::GasEngine);

        let preview = state.preview.expect("gas engine has a ground preview");
        assert_eq!(preview.spec.pose.rotation.quarter_turns_xyz(), [1, 0, 0]);
        let (minimum, maximum) = crate::builder::part_world_bounds(preview.spec.into());
        assert!((maximum.y - minimum.y - 0.75).abs() < 1.0e-6);
        assert!((maximum.z - minimum.z - 0.50).abs() < 1.0e-6);
    }

    #[test]
    fn bearing_shortcuts_are_gated_and_adjust_the_requested_diameter() {
        let mut keyboard = ButtonInput::default();
        keyboard.press(GameAction::BearingOuterIncrease);
        assert_eq!(
            requested_bearing_dimension_adjustment(&keyboard, Some(Tool::Bearing), false),
            Some((BearingDimensionTarget::Outer, 1))
        );
        assert_eq!(
            requested_bearing_dimension_adjustment(&keyboard, Some(Tool::Block), false),
            None
        );
        assert_eq!(
            requested_bearing_dimension_adjustment(&keyboard, Some(Tool::Bearing), true),
            None
        );

        keyboard.reset_all();
        keyboard.press(GameAction::NudgeUp);
        assert_eq!(
            requested_bearing_dimension_adjustment(&keyboard, Some(Tool::Bearing), false),
            None
        );

        let increased = adjusted_bearing_dimensions(
            BearingDimensions::default(),
            BearingDimensionTarget::Outer,
            1,
        );
        assert!((increased.outer_diameter() - 0.30).abs() < 1.0e-6);
        assert!((increased.inner_diameter() - 0.10).abs() < f32::EPSILON);

        keyboard.reset_all();
        keyboard.press(GameAction::BearingInnerIncrease);
        assert_eq!(
            requested_bearing_dimension_adjustment(&keyboard, Some(Tool::Bearing), false),
            Some((BearingDimensionTarget::Inner, 1))
        );
    }

    #[test]
    fn cylinder_shortcuts_adjust_and_clamp_without_graph_history() {
        let mut keyboard = ButtonInput::default();
        keyboard.press(GameAction::CylinderOuterIncrease);
        assert_eq!(
            requested_cylinder_dimension_adjustment(&keyboard, Some(Tool::Cylinder), false),
            Some((CylinderDimensionTarget::Outer, 1))
        );
        keyboard.reset_all();
        keyboard.press(GameAction::CylinderInnerIncrease);
        assert_eq!(
            requested_cylinder_dimension_adjustment(&keyboard, Some(Tool::Cylinder), false),
            Some((CylinderDimensionTarget::Inner, 1))
        );
        keyboard.reset_all();
        keyboard.press(GameAction::CylinderSweepDecrease);
        assert_eq!(
            requested_cylinder_dimension_adjustment(&keyboard, Some(Tool::Cylinder), false),
            Some((CylinderDimensionTarget::Sweep, -1))
        );
        assert!(
            requested_cylinder_dimension_adjustment(&keyboard, Some(Tool::Block), false).is_none()
        );

        let dimensions = CylinderDimensions::new(0.25, 0.20, 0.25).unwrap();
        let reduced = adjusted_cylinder_dimensions(dimensions, CylinderDimensionTarget::Outer, -1);
        assert!((reduced.outer_diameter() - 0.20).abs() < 1.0e-6);
        assert!((reduced.inner_diameter() - 0.15).abs() < 1.0e-6);
        let minimum = adjusted_cylinder_dimensions(reduced, CylinderDimensionTarget::Length, -1);
        assert!((minimum.axial_length() - 0.25).abs() < f32::EPSILON);
        let slice = adjusted_cylinder_dimensions(minimum, CylinderDimensionTarget::Sweep, -1);
        assert_eq!(slice.sweep_angle_degrees(), 345);
        let minimum_sweep = (0..30).fold(slice, |dimensions, _| {
            adjusted_cylinder_dimensions(dimensions, CylinderDimensionTarget::Sweep, -1)
        });
        assert_eq!(minimum_sweep.sweep_angle_degrees(), 15);

        let graph = ConstructionGraph::new();
        let history = EditorHistory::default();
        let settings = CylinderToolSettings {
            dimensions: minimum,
            ..Default::default()
        };
        assert_eq!(graph.part_count(), 0);
        assert!(history.undo.is_empty() && history.redo.is_empty());
        assert_eq!(settings.dimensions, minimum);
    }

    #[test]
    fn bearing_adjustments_clamp_and_remain_outside_history() {
        let mut settings = BearingToolSettings {
            dimensions: BearingDimensions::new(0.20, 0.15).unwrap(),
        };
        let history = EditorHistory::default();
        settings.dimensions =
            adjusted_bearing_dimensions(settings.dimensions, BearingDimensionTarget::Outer, -1);
        assert!((settings.dimensions.outer_diameter() - 0.15).abs() < 1.0e-6);
        assert!((settings.dimensions.inner_diameter() - 0.10).abs() < 1.0e-6);
        assert!(history.undo.is_empty());
        assert!(history.redo.is_empty());

        let minimum = adjusted_bearing_dimensions(
            BearingDimensions::new(0.05, 0.0).unwrap(),
            BearingDimensionTarget::Outer,
            -1,
        );
        assert_eq!(minimum, BearingDimensions::new(0.05, 0.0).unwrap());
        let maximum_inner = adjusted_bearing_dimensions(
            BearingDimensions::default(),
            BearingDimensionTarget::Inner,
            1,
        );
        assert!((maximum_inner.inner_diameter() - 0.15).abs() < 1.0e-6);

        let hud = tool_status_line(
            Tool::Bearing,
            settings.dimensions,
            CylinderDimensions::default(),
            None,
            mechanic_core::ConstructionMaterial::Steel,
        );
        assert!(hud.contains("Outer: 0.15 m"));
        assert!(hud.contains("Inner: 0.10 m"));
        assert!(hud.contains("Shift+←/→"));
    }

    #[test]
    fn hammer_charge_is_monotonic_and_clamped() {
        let tap = hammer_impulse_magnitude(0.0);
        let half = hammer_impulse_magnitude(HAMMER_CHARGE_SECONDS * 0.5);
        let full = hammer_impulse_magnitude(HAMMER_CHARGE_SECONDS);
        assert!((tap - HAMMER_MIN_IMPULSE).abs() < f32::EPSILON);
        assert!(tap < half && half < full);
        assert!((full - HAMMER_MAX_IMPULSE).abs() < f32::EPSILON);
        assert!((full - 4_000.0).abs() < f32::EPSILON);
        assert!((hammer_impulse_magnitude(100.0) - HAMMER_MAX_IMPULSE).abs() < f32::EPSILON);
    }

    #[test]
    fn hard_hammer_impulses_are_delivered_in_collision_safe_steps() {
        let mut graph = ConstructionGraph::new();
        let spec = CuboidSpec::new(
            [1, 1, 1],
            BuildPose::from_half_grid(IVec3::new(0, 1, 0), GridRotation::default()),
        )
        .unwrap();
        graph.apply(BuildCommand::Spawn(spec)).unwrap();
        let creation = graph.compile().unwrap();
        let root = creation.compounds[0].root_translation;
        let transform = GpuTransform {
            position: [root.x, root.y, root.z, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
        };
        let impulse = Vec3::X * HAMMER_MAX_IMPULSE;
        let local_point = Vec3::new(0.0, 0.125, 0.0);

        let (ticks, impulse_per_tick) =
            hammer_delivery(&creation, transform, 0, local_point, impulse);

        assert!(ticks > 1);
        assert!(ticks <= crate::editor::hammer::HAMMER_MAX_DELIVERY_TICKS);
        assert!(impulse_per_tick.length() * f32::from(ticks) < impulse.length());
        assert!(
            hammer_point_travel(&creation, transform, 0, local_point, impulse_per_tick)
                <= crate::editor::hammer::HAMMER_MAX_POINT_TRAVEL_PER_TICK + f32::EPSILON
        );

        let mut heavy_graph = ConstructionGraph::new();
        let heavy = CuboidSpec::new(
            [4, 4, 4],
            BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
        )
        .unwrap();
        heavy_graph.apply(BuildCommand::Spawn(heavy)).unwrap();
        let heavy_creation = heavy_graph.compile().unwrap();
        let heavy_root = heavy_creation.compounds[0].root_translation;
        let heavy_transform = GpuTransform {
            position: [heavy_root.x, heavy_root.y, heavy_root.z, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
        };
        let (heavy_ticks, heavy_impulse_per_tick) =
            hammer_delivery(&heavy_creation, heavy_transform, 0, Vec3::ZERO, impulse);
        assert!(
            (heavy_impulse_per_tick.length() * f32::from(heavy_ticks) - impulse.length()).abs()
                < 1.0e-3
        );
    }

    #[test]
    fn moving_frame_raycast_matches_exact_authored_feature_geometry() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([4; 3], BuildPose::default()).unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let owner = mechanic_core::SolidOwner::Part(part);
        let edge = graph.evaluated_solid(owner).unwrap().logical_edges[0].key;
        graph
            .apply(BuildCommand::AddShapeFeature(
                mechanic_core::ShapeFeature::new(
                    [mechanic_core::EdgeChainRef { owner, edge }],
                    mechanic_core::EdgeTreatment::Chamfer,
                    20,
                ),
            ))
            .unwrap();
        let frame = mechanic_core::ConstructionFrame::new(
            Vec3::new(3.0, 1.0, -2.0),
            Quat::from_rotation_z(0.61) * Quat::from_rotation_x(-0.4),
        )
        .unwrap();
        graph.reframe_parts([part], frame).unwrap();
        let creation = graph.compile().unwrap();
        let position = Vec3::new(-4.0, 5.0, 8.0);
        let rotation = Quat::from_rotation_y(0.7) * Quat::from_rotation_x(0.3);
        let transforms = [GpuTransform {
            position: position.extend(0.0).to_array(),
            rotation: rotation.to_array(),
        }];
        for x in [-0.49, 0.0, 0.49] {
            for z in [-0.49, 0.0, 0.49] {
                let origin = frame.point(Vec3::new(x, 2.0, z));
                let direction = frame.vector(Vec3::NEG_Y);
                let authored = crate::builder::raycast_construction_with_ground(
                    &graph, origin, direction, None,
                );
                let world_origin =
                    position + rotation * (origin - creation.compounds[0].root_translation);
                let world_direction = rotation * direction;
                let live = raycast_simulation(
                    &graph,
                    &creation,
                    &transforms,
                    world_origin,
                    world_direction,
                );
                assert_eq!(authored.is_some(), live.is_some());
                if let (Some(authored), Some(live)) = (authored, live) {
                    assert_eq!(live.part, part);
                    assert!((live.distance - authored.distance).abs() < 1.0e-4);
                    let expected = position
                        + rotation * (authored.point - creation.compounds[0].root_translation);
                    assert!(live.point.abs_diff_eq(expected, 1.0e-4));
                }
            }
        }
    }

    #[test]
    fn moving_frame_bearing_raycast_uses_composed_anchor_and_axis() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([4; 3], BuildPose::default()).unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let frame = mechanic_core::ConstructionFrame::new(
            Vec3::new(2.0, 3.0, 4.0),
            Quat::from_rotation_z(0.65),
        )
        .unwrap();
        graph.reframe_parts([part], frame).unwrap();
        let bearing = PlacedBearing {
            kind: mechanic_core::BearingKind::Rotational,
            axis: Vec3::ZERO,
            source: FaceRef::part(part, FaceKind::PositiveY),
            anchor: frame.point(Vec3::Y * 0.5),
            dimensions: BearingDimensions::new(0.5, 0.1).unwrap(),
        };
        let creation = graph.compile().unwrap();
        let position = Vec3::new(-2.0, 4.0, 1.0);
        let rotation = Quat::from_rotation_x(-0.4);
        let transforms = [GpuTransform {
            position: position.extend(0.0).to_array(),
            rotation: rotation.to_array(),
        }];
        let anchor =
            position + rotation * (bearing.anchor - creation.compounds[0].root_translation);
        let axis = rotation * frame.vector(Vec3::Y);
        let origin = anchor + axis * 2.0 + rotation * frame.vector(Vec3::X * 0.15);
        let expected = crate::editor::raycast::raycast_bearing_annulus(
            origin,
            -axis,
            anchor,
            axis,
            bearing.dimensions,
        )
        .unwrap();
        let actual = crate::editor::raycast::raycast_simulation_bearings(
            &graph,
            &creation,
            &transforms,
            &[bearing],
            origin,
            -axis,
        )
        .unwrap();
        assert!((actual.1 - expected).abs() < 1.0e-5);
    }

    #[test]
    fn hammer_raycast_uses_the_current_simulated_pose() {
        let mut graph = ConstructionGraph::new();
        let spec = CuboidSpec::new(
            [4, 4, 4],
            BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(_) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
            unreachable!()
        };
        let creation = graph.compile().unwrap();
        let transforms = [GpuTransform {
            position: [5.0, 1.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
        }];

        let hit = raycast_simulation(
            &graph,
            &creation,
            &transforms,
            Vec3::new(5.0, 1.0, 5.0),
            Vec3::NEG_Z,
        )
        .unwrap();
        assert_eq!(hit.body_index, 0);
        assert!(hit.point.abs_diff_eq(Vec3::new(5.0, 1.0, 0.5), 1.0e-5));
        assert!(
            raycast_simulation(
                &graph,
                &creation,
                &transforms,
                Vec3::new(0.0, 1.0, 5.0),
                Vec3::NEG_Z,
            )
            .is_none()
        );
    }

    #[test]
    fn hammer_hits_pipe_walls_before_through_and_after_a_bend() {
        let pieces = crate::builder::pipe_run_pieces(
            &[
                Vec3::Y,
                Vec3::new(0.875, 1.0, 0.0),
                Vec3::new(0.875, 1.875, 0.0),
            ],
            &[crate::builder::PipeNode::Bend { span: 1 }],
            CylinderDimensions::new(0.25, 0.0, 1.0).unwrap(),
            ConstructionMaterial::Steel,
        )
        .unwrap();
        let graph = crate::builder::stage_pipe_run(
            &ConstructionGraph::new(),
            &pieces,
            crate::builder::PipeRunAttachment::Free,
        )
        .unwrap();
        let creation = graph.compile().unwrap();
        assert_eq!(creation.compounds.len(), 1);
        let root = &creation.compounds[0];
        for (position, rotation) in [
            (root.root_translation, root.root_rotation),
            (Vec3::new(3.0, 4.0, -2.0), Quat::from_rotation_y(0.7)),
        ] {
            let transforms = [GpuTransform {
                position: position.extend(0.0).to_array(),
                rotation: rotation.to_array(),
            }];
            let world_from_build = rotation * root.root_rotation.inverse();
            for point in [
                Vec3::new(0.375, 0.0, 0.0),
                Vec3::new(
                    0.75 + 0.25 / 2.0_f32.sqrt(),
                    0.25 - 0.25 / 2.0_f32.sqrt(),
                    0.0,
                ),
                Vec3::new(1.0, 0.625, 0.0),
            ] {
                let build_origin = point + Vec3::Y + Vec3::Z;
                let authored = crate::builder::raycast_construction_with_ground(
                    &graph,
                    build_origin,
                    Vec3::NEG_Z,
                    None,
                )
                .expect("pipe wall must be visible");
                let direction = world_from_build * Vec3::NEG_Z;
                let hit = raycast_simulation(
                    &graph,
                    &creation,
                    &transforms,
                    position + world_from_build * (build_origin - root.root_translation),
                    direction,
                )
                .expect("hammer must accept curved pipe walls");
                assert_eq!(hit.body_index, 0);
                assert!((hit.distance - authored.distance).abs() < 1.0e-5);
                assert!(hit.normal.abs_diff_eq(-direction, 1.0e-5));
            }
        }
    }

    #[test]
    fn hammer_raycast_respects_a_cylinder_slice() {
        let mut graph = ConstructionGraph::new();
        let dimensions = CylinderDimensions::new(1.0, 0.0, 1.0)
            .unwrap()
            .with_sweep_angle_degrees(90)
            .unwrap();
        let spec = CylinderSpec::new(
            dimensions,
            BuildPose::new(IVec3::ZERO, GridRotation::default()),
        );
        let BuildOutcome::Spawned(_) = graph.apply(BuildCommand::SpawnCylinder(spec)).unwrap()
        else {
            unreachable!()
        };
        let creation = graph.compile().unwrap();
        let root = creation.compounds[0].root_translation;
        let transforms = [GpuTransform {
            position: [root.x, root.y, root.z, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
        }];

        assert!(
            raycast_simulation(
                &graph,
                &creation,
                &transforms,
                Vec3::new(0.3, 2.0, 0.0),
                Vec3::NEG_Y,
            )
            .is_some()
        );
        assert!(
            raycast_simulation(
                &graph,
                &creation,
                &transforms,
                Vec3::new(-0.3, 2.0, 0.0),
                Vec3::NEG_Y,
            )
            .is_none()
        );
    }

    /// A solid, welded slab of one-cell blocks with its minimum corner at the
    /// origin, which is what a region drag needs underneath it.
    fn welded_slab(size: IVec3) -> ConstructionGraph {
        let mut graph = ConstructionGraph::new();
        let mut previous: Option<PartId> = None;
        for z in 0..size.z {
            for y in 0..size.y {
                for x in 0..size.x {
                    let spec = CuboidSpec::new(
                        [1, 1, 1],
                        BuildPose::from_half_grid(
                            IVec3::ONE + IVec3::new(x, y, z) * 2,
                            GridRotation::default(),
                        ),
                    )
                    .unwrap();
                    let BuildOutcome::Spawned(id) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
                    else {
                        unreachable!()
                    };
                    if let Some(first) = previous {
                        graph
                            .apply(BuildCommand::RigidLink(RigidLinkSpec { first, second: id }))
                            .unwrap();
                    }
                    previous = Some(id);
                }
            }
        }
        graph
    }

    fn region_drag_on(
        graph: &ConstructionGraph,
        plane: PlacementPlane,
        press: PointerSample,
    ) -> RegionDrag {
        let (_, spec) = graph.parts().next().expect("the slab has blocks");
        let start = spec.as_cuboid().expect("the slab is made of blocks");
        RegionDrag {
            start,
            press,
            plane,
            anchor_span: IVec3::ZERO,
            span: IVec3::ZERO,
            last_span: None,
            region: region_area(start, IVec3::ZERO),
            error: None,
        }
    }

    #[test]
    fn shape_selection_uses_the_world_resolved_hover_hit() {
        let mut graph = welded_slab(IVec3::ONE);
        let ray_origin = Vec3::new(0.125, 2.0, 0.125);
        let ray_direction = Vec3::NEG_Y;
        let hit = raycast_construction(&graph, ray_origin, ray_direction)
            .expect("the world-resolved ray hits the block");
        let mut state = EditorState {
            hovered: Some(hit),
            pointer_position: Some(Vec2::ZERO),
            pointer_ray: Some((ray_origin, ray_direction)),
            ..EditorState::default()
        };
        let mut actions = ButtonInput::default();
        actions.press(GameAction::Primary);

        choose_region(
            &actions,
            &mut graph,
            &mut state,
            &mut EditorHistory::default(),
            Some(Vec2::ZERO),
            ray_origin,
            ray_direction,
        );

        assert!(state.region_drag.is_some());
    }

    #[test]
    fn shape_selection_preserves_a_fine_placed_blocks_origin() {
        let mut graph = ConstructionGraph::new();
        let spec = CuboidSpec::new(
            [1, 1, 1],
            BuildPose::from_position_ticks(IVec3::new(60, 50, 50), GridRotation::default()),
        )
        .unwrap();
        graph.apply(BuildCommand::Spawn(spec)).unwrap();
        let ray_origin = Vec3::new(0.15, 2.0, 0.125);
        let ray_direction = Vec3::NEG_Y;
        let hit = raycast_construction(&graph, ray_origin, ray_direction)
            .expect("the ray hits the fine-placed block");
        let FaceOwner::Part(part) = hit.face.owner else {
            unreachable!("the ray hit a block")
        };
        let mut state = EditorState {
            hovered: Some(hit),
            pointer_position: Some(Vec2::ZERO),
            pointer_ray: Some((ray_origin, ray_direction)),
            ..EditorState::default()
        };
        let mut actions = ButtonInput::default();
        actions.press(GameAction::Primary);

        choose_region(
            &actions,
            &mut graph,
            &mut state,
            &mut EditorHistory::default(),
            Some(Vec2::ZERO),
            ray_origin,
            ray_direction,
        );
        commit_region_drag(&mut graph, &mut state, &mut EditorHistory::default());

        let region = state.active_region.and_then(|id| graph.region(id)).unwrap();
        assert_eq!(region.origin_steps(), IVec3::new(10, 0, 0));
        assert!(
            (region.bounds_steps().0.as_vec3() * POSITION_TICK_METERS)
                .abs_diff_eq(Vec3::new(0.025, 0.0, 0.0), 1.0e-7)
        );
        assert_eq!(graph.region_of(part), state.active_region);
    }

    #[test]
    fn dragging_across_blocks_claims_all_of_them_as_one_region() {
        let graph = welded_slab(IVec3::new(3, 2, 1));
        // Straight down onto the top of the first block, which is the XZ plane
        // the pointer then slides along.
        let press = pointer_sample(Vec2::ZERO, Vec3::new(0.125, 2.0, 0.125), Vec3::NEG_Y);
        let mut state = EditorState {
            region_drag: Some(region_drag_on(&graph, PlacementPlane::Xz, press)),
            ..Default::default()
        };

        refresh_region_drag(
            &graph,
            &mut state,
            Vec2::new(100.0, 0.0),
            press.ray_origin,
            (Vec3::new(0.625, 0.0, 0.125) - press.ray_origin).normalize(),
        );

        let drag = state.region_drag.as_ref().unwrap();
        assert_eq!(drag.span, IVec3::new(2, 0, 0));
        assert_eq!(drag.region.size_cells(), IVec3::new(3, 1, 1));
        assert_eq!(drag.error, None, "three welded blocks are a valid area");
    }

    #[test]
    fn rotate_mid_area_drag_extrudes_the_selection_into_a_box() {
        let graph = welded_slab(IVec3::new(3, 2, 1));
        let press = pointer_sample(Vec2::ZERO, Vec3::new(0.125, 2.0, 0.125), Vec3::NEG_Y);
        let mut state = EditorState {
            region_drag: Some(region_drag_on(&graph, PlacementPlane::Xz, press)),
            ..Default::default()
        };
        refresh_region_drag(
            &graph,
            &mut state,
            Vec2::new(100.0, 0.0),
            press.ray_origin,
            (Vec3::new(0.625, 0.0, 0.125) - press.ray_origin).normalize(),
        );

        // What Rotate does: keep the rectangle already dragged and re-anchor here.
        let rotated = pointer_sample(Vec2::ZERO, Vec3::new(0.125, 0.125, 2.0), Vec3::NEG_Z);
        {
            let drag = state.region_drag.as_mut().unwrap();
            drag.plane = drag.plane.cycle();
            assert_eq!(drag.plane, PlacementPlane::Xy);
            drag.anchor_span = drag.span;
            drag.press = rotated;
            drag.last_span = None;
        }

        refresh_region_drag(
            &graph,
            &mut state,
            Vec2::new(0.0, 100.0),
            rotated.ray_origin,
            (Vec3::new(0.125, 0.375, 0.125) - rotated.ray_origin).normalize(),
        );

        let drag = state.region_drag.as_ref().unwrap();
        assert_eq!(
            drag.span,
            IVec3::new(2, 1, 0),
            "the rotation keeps the extent and grows the third axis"
        );
        assert_eq!(drag.region.size_cells(), IVec3::new(3, 2, 1));
        assert_eq!(drag.error, None);
    }

    #[test]
    fn releasing_a_valid_area_opens_it_for_editing() {
        let mut graph = welded_slab(IVec3::new(2, 1, 1));
        let press = pointer_sample(Vec2::ZERO, Vec3::new(0.125, 2.0, 0.125), Vec3::NEG_Y);
        let mut state = EditorState {
            region_drag: Some(region_drag_on(&graph, PlacementPlane::Xz, press)),
            ..Default::default()
        };
        refresh_region_drag(
            &graph,
            &mut state,
            Vec2::new(100.0, 0.0),
            press.ray_origin,
            (Vec3::new(0.375, 0.0, 0.125) - press.ray_origin).normalize(),
        );
        let mut history = EditorHistory::default();

        commit_region_drag(&mut graph, &mut state, &mut history);

        assert!(state.region_drag.is_none());
        let region = state.active_region.and_then(|id| graph.region(id)).unwrap();
        assert_eq!(region.size_cells(), IVec3::new(2, 1, 1));
        assert_eq!(history.undo.len(), 1);
    }

    #[test]
    fn an_area_reaching_past_the_blocks_is_refused_rather_than_claimed() {
        let mut graph = welded_slab(IVec3::new(2, 1, 1));
        let press = pointer_sample(Vec2::ZERO, Vec3::new(0.125, 2.0, 0.125), Vec3::NEG_Y);
        let mut state = EditorState {
            region_drag: Some(region_drag_on(&graph, PlacementPlane::Xz, press)),
            ..Default::default()
        };
        // Three cells wide over a two-block slab: the far cell is empty.
        refresh_region_drag(
            &graph,
            &mut state,
            Vec2::new(100.0, 0.0),
            press.ray_origin,
            (Vec3::new(0.625, 0.0, 0.125) - press.ray_origin).normalize(),
        );
        assert!(state.region_drag.as_ref().unwrap().error.is_some());

        let mut history = EditorHistory::default();
        commit_region_drag(&mut graph, &mut state, &mut history);

        assert_eq!(graph.regions().count(), 0);
        assert!(state.active_region.is_none());
        assert!(history.undo.is_empty());
    }

    #[test]
    fn placing_blocks_shows_the_same_plane_as_choosing_an_area() {
        let graph = ConstructionGraph::new();
        let hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: FaceRef::ground(),
        };
        let candidate = candidate_from_hit(&graph, hit);
        let specs =
            block_sheet_specs(candidate.spec, IVec3::new(4, 1, 2), PlacementPlane::Xz).unwrap();
        let state = EditorState {
            block_drag: Some(BlockDrag {
                start: candidate,
                attachment: BlockAttachment::AutoWeld {
                    source: FaceOwner::Ground,
                },
                start_guides: Vec::new(),
                press: pointer_sample(Vec2::ZERO, Vec3::Y, Vec3::NEG_Y),
                plane: PlacementPlane::Xz,
                anchor_span: IVec3::ZERO,
                span: IVec3::new(2, 0, 1),
                last_span: Some(IVec3::new(2, 0, 1)),
                volume: BlockVolume::new(candidate.spec, IVec3::new(2, 0, 1)).unwrap(),
                error: None,
            }),
            ..Default::default()
        };

        let (low, high, plane) =
            active_drag_plane(&state, &AppSimulation::default()).expect("a block drag has a plane");
        assert_eq!(plane, PlacementPlane::Xz);
        // Centred on the blocks about to be placed, exactly as an area is.
        let (sheet_low, sheet_high) = block_sheet_bounds(&specs).unwrap();
        assert!(low.abs_diff_eq(sheet_low, 1.0e-6));
        assert!(high.abs_diff_eq(sheet_high, 1.0e-6));

        assert!(
            active_drag_plane(&EditorState::default(), &AppSimulation::default()).is_none(),
            "no drag, no plane"
        );
    }

    #[test]
    fn selecting_a_tool_cancels_pending_editor_state() {
        let mut graph = ConstructionGraph::new();
        let spec = CuboidSpec::new(
            [4, 4, 4],
            BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
            unreachable!()
        };
        graph
            .apply(BuildCommand::BeginPending(PendingOperation::Weld(
                FaceRef::part(part, FaceKind::PositiveY),
            )))
            .unwrap();

        let mut app = App::new();
        app.insert_resource(EditorGraph(graph))
            .insert_resource(EditorState::default())
            .insert_resource(SelectedTool::from_editor_tool(Tool::Bearing))
            .add_systems(Update, handle_tool_change);

        app.update();

        assert!(app.world().resource::<EditorGraph>().0.pending().is_none());
    }

    #[test]
    fn weld_highlight_contains_the_entire_rigid_body_only() {
        let mut graph = ConstructionGraph::new();
        let specs = [
            CuboidSpec::new(
                [4, 4, 4],
                BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
            )
            .unwrap(),
            CuboidSpec::new(
                [4, 4, 4],
                BuildPose::new(IVec3::new(0, 6, 0), GridRotation::default()),
            )
            .unwrap(),
            CuboidSpec::new(
                [4, 4, 4],
                BuildPose::new(IVec3::new(0, 10, 0), GridRotation::default()),
            )
            .unwrap(),
        ];
        let parts = specs.map(|spec| {
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                unreachable!()
            };
            part
        });
        graph
            .apply(BuildCommand::Weld(mechanic_core::WeldSpec {
                first: FaceRef::part(parts[0], FaceKind::PositiveY),
                second: FaceRef::part(parts[1], FaceKind::NegativeY),
            }))
            .unwrap();
        graph
            .apply(BuildCommand::AddBearing(mechanic_core::BearingSpec::new(
                FaceRef::part(parts[1], FaceKind::PositiveY),
                FaceRef::part(parts[2], FaceKind::NegativeY),
                Vec3::new(0.0, 2.0, 0.0),
                Vec3::Y,
            )))
            .unwrap();

        assert_eq!(
            crate::builder::rigid_body_parts(&graph, parts[0]),
            parts[..2]
        );
        assert_eq!(
            crate::builder::rigid_body_parts(&graph, parts[2]),
            vec![parts[2]]
        );
    }

    #[test]
    fn block_click_places_on_release_through_drag_path() {
        let mut graph = ConstructionGraph::new();
        let hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: mechanic_core::FaceRef::ground(),
        };
        let candidate = candidate_from_hit(&graph, hit);
        let press = pointer_sample(
            Vec2::new(320.0, 240.0),
            Vec3::new(-0.3, 2.0, -0.2),
            Vec3::new(0.2, -1.0, 0.3),
        );
        let mut state = EditorState {
            hovered: Some(hit),
            preview: Some(candidate),
            pointer_position: Some(press.cursor),
            pointer_ray: Some((press.ray_origin, press.ray_direction)),
            ..Default::default()
        };
        let mut mouse = ButtonInput::default();
        let mut history = EditorHistory::default();

        mouse.press(GameAction::Primary);
        handle_block_actions(&mouse, &mut graph, &mut state, &mut history);
        assert_eq!(graph.part_count(), 0);
        assert!(state.block_drag.is_some());
        refresh_block_drag(
            &graph,
            &mut state,
            press.cursor,
            press.ray_origin,
            press.ray_direction,
        );
        assert_eq!(state.block_drag.as_ref().unwrap().volume.count(), 1);

        mouse.clear();
        mouse.release(GameAction::Primary);
        handle_block_actions(&mouse, &mut graph, &mut state, &mut history);
        assert_eq!(graph.part_count(), 1);
        assert!(state.block_drag.is_none());
        assert_eq!(history.undo.len(), 1);
    }

    #[test]
    fn block_drag_dead_zone_and_motion_are_relative_to_the_press() {
        let graph = ConstructionGraph::new();
        let hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: FaceRef::ground(),
        };
        let candidate = candidate_from_hit(&graph, hit);
        let press = pointer_sample(Vec2::ZERO, Vec3::new(0.0, 2.0, 0.0), Vec3::NEG_Y);
        let start_guide = SmartGuide {
            axis: 0,
            coordinate: 0.0,
            from: Vec3::ZERO,
            to: Vec3::Z,
        };
        let mut state = EditorState {
            block_drag: Some(BlockDrag {
                start: candidate,
                attachment: BlockAttachment::AutoWeld {
                    source: FaceOwner::Ground,
                },
                start_guides: vec![start_guide],
                press,
                plane: PlacementPlane::Xz,
                anchor_span: IVec3::ZERO,
                span: IVec3::ZERO,
                last_span: None,
                volume: BlockVolume::new(candidate.spec, IVec3::ZERO).unwrap(),
                error: None,
            }),
            smart_guides: vec![start_guide],
            ..Default::default()
        };

        refresh_block_drag(
            &graph,
            &mut state,
            Vec2::new(4.99, 0.0),
            press.ray_origin,
            Quat::from_rotation_z(0.003) * Vec3::NEG_Y,
        );
        assert_eq!(state.block_drag.as_ref().unwrap().volume.count(), 1);

        for (target, expected) in [
            (Vec3::new(0.50, 2.0, 0.25), 6),
            (Vec3::new(-0.50, 2.0, -0.25), 6),
            (Vec3::new(0.50, 2.0, 0.0), 3),
            (Vec3::new(0.0, 2.0, -0.50), 3),
        ] {
            refresh_block_drag(
                &graph,
                &mut state,
                Vec2::new(10.0, 0.0),
                press.ray_origin,
                (Vec3::new(target.x, 0.0, target.z) - press.ray_origin).normalize(),
            );
            assert_eq!(state.block_drag.as_ref().unwrap().volume.count(), expected);
            assert!(state.smart_guides.contains(&start_guide));
        }
    }

    #[test]
    fn cycling_the_plane_without_pointer_motion_stays_one_by_one() {
        let graph = ConstructionGraph::new();
        let hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: FaceRef::ground(),
        };
        let candidate = candidate_from_hit(&graph, hit);
        let press = pointer_sample(
            Vec2::new(100.0, 100.0),
            Vec3::new(0.0, 2.0, 2.0),
            Vec3::new(0.0, -1.0, -1.0),
        );
        let mut state = EditorState {
            block_drag: Some(BlockDrag {
                start: candidate,
                attachment: BlockAttachment::AutoWeld {
                    source: FaceOwner::Ground,
                },
                start_guides: Vec::new(),
                press,
                plane: PlacementPlane::Xz,
                anchor_span: IVec3::ZERO,
                span: IVec3::ZERO,
                last_span: None,
                volume: BlockVolume::new(candidate.spec, IVec3::ZERO).unwrap(),
                error: None,
            }),
            ..Default::default()
        };

        let drag = state.block_drag.as_mut().unwrap();
        drag.plane = drag.plane.cycle();
        assert_eq!(drag.plane, PlacementPlane::Xy);
        refresh_block_drag(
            &graph,
            &mut state,
            press.cursor,
            press.ray_origin,
            press.ray_direction,
        );
        assert_eq!(state.block_drag.as_ref().unwrap().volume.count(), 1);

        refresh_block_drag(
            &graph,
            &mut state,
            Vec2::new(110.0, 100.0),
            press.ray_origin,
            (Vec3::new(0.5, 0.0, 0.0) - press.ray_origin).normalize(),
        );
        assert_eq!(state.block_drag.as_ref().unwrap().volume.count(), 3);
    }

    #[test]
    fn dragged_placement_is_one_atomic_history_step() {
        let mut graph = ConstructionGraph::new();
        let hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::ZERO,
            face: FaceRef::ground(),
        };
        let candidate = candidate_from_hit(&graph, hit);
        let mut state = EditorState {
            block_drag: Some(BlockDrag {
                start: candidate,
                attachment: BlockAttachment::AutoWeld {
                    source: FaceOwner::Ground,
                },
                start_guides: Vec::new(),
                press: pointer_sample(Vec2::ZERO, Vec3::Y, Vec3::NEG_Y),
                plane: PlacementPlane::Xz,
                anchor_span: IVec3::ZERO,
                span: IVec3::new(2, 0, 1),
                last_span: Some(IVec3::new(2, 0, 1)),
                volume: BlockVolume::new(candidate.spec, IVec3::new(2, 0, 1)).unwrap(),
                error: None,
            }),
            ..Default::default()
        };
        let mut history = EditorHistory::default();
        let mut mouse = ButtonInput::default();
        mouse.press(GameAction::Primary);
        mouse.clear();
        mouse.release(GameAction::Primary);

        handle_block_actions(&mouse, &mut graph, &mut state, &mut history);

        assert_eq!(graph.part_count(), 6);
        assert_eq!(graph.weld_count(), 13);
        assert_eq!(history.undo.len(), 1);
        apply_history_action(HistoryAction::Undo, &mut graph, &mut state, &mut history);
        assert_eq!(graph.part_count(), 0);
        assert_eq!(graph.weld_count(), 0);
    }

    #[test]
    fn wiring_picks_a_bearing_through_the_hole_the_ring_pick_misses() {
        let mut graph = ConstructionGraph::new();
        let support = CuboidSpec::new(
            [4, 4, 4],
            BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(support)).unwrap() else {
            unreachable!()
        };
        let bearing = PlacedBearing {
            kind: mechanic_core::BearingKind::Rotational,
            axis: Vec3::ZERO,
            source: FaceRef::part(part, FaceKind::PositiveY),
            anchor: Vec3::Y,
            dimensions: BearingDimensions::default(),
        };

        // Straight down the axis passes through the hole, and whatever is
        // threaded through it, so the ring pick finds nothing there.
        let axis = Vec3::new(0.0, 3.0, 0.0);
        assert!(raycast_placed_bearings(&graph, &[bearing], axis, Vec3::NEG_Y).is_none());
        assert_eq!(
            raycast_placed_bearing_discs(&graph, &[bearing], axis, Vec3::NEG_Y).map(|hit| hit.0),
            Some(0)
        );

        // Past the rim it still misses, so the disc does not swallow the block.
        let outside = Vec3::new(bearing.dimensions.outer_diameter(), 3.0, 0.0);
        assert!(raycast_placed_bearing_discs(&graph, &[bearing], outside, Vec3::NEG_Y).is_none());
    }

    #[test]
    fn connector_pick_follows_a_simulated_bearing() {
        let mut graph = ConstructionGraph::new();
        let support = CuboidSpec::new(
            [4, 4, 4],
            BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(support)).unwrap() else {
            unreachable!()
        };
        let bearing = PlacedBearing {
            kind: mechanic_core::BearingKind::Rotational,
            axis: Vec3::ZERO,
            source: FaceRef::part(part, FaceKind::PositiveY),
            anchor: Vec3::Y,
            dimensions: BearingDimensions::default(),
        };
        let creation = graph.compile().unwrap();
        let compound = creation
            .part_to_compound
            .iter()
            .find_map(|&(candidate, compound)| (candidate == part).then_some(compound))
            .unwrap();
        let translation = Vec3::new(4.0, 0.0, 0.0);
        let mut transforms = creation
            .compounds
            .iter()
            .map(|compound| GpuTransform {
                position: compound.root_translation.extend(0.0).to_array(),
                rotation: compound.root_rotation.to_array(),
            })
            .collect::<Vec<_>>();
        transforms[compound as usize].position =
            (creation.compounds[compound as usize].root_translation + translation)
                .extend(0.0)
                .to_array();
        let ray_origin = bearing.anchor + translation + Vec3::Y * 3.0;

        assert!(
            raycast_placed_bearing_discs(&graph, &[bearing], ray_origin, Vec3::NEG_Y).is_none(),
            "the authored bearing no longer sits under the pointer"
        );
        assert_eq!(
            raycast_placed_bearing_discs_with_pose(
                &[bearing],
                ray_origin,
                Vec3::NEG_Y,
                |bearing| simulation_placed_bearing_pose(&graph, &creation, &transforms, bearing,),
            )
            .map(|hit| hit.0),
            Some(0),
            "the connector should pick the bearing at its simulated pose"
        );
    }

    #[test]
    fn placed_bearing_is_picked_before_support_and_attaches_on_release() {
        let mut graph = ConstructionGraph::new();
        let support = CuboidSpec::new(
            [4, 4, 4],
            BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(support)).unwrap() else {
            unreachable!()
        };
        let bearing = PlacedBearing {
            kind: mechanic_core::BearingKind::Rotational,
            axis: Vec3::ZERO,
            source: FaceRef::part(part, FaceKind::PositiveY),
            anchor: Vec3::Y,
            dimensions: BearingDimensions::default(),
        };
        let mut state = EditorState {
            placed_bearings: vec![bearing],
            hovered_bearing: Some(0),
            attachment_bearing: Some(0),
            pointer_position: Some(Vec2::ZERO),
            pointer_ray: Some((Vec3::new(0.1, 3.0, 0.0), Vec3::NEG_Y)),
            preview: Some(bearing_attachment_candidate(
                &graph,
                bearing.source,
                bearing.anchor,
            )),
            ..Default::default()
        };

        let origin = Vec3::new(0.1, 3.0, 0.0);
        let (_, bearing_distance) =
            raycast_placed_bearings(&graph, &state.placed_bearings, origin, Vec3::NEG_Y).unwrap();
        let support_distance = raycast_construction(&graph, origin, Vec3::NEG_Y)
            .unwrap()
            .distance;
        assert!(bearing_distance < support_distance);
        assert!(
            raycast_placed_bearings(
                &graph,
                &state.placed_bearings,
                Vec3::new(0.0, 3.0, 0.0),
                Vec3::NEG_Y,
            )
            .is_none()
        );
        let tiny_hole = PlacedBearing {
            dimensions: BearingDimensions::new(0.25, 0.001).unwrap(),
            ..bearing
        };
        assert!(
            raycast_placed_bearings(&graph, &[tiny_hole], Vec3::new(0.0, 3.0, 0.0), Vec3::NEG_Y,)
                .is_none()
        );
        assert_eq!(graph.part_count(), 1);
        assert_eq!(graph.bearing_count(), 0);

        let mut mouse = ButtonInput::default();
        let mut history = EditorHistory::default();
        mouse.press(GameAction::Primary);
        handle_block_actions(&mouse, &mut graph, &mut state, &mut history);
        assert_eq!(graph.part_count(), 1);
        assert_eq!(graph.bearing_count(), 0);
        assert_eq!(state.placed_bearings.len(), 1);
        assert!(state.block_drag.is_some());

        mouse.clear();
        mouse.release(GameAction::Primary);
        handle_block_actions(&mouse, &mut graph, &mut state, &mut history);

        assert_eq!(state.placed_bearings, vec![bearing]);
        assert_eq!(graph.part_count(), 2);
        assert_eq!(graph.bearing_count(), 1);
        assert_eq!(graph.weld_count(), 0);
    }

    #[test]
    fn oversized_bearing_claims_offset_block_preview_and_highlights_attachment() {
        let mut graph = ConstructionGraph::new();
        let support = CuboidSpec::new(
            [4, 4, 4],
            BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(support)).unwrap() else {
            unreachable!()
        };
        let bearing = PlacedBearing {
            kind: mechanic_core::BearingKind::Rotational,
            axis: Vec3::ZERO,
            source: FaceRef::part(part, FaceKind::PositiveY),
            anchor: Vec3::Y,
            dimensions: BearingDimensions::new(0.80, 0.10).unwrap(),
        };
        let mut state = EditorState {
            hovered: Some(SurfaceHit {
                distance: 1.0,
                point: Vec3::new(0.36, 1.0, 0.0),
                face: bearing.source,
            }),
            placed_bearings: vec![bearing],
            pointer_position: Some(Vec2::ZERO),
            pointer_ray: Some((Vec3::new(0.36, 3.0, 0.0), Vec3::NEG_Y)),
            ..Default::default()
        };

        refresh_tool_preview(&graph, &mut state, Tool::Block);

        assert_eq!(state.hovered_bearing, None);
        assert_eq!(state.attachment_bearing, Some(0));
        assert!(state.preview_error.is_none());
        assert!(bearing_attachment_is_highlighted(
            Tool::Block,
            state.attachment_bearing,
            state.preview_error.as_ref(),
        ));
        let preview = state.preview.unwrap();
        assert!((preview.spec.pose.translation().x - 0.375).abs() < 1.0e-6);

        let mut mouse = ButtonInput::default();
        let mut history = EditorHistory::default();
        mouse.press(GameAction::Primary);
        handle_block_actions(&mouse, &mut graph, &mut state, &mut history);
        mouse.clear();
        mouse.release(GameAction::Primary);
        handle_block_actions(&mouse, &mut graph, &mut state, &mut history);

        assert_eq!(state.placed_bearings, vec![bearing]);
        assert_eq!(graph.bearing_count(), 1);
        assert_eq!(graph.weld_count(), 0);
        assert_eq!(
            graph.bearings().next().unwrap().1.dimensions,
            bearing.dimensions
        );
    }

    #[test]
    fn bearing_claims_an_offset_pipe_preview_but_centres_it_by_default() {
        let mut graph = ConstructionGraph::new();
        let support = CuboidSpec::new(
            [4, 4, 4],
            BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(support)).unwrap() else {
            unreachable!()
        };
        let bearing = PlacedBearing {
            kind: mechanic_core::BearingKind::Rotational,
            axis: Vec3::ZERO,
            source: FaceRef::part(part, FaceKind::PositiveY),
            anchor: Vec3::Y,
            dimensions: BearingDimensions::new(0.80, 0.10).unwrap(),
        };
        let mut state = EditorState {
            hovered: Some(SurfaceHit {
                distance: 1.0,
                point: Vec3::new(0.36, 1.0, 0.0),
                face: bearing.source,
            }),
            placed_bearings: vec![bearing],
            pointer_position: Some(Vec2::ZERO),
            pointer_ray: Some((Vec3::new(0.36, 3.0, 0.0), Vec3::NEG_Y)),
            ..Default::default()
        };

        refresh_tool_preview(&graph, &mut state, Tool::Cylinder);

        assert_eq!(state.attachment_bearing, Some(0));
        let preview = state.cylinder_preview.unwrap();
        let direction = preview.spec.pose.rotation.quaternion() * Vec3::Y;
        let inlet_center = preview.spec.pose.translation()
            - direction * preview.spec.dimensions.axial_length() * 0.5;
        assert!(inlet_center.abs_diff_eq(bearing.anchor, 1.0e-5));
    }

    #[test]
    fn right_click_through_bearing_hole_deletes_block_but_keeps_bearing() {
        let mut graph = ConstructionGraph::new();
        let parts = [IVec3::new(0, 1, 0), IVec3::new(2, 1, 0)].map(|center| {
            let spec = CuboidSpec::new(
                [1; 3],
                BuildPose::from_half_grid(center, GridRotation::default()),
            )
            .unwrap();
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                unreachable!()
            };
            part
        });
        let center_face = FaceRef::part(parts[0], FaceKind::PositiveY);
        let bearing = PlacedBearing {
            kind: mechanic_core::BearingKind::Rotational,
            axis: Vec3::ZERO,
            source: FaceRef::part(parts[1], FaceKind::PositiveY),
            anchor: Vec3::new(0.0, 0.25, 0.0),
            dimensions: BearingDimensions::new(0.75, 0.40).unwrap(),
        };
        let state = EditorState {
            hovered: Some(SurfaceHit {
                distance: 1.0,
                point: bearing.anchor,
                face: center_face,
            }),
            hovered_bearing: None,
            attachment_bearing: Some(0),
            placed_bearings: vec![bearing],
            // A delete drag anchors on the press, so it needs the pointer.
            pointer_position: Some(Vec2::ZERO),
            pointer_ray: Some((Vec3::Y, Vec3::NEG_Y)),
            ..Default::default()
        };
        let mut mouse = ButtonInput::default();
        mouse.press(GameAction::Secondary);
        let mut app = App::new();
        app.insert_resource(mouse)
            .insert_resource(ButtonInput::<KeyCode>::default())
            .insert_resource(EditorGraph(graph))
            .insert_resource(state)
            .insert_resource(EditorHistory::default())
            .insert_resource(crate::chroma::ChromaBrush::default())
            .insert_resource(AppSimulation::default())
            .insert_resource(SelectedTool::from_editor_tool(Tool::Block))
            .insert_resource(BearingToolSettings::default())
            .insert_resource(CylinderToolSettings::default())
            .insert_resource(crate::ui::UiInput::default())
            .insert_resource(MaterialWheelState::default())
            .insert_resource(PlayerState {
                input_captured: true,
                ..Default::default()
            })
            .init_resource::<bevy::input::mouse::AccumulatedMouseMotion>()
            .add_systems(Update, handle_build_actions);

        app.update();
        {
            let state = app.world().resource::<EditorState>();
            assert!(state.delete_target.is_none());
            assert!(state.delete_drag.is_some());
        }
        {
            let mut mouse = app.world_mut().resource_mut::<ButtonInput<GameAction>>();
            mouse.clear();
            mouse.release(GameAction::Secondary);
        }
        app.update();

        let graph = app.world().resource::<EditorGraph>();
        let state = app.world().resource::<EditorState>();
        assert!(graph.0.part(parts[0]).is_none());
        assert!(graph.0.part(parts[1]).is_some());
        assert_eq!(state.placed_bearings, vec![bearing]);
    }

    #[test]
    fn deleting_current_support_rehomes_bearing_to_remaining_ring_support() {
        let mut graph = ConstructionGraph::new();
        let supports = [IVec3::new(-1, 1, 0), IVec3::new(1, 1, 0)].map(|center| {
            let spec = CuboidSpec::new(
                [1, 1, 1],
                BuildPose::from_half_grid(center, GridRotation::default()),
            )
            .unwrap();
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                unreachable!()
            };
            part
        });
        let target_spec = CuboidSpec::new(
            [1, 1, 1],
            BuildPose::from_half_grid(IVec3::new(0, 3, 0), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(target) = graph.apply(BuildCommand::Spawn(target_spec)).unwrap()
        else {
            unreachable!()
        };
        let socket = PlacedBearing {
            kind: mechanic_core::BearingKind::Rotational,
            axis: Vec3::ZERO,
            source: FaceRef::part(supports[0], FaceKind::PositiveY),
            anchor: Vec3::new(0.0, 0.25, 0.0),
            dimensions: BearingDimensions::new(0.50, 0.10).unwrap(),
        };
        graph
            .apply(BuildCommand::AddBearing(
                mechanic_core::BearingSpec::new(
                    socket.source,
                    FaceRef::part(target, FaceKind::NegativeY),
                    socket.anchor,
                    Vec3::Y,
                )
                .with_dimensions(socket.dimensions),
            ))
            .unwrap();

        let (graph, sockets, migrated) =
            stage_part_deletion_preserving_bearings(&graph, &[socket], &[supports[0]]).unwrap();

        assert_eq!(migrated, 1);
        assert!(graph.part(supports[0]).is_none());
        assert!(graph.part(supports[1]).is_some());
        assert_eq!(sockets.len(), 1);
        assert_eq!(
            sockets[0].source,
            FaceRef::part(supports[1], FaceKind::PositiveY)
        );
        let bearing = graph.bearings().next().unwrap().1;
        assert_eq!(bearing.source, sockets[0].source);
        assert_eq!(bearing.target, FaceRef::part(target, FaceKind::NegativeY));
        assert_eq!(graph.compile().unwrap().bearings.len(), 1);

        let (graph, sockets, migrated) =
            stage_part_deletion_preserving_bearings(&graph, &sockets, &[supports[1]]).unwrap();
        assert_eq!(migrated, 0);
        assert!(sockets.is_empty());
        assert_eq!(graph.bearing_count(), 0);
        assert!(graph.part(target).is_some());
    }

    #[test]
    fn deleting_linear_support_preserves_occupied_side_and_travel_axis() {
        use mechanic_core::{BearingKind, CarriageFace, LinearBearing, LinearBearingDimensions};
        let mut graph = ConstructionGraph::new();
        let supports = [IVec3::new(0, 1, 0), IVec3::new(1, 1, 0)].map(|center| {
            let spec =
                CuboidSpec::new([1; 3], BuildPose::new(center, GridRotation::default())).unwrap();
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                unreachable!()
            };
            part
        });
        let rail = LinearBearing {
            dimensions: LinearBearingDimensions::new(1.0, 0.4).unwrap(),
            mount_normal: Vec3::Y,
            face: CarriageFace::PositiveSide,
        };
        let socket = PlacedBearing {
            source: FaceRef::part(supports[0], FaceKind::PositiveY),
            anchor: Vec3::new(0.0, 0.375, 0.0),
            axis: Vec3::X,
            dimensions: BearingDimensions::default(),
            kind: BearingKind::Linear(LinearBearing {
                face: CarriageFace::Top,
                ..rail
            }),
        };
        let surface =
            super::builder::linear_carriage_face(socket.anchor, rail, socket.axis).unwrap();
        let candidate = super::builder::linear_block_candidate(
            socket.anchor,
            rail,
            socket.axis,
            surface.center,
            [1; 3],
            GridRotation::default(),
        )
        .unwrap();
        let graph = super::builder::stage_linear_block_batch_in_bounds(
            &graph,
            candidate,
            &[candidate.spec],
            super::builder::LinearAttachment {
                source: socket.source,
                anchor: socket.anchor,
                rail,
                axis: socket.axis,
                rigid_targets: &[],
            },
            super::PlacementBounds::Garage,
        )
        .unwrap();
        let (graph, sockets, migrated) =
            stage_part_deletion_preserving_bearings(&graph, &[socket], &[supports[0]]).unwrap();
        assert_eq!(migrated, 1);
        assert_eq!(
            sockets[0].source.owner,
            mechanic_core::FaceOwner::Part(supports[1])
        );
        let bearing = graph.bearings().next().unwrap().1;
        assert_eq!(bearing.kind, BearingKind::Linear(rail));
        assert_eq!(bearing.axis, Vec3::X);
        assert_eq!(graph.compile().unwrap().bearings.len(), 1);
    }

    #[test]
    fn deleting_reusable_socket_removes_all_of_its_joint_attachments() {
        let mut graph = ConstructionGraph::new();
        let support_spec = CuboidSpec::new(
            [4, 4, 4],
            BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(support) =
            graph.apply(BuildCommand::Spawn(support_spec)).unwrap()
        else {
            unreachable!()
        };
        let targets = [IVec3::new(0, 9, 0), IVec3::new(2, 9, 0)].map(|center| {
            let target_spec = CuboidSpec::new(
                [1, 1, 1],
                BuildPose::from_half_grid(center, GridRotation::default()),
            )
            .unwrap();
            let BuildOutcome::Spawned(target) =
                graph.apply(BuildCommand::Spawn(target_spec)).unwrap()
            else {
                unreachable!()
            };
            target
        });
        let socket = PlacedBearing {
            kind: mechanic_core::BearingKind::Rotational,
            axis: Vec3::ZERO,
            source: FaceRef::part(support, FaceKind::PositiveY),
            anchor: Vec3::Y,
            dimensions: BearingDimensions::new(0.80, 0.10).unwrap(),
        };
        for target in targets {
            graph
                .apply(BuildCommand::AddBearing(
                    mechanic_core::BearingSpec::new(
                        socket.source,
                        FaceRef::part(target, FaceKind::NegativeY),
                        socket.anchor,
                        Vec3::Y,
                    )
                    .with_dimensions(socket.dimensions),
                ))
                .unwrap();
        }
        graph
            .apply(BuildCommand::RigidLink(RigidLinkSpec {
                first: targets[0],
                second: targets[1],
            }))
            .unwrap();
        let state = EditorState {
            hovered_bearing: Some(0),
            placed_bearings: vec![socket],
            ..Default::default()
        };
        let mut mouse = ButtonInput::default();
        mouse.press(GameAction::Secondary);
        let mut app = App::new();
        app.insert_resource(mouse)
            .insert_resource(ButtonInput::<KeyCode>::default())
            .insert_resource(EditorGraph(graph))
            .insert_resource(state)
            .insert_resource(EditorHistory::default())
            .insert_resource(crate::chroma::ChromaBrush::default())
            .insert_resource(AppSimulation::default())
            .insert_resource(SelectedTool::from_editor_tool(Tool::Block))
            .insert_resource(BearingToolSettings::default())
            .insert_resource(CylinderToolSettings::default())
            .insert_resource(crate::ui::UiInput::default())
            .insert_resource(MaterialWheelState::default())
            .insert_resource(PlayerState {
                input_captured: true,
                ..Default::default()
            })
            .init_resource::<bevy::input::mouse::AccumulatedMouseMotion>()
            .add_systems(Update, handle_build_actions);

        app.update();
        {
            let mut mouse = app.world_mut().resource_mut::<ButtonInput<GameAction>>();
            mouse.clear();
            mouse.release(GameAction::Secondary);
        }
        app.update();

        let graph = app.world().resource::<EditorGraph>();
        let state = app.world().resource::<EditorState>();
        assert_eq!(graph.0.part_count(), 3);
        assert_eq!(graph.0.bearing_count(), 0);
        assert_eq!(graph.0.rigid_link_count(), 0);
        assert!(state.placed_bearings.is_empty());
    }

    #[test]
    fn delete_drag_uses_composed_centers_instead_of_other_frames_local_grid() {
        let mut graph = ConstructionGraph::new();
        let start = CuboidSpec::new([1; 3], BuildPose::default()).unwrap();
        let BuildOutcome::Spawned(local) = graph.apply(BuildCommand::Spawn(start)).unwrap() else {
            unreachable!()
        };
        let BuildOutcome::Spawned(remote) = graph.apply(BuildCommand::Spawn(start)).unwrap() else {
            unreachable!()
        };
        graph
            .reframe_parts(
                [remote],
                mechanic_core::ConstructionFrame::new(Vec3::X * 10.0, Quat::IDENTITY).unwrap(),
            )
            .unwrap();
        assert_eq!(
            delete_box_parts(&graph, start, IVec3::ZERO).unwrap(),
            vec![local]
        );
    }

    #[test]
    fn delete_drag_selects_only_the_box_it_spans() {
        let mut graph = ConstructionGraph::new();
        let centers = [
            IVec3::new(1, 1, 1),
            IVec3::new(3, 1, 1),
            IVec3::new(1, 1, 3),
            IVec3::new(3, 1, 3),
            IVec3::new(1, 3, 1),
        ];
        let mut parts = Vec::new();
        for center in centers {
            let spec = CuboidSpec::new(
                [1; 3],
                BuildPose::from_half_grid(center, GridRotation::default()),
            )
            .unwrap();
            let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap()
            else {
                unreachable!()
            };
            parts.push(part);
        }
        let start = graph.part(parts[0]).copied().unwrap().as_cuboid().unwrap();

        // A flat span is the four in that plane; the block above is untouched.
        let flat = delete_box_parts(&graph, start, IVec3::new(1, 0, 1)).unwrap();
        assert_eq!(flat.len(), 4);
        assert!(!flat.contains(&parts[4]));

        // Rotating into the third axis reaches the one above too.
        let boxed = delete_box_parts(&graph, start, IVec3::new(1, 1, 1)).unwrap();
        assert_eq!(boxed.len(), 5);
        assert!(boxed.contains(&parts[4]));
    }

    fn wired_socket_graph() -> (ConstructionGraph, PlacedBearing, PartId) {
        let mut graph = ConstructionGraph::new();
        let spawn = |graph: &mut ConstructionGraph, x: i32| {
            let BuildOutcome::Spawned(id) = graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new(
                        [4, 4, 4],
                        BuildPose::new(IVec3::new(x, 2, 0), GridRotation::default()),
                    )
                    .unwrap(),
                ))
                .unwrap()
            else {
                unreachable!()
            };
            id
        };
        let left = spawn(&mut graph, 0);
        let right = spawn(&mut graph, 4);
        let source = FaceRef::part(left, FaceKind::PositiveX);
        let anchor = Vec3::new(0.5, 0.5, 0.0);
        graph
            .apply(BuildCommand::AddBearing(mechanic_core::BearingSpec::new(
                source,
                FaceRef::part(right, FaceKind::NegativeX),
                anchor,
                Vec3::X,
            )))
            .unwrap();
        let BuildOutcome::Spawned(controller) = graph
            .apply(BuildCommand::SpawnController(ControllerSpec::new(
                BuildPose::new(IVec3::new(0, 12, 0), GridRotation::default()),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        let socket = PlacedBearing {
            kind: mechanic_core::BearingKind::Rotational,
            axis: Vec3::ZERO,
            source,
            anchor,
            dimensions: BearingDimensions::default(),
        };
        (graph, socket, controller)
    }

    #[test]
    fn dragging_a_control_block_onto_a_bearing_wires_every_row_of_that_socket() {
        let (mut graph, socket, controller) = wired_socket_graph();
        let mut state = EditorState {
            hovered_bearing: Some(0),
            placed_bearings: vec![socket],
            ..Default::default()
        };
        let mut history = EditorHistory::default();
        let block = WireEnd::Controller(controller);

        assert_eq!(
            wire_drag_step(None, Some(block), true),
            WireDragStep::Begin(block)
        );
        assert_eq!(
            wire_drag_step(
                Some(WireDrag {
                    from: block,
                    armed: false
                }),
                Some(WireEnd::Bearing(0)),
                false
            ),
            WireDragStep::Connect(WireConnection::Drive {
                controller,
                bearing: 0
            })
        );

        let message = connect_drive_wire(&mut graph, &mut state, &mut history, controller, 0);
        assert!(message.contains("Wired"), "{message}");
        assert_eq!(graph.drive_link_count(), 1);
        assert!(state.construction_mesh_dirty);
    }

    #[test]
    fn making_the_same_control_connection_twice_removes_it() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(input) = graph
            .apply(BuildCommand::SpawnInput(mechanic_core::InputSpec::new(
                BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        let BuildOutcome::Spawned(seat) = graph
            .apply(BuildCommand::SpawnSeat(mechanic_core::SeatSpec::new(
                BuildPose::new(IVec3::new(8, 2, 0), GridRotation::default()),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        let BuildOutcome::Spawned(controller) = graph
            .apply(BuildCommand::SpawnController(ControllerSpec::new(
                BuildPose::new(IVec3::new(16, 2, 0), GridRotation::default()),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        let mut state = EditorState::default();
        let mut history = EditorHistory::default();

        let message = connect_control_link(
            &mut graph,
            &mut state,
            &mut history,
            BuildCommand::AddInputSeatLink(mechanic_core::InputSeatLinkSpec { input, seat }),
            "Linked Input to Seat",
        );

        assert_eq!(message, "Linked Input to Seat");
        assert_eq!(graph.input_seat_links().count(), 1);
        // The wire is a line in the drive overlay, and that overlay is only
        // rebuilt on request: without the flag it stays invisible until some
        // unrelated edit happens to dirty the mesh.
        assert!(state.construction_mesh_dirty);

        state.construction_mesh_dirty = false;
        let message = connect_control_link(
            &mut graph,
            &mut state,
            &mut history,
            BuildCommand::AddInputSeatLink(mechanic_core::InputSeatLinkSpec { input, seat }),
            "Linked Input to Seat",
        );

        assert_eq!(message, "Removed Input-to-Seat link");
        assert_eq!(graph.input_seat_links().count(), 0);
        assert!(state.construction_mesh_dirty);

        for expected in [
            "Linked Seat to Controller",
            "Removed Seat-to-Controller link",
        ] {
            state.construction_mesh_dirty = false;
            let message = connect_control_link(
                &mut graph,
                &mut state,
                &mut history,
                BuildCommand::AddSeatControllerLink(mechanic_core::SeatControllerLinkSpec {
                    seat,
                    controller,
                }),
                "Linked Seat to Controller",
            );

            assert_eq!(message, expected);
            assert!(state.construction_mesh_dirty);
        }
        assert_eq!(graph.seat_controller_links().count(), 0);
    }

    #[test]
    fn a_wire_can_be_dragged_from_the_bearing_end_as_well() {
        let (_, _, controller) = wired_socket_graph();
        let drag = Some(WireDrag {
            from: WireEnd::Bearing(0),
            armed: false,
        });
        assert_eq!(
            wire_drag_step(drag, Some(WireEnd::Controller(controller)), false),
            WireDragStep::Connect(WireConnection::Drive {
                controller,
                bearing: 0
            })
        );
        // Two ends of the same kind never pair up.
        assert_eq!(
            wire_drag_step(drag, Some(WireEnd::Bearing(1)), false),
            WireDragStep::Cancel
        );
    }

    #[test]
    fn releasing_where_the_wire_started_leaves_it_armed_for_a_second_click() {
        let (_, _, controller) = wired_socket_graph();
        let block = WireEnd::Controller(controller);
        let drag = WireDrag {
            from: block,
            armed: false,
        };

        assert_eq!(
            wire_drag_step(Some(drag), Some(block), false),
            WireDragStep::Arm
        );
        assert_eq!(
            wire_drag_step(
                Some(WireDrag {
                    armed: true,
                    ..drag
                }),
                Some(WireEnd::Bearing(0)),
                true
            ),
            WireDragStep::Connect(WireConnection::Drive {
                controller,
                bearing: 0
            })
        );
        // Letting go over empty space drops the wire instead.
        assert_eq!(
            wire_drag_step(Some(drag), None, false),
            WireDragStep::Cancel
        );
    }

    #[test]
    fn making_the_same_drive_connection_twice_removes_it() {
        let (mut graph, socket, controller) = wired_socket_graph();
        let bearing = graph.bearings().next().unwrap().0;
        graph
            .apply(BuildCommand::AddDriveLink(DriveLinkSpec::new(
                controller, bearing,
            )))
            .unwrap();
        let mut state = EditorState {
            hovered_bearing: Some(0),
            placed_bearings: vec![socket],
            ..Default::default()
        };
        let mut history = EditorHistory::default();

        let message = connect_drive_wire(&mut graph, &mut state, &mut history, controller, 0);
        assert!(message.contains("Removed"), "{message}");
        assert_eq!(graph.drive_link_count(), 0);
        assert!(graph.bearing_drive_link(bearing).is_none());
        assert!(state.construction_mesh_dirty);
    }

    #[test]
    fn right_clicking_a_wired_bearing_changes_its_default_direction() {
        let (mut graph, socket, controller) = wired_socket_graph();
        let bearing = graph.bearings().next().unwrap().0;
        graph
            .apply(BuildCommand::AddDriveLink(DriveLinkSpec::new(
                controller, bearing,
            )))
            .unwrap();
        let state = EditorState {
            hovered_bearing: Some(0),
            placed_bearings: vec![socket],
            ..Default::default()
        };
        let mut mouse = ButtonInput::default();
        mouse.press(GameAction::Secondary);
        let mut app = App::new();
        app.insert_resource(mouse)
            .insert_resource(ButtonInput::<KeyCode>::default())
            .insert_resource(EditorGraph(graph))
            .insert_resource(state)
            .insert_resource(EditorHistory::default())
            .insert_resource(crate::chroma::ChromaBrush::default())
            .insert_resource(AppSimulation::default())
            .insert_resource(SelectedTool::from_editor_tool(Tool::Block))
            .insert_resource(BearingToolSettings::default())
            .insert_resource(CylinderToolSettings::default())
            .insert_resource(crate::ui::UiInput::default())
            .insert_resource(MaterialWheelState::default())
            .insert_resource(PlayerState {
                input_captured: true,
                ..Default::default()
            })
            .init_resource::<bevy::input::mouse::AccumulatedMouseMotion>()
            .add_systems(Update, handle_build_actions);

        app.update();
        {
            let mut mouse = app.world_mut().resource_mut::<ButtonInput<GameAction>>();
            mouse.clear();
            mouse.release(GameAction::Secondary);
        }
        app.update();

        let graph = app.world().resource::<EditorGraph>();
        let state = app.world().resource::<EditorState>();
        assert_eq!(graph.0.drive_link_count(), 1);
        assert!(graph.0.drive_links().next().unwrap().1.reversed);
        assert_eq!(state.placed_bearings, vec![socket]);
        assert!(
            state
                .feedback
                .as_deref()
                .is_some_and(|message| message.contains("default direction"))
        );
    }

    #[test]
    fn wiring_an_unattached_socket_reports_that_it_has_no_joint_yet() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(block) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [4, 4, 4],
                    BuildPose::new(IVec3::new(0, 2, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let BuildOutcome::Spawned(controller) = graph
            .apply(BuildCommand::SpawnController(ControllerSpec::new(
                BuildPose::new(IVec3::new(0, 12, 0), GridRotation::default()),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        let mut state = EditorState {
            hovered_bearing: Some(0),
            placed_bearings: vec![PlacedBearing {
                kind: mechanic_core::BearingKind::Rotational,
                axis: Vec3::ZERO,
                source: FaceRef::part(block, FaceKind::PositiveX),
                anchor: Vec3::new(0.5, 0.5, 0.0),
                dimensions: BearingDimensions::default(),
            }],
            ..Default::default()
        };
        let mut history = EditorHistory::default();

        let message = connect_drive_wire(&mut graph, &mut state, &mut history, controller, 0);
        assert!(message.contains("Attach a part"), "{message}");
        assert_eq!(graph.drive_link_count(), 0);
    }

    #[test]
    fn control_block_status_line_reports_how_many_bearings_are_wired() {
        let selected = tool_status_line(
            Tool::Controller,
            BearingDimensions::default(),
            CylinderDimensions::default(),
            Some(2),
            mechanic_core::ConstructionMaterial::Steel,
        );
        assert!(selected.contains("2 bearings wired"), "{selected}");
        assert!(
            selected.contains("Interact opens its program"),
            "{selected}"
        );

        let single = tool_status_line(
            Tool::Connector,
            BearingDimensions::default(),
            CylinderDimensions::default(),
            Some(1),
            mechanic_core::ConstructionMaterial::Steel,
        );
        assert!(single.contains("1 bearing wired"), "{single}");

        let none = tool_status_line(
            Tool::Controller,
            BearingDimensions::default(),
            CylinderDimensions::default(),
            None,
            mechanic_core::ConstructionMaterial::Steel,
        );
        assert!(none.contains("No block selected"), "{none}");
    }

    #[test]
    fn pipette_copies_material_dimensions_authored_orientation_and_bearing_setup() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(cuboid) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1; 3],
                    BuildPose::new(IVec3::new(0, 1, 0), GridRotation::default()),
                )
                .unwrap()
                .with_material(ConstructionMaterial::Wood),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let cylinder_dimensions = CylinderDimensions::new(0.75, 0.25, 1.5).unwrap();
        let BuildOutcome::Spawned(cylinder) = graph
            .apply(BuildCommand::SpawnCylinder(
                CylinderSpec::new(
                    cylinder_dimensions,
                    BuildPose::new(IVec3::new(8, 2, 0), GridRotation::default()),
                )
                .with_material(ConstructionMaterial::Concrete),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let orientation = AUTHORED_ORIENTATIONS[17];
        let BuildOutcome::Spawned(controller) = graph
            .apply(BuildCommand::SpawnController(ControllerSpec::new(
                BuildPose::new(IVec3::new(16, 2, 0), orientation),
            )))
            .unwrap()
        else {
            unreachable!()
        };

        let shaped_spec = CuboidSpec::new(
            [1; 3],
            BuildPose::new(IVec3::new(24, 1, 0), GridRotation::default()),
        )
        .unwrap()
        .with_material(ConstructionMaterial::Concrete);
        let BuildOutcome::Spawned(shaped) = graph.apply(BuildCommand::Spawn(shaped_spec)).unwrap()
        else {
            unreachable!()
        };
        let BuildOutcome::RegionAdded(region) = graph
            .apply(BuildCommand::AddRegion(
                crate::editor::shape_actions::region_area(shaped_spec, IVec3::ZERO),
            ))
            .unwrap()
        else {
            unreachable!()
        };

        let mut state = EditorState::default();
        let mut selection = SelectedTool::default();
        selection.clear();
        let mut material = super::SelectedMaterial(ConstructionMaterial::Steel);
        let mut bearing = BearingToolSettings::default();
        let mut cylinder_settings = CylinderToolSettings::default();
        macro_rules! apply {
            ($setup:expr) => {
                crate::editor::shortcuts::apply_pipette_setup(
                    $setup,
                    &graph,
                    &mut state,
                    &mut selection,
                    &mut material,
                    &mut bearing,
                    &mut cylinder_settings,
                )
            };
        }

        apply!(crate::editor::shortcuts::PipetteSetup::Part(cuboid));
        assert_eq!(selection.active_editor_tool(), Some(Tool::Block));
        assert_eq!(material.0, ConstructionMaterial::Wood);

        apply!(crate::editor::shortcuts::PipetteSetup::Part(cylinder));
        assert_eq!(selection.active_editor_tool(), Some(Tool::Cylinder));
        assert_eq!(material.0, ConstructionMaterial::Concrete);
        assert_eq!(cylinder_settings.dimensions, cylinder_dimensions);

        apply!(crate::editor::shortcuts::PipetteSetup::Part(controller));
        assert_eq!(selection.active_editor_tool(), Some(Tool::Controller));
        assert_eq!(state.authored_orientation, 17);

        let dimensions = BearingDimensions::new(0.9, 0.4).unwrap();
        apply!(crate::editor::shortcuts::PipetteSetup::Bearing(dimensions));
        assert_eq!(selection.active_editor_tool(), Some(Tool::Bearing));
        assert_eq!(bearing.dimensions, dimensions);

        material.0 = ConstructionMaterial::Wood;
        apply!(crate::editor::shortcuts::PipetteSetup::Ground);
        assert_eq!(selection.active_editor_tool(), Some(Tool::Block));
        assert_eq!(material.0, ConstructionMaterial::Wood);

        apply!(crate::editor::shortcuts::PipetteSetup::Part(shaped));
        assert_eq!(selection.active_editor_tool(), Some(Tool::Shape));
        assert_eq!(state.active_region, Some(region));
        assert_eq!(material.0, ConstructionMaterial::Concrete);
    }

    #[test]
    fn pipette_uses_simulation_space_and_reports_no_target() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([1; 3], BuildPose::default()).unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        let creation = graph.compile().unwrap();
        let simulation = AppSimulation {
            transforms: vec![GpuTransform {
                position: [10.0, 0.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            }],
            creation: Some(creation),
            published_graph: graph.clone(),
            ..Default::default()
        };
        assert_eq!(
            crate::editor::shortcuts::pipette_at_ray(
                &graph,
                &EditorState::default(),
                &simulation,
                Vec3::new(10.0, 0.0, 5.0),
                Vec3::NEG_Z,
            ),
            Some(crate::editor::shortcuts::PipetteSetup::Part(part)),
        );
        assert_eq!(
            crate::editor::shortcuts::pipette_at_ray(
                &ConstructionGraph::new(),
                &EditorState::default(),
                &AppSimulation::default(),
                Vec3::Y,
                Vec3::Y,
            ),
            None,
        );
    }

    #[test]
    fn clearing_the_hand_cancels_pending_edits_and_hammer_charge() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([1; 3], BuildPose::default()).unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        crate::builder::begin_weld(&mut graph, FaceRef::part(part, FaceKind::PositiveY)).unwrap();
        let mut state = EditorState::default();
        let mut selection = SelectedTool::from_editor_tool(Tool::Weld);
        let mut hammer = crate::editor::hammer::HammerInteraction {
            charging: Some(crate::editor::hammer::HammerCharge {
                body_index: 0,
                local_point: Vec3::ZERO,
                direction: Vec3::Y,
                elapsed_seconds: 1.0,
                local_normal: Vec3::Y,
            }),
            pending: None,
        };
        crate::editor::shortcuts::clear_held_tool(
            &mut graph,
            &mut state,
            &mut selection,
            &mut hammer,
        );
        assert_eq!(selection.active_editor_tool(), None);
        assert!(graph.pending().is_none());
        assert!(hammer.charging.is_none() && hammer.pending.is_none());
    }

    #[test]
    fn the_panel_opens_on_a_hovered_control_block_and_blocks_the_keyboard() {
        let (graph, _, controller) = wired_socket_graph();
        let mut panel = crate::control_panel::ControlPanelState::default();
        assert!(!panel.is_open());
        assert!(!panel.blocks_keyboard());

        panel.open(controller);
        assert_eq!(panel.controller(), Some(controller));
        assert!(panel.blocks_keyboard(), "typing must not fire shortcuts");

        // One row per wired bearing, and none until the block is wired.
        assert!(crate::control_panel::panel_rows(&graph, controller).is_empty());

        panel.close();
        assert!(!panel.is_open());
    }

    #[test]
    fn the_panel_opens_on_a_moving_simulation_control_block() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(controller) = graph
            .apply(BuildCommand::SpawnController(ControllerSpec::new(
                BuildPose::default(),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        let creation = graph.compile().unwrap();
        let mut actions = ButtonInput::default();
        actions.press(GameAction::Interact);
        let mut state = crate::editor::state::EditorState {
            hovered_simulation: Some(SimulationHit {
                part: controller,
                body_index: 0,
                distance: 1.0,
                point: Vec3::ZERO,
                normal: Vec3::Y,
            }),
            ..Default::default()
        };
        // `update_hover` clears ordinary editor targeting when the live hit is
        // nearer than the authored surface. The live target must survive.
        clear_editor_hover(&mut state);
        let mut app = App::new();
        app.insert_resource(actions)
            .insert_resource(crate::creation_menu::CreationMenuState::default())
            .insert_resource(crate::editor::state::EditorGraph(graph.clone()))
            .insert_resource(state)
            .insert_resource(crate::control_panel::ControlPanelState::default())
            .insert_resource(State::new(crate::world::AppSpace::World))
            .insert_resource(crate::simulation::state::AppSimulation {
                creation: Some(creation),
                published_graph: graph,
                ..Default::default()
            })
            .insert_resource(PlayerState {
                input_captured: true,
                ..Default::default()
            })
            .insert_resource(MaterialWheelState::default())
            .insert_resource(crate::pause_menu::PauseMenuState::default())
            .init_resource::<SelectedTool>()
            .add_systems(
                Update,
                crate::editor::shortcuts::handle_control_panel_shortcut,
            );

        app.update();

        assert_eq!(
            app.world()
                .resource::<crate::control_panel::ControlPanelState>()
                .controller(),
            Some(controller)
        );
    }

    #[test]
    fn remembered_controller_does_not_capture_seat_or_empty_space_interactions() {
        let (mut graph, _, controller) = wired_socket_graph();
        let BuildOutcome::Spawned(seat) = graph
            .apply(BuildCommand::SpawnSeat(mechanic_core::SeatSpec::new(
                BuildPose::new(IVec3::new(20, 0, 0), GridRotation::default()),
            )))
            .unwrap()
        else {
            panic!("seat");
        };
        let mut actions = ButtonInput::default();
        actions.press(GameAction::Interact);
        let mut panel = crate::control_panel::ControlPanelState::default();
        panel.open(controller);
        panel.close();
        let mut app = App::new();
        app.insert_resource(actions)
            .insert_resource(crate::creation_menu::CreationMenuState::default())
            .insert_resource(crate::editor::state::EditorGraph(graph))
            .insert_resource(EditorState {
                selected_controller: Some(controller),
                ..Default::default()
            })
            .insert_resource(panel)
            .insert_resource(PlayerState {
                input_captured: true,
                ..Default::default()
            })
            .insert_resource(MaterialWheelState::default())
            .insert_resource(crate::pause_menu::PauseMenuState::default())
            .init_resource::<SelectedTool>()
            .add_systems(
                Update,
                crate::editor::shortcuts::handle_control_panel_shortcut,
            );

        for (aimed, seated) in [(None, false), (Some(seat), false), (Some(controller), true)] {
            app.world_mut()
                .resource_mut::<EditorState>()
                .hovered_simulation = aimed.map(|part| SimulationHit {
                part,
                body_index: 0,
                distance: 1.0,
                point: Vec3::ZERO,
                normal: Vec3::Y,
            });
            app.world_mut().resource_mut::<PlayerState>().seat = seated.then_some(seat);
            app.update();
            assert!(
                !app.world()
                    .resource::<crate::control_panel::ControlPanelState>()
                    .is_open(),
                "aim {aimed:?}, seated {seated}: remembered controller must not capture E",
            );
        }
    }

    #[test]
    fn every_wire_of_one_socket_is_written_by_a_single_row_edit() {
        let (mut graph, socket, controller) = wired_socket_graph();
        let mut state = EditorState {
            hovered_bearing: Some(0),
            placed_bearings: vec![socket],
            ..Default::default()
        };
        let mut history = EditorHistory::default();
        connect_drive_wire(&mut graph, &mut state, &mut history, controller, 0);

        let rows = crate::control_panel::panel_rows(&graph, controller);
        assert_eq!(rows.len(), 1, "one socket is one joint row");
        let commands = crate::control_panel::set_row_commands(
            &rows[0],
            mechanic_core::DriveLimits::new(2.0, 30.0, None).unwrap(),
            mechanic_core::DriveProgram::default(),
            mechanic_core::DriveName::new("Tipper arm"),
            mechanic_core::ActuatorAssignment::Unpowered,
        );
        assert_eq!(commands.len(), rows[0].links.len());
        graph.apply_batch(commands).unwrap();
        for (_, link) in graph.controller_links(controller) {
            assert!((link.limits.max_torque_newton_meters() - 30.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn pipe_validation_rechecks_changes_to_world_geometry_and_placement_bounds() {
        use super::PlacementBounds;
        let mut graph = ConstructionGraph::new();
        let mut cached = None;
        let pipe = CylinderSpec::new(
            CylinderDimensions::default(),
            BuildPose::new(IVec3::Y * 8, GridRotation::default()),
        );
        let pieces = [crate::builder::PipeRunPiece {
            spec: PartSpec::Cylinder(pipe),
            inlet: FaceKind::NegativeY,
            outlet: FaceKind::PositiveY,
        }];
        for _ in 0..2 {
            assert!(
                crate::editor::pipe::PipeValidation::validate(
                    &mut cached,
                    &graph,
                    &pieces,
                    PlacementBounds::Garage
                )
                .is_ok()
            );
        }
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::SpawnCylinder(pipe)).unwrap()
        else {
            unreachable!()
        };
        assert!(
            crate::editor::pipe::PipeValidation::validate(
                &mut cached,
                &graph,
                &pieces,
                PlacementBounds::Garage
            )
            .is_err()
        );
        graph.apply(BuildCommand::Remove(part)).unwrap();
        assert!(
            crate::editor::pipe::PipeValidation::validate(
                &mut cached,
                &graph,
                &pieces,
                PlacementBounds::Garage
            )
            .is_ok()
        );
        let distant = [crate::builder::PipeRunPiece {
            spec: pieces[0].spec.with_pose(BuildPose::new(
                IVec3::new(1000, 8, 0),
                GridRotation::default(),
            )),
            ..pieces[0]
        }];
        assert!(
            crate::editor::pipe::PipeValidation::validate(
                &mut cached,
                &graph,
                &distant,
                PlacementBounds::Garage
            )
            .is_err()
        );
        assert!(
            crate::editor::pipe::PipeValidation::validate(
                &mut cached,
                &graph,
                &distant,
                PlacementBounds::World {
                    origin: bevy::math::DVec2::ZERO
                }
            )
            .is_ok()
        );
    }

    #[test]
    fn pipe_dimension_modes_cycle_without_mutating_the_current_value() {
        let endpoint = Vec3::new(0.0, 1.0, 0.0);
        let dimensions = CylinderDimensions::new(0.50, 0.25, 1.0).unwrap();
        let mut mode = PipeEditMode::Length;
        for expected in [
            PipeEditMode::OuterDiameter,
            PipeEditMode::InnerDiameter,
            PipeEditMode::Length,
        ] {
            mode = mode.next();
            assert_eq!(mode, expected);
            assert_eq!(endpoint, Vec3::new(0.0, 1.0, 0.0));
            assert_eq!(
                dimensions,
                CylinderDimensions::new(0.50, 0.25, 1.0).unwrap()
            );
        }
    }

    #[test]
    fn pipe_bend_span_steps_freely_above_the_channel_width() {
        for outer_diameter in [0.05, 0.20, 0.25] {
            assert_eq!(constrained_pipe_bend_span(outer_diameter, 0), 1);
            assert_eq!(constrained_pipe_bend_span(outer_diameter, 3), 3);
        }
        assert_eq!(constrained_pipe_bend_span(0.30, 1), 2);
        assert_eq!(constrained_pipe_bend_span(0.30, 4), 4);
        assert_eq!(constrained_pipe_bend_span(0.20, 40), 32);
    }

    #[test]
    fn bend_corner_sits_at_the_middle_of_the_last_channel_cell() {
        assert!((pipe_corner_inset(0.25) - 0.125).abs() < f32::EPSILON);
        assert!((pipe_corner_inset(0.30) - 0.25).abs() < f32::EPSILON);
        assert!((pipe_corner_inset(0.60) - 0.375).abs() < f32::EPSILON);
    }

    #[test]
    fn widening_a_bent_pipe_rebases_corners_to_the_new_channel() {
        let mut corners = [Vec3::X * 0.875];
        let mut endpoint = Vec3::new(0.875, 0.875, 0.0);
        rebase_pipe_path(
            Vec3::ZERO,
            &mut corners,
            &mut endpoint,
            &[Vec3::X, Vec3::Y],
            &[2],
            pipe_corner_inset(0.25),
            pipe_corner_inset(0.50),
        );
        assert!(corners[0].abs_diff_eq(Vec3::X * 0.75, 1.0e-5));
        assert!(endpoint.abs_diff_eq(Vec3::new(0.75, 0.75, 0.0), 1.0e-5));
        let pieces = crate::builder::pipe_run_pieces(
            &[Vec3::ZERO, corners[0], endpoint],
            &[crate::builder::PipeNode::Bend { span: 2 }],
            CylinderDimensions::new(0.50, 0.0, 0.25).unwrap(),
            ConstructionMaterial::Steel,
        )
        .expect("the rebased run keeps whole-block straights");
        assert_eq!(pieces.len(), 3);
    }

    #[test]
    fn bearing_pipe_drag_moves_only_across_the_bearing_plane() {
        let camera = Vec3::new(0.0, 2.0, -4.0);
        let press = (Vec3::ZERO - camera).normalize();
        let current = (Vec3::new(0.31, 0.0, 0.12) - camera).normalize();

        let offset = bearing_offset_from_rays(
            Vec3::ZERO,
            Vec3::Y,
            PlacementGrid::Centimetres25,
            camera,
            press,
            camera,
            current,
        )
        .unwrap();

        assert!(offset.abs_diff_eq(Vec3::X * 0.25, 1.0e-5));
    }

    #[test]
    fn bearing_offset_drag_translates_the_whole_pipe_before_release() {
        let graph = ConstructionGraph::new();
        let dimensions = CylinderDimensions::default();
        let start = Vec3::ZERO;
        let endpoint = Vec3::Y * dimensions.axial_length();
        let camera = Vec3::new(0.0, 2.0, -4.0);
        let press_direction = (start - camera).normalize();
        let pieces = crate::builder::pipe_run_pieces(
            &[start, endpoint],
            &[],
            dimensions,
            ConstructionMaterial::Steel,
        )
        .unwrap();
        let mut state = EditorState {
            pipe_drag: Some(PipeDrag {
                attachment: BlockAttachment::Bearing {
                    source: FaceRef::ground(),
                    anchor: start,
                    dimensions: BearingDimensions::default(),
                },
                start,
                corners: Vec::new(),
                endpoint,
                directions: vec![Vec3::Y],
                nodes: Vec::new(),
                pending_span: 1,
                branch: None,
                dimensions,
                material: ConstructionMaterial::Steel,
                appearance: MaterialAppearance::BAKED,
                mode: PipeEditMode::Length,
                bearing_offset: Some(BearingOffsetDrag {
                    start,
                    endpoint,
                    normal: Vec3::Y,
                }),
                choosing_direction: false,
                press: pointer_sample(Vec2::ZERO, camera, press_direction),
                anchor_endpoint: endpoint,
                anchor_dimensions: dimensions,
                pieces,
                error: None,
            }),
            ..Default::default()
        };
        let current_direction = (Vec3::new(0.31, 0.0, 0.12) - camera).normalize();

        assert!(refresh_bearing_offset_drag(
            &graph,
            &mut state,
            camera,
            current_direction,
        ));

        let drag = state.pipe_drag.unwrap();
        assert!(drag.start.abs_diff_eq(Vec3::X * 0.25, 1.0e-5));
        assert!(
            drag.endpoint
                .abs_diff_eq(Vec3::X * 0.25 + Vec3::Y * 0.25, 1.0e-5)
        );
        let PartSpec::Cylinder(pipe) = drag.pieces[0].spec else {
            panic!("a straight run remains a cylinder")
        };
        assert!(
            pipe.pose
                .translation()
                .abs_diff_eq(Vec3::new(0.25, 0.125, 0.0), 1.0e-5)
        );
    }

    #[test]
    fn pipe_turn_chooser_locks_only_perpendicular_aim_beyond_the_dead_zone() {
        let anchor = Vec3::Z;
        assert_eq!(
            pipe_turn_direction(Vec3::X, anchor, (Vec3::Z + Vec3::Y * 0.1).normalize()),
            Some(Vec3::Y)
        );
        assert!(
            pipe_turn_direction(Vec3::X, anchor, (Vec3::Z + Vec3::X * 0.1).normalize()).is_none(),
            "aim along the incoming axis cannot select a perpendicular direction"
        );
        assert!(pipe_turn_direction(Vec3::X, anchor, anchor).is_none());
    }

    #[test]
    fn pipe_drag_reanchors_length_and_diameter_measurements() {
        let axis_origin = Vec3::ZERO;
        let direction = Vec3::Y;
        let camera = Vec3::new(0.0, 0.0, -4.0);
        let first_ray = (Vec3::new(0.0, 1.0, 0.0) - camera).normalize();
        let second_ray = (Vec3::new(0.0, 1.5, 0.0) - camera).normalize();
        let first = closest_axis_parameter(axis_origin, direction, camera, first_ray).unwrap();
        let second = closest_axis_parameter(axis_origin, direction, camera, second_ray).unwrap();
        assert!((first - 1.0).abs() < 1.0e-5);
        assert!((second - 1.5).abs() < 1.0e-5);
        assert!(pipe_pointer_delta(Vec3::Z, (Vec3::Z + Vec3::Y * 0.05).normalize()).abs() > 0.0);
    }

    #[test]
    fn bend_activity_owns_wheel_only_after_turning_starts() {
        let dimensions = CylinderDimensions::default();
        let cylinder = CylinderSpec::new(dimensions, BuildPose::default());
        let make_drag = || PipeDrag {
            attachment: BlockAttachment::AutoWeld {
                source: FaceOwner::Ground,
            },
            start: Vec3::ZERO,
            corners: Vec::new(),
            endpoint: Vec3::Y * 0.25,
            directions: vec![Vec3::Y],
            nodes: Vec::new(),
            pending_span: 1,
            branch: None,
            dimensions,
            material: ConstructionMaterial::Steel,
            appearance: MaterialAppearance::BAKED,
            mode: PipeEditMode::Length,
            bearing_offset: None,
            choosing_direction: false,
            press: PointerSample {
                cursor: Vec2::ZERO,
                ray_origin: Vec3::ZERO,
                ray_direction: Vec3::Z,
            },
            anchor_endpoint: Vec3::Y * 0.25,
            anchor_dimensions: dimensions,
            pieces: vec![crate::builder::PipeRunPiece {
                spec: PartSpec::Cylinder(cylinder),
                inlet: FaceKind::NegativeY,
                outlet: FaceKind::PositiveY,
            }],
            error: None,
        };
        let mut state = EditorState {
            pipe_drag: Some(make_drag()),
            ..Default::default()
        };
        assert!(!state.pipe_bend_active());
        state.pipe_drag.as_mut().unwrap().choosing_direction = true;
        assert!(state.pipe_bend_active());
        state.pipe_drag.as_mut().unwrap().choosing_direction = false;
        state
            .pipe_drag
            .as_mut()
            .unwrap()
            .nodes
            .push(crate::builder::PipeNode::Bend { span: 1 });
        assert!(state.pipe_bend_active());
    }

    #[test]
    fn first_leg_grows_to_fit_a_bend_pressed_straight_after_clicking() {
        let dimensions = CylinderDimensions::default();
        let press = PointerSample {
            cursor: Vec2::ZERO,
            ray_origin: Vec3::ZERO,
            ray_direction: Vec3::Z,
        };
        let graph = ConstructionGraph::new();
        let mut state = EditorState {
            pipe_drag: Some(PipeDrag {
                attachment: BlockAttachment::Free,
                start: Vec3::Y,
                corners: Vec::new(),
                endpoint: Vec3::Y * 1.25,
                directions: vec![Vec3::Y],
                nodes: Vec::new(),
                pending_span: 2,
                branch: None,
                dimensions,
                material: ConstructionMaterial::Steel,
                appearance: MaterialAppearance::BAKED,
                mode: PipeEditMode::Length,
                bearing_offset: None,
                choosing_direction: false,
                press,
                anchor_endpoint: Vec3::Y * 1.25,
                anchor_dimensions: dimensions,
                pieces: Vec::new(),
                error: None,
            }),
            ..Default::default()
        };

        crate::editor::pipe::begin_pipe_node(&graph, &mut state);
        let drag = state.pipe_drag.as_ref().unwrap();
        assert!(drag.choosing_direction, "the bend starts on the first leg");
        assert!(drag.endpoint.abs_diff_eq(Vec3::Y * 1.5, 1.0e-5));

        crate::editor::pipe::adjust_pipe_bend_span(&graph, &mut state, 1);
        let drag = state.pipe_drag.as_ref().unwrap();
        assert!(drag.endpoint.abs_diff_eq(Vec3::Y * 1.75, 1.0e-5));

        crate::editor::pipe::lock_pipe_node(&graph, &mut state, Vec3::Y, Vec3::X, press);
        let drag = state.pipe_drag.as_ref().unwrap();
        assert_eq!(drag.error, None);
        assert_eq!(drag.pieces.len(), 1, "the whole run is one 3 × 3 bend");
    }
}

#[cfg(test)]
mod history_tests {
    use bevy::prelude::{ButtonInput, IVec3, Vec2, Vec3};
    use mechanic_core::{
        BearingDimensions, BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph,
        CuboidSpec, FaceKind, FaceRef, GridRotation, PendingOperation, WeldSpec,
    };

    use super::{
        BlockVolume, PlacementPlane, SurfaceHit, bearing_attachment_candidate,
        stage_bearing_attachment,
    };
    use crate::controls::GameAction;
    use crate::editor::build_actions::PlacedBearing;
    use crate::editor::history::{
        EditorHistory, EditorSnapshot, HISTORY_CAPACITY, HistoryAction, apply_history_action,
        requested_history_action,
    };
    use crate::editor::hover::{
        BlockAttachment, BlockDrag, DeleteDrag, DeleteTarget, PointerSample,
    };
    use crate::editor::state::EditorState;

    fn spawn_cube(graph: &mut ConstructionGraph, center: IVec3) -> mechanic_core::PartId {
        let spec =
            CuboidSpec::new([4; 3], BuildPose::new(center, GridRotation::default())).unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
            unreachable!()
        };
        part
    }

    #[test]
    fn control_and_command_z_choose_undo_and_shift_redo() {
        for _ in 0..2 {
            let mut keyboard = ButtonInput::default();
            keyboard.press(GameAction::Undo);
            assert_eq!(
                requested_history_action(&keyboard),
                Some(HistoryAction::Undo)
            );

            keyboard.reset_all();
            keyboard.press(GameAction::Redo);
            assert_eq!(
                requested_history_action(&keyboard),
                Some(HistoryAction::Redo)
            );
        }

        let mut keyboard = ButtonInput::default();
        keyboard.press(GameAction::Save);
        assert_eq!(requested_history_action(&keyboard), None);
    }

    #[test]
    #[expect(clippy::too_many_lines)]
    fn bearing_attachment_round_trips_exact_ids_and_cancels_transients() {
        let mut graph = ConstructionGraph::new();
        let support = spawn_cube(&mut graph, IVec3::new(0, 2, 0));
        let socket = PlacedBearing {
            kind: mechanic_core::BearingKind::Rotational,
            axis: Vec3::ZERO,
            source: FaceRef::part(support, FaceKind::PositiveY),
            anchor: Vec3::Y,
            dimensions: BearingDimensions::new(0.70, 0.35).unwrap(),
        };
        let mut state = EditorState {
            placed_bearings: vec![socket],
            ..Default::default()
        };
        let mut history = EditorHistory::default();
        let previous = EditorSnapshot::capture(&graph, &state);
        let candidate = bearing_attachment_candidate(&graph, socket.source, socket.anchor);
        graph = stage_bearing_attachment(
            &graph,
            candidate,
            socket.source,
            socket.anchor,
            socket.dimensions,
        )
        .unwrap();
        history.commit(previous);
        let attached_parts = graph.parts().map(|(id, _)| id).collect::<Vec<_>>();
        let attached_bearings = graph.bearings().map(|(id, _)| id).collect::<Vec<_>>();
        assert_eq!(
            graph.bearings().next().unwrap().1.dimensions,
            socket.dimensions
        );

        graph
            .apply(BuildCommand::BeginPending(PendingOperation::Weld(
                FaceRef::part(support, FaceKind::PositiveX),
            )))
            .unwrap();
        let hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::Y,
            face: FaceRef::part(support, FaceKind::PositiveY),
        };
        state.hovered = Some(hit);
        state.preview = Some(candidate);
        state.block_drag = Some(BlockDrag {
            start: candidate,
            attachment: BlockAttachment::AutoWeld {
                source: hit.face.owner,
            },
            start_guides: Vec::new(),
            press: PointerSample {
                cursor: Vec2::ZERO,
                ray_origin: Vec3::Y,
                ray_direction: Vec3::NEG_Y,
            },
            plane: PlacementPlane::Xz,
            anchor_span: IVec3::ZERO,
            span: IVec3::ZERO,
            last_span: None,
            volume: BlockVolume::new(candidate.spec, IVec3::ZERO).unwrap(),
            error: None,
        });
        state.delete_drag = Some(DeleteDrag {
            start: graph.part(support).copied().unwrap().as_cuboid().unwrap(),
            press: PointerSample {
                cursor: Vec2::ZERO,
                ray_origin: Vec3::Y,
                ray_direction: Vec3::NEG_Y,
            },
            plane: PlacementPlane::Xz,
            anchor_span: IVec3::ZERO,
            span: IVec3::ZERO,
            last_span: None,
            parts: vec![support],
            error: None,
        });
        state.delete_target = Some(DeleteTarget::PlacedBearing(0));

        apply_history_action(HistoryAction::Undo, &mut graph, &mut state, &mut history);

        assert_eq!(graph.part_count(), 1);
        assert_eq!(graph.bearing_count(), 0);
        assert_eq!(state.placed_bearings, vec![socket]);
        assert!(graph.pending().is_none());
        assert!(state.block_drag.is_none());
        assert!(state.delete_drag.is_none());
        assert!(state.delete_target.is_none());
        assert!(state.hovered.is_none());
        assert!(state.preview.is_none());
        assert!(state.construction_mesh_dirty);

        apply_history_action(HistoryAction::Redo, &mut graph, &mut state, &mut history);

        assert_eq!(
            graph.parts().map(|(id, _)| id).collect::<Vec<_>>(),
            attached_parts
        );
        assert_eq!(
            graph.bearings().map(|(id, _)| id).collect::<Vec<_>>(),
            attached_bearings
        );
        assert_eq!(
            graph.bearings().next().unwrap().1.dimensions,
            socket.dimensions
        );
        assert_eq!(state.placed_bearings, vec![socket]);
    }

    #[test]
    fn dragged_deletion_restores_connections_atomically() {
        let mut graph = ConstructionGraph::new();
        let parts = [
            spawn_cube(&mut graph, IVec3::new(0, 2, 0)),
            spawn_cube(&mut graph, IVec3::new(0, 6, 0)),
            spawn_cube(&mut graph, IVec3::new(0, 10, 0)),
        ];
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::ground(),
                second: FaceRef::part(parts[0], FaceKind::NegativeY),
            }))
            .unwrap();
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(parts[0], FaceKind::PositiveY),
                second: FaceRef::part(parts[1], FaceKind::NegativeY),
            }))
            .unwrap();
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(parts[1], FaceKind::PositiveY),
                FaceRef::part(parts[2], FaceKind::NegativeY),
                Vec3::new(0.0, 2.0, 0.0),
                Vec3::Y,
            )))
            .unwrap();
        let original_part_ids = graph.parts().map(|(id, _)| id).collect::<Vec<_>>();
        let original_weld_ids = graph.welds().map(|(id, _)| id).collect::<Vec<_>>();
        let original_bearing_ids = graph.bearings().map(|(id, _)| id).collect::<Vec<_>>();
        let mut state = EditorState::default();
        let mut history = EditorHistory::default();
        let previous = EditorSnapshot::capture(&graph, &state);

        graph
            .apply_batch(parts[..2].iter().copied().map(BuildCommand::Remove))
            .unwrap();
        history.commit(previous);
        assert_eq!(history.undo.len(), 1);
        assert_eq!(graph.part_count(), 1);
        assert_eq!(graph.weld_count(), 0);
        assert_eq!(graph.bearing_count(), 0);

        apply_history_action(HistoryAction::Undo, &mut graph, &mut state, &mut history);
        assert_eq!(
            graph.parts().map(|(id, _)| id).collect::<Vec<_>>(),
            original_part_ids
        );
        assert_eq!(
            graph.welds().map(|(id, _)| id).collect::<Vec<_>>(),
            original_weld_ids
        );
        assert_eq!(
            graph.bearings().map(|(id, _)| id).collect::<Vec<_>>(),
            original_bearing_ids
        );
    }

    #[test]
    fn history_is_bounded_and_new_edits_clear_only_redo() {
        let graph = ConstructionGraph::new();
        let mut state = EditorState::default();
        let mut history = EditorHistory::default();
        for _ in 0..=HISTORY_CAPACITY {
            history.commit(EditorSnapshot::capture(&graph, &state));
        }
        assert_eq!(history.undo.len(), HISTORY_CAPACITY);

        apply_history_action(
            HistoryAction::Undo,
            &mut graph.clone(),
            &mut state,
            &mut history,
        );
        assert_eq!(history.redo.len(), 1);
        state.feedback = Some("camera and tool changes are transient".to_owned());
        assert_eq!(history.redo.len(), 1);

        history.commit(EditorSnapshot::capture(&graph, &state));
        assert!(history.redo.is_empty());
        assert_eq!(history.undo.len(), HISTORY_CAPACITY);
    }

    #[test]
    fn empty_history_stacks_report_guidance_without_mutation() {
        let mut graph = ConstructionGraph::new();
        let part = spawn_cube(&mut graph, IVec3::new(0, 2, 0));
        let mut state = EditorState::default();
        let mut history = EditorHistory::default();

        apply_history_action(HistoryAction::Undo, &mut graph, &mut state, &mut history);
        assert_eq!(graph.parts().next().unwrap().0, part);
        assert_eq!(state.feedback.as_deref(), Some("Nothing to undo"));
        apply_history_action(HistoryAction::Redo, &mut graph, &mut state, &mut history);
        assert_eq!(state.feedback.as_deref(), Some("Nothing to redo"));
    }
}

#[cfg(test)]
mod showcase_loading_tests {
    use std::time::Duration;

    use crate::scheduler::FixedStepScheduler;
    use bevy::prelude::{IVec3, Quat, Vec3};
    use mechanic_core::{
        BuildCommand, BuildOutcome, BuildPose, CuboidSpec, GridRotation, PartId, SeatSpec,
        TopologyError,
    };
    use mechanic_gpu::GpuTransform;

    use super::{ConstructionGraph, showcase};
    use crate::editor::creation::install_editor_graph;
    use crate::editor::history::{
        EditorHistory, EditorSnapshot, HistoryAction, apply_history_action,
    };
    use crate::editor::state::EditorState;
    use crate::seat::{
        raycast_seat_interaction, seat_exit_position, seat_surface_distance, seat_world_pose,
    };
    use crate::simulation::publication::creation_requires_live_physics;
    use crate::simulation::state::{AppSimulation, next_simulation_ticks};
    use crate::simulation::visuals::visual_snapshot_is_due;

    fn graph_with_seat() -> (ConstructionGraph, PartId) {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(seat) = graph
            .apply(BuildCommand::SpawnSeat(SeatSpec::new(BuildPose::default())))
            .unwrap()
        else {
            unreachable!()
        };
        (graph, seat)
    }

    #[test]
    fn terrain_anchored_construction_does_not_start_live_physics() {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([2; 3], BuildPose::default()).unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };

        assert!(creation_requires_live_physics(&graph.compile().unwrap()));
        assert!(!creation_requires_live_physics(
            &graph.compile_with_static_parts([part]).unwrap()
        ));
    }

    #[test]
    fn stopped_published_construction_supports_seat_interaction() {
        let (graph, seat) = graph_with_seat();
        let creation = graph.compile_with_static_parts([seat]).unwrap();
        let simulation = AppSimulation {
            transforms: vec![GpuTransform {
                position: creation.compounds[0]
                    .root_translation
                    .extend(0.0)
                    .to_array(),
                rotation: Quat::IDENTITY.to_array(),
            }],
            creation: Some(creation),
            published_graph: graph.clone(),
            failure: Some("stopped for test".to_owned()),
            ..AppSimulation::default()
        };

        assert!(!simulation.is_running());
        assert_eq!(
            raycast_seat_interaction(&graph, &simulation, Vec3::Z * 2.0, Vec3::NEG_Z)
                .map(|hit| hit.0),
            Some(seat)
        );
    }

    #[test]
    fn authored_seat_is_interactive_before_physics_publication() {
        let (graph, seat) = graph_with_seat();
        let simulation = AppSimulation::default();

        assert_eq!(
            raycast_seat_interaction(&graph, &simulation, Vec3::Z * 2.0, Vec3::NEG_Z)
                .map(|hit| hit.0),
            Some(seat)
        );
        let (position, rotation) = seat_world_pose(&graph, &simulation, seat).unwrap();
        assert!(position.abs_diff_eq(Vec3::ZERO, 1.0e-6));
        assert!(rotation.abs_diff_eq(Quat::IDENTITY, 1.0e-6));
    }

    #[test]
    fn seat_exit_position_is_just_above_the_seat_surface() {
        let (graph, seat) = graph_with_seat();

        let exit = seat_exit_position(&graph, &AppSimulation::default(), seat).unwrap();

        assert!(exit.abs_diff_eq(Vec3::new(0.0, 0.135, 0.0), 1.0e-6));
    }

    #[test]
    fn framed_seat_pose_matches_authored_and_stopped_published_geometry() {
        let (mut graph, seat) = graph_with_seat();
        let frame = mechanic_core::ConstructionFrame::new(
            Vec3::new(5.0, 3.0, -2.0),
            Quat::from_rotation_y(0.7) * Quat::from_rotation_z(0.2),
        )
        .unwrap();
        graph.reframe_parts([seat], frame).unwrap();
        let (position, rotation) =
            seat_world_pose(&graph, &AppSimulation::default(), seat).unwrap();
        assert!(position.distance(frame.translation()) < 1.0e-5);
        assert!(rotation.angle_between(frame.rotation()) < 1.0e-3);
        let creation = graph.compile().unwrap();
        let transforms = creation
            .compounds
            .iter()
            .map(|body| GpuTransform {
                position: body.root_translation.extend(0.0).to_array(),
                rotation: body.root_rotation.to_array(),
            })
            .collect();
        let mut simulation = AppSimulation {
            creation: Some(creation),
            transforms,
            published_graph: graph.clone(),
            failure: Some("stopped".to_owned()),
            ..Default::default()
        };
        let (published_position, published_rotation) =
            seat_world_pose(&graph, &simulation, seat).unwrap();
        assert!(published_position.distance(position) < 1.0e-5);
        assert!(published_rotation.angle_between(rotation) < 1.0e-3);
        simulation.transforms[0].position[0] += 2.0;
        assert!(
            seat_world_pose(&graph, &simulation, seat)
                .unwrap()
                .0
                .distance(position + Vec3::X * 2.0)
                < 1.0e-5
        );
        let streaming_focus = crate::world::terrain_streaming_focus(
            &crate::camera::PlayerState {
                position: Vec3::new(-100.0, 0.0, -100.0),
                seat: Some(seat),
                input_captured: true,
            },
            &graph,
            &simulation,
            mechanic_world::FloatingOrigin::default(),
        );
        assert!(
            streaming_focus
                .0
                .distance((position + Vec3::X * 2.0).as_dvec3())
                < 1.0e-5,
            "terrain follows the moving seat instead of its entry point",
        );
        let standing = crate::camera::PlayerState {
            position: Vec3::new(7.0, 2.0, -3.0),
            ..Default::default()
        };
        assert_eq!(
            crate::world::terrain_streaming_focus(
                &standing,
                &graph,
                &simulation,
                mechanic_world::FloatingOrigin::default(),
            ),
            mechanic_world::WorldPosition(standing.position.as_dvec3()),
            "walking terrain still follows player position",
        );
        assert!(
            crate::editor::wiring::wire_end_position(
                &graph,
                &EditorState::default(),
                &simulation,
                crate::editor::wiring::WireEnd::Seat(seat)
            )
            .unwrap()
            .distance(position + Vec3::X * 2.0)
                < 1.0e-5
        );
    }

    #[test]
    fn wire_drag_maps_the_local_pointer_ray_to_world_coordinates() {
        let (mut graph, seat) = graph_with_seat();
        let frame = mechanic_core::ConstructionFrame::new(
            Vec3::new(5.0, 3.0, -2.0),
            Quat::from_rotation_y(0.7),
        )
        .unwrap();
        graph.reframe_parts([seat], frame).unwrap();
        let state = EditorState {
            edit_context: Some(super::live_edit::EditContext {
                anchor: seat,
                frame: graph.part_frame_id(seat).unwrap(),
                frame_to_world: frame,
            }),
            wire_drag: Some(crate::editor::wiring::WireDrag {
                from: crate::editor::wiring::WireEnd::Seat(seat),
                armed: true,
            }),
            pointer_ray: Some((Vec3::new(1.0, 0.0, 5.0), Vec3::NEG_Z)),
            ..Default::default()
        };
        let (from, to) =
            crate::editor::wiring::wire_drag_endpoints(&graph, &state, &AppSimulation::default())
                .unwrap();
        assert!(from.distance(frame.translation()) < 1.0e-5);
        assert!(to.distance(frame.point(Vec3::X)) < 1.0e-5);
    }

    #[test]
    fn seat_range_is_measured_from_the_player_in_third_person() {
        let (graph, seat) = graph_with_seat();
        let simulation = AppSimulation::default();
        let camera_hit =
            raycast_seat_interaction(&graph, &simulation, Vec3::Z * 6.0, Vec3::NEG_Z).unwrap();
        let player_distance =
            seat_surface_distance(&graph, &simulation, seat, Vec3::Z * 2.0).unwrap();

        assert!(camera_hit.1 > 3.0);
        assert!(player_distance < 3.0);
    }

    #[test]
    fn app_simulation_stages_catch_up_ticks_up_to_the_backlog_cap() {
        let mut scheduler = FixedStepScheduler::new();
        let mut next_tick = 1;
        let mut backlog = 0;
        let mut dropped = 0;

        // A one-second hitch owes sixty ticks. Only the cap is kept, and the
        // discarded ticks advance the index so simulated time stays a fixed
        // distance behind wall time instead of an ever-growing one.
        assert_eq!(
            next_simulation_ticks(
                &mut scheduler,
                &mut next_tick,
                &mut backlog,
                &mut dropped,
                Duration::from_secs(1),
                false,
                3,
            ),
            31..34
        );
        assert_eq!(scheduler.next_tick(), 61);
        assert_eq!(dropped, 30);
        assert_eq!(backlog, 27);
        assert_eq!(
            next_simulation_ticks(
                &mut scheduler,
                &mut next_tick,
                &mut backlog,
                &mut dropped,
                Duration::from_millis(17),
                false,
                3,
            ),
            34..37
        );
        assert_eq!(backlog, 25);
        assert_eq!(
            next_simulation_ticks(
                &mut scheduler,
                &mut next_tick,
                &mut backlog,
                &mut dropped,
                Duration::ZERO,
                false,
                u64::MAX,
            ),
            37..62
        );
        assert_eq!(next_tick, 62);
        assert_eq!(backlog, 0);
        assert_eq!(dropped, 30, "nothing is dropped once the batch keeps up");
    }

    #[test]
    fn a_simulation_that_keeps_up_drops_no_ticks() {
        let mut scheduler = FixedStepScheduler::new();
        let mut next_tick = 1;
        let mut backlog = 0;
        let mut dropped = 0;
        for _ in 0..600 {
            next_simulation_ticks(
                &mut scheduler,
                &mut next_tick,
                &mut backlog,
                &mut dropped,
                Duration::from_millis(16),
                false,
                u64::MAX,
            );
        }
        assert_eq!(dropped, 0);
        assert_eq!(backlog, 0);
    }

    #[test]
    fn paused_simulation_does_not_advance_or_accumulate_time() {
        let mut scheduler = FixedStepScheduler::new();
        let mut next_tick = 7;
        let mut backlog = 5;
        let mut dropped = 0;
        let scheduler_tick = scheduler.next_tick();

        assert_eq!(
            next_simulation_ticks(
                &mut scheduler,
                &mut next_tick,
                &mut backlog,
                &mut dropped,
                Duration::from_secs(10),
                true,
                3,
            ),
            7..7
        );
        assert_eq!(next_tick, 7);
        assert_eq!(backlog, 5);
        assert_eq!(dropped, 0);
        assert_eq!(scheduler.next_tick(), scheduler_tick);
    }

    #[test]
    fn prototype_meshes_publish_every_second_completed_physics_tick() {
        assert!(!visual_snapshot_is_due(10, 10));
        assert!(!visual_snapshot_is_due(10, 11));
        assert!(visual_snapshot_is_due(10, 12));
        assert!(visual_snapshot_is_due(10, 14));
    }

    #[test]
    fn selected_creation_replaces_editor_and_round_trips_history() {
        let mut graph = ConstructionGraph::new();
        let mut state = EditorState::default();
        let mut history = EditorHistory::default();
        let previous = EditorSnapshot::capture(&graph, &state);
        let preset = showcase::CreationPreset::PendulumGarden256;
        let creation =
            install_editor_graph(&mut graph, showcase::build_preset(preset).unwrap()).unwrap();
        history.commit(previous);
        assert_eq!(graph.part_count(), preset.part_count());
        assert_eq!(creation.compounds.len(), preset.part_count());
        assert!(preset.matches(&graph));

        apply_history_action(HistoryAction::Undo, &mut graph, &mut state, &mut history);
        assert_eq!(graph.part_count(), 0);
        apply_history_action(HistoryAction::Redo, &mut graph, &mut state, &mut history);
        assert!(preset.matches(&graph));
    }

    #[test]
    fn failed_install_preserves_the_current_graph() {
        let mut graph = ConstructionGraph::new();
        let spec = CuboidSpec::new(
            [2, 2, 2],
            BuildPose::new(IVec3::new(0, 1, 0), GridRotation::default()),
        )
        .unwrap();
        graph.apply(BuildCommand::Spawn(spec)).unwrap();

        let result = install_editor_graph(&mut graph, ConstructionGraph::new());
        assert!(matches!(result, Err(TopologyError::EmptyConstruction)));
        assert_eq!(graph.part_count(), 1);
        assert_eq!(graph.parts().next().unwrap().1, &spec);
    }
}

#[cfg(test)]
mod joint_number_tests {
    use mechanic_core::{
        BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, ControllerSpec,
        CuboidSpec, DriveLinkSpec, FaceKind, FaceRef, GridRotation, PartId,
    };

    use bevy::prelude::*;

    use super::control_panel;
    use crate::editor::preview::{
        drive_xray_is_visible, driven_bearing_count, joint_number_labels,
    };
    use crate::editor::state::EditorGraph;
    use crate::hotbar::Tool;
    use crate::simulation::state::AppSimulation;
    use crate::ui::markers;

    fn spawned(outcome: BuildOutcome) -> PartId {
        match outcome {
            BuildOutcome::Spawned(part) => part,
            other => panic!("expected a spawn, got {other:?}"),
        }
    }

    fn cuboid(dimensions: [u8; 3], units: IVec3) -> CuboidSpec {
        CuboidSpec::new(dimensions, BuildPose::new(units, GridRotation::default()))
            .expect("test dimensions are in range")
    }

    /// One control block driving two joints, each on its own rotor.
    fn two_driven_joints() -> (ConstructionGraph, PartId, [Vec3; 2]) {
        let mut graph = ConstructionGraph::new();
        let base = spawned(
            graph
                .apply(BuildCommand::Spawn(cuboid([16, 2, 4], IVec3::new(0, 1, 0))))
                .expect("the base spawns"),
        );
        let controller = spawned(
            graph
                .apply(BuildCommand::SpawnController(ControllerSpec::new(
                    BuildPose::from_half_grid(IVec3::new(0, 5, 0), GridRotation::default()),
                )))
                .expect("the control block spawns"),
        );
        let mut anchors = Vec::new();
        for offset in [-6_i8, 6_i8] {
            let rotor = spawned(
                graph
                    .apply(BuildCommand::Spawn(cuboid(
                        [2, 2, 2],
                        IVec3::new(i32::from(offset), 3, 0),
                    )))
                    .expect("the rotor spawns"),
            );
            let anchor = Vec3::new(f32::from(offset) * 0.25, 0.5, 0.0);
            let BuildOutcome::BearingAdded(bearing) = graph
                .apply(BuildCommand::AddBearing(BearingSpec::new(
                    FaceRef::part(base, FaceKind::PositiveY),
                    FaceRef::part(rotor, FaceKind::NegativeY),
                    anchor,
                    Vec3::Y,
                )))
                .expect("the bearing is added")
            else {
                panic!("expected a bearing outcome");
            };
            graph
                .apply(BuildCommand::AddDriveLink(DriveLinkSpec::new(
                    controller, bearing,
                )))
                .expect("the wire is added");
            anchors.push(anchor);
        }
        (graph, controller, [anchors[0], anchors[1]])
    }

    #[test]
    fn floating_numbers_match_the_rows_the_panel_lists() {
        let (graph, controller, anchors) = two_driven_joints();
        let rows = control_panel::panel_rows(&graph, controller);
        let labels = joint_number_labels(&graph, |bearing| Some(bearing.shared_anchor));

        assert_eq!(rows.len(), 2, "each joint gets its own panel row");
        assert_eq!(
            labels,
            vec![(1, anchors[0]), (2, anchors[1])],
            "the number floating over a joint is the row number the panel shows"
        );
    }

    #[test]
    fn two_wires_on_one_joint_share_a_single_number() {
        let (mut graph, controller, anchors) = two_driven_joints();
        // A second group hung from the first joint's socket adds a wire
        // describing the same physical joint, which the panel folds into one
        // row and which therefore earns one number, not two.
        let extra = spawned(
            graph
                .apply(BuildCommand::Spawn(cuboid([2, 2, 2], IVec3::new(-6, 3, 0))))
                .expect("the extra rotor spawns"),
        );
        let base = graph.parts().next().expect("the base is the first part").0;
        let BuildOutcome::BearingAdded(bearing) = graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(base, FaceKind::PositiveY),
                FaceRef::part(extra, FaceKind::NegativeY),
                anchors[0],
                Vec3::Y,
            )))
            .expect("the second bearing is added")
        else {
            panic!("expected a bearing outcome");
        };
        graph
            .apply(BuildCommand::AddDriveLink(DriveLinkSpec::new(
                controller, bearing,
            )))
            .expect("the second wire is added");

        assert_eq!(driven_bearing_count(&graph), 3, "three wires exist");
        assert_eq!(
            control_panel::panel_rows(&graph, controller).len(),
            2,
            "but they describe two joints"
        );
        assert_eq!(
            joint_number_labels(&graph, |bearing| Some(bearing.shared_anchor)),
            vec![(1, anchors[0]), (2, anchors[1])],
            "one joint carries one number no matter how many wires reach it"
        );
    }

    /// Builds an app running `update_joint_numbers` over the given graph.
    ///
    /// The camera has no viewport, so every projection fails and the labels
    /// stay hidden. That is deliberate: this exercises the spawn and despawn
    /// bookkeeping, which is what breaks, without needing a render target.
    /// What the overlay would number, with the connector in hand.
    fn numbered(graph: &ConstructionGraph) -> Vec<usize> {
        markers::wanted(
            &EditorGraph(graph.clone()),
            &AppSimulation::default(),
            Tool::Connector,
        )
        .into_iter()
        .map(|(number, _)| number)
        .collect()
    }

    #[test]
    fn every_driven_joint_is_numbered_once() {
        let (graph, _, _) = two_driven_joints();
        assert_eq!(numbered(&graph), vec![1, 2], "each joint gets one number");
    }

    #[test]
    fn numbers_track_joints_appearing_and_disappearing() {
        let (graph, _, _) = two_driven_joints();
        assert!(
            numbered(&ConstructionGraph::new()).is_empty(),
            "nothing driven, nothing numbered",
        );
        assert_eq!(numbered(&graph).len(), 2);
        assert!(
            numbered(&ConstructionGraph::new()).is_empty(),
            "removing the joints clears them",
        );
    }

    #[test]
    fn putting_away_the_connector_clears_the_numbers() {
        let (graph, _, _) = two_driven_joints();
        assert_eq!(numbered(&graph).len(), 2);
        assert!(
            markers::wanted(&EditorGraph(graph), &AppSimulation::default(), Tool::Hammer,)
                .is_empty(),
            "the numbers belong to the tools that show the wires",
        );
    }

    #[test]
    fn numbers_show_with_the_tools_that_show_the_wires() {
        for tool in [Tool::Connector, Tool::Controller] {
            assert!(
                drive_xray_is_visible(tool, 1),
                "{tool:?} should show joint numbers"
            );
        }
        for tool in [Tool::Block, Tool::Bearing, Tool::Weld, Tool::Hammer] {
            assert!(
                !drive_xray_is_visible(tool, 1),
                "{tool:?} should not show joint numbers"
            );
        }
        assert!(
            !drive_xray_is_visible(Tool::Connector, 0),
            "nothing driven means nothing to number"
        );
    }
}

#[cfg(test)]
mod placement_snap_tests {
    use crate::editor::overlay::coordinate_inside;
    use crate::editor::overlay::lattice_coordinates;
    use crate::editor::overlay::lattice_thickness;
    use crate::editor::overlay::placement_lattice_geometry;
    use crate::editor::overlay::smart_snap_range_geometry;
    use crate::editor::placement::FreePlacementSettings;
    use crate::editor::placement::SmartSnapSettings;
    use crate::render::mesh::construction::CUBE_POSITIONS;
    use bevy::math::DVec2;

    use super::*;

    #[test]
    fn modifier_precedence_selects_precision_only_with_shift_and_control() {
        assert_eq!(
            PlacementGrid::from_modifiers(false, false),
            PlacementGrid::Centimetres25
        );
        assert_eq!(
            PlacementGrid::from_modifiers(false, true),
            PlacementGrid::Centimetres25
        );
        assert_eq!(
            PlacementGrid::from_modifiers(true, false),
            PlacementGrid::Centimetres5
        );
        assert_eq!(
            PlacementGrid::from_modifiers(true, true),
            PlacementGrid::Centimetres1
        );
    }

    #[test]
    fn alt_tap_toggles_but_range_adjustment_does_not() {
        let mut settings = SmartSnapSettings::default();
        settings.update(true, 0.0, false);
        assert!(!settings.enabled);

        settings.update(false, 1.0, false);
        assert!((settings.range - 1.25).abs() <= f32::EPSILON);
        assert!(settings.range_adjusted_this_frame);
        settings.update(true, 0.0, false);
        assert!(!settings.enabled);

        settings.update(false, 100.0, false);
        assert!((settings.range - 5.0).abs() <= f32::EPSILON);
        settings.update(false, -100.0, false);
        assert!((settings.range - 0.25).abs() <= f32::EPSILON);
    }

    #[test]
    fn free_range_adjustment_is_contextual_clamped_and_yields_to_object_snap() {
        let mut settings = FreePlacementSettings::default();
        settings.update(1.0, false, false);
        assert!((settings.range - 5.0).abs() <= f32::EPSILON);
        assert!(!settings.range_adjusted_this_frame);

        settings.update(1.0, true, false);
        assert!((settings.range - 5.25).abs() <= f32::EPSILON);
        assert!(settings.range_adjusted_this_frame);

        settings.update(1.0, true, true);
        assert!((settings.range - 5.25).abs() <= f32::EPSILON);
        assert!(!settings.range_adjusted_this_frame);

        settings.update(1000.0, true, false);
        assert!((settings.range - 30.0).abs() <= f32::EPSILON);
        settings.update(-1000.0, true, false);
        assert!((settings.range - 0.25).abs() <= f32::EPSILON);
    }

    #[test]
    fn free_fallback_is_garage_only_and_requires_an_eligible_tool() {
        let origin = Vec3::new(1.0, 6.0, 2.0);
        let direction = Vec3::NEG_Z;
        assert_eq!(
            free_placement_point_on_miss(
                Tool::Block,
                PlacementBounds::GarageBuild,
                origin,
                direction,
                5.0,
                false,
            ),
            Some(Vec3::new(1.0, 6.0, -3.0))
        );
        assert!(
            free_placement_point_on_miss(
                Tool::Bearing,
                PlacementBounds::GarageBuild,
                origin,
                direction,
                5.0,
                false,
            )
            .is_none()
        );
        assert!(
            free_placement_point_on_miss(
                Tool::Block,
                PlacementBounds::World {
                    origin: DVec2::ZERO,
                },
                origin,
                direction,
                5.0,
                false,
            )
            .is_none()
        );
        assert!(
            free_placement_point_on_miss(
                Tool::Block,
                PlacementBounds::GarageBuild,
                origin,
                direction,
                5.0,
                true,
            )
            .is_none()
        );
    }

    #[test]
    fn lattice_coordinates_keep_global_phase_and_emphasis_hierarchy() {
        let coordinates = lattice_coordinates(-0.25, 0.25, 0, PlacementGrid::Centimetres1);
        assert_eq!(coordinates.len(), 50);
        let mut expected = -0.245;
        for coordinate in coordinates {
            assert!((coordinate - expected).abs() < 1.0e-5);
            expected += 0.01;
        }
        assert!(lattice_thickness(0, 50) > lattice_thickness(0, 10));
        assert!(lattice_thickness(0, 10) > lattice_thickness(0, 2));
    }

    #[test]
    fn lattice_wraps_only_one_cell_beyond_the_preview() {
        let selection_low = Vec3::ZERO;
        let selection_high = Vec3::splat(0.25);
        let geometry = placement_lattice_geometry(
            selection_low,
            selection_high,
            Vec3::ZERO,
            PlacementGrid::Centimetres5,
            None,
        );

        let centers = lattice_line_centers(&geometry);
        assert!(!centers.is_empty());
        for center in centers {
            assert!(
                center.cmpge(Vec3::splat(-0.050_01)).all()
                    && center.cmple(Vec3::splat(0.300_01)).all(),
                "line centre {center:?} escaped the one-cell envelope"
            );
            assert!(
                !(0..3).all(|axis| {
                    coordinate_inside(center[axis], selection_low[axis], selection_high[axis])
                }),
                "line centre {center:?} crossed the preview interior"
            );
        }
    }

    #[test]
    fn dragging_shows_only_a_one_cell_border_on_the_whole_active_plane() {
        let selection_low = Vec3::ZERO;
        let selection_high = Vec3::new(0.75, 0.25, 0.5);
        let geometry = placement_lattice_geometry(
            selection_low,
            selection_high,
            Vec3::ZERO,
            PlacementGrid::Centimetres5,
            Some(PlacementPlane::Xz),
        );

        let centers = lattice_line_centers(&geometry);
        assert!(!centers.is_empty());
        assert!(centers.iter().all(|center| {
            (center.y - 0.125).abs() < 1.0e-6
                && center.x >= -0.05
                && center.x <= 0.80
                && center.z >= -0.05
                && center.z <= 0.55
                && !(coordinate_inside(center.x, selection_low.x, selection_high.x)
                    && coordinate_inside(center.z, selection_low.z, selection_high.z))
        }));

        for vertices in geometry.positions.chunks_exact(CUBE_POSITIONS.len()) {
            let low = vertices
                .iter()
                .map(|position| Vec3::from_array(*position))
                .fold(Vec3::splat(f32::INFINITY), Vec3::min);
            let high = vertices
                .iter()
                .map(|position| Vec3::from_array(*position))
                .fold(Vec3::splat(f32::NEG_INFINITY), Vec3::max);
            let extent = high - low;
            assert!(extent.y <= 0.004_1, "no line may run normal to XZ");
            assert!(extent.x > 0.01 || extent.z > 0.01);
        }
    }

    #[test]
    fn snap_range_wraps_the_whole_selection_on_the_active_plane() {
        let geometry = smart_snap_range_geometry(
            Vec3::ZERO,
            Vec3::new(0.75, 0.25, 0.5),
            1.0,
            Some(PlacementPlane::Xz),
        );

        let centers = lattice_line_centers(&geometry);
        assert_eq!(centers.len(), 36);
        assert!(
            centers
                .iter()
                .all(|center| (center.y - 0.125).abs() < 1.0e-6)
        );
        assert!(centers.iter().any(|center| center.x < -0.99));
        assert!(centers.iter().any(|center| center.x > 1.74));
        assert!(centers.iter().any(|center| center.z < -0.99));
        assert!(centers.iter().any(|center| center.z > 1.49));
    }

    #[test]
    fn free_preview_snap_range_uses_three_orthogonal_outlines() {
        let geometry = smart_snap_range_geometry(Vec3::ZERO, Vec3::splat(0.25), 0.5, None);

        let centers = lattice_line_centers(&geometry);
        assert_eq!(centers.len(), 108);
        for axis in 0..3 {
            assert!(
                centers
                    .iter()
                    .filter(|center| (center[axis] - 0.125).abs() < 1.0e-6)
                    .count()
                    >= 36
            );
        }
    }

    fn lattice_line_centers(geometry: &OverlayGeometry) -> Vec<Vec3> {
        geometry
            .positions
            .chunks_exact(CUBE_POSITIONS.len())
            .map(|vertices| {
                let low = vertices
                    .iter()
                    .map(|position| Vec3::from_array(*position))
                    .fold(Vec3::splat(f32::INFINITY), Vec3::min);
                let high = vertices
                    .iter()
                    .map(|position| Vec3::from_array(*position))
                    .fold(Vec3::splat(f32::NEG_INFINITY), Vec3::max);
                (low + high) * 0.5
            })
            .collect()
    }
}

#[cfg(test)]
mod creation_file_tests {
    use bevy::prelude::Vec3;
    use mechanic_core::{BearingDimensions, FaceKind, FaceRef};

    use super::{
        ConstructionGraph,
        creation_store::{CreationStore, read_document},
        showcase,
    };
    use crate::editor::build_actions::PlacedBearing;
    use crate::editor::creation::{capture_creation, install_editor_graph};
    use crate::editor::state::EditorState;

    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new(label: &str) -> Self {
            let path = std::env::temp_dir()
                .join(format!("mechanic-creations-{}-{label}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A preset construction plus one bearing ring the editor is still holding.
    fn editor_with_a_loose_ring() -> (ConstructionGraph, EditorState) {
        let graph = showcase::build_preset(showcase::CreationPreset::PendulumGarden256)
            .expect("the preset builds");
        let (part, _) = graph.parts().next().expect("the preset has parts");
        let mut state = EditorState::default();
        state.placed_bearings.push(PlacedBearing {
            kind: mechanic_core::BearingKind::Rotational,
            axis: Vec3::ZERO,
            source: FaceRef::part(part, FaceKind::PositiveY),
            anchor: Vec3::new(0.25, 1.5, -0.75),
            dimensions: BearingDimensions::new(0.4, 0.15).expect("the ring is in range"),
        });
        (graph, state)
    }

    #[test]
    fn saving_then_opening_restores_the_construction_and_its_loose_rings() {
        let temporary = TempDir::new("round-trip");
        let store = CreationStore::new(&temporary.0);
        let (graph, state) = editor_with_a_loose_ring();

        let path = store
            .save(&capture_creation(&graph, &state, "Pendulum Rig"))
            .expect("the creation is written");

        let loaded = read_document(&path)
            .expect("the file parses")
            .into_graph()
            .expect("the document rebuilds");
        let mut installed = ConstructionGraph::new();
        let creation =
            install_editor_graph(&mut installed, loaded.graph).expect("the rebuild compiles");

        assert_eq!(loaded.name, "Pendulum Rig");
        assert_eq!(installed.part_count(), graph.part_count());
        assert_eq!(installed.weld_count(), graph.weld_count());
        assert_eq!(installed.bearing_count(), graph.bearing_count());
        assert_eq!(
            creation.compounds.len(),
            graph
                .compile()
                .expect("the original compiles")
                .compounds
                .len()
        );

        let restored = loaded.sockets.first().expect("the loose ring comes back");
        let original = &state.placed_bearings[0];
        assert_eq!(restored.anchor, original.anchor);
        assert_eq!(restored.dimensions, original.dimensions);
        assert_eq!(restored.source.face, original.source.face);
        assert_eq!(
            installed.parts().next().map(|(id, _)| id),
            match restored.source.owner {
                mechanic_core::FaceOwner::Part(part) => Some(part),
                mechanic_core::FaceOwner::Ground => None,
            },
            "the ring still hangs off the first part"
        );
    }

    #[test]
    fn the_listing_summarises_what_a_saved_creation_holds() {
        let temporary = TempDir::new("listing");
        let store = CreationStore::new(&temporary.0);
        let (graph, state) = editor_with_a_loose_ring();
        store
            .save(&capture_creation(&graph, &state, "Pendulum Rig"))
            .expect("the creation is written");

        let listed = store.list();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "Pendulum Rig");
        assert_eq!(listed[0].part_count, graph.part_count());
        assert_eq!(listed[0].joint_count, graph.bearing_count());
    }

    use mechanic_core::{
        AppearanceTarget, BuildCommand, BuildOutcome, ConstructionMaterial, MaterialAppearance,
        PartId,
    };

    /// A 1 m steel block resting on the ground with a 25 cm rubber top layer.
    #[test]
    fn layer_ghost_over_a_block_slab_draws_only_its_outer_skin() {
        let block = |x: i32| {
            mechanic_core::PartSpec::Cuboid(
                mechanic_core::CuboidSpec::new(
                    [4, 4, 4],
                    mechanic_core::BuildPose::from_position_ticks(
                        [x, 200, 0].into(),
                        mechanic_core::GridRotation::default(),
                    ),
                )
                .unwrap(),
            )
            .with_layer(
                mechanic_core::LayerFace::Face(mechanic_core::FaceKind::PositiveY),
                0.25,
                ConstructionMaterial::Rubber,
                MaterialAppearance::BAKED,
            )
            .unwrap()
        };
        let mesh = crate::render::mesh::preview::layer_preview_mesh(&[block(0), block(400)]);
        let Some(bevy::mesh::Indices::U32(indices)) = mesh.indices() else {
            panic!("the ghost has 32-bit indices");
        };
        // Two touching boxes keep five faces each; the shared wall is gone.
        assert_eq!(indices.len(), 10 * 6);
    }

    fn layered_steel_block() -> (ConstructionGraph, PartId) {
        let mut graph = ConstructionGraph::new();
        let block = mechanic_core::PartSpec::Cuboid(
            mechanic_core::CuboidSpec::new(
                [4, 4, 4],
                mechanic_core::BuildPose::from_position_ticks(
                    [0, 200, 0].into(),
                    mechanic_core::GridRotation::default(),
                ),
            )
            .unwrap(),
        )
        .with_layer(
            mechanic_core::LayerFace::Face(mechanic_core::FaceKind::PositiveY),
            0.25,
            ConstructionMaterial::Rubber,
            MaterialAppearance::BAKED,
        )
        .unwrap();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(block.as_cuboid().unwrap()))
            .unwrap()
        else {
            panic!("spawning a block reports its part");
        };
        (graph, part)
    }

    #[test]
    fn layered_part_renders_into_each_band_material_mesh() {
        let (graph, _) = layered_steel_block();
        for material in [ConstructionMaterial::Steel, ConstructionMaterial::Rubber] {
            let mesh = crate::render::mesh::construction::combined_material_construction_mesh(
                &graph, None, material,
            );
            assert!(mesh.count_vertices() > 0, "{material:?} band is drawn");
        }
        let concrete = ConstructionMaterial::ALL
            .into_iter()
            .find(|material| {
                !matches!(
                    material,
                    ConstructionMaterial::Steel | ConstructionMaterial::Rubber
                )
            })
            .unwrap();
        assert_eq!(
            crate::render::mesh::construction::combined_material_construction_mesh(
                &graph, None, concrete
            )
            .count_vertices(),
            0
        );
    }

    #[test]
    fn chroma_paints_only_the_hovered_band() {
        let (graph, part) = layered_steel_block();
        let mut state = crate::editor::state::EditorState::default();
        let hit = |point| crate::builder::SurfaceHit {
            distance: 1.0,
            point,
            face: mechanic_core::FaceRef::part(part, mechanic_core::FaceKind::PositiveZ),
        };
        state.hovered = Some(hit(Vec3::new(0.0, 0.9, 0.5)));
        assert_eq!(
            crate::editor::build_actions::appearance_target(&graph, &state),
            Some(AppearanceTarget::PartBand { part, band: 0 })
        );
        state.hovered = Some(hit(Vec3::new(0.0, 1.1, 0.5)));
        assert_eq!(
            crate::editor::build_actions::appearance_target(&graph, &state),
            Some(AppearanceTarget::PartBand { part, band: 1 })
        );
    }
}

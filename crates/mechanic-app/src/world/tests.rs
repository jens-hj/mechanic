mod render;
mod terrain_shader;

use super::brush::terrain_edit_commands;
use super::foundations::foundation_edit_is_ready;
use super::list::install_world;
use super::streaming::nodes_touch_on_face;
use super::streaming::ready_obsolete_nodes;
use super::terrain_render::full_rgba8_mip_byte_count;
use super::terrain_render::terrain_chunk_mesh;
use super::terrain_render::terrain_mesh_is_renderable;
use super::transfer::place_in_world;
use super::transfer::remove_cached_foundations;
use super::transfer::returned_component_parts;
use super::walking::advance_controller;
use super::walking::compile_player_collision;
use super::walking::player_collision_nodes;
use super::walking::smooth_step_visual_offset;
use super::walking::terrain_chunk_has_collision_near;
use crate::builder::bounds::graph_part_bounds;
use crate::testing::TempDir;
use std::collections::BTreeSet;

use bevy::{
    asset::RenderAssetUsages,
    camera::Exposure,
    math::DVec3,
    mesh::VertexAttributeValues,
    prelude::{App, IVec3, Image, State, Update, Vec3},
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
    state::app::AppExtStates,
};
use mechanic_core::{
    BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, CuboidSpec, DimensionLinkId,
    DimensionLinkSpec, FaceKind, FaceOwner, FaceRef, GridRotation, WeldSpec,
};
use mechanic_world::{
    ActiveTerrainNode, BrickCoord, FloatingOrigin, FoundationSample, FoundationSpatialIndex,
    FoundationSupport, KinematicCapsule, TerrainDensity, TerrainEditBatch, TerrainFace,
    TerrainField, TerrainMaterial, TerrainMeshChunk, TerrainMeshRequest, TerrainNodeId,
    TerrainOctree, TerrainRayHit, TerrainReadiness, TerrainTransitionMask, WorldBounds,
    WorldPosition, WorldSeed, WorldStore, mesh_chunk, select_active_nodes,
};

use super::{
    AppSpace, SpaceEditorState, TerrainAcknowledgements, TerrainEditOperation, TerrainStrokeSample,
    WorldDiagnostics, WorldListPhase, WorldListState, WorldPrototypePlugin, WorldRuntime,
    exposure_for_space, generate_rgba8_mip_chain, handle_world_list, load_space_editors,
    static_parts_for_physics, sync_world_foundations,
};
use super::{PendingFoundationSync, TerrainFoundation};
use crate::editor::history::EditorHistory;
use crate::editor::state::{EditorGraph, EditorState};
use crate::{garage, showcase};

#[test]
fn saved_floor_creation_is_centered_in_editable_garage_and_detached_from_ground() {
    let mut graph = mechanic_core::ConstructionGraph::new();
    let mechanic_core::BuildOutcome::Spawned(part) = graph
        .apply(mechanic_core::BuildCommand::Spawn(
            mechanic_core::CuboidSpec::new(
                [4, 1, 4],
                mechanic_core::BuildPose::from_half_grid(
                    IVec3::new(32, 1, -24),
                    mechanic_core::GridRotation::default(),
                ),
            )
            .unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    graph
        .apply(mechanic_core::BuildCommand::Weld(mechanic_core::WeldSpec {
            first: mechanic_core::FaceRef::ground(),
            second: mechanic_core::FaceRef::part(part, mechanic_core::FaceKind::NegativeY),
        }))
        .unwrap();
    let loaded = mechanic_core::CreationDocument::from_graph(&graph, "Floor example", &[])
        .into_graph()
        .unwrap();
    let placed = super::place_loaded_creation_in_garage(loaded).unwrap();
    let (low, high) = graph_part_bounds(&placed.graph).unwrap();
    assert!((low.y - garage::BUILD_MIN_Y).abs() < 1.0e-5);
    assert!((low.x + high.x).abs() < 1.0e-5);
    assert!((low.z + high.z).abs() < 1.0e-5);
    assert_eq!(placed.name, "Floor example");
    assert_eq!(placed.graph.weld_count(), 0);
    placed.graph.compile().unwrap();
}

struct FlatTerrain(f64);

impl TerrainDensity for FlatTerrain {
    fn density(&self, position: WorldPosition) -> f32 {
        (self.0 - position.0.y) as f32
    }

    fn material(&self, _position: WorldPosition) -> TerrainMaterial {
        TerrainMaterial::Soil
    }
}

struct SlopedTerrain;

impl TerrainDensity for SlopedTerrain {
    fn density(&self, position: WorldPosition) -> f32 {
        (position.0.x * 0.25 - position.0.y) as f32
    }

    fn material(&self, _position: WorldPosition) -> TerrainMaterial {
        TerrainMaterial::Soil
    }
}

#[test]
fn player_collision_compiles_before_foundation_classification() {
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [1, 8, 8],
                BuildPose::new(IVec3::new(4, 4, 0), GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap();

    let mut collision = compile_player_collision(&graph).unwrap();
    assert!(
        collision
            .cast_capsule(
                bevy::prelude::Vec3::ZERO,
                bevy::prelude::Vec3::X * 2.0,
                mechanic_world::KinematicCapsuleConfig::default(),
            )
            .is_some()
    );
}

#[test]
fn automatic_step_rise_is_smoothed_without_changing_the_collision_height() {
    let mut offset = smooth_step_visual_offset(0.0, 0.25, 1.0 / 60.0);
    assert!(offset > -0.25 && offset < 0.0, "{offset}");
    let first = offset;
    for _ in 0..60 {
        offset = smooth_step_visual_offset(offset, 0.0, 1.0 / 60.0);
    }
    assert!(offset.abs() < 1.0e-4, "{offset}");
    assert!(first.abs() < 0.25);
}

#[test]
fn jump_request_is_buffered_until_the_next_fixed_controller_tick() {
    let mut accumulator = 0.0;
    let mut jump_queued = false;

    assert_eq!(
        advance_controller(
            &mut accumulator,
            &mut jump_queued,
            mechanic_core::TICK_SECONDS * 0.5,
            true,
        ),
        (0, false)
    );
    assert_eq!(
        advance_controller(
            &mut accumulator,
            &mut jump_queued,
            mechanic_core::TICK_SECONDS * 0.5,
            false,
        ),
        (1, true)
    );
    assert_eq!(
        advance_controller(
            &mut accumulator,
            &mut jump_queued,
            mechanic_core::TICK_SECONDS,
            false,
        ),
        (1, false)
    );
}

#[test]
fn prototype_starts_in_garage_space() {
    let mut app = App::new();
    app.add_plugins(bevy::state::app::StatesPlugin);
    app.add_plugins(WorldPrototypePlugin);
    assert_eq!(
        *app.world().resource::<State<AppSpace>>().get(),
        AppSpace::Garage
    );
}

#[test]
fn transfer_collision_tests_composed_boxes_instead_of_local_overlap() {
    let spawn = |graph: &mut ConstructionGraph| {
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new([4; 3], BuildPose::new(IVec3::ZERO, GridRotation::default()))
                    .unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        part
    };
    let mut candidate = ConstructionGraph::new();
    let part = spawn(&mut candidate);
    let mut destination = ConstructionGraph::new();
    let other = spawn(&mut destination);
    let far = mechanic_core::ConstructionFrame::new(
        Vec3::new(5.0, 0.0, 0.0),
        bevy::prelude::Quat::from_rotation_y(0.4),
    )
    .unwrap();
    candidate.reframe_parts([part], far).unwrap();
    let mut index = crate::builder::PlacementSnapIndex::default();
    index.rebuild(&destination);
    assert!(super::transfer::collision_free(
        &candidate,
        &destination,
        &index
    ));
    destination.reframe_parts([other], far).unwrap();
    index.rebuild(&destination);
    assert!(!super::transfer::collision_free(
        &candidate,
        &destination,
        &index
    ));
}

#[test]
fn returned_framed_creation_accepts_blocks_in_its_local_grid() {
    use crate::builder::{self, PlacementBounds, PlacementGrid};
    use mechanic_core::ConstructionFrame;

    for rotation in [
        bevy::prelude::Quat::IDENTITY,
        bevy::prelude::Quat::from_rotation_z(0.4) * bevy::prelude::Quat::from_rotation_y(0.3),
    ] {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [8, 2, 4],
                    BuildPose::new(IVec3::new(0, 80, 0), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        graph
            .reframe_parts(
                [part],
                ConstructionFrame::new(Vec3::new(30.03, 4.07, -20.02), rotation).unwrap(),
            )
            .unwrap();
        // Repeat the user's Garage -> World -> Garage recovery path.
        for _ in 0..2 {
            let garage =
                super::transfer::place_in_garage(&graph, &[], &SpaceEditorState::default())
                    .unwrap();
            let part = garage.graph.parts().next().unwrap().0;
            let context = crate::live_edit::EditContext::resolve(
                &garage.graph,
                &crate::simulation::state::AppSimulation::default(),
                part,
            )
            .unwrap();
            let local = garage.graph.in_edit_frame(context.frame).unwrap();
            let spec = local.part(part).unwrap().as_cuboid().unwrap();
            let (low, high) = builder::part_world_bounds(mechanic_core::PartSpec::Cuboid(spec));
            let hit = builder::raycast::raycast_construction_with_ground(
                &local,
                Vec3::new((low.x + high.x) * 0.5, high.y + 1.0, (low.z + high.z) * 0.5),
                Vec3::NEG_Y,
                None,
            )
            .unwrap();
            let bounds = PlacementBounds::GarageBuild.in_edit_frame(context.frame_to_world);
            let candidate = builder::candidates::candidate_from_hit_with_grid(
                &local,
                hit,
                PlacementGrid::Centimetres25,
                bounds,
            );
            let mut index = builder::PlacementSnapIndex::default();
            index.rebuild(&local);
            assert_eq!(
                builder::validate_indexed_block_batch_in_bounds(
                    &index,
                    candidate,
                    &[candidate.spec],
                    PlacementBounds::GarageBuild,
                ),
                Err(builder::PlacementError::OutsidePlatform),
            );
            builder::validate_indexed_block_batch_in_bounds(
                &index,
                candidate,
                &[candidate.spec],
                bounds,
            )
            .unwrap();
            let edited =
                builder::stage_block_batch_in_bounds(&local, candidate, &[candidate.spec], bounds)
                    .unwrap()
                    .canonicalized();
            assert_eq!(edited.parts().count(), garage.graph.parts().count() + 1);
            edited.compile().unwrap();
            graph = place_in_world(
                &garage.graph,
                &[],
                &SpaceEditorState::default(),
                Vec3::ZERO,
                &FlatTerrain(0.0),
                FloatingOrigin::default(),
            )
            .unwrap()
            .graph;
        }
    }
}

#[test]
fn framed_creation_transfers_preserve_orientation_and_composed_size() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [8, 2, 4],
                BuildPose::new(IVec3::new(0, 4, 0), GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let frame = mechanic_core::ConstructionFrame::new(
        Vec3::new(30.0, 4.0, -20.0),
        bevy::prelude::Quat::from_rotation_z(0.4) * bevy::prelude::Quat::from_rotation_y(0.3),
    )
    .unwrap();
    graph.reframe_parts([part], frame).unwrap();
    let (low, high) = graph_part_bounds(&graph).unwrap();
    let size = high - low;
    let original_rotation = graph.part_rotation(part).unwrap();
    let garage =
        super::transfer::place_in_garage(&graph, &[], &SpaceEditorState::default()).unwrap();
    let (garage_low, garage_high) = graph_part_bounds(&garage.graph).unwrap();
    assert!((garage_high - garage_low).distance(size) < 1.0e-4);
    assert!(garage_low.y >= crate::garage::BUILD_MIN_Y - 1.0e-4);
    let garage_part = garage.graph.parts().next().unwrap().0;
    assert!(
        garage
            .graph
            .part_rotation(garage_part)
            .unwrap()
            .angle_between(original_rotation)
            < 1.0e-3
    );
    let world = place_in_world(
        &garage.graph,
        &[],
        &SpaceEditorState::default(),
        Vec3::ZERO,
        &FlatTerrain(0.0),
        FloatingOrigin::default(),
    )
    .unwrap();
    let (world_low, world_high) = graph_part_bounds(&world.graph).unwrap();
    assert!((world_high - world_low).distance(size) < 1.0e-4);
    assert!(world_low.y >= 0.125 - 1.0e-4);
    let world_part = world.graph.parts().next().unwrap().0;
    assert!(
        world
            .graph
            .part_rotation(world_part)
            .unwrap()
            .angle_between(original_rotation)
            < 1.0e-3
    );
}

#[test]
fn framed_foundation_sampling_uses_global_composed_bottom_bounds() {
    let spec = mechanic_core::PartSpec::Cuboid(
        CuboidSpec::new(
            [4, 2, 2],
            BuildPose::new(IVec3::ZERO, GridRotation::default()),
        )
        .unwrap(),
    );
    let frame = mechanic_core::ConstructionFrame::new(
        Vec3::new(5.0, 0.5, -3.0),
        bevy::prelude::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
    )
    .unwrap();
    let bounds = super::transfer::framed_part_bounds(spec, frame);
    let origin = FloatingOrigin(DVec3::new(100.0, 10.0, 200.0));
    let support = super::transfer::bounds_foundation_support(&FlatTerrain(10.0), bounds, origin);
    assert!(support.has_valid_anchor());
    assert!(support.samples.iter().all(|sample| {
        sample.position.0.x > 104.0
            && sample.position.0.x < 106.0
            && sample.position.0.z > 196.0
            && sample.position.0.z < 198.0
    }));
    assert!(
        super::transfer::bounds_foundation_support(
            &FlatTerrain(10.0),
            crate::builder::part_world_bounds(spec),
            origin
        )
        .samples
        .iter()
        .all(|sample| sample.position.0.x < 101.0)
    );
}

#[test]
fn frame_only_edit_invalidates_foundation_cache_with_unchanged_part_spec() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [2; 3],
                BuildPose::new(IVec3::new(0, 1, 0), GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let original_spec = *graph.part(part).unwrap();
    let mut app = App::new();
    app.init_resource::<WorldRuntime>();
    app.init_resource::<WorldListState>();
    app.init_resource::<EditorState>();
    app.init_resource::<WorldDiagnostics>();
    {
        let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
        runtime.known_world_parts.insert(part, original_spec);
        runtime
            .known_world_frames
            .insert(part, mechanic_core::ConstructionFrame::IDENTITY);
        runtime.synced_editor_revision = 1;
    }
    let frame = mechanic_core::ConstructionFrame::new(
        Vec3::new(5.0, 0.0, 0.0),
        bevy::prelude::Quat::from_rotation_y(0.4),
    )
    .unwrap();
    graph.reframe_parts([part], frame).unwrap();
    assert_eq!(*graph.part(part).unwrap(), original_spec);
    app.insert_resource(EditorGraph(graph));
    app.insert_resource(EditorHistory {
        current_revision: 2,
        next_revision: 2,
        ..Default::default()
    });
    app.world_mut().resource_mut::<WorldListState>().phase = WorldListPhase::Playing;
    app.add_systems(Update, sync_world_foundations);
    app.update();
    let runtime = app.world().resource::<WorldRuntime>();
    assert_eq!(runtime.known_world_frames.get(&part), Some(&frame));
    assert!(runtime.foundations_match_editor_revision(2));
    assert_eq!(
        app.world()
            .resource::<WorldDiagnostics>()
            .foundation_candidate_count,
        1
    );
    assert!(
        app.world()
            .resource::<WorldDiagnostics>()
            .foundation_sample_count
            > 0
    );
}

#[test]
fn garage_creation_enters_world_detached_and_clear_of_terrain() {
    const TERRAIN_HEIGHT: f64 = 0.06;

    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [2; 3],
                BuildPose::new(IVec3::new(0, 1, 0), GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(part, FaceKind::NegativeY),
            second: FaceRef::ground(),
        }))
        .unwrap();

    let placed = place_in_world(
        &graph,
        &[],
        &SpaceEditorState::default(),
        Vec3::ZERO,
        &FlatTerrain(TERRAIN_HEIGHT),
        FloatingOrigin::default(),
    )
    .unwrap();

    assert!(placed.graph.welds().all(|(_, weld)| {
        !matches!(weld.first.owner, FaceOwner::Ground)
            && !matches!(weld.second.owner, FaceOwner::Ground)
    }));
    let (minimum, maximum) = graph_part_bounds(&placed.graph).unwrap();
    assert!(f64::from(minimum.y) - TERRAIN_HEIGHT >= 0.125 - 1.0e-6);
    let closest_x = 0.0_f32.clamp(minimum.x, maximum.x);
    let closest_z = 0.0_f32.clamp(minimum.z, maximum.z);
    assert!(closest_x.mul_add(closest_x, closest_z * closest_z) > 4.0);
    let support = FoundationSupport::rectangular(
        &FlatTerrain(TERRAIN_HEIGHT),
        TerrainRayHit {
            position: WorldPosition(
                Vec3::new(
                    (minimum.x + maximum.x) * 0.5,
                    minimum.y,
                    (minimum.z + maximum.z) * 0.5,
                )
                .as_dvec3(),
            ),
            normal: Vec3::Y,
            distance: 0.0,
            material_weights: [0.0; TerrainMaterial::COUNT],
            chunk_generation: 0,
            triangle: 0,
        },
        f64::from(maximum.x - minimum.x),
        f64::from(maximum.z - minimum.z),
    );
    assert!(!support.has_valid_anchor());
}

#[test]
fn garage_creation_enters_sloped_world_without_becoming_a_foundation() {
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([8, 2, 2], BuildPose::new(IVec3::Y, GridRotation::default())).unwrap(),
        ))
        .unwrap();

    let placed = place_in_world(
        &graph,
        &[],
        &SpaceEditorState::default(),
        Vec3::new(0.0, -10.0, 0.0),
        &SlopedTerrain,
        FloatingOrigin::default(),
    )
    .unwrap();

    for (_, part) in placed.graph.parts() {
        let (minimum, maximum) = crate::builder::part_world_bounds(*part);
        let support = FoundationSupport::rectangular(
            &SlopedTerrain,
            TerrainRayHit {
                position: WorldPosition(
                    Vec3::new(
                        (minimum.x + maximum.x) * 0.5,
                        minimum.y,
                        (minimum.z + maximum.z) * 0.5,
                    )
                    .as_dvec3(),
                ),
                normal: Vec3::Y,
                distance: 0.0,
                material_weights: [0.0; TerrainMaterial::COUNT],
                chunk_generation: 0,
                triangle: 0,
            },
            f64::from(maximum.x - minimum.x),
            f64::from(maximum.z - minimum.z),
        );
        assert!(!support.has_valid_anchor());
    }
}

#[test]
fn returning_component_discards_foundations_cached_under_reused_part_ids() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(chassis) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([2, 1, 1], BuildPose::new(IVec3::Y, GridRotation::default())).unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let BuildOutcome::Spawned(link) = graph
        .apply(BuildCommand::SpawnDimensionLink(DimensionLinkSpec::new(
            DimensionLinkId(4),
            BuildPose::new(IVec3::new(2, 1, 0), GridRotation::default()),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(chassis, FaceKind::PositiveX),
            second: FaceRef::part(link, FaceKind::NegativeX),
        }))
        .unwrap();
    let BuildOutcome::Spawned(unrelated) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [1; 3],
                BuildPose::new(IVec3::new(20, 0, 0), GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let support = || FoundationSupport {
        samples: vec![FoundationSample {
            position: WorldPosition::default(),
            valid: true,
        }],
    };
    let mut foundations = vec![
        TerrainFoundation {
            part: chassis,
            support: support(),
        },
        TerrainFoundation {
            part: unrelated,
            support: support(),
        },
    ];
    let mut index = FoundationSpatialIndex::default();
    for foundation in &foundations {
        index.insert(foundation.part, &foundation.support);
    }

    let returned = returned_component_parts(&graph, DimensionLinkId(4)).unwrap();
    assert!(remove_cached_foundations(
        &mut foundations,
        &mut index,
        &returned,
    ));

    assert_eq!(foundations.len(), 1);
    assert_eq!(foundations[0].part, unrelated);
    assert_eq!(index.len(), 1);
    assert!(!returned.contains(&unrelated));
    let compiled = graph
        .compile_with_static_parts(foundations.iter().map(|foundation| foundation.part))
        .unwrap();
    let body_for = |part| {
        compiled
            .part_to_compound
            .iter()
            .find_map(|(candidate, body)| (*candidate == part).then_some(*body as usize))
            .unwrap()
    };
    assert!(!compiled.compounds[body_for(chassis)].is_static);
    assert!(compiled.compounds[body_for(unrelated)].is_static);
}

#[test]
fn dimension_link_keeps_ground_anchors_after_foundation_sync() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(chassis) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([2, 1, 1], BuildPose::new(IVec3::Y, GridRotation::default())).unwrap(),
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
            first: FaceRef::part(chassis, FaceKind::PositiveX),
            second: FaceRef::part(link, FaceKind::NegativeX),
        }))
        .unwrap();
    let BuildOutcome::Spawned(unrelated) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [1; 3],
                BuildPose::new(IVec3::new(20, 0, 0), GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let valid_support = || FoundationSupport {
        samples: vec![FoundationSample {
            position: WorldPosition::default(),
            valid: true,
        }],
    };
    let foundations = vec![
        TerrainFoundation {
            part: chassis,
            support: valid_support(),
        },
        TerrainFoundation {
            part: unrelated,
            support: valid_support(),
        },
    ];
    let pending = PendingFoundationSync {
        editor_revision: 8,
        parts: graph.parts().map(|(part, spec)| (part, *spec)).collect(),
        frames: graph
            .parts()
            .map(|(part, _)| (part, graph.part_frame(part).unwrap()))
            .collect(),
        replaced_parts: BTreeSet::new(),
        new_parts: graph.parts().map(|(part, _)| part).collect(),
        next_part: 32,
        foundations: Vec::new(),
        index: FoundationSpatialIndex::default(),
    };

    let static_parts = static_parts_for_physics(
        &pending.parts,
        &foundations,
        None,
        pending.editor_revision,
        pending.editor_revision,
    )
    .expect("completed support sampling permits publication");
    assert_eq!(static_parts, vec![chassis, unrelated]);

    let compiled = graph.compile_with_static_parts(static_parts).unwrap();
    let body_for = |part| {
        compiled
            .part_to_compound
            .iter()
            .find_map(|(candidate, body)| (*candidate == part).then_some(*body as usize))
            .unwrap()
    };
    assert!(compiled.compounds[body_for(chassis)].is_static);
    assert!(compiled.compounds[body_for(link)].is_static);
    assert!(compiled.compounds[body_for(unrelated)].is_static);
}

#[test]
fn linked_creation_does_not_mobilize_unrelated_world_construction() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(chassis) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([2, 1, 1], BuildPose::new(IVec3::Y, GridRotation::default())).unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let BuildOutcome::Spawned(link) = graph
        .apply(BuildCommand::SpawnDimensionLink(DimensionLinkSpec::new(
            DimensionLinkId(12),
            BuildPose::new(IVec3::new(2, 1, 0), GridRotation::default()),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(chassis, FaceKind::PositiveX),
            second: FaceRef::part(link, FaceKind::NegativeX),
        }))
        .unwrap();
    let BuildOutcome::Spawned(unrelated) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [1; 3],
                BuildPose::new(IVec3::new(20, 0, 0), GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let known_parts = graph.parts().map(|(part, spec)| (part, *spec)).collect();

    let foundations = [TerrainFoundation {
        part: unrelated,
        support: FoundationSupport {
            samples: vec![FoundationSample {
                position: WorldPosition::default(),
                valid: true,
            }],
        },
    }];
    let static_parts = static_parts_for_physics(&known_parts, &foundations, None, 3, 3).unwrap();
    assert_eq!(static_parts, vec![unrelated]);

    let compiled = graph.compile_with_static_parts(static_parts).unwrap();
    let body_for = |part| {
        compiled
            .part_to_compound
            .iter()
            .find_map(|(candidate, body)| (*candidate == part).then_some(*body as usize))
            .unwrap()
    };
    assert!(!compiled.compounds[body_for(chassis)].is_static);
    assert!(!compiled.compounds[body_for(link)].is_static);
    assert!(compiled.compounds[body_for(unrelated)].is_static);
}

#[test]
#[ignore = "requires a real GPU adapter"]
#[expect(
    clippy::too_many_lines,
    reason = "exercise the app owner and GPU publication together"
)]
fn real_gpu_terrain_publication_preserves_motion_and_rejects_failed_replacements() {
    use mechanic_world::{TerrainTriangleGroupMask, TriangleBvh, TriangleBvhTriangle};
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("terrain publication requires a real adapter");
    eprintln!("Terrain publication adapter: {:?}", adapter.get_info());
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
    let mut app = App::new();
    app.init_resource::<WorldRuntime>();
    let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
    runtime.floating_origin = FloatingOrigin(DVec3::splat(1000.0));
    let mut weights = [0.0; TerrainMaterial::COUNT];
    weights[usize::from(TerrainMaterial::Rock.code())] = 1.0;
    let mut chunk = TerrainMeshChunk {
        origin: WorldPosition(runtime.floating_origin.0),
        generation: 1,
        vertices: vec![[-5.0, 0.0, -5.0], [0.0, 0.0, 5.0], [5.0, 0.0, -5.0]],
        material_weights: vec![weights; 3],
        triangle_bvh: TriangleBvh {
            triangles: vec![TriangleBvhTriangle {
                indices: [0, 1, 2],
                group_mask: TerrainTriangleGroupMask::REGULAR,
            }],
            ..Default::default()
        },
        ..Default::default()
    };
    chunk.index_groups.regular = vec![0, 1, 2];
    let node = chunk.node;
    runtime.active_terrain_index.insert(node);
    runtime.active_terrain.insert(node, chunk);
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [4; 3],
                BuildPose::from_half_grid(IVec3::new(0, 5, 0), GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap();
    let mut creation = graph.compile().unwrap();
    for collider in &mut creation.colliders {
        collider.material_properties.restitution = 0.0;
        collider.material_properties.youngs_modulus_pa = 200.0e9;
    }
    let gpu = mechanic_gpu::GpuPhysics::new_with_config(
        &device,
        &queue,
        &creation,
        crate::GpuPhysicsConfig {
            ground_plane_enabled: false,
            ..Default::default()
        },
    )
    .unwrap();
    gpu.enable_async_readback();
    gpu.apply_impulse(
        &device,
        &queue,
        0,
        creation.compounds[0].root_translation,
        Vec3::NEG_Y * 20.0 * creation.compounds[0].mass_properties.mass,
    )
    .unwrap();
    let mut simulation = crate::simulation::state::AppSimulation {
        gpu: Some(gpu),
        ..Default::default()
    };
    let publish = |simulation: &mut crate::simulation::state::AppSimulation,
                   runtime: &WorldRuntime| {
        crate::terrain_publication::publish(simulation, runtime, &device, &queue)
    };
    assert!(publish(&mut simulation, &runtime).unwrap());
    assert!(!publish(&mut simulation, &runtime).unwrap());
    let gpu = simulation.gpu.as_ref().unwrap();
    gpu.dispatch_tick(&device, &queue, 1);
    // Replace the scene before waiting for the submitted tick. Its captured
    // buffers and readback must still represent the previously accepted cut.

    let chunk = runtime.active_terrain.get_mut(&node).unwrap();
    chunk.generation = 2;
    chunk.triangle_bvh.triangles[0].indices[0] = u32::MAX;
    assert!(publish(&mut simulation, &runtime).is_err());
    let chunk = runtime.active_terrain.get_mut(&node).unwrap();
    chunk.generation = 1;
    chunk.triangle_bvh.triangles[0].indices[0] = 0;
    assert!(!publish(&mut simulation, &runtime).unwrap());

    // Retirement removes contacts without rebuilding or resetting bodies.
    runtime.active_terrain_index.remove(node);
    assert!(publish(&mut simulation, &runtime).unwrap());
    let gpu = simulation.gpu.as_ref().unwrap();
    gpu.dispatch_tick(&device, &queue, 2);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let first = gpu.poll_tick_readback(&device).unwrap().unwrap();
    assert_eq!(first.diagnostics.error_flags, 0);
    assert_eq!(first.diagnostics.contact_count, 4);
    assert!((first.transforms[0].position[1] - 0.5).abs() <= 0.005);
    assert!(first.velocities[0].linear[1] <= 0.001);
    let retired = gpu.poll_tick_readback(&device).unwrap().unwrap();
    assert_eq!(retired.diagnostics.error_flags, 0);
    assert_eq!(retired.diagnostics.contact_count, 0);
    assert!(retired.transforms[0].position[1] < first.transforms[0].position[1]);
    assert!(retired.velocities[0].linear[1] < first.velocities[0].linear[1]);

    // The same chunk generation must be re-uploaded in a new local frame.
    runtime.active_terrain_index.insert(node);
    runtime.floating_origin.0 += DVec3::X * 20.0;
    assert!(publish(&mut simulation, &runtime).unwrap());
    let gpu = simulation.gpu.as_ref().unwrap();
    gpu.dispatch_tick(&device, &queue, 3);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let shifted = gpu.poll_tick_readback(&device).unwrap().unwrap();
    assert_eq!(shifted.diagnostics.error_flags, 0);
    assert_eq!(shifted.diagnostics.contact_count, 0);
    runtime.floating_origin.0 -= DVec3::X * 20.0;
    assert!(publish(&mut simulation, &runtime).unwrap());
    let gpu = simulation.gpu.as_ref().unwrap();
    gpu.dispatch_tick(&device, &queue, 4);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let restored = gpu.poll_tick_readback(&device).unwrap().unwrap();
    assert_eq!(restored.diagnostics.error_flags, 0);
    assert_eq!(restored.diagnostics.contact_count, 4);
}

#[test]
fn physics_terrain_excludes_hidden_replacements_and_tracks_retirement() {
    let mut app = App::new();
    app.init_resource::<WorldRuntime>();
    let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
    let parent = TerrainNodeId::ROOT;
    let child = parent.children().unwrap()[0];
    for (node, generation) in [(parent, 1), (child, 2)] {
        runtime.active_terrain.insert(
            node,
            TerrainMeshChunk {
                node,
                generation,
                ..Default::default()
            },
        );
    }
    let everywhere = [WorldBounds {
        minimum: WorldPosition(DVec3::splat(-1.0e9)),
        maximum: WorldPosition(DVec3::splat(1.0e9)),
    }];
    runtime.active_terrain_index.insert(parent);
    assert_eq!(
        runtime
            .physics_terrain_near(&everywhere)
            .map(|chunk| chunk.node)
            .collect::<Vec<_>>(),
        vec![parent]
    );
    runtime.active_terrain_index.remove(parent);
    runtime.active_terrain_index.insert(child);
    assert_eq!(
        runtime
            .physics_terrain_near(&everywhere)
            .map(|chunk| (chunk.node, chunk.generation))
            .collect::<Vec<_>>(),
        vec![(child, 2)]
    );
    runtime.active_terrain_index.remove(child);
    assert_eq!(runtime.physics_terrain_near(&everywhere).count(), 0);

    // A region that no chunk reaches publishes nothing at all.
    runtime.active_terrain_index.insert(child);
    let elsewhere = [WorldBounds {
        minimum: WorldPosition(DVec3::splat(1.0e8)),
        maximum: WorldPosition(DVec3::splat(1.0e8 + 1.0)),
    }];
    assert_eq!(runtime.physics_terrain_near(&elsewhere).count(), 0);
}

#[test]
fn physics_waits_until_local_terrain_is_current() {
    let mut app = App::new();
    app.init_resource::<WorldRuntime>();
    let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
    let id = TerrainNodeId::default();
    let node = ActiveTerrainNode {
        id,
        generation: 1,
        transition_mask: TerrainTransitionMask::NONE,
    };
    assert!(
        !runtime.physics_terrain_ready(),
        "no selected region is unsafe"
    );
    runtime.terrain_streamer.set_critical_nodes([id]);
    runtime.terrain_streamer.set_desired([node]);
    assert!(
        !runtime.physics_terrain_ready(),
        "pending terrain is unsafe"
    );
    runtime.terrain_streamer.mark_started(node);
    assert!(runtime.terrain_streamer.stage(node));
    assert_eq!(runtime.terrain_streamer.activate(id), vec![node]);
    assert!(
        runtime.physics_terrain_ready(),
        "active current terrain is safe"
    );
}

#[test]
fn bearing_and_upper_block_edits_preserve_ground_weld_until_last_foot_is_removed() {
    let mut graph = ConstructionGraph::new();
    let mut parts = Vec::new();
    for (size, position) in [
        ([1; 3], IVec3::ZERO),
        ([1; 3], IVec3::X),
        ([2, 1, 1], IVec3::Y),
        ([1; 3], IVec3::Y * 2),
    ] {
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(size, BuildPose::new(position, GridRotation::default())).unwrap(),
            ))
            .unwrap()
        else {
            unreachable!()
        };
        parts.push(part);
    }
    let [first_foot, second_foot, platform, upper] = parts.try_into().unwrap();
    for (below, above) in [
        (first_foot, platform),
        (second_foot, platform),
        (platform, upper),
    ] {
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(below, FaceKind::PositiveY),
                second: FaceRef::part(above, FaceKind::NegativeY),
            }))
            .unwrap();
    }
    let mut app = App::new();
    app.init_resource::<WorldRuntime>()
        .init_resource::<WorldListState>()
        .init_resource::<EditorState>()
        .init_resource::<WorldDiagnostics>()
        .init_resource::<EditorHistory>()
        .insert_resource(EditorGraph(graph.clone()))
        .add_systems(Update, sync_world_foundations);
    app.world_mut().resource_mut::<WorldListState>().phase = WorldListPhase::Playing;
    {
        let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
        runtime.known_world_parts = graph.parts().map(|(part, spec)| (part, *spec)).collect();
        runtime.known_world_frames = graph
            .parts()
            .map(|(part, _)| (part, graph.part_frame(part).unwrap()))
            .collect();
        for part in [first_foot, second_foot] {
            let support = FoundationSupport {
                samples: vec![FoundationSample {
                    position: WorldPosition::default(),
                    valid: true,
                }],
            };
            runtime.foundation_index.insert(part, &support);
            runtime
                .foundations
                .push(TerrainFoundation { part, support });
        }
    }
    // Placing a bearing socket changes editor history without changing part geometry.
    app.world_mut()
        .resource_mut::<EditorState>()
        .placed_bearings
        .push(crate::editor::build_actions::PlacedBearing {
            kind: mechanic_core::JointKind::Rotational,
            axis: Vec3::ZERO,
            source: FaceRef::part(platform, FaceKind::PositiveY),
            anchor: Vec3::new(0.25, 0.5, 0.125),
            dimensions: mechanic_core::BearingDimensions::default(),
        });
    for (revision, removed, expected_anchors) in [
        (1, None, vec![first_foot, second_foot]),
        (2, Some(upper), vec![first_foot, second_foot]),
        (3, Some(first_foot), vec![second_foot]),
        (4, Some(second_foot), vec![]),
    ] {
        if let Some(part) = removed {
            app.world_mut()
                .resource_mut::<EditorGraph>()
                .0
                .apply(BuildCommand::Remove(part))
                .unwrap();
        }
        app.world_mut()
            .resource_mut::<EditorHistory>()
            .current_revision = revision;
        app.update();
        let runtime = app.world().resource::<WorldRuntime>();
        let anchors = runtime.static_parts_for_physics(revision).unwrap();
        assert_eq!(anchors, expected_anchors);
        let graph = &app.world().resource::<EditorGraph>().0;
        let compiled = graph.compile_with_static_parts(anchors).unwrap();
        let body = compiled
            .compounds
            .iter()
            .find(|body| body.source_parts.contains(&platform))
            .unwrap();
        assert_eq!(body.is_static, !expected_anchors.is_empty());
    }
}

#[test]
fn ground_weld_sampling_does_not_depend_on_streamed_meshes() {
    let mut app = App::new();
    app.init_resource::<WorldRuntime>()
        .init_resource::<WorldListState>()
        .init_resource::<EditorState>()
        .init_resource::<WorldDiagnostics>()
        .init_resource::<EditorHistory>()
        .add_systems(Update, sync_world_foundations);
    {
        let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
        let spawn = runtime.field.safe_spawn();
        let scene = mechanic_world::TerrainScene {
            field: &runtime.field,
            edits: &runtime.edits,
        };
        let hit = mechanic_world::raycast_density(
            &scene,
            WorldPosition(spawn.0 + DVec3::Y * 64.0),
            -DVec3::Y,
            128.0,
        )
        .unwrap();
        runtime.floating_origin = FloatingOrigin(hit.position.0 + DVec3::Y * 0.125);
        assert!(runtime.active_terrain.is_empty());
    }
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([1; 3], BuildPose::default()).unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    app.insert_resource(EditorGraph(graph));
    app.world_mut().resource_mut::<WorldListState>().phase = WorldListPhase::Playing;
    app.update();
    assert_eq!(
        app.world()
            .resource::<WorldRuntime>()
            .static_parts_for_physics(0),
        Some(vec![part])
    );
}

#[test]
fn world_physics_waits_for_foundation_reconciliation() {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([1; 3], BuildPose::default()).unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let pending = PendingFoundationSync {
        editor_revision: 2,
        parts: graph.parts().map(|(part, spec)| (part, *spec)).collect(),
        frames: graph
            .parts()
            .map(|(part, _)| (part, graph.part_frame(part).unwrap()))
            .collect(),
        replaced_parts: BTreeSet::new(),
        new_parts: vec![part],
        next_part: 0,
        foundations: Vec::new(),
        index: FoundationSpatialIndex::default(),
    };

    assert_eq!(
        static_parts_for_physics(&pending.parts, &[], Some(&pending), 1, 2,),
        None
    );
}

#[test]
fn loading_world_ignores_picker_actions_until_playing() {
    let mut state = WorldListState {
        phase: WorldListPhase::Loading,
        entries: Vec::new(),
        notice: None,
        loading_progress: TerrainReadiness::default(),
        confirming_delete: None,
        requested: None,
    };
    state.act(crate::ui::WorldAction::Create {
        name: "ignored".to_owned(),
        seed: String::new(),
    });
    assert!(state.requested.is_none());
    assert!(state.is_open());
    state.phase = WorldListPhase::Playing;
    assert!(!state.is_open());
    state.act(crate::ui::WorldAction::ExitToSelector);
    assert_eq!(
        state.requested,
        Some(crate::ui::WorldAction::ExitToSelector)
    );
}

#[test]
fn exiting_to_the_selector_saves_and_allows_the_world_to_reload() {
    let temporary = TempDir::new("world-install");
    let store = WorldStore::new(&temporary.0);
    let document = store.create_world("Reloadable", Some(7)).unwrap();
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([1; 3], BuildPose::new(IVec3::ZERO, GridRotation::default())).unwrap(),
        ))
        .unwrap();

    let mut app = App::new();
    app.add_plugins(bevy::state::app::StatesPlugin);
    app.init_state::<AppSpace>();
    app.init_resource::<WorldRuntime>();
    {
        let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
        runtime.store = store;
        install_world(&mut runtime, document).unwrap();
    }
    app.init_resource::<WorldListState>();
    app.insert_resource(EditorGraph(graph));
    app.init_resource::<EditorState>();
    app.add_systems(Update, handle_world_list);
    {
        let mut list = app.world_mut().resource_mut::<WorldListState>();
        list.phase = WorldListPhase::Playing;
        list.act(crate::ui::WorldAction::ExitToSelector);
    }

    app.update();

    let path = {
        let list = app.world().resource::<WorldListState>();
        assert_eq!(list.phase(), WorldListPhase::Picking);
        list.entries()[0].path.clone()
    };
    app.world_mut()
        .resource_mut::<WorldListState>()
        .act(crate::ui::WorldAction::Open(path));

    app.update();

    assert_eq!(
        app.world().resource::<WorldListState>().phase(),
        WorldListPhase::Loading
    );
    assert_eq!(
        app.world()
            .resource::<WorldRuntime>()
            .pending_garage_editor
            .as_ref()
            .unwrap()
            .graph
            .part_count(),
        1
    );
}

fn frozen_save_fixture() -> (TempDir, App, ConstructionGraph) {
    let temporary = TempDir::new("world-install");
    let store = WorldStore::new(&temporary.0);
    let document = store.create_world("Frozen publication", Some(7)).unwrap();
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::SpawnDimensionLink(DimensionLinkSpec::new(
            DimensionLinkId(1),
            BuildPose::new(IVec3::new(0, 4, 0), GridRotation::default()),
        )))
        .unwrap();
    let mut app = App::new();
    app.init_resource::<WorldRuntime>();
    let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
    runtime.store = store;
    install_world(&mut runtime, document).unwrap();
    runtime.document.active_dimension_link = Some(DimensionLinkId(1));
    runtime.document.next_dimension_link_id = 2;
    let construction_generation = runtime.document.construction_generation;
    runtime
        .persist_frozen_creation(
            Some(mechanic_world::FrozenCreationDoc {
                link: DimensionLinkId(1),
                target: WorldPosition(DVec3::new(20.0, 8.0, 30.0)),
                heading: 2,
                construction_generation,
            }),
            &graph,
            &EditorState::default(),
        )
        .unwrap();
    (temporary, app, graph)
}

#[test]
fn toggling_active_dimension_link_off_clears_and_persists_its_frozen_hold() {
    let (_temporary, mut app, graph) = frozen_save_fixture();
    let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
    let link = graph.dimension_link(DimensionLinkId(1)).unwrap();
    // Deactivation must work even when activation would reject an occupied Garage.
    runtime.garage_editor = Some(SpaceEditorState {
        graph: graph.clone(),
        ..Default::default()
    });
    assert_eq!(
        runtime.toggle_dimension_link(AppSpace::World, &graph, link),
        Ok(None)
    );
    assert_eq!(runtime.active_dimension_link(), None);
    assert!(runtime.document.frozen_creation.is_none());
    let saved = runtime
        .store
        .load_world(&runtime.store.directory_for(&runtime.document.name))
        .unwrap();
    assert_eq!(saved.active_dimension_link, None);
    assert!(saved.frozen_creation.is_none());
    assert_eq!(
        runtime.toggle_dimension_link(AppSpace::Garage, &graph, link),
        Ok(Some(DimensionLinkId(1)))
    );
    assert_eq!(runtime.active_dimension_link(), Some(DimensionLinkId(1)));
}

fn spawn_unaccepted_frozen_edit(graph: &mut ConstructionGraph) {
    // This block occupies the held target, so it has not passed freeze clearance.
    graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [4; 3],
                BuildPose::new(IVec3::new(80, 32, 120), GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap();
}

#[test]
fn leaving_world_discards_body_state_and_keeps_the_construction_frame() {
    let temporary = TempDir::new("world-install");
    let mut app = App::new();
    app.init_resource::<WorldRuntime>();
    app.init_resource::<EditorGraph>();
    app.init_resource::<EditorHistory>();
    app.init_resource::<EditorState>();
    app.init_resource::<bevy::prelude::ClearColor>();
    app.world_mut().spawn((
        crate::MainCamera,
        bevy::prelude::DistanceFog::default(),
        super::Exposure::default(),
    ));
    let graph = showcase::build_preset(showcase::CreationPreset::PendulumGarden256).unwrap();
    app.insert_resource(crate::simulation::state::AppSimulation {
        creation: Some(graph.compile().unwrap()),
        published_graph: graph.clone(),
        world_revision: Some((1, 1)),
        transforms: vec![mechanic_gpu::GpuTransform {
            position: [1.0; 4],
            rotation: [0.0, 0.0, 0.0, 1.0],
        }],
        ..Default::default()
    });
    app.insert_resource(EditorGraph(graph));
    let origin = FloatingOrigin(DVec3::new(12.0, 34.0, 56.0));
    {
        let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
        runtime.store = WorldStore::new(&temporary.0);
        runtime.document = runtime.store.create_world("Leave", Some(42)).unwrap();
        runtime.floating_origin = origin;
        runtime.garage_editor = Some(SpaceEditorState::default());
    }
    app.add_systems(Update, super::leave_world);
    app.update();
    let simulation = app
        .world()
        .resource::<crate::simulation::state::AppSimulation>();
    assert!(simulation.creation.is_none());
    assert!(simulation.transforms.is_empty());
    assert!(simulation.live_state.is_none());
    assert_eq!(simulation.published_graph.part_count(), 0);
    let runtime = app.world().resource::<WorldRuntime>();
    assert_eq!(runtime.world_editor.as_ref().unwrap().origin, origin);
    assert!(runtime.world_editor.as_ref().unwrap().graph.part_count() > 0);
}

#[test]
fn world_reload_preserves_construction_origin_after_player_moves() {
    let temporary = TempDir::new("world-install");
    let mut app = App::new();
    app.init_resource::<WorldRuntime>();
    app.init_resource::<WorldListState>();
    app.init_resource::<EditorState>();
    app.init_resource::<WorldDiagnostics>();
    app.init_resource::<EditorHistory>();
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [4; 3],
                BuildPose::new(IVec3::new(8, 12, 16), GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap();
    let origin = FloatingOrigin(DVec3::new(123.0, 45.0, -678.0));
    let bounds = graph_part_bounds(&graph).unwrap();
    {
        let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
        runtime.store = WorldStore::new(&temporary.0);
        runtime.document = runtime.store.create_world("Origin", Some(42)).unwrap();
        runtime.floating_origin = origin;
        runtime.capsule.position = WorldPosition(DVec3::new(-89.0, 123.0, 456.0));
        runtime.document.return_anchor = Some(runtime.capsule.position);
        super::saving::save_all(&mut runtime).unwrap();
        super::save_world_instance(&mut runtime, &graph, &EditorState::default()).unwrap();
        let saved = runtime
            .store
            .load_world(&runtime.store.directory_for("Origin"))
            .unwrap();
        let (world, _) = load_space_editors(&runtime.store, &saved).unwrap();
        assert_eq!(world.origin, origin);
        assert_eq!(graph_part_bounds(&world.graph).unwrap(), bounds);
        // Saving again from the Garage must preserve the inactive World's root.
        runtime.world_editor = Some(world);
        super::saving::save_garage_instance(
            &mut runtime,
            &ConstructionGraph::new(),
            &EditorState::default(),
        )
        .unwrap();
        let saved = runtime
            .store
            .load_world(&runtime.store.directory_for("Origin"))
            .unwrap();
        install_world(&mut runtime, saved).unwrap();
        assert_eq!(runtime.floating_origin, origin);
        let world = runtime.world_editor.take().unwrap();
        assert_eq!(world.origin, origin);
        assert_eq!(graph_part_bounds(&world.graph).unwrap(), bounds);
        let mut player = crate::camera::PlayerState::default();
        super::space::restore_world_player(&mut runtime, &mut player, world.origin);
        assert_eq!(runtime.floating_origin, origin);
        assert_eq!(
            runtime.local_to_global(player.position),
            runtime.document.return_anchor.unwrap()
        );
        graph = world.graph;
    }
    app.insert_resource(EditorGraph(graph));
    app.world_mut().resource_mut::<WorldListState>().phase = WorldListPhase::Playing;
    app.add_systems(Update, sync_world_foundations);
    app.update();
    let runtime = app.world().resource::<WorldRuntime>();
    let graph = app.world().resource::<EditorGraph>();
    let compiled = graph
        .0
        .compile_with_static_parts(runtime.static_parts_for_physics(0).unwrap())
        .unwrap();
    // This fixture has no terrain contact; loading must not invent a ground weld.
    assert!(compiled.compounds.iter().all(|body| !body.is_static));
    assert_eq!(
        runtime.local_to_global(bounds.0),
        WorldPosition(origin.0 + bounds.0.as_dvec3())
    );
}

#[test]
fn frozen_save_pairs_target_with_only_the_last_accepted_construction() {
    let (_temporary, mut app, mut graph) = frozen_save_fixture();
    let editor = EditorState::default();
    let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
    let target = runtime.frozen_creation().unwrap().target;
    spawn_unaccepted_frozen_edit(&mut graph);
    super::save_world_instance(&mut runtime, &graph, &editor).unwrap();
    let saved = runtime
        .store
        .load_world(&runtime.store.directory_for(&runtime.document.name))
        .unwrap();
    let (world, _) = load_space_editors(&runtime.store, &saved).unwrap();
    assert_eq!(world.graph.part_count(), 1);
    assert!(world.graph.dimension_link(DimensionLinkId(1)).is_some());
    let hold = saved.frozen_creation.unwrap();
    assert_eq!(hold.target, target);
    assert_eq!(hold.construction_generation, saved.construction_generation);
    let previous_generation = saved.construction_generation;

    // Publication acceptance is the explicit gate; save itself never validates geometry.
    runtime.accept_frozen_publication(&graph, &editor);
    super::save_world_instance(&mut runtime, &graph, &editor).unwrap();
    let saved = runtime
        .store
        .load_world(&runtime.store.directory_for(&runtime.document.name))
        .unwrap();
    let (world, _) = load_space_editors(&runtime.store, &saved).unwrap();
    assert_eq!(world.graph.part_count(), 2);
    assert!(saved.construction_generation > previous_generation);
    let hold = saved.frozen_creation.unwrap();
    assert_eq!(hold.target, target);
    assert_eq!(hold.construction_generation, saved.construction_generation);
}

#[test]
fn frozen_global_target_survives_origin_change_and_release_saves_current_edits() {
    let (_temporary, mut app, mut graph) = frozen_save_fixture();
    let editor = EditorState::default();
    let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
    runtime.floating_origin = FloatingOrigin(DVec3::new(1000.0, 0.0, -2000.0));
    let mut hold = runtime.frozen_creation().unwrap();
    hold.target = runtime.local_to_global(Vec3::new(3.0, 8.25, -4.0));
    let target = hold.target;
    runtime.set_frozen_target(hold);
    runtime.floating_origin = FloatingOrigin(DVec3::new(-500.0, 0.0, 1000.0));
    spawn_unaccepted_frozen_edit(&mut graph);
    super::save_world_instance(&mut runtime, &graph, &editor).unwrap();
    let saved = runtime
        .store
        .load_world(&runtime.store.directory_for(&runtime.document.name))
        .unwrap();
    assert_eq!(saved.frozen_creation.unwrap().target, target);
    assert_eq!(
        load_space_editors(&runtime.store, &saved)
            .unwrap()
            .0
            .graph
            .part_count(),
        1
    );
    install_world(&mut runtime, saved).unwrap();
    assert_eq!(runtime.frozen_creation().unwrap().target, target);
    assert_eq!(runtime.frozen_editor.as_ref().unwrap().0.part_count(), 1);

    runtime
        .persist_frozen_creation(None, &graph, &editor)
        .unwrap();
    assert!(runtime.frozen_editor.is_none());
    let saved = runtime
        .store
        .load_world(&runtime.store.directory_for(&runtime.document.name))
        .unwrap();
    assert!(saved.frozen_creation.is_none());
    assert_eq!(
        load_space_editors(&runtime.store, &saved)
            .unwrap()
            .0
            .graph
            .part_count(),
        2
    );
}

#[test]
fn installing_a_new_world_replaces_the_previous_world_editor() {
    let mut editor = super::SpaceEditorState {
        graph: showcase::build_preset(showcase::CreationPreset::PendulumGarden256).unwrap(),
        ..super::SpaceEditorState::default()
    };
    assert!(editor.graph.part_count() > 0);
    let temporary = TempDir::new("world-install");
    let store = WorldStore::new(&temporary.0);
    let document = store.create_world("Fresh", Some(42)).unwrap();

    editor = load_space_editors(&store, &document).unwrap().0;

    assert_eq!(editor.graph.part_count(), 0);
    assert!(editor.placed_bearings.is_empty());
}

#[test]
fn brush_paths_preserve_every_five_centimetre_sample() {
    let start = WorldPosition(DVec3::new(-1.0, 2.0, 3.0));
    let end = WorldPosition(DVec3::new(-0.77, 2.0, 3.0));
    let operation = TerrainEditOperation::Remove;
    let previous = TerrainStrokeSample {
        centre: start,
        radius_metres: 0.5,
        operation,
    };
    let commands = terrain_edit_commands(Some(previous), end, 0.5, operation);

    assert_eq!(commands.len(), 5);
    let mut previous = start;
    for command in &commands {
        assert_eq!(command.previous, Some((previous, 0.5)));
        assert!(previous.0.distance(command.centre.0) <= 0.05 + 1.0e-12);
        previous = command.centre;
    }
    assert_eq!(previous, end);
    assert!(
        terrain_edit_commands(
            Some(TerrainStrokeSample {
                centre: end,
                radius_metres: 0.5,
                operation,
            }),
            end,
            0.5,
            operation,
        )
        .is_empty()
    );
    assert_eq!(terrain_edit_commands(None, end, 0.5, operation).len(), 1);
}

#[test]
fn publication_waves_do_not_reinvalidate_an_acknowledged_foundation_edit() {
    let pending = TerrainEditBatch {
        generation: 7,
        changed_bricks: BTreeSet::from([BrickCoord::new(1, 2, 3)]),
    };
    let acknowledgements = TerrainAcknowledgements {
        edit: 7,
        mesh: 7,
        upload: 7,
        collision: 7,
    };
    assert!(foundation_edit_is_ready(
        acknowledgements,
        &pending,
        6,
        true
    ));
    assert!(!foundation_edit_is_ready(
        acknowledgements,
        &pending,
        7,
        true
    ));
    assert!(!foundation_edit_is_ready(
        TerrainAcknowledgements {
            upload: 6,
            ..acknowledgements
        },
        &pending,
        6,
        true
    ));
}

#[test]
fn continuous_stroke_waits_for_the_final_acknowledged_generation() {
    let pending = TerrainEditBatch {
        generation: 9,
        changed_bricks: BTreeSet::from([BrickCoord::new(0, 0, 0), BrickCoord::new(1, 0, 0)]),
    };
    let acknowledgements = TerrainAcknowledgements {
        edit: 9,
        mesh: 9,
        upload: 9,
        collision: 9,
    };
    assert!(!foundation_edit_is_ready(
        acknowledgements,
        &pending,
        8,
        false
    ));
    assert!(foundation_edit_is_ready(
        acknowledgements,
        &pending,
        8,
        true
    ));
}

#[test]
fn large_construction_foundations_publish_over_bounded_frames() {
    let mut graph = ConstructionGraph::new();
    let mut edit = graph.begin_edit();
    edit.reserve_parts_and_welds(4_096, 0);
    edit.spawn_cuboids((0..64).flat_map(|x| {
        (0..64).map(move |z| {
            CuboidSpec::new(
                [1; 3],
                BuildPose::new(IVec3::new(x, 0, z), GridRotation::default()),
            )
            .unwrap()
        })
    }));
    graph = edit.finish();

    let mut app = App::new();
    app.init_resource::<WorldRuntime>();
    app.init_resource::<WorldListState>();
    app.init_resource::<EditorState>();
    app.init_resource::<WorldDiagnostics>();
    app.insert_resource(EditorGraph(graph));
    let history = EditorHistory {
        current_revision: 1,
        next_revision: 1,
        ..Default::default()
    };
    app.insert_resource(history);
    app.world_mut().resource_mut::<WorldListState>().phase = WorldListPhase::Playing;
    app.add_systems(Update, sync_world_foundations);

    app.update();
    let runtime = app.world().resource::<WorldRuntime>();
    let pending = runtime
        .pending_foundation_sync
        .as_ref()
        .expect("the first frame leaves bounded foundation work pending");
    assert!(
        (1..=super::foundations::FOUNDATION_SYNC_MAX_PARTS_PER_FRAME).contains(&pending.next_part)
    );
    assert!(!runtime.foundations_match_editor_revision(1));

    for _ in 0..4_096 {
        if app
            .world()
            .resource::<WorldRuntime>()
            .foundations_match_editor_revision(1)
        {
            break;
        }
        app.update();
    }
    let runtime = app.world().resource::<WorldRuntime>();
    assert!(runtime.foundations_match_editor_revision(1));
    assert_eq!(runtime.known_world_parts.len(), 4_096);
}

#[test]
fn linked_creation_waits_for_bounded_world_foundation_sampling() {
    let mut graph = ConstructionGraph::new();
    let mut edit = graph.begin_edit();
    edit.reserve_parts_and_welds(129, 0);
    let _unrelated = edit.spawn_cuboids((0..128).map(|x| {
        CuboidSpec::new(
            [1; 3],
            BuildPose::new(IVec3::new(x, 0, 0), GridRotation::default()),
        )
        .unwrap()
    }));
    graph = edit.finish();
    let BuildOutcome::Spawned(_link) = graph
        .apply(BuildCommand::SpawnDimensionLink(DimensionLinkSpec::new(
            DimensionLinkId(11),
            BuildPose::new(IVec3::new(300, 4, 0), GridRotation::default()),
        )))
        .unwrap()
    else {
        unreachable!()
    };

    let mut app = App::new();
    app.init_resource::<WorldRuntime>();
    app.init_resource::<WorldListState>();
    app.init_resource::<EditorState>();
    app.init_resource::<WorldDiagnostics>();
    app.insert_resource(EditorGraph(graph));
    app.insert_resource(EditorHistory {
        current_revision: 1,
        next_revision: 1,
        ..Default::default()
    });
    app.world_mut().resource_mut::<WorldListState>().phase = WorldListPhase::Playing;
    app.add_systems(Update, sync_world_foundations);

    app.update();

    let runtime = app.world().resource::<WorldRuntime>();
    let pending = runtime.pending_foundation_sync.as_ref().unwrap();
    assert!(pending.next_part > 0);
    assert_eq!(runtime.static_parts_for_physics(1), None);
}

#[test]
fn adding_dimension_link_preserves_cached_ground_anchors() {
    let mut previous_graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(chassis) = previous_graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([2, 1, 1], BuildPose::new(IVec3::Y, GridRotation::default())).unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let BuildOutcome::Spawned(unrelated) = previous_graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new(
                [1; 3],
                BuildPose::new(IVec3::new(20, 0, 0), GridRotation::default()),
            )
            .unwrap(),
        ))
        .unwrap()
    else {
        unreachable!()
    };
    let mut graph = previous_graph.clone();
    let BuildOutcome::Spawned(link) = graph
        .apply(BuildCommand::SpawnDimensionLink(DimensionLinkSpec::new(
            DimensionLinkId(12),
            BuildPose::new(IVec3::new(2, 1, 0), GridRotation::default()),
        )))
        .unwrap()
    else {
        unreachable!()
    };
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(chassis, FaceKind::PositiveX),
            second: FaceRef::part(link, FaceKind::NegativeX),
        }))
        .unwrap();
    let valid_support = || FoundationSupport {
        samples: vec![FoundationSample {
            position: WorldPosition::default(),
            valid: true,
        }],
    };

    let mut app = App::new();
    app.init_resource::<WorldRuntime>();
    app.init_resource::<WorldListState>();
    app.init_resource::<EditorState>();
    app.init_resource::<WorldDiagnostics>();
    app.insert_resource(EditorGraph(graph));
    app.insert_resource(EditorHistory {
        current_revision: 2,
        next_revision: 2,
        ..Default::default()
    });
    app.world_mut().resource_mut::<WorldListState>().phase = WorldListPhase::Playing;
    {
        let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
        runtime.known_world_parts = previous_graph
            .parts()
            .map(|(part, spec)| (part, *spec))
            .collect();
        runtime.known_world_frames = previous_graph
            .parts()
            .map(|(part, _)| (part, previous_graph.part_frame(part).unwrap()))
            .collect();
        runtime.synced_editor_revision = 1;
        for part in [chassis, unrelated] {
            let support = valid_support();
            runtime.foundation_index.insert(part, &support);
            runtime
                .foundations
                .push(TerrainFoundation { part, support });
        }
    }
    app.add_systems(Update, sync_world_foundations);

    app.update();

    let runtime = app.world().resource::<WorldRuntime>();
    assert!(runtime.foundations_match_editor_revision(2));
    assert_eq!(
        runtime.anchored_parts().collect::<Vec<_>>(),
        vec![chassis, unrelated]
    );
    assert_eq!(
        runtime.static_parts_for_physics(2),
        Some(vec![chassis, unrelated])
    );
}

#[test]
fn empty_terrain_chunks_are_not_published_to_bevys_mesh_allocator() {
    let mut chunk = TerrainMeshChunk::default();
    assert!(!terrain_mesh_is_renderable(&chunk, 0));

    chunk.vertices = vec![[0.0; 3]; 3];
    assert!(!terrain_mesh_is_renderable(&chunk, 0));
    assert!(terrain_mesh_is_renderable(&chunk, 3));
}

#[test]
fn freeze_triangle_query_prunes_distant_geometry_and_converts_global_coordinates() {
    use mechanic_world::{
        TerrainTriangleGroupMask, TriangleBvh, TriangleBvhNode, TriangleBvhTriangle, WorldBounds,
    };
    let mut app = App::new();
    app.init_resource::<WorldRuntime>();
    let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
    runtime.floating_origin = FloatingOrigin(DVec3::splat(1000.0));
    runtime.player_terrain_ready = true;
    let bounds = WorldBounds {
        minimum: WorldPosition(DVec3::splat(1000.0)),
        maximum: WorldPosition(DVec3::splat(1002.0)),
    };
    let group_mask = TerrainTriangleGroupMask::REGULAR;
    let chunk = TerrainMeshChunk {
        origin: WorldPosition(DVec3::splat(1000.0)),
        vertices: vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
        triangle_bvh: TriangleBvh {
            bounds,
            nodes: vec![TriangleBvhNode {
                bounds,
                first_triangle: 0,
                triangle_count: 1,
                group_mask,
                ..Default::default()
            }],
            triangles: vec![TriangleBvhTriangle {
                indices: [0, 1, 2],
                group_mask,
            }],
        },
        ..Default::default()
    };
    runtime.active_terrain.insert(chunk.node, chunk);
    let mut visits = 0;
    assert!(
        runtime.freeze_triangles_clear(Vec3::ZERO, Vec3::ONE, |points| {
            visits += 1;
            assert_eq!(points[0], Vec3::ZERO);
            assert_eq!(points[1], Vec3::X * 2.0);
            true
        })
    );
    assert_eq!(visits, 1);
    assert!(
        runtime.freeze_triangles_clear(Vec3::splat(20.0), Vec3::splat(21.0), |_| panic!(
            "distant triangles must be pruned"
        ))
    );
    assert!(!runtime.freeze_triangles_clear(Vec3::ZERO, Vec3::ONE, |_| false));
}

#[test]
fn terrain_texture_coordinates_and_weights_are_chunk_seam_stable() {
    let chunk = TerrainMeshChunk {
        origin: WorldPosition(DVec3::new(15.0, 30.0, 45.0)),
        vertices: vec![[1.5, 3.0, 4.5]],
        normals: vec![[0.0, 1.0, 0.0]],
        material_weights: vec![[0.1, 0.2, 0.3, 0.15, 0.1, 0.15]],
        ..TerrainMeshChunk::default()
    };
    let mesh = terrain_chunk_mesh(&chunk, Vec::new());
    let Some(VertexAttributeValues::Float32x2(horizontal)) =
        mesh.attribute(bevy::mesh::Mesh::ATTRIBUTE_UV_0)
    else {
        panic!("terrain mesh must have horizontal texture coordinates")
    };
    let Some(VertexAttributeValues::Float32x2(vertical)) =
        mesh.attribute(bevy::mesh::Mesh::ATTRIBUTE_UV_1)
    else {
        panic!("terrain mesh must have vertical texture coordinates")
    };
    let Some(VertexAttributeValues::Float32x4(weights)) =
        mesh.attribute(bevy::mesh::Mesh::ATTRIBUTE_COLOR)
    else {
        panic!("terrain mesh must carry material weights as vertex colors")
    };
    assert_eq!(horizontal, &[[11.0, 33.0]]);
    assert_eq!(vertical, &[[22.0, 0.1]]);
    assert_eq!(weights, &[[0.1, 0.2, 0.3, 0.15]]);
}

#[test]
fn terrain_pbr_maps_keep_the_authored_1536_pixel_top_mip() {
    let maps: [&[u8]; 9] = [
        include_bytes!("../../assets/terrain/grass/grass_base_color.png"),
        include_bytes!("../../assets/terrain/grass/grass_normal.png"),
        include_bytes!("../../assets/terrain/grass/grass_orm.png"),
        include_bytes!("../../assets/terrain/dirt/dirt_base_color.png"),
        include_bytes!("../../assets/terrain/dirt/dirt_normal.png"),
        include_bytes!("../../assets/terrain/dirt/dirt_orm.png"),
        include_bytes!("../../assets/terrain/stone/stone_base_color.png"),
        include_bytes!("../../assets/terrain/stone/stone_normal.png"),
        include_bytes!("../../assets/terrain/stone/stone_orm.png"),
    ];
    for png in maps {
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(u32::from_be_bytes(png[16..20].try_into().unwrap()), 1536);
        assert_eq!(u32::from_be_bytes(png[20..24].try_into().unwrap()), 1536);
    }
}

#[test]
fn terrain_runtime_textures_receive_a_complete_mip_chain() {
    let pixel = [40_u8, 80, 120, 255];
    let top = pixel.repeat(8);
    let mut image = Image::new(
        Extent3d {
            width: 4,
            height: 2,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        top.clone(),
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::MAIN_WORLD,
    );

    generate_rgba8_mip_chain(&mut image).unwrap();

    assert_eq!(image.texture_descriptor.mip_level_count, 3);
    assert_eq!(
        image.data.as_ref().unwrap().len(),
        full_rgba8_mip_byte_count(4, 2)
    );
    assert_eq!(&image.data.as_ref().unwrap()[..top.len()], top);
    assert!(
        image
            .data
            .as_ref()
            .unwrap()
            .chunks_exact(4)
            .all(|sample| sample == pixel)
    );
}

#[test]
fn player_waits_for_collision_geometry_near_the_capsule() {
    let capsule = KinematicCapsule::new(WorldPosition(DVec3::new(0.0, 4.05, 0.0)));
    let mut chunk = TerrainMeshChunk {
        bounds: WorldBounds {
            minimum: WorldPosition(DVec3::new(-1.0, 3.0, -1.0)),
            maximum: WorldPosition(DVec3::new(1.0, 6.0, 1.0)),
        },
        ..TerrainMeshChunk::default()
    };
    assert!(!terrain_chunk_has_collision_near(
        &chunk,
        TerrainTransitionMask::NONE,
        &capsule,
    ));

    chunk.index_groups.regular = vec![0, 1, 2];
    assert!(terrain_chunk_has_collision_near(
        &chunk,
        TerrainTransitionMask::NONE,
        &capsule,
    ));
    chunk.bounds.minimum.0.x = 20.0;
    chunk.bounds.maximum.0.x = 25.0;
    assert!(!terrain_chunk_has_collision_near(
        &chunk,
        TerrainTransitionMask::NONE,
        &capsule,
    ));
}

#[test]
fn capsule_overlapping_nodes_are_streamed_first() {
    let capsule = KinematicCapsule::new(WorldPosition(DVec3::new(0.0, 4.05, 0.0)));
    let local = ActiveTerrainNode {
        id: TerrainNodeId::containing(BrickCoord::new(0, 2, 0), 2).unwrap(),
        generation: 0,
        transition_mask: TerrainTransitionMask::NONE,
    };
    let far = ActiveTerrainNode {
        id: TerrainNodeId::containing(BrickCoord::new(100, 100, 100), 2).unwrap(),
        ..local
    };
    assert_eq!(
        player_collision_nodes(&[far, local], &capsule).collect::<Vec<_>>(),
        vec![local.id],
    );
}

#[test]
fn generated_spawn_cut_contains_collision_pins() {
    let field = TerrainField::new(WorldSeed(7));
    let spawn = field.safe_spawn();
    let terrain = TerrainOctree::default().snapshot();
    let cut = select_active_nodes(&field, &terrain, spawn);
    let capsule = KinematicCapsule::new(spawn);
    let pins = player_collision_nodes(&cut, &capsule).collect::<Vec<_>>();
    assert!(!pins.is_empty());
    assert!(pins.into_iter().any(|id| {
        let node = cut
            .iter()
            .find(|node| node.id == id)
            .expect("a pin belongs to the selected cut");
        let chunk = mesh_chunk(
            &field,
            &terrain,
            TerrainMeshRequest {
                node: id,
                generation: node.generation,
                transition_mask: node.transition_mask,
            },
        );
        terrain_chunk_has_collision_near(&chunk, TerrainTransitionMask::NONE, &capsule)
    }));
}

#[test]
fn each_space_uses_exposure_matched_to_its_lighting() {
    assert!(
        (exposure_for_space(AppSpace::World).ev100 - Exposure::OVERCAST.ev100).abs() < f32::EPSILON
    );
    assert!(
        (exposure_for_space(AppSpace::Garage).ev100 - garage::EXPOSURE.ev100).abs() < f32::EPSILON
    );
}

#[test]
fn equal_and_two_to_one_nodes_share_the_expected_face() {
    let fine = TerrainNodeId::leaf(BrickCoord::new(1, 0, 0));
    let equal = TerrainNodeId::leaf(BrickCoord::new(2, 0, 0));
    let coarse = TerrainNodeId::containing(BrickCoord::new(2, 0, 0), 1).unwrap();
    assert!(nodes_touch_on_face(fine, equal, TerrainFace::PositiveX));
    assert!(nodes_touch_on_face(fine, coarse, TerrainFace::PositiveX));
}

#[test]
fn old_lod_waits_until_every_visible_replacement_is_published() {
    let parent = TerrainNodeId::containing(BrickCoord::new(0, 0, 0), 1).unwrap();
    let children = BTreeSet::from(parent.children().unwrap());
    let obsolete = BTreeSet::from([parent]);
    let mut published = children.clone();
    let missing = *published.first().unwrap();
    published.remove(&missing);

    assert!(ready_obsolete_nodes(&obsolete, &children, &published).is_empty());
    published.insert(missing);
    assert_eq!(
        ready_obsolete_nodes(&obsolete, &children, &published),
        obsolete
    );
}
#[test]
fn soil_commits_on_sixth_tick_and_survives_world_reload() {
    let temporary = TempDir::new("world-install");
    let store = WorldStore::new(&temporary.0);
    let document = store.create_world("Soil", Some(91)).unwrap();
    let mut app = App::new();
    app.init_resource::<WorldRuntime>();
    let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
    runtime.store = store;
    install_world(&mut runtime, document).unwrap();
    let patch = mechanic_world::SoilPatch {
        centre: WorldPosition(DVec3::new(0.0, runtime.field.surface_height(0.0, 0.0), 0.0)),
        normal: DVec3::Y,
        footprint: mechanic_world::LoadFootprint::square(DVec3::Y, 0.1),
        pressure_pa: 1.0e6,
        seconds: 1.0 / 60.0,
    };
    for _ in 0..5 {
        runtime.accumulate_soil(std::iter::once(patch));
    }
    assert!(runtime.pending_terrain_edits.is_empty());
    runtime.accumulate_soil(std::iter::once(patch));
    assert!(!runtime.pending_terrain_edits.is_empty());
    let commands = runtime.pending_terrain_edits.drain(..).collect();
    let result =
        super::brush::execute_terrain_edit_batch(runtime.edits.clone(), &runtime.field, commands)
            .unwrap();
    assert!(super::commit_terrain_edit_result(&mut runtime, result).0);
    assert!(!runtime.pending_foundation_edit.is_empty());
    let store = WorldStore::new(&temporary.0);
    let name = runtime.document.name.clone();
    store.save_dirty_leaves(&name, &mut runtime.edits).unwrap();
    let reloaded = store.load_octree(&name).unwrap();
    for brick in runtime.edits.snapshot().bricks() {
        assert_eq!(Some(brick), reloaded.brick(brick.coordinate()));
    }
    let command = super::TerrainEditCommand {
        centre: patch.centre,
        radius_metres: 0.1,
        previous: None,
        operation: TerrainEditOperation::Remove,
    };
    runtime.pending_terrain_edits =
        std::iter::repeat_n(command, super::MAX_PENDING_TERRAIN_EDITS).collect();
    runtime.accumulate_soil(std::iter::once(patch));
    assert_eq!(
        runtime.pending_terrain_edits.len(),
        super::MAX_PENDING_TERRAIN_EDITS
    );
    assert!(runtime.pending_soil.take_ready().is_empty());
}

#[test]
fn saving_an_unpublished_transfer_keeps_the_previous_material_owner() {
    let temporary = TempDir::new("world-install");
    let store = WorldStore::new(&temporary.0);
    let document = store.create_world("Material", Some(91)).unwrap();
    let mut app = App::new();
    app.init_resource::<WorldRuntime>();
    let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
    runtime.store = store;
    install_world(&mut runtime, document).unwrap();
    let centre = WorldPosition(DVec3::new(0.025, 200.025, 0.025));
    let field = runtime.field.clone();
    runtime
        .edits
        .add_sphere(&field, centre, 0.1, TerrainMaterial::Rock)
        .unwrap();
    let cell = centre.cell().unwrap();
    let source = mechanic_world::ExtractionCell {
        cell,
        sample: runtime.edits.sample_cell(&field, cell),
        throw: DVec3::ZERO,
    };
    let transfer = runtime
        .clumps
        .prepare_extraction(&runtime.edits, &field, &[source])
        .unwrap();
    runtime.pending_material = Some(super::PendingMaterialPublication {
        previous: runtime.edits.clone(),
        clumps: transfer.clumps,
        sources: vec![source],
    });
    super::commit_terrain_edit_result(
        &mut runtime,
        super::TerrainEditTaskResult {
            terrain: transfer.terrain,
            outcomes: vec![transfer.outcome],
            elapsed_ms: 0.0,
        },
    );
    super::saving::save_all(&mut runtime).unwrap();
    let (saved, clumps) = runtime.store.load_material_state("Material").unwrap();
    assert!(saved.sample_cell(&field, cell).is_solid());
    assert!(clumps.bodies.is_empty());
    super::brush::finish_terrain_edits(&mut runtime).unwrap();
    assert!(runtime.edits.sample_cell(&field, cell).is_solid());
    assert!(!runtime.material_publication_pending());
}

/// Updates until `done`, failing instead of waiting forever on a stalled pipeline.
fn update_until(app: &mut App, what: &str, done: impl Fn(&App) -> bool) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_mins(2);
    while !done(app) {
        assert!(std::time::Instant::now() < deadline, "timed out: {what}");
        app.update();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

#[test]
fn a_settled_saved_clump_neither_blocks_loading_nor_waits_for_physics_to_deposit() {
    use bevy::prelude::IntoScheduleConfigs;

    bevy::tasks::AsyncComputeTaskPool::get_or_init(bevy::tasks::TaskPool::new);
    let temporary = TempDir::new("world-install");
    let store = WorldStore::new(&temporary.0);
    let document = store.create_world("Settled", Some(91)).unwrap();
    let mut app = App::new();
    app.init_resource::<WorldRuntime>()
        .init_resource::<WorldListState>()
        .init_resource::<WorldDiagnostics>()
        .init_resource::<EditorState>()
        .init_resource::<EditorGraph>()
        .init_resource::<crate::simulation::state::AppSimulation>()
        .init_resource::<bevy::prelude::Assets<bevy::prelude::Mesh>>()
        .add_systems(
            Update,
            (
                super::coordinate_terrain_edits,
                super::schedule_terrain_remeshes,
                super::integrate_terrain_remeshes,
            )
                .chain(),
        );
    let player = {
        let mut runtime = app.world_mut().resource_mut::<WorldRuntime>();
        runtime.store = store;
        install_world(&mut runtime, document).unwrap();
        runtime.terrain_material = Some(bevy::prelude::Handle::default());
        let spawn = runtime.capsule.position.0;
        let surface = runtime.field.surface_height(spawn.x, spawn.z);
        // Saved exactly as a play session leaves it: soft, at rest, and
        // already past the deposit delay, with no physics scene yet.
        let clump = mechanic_world::MaterialClump {
            id: 1,
            material: TerrainMaterial::Soil,
            quanta: 510 * 8,
            half_extents: DVec3::splat(0.05),
            position: WorldPosition(DVec3::new(spawn.x, surface + 0.05, spawn.z)),
            rotation: bevy::math::DQuat::IDENTITY,
            linear_velocity: DVec3::ZERO,
            angular_velocity: DVec3::ZERO,
            settled_seconds: 300.0,
            sleeping: false,
        };
        assert!(clump.is_valid() && clump.can_deposit());
        runtime.clumps.bodies.insert(clump.id, clump);
        runtime.clumps.next_id = 2;
        (spawn - runtime.floating_origin.0).as_vec3()
    };
    app.insert_resource(crate::camera::PlayerState {
        position: player,
        ..Default::default()
    });
    app.world_mut().resource_mut::<WorldListState>().phase = WorldListPhase::Loading;

    update_until(&mut app, "the world never finished loading", |app| {
        let playing = app.world().resource::<WorldListState>().phase() == WorldListPhase::Playing;
        // A transfer holds terrain publication, and with it loading progress.
        assert!(
            playing
                || !app
                    .world()
                    .resource::<WorldRuntime>()
                    .material_publication_pending()
        );
        playing
    });
    assert!(
        app.world()
            .resource::<crate::simulation::state::AppSimulation>()
            .cpu
            .is_none()
    );

    update_until(&mut app, "the clump never deposited", |app| {
        let runtime = app.world().resource::<WorldRuntime>();
        runtime.clumps.bodies.is_empty() && !runtime.material_publication_pending()
    });
}

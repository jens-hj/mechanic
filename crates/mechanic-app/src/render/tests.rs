use std::time::Instant;

#[test]
#[ignore = "CPU-only saved-world visual rebuild measurement"]
#[expect(
    clippy::too_many_lines,
    reason = "keep the saved-world stage measurements together"
)]
fn measure_builder_body_mesh_rebuild() {
    let source = crate::env::text(crate::env::EDIT_FIXTURE).map_or_else(
        || {
            include_str!(
                "../../../mechanic-bench/tests/fixtures/builder-world/generations/20/world.ron"
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
            if !matches!(spec, crate::PartSpec::Cuboid(_)) {
                continue;
            }
            let owner = mechanic_core::SolidOwner::Part(part);
            let solid = graph.evaluated_solid_shared(owner).unwrap();
            for edge in solid.logical_edges.iter().take(1) {
                let mut preview = graph.clone();
                let step = Instant::now();
                let _ = preview.apply(crate::BuildCommand::AddShapeFeature(
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
            for material in crate::ConstructionMaterial::ALL {
                if crate::render::mesh::simulation::simulation_material_is_present_for_compound(
                    &graph, &creation, body, material,
                ) {
                    let mesh_started = Instant::now();
                    vertices += crate::render::mesh::simulation::local_simulation_material_mesh(
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
        AlphaMode, App, Color, EnvironmentMapLight, GeneratedEnvironmentMapLight, Handle, IVec3,
        Image, Mesh, Quat, StandardMaterial, Update, Vec2, Vec3,
    },
};
use mechanic_core::{
    BearingDimensions, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph,
    ConstructionMaterial, ControllerSpec, CuboidSpec, CylinderDimensions, CylinderSpec,
    DimensionLinkId, DimensionLinkSpec, DriveLimits, DriveLinkSpec, DriveProgram, DriveState,
    DriveTarget, EdgeChainRef, EdgeTreatment, EngineKind, EngineSpec, FaceKind, FaceOwner, FaceRef,
    GridRotation, InputSpec, MaterialAppearance, MaterialColor, MaterialDye, MaterialFinish,
    MaterialShift, PartSpec, PipeBendDimensions, SeatSpec, ServoSpec, ShapeFeature, SolidOwner,
};
use mechanic_gpu::GpuTransform;

use crate::builder::BEARING_DEPTH;
use crate::editor::build_actions::PlacedBearing;
use crate::editor::preview::{
    BLOCK_SHEET_PREVIEW_INSET_METERS, bearing_preview_dimensions_changed, drive_xray_is_visible,
    joint_xray_is_visible, should_sync_editor_visual_meshes,
};
use crate::pose::transform_from_gpu;
use crate::render::authored::AuthoredPart;
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
        .init_resource::<crate::world::WorldRuntime>()
        .init_resource::<crate::shape_tool::ShapeMirror>()
        .init_resource::<crate::sequencer::DriveSequencer>()
        .init_resource::<crate::hotbar::SelectedTool>()
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
    use crate::editor::preview::{BearingVisual, ConstructionVisual, EditorVisuals};
    use crate::editor::state::EditorState;
    use crate::hotbar::SelectedTool;
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
    use crate::editor::preview::{BearingVisual, ConstructionVisual, EditorVisuals};
    use crate::editor::state::EditorState;
    use crate::hotbar::SelectedTool;
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
        crate::render::mesh::construction::ordinary_material(*graph.part(part).unwrap()).unwrap();
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
        crate::GpuPhysicsConfig::default(),
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
    .init_resource::<crate::world::WorldRuntime>()
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
            BuildPose::from_position_ticks(IVec3::new(-3_150, 50, -3_150), GridRotation::default()),
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
    crate::testing::assert_within_budget(
        elapsed,
        std::time::Duration::from_millis(5),
        "publishing a 4,096-block sheet mesh",
    );
}

use crate::builder::PlacementPlane;
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
    let Some(VertexAttributeValues::Float32x3(values)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION)
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
    let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::SpawnDimensionLink(spec)).unwrap()
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
    let Some(VertexAttributeValues::Float32x2(uvs)) = mesh.attribute(Mesh::ATTRIBUTE_UV_0) else {
        panic!("material UVs use Float32x2")
    };
    assert!((uvs[1][0] - positions[1].x / MATERIAL_TEXTURE_METERS_PER_REPEAT).abs() < 1.0e-6);
    assert!((uvs[0][0] - positions[0].x / MATERIAL_TEXTURE_METERS_PER_REPEAT).abs() < 1.0e-6);
    let expected_span = 4.0 * MATERIAL_TEXTURE_PIXELS_PER_BLOCK / MATERIAL_TEXTURE_PIXELS_PER_SIDE;
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
        (maximum_radius - (dimensions.outer_diameter() * 0.5 - BEARING_RENDER_RADIAL_SKIN)).abs()
            < 1.0e-6
    );
    assert!(
        (minimum_radius - (dimensions.inner_diameter() * 0.5 + BEARING_RENDER_RADIAL_SKIN)).abs()
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
    let Some(VertexAttributeValues::Float32x2(uvs)) = mesh.attribute(Mesh::ATTRIBUTE_UV_0) else {
        panic!("bearing UVs use Float32x2")
    };
    let Some(VertexAttributeValues::Float32x4(tangents)) = mesh.attribute(Mesh::ATTRIBUTE_TANGENT)
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
    assert!(
        normals.iter().any(|normal| normal[1].abs() < 1.0e-6
            && normal[0].abs() > 0.5
            && normal[2].abs() > 0.5)
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
    let straight_length = cylinder_dimensions.axial_length() / MATERIAL_TEXTURE_METERS_PER_REPEAT;
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
    let inner_circumference_step = std::f32::consts::TAU / 24.0 * dimensions.inner_diameter() * 0.5
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
    let endpoint = crate::builder::face_geometry_from_ref(FaceRef::part(part, face), Some(graph));
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
        assert!((offsets[&first].v_angle - second_angle - offsets[&second].v_angle).abs() < 1.0e-6);
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
    let Some(VertexAttributeValues::Float32x3(normals)) = mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
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
        let Some(VertexAttributeValues::Float32x3(actual)) = mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
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
    let Some(VertexAttributeValues::Float32x3(normals)) = mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
    else {
        panic!("mesh must have float3 normals")
    };
    let Some(VertexAttributeValues::Float32x4(tangents)) = mesh.attribute(Mesh::ATTRIBUTE_TANGENT)
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
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
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
        (placed_radius - (placed_dimensions.outer_diameter() * 0.5 - BEARING_RENDER_RADIAL_SKIN))
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
    let BuildOutcome::Spawned(support) = graph.apply(BuildCommand::Spawn(support)).unwrap() else {
        unreachable!()
    };
    let targets = [IVec3::new(0, 9, 0), IVec3::new(2, 9, 0)].map(|center| {
        let spec = CuboidSpec::new(
            [1, 1, 1],
            BuildPose::from_half_grid(center, GridRotation::default()),
        )
        .unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
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
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
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
    let electric = combined_authored_construction_mesh(&graph, AuthoredPart::ElectricEngine, None);
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
    let Some(VertexAttributeValues::Float32x2(uvs)) = mesh.attribute(Mesh::ATTRIBUTE_UV_0) else {
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
fn a_sliding_joints_drive_arrow_runs_straight_along_its_travel_and_flips_with_the_target() {
    use crate::render::mesh::drive::{append_travel_indicator, travel_line};
    let piston = mechanic_core::BearingKind::Piston(mechanic_core::Piston::default());
    let (start, end) = travel_line(piston, Vec3::Y).expect("a piston slides");
    assert!(start.abs_diff_eq(Vec3::ZERO, 1.0e-6));
    assert!(travel_line(mechanic_core::BearingKind::Rotational, Vec3::Y).is_none());

    let arrow = |target| {
        let (mut positions, mut normals, mut indices) = (Vec::new(), Vec::new(), Vec::new());
        append_travel_indicator(
            start,
            end,
            DriveState::new(target).unwrap(),
            &mut positions,
            &mut normals,
            &mut indices,
        );
        positions.into_iter().map(Vec3::from).collect::<Vec<_>>()
    };
    let extending = arrow(DriveTarget::LinearPosition(1.0));
    assert!(
        extending
            .iter()
            .all(|point| point.x.abs() < 0.07 && point.z.abs() < 0.07)
    );
    assert!(extending.iter().any(|point| point.abs_diff_eq(end, 1.0e-6)));
    let retracting = arrow(DriveTarget::LinearSpeed(-1.0));
    assert!(
        retracting
            .iter()
            .any(|point| point.abs_diff_eq(start, 1.0e-6))
    );
    assert!(
        !retracting
            .iter()
            .any(|point| point.abs_diff_eq(end, 1.0e-6))
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

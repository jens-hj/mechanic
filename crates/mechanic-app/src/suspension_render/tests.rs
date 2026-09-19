use super::*;
use mechanic_core::{
    BearingDimensions, BuildCommand, BuildOutcome, BuildPose, CuboidSpec, FaceKind, FaceRef,
    ShockSpec,
};
fn fixture() -> (ConstructionGraph, PlacedBearing) {
    let mut graph = ConstructionGraph::new();
    let BuildOutcome::Spawned(part) = graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([1; 3], BuildPose::default()).unwrap(),
        ))
        .unwrap()
    else {
        panic!("part")
    };
    let spec = SuspensionSpec::new(None, Some(ShockSpec::default()), None).unwrap();
    let socket = PlacedBearing {
        kind: BearingKind::Suspension(spec),
        source: FaceRef::part(part, FaceKind::PositiveY),
        axis: Vec3::Y,
        anchor: Vec3::Y * 0.125,
        dimensions: BearingDimensions::default(),
    };
    (graph, socket)
}
#[test]
fn moving_ghost_and_fit_feedback_reuse_geometry() {
    let (_, socket) = fixture();
    let original = preview_spec(socket, true, None);
    let mut moved = original;
    moved.socket.anchor += Vec3::X;
    moved.socket.axis = Vec3::Z;
    moved.preview_valid = Some(false);
    assert!(original.same_geometry(moved));
    moved.socket.kind = BearingKind::Suspension(
        SuspensionSpec::new(Some(mechanic_core::SpringSpec::default()), None, None).unwrap(),
    );
    assert!(!original.same_geometry(moved));
}
#[test]
fn triangle_picking_returns_shock_body_owner_and_misses_empty_shaft_space() {
    let (graph, socket) = fixture();
    let hit = raycast_scene_component(
        &graph,
        None,
        &[socket],
        Vec3::new(0.2, 0.24, 0.0),
        Vec3::NEG_X,
    )
    .unwrap();
    assert_eq!(hit.0, 0);
    assert_eq!(hit.2, SuspensionMeshOwner::Source);
    assert!((hit.1 - 0.15).abs() < 0.002);
    assert!(
        raycast_scene(
            &graph,
            None,
            &[socket],
            Vec3::new(0.2, 0.5, 0.04),
            Vec3::NEG_X
        )
        .is_none()
    );
}
#[test]
fn rendered_mesh_uses_chroma_payload_and_ghost_omits_it() {
    let spec = SuspensionSpec::new(
        Some(mechanic_core::SpringSpec::default()),
        Some(ShockSpec::default()),
        Some(mechanic_core::BumpStopSpec::new(0.05, 0.06).unwrap()),
    )
    .unwrap();
    for chunk in suspension_meshes(spec, 0.0) {
        assert!(
            render_mesh(&chunk, None)
                .attribute(Mesh::ATTRIBUTE_TANGENT)
                .is_some(),
            "finish {}",
            chunk.finish
        );
    }
    let chunk = suspension_meshes(spec, 0.0).remove(0);
    let appearance = mechanic_core::MaterialAppearance::BAKED;
    let mesh = render_mesh(&chunk, Some(appearance));
    let bevy::mesh::VertexAttributeValues::Float32x4(colors) =
        mesh.attribute(Mesh::ATTRIBUTE_COLOR).unwrap()
    else {
        panic!("Chroma payload")
    };
    assert_eq!(colors.len(), chunk.positions.len());
    assert!(colors.iter().all(|color| color.map(f32::to_bits)
        == super::super::chroma::encode_appearance(appearance).map(f32::to_bits)));
    assert!(
        render_mesh(&chunk, None)
            .attribute(Mesh::ATTRIBUTE_COLOR)
            .is_none()
    );
}
#[test]
fn finish_modulation_preserves_guide_colour_over_existing_dark_textures() {
    let base = ConstructionRenderMaterial {
        base: StandardMaterial::default(),
        extension: crate::chroma::ChromaMaterialExtension {
            tint_mask: Handle::default(),
            base_lightness: Vec4::ZERO,
        },
    };
    for finish in SUSPENSION_FINISHES {
        let material = finish_material(&base, finish);
        let representative =
            super::super::chroma::material_profile(finish.material).representative_srgb;
        let baked =
            Color::srgb_u8(representative[0], representative[1], representative[2]).to_linear();
        let multiplier = material.base.base_color.to_linear();
        let target = Color::srgb_u8(finish.color[0], finish.color[1], finish.color[2]).to_linear();
        assert!((baked.red * multiplier.red - target.red).abs() < 1e-6);
        assert!((baked.green * multiplier.green - target.green).abs() < 1e-6);
        assert!((baked.blue * multiplier.blue - target.blue).abs() < 1e-6);
    }
}
#[test]
fn insertion_preview_retains_nonzero_joint_compression_and_source_pose() {
    let (graph, socket) = fixture();
    let candidate = crate::builder::plate_block_candidate(socket).unwrap();
    let graph = crate::builder::stage_plate_block(
        &graph,
        socket,
        candidate,
        &[],
        crate::PlacementBounds::Garage,
    )
    .unwrap();
    let creation = graph.compile().unwrap();
    let host = visual_specs(&graph, &[socket], Some(&creation))[0];
    let host_spec = host.suspension();
    let replacement = host_spec
        .with_components(
            Some(mechanic_core::SpringSpec::default()),
            host_spec.shock(),
            None,
            true,
        )
        .unwrap();
    let mut preview = preview_spec(
        PlacedBearing {
            kind: BearingKind::Suspension(replacement),
            ..socket
        },
        true,
        None,
    );
    preview.source_body = host.source_body;
    preview.joint = host.joint;
    preview.preview_host = Some(host_spec);
    let world_rotation = Quat::from_rotation_z(0.6);
    let world_offset = Vec3::new(3.0, 2.0, -1.0);
    let mut transforms = creation
        .compounds
        .iter()
        .map(|body| mechanic_gpu::GpuTransform {
            position: (world_offset + world_rotation * body.root_translation)
                .extend(0.0)
                .to_array(),
            rotation: (world_rotation * body.root_rotation).to_array(),
        })
        .collect::<Vec<_>>();
    let target = creation.bearings[host.joint.unwrap()].compound_b as usize;
    let initial = transforms[target].position;
    let key = preview;
    for compression in [0.025, 0.075, 0.1] {
        let position = Vec3::from_slice(&initial[..3]) - world_rotation * socket.axis * compression;
        transforms[target].position = position.extend(0.0).to_array();
        let host_pose = snapshot_pose(&host, &creation, &transforms);
        let ghost = insertion_pose(&preview, host_spec, host_pose);
        assert!((ghost.1 - compression).abs() < 1e-5);
        assert!(
            (replacement.extended_length() - ghost.1 - (host_spec.extended_length() - host_pose.1))
                .abs()
                < 1e-6
        );
        assert!(
            ghost
                .0
                .translation
                .distance(world_offset + world_rotation * socket.anchor)
                < 1e-5
        );
        assert!((ghost.0.rotation * Vec3::Y).distance(world_rotation * socket.axis) < 1e-5);
        assert!(preview == key);
    }
}
#[test]
fn local_preview_matches_authored_mount_and_follows_moving_frame_without_rekeying() {
    let (_, local_socket) = fixture();
    let authored =
        ConstructionFrame::new(Vec3::new(3.0, -1.0, 2.0), Quat::from_rotation_z(0.7)).unwrap();
    let preview = preview_spec(local_socket, true, Some(authored));
    let committed = VisualSpec {
        socket: super::super::live_edit::transform_bearing(local_socket, authored),
        source_body: None,
        joint: None,
        preview_valid: None,
        preview_host: None,
    };
    assert!(preview.same_mount(committed));
    for world in [
        authored,
        ConstructionFrame::new(Vec3::new(-2.0, 4.0, 8.0), Quat::from_rotation_x(1.2)).unwrap(),
    ] {
        let (transform, _) = render_pose(&preview, None, Some(world.compose(authored.inverse())));
        assert!(
            transform
                .translation
                .distance(world.point(local_socket.anchor))
                < 1e-5
        );
        assert!((transform.rotation * Vec3::Y).distance(world.vector(local_socket.axis)) < 1e-5);
        // Frame motion is outside VisualSpec's topology/material cache key.
        assert!(preview == preview_spec(local_socket, true, Some(authored)));
    }
    let (placed, _) = render_pose(&committed, None, Some(authored));
    assert!(placed.translation.distance(committed.socket.anchor) < 1e-6);
}
#[test]
fn stopped_snapshot_retains_build_pose_and_starting_compression() {
    let (graph, socket) = fixture();
    let creation = graph.compile().unwrap();
    let transforms = creation
        .compounds
        .iter()
        .map(|compound| mechanic_gpu::GpuTransform {
            position: (compound.root_translation + Vec3::splat(5.0))
                .extend(0.0)
                .to_array(),
            rotation: Quat::IDENTITY.to_array(),
        })
        .collect();
    let simulation = AppSimulation {
        creation: Some(creation),
        transforms,
        published_graph: graph.clone(),
        ..default()
    };
    let (pose, compression) = socket_pose(&graph, Some(&simulation), socket);
    assert_eq!(pose.translation, socket.anchor);
    assert!(compression.abs() < 1e-7);
}
#[test]
fn damping_and_camera_motion_share_render_and_pick_geometry_keys() {
    let (_, socket) = fixture();
    let BearingKind::Suspension(spec) = socket.kind else {
        panic!("suspension");
    };
    let changed = crate::suspension_controls::Parameter::Compression
        .edit(spec, 25.0, false)
        .unwrap();
    assert!(same_geometry(spec, changed));
    let a = preview_spec(socket, true, None);
    let mut b = a;
    b.socket.kind = BearingKind::Suspension(changed);
    b.socket.anchor += Vec3::X;
    b.preview_valid = None;
    assert!(a.same_geometry(b));
    assert!(!same_geometry(
        spec,
        crate::suspension_controls::Parameter::ShockOd
            .edit(spec, 0.1025, false)
            .unwrap()
    ));
}
#[test]
fn free_mount_preview_moves_while_attached_preview_preserves_live_spacing() {
    let original =
        SuspensionSpec::new(None, Some(mechanic_core::ShockSpec::default()), None).unwrap();
    let draft = crate::suspension_controls::Parameter::ShockLength
        .edit(original, 0.55, false)
        .unwrap();
    assert!(draft_compression(original, draft, 0.02, false).abs() < 1e-6);
    assert!((draft_compression(original, draft, 0.02, true) - 0.07).abs() < 1e-6);
}
#[test]
fn cached_mesh_has_exactly_one_material_during_preview_and_after_release() {
    let mut world = World::new();
    let entity = world.spawn(Mesh3d(Handle::default())).id();
    let mut queue = bevy::ecs::world::CommandQueue::default();
    for preview in [false, true, false] {
        set_chunk_material(
            &mut Commands::new(&mut queue, &world),
            entity,
            preview.then(Handle::default),
            Handle::default(),
        );
        queue.apply(&mut world);
        assert_eq!(
            world
                .get::<MeshMaterial3d<StandardMaterial>>(entity)
                .is_some(),
            preview
        );
        assert_eq!(
            world
                .get::<MeshMaterial3d<ConstructionRenderMaterial>>(entity)
                .is_some(),
            !preview
        );
        assert!(world.get::<Mesh3d>(entity).is_some());
    }
}

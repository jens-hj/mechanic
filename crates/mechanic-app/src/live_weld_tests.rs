use super::*;
use mechanic_core::{
    BearingSpec, BuildOutcome, BuildPose, CuboidSpec, FaceKind, FaceRef, GridRotation,
};

fn spawn(graph: &mut ConstructionGraph, units: IVec3) -> PartId {
    let spec = CuboidSpec::new([2, 2, 2], BuildPose::new(units, GridRotation::default())).unwrap();
    let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
        panic!("part expected")
    };
    part
}

fn pose(position: Vec3, rotation: Quat) -> GpuTransform {
    GpuTransform {
        position: position.extend(0.0).to_array(),
        rotation: rotation.to_array(),
    }
}

fn simulation(graph: &ConstructionGraph, poses: Vec<GpuTransform>) -> AppSimulation {
    AppSimulation {
        creation: Some(graph.compile().unwrap()),
        // Deliberately stale interpolation must not determine whether targets touch.
        transforms: vec![pose(Vec3::splat(100.0), Quat::IDENTITY); poses.len()],
        live_state: Some(crate::LivePhysicsState {
            tick: 12,
            velocities: vec![
                mechanic_gpu::GpuVelocity {
                    linear: [0.0; 4],
                    angular: [0.0; 4]
                };
                poses.len()
            ],
            coordinates: Vec::new(),
            transforms: poses,
        }),
        ..Default::default()
    }
}

#[test]
fn arbitrary_rotation_about_contact_normal_welds_without_changing_local_grids() {
    let mut graph = ConstructionGraph::new();
    let first = spawn(&mut graph, IVec3::ZERO);
    let second = spawn(&mut graph, IVec3::new(16, 0, 0));
    let original = graph.clone();
    let common =
        ConstructionFrame::new(Vec3::new(8.0, 3.0, -2.0), Quat::from_rotation_y(0.4)).unwrap();
    let second_rotation = Quat::from_rotation_x(0.37);
    let poses = vec![
        pose(common.translation(), common.rotation()),
        pose(
            common.point(Vec3::X * 0.5),
            common.rotation() * second_rotation,
        ),
    ];
    let simulation = simulation(&graph, poses.clone());
    let welded = stage(&graph, &simulation, first, second).unwrap();
    assert!(graph.shares_revision(&original));
    assert_eq!(welded.part(second), graph.part(second));
    assert_eq!(welded.compile().unwrap().compounds.len(), 1);
    assert!(
        common
            .point(welded.part_position(second).unwrap())
            .abs_diff_eq(Vec3::from_slice(&poses[1].position[..3]), 1.0e-5)
    );
    assert!(
        (common.rotation() * welded.part_rotation(second).unwrap())
            .abs_diff_eq(Quat::from_array(poses[1].rotation), 1.0e-5)
    );
    assert!(
        common
            .point(welded.part_position(first).unwrap())
            .abs_diff_eq(common.translation(), 1.0e-5)
    );
}

#[test]
fn edge_contact_uses_validated_rigid_membership_and_survives_save_round_trip() {
    let mut graph = ConstructionGraph::new();
    let first = spawn(&mut graph, IVec3::ZERO);
    let second = spawn(&mut graph, IVec3::new(16, 0, 0));
    graph
        .apply(BuildCommand::BeginPending(
            mechanic_core::PendingOperation::Weld(FaceRef::part(first, FaceKind::PositiveX)),
        ))
        .unwrap();
    let second_position = Vec3::new(0.25 + 0.25 * 2.0_f32.sqrt(), 3.0, 0.0);
    let simulation = simulation(
        &graph,
        vec![
            pose(Vec3::Y * 3.0, Quat::IDENTITY),
            pose(
                second_position,
                Quat::from_rotation_y(std::f32::consts::FRAC_PI_4),
            ),
        ],
    );
    let welded = stage(&graph, &simulation, first, second).unwrap();
    assert_eq!(welded.rigid_links().count(), 1);
    assert!(welded.pending().is_none());
    let document = mechanic_core::CreationDocument::from_graph(&welded, "Edge weld", &[]);
    let restored = document.into_graph().unwrap().graph;
    assert_eq!(restored.compile().unwrap().compounds.len(), 1);
    assert!(
        restored
            .part_position(second)
            .unwrap()
            .abs_diff_eq(second_position - Vec3::Y * 3.0, 1.0e-5)
    );
}

#[test]
fn separated_or_deeply_penetrating_world_targets_are_refused() {
    let mut graph = ConstructionGraph::new();
    let first = spawn(&mut graph, IVec3::ZERO);
    let second = spawn(&mut graph, IVec3::new(16, 0, 0));
    for distance in [1.0, 0.1] {
        let simulation = simulation(
            &graph,
            vec![
                pose(Vec3::Y * 3.0, Quat::IDENTITY),
                pose(Vec3::new(distance, 3.0, 0.0), Quat::IDENTITY),
            ],
        );
        assert!(stage(&graph, &simulation, first, second).is_err());
    }
    assert_eq!(graph.rigid_links().count(), 0);
    assert_eq!(graph.weld_count(), 0);
}

#[test]
fn touching_live_bodies_cannot_merge_when_articulated_defaults_intersect() {
    let mut graph = ConstructionGraph::new();
    let first = spawn(&mut graph, IVec3::ZERO);
    let second = spawn(&mut graph, IVec3::new(16, 0, 0));
    let arm = spawn(&mut graph, IVec3::new(16, 2, 0));
    let corner = spawn(&mut graph, IVec3::new(14, 2, 0));
    let foot = spawn(&mut graph, IVec3::new(14, 0, 0));
    for (a, b) in [(arm, corner), (corner, foot)] {
        graph
            .apply(BuildCommand::RigidLink(RigidLinkSpec {
                first: a,
                second: b,
            }))
            .unwrap();
    }
    graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(second, FaceKind::PositiveY),
            FaceRef::part(arm, FaceKind::NegativeY),
            Vec3::new(4.0, 0.25, 0.0),
            Vec3::Y,
        )))
        .unwrap();
    // A bent arm swings to the right, clear of the first creation. Its default
    // left-hand foot would return to x=0 and overlap the proposed weld partner.
    let turn = Quat::from_rotation_y(std::f32::consts::PI);
    let child_motion = ConstructionFrame::new(
        Vec3::new(0.5, 3.25, 0.0) - turn * Vec3::new(4.0, 0.25, 0.0),
        turn,
    )
    .unwrap();
    let compiled = graph.compile().unwrap();
    let poses = compiled
        .compounds
        .iter()
        .map(|body| {
            if body.source_parts.contains(&first) {
                pose(Vec3::Y * 3.0, Quat::IDENTITY)
            } else if body.source_parts.contains(&second) {
                pose(Vec3::new(0.5, 3.0, 0.0), Quat::IDENTITY)
            } else {
                pose(
                    child_motion.point(body.root_translation),
                    turn * body.root_rotation,
                )
            }
        })
        .collect();
    let simulation = simulation(&graph, poses);
    let original = graph.clone();
    let error = stage(&graph, &simulation, first, second).unwrap_err();
    assert!(error.contains("Garage default pose"), "{error}");
    assert!(graph.shares_revision(&original));
    assert_eq!(graph.bearing_count(), 1);
}

#[test]
fn convex_vertices_use_body_origin_without_adding_centroid_twice() {
    let mut graph = ConstructionGraph::new();
    spawn(&mut graph, IVec3::ZERO);
    let creation = graph.compile().unwrap();
    let mut collider = creation.colliders[0].clone();
    let vertices = [-0.25, 0.25]
        .into_iter()
        .flat_map(|x| {
            [-0.25, 0.25].into_iter().flat_map(move |y| {
                [-0.25, 0.25]
                    .into_iter()
                    .map(move |z| Vec3::new(x + 2.0, y, z))
            })
        })
        .collect();
    collider.local_center = Vec3::X * 2.0;
    collider.shape = ColliderShape::Convex(mechanic_core::CompiledConvex {
        vertices,
        face_planes: vec![
            Vec3::X.extend(2.25),
            Vec3::NEG_X.extend(-1.75),
            Vec3::Y.extend(0.25),
            Vec3::NEG_Y.extend(0.25),
            Vec3::Z.extend(0.25),
            Vec3::NEG_Z.extend(0.25),
        ],
        edge_directions: vec![Vec3::X, Vec3::Y, Vec3::Z],
    });
    let convex = geometry(&collider, pose(Vec3::ZERO, Quat::IDENTITY)).unwrap();
    let box_at_contact =
        geometry(&creation.colliders[0], pose(Vec3::X * 2.5, Quat::IDENTITY)).unwrap();
    assert!(penetration(&convex, &box_at_contact).abs() < 1.0e-5);
}

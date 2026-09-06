use super::*;
use mechanic_core::{
    BuildCommand, BuildOutcome, BuildPose, ColliderShape, ConstructionGraph, CuboidSpec,
    GridRotation, PartId,
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

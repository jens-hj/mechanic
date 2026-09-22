//! Which colliders of joined bodies may meet: whatever was built apart.

use super::super::*;
use crate::MachineState;
use bevy_math::{IVec3, Vec3};
use mechanic_core::{
    BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, ConstructionMaterial,
    CuboidSpec, FaceKind, FaceRef, GridRotation, PartId, RigidLinkSpec,
};

fn block(
    graph: &mut ConstructionGraph,
    ticks: IVec3,
    dimensions: [u8; 3],
    material: ConstructionMaterial,
) -> PartId {
    let mut spec = CuboidSpec::new(
        dimensions,
        BuildPose::from_position_ticks(ticks, GridRotation::default()),
    )
    .unwrap();
    spec.material = material;
    let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
        panic!("spawn expected");
    };
    part
}

fn weld(graph: &mut ConstructionGraph, first: PartId, second: PartId) {
    graph
        .apply(BuildCommand::RigidLink(RigidLinkSpec { first, second }))
        .unwrap();
}

#[test]
fn joined_bodies_meet_only_where_they_were_built_apart() {
    let mut graph = ConstructionGraph::new();
    // A 2 m base of two welded blocks in different materials, so they stay
    // two colliders, with a stop standing on the far end.
    let near = block(
        &mut graph,
        IVec3::new(-200, 0, 0),
        [4, 1, 1],
        ConstructionMaterial::Wood,
    );
    let far = block(
        &mut graph,
        IVec3::new(200, 0, 0),
        [4, 1, 1],
        ConstructionMaterial::Steel,
    );
    let stop = block(
        &mut graph,
        IVec3::new(350, 100, 0),
        [1, 1, 1],
        ConstructionMaterial::Steel,
    );
    weld(&mut graph, near, far);
    weld(&mut graph, far, stop);
    // A slider on the near block's top, hinged to it, 1.6 m short of the stop.
    let slider = block(
        &mut graph,
        IVec3::new(-300, 100, 0),
        [1, 1, 1],
        ConstructionMaterial::Wood,
    );
    graph
        .apply(BuildCommand::AddBearing(BearingSpec::new(
            FaceRef::part(near, FaceKind::PositiveY),
            FaceRef::part(slider, FaceKind::NegativeY),
            Vec3::new(-0.75, 0.125, 0.0),
            Vec3::Y,
        )))
        .unwrap();
    let creation = graph.compile_with_static_parts(vec![near]).unwrap();
    assert_eq!(creation.bearings[0].parts, [Some(near), Some(slider)]);
    let body_of = |part: PartId| {
        creation
            .part_to_compound
            .iter()
            .find(|(candidate, _)| *candidate == part)
            .unwrap()
            .1 as usize
    };
    let (base, mover) = (body_of(near), body_of(slider));
    assert_eq!(body_of(stop), base);
    let geometry = MachineCollisionGeometry::new(&creation, 7).unwrap();
    assert!(
        geometry.suppressed.is_empty(),
        "the stop was built apart, so the bodies are not exempt as a whole"
    );
    let rest = MachineState::at_rest(&creation).poses;
    let pairs_at = |slid: f64| {
        let mut poses = rest.clone();
        poses[mover].position.x += slid;
        let bounds = geometry
            .colliders
            .iter()
            .map(|collider| {
                let pose = poses[collider.body];
                collider
                    .local
                    .transformed_bounds(pose.position, pose.rotation)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        geometry
            .candidate_pairs(&bounds)
            .iter()
            .filter(|&&[a, b]| {
                let bodies = [geometry.colliders[a].body, geometry.colliders[b].body];
                bodies == [base, mover] || bodies == [mover, base]
            })
            .count()
    };
    assert_eq!(pairs_at(0.0), 0, "built touching the block it is hinged to");
    assert_eq!(
        pairs_at(1.0),
        0,
        "slid onto the far block, whose top is the face shared as built"
    );
    assert!(
        pairs_at(1.6) > 0,
        "slid into the stop, which was built apart"
    );
}

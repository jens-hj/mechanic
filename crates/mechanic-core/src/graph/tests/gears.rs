use super::{cube_at, spawn};
use crate::testing::{Y_TO_Z, gear_pair, rack, ring_gear, spawned, spur_gear, worm_drive};
use crate::{
    BuildCommand, BuildOutcome, ConstructionGraph, GearLinkError, GearLinkId, GearLinkKind,
    GearLinkSpec, GraphError, GridRotation, RigidLinkSpec,
};
use bevy_math::IVec3;

fn meshed(
    graph: &mut ConstructionGraph,
    first: crate::PartId,
    second: crate::PartId,
) -> GearLinkId {
    match graph
        .apply(BuildCommand::AddGearLink(GearLinkSpec { first, second }))
        .unwrap()
    {
        BuildOutcome::GearLinked(id) => id,
        other => panic!("expected a mesh, got {other:?}"),
    }
}

#[test]
fn toothed_cylinders_mesh_when_tangent_and_lose_the_mesh_with_their_teeth() {
    let mut graph = ConstructionGraph::new();
    let (pinion, wheel) = gear_pair(&mut graph);
    let link = meshed(&mut graph, pinion, wheel);
    let mesh = graph.gear_mesh(*graph.gear_link(link).unwrap()).unwrap();
    assert_eq!(mesh.kind, GearLinkKind::Gears);
    assert!((mesh.ratio().unwrap() - 1.5).abs() < 1.0e-6);
    assert_eq!(
        graph.apply(BuildCommand::AddGearLink(GearLinkSpec {
            first: wheel,
            second: pinion,
        })),
        Err(GraphError::AlreadyMeshed)
    );
    assert_eq!(graph.part_gear_links(wheel).count(), 1);

    let plain = graph
        .part(pinion)
        .unwrap()
        .as_cylinder()
        .unwrap()
        .without_gear();
    assert_eq!(
        graph.apply(BuildCommand::SetGear {
            part: pinion,
            spec: plain,
        }),
        Ok(BuildOutcome::GearUpdated)
    );
    assert!(graph.gear_link(link).is_none());
    assert_eq!(graph.part_gear_links(wheel).count(), 0);
    assert_eq!(
        graph.apply(BuildCommand::RemoveGearLink(link)),
        Err(GraphError::MissingGearLink(link))
    );
}

#[test]
fn meshes_are_refused_inside_one_body_across_modules_and_off_tangency() {
    let mut graph = ConstructionGraph::new();
    let (pinion, wheel) = gear_pair(&mut graph);
    let far = spawned(
        graph
            .apply(BuildCommand::SpawnCylinder(spur_gear(
                4,
                36,
                IVec3::new(0, 0, 140),
                GridRotation::default(),
            )))
            .unwrap(),
    );
    assert!(matches!(
        graph.apply(BuildCommand::AddGearLink(GearLinkSpec {
            first: pinion,
            second: far,
        })),
        Err(GraphError::GearLink(GearLinkError::NotTangent { .. }))
    ));
    let coarse = spur_gear(6, 16, IVec3::new(120, 0, 0), GridRotation::default());
    graph
        .apply(BuildCommand::SetGear {
            part: wheel,
            spec: coarse,
        })
        .unwrap();
    assert_eq!(
        graph.apply(BuildCommand::AddGearLink(GearLinkSpec {
            first: pinion,
            second: wheel,
        })),
        Err(GraphError::GearLink(GearLinkError::ModuleMismatch))
    );
    assert_eq!(
        graph.apply(BuildCommand::SetGear {
            part: far,
            spec: spur_gear(4, 36, IVec3::new(120, 0, 0), GridRotation::default()),
        }),
        Err(GraphError::GearTargetChanged(far))
    );
    assert_eq!(
        graph.apply(BuildCommand::AddGearLink(GearLinkSpec {
            first: pinion,
            second: pinion,
        })),
        Err(GraphError::SameGearLinkPart)
    );
    let block = spawn(&mut graph, cube_at(0));
    assert_eq!(
        graph.apply(BuildCommand::AddGearLink(GearLinkSpec {
            first: pinion,
            second: block,
        })),
        Err(GraphError::GearLink(GearLinkError::NeedsTeeth))
    );
}

#[test]
fn a_mesh_never_forms_inside_one_rigid_body() {
    let mut graph = ConstructionGraph::new();
    let (pinion, wheel) = gear_pair(&mut graph);
    graph
        .apply(BuildCommand::RigidLink(RigidLinkSpec {
            first: pinion,
            second: wheel,
        }))
        .unwrap();
    assert_eq!(
        graph.apply(BuildCommand::AddGearLink(GearLinkSpec {
            first: pinion,
            second: wheel,
        })),
        Err(GraphError::GearLinkWithinBody)
    );
}

#[test]
fn removing_a_meshed_part_cascades_its_mesh_and_the_component_crosses_it() {
    let mut graph = ConstructionGraph::new();
    let (pinion, wheel) = gear_pair(&mut graph);
    let link = meshed(&mut graph, pinion, wheel);
    let component = graph.structural_component(pinion, []).unwrap();
    assert!(component.contains(wheel));
    graph.apply(BuildCommand::Remove(wheel)).unwrap();
    assert!(graph.gear_link(link).is_none());
    assert_eq!(graph.gear_links().count(), 0);
}

#[test]
fn a_wheel_meshes_a_worm_and_any_part_rides_it_as_a_nut() {
    let mut graph = ConstructionGraph::new();
    let (wheel, worm, nut) = worm_drive(&mut graph);
    let drive = meshed(&mut graph, wheel, worm);
    let mesh = graph.gear_mesh(*graph.gear_link(drive).unwrap()).unwrap();
    assert_eq!(mesh.kind, GearLinkKind::Worm);
    // One worm turn advances the wheel's 18 cm pitch circle one 5 cm lead.
    let ratio = 0.05 / (core::f32::consts::TAU * 0.18);
    assert!((mesh.ratio().unwrap() - ratio).abs() < 1.0e-6);
    let travel = meshed(&mut graph, worm, nut);
    let mesh = graph.gear_mesh(*graph.gear_link(travel).unwrap()).unwrap();
    assert_eq!(mesh.kind, GearLinkKind::Screw);
    assert!(mesh.swapped);
    assert_eq!(
        graph.apply(BuildCommand::AddGearLink(GearLinkSpec {
            first: nut,
            second: wheel,
        })),
        Err(GraphError::GearLink(GearLinkError::NeedsTeeth))
    );

    let plain = graph
        .part(worm)
        .unwrap()
        .as_cylinder()
        .unwrap()
        .without_spiral();
    graph
        .apply(BuildCommand::SetSpiral {
            part: worm,
            spec: plain,
        })
        .unwrap();
    assert_eq!(graph.gear_links().count(), 0);
}

#[test]
fn a_pinion_meshes_a_rack_and_sits_inside_a_ring() {
    let mut graph = ConstructionGraph::new();
    let bar = spawned(
        graph
            .apply(BuildCommand::Spawn(rack(4, [8, 1, 1], IVec3::ZERO)))
            .unwrap(),
    );
    // Pitch plane 1 cm under the 12.5 cm high face, pinion pitch radius 12 cm.
    let pinion = spawned(
        graph
            .apply(BuildCommand::SpawnCylinder(spur_gear(
                4,
                24,
                IVec3::new(0, 94, 0),
                Y_TO_Z,
            )))
            .unwrap(),
    );
    let link = meshed(&mut graph, bar, pinion);
    let mesh = graph.gear_mesh(*graph.gear_link(link).unwrap()).unwrap();
    assert_eq!(mesh.kind, GearLinkKind::Rack);
    assert!(mesh.swapped);
    let plain = graph.part(bar).unwrap().as_cuboid().unwrap().without_rack();
    graph
        .apply(BuildCommand::SetRack {
            part: bar,
            spec: plain,
        })
        .unwrap();
    assert!(graph.gear_link(link).is_none());

    let ring = spawned(
        graph
            .apply(BuildCommand::SpawnCylinder(ring_gear(
                4,
                72,
                IVec3::new(0, 0, 200),
            )))
            .unwrap(),
    );
    let planet = spawned(
        graph
            .apply(BuildCommand::SpawnCylinder(spur_gear(
                4,
                24,
                IVec3::new(96, 0, 200),
                GridRotation::default(),
            )))
            .unwrap(),
    );
    let link = meshed(&mut graph, ring, planet);
    let mesh = graph.gear_mesh(*graph.gear_link(link).unwrap()).unwrap();
    assert_eq!(mesh.kind, GearLinkKind::Gears);
    assert!((mesh.ratio().unwrap() - 1.0 / 3.0).abs() < 1.0e-6);
}

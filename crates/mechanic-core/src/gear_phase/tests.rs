use super::*;
use crate::testing::{Y_TO_Z, gear_pair, rack, spawned, spur_gear};
use crate::{BuildCommand, GearLinkSpec, GridRotation};
use bevy_math::IVec3;
use core::f32::consts::{PI, TAU};

fn mesh(graph: &mut ConstructionGraph, first: PartId, second: PartId) {
    graph
        .apply(BuildCommand::AddGearLink(GearLinkSpec { first, second }))
        .unwrap();
}

/// Where the tooth centre of `part` nearest its pitch point with `partner`
/// falls along `reference`, as a fraction of the tooth pitch in [0, 1),
/// placing the teeth as the app draws them: a gear's at (k + 0.375) pitch
/// past its phase from its X axis towards Z, a rack's on its tooth line.
fn tooth_offset(
    graph: &ConstructionGraph,
    part: PartId,
    partner: PartId,
    phases: &BTreeMap<PartId, f32>,
    reference: Vec3,
) -> f32 {
    let link = GearLinkSpec {
        first: part,
        second: partner,
    };
    let (mine, theirs) = sides(link, graph.gear_mesh(link).unwrap(), part);
    let (offset, pitch) = match graph.part(part).unwrap() {
        PartSpec::Cylinder(cylinder) => {
            let gear = cylinder.gear().unwrap();
            let pitch = TAU / f32::from(gear.teeth());
            let angle = pitch_angle(graph, part, mine, theirs);
            let phase = phases.get(&part).copied().unwrap_or(0.0);
            let tooth = ((angle - phase) / pitch - GEAR_TOOTH_CENTER_FRACTION).round();
            let centre = phase + (tooth + GEAR_TOOTH_CENTER_FRACTION) * pitch;
            let arc = (centre - angle) * gear.pitch_radius();
            (
                tangent(graph, part, mine, theirs) * arc,
                pitch * gear.pitch_radius(),
            )
        }
        PartSpec::Cuboid(cuboid) => {
            let spec = cuboid.rack().unwrap();
            let local = world_rotation(graph, part).inverse()
                * (theirs.center - graph.part_frame_point(part, cuboid.pose.translation()));
            let position = local[spec.along().index()];
            let first_tooth = spec.tooth_line(cuboid.pose);
            let tooth = ((position - first_tooth) / spec.pitch()).round();
            let centre = first_tooth + tooth * spec.pitch();
            (
                tangent(graph, part, mine, theirs) * (centre - position),
                spec.pitch(),
            )
        }
        _ => unreachable!("toothed parts are cylinders and cuboids"),
    };
    (offset.dot(reference) / pitch).rem_euclid(1.0)
}

/// Asserts that the teeth of two meshed parts alternate at their pitch point.
fn assert_interleaved(graph: &ConstructionGraph, a: PartId, b: PartId) {
    let phases = gear_phases(graph);
    let link = GearLinkSpec {
        first: a,
        second: b,
    };
    let (mine, theirs) = sides(link, graph.gear_mesh(link).unwrap(), a);
    let reference = tangent(graph, a, mine, theirs);
    let first = tooth_offset(graph, a, b, &phases, reference);
    let second = tooth_offset(graph, b, a, &phases, reference);
    let apart = (first - second).rem_euclid(1.0);
    assert!(
        (apart - 0.5).abs() < 0.02,
        "{a:?} teeth at {first} and {b:?} teeth at {second} of a pitch along the mesh"
    );
}

#[test]
fn meshed_gears_interleave_their_teeth_at_the_pitch_point() {
    let mut graph = ConstructionGraph::new();
    let (pinion, wheel) = gear_pair(&mut graph);
    assert_eq!(gear_phases(&graph).len(), 2, "unmeshed gears keep phase 0");
    mesh(&mut graph, pinion, wheel);
    let phases = gear_phases(&graph);
    assert!(phases[&pinion].abs() < 1.0e-6, "the lowest gear stays");
    // The pinion's pitch point, on its X axis, falls 0.625 of a tooth past a
    // tooth centre. The wheel turns the other way, so its pitch point on its
    // -X axis must fall at 0.875, which its 36 teeth reach by turning three
    // quarters of a tooth.
    assert!(
        (phases[&wheel] - PI / 24.0).abs() < 1.0e-5,
        "{}",
        phases[&wheel]
    );
    assert_interleaved(&graph, pinion, wheel);
}

#[test]
fn a_wheel_whose_axis_points_the_other_way_interleaves_too() {
    let mut graph = ConstructionGraph::new();
    let pinion = spawned(
        graph
            .apply(BuildCommand::SpawnCylinder(spur_gear(
                4,
                24,
                IVec3::ZERO,
                GridRotation::default(),
            )))
            .unwrap(),
    );
    let wheel = spawned(
        graph
            .apply(BuildCommand::SpawnCylinder(spur_gear(
                4,
                36,
                IVec3::new(120, 0, 0),
                GridRotation::new(2, 0, 0),
            )))
            .unwrap(),
    );
    mesh(&mut graph, pinion, wheel);
    assert_interleaved(&graph, pinion, wheel);
}

#[test]
fn a_train_phases_each_gear_to_the_one_before_it() {
    let mut graph = ConstructionGraph::new();
    let (pinion, wheel) = gear_pair(&mut graph);
    let last = spawned(
        graph
            .apply(BuildCommand::SpawnCylinder(spur_gear(
                4,
                24,
                IVec3::new(240, 0, 0),
                GridRotation::new(0, 1, 0),
            )))
            .unwrap(),
    );
    mesh(&mut graph, pinion, wheel);
    mesh(&mut graph, wheel, last);
    assert_interleaved(&graph, pinion, wheel);
    assert_interleaved(&graph, wheel, last);
}

#[test]
fn a_rack_keeps_its_teeth_and_its_pinion_turns_to_them() {
    for pinion_rotation in [Y_TO_Z, GridRotation::new(3, 0, 0)] {
        let mut graph = ConstructionGraph::new();
        // A 2 m rack, its pitch line 11.5 cm up; a 24-tooth pinion on a Z axis
        // 2.5 cm along it, its pitch circle on that line, its axis pointing
        // either way.
        let bar = spawned(
            graph
                .apply(BuildCommand::Spawn(rack(4, [8, 1, 1], IVec3::ZERO)))
                .unwrap(),
        );
        let pinion = spawned(
            graph
                .apply(BuildCommand::SpawnCylinder(spur_gear(
                    4,
                    24,
                    IVec3::new(10, 94, 0),
                    pinion_rotation,
                )))
                .unwrap(),
        );
        mesh(&mut graph, pinion, bar);
        let phases = gear_phases(&graph);
        assert!(!phases.contains_key(&bar), "only gears carry a phase");
        assert_interleaved(&graph, bar, pinion);
    }
}

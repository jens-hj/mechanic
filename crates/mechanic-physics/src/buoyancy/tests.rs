//! Probes carry their colliders' volume and mass.

use bevy_math::IVec3;
use mechanic_core::{
    BuildCommand, BuildPose, ConstructionGraph, CuboidSpec, CylinderDimensions, CylinderSpec,
    GridRotation,
};

use super::BuoyancyProbes;

fn totals(probes: &BuoyancyProbes) -> (f64, f64) {
    probes
        .probes
        .iter()
        .fold((0.0, 0.0), |(volume, mass), probe| {
            (volume + probe.volume, mass + probe.mass)
        })
}

#[test]
fn a_cube_is_probed_as_its_whole_volume_and_mass() {
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::Spawn(
            CuboidSpec::new([4, 2, 4], BuildPose::default()).unwrap(),
        ))
        .unwrap();
    let creation = graph.compile().unwrap();
    let probes = BuoyancyProbes::new(&creation);
    let (volume, mass) = totals(&probes);
    assert!((volume - 0.5).abs() < 1.0e-9, "volume {volume}");
    let expected = f64::from(creation.compounds[0].mass_properties.mass);
    assert!((mass - expected).abs() < 1.0e-6 * expected);
}

#[test]
fn a_cylinder_is_probed_once_as_its_round_volume() {
    let mut graph = ConstructionGraph::new();
    graph
        .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
            CylinderDimensions::new(1.0, 0.0, 0.5).unwrap(),
            BuildPose::from_position_ticks(IVec3::ZERO, GridRotation::default()),
        )))
        .unwrap();
    let creation = graph.compile().unwrap();
    let probes = BuoyancyProbes::new(&creation);
    let (volume, _) = totals(&probes);
    // The sixteen overlapping tangent boxes would count far more.
    let round = std::f64::consts::PI * 0.25 * 0.5;
    assert!(
        (volume - round).abs() < 0.05 * round,
        "volume {volume:.4} m³ against {round:.4} m³"
    );
}

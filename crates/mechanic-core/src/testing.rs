//! Fixtures shared by tests across modules.

use crate::{
    BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, CuboidSpec, CylinderDimensions,
    CylinderSpec, FaceKind, GearKind, GearSpec, GridRotation, PartId, RackSpec,
};
use bevy_math::IVec3;

/// A cardinal rotation taking the local Y axis to world Z.
pub(crate) const Y_TO_Z: GridRotation = GridRotation::new(1, 0, 0);

/// A solid cylinder one block long carrying external teeth, sized so the
/// tooth tips are its outer wall.
pub(crate) fn spur_gear(
    module_ticks: u8,
    teeth: u16,
    ticks: IVec3,
    rotation: GridRotation,
) -> CylinderSpec {
    let gear = GearSpec::new(module_ticks, teeth, GearKind::Spur).unwrap();
    CylinderSpec::new(
        CylinderDimensions::new(gear.tip_diameter(), 0.0, 0.25).unwrap(),
        BuildPose::from_position_ticks(ticks, rotation),
    )
    .with_gear(gear)
    .unwrap()
}

/// A hollow cylinder one block long carrying internal teeth on its bore, with
/// a 5 cm rim outside them.
pub(crate) fn ring_gear(module_ticks: u8, teeth: u16, ticks: IVec3) -> CylinderSpec {
    let gear = GearSpec::new(module_ticks, teeth, GearKind::Internal).unwrap();
    CylinderSpec::new(
        CylinderDimensions::new(gear.tip_diameter() + 0.1, gear.tip_diameter(), 0.25).unwrap(),
        BuildPose::from_position_ticks(ticks, GridRotation::default()),
    )
    .with_gear(gear)
    .unwrap()
}

/// A cuboid with rack teeth cut into its positive-Y face, running along X.
pub(crate) fn rack(module_ticks: u8, dimensions: [u8; 3], ticks: IVec3) -> CuboidSpec {
    CuboidSpec::new(
        dimensions,
        BuildPose::from_position_ticks(ticks, GridRotation::default()),
    )
    .unwrap()
    .with_rack(RackSpec::new(module_ticks, FaceKind::PositiveY, crate::Axis::X).unwrap())
    .unwrap()
}

/// A 10 cm solid cylinder half a metre long carrying a single-start square
/// thread with a 5 cm pitch cut 1.25 cm deep: its pitch radius is 4.375 cm.
pub(crate) fn worm(ticks: IVec3, rotation: GridRotation) -> CylinderSpec {
    CylinderSpec::new(
        CylinderDimensions::new(0.1, 0.0, 0.5).unwrap(),
        BuildPose::from_position_ticks(ticks, rotation),
    )
    .with_spiral(
        crate::SpiralSpec::new(
            20,
            1,
            crate::SpiralHand::Right,
            crate::SpiralProfile::square(10, 5).unwrap(),
            crate::SpiralProfile::PLAIN,
            None,
        )
        .unwrap(),
    )
    .unwrap()
}

/// The part a spawn command created.
///
/// # Panics
///
/// Panics when the outcome is not a spawn.
pub(crate) fn spawned(outcome: BuildOutcome) -> PartId {
    let BuildOutcome::Spawned(id) = outcome else {
        panic!("expected a spawned part, got {outcome:?}")
    };
    id
}

/// A 24-tooth pinion at the origin and a 36-tooth wheel 30 cm along X, both
/// module 1 cm on world-Y axes: their pitch circles touch.
pub(crate) fn gear_pair(graph: &mut ConstructionGraph) -> (PartId, PartId) {
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
                GridRotation::default(),
            )))
            .unwrap(),
    );
    (pinion, wheel)
}

/// A 24-tooth module 1.5 cm wheel at the origin on world Y, the [`worm`]
/// 22.5 cm along X on world Z, and a one-block nut riding the worm 15 cm
/// further along it.
pub(crate) fn worm_drive(graph: &mut ConstructionGraph) -> (PartId, PartId, PartId) {
    let wheel = spawned(
        graph
            .apply(BuildCommand::SpawnCylinder(spur_gear(
                6,
                24,
                IVec3::ZERO,
                GridRotation::default(),
            )))
            .unwrap(),
    );
    let worm = spawned(
        graph
            .apply(BuildCommand::SpawnCylinder(worm(
                IVec3::new(90, 0, 0),
                Y_TO_Z,
            )))
            .unwrap(),
    );
    let nut = spawned(
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1; 3],
                    BuildPose::from_position_ticks(IVec3::new(90, 0, 60), GridRotation::default()),
                )
                .unwrap(),
            ))
            .unwrap(),
    );
    (wheel, worm, nut)
}

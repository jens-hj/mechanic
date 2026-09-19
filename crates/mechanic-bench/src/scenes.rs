//! The compiled scenes each GPU scenario runs.

use bevy_math::{IVec3, Vec3};
use mechanic_core::{
    BearingSpec, BuildCommand, BuildOutcome, BuildPose, CompiledCreation, ConstructionGraph,
    ConstructionMaterial, CoordinateDrive, CuboidSpec, CylinderDimensions, CylinderSpec, DriveMode,
    FaceKind, FaceRef, GridRotation, JointKind, PartId, PipeBendDimensions, PipeBendSpec,
    RigidLinkSpec, ShapeRegion, SpringSpec, SuspensionSpec, WeldSpec,
};
use mechanic_gpu::{DRIVE_MODE_ANGLE, DRIVE_MODE_PASSIVE, DRIVE_MODE_SPEED, GpuMechanismDrive};

pub(crate) const SCALE_BODY_COUNT: usize = 100_000;

pub(crate) fn spawned_part(outcome: BuildOutcome) -> Result<PartId, String> {
    match outcome {
        BuildOutcome::Spawned(part) => Ok(part),
        other => Err(format!("expected spawned part, got {other:?}")),
    }
}

pub(crate) fn build_suspension_one() -> Result<CompiledCreation, String> {
    let mut graph = ConstructionGraph::new();
    let spring = SpringSpec::default();
    let suspension =
        SuspensionSpec::new(Some(spring), None, None).map_err(|error| error.to_string())?;
    #[expect(clippy::cast_possible_truncation)]
    let spacing_ticks = ((suspension.initial_length() + 0.625) / 0.0025).round() as i32;
    let base = spawned_part(
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [4, 4, 4],
                    BuildPose::from_position_ticks(IVec3::new(0, 200, 0), GridRotation::default()),
                )
                .map_err(|error| error.to_string())?,
            ))
            .map_err(|error| error.to_string())?,
    )?;
    let load = spawned_part(
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1, 1, 1],
                    BuildPose::from_position_ticks(
                        IVec3::new(0, 200 + spacing_ticks, 0),
                        GridRotation::default(),
                    ),
                )
                .map_err(|error| error.to_string())?,
            ))
            .map_err(|error| error.to_string())?,
    )?;
    graph
        .apply(BuildCommand::Weld(WeldSpec {
            first: FaceRef::part(base, FaceKind::NegativeY),
            second: FaceRef::ground(),
        }))
        .map_err(|error| error.to_string())?;
    graph
        .apply(BuildCommand::AddBearing(
            BearingSpec::new(
                FaceRef::part(base, FaceKind::PositiveY),
                FaceRef::part(load, FaceKind::NegativeY),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::Y,
            )
            .with_kind(JointKind::Suspension(suspension)),
        ))
        .map_err(|error| error.to_string())?;
    graph.compile().map_err(|error| error.to_string())
}

/// Repository-owned reproduction of the TEST2 vehicle's expensive topology.
#[expect(clippy::too_many_lines)]
pub(crate) fn build_test2_car() -> Result<CompiledCreation, String> {
    let mut graph = ConstructionGraph::new();
    let chassis = spawned_part(
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [12, 1, 8],
                    BuildPose::from_position_ticks(IVec3::new(0, 500, 0), GridRotation::default()),
                )
                .map_err(|error| error.to_string())?
                .with_material(ConstructionMaterial::Stone),
            ))
            .map_err(|error| error.to_string())?,
    )?;

    // Seventy-seven source blocks are represented by one region collider. The
    // remaining eight rigidly linked blocks preserve the captured 94-part / ten
    // chassis-collider shape without changing the four expensive pipe bends.
    let mut region_parts = Vec::new();
    for z in -5..=5 {
        for x in -3..=3 {
            let part = spawned_part(
                graph
                    .apply(BuildCommand::Spawn(
                        CuboidSpec::new(
                            [1, 1, 1],
                            BuildPose::new(IVec3::new(x, 6, z), GridRotation::default()),
                        )
                        .map_err(|error| error.to_string())?
                        .with_material(ConstructionMaterial::Stone),
                    ))
                    .map_err(|error| error.to_string())?,
            )?;
            graph
                .apply(BuildCommand::RigidLink(RigidLinkSpec {
                    first: chassis,
                    second: part,
                }))
                .map_err(|error| error.to_string())?;
            region_parts.push(part);
        }
    }
    debug_assert_eq!(region_parts.len(), 77);
    graph
        .apply(BuildCommand::AddRegion(
            ShapeRegion::new(
                IVec3::new(-7, 11, -11),
                IVec3::new(7, 1, 11),
                ConstructionMaterial::Stone,
            )
            .map_err(|error| error.to_string())?,
        ))
        .map_err(|error| error.to_string())?;
    let detail_positions = [
        IVec3::new(-5, 8, -4),
        IVec3::new(-3, 8, -4),
        IVec3::new(-1, 8, -4),
        IVec3::new(1, 8, -4),
        IVec3::new(3, 8, -4),
        IVec3::new(5, 8, -4),
        IVec3::new(-5, 8, 4),
        IVec3::new(5, 8, 4),
    ];
    for position in detail_positions {
        let part = spawned_part(
            graph
                .apply(BuildCommand::Spawn(
                    CuboidSpec::new([1, 1, 1], BuildPose::new(position, GridRotation::default()))
                        .map_err(|error| error.to_string())?
                        .with_material(ConstructionMaterial::Stone),
                ))
                .map_err(|error| error.to_string())?,
        )?;
        graph
            .apply(BuildCommand::RigidLink(RigidLinkSpec {
                first: chassis,
                second: part,
            }))
            .map_err(|error| error.to_string())?;
    }

    let bend_dimensions =
        PipeBendDimensions::new(0.2, 0.0, 2).map_err(|error| error.to_string())?;
    // Two-block bends need the chassis an eighth of a metre higher than the
    // captured one-block-radius bends did to keep the wheel axles in place.
    let corners = [
        (600, 1.5, 1, 0.75, 1.125),
        (-600, -1.5, 1, 0.75, 1.125),
        (-600, -1.5, -1, -0.75, -1.125),
        (600, 1.5, -1, -0.75, -1.125),
    ];
    let mut bends = [None; 4];
    let mut wheels = [None; 4];
    for (wheel, corner) in [
        (true, 0),
        (false, 1),
        (true, 1),
        (false, 2),
        (true, 2),
        (false, 3),
        (false, 0),
        (true, 3),
    ] {
        let (x_ticks, _, z_sign, _, _) = corners[corner];
        if wheel {
            wheels[corner] = Some(spawned_part(
                graph
                    .apply(BuildCommand::SpawnCylinder(
                        CylinderSpec::new(
                            CylinderDimensions::new(0.95, 0.0, 0.25)
                                .map_err(|error| error.to_string())?,
                            BuildPose::from_position_ticks(
                                IVec3::new(x_ticks, 290, z_sign * 500),
                                if z_sign > 0 {
                                    GridRotation::new(1, 0, 0)
                                } else {
                                    GridRotation::new(1, 2, 2)
                                },
                            ),
                        )
                        .with_material(ConstructionMaterial::Rubber),
                    ))
                    .map_err(|error| error.to_string())?,
            )?);
        } else {
            bends[corner] = Some(spawned_part(
                graph
                    .apply(BuildCommand::SpawnPipeBend(PipeBendSpec::new(
                        bend_dimensions,
                        BuildPose::from_position_ticks(
                            IVec3::new(x_ticks, 300, z_sign * 300),
                            if z_sign > 0 {
                                GridRotation::new(0, 3, 3)
                            } else {
                                GridRotation::new(0, 1, 3)
                            },
                        ),
                    )))
                    .map_err(|error| error.to_string())?,
            )?);
        }
    }
    let bends = bends.map(Option::unwrap);
    let wheels = wheels.map(Option::unwrap);
    for corner in [0, 3, 2, 1] {
        let (_, x, _, bend_z, _) = corners[corner];
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(chassis, FaceKind::NegativeY),
                FaceRef::part(bends[corner], FaceKind::NegativeX),
                Vec3::new(x, 1.125, bend_z),
                Vec3::NEG_Y,
            )))
            .map_err(|error| error.to_string())?;
    }
    for corner in [1, 0, 3, 2] {
        let (_, x, z_sign, _, wheel_z) = corners[corner];
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(bends[corner], FaceKind::PositiveY),
                FaceRef::part(wheels[corner], FaceKind::NegativeY),
                Vec3::new(x, 0.75, wheel_z),
                if z_sign > 0 { Vec3::Z } else { Vec3::NEG_Z },
            )))
            .map_err(|error| error.to_string())?;
    }
    let mut creation = graph.compile().map_err(|error| error.to_string())?;
    if graph.part_count() != 94
        || creation.compounds.len() != 9
        || creation.colliders.len() != 842
        || creation.bearings.len() != 8
    {
        return Err(format!(
            "test2_car fixture generated {} parts, {} bodies, {} colliders, and {} bearings",
            graph.part_count(),
            creation.compounds.len(),
            creation.colliders.len(),
            creation.bearings.len(),
        ));
    }
    for coordinate in 0..4 {
        creation.coordinate_drives[coordinate] = CoordinateDrive {
            mode: DriveMode::Angle,
            max_speed: std::f32::consts::PI,
            max_acceleration: 20.0,
            source_a_max_acceleration: 20.0,
            source_a_no_load_speed: std::f32::consts::PI,
            min_angle: -std::f32::consts::FRAC_PI_4,
            max_angle: std::f32::consts::FRAC_PI_4,
            ..CoordinateDrive::default()
        };
    }
    for coordinate in 4..8 {
        creation.coordinate_drives[coordinate] = CoordinateDrive {
            mode: DriveMode::Speed,
            max_speed: std::f32::consts::TAU * 6.0,
            max_acceleration: 40.0,
            source_a_max_acceleration: 40.0,
            source_a_no_load_speed: std::f32::consts::TAU * 6.0,
            min_angle: f32::NEG_INFINITY,
            max_angle: f32::INFINITY,
            ..CoordinateDrive::default()
        };
    }
    Ok(creation)
}

pub(crate) fn test2_phase_drives(
    base: &[GpuMechanismDrive],
    creation: &CompiledCreation,
    phase: usize,
) -> Vec<GpuMechanismDrive> {
    let mut drives = base.to_vec();
    for drive in &mut drives[..4] {
        drive.mode = DRIVE_MODE_ANGLE;
        drive.target_angle = if phase == 3 {
            std::f32::consts::FRAC_PI_6
        } else {
            0.0
        };
    }
    for (coordinate, drive) in drives.iter_mut().enumerate().skip(4) {
        if matches!(phase, 0 | 1 | 4) {
            drive.mode = DRIVE_MODE_PASSIVE;
            drive.target_speed = 0.0;
            continue;
        }
        drive.mode = DRIVE_MODE_SPEED;
        let source_bearing = creation.loop_topology.tree_bearings[coordinate];
        let bearing = creation
            .bearings
            .iter()
            .find(|bearing| bearing.source_bearing == source_bearing)
            .expect("tree bearing remains compiled");
        let axis =
            creation.compounds[bearing.compound_a as usize].root_rotation * bearing.local_axis_a;
        drive.target_speed =
            axis.cross(Vec3::Y).dot(Vec3::X).signum() * std::f32::consts::TAU * 6.0;
    }
    drives
}

pub(crate) fn build_four_bar(invalid: bool) -> Result<CompiledCreation, String> {
    let mut graph = ConstructionGraph::new();
    let outcomes = graph
        .apply_batch([
            BuildCommand::Spawn(unit_cube(IVec3::ZERO)),
            BuildCommand::Spawn(unit_cube(IVec3::new(4, 0, 0))),
            BuildCommand::Spawn(unit_cube(IVec3::new(4, 4, 0))),
            BuildCommand::Spawn(unit_cube(IVec3::new(0, 4, 0))),
        ])
        .map_err(|error| format!("four-bar part generation failed: {error}"))?;
    let parts = outcomes
        .into_iter()
        .map(|outcome| match outcome {
            BuildOutcome::Spawned(part) => part,
            _ => unreachable!("batch contains only spawn commands"),
        })
        .collect::<Vec<_>>();
    let edges = [
        (
            parts[0],
            FaceKind::PositiveX,
            parts[1],
            FaceKind::NegativeX,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::X,
        ),
        (
            parts[1],
            FaceKind::PositiveY,
            parts[2],
            FaceKind::NegativeY,
            Vec3::new(1.0, 0.5, 0.0),
            Vec3::Y,
        ),
        (
            parts[2],
            FaceKind::NegativeX,
            parts[3],
            FaceKind::PositiveX,
            Vec3::new(0.5, 1.0, 0.0),
            Vec3::NEG_X,
        ),
        (
            parts[3],
            FaceKind::NegativeY,
            parts[0],
            FaceKind::PositiveY,
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::NEG_Y,
        ),
    ];
    graph
        .apply_batch(
            edges.map(|(source, source_face, target, target_face, anchor, axis)| {
                bearing_command(source, source_face, target, target_face, anchor, axis)
            }),
        )
        .map_err(|error| format!("four-bar bearing generation failed: {error}"))?;
    let mut creation = graph.compile().map_err(|error| error.to_string())?;
    if invalid {
        let closure = creation
            .bearings
            .iter_mut()
            .find(|bearing| bearing.coordinate_index.is_none())
            .ok_or_else(|| "four-bar did not compile a closure edge".to_owned())?;
        closure.local_anchor_b += Vec3::new(0.2, 0.13, 0.07);
    }
    Ok(creation)
}

pub(crate) fn build_dense(count: usize) -> Result<CompiledCreation, String> {
    let mut graph = ConstructionGraph::new();
    let commands = (0..count).map(|index| {
        let x = i32::try_from(index % 50).expect("x fits i32");
        let y = i32::try_from((index / 50) % 40).expect("y fits i32");
        let z = i32::try_from(index / 2_000).expect("z fits i32");
        BuildCommand::Spawn(unit_cube(IVec3::new(x * 4, y * 4 + 2, z * 4)))
    });
    graph
        .apply_batch(commands)
        .map_err(|error| format!("dense graph generation failed: {error}"))?;
    graph.compile().map_err(|error| error.to_string())
}

pub(crate) fn build_loops_100k() -> Result<CompiledCreation, String> {
    const WIDTH: usize = 100;
    const HEIGHT: usize = 100;
    const DEPTH: usize = 10;
    let mut graph = ConstructionGraph::new();
    let outcomes = graph
        .apply_batch((0..DEPTH).flat_map(|z| {
            (0..HEIGHT).flat_map(move |y| {
                (0..WIDTH).map(move |x| {
                    BuildCommand::Spawn(unit_cube(IVec3::new(
                        i32::try_from(x * 4).expect("x fits i32"),
                        i32::try_from(y * 4).expect("y fits i32"),
                        i32::try_from(z * 4).expect("z fits i32"),
                    )))
                })
            })
        }))
        .map_err(|error| format!("lattice part generation failed: {error}"))?;
    let parts = outcomes
        .into_iter()
        .map(|outcome| match outcome {
            BuildOutcome::Spawned(part) => part,
            _ => unreachable!("batch contains only spawn commands"),
        })
        .collect::<Vec<_>>();
    let part =
        |x: usize, y: usize, z: usize| -> PartId { parts[z * WIDTH * HEIGHT + y * WIDTH + x] };
    let mut bearings = Vec::with_capacity(198_009);
    for z in 0..DEPTH {
        for y in 0..HEIGHT {
            for x in 0..WIDTH - 1 {
                bearings.push(bearing_command(
                    part(x, y, z),
                    FaceKind::PositiveX,
                    part(x + 1, y, z),
                    FaceKind::NegativeX,
                    Vec3::new(grid_f32(x) + 0.5, grid_f32(y), grid_f32(z)),
                    Vec3::X,
                ));
            }
        }
        for y in 0..HEIGHT - 1 {
            for x in 0..WIDTH {
                bearings.push(bearing_command(
                    part(x, y, z),
                    FaceKind::PositiveY,
                    part(x, y + 1, z),
                    FaceKind::NegativeY,
                    Vec3::new(grid_f32(x), grid_f32(y) + 0.5, grid_f32(z)),
                    Vec3::Y,
                ));
            }
        }
    }
    for z in 0..DEPTH - 1 {
        bearings.push(bearing_command(
            part(0, 0, z),
            FaceKind::PositiveZ,
            part(0, 0, z + 1),
            FaceKind::NegativeZ,
            Vec3::new(0.0, 0.0, grid_f32(z) + 0.5),
            Vec3::Z,
        ));
    }
    graph
        .apply_batch(bearings)
        .map_err(|error| format!("lattice bearing generation failed: {error}"))?;
    graph.compile().map_err(|error| error.to_string())
}

pub(crate) fn unit_cube(units: IVec3) -> CuboidSpec {
    CuboidSpec::new([4, 4, 4], BuildPose::new(units, GridRotation::default()))
        .expect("one-metre cube is in range")
}

pub(crate) fn bearing_command(
    source: PartId,
    source_face: FaceKind,
    target: PartId,
    target_face: FaceKind,
    anchor: Vec3,
    axis: Vec3,
) -> BuildCommand {
    BuildCommand::AddBearing(BearingSpec::new(
        FaceRef::part(source, source_face),
        FaceRef::part(target, target_face),
        anchor,
        axis,
    ))
}

pub(crate) fn grid_f32(value: usize) -> f32 {
    f32::from(u16::try_from(value).unwrap_or(u16::MAX))
}

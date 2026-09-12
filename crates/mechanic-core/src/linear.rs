//! Geometry and physical dimensions shared by linear-bearing construction and simulation.

use bevy_math::{Quat, Vec2, Vec3};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Distance travelled per output revolution, in metres.
pub const LINEAR_METERS_PER_REVOLUTION: f32 = 0.25;
/// Distance travelled per output radian, in metres.
pub const LINEAR_METERS_PER_RADIAN: f32 = LINEAR_METERS_PER_REVOLUTION / core::f32::consts::TAU;

/// Invalid rail dimensions or attachment frame.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum LinearBearingError {
    /// Length must be finite and within the supported interval.
    #[error("linear bearing length must be between 0.25 and 8 m")]
    Length,
    /// Width must be finite and within the supported interval.
    #[error("linear bearing width must be between 0.05 and 0.40 m")]
    Width,
    /// Dimensions must lie on the construction position grid.
    #[error("linear bearing dimensions must be multiples of 2.5 mm")]
    Quantization,
    /// The mounting frame must be orthonormal.
    #[error("linear bearing travel and mounting axes must be perpendicular unit vectors")]
    Frame,
}

/// Validated, exactly quantized linear-bearing dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "[f32; 2]", into = "[f32; 2]")]
pub struct LinearBearingDimensions {
    length_ticks: u16,
    width_ticks: u16,
}

impl LinearBearingDimensions {
    /// Carriage top above the mounting plane, in metres.
    pub const HEIGHT: f32 = 0.100;
    /// Length of the moving carriage, in metres.
    pub const CARRIAGE_LENGTH: f32 = 0.120;
    /// Thickness of each physical end stop, in metres.
    pub const END_STOP: f32 = 0.015;
    /// Attachment lattice pitch, in metres.
    pub const ATTACHMENT_PITCH: f32 = 0.025;

    /// Creates dimensions on the 2.5 mm grid.
    ///
    /// # Errors
    /// Returns an error for non-finite, out-of-range, or off-grid dimensions.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // Validated positive, integral ticks fit u16.
    pub fn new(length: f32, width: f32) -> Result<Self, LinearBearingError> {
        if !(0.25..=8.0).contains(&length) {
            return Err(LinearBearingError::Length);
        }
        if !(0.05..=0.40).contains(&width) {
            return Err(LinearBearingError::Width);
        }
        let ticks = [length, width].map(|value| value * 400.0);
        if ticks
            .iter()
            .any(|value| (value - value.round()).abs() > 0.00025)
        {
            return Err(LinearBearingError::Quantization);
        }
        Ok(Self {
            length_ticks: ticks[0].round() as u16,
            width_ticks: ticks[1].round() as u16,
        })
    }

    /// Rail length, including both stops, in metres.
    pub fn length(self) -> f32 {
        f32::from(self.length_ticks) / 400.0
    }

    /// Rail width in metres.
    pub fn width(self) -> f32 {
        f32::from(self.width_ticks) / 400.0
    }

    /// Quantized outer carriage half-width; the inner clearance stays unchanged.
    pub fn carriage_half_width(self) -> f32 {
        ((self.width() / 2.0 + 0.016) / crate::POSITION_TICK_METERS).round()
            * crate::POSITION_TICK_METERS
    }

    /// Total physical travel, in metres.
    pub fn travel(self) -> f32 {
        self.length() - 0.150
    }

    /// Physical displacement limits relative to the centred build pose.
    pub fn bounds(self) -> [f32; 2] {
        [-self.travel() / 2.0, self.travel() / 2.0]
    }
}

impl Default for LinearBearingDimensions {
    fn default() -> Self {
        Self {
            length_ticks: 400,
            width_ticks: 40,
        }
    }
}

impl TryFrom<[f32; 2]> for LinearBearingDimensions {
    type Error = LinearBearingError;
    fn try_from(value: [f32; 2]) -> Result<Self, Self::Error> {
        Self::new(value[0], value[1])
    }
}

impl From<LinearBearingDimensions> for [f32; 2] {
    fn from(value: LinearBearingDimensions) -> Self {
        [value.length(), value.width()]
    }
}

/// The single carriage face occupied by direct attachments.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CarriageFace {
    /// Broad upper face.
    #[default]
    Top,
    /// Side in the positive rail-local Z direction.
    PositiveSide,
    /// Side in the negative rail-local Z direction.
    NegativeSide,
}

impl CarriageFace {
    /// Outward local normal; rail travel is local X and mounting normal is Y.
    pub const fn normal(self) -> Vec3 {
        match self {
            Self::Top => Vec3::Y,
            Self::PositiveSide => Vec3::Z,
            Self::NegativeSide => Vec3::NEG_Z,
        }
    }

    /// Centred local attachment-plane origin, in metres.
    pub fn origin(self, dimensions: LinearBearingDimensions) -> Vec3 {
        match self {
            Self::Top => Vec3::new(0.0, LinearBearingDimensions::HEIGHT, 0.0),
            Self::PositiveSide => Vec3::new(0.0, 0.055, dimensions.carriage_half_width()),
            Self::NegativeSide => Vec3::new(0.0, 0.055, -dimensions.carriage_half_width()),
        }
    }

    /// Rectangular attachment extent along travel and across the selected face.
    pub fn size(self, dimensions: LinearBearingDimensions) -> Vec2 {
        Vec2::new(
            LinearBearingDimensions::CARRIAGE_LENGTH,
            match self {
                Self::Top => dimensions.carriage_half_width() * 2.0,
                Self::PositiveSide | Self::NegativeSide => 0.09,
            },
        )
    }
}

/// A rail frame and the occupied carriage surface. The anchor is the rail's underside centre.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinearBearing {
    /// Rail dimensions.
    pub dimensions: LinearBearingDimensions,
    /// World-space mounting normal, perpendicular to the bearing travel axis.
    pub mount_normal: Vec3,
    /// Selected carriage attachment surface.
    pub face: CarriageFace,
}

impl LinearBearing {
    /// Validates the rail frame and returns its local-to-world rotation.
    ///
    /// # Errors
    /// Returns an error unless both axes form an orthonormal frame.
    pub fn rotation(self, travel_axis: Vec3) -> Result<Quat, LinearBearingError> {
        if !travel_axis.is_finite()
            || !self.mount_normal.is_finite()
            || (travel_axis.length_squared() - 1.0).abs() > 1.0e-5
            || (self.mount_normal.length_squared() - 1.0).abs() > 1.0e-5
            || travel_axis.dot(self.mount_normal).abs() > 1.0e-5
        {
            return Err(LinearBearingError::Frame);
        }
        Ok(Quat::from_mat3(&bevy_math::Mat3::from_cols(
            travel_axis,
            self.mount_normal,
            travel_axis.cross(self.mount_normal),
        )))
    }
}

/// Physical one-dimensional motion permitted by a bearing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum BearingKind {
    /// Unbounded rotation about the source-face normal.
    #[default]
    Rotational,
    /// Bounded translation along the rail.
    Linear(LinearBearing),
    /// Passive axial suspension with rigid mount orientations.
    Suspension(crate::SuspensionSpec),
}

impl BearingKind {
    /// Whether the physical coordinate is translation, in metres.
    pub const fn is_translational(self) -> bool {
        !matches!(self, Self::Rotational)
    }

    /// Physical coordinate bounds independent of drive programming.
    pub fn bounds(self) -> [f32; 2] {
        match self {
            Self::Rotational => [f32::NEG_INFINITY, f32::INFINITY],
            Self::Linear(rail) => rail.dimensions.bounds(),
            Self::Suspension(spec) => spec.bounds(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, CreationDocument,
        CuboidSpec, FaceKind, FaceRef, GridRotation, PartId,
    };
    use bevy_math::IVec3;

    fn spawn_controller(graph: &mut ConstructionGraph) -> PartId {
        let BuildOutcome::Spawned(controller) = graph
            .apply(BuildCommand::SpawnController(crate::ControllerSpec::new(
                BuildPose::new(IVec3::new(0, 40, 0), GridRotation::default()),
            )))
            .unwrap()
        else {
            panic!("expected controller")
        };
        controller
    }

    fn linear_wire() -> (ConstructionGraph, crate::DriveLinkSpec) {
        let (mut graph, bearing) = rail_graph(CarriageFace::Top, GridRotation::default());
        let BuildOutcome::BearingAdded(id) =
            graph.apply(BuildCommand::AddBearing(bearing)).unwrap()
        else {
            panic!("expected bearing")
        };
        let controller = spawn_controller(&mut graph);
        (
            graph,
            crate::DriveLinkSpec::new_linear(controller, id, LinearBearingDimensions::default()),
        )
    }

    fn target_program(target: crate::DriveTarget) -> crate::DriveProgram {
        crate::DriveProgram::new(&[crate::DriveState::new(target).unwrap()], false).unwrap()
    }

    #[test]
    fn live_compiled_rows_reject_mismatched_target_units() {
        let (mut graph, bearing) = rail_graph(CarriageFace::Top, GridRotation::default());
        graph.apply(BuildCommand::AddBearing(bearing)).unwrap();
        let mut creation = graph.compile().unwrap();
        let row = creation.coordinate_drive_row(
            0,
            crate::DriveTarget::Speed(1.0),
            crate::DriveLimits::default(),
        );
        assert_eq!(row.mode, crate::DriveMode::Passive);
        assert_eq!(
            [row.min_angle, row.max_angle].map(f32::to_bits),
            bearing.kind.bounds().map(f32::to_bits)
        );
        creation.bearings[0].kind = BearingKind::Rotational;
        let row = creation.coordinate_drive_row(
            0,
            crate::DriveTarget::LinearSpeed(1.0),
            crate::DriveLimits::default(),
        );
        assert_eq!(row.mode, crate::DriveMode::Passive);
        assert_eq!(
            [row.min_angle, row.max_angle].map(f32::to_bits),
            BearingKind::Rotational.bounds().map(f32::to_bits)
        );
    }

    #[test]
    fn attached_and_unattached_linear_sockets_survive_creation_transforms() {
        for attached in [false, true] {
            for face in [
                CarriageFace::Top,
                CarriageFace::PositiveSide,
                CarriageFace::NegativeSide,
            ] {
                let (mut graph, bearing) = rail_graph(face, GridRotation::new(1, 0, 0));
                if attached {
                    graph.apply(BuildCommand::AddBearing(bearing)).unwrap();
                }
                let socket = crate::BearingSocket {
                    kind: bearing.kind,
                    axis: bearing.axis,
                    source: bearing.source,
                    anchor: bearing.shared_anchor,
                    dimensions: bearing.dimensions,
                };
                let mut document = CreationDocument::from_graph(&graph, "Rail socket", &[socket]);
                document.transform_cardinal(1, IVec3::new(8, 16, -4));
                let decoded: CreationDocument =
                    ron::from_str(&ron::to_string(&document).unwrap()).unwrap();
                let restored = decoded.into_graph().unwrap();
                assert_eq!(restored.sockets.len(), 1);
                assert_eq!(restored.graph.bearing_count(), usize::from(attached));
                assert_eq!(
                    CreationDocument::from_graph(&restored.graph, "Rail socket", &restored.sockets),
                    document
                );
                let transformed = restored.sockets[0];
                let yaw = GridRotation::new(0, 1, 0).quaternion();
                assert!((transformed.axis - yaw * socket.axis).length() < 1.0e-6);
                assert!(
                    (transformed.anchor - (yaw * socket.anchor + Vec3::new(1.0, 2.0, -0.5)))
                        .length()
                        < 1.0e-6
                );
                let BearingKind::Linear(original) = socket.kind else {
                    unreachable!()
                };
                let BearingKind::Linear(rail) = transformed.kind else {
                    panic!("lost rail socket")
                };
                assert_eq!(rail.dimensions, original.dimensions);
                assert_eq!(rail.face, face);
                assert!((rail.mount_normal - yaw * original.mount_normal).length() < 1.0e-6);
            }
        }
    }

    #[test]
    fn a_carriage_cannot_attach_on_a_second_face() {
        let (mut graph, bearing) = rail_graph(CarriageFace::Top, GridRotation::default());
        graph.apply(BuildCommand::AddBearing(bearing)).unwrap();
        let target = spawn_block(&mut graph, IVec3::new(0, 954, 76), GridRotation::default());
        let mut side = bearing;
        side.target = FaceRef::part(target, FaceKind::NegativeZ);
        let BearingKind::Linear(ref mut rail) = side.kind else {
            unreachable!()
        };
        rail.face = CarriageFace::PositiveSide;
        assert!(graph.apply(BuildCommand::AddBearing(side)).is_err());
        assert_eq!(graph.bearing_count(), 1);
        assert_eq!(graph.compile().unwrap().bearings.len(), 1);
    }

    #[test]
    fn linear_drive_insertion_rejects_rotational_units_and_overextended_limits() {
        for target in [
            crate::DriveTarget::Angle(0.0),
            crate::DriveTarget::Speed(1.0),
        ] {
            let (mut graph, mut link) = linear_wire();
            link.program = target_program(target);
            assert!(graph.apply(BuildCommand::AddDriveLink(link)).is_err());
            assert_eq!(graph.drive_link_count(), 0);
        }
        let (mut graph, mut link) = linear_wire();
        link.linear_limits = None;
        assert!(graph.apply(BuildCommand::AddDriveLink(link)).is_err());
        link.linear_limits = Some(crate::LinearDriveLimits::new(1.0, 100.0, -0.5, 0.5).unwrap());
        assert!(graph.apply(BuildCommand::AddDriveLink(link)).is_err());
        assert_eq!(graph.drive_link_count(), 0);
    }

    #[test]
    fn live_linear_reprogramming_rejects_unit_changes_and_preserves_the_previous_row() {
        let (mut graph, link) = linear_wire();
        let BuildOutcome::DriveLinked(id) = graph.apply(BuildCommand::AddDriveLink(link)).unwrap()
        else {
            panic!("expected drive")
        };
        for target in [
            crate::DriveTarget::Angle(0.0),
            crate::DriveTarget::Speed(1.0),
        ] {
            assert!(
                graph
                    .apply(BuildCommand::SetDriveLink {
                        link: id,
                        limits: link.limits,
                        program: target_program(target),
                        name: link.name,
                        actuator: link.actuator,
                    })
                    .is_err()
            );
            assert_eq!(graph.drive_link(id), Some(&link));
        }
        assert!(
            graph
                .apply(BuildCommand::SetLinearDriveLimits {
                    link: id,
                    limits: crate::LinearDriveLimits::new(1.0, 100.0, -0.5, 0.5).unwrap(),
                })
                .is_err()
        );
        assert_eq!(graph.drive_link(id), Some(&link));
        let narrow = crate::LinearDriveLimits::new(0.25, 50.0, -0.1, 0.2).unwrap();
        graph
            .apply(BuildCommand::SetLinearDriveLimits {
                link: id,
                limits: narrow,
            })
            .unwrap();
        graph
            .apply(BuildCommand::SetDriveLink {
                link: id,
                limits: link.limits,
                program: target_program(crate::DriveTarget::LinearPosition(0.1)),
                name: link.name,
                actuator: link.actuator,
            })
            .unwrap();
        assert_eq!(graph.drive_link(id).unwrap().linear_limits, Some(narrow));
    }

    #[test]
    fn rotational_joints_reject_linear_programs_and_limits() {
        let (mut graph, rail) = rail_graph(CarriageFace::Top, GridRotation::default());
        let target = spawn_block(&mut graph, IVec3::new(0, 950, 0), GridRotation::default());
        let BuildOutcome::BearingAdded(id) = graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                rail.source,
                FaceRef::part(target, FaceKind::NegativeY),
                rail.shared_anchor,
                Vec3::Y,
            )))
            .unwrap()
        else {
            panic!("expected bearing")
        };
        let controller = spawn_controller(&mut graph);
        let valid = crate::DriveLinkSpec::new(controller, id);
        for target in [
            crate::DriveTarget::LinearPosition(0.1),
            crate::DriveTarget::LinearSpeed(0.1),
        ] {
            let mut link = valid;
            link.program = target_program(target);
            assert!(graph.apply(BuildCommand::AddDriveLink(link)).is_err());
        }
        let mut wrong_limits = valid;
        wrong_limits.linear_limits =
            Some(crate::LinearDriveLimits::new(1.0, 100.0, -0.1, 0.1).unwrap());
        assert!(
            graph
                .apply(BuildCommand::AddDriveLink(wrong_limits))
                .is_err()
        );
        let BuildOutcome::DriveLinked(link) =
            graph.apply(BuildCommand::AddDriveLink(valid)).unwrap()
        else {
            panic!("expected drive")
        };
        assert!(
            graph
                .apply(BuildCommand::SetLinearDriveLimits {
                    link,
                    limits: wrong_limits.linear_limits.unwrap(),
                })
                .is_err()
        );
        assert!(
            graph
                .apply(BuildCommand::SetDriveLink {
                    link,
                    limits: valid.limits,
                    program: target_program(crate::DriveTarget::LinearSpeed(0.1)),
                    name: valid.name,
                    actuator: valid.actuator,
                })
                .is_err()
        );
        assert_eq!(graph.drive_link(link), Some(&valid));
    }

    #[test]
    fn one_engine_shares_linear_force_and_gearing_preserves_output_power() {
        let (mut graph, mut first) = linear_wire();
        let original = *graph.bearing(first.bearing).unwrap();
        let second_target =
            spawn_block(&mut graph, IVec3::new(200, 990, 0), GridRotation::default());
        let second = BearingSpec {
            target: FaceRef::part(second_target, FaceKind::NegativeY),
            shared_anchor: original.shared_anchor + Vec3::X * 0.5,
            ..original
        };
        let BuildOutcome::BearingAdded(second_id) =
            graph.apply(BuildCommand::AddBearing(second)).unwrap()
        else {
            panic!("expected bearing")
        };
        let BuildOutcome::Spawned(engine) = graph
            .apply(BuildCommand::SpawnEngine(crate::EngineSpec::new(
                crate::EngineKind::Electric,
                BuildPose::new(IVec3::new(0, 42, 0), GridRotation::default()),
            )))
            .unwrap()
        else {
            panic!("expected engine")
        };
        graph
            .apply(BuildCommand::Weld(crate::WeldSpec {
                first: FaceRef::part(first.controller, FaceKind::PositiveY),
                second: FaceRef::part(engine, FaceKind::NegativeY),
            }))
            .unwrap();
        first.actuator = crate::ActuatorAssignment::motor(100, 0).unwrap();
        first.program = target_program(crate::DriveTarget::LinearSpeed(0.8));
        graph.apply(BuildCommand::AddDriveLink(first)).unwrap();
        graph
            .apply(BuildCommand::AddDriveLink(crate::DriveLinkSpec {
                bearing: second_id,
                ..first
            }))
            .unwrap();
        let compiled = graph.compile().unwrap();
        let direct = compiled.resolve_coordinate_drives(&graph);
        let geared = compiled.resolve_coordinate_drives_with_gears(
            &graph,
            &[crate::GearSelection {
                controller: first.controller,
                kind: crate::EngineKind::Electric,
                ratio: Some(4.0),
            }],
        );
        assert_eq!(direct.len(), 2);
        for (coordinate, (direct, geared)) in direct.iter().zip(&geared).enumerate() {
            let effective_mass = compiled.loop_topology.coordinate_axis_inertia[coordinate];
            let force = direct.source_a_max_acceleration * effective_mass;
            let geared_force = geared.source_a_max_acceleration * effective_mass;
            assert!((force * LINEAR_METERS_PER_RADIAN - 250.0).abs() < 0.001);
            assert!((direct.max_speed - 0.5).abs() < 1.0e-5);
            assert!((direct.target_speed - 0.5).abs() < 1.0e-5);
            assert!((geared.max_speed - 0.125).abs() < 1.0e-5);
            assert!((geared_force / force - 4.0).abs() < 1.0e-5);
            assert!((force * direct.max_speed - geared_force * geared.max_speed).abs() < 0.01);
        }
    }

    fn spawn_block(graph: &mut ConstructionGraph, ticks: IVec3, rotation: GridRotation) -> PartId {
        let spec =
            CuboidSpec::new([1, 1, 1], BuildPose::from_position_ticks(ticks, rotation)).unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
            panic!("expected a block");
        };
        part
    }

    fn rail_graph(face: CarriageFace, rotation: GridRotation) -> (ConstructionGraph, BearingSpec) {
        let mut graph = ConstructionGraph::new();
        let orientation = rotation.quaternion();
        let offset = IVec3::new(0, 800, 0);
        let rotated_ticks =
            |ticks: IVec3| (orientation * ticks.as_vec3()).round().as_ivec3() + offset;
        let base = spawn_block(&mut graph, rotated_ticks(IVec3::new(0, 50, 0)), rotation);
        let (target_ticks, target_face) = match face {
            CarriageFace::Top => (IVec3::new(0, 190, 0), FaceKind::NegativeY),
            CarriageFace::PositiveSide => (IVec3::new(0, 154, 76), FaceKind::NegativeZ),
            CarriageFace::NegativeSide => (IVec3::new(0, 154, -76), FaceKind::PositiveZ),
        };
        let target = spawn_block(&mut graph, rotated_ticks(target_ticks), rotation);
        let rail = LinearBearing {
            dimensions: LinearBearingDimensions::default(),
            mount_normal: orientation * Vec3::Y,
            face,
        };
        let bearing = BearingSpec::new(
            FaceRef::part(base, FaceKind::PositiveY),
            FaceRef::part(target, target_face),
            orientation * Vec3::new(0.0, 0.25, 0.0) + offset.as_vec3() / 400.0,
            orientation * Vec3::X,
        )
        .with_kind(BearingKind::Linear(rail));
        (graph, bearing)
    }

    #[test]
    fn every_supported_dimension_survives_serialization() {
        for ticks in 100_u16..=3200 {
            let dimensions = LinearBearingDimensions::new(f32::from(ticks) / 400.0, 0.1)
                .unwrap_or_else(|error| panic!("length tick {ticks}: {error}"));
            let encoded = ron::to_string(&dimensions).unwrap();
            let decoded: LinearBearingDimensions = ron::from_str(&encoded).unwrap();
            assert_eq!(decoded, dimensions);
        }
        for ticks in 20_u16..=160 {
            let dimensions = LinearBearingDimensions::new(1.0, f32::from(ticks) / 400.0).unwrap();
            let encoded = ron::to_string(&dimensions).unwrap();
            let decoded: LinearBearingDimensions = ron::from_str(&encoded).unwrap();
            assert_eq!(decoded, dimensions);
        }
    }

    #[test]
    fn top_and_side_attachments_compile_in_every_cardinal_orientation() {
        for x in 0..4 {
            for y in 0..4 {
                for z in 0..4 {
                    for face in [
                        CarriageFace::Top,
                        CarriageFace::PositiveSide,
                        CarriageFace::NegativeSide,
                    ] {
                        let (mut graph, bearing) = rail_graph(face, GridRotation::new(x, y, z));
                        graph
                            .apply(BuildCommand::AddBearing(bearing))
                            .unwrap_or_else(|error| {
                                panic!("rotation {x}, {y}, {z}, face {face:?}: {error}")
                            });
                        let compiled = graph.compile().unwrap();
                        assert_eq!(compiled.bearings.len(), 1);
                        assert_eq!(compiled.bearings[0].kind, bearing.kind);
                        assert_eq!(compiled.loop_topology.tree_bearings.len(), 1);
                    }
                }
            }
        }
    }

    #[test]
    fn maximum_rail_can_overhang_a_single_support_block() {
        let (mut graph, mut bearing) = rail_graph(CarriageFace::Top, GridRotation::default());
        let BearingKind::Linear(ref mut rail) = bearing.kind else {
            unreachable!()
        };
        rail.dimensions = LinearBearingDimensions::new(8.0, 0.1).unwrap();
        graph.apply(BuildCommand::AddBearing(bearing)).unwrap();
        let compiled = graph.compile().unwrap();
        let [min, max] = compiled.bearings[0].kind.bounds();
        assert!((min + 3.925).abs() < 1.0e-6 && (max - 3.925).abs() < 1.0e-6);
        assert_eq!(
            compiled.compounds.len(),
            2,
            "the connector adds no physical body"
        );
    }

    #[test]
    fn duplicate_linear_rows_share_one_physical_coordinate() {
        let (mut graph, bearing) = rail_graph(CarriageFace::Top, GridRotation::default());
        let BuildOutcome::BearingAdded(first) =
            graph.apply(BuildCommand::AddBearing(bearing)).unwrap()
        else {
            panic!("expected bearing");
        };
        let BuildOutcome::BearingAdded(second) =
            graph.apply(BuildCommand::AddBearing(bearing)).unwrap()
        else {
            panic!("expected duplicate bearing");
        };
        let compiled = graph.compile().unwrap();
        assert_eq!(compiled.bearings.len(), 1);
        let coordinates = &compiled.loop_topology.bearing_coordinates;
        assert!(coordinates.contains_key(&first));
        assert_eq!(coordinates[&first], coordinates[&second]);
    }

    #[test]
    fn deleting_either_attachment_removes_its_linear_joint() {
        for delete_source in [false, true] {
            let (mut graph, bearing) = rail_graph(CarriageFace::Top, GridRotation::default());
            graph.apply(BuildCommand::AddBearing(bearing)).unwrap();
            let face = if delete_source {
                bearing.source
            } else {
                bearing.target
            };
            let crate::FaceOwner::Part(part) = face.owner else {
                unreachable!()
            };
            graph.apply(BuildCommand::Remove(part)).unwrap();
            assert_eq!(graph.bearing_count(), 0);
            assert!(graph.compile().unwrap().bearings.is_empty());
        }
    }

    #[test]
    fn transformed_linear_creation_preserves_frame_face_and_dimensions() {
        for face in [
            CarriageFace::Top,
            CarriageFace::PositiveSide,
            CarriageFace::NegativeSide,
        ] {
            let (mut graph, bearing) = rail_graph(face, GridRotation::new(1, 0, 0));
            graph.apply(BuildCommand::AddBearing(bearing)).unwrap();
            let mut document = CreationDocument::from_graph(&graph, "Linear", &[]);
            document.transform_cardinal(1, IVec3::new(8, 16, -4));
            let encoded = ron::to_string(&document).unwrap();
            let decoded: CreationDocument = ron::from_str(&encoded).unwrap();
            let restored = decoded.into_graph().unwrap();
            assert_eq!(
                CreationDocument::from_graph(&restored.graph, "Linear", &[]),
                document
            );
            let compiled = restored.graph.compile().unwrap();
            let BearingKind::Linear(rail) = compiled.bearings[0].kind else {
                panic!("lost linear joint")
            };
            assert_eq!(rail.face, face);
            assert_eq!(rail.dimensions, LinearBearingDimensions::default());
            let yaw = GridRotation::new(0, 1, 0).quaternion();
            let BearingKind::Linear(original) = bearing.kind else {
                unreachable!()
            };
            assert!((rail.mount_normal - yaw * original.mount_normal).length() < 1.0e-6);
            let (_, restored_bearing) = restored.graph.bearings().next().unwrap();
            assert!((restored_bearing.axis - yaw * bearing.axis).length() < 1.0e-6);
            assert!(
                (restored_bearing.shared_anchor
                    - (yaw * bearing.shared_anchor + Vec3::new(1.0, 2.0, -0.5)))
                .length()
                    < 1.0e-6
            );
        }
    }

    #[test]
    fn default_carriage_is_130_mm_wide_with_850_mm_travel() {
        let dimensions = LinearBearingDimensions::default();
        assert!((dimensions.carriage_half_width() * 2.0 - 0.130).abs() < 1.0e-6);
        assert!((dimensions.travel() - 0.850).abs() < 1.0e-6);
    }

    #[test]
    fn invalid_and_off_grid_dimensions_are_rejected() {
        for length in [f32::NAN, f32::INFINITY, 0.249, 8.001] {
            assert!(LinearBearingDimensions::new(length, 0.1).is_err());
        }
        for width in [f32::NAN, f32::INFINITY, 0.049, 0.401] {
            assert!(LinearBearingDimensions::new(1.0, width).is_err());
        }
        assert_eq!(
            LinearBearingDimensions::new(1.001, 0.1),
            Err(LinearBearingError::Quantization)
        );
        for length in [0.25, 1.0, 8.0] {
            for ticks in 20_u16..=160 {
                let width = f32::from(ticks) / 400.0;
                let dimensions = LinearBearingDimensions::new(length, width).unwrap();
                let half_ticks = dimensions.carriage_half_width() / 0.0025;
                assert!((half_ticks - half_ticks.round()).abs() < 0.0001);
                assert!(dimensions.carriage_half_width() > width / 2.0 + 0.002);
            }
        }
    }

    #[test]
    fn all_mounting_orientations_preserve_flush_attachment_planes() {
        let dimensions = LinearBearingDimensions::default();
        for normal in [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::NEG_X,
            Vec3::NEG_Y,
            Vec3::NEG_Z,
        ] {
            for axis in [
                Vec3::X,
                Vec3::Y,
                Vec3::Z,
                Vec3::NEG_X,
                Vec3::NEG_Y,
                Vec3::NEG_Z,
            ] {
                if normal.dot(axis) != 0.0 {
                    continue;
                }
                for face in [
                    CarriageFace::Top,
                    CarriageFace::PositiveSide,
                    CarriageFace::NegativeSide,
                ] {
                    let rail = LinearBearing {
                        dimensions,
                        mount_normal: normal,
                        face,
                    };
                    let rotation = rail.rotation(axis).unwrap();
                    assert!((rotation * Vec3::X - axis).length() < 1.0e-6);
                    assert!((rotation * Vec3::Y - normal).length() < 1.0e-6);
                    let origin = rotation * face.origin(dimensions);
                    let face_normal = rotation * face.normal();
                    let offset = if face == CarriageFace::Top {
                        0.1
                    } else {
                        0.065
                    };
                    assert!((origin.dot(face_normal) - offset).abs() < 1.0e-6);
                }
            }
        }
    }

    #[test]
    fn transmission_conversion_conserves_power() {
        let angular_speed = 12.0;
        let torque = 42.0;
        let speed = angular_speed * LINEAR_METERS_PER_RADIAN;
        let force = torque / LINEAR_METERS_PER_RADIAN;
        assert!((speed * force - angular_speed * torque).abs() < 0.0001);
    }
}

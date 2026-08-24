//! Serializable form of a saved creation.
//!
//! [`ConstructionGraph`] stores its rows in generational arenas whose handles
//! are minted privately and are not stable across a rebuild, so a file cannot
//! reference them directly. A [`CreationDocument`] instead numbers each row by
//! its position in the file and rebuilds the graph by replaying
//! [`BuildCommand`]s, remapping those dense indices onto the handles the arenas
//! hand back. Every value passes through the same validating constructors the
//! editor uses, so a hand-edited file cannot produce an invalid graph.

use std::collections::HashMap;

use bevy_math::Vec3;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    ActuatorAssignment, BearingDimensionError, BearingDimensions, BearingSpec, BuildCommand,
    BuildOutcome, BuildPose, ConstructionGraph, ControllerSpec, CuboidSpec, CylinderDimensionError,
    CylinderDimensions, CylinderSpec, DimensionError, DriveDwell, DriveKey, DriveLimits,
    DriveLimitsError, DriveLinkSpec, DriveName, DriveProgram, DriveProgramError, DriveRelease,
    DriveState, DriveTarget, DriveTrigger, EngineKind, EngineSpec, FaceKind, FaceOwner, FaceRef,
    GraphError, GridDimension, GridRotation, InputSeatLinkSpec, InputSpec, PartId, PartSpec,
    RigidLinkSpec, SeatControllerLinkSpec, SeatSpec, ServoSpec, WeldSpec,
};

/// Format version written by this build. Files carrying anything else are
/// refused rather than guessed at.
pub const CREATION_FORMAT_VERSION: u32 = 3;
const OLDEST_CREATION_FORMAT_VERSION: u32 = 1;

/// A bearing ring placed on a face with nothing attached through it yet.
///
/// The graph cannot hold these: a bearing needs two endpoints. The editor owns
/// them, and a saved creation carries them alongside the graph so a half-built
/// machine reloads exactly as it was left.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BearingSocket {
    /// Face the ring sits on.
    pub source: FaceRef,
    /// World-space point the ring is centred on.
    pub anchor: Vec3,
    /// Visual outer and inner diameters.
    pub dimensions: BearingDimensions,
}

/// Reason a creation file could not be turned back into a graph.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum CreationError {
    /// The file was written by a different format version.
    #[error(
        "creation format version {0} is not supported; this build reads versions {OLDEST_CREATION_FORMAT_VERSION} through {CREATION_FORMAT_VERSION}"
    )]
    UnsupportedVersion(u32),
    /// A record referenced a part the file does not define.
    #[error("creation references part {0}, which the file does not define")]
    MissingPart(u32),
    /// A drive wire referenced a bearing the file does not define.
    #[error("creation references bearing {0}, which the file does not define")]
    MissingBearing(u32),
    /// A drive state was bound to something that is not a letter or a digit.
    #[error("drive state key {0:?} is not a letter or a digit")]
    InvalidDriveKey(char),
    /// A cuboid dimension was out of range.
    #[error(transparent)]
    Dimension(#[from] DimensionError),
    /// A cylinder dimension was out of range.
    #[error(transparent)]
    CylinderDimension(#[from] CylinderDimensionError),
    /// A bearing ring dimension was out of range.
    #[error(transparent)]
    BearingDimension(#[from] BearingDimensionError),
    /// A drive program was malformed.
    #[error(transparent)]
    DriveProgram(#[from] DriveProgramError),
    /// A drive envelope was out of range.
    #[error(transparent)]
    DriveLimits(#[from] DriveLimitsError),
    /// The replayed commands did not describe a valid construction.
    #[error(transparent)]
    Graph(#[from] GraphError),
}

/// Grid-aligned pose in its serialized form.
///
/// The translation is kept in eighth-metre half-grid units, which is the one
/// representation that survives a round trip for odd-sized cuboids resting
/// flush against a face.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoseDoc {
    /// Centre in integer eighth-metre units.
    pub translation_half_units: [i32; 3],
    /// Quarter turns around local x, y, and z.
    pub rotation: [u8; 3],
}

impl From<BuildPose> for PoseDoc {
    fn from(pose: BuildPose) -> Self {
        let translation = pose.translation_half_units();
        Self {
            translation_half_units: [translation.x, translation.y, translation.z],
            rotation: pose.rotation.quarter_turns_xyz(),
        }
    }
}

impl From<PoseDoc> for BuildPose {
    fn from(doc: PoseDoc) -> Self {
        let [x, y, z] = doc.rotation;
        Self::from_half_grid(
            doc.translation_half_units.into(),
            GridRotation::new(x, y, z),
        )
    }
}

/// One construction part in its serialized form.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum PartDoc {
    /// Rectangular cuboid, sized in quarter-metre grid units.
    Cuboid {
        /// Integer x/y/z side lengths.
        dimensions: [u8; 3],
        /// Centre and orientation.
        pose: PoseDoc,
    },
    /// Solid or hollow cylinder whose axis is local Y.
    Cylinder {
        /// Outer diameter in metres.
        outer_diameter: f32,
        /// Inner diameter in metres. Zero is solid.
        inner_diameter: f32,
        /// Axial length in quarter-metre grid units.
        length_units: u8,
        /// Retained angular sector in degrees.
        sweep_degrees: u16,
        /// Centre and orientation.
        pose: PoseDoc,
    },
    /// Fixed-size control block.
    Controller {
        /// Centre and orientation.
        pose: PoseDoc,
    },
    /// Fixed-size gas or electric engine.
    Engine {
        /// Authored engine family.
        kind: EngineKind,
        /// Centre and orientation.
        pose: PoseDoc,
    },
    /// Fixed-size servo.
    Servo {
        /// Centre and orientation.
        pose: PoseDoc,
    },
    /// Fixed-size seat cushion.
    Seat {
        /// Centre and orientation.
        pose: PoseDoc,
    },
    /// Fixed-size keyboard Input block.
    Input {
        /// Centre and orientation.
        pose: PoseDoc,
    },
}

/// Owner of a serialized face: a part index, or the ground plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FaceOwnerDoc {
    /// Index into the document's part list.
    Part(u32),
    /// The static ground plane.
    Ground,
}

/// Reference to one oriented face in its serialized form.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FaceRefDoc {
    /// Part index or ground.
    pub owner: FaceOwnerDoc,
    /// Oriented face on that owner.
    pub face: FaceKind,
}

/// Weld between two touching faces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeldDoc {
    /// First welded face.
    pub first: FaceRefDoc,
    /// Second welded face.
    pub second: FaceRefDoc,
}

/// Non-geometric rigid membership between two part indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RigidLinkDoc {
    /// First part index.
    pub first: u32,
    /// Second part index.
    pub second: u32,
}

/// One-degree-of-freedom bearing in its serialized form.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BearingDoc {
    /// Face whose outward normal establishes the axis.
    pub source: FaceRefDoc,
    /// Compatible face on the attached side.
    pub target: FaceRefDoc,
    /// Shared world-space anchor.
    pub anchor: [f32; 3],
    /// Unit world-space axis.
    pub axis: [f32; 3],
    /// Visual outer diameter in metres.
    pub outer_diameter: f32,
    /// Visual inner diameter in metres.
    pub inner_diameter: f32,
}

/// Unattached bearing ring in its serialized form.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BearingSocketDoc {
    /// Face the ring sits on.
    pub source: FaceRefDoc,
    /// World-space point the ring is centred on.
    pub anchor: [f32; 3],
    /// Visual outer diameter in metres.
    pub outer_diameter: f32,
    /// Visual inner diameter in metres.
    pub inner_diameter: f32,
}

/// Speed, torque, and travel envelope in its serialized form.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DriveLimitsDoc {
    /// Fastest the bearing may turn, in radians per second.
    pub max_speed_rad_s: f32,
    /// Maximum applied torque in newton metres. `None` is unlimited, which is
    /// how the panel's `inf` reads on disk without encoding a float infinity.
    pub max_torque_newton_meters: Option<f32>,
    /// Travel limits in radians, when the bearing stops and holds at its ends.
    pub angle_limits: Option<(f32, f32)>,
}

/// Automatic handoff in its serialized form.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DriveDwellDoc {
    /// How long the state stays active, in seconds.
    pub seconds: f32,
    /// Explicit handoff target, or `None` for the following state.
    pub next: Option<u8>,
}

/// Key binding in its serialized form.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DriveTriggerDoc {
    /// Bound character, always an uppercase letter or a digit.
    pub key: char,
    /// What happens when the key is released.
    pub release: DriveRelease,
}

/// One drive state in its serialized form.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DriveStateDoc {
    /// What this state asks of the bearing.
    pub target: DriveTarget,
    /// Automatic handoff, when this state advances on its own.
    pub dwell: Option<DriveDwellDoc>,
    /// Key binding, when this state can be triggered by hand.
    pub trigger: Option<DriveTriggerDoc>,
}

/// Ordered states of one driven bearing in their serialized form.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DriveProgramDoc {
    /// Whether the last state hands back to the first.
    pub loops: bool,
    /// The states, in order.
    pub states: Vec<DriveStateDoc>,
}

/// Wire from a control block to one bearing, in its serialized form.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DriveLinkDoc {
    /// Index of the control-block part this wire belongs to.
    pub controller: u32,
    /// Index of the bearing driven through this wire.
    pub bearing: u32,
    /// Whether this bearing runs opposite the programmed direction.
    pub reversed: bool,
    /// Hardware family assigned to this joint. Older files load unpowered.
    #[serde(default)]
    pub actuator: ActuatorAssignment,
    /// Speed, torque, and travel envelope.
    pub limits: DriveLimitsDoc,
    /// Ordered states this bearing moves through.
    pub program: DriveProgramDoc,
    /// What the panel calls this joint. Absent in files written before joints
    /// could be named, which read back as unnamed.
    #[serde(default)]
    pub name: String,
}

/// Logical link from an Input block to a Seat.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputSeatLinkDoc {
    /// Index of the Input part.
    pub input: u32,
    /// Index of the Seat part.
    pub seat: u32,
}

/// Logical link from a Seat to a Controller.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeatControllerLinkDoc {
    /// Index of the Seat part.
    pub seat: u32,
    /// Index of the Controller part.
    pub controller: u32,
}

/// A whole saved creation: everything the editor authors, and nothing it
/// derives.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CreationDocument {
    /// Format version. See [`CREATION_FORMAT_VERSION`].
    pub version: u32,
    /// Display name, kept in the file so renaming a file does not rename the
    /// creation and vice versa.
    pub name: String,
    /// Parts, in the order every other record indexes them by.
    pub parts: Vec<PartDoc>,
    /// Welds between touching faces.
    #[serde(default)]
    pub welds: Vec<WeldDoc>,
    /// Non-geometric rigid memberships.
    #[serde(default)]
    pub rigid_links: Vec<RigidLinkDoc>,
    /// Bearings, in the order drive wires index them by.
    #[serde(default)]
    pub bearings: Vec<BearingDoc>,
    /// Control-block wires.
    #[serde(default)]
    pub drive_links: Vec<DriveLinkDoc>,
    /// Logical Input-to-Seat links.
    #[serde(default)]
    pub input_seat_links: Vec<InputSeatLinkDoc>,
    /// Logical Seat-to-Controller links.
    #[serde(default)]
    pub seat_controller_links: Vec<SeatControllerLinkDoc>,
    /// Bearing rings placed but not yet attached through.
    #[serde(default)]
    pub sockets: Vec<BearingSocketDoc>,
}

/// A creation rebuilt from a document.
#[derive(Clone, Debug)]
pub struct LoadedCreation {
    /// Display name the file carried.
    pub name: String,
    /// The rebuilt construction.
    pub graph: ConstructionGraph,
    /// Unattached bearing rings, with their part references resolved.
    pub sockets: Vec<BearingSocket>,
}

impl CreationDocument {
    /// Captures a construction and its unattached bearing rings.
    ///
    /// Any pending two-step operation on `graph` is ignored: it is transient
    /// editor state, not part of the creation.
    ///
    /// # Panics
    ///
    /// Never in practice: the arenas already refuse to exceed `u32` indices.
    pub fn from_graph(graph: &ConstructionGraph, name: &str, sockets: &[BearingSocket]) -> Self {
        let part_indices = index_map(graph.parts().map(|(id, _)| id));
        let bearing_indices = index_map(graph.bearings().map(|(id, _)| id));
        let face = |face: FaceRef| face_doc(face, &part_indices);
        let part = |part: PartId| {
            *part_indices
                .get(&part)
                .expect("every referenced part is live in the graph it came from")
        };

        Self {
            version: CREATION_FORMAT_VERSION,
            name: name.to_owned(),
            parts: graph.parts().map(|(_, spec)| part_doc(*spec)).collect(),
            welds: graph
                .welds()
                .map(|(_, weld)| WeldDoc {
                    first: face(weld.first),
                    second: face(weld.second),
                })
                .collect(),
            rigid_links: graph
                .rigid_links()
                .map(|(_, link)| RigidLinkDoc {
                    first: part(link.first),
                    second: part(link.second),
                })
                .collect(),
            bearings: graph
                .bearings()
                .map(|(_, bearing)| BearingDoc {
                    source: face(bearing.source),
                    target: face(bearing.target),
                    anchor: bearing.shared_anchor.to_array(),
                    axis: bearing.axis.to_array(),
                    outer_diameter: bearing.dimensions.outer_diameter(),
                    inner_diameter: bearing.dimensions.inner_diameter(),
                })
                .collect(),
            drive_links: graph
                .drive_links()
                .map(|(_, link)| DriveLinkDoc {
                    controller: part(link.controller),
                    bearing: *bearing_indices
                        .get(&link.bearing)
                        .expect("every wired bearing is live in the graph it came from"),
                    reversed: link.reversed,
                    actuator: link.actuator,
                    limits: limits_doc(link.limits),
                    program: program_doc(&link.program),
                    name: link.name.to_string(),
                })
                .collect(),
            input_seat_links: graph
                .input_seat_links()
                .map(|(_, link)| InputSeatLinkDoc {
                    input: part(link.input),
                    seat: part(link.seat),
                })
                .collect(),
            seat_controller_links: graph
                .seat_controller_links()
                .map(|(_, link)| SeatControllerLinkDoc {
                    seat: part(link.seat),
                    controller: part(link.controller),
                })
                .collect(),
            sockets: sockets
                .iter()
                .map(|socket| BearingSocketDoc {
                    source: face(socket.source),
                    anchor: socket.anchor.to_array(),
                    outer_diameter: socket.dimensions.outer_diameter(),
                    inner_diameter: socket.dimensions.inner_diameter(),
                })
                .collect(),
        }
    }

    /// Rebuilds the construction this document describes.
    ///
    /// Parts are spawned first so the handles they return can resolve every
    /// later reference, then connections, then drive wires.
    ///
    /// # Errors
    ///
    /// Returns [`CreationError`] when the version is unsupported, an index
    /// names a row the file does not define, a value is outside its supported
    /// range, or the replayed commands do not describe a valid construction.
    #[allow(clippy::too_many_lines)] // One replay pass per serialized record family.
    pub fn into_graph(self) -> Result<LoadedCreation, CreationError> {
        if !(OLDEST_CREATION_FORMAT_VERSION..=CREATION_FORMAT_VERSION).contains(&self.version) {
            return Err(CreationError::UnsupportedVersion(self.version));
        }

        let mut graph = ConstructionGraph::new();
        let spawns = self
            .parts
            .iter()
            .map(|part| build_command(*part))
            .collect::<Result<Vec<_>, _>>()?;
        let part_ids = spawned_ids(graph.apply_batch(spawns)?);

        let mut connections =
            Vec::with_capacity(self.welds.len() + self.rigid_links.len() + self.bearings.len());
        for weld in &self.welds {
            connections.push(BuildCommand::Weld(WeldSpec {
                first: resolve_face(weld.first, &part_ids)?,
                second: resolve_face(weld.second, &part_ids)?,
            }));
        }
        for link in &self.rigid_links {
            connections.push(BuildCommand::RigidLink(RigidLinkSpec {
                first: resolve_part(link.first, &part_ids)?,
                second: resolve_part(link.second, &part_ids)?,
            }));
        }
        let first_bearing = connections.len();
        for bearing in &self.bearings {
            connections.push(BuildCommand::AddBearing(
                BearingSpec::new(
                    resolve_face(bearing.source, &part_ids)?,
                    resolve_face(bearing.target, &part_ids)?,
                    Vec3::from_array(bearing.anchor),
                    Vec3::from_array(bearing.axis),
                )
                .with_dimensions(BearingDimensions::new(
                    bearing.outer_diameter,
                    bearing.inner_diameter,
                )?),
            ));
        }
        let outcomes = graph.apply_batch(connections)?;
        let bearing_ids = outcomes[first_bearing..]
            .iter()
            .map(|outcome| match outcome {
                BuildOutcome::BearingAdded(bearing) => *bearing,
                _ => unreachable!("the bearing tail of the batch only contains bearing commands"),
            })
            .collect::<Vec<_>>();

        let wires = self
            .drive_links
            .iter()
            .map(|link| {
                Ok(BuildCommand::AddDriveLink(DriveLinkSpec {
                    controller: resolve_part(link.controller, &part_ids)?,
                    bearing: *bearing_ids
                        .get(link.bearing as usize)
                        .ok_or(CreationError::MissingBearing(link.bearing))?,
                    reversed: link.reversed,
                    actuator: link.actuator,
                    limits: resolve_limits(link.limits)?,
                    program: resolve_program(&link.program)?,
                    name: DriveName::new(&link.name),
                }))
            })
            .collect::<Result<Vec<_>, CreationError>>()?;
        graph.apply_batch(wires)?;

        let logical_links = self
            .input_seat_links
            .iter()
            .map(|link| {
                Ok(BuildCommand::AddInputSeatLink(InputSeatLinkSpec {
                    input: resolve_part(link.input, &part_ids)?,
                    seat: resolve_part(link.seat, &part_ids)?,
                }))
            })
            .chain(self.seat_controller_links.iter().map(|link| {
                Ok(BuildCommand::AddSeatControllerLink(
                    SeatControllerLinkSpec {
                        seat: resolve_part(link.seat, &part_ids)?,
                        controller: resolve_part(link.controller, &part_ids)?,
                    },
                ))
            }))
            .collect::<Result<Vec<_>, CreationError>>()?;
        graph.apply_batch(logical_links)?;

        let sockets = self
            .sockets
            .iter()
            .map(|socket| {
                Ok(BearingSocket {
                    source: resolve_face(socket.source, &part_ids)?,
                    anchor: Vec3::from_array(socket.anchor),
                    dimensions: BearingDimensions::new(
                        socket.outer_diameter,
                        socket.inner_diameter,
                    )?,
                })
            })
            .collect::<Result<Vec<_>, CreationError>>()?;

        Ok(LoadedCreation {
            name: self.name,
            graph,
            sockets,
        })
    }
}

fn index_map<I: Copy + Eq + std::hash::Hash>(ids: impl Iterator<Item = I>) -> HashMap<I, u32> {
    ids.enumerate()
        .map(|(index, id)| {
            (
                id,
                u32::try_from(index).expect("construction arena indices fit u32"),
            )
        })
        .collect()
}

fn face_doc(face: FaceRef, parts: &HashMap<PartId, u32>) -> FaceRefDoc {
    FaceRefDoc {
        owner: match face.owner {
            FaceOwner::Part(part) => FaceOwnerDoc::Part(
                *parts
                    .get(&part)
                    .expect("every referenced part is live in the graph it came from"),
            ),
            FaceOwner::Ground => FaceOwnerDoc::Ground,
        },
        face: face.face,
    }
}

fn part_doc(spec: PartSpec) -> PartDoc {
    match spec {
        PartSpec::Cuboid(cuboid) => PartDoc::Cuboid {
            dimensions: cuboid.dimensions.map(GridDimension::units),
            pose: cuboid.pose.into(),
        },
        PartSpec::Cylinder(cylinder) => PartDoc::Cylinder {
            outer_diameter: cylinder.dimensions.outer_diameter(),
            inner_diameter: cylinder.dimensions.inner_diameter(),
            length_units: cylinder.dimensions.axial_length_units(),
            sweep_degrees: cylinder.dimensions.sweep_angle_degrees(),
            pose: cylinder.pose.into(),
        },
        PartSpec::Controller(controller) => PartDoc::Controller {
            pose: controller.pose.into(),
        },
        PartSpec::Engine(engine) => PartDoc::Engine {
            kind: engine.kind,
            pose: engine.pose.into(),
        },
        PartSpec::Servo(servo) => PartDoc::Servo {
            pose: servo.pose.into(),
        },
        PartSpec::Seat(seat) => PartDoc::Seat {
            pose: seat.pose.into(),
        },
        PartSpec::Input(input) => PartDoc::Input {
            pose: input.pose.into(),
        },
    }
}

fn limits_doc(limits: DriveLimits) -> DriveLimitsDoc {
    let torque = limits.max_torque_newton_meters();
    DriveLimitsDoc {
        max_speed_rad_s: limits.max_speed_rad_s(),
        max_torque_newton_meters: torque.is_finite().then_some(torque),
        angle_limits: limits.angle_limits(),
    }
}

fn program_doc(program: &DriveProgram) -> DriveProgramDoc {
    DriveProgramDoc {
        loops: program.loops(),
        states: program
            .states()
            .iter()
            .map(|state| DriveStateDoc {
                target: state.target(),
                dwell: state.dwell().map(|dwell| DriveDwellDoc {
                    seconds: dwell.seconds(),
                    next: dwell.next(),
                }),
                trigger: state.trigger().map(|trigger| DriveTriggerDoc {
                    key: trigger.key().symbol(),
                    release: trigger.release(),
                }),
            })
            .collect(),
    }
}

fn build_command(part: PartDoc) -> Result<BuildCommand, CreationError> {
    Ok(match part {
        PartDoc::Cuboid { dimensions, pose } => {
            BuildCommand::Spawn(CuboidSpec::new(dimensions, pose.into())?)
        }
        PartDoc::Cylinder {
            outer_diameter,
            inner_diameter,
            length_units,
            sweep_degrees,
            pose,
        } => BuildCommand::SpawnCylinder(CylinderSpec::new(
            CylinderDimensions::new(
                outer_diameter,
                inner_diameter,
                f32::from(length_units) * crate::GRID_UNIT_METERS,
            )?
            .with_sweep_angle_degrees(sweep_degrees)?,
            pose.into(),
        )),
        PartDoc::Controller { pose } => {
            BuildCommand::SpawnController(ControllerSpec::new(pose.into()))
        }
        PartDoc::Engine { kind, pose } => {
            BuildCommand::SpawnEngine(EngineSpec::new(kind, pose.into()))
        }
        PartDoc::Servo { pose } => BuildCommand::SpawnServo(ServoSpec::new(pose.into())),
        PartDoc::Seat { pose } => BuildCommand::SpawnSeat(SeatSpec::new(pose.into())),
        PartDoc::Input { pose } => BuildCommand::SpawnInput(InputSpec::new(pose.into())),
    })
}

fn spawned_ids(outcomes: Vec<BuildOutcome>) -> Vec<PartId> {
    outcomes
        .into_iter()
        .map(|outcome| match outcome {
            BuildOutcome::Spawned(part) => part,
            _ => unreachable!("the spawn batch only contains spawn commands"),
        })
        .collect()
}

fn resolve_part(index: u32, parts: &[PartId]) -> Result<PartId, CreationError> {
    parts
        .get(index as usize)
        .copied()
        .ok_or(CreationError::MissingPart(index))
}

fn resolve_face(face: FaceRefDoc, parts: &[PartId]) -> Result<FaceRef, CreationError> {
    Ok(match face.owner {
        FaceOwnerDoc::Part(index) => FaceRef::part(resolve_part(index, parts)?, face.face),
        FaceOwnerDoc::Ground => FaceRef {
            owner: FaceOwner::Ground,
            face: face.face,
        },
    })
}

fn resolve_limits(limits: DriveLimitsDoc) -> Result<DriveLimits, CreationError> {
    Ok(DriveLimits::new(
        limits.max_speed_rad_s,
        limits.max_torque_newton_meters.unwrap_or(f32::INFINITY),
        limits.angle_limits,
    )?)
}

fn resolve_program(program: &DriveProgramDoc) -> Result<DriveProgram, CreationError> {
    let states = program
        .states
        .iter()
        .map(|state| {
            let dwell = state
                .dwell
                .map(|dwell| DriveDwell::new(dwell.seconds, dwell.next))
                .transpose()?;
            let trigger = state
                .trigger
                .map(|trigger| {
                    DriveKey::new(trigger.key)
                        .map(|key| DriveTrigger::new(key, trigger.release))
                        .ok_or(CreationError::InvalidDriveKey(trigger.key))
                })
                .transpose()?;
            Ok(DriveState::new(state.target)?
                .with_dwell(dwell)
                .with_trigger(trigger))
        })
        .collect::<Result<Vec<_>, CreationError>>()?;
    Ok(DriveProgram::new(&states, program.loops)?)
}

#[cfg(test)]
mod tests {
    use bevy_math::{IVec3, Vec3};

    use super::{
        BearingSocket, CREATION_FORMAT_VERSION, CreationDocument, CreationError, FaceOwnerDoc,
    };
    use crate::{
        ActuatorAssignment, BearingDimensions, BearingSpec, BuildCommand, BuildOutcome, BuildPose,
        ConstructionGraph, ControllerSpec, CuboidSpec, CylinderDimensions, CylinderSpec,
        DriveDwell, DriveKey, DriveLimits, DriveLinkSpec, DriveName, DriveProgram, DriveRelease,
        DriveState, DriveTarget, DriveTrigger, EngineKind, EngineSpec, FaceKind, FaceRef,
        GridRotation, InputSeatLinkSpec, InputSpec, PartSpec, RigidLinkSpec,
        SeatControllerLinkSpec, SeatSpec, ServoSpec, WeldSpec,
    };

    fn cuboid(dimensions: [u8; 3], units: IVec3) -> CuboidSpec {
        CuboidSpec::new(dimensions, BuildPose::new(units, GridRotation::default()))
            .expect("test dimensions are in range")
    }

    fn spawned(outcome: BuildOutcome) -> crate::PartId {
        match outcome {
            BuildOutcome::Spawned(part) => part,
            other => panic!("expected a spawn outcome, got {other:?}"),
        }
    }

    fn round_trip(document: &CreationDocument) -> CreationDocument {
        let text = ron::ser::to_string_pretty(document, ron::ser::PrettyConfig::default())
            .expect("a creation document serializes");
        ron::from_str(&text).expect("a serialized creation document parses")
    }

    /// A short tower welded to the ground, carrying a driven bearing, a
    /// control block, a hollow sliced cylinder, and one loose ring.
    fn sample() -> (ConstructionGraph, Vec<BearingSocket>) {
        let mut graph = ConstructionGraph::new();
        // A 1.0 x 0.5 x 1.0 m slab resting on the ground plane.
        let base = spawned(
            graph
                .apply(BuildCommand::Spawn(cuboid([4, 2, 4], IVec3::new(0, 1, 0))))
                .expect("the base spawns"),
        );
        // A smaller block sitting on the slab, held only by the bearing.
        let rotor = spawned(
            graph
                .apply(BuildCommand::Spawn(cuboid([2, 2, 2], IVec3::new(0, 3, 0))))
                .expect("the rotor spawns"),
        );
        // A quarter-turned control block flush on the slab's top face.
        let controller = spawned(
            graph
                .apply(BuildCommand::SpawnController(ControllerSpec::new(
                    BuildPose::from_half_grid(IVec3::new(2, 6, 0), GridRotation::new(0, 1, 0)),
                )))
                .expect("the control block spawns"),
        );
        // A detached hollow, sliced cylinder, joined to the base without contact.
        let column = spawned(
            graph
                .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
                    CylinderDimensions::new(1.0, 0.5, 0.75)
                        .expect("the cylinder dimensions are in range")
                        .with_sweep_angle_degrees(255)
                        .expect("255 degrees is a supported sweep"),
                    BuildPose::new(IVec3::new(-8, 4, 0), GridRotation::default()),
                )))
                .expect("the cylinder spawns"),
        );
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(base, FaceKind::NegativeY),
                second: FaceRef::ground(),
            }))
            .expect("the base welds to the ground");
        graph
            .apply(BuildCommand::RigidLink(RigidLinkSpec {
                first: base,
                second: column,
            }))
            .expect("the column joins the base rigidly");
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(base, FaceKind::PositiveY),
                second: FaceRef::part(controller, FaceKind::NegativeY),
            }))
            .expect("the control block welds to the base");
        let bearing = match graph
            .apply(BuildCommand::AddBearing(
                BearingSpec::new(
                    FaceRef::part(base, FaceKind::PositiveY),
                    FaceRef::part(rotor, FaceKind::NegativeY),
                    Vec3::new(0.0, 0.5, 0.0),
                    Vec3::Y,
                )
                .with_dimensions(
                    BearingDimensions::new(0.5, 0.2).expect("the ring dimensions are in range"),
                ),
            ))
            .expect("the bearing is added")
        {
            BuildOutcome::BearingAdded(bearing) => bearing,
            other => panic!("expected a bearing outcome, got {other:?}"),
        };

        let program = DriveProgram::new(
            &[
                DriveState::new(DriveTarget::Angle(0.0)).expect("zero degrees is in range"),
                DriveState::new(DriveTarget::Speed(2.5))
                    .expect("2.5 rad/s is in range")
                    .with_dwell(Some(
                        DriveDwell::new(1.5, Some(0)).expect("1.5 s is in range"),
                    ))
                    .with_trigger(Some(DriveTrigger::new(
                        DriveKey::new('w').expect("W is bindable"),
                        DriveRelease::RevertTo(0),
                    ))),
            ],
            true,
        )
        .expect("the program is valid");
        graph
            .apply(BuildCommand::AddDriveLink(DriveLinkSpec {
                controller,
                bearing,
                reversed: true,
                actuator: ActuatorAssignment::Unpowered,
                limits: DriveLimits::new(4.0, f32::INFINITY, Some((-1.0, 1.0)))
                    .expect("the limits are in range"),
                program,
                name: DriveName::new("Tipper arm"),
            }))
            .expect("the wire is added");

        // A ring placed on the rotor's top face with nothing attached through it.
        let sockets = vec![BearingSocket {
            source: FaceRef::part(rotor, FaceKind::PositiveY),
            anchor: Vec3::new(0.0, 1.0, 0.0),
            dimensions: BearingDimensions::new(0.3, 0.05)
                .expect("the ring dimensions are in range"),
        }];
        (graph, sockets)
    }

    #[test]
    fn sample_creation_survives_a_serialized_round_trip() {
        let (graph, sockets) = sample();
        let document = CreationDocument::from_graph(&graph, "Test Rig", &sockets);
        let restored = round_trip(&document)
            .into_graph()
            .expect("the document rebuilds");

        assert_eq!(restored.name, "Test Rig");
        assert_eq!(restored.sockets.len(), 1);
        assert_eq!(restored.graph.part_count(), graph.part_count());
        assert_eq!(restored.graph.weld_count(), graph.weld_count());
        assert_eq!(restored.graph.rigid_link_count(), graph.rigid_link_count());
        assert_eq!(restored.graph.bearing_count(), graph.bearing_count());
        assert_eq!(restored.graph.drive_link_count(), graph.drive_link_count());
        assert_eq!(
            CreationDocument::from_graph(&restored.graph, "Test Rig", &restored.sockets),
            document,
            "a second capture of the rebuilt graph must be byte-identical"
        );
    }

    #[test]
    fn engine_kinds_survive_a_serialized_round_trip() {
        let mut graph = ConstructionGraph::new();
        for (kind, x) in [(EngineKind::Gas, -2), (EngineKind::Electric, 2)] {
            graph
                .apply(BuildCommand::SpawnEngine(EngineSpec::new(
                    kind,
                    BuildPose::new(IVec3::new(x, 2, 0), GridRotation::default()),
                )))
                .expect("the engine spawns");
        }

        let document = CreationDocument::from_graph(&graph, "Engines", &[]);
        assert_eq!(document.version, CREATION_FORMAT_VERSION);
        let restored = round_trip(&document)
            .into_graph()
            .expect("the engine document rebuilds");
        let kinds = restored
            .graph
            .parts()
            .map(|(_, spec)| match spec {
                PartSpec::Engine(engine) => engine.kind,
                _ => panic!("engine document rebuilt a different part kind"),
            })
            .collect::<Vec<_>>();
        assert_eq!(kinds, [EngineKind::Gas, EngineKind::Electric]);
    }

    #[test]
    fn servo_seat_input_and_their_routes_survive_a_round_trip() {
        let mut graph = ConstructionGraph::new();
        let servo = spawned(
            graph
                .apply(BuildCommand::SpawnServo(ServoSpec::new(
                    BuildPose::default(),
                )))
                .unwrap(),
        );
        let input = spawned(
            graph
                .apply(BuildCommand::SpawnInput(InputSpec::new(
                    BuildPose::default(),
                )))
                .unwrap(),
        );
        let seat = spawned(
            graph
                .apply(BuildCommand::SpawnSeat(SeatSpec::new(BuildPose::default())))
                .unwrap(),
        );
        let controller = spawned(
            graph
                .apply(BuildCommand::SpawnController(ControllerSpec::new(
                    BuildPose::default(),
                )))
                .unwrap(),
        );
        graph
            .apply(BuildCommand::AddInputSeatLink(InputSeatLinkSpec {
                input,
                seat,
            }))
            .unwrap();
        graph
            .apply(BuildCommand::AddSeatControllerLink(
                SeatControllerLinkSpec { seat, controller },
            ))
            .unwrap();

        let restored = round_trip(&CreationDocument::from_graph(&graph, "Controls", &[]))
            .into_graph()
            .unwrap()
            .graph;
        assert_eq!(
            restored
                .parts()
                .filter(|(_, part)| matches!(part, PartSpec::Servo(_)))
                .count(),
            1
        );
        assert_eq!(restored.input_seat_links().count(), 1);
        assert_eq!(restored.seat_controller_links().count(), 1);
        assert!(
            restored.part(servo).is_some(),
            "canonical replay keeps part ids"
        );
    }

    #[test]
    fn version_two_drive_links_without_assignments_load_unpowered() {
        let (graph, sockets) = sample();
        let mut document = CreationDocument::from_graph(&graph, "Old Drives", &sockets);
        document.version = 2;
        let text =
            ron::ser::to_string_pretty(&document, ron::ser::PrettyConfig::default()).unwrap();
        let legacy = text.replace("actuator: Unpowered,", "");
        assert_ne!(legacy, text, "the fixture removed the version-three field");

        let restored: CreationDocument = ron::from_str(&legacy).unwrap();
        assert!(
            restored
                .drive_links
                .iter()
                .all(|link| link.actuator == ActuatorAssignment::Unpowered)
        );
        restored.into_graph().unwrap();
    }

    #[test]
    fn version_one_creation_remains_readable() {
        let (graph, sockets) = sample();
        let mut document = CreationDocument::from_graph(&graph, "Old Rig", &sockets);
        document.version = 1;

        let restored = round_trip(&document)
            .into_graph()
            .expect("version-one documents migrate on read");
        assert_eq!(restored.name, "Old Rig");
        assert_eq!(restored.graph.part_count(), graph.part_count());
    }

    #[test]
    fn rebuilt_creation_compiles_to_the_same_bodies() {
        let (graph, sockets) = sample();
        let expected = graph.compile().expect("the sample compiles");
        let restored = CreationDocument::from_graph(&graph, "Test Rig", &sockets)
            .into_graph()
            .expect("the document rebuilds");
        let actual = restored.graph.compile().expect("the rebuild compiles");

        assert_eq!(actual.compounds.len(), expected.compounds.len());
        assert_eq!(actual.bearings.len(), expected.bearings.len());
        assert_eq!(
            actual.coordinate_drives.len(),
            expected.coordinate_drives.len()
        );
    }

    #[test]
    fn unlimited_torque_round_trips_without_encoding_an_infinity() {
        let (graph, _) = sample();
        let document = CreationDocument::from_graph(&graph, "Test Rig", &[]);
        let wire = &document.drive_links[0];
        assert_eq!(wire.limits.max_torque_newton_meters, None);

        let text = ron::ser::to_string(&document).expect("the document serializes");
        assert!(
            !text.contains("inf"),
            "an unlimited torque must not encode as a float infinity: {text}"
        );

        let restored = round_trip(&document)
            .into_graph()
            .expect("the document rebuilds");
        let (_, link) = restored
            .graph
            .drive_links()
            .next()
            .expect("the rebuilt graph keeps its wire");
        assert!(link.limits.max_torque_newton_meters().is_infinite());
        assert_eq!(link.limits.angle_limits(), Some((-1.0, 1.0)));
        assert_eq!(link.name.as_str(), "Tipper arm");
        assert!(link.reversed);
        assert!(link.program.loops());
        assert_eq!(link.program.len(), 2);
        let triggered = link.program.state(1).expect("the second state exists");
        assert_eq!(
            triggered.trigger().map(|trigger| trigger.key().symbol()),
            Some('W')
        );
        assert_eq!(
            triggered.trigger().map(DriveTrigger::release),
            Some(DriveRelease::RevertTo(0))
        );
        assert_eq!(
            triggered
                .dwell()
                .map(|dwell| (dwell.seconds(), dwell.next())),
            Some((1.5, Some(0)))
        );
    }

    #[test]
    fn hollow_sliced_cylinder_keeps_its_bore_and_sweep() {
        let (graph, _) = sample();
        let restored = CreationDocument::from_graph(&graph, "Test Rig", &[])
            .into_graph()
            .expect("the document rebuilds");
        let dimensions = restored
            .graph
            .parts()
            .find_map(|(_, spec)| spec.as_cylinder())
            .expect("the rebuilt graph keeps its cylinder")
            .dimensions;

        assert!((dimensions.outer_diameter() - 1.0).abs() < 1.0e-6);
        assert!((dimensions.inner_diameter() - 0.5).abs() < 1.0e-6);
        assert_eq!(dimensions.axial_length_units(), 3);
        assert_eq!(dimensions.sweep_angle_degrees(), 255);
    }

    #[test]
    fn odd_sized_cuboid_keeps_its_half_grid_offset() {
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [3, 1, 3],
                    BuildPose::from_half_grid(IVec3::new(1, 3, 1), GridRotation::new(1, 2, 3)),
                )
                .expect("the dimensions are in range"),
            ))
            .expect("the cuboid spawns");
        let restored = CreationDocument::from_graph(&graph, "Offset", &[])
            .into_graph()
            .expect("the document rebuilds");
        let (_, original) = graph.parts().next().expect("the graph holds its cuboid");
        let (_, rebuilt) = restored
            .graph
            .parts()
            .next()
            .expect("the rebuild holds its cuboid");

        assert_eq!(rebuilt.pose(), original.pose());
    }

    #[test]
    fn unsupported_version_is_refused() {
        let (graph, _) = sample();
        let mut document = CreationDocument::from_graph(&graph, "Test Rig", &[]);
        document.version = CREATION_FORMAT_VERSION + 1;

        assert_eq!(
            document.into_graph().err(),
            Some(CreationError::UnsupportedVersion(
                CREATION_FORMAT_VERSION + 1
            ))
        );
    }

    #[test]
    fn weld_to_a_missing_part_is_refused() {
        let (graph, _) = sample();
        let mut document = CreationDocument::from_graph(&graph, "Test Rig", &[]);
        document.welds[0].first.owner = FaceOwnerDoc::Part(99);

        assert_eq!(
            document.into_graph().err(),
            Some(CreationError::MissingPart(99))
        );
    }

    #[test]
    fn wire_to_a_missing_bearing_is_refused() {
        let (graph, _) = sample();
        let mut document = CreationDocument::from_graph(&graph, "Test Rig", &[]);
        document.drive_links[0].bearing = 7;

        assert_eq!(
            document.into_graph().err(),
            Some(CreationError::MissingBearing(7))
        );
    }

    #[test]
    fn out_of_range_cuboid_dimension_is_refused() {
        let (graph, _) = sample();
        let mut document = CreationDocument::from_graph(&graph, "Test Rig", &[]);
        document.parts[0] = super::PartDoc::Cuboid {
            dimensions: [0, 1, 1],
            pose: super::PoseDoc {
                translation_half_units: [0, 0, 0],
                rotation: [0, 0, 0],
            },
        };

        assert!(matches!(
            document.into_graph(),
            Err(CreationError::Dimension(_))
        ));
    }
}

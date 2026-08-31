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

use bevy_math::{IVec3, Vec3};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    ActuatorAssignment, BearingDimensionError, BearingDimensions, BearingSpec, BuildCommand,
    BuildOutcome, BuildPose, CageIndex, ConstructionGraph, ConstructionMaterial, ControllerSpec,
    CuboidSpec, CylinderDimensionError, CylinderDimensions, CylinderSpec, DimensionError,
    DimensionLinkId, DimensionLinkSpec, DriveDwell, DriveKey, DriveLimits, DriveLimitsError,
    DriveLinkSpec, DriveName, DriveProgram, DriveProgramError, DriveRelease, DriveState,
    DriveTarget, DriveTrigger, EngineKind, EngineSpec, FaceKind, FaceOwner, FaceRef, GearKeyChord,
    GraphError, GridDimension, GridRotation, InputSeatLinkSpec, InputSpec, MaterialAppearance,
    PartId, PartSpec, PipeBendDimensionError, PipeBendDimensions, PipeBendSpec, RigidLinkSpec,
    SeatControllerLinkSpec, SeatSpec, ServoSpec, ShapeRegion, ShiftMode, TransmissionSpec,
    WeldSpec,
};

/// Format version written by this build. Files carrying anything else are
/// refused rather than guessed at.
pub const CREATION_FORMAT_VERSION: u32 = 12;
const OLDEST_CREATION_FORMAT_VERSION: u32 = CREATION_FORMAT_VERSION;

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
    /// Combining documents exceeded the on-disk 32-bit row index space.
    #[error("creation has too many rows to combine")]
    TooManyRows,
    /// A drive state was bound to something that is not a letter or a digit.
    #[error("drive state key {0:?} is not a letter or a digit")]
    InvalidDriveKey(char),
    /// A cuboid dimension was out of range.
    #[error(transparent)]
    Dimension(#[from] DimensionError),
    /// A cylinder dimension was out of range.
    #[error(transparent)]
    CylinderDimension(#[from] CylinderDimensionError),
    /// A pipe-bend dimension was out of range.
    #[error(transparent)]
    PipeBendDimension(#[from] PipeBendDimensionError),
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
/// Translation uses exact 2.5 mm ticks. Older coordinate encodings are rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoseDoc {
    /// Centre in exact integer 2.5 mm ticks.
    pub translation_ticks: [i32; 3],
    /// Quarter turns around local x, y, and z.
    pub rotation: [u8; 3],
}

impl From<BuildPose> for PoseDoc {
    fn from(pose: BuildPose) -> Self {
        let translation = pose.translation_position_ticks();
        Self {
            translation_ticks: [translation.x, translation.y, translation.z],
            rotation: pose.rotation.quarter_turns_xyz(),
        }
    }
}

impl From<PoseDoc> for BuildPose {
    fn from(doc: PoseDoc) -> Self {
        let [x, y, z] = doc.rotation;
        Self::from_position_ticks(doc.translation_ticks.into(), GridRotation::new(x, y, z))
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
        /// Physical material.
        material: ConstructionMaterial,
        /// Color and finish treatment.
        appearance: MaterialAppearance,
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
        /// Physical material.
        material: ConstructionMaterial,
        /// Color and finish treatment.
        appearance: MaterialAppearance,
    },
    /// Cardinal 90-degree quarter-torus pipe bend.
    PipeBend {
        /// Outer diameter in metres.
        outer_diameter: f32,
        /// Inner diameter in metres. Zero is solid.
        inner_diameter: f32,
        /// Centreline radius in quarter-metre grid units.
        radius_units: u8,
        /// Sharp-corner position and cardinal orientation.
        pose: PoseDoc,
        /// Physical material.
        material: ConstructionMaterial,
        /// Color and finish treatment.
        appearance: MaterialAppearance,
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
    /// Fixed-size transmission with a graph-owned upstream relation.
    Transmission {
        /// Index of the engine or transmission whose positive-Z output it extends.
        parent: u32,
        /// Centre and inherited root-engine orientation.
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
    /// Fixed-size Dimension Link portal anchor.
    DimensionLink {
        /// Stable identity within its owning world and Garage.
        id: DimensionLinkId,
        /// Centre and orientation.
        pose: PoseDoc,
    },
}

/// One shape region in its serialized form.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RegionDoc {
    /// Minimum corner in shape steps (1.25 cm).
    pub origin_steps: [i32; 3],
    /// Extent in construction cells.
    pub size_cells: [i32; 3],
    /// Material every block in the region shares.
    pub material: ConstructionMaterial,
    /// Color and finish shared by every region member.
    pub appearance: MaterialAppearance,
    /// Cage planes beyond the two the extent implies, in cells, per axis.
    #[serde(default)]
    pub divisions: [Vec<i32>; 3],
    /// Displaced cage vertices.
    #[serde(default)]
    pub vertices: Vec<(CageIndex, [i16; 3])>,
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

/// Persistent gearbox settings for one Controller and engine family.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GearboxConfigDoc {
    /// Index of the Controller part owning this lane.
    pub controller: u32,
    /// Engine family being configured.
    pub kind: EngineKind,
    /// Automatic or manual shifting.
    pub mode: ShiftMode,
    /// Strictly descending input-to-output ratios.
    pub ratios: Vec<f32>,
    /// Number of leading gas ratios assigned to reverse.
    pub reverse_gears: u8,
    /// Manual upshift chord.
    pub gear_up: GearKeyChord,
    /// Manual downshift chord.
    pub gear_down: GearKeyChord,
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
    /// Per-controller, per-engine-family gearbox settings.
    #[serde(default)]
    pub gearbox_configs: Vec<GearboxConfigDoc>,
    /// Editable shape regions. Absent in files written before regions existed.
    #[serde(default)]
    pub regions: Vec<RegionDoc>,
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
    /// Reassigns every Dimension Link identity for insertion as a reusable creation.
    ///
    /// Transfers between a world and its Garage must not call this: those retain IDs.
    pub fn remap_dimension_links(&mut self, next_id: &mut u64) {
        for part in &mut self.parts {
            if let PartDoc::DimensionLink { id, .. } = part {
                *id = DimensionLinkId(*next_id);
                *next_id = next_id.saturating_add(1);
            }
        }
    }

    /// Appends another complete construction document, remapping all dense row references.
    ///
    /// # Errors
    ///
    /// Returns [`CreationError::TooManyRows`] if a remapped row index exceeds `u32`.
    pub fn append(&mut self, mut other: Self) -> Result<(), CreationError> {
        let part_offset =
            u32::try_from(self.parts.len()).map_err(|_| CreationError::TooManyRows)?;
        let bearing_offset =
            u32::try_from(self.bearings.len()).map_err(|_| CreationError::TooManyRows)?;
        let add_part = |index: &mut u32| -> Result<(), CreationError> {
            *index = index
                .checked_add(part_offset)
                .ok_or(CreationError::TooManyRows)?;
            Ok(())
        };
        let add_face = |face: &mut FaceRefDoc| -> Result<(), CreationError> {
            if let FaceOwnerDoc::Part(index) = &mut face.owner {
                add_part(index)?;
            }
            Ok(())
        };
        for part in &mut other.parts {
            if let PartDoc::Transmission { parent, .. } = part {
                add_part(parent)?;
            }
        }
        for weld in &mut other.welds {
            add_face(&mut weld.first)?;
            add_face(&mut weld.second)?;
        }
        for link in &mut other.rigid_links {
            add_part(&mut link.first)?;
            add_part(&mut link.second)?;
        }
        for bearing in &mut other.bearings {
            add_face(&mut bearing.source)?;
            add_face(&mut bearing.target)?;
        }
        for link in &mut other.drive_links {
            add_part(&mut link.controller)?;
            link.bearing = link
                .bearing
                .checked_add(bearing_offset)
                .ok_or(CreationError::TooManyRows)?;
        }
        for link in &mut other.input_seat_links {
            add_part(&mut link.input)?;
            add_part(&mut link.seat)?;
        }
        for link in &mut other.seat_controller_links {
            add_part(&mut link.seat)?;
            add_part(&mut link.controller)?;
        }
        for config in &mut other.gearbox_configs {
            add_part(&mut config.controller)?;
        }
        for socket in &mut other.sockets {
            add_face(&mut socket.source)?;
        }
        self.parts.append(&mut other.parts);
        self.welds.append(&mut other.welds);
        self.rigid_links.append(&mut other.rigid_links);
        self.bearings.append(&mut other.bearings);
        self.drive_links.append(&mut other.drive_links);
        self.input_seat_links.append(&mut other.input_seat_links);
        self.seat_controller_links
            .append(&mut other.seat_controller_links);
        self.gearbox_configs.append(&mut other.gearbox_configs);
        self.regions.append(&mut other.regions);
        self.sockets.append(&mut other.sockets);
        Ok(())
    }

    /// Rotates the authored construction around the world origin and then translates it.
    ///
    /// Translation uses half-grid units so shape-region cage data remains exact.
    pub fn transform_cardinal(&mut self, yaw_quarter_turns: u8, translation_half_units: IVec3) {
        let yaw = yaw_quarter_turns % 4;
        for part in &mut self.parts {
            let pose = match part {
                PartDoc::Cuboid { pose, .. }
                | PartDoc::Cylinder { pose, .. }
                | PartDoc::PipeBend { pose, .. }
                | PartDoc::Controller { pose }
                | PartDoc::Engine { pose, .. }
                | PartDoc::Transmission { pose, .. }
                | PartDoc::Servo { pose }
                | PartDoc::Seat { pose }
                | PartDoc::Input { pose }
                | PartDoc::DimensionLink { pose, .. } => pose,
            };
            let rotated = rotate_y_i32(IVec3::from_array(pose.translation_ticks), yaw)
                + translation_half_units * crate::POSITION_TICKS_PER_HALF_GRID_UNIT;
            pose.translation_ticks = rotated.to_array();
            let [x, y, z] = pose.rotation;
            pose.rotation = GridRotation::new(x, y, z)
                .rotated_y(yaw)
                .quarter_turns_xyz();
        }
        let translation = translation_half_units.as_vec3() * (crate::GRID_UNIT_METERS * 0.5);
        for bearing in &mut self.bearings {
            bearing.anchor =
                (rotate_y_vec3(Vec3::from_array(bearing.anchor), yaw) + translation).to_array();
            bearing.axis = rotate_y_vec3(Vec3::from_array(bearing.axis), yaw).to_array();
        }
        for socket in &mut self.sockets {
            socket.anchor =
                (rotate_y_vec3(Vec3::from_array(socket.anchor), yaw) + translation).to_array();
        }
        for region in &mut self.regions {
            transform_region_doc(region, yaw, translation_half_units);
        }
    }

    /// Captures a construction and its unattached bearing rings.
    ///
    /// Any pending two-step operation on `graph` is ignored: it is transient
    /// editor state, not part of the creation.
    ///
    /// # Panics
    ///
    /// Never in practice: the arenas already refuse to exceed `u32` indices.
    #[allow(clippy::too_many_lines)] // The document snapshot keeps all index remapping together.
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
            parts: graph
                .parts()
                .map(|(id, spec)| part_doc(*spec, graph.transmission_parent(id).map(&part)))
                .collect(),
            regions: graph
                .regions()
                .map(|(_, region)| RegionDoc {
                    origin_steps: region.origin_steps().to_array(),
                    size_cells: region.size_cells().to_array(),
                    material: region.material(),
                    appearance: region.appearance(),
                    divisions: core::array::from_fn(|axis| {
                        // The first and last planes are implied by the extent.
                        let grid = region.grid();
                        let planes = grid.planes(axis);
                        let origin = planes[0];
                        planes[1..planes.len() - 1]
                            .iter()
                            .map(|half_units| (half_units - origin) / 2)
                            .collect()
                    }),
                    vertices: region.offsets().collect(),
                })
                .collect(),
            welds: graph
                .welds()
                .filter(|(id, _)| !graph.transmission_welds.values().any(|weld| weld == id))
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
            gearbox_configs: graph
                .gearbox_configs()
                .filter_map(|((controller, kind), _)| {
                    let config = graph.gearbox_config(controller, kind).ok()?;
                    Some(GearboxConfigDoc {
                        controller: part(controller),
                        kind,
                        mode: config.mode(),
                        ratios: config.ratios().to_vec(),
                        reverse_gears: config.reverse_gears(),
                        gear_up: config.gear_up(),
                        gear_down: config.gear_down(),
                    })
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
        let mut part_ids = Vec::with_capacity(self.parts.len());
        for (index, part) in self.parts.iter().copied().enumerate() {
            let command = match part {
                PartDoc::Transmission { parent, pose } => {
                    let parent = part_ids
                        .get(parent as usize)
                        .copied()
                        .ok_or(CreationError::MissingPart(parent))?;
                    BuildCommand::AttachTransmission {
                        parent,
                        spec: TransmissionSpec::new(pose.into()),
                    }
                }
                other => build_command(other)?,
            };
            let BuildOutcome::Spawned(id) = graph.apply(command)? else {
                unreachable!("part {index} replay uses a spawn command")
            };
            part_ids.push(id);
        }

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
        for document in &self.regions {
            let region = ShapeRegion::from_origin_steps(
                IVec3::from_array(document.origin_steps),
                IVec3::from_array(document.size_cells),
                document.material,
            )
            .map_err(GraphError::from)?
            .with_appearance(document.appearance);
            let BuildOutcome::RegionAdded(id) = graph.apply(BuildCommand::AddRegion(region))?
            else {
                unreachable!("adding a region reports the region it added")
            };
            for (axis, positions) in document.divisions.iter().enumerate() {
                for &position in positions {
                    graph.apply(BuildCommand::SubdivideRegion {
                        region: id,
                        axis,
                        position,
                    })?;
                }
            }
            if !document.vertices.is_empty() {
                graph.apply(BuildCommand::SetRegionVertices {
                    region: id,
                    vertices: document.vertices.clone(),
                })?;
            }
        }

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

        for gearbox in &self.gearbox_configs {
            let controller = resolve_part(gearbox.controller, &part_ids)?;
            graph.apply_batch([
                BuildCommand::SetGearboxMode {
                    controller,
                    kind: gearbox.kind,
                    mode: gearbox.mode,
                },
                BuildCommand::SetGearboxRatios {
                    controller,
                    kind: gearbox.kind,
                    ratios: gearbox.ratios.clone(),
                },
                BuildCommand::SetGearboxBindings {
                    controller,
                    kind: gearbox.kind,
                    up: gearbox.gear_up,
                    down: gearbox.gear_down,
                },
            ])?;
            if gearbox.kind == EngineKind::Gas {
                graph.apply(BuildCommand::SetGasDivider {
                    controller,
                    reverse_gears: gearbox.reverse_gears,
                })?;
            }
        }

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

const fn rotate_y_i32(position: IVec3, yaw: u8) -> IVec3 {
    match yaw % 4 {
        0 => position,
        1 => IVec3::new(position.z, position.y, -position.x),
        2 => IVec3::new(-position.x, position.y, -position.z),
        _ => IVec3::new(-position.z, position.y, position.x),
    }
}

fn rotate_y_vec3(position: Vec3, yaw: u8) -> Vec3 {
    match yaw % 4 {
        0 => position,
        1 => Vec3::new(position.z, position.y, -position.x),
        2 => Vec3::new(-position.x, position.y, -position.z),
        _ => Vec3::new(-position.z, position.y, position.x),
    }
}

fn transform_region_doc(region: &mut RegionDoc, yaw: u8, translation: IVec3) {
    let origin = IVec3::from_array(region.origin_steps);
    let size = IVec3::from_array(region.size_cells);
    let maximum = origin + size * crate::STEPS_PER_CELL;
    let corners = [
        IVec3::new(origin.x, origin.y, origin.z),
        IVec3::new(maximum.x, origin.y, origin.z),
        IVec3::new(origin.x, maximum.y, origin.z),
        IVec3::new(origin.x, origin.y, maximum.z),
        IVec3::new(maximum.x, maximum.y, maximum.z),
    ]
    .map(|corner| rotate_y_i32(corner, yaw) + translation * crate::STEPS_PER_HALF_UNIT);
    let minimum = corners
        .iter()
        .copied()
        .reduce(IVec3::min)
        .expect("region has corners");
    let maximum = corners
        .iter()
        .copied()
        .reduce(IVec3::max)
        .expect("region has corners");
    let old_divisions = core::mem::take(&mut region.divisions);
    let old_vertices = core::mem::take(&mut region.vertices);
    let old_counts = old_divisions.each_ref().map(|axis| axis.len() + 2);
    region.origin_steps = minimum.to_array();
    region.size_cells = ((maximum - minimum) / crate::STEPS_PER_CELL).to_array();
    region.divisions = match yaw % 4 {
        0 => old_divisions,
        1 => [
            old_divisions[2].clone(),
            old_divisions[1].clone(),
            reflected_divisions(&old_divisions[0], size.x),
        ],
        2 => [
            reflected_divisions(&old_divisions[0], size.x),
            old_divisions[1].clone(),
            reflected_divisions(&old_divisions[2], size.z),
        ],
        _ => [
            reflected_divisions(&old_divisions[2], size.z),
            old_divisions[1].clone(),
            old_divisions[0].clone(),
        ],
    };
    region.vertices = old_vertices
        .into_iter()
        .map(|([i, j, k], [x, y, z])| match yaw % 4 {
            0 => ([i, j, k], [x, y, z]),
            1 => (
                [
                    k,
                    j,
                    u16::try_from(old_counts[0] - 1).unwrap_or(u16::MAX) - i,
                ],
                [z, y, -x],
            ),
            2 => (
                [
                    u16::try_from(old_counts[0] - 1).unwrap_or(u16::MAX) - i,
                    j,
                    u16::try_from(old_counts[2] - 1).unwrap_or(u16::MAX) - k,
                ],
                [-x, y, -z],
            ),
            _ => (
                [
                    u16::try_from(old_counts[2] - 1).unwrap_or(u16::MAX) - k,
                    j,
                    i,
                ],
                [-z, y, x],
            ),
        })
        .collect();
}

fn reflected_divisions(divisions: &[i32], size: i32) -> Vec<i32> {
    divisions
        .iter()
        .rev()
        .map(|position| size - position)
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

fn part_doc(spec: PartSpec, transmission_parent: Option<u32>) -> PartDoc {
    match spec {
        PartSpec::Cuboid(cuboid) => PartDoc::Cuboid {
            dimensions: cuboid.dimensions.map(GridDimension::units),
            pose: cuboid.pose.into(),
            material: cuboid.material,
            appearance: cuboid.appearance,
        },
        PartSpec::Cylinder(cylinder) => PartDoc::Cylinder {
            outer_diameter: cylinder.dimensions.outer_diameter(),
            inner_diameter: cylinder.dimensions.inner_diameter(),
            length_units: cylinder.dimensions.axial_length_units(),
            sweep_degrees: cylinder.dimensions.sweep_angle_degrees(),
            pose: cylinder.pose.into(),
            material: cylinder.material,
            appearance: cylinder.appearance,
        },
        PartSpec::PipeBend(bend) => PartDoc::PipeBend {
            outer_diameter: bend.dimensions.outer_diameter(),
            inner_diameter: bend.dimensions.inner_diameter(),
            radius_units: bend.dimensions.radius_units(),
            pose: bend.pose.into(),
            material: bend.material,
            appearance: bend.appearance,
        },
        PartSpec::Controller(controller) => PartDoc::Controller {
            pose: controller.pose.into(),
        },
        PartSpec::Engine(engine) => PartDoc::Engine {
            kind: engine.kind,
            pose: engine.pose.into(),
        },
        PartSpec::Transmission(transmission) => PartDoc::Transmission {
            parent: transmission_parent.expect("every transmission has a live graph parent"),
            pose: transmission.pose.into(),
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
        PartSpec::DimensionLink(link) => PartDoc::DimensionLink {
            id: link.id,
            pose: link.pose.into(),
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
        PartDoc::Cuboid {
            dimensions,
            pose,
            material,
            appearance,
        } => BuildCommand::Spawn(
            CuboidSpec::new(dimensions, pose.into())?
                .with_material(material)
                .with_appearance(appearance),
        ),
        PartDoc::Cylinder {
            outer_diameter,
            inner_diameter,
            length_units,
            sweep_degrees,
            pose,
            material,
            appearance,
        } => BuildCommand::SpawnCylinder(
            CylinderSpec::new(
                CylinderDimensions::new(
                    outer_diameter,
                    inner_diameter,
                    f32::from(length_units) * crate::GRID_UNIT_METERS,
                )?
                .with_sweep_angle_degrees(sweep_degrees)?,
                pose.into(),
            )
            .with_material(material)
            .with_appearance(appearance),
        ),
        PartDoc::PipeBend {
            outer_diameter,
            inner_diameter,
            radius_units,
            pose,
            material,
            appearance,
        } => BuildCommand::SpawnPipeBend(
            PipeBendSpec::new(
                PipeBendDimensions::new(
                    outer_diameter,
                    inner_diameter,
                    f32::from(radius_units) * crate::GRID_UNIT_METERS,
                )?,
                pose.into(),
            )
            .with_material(material)
            .with_appearance(appearance),
        ),
        PartDoc::Controller { pose } => {
            BuildCommand::SpawnController(ControllerSpec::new(pose.into()))
        }
        PartDoc::Engine { kind, pose } => {
            BuildCommand::SpawnEngine(EngineSpec::new(kind, pose.into()))
        }
        PartDoc::Transmission { .. } => {
            unreachable!("transmissions are replayed with their parent relation")
        }
        PartDoc::Servo { pose } => BuildCommand::SpawnServo(ServoSpec::new(pose.into())),
        PartDoc::Seat { pose } => BuildCommand::SpawnSeat(SeatSpec::new(pose.into())),
        PartDoc::Input { pose } => BuildCommand::SpawnInput(InputSpec::new(pose.into())),
        PartDoc::DimensionLink { id, pose } => {
            BuildCommand::SpawnDimensionLink(DimensionLinkSpec::new(id, pose.into()))
        }
    })
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
        PartDoc,
    };
    use crate::{
        ActuatorAssignment, BearingDimensions, BearingSpec, BuildCommand, BuildOutcome, BuildPose,
        ConstructionGraph, ConstructionMaterial, ControllerSpec, CuboidSpec, CylinderDimensions,
        CylinderSpec, DimensionLinkId, DimensionLinkSpec, DriveDwell, DriveKey, DriveLimits,
        DriveLinkSpec, DriveName, DriveProgram, DriveRelease, DriveState, DriveTarget,
        DriveTrigger, EngineKind, EngineSpec, FaceKind, FaceRef, GearKey, GearKeyChord,
        GridRotation, InputSeatLinkSpec, InputSpec, MaterialAppearance, MaterialColor, MaterialDye,
        MaterialFinish, PartSpec, PipeBendDimensions, PipeBendSpec, RigidLinkSpec,
        SeatControllerLinkSpec, SeatSpec, ServoSpec, ShapeRegion, ShiftMode, WeldSpec,
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
    fn reusable_creation_remaps_every_dimension_link_id() {
        let mut graph = ConstructionGraph::new();
        for id in [2, 9] {
            graph
                .apply(BuildCommand::SpawnDimensionLink(DimensionLinkSpec::new(
                    DimensionLinkId(id),
                    BuildPose::default(),
                )))
                .unwrap();
        }
        let mut document = CreationDocument::from_graph(&graph, "Links", &[]);
        let mut next_id = 40;
        document.remap_dimension_links(&mut next_id);
        let restored = document.into_graph().unwrap().graph;
        assert!(restored.dimension_link(DimensionLinkId(40)).is_some());
        assert!(restored.dimension_link(DimensionLinkId(41)).is_some());
        assert_eq!(next_id, 42);
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
    fn obsolete_creation_versions_are_rejected() {
        for version in 1..CREATION_FORMAT_VERSION {
            let document = CreationDocument {
                version,
                name: format!("Obsolete {version}"),
                ..CreationDocument::from_graph(&ConstructionGraph::new(), "obsolete", &[])
            };
            assert!(matches!(
                document.into_graph(),
                Err(CreationError::UnsupportedVersion(found)) if found == version
            ));
        }
    }

    #[test]
    fn current_version_round_trips_all_construction_materials() {
        let mut graph = ConstructionGraph::new();
        for (index, material) in ConstructionMaterial::ALL.into_iter().enumerate() {
            let position = IVec3::new(i32::try_from(index).unwrap() * 8, 0, 0);
            graph
                .apply(BuildCommand::Spawn(
                    cuboid([1, 1, 1], position).with_material(material),
                ))
                .unwrap();
        }
        let document = CreationDocument::from_graph(&graph, "Materials", &[]);
        assert_eq!(document.version, CREATION_FORMAT_VERSION);
        let restored = round_trip(&document).into_graph().unwrap();
        let materials = restored
            .graph
            .parts()
            .filter_map(|(_, spec)| spec.as_cuboid().map(|cuboid| cuboid.material))
            .collect::<Vec<_>>();
        assert_eq!(materials, ConstructionMaterial::ALL);
    }

    #[test]
    fn current_version_round_trips_construction_appearances() {
        let appearance = MaterialAppearance::new(
            MaterialColor::Dye(MaterialDye::new([224, 86, 31], 2.0).unwrap()),
            MaterialFinish::Anodised,
        );
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::Spawn(
                cuboid([1, 1, 1], IVec3::ZERO).with_appearance(appearance),
            ))
            .unwrap();
        let document = CreationDocument::from_graph(&graph, "Chroma", &[]);
        let encoded = ron::ser::to_string(&document).unwrap();
        assert!(encoded.contains("appearance"));
        let restored = ron::from_str::<CreationDocument>(&encoded)
            .unwrap()
            .into_graph()
            .unwrap();
        assert_eq!(
            restored.graph.parts().next().unwrap().1.appearance(),
            Some(appearance)
        );
    }

    #[test]
    fn construction_documents_without_appearance_are_rejected() {
        let missing = r"Cuboid(
            dimensions: [1, 1, 1],
            pose: (translation_ticks: [0, 0, 0], rotation: [0, 0, 0]),
            material: Steel,
        )";
        assert!(ron::from_str::<PartDoc>(missing).is_err());
    }

    #[test]
    fn legacy_carbon_material_deserializes_as_graphite_and_writes_graphite() {
        let parsed: ConstructionMaterial = ron::from_str("Carbon").unwrap();
        assert_eq!(parsed, ConstructionMaterial::Graphite);
        assert_eq!(ron::to_string(&parsed).unwrap(), "Graphite");
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
    fn current_version_round_trips_pipe_bends() {
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::SpawnPipeBend(
                PipeBendSpec::new(
                    PipeBendDimensions::new(0.75, 0.25, 1.0).unwrap(),
                    BuildPose::new(IVec3::new(4, 8, 12), GridRotation::new(1, 2, 3)),
                )
                .with_material(ConstructionMaterial::Aluminium),
            ))
            .unwrap();
        let document = CreationDocument::from_graph(&graph, "Bent Pipe", &[]);
        assert_eq!(document.version, CREATION_FORMAT_VERSION);
        let loaded = round_trip(&document).into_graph().unwrap();
        let bend = loaded
            .graph
            .parts()
            .find_map(|(_, part)| part.as_pipe_bend())
            .unwrap();
        assert_eq!(bend.material, ConstructionMaterial::Aluminium);
        assert!((bend.dimensions.radius() - 1.0).abs() < f32::EPSILON);
        assert!((bend.dimensions.inner_diameter() - 0.25).abs() < f32::EPSILON);
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
    fn current_document_round_trips_five_centimetre_fine_placement() {
        let mut graph = ConstructionGraph::new();
        graph
            .apply(BuildCommand::Spawn(
                CuboidSpec::new(
                    [1, 1, 1],
                    BuildPose::from_position_ticks(
                        IVec3::new(20, 50, -20),
                        GridRotation::default(),
                    ),
                )
                .unwrap(),
            ))
            .unwrap();
        let document = round_trip(&CreationDocument::from_graph(&graph, "Fine", &[]));
        let PartDoc::Cuboid { pose, .. } = document.parts[0] else {
            unreachable!()
        };
        assert_eq!(pose.translation_ticks, [20, 50, -20]);
        let rebuilt = document.into_graph().unwrap().graph;
        let (_, part) = rebuilt.parts().next().unwrap();
        assert_eq!(
            part.pose().translation_position_ticks(),
            IVec3::new(20, 50, -20)
        );
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
                translation_ticks: [0, 0, 0],
                rotation: [0, 0, 0],
            },
            material: crate::ConstructionMaterial::Steel,
            appearance: crate::MaterialAppearance::BAKED,
        };

        assert!(matches!(
            document.into_graph(),
            Err(CreationError::Dimension(_))
        ));
    }

    #[test]
    fn a_shaped_creation_round_trips_its_regions() {
        let spec = CuboidSpec::new(
            [1, 1, 1],
            BuildPose::from_half_grid(IVec3::ONE, GridRotation::default()),
        )
        .unwrap();
        let mut graph = ConstructionGraph::new();
        graph.apply(BuildCommand::Spawn(spec)).unwrap();
        let region =
            ShapeRegion::new(IVec3::ZERO, IVec3::ONE, ConstructionMaterial::Steel).unwrap();
        let BuildOutcome::RegionAdded(id) = graph.apply(BuildCommand::AddRegion(region)).unwrap()
        else {
            panic!("wrong outcome")
        };
        graph
            .apply(BuildCommand::SetRegionVertices {
                region: id,
                vertices: vec![([1, 1, 1], [-3, -4, -5])],
            })
            .unwrap();

        let document = CreationDocument::from_graph(&graph, "shaped", &[]);
        assert_eq!(document.regions.len(), 1);
        let restored = document.into_graph().unwrap().graph;
        let (_, original) = graph.regions().next().unwrap();
        let (_, replayed) = restored.regions().next().unwrap();
        assert_eq!(replayed, original);
    }

    #[test]
    fn a_fine_placed_shape_region_round_trips_its_exact_origin() {
        let spec = CuboidSpec::new(
            [1, 1, 1],
            BuildPose::from_position_ticks(IVec3::new(60, 50, 50), GridRotation::default()),
        )
        .unwrap();
        let mut graph = ConstructionGraph::new();
        graph.apply(BuildCommand::Spawn(spec)).unwrap();
        let region = ShapeRegion::from_origin_steps(
            IVec3::new(10, 0, 0),
            IVec3::ONE,
            ConstructionMaterial::Steel,
        )
        .unwrap();
        graph.apply(BuildCommand::AddRegion(region)).unwrap();

        let document = CreationDocument::from_graph(&graph, "fine shaped", &[]);
        assert_eq!(document.regions[0].origin_steps, [10, 0, 0]);
        let restored = document.into_graph().unwrap().graph;
        let (_, replayed) = restored.regions().next().unwrap();
        assert_eq!(replayed.origin_steps(), IVec3::new(10, 0, 0));
    }

    #[test]
    fn a_subdivided_region_round_trips_its_cage_planes() {
        let mut graph = ConstructionGraph::new();
        for x in 0..2 {
            let spec = CuboidSpec::new(
                [1, 1, 1],
                BuildPose::from_half_grid(IVec3::new(1 + x * 2, 1, 1), GridRotation::default()),
            )
            .unwrap();
            graph.apply(BuildCommand::Spawn(spec)).unwrap();
        }
        let parts = graph.parts().map(|(id, _)| id).collect::<Vec<_>>();
        graph
            .apply(BuildCommand::RigidLink(RigidLinkSpec {
                first: parts[0],
                second: parts[1],
            }))
            .unwrap();
        let region = ShapeRegion::new(
            IVec3::ZERO,
            IVec3::new(2, 1, 1),
            ConstructionMaterial::Steel,
        )
        .unwrap();
        let BuildOutcome::RegionAdded(id) = graph.apply(BuildCommand::AddRegion(region)).unwrap()
        else {
            panic!("wrong outcome")
        };
        graph
            .apply(BuildCommand::SubdivideRegion {
                region: id,
                axis: 0,
                position: 1,
            })
            .unwrap();

        let document = CreationDocument::from_graph(&graph, "subdivided", &[]);
        let restored = document.into_graph().unwrap().graph;
        let (_, replayed) = restored.regions().next().unwrap();
        assert_eq!(replayed.plane_counts(), [3, 2, 2]);
    }

    #[test]
    fn transmission_parents_and_gearbox_settings_round_trip() {
        let mut graph = ConstructionGraph::new();
        let controller = spawned(
            graph
                .apply(BuildCommand::SpawnController(ControllerSpec::new(
                    BuildPose::new(IVec3::ZERO, GridRotation::default()),
                )))
                .unwrap(),
        );
        let engine = spawned(
            graph
                .apply(BuildCommand::SpawnEngine(EngineSpec::new(
                    EngineKind::Gas,
                    BuildPose::new(IVec3::new(2, 0, 0), GridRotation::default()),
                )))
                .unwrap(),
        );
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(controller, FaceKind::PositiveX),
                second: FaceRef::part(engine, FaceKind::NegativeX),
            }))
            .unwrap();
        let spec = graph.next_transmission_spec(engine).unwrap();
        let transmission = spawned(
            graph
                .apply(BuildCommand::AttachTransmission {
                    parent: engine,
                    spec,
                })
                .unwrap(),
        );
        graph
            .apply(BuildCommand::SetGearboxMode {
                controller,
                kind: EngineKind::Gas,
                mode: ShiftMode::Manual,
            })
            .unwrap();
        graph
            .apply(BuildCommand::SetGearboxRatios {
                controller,
                kind: EngineKind::Gas,
                ratios: vec![4.0, 0.8],
            })
            .unwrap();
        graph
            .apply(BuildCommand::SetGearboxBindings {
                controller,
                kind: EngineKind::Gas,
                up: GearKeyChord::new(GearKey::PageUp),
                down: GearKeyChord {
                    shift: true,
                    ..GearKeyChord::new(GearKey::PageDown)
                },
            })
            .unwrap();
        graph
            .apply(BuildCommand::SetGasDivider {
                controller,
                reverse_gears: 2,
            })
            .unwrap();

        let document = CreationDocument::from_graph(&graph, "Geared", &[]);
        assert_eq!(document.gearbox_configs.len(), 1);
        assert_eq!(
            document.welds.len(),
            1,
            "the required weld is derived from its parent"
        );
        let restored = round_trip(&document).into_graph().unwrap().graph;
        let restored_transmission = restored
            .parts()
            .find_map(|(id, spec)| matches!(spec, PartSpec::Transmission(_)).then_some(id))
            .unwrap();
        let restored_engine = restored.transmission_parent(restored_transmission).unwrap();
        assert!(matches!(
            restored.part(restored_engine),
            Some(PartSpec::Engine(_))
        ));
        let restored_controller = restored
            .parts()
            .find_map(|(id, spec)| matches!(spec, PartSpec::Controller(_)).then_some(id))
            .unwrap();
        let config = restored
            .gearbox_config(restored_controller, EngineKind::Gas)
            .unwrap();
        assert_eq!(config.mode(), ShiftMode::Manual);
        assert_eq!(config.ratios(), &[4.0, 0.8]);
        assert_eq!(config.reverse_gears(), 2);
        assert_eq!(config.gear_up().key, GearKey::PageUp);
        assert!(config.gear_down().shift);
        assert!(graph.part(transmission).is_some());
    }

    #[test]
    fn incomplete_matching_stacks_remain_saveable_and_loadable() {
        let mut graph = ConstructionGraph::new();
        let controller = spawned(
            graph
                .apply(BuildCommand::SpawnController(ControllerSpec::new(
                    BuildPose::new(IVec3::ZERO, GridRotation::default()),
                )))
                .unwrap(),
        );
        let first = spawned(
            graph
                .apply(BuildCommand::SpawnEngine(EngineSpec::new(
                    EngineKind::Electric,
                    BuildPose::new(IVec3::new(2, 0, 0), GridRotation::default()),
                )))
                .unwrap(),
        );
        let second = spawned(
            graph
                .apply(BuildCommand::SpawnEngine(EngineSpec::new(
                    EngineKind::Electric,
                    BuildPose::new(IVec3::new(4, 0, 0), GridRotation::default()),
                )))
                .unwrap(),
        );
        for (left, right) in [(controller, first), (first, second)] {
            graph
                .apply(BuildCommand::Weld(WeldSpec {
                    first: FaceRef::part(left, FaceKind::PositiveX),
                    second: FaceRef::part(right, FaceKind::NegativeX),
                }))
                .unwrap();
        }
        let spec = graph.next_transmission_spec(first).unwrap();
        graph
            .apply(BuildCommand::AttachTransmission {
                parent: first,
                spec,
            })
            .unwrap();

        let document = CreationDocument::from_graph(&graph, "Incomplete", &[]);
        let restored = round_trip(&document).into_graph().unwrap().graph;
        assert!(matches!(
            restored.compile(),
            Err(crate::TopologyError::TransmissionDepthMismatch {
                kind: EngineKind::Electric,
                ..
            })
        ));
    }
}

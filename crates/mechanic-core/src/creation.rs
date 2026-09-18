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

use bevy_math::{IVec3, Quat, Vec3};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    ActuatorAssignment, BearingDimensionError, BearingDimensions, BearingSpec, BuildCommand,
    BuildOutcome, BuildPose, CageIndex, ConstructionGraph, ConstructionMaterial, ControllerSpec,
    CuboidSpec, CylinderDimensionError, CylinderDimensions, CylinderSpec, DimensionError,
    DimensionLinkId, DimensionLinkSpec, DriveDwell, DriveKey, DriveLimits, DriveLimitsError,
    DriveLinkSpec, DriveName, DriveProgram, DriveProgramError, DriveRelease, DriveState,
    DriveTarget, DriveTrigger, EdgeChainRef, EdgeTreatment, EngineKind, EngineSpec, FaceKind,
    FaceOwner, FaceRef, GearKeyChord, GraphError, GridDimension, GridRotation, InputSeatLinkSpec,
    InputSpec, MaterialAppearance, PartId, PartSpec, PipeArms, PipeBendDimensionError,
    PipeBendDimensions, PipeBendSpec, PipeJunctionDimensions, PipeJunctionError, PipeJunctionSpec,
    RigidLinkSpec, SeatControllerLinkSpec, SeatSpec, ServoSpec, ShapeFeature, ShapeFeatureId,
    ShapeRegion, ShiftMode, SolidOwner, TopologyKey, TopologySource, TransmissionSpec, WeldSpec,
};

/// Format version written by this build. Files carrying anything else are
/// refused rather than guessed at.
pub const CREATION_FORMAT_VERSION: u32 = 17;

/// A bearing ring placed on a face with nothing attached through it yet.
///
/// The graph cannot hold these: a bearing needs two endpoints. The editor owns
/// them, and a saved creation carries them alongside the graph so a half-built
/// machine reloads exactly as it was left.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BearingSocket {
    /// Physical motion variant and rail frame.
    pub kind: crate::BearingKind,
    /// World-space travel axis for a linear socket.
    pub axis: Vec3,
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
    /// Invalid saved linear socket frame.
    #[error(transparent)]
    LinearBearing(#[from] crate::LinearBearingError),
    /// The file was written by a different format version.
    #[error(
        "creation format version {0} is not supported; this build reads only version {CREATION_FORMAT_VERSION}"
    )]
    UnsupportedVersion(u32),
    /// Frame transform is not finite and rigid.
    #[error(transparent)]
    Frame(#[from] crate::FrameError),
    /// Membership must provide exactly one frame for each saved part.
    #[error("construction frame membership count does not match part count")]
    FrameMembershipCount,
    /// Membership must provide exactly one frame for each saved shape region.
    #[error("construction frame membership count does not match region count")]
    RegionFrameMembershipCount,
    /// A frame membership names a missing frame row.
    #[error("creation references construction frame {0}, which the file does not define")]
    MissingFrame(u32),
    /// Transmission attachments share their parent's authored grid.
    #[error("transmission part {0} must share its parent's construction frame")]
    TransmissionFrame(u32),
    /// A record referenced a part the file does not define.
    #[error("creation references part {0}, which the file does not define")]
    MissingPart(u32),
    /// A drive wire referenced a bearing the file does not define.
    #[error("creation references bearing {0}, which the file does not define")]
    MissingBearing(u32),
    /// A feature target referenced a Shape region the file does not define.
    #[error("creation references region {0}, which the file does not define")]
    MissingRegion(u32),
    /// A topology key referenced an earlier feature the file does not define.
    #[error("creation references shape feature {0}, which the file does not define")]
    MissingShapeFeature(u32),
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
    /// A saved material layer does not fit its part.
    #[error(transparent)]
    Layer(#[from] crate::LayerError),
    /// A pipe-bend dimension was out of range.
    #[error(transparent)]
    PipeBendDimension(#[from] PipeBendDimensionError),
    /// A pipe-junction cross-section or opening set was invalid.
    #[error(transparent)]
    PipeJunction(#[from] PipeJunctionError),
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

/// One material layer in its serialized form.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaterialLayerDoc {
    /// Surface the layer was laid on.
    pub face: crate::LayerFace,
    /// Thickness in metres.
    pub thickness: f32,
    /// Physical material.
    pub material: ConstructionMaterial,
    /// Color and finish treatment.
    pub appearance: MaterialAppearance,
}

/// One construction part in its serialized form.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PartDoc {
    /// Rectangular cuboid, sized in quarter-metre grid units.
    Cuboid {
        /// Integer x/y/z core side lengths.
        dimensions: [u8; 3],
        /// Core centre and orientation.
        pose: PoseDoc,
        /// Core physical material.
        material: ConstructionMaterial,
        /// Core color and finish treatment.
        appearance: MaterialAppearance,
        /// Material layers replayed over the core, oldest first.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        layers: Vec<MaterialLayerDoc>,
    },
    /// Solid or hollow cylinder whose axis is local Y.
    Cylinder {
        /// Core outer diameter in metres.
        outer_diameter: f32,
        /// Core inner diameter in metres. Zero is solid.
        inner_diameter: f32,
        /// Core axial length in quarter-metre grid units.
        length_units: u8,
        /// Retained angular sector in degrees.
        sweep_degrees: u16,
        /// Core centre and orientation.
        pose: PoseDoc,
        /// Core physical material.
        material: ConstructionMaterial,
        /// Core color and finish treatment.
        appearance: MaterialAppearance,
        /// Material layers replayed over the core, oldest first.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        layers: Vec<MaterialLayerDoc>,
    },
    /// Cardinal 90-degree quarter-torus pipe bend.
    PipeBend {
        /// Outer diameter in metres.
        outer_diameter: f32,
        /// Inner diameter in metres. Zero is solid.
        inner_diameter: f32,
        /// Side of the bend's square footprint, in blocks.
        span_blocks: u8,
        /// Sharp-corner position and cardinal orientation.
        pose: PoseDoc,
        /// Physical material.
        material: ConstructionMaterial,
        /// Color and finish treatment.
        appearance: MaterialAppearance,
    },
    /// Cube fitting joining pipe ends on any of its faces.
    PipeJunction {
        /// Outer diameter of each pipe end in metres.
        outer_diameter: f32,
        /// Bore diameter in metres. Zero is solid.
        inner_diameter: f32,
        /// Open faces as bits ordered +X, −X, +Y, −Y, +Z, −Z.
        arms: u8,
        /// Cube centre and cardinal orientation.
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
    /// Stable evaluated surface patch, when this connection uses trimmed or
    /// feature-generated geometry.
    #[serde(default)]
    pub patch: Option<TopologyKeyDoc>,
}

/// Serialized owner of a parametric feature target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolidOwnerDoc {
    /// Index into the document's part list.
    Part(u32),
    /// Index into the document's Shape-region list.
    Region(u32),
}

/// Serialized provenance of a stable topology key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologySourceDoc {
    /// Base-generator topology.
    Base,
    /// Index of an earlier record in the ordered feature list.
    Feature(u32),
}

/// Serialized stable logical-curve key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyKeyDoc {
    /// Base or generating-feature provenance.
    pub source: TopologySourceDoc,
    /// Deterministic identity within that provenance.
    pub local: u32,
}

/// One serialized feature target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeChainRefDoc {
    /// Part or region carrying the chain.
    pub owner: SolidOwnerDoc,
    /// Stable logical edge key.
    pub edge: TopologyKeyDoc,
}

/// One record in explicit global feature order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShapeFeatureDoc {
    /// Logical chains treated together.
    pub targets: Vec<EdgeChainRefDoc>,
    /// Chamfer or fillet.
    pub treatment: EdgeTreatment,
    /// Equal setback or radius in exact position ticks.
    pub amount_ticks: u32,
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
    /// Motion kind and complete rail configuration.
    pub kind: crate::BearingKind,
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
    /// Physical motion variant and rail frame.
    pub kind: crate::BearingKind,
    /// World-space travel axis for a linear socket.
    pub axis: [f32; 3],
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
    /// Typed linear limits in SI units.
    pub linear_limits: Option<crate::LinearDriveLimits>,
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

/// Rigid transform of an authored local construction grid.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstructionFrameDoc {
    /// Translation from the local grid into build space, in metres.
    pub translation: [f32; 3],
    /// Unit quaternion in x/y/z/w order.
    pub rotation: [f32; 4],
}

/// Serializable authored construction and its dense relationship rows.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CreationDocument {
    /// Format version. See [`CREATION_FORMAT_VERSION`].
    pub version: u32,
    /// Display name, kept in the file so renaming a file does not rename the
    /// creation and vice versa.
    pub name: String,
    /// Parts, in the order every other record indexes them by.
    pub parts: Vec<PartDoc>,
    /// Rigid authored grids, indexed by `part_frames`.
    pub frames: Vec<ConstructionFrameDoc>,
    /// Exactly one dense frame index per saved part.
    pub part_frames: Vec<u32>,
    /// Exactly one dense frame index per saved shape region.
    pub region_frames: Vec<u32>,
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
    /// Parametric edge features in explicit replay order.
    #[serde(default)]
    pub shape_features: Vec<ShapeFeatureDoc>,
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
    #[allow(clippy::too_many_lines)] // Dense relationship remapping is one transaction.
    pub fn append(&mut self, mut other: Self) -> Result<(), CreationError> {
        let frame_offset =
            u32::try_from(self.frames.len()).map_err(|_| CreationError::TooManyRows)?;
        for frame in other.part_frames.iter_mut().chain(&mut other.region_frames) {
            *frame = frame
                .checked_add(frame_offset)
                .ok_or(CreationError::TooManyRows)?;
        }
        let part_offset =
            u32::try_from(self.parts.len()).map_err(|_| CreationError::TooManyRows)?;
        let bearing_offset =
            u32::try_from(self.bearings.len()).map_err(|_| CreationError::TooManyRows)?;
        let region_offset =
            u32::try_from(self.regions.len()).map_err(|_| CreationError::TooManyRows)?;
        let feature_offset =
            u32::try_from(self.shape_features.len()).map_err(|_| CreationError::TooManyRows)?;
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
            if let Some(TopologyKeyDoc {
                source: TopologySourceDoc::Feature(index),
                ..
            }) = &mut face.patch
            {
                *index = index
                    .checked_add(feature_offset)
                    .ok_or(CreationError::TooManyRows)?;
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
        for feature in &mut other.shape_features {
            for target in &mut feature.targets {
                match &mut target.owner {
                    SolidOwnerDoc::Part(index) => add_part(index)?,
                    SolidOwnerDoc::Region(index) => {
                        *index = index
                            .checked_add(region_offset)
                            .ok_or(CreationError::TooManyRows)?;
                    }
                }
                if let TopologySourceDoc::Feature(index) = &mut target.edge.source {
                    *index = index
                        .checked_add(feature_offset)
                        .ok_or(CreationError::TooManyRows)?;
                }
            }
        }
        self.frames.append(&mut other.frames);
        self.part_frames.append(&mut other.part_frames);
        self.region_frames.append(&mut other.region_frames);
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
        self.shape_features.append(&mut other.shape_features);
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
                | PartDoc::PipeJunction { pose, .. }
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
        let cardinal = GridRotation::new(0, yaw, 0).quaternion();
        for frame in &mut self.frames {
            if Vec3::from_array(frame.translation) == Vec3::ZERO
                && Quat::from_array(frame.rotation) == Quat::IDENTITY
            {
                continue;
            }
            let rotation = cardinal * Quat::from_array(frame.rotation) * cardinal.conjugate();
            frame.translation = (cardinal * Vec3::from_array(frame.translation) + translation
                - rotation * translation)
                .to_array();
            frame.rotation = rotation.to_array();
        }
        for bearing in &mut self.bearings {
            bearing.anchor =
                (rotate_y_vec3(Vec3::from_array(bearing.anchor), yaw) + translation).to_array();
            bearing.axis = rotate_y_vec3(Vec3::from_array(bearing.axis), yaw).to_array();
            if let crate::BearingKind::Linear(rail) = &mut bearing.kind {
                rail.mount_normal = rotate_y_vec3(rail.mount_normal, yaw);
            }
        }
        for socket in &mut self.sockets {
            socket.axis = rotate_y_vec3(Vec3::from_array(socket.axis), yaw).to_array();
            if let crate::BearingKind::Linear(rail) = &mut socket.kind {
                rail.mount_normal = rotate_y_vec3(rail.mount_normal, yaw);
            }
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
        let view_to_build = graph.view_to_build();
        let graph = graph.canonicalized();
        let frame_indices = index_map(graph.construction_frames().map(|(id, _)| id));
        let part_indices = index_map(graph.parts().map(|(id, _)| id));
        let bearing_indices = index_map(graph.bearings().map(|(id, _)| id));
        let region_indices = index_map(graph.regions().map(|(id, _)| id));
        let feature_indices = index_map(graph.shape_features().map(|(id, _)| id));
        let face = |face: FaceRef| face_doc(face, &part_indices, &feature_indices);
        let part = |part: PartId| {
            *part_indices
                .get(&part)
                .expect("every referenced part is live in the graph it came from")
        };

        Self {
            version: CREATION_FORMAT_VERSION,
            name: name.to_owned(),
            region_frames: graph
                .regions()
                .map(|(region, _)| {
                    frame_indices[&graph
                        .region_frame_id(region)
                        .expect("live regions have a frame")]
                })
                .collect(),
            frames: graph
                .construction_frames()
                .map(|(_, frame)| ConstructionFrameDoc {
                    translation: frame.translation().to_array(),
                    rotation: frame.rotation().to_array(),
                })
                .collect(),
            part_frames: graph
                .parts()
                .map(|(part, _)| {
                    frame_indices[&graph.part_frame_id(part).expect("live parts have a frame")]
                })
                .collect(),
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
            shape_features: graph
                .shape_features()
                .map(|(_, feature)| ShapeFeatureDoc {
                    targets: feature
                        .targets
                        .iter()
                        .map(|target| {
                            edge_chain_doc(
                                *target,
                                &part_indices,
                                &region_indices,
                                &feature_indices,
                            )
                        })
                        .collect(),
                    treatment: feature.treatment,
                    amount_ticks: feature.amount_ticks,
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
                    kind: bearing.kind,
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
                    linear_limits: link.linear_limits,
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
                .map(|socket| {
                    let mut socket = *socket;
                    // Preserve canonical rows exactly, including floating-point
                    // bit patterns, instead of applying a nominal identity map.
                    if view_to_build != crate::ConstructionFrame::IDENTITY {
                        socket.anchor = view_to_build.point(socket.anchor);
                        socket.axis = view_to_build.vector(socket.axis);
                        if let crate::BearingKind::Linear(ref mut rail) = socket.kind {
                            rail.mount_normal = view_to_build.vector(rail.mount_normal);
                        }
                    }
                    socket
                })
                .map(|socket| BearingSocketDoc {
                    kind: socket.kind,
                    axis: socket.axis.to_array(),
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
        if self.version != CREATION_FORMAT_VERSION {
            return Err(CreationError::UnsupportedVersion(self.version));
        }

        if self.part_frames.len() != self.parts.len() {
            return Err(CreationError::FrameMembershipCount);
        }
        if self.region_frames.len() != self.regions.len() {
            return Err(CreationError::RegionFrameMembershipCount);
        }
        let frames = self
            .frames
            .iter()
            .map(|frame| {
                crate::ConstructionFrame::new(
                    Vec3::from_array(frame.translation),
                    Quat::from_array(frame.rotation),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        for &frame in self.part_frames.iter().chain(&self.region_frames) {
            if frame as usize >= frames.len() {
                return Err(CreationError::MissingFrame(frame));
            }
        }
        let mut graph = ConstructionGraph::new();
        let frame_ids = frames
            .into_iter()
            .enumerate()
            .map(|(index, frame)| {
                if index == 0 && frame == crate::ConstructionFrame::IDENTITY {
                    Ok(crate::ConstructionFrameId::default())
                } else {
                    graph.add_construction_frame(frame)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut part_ids = vec![None; self.parts.len()];
        let mut transmission_children = vec![Vec::new(); self.parts.len()];
        let mut unresolved_transmissions = 0;
        for (index, part) in self.parts.iter().cloned().enumerate() {
            let PartDoc::Transmission { parent, .. } = part else {
                let BuildOutcome::Spawned(id) = graph.apply(build_command(part)?)? else {
                    unreachable!("part {index} replay uses a spawn command")
                };
                part_ids[index] = Some(id);
                continue;
            };
            let Some(children) = transmission_children.get_mut(parent as usize) else {
                return Err(CreationError::MissingPart(parent));
            };
            if self.part_frames[index] != self.part_frames[parent as usize] {
                return Err(CreationError::TransmissionFrame(
                    u32::try_from(index).map_err(|_| CreationError::TooManyRows)?,
                ));
            }
            children.push(index);
            unresolved_transmissions += 1;
        }

        let mut ready = self
            .parts
            .iter()
            .enumerate()
            .filter_map(|(index, part)| match part {
                PartDoc::Transmission { parent, .. }
                    if part_ids.get(*parent as usize).is_some_and(Option::is_some) =>
                {
                    Some(index)
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        while let Some(index) = ready.pop() {
            let PartDoc::Transmission { parent, pose } = self.parts[index] else {
                unreachable!("only transmissions wait for their parents")
            };
            let parent_id = part_ids
                .get(parent as usize)
                .copied()
                .flatten()
                .ok_or(CreationError::MissingPart(parent))?;
            let BuildOutcome::Spawned(id) = graph.apply(BuildCommand::AttachTransmission {
                parent: parent_id,
                spec: TransmissionSpec::new(pose.into()),
            })?
            else {
                unreachable!("transmission replay uses a spawn command")
            };
            part_ids[index] = Some(id);
            unresolved_transmissions -= 1;
            ready.extend(
                transmission_children
                    .get(index)
                    .into_iter()
                    .flatten()
                    .copied(),
            );
        }
        if unresolved_transmissions != 0 {
            let Some(parent) = self
                .parts
                .iter()
                .enumerate()
                .find_map(|(index, part)| match part {
                    PartDoc::Transmission { parent, .. } if part_ids[index].is_none() => {
                        Some(*parent)
                    }
                    _ => None,
                })
            else {
                return Err(CreationError::MissingPart(u32::MAX));
            };
            return Err(CreationError::MissingPart(parent));
        }
        let part_ids = part_ids
            .into_iter()
            .enumerate()
            .map(|(index, id)| {
                id.ok_or_else(|| {
                    CreationError::MissingPart(u32::try_from(index).unwrap_or(u32::MAX))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        for (index, &part) in part_ids.iter().enumerate() {
            graph.assign_part_frame(part, frame_ids[self.part_frames[index] as usize])?;
        }

        // Primitive welds establish rigid membership before Shape regions are
        // claimed. Connections on generated patches wait until feature replay.
        let mut initial_connections = Vec::with_capacity(self.welds.len() + self.rigid_links.len());
        for weld in self
            .welds
            .iter()
            .filter(|weld| weld.first.patch.is_none() && weld.second.patch.is_none())
        {
            initial_connections.push(BuildCommand::Weld(WeldSpec {
                first: resolve_face(weld.first, &part_ids, &[])?,
                second: resolve_face(weld.second, &part_ids, &[])?,
            }));
        }
        for link in &self.rigid_links {
            initial_connections.push(BuildCommand::RigidLink(RigidLinkSpec {
                first: resolve_part(link.first, &part_ids)?,
                second: resolve_part(link.second, &part_ids)?,
            }));
        }
        graph.apply_batch(initial_connections)?;
        for (index, document) in self.regions.iter().enumerate() {
            graph.set_edit_frame(frame_ids[self.region_frames[index] as usize])?;
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

        let region_ids = graph.regions().map(|(id, _)| id).collect::<Vec<_>>();
        let mut feature_ids = Vec::<ShapeFeatureId>::with_capacity(self.shape_features.len());
        for document in &self.shape_features {
            let targets = document
                .targets
                .iter()
                .copied()
                .map(|target| resolve_edge_chain(target, &part_ids, &region_ids, &feature_ids))
                .collect::<Result<Vec<_>, CreationError>>()?;
            let outcome = graph.apply(BuildCommand::AddShapeFeature(ShapeFeature::new(
                targets,
                document.treatment,
                document.amount_ticks,
            )))?;
            let BuildOutcome::ShapeFeatureAdded(id) = outcome else {
                unreachable!("adding a feature reports the feature it added")
            };
            feature_ids.push(id);
        }

        let mut final_connections = Vec::with_capacity(self.welds.len() + self.bearings.len());
        for weld in self
            .welds
            .iter()
            .filter(|weld| weld.first.patch.is_some() || weld.second.patch.is_some())
        {
            final_connections.push(BuildCommand::Weld(WeldSpec {
                first: resolve_face(weld.first, &part_ids, &feature_ids)?,
                second: resolve_face(weld.second, &part_ids, &feature_ids)?,
            }));
        }
        let first_bearing = final_connections.len();
        for bearing in &self.bearings {
            final_connections.push(BuildCommand::AddBearing(
                BearingSpec::new(
                    resolve_face(bearing.source, &part_ids, &feature_ids)?,
                    resolve_face(bearing.target, &part_ids, &feature_ids)?,
                    Vec3::from_array(bearing.anchor),
                    Vec3::from_array(bearing.axis),
                )
                .with_kind(bearing.kind)
                .with_dimensions(BearingDimensions::new(
                    bearing.outer_diameter,
                    bearing.inner_diameter,
                )?),
            ));
        }
        let outcomes = graph.apply_batch(final_connections)?;

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
                    linear_limits: link.linear_limits,
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
                if let crate::BearingKind::Linear(rail) = socket.kind {
                    rail.rotation(Vec3::from_array(socket.axis))?;
                }
                let socket = BearingSocket {
                    kind: socket.kind,
                    axis: Vec3::from_array(socket.axis),
                    source: resolve_face(socket.source, &part_ids, &feature_ids)?,
                    anchor: Vec3::from_array(socket.anchor),
                    dimensions: BearingDimensions::new(
                        socket.outer_diameter,
                        socket.inner_diameter,
                    )?,
                };
                graph.validate_suspension_socket(socket)?;
                Ok(socket)
            })
            .collect::<Result<Vec<_>, CreationError>>()?;

        graph.set_edit_frame(crate::ConstructionFrameId::default())?;
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
    let maximum = origin + size * crate::POSITION_TICKS_PER_GRID_UNIT;
    let corners = [
        IVec3::new(origin.x, origin.y, origin.z),
        IVec3::new(maximum.x, origin.y, origin.z),
        IVec3::new(origin.x, maximum.y, origin.z),
        IVec3::new(origin.x, origin.y, maximum.z),
        IVec3::new(maximum.x, maximum.y, maximum.z),
    ]
    .map(|corner| {
        rotate_y_i32(corner, yaw) + translation * crate::POSITION_TICKS_PER_HALF_GRID_UNIT
    });
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
    region.size_cells = ((maximum - minimum) / crate::POSITION_TICKS_PER_GRID_UNIT).to_array();
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

fn face_doc(
    face: FaceRef,
    parts: &HashMap<PartId, u32>,
    features: &HashMap<ShapeFeatureId, u32>,
) -> FaceRefDoc {
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
        patch: face.patch.map(|patch| TopologyKeyDoc {
            source: match patch.source {
                TopologySource::Base => TopologySourceDoc::Base,
                TopologySource::Feature(feature) => TopologySourceDoc::Feature(
                    *features
                        .get(&feature)
                        .expect("a referenced generated patch has a live feature"),
                ),
            },
            local: patch.local,
        }),
    }
}

fn edge_chain_doc(
    target: EdgeChainRef,
    parts: &HashMap<PartId, u32>,
    regions: &HashMap<crate::RegionId, u32>,
    features: &HashMap<ShapeFeatureId, u32>,
) -> EdgeChainRefDoc {
    EdgeChainRefDoc {
        owner: match target.owner {
            SolidOwner::Part(part) => SolidOwnerDoc::Part(
                *parts
                    .get(&part)
                    .expect("every feature part owner is live in its graph"),
            ),
            SolidOwner::Region(region) => SolidOwnerDoc::Region(
                *regions
                    .get(&region)
                    .expect("every feature region owner is live in its graph"),
            ),
        },
        edge: TopologyKeyDoc {
            source: match target.edge.source {
                TopologySource::Base => TopologySourceDoc::Base,
                TopologySource::Feature(feature) => TopologySourceDoc::Feature(
                    *features
                        .get(&feature)
                        .expect("generated topology references a live earlier feature"),
                ),
            },
            local: target.edge.local,
        },
    }
}

fn resolve_edge_chain(
    target: EdgeChainRefDoc,
    parts: &[PartId],
    regions: &[crate::RegionId],
    features: &[ShapeFeatureId],
) -> Result<EdgeChainRef, CreationError> {
    let owner = match target.owner {
        SolidOwnerDoc::Part(index) => SolidOwner::Part(resolve_part(index, parts)?),
        SolidOwnerDoc::Region(index) => SolidOwner::Region(
            *regions
                .get(index as usize)
                .ok_or(CreationError::MissingRegion(index))?,
        ),
    };
    let source = match target.edge.source {
        TopologySourceDoc::Base => TopologySource::Base,
        TopologySourceDoc::Feature(index) => TopologySource::Feature(
            *features
                .get(index as usize)
                .ok_or(CreationError::MissingShapeFeature(index))?,
        ),
    };
    Ok(EdgeChainRef {
        owner,
        edge: TopologyKey {
            source,
            local: target.edge.local,
        },
    })
}

fn layer_docs(layers: crate::MaterialLayers) -> Vec<MaterialLayerDoc> {
    layers
        .iter()
        .map(|layer| MaterialLayerDoc {
            face: layer.face,
            thickness: layer.thickness,
            material: layer.material,
            appearance: layer.appearance,
        })
        .collect()
}

/// Replays saved layers over a part's core.
fn with_layer_docs(spec: PartSpec, layers: &[MaterialLayerDoc]) -> Result<PartSpec, CreationError> {
    layers.iter().try_fold(spec, |spec, layer| {
        Ok(spec.with_layer(
            layer.face,
            layer.thickness,
            layer.material,
            layer.appearance,
        )?)
    })
}

fn part_doc(spec: PartSpec, transmission_parent: Option<u32>) -> PartDoc {
    match spec {
        PartSpec::Cuboid(cuboid) => {
            let core = cuboid.without_layers();
            PartDoc::Cuboid {
                dimensions: core.dimensions.map(GridDimension::units),
                pose: core.pose.into(),
                material: core.material,
                appearance: core.appearance,
                layers: layer_docs(cuboid.layers()),
            }
        }
        PartSpec::Cylinder(cylinder) => {
            let core = cylinder.without_layers();
            PartDoc::Cylinder {
                outer_diameter: core.dimensions.outer_diameter(),
                inner_diameter: core.dimensions.inner_diameter(),
                length_units: core.dimensions.axial_length_units(),
                sweep_degrees: core.dimensions.sweep_angle_degrees(),
                pose: core.pose.into(),
                material: core.material,
                appearance: core.appearance,
                layers: layer_docs(cylinder.layers()),
            }
        }
        PartSpec::PipeBend(bend) => PartDoc::PipeBend {
            outer_diameter: bend.dimensions.outer_diameter(),
            inner_diameter: bend.dimensions.inner_diameter(),
            span_blocks: bend.dimensions.span_blocks(),
            pose: bend.pose.into(),
            material: bend.material,
            appearance: bend.appearance,
        },
        PartSpec::PipeJunction(junction) => PartDoc::PipeJunction {
            outer_diameter: junction.dimensions.outer_diameter(),
            inner_diameter: junction.dimensions.inner_diameter(),
            arms: junction.arms.bits(),
            pose: junction.pose.into(),
            material: junction.material,
            appearance: junction.appearance,
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
            layers,
        } => {
            let core = CuboidSpec::new(dimensions, pose.into())?
                .with_material(material)
                .with_appearance(appearance);
            match with_layer_docs(PartSpec::Cuboid(core), &layers)? {
                PartSpec::Cuboid(cuboid) => BuildCommand::Spawn(cuboid),
                _ => unreachable!("layers keep the part kind"),
            }
        }
        PartDoc::Cylinder {
            outer_diameter,
            inner_diameter,
            length_units,
            sweep_degrees,
            pose,
            material,
            appearance,
            layers,
        } => {
            let core = CylinderSpec::new(
                CylinderDimensions::new(
                    outer_diameter,
                    inner_diameter,
                    f32::from(length_units) * crate::GRID_UNIT_METERS,
                )?
                .with_sweep_angle_degrees(sweep_degrees)?,
                pose.into(),
            )
            .with_material(material)
            .with_appearance(appearance);
            match with_layer_docs(PartSpec::Cylinder(core), &layers)? {
                PartSpec::Cylinder(cylinder) => BuildCommand::SpawnCylinder(cylinder),
                _ => unreachable!("layers keep the part kind"),
            }
        }
        PartDoc::PipeBend {
            outer_diameter,
            inner_diameter,
            span_blocks,
            pose,
            material,
            appearance,
        } => BuildCommand::SpawnPipeBend(
            PipeBendSpec::new(
                PipeBendDimensions::new(outer_diameter, inner_diameter, span_blocks)?,
                pose.into(),
            )
            .with_material(material)
            .with_appearance(appearance),
        ),
        PartDoc::PipeJunction {
            outer_diameter,
            inner_diameter,
            arms,
            pose,
            material,
            appearance,
        } => BuildCommand::SpawnPipeJunction(
            PipeJunctionSpec::new(
                PipeJunctionDimensions::new(outer_diameter, inner_diameter)?,
                PipeArms::from_bits(arms)?,
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

fn resolve_face(
    face: FaceRefDoc,
    parts: &[PartId],
    features: &[ShapeFeatureId],
) -> Result<FaceRef, CreationError> {
    let patch = face
        .patch
        .map(|patch| -> Result<crate::SurfacePatchKey, CreationError> {
            Ok(crate::SurfacePatchKey {
                source: match patch.source {
                    TopologySourceDoc::Base => TopologySource::Base,
                    TopologySourceDoc::Feature(index) => TopologySource::Feature(
                        *features
                            .get(index as usize)
                            .ok_or(CreationError::MissingShapeFeature(index))?,
                    ),
                },
                local: patch.local,
            })
        })
        .transpose()?;
    Ok(match face.owner {
        FaceOwnerDoc::Part(index) => FaceRef {
            owner: FaceOwner::Part(resolve_part(index, parts)?),
            face: face.face,
            patch,
        },
        FaceOwnerDoc::Ground => FaceRef {
            owner: FaceOwner::Ground,
            face: face.face,
            patch: None,
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
mod tests;

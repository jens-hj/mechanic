//! Serializable form of a saved creation.
//!
//! [`ConstructionGraph`] stores its rows in generational arenas whose handles
//! are minted privately and are not stable across a rebuild, so a file cannot
//! reference them directly. A [`CreationDocument`] instead numbers each row by
//! its position in the file and rebuilds the graph by replaying
//! [`BuildCommand`]s, remapping those dense indices onto the handles the arenas
//! hand back. Every value passes through the same validating constructors the
//! editor uses, so a hand-edited file cannot produce an invalid graph.

mod decode;
mod doc;
mod encode;
mod transform;

use decode::{
    build_command, index_map, resolve_edge_chain, resolve_face, resolve_limits, resolve_part,
    resolve_program,
};
pub use doc::{
    BearingDoc, BearingSocketDoc, ConstructionFrameDoc, DriveDwellDoc, DriveLimitsDoc,
    DriveLinkDoc, DriveProgramDoc, DriveStateDoc, DriveTriggerDoc, EdgeChainRefDoc, FaceOwnerDoc,
    FaceRefDoc, GearboxConfigDoc, InputSeatLinkDoc, MaterialLayerDoc, PartDoc, PoseDoc, RegionDoc,
    RigidLinkDoc, SeatControllerLinkDoc, ShapeFeatureDoc, SolidOwnerDoc, TopologyKeyDoc,
    TopologySourceDoc, WeldDoc,
};
use encode::{edge_chain_doc, face_doc, limits_doc, part_doc, program_doc};
use transform::{rotate_y_i32, rotate_y_vec3, transform_region_doc};

use bevy_math::{IVec3, Quat, Vec3};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    BearingDimensionError, BearingDimensions, BearingSpec, BuildCommand, BuildOutcome, BuildPose,
    ConstructionGraph, CylinderDimensionError, DimensionError, DimensionLinkId, DriveLimitsError,
    DriveLinkSpec, DriveName, DriveProgramError, EngineKind, FaceRef, GraphError, GridRotation,
    InputSeatLinkSpec, PartId, PipeBendDimensionError, PipeJunctionError, RigidLinkSpec,
    SeatControllerLinkSpec, ShapeFeature, ShapeFeatureId, ShapeRegion, TransmissionSpec, WeldSpec,
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

impl From<PoseDoc> for BuildPose {
    fn from(doc: PoseDoc) -> Self {
        let [x, y, z] = doc.rotation;
        Self::from_position_ticks(doc.translation_ticks.into(), GridRotation::new(x, y, z))
    }
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
    #[expect(
        clippy::too_many_lines,
        reason = "dense relationship remapping is one transaction"
    )]
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
            if let Some(target) = &mut bearing.target {
                add_face(target)?;
            }
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
            bearing.kind.rotate(|vector| rotate_y_vec3(vector, yaw));
        }
        for socket in &mut self.sockets {
            socket.axis = rotate_y_vec3(Vec3::from_array(socket.axis), yaw).to_array();
            socket.kind.rotate(|vector| rotate_y_vec3(vector, yaw));
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
    #[expect(
        clippy::too_many_lines,
        reason = "the document snapshot keeps all index remapping together"
    )]
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
                    target: bearing.target.map(&face),
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
                        socket.kind.rotate(|vector| view_to_build.vector(vector));
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
    #[expect(
        clippy::too_many_lines,
        reason = "one replay pass per serialized record family"
    )]
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
            let source = resolve_face(bearing.source, &part_ids, &feature_ids)?;
            let target = bearing
                .target
                .map(|target| resolve_face(target, &part_ids, &feature_ids))
                .transpose()?;
            let mut spec = BearingSpec::new(
                source,
                target.unwrap_or(source),
                Vec3::from_array(bearing.anchor),
                Vec3::from_array(bearing.axis),
            );
            spec.target = target;
            final_connections.push(BuildCommand::AddBearing(
                spec.with_kind(bearing.kind)
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
                graph.validate_socket(socket)?;
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

#[cfg(test)]
mod tests;

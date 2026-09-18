//! Document rows replayed into validated build commands.

use super::CreationError;
use super::doc::{
    DriveLimitsDoc, DriveProgramDoc, EdgeChainRefDoc, FaceOwnerDoc, FaceRefDoc, MaterialLayerDoc,
    PartDoc, SolidOwnerDoc, TopologySourceDoc,
};
use crate::{
    BuildCommand, ControllerSpec, CuboidSpec, CylinderDimensions, CylinderSpec, DimensionLinkSpec,
    DriveDwell, DriveKey, DriveLimits, DriveProgram, DriveState, DriveTrigger, EdgeChainRef,
    EngineSpec, FaceOwner, FaceRef, InputSpec, PartId, PartSpec, PipeArms, PipeBendDimensions,
    PipeBendSpec, PipeJunctionDimensions, PipeJunctionSpec, SeatSpec, ServoSpec, ShapeFeatureId,
    SolidOwner, TopologyKey, TopologySource,
};
use std::collections::HashMap;

pub(super) fn index_map<I: Copy + Eq + std::hash::Hash>(
    ids: impl Iterator<Item = I>,
) -> HashMap<I, u32> {
    ids.enumerate()
        .map(|(index, id)| {
            (
                id,
                u32::try_from(index).expect("construction arena indices fit u32"),
            )
        })
        .collect()
}

pub(super) fn resolve_edge_chain(
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

/// Replays saved layers over a part's core.
pub(super) fn with_layer_docs(
    spec: PartSpec,
    layers: &[MaterialLayerDoc],
) -> Result<PartSpec, CreationError> {
    layers.iter().try_fold(spec, |spec, layer| {
        Ok(spec.with_layer(
            layer.face,
            layer.thickness,
            layer.material,
            layer.appearance,
        )?)
    })
}

pub(super) fn build_command(part: PartDoc) -> Result<BuildCommand, CreationError> {
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

pub(super) fn resolve_part(index: u32, parts: &[PartId]) -> Result<PartId, CreationError> {
    parts
        .get(index as usize)
        .copied()
        .ok_or(CreationError::MissingPart(index))
}

pub(super) fn resolve_face(
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

pub(super) fn resolve_limits(limits: DriveLimitsDoc) -> Result<DriveLimits, CreationError> {
    Ok(DriveLimits::new(
        limits.max_speed_rad_s,
        limits.max_torque_newton_meters.unwrap_or(f32::INFINITY),
        limits.angle_limits,
    )?)
}

pub(super) fn resolve_program(program: &DriveProgramDoc) -> Result<DriveProgram, CreationError> {
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

//! Dense on-disk references for physical input configuration.

use crate::{
    AnalogMapping, AnalogRange, BuildCommand, ButtonMode, ConstructionGraph, CreationError,
    DriveKey, DriveParameter, EngineKind, GearParameter, InputConfiguration, NumericParameter,
    PartId,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Saved numeric reference, using document rows rather than arena handles.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum NumericParameterDoc {
    /// Driven joint setting.
    Drive {
        /// Dense drive-link row.
        link: u32,
        /// Typed numeric field.
        parameter: DriveParameter,
    },
    /// Controller gearbox setting.
    Gear {
        /// Dense controller part row.
        controller: u32,
        /// Engine family.
        kind: EngineKind,
        /// Typed numeric field.
        parameter: GearParameter,
    },
}

/// Saved dial mapping in native units.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnalogMappingDoc {
    /// Dense controller target.
    pub target: NumericParameterDoc,
    /// Native-unit endpoints and inversion.
    pub range: AnalogRange,
}

/// Persistent input configuration, excluding all transient feedback and values.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InputConfigurationDoc {
    /// Dense physical-input part row.
    pub input: u32,
    /// Player-facing name.
    pub name: String,
    /// Dense controller part row, when connected.
    pub controller: Option<u32>,
    /// Momentary or toggle behavior.
    pub button_mode: ButtonMode,
    /// Optional ASCII controller key.
    pub key: Option<char>,
    /// Dial mappings.
    pub analog: Vec<AnalogMappingDoc>,
}

pub(super) fn input_docs(
    graph: &ConstructionGraph,
    part: &impl Fn(PartId) -> u32,
) -> Vec<InputConfigurationDoc> {
    let drives: BTreeMap<_, _> = graph
        .drive_links()
        .enumerate()
        .map(|(index, (id, _))| (id, u32::try_from(index).expect("bounded drive count")))
        .collect();
    graph
        .physical_inputs()
        .map(|(id, config)| InputConfigurationDoc {
            input: part(id),
            name: config.name.clone(),
            controller: config.controller.map(part),
            button_mode: config.button_mode,
            key: config.key.map(DriveKey::symbol),
            analog: config
                .analog
                .iter()
                .map(|mapping| AnalogMappingDoc {
                    range: mapping.range,
                    target: match mapping.target {
                        NumericParameter::Drive { link, parameter } => NumericParameterDoc::Drive {
                            link: drives[&link],
                            parameter,
                        },
                        NumericParameter::Gear {
                            controller,
                            kind,
                            parameter,
                        } => NumericParameterDoc::Gear {
                            controller: part(controller),
                            kind,
                            parameter,
                        },
                    },
                })
                .collect(),
        })
        .collect()
}

pub(super) fn apply_input_docs(
    graph: &mut ConstructionGraph,
    docs: &[InputConfigurationDoc],
    parts: &[PartId],
) -> Result<(), CreationError> {
    let drives: Vec<_> = graph.drive_links().map(|(id, _)| id).collect();
    let part = |index| super::decode::resolve_part(index, parts);
    let drive = |index: u32| {
        drives
            .get(index as usize)
            .copied()
            .ok_or(CreationError::MissingDriveLink(index))
    };
    let mut seen = std::collections::BTreeSet::new();
    for doc in docs {
        let input = part(doc.input)?;
        if !seen.insert(input) {
            return Err(crate::GraphError::from(crate::InputBindingError::InvalidTarget).into());
        }
        let configuration = InputConfiguration {
            name: doc.name.clone(),
            controller: doc.controller.map(part).transpose()?,
            button_mode: doc.button_mode,
            key: doc
                .key
                .map(|key| DriveKey::new(key).ok_or(CreationError::InvalidDriveKey(key)))
                .transpose()?,
            analog: doc
                .analog
                .iter()
                .map(|mapping| {
                    Ok(AnalogMapping {
                        range: mapping.range,
                        target: match mapping.target {
                            NumericParameterDoc::Drive { link, parameter } => {
                                NumericParameter::Drive {
                                    link: drive(link)?,
                                    parameter,
                                }
                            }
                            NumericParameterDoc::Gear {
                                controller,
                                kind,
                                parameter,
                            } => NumericParameter::Gear {
                                controller: part(controller)?,
                                kind,
                                parameter,
                            },
                        },
                    })
                })
                .collect::<Result<_, CreationError>>()?,
        };
        // Establish ownership first: reconnect intentionally clears any bindings.
        graph.apply(BuildCommand::SetInputConfiguration {
            input,
            configuration: InputConfiguration {
                controller: configuration.controller,
                ..InputConfiguration::default()
            },
        })?;
        graph.apply(BuildCommand::SetInputConfiguration {
            input,
            configuration,
        })?;
    }
    Ok(())
}

pub(super) fn offset_input_docs(
    docs: &mut [InputConfigurationDoc],
    parts: u32,
    drives: u32,
) -> Result<(), CreationError> {
    let add = |index: &mut u32, offset| {
        *index = index
            .checked_add(offset)
            .ok_or(CreationError::TooManyRows)?;
        Ok::<_, CreationError>(())
    };
    for doc in docs {
        add(&mut doc.input, parts)?;
        if let Some(controller) = &mut doc.controller {
            add(controller, parts)?;
        }
        for mapping in &mut doc.analog {
            match &mut mapping.target {
                NumericParameterDoc::Drive { link, .. } => add(link, drives)?,
                NumericParameterDoc::Gear { controller, .. } => add(controller, parts)?,
            }
        }
    }
    Ok(())
}

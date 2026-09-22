//! Persistent dial assignment transactions, independent of the Mosaic view.
use crate::editor::{
    history::{EditorHistory, EditorSnapshot},
    state::{EditorGraph, EditorState},
};
use bevy::prelude::*;
use mechanic_core::{AnalogMapping, AnalogRange, NumericParameter, PartId};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Draft {
    pub targets: Vec<NumericParameter>,
    pub name: String,
    pub current: f32,
    pub dial: Option<PartId>,
    pub minimum: f32,
    pub maximum: f32,
    pub reverse: bool,
    pub factor: f32,
    pub unit: String,
    pub integer: bool,
    pub error: Option<String>,
}
#[derive(Resource, Default)]
pub(crate) struct DialAssignments {
    pub draft: Option<Draft>,
    pub located: Option<PartId>,
}

/// Renames a physical dial without changing its assignments or runtime values.
pub(crate) fn rename(
    dial: PartId,
    name: &str,
    graph: &mut EditorGraph,
    editor: &mut EditorState,
    history: &mut EditorHistory,
) {
    if !matches!(graph.0.part(dial), Some(mechanic_core::PartSpec::Dial(_))) {
        return;
    }
    let Some(mut configuration) = graph.0.input_configuration(dial).cloned() else {
        return;
    };
    let name = name.trim();
    if name.is_empty() || configuration.name == name {
        return;
    }
    name.clone_into(&mut configuration.name);
    let previous = EditorSnapshot::capture(&graph.0, editor);
    if graph
        .0
        .apply(mechanic_core::BuildCommand::SetInputConfiguration {
            input: dial,
            configuration,
        })
        .is_ok()
    {
        history.commit(previous);
    }
}

pub(crate) fn apply(
    state: &mut DialAssignments,
    graph: &mut EditorGraph,
    editor: &mut EditorState,
    history: &mut EditorHistory,
    unlink: bool,
) {
    let Some(draft) = state.draft.as_ref() else {
        return;
    };
    let previous = EditorSnapshot::capture(&graph.0, editor);
    let result = if unlink {
        graph.0.unlink_dial_targets(&draft.targets);
        Ok(())
    } else {
        let Some(dial) = draft.dial else {
            if let Some(draft) = &mut state.draft {
                draft.error = Some("Select a connected dial before applying.".into());
            }
            return;
        };
        let mappings = AnalogRange::new(
            [draft.minimum / draft.factor, draft.maximum / draft.factor],
            draft.reverse,
        )
        .map(|range| {
            draft
                .targets
                .iter()
                .map(|&target| AnalogMapping { target, range })
                .collect::<Vec<_>>()
        });
        match mappings {
            Ok(mappings) => graph.0.assign_dial(dial, &mappings),
            Err(error) => Err(error.into()),
        }
    };
    match result {
        Ok(()) => {
            history.commit(previous);
            state.draft = None;
        }
        Err(error) => {
            if let Some(draft) = &mut state.draft {
                draft.error = Some(error.to_string());
            }
        }
    }
}

/// Highlights controller Locate selections at their current physical pose.
pub(crate) fn highlight(
    assignments: Res<DialAssignments>,
    graph: Res<EditorGraph>,
    simulation: Res<crate::simulation::state::AppSimulation>,
    mut gizmos: Gizmos,
) {
    let part = assignments.located;
    let Some(part) = part else { return };
    let Some((position, rotation)) = simulation.live_part_pose(&graph.0, part) else {
        return;
    };
    let Some(mechanic_core::PartSpec::Dial(spec)) = graph.0.part(part) else {
        return;
    };
    gizmos.cube(
        Transform::from_translation(position)
            .with_rotation(rotation)
            .with_scale(spec.size_meters() + Vec3::splat(0.02)),
        Color::srgb(1.0, 0.65, 0.08),
    );
}

//! The Tread tool: cutting the brush's tread into the surfaces dragged over,
//! smoothing them again, and sampling a surface's tread into the brush.

use crate::builder::{PlacementError, layer_target_from_hit};
use crate::controls::GameAction;
use crate::editor::history::{EditorHistory, EditorSnapshot};
use crate::editor::state::EditorState;
use crate::tread::tread_label;
use bevy::prelude::ButtonInput;
use mechanic_core::{
    BuildCommand, BuildOutcome, ConstructionGraph, FaceKind, LayerFace, PartId, TreadSpec,
};
use std::collections::HashSet;

/// One press-and-drag of the Tread tool, undone as one step.
#[derive(Clone, Debug)]
pub(crate) struct TreadStroke {
    previous: EditorSnapshot,
    surfaces: HashSet<(PartId, LayerFace)>,
    smooth: bool,
    changed: bool,
}

/// The part surface under the cursor that can take a tread.
pub(crate) fn hovered_surface(
    graph: &ConstructionGraph,
    state: &EditorState,
) -> Result<(PartId, LayerFace), PlacementError> {
    let hit = state.hovered.ok_or(PlacementError::NotLayerSurface)?;
    layer_target_from_hit(graph, hit).map(|target| (target.part, target.face))
}

/// The tread on the surface under the cursor, if it has one.
pub(crate) fn sample(graph: &ConstructionGraph, state: &EditorState) -> Option<TreadSpec> {
    let (part, surface) = hovered_surface(graph, state).ok()?;
    graph.part_treads(part).get(surface)
}

/// A surface's name for the status line.
pub(crate) const fn surface_label(surface: LayerFace) -> &'static str {
    match surface {
        LayerFace::OuterWall => "outer wall",
        LayerFace::Bore => "bore",
        LayerFace::Face(FaceKind::PositiveX) => "+X face",
        LayerFace::Face(FaceKind::NegativeX) => "−X face",
        LayerFace::Face(FaceKind::PositiveY) => "+Y face",
        LayerFace::Face(FaceKind::NegativeY) => "−Y face",
        LayerFace::Face(FaceKind::PositiveZ) => "+Z face",
        LayerFace::Face(FaceKind::NegativeZ) => "−Z face",
    }
}

/// Says once what the surface under the cursor has and what a click would
/// make of it, or why it takes no tread.
pub(crate) fn announce_hover(graph: &ConstructionGraph, state: &mut EditorState, brush: TreadSpec) {
    let Ok((part, surface)) = hovered_surface(graph, state) else {
        return;
    };
    let Some(spec) = graph.part(part) else {
        return;
    };
    let label = surface_label(surface);
    let line = if let Err(error) = spec.tread_fits(surface) {
        format!("{label}: {error}")
    } else {
        match graph.part_treads(part).get(surface) {
            Some(current) if current == brush => {
                format!("{label}: {} · Right-drag to smooth", tread_label(current))
            }
            Some(current) => format!(
                "{label}: {} · Click to recut as {}",
                tread_label(current),
                tread_label(brush)
            ),
            None => format!("{label}: smooth · Click to cut {}", tread_label(brush)),
        }
    };
    if state.tread_announced.as_deref() != Some(line.as_str()) {
        state.tread_announced = Some(line.clone());
        state.feedback = Some(line);
    }
}

/// Left-drag cuts the brush's tread into every surface passed over;
/// right-drag smooths them. One drag is one undo step.
pub(crate) fn handle_tread_actions(
    actions: &ButtonInput<GameAction>,
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    brush: TreadSpec,
) {
    let smooth = actions.just_pressed(GameAction::Secondary);
    if actions.just_pressed(GameAction::Primary) || smooth {
        state.tread_stroke = Some(TreadStroke {
            previous: EditorSnapshot::capture(graph, state),
            surfaces: HashSet::new(),
            smooth,
            changed: false,
        });
    }
    if (actions.pressed(GameAction::Primary) || actions.pressed(GameAction::Secondary))
        && let Ok((part, surface)) = hovered_surface(graph, state)
        && let Some(stroke) = state.tread_stroke.as_mut()
        && stroke.surfaces.insert((part, surface))
    {
        let wanted = (!stroke.smooth).then_some(brush);
        if graph.part_treads(part).get(surface) != wanted {
            match graph.apply(BuildCommand::SetTread {
                part,
                surface,
                tread: wanted,
            }) {
                Ok(BuildOutcome::TreadUpdated) => {
                    stroke.changed = true;
                    state.construction_mesh_dirty = true;
                    state.tread_announced = None;
                }
                Ok(_) => unreachable!("tread edits report their outcome"),
                Err(error) => state.feedback = Some(error.to_string()),
            }
        }
    }
    if (actions.just_released(GameAction::Primary) || actions.just_released(GameAction::Secondary))
        && let Some(stroke) = state.tread_stroke.take()
        && stroke.changed
    {
        history.commit(stroke.previous);
        state.feedback = Some(if stroke.smooth {
            "Smoothed treaded surfaces".to_owned()
        } else {
            format!("Cut {} treads", tread_label(brush))
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::SurfaceHit;
    use crate::editor::history::{HistoryAction, apply_history_action};
    use bevy::prelude::{IVec3, Vec3};
    use mechanic_core::{BuildPose, CuboidSpec, FaceRef, GridRotation, TreadPattern};

    /// Two loose blocks side by side, and a hit on the top face of either.
    fn blocks() -> (ConstructionGraph, [PartId; 2]) {
        let mut graph = ConstructionGraph::new();
        let ids = [0, 4].map(|x| {
            let spec = CuboidSpec::new(
                [1, 1, 1],
                BuildPose::new(IVec3::new(x, 0, 0), GridRotation::default()),
            )
            .unwrap();
            let BuildOutcome::Spawned(id) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
                unreachable!("spawning reports the part")
            };
            id
        });
        (graph, ids)
    }

    fn top(graph: &ConstructionGraph, part: PartId) -> SurfaceHit {
        SurfaceHit {
            distance: 1.0,
            point: graph.part_position(part).unwrap() + Vec3::Y * 0.125,
            face: FaceRef::part(part, FaceKind::PositiveY),
        }
    }

    fn drag(
        button: GameAction,
        graph: &mut ConstructionGraph,
        state: &mut EditorState,
        history: &mut EditorHistory,
        over: &[PartId],
        brush: TreadSpec,
    ) {
        let mut mouse = ButtonInput::default();
        for (index, &part) in over.iter().enumerate() {
            state.hovered = Some(top(graph, part));
            if index == 0 {
                mouse.press(button);
            }
            handle_tread_actions(&mouse, graph, state, history, brush);
            mouse.clear();
        }
        mouse.release(button);
        handle_tread_actions(&mouse, graph, state, history, brush);
    }

    #[test]
    fn a_drag_cuts_every_surface_it_crosses_as_one_undoable_step() {
        let (mut graph, ids) = blocks();
        let brush = TreadSpec::new(TreadPattern::Chevron, 14).unwrap();
        let mut state = EditorState::default();
        let mut history = EditorHistory::default();
        drag(
            GameAction::Primary,
            &mut graph,
            &mut state,
            &mut history,
            &ids,
            brush,
        );
        let face = LayerFace::Face(FaceKind::PositiveY);
        assert!(
            ids.iter()
                .all(|&part| graph.part_treads(part).get(face) == Some(brush))
        );
        assert_eq!(history.undo.len(), 1);

        state.hovered = Some(top(&graph, ids[1]));
        assert_eq!(sample(&graph, &state), Some(brush));

        assert!(apply_history_action(
            HistoryAction::Undo,
            &mut graph,
            &mut state,
            &mut history,
        ));
        assert_eq!(graph.treaded_parts().count(), 0);
    }

    #[test]
    fn a_right_drag_smooths_and_a_drag_that_changes_nothing_adds_no_step() {
        let (mut graph, ids) = blocks();
        let brush = TreadSpec::new(TreadPattern::Studded, 8).unwrap();
        let mut state = EditorState::default();
        let mut history = EditorHistory::default();
        for button in [
            GameAction::Primary,
            GameAction::Primary,
            GameAction::Secondary,
        ] {
            drag(button, &mut graph, &mut state, &mut history, &ids, brush);
        }
        assert_eq!(history.undo.len(), 2);
        assert_eq!(graph.treaded_parts().count(), 0);
    }
}

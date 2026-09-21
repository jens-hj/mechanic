//! The Spiral tool: settings, the cylinder under the cursor, and applying one to the other.

use crate::builder::spiral::{
    SpiralDimension, SpiralMode, SpiralSettings, SpiralTarget, SpiralWall, spiral_target_from_hit,
    spiralled, stage_spiral, unspiralled, validate_spiral,
};
use crate::builder::{PlacementError, SurfaceHit};
use crate::controls::GameAction;
use crate::editor::history::{EditorHistory, EditorSnapshot};
use crate::editor::hover::clear_hover;
use crate::editor::state::EditorState;
use bevy::prelude::ButtonInput;
use mechanic_core::{ConstructionGraph, CylinderSpec, SpiralHand};

/// The cylinder the tool is pointed at and what the settings would make of it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SpiralPreview {
    pub(crate) target: SpiralTarget,
    pub(crate) spec: CylinderSpec,
}

/// The Spiral tool's state.
#[derive(Clone, Debug, Default)]
pub(crate) struct SpiralTool {
    pub(crate) settings: SpiralSettings,
    pub(crate) preview: Option<SpiralPreview>,
}

/// Works out what the settings make of the cylinder under the cursor. Returns
/// why they cannot, if they cannot.
pub(crate) fn hover(
    graph: &ConstructionGraph,
    state: &mut EditorState,
    hit: SurfaceHit,
) -> Option<PlacementError> {
    let target = match spiral_target_from_hit(graph, hit) {
        Ok(target) => target,
        Err(error) => return Some(error),
    };
    let settings = state.spiral.settings.fitted_to(target.spec);
    match spiralled(settings, &target) {
        Ok(spec) => {
            let error = validate_spiral(graph, &target, spec, state.placement_bounds).err();
            state.spiral.preview = Some(SpiralPreview { target, spec });
            error
        }
        Err(error) => Some(error),
    }
}

// Keys that change the settings, as (action, what it does).
fn adjusted(actions: &ButtonInput<GameAction>, mut settings: SpiralSettings) -> SpiralSettings {
    for (decrease, increase, dimension) in [
        (
            GameAction::CylinderLengthDecrease,
            GameAction::CylinderLengthIncrease,
            SpiralDimension::Pitch,
        ),
        (
            GameAction::CylinderInnerDecrease,
            GameAction::CylinderInnerIncrease,
            SpiralDimension::Width,
        ),
        (
            GameAction::CylinderOuterDecrease,
            GameAction::CylinderOuterIncrease,
            SpiralDimension::Depth,
        ),
        (
            GameAction::CylinderSweepDecrease,
            GameAction::CylinderSweepIncrease,
            SpiralDimension::Starts,
        ),
        (
            GameAction::ZoomOut,
            GameAction::ZoomIn,
            SpiralDimension::Taper,
        ),
        (
            GameAction::NudgeDown,
            GameAction::NudgeUp,
            SpiralDimension::Tip,
        ),
    ] {
        let direction =
            i8::from(actions.just_pressed(increase)) - i8::from(actions.just_pressed(decrease));
        if direction != 0 {
            settings = settings.adjusted(dimension, direction);
        }
    }
    if actions.just_pressed(GameAction::PipeTurn) {
        settings.preset = settings.preset.next();
    }
    if actions.just_pressed(GameAction::Rotate) {
        settings.hand = match settings.hand {
            SpiralHand::Right => SpiralHand::Left,
            SpiralHand::Left => SpiralHand::Right,
        };
    }
    if actions.just_pressed(GameAction::ShapeMirrorX) {
        settings.mode = match settings.mode {
            SpiralMode::Cut => SpiralMode::Add,
            SpiralMode::Add => SpiralMode::Cut,
        };
    }
    if actions.just_pressed(GameAction::ShapeMirrorZ) {
        settings.wall = match settings.wall {
            SpiralWall::Outer => SpiralWall::Bore,
            SpiralWall::Bore => SpiralWall::Outer,
        };
    }
    settings
}

/// Click a cylinder to put the spiral on it. Changing a setting while pointing
/// at a cylinder that already carries one reshapes it at once, so the part
/// itself is the preview. Secondary takes the spiral off again.
pub(crate) fn handle_spiral_actions(
    actions: &ButtonInput<GameAction>,
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
) {
    let target = state
        .spiral
        .preview
        .map(|preview| preview.target)
        .or_else(|| {
            state
                .hovered
                .and_then(|hit| spiral_target_from_hit(graph, hit).ok())
        });
    let carries_spiral = target.is_some_and(|target| target.spec.spiral().is_some());

    if actions.just_pressed(GameAction::Interact)
        && let Some(picked) =
            target.and_then(|target| state.spiral.settings.picked_from(target.spec))
    {
        state.spiral.settings = picked;
        state.feedback = Some(format!("Picked up: {}", picked.summary()));
        return;
    }
    let before = state.spiral.settings;
    let settings = adjusted(actions, before);
    let changed = settings != before;
    state.spiral.settings = settings;
    if changed {
        state.feedback = Some(settings.summary());
    }

    if actions.just_pressed(GameAction::Secondary) {
        let Some(target) = target.filter(|_| carries_spiral) else {
            state.feedback = Some("Point at a spiral cylinder to take its spiral off".to_owned());
            return;
        };
        let plain = unspiralled(settings, &target);
        commit(graph, state, history, &target, plain, "Spiral removed");
        return;
    }
    if !(actions.just_pressed(GameAction::Primary) || (changed && carries_spiral)) {
        return;
    }
    let Some(target) = target else {
        if actions.just_pressed(GameAction::Primary) {
            state.feedback = Some(state.preview_error.as_ref().map_or_else(
                || "Point at a full cylinder".to_owned(),
                ToString::to_string,
            ));
        }
        return;
    };
    let fitted = settings.fitted_to(target.spec);
    state.spiral.settings = fitted;
    let spec = spiralled(fitted, &target);
    commit(graph, state, history, &target, spec, &fitted.summary());
}

fn commit(
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
    target: &SpiralTarget,
    spec: Result<CylinderSpec, PlacementError>,
    done: &str,
) {
    let staged = spec.and_then(|spec| stage_spiral(graph, target, spec, state.placement_bounds));
    match staged {
        Ok(staged) => {
            let previous = EditorSnapshot::capture(graph, state);
            *graph = staged;
            history.commit(previous);
            state.construction_mesh_dirty = true;
            clear_hover(state);
            state.spiral.preview = None;
            state.feedback = Some(done.to_owned());
        }
        Err(error) => state.feedback = Some(error.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mechanic_core::{
        BuildCommand, BuildOutcome, BuildPose, CylinderDimensions, GridRotation, PartId, SpiralEnd,
    };

    fn shaft() -> (ConstructionGraph, PartId) {
        let mut graph = ConstructionGraph::new();
        let BuildOutcome::Spawned(part) = graph
            .apply(BuildCommand::SpawnCylinder(CylinderSpec::new(
                CylinderDimensions::new(0.5, 0.0, 2.0).unwrap(),
                BuildPose::from_position_ticks(bevy::math::IVec3::Y * 800, GridRotation::default()),
            )))
            .unwrap()
        else {
            unreachable!()
        };
        (graph, part)
    }

    // Points the tool at the part, as hovering it does.
    fn aim(graph: &ConstructionGraph, state: &mut EditorState, part: PartId) {
        let spec = graph
            .part(part)
            .and_then(|spec| spec.as_cylinder())
            .unwrap();
        state.spiral.preview = Some(SpiralPreview {
            target: SpiralTarget {
                part,
                spec,
                frame: graph.part_frame(part).unwrap(),
                near_end: SpiralEnd::NegativeY,
            },
            spec,
        });
    }

    fn tap(action: GameAction) -> ButtonInput<GameAction> {
        let mut actions = ButtonInput::default();
        actions.press(action);
        actions
    }

    #[test]
    fn a_click_spirals_a_cylinder_a_key_reshapes_it_and_right_click_takes_it_off() {
        let (mut graph, part) = shaft();
        let mut state = EditorState::default();
        let mut history = EditorHistory::default();
        let spiral = |graph: &ConstructionGraph| {
            graph
                .part(part)
                .and_then(|spec| spec.as_cylinder())
                .and_then(CylinderSpec::spiral)
        };

        aim(&graph, &mut state, part);
        handle_spiral_actions(
            &tap(GameAction::Primary),
            &mut graph,
            &mut state,
            &mut history,
        );
        let first = spiral(&graph).expect("the click put a spiral on the cylinder");
        assert_eq!(first.pitch_ticks(), 200, "sized to the cylinder it met");
        assert_eq!(history.undo.len(), 1);

        // Pointing at it, a longer pitch reshapes it without another click.
        aim(&graph, &mut state, part);
        handle_spiral_actions(
            &tap(GameAction::CylinderLengthIncrease),
            &mut graph,
            &mut state,
            &mut history,
        );
        assert_eq!(spiral(&graph).unwrap().pitch_ticks(), 210);
        assert_eq!(history.undo.len(), 2);

        // Pointing at nothing, a key only changes the settings.
        handle_spiral_actions(
            &tap(GameAction::Rotate),
            &mut graph,
            &mut state,
            &mut history,
        );
        assert_eq!(state.spiral.settings.hand, SpiralHand::Left);
        assert_eq!(spiral(&graph).unwrap().hand(), SpiralHand::Right);

        aim(&graph, &mut state, part);
        handle_spiral_actions(
            &tap(GameAction::Secondary),
            &mut graph,
            &mut state,
            &mut history,
        );
        assert!(spiral(&graph).is_none());
        let plain = graph
            .part(part)
            .and_then(|spec| spec.as_cylinder())
            .unwrap();
        assert!((plain.dimensions.outer_diameter() - 0.5).abs() < 1.0e-6);
    }
}

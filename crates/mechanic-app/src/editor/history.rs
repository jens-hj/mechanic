//! Undo and redo: whole-creation snapshots and the shortcuts that walk them.

use crate::controls::GameAction;
use crate::editor::build_actions::PlacedBearing;
use crate::editor::hover::clear_hover;
use crate::editor::state::{EditorGraph, EditorState};
use crate::simulation::state::AppSimulation;
use crate::{freeze, live_edit, ui, weld_publication};
use bevy::prelude::{ButtonInput, Res, ResMut, Resource};
use mechanic_core::{AppearanceTarget, BuildCommand, ConstructionGraph};
use std::collections::{HashSet, VecDeque};
use std::sync::Arc;

pub(crate) const HISTORY_CAPACITY: usize = 64;

#[derive(Clone, Debug)]
pub(crate) struct EditorSnapshot {
    pub(crate) weld_restore: Option<weld_publication::Restore>,
    pub(crate) graph: Arc<ConstructionGraph>,
    pub(crate) placed_bearings: Vec<PlacedBearing>,
    pub(crate) revision: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct ChromaStroke {
    pub(crate) previous: EditorSnapshot,
    pub(crate) targets: HashSet<AppearanceTarget>,
    pub(crate) remove: bool,
    pub(crate) changed: bool,
}

impl EditorSnapshot {
    pub(crate) fn capture(graph: &ConstructionGraph, state: &EditorState) -> Self {
        let basis = graph.view_to_build();
        let mut graph = graph.canonicalized();
        if graph.pending().is_some() {
            graph
                .apply(BuildCommand::CancelPending)
                .expect("captured pending editor operation can be cancelled");
        }
        Self {
            weld_restore: None,
            graph: Arc::new(graph),
            placed_bearings: state
                .placed_bearings
                .iter()
                .map(|&bearing| live_edit::transform_bearing(bearing, basis))
                .collect(),
            revision: 0,
        }
    }
}

#[derive(Clone, Resource, Default)]
pub(crate) struct EditorHistory {
    pub(crate) undo: VecDeque<EditorSnapshot>,
    pub(crate) redo: VecDeque<EditorSnapshot>,
    pub(crate) current_revision: u64,
    pub(crate) clean_revision: u64,
    pub(crate) next_revision: u64,
}

impl EditorHistory {
    pub(crate) fn commit(&mut self, mut previous: EditorSnapshot) {
        previous.revision = self.current_revision;
        self.redo.clear();
        if self.undo.len() == HISTORY_CAPACITY {
            self.undo.pop_front();
        }
        self.undo.push_back(previous);
        self.next_revision = self.next_revision.saturating_add(1);
        self.current_revision = self.next_revision;
    }

    pub(crate) fn undo(&mut self, mut current: EditorSnapshot) -> Option<EditorSnapshot> {
        current.revision = self.current_revision;
        let previous = self.undo.pop_back()?;
        self.current_revision = previous.revision;
        self.redo.push_back(current);
        Some(previous)
    }

    pub(crate) fn redo(&mut self, mut current: EditorSnapshot) -> Option<EditorSnapshot> {
        current.revision = self.current_revision;
        let next = self.redo.pop_back()?;
        self.current_revision = next.revision;
        self.undo.push_back(current);
        Some(next)
    }

    pub(crate) const fn is_dirty(&self) -> bool {
        self.current_revision != self.clean_revision
    }

    pub(crate) fn mark_clean(&mut self) {
        self.clean_revision = self.current_revision;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HistoryAction {
    Undo,
    Redo,
}

pub(crate) fn requested_history_action(actions: &ButtonInput<GameAction>) -> Option<HistoryAction> {
    if actions.just_pressed(GameAction::Redo) {
        Some(HistoryAction::Redo)
    } else if actions.just_pressed(GameAction::Undo) {
        Some(HistoryAction::Undo)
    } else {
        None
    }
}

pub(crate) fn handle_history_shortcut(
    actions: Res<ButtonInput<GameAction>>,
    mut graph: ResMut<EditorGraph>,
    mut state: ResMut<EditorState>,
    mut history: ResMut<EditorHistory>,
    overlay: Res<ui::UiInput>,
    simulation: Res<AppSimulation>,
    frozen: Res<freeze::DimensionFreeze>,
) {
    if overlay.blocks_keyboard() {
        return;
    }
    let Some(action) = requested_history_action(&actions) else {
        return;
    };
    state.history_capture = match action {
        HistoryAction::Undo => history.undo.back(),
        HistoryAction::Redo => history.redo.back(),
    }
    .and_then(|snapshot| snapshot.weld_restore.as_ref())
    .and_then(|restore| restore.recapture(&graph.0, &simulation, &frozen));
    apply_history_action(action, &mut graph.0, &mut state, &mut history);
}

pub(crate) fn apply_history_action(
    action: HistoryAction,
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
    history: &mut EditorHistory,
) -> bool {
    let mut current = EditorSnapshot::capture(graph, state);
    current.weld_restore = state.history_capture.take();
    let restored = match action {
        HistoryAction::Undo => history.undo(current),
        HistoryAction::Redo => history.redo(current),
    };
    let Some(restored) = restored else {
        state.feedback = Some(match action {
            HistoryAction::Undo => "Nothing to undo".to_owned(),
            HistoryAction::Redo => "Nothing to redo".to_owned(),
        });
        return false;
    };

    *graph = Arc::unwrap_or_clone(restored.graph);
    state.placed_bearings = restored.placed_bearings;
    state.weld_restore = restored.weld_restore;
    cancel_transient_editor_state(graph, state);
    state.construction_mesh_dirty = true;
    state.feedback = Some(match action {
        HistoryAction::Undo => "Undid construction edit".to_owned(),
        HistoryAction::Redo => "Redid construction edit".to_owned(),
    });
    true
}

pub(crate) fn cancel_transient_editor_state(
    graph: &mut ConstructionGraph,
    state: &mut EditorState,
) {
    if graph.pending().is_some() {
        graph
            .apply(BuildCommand::CancelPending)
            .expect("restored pending editor operation can be cancelled");
    }
    state.suspension.controls.dismiss();
    state.suspension.drag = None;
    state.block_drag = None;
    state.pipe_drag = None;
    state.delete_drag = None;
    state.delete_target = None;
    state.region_drag = None;
    state.vertex_drag = None;
    state.feature_drag = None;
    state.wire_drag = None;
    state.edit_context = None;
    clear_hover(state);
}

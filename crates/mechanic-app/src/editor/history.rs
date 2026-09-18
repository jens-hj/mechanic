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

#[cfg(test)]
mod tests {
    use bevy::prelude::{ButtonInput, IVec3, Vec2, Vec3};
    use mechanic_core::{
        BearingDimensions, BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph,
        CuboidSpec, FaceKind, FaceRef, GridRotation, PendingOperation, WeldSpec,
    };

    use crate::builder::{
        BlockVolume, PlacementPlane, SurfaceHit, bearing_attachment_candidate,
        stage_bearing_attachment,
    };
    use crate::controls::GameAction;
    use crate::editor::build_actions::PlacedBearing;
    use crate::editor::history::{
        EditorHistory, EditorSnapshot, HISTORY_CAPACITY, HistoryAction, apply_history_action,
        requested_history_action,
    };
    use crate::editor::hover::{
        BlockAttachment, BlockDrag, DeleteDrag, DeleteTarget, PointerSample,
    };
    use crate::editor::state::EditorState;

    fn spawn_cube(graph: &mut ConstructionGraph, center: IVec3) -> mechanic_core::PartId {
        let spec =
            CuboidSpec::new([4; 3], BuildPose::new(center, GridRotation::default())).unwrap();
        let BuildOutcome::Spawned(part) = graph.apply(BuildCommand::Spawn(spec)).unwrap() else {
            unreachable!()
        };
        part
    }

    #[test]
    fn control_and_command_z_choose_undo_and_shift_redo() {
        for _ in 0..2 {
            let mut keyboard = ButtonInput::default();
            keyboard.press(GameAction::Undo);
            assert_eq!(
                requested_history_action(&keyboard),
                Some(HistoryAction::Undo)
            );

            keyboard.reset_all();
            keyboard.press(GameAction::Redo);
            assert_eq!(
                requested_history_action(&keyboard),
                Some(HistoryAction::Redo)
            );
        }

        let mut keyboard = ButtonInput::default();
        keyboard.press(GameAction::Save);
        assert_eq!(requested_history_action(&keyboard), None);
    }

    #[test]
    #[expect(clippy::too_many_lines)]
    fn bearing_attachment_round_trips_exact_ids_and_cancels_transients() {
        let mut graph = ConstructionGraph::new();
        let support = spawn_cube(&mut graph, IVec3::new(0, 2, 0));
        let socket = PlacedBearing {
            kind: mechanic_core::BearingKind::Rotational,
            axis: Vec3::ZERO,
            source: FaceRef::part(support, FaceKind::PositiveY),
            anchor: Vec3::Y,
            dimensions: BearingDimensions::new(0.70, 0.35).unwrap(),
        };
        let mut state = EditorState {
            placed_bearings: vec![socket],
            ..Default::default()
        };
        let mut history = EditorHistory::default();
        let previous = EditorSnapshot::capture(&graph, &state);
        let candidate = bearing_attachment_candidate(&graph, socket.source, socket.anchor);
        graph = stage_bearing_attachment(
            &graph,
            candidate,
            socket.source,
            socket.anchor,
            socket.dimensions,
        )
        .unwrap();
        history.commit(previous);
        let attached_parts = graph.parts().map(|(id, _)| id).collect::<Vec<_>>();
        let attached_bearings = graph.bearings().map(|(id, _)| id).collect::<Vec<_>>();
        assert_eq!(
            graph.bearings().next().unwrap().1.dimensions,
            socket.dimensions
        );

        graph
            .apply(BuildCommand::BeginPending(PendingOperation::Weld(
                FaceRef::part(support, FaceKind::PositiveX),
            )))
            .unwrap();
        let hit = SurfaceHit {
            distance: 1.0,
            point: Vec3::Y,
            face: FaceRef::part(support, FaceKind::PositiveY),
        };
        state.hovered = Some(hit);
        state.preview = Some(candidate);
        state.block_drag = Some(BlockDrag {
            start: candidate,
            attachment: BlockAttachment::AutoWeld {
                source: hit.face.owner,
            },
            start_guides: Vec::new(),
            press: PointerSample {
                cursor: Vec2::ZERO,
                ray_origin: Vec3::Y,
                ray_direction: Vec3::NEG_Y,
            },
            plane: PlacementPlane::Xz,
            anchor_span: IVec3::ZERO,
            span: IVec3::ZERO,
            last_span: None,
            volume: BlockVolume::new(candidate.spec, IVec3::ZERO).unwrap(),
            error: None,
        });
        state.delete_drag = Some(DeleteDrag {
            start: graph.part(support).copied().unwrap().as_cuboid().unwrap(),
            press: PointerSample {
                cursor: Vec2::ZERO,
                ray_origin: Vec3::Y,
                ray_direction: Vec3::NEG_Y,
            },
            plane: PlacementPlane::Xz,
            anchor_span: IVec3::ZERO,
            span: IVec3::ZERO,
            last_span: None,
            parts: vec![support],
            error: None,
        });
        state.delete_target = Some(DeleteTarget::PlacedBearing(0));

        apply_history_action(HistoryAction::Undo, &mut graph, &mut state, &mut history);

        assert_eq!(graph.part_count(), 1);
        assert_eq!(graph.bearing_count(), 0);
        assert_eq!(state.placed_bearings, vec![socket]);
        assert!(graph.pending().is_none());
        assert!(state.block_drag.is_none());
        assert!(state.delete_drag.is_none());
        assert!(state.delete_target.is_none());
        assert!(state.hovered.is_none());
        assert!(state.preview.is_none());
        assert!(state.construction_mesh_dirty);

        apply_history_action(HistoryAction::Redo, &mut graph, &mut state, &mut history);

        assert_eq!(
            graph.parts().map(|(id, _)| id).collect::<Vec<_>>(),
            attached_parts
        );
        assert_eq!(
            graph.bearings().map(|(id, _)| id).collect::<Vec<_>>(),
            attached_bearings
        );
        assert_eq!(
            graph.bearings().next().unwrap().1.dimensions,
            socket.dimensions
        );
        assert_eq!(state.placed_bearings, vec![socket]);
    }

    #[test]
    fn dragged_deletion_restores_connections_atomically() {
        let mut graph = ConstructionGraph::new();
        let parts = [
            spawn_cube(&mut graph, IVec3::new(0, 2, 0)),
            spawn_cube(&mut graph, IVec3::new(0, 6, 0)),
            spawn_cube(&mut graph, IVec3::new(0, 10, 0)),
        ];
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::ground(),
                second: FaceRef::part(parts[0], FaceKind::NegativeY),
            }))
            .unwrap();
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(parts[0], FaceKind::PositiveY),
                second: FaceRef::part(parts[1], FaceKind::NegativeY),
            }))
            .unwrap();
        graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(parts[1], FaceKind::PositiveY),
                FaceRef::part(parts[2], FaceKind::NegativeY),
                Vec3::new(0.0, 2.0, 0.0),
                Vec3::Y,
            )))
            .unwrap();
        let original_part_ids = graph.parts().map(|(id, _)| id).collect::<Vec<_>>();
        let original_weld_ids = graph.welds().map(|(id, _)| id).collect::<Vec<_>>();
        let original_bearing_ids = graph.bearings().map(|(id, _)| id).collect::<Vec<_>>();
        let mut state = EditorState::default();
        let mut history = EditorHistory::default();
        let previous = EditorSnapshot::capture(&graph, &state);

        graph
            .apply_batch(parts[..2].iter().copied().map(BuildCommand::Remove))
            .unwrap();
        history.commit(previous);
        assert_eq!(history.undo.len(), 1);
        assert_eq!(graph.part_count(), 1);
        assert_eq!(graph.weld_count(), 0);
        assert_eq!(graph.bearing_count(), 0);

        apply_history_action(HistoryAction::Undo, &mut graph, &mut state, &mut history);
        assert_eq!(
            graph.parts().map(|(id, _)| id).collect::<Vec<_>>(),
            original_part_ids
        );
        assert_eq!(
            graph.welds().map(|(id, _)| id).collect::<Vec<_>>(),
            original_weld_ids
        );
        assert_eq!(
            graph.bearings().map(|(id, _)| id).collect::<Vec<_>>(),
            original_bearing_ids
        );
    }

    #[test]
    fn history_is_bounded_and_new_edits_clear_only_redo() {
        let graph = ConstructionGraph::new();
        let mut state = EditorState::default();
        let mut history = EditorHistory::default();
        for _ in 0..=HISTORY_CAPACITY {
            history.commit(EditorSnapshot::capture(&graph, &state));
        }
        assert_eq!(history.undo.len(), HISTORY_CAPACITY);

        apply_history_action(
            HistoryAction::Undo,
            &mut graph.clone(),
            &mut state,
            &mut history,
        );
        assert_eq!(history.redo.len(), 1);
        state.feedback = Some("camera and tool changes are transient".to_owned());
        assert_eq!(history.redo.len(), 1);

        history.commit(EditorSnapshot::capture(&graph, &state));
        assert!(history.redo.is_empty());
        assert_eq!(history.undo.len(), HISTORY_CAPACITY);
    }

    #[test]
    fn empty_history_stacks_report_guidance_without_mutation() {
        let mut graph = ConstructionGraph::new();
        let part = spawn_cube(&mut graph, IVec3::new(0, 2, 0));
        let mut state = EditorState::default();
        let mut history = EditorHistory::default();

        apply_history_action(HistoryAction::Undo, &mut graph, &mut state, &mut history);
        assert_eq!(graph.parts().next().unwrap().0, part);
        assert_eq!(state.feedback.as_deref(), Some("Nothing to undo"));
        apply_history_action(HistoryAction::Redo, &mut graph, &mut state, &mut history);
        assert_eq!(state.feedback.as_deref(), Some("Nothing to redo"));
    }
}

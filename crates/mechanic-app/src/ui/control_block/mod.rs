//! The control block panel: one lane per driven joint.
//!
//! The construction graph stays the truth about the machine. This module is
//! the seam: it reads the graph into a [`PanelModel`] the tree renders, and
//! folds the [`Intent`]s the tree produces back into build commands. Nothing
//! in the view touches the graph, and nothing in the graph knows the panel
//! exists.
//!
//! The panel does not mount itself — [`crate::ui`] owns the tree and hangs this
//! view in it, along with every other panel.

mod geometry;
mod model;
mod view;

use bevy::prelude::*;
use mechanic_core::{BuildCommand, DriveLinkId, PartId};

use crate::control_panel::{ControlPanelState, panel_rows, set_row_commands};
use crate::{AppSimulation, EditorGraph, EditorHistory, EditorSnapshot, EditorState};

pub(crate) use model::{Intent, PanelEdit, PanelModel};
pub(crate) use view::{Handles, panel};

use model::{LaneModel, apply_edit};

/// The joint the panel is pointing out in the world, if any.
///
/// A plain resource rather than something read off the tree, so the systems
/// that draw the world do not have to be pinned to the main thread.
#[derive(Resource, Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LocatedJoint(pub(crate) Option<DriveLinkId>);

/// The construction the panel edits, and everything one edit touches.
#[derive(bevy::ecs::system::SystemParam)]
pub(crate) struct EditTarget<'w> {
    graph: ResMut<'w, EditorGraph>,
    editor: ResMut<'w, EditorState>,
    history: ResMut<'w, EditorHistory>,
    simulation: Res<'w, AppSimulation>,
}

/// Writes one intent to every wire behind the joint it names.
///
/// While building this is an undoable edit like any other, except part-way
/// through a drag: a drag writes on every pointer move, and only the move that
/// ends it belongs in history. While simulating it skips history entirely and
/// marks the drive rows dirty, which is the one write running GPU state takes.
pub(crate) fn write_joint(panel: &ControlPanelState, target: &mut EditTarget, intent: &Intent) {
    let Some(controller) = panel.controller() else {
        return;
    };
    write_to(controller, target, intent);
}

/// The half of [`write_joint`] that already knows which control block is open.
fn write_to(controller: PartId, target: &mut EditTarget, intent: &Intent) {
    let rows = panel_rows(&target.graph.0, controller);
    let Some(row) = rows.iter().find(|row| row.links.contains(&intent.lane)) else {
        return;
    };
    let Some(spec) = target.graph.0.drive_link(row.primary).copied() else {
        return;
    };
    let Some((limits, program, name)) =
        apply_edit(spec.limits, spec.program, spec.name, &intent.edit)
    else {
        return;
    };
    let commands: Vec<BuildCommand> = set_row_commands(row, limits, program, name);
    let previous = EditorSnapshot::capture(&target.graph.0, &target.editor);
    match target.graph.0.apply_batch(commands) {
        Ok(_) => {
            if target.simulation.is_running() {
                target.editor.drive_rows_dirty = true;
            } else {
                if !intent.transient {
                    target.history.commit(previous);
                }
                target.editor.construction_mesh_dirty = true;
            }
        }
        Err(error) => target.editor.feedback = Some(error.to_string()),
    }
}

/// Reads the open control block's wires into what the panel draws.
pub(crate) fn capture(panel: &ControlPanelState, graph: &EditorGraph) -> PanelModel {
    let Some(controller) = panel.controller() else {
        return PanelModel::default();
    };
    let lanes = panel_rows(&graph.0, controller)
        .iter()
        .enumerate()
        .filter_map(|(index, row)| {
            let spec = graph.0.drive_link(row.primary)?;
            Some(LaneModel::capture(
                row.primary,
                index + 1,
                spec.limits,
                &spec.program,
                &spec.name,
            ))
        })
        .collect();
    PanelModel { open: true, lanes }
}

/// Binds the next key pressed while a keycap is waiting for one.
///
/// The panel owns the whole keyboard while it is open, so this reads the raw
/// key rather than going through the tree: nothing else may act on the press
/// that binds a state.
pub(crate) fn capture_key(handles: &Handles, keyboard: &ButtonInput<KeyCode>) {
    let Some((link, slot)) = handles.capturing.get_untracked() else {
        return;
    };
    if keyboard.just_pressed(KeyCode::Escape) {
        handles.capturing.set(None);
        return;
    }
    // `E` opens and closes the panel, so binding it would take the way out.
    for pressed in keyboard.get_just_pressed() {
        if *pressed == KeyCode::KeyE {
            continue;
        }
        let Some(key) = crate::sequencer::drive_key(*pressed) else {
            continue;
        };
        handles.edit(
            link,
            PanelEdit::BindKey {
                state: slot,
                key: key.symbol(),
            },
        );
        handles.capturing.set(None);
        return;
    }
}

#[cfg(test)]
mod tests {
    use bevy::math::{IVec3, Vec3};
    use mechanic_core::{
        BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, ControllerSpec,
        CuboidSpec, DriveLinkId, DriveLinkSpec, FaceKind, FaceRef, GridRotation,
    };
    use mosaic_core::{Rect, Vector2};
    use mosaic_widgets::input::{Key, PointerButton, PointerEventKind};

    use super::geometry;
    use super::model::{Mode, PanelEdit, StateModel};
    use crate::ui::UiIntent;
    use crate::ui::testing::{Overlay, away};

    /// A control block driving one bearing.
    fn wired() -> (ConstructionGraph, DriveLinkId) {
        let mut graph = ConstructionGraph::new();
        let cuboid = |dimensions: [u8; 3], units: IVec3| {
            CuboidSpec::new(dimensions, BuildPose::new(units, GridRotation::default()))
                .expect("test dimensions are in range")
        };
        let spawned = |outcome: BuildOutcome| match outcome {
            BuildOutcome::Spawned(part) => part,
            other => panic!("expected a spawn, got {other:?}"),
        };
        let base = spawned(
            graph
                .apply(BuildCommand::Spawn(cuboid([4, 2, 4], IVec3::new(0, 1, 0))))
                .expect("the base spawns"),
        );
        let rotor = spawned(
            graph
                .apply(BuildCommand::Spawn(cuboid([2, 2, 2], IVec3::new(0, 3, 0))))
                .expect("the rotor spawns"),
        );
        let controller = spawned(
            graph
                .apply(BuildCommand::SpawnController(ControllerSpec::new(
                    BuildPose::from_half_grid(IVec3::new(2, 5, 0), GridRotation::default()),
                )))
                .expect("the control block spawns"),
        );
        let BuildOutcome::BearingAdded(bearing) = graph
            .apply(BuildCommand::AddBearing(BearingSpec::new(
                FaceRef::part(base, FaceKind::PositiveY),
                FaceRef::part(rotor, FaceKind::NegativeY),
                Vec3::new(0.0, 0.5, 0.0),
                Vec3::Y,
            )))
            .expect("the bearing is added")
        else {
            panic!("expected a bearing outcome");
        };
        graph
            .apply(BuildCommand::AddDriveLink(DriveLinkSpec::new(
                controller, bearing,
            )))
            .expect("the wire is added");
        let link = graph
            .controller_links(controller)
            .next()
            .expect("the wire is there")
            .0;
        (graph, link)
    }

    /// The overlay with a control block open on one wired joint.
    fn open() -> (Overlay, DriveLinkId) {
        let (graph, link) = wired();
        let controller = graph
            .drive_link(link)
            .expect("the wire is there")
            .controller;
        let overlay = Overlay::mount();
        let mut state = crate::control_panel::ControlPanelState::default();
        state.open(controller);
        overlay
            .handles
            .block
            .model
            .set(super::capture(&state, &crate::EditorGraph(graph)));
        overlay.settle();
        (overlay, link)
    }

    /// Gives the joint travel limits and puts its first state on an angle,
    /// which is what brings the travel grips out.
    fn limit_travel(overlay: &Overlay, low: f32, high: f32) {
        overlay.handles.block.model.update(|model| {
            let Some(lane) = model.lanes.first_mut() else {
                return;
            };
            lane.travel = Some((low, high));
            if let Some(state) = lane.states.first_mut() {
                state.mode = Mode::Angle;
            }
        });
        overlay.settle();
    }

    /// The drive edits the panel asked for.
    fn edits(overlay: &Overlay) -> Vec<super::Intent> {
        overlay
            .intents()
            .into_iter()
            .filter_map(|intent| match intent {
                UiIntent::Drive(edit) => Some(edit),
                _ => None,
            })
            .collect()
    }

    /// The dial's box, which everything on the dial is placed from.
    fn dial_box(overlay: &Overlay) -> Rect {
        overlay
            .rects()
            .into_iter()
            .find(|(_, rect)| {
                (rect.size.width - 132.0).abs() < 0.5 && (rect.size.height - 132.0).abs() < 0.5
            })
            .expect("the dial is laid out")
            .1
    }

    #[test]
    fn a_mounted_panel_builds_a_tree_rather_than_an_empty_root() {
        let (overlay, link) = open();
        let states = overlay
            .handles
            .block
            .model
            .with(|model| model.lane(link).map(|lane| lane.states.len()))
            .expect("the joint is in the model");
        assert_eq!(states, 1, "a fresh wire holds one state");
        assert!(
            overlay.element_count() > 20,
            "the panel builds a tree; it had {} elements",
            overlay.element_count(),
        );
    }

    /// The bug this guards against: a full-bleed grouping wrapper marked
    /// `nohit` takes its whole subtree out of reach with it, and every control
    /// on the card stops responding while still looking perfectly right.
    #[test]
    fn the_controls_on_a_card_can_be_reached_by_the_pointer() {
        let (overlay, _link) = open();
        let boxes = overlay.reachable_boxes();
        for (what, width, height) in [
            ("the keycap", 46.0, 34.0),
            ("a mode switch", 26.0, 20.0),
            ("the delete button", 20.0, 20.0),
            ("the dial", 132.0, 132.0),
            ("a port", 22.0, 22.0),
            ("a preset", 34.0, 34.0),
        ] {
            assert!(
                overlay.reaches_box(&boxes, width, height),
                "{what} must take the pointer; \
                 nothing {width}×{height} was reachable anywhere in the panel",
            );
        }
    }

    #[test]
    fn the_header_close_button_requests_that_the_panel_close() {
        let (overlay, _link) = open();
        let close = overlay
            .reachable_boxes()
            .into_iter()
            .find(|rect| {
                (rect.size.width - 32.0).abs() < 0.5 && (rect.size.height - 32.0).abs() < 0.5
            })
            .expect("the close button is reachable in the header");

        overlay.click(close.center());

        assert_eq!(overlay.intents(), vec![UiIntent::CloseControlPanel]);
    }

    #[test]
    fn the_header_close_icon_is_geometrically_centred() {
        let (overlay, _link) = open();
        let tree = overlay.rects();
        let (button_index, (button_depth, button)) = tree
            .iter()
            .enumerate()
            .find(|(_, (_, rect))| {
                (rect.size.width - 32.0).abs() < 0.5 && (rect.size.height - 32.0).abs() < 0.5
            })
            .expect("the close button is in the header");
        let icon = tree[button_index + 1..]
            .iter()
            .take_while(|(depth, _)| depth > button_depth)
            .find(|(_, rect)| {
                (rect.size.width - 18.0).abs() < 0.5 && (rect.size.height - 18.0).abs() < 0.5
            })
            .expect("the close button contains its canvas")
            .1;

        assert!(away(icon.center(), button.center()) < 0.5);
    }

    /// The bug this guards against: a canvas with no size of its own shrinks
    /// onto the drawing inside it and pulls that drawing flush with its own
    /// corner. The dial then sits off the number at its centre by however far
    /// the sweep happened to reach — and every mark on it, the travel grips
    /// included, moves out from under the pointer with it.
    #[test]
    fn the_dial_is_drawn_from_the_corner_of_its_own_box() {
        let (overlay, _link) = open();
        let tree = overlay.rects();
        let dial = tree
            .iter()
            .position(|(_, rect)| {
                (rect.size.width - 132.0).abs() < 0.5 && (rect.size.height - 132.0).abs() < 0.5
            })
            .expect("the dial is laid out");
        let (depth, box_) = tree[dial];
        let (canvas_depth, canvas) = tree[dial + 1];
        assert_eq!(canvas_depth, depth + 1);
        assert_eq!(canvas, box_, "the drawing surface is the dial's own box");
        for (mark_depth, mark) in &tree[dial + 2..] {
            if *mark_depth <= canvas_depth {
                break;
            }
            assert_eq!(
                mark.origin, box_.origin,
                "every mark on the dial is placed from the dial's own corner",
            );
        }
    }

    /// A grip is drawn on the dial and grabbed by a box of its own, so the two
    /// have to agree about where it is.
    #[test]
    fn a_travel_grip_is_grabbed_where_it_is_drawn() {
        let (overlay, _link) = open();
        limit_travel(&overlay, -45.0, 60.0);
        let dial = dial_box(&overlay);
        // Only the boxes out on the dial's rim: an 18×18 box is a common
        // enough size that the header's own marks would otherwise count.
        let grips: Vec<Rect> = overlay
            .reachable_boxes()
            .into_iter()
            .filter(|rect| {
                (rect.size.width - 18.0).abs() < 0.5
                    && (rect.size.height - 18.0).abs() < 0.5
                    && away(rect.center(), dial.center()) < geometry::GRIP_RADIUS + 1.0
            })
            .collect();
        assert_eq!(grips.len(), 2, "both ends of the travel take the pointer");
        for (degrees, grip) in [-45.0_f32, 60.0].into_iter().zip(grips) {
            let (x, y) = geometry::polar(geometry::GRIP_RADIUS, degrees);
            let wanted = dial.origin + Vector2::new(x, y);
            let found = grip.center();
            assert!(
                (found.x - wanted.x).abs() < 0.5 && (found.y - wanted.y).abs() < 0.5,
                "the grip for {degrees}° is grabbed at {found:?}, but drawn at {wanted:?}",
            );
        }
    }

    /// Travel is a switch like any other, so its grips come and go — and an
    /// element built once outside the branch that shows it is freed the first
    /// time that branch closes.
    #[test]
    fn travel_grips_survive_being_switched_off_and_on() {
        let (overlay, _link) = open();
        for _ in 0..2 {
            limit_travel(&overlay, -45.0, 60.0);
            overlay.handles.block.model.update(|model| {
                if let Some(lane) = model.lanes.first_mut() {
                    lane.travel = None;
                }
            });
            overlay.settle();
        }
    }

    /// Dragging the dial is how a state's number is set without typing it.
    #[test]
    fn dragging_the_dial_moves_the_number_it_reads() {
        let (overlay, link) = open();
        let centre = dial_box(&overlay).center();
        overlay.drag(
            centre + Vector2::new(0.0, -40.0),
            centre + Vector2::new(40.0, 0.0),
        );
        let queued = edits(&overlay);
        assert!(
            queued.iter().any(|edit| edit.lane == link
                && matches!(edit.edit, PanelEdit::SetValue { state: 0, .. })),
            "a quarter turn round the dial sets the state's number",
        );
        assert!(
            queued.iter().all(|edit| edit.transient),
            "nothing part-way through a drag belongs in history",
        );
    }

    /// A grip sits inside the dial, which reads the same gesture as a change of
    /// value — so the grip has to keep the pointer to itself.
    #[test]
    fn dragging_a_travel_grip_moves_the_limit_and_not_the_reading() {
        let (overlay, link) = open();
        limit_travel(&overlay, -45.0, 60.0);
        let dial = dial_box(&overlay);
        let (x, y) = geometry::polar(geometry::GRIP_RADIUS, -45.0);
        let grip = dial.origin + Vector2::new(x, y);
        // Round to nine o'clock, which is a limit of -90°.
        overlay.drag(
            grip,
            dial.center() + Vector2::new(-geometry::GRIP_RADIUS, 0.0),
        );

        let queued = edits(&overlay);
        assert!(
            queued.iter().any(|edit| edit.lane == link
                && matches!(
                    edit.edit,
                    PanelEdit::SetTravel { min, max }
                        if (min + 90.0).abs() < 0.01 && (max - 60.0).abs() < 0.01
                )),
            "the grip moves the end of the travel it belongs to; got {queued:?}",
        );
        assert!(
            queued
                .iter()
                .all(|edit| !matches!(edit.edit, PanelEdit::SetValue { .. })),
            "and the dial underneath it must not read the same gesture as a value",
        );
    }

    /// Dragging out of a port draws the wire it is about to make, rather than
    /// leaving the pointer to be aimed at nothing.
    #[test]
    fn dragging_out_of_a_port_draws_the_wire_before_it_lands() {
        let (overlay, link) = open();
        overlay.handles.block.model.update(|model| {
            let Some(lane) = model.lanes.first_mut() else {
                return;
            };
            lane.states[0].key = Some('W');
            lane.states.push(StateModel {
                mode: Mode::Speed,
                value: 90.0,
                key: None,
                release: None,
                dwell: None,
            });
        });
        overlay.settle();

        let resting = overlay.element_count();
        let card = overlay
            .rects()
            .into_iter()
            .find(|(_, rect)| {
                (rect.size.width - 204.0).abs() < 0.5 && (rect.size.height - 214.0).abs() < 0.5
            })
            .expect("a card is laid out")
            .1;
        // The release port hangs off the card's top-right corner. Picked by
        // where it sits rather than by its size: a 22-pixel box is also what
        // the header's legend tiles measure.
        let corner = Vector2::new(card.origin.x + card.size.width, card.origin.y);
        let port = overlay
            .reachable_boxes()
            .into_iter()
            .find(|rect| {
                (rect.size.width - 22.0).abs() < 0.5
                    && (rect.size.height - 22.0).abs() < 0.5
                    && away(rect.center(), corner) < 2.0
            })
            .expect("the release port is reachable");
        let second = Vector2::new(
            card.center().x + geometry::NODE_W + geometry::GAP,
            card.center().y,
        );

        overlay.drag(port.center(), second);
        assert!(
            overlay.element_count() > resting,
            "the wire being dragged is drawn while the pointer holds it",
        );

        overlay.dispatch(PointerEventKind::Up(PointerButton::Primary), second);
        assert_eq!(
            overlay.element_count(),
            resting,
            "and is put away once it is let go",
        );
        let queued = edits(&overlay);
        assert!(
            queued.iter().any(|edit| edit.lane == link
                && matches!(
                    edit.edit,
                    PanelEdit::SetRelease {
                        state: 0,
                        target: Some(1)
                    }
                )),
            "letting go on a card is what wires the state to it; got {queued:?}",
        );
    }

    /// The bug this guards against: an element built once and then adopted into
    /// a branch is freed when that branch closes, so opening the field a second
    /// time reaches a layout node that no longer exists and the whole app goes
    /// down.
    #[test]
    fn a_number_chip_survives_being_opened_twice() {
        let (overlay, _link) = open();
        let chip = overlay
            .reachable_boxes()
            .into_iter()
            .find(|rect| {
                (rect.size.width - 114.0).abs() < 0.5 && (rect.size.height - 44.0).abs() < 0.5
            })
            .expect("the speed chip is reachable");

        for _ in 0..2 {
            // Opening the field, then putting the pointer in it so the keyboard
            // reaches it, then giving up on the edit.
            overlay.click(chip.center());
            overlay.click(chip.center());
            overlay.press(Key::Escape);
        }

        // And the edit itself still lands, which is what the field is for.
        overlay.click(chip.center());
        overlay.click(chip.center());
        overlay.press(Key::Character("3".to_owned()));
        overlay.press(Key::Enter);
        let queued = edits(&overlay);
        assert!(
            queued
                .iter()
                .any(|edit| matches!(edit.edit, PanelEdit::SetMaxSpeed(_))),
            "typing a number into the speed chip sets the joint's ceiling; got {queued:?}",
        );
    }

    #[test]
    fn clicking_an_unbound_keycap_arms_a_key_capture() {
        let (overlay, link) = open();
        assert_eq!(overlay.handles.block.capturing.get_untracked(), None);

        let keycap = overlay
            .reachable_boxes()
            .into_iter()
            .find(|rect| {
                (rect.size.width - 46.0).abs() < 0.5 && (rect.size.height - 34.0).abs() < 0.5
            })
            .expect("the keycap is reachable");
        overlay.click(keycap.center());

        assert_eq!(
            overlay.handles.block.capturing.get_untracked(),
            Some((link, 0)),
            "clicking an empty keycap is how a state waits for its key",
        );
    }
}

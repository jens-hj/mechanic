use bevy::math::{IVec3, Vec3};
use bevy_mosaic::ui::TextStyle;
use mechanic_core::{
    ActuatorAssignment, BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph,
    ControllerSpec, CuboidSpec, DriveLinkId, DriveLinkSpec, EngineKind, FaceKind, FaceRef,
    GearboxConfig, GridRotation,
};
use mosaic_core::{Rect, Vector2};
use mosaic_widgets::input::{Key, Modifiers, PointerButton, PointerEventKind};

use super::geometry;
use super::model::{BearingSlots, EngineLaneModel, Mode, PanelEdit, StateModel};
use crate::ui::UiIntent;
use crate::ui::testing::{Overlay, away};
use crate::ui::theme::typeface;

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

/// Exercises the same graph writer as the controller's preset buttons.
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "hardware fixture plus assertions across both gear ranges"
)]
fn drive_preset_uses_two_gas_engines_and_the_full_gear_range() {
    use bevy::ecs::system::SystemState;
    use bevy::prelude::World;
    use mechanic_core::{DriveTarget, EngineSpec, WeldSpec};

    let (mut graph, link) = wired();
    let controller = graph.drive_link(link).unwrap().controller;
    for (position, first, second) in [
        (
            IVec3::new(2, 9, 0),
            FaceKind::PositiveY,
            FaceKind::NegativeY,
        ),
        (
            IVec3::new(6, 5, 0),
            FaceKind::PositiveX,
            FaceKind::NegativeX,
        ),
    ] {
        let BuildOutcome::Spawned(engine) = graph
            .apply(BuildCommand::SpawnEngine(EngineSpec::new(
                EngineKind::Gas,
                BuildPose::from_half_grid(position, GridRotation::default()),
            )))
            .unwrap()
        else {
            panic!("engine");
        };
        graph
            .apply(BuildCommand::Weld(WeldSpec {
                first: FaceRef::part(controller, first),
                second: FaceRef::part(engine, second),
            }))
            .unwrap();
        let mut parent = engine;
        for _ in 0..2 {
            let spec = graph.next_transmission_spec(parent).unwrap();
            let BuildOutcome::Spawned(child) = graph
                .apply(BuildCommand::AttachTransmission { parent, spec })
                .unwrap()
            else {
                panic!("transmission");
            };
            parent = child;
        }
    }
    assert_eq!(graph.actuator_inventory(controller).unwrap().gas_engines, 2);
    let mut world = World::new();
    world.insert_resource(crate::editor::state::EditorGraph(graph));
    world.init_resource::<crate::editor::state::EditorState>();
    world.init_resource::<crate::editor::history::EditorHistory>();
    world.init_resource::<crate::simulation::state::AppSimulation>();
    let mut system = SystemState::<super::EditTarget>::new(&mut world);
    let intent = super::Intent {
        lane: link,
        edit: PanelEdit::ApplyPreset(super::model::Preset::Drive),
        transient: false,
    };
    for ratios in [vec![4.0, 3.0, 1.0], vec![4.0, 3.0, 0.25]] {
        world
            .resource_mut::<crate::editor::state::EditorGraph>()
            .0
            .apply(BuildCommand::SetGearboxRatios {
                controller,
                kind: EngineKind::Gas,
                ratios: ratios.clone(),
            })
            .unwrap();
        super::write_to(
            controller,
            &mut system.get_mut(&mut world).unwrap(),
            &intent,
        );
        let graph = &world.resource::<crate::editor::state::EditorGraph>().0;
        let drive = graph.drive_link(link).unwrap();
        let top = mechanic_core::rpm_to_rad_s(EngineKind::Gas.no_load_rpm()) / ratios[2];
        assert!(
            matches!(drive.program.state(1).unwrap().target(), DriveTarget::Speed(speed) if (speed - top).abs() < 0.0001),
            "W must request {top} rad/s, got {:?}",
            drive.program
        );
        assert!(
            matches!(drive.program.state(2).unwrap().target(), DriveTarget::Speed(speed) if (speed + top * 0.7).abs() < 0.0001)
        );
        assert_eq!(
            drive
                .program
                .state(1)
                .unwrap()
                .trigger()
                .unwrap()
                .key()
                .symbol(),
            'W'
        );
        assert_eq!(
            drive
                .program
                .state(2)
                .unwrap()
                .trigger()
                .unwrap()
                .key()
                .symbol(),
            'S'
        );
        assert!(
            world
                .resource::<crate::editor::state::EditorState>()
                .feedback
                .is_none()
        );
        graph.compile().unwrap();
    }
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
    overlay.handles.block.model.set(super::capture(
        &state,
        &crate::editor::state::EditorGraph(graph),
        &crate::sequencer::GearboxRuntime::default(),
        false,
    ));
    overlay.settle();
    (overlay, link)
}

/// The overlay with a five-speed gas engine line and no joint rows.
fn open_gas_gearbox() -> Overlay {
    let (overlay, _link) = open();
    overlay.handles.block.model.update(|model| {
        model.engine_lanes = vec![EngineLaneModel {
            kind: EngineKind::Gas,
            engine_count: 1,
            combined_stall_torque: EngineKind::Gas.stall_torque_newton_meters(),
            base_rpm: EngineKind::Gas.no_load_rpm(),
            slots: BearingSlots::new(0, EngineKind::Gas.bearing_capacity()),
            transmission_depth: Some(4),
            physical_depths: vec![4],
            mismatch: false,
            config: Some(GearboxConfig::for_depth(4, true)),
            active_gear: None,
            binding_conflict: false,
        }];
    });
    overlay.settle();
    overlay
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
        .find(|rect| (rect.size.width - 32.0).abs() < 0.5 && (rect.size.height - 32.0).abs() < 0.5)
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

#[test]
fn the_header_keeps_all_three_capacity_stats_without_hardware() {
    let (overlay, _link) = open();
    let stats = overlay
        .rects()
        .into_iter()
        .filter(|(_, rect)| {
            (rect.size.width - 110.0).abs() < 0.5 && (rect.size.height - 38.0).abs() < 0.5
        })
        .count();

    assert_eq!(stats, 3, "electric, gas, and Servo stats stay visible");
}

#[test]
fn compact_powertrain_text_fits_its_tiles_in_the_body_font() {
    let (overlay, link) = open();
    overlay.handles.block.model.update(|model| {
        let lane = model.lanes.first_mut().expect("the joint is present");
        lane.actuator = ActuatorAssignment::motor(100, 100).expect("valid percentages");
        lane.torque = 9_999.0;
    });
    let (heading, speed, torque, electric, gas) = overlay.handles.block.model.with(|model| {
        let lane = model.lane(link).expect("the joint is present");
        (
            lane.torque_label(),
            lane.speed_text(),
            lane.torque_text(),
            lane.electric_text(),
            lane.gas_text(),
        )
    });
    let heading_style = TextStyle::new(8.0)
        .family(mosaic_core::theme::typed(
            typeface.body,
            bevy_mosaic::ui::FontFamily::default,
        ))
        .weight(700)
        .letter_spacing(0.45);
    let value_style = TextStyle::new(11.0)
        .family(mosaic_core::theme::typed(
            typeface.body,
            bevy_mosaic::ui::FontFamily::default,
        ))
        .letter_spacing(-0.12);

    for text in ["ELECTRIC", "SPEED", heading] {
        assert!(
            overlay.text_width(text, &heading_style) <= super::view::CAPABILITY_TEXT_WIDTH,
            "capability heading {text:?} overflows its tile",
        );
    }
    for text in [
        speed.as_str(),
        torque.as_str(),
        electric.as_str(),
        gas.as_str(),
    ] {
        assert!(
            overlay.text_width(text, &value_style) <= super::view::CAPABILITY_TEXT_WIDTH,
            "capability value {text:?} overflows its tile",
        );
    }

    let capacity_style = TextStyle::new(9.0).family(mosaic_core::theme::typed(
        typeface.body,
        bevy_mosaic::ui::FontFamily::default,
    ));
    let capacity = BearingSlots::new(99, 99).text();
    assert!(
        overlay.text_width(&capacity, &capacity_style) <= super::view::CAPACITY_TEXT_WIDTH,
        "capacity value {capacity:?} overflows its header tile",
    );
}

#[test]
fn gas_direction_split_uses_the_available_lane_width() {
    let overlay = open_gas_gearbox();
    let gear_card_rects = overlay
        .rects()
        .into_iter()
        .map(|(_, rect)| rect)
        .filter(|rect| {
            (rect.size.width - 128.0).abs() < 0.5 && (rect.size.height - 82.0).abs() < 0.5
        })
        .collect::<Vec<_>>();
    let widest_track = overlay
        .rects()
        .into_iter()
        .map(|(_, rect)| rect)
        .filter(|rect| (rect.size.height - 30.0).abs() < 0.5)
        .map(|rect| rect.size.width)
        .fold(0.0_f32, f32::max);

    assert!(
        widest_track > crate::ui::testing::VIEWPORT.width * 0.5,
        "the direction track should fill the gearbox lane; it was only {widest_track}px wide",
    );
    assert_eq!(
        gear_card_rects.len(),
        5,
        "all five ratio cards should be visible"
    );
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
        queued
            .iter()
            .any(|edit| edit.lane == link
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

/// The speed capability is a stable control even while its displayed unit
/// changes beneath it.
#[test]
fn the_speed_chip_survives_repeated_unit_toggles() {
    let (overlay, _link) = open();
    let chip = overlay
        .reachable_boxes()
        .into_iter()
        .find(|rect| (rect.size.width - 114.0).abs() < 0.5 && (rect.size.height - 44.0).abs() < 0.5)
        .expect("the speed chip is reachable");

    for _ in 0..3 {
        overlay.click(chip.center());
    }

    let queued = edits(&overlay);
    assert!(
        queued
            .iter()
            .filter(|edit| matches!(edit.edit, PanelEdit::ToggleSpeedUnit))
            .count()
            >= 3,
        "clicking the hardware speed chip toggles its unit; got {queued:?}",
    );
}

#[test]
fn clicking_an_unbound_keycap_arms_a_key_capture() {
    let (overlay, link) = open();
    assert_eq!(overlay.handles.block.capturing.get_untracked(), None);

    let keycap = overlay
        .reachable_boxes()
        .into_iter()
        .find(|rect| (rect.size.width - 46.0).abs() < 0.5 && (rect.size.height - 34.0).abs() < 0.5)
        .expect("the keycap is reachable");
    overlay.click(keycap.center());

    assert_eq!(
        overlay.handles.block.capturing.get_untracked(),
        Some((link, 0)),
        "clicking an empty keycap is how a state waits for its key",
    );
}

/// A lane whose only state waits a second before handing back to itself,
/// which is what puts a dwell pill on the wire below the cards.
fn waiting(overlay: &Overlay, seconds: f32) {
    overlay.handles.block.model.update(|model| {
        let Some(lane) = model.lanes.first_mut() else {
            return;
        };
        if let Some(state) = lane.states.first_mut() {
            state.dwell = Some((seconds, 0));
        }
        lane.dwell_wires = vec![super::model::WireModel {
            source: 0,
            target: 0,
            label: format!("{} s", super::model::dwell_text(seconds)),
        }];
    });
    overlay.settle();
}

/// The dwell pill on the wire below the cards: the widest of the pill-high
/// boxes down there, the narrower ones being its own icon and text.
fn dwell_pill(overlay: &Overlay) -> Rect {
    overlay
        .rects()
        .into_iter()
        .map(|(_, rect)| rect)
        .filter(|rect| {
            (rect.size.height - 24.0).abs() < 0.5
                && rect.size.width < 200.0
                && rect.origin.y > geometry::NODE_H
        })
        .max_by(|a, b| a.size.width.total_cmp(&b.size.width))
        .expect("the dwell pill is laid out below the cards")
}

/// The bug this guards against: the field asked for a share of the leftover
/// room inside a pill that hugs its contents and has none, so it laid out
/// zero pixels wide — nothing to see, and nothing to put a caret in.
#[test]
fn the_dwell_field_opens_wide_enough_to_type_in() {
    let (overlay, _link) = open();
    waiting(&overlay, 1.0);
    let pill = dwell_pill(&overlay);

    overlay.click(pill.center());

    let widths: Vec<f32> = overlay
        .rects()
        .into_iter()
        .map(|(_, rect)| rect)
        .filter(|rect| (rect.size.height - 24.0).abs() < 0.5 && rect.origin.y > geometry::NODE_H)
        .map(|rect| rect.size.width)
        .collect();
    assert!(
        widths.iter().any(|width| *width > 20.0 && *width < 100.0),
        "the opened field must have room to type in; the boxes on the wire were {widths:?}",
    );
}

#[test]
fn typing_a_dwell_and_pressing_enter_sets_it() {
    let (overlay, link) = open();
    waiting(&overlay, 1.0);
    let pill = dwell_pill(&overlay);

    overlay.click(pill.center());
    overlay.type_text("2.5");
    overlay.press(Key::Enter);

    assert_eq!(
        edits(&overlay)
            .into_iter()
            .map(|intent| intent.edit)
            .collect::<Vec<_>>(),
        vec![PanelEdit::SetDwell {
            state: 0,
            seconds: 2.5
        }],
        "the joint is {link:?}",
    );
}

/// Opening the field, backing out, and opening it again: the second open
/// builds a fresh field rather than re-adopting the freed one, and it takes
/// the keyboard just as the first did.
#[test]
fn a_dwell_field_can_be_backed_out_of_and_opened_again() {
    let (overlay, _link) = open();
    waiting(&overlay, 1.0);
    let pill = dwell_pill(&overlay);

    overlay.click(pill.center());
    overlay.press(Key::Escape);
    overlay.click(dwell_pill(&overlay).center());
    overlay.type_text("3");
    overlay.press(Key::Enter);

    assert_eq!(last_dwell(&overlay), Some(3.0));
}

#[test]
fn dragging_a_dwell_pill_steps_the_wait_by_whole_seconds() {
    let (overlay, _link) = open();
    waiting(&overlay, 1.0);
    let pill = dwell_pill(&overlay);
    let at = pill.center();

    overlay.drag_held(at, at + Vector2::new(36.0, 0.0), Modifiers::default());

    let seconds = last_dwell(&overlay).expect("the drag set a dwell");
    assert!(
        (seconds - 4.0).abs() < 0.001,
        "three steps right adds three seconds; got {seconds}",
    );
}

#[test]
fn dragging_a_dwell_pill_down_shortens_the_wait() {
    let (overlay, _link) = open();
    waiting(&overlay, 5.0);
    let pill = dwell_pill(&overlay);
    let at = pill.center();

    overlay.drag_held(at, at + Vector2::new(0.0, 24.0), Modifiers::default());

    let seconds = last_dwell(&overlay).expect("the drag set a dwell");
    assert!(
        (seconds - 3.0).abs() < 0.001,
        "two steps down takes two seconds off; got {seconds}",
    );
}

#[test]
fn a_held_modifier_makes_a_scrubbed_step_finer() {
    for (modifiers, step) in [
        (
            Modifiers {
                shift: true,
                ..Modifiers::default()
            },
            0.25,
        ),
        (
            Modifiers {
                ctrl: true,
                ..Modifiers::default()
            },
            0.1,
        ),
    ] {
        let (overlay, _link) = open();
        waiting(&overlay, 1.0);
        let pill = dwell_pill(&overlay);
        let at = pill.center();

        overlay.drag_held(at, at + Vector2::new(24.0, 0.0), modifiers);

        let seconds = last_dwell(&overlay).expect("the drag set a dwell");
        assert!(
            (seconds - (1.0 + 2.0 * step)).abs() < 0.001,
            "two steps of {step} from one second; got {seconds}",
        );
    }
}

/// The wait the last drive edit asked for.
fn last_dwell(overlay: &Overlay) -> Option<f32> {
    edits(overlay)
        .into_iter()
        .rev()
        .find_map(|intent| match intent.edit {
            PanelEdit::SetDwell { seconds, .. } => Some(seconds),
            _ => None,
        })
}

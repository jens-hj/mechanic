use super::*;

#[test]
fn hammer_bindings_only_apply_in_the_hammer_context() {
    use crate::hotbar::MainTool;
    let mut controls = Controls::default();
    let mut keyboard = ButtonInput::default();
    let mouse = ButtonInput::default();
    for (action, key) in [
        (GameAction::FreezeCreation, KeyCode::KeyF),
        (GameAction::RaiseFrozenCreation, KeyCode::ArrowUp),
        (GameAction::LowerFrozenCreation, KeyCode::ArrowDown),
    ] {
        assert!(GameAction::ALL.contains(&action));
        assert_eq!(controls.binding(action).0[0], Some(InputChord::key(key)));
        assert_eq!(action.group(), "Hammer");
        assert!(!controls.conflicts(action));
    }
    keyboard.press(KeyCode::KeyF);
    keyboard.press(KeyCode::ArrowUp);
    let input = ActionInput::without_wheel(&controls, &keyboard, &mouse);
    assert!(!controls.conflicts(GameAction::FreezeCreation));
    assert!(!controls.conflicts(GameAction::RaiseFrozenCreation));
    assert!(!input.just_pressed(GameAction::FreezeCreation));
    assert!(input.just_pressed(GameAction::PipeTurn));
    assert!(input.just_pressed_for_tool(GameAction::FreezeCreation, MainTool::Hammer));
    assert!(input.pressed_for_tool(GameAction::RaiseFrozenCreation, MainTool::Hammer));
    assert!(!input.just_pressed_for_tool(GameAction::PipeTurn, MainTool::Hammer));
    assert!(!input.pressed_for_tool(GameAction::RaiseFrozenCreation, MainTool::Welder));

    controls.set(
        GameAction::FreezeCreation,
        0,
        Some(InputChord::key(KeyCode::KeyF).with_shift()),
    );
    keyboard.press(KeyCode::ShiftLeft);
    let input = ActionInput::without_wheel(&controls, &keyboard, &mouse);
    assert!(input.just_pressed(GameAction::PipeTurn));
    assert!(input.just_pressed_for_tool(GameAction::FreezeCreation, MainTool::Hammer));
    assert!(input.pressed_for_tool(GameAction::RaiseFrozenCreation, MainTool::Hammer));
}

#[test]
fn linear_dimensions_have_distinct_rebindable_coarse_and_fine_chords() {
    let controls = Controls::default();
    let actions = [
        GameAction::LinearLengthDecrease,
        GameAction::LinearLengthIncrease,
        GameAction::LinearWidthDecrease,
        GameAction::LinearWidthIncrease,
        GameAction::LinearLengthFineDecrease,
        GameAction::LinearLengthFineIncrease,
        GameAction::LinearWidthFineDecrease,
        GameAction::LinearWidthFineIncrease,
    ];
    for (index, expected) in actions.into_iter().enumerate() {
        let mut keyboard = ButtonInput::default();
        keyboard.press(if index % 2 == 0 {
            KeyCode::ArrowLeft
        } else {
            KeyCode::ArrowRight
        });
        if index % 4 >= 2 {
            keyboard.press(KeyCode::ShiftLeft);
        }
        if index >= 4 {
            keyboard.press(KeyCode::ControlLeft);
        }
        let mouse = ButtonInput::default();
        let input = ActionInput::without_wheel(&controls, &keyboard, &mouse);
        for action in actions {
            assert_eq!(
                input.just_pressed(action),
                action == expected,
                "{expected:?} versus {action:?}"
            );
        }
        assert!(GameAction::ALL.contains(&expected));
        assert_eq!(expected.group(), "Dimensions");
        assert!(expected.instantaneous());
    }
    let mut rebound = controls;
    rebound.set(
        GameAction::LinearLengthFineIncrease,
        0,
        Some(InputChord::key(KeyCode::KeyL)),
    );
    assert_eq!(rebound.label(GameAction::LinearLengthFineIncrease), "L");
}

#[test]
fn linear_fine_shortcuts_preserve_other_placeables_control_arrows() {
    let controls = Controls::default();
    let mut keyboard = ButtonInput::default();
    let mouse = ButtonInput::default();
    keyboard.press(KeyCode::ControlLeft);
    keyboard.press(KeyCode::ArrowRight);
    let input = ActionInput::without_wheel(&controls, &keyboard, &mouse);
    assert!(input.just_pressed(GameAction::BearingOuterIncrease));
    assert!(input.just_pressed(GameAction::CylinderOuterIncrease));
    assert!(input.just_pressed(GameAction::LinearLengthFineIncrease));
    keyboard.press(KeyCode::ShiftLeft);
    let input = ActionInput::without_wheel(&controls, &keyboard, &mouse);
    assert!(input.just_pressed(GameAction::BearingInnerIncrease));
    assert!(input.just_pressed(GameAction::CylinderInnerIncrease));
    assert!(input.just_pressed(GameAction::LinearWidthFineIncrease));
    assert!(!input.just_pressed(GameAction::BearingOuterIncrease));
}

#[test]
fn defaults_use_r_for_rotate_and_q_for_clear_pipette() {
    let controls = Controls::default();
    assert_eq!(controls.label(GameAction::Rotate), "R");
    assert_eq!(controls.label(GameAction::ClearPipette), "Q");
    assert_eq!(controls.label(GameAction::Sprint), "ShiftLeft");
    assert_eq!(controls.label(GameAction::Jump), "Space");
    assert_eq!(controls[GameAction::Save].0.iter().flatten().count(), 2);
    assert!(!controls.conflicts(GameAction::Sprint));
    assert!(!controls.conflicts(GameAction::Jump));
    for action in [
        GameAction::NudgeLeft,
        GameAction::NudgeRight,
        GameAction::NudgeUp,
        GameAction::NudgeDown,
    ] {
        assert_eq!(controls.binding(action).0[1], None);
    }
    assert_eq!(
        controls.label(GameAction::FreePlacementRangeIncrease),
        "Shift+Wheel Up"
    );
    assert_eq!(
        controls.label(GameAction::FreePlacementRangeDecrease),
        "Shift+Wheel Down"
    );
}

#[test]
fn descend_and_precision_placement_have_distinct_defaults() {
    let controls = Controls::default();

    assert_eq!(controls.label(GameAction::Descend), "C");
    assert_eq!(
        controls.binding(GameAction::PrecisionPlacement).0[0],
        Some(InputChord::key(KeyCode::ControlLeft))
    );
    assert!(!controls.conflicts(GameAction::Descend));
    assert!(!controls.conflicts(GameAction::PrecisionPlacement));
}

#[test]
fn crouch_shares_control_with_precision_placement_without_conflicting() {
    let controls = Controls::default();

    assert_eq!(
        controls.binding(GameAction::Crouch).0[0],
        Some(InputChord::key(KeyCode::ControlLeft))
    );
    assert!(!controls.conflicts(GameAction::Crouch));
    assert!(!controls.conflicts(GameAction::PrecisionPlacement));
}

#[test]
fn shift_wheel_exposes_contextual_free_range_and_zoom_actions() {
    let controls = Controls::default();
    let mut keyboard = ButtonInput::default();
    keyboard.press(KeyCode::ShiftLeft);
    let mouse = ButtonInput::default();
    let input = ActionInput {
        controls: &controls,
        keyboard: &keyboard,
        mouse: &mouse,
        scroll: Vec2::Y,
    };

    assert!(input.just_pressed(GameAction::FreePlacementRangeIncrease));
    assert!(input.just_pressed(GameAction::ZoomIn));
    assert!(!input.just_pressed(GameAction::ObjectSnapRangeIncrease));
}

#[test]
fn movement_bindings_never_also_nudge_shapes() {
    let mut controls = Controls::default();
    controls.set(GameAction::NudgeUp, 1, Some(InputChord::key(KeyCode::KeyW)));
    let mut keyboard = ButtonInput::default();
    keyboard.press(KeyCode::KeyW);
    let mouse = ButtonInput::default();
    let input = ActionInput::without_wheel(&controls, &keyboard, &mouse);

    assert!(input.just_pressed(GameAction::MoveForward));
    assert!(!input.just_pressed(GameAction::NudgeUp));
}

#[test]
fn defaults_bind_four_tools_and_eight_shift_modes_without_shadowing() {
    let controls = Controls::default();
    for ((action, _), digit) in GameAction::TOOL_ACTIONS.into_iter().zip([
        KeyCode::Digit1,
        KeyCode::Digit2,
        KeyCode::Digit3,
        KeyCode::Digit4,
    ]) {
        assert_eq!(controls.binding(action).0[0], Some(InputChord::key(digit)));
    }
    for ((action, _), digit) in GameAction::MODE_ACTIONS.into_iter().zip([
        KeyCode::Digit1,
        KeyCode::Digit2,
        KeyCode::Digit3,
        KeyCode::Digit4,
        KeyCode::Digit5,
        KeyCode::Digit6,
        KeyCode::Digit7,
        KeyCode::Digit8,
    ]) {
        assert_eq!(
            controls.binding(action).0[0],
            Some(InputChord::key(digit).with_shift())
        );
    }

    let mut keyboard = ButtonInput::default();
    let mouse = ButtonInput::default();
    keyboard.press(KeyCode::ShiftLeft);
    keyboard.press(KeyCode::Digit1);
    let input = ActionInput::without_wheel(&controls, &keyboard, &mouse);
    assert!(input.just_pressed(GameAction::MatterBlock));
    assert!(!input.just_pressed(GameAction::ToolMatterManipulator));
    assert_eq!(
        controls.binding(GameAction::MatterLayer).0[0],
        Some(InputChord::key(KeyCode::Digit3).with_shift())
    );
}

#[test]
fn binding_labels_are_user_facing_keycaps() {
    assert_eq!(InputChord::key(KeyCode::Digit1).label(), "1");
    assert_eq!(InputChord::key(KeyCode::BracketLeft).label(), "[");
    assert_eq!(InputChord::key(KeyCode::BracketRight).label(), "]");
    assert_eq!(InputChord::key(KeyCode::Minus).label(), "-");
    assert_eq!(InputChord::key(KeyCode::Equal).label(), "=");
    assert_eq!(
        InputChord::key(KeyCode::Digit7).with_control().label(),
        "Ctrl+7"
    );
}

#[test]
fn chords_dual_slots_mouse_wheel_and_duplicates_activate() {
    let mut controls = Controls::default();
    controls.set(
        GameAction::Rotate,
        1,
        Some(InputChord::mouse(MouseButton::Middle)),
    );
    controls.set(
        GameAction::ToggleHelp,
        0,
        Some(InputChord::wheel(WheelDirection::Up)),
    );
    controls.set(
        GameAction::Creations,
        0,
        Some(InputChord::wheel(WheelDirection::Up)),
    );
    let mut keyboard = ButtonInput::default();
    let mut mouse = ButtonInput::default();
    mouse.press(MouseButton::Middle);
    let input = ActionInput {
        controls: &controls,
        keyboard: &keyboard,
        mouse: &mouse,
        scroll: Vec2::Y,
    };
    assert!(input.just_pressed(GameAction::Rotate));
    assert!(input.just_pressed(GameAction::ToggleHelp));
    assert!(input.just_pressed(GameAction::Creations));
    assert!(controls.conflicts(GameAction::ToggleHelp));
    keyboard.press(KeyCode::ShiftLeft);
    keyboard.press(KeyCode::Space);
    let input = ActionInput {
        controls: &controls,
        keyboard: &keyboard,
        mouse: &mouse,
        scroll: Vec2::ZERO,
    };
    assert!(!input.just_pressed(GameAction::RestartSimulation));
    assert!(!input.just_pressed(GameAction::ToggleSimulation));
    assert!(input.just_pressed(GameAction::Jump));
    assert!(input.just_pressed(GameAction::Sprint));

    mouse.press(MouseButton::Left);
    let input = ActionInput {
        controls: &controls,
        keyboard: &keyboard,
        mouse: &mouse,
        scroll: Vec2::ZERO,
    };
    assert!(
        input.just_pressed(GameAction::Primary),
        "unclaimed modifiers may accompany a bare binding"
    );
}

#[test]
fn release_and_clear_semantics_follow_each_slot() {
    let mut controls = Controls::default();
    controls.set(GameAction::Interact, 0, None);
    controls.set(
        GameAction::Interact,
        1,
        Some(InputChord::mouse(MouseButton::Back)),
    );
    let keyboard = ButtonInput::default();
    let mut mouse = ButtonInput::default();
    mouse.press(MouseButton::Back);
    mouse.clear();
    mouse.release(MouseButton::Back);
    let input = ActionInput::without_wheel(&controls, &keyboard, &mouse);
    assert!(input.just_released(GameAction::Interact));
}

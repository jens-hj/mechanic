//! Rebindable gameplay actions shared by input systems, HUD text, and settings.

use std::{collections::HashMap, fmt};

use bevy::{input::mouse::AccumulatedMouseScroll, prelude::*};
use serde::{Deserialize, Serialize};

use crate::settings::AppSettings;
use mechanic_core::{ConstructionGraph, EngineKind, GearKey, GearKeyChord};

/// A gameplay action whose bindings can be changed independently of vehicle programs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) enum GameAction {
    MoveForward,
    MoveBackward,
    MoveLeft,
    MoveRight,
    Sprint,
    Jump,
    Primary,
    Secondary,
    Interact,
    ToggleSpace,
    FinePlacement,
    ToggleSimulation,
    RestartSimulation,
    MaterialWheel,
    ZoomIn,
    ZoomOut,
    ToggleHelp,
    Creations,
    Save,
    Undo,
    Redo,
    ToolMatterManipulator,
    ToolWelder,
    ToolConnector,
    ToolHammer,
    MatterBlock,
    MatterCylinder,
    MatterItem,
    MatterTerrain,
    MatterManipulate,
    ClearPipette,
    Rotate,
    PipeTurn,
    ShapeMirrorX,
    ShapeMirrorZ,
    ShapeSnap,
    NudgeLeft,
    NudgeRight,
    NudgeUp,
    NudgeDown,
    SelectionModifier,
    BearingOuterDecrease,
    BearingOuterIncrease,
    BearingInnerDecrease,
    BearingInnerIncrease,
    CylinderOuterDecrease,
    CylinderOuterIncrease,
    CylinderInnerDecrease,
    CylinderInnerIncrease,
    CylinderLengthDecrease,
    CylinderLengthIncrease,
    CylinderSweepDecrease,
    CylinderSweepIncrease,
}

impl GameAction {
    pub(crate) const ALL: [Self; 51] = [
        Self::MoveForward,
        Self::MoveBackward,
        Self::MoveLeft,
        Self::MoveRight,
        Self::Sprint,
        Self::Jump,
        Self::Primary,
        Self::Secondary,
        Self::Interact,
        Self::ToggleSpace,
        Self::FinePlacement,
        Self::MaterialWheel,
        Self::ZoomIn,
        Self::ZoomOut,
        Self::ToggleHelp,
        Self::Creations,
        Self::Save,
        Self::Undo,
        Self::Redo,
        Self::ToolMatterManipulator,
        Self::ToolWelder,
        Self::ToolConnector,
        Self::ToolHammer,
        Self::MatterBlock,
        Self::MatterCylinder,
        Self::MatterItem,
        Self::MatterTerrain,
        Self::MatterManipulate,
        Self::ClearPipette,
        Self::Rotate,
        Self::PipeTurn,
        Self::ShapeMirrorX,
        Self::ShapeMirrorZ,
        Self::ShapeSnap,
        Self::NudgeLeft,
        Self::NudgeRight,
        Self::NudgeUp,
        Self::NudgeDown,
        Self::SelectionModifier,
        Self::BearingOuterDecrease,
        Self::BearingOuterIncrease,
        Self::BearingInnerDecrease,
        Self::BearingInnerIncrease,
        Self::CylinderOuterDecrease,
        Self::CylinderOuterIncrease,
        Self::CylinderInnerDecrease,
        Self::CylinderInnerIncrease,
        Self::CylinderLengthDecrease,
        Self::CylinderLengthIncrease,
        Self::CylinderSweepDecrease,
        Self::CylinderSweepIncrease,
    ];

    pub(crate) const TOOL_ACTIONS: [(Self, crate::hotbar::MainTool); 4] = [
        (
            Self::ToolMatterManipulator,
            crate::hotbar::MainTool::MatterManipulator,
        ),
        (Self::ToolWelder, crate::hotbar::MainTool::Welder),
        (Self::ToolConnector, crate::hotbar::MainTool::Connector),
        (Self::ToolHammer, crate::hotbar::MainTool::Hammer),
    ];

    pub(crate) const MODE_ACTIONS: [(Self, crate::hotbar::MatterMode); 5] = [
        (Self::MatterBlock, crate::hotbar::MatterMode::Block),
        (Self::MatterCylinder, crate::hotbar::MatterMode::Cylinder),
        (Self::MatterItem, crate::hotbar::MatterMode::Item),
        (Self::MatterTerrain, crate::hotbar::MatterMode::Terrain),
        (
            Self::MatterManipulate,
            crate::hotbar::MatterMode::Manipulate,
        ),
    ];

    pub(crate) const fn for_tool(tool: crate::hotbar::MainTool) -> Self {
        match tool {
            crate::hotbar::MainTool::MatterManipulator => Self::ToolMatterManipulator,
            crate::hotbar::MainTool::Welder => Self::ToolWelder,
            crate::hotbar::MainTool::Connector => Self::ToolConnector,
            crate::hotbar::MainTool::Hammer => Self::ToolHammer,
        }
    }

    pub(crate) const fn for_mode(mode: crate::hotbar::MatterMode) -> Self {
        match mode {
            crate::hotbar::MatterMode::Block => Self::MatterBlock,
            crate::hotbar::MatterMode::Cylinder => Self::MatterCylinder,
            crate::hotbar::MatterMode::Item => Self::MatterItem,
            crate::hotbar::MatterMode::Terrain => Self::MatterTerrain,
            crate::hotbar::MatterMode::Manipulate => Self::MatterManipulate,
        }
    }

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::MoveForward => "Move Forward",
            Self::MoveBackward => "Move Backward",
            Self::MoveLeft => "Move Left",
            Self::MoveRight => "Move Right",
            Self::Sprint => "Sprint",
            Self::Jump => "Jump",
            Self::Primary => "Primary Action",
            Self::Secondary => "Secondary Action",
            Self::Interact => "Interact",
            Self::ToggleSpace => "Toggle Garage / World",
            Self::FinePlacement => "Fine Placement",
            Self::ToggleSimulation => "Toggle Simulation",
            Self::RestartSimulation => "Restart Simulation",
            Self::MaterialWheel => "Material Wheel",
            Self::ZoomIn => "Zoom In",
            Self::ZoomOut => "Zoom Out",
            Self::ToggleHelp => "Toggle Help",
            Self::Creations => "Creations",
            Self::Save => "Save",
            Self::Undo => "Undo",
            Self::Redo => "Redo",
            Self::ToolMatterManipulator => "Matter Manipulator",
            Self::ToolWelder => "Welder",
            Self::ToolConnector => "Connector",
            Self::ToolHammer => "Hammer",
            Self::MatterBlock => "Matter: Block",
            Self::MatterCylinder => "Matter: Cylinder",
            Self::MatterItem => "Matter: Item",
            Self::MatterTerrain => "Matter: Terrain",
            Self::MatterManipulate => "Matter: Manipulate",
            Self::ClearPipette => "Clear / Pipette",
            Self::Rotate => "Rotate / Cycle",
            Self::PipeTurn => "Add Pipe Bend",
            Self::ShapeMirrorX => "Mirror X",
            Self::ShapeMirrorZ => "Mirror Z",
            Self::ShapeSnap => "Shape Snap",
            Self::NudgeLeft => "Nudge Left",
            Self::NudgeRight => "Nudge Right",
            Self::NudgeUp => "Nudge Up",
            Self::NudgeDown => "Nudge Down",
            Self::SelectionModifier => "Extend Selection",
            Self::BearingOuterDecrease => "Bearing Outer -",
            Self::BearingOuterIncrease => "Bearing Outer +",
            Self::BearingInnerDecrease => "Bearing Inner -",
            Self::BearingInnerIncrease => "Bearing Inner +",
            Self::CylinderOuterDecrease => "Cylinder Outer -",
            Self::CylinderOuterIncrease => "Cylinder Outer +",
            Self::CylinderInnerDecrease => "Cylinder Inner -",
            Self::CylinderInnerIncrease => "Cylinder Inner +",
            Self::CylinderLengthDecrease => "Cylinder Length -",
            Self::CylinderLengthIncrease => "Cylinder Length +",
            Self::CylinderSweepDecrease => "Cylinder Sweep -",
            Self::CylinderSweepIncrease => "Cylinder Sweep +",
        }
    }

    pub(crate) const fn group(self) -> &'static str {
        match self {
            Self::MoveForward
            | Self::MoveBackward
            | Self::MoveLeft
            | Self::MoveRight
            | Self::Sprint
            | Self::Jump
            | Self::Primary
            | Self::Secondary
            | Self::Interact
            | Self::ZoomIn
            | Self::ZoomOut => "Movement & Camera",
            Self::ToggleSpace
            | Self::FinePlacement
            | Self::ToggleSimulation
            | Self::RestartSimulation
            | Self::MaterialWheel
            | Self::ToggleHelp
            | Self::Creations
            | Self::Save
            | Self::Undo
            | Self::Redo
            | Self::ClearPipette
            | Self::Rotate
            | Self::PipeTurn => "General",
            Self::ToolMatterManipulator
            | Self::ToolWelder
            | Self::ToolConnector
            | Self::ToolHammer
            | Self::MatterBlock
            | Self::MatterCylinder
            | Self::MatterItem
            | Self::MatterTerrain
            | Self::MatterManipulate => "Tools",
            Self::ShapeMirrorX
            | Self::ShapeMirrorZ
            | Self::ShapeSnap
            | Self::NudgeLeft
            | Self::NudgeRight
            | Self::NudgeUp
            | Self::NudgeDown
            | Self::SelectionModifier => "Shape Editing",
            _ => "Dimensions",
        }
    }

    pub(crate) const fn instantaneous(self) -> bool {
        !matches!(
            self,
            Self::MoveForward
                | Self::MoveBackward
                | Self::MoveLeft
                | Self::MoveRight
                | Self::Sprint
                | Self::Primary
                | Self::Secondary
                | Self::MaterialWheel
                | Self::FinePlacement
                | Self::SelectionModifier
        )
    }

    const fn intentionally_shares_binding_with(self, other: Self) -> bool {
        matches!(
            (self, other),
            (Self::Sprint, Self::FinePlacement | Self::SelectionModifier)
                | (Self::FinePlacement | Self::SelectionModifier, Self::Sprint)
        )
    }
}

/// Modifier keys attached to one input.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct Modifiers {
    pub(crate) shift: bool,
    pub(crate) control: bool,
    pub(crate) alt: bool,
    pub(crate) super_key: bool,
}

impl Modifiers {
    pub(crate) fn from_keyboard(keyboard: &ButtonInput<KeyCode>) -> Self {
        Self {
            shift: keyboard.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]),
            control: keyboard.any_pressed([KeyCode::ControlLeft, KeyCode::ControlRight]),
            alt: keyboard.any_pressed([KeyCode::AltLeft, KeyCode::AltRight]),
            super_key: keyboard.any_pressed([KeyCode::SuperLeft, KeyCode::SuperRight]),
        }
    }

    const fn is_subset_of(self, other: Self) -> bool {
        (!self.shift || other.shift)
            && (!self.control || other.control)
            && (!self.alt || other.alt)
            && (!self.super_key || other.super_key)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) enum WheelDirection {
    Up,
    Down,
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) enum BindingInput {
    Key(KeyCode),
    Mouse(MouseButton),
    Wheel(WheelDirection),
}

/// One physical input plus its exact modifier chord.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct InputChord {
    pub(crate) input: BindingInput,
    #[serde(default)]
    pub(crate) modifiers: Modifiers,
}

impl InputChord {
    pub(crate) const fn key(key: KeyCode) -> Self {
        Self {
            input: BindingInput::Key(key),
            modifiers: Modifiers {
                shift: false,
                control: false,
                alt: false,
                super_key: false,
            },
        }
    }
    pub(crate) const fn mouse(button: MouseButton) -> Self {
        Self {
            input: BindingInput::Mouse(button),
            modifiers: Modifiers {
                shift: false,
                control: false,
                alt: false,
                super_key: false,
            },
        }
    }
    pub(crate) const fn wheel(direction: WheelDirection) -> Self {
        Self {
            input: BindingInput::Wheel(direction),
            modifiers: Modifiers {
                shift: false,
                control: false,
                alt: false,
                super_key: false,
            },
        }
    }
    pub(crate) const fn with_shift(mut self) -> Self {
        self.modifiers.shift = true;
        self
    }
    pub(crate) const fn with_control(mut self) -> Self {
        self.modifiers.control = true;
        self
    }
    pub(crate) const fn with_super(mut self) -> Self {
        self.modifiers.super_key = true;
        self
    }

    pub(crate) fn label(self) -> String {
        let mut parts = Vec::new();
        if self.modifiers.control {
            parts.push("Ctrl".to_owned());
        }
        if self.modifiers.super_key {
            parts.push("Cmd".to_owned());
        }
        if self.modifiers.alt {
            parts.push("Alt".to_owned());
        }
        if self.modifiers.shift {
            parts.push("Shift".to_owned());
        }
        parts.push(match self.input {
            BindingInput::Key(key) => key_label(key),
            BindingInput::Mouse(MouseButton::Left) => "Left Mouse".to_owned(),
            BindingInput::Mouse(MouseButton::Right) => "Right Mouse".to_owned(),
            BindingInput::Mouse(MouseButton::Middle) => "Middle Mouse".to_owned(),
            BindingInput::Mouse(MouseButton::Back) => "Mouse Back".to_owned(),
            BindingInput::Mouse(MouseButton::Forward) => "Mouse Forward".to_owned(),
            BindingInput::Mouse(MouseButton::Other(number)) => format!("Mouse {number}"),
            BindingInput::Wheel(direction) => format!("Wheel {direction:?}"),
        });
        parts.join("+")
    }
}

fn key_label(key: KeyCode) -> String {
    match key {
        KeyCode::Backquote => return "`".to_owned(),
        KeyCode::Backslash | KeyCode::IntlBackslash => return "\\".to_owned(),
        KeyCode::BracketLeft => return "[".to_owned(),
        KeyCode::BracketRight => return "]".to_owned(),
        KeyCode::Comma => return ",".to_owned(),
        KeyCode::Equal => return "=".to_owned(),
        KeyCode::Minus => return "-".to_owned(),
        KeyCode::Period => return ".".to_owned(),
        KeyCode::Quote => return "'".to_owned(),
        KeyCode::Semicolon => return ";".to_owned(),
        KeyCode::Slash => return "/".to_owned(),
        KeyCode::PageDown => return "Page Down".to_owned(),
        KeyCode::PageUp => return "Page Up".to_owned(),
        KeyCode::PrintScreen => return "Print Screen".to_owned(),
        KeyCode::CapsLock => return "Caps Lock".to_owned(),
        KeyCode::NumLock => return "Num Lock".to_owned(),
        KeyCode::ScrollLock => return "Scroll Lock".to_owned(),
        _ => {}
    }
    let debug = format!("{key:?}");
    if let Some(letter) = debug.strip_prefix("Key") {
        letter.to_owned()
    } else if let Some(digit) = debug.strip_prefix("Digit") {
        digit.to_owned()
    } else if let Some(direction) = debug.strip_prefix("Arrow") {
        direction.to_owned()
    } else if let Some(numpad) = debug.strip_prefix("Numpad") {
        format!("Num {numpad}")
    } else {
        debug
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ActionBinding(pub(crate) [Option<InputChord>; 2]);

/// All gameplay bindings. Missing actions are filled from defaults after loading.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Controls {
    bindings: HashMap<GameAction, ActionBinding>,
}

impl Default for Controls {
    #[allow(clippy::too_many_lines)]
    fn default() -> Self {
        use GameAction as A;
        use KeyCode as K;
        let mut bindings = HashMap::new();
        let mut set = |action, first, second| {
            bindings.insert(action, ActionBinding([first, second]));
        };
        set(A::MoveForward, Some(InputChord::key(K::KeyW)), None);
        set(A::MoveBackward, Some(InputChord::key(K::KeyS)), None);
        set(A::MoveLeft, Some(InputChord::key(K::KeyA)), None);
        set(A::MoveRight, Some(InputChord::key(K::KeyD)), None);
        set(
            A::Sprint,
            Some(InputChord::key(K::ShiftLeft)),
            Some(InputChord::key(K::ShiftRight)),
        );
        set(A::Jump, Some(InputChord::key(K::Space)), None);
        set(A::Primary, Some(InputChord::mouse(MouseButton::Left)), None);
        set(
            A::Secondary,
            Some(InputChord::mouse(MouseButton::Right)),
            None,
        );
        set(A::Interact, Some(InputChord::key(K::KeyE)), None);
        set(A::ToggleSpace, Some(InputChord::key(K::F6)), None);
        set(
            A::FinePlacement,
            Some(InputChord::key(K::ShiftLeft)),
            Some(InputChord::key(K::ShiftRight)),
        );
        set(A::ToggleSimulation, None, None);
        set(A::RestartSimulation, None, None);
        set(A::MaterialWheel, Some(InputChord::key(K::Tab)), None);
        set(A::ZoomIn, Some(InputChord::wheel(WheelDirection::Up)), None);
        set(
            A::ZoomOut,
            Some(InputChord::wheel(WheelDirection::Down)),
            None,
        );
        set(
            A::ToggleHelp,
            Some(InputChord::key(K::Slash).with_shift()),
            None,
        );
        set(A::Creations, Some(InputChord::key(K::KeyP)), None);
        set(
            A::Save,
            Some(InputChord::key(K::KeyS).with_control()),
            Some(InputChord::key(K::KeyS).with_super()),
        );
        set(
            A::Undo,
            Some(InputChord::key(K::KeyZ).with_control()),
            Some(InputChord::key(K::KeyZ).with_super()),
        );
        set(
            A::Redo,
            Some(InputChord::key(K::KeyZ).with_control().with_shift()),
            Some(InputChord::key(K::KeyZ).with_super().with_shift()),
        );
        let tool_keys = [K::Digit1, K::Digit2, K::Digit3, K::Digit4];
        for ((action, _), key) in A::TOOL_ACTIONS.into_iter().zip(tool_keys) {
            set(action, Some(InputChord::key(key)), None);
        }
        let mode_keys = [K::Digit1, K::Digit2, K::Digit3, K::Digit4, K::Digit5];
        for ((action, _), key) in A::MODE_ACTIONS.into_iter().zip(mode_keys) {
            set(action, Some(InputChord::key(key).with_shift()), None);
        }
        set(A::ClearPipette, Some(InputChord::key(K::KeyQ)), None);
        set(A::Rotate, Some(InputChord::key(K::KeyR)), None);
        set(A::PipeTurn, Some(InputChord::key(K::KeyF)), None);
        set(A::ShapeMirrorX, Some(InputChord::key(K::KeyX)), None);
        set(A::ShapeMirrorZ, Some(InputChord::key(K::KeyZ)), None);
        set(A::ShapeSnap, Some(InputChord::key(K::KeyG)), None);
        set(
            A::NudgeLeft,
            Some(InputChord::key(K::ArrowLeft)),
            Some(InputChord::key(K::KeyA)),
        );
        set(
            A::NudgeRight,
            Some(InputChord::key(K::ArrowRight)),
            Some(InputChord::key(K::KeyD)),
        );
        set(
            A::NudgeUp,
            Some(InputChord::key(K::ArrowUp)),
            Some(InputChord::key(K::KeyW)),
        );
        set(
            A::NudgeDown,
            Some(InputChord::key(K::ArrowDown)),
            Some(InputChord::key(K::KeyS)),
        );
        set(
            A::SelectionModifier,
            Some(InputChord::key(K::ShiftLeft)),
            Some(InputChord::key(K::ShiftRight)),
        );
        set(
            A::BearingOuterDecrease,
            Some(InputChord::key(K::ArrowLeft)),
            None,
        );
        set(
            A::BearingOuterIncrease,
            Some(InputChord::key(K::ArrowRight)),
            None,
        );
        set(
            A::BearingInnerDecrease,
            Some(InputChord::key(K::ArrowLeft).with_shift()),
            None,
        );
        set(
            A::BearingInnerIncrease,
            Some(InputChord::key(K::ArrowRight).with_shift()),
            None,
        );
        set(
            A::CylinderOuterDecrease,
            Some(InputChord::key(K::ArrowLeft)),
            None,
        );
        set(
            A::CylinderOuterIncrease,
            Some(InputChord::key(K::ArrowRight)),
            None,
        );
        set(
            A::CylinderInnerDecrease,
            Some(InputChord::key(K::ArrowLeft).with_shift()),
            None,
        );
        set(
            A::CylinderInnerIncrease,
            Some(InputChord::key(K::ArrowRight).with_shift()),
            None,
        );
        set(
            A::CylinderLengthDecrease,
            Some(InputChord::key(K::ArrowDown)),
            None,
        );
        set(
            A::CylinderLengthIncrease,
            Some(InputChord::key(K::ArrowUp)),
            None,
        );
        set(
            A::CylinderSweepDecrease,
            Some(InputChord::key(K::ArrowDown).with_shift()),
            None,
        );
        set(
            A::CylinderSweepIncrease,
            Some(InputChord::key(K::ArrowUp).with_shift()),
            None,
        );
        Self { bindings }
    }
}

impl Controls {
    pub(crate) fn normalize(&mut self) {
        let defaults = Self::default();
        for action in GameAction::ALL {
            self.bindings.entry(action).or_insert(defaults[action]);
        }
    }
    pub(crate) fn binding(&self, action: GameAction) -> ActionBinding {
        self[action]
    }
    pub(crate) fn set(&mut self, action: GameAction, slot: usize, chord: Option<InputChord>) {
        if slot < 2 {
            self.bindings.entry(action).or_default().0[slot] = chord;
        }
    }
    pub(crate) fn label(&self, action: GameAction) -> String {
        self[action]
            .0
            .into_iter()
            .flatten()
            .next()
            .map_or_else(|| "Unbound".to_owned(), InputChord::label)
    }
    pub(crate) fn conflicts(&self, action: GameAction) -> bool {
        self[action].0.into_iter().flatten().any(|chord| {
            GameAction::ALL.into_iter().any(|other| {
                other != action
                    && !action.intentionally_shares_binding_with(other)
                    && self[other].0.contains(&Some(chord))
            })
        })
    }
    pub(crate) fn conflicts_with_vehicle(
        &self,
        graph: &ConstructionGraph,
        action: GameAction,
    ) -> bool {
        let binding = self[action];
        vehicle_chords(graph)
            .into_iter()
            .any(|vehicle| binding.0.contains(&Some(vehicle)))
    }
}

fn vehicle_chords(graph: &ConstructionGraph) -> Vec<InputChord> {
    let mut chords = graph
        .drive_links()
        .flat_map(|(_, link)| link.program.states())
        .filter_map(|state| state.trigger())
        .filter_map(|trigger| symbol_key(trigger.key().symbol()))
        .map(InputChord::key)
        .collect::<Vec<_>>();
    for (controller, _) in graph.parts().filter(|(part, _)| graph.is_controller(*part)) {
        for kind in [EngineKind::Electric, EngineKind::Gas] {
            if let Ok(config) = graph.gearbox_config(controller, kind) {
                chords.extend(
                    [config.gear_up(), config.gear_down()]
                        .into_iter()
                        .filter_map(gear_chord),
                );
            }
        }
    }
    chords
}

fn symbol_key(symbol: char) -> Option<KeyCode> {
    Some(match symbol.to_ascii_uppercase() {
        'A' => KeyCode::KeyA,
        'B' => KeyCode::KeyB,
        'C' => KeyCode::KeyC,
        'D' => KeyCode::KeyD,
        'E' => KeyCode::KeyE,
        'F' => KeyCode::KeyF,
        'G' => KeyCode::KeyG,
        'H' => KeyCode::KeyH,
        'I' => KeyCode::KeyI,
        'J' => KeyCode::KeyJ,
        'K' => KeyCode::KeyK,
        'L' => KeyCode::KeyL,
        'M' => KeyCode::KeyM,
        'N' => KeyCode::KeyN,
        'O' => KeyCode::KeyO,
        'P' => KeyCode::KeyP,
        'Q' => KeyCode::KeyQ,
        'R' => KeyCode::KeyR,
        'S' => KeyCode::KeyS,
        'T' => KeyCode::KeyT,
        'U' => KeyCode::KeyU,
        'V' => KeyCode::KeyV,
        'W' => KeyCode::KeyW,
        'X' => KeyCode::KeyX,
        'Y' => KeyCode::KeyY,
        'Z' => KeyCode::KeyZ,
        '0' => KeyCode::Digit0,
        '1' => KeyCode::Digit1,
        '2' => KeyCode::Digit2,
        '3' => KeyCode::Digit3,
        '4' => KeyCode::Digit4,
        '5' => KeyCode::Digit5,
        '6' => KeyCode::Digit6,
        '7' => KeyCode::Digit7,
        '8' => KeyCode::Digit8,
        '9' => KeyCode::Digit9,
        _ => return None,
    })
}

fn gear_chord(chord: GearKeyChord) -> Option<InputChord> {
    let key = match chord.key {
        GearKey::Letter(symbol) => symbol_key(symbol)?,
        GearKey::Digit(digit) => symbol_key(char::from_digit(u32::from(digit), 10)?)?,
        GearKey::Space => KeyCode::Space,
        GearKey::ArrowUp => KeyCode::ArrowUp,
        GearKey::ArrowDown => KeyCode::ArrowDown,
        GearKey::ArrowLeft => KeyCode::ArrowLeft,
        GearKey::ArrowRight => KeyCode::ArrowRight,
        GearKey::PageUp => KeyCode::PageUp,
        GearKey::PageDown => KeyCode::PageDown,
    };
    Some(InputChord {
        input: BindingInput::Key(key),
        modifiers: Modifiers {
            shift: chord.shift,
            control: chord.control,
            alt: chord.alt,
            super_key: chord.super_key,
        },
    })
}

impl std::ops::Index<GameAction> for Controls {
    type Output = ActionBinding;
    fn index(&self, action: GameAction) -> &Self::Output {
        self.bindings
            .get(&action)
            .expect("normalized controls contain every action")
    }
}

/// Current raw input interpreted through a set of bindings.
pub(crate) struct ActionInput<'a> {
    pub(crate) controls: &'a Controls,
    pub(crate) keyboard: &'a ButtonInput<KeyCode>,
    pub(crate) mouse: &'a ButtonInput<MouseButton>,
    pub(crate) scroll: Vec2,
}

impl<'a> ActionInput<'a> {
    pub(crate) fn new(
        controls: &'a Controls,
        keyboard: &'a ButtonInput<KeyCode>,
        mouse: &'a ButtonInput<MouseButton>,
        scroll: &AccumulatedMouseScroll,
    ) -> Self {
        Self {
            controls,
            keyboard,
            mouse,
            scroll: scroll.delta,
        }
    }
    #[cfg(test)]
    pub(crate) fn without_wheel(
        controls: &'a Controls,
        keyboard: &'a ButtonInput<KeyCode>,
        mouse: &'a ButtonInput<MouseButton>,
    ) -> Self {
        Self {
            controls,
            keyboard,
            mouse,
            scroll: Vec2::ZERO,
        }
    }
    pub(crate) fn pressed(&self, action: GameAction) -> bool {
        self.matches(action, MatchKind::Pressed)
    }
    pub(crate) fn just_pressed(&self, action: GameAction) -> bool {
        self.matches(action, MatchKind::JustPressed)
    }
    #[cfg(test)]
    pub(crate) fn just_released(&self, action: GameAction) -> bool {
        self.matches(action, MatchKind::JustReleased)
    }
    fn matches(&self, action: GameAction, kind: MatchKind) -> bool {
        self.controls[action].0.into_iter().flatten().any(|chord| {
            let current = Modifiers::from_keyboard(self.keyboard);
            if !chord.modifiers.is_subset_of(current) {
                return false;
            }
            // A more specific chord on the same physical input owns the press.
            // This keeps Shift+Space from also pausing while still allowing
            // Shift+left mouse and Shift+W when no such chord is configured.
            let shadowed = GameAction::ALL.into_iter().any(|candidate| {
                if action.intentionally_shares_binding_with(candidate) {
                    return false;
                }
                self.controls[candidate]
                    .0
                    .into_iter()
                    .flatten()
                    .any(|other| {
                        other.input == chord.input
                            && other.modifiers != chord.modifiers
                            && chord.modifiers.is_subset_of(other.modifiers)
                            && other.modifiers.is_subset_of(current)
                    })
            });
            if shadowed {
                return false;
            }
            match (chord.input, kind) {
                (BindingInput::Key(key), MatchKind::Pressed) => self.keyboard.pressed(key),
                (BindingInput::Key(key), MatchKind::JustPressed) => self.keyboard.just_pressed(key),
                (BindingInput::Key(key), MatchKind::JustReleased) => {
                    self.keyboard.just_released(key)
                }
                (BindingInput::Mouse(button), MatchKind::Pressed) => self.mouse.pressed(button),
                (BindingInput::Mouse(button), MatchKind::JustPressed) => {
                    self.mouse.just_pressed(button)
                }
                (BindingInput::Mouse(button), MatchKind::JustReleased) => {
                    self.mouse.just_released(button)
                }
                (BindingInput::Wheel(direction), MatchKind::JustPressed) => match direction {
                    WheelDirection::Up => self.scroll.y > 0.0,
                    WheelDirection::Down => self.scroll.y < 0.0,
                    WheelDirection::Left => self.scroll.x < 0.0,
                    WheelDirection::Right => self.scroll.x > 0.0,
                },
                (BindingInput::Wheel(_), MatchKind::Pressed | MatchKind::JustReleased) => false,
            }
        })
    }
}

#[derive(Clone, Copy)]
#[cfg_attr(not(test), allow(dead_code))]
enum MatchKind {
    Pressed,
    JustPressed,
    JustReleased,
}

/// Rebuilds the action-level button state from raw device inputs once per frame.
pub(crate) fn update_action_state(
    settings: Res<AppSettings>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    scroll: Res<AccumulatedMouseScroll>,
    mut actions: ResMut<ButtonInput<GameAction>>,
) {
    actions.clear();
    let input = ActionInput::new(settings.controls(), &keyboard, &mouse, &scroll);
    for action in GameAction::ALL {
        let active = input.pressed(action) || input.just_pressed(action);
        if active {
            actions.press(action);
        } else {
            actions.release(action);
        }
    }
}

impl fmt::Display for WheelDirection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    }

    #[test]
    fn defaults_bind_four_tools_and_five_shift_modes_without_shadowing() {
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
}

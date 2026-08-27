//! The tools, and which one is in hand.
//!
//! What the bar looks like lives in [`crate::ui::hotbar`]; this is what a tool
//! is, which key picks it, and what it may do in which mode.

use bevy::prelude::*;
use mechanic_core::ConstructionMaterial;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) enum Tool {
    #[default]
    Block,
    Cylinder,
    Bearing,
    Weld,
    Hammer,
    Controller,
    Connector,
    GasEngine,
    ElectricEngine,
    Transmission,
    Servo,
    Seat,
    Input,
    Shape,
}

impl Tool {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Block => "Blocker Placer",
            Self::Cylinder => "Pipe / Cylinder",
            Self::Bearing => "Bearing",
            Self::Weld => "Weld",
            Self::Hammer => "Hammer",
            Self::Controller => "Control Block",
            Self::Connector => "Connector",
            Self::GasEngine => "Gas Engine",
            Self::ElectricEngine => "Electric Engine",
            Self::Transmission => "Transmission",
            Self::Servo => "Servo",
            Self::Seat => "Seat",
            Self::Input => "Input",
            Self::Shape => "Shape",
        }
    }

    pub(crate) const fn works_while_simulating(self) -> bool {
        matches!(self, Self::Hammer)
    }

    /// Whether this tool works with control blocks and their wires.
    pub(crate) const fn edits_drives(self) -> bool {
        matches!(self, Self::Controller | Self::Connector)
    }

    pub(crate) const fn works_in_mode(self, simulating: bool) -> bool {
        self.works_while_simulating() == simulating
    }
}

#[cfg(test)]
pub(crate) const fn shortcut_tool(key: KeyCode) -> Option<Tool> {
    match key {
        KeyCode::Digit1 => Some(Tool::Block),
        KeyCode::Digit2 => Some(Tool::Cylinder),
        KeyCode::Digit3 => Some(Tool::Bearing),
        KeyCode::Digit4 => Some(Tool::Weld),
        KeyCode::Digit5 => Some(Tool::Hammer),
        KeyCode::Digit6 => Some(Tool::Controller),
        KeyCode::Digit7 => Some(Tool::Connector),
        KeyCode::Digit8 => Some(Tool::GasEngine),
        KeyCode::Digit9 => Some(Tool::ElectricEngine),
        KeyCode::BracketRight => Some(Tool::Transmission),
        KeyCode::Digit0 => Some(Tool::Servo),
        KeyCode::Minus => Some(Tool::Seat),
        KeyCode::Equal => Some(Tool::Input),
        KeyCode::BracketLeft => Some(Tool::Shape),
        _ => None,
    }
}

#[derive(Resource, Debug)]
pub(crate) struct SelectedTool(pub(crate) Option<Tool>);

impl Default for SelectedTool {
    fn default() -> Self {
        Self(Some(Tool::Block))
    }
}

/// Material shared by the Blocker Placer and Cylinder for this process.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SelectedMaterial(pub(crate) ConstructionMaterial);

#[cfg(test)]
mod tests {
    use super::{Tool, shortcut_tool};
    use bevy::prelude::KeyCode;

    #[test]
    fn every_tool_has_a_keyboard_shortcut_including_brackets() {
        let mappings = [
            (KeyCode::Digit1, Tool::Block),
            (KeyCode::Digit2, Tool::Cylinder),
            (KeyCode::Digit3, Tool::Bearing),
            (KeyCode::Digit4, Tool::Weld),
            (KeyCode::Digit5, Tool::Hammer),
            (KeyCode::Digit6, Tool::Controller),
            (KeyCode::Digit7, Tool::Connector),
            (KeyCode::Digit8, Tool::GasEngine),
            (KeyCode::Digit9, Tool::ElectricEngine),
            (KeyCode::BracketRight, Tool::Transmission),
            (KeyCode::Digit0, Tool::Servo),
            (KeyCode::Minus, Tool::Seat),
            (KeyCode::Equal, Tool::Input),
            (KeyCode::BracketLeft, Tool::Shape),
        ];
        for (key, tool) in mappings {
            assert_eq!(shortcut_tool(key), Some(tool));
        }
    }
}

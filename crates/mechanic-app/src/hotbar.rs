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
            Self::Cylinder => "Cylinder",
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

    pub(crate) const fn shortcut(self) -> &'static str {
        match self {
            Self::Block => "1",
            Self::Cylinder => "2",
            Self::Bearing => "3",
            Self::Weld => "4",
            Self::Hammer => "5",
            Self::Controller => "6",
            Self::Connector => "7",
            Self::GasEngine => "8",
            Self::ElectricEngine => "9",
            Self::Transmission => "]",
            Self::Servo => "0",
            Self::Seat => "-",
            Self::Input => "=",
            Self::Shape => "[",
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

#[derive(Resource, Debug, Default)]
pub(crate) struct SelectedTool(pub(crate) Tool);

/// Material shared by the Blocker Placer and Cylinder for this process.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SelectedMaterial(pub(crate) ConstructionMaterial);

//! Which control block is open, and how its wires group into joints.
//!
//! The panel itself is drawn by [`crate::ui::control_block`] in Mosaic. What
//! lives here is the part the rest of the app talks to: whether a block is open
//! — which is what every other system gates its own input on — and the grouping
//! that turns a socket's several wires into the one joint a person sees.

use bevy::prelude::*;
use mechanic_core::{
    ActuatorAssignment, BuildCommand, ConstructionGraph, DriveLimits, DriveLinkId, DriveName,
    DriveProgram, PartId,
};

/// Which control block is open, and what it is holding onto.
#[derive(Resource, Debug, Default)]
pub(crate) struct ControlPanelState {
    controller: Option<PartId>,
    speed_unit: SpeedUnit,
}

/// Display unit for continuous rotation targets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum SpeedUnit {
    /// Revolutions per minute.
    #[default]
    Rpm,
    /// Degrees per second.
    DegreesPerSecond,
}

impl SpeedUnit {
    pub(crate) const fn toggled(self) -> Self {
        match self {
            Self::Rpm => Self::DegreesPerSecond,
            Self::DegreesPerSecond => Self::Rpm,
        }
    }
}

impl ControlPanelState {
    /// Opens the panel on one control block.
    pub(crate) const fn open(&mut self, controller: PartId) {
        self.controller = Some(controller);
    }

    /// Closes the panel.
    pub(crate) const fn close(&mut self) {
        self.controller = None;
    }

    /// Control block the panel is showing.
    pub(crate) const fn controller(&self) -> Option<PartId> {
        self.controller
    }

    /// Whether the panel is showing.
    pub(crate) const fn is_open(&self) -> bool {
        self.controller.is_some()
    }

    /// Whether the panel owns the keyboard.
    ///
    /// True whenever it is open: its dials are typed into and its keycaps are
    /// bound by pressing a key, and a key press must never both bind a state
    /// and fire a global shortcut.
    pub(crate) const fn blocks_keyboard(&self) -> bool {
        self.controller.is_some()
    }

    pub(crate) const fn speed_unit(&self) -> SpeedUnit {
        self.speed_unit
    }

    pub(crate) fn toggle_speed_unit(&mut self) {
        self.speed_unit = self.speed_unit.toggled();
    }
}

/// Rows of the table: one physical joint, and every wire backing it.
///
/// One socket can produce several bearing rows in the graph. They describe the
/// same joint, so the panel shows one row and writes every wire behind it.
pub(crate) struct PanelRow {
    /// Wires this row edits, all describing one joint.
    pub(crate) links: Vec<DriveLinkId>,
    /// Wire whose values the row displays.
    pub(crate) primary: DriveLinkId,
}

/// Groups a control block's wires into displayed rows.
///
/// Wires are grouped by the anchor and axis of their bearings, which is what
/// makes two rows the same physical joint.
pub(crate) fn panel_rows(graph: &ConstructionGraph, controller: PartId) -> Vec<PanelRow> {
    let mut rows: Vec<(Vec3, Vec3, PanelRow)> = Vec::new();
    for (link, spec) in graph.controller_links(controller) {
        let Some(bearing) = graph.bearing(spec.bearing) else {
            continue;
        };
        let existing = rows.iter_mut().find(|(anchor, axis, _)| {
            anchor.abs_diff_eq(bearing.shared_anchor, 1.0e-5)
                && axis.abs_diff_eq(bearing.axis, 1.0e-5)
        });
        match existing {
            Some((_, _, row)) => row.links.push(link),
            None => rows.push((
                bearing.shared_anchor,
                bearing.axis,
                PanelRow {
                    links: vec![link],
                    primary: link,
                },
            )),
        }
    }
    rows.into_iter().map(|(_, _, row)| row).collect()
}

/// Commands that write one row's limits, program, and name to every wire
/// behind it.
pub(crate) fn set_row_commands(
    row: &PanelRow,
    limits: DriveLimits,
    program: DriveProgram,
    name: DriveName,
    actuator: ActuatorAssignment,
) -> Vec<BuildCommand> {
    row.links
        .iter()
        .map(|&link| BuildCommand::SetDriveLink {
            link,
            limits,
            program,
            name,
            actuator,
        })
        .collect()
}

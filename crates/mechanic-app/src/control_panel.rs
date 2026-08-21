//! Table UI for one control block's drive programs.
//!
//! The panel shows one row per driven bearing and one column per state. Cells
//! are clicked to cycle discrete choices, or focused and typed into for
//! numbers. It stays usable while the simulation runs, because reprogramming a
//! wire changes no topology and no buffer size.

use bevy::{
    input::{
        keyboard::{Key, KeyboardInput},
        mouse::{AccumulatedMouseScroll, MouseScrollUnit},
    },
    prelude::*,
    ui::{FocusPolicy, ScrollPosition},
};
use mechanic_core::{
    BuildCommand, ConstructionGraph, DriveDwell, DriveLimits, DriveLinkId, DriveProgram,
    DriveRelease, DriveState, DriveTarget, DriveTrigger, MAX_DRIVE_LIMIT_RADIANS,
    MAX_DRIVE_SPEED_RAD_S, MAX_DRIVE_STATES, PartId,
};

use crate::sequencer::drive_key;

const PANEL_BACKGROUND: Color = Color::srgba(0.018, 0.026, 0.038, 0.99);
const PANEL_BORDER: Color = Color::srgba(0.30, 0.62, 0.66, 0.95);
const CELL_BACKGROUND: Color = Color::srgba(0.06, 0.085, 0.12, 0.98);
const CELL_HOVER: Color = Color::srgba(0.10, 0.16, 0.21, 0.98);
const CELL_FOCUSED: Color = Color::srgba(0.08, 0.30, 0.36, 0.99);
const HEADING: Color = Color::srgb(0.72, 0.82, 0.86);
/// Width of the `S1`, `S2`, ... gutter, so headers line up with their cells.
const STATE_LABEL_WIDTH: u32 = 34;
/// Fixed panel width. A state line is the gutter plus seven cells, and a cell
/// grows past its minimum for wide words like `unlimited`, so this leaves room
/// for the widest line rather than resizing the panel as values change.
const PANEL_WIDTH: u32 = 700;
/// Inset from the window edges on the left, top, and bottom.
const PANEL_MARGIN: u32 = 24;
/// Pixels one wheel line scrolls the table by.
const SCROLL_LINE_PIXELS: f32 = 24.0;
const LABEL: Color = Color::srgb(0.88, 0.92, 0.95);

/// Which value one cell of the table edits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DriveCellKind {
    /// Row envelope: fastest the bearing may turn.
    MaxSpeed,
    /// Row envelope: strongest torque the bearing may apply.
    Torque,
    /// Row envelope: whether travel stops are enabled.
    ToggleTravelLimits,
    /// Whether the row's last state hands back to its first.
    ToggleLoop,
    /// Angle or speed, for one state.
    Mode(u8),
    /// The state's target value.
    Value(u8),
    /// The state's dwell time, empty to clear it.
    Dwell(u8),
    /// Where the state's dwell hands off to.
    Next(u8),
    /// The state's key binding.
    Key(u8),
    /// What the state does when its key is released.
    Release(u8),
    /// Append a state to the row.
    AddState,
    /// Remove one state from the row.
    RemoveState(u8),
}

impl DriveCellKind {
    /// Whether this cell is edited by typing rather than clicking.
    pub(crate) const fn is_typed(self) -> bool {
        matches!(
            self,
            Self::MaxSpeed | Self::Torque | Self::Value(_) | Self::Dwell(_)
        )
    }

    /// What this cell means when nothing is typed into it, if anything.
    ///
    /// These two cells have an unbounded setting that no number can express, so
    /// they take a word instead. It is also what an empty entry commits, and it
    /// is shown in place of the empty buffer so the choice is visible.
    pub(crate) const fn open_ended_value(self) -> Option<&'static str> {
        match self {
            Self::Torque => Some("unlimited"),
            Self::Dwell(_) => Some("none"),
            _ => None,
        }
    }
}

/// Words accepted for a cell with no upper bound. An infinite dwell and no
/// dwell are the same thing -- the state never advances on its own -- so both
/// spellings are taken.
fn is_open_ended_word(text: &str) -> bool {
    matches!(
        text.trim().to_ascii_lowercase().as_str(),
        "" | "inf" | "infinite" | "infinity" | "unlimited" | "none" | "never"
    )
}

/// One editable cell: which wire, and which of its values.
#[derive(Clone, Copy, Component, Debug, PartialEq, Eq)]
pub(crate) struct DriveCell {
    /// Wire the cell edits.
    pub(crate) link: DriveLinkId,
    /// Value within that wire.
    pub(crate) kind: DriveCellKind,
}

/// Root of the panel's UI tree.
#[derive(Component)]
pub(crate) struct ControlPanelRoot;

/// The scrolling table inside the panel. The title and the hint line sit
/// outside it, so they stay put while the joints scroll under them.
#[derive(Component)]
pub(crate) struct ControlPanelScroll;

/// What a keystroke does to the cell being typed into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NumberEntryAction {
    /// Append a character to the buffer.
    Insert(char),
    /// Drop the last character.
    Backspace,
    /// Parse the buffer and write it back.
    Commit,
    /// Discard the buffer and leave the old value alone.
    Cancel,
    /// Not part of numeric entry.
    Ignore,
}

/// Maps one logical key to its numeric-entry action.
///
/// Only what a number needs: digits, a leading minus, one decimal point. Every
/// other key is ignored rather than swallowed, so a stray press cannot corrupt
/// a value.
pub(crate) fn number_entry_action(key: &Key, accepts_words: bool) -> NumberEntryAction {
    match key {
        Key::Character(text) => text
            .chars()
            .next()
            .map_or(NumberEntryAction::Ignore, |symbol| match symbol {
                '0'..='9' | '-' | '.' => NumberEntryAction::Insert(symbol),
                // Only cells with an open-ended setting take letters, so a
                // stray keystroke cannot corrupt a plain number.
                letter if accepts_words && letter.is_ascii_alphabetic() => {
                    NumberEntryAction::Insert(letter)
                }
                _ => NumberEntryAction::Ignore,
            }),
        Key::Backspace => NumberEntryAction::Backspace,
        Key::Enter => NumberEntryAction::Commit,
        Key::Escape => NumberEntryAction::Cancel,
        _ => NumberEntryAction::Ignore,
    }
}

/// Applies one action to a typing buffer, reporting what to do next.
pub(crate) fn edited_buffer(buffer: &str, action: NumberEntryAction) -> Option<String> {
    match action {
        NumberEntryAction::Insert(symbol) => {
            let mut edited = buffer.to_owned();
            edited.push(symbol);
            Some(edited)
        }
        NumberEntryAction::Backspace => {
            let mut edited = buffer.to_owned();
            edited.pop();
            Some(edited)
        }
        NumberEntryAction::Commit | NumberEntryAction::Cancel | NumberEntryAction::Ignore => None,
    }
}

/// Live panel state: which block it shows, and which cell is being typed into.
#[derive(Resource, Debug, Default)]
pub(crate) struct ControlPanelState {
    controller: Option<PartId>,
    typing: Option<(DriveCell, String)>,
    capturing: Option<DriveCell>,
    /// Cell held down last frame. `Interaction::Pressed` stays set for as long
    /// as the button is held, so a cell would otherwise cycle every frame.
    held: Option<DriveCell>,
    /// How far the table is scrolled. A rebuild respawns it, so the offset
    /// lives here rather than only on the node.
    scroll: Vec2,
    blocks_pointer: bool,
    dirty: bool,
}

impl ControlPanelState {
    /// Clears the one-frame pointer latch at the top of each frame.
    pub(crate) fn begin_frame(&mut self) {
        self.blocks_pointer = false;
    }

    /// Opens the panel on one control block.
    pub(crate) fn open(&mut self, controller: PartId) {
        self.controller = Some(controller);
        self.typing = None;
        self.capturing = None;
        self.held = None;
        self.scroll = Vec2::ZERO;
        self.blocks_pointer = true;
        self.dirty = true;
    }

    /// Closes the panel, discarding any half-typed value.
    pub(crate) fn close(&mut self) {
        self.controller = None;
        self.typing = None;
        self.capturing = None;
        self.held = None;
        self.scroll = Vec2::ZERO;
        self.blocks_pointer = true;
        self.dirty = true;
    }

    /// Control block the panel is showing.
    pub(crate) const fn controller(&self) -> Option<PartId> {
        self.controller
    }

    /// Whether the panel is showing.
    pub(crate) const fn is_open(&self) -> bool {
        self.controller.is_some()
    }

    /// Whether the panel is swallowing pointer input this frame.
    pub(crate) const fn blocks_pointer(&self) -> bool {
        self.blocks_pointer || self.controller.is_some()
    }

    /// Whether the panel owns the keyboard.
    ///
    /// True whenever it is open: its cells are clicked and typed into, and a
    /// key press must never both bind a state and fire a global shortcut.
    pub(crate) const fn blocks_keyboard(&self) -> bool {
        self.controller.is_some()
    }

    /// Whether the panel is waiting for a key to bind.
    pub(crate) const fn is_capturing(&self) -> bool {
        self.capturing.is_some()
    }

    /// Whether Escape belongs to the panel's own contents rather than to the
    /// panel itself. Escape cancels a value being typed or a key being bound
    /// before it closes the window.
    pub(crate) const fn escape_is_consumed(&self) -> bool {
        self.capturing.is_some() || self.typing.is_some()
    }

    /// Marks the table for a rebuild.
    pub(crate) const fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Takes the pending rebuild request.
    pub(crate) const fn take_dirty(&mut self) -> bool {
        let dirty = self.dirty;
        self.dirty = false;
        dirty
    }

    /// Focuses one cell for typing, starting from an empty buffer.
    pub(crate) fn begin_typing(&mut self, cell: DriveCell) {
        self.typing = Some((cell, String::new()));
        self.dirty = true;
    }

    /// Cell currently being typed into, and its buffer.
    pub(crate) fn typing(&self) -> Option<(DriveCell, &str)> {
        self.typing
            .as_ref()
            .map(|(cell, buffer)| (*cell, buffer.as_str()))
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

/// Commands that write one row's limits and program to every wire behind it.
pub(crate) fn set_row_commands(
    row: &PanelRow,
    limits: DriveLimits,
    program: DriveProgram,
) -> Vec<BuildCommand> {
    row.links
        .iter()
        .map(|&link| BuildCommand::SetDriveLink {
            link,
            limits,
            program,
        })
        .collect()
}

/// Applies a typed number to the cell it was typed into.
///
/// Angles are typed and shown in degrees; everything else is in the unit the
/// label states. Returns `None` when the text is not a number or the value is
/// out of range, leaving the old value in place.
pub(crate) fn committed_value(
    kind: DriveCellKind,
    text: &str,
    limits: DriveLimits,
    program: DriveProgram,
) -> Option<(DriveLimits, DriveProgram)> {
    let trimmed = text.trim();
    match kind {
        DriveCellKind::MaxSpeed => {
            let degrees = trimmed.parse::<f32>().ok()?;
            Some((limits.with_max_speed(degrees.to_radians()).ok()?, program))
        }
        DriveCellKind::Torque => {
            // An empty cell, or "unlimited", means "as much as it takes".
            let torque = if is_open_ended_word(trimmed) {
                f32::INFINITY
            } else {
                trimmed.parse::<f32>().ok()?
            };
            Some((limits.with_max_torque(torque).ok()?, program))
        }
        DriveCellKind::Value(index) => {
            let value = trimmed.parse::<f32>().ok()?;
            let state = program.state(index)?;
            let target = match state.target() {
                DriveTarget::Angle(_) => DriveTarget::Angle(value.to_radians()),
                DriveTarget::Speed(_) => DriveTarget::Speed(value),
            };
            Some((
                limits,
                program
                    .with_state(index, state.with_target(target).ok()?)
                    .ok()?,
            ))
        }
        DriveCellKind::Dwell(index) => {
            let state = program.state(index)?;
            // No dwell and an infinite one both mean the state waits for a key
            // rather than advancing itself, so they resolve to the same thing.
            let dwell = if is_open_ended_word(trimmed) {
                None
            } else {
                Some(
                    DriveDwell::new(
                        trimmed.parse::<f32>().ok()?,
                        state.dwell().and_then(DriveDwell::next),
                    )
                    .ok()?,
                )
            };
            Some((
                limits,
                program.with_state(index, state.with_dwell(dwell)).ok()?,
            ))
        }
        _ => None,
    }
}

/// Applies a click to a cell that cycles through discrete choices.
pub(crate) fn clicked_value(
    kind: DriveCellKind,
    limits: DriveLimits,
    program: DriveProgram,
) -> Option<(DriveLimits, DriveProgram)> {
    match kind {
        DriveCellKind::ToggleTravelLimits => {
            let travel = if limits.angle_limits().is_some() {
                None
            } else {
                Some((-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2))
            };
            Some((limits.with_angle_limits(travel).ok()?, program))
        }
        DriveCellKind::ToggleLoop => Some((limits, program.with_loops(!program.loops()))),
        DriveCellKind::Mode(index) => {
            let state = program.state(index)?;
            // Switching mode carries the number across and clamps it into the
            // new unit's range, so the click never fails validation and the
            // player does not silently lose what they typed.
            let target = match state.target() {
                DriveTarget::Angle(angle) => {
                    DriveTarget::Speed(angle.clamp(-MAX_DRIVE_SPEED_RAD_S, MAX_DRIVE_SPEED_RAD_S))
                }
                DriveTarget::Speed(speed) => DriveTarget::Angle(
                    speed.clamp(-MAX_DRIVE_LIMIT_RADIANS, MAX_DRIVE_LIMIT_RADIANS),
                ),
            };
            Some((
                limits,
                program
                    .with_state(index, state.with_target(target).ok()?)
                    .ok()?,
            ))
        }
        DriveCellKind::Next(index) => {
            let state = program.state(index)?;
            // With a dwell this names where the dwell hands off. Without one the
            // state is left by its key going up instead, so it names that
            // target -- the same setting the release cell shows, reachable from
            // whichever column is looked at first.
            let Some(dwell) = state.dwell() else {
                let released = cycled_release(state, program.len())?;
                return Some((limits, program.with_state(index, released).ok()?));
            };
            let next = cycled_state(dwell.next(), program.len());
            Some((
                limits,
                program
                    .with_state(
                        index,
                        state.with_dwell(Some(DriveDwell::new(dwell.seconds(), next).ok()?)),
                    )
                    .ok()?,
            ))
        }
        DriveCellKind::Release(index) => {
            let released = cycled_release(program.state(index)?, program.len())?;
            Some((limits, program.with_state(index, released).ok()?))
        }
        DriveCellKind::AddState => {
            if program.len() >= MAX_DRIVE_STATES {
                return None;
            }
            let appended = DriveState::new(program.state(0)?.target()).ok()?;
            Some((limits, program.with_pushed_state(appended).ok()?))
        }
        DriveCellKind::RemoveState(index) => {
            Some((limits, program.with_removed_state(index).ok()?))
        }
        DriveCellKind::Key(index) => {
            // Clicking a bound key clears it; clicking an empty one arms
            // capture, which the caller handles.
            let state = program.state(index)?;
            state.trigger()?;
            Some((
                limits,
                program.with_state(index, state.with_trigger(None)).ok()?,
            ))
        }
        DriveCellKind::MaxSpeed
        | DriveCellKind::Torque
        | DriveCellKind::Value(_)
        | DriveCellKind::Dwell(_) => None,
    }
}

/// Binds a captured key to a state, replacing any existing binding.
pub(crate) fn captured_key(index: u8, key: KeyCode, program: DriveProgram) -> Option<DriveProgram> {
    let bound = drive_key(key)?;
    let state = program.state(index)?;
    let release = state
        .trigger()
        .map_or(DriveRelease::Latch, DriveTrigger::release);
    program
        .with_state(
            index,
            state.with_trigger(Some(DriveTrigger::new(bound, release))),
        )
        .ok()
}

/// Steps a state reference through `default -> S1 -> .. -> Sn -> default`.
/// Cycles what a state does when its key goes up: stay, or hand off to each
/// state in turn.
fn cycled_release(state: DriveState, len: usize) -> Option<DriveState> {
    let trigger = state.trigger()?;
    let release = cycled_state(trigger.release().target(), len)
        .map_or(DriveRelease::Latch, DriveRelease::RevertTo);
    state
        .with_trigger(Some(DriveTrigger::new(trigger.key(), release)))
        .into()
}

fn release_text(release: DriveRelease) -> String {
    match release {
        DriveRelease::Latch => "stay".to_owned(),
        DriveRelease::RevertTo(state) => format!("→S{}", state + 1),
    }
}

fn cycled_state(current: Option<u8>, len: usize) -> Option<u8> {
    let last = u8::try_from(len).unwrap_or(u8::MAX).saturating_sub(1);
    match current {
        None => Some(0),
        Some(state) if state >= last => None,
        Some(state) => Some(state + 1),
    }
}

/// Why a click on this cell changed nothing, when that is worth saying.
///
/// A cell that silently ignores clicks reads as broken, and these are the two
/// that legitimately have nothing to cycle through.
fn cell_click_hint(kind: DriveCellKind, program: DriveProgram) -> Option<String> {
    match kind {
        DriveCellKind::Next(index) | DriveCellKind::Release(index) => {
            let state = program.state(index)?;
            (state.dwell().is_none() && state.trigger().is_none()).then(|| {
                format!(
                    "S{} has no key and no dwell, so nothing makes it hand off — bind a key or give it a dwell",
                    index + 1
                )
            })
        }
        DriveCellKind::AddState => Some(format!("A row holds at most {MAX_DRIVE_STATES} states")),
        DriveCellKind::RemoveState(_) => Some("A row keeps at least one state".to_owned()),
        _ => None,
    }
}

/// Text shown in one cell.
pub(crate) fn cell_text(kind: DriveCellKind, limits: DriveLimits, program: DriveProgram) -> String {
    match kind {
        DriveCellKind::MaxSpeed => format!("{:.0}°/s", limits.max_speed_rad_s().to_degrees()),
        DriveCellKind::Torque => {
            let torque = limits.max_torque_newton_meters();
            if torque.is_infinite() {
                "unlimited".to_owned()
            } else {
                format!("{torque:.0} N·m")
            }
        }
        DriveCellKind::ToggleTravelLimits => limits.angle_limits().map_or_else(
            || "free".to_owned(),
            |(minimum, maximum)| {
                format!("{:.0}°..{:.0}°", minimum.to_degrees(), maximum.to_degrees())
            },
        ),
        DriveCellKind::ToggleLoop => if program.loops() { "loop" } else { "once" }.to_owned(),
        DriveCellKind::Mode(index) => {
            program
                .state(index)
                .map_or_else(String::new, |state| match state.target() {
                    DriveTarget::Angle(_) => "angle".to_owned(),
                    DriveTarget::Speed(_) => "speed".to_owned(),
                })
        }
        DriveCellKind::Value(index) => {
            program
                .state(index)
                .map_or_else(String::new, |state| match state.target() {
                    DriveTarget::Angle(angle) => format!("{:.0}°", angle.to_degrees()),
                    DriveTarget::Speed(speed) => format!("{speed:.1}/s"),
                })
        }
        DriveCellKind::Dwell(index) => program
            .state(index)
            .and_then(DriveState::dwell)
            .map_or_else(
                || "—".to_owned(),
                |dwell| format!("{:.1}s", dwell.seconds()),
            ),
        DriveCellKind::Next(index) => {
            program
                .state(index)
                .map_or_else(String::new, |state| match state.dwell() {
                    Some(dwell) => dwell
                        .next()
                        .map_or_else(|| "next".to_owned(), |next| format!("→S{}", next + 1)),
                    // No dwell, so the hand-off belongs to the key going up.
                    None => state
                        .trigger()
                        .map_or_else(|| "—".to_owned(), |trigger| release_text(trigger.release())),
                })
        }
        DriveCellKind::Key(index) => program
            .state(index)
            .and_then(DriveState::trigger)
            .map_or_else(|| "—".to_owned(), |trigger| trigger.key().to_string()),
        DriveCellKind::Release(index) => program
            .state(index)
            .and_then(DriveState::trigger)
            .map_or_else(|| "—".to_owned(), |trigger| release_text(trigger.release())),
        DriveCellKind::AddState => "+".to_owned(),
        DriveCellKind::RemoveState(index) => format!("−S{}", index + 1),
    }
}

/// Spawns the empty panel shell. Rows are filled in as the table rebuilds.
pub(crate) fn spawn(commands: &mut Commands) {
    commands.spawn((
        Name::new("Control block panel"),
        ControlPanelRoot,
        Node {
            position_type: PositionType::Absolute,
            left: px(PANEL_MARGIN),
            top: px(PANEL_MARGIN),
            bottom: px(PANEL_MARGIN),
            width: px(PANEL_WIDTH),
            max_width: percent(94),
            padding: UiRect::all(px(18)),
            flex_direction: FlexDirection::Column,
            row_gap: px(10),
            border: UiRect::all(px(2)),
            border_radius: BorderRadius::all(px(10)),
            ..default()
        },
        BackgroundColor(PANEL_BACKGROUND),
        BorderColor::all(PANEL_BORDER),
        GlobalZIndex(90),
        Visibility::Hidden,
        FocusPolicy::Block,
    ));
}

/// Rebuilds the table for the control block the panel is showing.
pub(crate) fn rebuild(
    commands: &mut Commands,
    root: Entity,
    state: &ControlPanelState,
    graph: &ConstructionGraph,
) {
    commands.entity(root).despawn_related::<Children>();
    let Some(controller) = state.controller() else {
        return;
    };
    let rows = panel_rows(graph, controller);
    let typing = state.typing();
    let scroll = state.scroll;
    commands.entity(root).with_children(|panel| {
        panel.spawn((
            Text::new("CONTROL BLOCK"),
            TextFont {
                font_size: FontSize::Px(20.0),
                ..default()
            },
            TextColor(LABEL),
        ));
        if rows.is_empty() {
            panel.spawn((
                Text::new("No bearings wired. Use the Connector tool, then reopen with E."),
                TextFont {
                    font_size: FontSize::Px(14.0),
                    ..default()
                },
                TextColor(HEADING),
            ));
            return;
        }
        panel.spawn((
            Text::new(if state.is_capturing() {
                "Press a key to bind it, or Escape to cancel."
            } else {
                "Click a cell to change it. Speed, torque, target, and dwell accept typed numbers: Enter commits, Escape cancels. Torque and dwell also take \"none\" or \"inf\", which is what an empty cell commits. With no dwell, a state hands off when its key goes up, and \"then\" names where."
            }),
            TextFont {
                font_size: FontSize::Px(14.0),
                ..default()
            },
            TextColor(HEADING),
        ));
        panel
            .spawn((
                ControlPanelScroll,
                ScrollPosition(scroll),
                Node {
                    flex_direction: FlexDirection::Column,
                    // `flex_grow` claims the space the title and hint leave, and
                    // a zero minimum lets it shrink to that instead of pushing
                    // the table past the panel's fixed height.
                    flex_grow: 1.0,
                    min_height: px(0),
                    overflow: Overflow::scroll(),
                    ..default()
                },
            ))
            .with_children(|table| {
                for (index, row) in rows.iter().enumerate() {
                    let Some(spec) = graph.drive_link(row.primary) else {
                        continue;
                    };
                    spawn_row(table, index, row.primary, spec.limits, spec.program, typing);
                }
            });
    });
}

fn spawn_row(
    parent: &mut ChildSpawnerCommands<'_>,
    index: usize,
    link: DriveLinkId,
    limits: DriveLimits,
    program: DriveProgram,
    typing: Option<(DriveCell, &str)>,
) {
    parent
        .spawn(Node {
            flex_direction: FlexDirection::Column,
            row_gap: px(4),
            margin: UiRect::top(px(8)),
            ..default()
        })
        .with_children(|group| {
            group.spawn((
                Text::new(format!("Joint {}", index + 1)),
                TextFont {
                    font_size: FontSize::Px(15.0),
                    ..default()
                },
                TextColor(LABEL),
            ));
            spawn_header(group, &["max speed", "torque", "travel", "repeat"], 0);
            group
                .spawn(Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: px(6),
                    ..default()
                })
                .with_children(|line| {
                    for kind in [
                        DriveCellKind::MaxSpeed,
                        DriveCellKind::Torque,
                        DriveCellKind::ToggleTravelLimits,
                        DriveCellKind::ToggleLoop,
                    ] {
                        spawn_cell(line, link, kind, limits, program, typing);
                    }
                });
            spawn_header(
                group,
                &["mode", "target", "key", "on release", "dwell", "then", ""],
                STATE_LABEL_WIDTH,
            );
            for state in 0..u8::try_from(program.len()).unwrap_or(u8::MAX) {
                group
                    .spawn(Node {
                        flex_direction: FlexDirection::Row,
                        column_gap: px(6),
                        ..default()
                    })
                    .with_children(|line| {
                        line.spawn((
                            Text::new(format!("S{}", state + 1)),
                            TextFont {
                                font_size: FontSize::Px(14.0),
                                ..default()
                            },
                            TextColor(HEADING),
                            Node {
                                width: px(STATE_LABEL_WIDTH),
                                ..default()
                            },
                        ));
                        for kind in [
                            DriveCellKind::Mode(state),
                            DriveCellKind::Value(state),
                            DriveCellKind::Key(state),
                            DriveCellKind::Release(state),
                            DriveCellKind::Dwell(state),
                            DriveCellKind::Next(state),
                            DriveCellKind::RemoveState(state),
                        ] {
                            spawn_cell(line, link, kind, limits, program, typing);
                        }
                    });
            }
            group
                .spawn(Node {
                    flex_direction: FlexDirection::Row,
                    ..default()
                })
                .with_children(|line| {
                    spawn_cell(line, link, DriveCellKind::AddState, limits, program, typing);
                });
        });
}

/// Column labels above a line of cells, indented past the state label.
fn spawn_header(parent: &mut ChildSpawnerCommands<'_>, labels: &[&str], indent: u32) {
    parent
        .spawn(Node {
            flex_direction: FlexDirection::Row,
            column_gap: px(6),
            margin: UiRect::top(px(4)),
            ..default()
        })
        .with_children(|line| {
            if indent > 0 {
                line.spawn(Node {
                    width: px(indent),
                    ..default()
                });
            }
            for label in labels {
                line.spawn((
                    Text::new((*label).to_owned()),
                    TextFont {
                        font_size: FontSize::Px(11.0),
                        ..default()
                    },
                    TextColor(HEADING),
                    Node {
                        min_width: px(74),
                        ..default()
                    },
                ));
            }
        });
}

fn spawn_cell(
    parent: &mut ChildSpawnerCommands<'_>,
    link: DriveLinkId,
    kind: DriveCellKind,
    limits: DriveLimits,
    program: DriveProgram,
    typing: Option<(DriveCell, &str)>,
) {
    let cell = DriveCell { link, kind };
    let focused = typing.filter(|(typed, _)| *typed == cell);
    let text = focused.map_or_else(
        || cell_text(kind, limits, program),
        |(_, buffer)| match kind.open_ended_value() {
            // An empty buffer shows what Enter would commit, so the setting no
            // number can express is visible rather than folklore.
            Some(word) if buffer.is_empty() => format!("{word}_"),
            _ => format!("{buffer}_"),
        },
    );
    parent.spawn((
        Button,
        cell,
        Node {
            min_width: px(74),
            height: px(26),
            padding: UiRect::axes(px(8), px(4)),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            border_radius: BorderRadius::all(px(5)),
            ..default()
        },
        BackgroundColor(if focused.is_some() {
            CELL_FOCUSED
        } else {
            CELL_BACKGROUND
        }),
        children![(
            Text::new(text),
            TextFont {
                font_size: FontSize::Px(14.0),
                ..default()
            },
            TextColor(LABEL),
        )],
    ));
}

/// Handles clicks, typing, and key capture, writing edits back to the graph.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn update(
    mut commands: Commands,
    mut state: ResMut<ControlPanelState>,
    mut graph: ResMut<crate::EditorGraph>,
    mut editor: ResMut<crate::EditorState>,
    mut history: ResMut<crate::EditorHistory>,
    simulation: Res<crate::AppSimulation>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut keystrokes: MessageReader<KeyboardInput>,
    wheel: Res<AccumulatedMouseScroll>,
    root: Single<(Entity, &mut Visibility), With<ControlPanelRoot>>,
    mut table: Query<&mut ScrollPosition, With<ControlPanelScroll>>,
    mut cells: Query<(&Interaction, &DriveCell, &mut BackgroundColor), With<Button>>,
) {
    let (root_entity, mut visibility) = root.into_inner();
    *visibility = if state.is_open() {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };

    if state.is_open() {
        // A control block deleted from under an open panel closes it.
        if !state
            .controller()
            .is_some_and(|part| graph.0.is_controller(part))
        {
            state.close();
        }
    }

    if state.is_open() {
        handle_key_capture(
            &mut state,
            &mut graph,
            &mut editor,
            &mut history,
            &simulation,
            &keyboard,
        );
        handle_typing(
            &mut state,
            &mut graph,
            &mut editor,
            &mut history,
            &simulation,
            &mut keystrokes,
        );
        handle_clicks(
            &mut state,
            &mut graph,
            &mut editor,
            &mut history,
            &simulation,
            &mut cells,
        );
        handle_scroll(&mut state, &wheel, &mut table);
    } else {
        keystrokes.clear();
    }

    if state.take_dirty() {
        rebuild(&mut commands, root_entity, &state, &graph.0);
    }
}

/// Scrolls the table with the wheel.
///
/// The camera already ignores the wheel while the panel blocks the pointer, so
/// the whole gesture belongs to the table. Layout clamps an offset past the
/// end, so the clamped value is read back rather than tracked separately.
fn handle_scroll(
    state: &mut ControlPanelState,
    wheel: &AccumulatedMouseScroll,
    table: &mut Query<&mut ScrollPosition, With<ControlPanelScroll>>,
) {
    let Ok(mut position) = table.single_mut() else {
        return;
    };
    let scale = match wheel.unit {
        MouseScrollUnit::Line => SCROLL_LINE_PIXELS,
        MouseScrollUnit::Pixel => 1.0,
    };
    let delta = wheel.delta * scale;
    if delta != Vec2::ZERO {
        position.0 = (position.0 - delta).max(Vec2::ZERO);
    }
    state.scroll = position.0;
}

fn handle_key_capture(
    state: &mut ControlPanelState,
    graph: &mut crate::EditorGraph,
    editor: &mut crate::EditorState,
    history: &mut crate::EditorHistory,
    simulation: &crate::AppSimulation,
    keyboard: &ButtonInput<KeyCode>,
) {
    let Some(cell) = state.capturing else {
        return;
    };
    let DriveCellKind::Key(index) = cell.kind else {
        state.capturing = None;
        return;
    };
    if keyboard.just_pressed(KeyCode::Escape) {
        state.capturing = None;
        state.mark_dirty();
        return;
    }
    let Some(pressed) = keyboard
        .get_just_pressed()
        .find_map(|key| drive_key(*key).map(|_| *key))
    else {
        return;
    };
    let Some(spec) = graph.0.drive_link(cell.link).copied() else {
        state.capturing = None;
        return;
    };
    state.capturing = None;
    let Some(program) = captured_key(index, pressed, spec.program) else {
        editor.feedback = Some("That key is already bound on this joint".to_owned());
        state.mark_dirty();
        return;
    };
    write_row(
        state,
        graph,
        editor,
        history,
        simulation,
        cell.link,
        spec.limits,
        program,
    );
}

fn handle_typing(
    state: &mut ControlPanelState,
    graph: &mut crate::EditorGraph,
    editor: &mut crate::EditorState,
    history: &mut crate::EditorHistory,
    simulation: &crate::AppSimulation,
    keystrokes: &mut MessageReader<KeyboardInput>,
) {
    let pressed = keystrokes
        .read()
        .filter(|stroke| stroke.state.is_pressed())
        .map(|stroke| stroke.logical_key.clone())
        .collect::<Vec<_>>();
    let Some((cell, buffer)) = state.typing.clone() else {
        return;
    };
    let mut buffer = buffer;
    for key in pressed {
        match number_entry_action(&key, cell.kind.open_ended_value().is_some()) {
            NumberEntryAction::Commit => {
                let Some(spec) = graph.0.drive_link(cell.link).copied() else {
                    state.typing = None;
                    return;
                };
                state.typing = None;
                let Some((limits, program)) =
                    committed_value(cell.kind, &buffer, spec.limits, spec.program)
                else {
                    editor.feedback =
                        Some(format!("\"{buffer}\" is not a value this cell accepts"));
                    state.mark_dirty();
                    return;
                };
                write_row(
                    state, graph, editor, history, simulation, cell.link, limits, program,
                );
                return;
            }
            NumberEntryAction::Cancel => {
                state.typing = None;
                state.mark_dirty();
                return;
            }
            action => {
                if let Some(edited) = edited_buffer(&buffer, action) {
                    buffer = edited;
                    state.mark_dirty();
                }
            }
        }
    }
    state.typing = Some((cell, buffer));
}

fn handle_clicks(
    state: &mut ControlPanelState,
    graph: &mut crate::EditorGraph,
    editor: &mut crate::EditorState,
    history: &mut crate::EditorHistory,
    simulation: &crate::AppSimulation,
    cells: &mut Query<(&Interaction, &DriveCell, &mut BackgroundColor), With<Button>>,
) {
    let mut pressed_now = None;
    for (interaction, cell, mut background) in cells.iter_mut() {
        let focused = state.typing().is_some_and(|(typed, _)| typed == *cell)
            || state.capturing == Some(*cell);
        *background = BackgroundColor(match interaction {
            Interaction::Pressed => CELL_FOCUSED,
            Interaction::Hovered => CELL_HOVER,
            Interaction::None if focused => CELL_FOCUSED,
            Interaction::None => CELL_BACKGROUND,
        });
        if *interaction == Interaction::Pressed {
            pressed_now = Some(*cell);
        }
    }
    // Act once per press, not once per frame the button stays down.
    let clicked = pressed_now.filter(|cell| state.held != Some(*cell));
    state.held = pressed_now;
    let Some(cell) = clicked else {
        return;
    };
    let Some(spec) = graph.0.drive_link(cell.link).copied() else {
        return;
    };

    if cell.kind.is_typed() {
        state.begin_typing(cell);
        return;
    }
    if let DriveCellKind::Key(index) = cell.kind
        && spec
            .program
            .state(index)
            .and_then(DriveState::trigger)
            .is_none()
    {
        state.capturing = Some(cell);
        state.mark_dirty();
        return;
    }
    let Some((limits, program)) = clicked_value(cell.kind, spec.limits, spec.program) else {
        if let Some(hint) = cell_click_hint(cell.kind, spec.program) {
            editor.feedback = Some(hint);
            state.mark_dirty();
        }
        return;
    };
    write_row(
        state, graph, editor, history, simulation, cell.link, limits, program,
    );
}

/// Writes one row's edit to every wire behind it.
///
/// While building this is an undoable edit like any other. While simulating it
/// skips history and only marks the drive rows dirty, which is the one write
/// the running GPU state accepts.
#[allow(clippy::too_many_arguments)] // One edit touches panel, graph, editor, history, and mode.
fn write_row(
    state: &mut ControlPanelState,
    graph: &mut crate::EditorGraph,
    editor: &mut crate::EditorState,
    history: &mut crate::EditorHistory,
    simulation: &crate::AppSimulation,
    link: DriveLinkId,
    limits: DriveLimits,
    program: DriveProgram,
) {
    let Some(controller) = state.controller() else {
        return;
    };
    let rows = panel_rows(&graph.0, controller);
    let Some(row) = rows.iter().find(|row| row.links.contains(&link)) else {
        return;
    };
    let commands = set_row_commands(row, limits, program);
    let previous = crate::EditorSnapshot::capture(&graph.0, editor);
    match graph.0.apply_batch(commands) {
        Ok(_) => {
            if simulation.is_running() {
                editor.drive_rows_dirty = true;
            } else {
                history.commit(previous);
                editor.construction_mesh_dirty = true;
            }
            state.mark_dirty();
        }
        Err(error) => {
            editor.feedback = Some(error.to_string());
            state.mark_dirty();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ControlPanelRoot, ControlPanelScroll, ControlPanelState, DriveCell, DriveCellKind,
        NumberEntryAction, SCROLL_LINE_PIXELS, captured_key, cell_click_hint, cell_text,
        clicked_value, committed_value, edited_buffer, handle_scroll, number_entry_action, rebuild,
    };
    use bevy::ecs::world::CommandQueue;
    use bevy::input::keyboard::Key;
    use bevy::input::mouse::{AccumulatedMouseScroll, MouseScrollUnit};
    use bevy::prelude::*;
    use bevy::ui::ScrollPosition;
    use mechanic_core::{
        BearingSpec, BuildCommand, BuildOutcome, BuildPose, ConstructionGraph, ControllerSpec,
        CuboidSpec, DriveDwell, DriveKey, DriveLimits, DriveLinkSpec, DriveProgram, DriveRelease,
        DriveState, DriveTarget, DriveTrigger, FaceKind, FaceRef, GridRotation, PartId,
    };

    fn program(states: &[DriveState]) -> DriveProgram {
        DriveProgram::new(states, false).expect("test program is valid")
    }

    fn angle_state(degrees: f32) -> DriveState {
        DriveState::new(DriveTarget::Angle(degrees.to_radians())).expect("angle is in range")
    }

    #[test]
    fn number_entry_accepts_digits_sign_and_point_and_ignores_everything_else() {
        assert_eq!(
            number_entry_action(&Key::Character("4".into()), false),
            NumberEntryAction::Insert('4')
        );
        assert_eq!(
            number_entry_action(&Key::Character("-".into()), false),
            NumberEntryAction::Insert('-')
        );
        assert_eq!(
            number_entry_action(&Key::Character("x".into()), false),
            NumberEntryAction::Ignore
        );
        assert_eq!(
            number_entry_action(&Key::Enter, false),
            NumberEntryAction::Commit
        );
        assert_eq!(
            number_entry_action(&Key::Escape, false),
            NumberEntryAction::Cancel
        );
        assert_eq!(
            number_entry_action(&Key::Space, false),
            NumberEntryAction::Ignore
        );
        // Letters reach only the cells that mean something by a word.
        assert_eq!(
            number_entry_action(&Key::Character("n".into()), true),
            NumberEntryAction::Insert('n')
        );

        assert_eq!(
            edited_buffer("3", NumberEntryAction::Insert('7')).as_deref(),
            Some("37")
        );
        assert_eq!(
            edited_buffer("37", NumberEntryAction::Backspace).as_deref(),
            Some("3")
        );
        assert_eq!(edited_buffer("3", NumberEntryAction::Commit), None);
    }

    #[test]
    fn typed_angles_are_degrees_and_bad_input_leaves_the_value_alone() {
        let limits = DriveLimits::default();
        let base = program(&[angle_state(0.0)]);

        let (_, edited) =
            committed_value(DriveCellKind::Value(0), "90", limits, base).expect("90 is a number");
        let angle = edited
            .state(0)
            .and_then(|state| state.target().angle())
            .expect("state 0 holds an angle");
        assert!((angle - std::f32::consts::FRAC_PI_2).abs() < 1.0e-5);

        assert!(committed_value(DriveCellKind::Value(0), "many", limits, base).is_none());
        // 400° is beyond one full turn, so the core setter rejects it.
        assert!(committed_value(DriveCellKind::Value(0), "400", limits, base).is_none());
    }

    #[test]
    fn an_empty_torque_cell_means_unlimited_and_an_empty_dwell_clears_it() {
        let limits = DriveLimits::new(2.0, 40.0, None).expect("limits are valid");
        let timed = angle_state(0.0)
            .with_dwell(Some(DriveDwell::new(2.0, None).expect("dwell is in range")));
        let base = program(&[timed]);

        let (unlimited, _) =
            committed_value(DriveCellKind::Torque, "  ", limits, base).expect("empty is accepted");
        assert!(unlimited.max_torque_newton_meters().is_infinite());

        let (_, cleared) =
            committed_value(DriveCellKind::Dwell(0), "", limits, base).expect("empty is accepted");
        assert!(cleared.state(0).and_then(DriveState::dwell).is_none());
    }

    #[test]
    fn dwell_and_torque_can_be_typed_as_none_or_inf() {
        let limits = DriveLimits::new(2.0, 40.0, None).expect("limits are valid");
        let timed = angle_state(0.0)
            .with_dwell(Some(DriveDwell::new(2.0, None).expect("dwell is in range")));
        let base = program(&[timed]);

        // A dwell that never elapses and no dwell at all are the same setting:
        // the state waits for a key instead of advancing itself.
        for spelling in ["none", "inf", "Never", " INFINITE "] {
            let (_, cleared) = committed_value(DriveCellKind::Dwell(0), spelling, limits, base)
                .unwrap_or_else(|| panic!("{spelling} is accepted"));
            assert!(
                cleared.state(0).and_then(DriveState::dwell).is_none(),
                "{spelling}"
            );
        }
        for spelling in ["unlimited", "inf", "none"] {
            let (open, _) = committed_value(DriveCellKind::Torque, spelling, limits, base)
                .unwrap_or_else(|| panic!("{spelling} is accepted"));
            assert!(open.max_torque_newton_meters().is_infinite(), "{spelling}");
        }

        // A word that is not one of those is still rejected.
        assert!(committed_value(DriveCellKind::Dwell(0), "soon", limits, base).is_none());
        assert!(committed_value(DriveCellKind::Torque, "lots", limits, base).is_none());
    }

    #[test]
    fn a_state_with_no_dwell_hands_off_through_its_then_cell_when_the_key_goes_up() {
        let limits = DriveLimits::default();
        let held = angle_state(30.0).with_trigger(Some(DriveTrigger::new(
            DriveKey::new('A').expect("A is bindable"),
            DriveRelease::Latch,
        )));
        let base = program(&[angle_state(0.0), held]);
        assert!(base.state(1).and_then(DriveState::dwell).is_none());
        assert_eq!(cell_text(DriveCellKind::Next(1), limits, base), "stay");

        let (_, cycled) = clicked_value(DriveCellKind::Next(1), limits, base)
            .expect("a keyed state hands off when its key goes up");
        assert_eq!(
            cycled
                .state(1)
                .and_then(DriveState::trigger)
                .map(DriveTrigger::release),
            Some(DriveRelease::RevertTo(0))
        );
        assert_eq!(cell_text(DriveCellKind::Next(1), limits, cycled), "→S1");
        // The release column is the same setting seen from the other side.
        assert_eq!(cell_text(DriveCellKind::Release(1), limits, cycled), "→S1");

        // With neither a key nor a dwell there is nothing to hand off, and the
        // dead click says why instead of looking broken.
        assert!(clicked_value(DriveCellKind::Next(0), limits, base).is_none());
        assert!(
            cell_click_hint(DriveCellKind::Next(0), base)
                .is_some_and(|hint| hint.contains("no key and no dwell"))
        );
    }

    #[test]
    fn clicking_cycles_mode_loop_and_state_references() {
        let limits = DriveLimits::default();
        let timed = angle_state(0.0)
            .with_dwell(Some(DriveDwell::new(1.0, None).expect("dwell is in range")));
        let base = program(&[timed, angle_state(30.0)]);

        let (_, moded) = clicked_value(DriveCellKind::Mode(0), limits, base).expect("mode cycles");
        assert!(
            moded
                .state(0)
                .and_then(|state| state.target().speed())
                .is_some()
        );

        // A full turn is a valid angle but not a valid speed, so the switch
        // clamps rather than refusing the click.
        let wide =
            program(&[
                DriveState::new(DriveTarget::Angle(mechanic_core::MAX_DRIVE_LIMIT_RADIANS))
                    .expect("a full turn is a valid angle"),
            ]);
        let (_, clamped) =
            clicked_value(DriveCellKind::Mode(0), limits, wide).expect("mode still cycles");
        let speed = clamped
            .state(0)
            .and_then(|state| state.target().speed())
            .expect("the state is now a speed");
        assert!(speed.abs() <= mechanic_core::MAX_DRIVE_SPEED_RAD_S);

        let (_, looped) =
            clicked_value(DriveCellKind::ToggleLoop, limits, base).expect("loop toggles");
        assert!(looped.loops());

        // default -> S1 -> S2 -> default
        let (_, first) = clicked_value(DriveCellKind::Next(0), limits, base).expect("next cycles");
        assert_eq!(
            first
                .state(0)
                .and_then(DriveState::dwell)
                .and_then(DriveDwell::next),
            Some(0)
        );
        let (_, second) =
            clicked_value(DriveCellKind::Next(0), limits, first).expect("next cycles");
        assert_eq!(
            second
                .state(0)
                .and_then(DriveState::dwell)
                .and_then(DriveDwell::next),
            Some(1)
        );
        let (_, wrapped) =
            clicked_value(DriveCellKind::Next(0), limits, second).expect("next cycles");
        assert_eq!(
            wrapped
                .state(0)
                .and_then(DriveState::dwell)
                .and_then(DriveDwell::next),
            None
        );
    }

    #[test]
    fn adding_and_removing_states_keeps_the_program_valid() {
        let limits = DriveLimits::default();
        let base = program(&[angle_state(0.0)]);

        let (_, grown) = clicked_value(DriveCellKind::AddState, limits, base).expect("adds");
        assert_eq!(grown.len(), 2);

        let (_, shrunk) =
            clicked_value(DriveCellKind::RemoveState(1), limits, grown).expect("removes");
        assert_eq!(shrunk.len(), 1);

        // The last state cannot be removed: a program always has one.
        assert!(clicked_value(DriveCellKind::RemoveState(0), limits, shrunk).is_none());
    }

    #[test]
    fn key_capture_binds_and_clicking_a_bound_key_clears_it() {
        let limits = DriveLimits::default();
        let base = program(&[angle_state(0.0), angle_state(30.0)]);

        let bound = captured_key(1, KeyCode::KeyA, base).expect("A is bindable");
        assert_eq!(
            bound
                .state(1)
                .and_then(DriveState::trigger)
                .map(DriveTrigger::key),
            DriveKey::new('A')
        );
        assert!(
            captured_key(1, KeyCode::F1, base).is_none(),
            "F1 is not bindable"
        );

        // The same key twice on one joint is ambiguous, so the bind is refused.
        assert!(captured_key(0, KeyCode::KeyA, bound).is_none());

        let (_, cleared) =
            clicked_value(DriveCellKind::Key(1), limits, bound).expect("a bound key clears");
        assert!(cleared.state(1).and_then(DriveState::trigger).is_none());
    }

    #[test]
    fn cells_read_back_what_they_hold() {
        let limits = DriveLimits::new(1.5, 40.0, Some((-0.5, 0.5))).expect("limits are valid");
        let state = angle_state(30.0)
            .with_trigger(Some(DriveTrigger::new(
                DriveKey::new('A').unwrap(),
                DriveRelease::RevertTo(0),
            )))
            .with_dwell(Some(
                DriveDwell::new(2.0, Some(0)).expect("dwell is in range"),
            ));
        let base = program(&[state]);

        assert_eq!(cell_text(DriveCellKind::Value(0), limits, base), "30°");
        assert_eq!(cell_text(DriveCellKind::Mode(0), limits, base), "angle");
        assert_eq!(cell_text(DriveCellKind::Key(0), limits, base), "A");
        assert_eq!(cell_text(DriveCellKind::Release(0), limits, base), "→S1");
        assert_eq!(cell_text(DriveCellKind::Dwell(0), limits, base), "2.0s");
        assert_eq!(cell_text(DriveCellKind::Next(0), limits, base), "→S1");
        assert_eq!(cell_text(DriveCellKind::Torque, limits, base), "40 N·m");
        assert_eq!(cell_text(DriveCellKind::ToggleLoop, limits, base), "once");
    }

    /// Runs `handle_scroll` once over a table node holding `start`.
    fn scrolled(start: Vec2, delta: Vec2, unit: MouseScrollUnit) -> Vec2 {
        let mut app = App::new();
        app.insert_resource(AccumulatedMouseScroll { delta, unit })
            .init_resource::<ControlPanelState>();
        let table = app
            .world_mut()
            .spawn((ControlPanelScroll, ScrollPosition(start)))
            .id();
        app.add_systems(
            Update,
            |mut state: ResMut<ControlPanelState>,
             wheel: Res<AccumulatedMouseScroll>,
             mut table: Query<&mut ScrollPosition, With<ControlPanelScroll>>| {
                handle_scroll(&mut state, &wheel, &mut table);
            },
        );
        app.update();

        let position = app
            .world()
            .entity(table)
            .get::<ScrollPosition>()
            .expect("the table keeps its scroll position");
        assert_eq!(
            app.world().resource::<ControlPanelState>().scroll,
            position.0,
            "the state must mirror the node"
        );
        position.0
    }

    #[test]
    fn wheel_scrolls_the_table_a_line_at_a_time() {
        assert_eq!(
            scrolled(Vec2::ZERO, Vec2::new(0.0, -2.0), MouseScrollUnit::Line),
            Vec2::new(0.0, 2.0 * SCROLL_LINE_PIXELS),
        );
        assert_eq!(
            scrolled(Vec2::ZERO, Vec2::new(0.0, -30.0), MouseScrollUnit::Pixel),
            Vec2::new(0.0, 30.0),
        );
    }

    #[test]
    fn scrolling_back_past_the_top_stops_there() {
        assert_eq!(
            scrolled(
                Vec2::new(0.0, 10.0),
                Vec2::new(0.0, 40.0),
                MouseScrollUnit::Pixel,
            ),
            Vec2::ZERO,
            "a wide panel must not drift above its first joint"
        );
    }

    /// A control block wired to one bearing between a base and a rotor.
    fn wired_control_block() -> (ConstructionGraph, PartId) {
        fn spawned(outcome: BuildOutcome) -> PartId {
            match outcome {
                BuildOutcome::Spawned(part) => part,
                other => panic!("expected a spawn, got {other:?}"),
            }
        }

        let mut graph = ConstructionGraph::new();
        let cuboid = |dimensions: [u8; 3], units: IVec3| {
            CuboidSpec::new(dimensions, BuildPose::new(units, GridRotation::default()))
                .expect("test dimensions are in range")
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
        (graph, controller)
    }

    /// Every entity beneath `root`, so a test can ask what a subtree contains.
    fn descendants(world: &World, root: Entity) -> Vec<Entity> {
        let mut found = Vec::new();
        let mut pending = vec![root];
        while let Some(entity) = pending.pop() {
            if let Some(children) = world.entity(entity).get::<Children>() {
                for &child in children {
                    found.push(child);
                    pending.push(child);
                }
            }
        }
        found
    }

    #[test]
    fn a_wired_block_puts_its_joints_inside_the_scrolling_table() {
        let (graph, controller) = wired_control_block();
        let mut app = App::new();
        let root = app.world_mut().spawn(ControlPanelRoot).id();
        let mut state = ControlPanelState::default();
        state.open(controller);

        let mut queue = CommandQueue::default();
        rebuild(
            &mut Commands::new(&mut queue, app.world()),
            root,
            &state,
            &graph,
        );
        queue.apply(app.world_mut());

        let table = descendants(app.world(), root)
            .into_iter()
            .find(|entity| app.world().entity(*entity).contains::<ControlPanelScroll>())
            .expect("a wired block builds a scrolling table");
        assert!(
            app.world().entity(table).get::<ScrollPosition>().is_some(),
            "the table carries a scroll position"
        );
        assert!(
            descendants(app.world(), table)
                .into_iter()
                .any(|entity| app.world().entity(entity).contains::<DriveCell>()),
            "the joint's cells must sit inside the table, so they scroll with it"
        );
    }
}

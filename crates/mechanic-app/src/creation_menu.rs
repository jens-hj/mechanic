//! Modal that saves the current creation and opens a saved or preset one.
//!
//! One screen does both jobs: type a name and press Enter to save, or click a
//! row to open it. Rows come from the creations directory, so the list is
//! rebuilt from disk each time the modal opens rather than tracked live.

use std::path::{Path, PathBuf};

use bevy::{
    input::{
        keyboard::{Key, KeyboardInput},
        mouse::{AccumulatedMouseScroll, MouseScrollUnit},
    },
    prelude::*,
    ui::{FocusPolicy, ScrollPosition},
};

use crate::{
    creation_store::{SavedCreation, slug},
    showcase::CreationPreset,
};

const BUTTON_BACKGROUND: Color = Color::srgba(0.06, 0.085, 0.12, 0.98);
const BUTTON_HOVER_BACKGROUND: Color = Color::srgba(0.10, 0.16, 0.21, 0.98);
const BUTTON_PRESSED_BACKGROUND: Color = Color::srgba(0.08, 0.28, 0.35, 0.98);
const DANGER_BACKGROUND: Color = Color::srgba(0.24, 0.07, 0.08, 0.98);
const FIELD_BACKGROUND: Color = Color::srgba(0.02, 0.05, 0.07, 0.99);
const HEADING_COLOR: Color = Color::srgb(0.55, 0.66, 0.75);
const SUBTEXT_COLOR: Color = Color::srgb(0.68, 0.75, 0.81);
const NOTICE_COLOR: Color = Color::srgb(1.0, 0.76, 0.28);

/// Longest display name the field accepts.
const MAX_NAME_LENGTH: usize = 60;

/// Pixels one wheel line scrolls the list by.
const SCROLL_LINE_PIXELS: f32 = 24.0;

/// What a keystroke does to the name being typed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NameEntryAction {
    /// Append a character to the name.
    Insert(char),
    /// Drop the last character.
    Backspace,
    /// Save under the typed name.
    Commit,
    /// Clear the name, or close the modal when it is already empty.
    Cancel,
    /// Not part of name entry.
    Ignore,
}

/// Maps one logical key to its name-entry action.
///
/// A creation name is free text, so unlike the control panel's numeric cells
/// this takes any single printable character, space included. Everything else
/// is ignored rather than swallowed.
pub(crate) fn name_entry_action(key: &Key) -> NameEntryAction {
    match key {
        Key::Character(text) => {
            let mut characters = text.chars();
            match (characters.next(), characters.next()) {
                (Some(symbol), None) if !symbol.is_control() => NameEntryAction::Insert(symbol),
                _ => NameEntryAction::Ignore,
            }
        }
        Key::Space => NameEntryAction::Insert(' '),
        Key::Backspace => NameEntryAction::Backspace,
        Key::Enter => NameEntryAction::Commit,
        Key::Escape => NameEntryAction::Cancel,
        _ => NameEntryAction::Ignore,
    }
}

/// A destructive action waiting for a second press before it happens.
#[derive(Clone, Debug, PartialEq, Eq)]
enum PendingConfirm {
    /// Enter would write over a creation that already exists.
    Replace,
    /// This creation's file would be removed.
    Delete(PathBuf),
}

/// What the modal decided, handed to a world-mutating system one step later.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum CreationRequest {
    /// Write the current creation under this display name.
    Save(String),
    /// Open the creation stored at this path.
    Load(PathBuf),
    /// Remove the creation stored at this path.
    Delete(PathBuf),
    /// Open one of the built-in scenes.
    LoadPreset(CreationPreset),
}

/// Live modal state: what is typed, what is on disk, and what was decided.
#[derive(Resource, Debug, Default)]
pub(crate) struct CreationMenuState {
    open: bool,
    name: String,
    entries: Vec<SavedCreation>,
    /// Directory the rows came from, shown so the standard location is never a
    /// mystery.
    directory: PathBuf,
    confirming: Option<PendingConfirm>,
    notice: Option<String>,
    /// How far the list is scrolled. A rebuild respawns the dialog, so the
    /// offset lives here rather than only on the node.
    scroll: f32,
    /// Action held down last frame. `Interaction::Pressed` stays set while a
    /// button is held, and a rebuild re-spawns it under the same cursor, so a
    /// confirmation would otherwise resolve itself in one click.
    held: Option<CreationMenuAction>,
    dirty: bool,
    blocks_pointer: bool,
    requested: Option<CreationRequest>,
}

impl CreationMenuState {
    /// Clears the one-frame pointer latch at the top of each frame.
    pub(crate) fn begin_frame(&mut self) {
        self.blocks_pointer = false;
    }

    /// Opens the modal on a freshly read directory listing.
    pub(crate) fn open(&mut self, entries: Vec<SavedCreation>, name: String, directory: PathBuf) {
        self.open = true;
        self.name = name;
        self.entries = entries;
        self.directory = directory;
        self.confirming = None;
        self.notice = None;
        self.held = None;
        self.scroll = 0.0;
        self.blocks_pointer = true;
        self.dirty = true;
    }

    /// Closes the modal, discarding a half-typed name.
    pub(crate) fn close(&mut self) {
        self.open = false;
        self.name.clear();
        self.confirming = None;
        self.notice = None;
        self.held = None;
        self.scroll = 0.0;
        self.blocks_pointer = true;
        self.dirty = true;
    }

    /// Replaces the directory listing shown by an already-open modal.
    pub(crate) fn set_entries(&mut self, entries: Vec<SavedCreation>) {
        self.entries = entries;
        self.dirty = true;
    }

    /// Shows a one-line message inside the modal.
    pub(crate) fn notify(&mut self, notice: impl Into<String>) {
        self.notice = Some(notice.into());
        self.dirty = true;
    }

    /// Whether the modal is showing.
    pub(crate) const fn is_open(&self) -> bool {
        self.open
    }

    /// Whether the modal is swallowing pointer input this frame.
    pub(crate) const fn blocks_pointer(&self) -> bool {
        self.blocks_pointer || self.open
    }

    /// Whether the modal owns the keyboard.
    ///
    /// True whenever it is open: the name field takes letters and digits, and a
    /// keystroke must never both type a character and fire a global shortcut.
    pub(crate) const fn blocks_keyboard(&self) -> bool {
        self.open
    }

    /// Takes the pending decision.
    pub(crate) fn take_request(&mut self) -> Option<CreationRequest> {
        self.requested.take()
    }

    /// Marks the modal for a rebuild.
    const fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Takes the pending rebuild request.
    const fn take_dirty(&mut self) -> bool {
        let dirty = self.dirty;
        self.dirty = false;
        dirty
    }

    /// The saved creation the typed name would land on, when one exists.
    fn colliding_entry(&self) -> Option<&SavedCreation> {
        let stem = slug(self.name.trim());
        self.entries.iter().find(|entry| {
            entry
                .path
                .file_stem()
                .is_some_and(|candidate| candidate == stem.as_str())
        })
    }

    /// Requests a save, asking once before it writes over an existing file.
    fn commit_name(&mut self) {
        let name = self.name.trim().to_owned();
        if name.is_empty() {
            self.notify("Type a name first");
            return;
        }
        if self.confirming != Some(PendingConfirm::Replace)
            && let Some(existing) = self.colliding_entry()
        {
            let message = format!("Press Enter again to replace \"{}\"", existing.name);
            self.confirming = Some(PendingConfirm::Replace);
            self.notify(message);
            return;
        }
        self.requested = Some(CreationRequest::Save(name));
        self.close();
    }

    /// Requests a delete, asking once before it removes a file.
    fn confirm_delete(&mut self, path: &Path) {
        if self.confirming == Some(PendingConfirm::Delete(path.to_path_buf())) {
            self.requested = Some(CreationRequest::Delete(path.to_path_buf()));
            self.confirming = None;
            self.notice = None;
            self.mark_dirty();
            return;
        }
        self.confirming = Some(PendingConfirm::Delete(path.to_path_buf()));
        self.notify("Click again to delete for good");
    }

    /// Clears a half-typed name, or closes the modal when it is already empty.
    fn cancel(&mut self) {
        if self.name.is_empty() {
            self.close();
            return;
        }
        self.name.clear();
        self.confirming = None;
        self.notice = None;
        self.mark_dirty();
    }

    fn edit_name(&mut self, action: NameEntryAction) {
        match action {
            NameEntryAction::Insert(symbol) => {
                if self.name.chars().count() >= MAX_NAME_LENGTH {
                    return;
                }
                self.name.push(symbol);
            }
            NameEntryAction::Backspace => {
                if self.name.pop().is_none() {
                    return;
                }
            }
            NameEntryAction::Commit | NameEntryAction::Cancel | NameEntryAction::Ignore => return,
        }
        // Editing the name withdraws a replace prompt that named the old one.
        if self.confirming == Some(PendingConfirm::Replace) {
            self.confirming = None;
        }
        self.notice = None;
        self.mark_dirty();
    }
}

/// A clickable target in the modal.
#[derive(Clone, Debug, PartialEq, Component)]
pub(crate) enum CreationMenuAction {
    /// Save under the typed name.
    Save,
    /// Open this saved creation.
    LoadFile(PathBuf),
    /// Remove this saved creation.
    DeleteFile(PathBuf),
    /// Open a built-in scene.
    LoadPreset(CreationPreset),
    /// Close without doing anything.
    Cancel,
}

/// Root of the modal's UI tree. Rows are filled in as the list rebuilds.
#[derive(Component)]
pub(crate) struct CreationMenuRoot;

/// The scrollable dialog inside the backdrop.
#[derive(Component)]
pub(crate) struct CreationMenuDialog;

/// Spawns the empty modal shell.
pub(crate) fn spawn(commands: &mut Commands) {
    commands.spawn((
        Name::new("Creation picker backdrop"),
        CreationMenuRoot,
        Node {
            position_type: PositionType::Absolute,
            left: px(0),
            top: px(0),
            width: percent(100),
            height: percent(100),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        },
        BackgroundColor(Color::srgba(0.005, 0.008, 0.012, 0.76)),
        GlobalZIndex(100),
        Visibility::Hidden,
        FocusPolicy::Block,
    ));
}

/// Rebuilds the modal's contents from the current state.
fn rebuild(commands: &mut Commands, root: Entity, state: &CreationMenuState) {
    commands.entity(root).despawn_related::<Children>();
    if !state.open {
        return;
    }
    let confirming = state.confirming.clone();
    let scroll = state.scroll;
    commands.entity(root).with_children(|backdrop| {
        backdrop
            .spawn((
                Name::new("Creation picker"),
                CreationMenuDialog,
                ScrollPosition(Vec2::new(0.0, scroll)),
                Node {
                    width: px(640),
                    max_height: percent(88),
                    padding: UiRect::all(px(24)),
                    flex_direction: FlexDirection::Column,
                    row_gap: px(10),
                    border: UiRect::all(px(2)),
                    border_radius: BorderRadius::all(px(10)),
                    overflow: Overflow::scroll_y(),
                    ..default()
                },
                BackgroundColor(Color::srgba(0.018, 0.026, 0.038, 0.99)),
                BorderColor::all(Color::srgba(0.36, 0.48, 0.58, 0.95)),
            ))
            .with_children(|dialog| {
                spawn_title(dialog);
                spawn_name_field(
                    dialog,
                    &state.name,
                    confirming == Some(PendingConfirm::Replace),
                );
                if let Some(notice) = &state.notice {
                    dialog.spawn((
                        Text::new(notice.clone()),
                        TextFont {
                            font_size: FontSize::Px(14.0),
                            ..default()
                        },
                        TextColor(NOTICE_COLOR),
                    ));
                }
                spawn_heading(dialog, "YOUR CREATIONS");
                dialog.spawn((
                    Text::new(format!("Stored in {}", state.directory.display())),
                    TextFont {
                        font_size: FontSize::Px(12.0),
                        ..default()
                    },
                    TextColor(HEADING_COLOR),
                ));
                if state.entries.is_empty() {
                    dialog.spawn((
                        Text::new("Nothing saved yet. Type a name above and press Enter."),
                        TextFont {
                            font_size: FontSize::Px(14.0),
                            ..default()
                        },
                        TextColor(SUBTEXT_COLOR),
                    ));
                }
                for entry in &state.entries {
                    spawn_saved_row(
                        dialog,
                        entry,
                        confirming == Some(PendingConfirm::Delete(entry.path.clone())),
                    );
                }
                spawn_heading(dialog, "PRESETS");
                for preset in CreationPreset::ALL {
                    spawn_preset_button(dialog, preset);
                }
                spawn_cancel_button(dialog);
            });
    });
}

fn spawn_title(parent: &mut ChildSpawnerCommands<'_>) {
    parent.spawn((
        Text::new("CREATIONS"),
        TextFont {
            font_size: FontSize::Px(25.0),
            ..default()
        },
        TextColor(Color::WHITE),
    ));
    parent.spawn((
        Text::new(
            "Type a name and press Enter to save. Click a creation to open it — that replaces the current construction and can be undone.",
        ),
        TextFont {
            font_size: FontSize::Px(15.0),
            ..default()
        },
        TextColor(SUBTEXT_COLOR),
        Node {
            margin: UiRect::bottom(px(5)),
            ..default()
        },
    ));
}

fn spawn_heading(parent: &mut ChildSpawnerCommands<'_>, text: &str) {
    parent.spawn((
        Text::new(text.to_owned()),
        TextFont {
            font_size: FontSize::Px(13.0),
            ..default()
        },
        TextColor(HEADING_COLOR),
        Node {
            margin: UiRect::top(px(8)),
            ..default()
        },
    ));
}

fn spawn_name_field(parent: &mut ChildSpawnerCommands<'_>, name: &str, replacing: bool) {
    parent
        .spawn(Node {
            width: percent(100),
            column_gap: px(10),
            align_items: AlignItems::Center,
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new("Save current as"),
                TextFont {
                    font_size: FontSize::Px(15.0),
                    ..default()
                },
                TextColor(SUBTEXT_COLOR),
            ));
            row.spawn((
                Node {
                    flex_grow: 1.0,
                    min_height: px(34),
                    padding: UiRect::axes(px(10), px(7)),
                    justify_content: JustifyContent::FlexStart,
                    align_items: AlignItems::Center,
                    border_radius: BorderRadius::all(px(6)),
                    ..default()
                },
                BackgroundColor(FIELD_BACKGROUND),
                children![(
                    Text::new(format!("{name}_")),
                    TextFont {
                        font_size: FontSize::Px(16.0),
                        ..default()
                    },
                    TextColor(Color::WHITE),
                )],
            ));
            row.spawn((
                Button,
                CreationMenuAction::Save,
                Node {
                    min_width: px(96),
                    height: px(34),
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    border_radius: BorderRadius::all(px(6)),
                    ..default()
                },
                BackgroundColor(BUTTON_BACKGROUND),
                children![(
                    Text::new(if replacing { "Replace" } else { "Save" }),
                    TextFont {
                        font_size: FontSize::Px(15.0),
                        ..default()
                    },
                    TextColor(Color::WHITE),
                )],
            ));
        });
}

fn spawn_saved_row(parent: &mut ChildSpawnerCommands<'_>, entry: &SavedCreation, confirming: bool) {
    let summary = format!(
        "{} part{}, {} joint{}",
        entry.part_count,
        if entry.part_count == 1 { "" } else { "s" },
        entry.joint_count,
        if entry.joint_count == 1 { "" } else { "s" },
    );
    parent
        .spawn(Node {
            width: percent(100),
            column_gap: px(8),
            align_items: AlignItems::Stretch,
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Button,
                CreationMenuAction::LoadFile(entry.path.clone()),
                Node {
                    flex_grow: 1.0,
                    min_height: px(52),
                    padding: UiRect::axes(px(16), px(9)),
                    flex_direction: FlexDirection::Column,
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::FlexStart,
                    row_gap: px(3),
                    border_radius: BorderRadius::all(px(7)),
                    ..default()
                },
                BackgroundColor(BUTTON_BACKGROUND),
                children![
                    (
                        Text::new(entry.name.clone()),
                        TextFont {
                            font_size: FontSize::Px(18.0),
                            ..default()
                        },
                        TextColor(Color::WHITE),
                    ),
                    (
                        Text::new(summary),
                        TextFont {
                            font_size: FontSize::Px(13.0),
                            ..default()
                        },
                        TextColor(SUBTEXT_COLOR),
                    )
                ],
            ));
            row.spawn((
                Button,
                CreationMenuAction::DeleteFile(entry.path.clone()),
                Node {
                    min_width: if confirming { px(96) } else { px(44) },
                    justify_content: JustifyContent::Center,
                    align_items: AlignItems::Center,
                    border_radius: BorderRadius::all(px(7)),
                    ..default()
                },
                BackgroundColor(if confirming {
                    DANGER_BACKGROUND
                } else {
                    BUTTON_BACKGROUND
                }),
                children![(
                    Text::new(if confirming { "Delete?" } else { "×" }),
                    TextFont {
                        font_size: FontSize::Px(if confirming { 14.0 } else { 20.0 }),
                        ..default()
                    },
                    TextColor(Color::WHITE),
                )],
            ));
        });
}

fn spawn_preset_button(parent: &mut ChildSpawnerCommands<'_>, preset: CreationPreset) {
    parent
        .spawn((
            Button,
            CreationMenuAction::LoadPreset(preset),
            Node {
                width: percent(100),
                min_height: px(56),
                padding: UiRect::axes(px(16), px(9)),
                flex_direction: FlexDirection::Column,
                justify_content: JustifyContent::Center,
                align_items: AlignItems::FlexStart,
                row_gap: px(3),
                border_radius: BorderRadius::all(px(7)),
                ..default()
            },
            BackgroundColor(BUTTON_BACKGROUND),
        ))
        .with_children(|button| {
            button.spawn((
                Text::new(preset.label()),
                TextFont {
                    font_size: FontSize::Px(17.0),
                    ..default()
                },
                TextColor(Color::WHITE),
            ));
            button.spawn((
                Text::new(preset.description()),
                TextFont {
                    font_size: FontSize::Px(13.0),
                    ..default()
                },
                TextColor(SUBTEXT_COLOR),
            ));
        });
}

fn spawn_cancel_button(parent: &mut ChildSpawnerCommands<'_>) {
    parent.spawn((
        Button,
        CreationMenuAction::Cancel,
        Node {
            width: percent(100),
            height: px(42),
            margin: UiRect::top(px(10)),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            border_radius: BorderRadius::all(px(7)),
            ..default()
        },
        BackgroundColor(BUTTON_BACKGROUND),
        children![(
            Text::new("Cancel"),
            TextFont {
                font_size: FontSize::Px(16.0),
                ..default()
            },
            TextColor(Color::WHITE),
        )],
    ));
}

/// Drives the modal: typing, clicks, visibility, and rebuilds.
#[allow(clippy::too_many_arguments)] // Bevy system parameters are explicit.
pub(crate) fn update(
    mut commands: Commands,
    mut state: ResMut<CreationMenuState>,
    mut keystrokes: MessageReader<KeyboardInput>,
    wheel: Res<AccumulatedMouseScroll>,
    root: Single<(Entity, &mut Visibility), With<CreationMenuRoot>>,
    mut dialog: Query<&mut ScrollPosition, With<CreationMenuDialog>>,
    mut buttons: Query<(&Interaction, &CreationMenuAction, &mut BackgroundColor), With<Button>>,
) {
    let (root_entity, mut visibility) = root.into_inner();
    if state.open {
        state.blocks_pointer = true;
        handle_typing(&mut state, &mut keystrokes);
    } else {
        keystrokes.clear();
    }
    if state.open {
        handle_clicks(&mut state, &mut buttons);
        handle_scroll(&mut state, &wheel, &mut dialog);
    } else {
        state.held = None;
    }

    *visibility = if state.open {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };
    if state.take_dirty() {
        rebuild(&mut commands, root_entity, &state);
    }
}

/// Scrolls the list with the wheel.
///
/// The camera already ignores the wheel while the modal blocks the pointer, so
/// the whole gesture belongs to the list. Layout clamps an offset past the end,
/// so the clamped value is read back rather than tracked separately.
fn handle_scroll(
    state: &mut CreationMenuState,
    wheel: &AccumulatedMouseScroll,
    dialog: &mut Query<&mut ScrollPosition, With<CreationMenuDialog>>,
) {
    let Ok(mut position) = dialog.single_mut() else {
        return;
    };
    let pixels = wheel.delta.y
        * match wheel.unit {
            MouseScrollUnit::Line => SCROLL_LINE_PIXELS,
            MouseScrollUnit::Pixel => 1.0,
        };
    if pixels != 0.0 {
        position.0.y = (position.0.y - pixels).max(0.0);
    }
    state.scroll = position.0.y;
}

fn handle_typing(state: &mut CreationMenuState, keystrokes: &mut MessageReader<KeyboardInput>) {
    let pressed = keystrokes
        .read()
        .filter(|stroke| stroke.state.is_pressed())
        .map(|stroke| stroke.logical_key.clone())
        .collect::<Vec<_>>();
    for key in pressed {
        match name_entry_action(&key) {
            NameEntryAction::Commit => {
                state.commit_name();
                if !state.open {
                    return;
                }
            }
            NameEntryAction::Cancel => {
                state.cancel();
                return;
            }
            action => state.edit_name(action),
        }
    }
}

fn handle_clicks(
    state: &mut CreationMenuState,
    buttons: &mut Query<(&Interaction, &CreationMenuAction, &mut BackgroundColor), With<Button>>,
) {
    let mut pressed_now = None;
    for (interaction, action, mut background) in buttons.iter_mut() {
        let danger = match (action, state.confirming.as_ref()) {
            (CreationMenuAction::DeleteFile(path), Some(PendingConfirm::Delete(pending))) => {
                path == pending
            }
            _ => false,
        };
        background.0 = match interaction {
            Interaction::Pressed => BUTTON_PRESSED_BACKGROUND,
            Interaction::Hovered => BUTTON_HOVER_BACKGROUND,
            Interaction::None if danger => DANGER_BACKGROUND,
            Interaction::None => BUTTON_BACKGROUND,
        };
        if *interaction == Interaction::Pressed {
            pressed_now = Some(action.clone());
        }
    }
    let Some(action) = pressed_now else {
        state.held = None;
        return;
    };
    // A held button reports `Pressed` every frame, and a rebuild re-spawns it
    // under the same cursor, so only the first frame of a press counts.
    if state.held.as_ref() == Some(&action) {
        return;
    }
    state.held = Some(action.clone());
    match action {
        CreationMenuAction::Save => state.commit_name(),
        CreationMenuAction::LoadFile(path) => {
            state.requested = Some(CreationRequest::Load(path));
            state.close();
        }
        CreationMenuAction::DeleteFile(path) => state.confirm_delete(&path),
        CreationMenuAction::LoadPreset(preset) => {
            state.requested = Some(CreationRequest::LoadPreset(preset));
            state.close();
        }
        CreationMenuAction::Cancel => state.close(),
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use bevy::{
        input::{
            keyboard::{Key, KeyboardInput},
            mouse::AccumulatedMouseScroll,
        },
        prelude::*,
    };

    use super::{
        CreationMenuAction, CreationMenuRoot, CreationMenuState, CreationPreset, CreationRequest,
        NameEntryAction, name_entry_action, update,
    };
    use crate::creation_store::SavedCreation;

    fn entry(name: &str, stem: &str) -> SavedCreation {
        SavedCreation {
            name: name.to_owned(),
            path: PathBuf::from(format!("/creations/{stem}.mech")),
            part_count: 3,
            joint_count: 1,
        }
    }

    /// Builds an app whose only pre-existing button is the one under test.
    ///
    /// A rebuild spawns the modal's real buttons too, so the tests drive this
    /// one entity by id rather than every `Interaction` in the world.
    fn app_with(state: CreationMenuState, action: CreationMenuAction) -> (App, Entity) {
        let mut app = App::new();
        app.insert_resource(state)
            .add_message::<KeyboardInput>()
            .init_resource::<AccumulatedMouseScroll>()
            .add_systems(Update, update);
        app.world_mut()
            .spawn((CreationMenuRoot, Visibility::Hidden));
        let button = app
            .world_mut()
            .spawn((
                Button,
                action,
                Interaction::Pressed,
                BackgroundColor::default(),
            ))
            .id();
        (app, button)
    }

    fn set(app: &mut App, button: Entity, interaction: Interaction) {
        *app.world_mut()
            .entity_mut(button)
            .get_mut::<Interaction>()
            .expect("the test button keeps its interaction") = interaction;
    }

    fn taken(app: &mut App) -> Option<CreationRequest> {
        app.world_mut()
            .resource_mut::<CreationMenuState>()
            .take_request()
    }

    fn opened(entries: Vec<SavedCreation>) -> CreationMenuState {
        let mut state = CreationMenuState::default();
        state.open(entries, String::new(), PathBuf::from("/creations"));
        state
    }

    #[test]
    fn name_entry_takes_printable_characters_and_ignores_the_rest() {
        assert_eq!(
            name_entry_action(&Key::Character("g".into())),
            NameEntryAction::Insert('g')
        );
        assert_eq!(
            name_entry_action(&Key::Character("3".into())),
            NameEntryAction::Insert('3')
        );
        assert_eq!(name_entry_action(&Key::Space), NameEntryAction::Insert(' '));
        assert_eq!(
            name_entry_action(&Key::Backspace),
            NameEntryAction::Backspace
        );
        assert_eq!(name_entry_action(&Key::Enter), NameEntryAction::Commit);
        assert_eq!(name_entry_action(&Key::Escape), NameEntryAction::Cancel);
        assert_eq!(name_entry_action(&Key::F1), NameEntryAction::Ignore);
        assert_eq!(name_entry_action(&Key::ArrowLeft), NameEntryAction::Ignore);
    }

    #[test]
    fn pressed_preset_requests_load_and_closes_modal() {
        let (mut app, _) = app_with(
            opened(Vec::new()),
            CreationMenuAction::LoadPreset(CreationPreset::MobileWorkshop1024),
        );

        app.update();

        let mut state = app.world_mut().resource_mut::<CreationMenuState>();
        assert!(!state.is_open());
        assert!(state.blocks_pointer());
        assert_eq!(
            state.take_request(),
            Some(CreationRequest::LoadPreset(
                CreationPreset::MobileWorkshop1024
            ))
        );
    }

    #[test]
    fn pressed_saved_row_requests_that_file() {
        let saved = entry("Walker v3", "walker-v3");
        let (mut app, _) = app_with(
            opened(vec![saved.clone()]),
            CreationMenuAction::LoadFile(saved.path.clone()),
        );

        app.update();

        let mut state = app.world_mut().resource_mut::<CreationMenuState>();
        assert!(!state.is_open());
        assert_eq!(
            state.take_request(),
            Some(CreationRequest::Load(saved.path))
        );
    }

    #[test]
    fn delete_asks_once_and_only_acts_on_a_second_press() {
        let saved = entry("Doomed", "doomed");
        let (mut app, button) = app_with(
            opened(vec![saved.clone()]),
            CreationMenuAction::DeleteFile(saved.path.clone()),
        );

        app.update();
        assert!(
            app.world().resource::<CreationMenuState>().is_open(),
            "the first press only asks"
        );
        assert_eq!(taken(&mut app), None);

        // Holding the button down must not resolve the confirmation by itself.
        app.update();
        assert_eq!(
            taken(&mut app),
            None,
            "a held button must not confirm itself"
        );

        set(&mut app, button, Interaction::None);
        app.update();
        set(&mut app, button, Interaction::Pressed);
        app.update();

        assert_eq!(taken(&mut app), Some(CreationRequest::Delete(saved.path)));
    }

    #[test]
    fn save_asks_once_before_replacing_an_existing_name() {
        let mut state = opened(vec![entry("Walker v3", "walker-v3")]);
        state.name = "Walker V3".to_owned();
        let (mut app, button) = app_with(state, CreationMenuAction::Save);

        app.update();
        assert!(
            app.world().resource::<CreationMenuState>().is_open(),
            "a colliding name only asks first"
        );
        assert_eq!(taken(&mut app), None);

        set(&mut app, button, Interaction::None);
        app.update();
        set(&mut app, button, Interaction::Pressed);
        app.update();

        assert_eq!(
            taken(&mut app),
            Some(CreationRequest::Save("Walker V3".to_owned()))
        );
    }

    #[test]
    fn save_under_a_fresh_name_writes_immediately() {
        let mut state = opened(vec![entry("Walker v3", "walker-v3")]);
        state.name = "  Gearbox  ".to_owned();
        let (mut app, _) = app_with(state, CreationMenuAction::Save);

        app.update();

        let mut state = app.world_mut().resource_mut::<CreationMenuState>();
        assert!(!state.is_open());
        assert_eq!(
            state.take_request(),
            Some(CreationRequest::Save("Gearbox".to_owned())),
            "the stored name is trimmed"
        );
    }

    #[test]
    fn empty_name_is_refused_without_closing() {
        let (mut app, _) = app_with(opened(Vec::new()), CreationMenuAction::Save);

        app.update();

        let mut state = app.world_mut().resource_mut::<CreationMenuState>();
        assert!(state.is_open());
        assert_eq!(state.take_request(), None);
    }

    #[test]
    fn open_modal_owns_the_keyboard() {
        let mut state = CreationMenuState::default();
        assert!(!state.blocks_keyboard());
        state.open(Vec::new(), String::new(), PathBuf::new());
        assert!(state.blocks_keyboard(), "typing must not fire shortcuts");
        state.close();
        assert!(!state.blocks_keyboard());
    }

    #[test]
    fn escape_clears_the_name_before_it_closes() {
        let mut state = opened(Vec::new());
        state.name = "Gear".to_owned();

        state.cancel();
        assert!(state.is_open(), "the first Escape only clears the name");
        assert!(state.name.is_empty());

        state.cancel();
        assert!(!state.is_open());
    }
}

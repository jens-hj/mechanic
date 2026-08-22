//! Modal that saves the current creation and opens a saved or preset one.
//!
//! One screen does both jobs: type a name and press Enter to save, or click a
//! row to open it. Rows come from the creations directory, so the list is
//! rebuilt from disk each time the modal opens rather than tracked live.
//!
//! What it looks like lives in [`crate::ui::creations`]; this is the state
//! behind it — what is typed, what is on disk, and what has been asked for
//! twice.

use std::path::{Path, PathBuf};

use bevy::prelude::*;

use crate::creation_store::{SavedCreation, slug};
pub(crate) use crate::showcase::CreationPreset;
use crate::ui::CreationsAction;

/// Longest display name the field accepts.
const MAX_NAME_LENGTH: usize = 60;

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
    requested: Option<CreationRequest>,
}

impl CreationMenuState {
    /// Opens the modal on a freshly read directory listing.
    pub(crate) fn open(&mut self, entries: Vec<SavedCreation>, name: String, directory: PathBuf) {
        self.open = true;
        self.name = name;
        self.entries = entries;
        self.directory = directory;
        self.confirming = None;
        self.notice = None;
    }

    /// Closes the modal, discarding a half-typed name.
    pub(crate) fn close(&mut self) {
        self.open = false;
        self.name.clear();
        self.confirming = None;
        self.notice = None;
    }

    /// Replaces the directory listing shown by an already-open modal.
    pub(crate) fn set_entries(&mut self, entries: Vec<SavedCreation>) {
        self.entries = entries;
    }

    /// Shows a one-line message inside the modal.
    pub(crate) fn notify(&mut self, notice: impl Into<String>) {
        self.notice = Some(notice.into());
    }

    /// Whether the modal is showing.
    pub(crate) const fn is_open(&self) -> bool {
        self.open
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
    }

    /// Takes what was typed into the name field.
    fn set_name(&mut self, name: String) {
        let name: String = name.chars().take(MAX_NAME_LENGTH).collect();
        if self.name == name {
            return;
        }
        self.name = name;
        // Editing the name withdraws a replace prompt that named the old one.
        if self.confirming == Some(PendingConfirm::Replace) {
            self.confirming = None;
        }
        self.notice = None;
    }

    /// Does what the picker was asked to do.
    ///
    /// The one way in: the overlay reports what a person did, and every
    /// decision about what that means is made here.
    pub(crate) fn act(&mut self, action: CreationsAction) {
        match action {
            CreationsAction::Name(name) => self.set_name(name),
            CreationsAction::Save => self.commit_name(),
            CreationsAction::Load(path) => {
                self.requested = Some(CreationRequest::Load(path));
                self.close();
            }
            CreationsAction::Delete(path) => self.confirm_delete(&path),
            CreationsAction::Preset(preset) => {
                self.requested = Some(CreationRequest::LoadPreset(preset));
                self.close();
            }
            CreationsAction::Cancel => self.cancel(),
        }
    }

    /// The name typed so far.
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    /// Where the listed creations came from.
    pub(crate) fn directory(&self) -> &Path {
        &self.directory
    }

    /// The message showing inside the modal, when there is one.
    pub(crate) fn notice(&self) -> Option<&str> {
        self.notice.as_deref()
    }

    /// What is on disk.
    pub(crate) fn entries(&self) -> &[SavedCreation] {
        &self.entries
    }

    /// Whether saving would write over an existing creation.
    pub(crate) fn is_replacing(&self) -> bool {
        self.confirming == Some(PendingConfirm::Replace)
    }

    /// Whether this creation has already been asked to be deleted once.
    pub(crate) fn is_confirming_delete(&self, path: &Path) -> bool {
        self.confirming == Some(PendingConfirm::Delete(path.to_path_buf()))
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{CreationMenuState, CreationPreset, CreationRequest};
    use crate::creation_store::SavedCreation;
    use crate::ui::CreationsAction;

    fn entry(name: &str, stem: &str) -> SavedCreation {
        SavedCreation {
            name: name.to_owned(),
            path: PathBuf::from(format!("/creations/{stem}.mech")),
            part_count: 3,
            joint_count: 1,
        }
    }

    fn opened(entries: Vec<SavedCreation>) -> CreationMenuState {
        let mut state = CreationMenuState::default();
        state.open(entries, String::new(), PathBuf::from("/creations"));
        state
    }

    #[test]
    fn a_preset_is_opened_and_the_picker_closes_behind_it() {
        let mut state = opened(Vec::new());
        state.act(CreationsAction::Preset(CreationPreset::MobileWorkshop1024));
        assert!(!state.is_open());
        assert_eq!(
            state.take_request(),
            Some(CreationRequest::LoadPreset(
                CreationPreset::MobileWorkshop1024
            ))
        );
    }

    #[test]
    fn a_saved_row_opens_that_file() {
        let saved = entry("Walker v3", "walker-v3");
        let mut state = opened(vec![saved.clone()]);
        state.act(CreationsAction::Load(saved.path.clone()));
        assert!(!state.is_open());
        assert_eq!(
            state.take_request(),
            Some(CreationRequest::Load(saved.path))
        );
    }

    #[test]
    fn delete_asks_once_and_only_acts_on_a_second_press() {
        let saved = entry("Doomed", "doomed");
        let mut state = opened(vec![saved.clone()]);

        state.act(CreationsAction::Delete(saved.path.clone()));
        assert!(state.is_open(), "the first press only asks");
        assert_eq!(state.take_request(), None);
        assert!(state.is_confirming_delete(&saved.path));

        state.act(CreationsAction::Delete(saved.path.clone()));
        assert_eq!(
            state.take_request(),
            Some(CreationRequest::Delete(saved.path))
        );
    }

    #[test]
    fn save_asks_once_before_replacing_an_existing_name() {
        let mut state = opened(vec![entry("Walker v3", "walker-v3")]);
        state.act(CreationsAction::Name("Walker V3".to_owned()));

        state.act(CreationsAction::Save);
        assert!(state.is_open(), "a colliding name only asks first");
        assert!(state.is_replacing());
        assert_eq!(state.take_request(), None);

        state.act(CreationsAction::Save);
        assert_eq!(
            state.take_request(),
            Some(CreationRequest::Save("Walker V3".to_owned()))
        );
    }

    /// Typing after the prompt withdraws it: the warning named the creation the
    /// old text collided with, and the new text may collide with nothing.
    #[test]
    fn editing_the_name_withdraws_a_replace_prompt() {
        let mut state = opened(vec![entry("Walker v3", "walker-v3")]);
        state.act(CreationsAction::Name("Walker V3".to_owned()));
        state.act(CreationsAction::Save);
        assert!(state.is_replacing());

        state.act(CreationsAction::Name("Walker V4".to_owned()));
        assert!(!state.is_replacing());
        assert_eq!(state.notice(), None);
    }

    #[test]
    fn save_under_a_fresh_name_writes_immediately() {
        let mut state = opened(vec![entry("Walker v3", "walker-v3")]);
        state.act(CreationsAction::Name("  Gearbox  ".to_owned()));
        state.act(CreationsAction::Save);

        assert!(!state.is_open());
        assert_eq!(
            state.take_request(),
            Some(CreationRequest::Save("Gearbox".to_owned())),
            "the stored name is trimmed",
        );
    }

    #[test]
    fn an_empty_name_is_refused_without_closing() {
        let mut state = opened(Vec::new());
        state.act(CreationsAction::Save);
        assert!(state.is_open());
        assert_eq!(state.take_request(), None);
        assert_eq!(state.notice(), Some("Type a name first"));
    }

    /// Backing out clears a half-typed name before it closes the picker, so a
    /// stray Escape does not throw away the list as well as the word.
    #[test]
    fn cancel_clears_the_name_before_it_closes() {
        let mut state = opened(Vec::new());
        state.act(CreationsAction::Name("Half typed".to_owned()));

        state.act(CreationsAction::Cancel);
        assert!(state.is_open(), "the first cancel clears the field");
        assert_eq!(state.name(), "");

        state.act(CreationsAction::Cancel);
        assert!(!state.is_open());
    }

    #[test]
    fn a_name_is_cut_at_the_length_the_field_accepts() {
        let mut state = opened(Vec::new());
        state.act(CreationsAction::Name(
            "x".repeat(super::MAX_NAME_LENGTH + 20),
        ));
        assert_eq!(state.name().chars().count(), super::MAX_NAME_LENGTH);
    }

    #[test]
    fn an_open_picker_owns_the_keyboard() {
        let mut state = CreationMenuState::default();
        assert!(!state.blocks_keyboard());
        state.open(Vec::new(), String::new(), PathBuf::new());
        assert!(state.blocks_keyboard(), "typing must not fire shortcuts");
        state.close();
        assert!(!state.blocks_keyboard());
    }
}

//! When a dirty world is due to be written.

use std::time::Duration;

/// Delay after the last mutation before an ordinary autosave.
pub const AUTOSAVE_DEBOUNCE: Duration = Duration::from_secs(2);

/// Maximum time dirty data waits even while mutations continue.
pub const AUTOSAVE_DIRTY_INTERVAL: Duration = Duration::from_secs(30);

/// Autosave timing state independent of Bevy's scheduling layer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AutosaveState {
    pub(super) dirty_since: Option<Duration>,
    pub(super) last_mutation: Option<Duration>,
}

impl AutosaveState {
    /// Marks persistent world state dirty at monotonic `now`.
    pub fn mutate(&mut self, now: Duration) {
        self.dirty_since.get_or_insert(now);
        self.last_mutation = Some(now);
    }

    /// Whether the debounce or maximum dirty interval requires a save.
    pub fn due(self, now: Duration) -> bool {
        self.dirty_since
            .is_some_and(|dirty| now.saturating_sub(dirty) >= AUTOSAVE_DIRTY_INTERVAL)
            || self
                .last_mutation
                .is_some_and(|mutation| now.saturating_sub(mutation) >= AUTOSAVE_DEBOUNCE)
    }

    /// Clears dirty timing only after every atomic write succeeds.
    pub fn saved(&mut self) {
        self.dirty_since = None;
        self.last_mutation = None;
    }

    /// True when any persistent mutation remains unsaved.
    pub const fn is_dirty(self) -> bool {
        self.dirty_since.is_some()
    }
}

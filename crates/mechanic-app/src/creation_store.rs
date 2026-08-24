//! Where saved creations live on disk, and how they get there.
//!
//! Files land in the platform's application-data directory so the app finds
//! them by itself, from any working directory, and so `cargo clean` cannot take
//! a machine with it. The store owns its directory rather than resolving it on
//! every call, which keeps the tests off the real home directory without
//! mutating process environment.

use std::{
    env, fs, io,
    path::{Path, PathBuf},
};

use bevy::prelude::Resource;
use mechanic_core::{CreationDocument, CreationError};
use thiserror::Error;

/// Extension every saved creation carries. The contents are RON.
pub(crate) const CREATION_EXTENSION: &str = "mech";

/// Directory that overrides the platform default, for tests and for anyone who
/// wants their creations somewhere specific.
const DIRECTORY_OVERRIDE: &str = "MECHANIC_CREATIONS_DIR";

/// Name a creation falls back to when its own reduces to nothing usable.
const FALLBACK_SLUG: &str = "creation";

/// Reason a creation could not be read from or written to disk.
#[derive(Debug, Error)]
pub(crate) enum StoreError {
    /// The file could not be read, written, or removed.
    #[error("{0}")]
    Io(#[from] io::Error),
    /// The file is not readable RON.
    #[error("file is not a readable creation: {0}")]
    Parse(#[from] ron::error::SpannedError),
    /// The document could not be encoded.
    #[error("creation could not be encoded: {0}")]
    Encode(#[from] ron::Error),
    /// The file parsed but does not describe a valid construction.
    #[error("{0}")]
    Creation(#[from] CreationError),
}

/// One creation on disk, summarised for a menu row without loading it twice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SavedCreation {
    /// Display name the file carries, which is not necessarily its file stem.
    pub(crate) name: String,
    /// Full path, used to load or delete it.
    pub(crate) path: PathBuf,
    /// Number of parts, for the row's summary line.
    pub(crate) part_count: usize,
    /// Number of bearings, for the row's summary line.
    pub(crate) joint_count: usize,
}

/// The directory saved creations are read from and written to.
#[derive(Resource, Clone, Debug)]
pub(crate) struct CreationStore {
    directory: PathBuf,
}

impl Default for CreationStore {
    fn default() -> Self {
        Self::from_environment()
    }
}

impl CreationStore {
    /// Resolves the platform's creations directory.
    pub(crate) fn from_environment() -> Self {
        Self {
            directory: env::var_os(DIRECTORY_OVERRIDE).map_or_else(
                || {
                    data_root().map_or_else(
                        || PathBuf::from("creations"),
                        |root| root.join("mechanic").join("creations"),
                    )
                },
                PathBuf::from,
            ),
        }
    }

    /// Uses an explicit directory. Tests build stores this way so they never
    /// touch a real home directory.
    #[cfg(test)]
    pub(crate) fn new(directory: impl Into<PathBuf>) -> Self {
        Self {
            directory: directory.into(),
        }
    }

    /// The directory this store reads and writes.
    pub(crate) fn directory(&self) -> &Path {
        &self.directory
    }

    /// Where a creation with this display name would be written.
    pub(crate) fn path_for(&self, name: &str) -> PathBuf {
        self.directory
            .join(slug(name))
            .with_extension(CREATION_EXTENSION)
    }

    /// Every readable creation in the directory, ordered by display name.
    ///
    /// A file that cannot be read or parsed is skipped rather than fatal, so
    /// one bad file never hides the rest.
    pub(crate) fn list(&self) -> Vec<SavedCreation> {
        let Ok(entries) = fs::read_dir(&self.directory) else {
            return Vec::new();
        };
        let mut creations = entries
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .is_some_and(|ext| ext == CREATION_EXTENSION)
            })
            .filter_map(|path| {
                let document = read_document(&path).ok()?;
                Some(SavedCreation {
                    name: document.name,
                    path,
                    part_count: document.parts.len(),
                    joint_count: document.bearings.len(),
                })
            })
            .collect::<Vec<_>>();
        creations.sort_by(|first, second| {
            first
                .name
                .to_lowercase()
                .cmp(&second.name.to_lowercase())
                .then_with(|| first.path.cmp(&second.path))
        });
        creations
    }

    /// Writes a creation, replacing any file with the same slug.
    ///
    /// The document goes to a temporary file that is then renamed over the
    /// target, so an interrupted write cannot destroy an existing good save.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError`] when the document cannot be encoded or the
    /// directory cannot be created or written.
    pub(crate) fn save(&self, document: &CreationDocument) -> Result<PathBuf, StoreError> {
        fs::create_dir_all(&self.directory)?;
        let text = ron::ser::to_string_pretty(document, ron::ser::PrettyConfig::default())?;
        let path = self.path_for(&document.name);
        let temporary = path.with_extension(format!("{CREATION_EXTENSION}.tmp"));
        fs::write(&temporary, text)?;
        fs::rename(&temporary, &path)?;
        Ok(path)
    }
}

/// Reads one creation file.
///
/// # Errors
///
/// Returns [`StoreError`] when the file cannot be read or is not readable RON.
pub(crate) fn read_document(path: &Path) -> Result<CreationDocument, StoreError> {
    Ok(ron::from_str(&fs::read_to_string(path)?)?)
}

/// Removes one creation file.
///
/// # Errors
///
/// Returns [`StoreError`] when the file cannot be removed.
pub(crate) fn delete(path: &Path) -> Result<(), StoreError> {
    Ok(fs::remove_file(path)?)
}

/// The platform's per-user application-data directory.
fn data_root() -> Option<PathBuf> {
    if cfg!(target_os = "macos") {
        env::var_os("HOME").map(|home| Path::new(&home).join("Library/Application Support"))
    } else if cfg!(target_os = "windows") {
        env::var_os("APPDATA").map(PathBuf::from)
    } else {
        env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| env::var_os("HOME").map(|home| Path::new(&home).join(".local/share")))
    }
}

/// Reduces a display name to a safe file stem.
///
/// Only ASCII alphanumerics survive; every run of anything else becomes a
/// single hyphen. The display name itself is kept inside the file, so this
/// losing information is fine — renaming stays lossless.
pub(crate) fn slug(name: &str) -> String {
    let mut slug = String::with_capacity(name.len());
    for character in name.chars() {
        if character.is_ascii_alphanumeric() {
            slug.push(character.to_ascii_lowercase());
        } else if !slug.ends_with('-') {
            slug.push('-');
        }
    }
    let trimmed = slug.trim_matches('-');
    if trimmed.is_empty() {
        FALLBACK_SLUG.to_owned()
    } else {
        trimmed.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use mechanic_core::{CREATION_FORMAT_VERSION, CreationDocument, PartDoc, PoseDoc};

    use super::{CreationStore, SavedCreation, delete, read_document, slug};

    /// A directory of its own per test, so they stay parallel-safe without
    /// mutating process environment.
    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new() -> Self {
            static COUNTER: AtomicU32 = AtomicU32::new(0);
            let path = std::env::temp_dir().join(format!(
                "mechanic-store-{}-{}",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::Relaxed)
            ));
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn document(name: &str, parts: usize) -> CreationDocument {
        CreationDocument {
            version: CREATION_FORMAT_VERSION,
            name: name.to_owned(),
            parts: (0..parts)
                .map(|index| PartDoc::Cuboid {
                    dimensions: [1, 1, 1],
                    pose: PoseDoc {
                        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                        translation_half_units: [index as i32 * 2, 1, 0],
                        rotation: [0, 0, 0],
                    },
                    material: mechanic_core::ConstructionMaterial::Steel,
                })
                .collect(),
            welds: Vec::new(),
            rigid_links: Vec::new(),
            bearings: Vec::new(),
            drive_links: Vec::new(),
            input_seat_links: Vec::new(),
            seat_controller_links: Vec::new(),
            sockets: Vec::new(),
        }
    }

    #[test]
    fn slug_keeps_alphanumerics_and_collapses_everything_else() {
        assert_eq!(slug("Walker v3"), "walker-v3");
        assert_eq!(slug("  Gear__box!!  "), "gear-box");
        assert_eq!(slug("Åäö"), "creation");
        assert_eq!(slug(""), "creation");
        assert_eq!(slug("../../etc/passwd"), "etc-passwd");
    }

    #[test]
    fn saving_then_listing_and_loading_round_trips() {
        let temporary = TempDir::new();
        let store = CreationStore::new(&temporary.0);
        assert!(
            store.list().is_empty(),
            "a missing directory lists as empty"
        );

        let path = store
            .save(&document("Walker v3", 3))
            .expect("the save writes");
        assert_eq!(path, store.path_for("Walker v3"));
        assert_eq!(path.file_name().unwrap(), "walker-v3.mech");

        store
            .save(&document("Gearbox", 1))
            .expect("the save writes");

        let listed = store.list();
        assert_eq!(
            listed
                .iter()
                .map(|entry| entry.name.as_str())
                .collect::<Vec<_>>(),
            ["Gearbox", "Walker v3"],
            "rows come back ordered by display name"
        );
        assert_eq!(
            listed[1],
            SavedCreation {
                name: "Walker v3".to_owned(),
                path: path.clone(),
                part_count: 3,
                joint_count: 0,
            }
        );

        let loaded = read_document(&path).expect("the file parses");
        assert_eq!(loaded, document("Walker v3", 3));
    }

    #[test]
    fn saving_the_same_name_replaces_rather_than_duplicates() {
        let temporary = TempDir::new();
        let store = CreationStore::new(&temporary.0);
        store
            .save(&document("Rig", 1))
            .expect("the first save writes");
        store
            .save(&document("Rig", 5))
            .expect("the second save writes");

        let listed = store.list();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].part_count, 5);
    }

    #[test]
    fn deleting_removes_only_that_creation() {
        let temporary = TempDir::new();
        let store = CreationStore::new(&temporary.0);
        let doomed = store.save(&document("Doomed", 1)).expect("the save writes");
        store.save(&document("Kept", 1)).expect("the save writes");

        delete(&doomed).expect("the file is removed");

        let listed = store.list();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "Kept");
    }

    #[test]
    fn unreadable_files_are_skipped_rather_than_hiding_the_rest() {
        let temporary = TempDir::new();
        let store = CreationStore::new(&temporary.0);
        store.save(&document("Good", 1)).expect("the save writes");
        std::fs::write(temporary.0.join("broken.mech"), "this is not RON")
            .expect("the broken file is written");
        std::fs::write(temporary.0.join("notes.txt"), "ignored").expect("the note is written");

        let listed = store.list();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "Good");
    }

    #[test]
    fn environment_store_resolves_a_creations_directory() {
        let store = CreationStore::from_environment();
        assert!(
            store.directory().ends_with("creations"),
            "expected a creations directory, got {}",
            store.directory().display()
        );
    }
}

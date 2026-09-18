//! File-level helpers: atomic writes, manifests, the data root, and slugs.

use super::error::WorldSaveError;
use crate::{WorldGeneratorVersion, WorldSeed};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Deserialize)]
pub(super) struct ManifestHeader {
    pub(super) version: u32,
    pub(super) name: String,
    pub(super) generator_version: WorldGeneratorVersion,
    pub(super) seed: WorldSeed,
    pub(super) last_played_unix_seconds: u64,
}

pub(super) fn inspect_manifest(path: &Path) -> Result<ManifestHeader, WorldSaveError> {
    let text = read_exact(path)?;
    ron::from_str(&text).map_err(|source| WorldSaveError::Decode {
        path: path.to_owned(),
        source: Box::new(source),
    })
}

pub(super) fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), WorldSaveError> {
    let parent = path
        .parent()
        .expect("every world file has a parent directory");
    fs::create_dir_all(parent).map_err(|source| WorldSaveError::Io {
        path: parent.to_owned(),
        source,
    })?;
    let temporary = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or("save")
    ));
    fs::write(&temporary, bytes).map_err(|source| WorldSaveError::Io {
        path: temporary.clone(),
        source,
    })?;
    fs::rename(&temporary, path).map_err(|source| WorldSaveError::Io {
        path: path.to_owned(),
        source,
    })
}

pub(super) fn read_exact(path: &Path) -> Result<String, WorldSaveError> {
    fs::read_to_string(path).map_err(|source| WorldSaveError::Io {
        path: path.to_owned(),
        source,
    })
}

pub(super) fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub(super) fn data_root() -> Option<PathBuf> {
    if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map(|home| Path::new(&home).join("Library/Application Support"))
    } else if cfg!(target_os = "windows") {
        std::env::var_os("APPDATA").map(PathBuf::from)
    } else {
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|home| Path::new(&home).join(".local/share")))
    }
}

pub(super) fn slug(name: &str) -> String {
    let mut slug = String::with_capacity(name.len());
    for character in name.chars() {
        if character.is_ascii_alphanumeric() {
            slug.push(character.to_ascii_lowercase());
        } else if !slug.ends_with('-') {
            slug.push('-');
        }
    }
    let slug = slug.trim_matches('-');
    if slug.is_empty() {
        "world".to_owned()
    } else {
        slug.to_owned()
    }
}

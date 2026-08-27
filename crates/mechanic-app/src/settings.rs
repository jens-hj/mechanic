//! Persistent application preferences that are not part of a construction.

use std::{fs, io, path::PathBuf};

use bevy::prelude::{Resource, warn};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::controls::{Controls, GameAction, InputChord};
use crate::creation_store::CreationStore;

const SETTINGS_VERSION: u32 = 1;
#[cfg(test)]
const SETTINGS_FILE: &str = "settings.ron";
pub(crate) const DEFAULT_CAMERA_FOV_DEGREES: f32 = 45.0;
pub(crate) const MIN_CAMERA_FOV_DEGREES: f32 = 45.0;
pub(crate) const MAX_CAMERA_FOV_DEGREES: f32 = 100.0;
pub(crate) const CAMERA_FOV_STEP_DEGREES: f32 = 5.0;

#[derive(Clone, Debug, Deserialize, Serialize)]
struct SettingsDocument {
    version: u32,
    camera_fov_degrees: f32,
    #[serde(default)]
    controls: Controls,
}

#[derive(Debug, Error)]
pub(crate) enum SettingsError {
    #[error("{0}")]
    Io(#[from] io::Error),
    #[error("settings are not readable: {0}")]
    Parse(#[from] ron::error::SpannedError),
    #[error("settings could not be encoded: {0}")]
    Encode(#[from] ron::Error),
    #[error("unsupported settings version {0}")]
    UnsupportedVersion(u32),
}

/// The settings file and the preference currently applied by the app.
#[derive(Resource, Clone, Debug)]
pub(crate) struct AppSettings {
    path: PathBuf,
    camera_fov_degrees: f32,
    controls: Controls,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self::load(CreationStore::from_environment().settings_path())
    }
}

impl AppSettings {
    fn load(path: PathBuf) -> Self {
        let (camera_fov_degrees, controls) = match read_document(&path) {
            Ok(mut document) => {
                document.controls.normalize();
                (normalized_fov(&document), document.controls)
            }
            Err(SettingsError::Io(error)) if error.kind() == io::ErrorKind::NotFound => {
                (DEFAULT_CAMERA_FOV_DEGREES, Controls::default())
            }
            Err(error) => {
                warn!("could not load settings from {}: {error}", path.display());
                (DEFAULT_CAMERA_FOV_DEGREES, Controls::default())
            }
        };
        Self {
            path,
            camera_fov_degrees,
            controls,
        }
    }

    #[cfg(test)]
    pub(crate) fn from_path(path: PathBuf) -> Self {
        Self::load(path)
    }

    pub(crate) const fn camera_fov_degrees(&self) -> f32 {
        self.camera_fov_degrees
    }

    pub(crate) const fn controls(&self) -> &Controls {
        &self.controls
    }

    /// Applies and atomically persists a camera field of view.
    pub(crate) fn set_camera_fov_degrees(&mut self, degrees: f32) -> Result<(), SettingsError> {
        self.camera_fov_degrees = normalized_fov(&SettingsDocument {
            version: SETTINGS_VERSION,
            camera_fov_degrees: degrees,
            controls: self.controls.clone(),
        });
        self.save()
    }

    /// Changes one binding slot and atomically persists the whole settings document.
    pub(crate) fn set_binding(
        &mut self,
        action: GameAction,
        slot: usize,
        chord: Option<InputChord>,
    ) -> Result<(), SettingsError> {
        self.controls.set(action, slot, chord);
        self.save()
    }

    /// Restores every gameplay action to its shipped bindings.
    pub(crate) fn reset_controls(&mut self) -> Result<(), SettingsError> {
        self.controls = Controls::default();
        self.save()
    }

    fn save(&self) -> Result<(), SettingsError> {
        if let Some(parent) = self
            .path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        let document = SettingsDocument {
            version: SETTINGS_VERSION,
            camera_fov_degrees: self.camera_fov_degrees,
            controls: self.controls.clone(),
        };
        let text = ron::ser::to_string_pretty(&document, ron::ser::PrettyConfig::default())?;
        let temporary = self.path.with_extension("ron.tmp");
        fs::write(&temporary, text)?;
        fs::rename(temporary, &self.path)?;
        Ok(())
    }
}

fn read_document(path: &std::path::Path) -> Result<SettingsDocument, SettingsError> {
    let document: SettingsDocument = ron::from_str(&fs::read_to_string(path)?)?;
    if document.version != SETTINGS_VERSION {
        return Err(SettingsError::UnsupportedVersion(document.version));
    }
    Ok(document)
}

fn normalized_fov(document: &SettingsDocument) -> f32 {
    if !document.camera_fov_degrees.is_finite() {
        return DEFAULT_CAMERA_FOV_DEGREES;
    }
    let clamped = document
        .camera_fov_degrees
        .clamp(MIN_CAMERA_FOV_DEGREES, MAX_CAMERA_FOV_DEGREES);
    (clamped / CAMERA_FOV_STEP_DEGREES).round() * CAMERA_FOV_STEP_DEGREES
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            static COUNTER: AtomicU32 = AtomicU32::new(0);
            let path = std::env::temp_dir().join(format!(
                "mechanic-settings-{}-{}",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&path).expect("temporary directory");
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn absent_and_malformed_settings_use_the_default() {
        let temporary = TempDir::new();
        let path = temporary.0.join(SETTINGS_FILE);
        assert_eq!(
            AppSettings::from_path(path.clone()).camera_fov_degrees(),
            DEFAULT_CAMERA_FOV_DEGREES
        );
        fs::write(&path, "not ron").expect("malformed fixture");
        assert_eq!(
            AppSettings::from_path(path).camera_fov_degrees(),
            DEFAULT_CAMERA_FOV_DEGREES
        );
    }

    #[test]
    fn values_are_clamped_and_rounded_to_slider_steps() {
        let temporary = TempDir::new();
        for (value, expected) in [(20.0, 45.0), (87.0, 85.0), (140.0, 100.0)] {
            let path = temporary.0.join(format!("{value}.ron"));
            let text = ron::ser::to_string(&SettingsDocument {
                version: SETTINGS_VERSION,
                camera_fov_degrees: value,
                controls: Controls::default(),
            })
            .expect("fixture encodes");
            fs::write(&path, text).expect("fixture writes");
            assert_eq!(AppSettings::from_path(path).camera_fov_degrees(), expected);
        }
    }

    #[test]
    fn atomic_save_round_trips_without_leaving_the_temporary_file() {
        let temporary = TempDir::new();
        let path = temporary.0.join(SETTINGS_FILE);
        let mut settings = AppSettings::from_path(path.clone());
        settings
            .set_camera_fov_degrees(75.0)
            .expect("settings save");

        assert_eq!(AppSettings::from_path(path).camera_fov_degrees(), 75.0);
        assert!(!temporary.0.join("settings.ron.tmp").exists());
    }

    #[test]
    fn old_fov_only_document_loads_default_controls() {
        let temporary = TempDir::new();
        let path = temporary.0.join(SETTINGS_FILE);
        fs::write(&path, "(version:1,camera_fov_degrees:65.0)").expect("old fixture writes");
        let settings = AppSettings::from_path(path);
        assert_eq!(settings.camera_fov_degrees(), 65.0);
        assert_eq!(settings.controls().label(GameAction::Rotate), "R");
    }

    #[test]
    fn binding_changes_round_trip_and_reset() {
        let temporary = TempDir::new();
        let path = temporary.0.join(SETTINGS_FILE);
        let mut settings = AppSettings::from_path(path.clone());
        settings
            .set_binding(
                GameAction::Rotate,
                0,
                Some(InputChord::key(bevy::prelude::KeyCode::KeyT)),
            )
            .expect("binding saves");
        assert_eq!(
            AppSettings::from_path(path.clone())
                .controls()
                .label(GameAction::Rotate),
            "T"
        );
        settings.reset_controls().expect("reset saves");
        assert_eq!(
            AppSettings::from_path(path)
                .controls()
                .label(GameAction::Rotate),
            "R"
        );
    }
}
